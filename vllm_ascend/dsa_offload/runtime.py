# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 数据面的 worker-lifetime tensor owner 与逐层执行上下文。

``DSAInputBatchCacheLayout`` 仍是请求行状态的唯一 worker 投影；本模块只把
其 active-prefix 转成设备算子需要的固定地址 view：

* resident slot-mapping position；
* active DRAM block table；
* 所有层串行复用的 caller-owned LIDU scratch 输出；
* 本轮满块 dump 的紧凑 src/dst 列。
* prefix hit 的 DRAM src 与 target Indexer/MLA/可选 MTP HBM dst 固定 restore 列。

固定 MTP3 使用两块 parity tail，位置映射化简为 ``position % 256``。
已确认满块在下一 outer step、当前 target 层写 KV 前 dump；每个 target
物理层只发射一次；logical block id 由 scheduler 分配，worker 只维护固定行
镜像，同一 assignment 在 target Indexer/MLA/可选 MTP arena 间共享。
MTP proposer 仍使用原生 BF16 full cache，只绑定不改变 attention 语义的 durable
full-block copy hook。

所有大 tensor 在 model runner 初始化期预分配。steady step 只刷新 active
prefix；DRAM 逻辑表未变化时不会重复 H2D。图模式复用这些 owner 的
captured-prefix + PAD view，而不是另建一套 graph-only 语义。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer

from vllm_ascend.dsa_offload.contracts import (
    DSA_A5_MTP_ATTENTION_CAPACITY,
    DSA_A5_NOMTP_ATTENTION_CAPACITY,
    DSA_DRAM_NULL_BLOCK_ID,
    DSA_DUMP_NOOP_DST_BLOCK_ID,
    DSA_LIDU_OUTPUT_CAPACITY,
    DSA_ROW_MODE_SPARSE,
)
from vllm_ascend.dsa_offload.ops import (
    DSALightningIndexerOutputs,
    DSAOffloadSelectionOutput,
    a5_kvcache_scatter_copy_c8,
    a5_lightning_indexer_decode_update_c8,
    a5_lightning_indexer_decode_update_mtp_c8,
    copy_full_cache_plane_bytes,
    dump_full_cache_plane_bytes,
    dump_full_kv_cache_blocks,
    dump_full_kv_cache_blocks_c8,
    kvcache_scatter_copy,
    lightning_indexer_decode_update,
    sparse_flash_attention_for_offload,
    sparse_flash_attention_for_offload_c8,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.resident_pool import DSAResidentTokenPool
from vllm_ascend.dsa_offload.worker_dram_store import DSAWorkerDRAMStore

if TYPE_CHECKING:
    from vllm_ascend.dsa_offload.input_batch import (
        DSAInputBatchCacheLayout,
    )
    from vllm_ascend.worker.npu_input_batch import NPUInputBatch


class DSAOffloadRuntime:
    """持有 eager 与后续 graph 共用的固定地址设备元数据。"""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        max_num_tokens: int,
        num_layers: int,
        max_model_len: int,
        block_size: int,
        resident_token_pool: DSAResidentTokenPool,
        device: torch.device,
        pin_memory: bool,
        packed_c8: bool = False,
        max_decode_query_len: int = 1,
        enable_prefix_caching: bool = False,
        mtp_durable_offload: bool = False,
    ) -> None:
        self.max_num_reqs = int(max_num_reqs)
        self.max_num_tokens = int(max_num_tokens)
        self.num_layers = int(num_layers)
        self.max_model_len = int(max_model_len)
        self.block_size = int(block_size)
        self.device = torch.device(device)
        self.packed_c8 = bool(packed_c8)
        self.max_decode_query_len = int(max_decode_query_len)
        self.enable_prefix_caching = bool(enable_prefix_caching)
        self.mtp_durable_offload = bool(mtp_durable_offload)
        if self.mtp_durable_offload and self.max_decode_query_len <= 1:
            raise ValueError(
                "DSA MTP durable offload requires an MTP decode query width"
            )
        if self.max_decode_query_len <= 0:
            raise ValueError("DSA max_decode_query_len must be positive")
        self.resident_token_pool = resident_token_pool
        self.sparse_tail_block_count = (
            2 if self.max_decode_query_len > 1 else 1
        )
        self.sparse_tail_capacity_tokens = (
            self.sparse_tail_block_count * self.block_size
        )
        self.max_logical_blocks = cdiv(
            self.max_model_len,
            self.block_size,
        )

        # LIDU 输出在当前 selection group 内存活：full 层写入，所属 shared
        # 层依次消费，下一 full 层才允许覆盖。层间存在 hidden-state 数据
        # 依赖且当前为单流顺序执行，因此所有 group 可复用同一套固定地址
        # scratch。跨 step 持久状态只保存在 resident_token_pool.cache_slots。
        output_shape = (
            self.max_num_reqs,
            1,
            DSA_LIDU_OUTPUT_CAPACITY,
        )
        self._lidu_topk_index = torch.empty(
            output_shape,
            dtype=torch.int32,
            device=self.device,
        )
        self._lidu_topk_slots = torch.empty_like(self._lidu_topk_index)
        self._lidu_miss_count = torch.empty(
            self.max_num_reqs,
            dtype=torch.int32,
            device=self.device,
        )
        self._lidu_tail_info = torch.empty(
            (self.max_num_reqs, 2),
            dtype=torch.int32,
            device=self.device,
        )
        self._lidu_output_views: dict[
            int,
            DSALightningIndexerOutputs,
        ] = {}
        self._a5_attention_slots: torch.Tensor | None = None
        self._a5_attention_src_ids: torch.Tensor | None = None
        self._a5_per_query_miss_counts: torch.Tensor | None = None
        self._a5_resident_seq_lengths: torch.Tensor | None = None
        if self.packed_c8:
            attention_capacity = (
                DSA_A5_MTP_ATTENTION_CAPACITY if self.max_decode_query_len > 1 else DSA_A5_NOMTP_ATTENTION_CAPACITY
            )
            attention_shape = (
                self.max_num_reqs * self.max_decode_query_len,
                1,
                attention_capacity,
            )
            self._a5_attention_slots = torch.empty(
                attention_shape,
                dtype=torch.int32,
                device=self.device,
            )
            if self.max_decode_query_len > 1:
                self._a5_attention_src_ids = torch.empty(
                    attention_shape,
                    dtype=torch.int32,
                    device=self.device,
                )
                self._a5_per_query_miss_counts = torch.empty(
                    attention_shape[0],
                    dtype=torch.int32,
                    device=self.device,
                )
            self._a5_resident_seq_lengths = torch.empty(
                self.max_num_reqs,
                dtype=torch.int32,
                device=self.device,
            )

        self.resident_positions = torch.empty(
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        self._token_row_modes = torch.empty(
            self.max_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self._token_sparse_budgets = torch.empty_like(self._token_row_modes)
        self._sparse_token_mask = torch.empty(
            self.max_num_tokens,
            dtype=torch.bool,
            device=self.device,
        )

        self.active_dram_block_table = CpuGpuBuffer(
            self.max_num_reqs,
            self.max_logical_blocks,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.active_dram_block_table.np.fill(DSA_DRAM_NULL_BLOCK_ID)
        self.active_dram_block_table.gpu.zero_()
        # 普通 forward 只会新增约 ``max_num_tokens / block_size`` 个
        # 满块；HBM-only prefix 首轮却可能回填每个请求的完整历史。
        # 仍在初始化期一次预分配，热路径只切实际 job 前缀。
        max_dump_jobs = (
            self.max_num_reqs * self.max_logical_blocks
            if self.enable_prefix_caching
            else cdiv(self.max_num_tokens, self.block_size)
            + self.max_num_reqs
        )
        self.dump_src_block_ids = CpuGpuBuffer(
            max_dump_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.dump_dst_block_ids = CpuGpuBuffer(
            max_dump_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.dump_indexer_src_block_ids = CpuGpuBuffer(
            max_dump_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.dump_mtp_src_block_ids = CpuGpuBuffer(
            max_dump_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.dump_src_block_ids.np.fill(0)
        self.dump_indexer_src_block_ids.np.fill(0)
        self.dump_mtp_src_block_ids.np.fill(0)
        self.dump_dst_block_ids.np.fill(DSA_DUMP_NOOP_DST_BLOCK_ID)
        self.dump_src_block_ids.gpu.zero_()
        self.dump_indexer_src_block_ids.gpu.zero_()
        self.dump_mtp_src_block_ids.gpu.zero_()
        self.dump_dst_block_ids.gpu.fill_(DSA_DUMP_NOOP_DST_BLOCK_ID)
        self._dump_pool_indices = np.empty(max_dump_jobs, dtype=np.intp)
        self._dump_logical_indices = np.empty(max_dump_jobs, dtype=np.intp)
        self._dump_source_blocks = np.empty(max_dump_jobs, dtype=np.int32)
        self._dump_indexer_source_blocks = np.empty(
            max_dump_jobs,
            dtype=np.int32,
        )
        self._dump_mtp_source_blocks = np.empty(
            max_dump_jobs,
            dtype=np.int32,
        )
        self._logical_block_indices = np.arange(
            self.max_logical_blocks,
            dtype=np.intp,
        )
        # 满块边界判定每个 forward 都会执行，即使最终没有 dump。
        # 这些逐请求 scratch 固定复用，避免 steady decode 为几次简单的
        # add/floor-divide/subtract 现场创建 NumPy 临时数组。
        self._scheduled_tokens_i64 = np.empty(
            self.max_num_reqs,
            dtype=np.int64,
        )
        self._computed_tokens_i64 = np.empty_like(self._scheduled_tokens_i64)
        self._tokens_after_schedule = np.empty_like(self._scheduled_tokens_i64)
        self._completed_after_blocks = np.empty_like(self._scheduled_tokens_i64)
        self._completed_block_counts = np.empty_like(self._scheduled_tokens_i64)
        self._dump_boundary_mask = np.empty(
            self.max_num_reqs,
            dtype=np.bool_,
        )
        self._durable_full_blocks = np.empty(
            self.max_num_reqs,
            dtype=np.int32,
        )
        self._target_durable_full_blocks = np.empty_like(
            self._durable_full_blocks
        )
        max_restore_jobs = (
            self.max_num_reqs
            * cdiv(self.max_model_len, self.block_size)
            if self.enable_prefix_caching
            else 1
        )
        self.restore_src_dram_block_ids = CpuGpuBuffer(
            max_restore_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.restore_dst_indexer_block_ids = CpuGpuBuffer(
            max_restore_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.restore_dst_resident_block_ids = CpuGpuBuffer(
            max_restore_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.restore_dst_mtp_block_ids = CpuGpuBuffer(
            max_restore_jobs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )
        self.restore_src_dram_block_ids.np.fill(0)
        self.restore_dst_indexer_block_ids.np.fill(0)
        self.restore_dst_resident_block_ids.np.fill(0)
        self.restore_dst_mtp_block_ids.np.fill(0)
        self.restore_src_dram_block_ids.gpu.zero_()
        self.restore_dst_indexer_block_ids.gpu.zero_()
        self.restore_dst_resident_block_ids.gpu.zero_()
        self.restore_dst_mtp_block_ids.gpu.zero_()
        self._restore_pool_indices = np.empty(
            self.max_num_reqs,
            dtype=np.intp,
        )
        self._restore_ready_counts = np.empty(
            self.max_num_reqs,
            dtype=np.int32,
        )

        self.dram_store: DSAWorkerDRAMStore | None = None
        self.active_num_reqs = 0
        self.execution_num_reqs = 0
        self.dump_job_count = 0
        self.dump_launch_count = 0
        self.dump_before_cache_write = False
        self._forward_epoch = 0
        self._layer_dump_epochs = np.full(
            self.num_layers,
            -1,
            dtype=np.int64,
        )
        self._mtp_layer_dump_epochs: dict[int, int] = {}
        self._logged_mtp_dump_timings: set[bool] = set()
        self._graph_capture_row_count = 0
        self._runtime_dummy_row_count = 0
        self._dram_table_row_count = 0
        self._dram_table_signature: tuple[int, int, int] | None = None
        # shared indexer 新鲜度守卫：每个执行 view 先清空，再记录最近一次
        # 跑过 LIDU 的 full 层；shared 层据此拒绝跨 step 的陈旧输出。
        self._selection_source_layer: int | None = None

    def bind_dram_store(self, store: DSAWorkerDRAMStore) -> None:
        if self.dram_store is not None:
            raise RuntimeError("DSA offload runtime DRAM store was bound twice")
        if store.storage_rows != self.resident_token_pool.storage_rows:
            raise RuntimeError(
                "DSA DRAM/resident pool row capacities differ: "
                f"dram={store.storage_rows}, "
                f"resident={self.resident_token_pool.storage_rows}"
            )
        if store.max_logical_blocks != self.max_logical_blocks:
            raise RuntimeError(
                "DSA DRAM/runtime logical widths differ: "
                f"dram={store.max_logical_blocks}, "
                f"runtime={self.max_logical_blocks}"
            )
        self.dram_store = store

    def prepare_prefix_restore_jobs(
        self,
        *,
        restores: Sequence[object],
        state: DSAInputBatchCacheLayout,
    ) -> None:
        """写入冷路径 restore，并提交恢复后的连续 L2 durable 水位。"""

        store = self.dram_store
        if store is None:
            raise RuntimeError("DSA prefix restore requires a bound DRAM store")
        if not restores:
            return
        if self._graph_capture_row_count or self._runtime_dummy_row_count:
            raise RuntimeError("DSA prefix restore cannot execute inside a dummy batch")
        if not self.enable_prefix_caching:
            raise RuntimeError(
                "DSA received prefix restore jobs while prefix caching is "
                "disabled"
            )

        offset = 0
        restore_rows = 0
        capacity = int(self.restore_src_dram_block_ids.np.shape[0])
        for restore in restores:
            request_id = restore.request_id
            pool_index = state.resident_token_pool.get_index(request_id)
            if pool_index is None:
                raise RuntimeError(
                    "DSA prefix restore has no resident pool row: "
                    f"request_id={request_id!r}"
                )
            block_count = len(restore.source_dram_block_ids)
            if not (
                block_count
                == len(restore.destination_indexer_block_ids)
                == len(restore.destination_resident_block_ids)
            ):
                raise RuntimeError(
                    "DSA prefix restore contains inconsistent block columns"
                )
            mtp_block_count = len(restore.destination_mtp_block_ids)
            store_has_mtp = bool(store.mtp_layer_ids)
            if (
                (block_count and (mtp_block_count != 0) != store_has_mtp)
                or mtp_block_count != (block_count if store_has_mtp else 0)
            ):
                raise RuntimeError(
                    "DSA prefix restore MTP column disagrees with the worker "
                    f"bundle: blocks={block_count}, mtp={mtp_block_count}, "
                    f"store_has_mtp={store_has_mtp}"
                )
            logical_start = int(restore.logical_block_start)
            ready_block_count = int(restore.ready_block_count)
            logical_end = logical_start + block_count
            if (
                logical_start < 0
                or ready_block_count < 0
                or logical_end > store.max_logical_blocks
                or ready_block_count > store.max_logical_blocks
            ):
                raise RuntimeError(
                    "DSA prefix restore contains an invalid logical range: "
                    f"request_id={request_id!r}, start={logical_start}, "
                    f"count={block_count}, ready={ready_block_count}"
                )
            if ready_block_count and np.any(
                store.logical_block_table[
                    pool_index,
                    :ready_block_count,
                ]
                == DSA_DRAM_NULL_BLOCK_ID
            ):
                raise RuntimeError(
                    "DSA prefix restore ready watermark crosses an "
                    "unassigned L2 block: "
                    f"request_id={request_id!r}, "
                    f"ready={ready_block_count}"
                )
            end = offset + block_count
            if end > capacity:
                raise RuntimeError(
                    "DSA prefix restore exceeds fixed job capacity: "
                    f"required={end}, capacity={capacity}"
                )
            assigned = store.logical_block_table[
                pool_index,
                logical_start : logical_start + block_count,
            ]
            self.restore_src_dram_block_ids.np[offset:end] = (
                restore.source_dram_block_ids
            )
            self.restore_dst_indexer_block_ids.np[offset:end] = (
                restore.destination_indexer_block_ids
            )
            self.restore_dst_resident_block_ids.np[offset:end] = (
                restore.destination_resident_block_ids
            )
            if store_has_mtp:
                self.restore_dst_mtp_block_ids.np[offset:end] = (
                    restore.destination_mtp_block_ids
                )
            else:
                self.restore_dst_mtp_block_ids.np[offset:end] = 0
            if not np.array_equal(
                assigned,
                self.restore_src_dram_block_ids.np[offset:end],
            ):
                raise RuntimeError(
                    "DSA prefix restore source disagrees with the worker "
                    f"DRAM table: request_id={request_id!r}"
                )
            self._restore_pool_indices[restore_rows] = int(pool_index)
            self._restore_ready_counts[restore_rows] = ready_block_count
            restore_rows += 1
            offset = end

        if offset:
            self.restore_src_dram_block_ids.gpu[:offset].copy_(
                self.restore_src_dram_block_ids.cpu[:offset],
                non_blocking=True,
            )
            self.restore_dst_indexer_block_ids.gpu[:offset].copy_(
                self.restore_dst_indexer_block_ids.cpu[:offset],
                non_blocking=True,
            )
            self.restore_dst_resident_block_ids.gpu[:offset].copy_(
                self.restore_dst_resident_block_ids.cpu[:offset],
                non_blocking=True,
            )
            self.restore_dst_mtp_block_ids.gpu[:offset].copy_(
                self.restore_dst_mtp_block_ids.cpu[:offset],
                non_blocking=True,
            )
        if offset:
            self._execute_prefix_restore_jobs(offset)
        store.commit_durable_full_block_counts(
            pool_indices=self._restore_pool_indices[:restore_rows],
            full_block_counts=self._restore_ready_counts[:restore_rows],
        )
        logger.info(
            "DSA prefix restore submitted: requests=%d blocks=%d "
            "target_layers=%d mtp_layers=%d",
            restore_rows,
            offset,
            self.num_layers,
            len(store.mtp_layer_ids),
        )

    def _execute_prefix_restore_jobs(self, job_count: int) -> None:
        """在 model forward 前恢复所有 target layer，复用当前 NPU stream。"""

        store = self.dram_store
        if store is None:
            raise RuntimeError("DSA prefix restore requires a bound DRAM store")
        source_ids = self.restore_src_dram_block_ids.gpu[:job_count]
        resident_destination_ids = (
            self.restore_dst_resident_block_ids.gpu[:job_count]
        )
        indexer_destination_ids = (
            self.restore_dst_indexer_block_ids.gpu[:job_count]
        )
        mtp_destination_ids = self.restore_dst_mtp_block_ids.gpu[:job_count]
        for layer_id in range(self.num_layers):
            arenas = store.get_layer_arenas(layer_id)
            if len(arenas.resident_source_bytes) != len(
                arenas.resident_dram_bytes
            ):
                raise RuntimeError(
                    "DSA resident HBM/DRAM restore plane count mismatch"
                )
            for dram_bytes, resident_bytes in zip(
                arenas.resident_dram_bytes,
                arenas.resident_source_bytes,
                strict=True,
            ):
                copy_full_cache_plane_bytes(
                    source_bytes=dram_bytes,
                    destination_bytes=resident_bytes,
                    src_block_ids=source_ids,
                    dst_block_ids=resident_destination_ids,
                )
            if len(arenas.indexer_source_bytes) != len(
                arenas.indexer_dram_bytes
            ):
                raise RuntimeError(
                    "DSA Indexer HBM/DRAM restore plane count mismatch"
                )
            for dram_bytes, indexer_bytes in zip(
                arenas.indexer_dram_bytes,
                arenas.indexer_source_bytes,
                strict=True,
            ):
                copy_full_cache_plane_bytes(
                    source_bytes=dram_bytes,
                    destination_bytes=indexer_bytes,
                    src_block_ids=source_ids,
                    dst_block_ids=indexer_destination_ids,
                )
        for layer_id in store.mtp_layer_ids:
            arenas = store.get_mtp_layer_arenas(layer_id)
            if len(arenas.source_bytes) != len(arenas.dram_bytes):
                raise RuntimeError(
                    "DSA MTP HBM/DRAM restore plane count mismatch"
                )
            for dram_bytes, mtp_bytes in zip(
                arenas.dram_bytes,
                arenas.source_bytes,
                strict=True,
            ):
                copy_full_cache_plane_bytes(
                    source_bytes=dram_bytes,
                    destination_bytes=mtp_bytes,
                    src_block_ids=source_ids,
                    dst_block_ids=mtp_destination_ids,
                )

    def release_pool_index(self, pool_index: int) -> None:
        store = self.dram_store
        if store is not None:
            store.release_pool_index(pool_index)

    def _reset_selection_source(self) -> None:
        """进入新的执行 view 前清空上一步的 full 层来源标记。"""
        self._selection_source_layer = None

    def prepare_forward(
        self,
        *,
        input_batch: NPUInputBatch,
        state: DSAInputBatchCacheLayout,
        num_scheduled_tokens: np.ndarray,
        req_indices: torch.Tensor,
        positions: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
        resident_group_id: int,
        indexer_group_id: int,
        mtp_group_id: int | None = None,
        decode_forward: bool,
    ) -> torch.Tensor:
        """刷新本轮统一元数据并返回 resident slot-mapping positions。"""

        if self.dram_store is None:
            raise RuntimeError("DSA offload runtime has no DRAM store")
        if not state.valid or state.row_count != num_reqs:
            raise RuntimeError(
                "DSA InputBatch state is not ready for eager execution: "
                f"valid={state.valid}, rows={state.row_count}, "
                f"num_reqs={num_reqs}"
            )
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(f"DSA token capacity exceeded: {num_tokens} > {self.max_num_tokens}")

        self.active_num_reqs = int(num_reqs)
        self.execution_num_reqs = 0
        self.dump_launch_count = 0
        self.dump_before_cache_write = bool(
            self.max_decode_query_len > 1 and decode_forward
        )
        self._forward_epoch += 1

        self._prepare_dump_plan(
            input_batch=input_batch,
            state=state,
            num_scheduled_tokens=num_scheduled_tokens,
            resident_group_id=resident_group_id,
            indexer_group_id=indexer_group_id,
            mtp_group_id=mtp_group_id,
        )
        self._refresh_candidate_lens(state)
        # 七列共用一个 owner，仅一次 H2D；后续 tensor 均为该 owner 的 view。
        state.copy_to_device(num_reqs)
        dram_table_refreshed = self._refresh_active_dram_table(state)
        self._validate_sparse_dram_rows(
            input_batch=input_batch,
            state=state,
            validate_all_sparse=dram_table_refreshed,
        )

        active_modes = state.row_modes_cpu[:num_reqs]
        if not np.any(active_modes == DSA_ROW_MODE_SPARSE):
            return positions[:num_tokens]

        return self.map_resident_positions(
            positions=positions[:num_tokens],
            row_modes=state.row_modes,
            sparse_budgets=state.sparse_budget_tokens,
            req_indices=req_indices[:num_tokens],
        )

    def map_resident_positions(
        self,
        *,
        positions: torch.Tensor,
        row_modes: torch.Tensor,
        sparse_budgets: torch.Tensor,
        req_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """把真实序列位置映射到 dense 或 parity-tail 逻辑位置。

        target TND 通过 ``req_indices`` 展开请求列；普通 Q=1 与 MTP target
        Q=4 验算复用同一组固定 scratch，双尾公式可化简为
        ``position % (2 * block_size)``。MTP proposer 不调用该函数。
        """

        num_tokens = int(positions.shape[0])
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                "DSA resident position capacity exceeded: "
                f"tokens={num_tokens}, capacity={self.max_num_tokens}"
            )
        token_modes = self._token_row_modes[:num_tokens]
        token_budgets = self._token_sparse_budgets[:num_tokens]
        if req_indices is None:
            token_modes.copy_(row_modes[:num_tokens])
            token_budgets.copy_(sparse_budgets[:num_tokens])
        else:
            torch.index_select(
                row_modes,
                0,
                req_indices,
                out=token_modes,
            )
            torch.index_select(
                sparse_budgets,
                0,
                req_indices,
                out=token_budgets,
            )
        resident_positions = self.resident_positions[:num_tokens]
        torch.remainder(
            positions,
            self.sparse_tail_capacity_tokens,
            out=resident_positions,
        )
        resident_positions.add_(token_budgets)
        sparse_mask = self._sparse_token_mask[:num_tokens]
        torch.eq(
            token_modes,
            DSA_ROW_MODE_SPARSE,
            out=sparse_mask,
        )
        torch.where(
            sparse_mask,
            resident_positions,
            positions,
            out=resident_positions,
        )
        return resident_positions

    def _refresh_candidate_lens(
        self,
        state: DSAInputBatchCacheLayout,
    ) -> None:
        """刷新 LI/KSC 共享的历史候选长度，不创建逐 step 临时数组。

        普通 Q=1 的 DENSE 行候选区是当前完整序列，SPARSE 行只覆盖最后
        一个物理 tail 之前的完整逻辑块。MTP target Q=4 验算时，DENSE 写
        完整可见长度；SPARSE 的该列专门表示已经持久化到 DRAM 的 durable
        prefix D。同一请求的所有 query 共享 [0,D) 的 LI 候选域；LIM 根据
        final length 逐 query 追加因果可见的双尾 [D,V)，不让 tail 参与 LI。
        """

        num_reqs = self.active_num_reqs
        tokens_after = self._tokens_after_schedule[:num_reqs]
        candidate_lens = state.candidate_lens_cpu[:num_reqs]
        if self.max_decode_query_len > 1:
            np.copyto(
                candidate_lens,
                tokens_after,
                casting="unsafe",
            )
            sparse_mask = self._dump_boundary_mask[:num_reqs]
            np.equal(
                state.row_modes_cpu[:num_reqs],
                DSA_ROW_MODE_SPARSE,
                out=sparse_mask,
            )
            completed_before_tail = self._completed_after_blocks[:num_reqs]
            np.multiply(
                self._durable_full_blocks[:num_reqs],
                self.block_size,
                out=completed_before_tail,
            )
            np.copyto(
                candidate_lens,
                completed_before_tail,
                where=sparse_mask,
                casting="unsafe",
            )
            return
        np.copyto(candidate_lens, tokens_after, casting="unsafe")

        sparse_mask = self._dump_boundary_mask[:num_reqs]
        np.equal(
            state.row_modes_cpu[:num_reqs],
            DSA_ROW_MODE_SPARSE,
            out=sparse_mask,
        )
        completed_before_tail = self._completed_after_blocks[:num_reqs]
        np.subtract(tokens_after, 1, out=completed_before_tail)
        np.floor_divide(
            completed_before_tail,
            self.block_size,
            out=completed_before_tail,
        )
        np.multiply(
            completed_before_tail,
            self.block_size,
            out=completed_before_tail,
        )
        np.copyto(
            candidate_lens,
            completed_before_tail,
            where=sparse_mask,
            casting="unsafe",
        )

    @property
    def graph_capture_row_count(self) -> int:
        return self._graph_capture_row_count

    @property
    def runtime_dummy_row_count(self) -> int:
        return self._runtime_dummy_row_count

    def prepare_execution_view(
        self,
        *,
        active_num_reqs: int,
        graph_row_count: int | None,
    ) -> int:
        """把 active-prefix 绑定成 eager 或 FULL-graph 的执行 view。

        eager 仅复制紧凑 dump jobs；graph 固定复制 captured-row 宽度，未
        使用行以 ``dst=-1`` 空转。DRAM table 和请求列仍是同一固定 owner
        的前缀，不创建 graph 专属副本。
        """

        if self._graph_capture_row_count or self._runtime_dummy_row_count:
            raise RuntimeError(
                "DSA real execution cannot reuse an active dummy state"
            )
        active_num_reqs = int(active_num_reqs)
        if active_num_reqs != self.active_num_reqs:
            raise RuntimeError(
                "DSA runtime active rows changed before execution view "
                f"was finalized: prepared={self.active_num_reqs}, "
                f"requested={active_num_reqs}"
            )

        if graph_row_count is None:
            execution_num_reqs = active_num_reqs
            launch_count = self.dump_job_count
        else:
            execution_num_reqs = int(graph_row_count)
            if not (active_num_reqs <= execution_num_reqs <= self.max_num_reqs):
                raise RuntimeError(
                    "DSA graph rows must cover active rows and fit runtime "
                    f"capacity: active={active_num_reqs}, "
                    f"graph={execution_num_reqs}, "
                    f"capacity={self.max_num_reqs}"
                )
            if self.dump_job_count > active_num_reqs:
                raise RuntimeError(
                    "DSA single-token graph step produced more dump jobs "
                    f"than active rows: jobs={self.dump_job_count}, "
                    f"rows={active_num_reqs}"
                )
            launch_count = execution_num_reqs
            if self.dump_job_count < launch_count:
                tail = slice(self.dump_job_count, launch_count)
                self.dump_src_block_ids.np[tail] = 0
                self.dump_indexer_src_block_ids.np[tail] = 0
                if self.mtp_durable_offload:
                    self.dump_mtp_src_block_ids.np[tail] = 0
                self.dump_dst_block_ids.np[tail] = DSA_DUMP_NOOP_DST_BLOCK_ID

        if launch_count:
            self.dump_src_block_ids.gpu[:launch_count].copy_(
                self.dump_src_block_ids.cpu[:launch_count],
                non_blocking=True,
            )
            self.dump_indexer_src_block_ids.gpu[:launch_count].copy_(
                self.dump_indexer_src_block_ids.cpu[:launch_count],
                non_blocking=True,
            )
            if self.mtp_durable_offload:
                self.dump_mtp_src_block_ids.gpu[:launch_count].copy_(
                    self.dump_mtp_src_block_ids.cpu[:launch_count],
                    non_blocking=True,
                )
            self.dump_dst_block_ids.gpu[:launch_count].copy_(
                self.dump_dst_block_ids.cpu[:launch_count],
                non_blocking=True,
            )
        self.execution_num_reqs = execution_num_reqs
        self.dump_launch_count = int(launch_count)
        self._reset_selection_source()
        return execution_num_reqs

    def dummy_query_lengths(self, *, row_count: int | None, num_tokens: int, query_len: int) -> np.ndarray:
        """按最终 native descriptor 构造 uniform dummy 行，不按原 batch 倍增。"""

        if not 0 < query_len <= self.max_decode_query_len:
            raise ValueError("DSA dummy query shape is outside runtime capacity")
        if row_count is None:
            row_count = num_tokens // query_len
        if not 0 < row_count <= self.max_num_reqs:
            raise ValueError("DSA dummy query shape is outside runtime capacity")
        if row_count * query_len != num_tokens:
            raise ValueError("DSA uniform dummy descriptor has inconsistent row/token counts")
        return np.full(row_count, query_len, dtype=np.int32)

    def prepare_graph_capture(self, *, row_count: int) -> None:
        """为原生 dummy-run 安装固定地址 DRAM/dump 输入。"""

        if self.dram_store is None:
            raise RuntimeError("DSA graph capture requires an initialized DRAM store")
        row_count = int(row_count)
        if self._graph_capture_row_count or self._runtime_dummy_row_count:
            raise RuntimeError("DSA runtime dummy state was installed twice")
        if not 0 < row_count <= self.max_num_reqs:
            raise ValueError(
                "DSA runtime graph-capture row count is outside capacity: "
                f"rows={row_count}, capacity={self.max_num_reqs}"
            )

        self._graph_capture_row_count = row_count
        try:
            self.active_dram_block_table.np[:row_count].fill(DSA_DRAM_NULL_BLOCK_ID)
            self.active_dram_block_table.gpu[:row_count].zero_()
            self.dump_src_block_ids.np[:row_count] = 0
            self.dump_indexer_src_block_ids.np[:row_count] = 0
            if self.mtp_durable_offload:
                self.dump_mtp_src_block_ids.np[:row_count] = 0
            self.dump_dst_block_ids.np[:row_count] = DSA_DUMP_NOOP_DST_BLOCK_ID
            self.dump_src_block_ids.gpu[:row_count].zero_()
            self.dump_indexer_src_block_ids.gpu[:row_count].zero_()
            if self.mtp_durable_offload:
                self.dump_mtp_src_block_ids.gpu[:row_count].zero_()
            self.dump_dst_block_ids.gpu[:row_count].fill_(DSA_DUMP_NOOP_DST_BLOCK_ID)
            self.active_num_reqs = row_count
            self.execution_num_reqs = row_count
            self.dump_job_count = 0
            self.dump_launch_count = row_count
            self.dump_before_cache_write = self.max_decode_query_len > 1
            self._forward_epoch += 1
            self._reset_selection_source()
        except Exception:
            try:
                self.restore_after_graph_capture()
            except Exception:
                # 保留首次安装失败作为根因，并至少解除 host 侧 installed 状态。
                self.active_num_reqs = 0
                self.execution_num_reqs = 0
                self.dump_job_count = 0
                self.dump_launch_count = 0
                self._graph_capture_row_count = 0
            raise

    def restore_after_graph_capture(self) -> None:
        """恢复 dummy-run 之外的空 runtime 状态。"""

        if not self._graph_capture_row_count:
            return
        self.active_num_reqs = 0
        self.execution_num_reqs = 0
        self.dump_job_count = 0
        self.dump_launch_count = 0
        self.dump_before_cache_write = False
        self._graph_capture_row_count = 0
        self._dram_table_row_count = 0
        self._dram_table_signature = None

    def prepare_runtime_dummy(self, *, row_count: int) -> None:
        """Install fixed-address no-op IO inputs for an idle graph replay."""

        if self.dram_store is None:
            raise RuntimeError(
                "DSA runtime dummy requires an initialized DRAM store"
            )
        row_count = int(row_count)
        if self._graph_capture_row_count or self._runtime_dummy_row_count:
            raise RuntimeError("DSA runtime dummy state was installed twice")
        if not 0 < row_count <= self.max_num_reqs:
            raise ValueError(
                "DSA runtime-dummy row count is outside runtime capacity: "
                f"rows={row_count}, capacity={self.max_num_reqs}"
            )

        self._runtime_dummy_row_count = row_count
        try:
            self.active_dram_block_table.np[:row_count].fill(
                DSA_DRAM_NULL_BLOCK_ID
            )
            self.active_dram_block_table.gpu[:row_count].zero_()
            self.dump_src_block_ids.np[:row_count] = 0
            self.dump_indexer_src_block_ids.np[:row_count] = 0
            if self.mtp_durable_offload:
                self.dump_mtp_src_block_ids.np[:row_count] = 0
            self.dump_dst_block_ids.np[:row_count] = (
                DSA_DUMP_NOOP_DST_BLOCK_ID
            )
            self.dump_src_block_ids.gpu[:row_count].zero_()
            self.dump_indexer_src_block_ids.gpu[:row_count].zero_()
            if self.mtp_durable_offload:
                self.dump_mtp_src_block_ids.gpu[:row_count].zero_()
            self.dump_dst_block_ids.gpu[:row_count].fill_(
                DSA_DUMP_NOOP_DST_BLOCK_ID
            )
            self.active_num_reqs = row_count
            self.execution_num_reqs = row_count
            self.dump_job_count = 0
            self.dump_launch_count = row_count
            self.dump_before_cache_write = self.max_decode_query_len > 1
            self._reset_selection_source()
        except Exception:
            self._runtime_dummy_row_count = 0
            self.active_num_reqs = 0
            self.execution_num_reqs = 0
            self.dump_job_count = 0
            self.dump_launch_count = 0
            raise

    def restore_after_runtime_dummy(self) -> None:
        """Release the transient runtime-dummy execution view."""

        if not self._runtime_dummy_row_count:
            return
        self.active_num_reqs = 0
        self.execution_num_reqs = 0
        self.dump_job_count = 0
        self.dump_launch_count = 0
        self.dump_before_cache_write = False
        self._runtime_dummy_row_count = 0
        self._dram_table_row_count = 0
        self._dram_table_signature = None

    def _prepare_dump_plan(
        self,
        *,
        input_batch: NPUInputBatch,
        state: DSAInputBatchCacheLayout,
        num_scheduled_tokens: np.ndarray,
        resident_group_id: int,
        indexer_group_id: int,
        mtp_group_id: int | None,
    ) -> None:
        """按 durable 水位差构造本轮紧凑 dump jobs。

        普通 decode/prefill 在本层 attention 后持久化本轮新满块；MTP
        target 验算只持久化 forward 开始时已经由 scheduler 确认的前缀，
        且必须在本层写入新 KV 前执行，防止 parity tail 被复用后覆盖源块。
        """

        store = self.dram_store
        assert store is not None
        if (mtp_group_id is not None) != self.mtp_durable_offload:
            raise RuntimeError(
                "DSA MTP durable runtime/group capability disagrees: "
                f"runtime={self.mtp_durable_offload}, group={mtp_group_id}"
            )
        num_reqs = self.active_num_reqs
        scheduled = self._scheduled_tokens_i64[:num_reqs]
        computed = self._computed_tokens_i64[:num_reqs]
        tokens_after = self._tokens_after_schedule[:num_reqs]
        durable_before = self._durable_full_blocks[:num_reqs]
        durable_target = self._target_durable_full_blocks[:num_reqs]
        completed_counts = self._completed_block_counts[:num_reqs]
        boundary_mask = self._dump_boundary_mask[:num_reqs]

        np.copyto(
            scheduled,
            num_scheduled_tokens[:num_reqs],
            casting="unsafe",
        )
        np.copyto(
            computed,
            input_batch.num_computed_tokens_cpu[:num_reqs],
            casting="unsafe",
        )
        np.add(computed, scheduled, out=tokens_after)
        pool_indices = state.resident_pool_indices_cpu[:num_reqs]
        store.gather_durable_full_block_counts(
            pool_indices=pool_indices,
            output=durable_before,
        )
        durable_source_tokens = (
            computed if self.dump_before_cache_write else tokens_after
        )
        np.floor_divide(
            durable_source_tokens,
            self.block_size,
            out=durable_target,
        )
        np.subtract(
            durable_target,
            durable_before,
            out=completed_counts,
        )
        if np.any(completed_counts < 0):
            raise RuntimeError(
                "DSA durable full-block watermark moved backwards"
            )
        np.greater(completed_counts, 0, out=boundary_mask)
        if not np.any(boundary_mask):
            self.dump_job_count = 0
            return
        boundary_rows = np.flatnonzero(boundary_mask)

        resident_table = input_batch.block_table[int(resident_group_id)]
        resident_blocks = resident_table.get_numpy_array()
        resident_row_widths = resident_table.num_blocks_per_row
        indexer_table = input_batch.block_table[int(indexer_group_id)]
        indexer_blocks = indexer_table.get_numpy_array()
        indexer_row_widths = indexer_table.num_blocks_per_row
        mtp_blocks: np.ndarray | None = None
        mtp_row_widths: np.ndarray | None = None
        if mtp_group_id is not None:
            mtp_table = input_batch.block_table[int(mtp_group_id)]
            mtp_blocks = mtp_table.get_numpy_array()
            mtp_row_widths = mtp_table.num_blocks_per_row
        job_count = 0
        for row_value in boundary_rows:
            row = int(row_value)
            count = int(completed_counts[row])
            logical_start = int(durable_before[row])
            logical_end = logical_start + count
            next_job_count = job_count + count
            if next_job_count > self._dump_pool_indices.size:
                raise RuntimeError(
                    "DSA full-block dump job capacity exceeded: "
                    f"required={next_job_count}, "
                    f"capacity={self._dump_pool_indices.size}"
                )

            jobs = slice(job_count, next_job_count)
            self._dump_pool_indices[jobs] = state.resident_pool_indices_cpu[row]
            self._dump_logical_indices[jobs] = self._logical_block_indices[logical_start:logical_end]
            if logical_end > int(indexer_row_widths[row]):
                raise RuntimeError(
                    "DSA complete dump references an unallocated Indexer "
                    f"block: row={row}, logical_end={logical_end}, "
                    f"indexer_blocks={int(indexer_row_widths[row])}"
                )
            self._dump_indexer_source_blocks[jobs] = indexer_blocks[
                row,
                logical_start:logical_end,
            ]
            if mtp_blocks is not None and mtp_row_widths is not None:
                if logical_end > int(mtp_row_widths[row]):
                    raise RuntimeError(
                        "DSA complete dump references an unallocated MTP "
                        f"block: row={row}, logical_end={logical_end}, "
                        f"mtp_blocks={int(mtp_row_widths[row])}"
                    )
                self._dump_mtp_source_blocks[jobs] = mtp_blocks[
                    row,
                    logical_start:logical_end,
                ]
            else:
                self._dump_mtp_source_blocks[jobs] = 0
            if state.row_modes_cpu[row] == DSA_ROW_MODE_SPARSE:
                if count > 1:
                    raise RuntimeError(
                        "DSA sparse decode advanced by more than one durable "
                        f"full block: row={row}, count={count}"
                    )
                budget_blocks = (
                    int(state.sparse_budget_tokens_cpu[row])
                    // self.block_size
                )
                for local_job, logical_block in enumerate(
                    range(logical_start, logical_end)
                ):
                    tail_column = (
                        budget_blocks
                        + logical_block % self.sparse_tail_block_count
                    )
                    if tail_column >= int(resident_row_widths[row]):
                        raise RuntimeError(
                            "DSA sparse dump tail column is unallocated: "
                            f"row={row}, column={tail_column}, "
                            f"resident_blocks={int(resident_row_widths[row])}"
                        )
                    self._dump_source_blocks[job_count + local_job] = (
                        resident_blocks[row, tail_column]
                    )
            else:
                if logical_end > int(resident_row_widths[row]):
                    raise RuntimeError(
                        "DSA dense dump references an unallocated resident "
                        f"block: row={row}, logical_end={logical_end}, "
                        f"resident_blocks={int(resident_row_widths[row])}"
                    )
                self._dump_source_blocks[jobs] = resident_blocks[
                    row,
                    logical_start:logical_end,
                ]
            job_count = next_job_count

        if job_count == 0:
            self.dump_job_count = 0
            return

        physical_block_ids = store.resolve_blocks(
            pool_indices=self._dump_pool_indices[:job_count],
            logical_block_indices=self._dump_logical_indices[:job_count],
        )
        self.dump_src_block_ids.np[:job_count] = (
            self._dump_source_blocks[:job_count]
        )
        self.dump_indexer_src_block_ids.np[:job_count] = (
            self._dump_indexer_source_blocks[:job_count]
        )
        self.dump_mtp_src_block_ids.np[:job_count] = (
            self._dump_mtp_source_blocks[:job_count]
        )
        self.dump_dst_block_ids.np[:job_count] = (
            physical_block_ids
        )
        store.commit_durable_full_block_counts(
            pool_indices=pool_indices[boundary_rows],
            full_block_counts=durable_target[boundary_rows],
        )
        np.copyto(
            durable_before,
            durable_target,
            where=boundary_mask,
            casting="unsafe",
        )
        self.dump_job_count = job_count

    def claim_layer_dump(
        self,
        *,
        layer_id: int,
        before_cache_write: bool,
    ) -> bool:
        """同一 outer forward 内让每个物理层只发射一次 dump。"""

        layer_id = int(layer_id)
        if bool(before_cache_write) != self.dump_before_cache_write:
            return False
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(
                f"DSA dump layer {layer_id} is outside runtime capacity"
            )
        if self._layer_dump_epochs[layer_id] == self._forward_epoch:
            return False
        self._layer_dump_epochs[layer_id] = self._forward_epoch
        return True

    def claim_mtp_layer_dump(
        self,
        *,
        layer_id: int,
        before_cache_write: bool,
    ) -> bool:
        """同一 outer forward 内让每个 MTP 层只发射一次 full dump。"""

        layer_id = int(layer_id)
        if bool(before_cache_write) != self.dump_before_cache_write:
            return False
        if self.dram_store is None or layer_id not in self.dram_store.mtp_layer_ids:
            raise IndexError(
                f"DSA MTP dump layer {layer_id} has no bound DRAM arena"
            )
        if self._mtp_layer_dump_epochs.get(layer_id) == self._forward_epoch:
            return False
        self._mtp_layer_dump_epochs[layer_id] = self._forward_epoch
        return True

    def validate_mtp_dump_completion(self) -> None:
        """Reject publishing a durable bundle with stale MTP cache planes.

        Target Indexer/MLA dumps run inside the target forward, while the MTP
        cache dump runs later inside the eager proposer.  The L3 connector is
        finalized only after both model paths.  This guard makes that ordering
        an explicit correctness contract instead of allowing an incomplete
        L2 bundle to be published silently.
        """

        if not self.mtp_durable_offload or self.dump_job_count == 0:
            return
        store = self.dram_store
        if store is None:
            raise RuntimeError(
                "DSA MTP dump completion check has no bound DRAM store"
            )
        expected_layers = store.mtp_layer_ids
        if not expected_layers:
            raise RuntimeError(
                "DSA MTP durable offload has no bound MTP cache layers"
            )
        completed_layers = tuple(
            layer_id
            for layer_id in expected_layers
            if self._mtp_layer_dump_epochs.get(layer_id)
            == self._forward_epoch
        )
        if completed_layers != expected_layers:
            missing_layers = tuple(
                layer_id
                for layer_id in expected_layers
                if layer_id not in completed_layers
            )
            raise RuntimeError(
                "DSA cannot publish a durable MTP bundle before every MTP "
                "cache layer is copied: "
                f"forward_epoch={self._forward_epoch}, "
                f"dump_jobs={self.dump_job_count}, "
                f"before_cache_write={self.dump_before_cache_write}, "
                f"expected_layers={expected_layers}, "
                f"completed_layers={completed_layers}, "
                f"missing_layers={missing_layers}, "
                "first_src_block="
                f"{int(self.dump_mtp_src_block_ids.np[0])}, "
                "first_dst_block="
                f"{int(self.dump_dst_block_ids.np[0])}"
            )
        timing = bool(self.dump_before_cache_write)
        if timing not in self._logged_mtp_dump_timings:
            last_job = self.dump_job_count - 1
            logger.info(
                "DSA MTP durable dump completed: epoch=%d jobs=%d "
                "layers=%s timing=%s src_blocks=(%d,%d) "
                "dst_blocks=(%d,%d)",
                self._forward_epoch,
                self.dump_job_count,
                expected_layers,
                "prewrite" if timing else "postwrite",
                int(self.dump_mtp_src_block_ids.np[0]),
                int(self.dump_mtp_src_block_ids.np[last_job]),
                int(self.dump_dst_block_ids.np[0]),
                int(self.dump_dst_block_ids.np[last_job]),
            )
            self._logged_mtp_dump_timings.add(timing)

    def _refresh_active_dram_table(
        self,
        state: DSAInputBatchCacheLayout,
    ) -> bool:
        store = self.dram_store
        assert store is not None
        signature = (
            int(state.mapping_version),
            int(store.table_version),
            self.active_num_reqs,
        )
        if signature == self._dram_table_signature:
            return False
        previous_row_count = self._dram_table_row_count
        active_cpu = self.active_dram_block_table.np[: self.active_num_reqs]
        store.gather_rows(
            pool_indices=state.resident_pool_indices_cpu[: self.active_num_reqs],
            output=active_cpu,
        )
        if self.active_num_reqs < previous_row_count:
            self.active_dram_block_table.np[self.active_num_reqs : previous_row_count].fill(DSA_DRAM_NULL_BLOCK_ID)
        copy_row_count = max(
            self.active_num_reqs,
            previous_row_count,
        )
        self.active_dram_block_table.gpu[:copy_row_count].copy_(
            self.active_dram_block_table.cpu[:copy_row_count],
            non_blocking=True,
        )
        self._dram_table_row_count = self.active_num_reqs
        self._dram_table_signature = signature
        return True

    def _validate_sparse_dram_rows(
        self,
        *,
        input_batch: NPUInputBatch,
        state: DSAInputBatchCacheLayout,
        validate_all_sparse: bool,
    ) -> None:
        """在低频布局边界拒绝不完整的 sparse DRAM 映射。

        KSC 的 block 0 是合法可寻址的空 arena，因此缺失逻辑映射若不在
        host 侧拦住，会静默把零 payload 写入 resident HBM。ENTER 每个请求
        只检查一次；稳定 SPARSE 仅在 pool-row 或 DRAM table 版本变化时
        检查，不增加 steady decode 的逐 step 扫描。
        """

        num_reqs = self.active_num_reqs
        stages = state.stages_cpu[:num_reqs]
        if validate_all_sparse:
            rows = np.flatnonzero(stages >= int(DSARequestCacheStage.ENTER_SPARSE_DECODE))
        else:
            rows = np.flatnonzero(stages == int(DSARequestCacheStage.ENTER_SPARSE_DECODE))
        if rows.size == 0:
            return

        dram_table = self.active_dram_block_table.np
        for row_value in rows:
            row = int(row_value)
            # candidate 是 KSC 可读的完整块前缀。MTP 使用共同 durable D，
            # 未 durable 的因果双尾仍在 HBM，不因尾块已填满而纳入此范围。
            required_blocks = (
                int(state.candidate_lens_cpu[row]) // self.block_size
            )
            if required_blocks == 0:
                continue
            missing = np.flatnonzero(dram_table[row, :required_blocks] == DSA_DRAM_NULL_BLOCK_ID)
            if missing.size:
                request_id = input_batch.req_ids[row]
                raise RuntimeError(
                    "DSA sparse decode has an incomplete DRAM block table: "
                    f"request_id={request_id!r}, batch_row={row}, "
                    "resident_pool_row="
                    f"{int(state.resident_pool_indices_cpu[row])}, "
                    f"required_blocks={required_blocks}, "
                    f"first_missing_logical_block={int(missing[0])}. "
                    "Refusing to let KSC read the null DRAM block."
                )

    def get_lidu_outputs(
        self,
        *,
        num_reqs: int,
    ) -> DSALightningIndexerOutputs:
        num_reqs = int(num_reqs)
        view = self._lidu_output_views.get(num_reqs)
        if view is None:
            if not 0 < num_reqs <= self.max_num_reqs:
                raise RuntimeError(f"DSA LIDU row count {num_reqs} is outside capacity")
            view = DSALightningIndexerOutputs(
                topk_index=self._lidu_topk_index[:num_reqs],
                topk_slots=self._lidu_topk_slots[:num_reqs],
                miss_count=self._lidu_miss_count[:num_reqs],
                tail_info=self._lidu_tail_info[:num_reqs],
            )
            self._lidu_output_views[num_reqs] = view
        return view


@dataclass(frozen=True)
class DSAMTPFullCacheOffloadContext:
    """MTP proposer 的 full-cache 满块持久化钩子。

    它不改变 MTP attention 或 cache slot 语义，只在既有 durable dump
    boundary 上复制原生 BF16 planes。decode 在第一轮 proposer 写前复制，
    prefill 在 cache 写后复制；同一 outer step 的后续 draft 调用自动跳过。
    """

    layer_id: int
    runtime: DSAOffloadRuntime

    def dump_full_blocks(self, *, before_cache_write: bool) -> None:
        # GLM-5.x MTP proposer currently remains eager. Use the compact real
        # job count instead of target FULL-graph padding so steady decode does
        # not launch three no-op byte copies on every draft step.
        job_count = self.runtime.dump_job_count
        if job_count == 0:
            return
        if not self.runtime.claim_mtp_layer_dump(
            layer_id=self.layer_id,
            before_cache_write=before_cache_write,
        ):
            return
        store = self.runtime.dram_store
        if store is None:
            raise RuntimeError("DSA MTP layer has no bound DRAM store")
        arenas = store.get_mtp_layer_arenas(self.layer_id)
        if len(arenas.source_bytes) != len(arenas.dram_bytes):
            raise RuntimeError(
                "DSA MTP source/DRAM plane count mismatch"
            )
        source_ids = self.runtime.dump_mtp_src_block_ids.gpu[:job_count]
        destination_ids = self.runtime.dump_dst_block_ids.gpu[:job_count]
        for source_bytes, dram_bytes in zip(
            arenas.source_bytes,
            arenas.dram_bytes,
            strict=True,
        ):
            dump_full_cache_plane_bytes(
                source_bytes=source_bytes,
                dram_bytes=dram_bytes,
                src_block_ids=source_ids,
                dst_block_ids=destination_ids,
            )


@dataclass(frozen=True)
class DSALayerOffloadContext:
    """绑定到一个 ``AscendSFAImpl`` 的逐层稳定资源。

    full indexer 层持有本层 ``indexer_cache``（``selection_source_layer_id``
    为 None）；GLM-5.2 的 shared indexer 层 ``indexer_cache`` 为 None，
    用 ``selection_source_layer_id`` 指向所属 full 层，复用其 LIDU 输出。

    ``selection_state_id`` 只为 full 层分配，用于索引紧凑的 LIDU
    ``cache_slots``；``layer_id`` 始终索引全部 target resident cache/DRAM
    arena。MTP draft 层使用独立的原生 BF16 full cache，不绑定本上下文。
    """

    layer_id: int
    indexer_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None
    runtime: DSAOffloadRuntime
    selection_state_id: int | None = None
    selection_source_layer_id: int | None = None
    packed_c8: bool = False

    def execute_decode_selection(
        self,
        *,
        query: torch.Tensor,
        weights: torch.Tensor,
        row_modes: torch.Tensor,
        resident_pool_indices: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        indexer_block_table: torch.Tensor,
        resident_cache: tuple[torch.Tensor, ...],
        resident_block_table: torch.Tensor,
        dram_block_table: torch.Tensor,
        candidate_lens: torch.Tensor | None = None,
        query_dequant_scale: torch.Tensor | None = None,
        query_shape: tuple[int, ...] | None = None,
    ) -> DSAOffloadSelectionOutput:
        if self.indexer_cache is None or self.selection_state_id is None:
            raise RuntimeError(f"DSA shared-indexer layer must not run LIDU selection: layer_id={self.layer_id}")
        num_reqs = int(actual_seq_lengths_key.shape[0])
        num_query_rows = int(query_shape[0]) if query_shape is not None else num_reqs
        outputs = self.runtime.get_lidu_outputs(
            num_reqs=num_reqs,
        )
        if self.packed_c8:
            if not isinstance(self.indexer_cache, tuple) or len(self.indexer_cache) != 2 or len(resident_cache) != 1:
                raise RuntimeError("DSA A5 C8 context requires packed resident cache and Indexer key/scale caches")
            if candidate_lens is None or query_dequant_scale is None or query_shape is None:
                raise RuntimeError("DSA A5 C8 decode metadata is incomplete")
            indexer_key, indexer_scale = self.indexer_cache
            attention_slots_buffer = self.runtime._a5_attention_slots
            attention_src_ids_buffer = self.runtime._a5_attention_src_ids
            per_query_miss_counts_buffer = self.runtime._a5_per_query_miss_counts
            resident_seq_lengths_buffer = self.runtime._a5_resident_seq_lengths
            if attention_slots_buffer is None or resident_seq_lengths_buffer is None:
                raise RuntimeError("DSA A5 C8 context has no preallocated selection scratch")
            if num_query_rows > int(attention_slots_buffer.shape[0]):
                raise RuntimeError(
                    "DSA A5 query-row capacity exceeded: "
                    f"required={num_query_rows}, "
                    f"capacity={int(attention_slots_buffer.shape[0])}"
                )
            attention_slots = attention_slots_buffer[:num_query_rows]
            resident_seq_lengths = resident_seq_lengths_buffer[:num_reqs]
            cache_slots = self.runtime.resident_token_pool.get_cache_slots(self.selection_state_id)
            if self.runtime.max_decode_query_len > 1:
                if attention_src_ids_buffer is None or per_query_miss_counts_buffer is None:
                    raise RuntimeError("DSA A5 MTP context has no per-query selection scratch")
                a5_lightning_indexer_decode_update_mtp_c8(
                    index_weights=weights,
                    query=query.view(query_shape),
                    query_dequant_scale=query_dequant_scale.view(query_shape[:-1]),
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    index_key_cache=indexer_key,
                    index_key_dequant_scale=indexer_scale,
                    index_block_table=indexer_block_table,
                    candidate_lens=candidate_lens,
                    final_seq_lengths_kv=actual_seq_lengths_key,
                    row_modes=row_modes,
                    req_pool_entries=resident_pool_indices,
                    cache_slots=cache_slots,
                    attention_slots=attention_slots,
                    attention_src_ids=attention_src_ids_buffer[:num_query_rows],
                    per_query_miss_counts=per_query_miss_counts_buffer[:num_query_rows],
                    resident_seq_lengths=resident_seq_lengths,
                    outputs=outputs,
                )
            else:
                a5_lightning_indexer_decode_update_c8(
                    index_weights=weights,
                    query=query.view(query_shape),
                    query_dequant_scale=query_dequant_scale.view(query_shape[:-1]),
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    index_key_cache=indexer_key,
                    index_key_dequant_scale=indexer_scale,
                    index_block_table=indexer_block_table,
                    candidate_lens=candidate_lens,
                    final_seq_lengths_kv=actual_seq_lengths_key,
                    row_modes=row_modes,
                    req_pool_entries=resident_pool_indices,
                    cache_slots=cache_slots,
                    attention_slots=attention_slots,
                    resident_seq_lengths=resident_seq_lengths,
                    outputs=outputs,
                )
            store = self.runtime.dram_store
            if store is None:
                raise RuntimeError("DSA layer has no bound DRAM store")
            packed_arena = store.get_layer_arenas(self.layer_id).packed
            if packed_arena is None:
                raise RuntimeError("DSA A5 layer has no packed DRAM arena")
            a5_kvcache_scatter_copy_c8(
                resident_packed_cache=resident_cache[0],
                dram_packed_arena=packed_arena,
                resident_block_table=resident_block_table,
                dram_block_table=dram_block_table,
                source_token_ids=outputs.topk_index,
                destination_slots=outputs.topk_slots,
                copy_counts=outputs.miss_count,
            )
            # KSC 已发射后再公布 full 层来源，shared 层只会消费一份
            # 同时包含 copy plan 和 attention metadata 的完整融合输出。
            self.runtime._selection_source_layer = self.layer_id
            return DSAOffloadSelectionOutput(
                sparse_indices=attention_slots,
                tail_info=outputs.tail_info,
                resident_seq_lengths=resident_seq_lengths,
            )

        if not isinstance(self.indexer_cache, torch.Tensor):
            raise RuntimeError("DSA bf16 Indexer cache must be one tensor")
        if len(resident_cache) < 2:
            raise RuntimeError("DSA bf16 resident cache must contain NOPE and ROPE planes")
        lightning_indexer_decode_update(
            query=query,
            key=self.indexer_cache,
            weights=weights,
            req_pool_entries=resident_pool_indices,
            cache_slots=self.runtime.resident_token_pool.get_cache_slots(self.selection_state_id),
            row_modes=row_modes,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=indexer_block_table,
            outputs=outputs,
        )
        # 记录本步 full 层选择来源，供其 shared 跟随层复用前校验。
        self.runtime._selection_source_layer = self.layer_id
        store = self.runtime.dram_store
        if store is None:
            raise RuntimeError("DSA layer has no bound DRAM store")
        arenas = store.get_layer_arenas(self.layer_id)
        kvcache_scatter_copy(
            resident_nope_cache=resident_cache[0],
            resident_rope_cache=resident_cache[1],
            dram_nope_arena=arenas.nope,
            dram_rope_arena=arenas.rope,
            resident_block_table=resident_block_table,
            dram_block_table=dram_block_table,
            src_token_ids=outputs.topk_index,
            dst_slots=outputs.topk_slots,
            copy_counts=outputs.miss_count,
        )
        return DSAOffloadSelectionOutput(
            sparse_indices=outputs.topk_slots,
            tail_info=outputs.tail_info,
        )

    def execute_shared_decode_selection(
        self,
        *,
        resident_cache: tuple[torch.Tensor, ...],
        resident_block_table: torch.Tensor,
        dram_block_table: torch.Tensor,
        num_reqs: int,
        num_query_rows: int | None = None,
    ) -> DSAOffloadSelectionOutput:
        """shared indexer 层的 decode 选择：复用所属 full 层的 LIDU 输出。

        不跑 LIDU（shared 层无 indexer 权重），只对本层 resident/DRAM
        arena 跑 KSC。新鲜度守卫确保所属 full 层在本步已先跑过 LIDU。
        """
        if self.indexer_cache is not None or self.selection_source_layer_id is None:
            raise RuntimeError(f"DSA full-indexer layer must use execute_decode_selection: layer_id={self.layer_id}")
        if self.runtime._selection_source_layer != self.selection_source_layer_id:
            raise RuntimeError(
                "DSA shared-indexer selection source is stale: "
                f"layer_id={self.layer_id}, "
                f"expected_source={self.selection_source_layer_id}, "
                f"current_source={self.runtime._selection_source_layer}"
            )
        if num_query_rows is None:
            num_query_rows = int(num_reqs)
        outputs = self.runtime.get_lidu_outputs(num_reqs=int(num_reqs))
        store = self.runtime.dram_store
        if store is None:
            raise RuntimeError("DSA layer has no bound DRAM store")
        arenas = store.get_layer_arenas(self.layer_id)
        if self.packed_c8:
            if len(resident_cache) != 1 or arenas.packed is None:
                raise RuntimeError("DSA A5 shared layer requires one packed resident cache and packed DRAM arena")
            attention_slots_buffer = self.runtime._a5_attention_slots
            resident_seq_lengths_buffer = self.runtime._a5_resident_seq_lengths
            if attention_slots_buffer is None or resident_seq_lengths_buffer is None:
                raise RuntimeError("DSA A5 C8 context has no preallocated selection scratch")
            if num_query_rows > int(attention_slots_buffer.shape[0]):
                raise RuntimeError(
                    "DSA A5 shared query-row capacity exceeded: "
                    f"required={num_query_rows}, "
                    f"capacity={int(attention_slots_buffer.shape[0])}"
                )
            attention_slots = attention_slots_buffer[:num_query_rows]
            resident_seq_lengths = resident_seq_lengths_buffer[:num_reqs]
            a5_kvcache_scatter_copy_c8(
                resident_packed_cache=resident_cache[0],
                dram_packed_arena=arenas.packed,
                resident_block_table=resident_block_table,
                dram_block_table=dram_block_table,
                source_token_ids=outputs.topk_index,
                destination_slots=outputs.topk_slots,
                copy_counts=outputs.miss_count,
            )
            return DSAOffloadSelectionOutput(
                sparse_indices=attention_slots,
                tail_info=outputs.tail_info,
                resident_seq_lengths=resident_seq_lengths,
            )
        if len(resident_cache) < 2:
            raise RuntimeError("DSA bf16 shared layer requires NOPE and ROPE resident caches")
        kvcache_scatter_copy(
            resident_nope_cache=resident_cache[0],
            resident_rope_cache=resident_cache[1],
            dram_nope_arena=arenas.nope,
            dram_rope_arena=arenas.rope,
            resident_block_table=resident_block_table,
            dram_block_table=dram_block_table,
            src_token_ids=outputs.topk_index,
            dst_slots=outputs.topk_slots,
            copy_counts=outputs.miss_count,
        )
        return DSAOffloadSelectionOutput(
            sparse_indices=outputs.topk_slots,
            tail_info=outputs.tail_info,
        )

    def execute_sparse_attention(
        self,
        *,
        query: torch.Tensor,
        resident_cache: tuple[torch.Tensor, ...],
        selection: DSAOffloadSelectionOutput,
        scale_value: float,
        resident_block_table: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        query_rope: torch.Tensor,
    ) -> torch.Tensor:
        if self.packed_c8:
            if len(resident_cache) != 1:
                raise RuntimeError("DSA A5 C8 attention requires one packed resident cache")
            resident_seq_lengths = selection.resident_seq_lengths
            if resident_seq_lengths is None:
                raise RuntimeError("DSA A5 C8 selection has no resident sequence lengths")
            return sparse_flash_attention_for_offload_c8(
                query=torch.cat([query, query_rope], dim=-1),
                packed_kv=resident_cache[0],
                sparse_indices=selection.sparse_indices,
                scale_value=scale_value,
                block_table=resident_block_table,
                actual_seq_lengths_query=actual_seq_lengths_query,
                resident_seq_lengths=resident_seq_lengths,
            )
        if len(resident_cache) < 2:
            raise RuntimeError("DSA bf16 attention requires NOPE and ROPE resident caches")
        return sparse_flash_attention_for_offload(
            query=query,
            key=resident_cache[0],
            sparse_indices=selection.sparse_indices,
            tail_info=selection.tail_info,
            scale_value=scale_value,
            block_table=resident_block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope,
            key_rope=resident_cache[1],
        )

    def dump_full_blocks(
        self,
        *,
        resident_cache: tuple[torch.Tensor, ...],
        before_cache_write: bool,
    ) -> None:
        job_count = self.runtime.dump_launch_count
        if job_count == 0:
            return
        if not self.runtime.claim_layer_dump(
            layer_id=self.layer_id,
            before_cache_write=before_cache_write,
        ):
            return
        store = self.runtime.dram_store
        if store is None:
            raise RuntimeError("DSA layer has no bound DRAM store")
        arenas = store.get_layer_arenas(self.layer_id)
        if self.packed_c8:
            if len(resident_cache) != 1 or arenas.packed is None:
                raise RuntimeError("DSA A5 C8 dump requires packed HBM and DRAM caches")
            dump_full_kv_cache_blocks_c8(
                resident_packed_cache=resident_cache[0],
                dram_packed_arena=arenas.packed,
                src_block_ids=self.runtime.dump_src_block_ids.gpu[:job_count],
                dst_block_ids=self.runtime.dump_dst_block_ids.gpu[:job_count],
            )
        else:
            if len(resident_cache) < 2 or arenas.nope is None or arenas.rope is None:
                raise RuntimeError("DSA bf16 dump requires NOPE and ROPE HBM/DRAM planes")
            dump_full_kv_cache_blocks(
                resident_nope_cache=resident_cache[0],
                resident_rope_cache=resident_cache[1],
                dram_nope_arena=arenas.nope,
                dram_rope_arena=arenas.rope,
                src_block_ids=self.runtime.dump_src_block_ids.gpu[:job_count],
                dst_block_ids=self.runtime.dump_dst_block_ids.gpu[:job_count],
            )

        if len(arenas.indexer_source_bytes) != len(
            arenas.indexer_dram_bytes
        ):
            raise RuntimeError(
                "DSA Indexer source/DRAM plane count mismatch"
            )
        for source_bytes, dram_bytes in zip(
            arenas.indexer_source_bytes,
            arenas.indexer_dram_bytes,
            strict=True,
        ):
            dump_full_cache_plane_bytes(
                source_bytes=source_bytes,
                dram_bytes=dram_bytes,
                src_block_ids=(
                    self.runtime.dump_indexer_src_block_ids.gpu[:job_count]
                ),
                dst_block_ids=self.runtime.dump_dst_block_ids.gpu[:job_count],
            )
