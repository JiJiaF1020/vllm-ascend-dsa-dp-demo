# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 多级缓存接入 vLLM 原生 KVConnector 生命周期的入口。

根据 connector 所在角色组合 scheduler handler 或 worker handler：调度侧
完成外部 prefix 查询、分配后的 load 计划与 store 任务下发；worker 侧在
既有回调中处理复用 fence、同步加载和异步卸载，并经 worker metadata
回传完成结果。实际 payload 由 DSA worker DRAM store 持有。

含 MTP 的真实 forward 先暂存 store metadata，等 proposer 完成对应 durable
dump 后，再由 ``wait_for_save`` 放行上传，避免读取尚未写完整的 MTP plane。
当前加载一次处理完整任务，``wait_for_layer_load`` 和 ``save_kv_layer``
不提供 layerwise IO；L2 -> HBM restore 仍由 DSA runtime 执行。

本模块负责生命周期衔接，不建立第二套 Scheduler，也不直接实现存储协议。
单机DP正常退出在全域排空后，经原生worker RPC提前调用同一个shutdown路径；
使用原生ensure_kv_transfer_shutdown清除全局引用，后续WorkerProc关闭自然幂等。
"""

from __future__ import annotations

import time
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.forward_context import ForwardContext
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_ascend.dsa_offload.l3.backend_adapter import DSAL3BackendAdapter
from vllm_ascend.dsa_offload.l3.metadata import (
    DSAL3ConnectorMetadata,
    DSAL3WorkerMetadata,
)
from vllm_ascend.dsa_offload.l3.scheduler import DSAL3StoreScheduler
from vllm_ascend.dsa_offload.l3.schema import DSAL3KeySchema
from vllm_ascend.dsa_offload.l3.worker import DSAL3StoreWorker
from vllm_ascend.dsa_offload.worker_dram_store import DSAWorkerDRAMStore


class DSAL3Connector(KVConnectorBase_V1, SupportsHMA):
    """DSA rank0 write-behind 与同步 all-rank prefix load。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        schema = DSAL3KeySchema.from_config(vllm_config, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        kv_role = str(self._kv_transfer_config.kv_role)
        self.enable_store = kv_role in ("kv_producer", "kv_both")
        self.enable_load = kv_role in ("kv_consumer", "kv_both")
        if not self.enable_store and not self.enable_load:
            raise ValueError(f"Unsupported DSA L3 kv_role={kv_role!r}")
        backend_name = str(extra.get("backend", "mooncake"))
        if backend_name != "mooncake":
            raise ValueError(f"DSA L3 first version supports backend='mooncake' only, got {backend_name!r}")
        max_inflight = int(extra.get("max_inflight_l3_store_jobs", 8))
        max_active_channels = int(extra.get("max_active_channels", 8))
        max_descriptors_per_put = int(
            extra.get("max_descriptors_per_l3_put", 128)
        )
        direct_hbm_load = extra.get("direct_hbm_load", False)
        if not isinstance(direct_hbm_load, bool):
            raise TypeError(
                "DSA direct_hbm_load must be a JSON boolean, got "
                f"{direct_hbm_load!r}"
            )
        if direct_hbm_load:
            raise ValueError(
                "DSA direct_hbm_load is frozen; use L3->L2->HBM"
            )
        self.direct_hbm_load = direct_hbm_load
        self._has_mtp_component = schema.has_mtp
        self.data_parallel_rank = int(getattr(vllm_config.parallel_config, "data_parallel_rank", 0))
        self.data_parallel_size = int(getattr(vllm_config.parallel_config, "data_parallel_size", 1))
        self.supports_dsa_namespace_reset = self.data_parallel_size == 1
        self._defer_store_until_wait = False
        self._deferred_store_metadata: DSAL3ConnectorMetadata | None = None
        self.scheduler_handler: DSAL3StoreScheduler | None = None
        self.worker_handler: DSAL3StoreWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            logger.info(
                "DSA prefix tiers: target_hbm=%s local_l2=%s l3=%s dp_rank=%d fingerprint=%s mtp=%s",
                bool(vllm_config.cache_config.enable_prefix_caching),
                bool(vllm_config.cache_config.enable_prefix_caching),
                self.enable_load,
                self.data_parallel_rank,
                schema.fingerprint,
                schema.has_mtp,
            )
            lookup_backend = (
                DSAL3BackendAdapter.create_mooncake(
                    vllm_config=vllm_config,
                    schema=schema,
                    max_active_channels=max_active_channels,
                    max_descriptors_per_put=(
                        max_descriptors_per_put
                    ),
                    contribute_memory=False,
                    direct_hbm_load=self.direct_hbm_load,
                )
                if self.enable_load
                else None
            )
            self.scheduler_handler = DSAL3StoreScheduler(
                schema,
                max_inflight_jobs=max_inflight,
                lookup_backend=lookup_backend,
                enable_load=self.enable_load,
                direct_hbm_load=self.direct_hbm_load,
            )
        elif role == KVConnectorRole.WORKER:
            tp_rank = get_tensor_model_parallel_rank()
            # 对齐 AscendStore 的 embedded-client 范式：每个 TP worker
            # 都挂载自己的 Mooncake segment，组成全局池。Indexer/MLA
            # payload 是 TP 副本，因此仍只允许 TP0 发起一次 put。
            backend = DSAL3BackendAdapter.create_mooncake(
                vllm_config=vllm_config,
                schema=schema,
                max_active_channels=max_active_channels,
                max_descriptors_per_put=max_descriptors_per_put,
                direct_hbm_load=self.direct_hbm_load,
            )
            logger.info(
                "DSA L3 worker backend active: tp_rank=%d "
                "contributes_segment=True store_producer=%s "
                "load_consumer=%s direct_hbm=%s dp_rank=%d fingerprint=%s",
                tp_rank,
                self.enable_store and tp_rank == 0,
                self.enable_load,
                self.direct_hbm_load,
                self.data_parallel_rank,
                schema.fingerprint,
            )
            self.worker_handler = DSAL3StoreWorker(
                tp_rank=tp_rank,
                device=torch.device(f"npu:{torch.npu.current_device()}"),
                backend=backend,
                max_inflight_jobs=max_inflight,
                enable_load=self.enable_load,
                direct_hbm_load=self.direct_hbm_load,
            )

    def bind_dsa_coordinator(self, coordinator: Any) -> None:
        if self.scheduler_handler is None:
            raise RuntimeError("DSA coordinator can only bind on scheduler")
        self.scheduler_handler.bind_coordinator(coordinator)

    def bind_dsa_dram_store(self, store: DSAWorkerDRAMStore) -> None:
        if self.worker_handler is None:
            raise RuntimeError("DSA L3 DRAM store can only bind on a worker")
        self.worker_handler.bind_store(store)

    def handle_preemptions(self, kv_connector_metadata: Any) -> None:
        if not isinstance(kv_connector_metadata, DSAL3ConnectorMetadata):
            raise TypeError("DSA L3 worker received invalid connector metadata")
        assert self.worker_handler is not None
        self.worker_handler.prepare_step(kv_connector_metadata)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del kwargs
        assert self.worker_handler is not None
        if getattr(forward_context, "skip_kv_transfer", False):
            # FULL dummy也有attention metadata，但不能暂存/完成真实MTP上传任务。
            raise RuntimeError("DSA dummy forward must not enter the L3 connector lifecycle")
        if self._deferred_store_metadata is not None:
            raise RuntimeError(
                "DSA MTP L3 store metadata survived into the next step; "
                "the proposer-side connector finalization was skipped"
            )
        # ``KVConnectorModelRunnerMixin`` always calls get_finished() when the
        # target-forward context exits, even when connector finalization is
        # deferred for speculative decoding.  MTP cache planes are written by
        # the proposer after that point, so only stage store jobs in
        # get_finished() and submit them from wait_for_save().  Connector-only
        # progress steps have no attention metadata and can submit immediately.
        has_model_forward = bool(
            forward_context is not None
            and forward_context.attn_metadata is not None
        )
        self._defer_store_until_wait = bool(
            self._has_mtp_component and has_model_forward
        )
        self.worker_handler.start_step()

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        if self.worker_handler is None:
            return
        metadata = self._deferred_store_metadata
        if metadata is None:
            return
        self._deferred_store_metadata = None
        self._defer_store_until_wait = False
        logger.info(
            "DSA MTP L3 store released after proposer completion: "
            "jobs=%d blocks=%d",
            len(metadata.store_jobs),
            sum(len(job.blocks) for job in metadata.store_jobs.values()),
        )
        self.worker_handler.finish_step(metadata)

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str], set[str]]:
        del finished_req_ids
        assert self.worker_handler is not None
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DSAL3ConnectorMetadata):
            raise TypeError("DSA L3 worker metadata is not bound")
        if self._defer_store_until_wait and metadata.store_jobs:
            if self._deferred_store_metadata is not None:
                raise RuntimeError(
                    "DSA MTP L3 received another store batch before the "
                    "previous proposer-side finalization"
                )
            self._deferred_store_metadata = metadata
            logger.info(
                "DSA MTP L3 store staged until proposer completion: "
                "jobs=%d blocks=%d",
                len(metadata.store_jobs),
                sum(
                    len(job.blocks)
                    for job in metadata.store_jobs.values()
                ),
            )
        else:
            self.worker_handler.finish_step(metadata)
        self._defer_store_until_wait = False
        return set(), set()

    def build_connector_worker_meta(self) -> DSAL3WorkerMetadata | None:
        assert self.worker_handler is not None
        return self.worker_handler.build_worker_meta()

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self.scheduler_handler is not None
        external_tokens = self.scheduler_handler.lookup_external_prefix(
            request,
            num_computed_tokens,
        )
        return external_tokens, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        del blocks
        assert self.scheduler_handler is not None
        self.scheduler_handler.stage_external_load(
            request.request_id,
            num_external_tokens,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> DSAL3ConnectorMetadata:
        del scheduler_output
        assert self.scheduler_handler is not None
        return self.scheduler_handler.build_connector_meta()

    def extend_dsa_store_jobs(
        self,
        *,
        scheduler_output: SchedulerOutput,
        projection: Any,
        coordinator: Any,
        requests: dict[str, Request],
    ) -> None:
        assert self.scheduler_handler is not None
        metadata = scheduler_output.kv_connector_metadata
        if not isinstance(metadata, DSAL3ConnectorMetadata):
            raise TypeError("DSA scheduler output is missing DSAL3ConnectorMetadata")
        self.scheduler_handler.extend_store_jobs(
            metadata=metadata,
            projection=projection,
            coordinator=coordinator,
            requests=requests,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self.scheduler_handler is not None
        self.scheduler_handler.update_connector_output(connector_output)

    def has_pending_transfers(self) -> bool:
        """Scheduler-side progress hint for asynchronous write-behind."""

        if self.scheduler_handler is None:
            return False
        return self.scheduler_handler.has_pending_jobs()

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        assert self.scheduler_handler is not None
        self.scheduler_handler.request_finished(request.request_id)
        return False, None

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        assert self.scheduler_handler is not None
        self.scheduler_handler.request_finished(request.request_id)
        return False, None

    def get_finished_count(self) -> int | None:
        return 1

    def reset_cache_for_dp(self, namespace_epoch: str) -> bool:
        """仅供已暂停且持有管理租约的EngineCore事务提交统一epoch。"""
        if self.data_parallel_size <= 1 or self.scheduler_handler is None:
            raise RuntimeError("DSA coordinated namespace reset requires a DP scheduler connector")
        return self.scheduler_handler.reset_cache(namespace_epoch=namespace_epoch)

    def reset_cache(self) -> bool:
        if not self.supports_dsa_namespace_reset:
            logger.warning("Uncoordinated DSA DP namespace reset rejected; use the online all-DP reset transaction")
            return False
        if self.scheduler_handler is None:
            # reset_cache is scheduler-owned; worker payload is overwritten by
            # subsequent assignments and does not own the L3 key namespace.
            return True
        return self.scheduler_handler.reset_cache()

    def shutdown(self) -> None:
        if self._deferred_store_metadata is not None:
            logger.warning(
                "Discarding DSA MTP L3 store jobs because proposer-side "
                "finalization did not complete: jobs=%d",
                len(self._deferred_store_metadata.store_jobs),
            )
            self._deferred_store_metadata = None
        if self.worker_handler is not None:
            self.worker_handler.shutdown()
        if self.scheduler_handler is not None:
            self.scheduler_handler.shutdown()


def shutdown_dsa_l3_worker(worker: Any) -> bool:
    """全域排空后的原生collective_rpc回调，只关闭DSA L3，不关闭模型或通信组。"""
    from vllm.distributed.kv_transfer.kv_transfer_state import (
        ensure_kv_transfer_shutdown,
        get_kv_transfer_group,
        has_kv_transfer_group,
    )

    config = worker.vllm_config.kv_transfer_config
    if getattr(config, "kv_connector", None) != "DSAL3Connector":
        raise RuntimeError("DSA resource shutdown RPC cannot close a different connector")
    if has_kv_transfer_group():
        connector = get_kv_transfer_group()
        if not isinstance(connector, DSAL3Connector) or connector.worker_handler is None:
            raise RuntimeError("DSA resource shutdown RPC requires its worker connector")
        if connector._deferred_store_metadata is not None:
            raise RuntimeError("DSA resource shutdown still has deferred MTP store metadata")
        # DSA dump与record_event都在当前模型流；不要等待SDK可能常驻的通信流。
        started = time.monotonic()
        logger.info("DSA L3 resource close begin: stage=model-stream")
        torch.npu.current_stream().synchronize()
        logger.info(
            "DSA L3 resource close completed: stage=model-stream elapsed_ms=%.3f",
            (time.monotonic() - started) * 1000,
        )
        ensure_kv_transfer_shutdown()
    return True
