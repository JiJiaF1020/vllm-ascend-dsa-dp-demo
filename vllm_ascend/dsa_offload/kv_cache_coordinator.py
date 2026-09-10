# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA target 双物理池与可选 MTP full-cache 池 coordinator。

vLLM v0.23 的多 group coordinator 仍默认所有 manager 共用一个
``BlockPool``，这适用于“不同逻辑 cache 共享同一物理 block id 空间”的
hybrid cache。DSA target 的 Indexer dense plane、MLA resident plane，
以及方案二中可选的 MTP 原生 BF16 full-cache plane 容量不同，block id 也
分别索引各自 tensor，因此必须为每个 group 建立独立 BlockPool，并在
admission 时逐 component 检查。

本模块同时持有请求 cache 布局 planner：manager 先生成不可变布局计划，完成
各 pool 容量检查和物理修改后再 commit；prefill 输出返回后，薄 Scheduler
也通过这里释放已卸载的 resident 满块。它仍不复制 Scheduler 主循环。
本地 prefix cache 给完整 DRAM bundle 与 HBM component 建 hash：no-MTP
使用 target pair，MTP 使用 target pair+MTP full-cache triple。完整 HBM
prefix 可以比 L2 更长，缺失的 L2 由首个 prefill forward 回填。固定 MTP3 的 target
验算仍由 resident 双尾块承接 provisional 写入；MTP proposer 自身则使用
第三个原生 full-cache group。active decode 不做 MTP token-wise swap，只有
verified full blocks 进入完整 DRAM/L3 bundle。

这里的 block table 是 scheduler/core 侧逻辑真源。v0.23 原生
``SchedulerOutput`` 对 cached request 只表达“追加的新块”，尚不能表达
ENTER 对 resident 表的整表替换；该传输由类型化 scheduler projection 和
worker ``DSAInputBatchCacheLayout`` 共同完成，不能由各 worker 根据长度
自行重新分配。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashListWithBlockSize,
    KVCacheBlock,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    FullAttentionManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request

from vllm_ascend.dsa_offload.kv_cache import (
    DSAKVCacheConfig,
    get_dsa_group_num_blocks,
    get_dsa_kv_cache_group_ids,
    validate_dsa_kv_cache_config,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCachePlanner,
    DSARequestCacheStage,
    DSARequestCacheState,
)
from vllm_ascend.dsa_offload.scheduler_dram_pool import (
    DSASchedulerDRAMBlockDirectory,
)
from vllm_ascend.dsa_offload.scheduler_output import (
    DSADRAMBlockRestore,
)


class DSABlockPoolView:
    """向 vLLM 顶层暴露多个物理 BlockPool 的只读聚合视图。

    single-type manager 不通过该对象分配，而是直接持有各自的真实 pool。
    聚合视图仅服务于 usage、reset、event 等 ``KVCacheManager`` 公共接口；
    admission 必须调用 ``DSAKVCacheCoordinator.can_allocate``，不能使用
    聚合 free-block 数做标量判断。
    """

    def __init__(
        self,
        block_pools: Sequence[BlockPool],
        *,
        extra_prefix_cache_pools: Sequence[BlockPool] = (),
    ) -> None:
        if not block_pools:
            raise ValueError("DSA requires at least one physical block pool")
        self.block_pools = tuple(block_pools)
        self.prefix_cache_pools = (
            *self.block_pools,
            *tuple(extra_prefix_cache_pools),
        )
        self.num_gpu_blocks = sum(pool.num_gpu_blocks for pool in self.block_pools)
        self.enable_caching = any(
            pool.enable_caching for pool in self.prefix_cache_pools
        )

    def get_num_free_blocks(self) -> int:
        return sum(pool.get_num_free_blocks() for pool in self.block_pools)

    def get_usage(self) -> float:
        # 调度压力由最先耗尽的 component 决定，使用最大利用率比按总块数
        # 加权平均更能反映 DSA 的真实可调度容量。
        return max(pool.get_usage() for pool in self.block_pools)

    def can_reset_prefix_cache(self) -> bool:
        """复用基线BlockPool的零引用判据，只检查而不清理任何一个pool。"""
        return all(pool.num_gpu_blocks - pool.get_num_free_blocks() == 1 for pool in self.prefix_cache_pools)

    def reset_prefix_cache(self) -> bool:
        # ``all(generator)`` 会在首个 False 处短路，导致后续物理池没有被
        # reset。所有 pool 都必须尝试，避免 HBM MTP 与 DRAM bundle 只
        # 清掉一侧后产生不一致命中。
        results = [
            pool.reset_prefix_cache() for pool in self.prefix_cache_pools
        ]
        return all(results)

    def evict_blocks(self, block_ids: set[int]) -> None:
        # 各物理池的 block id 均从 0 开始。首版禁用 KV connector 的
        # 外部按-id 驱逐；若未来开放该接口，必须增加 group/tier id，
        # 不能把同一组裸 block id 无差别广播给各 pool。
        if block_ids:
            raise RuntimeError("DSA split KV-cache eviction requires group-qualified block ids")

    def take_events(self) -> list:
        events = []
        for pool in self.prefix_cache_pools:
            events.extend(pool.take_events())
        return events


@dataclass(frozen=True, slots=True)
class DSAPrefixCacheHitPlan:
    """查询阶段的零引用候选；只有联合 admission 成功后才能提交。

    ``dram_blocks`` 与完整 HBM component blocks 分别表示两个独立本地
    tier 的连续前缀。二者允许长度不同；``local_computed_tokens`` 取二者
    的较长者。若 HBM 前缀更长，缺失的 L2 bundle 会在首个 prefill
    forward 中由既有 full-block dump 同步补齐。
    """

    request_id: str
    num_computed_tokens: int
    dram_blocks: tuple[KVCacheBlock, ...]
    hbm_indexer_blocks: tuple[KVCacheBlock, ...] = ()
    hbm_resident_blocks: tuple[KVCacheBlock, ...] = ()
    mtp_blocks: tuple[KVCacheBlock, ...] = ()
    external_computed_tokens: int = 0
    external_block_hashes: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        if self.num_computed_tokens < 0:
            raise ValueError("DSA prefix token count cannot be negative")
        if not 0 <= self.external_computed_tokens <= self.num_computed_tokens:
            raise ValueError("DSA external prefix token count is invalid")
        if len(self.hbm_indexer_blocks) != len(
            self.hbm_resident_blocks
        ):
            raise ValueError(
                "DSA paired HBM prefix components have different lengths"
            )
        if self.mtp_blocks and len(self.mtp_blocks) != len(
            self.hbm_indexer_blocks
        ):
            raise ValueError(
                "DSA MTP HBM prefix has a different component length"
            )
        if not self.request_id:
            raise ValueError("DSA prefix hit requires a request ID")

    @property
    def local_computed_tokens(self) -> int:
        return self.num_computed_tokens - self.external_computed_tokens

    @property
    def hbm_block_count(self) -> int:
        return len(self.hbm_indexer_blocks)

    @property
    def l2_block_count(self) -> int:
        return len(self.dram_blocks)


@dataclass(frozen=True, slots=True)
class DSAExternalPrefixLoadPlan:
    """已完成 admission、等待 Connector 下发的 L3-only suffix。"""

    request_id: str
    logical_block_start: int
    block_hashes: tuple[bytes, ...]
    destination_dram_block_ids: tuple[int, ...]
    destination_indexer_hbm_block_ids: tuple[int, ...]
    destination_resident_hbm_block_ids: tuple[int, ...]
    destination_mtp_hbm_block_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("DSA external prefix load requires a request ID")
        if self.logical_block_start < 0:
            raise ValueError("DSA external prefix load start must be non-negative")
        if not self.block_hashes:
            raise ValueError("DSA external prefix load cannot be empty")
        if len(self.block_hashes) != len(self.destination_dram_block_ids):
            raise ValueError(
                "DSA external prefix hash/destination columns disagree"
            )
        if not (
            len(self.block_hashes)
            == len(self.destination_indexer_hbm_block_ids)
            == len(self.destination_resident_hbm_block_ids)
        ):
            raise ValueError(
                "DSA external prefix HBM destination columns disagree"
            )
        if self.destination_mtp_hbm_block_ids and len(
            self.destination_mtp_hbm_block_ids
        ) != len(self.block_hashes):
            raise ValueError(
                "DSA external prefix MTP destination column disagrees"
            )
        if any(block_id <= 0 for block_id in self.destination_dram_block_ids):
            raise ValueError(
                "DSA external prefix destinations must be positive"
            )
        if any(
            block_id <= 0
            for block_ids in (
                self.destination_indexer_hbm_block_ids,
                self.destination_resident_hbm_block_ids,
                self.destination_mtp_hbm_block_ids,
            )
            for block_id in block_ids
        ):
            raise ValueError(
                "DSA external prefix HBM destinations must be positive"
            )


class DSAKVCacheCoordinator(KVCacheCoordinator):
    """为 DSA layout 中每个物理 cache group 构造独立 pool/manager。"""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_num_batched_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        if enable_kv_cache_events:
            raise ValueError(
                "DSA split KV-cache does not yet support KV-cache events: "
                "the independent pools currently reuse block IDs"
            )
        if metrics_collector is not None:
            raise ValueError(
                "DSA split KV-cache does not yet support KV-cache metrics: "
                "the independent pools currently reuse block IDs"
            )
        if scheduler_block_size % hash_block_size != 0 or any(
            scheduler_block_size % group.kv_cache_spec.block_size != 0 for group in kv_cache_config.kv_cache_groups
        ):
            raise ValueError(
                "DSA scheduler block size must be divisible by the hash block size and every KV-cache group block size"
            )

        validate_dsa_kv_cache_config(kv_cache_config)
        if (
            not isinstance(kv_cache_config, DSAKVCacheConfig)
            or kv_cache_config.dram_capacity is None
        ):
            raise RuntimeError(
                "DSA coordinator requires the scheduler-finalized DRAM "
                "capacity plan"
            )
        self.kv_cache_config = kv_cache_config
        self.data_parallel_rank = kv_cache_config.dram_capacity.data_parallel_rank
        self.max_model_len = max_model_len
        self.enable_caching = bool(enable_caching)
        self.scheduler_block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        self.retention_interval = None
        self.group_ids = get_dsa_kv_cache_group_ids(kv_cache_config)
        physical_pools = tuple(
            BlockPool(
                num_gpu_blocks=get_dsa_group_num_blocks(
                    kv_cache_config,
                    group,
                ),
                enable_caching=(
                    self.enable_caching
                    and group_id
                    in {
                        self.group_ids.indexer,
                        self.group_ids.resident_mla,
                        self.group_ids.mtp_full,
                    }
                ),
                hash_block_size=hash_block_size,
                enable_kv_cache_events=enable_kv_cache_events,
                metrics_collector=metrics_collector,
            )
            for group_id, group in enumerate(
                kv_cache_config.kv_cache_groups
            )
        )
        self.physical_block_pools = physical_pools

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                block_pool=physical_pools[group_id],
                enable_caching=physical_pools[group_id].enable_caching,
                kv_cache_group_id=group_id,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                scheduler_block_size=scheduler_block_size,
            )
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        )
        self.num_single_type_manager = len(self.single_type_managers)
        indexer_block_size = int(
            kv_cache_config.kv_cache_groups[
                self.group_ids.indexer
            ].kv_cache_spec.block_size
        )
        # vLLM Scheduler.block_size 在 hybrid/hash/CP 配置下可能与单个
        # 物理 group 不同，不能用它推导 DSA DRAM logical block 数。
        # DRAM bundle 与 Indexer/MLA payload 一一对应，唯一真源是最终
        # Indexer spec 的物理 block size。
        self.dram_block_size = indexer_block_size
        if self.enable_caching and hash_block_size != indexer_block_size:
            raise ValueError(
                "DSA first-version DRAM prefix cache requires hash and "
                "physical bundle block sizes to match: "
                f"hash={hash_block_size}, dram={indexer_block_size}"
            )
        self.dram_directory = DSASchedulerDRAMBlockDirectory(
            usable_blocks=kv_cache_config.dram_capacity.usable_blocks,
            max_logical_blocks=cdiv(max_model_len, indexer_block_size),
            block_size=indexer_block_size,
            enable_caching=self.enable_caching,
        )
        self.block_pool = DSABlockPoolView(
            physical_pools,
            extra_prefix_cache_pools=(
                self.dram_directory.block_pool,
            ),
        )
        self._pending_prefix_hits: dict[
            str,
            DSAPrefixCacheHitPlan,
        ] = {}
        self._pending_prefix_restores: dict[
            str,
            DSADRAMBlockRestore,
        ] = {}
        self._pending_external_prefix_loads: dict[
            str,
            DSAExternalPrefixLoadPlan,
        ] = {}
        self.l3_direct_hbm_load_enabled = False
        self.eagle_group_ids = (
            {self.group_ids.mtp_full}
            if self.group_ids.mtp_full is not None
            else set()
        )

        from vllm_ascend.ascend_config import get_ascend_config
        from vllm_ascend.dsa_offload.kv_cache_manager import (
            DSAIndexerKVCacheManager,
            DSAResidentMLAKVCacheManager,
        )

        indexer_manager = self.single_type_managers[self.group_ids.indexer]
        resident_manager = self.single_type_managers[self.group_ids.resident_mla]
        if not isinstance(indexer_manager, DSAIndexerKVCacheManager):
            raise RuntimeError(
                f"DSA Indexer group did not resolve to DSAIndexerKVCacheManager: {type(indexer_manager).__name__}"
            )
        if not isinstance(
            resident_manager,
            DSAResidentMLAKVCacheManager,
        ):
            raise RuntimeError(
                "DSA resident MLA group did not resolve to "
                "DSAResidentMLAKVCacheManager: "
                f"{type(resident_manager).__name__}"
            )
        self.indexer_manager = indexer_manager
        self.resident_manager = resident_manager
        self.mtp_full_manager: FullAttentionManager | None = None
        if self.group_ids.mtp_full is not None:
            mtp_full_manager = self.single_type_managers[
                self.group_ids.mtp_full
            ]
            if not isinstance(mtp_full_manager, FullAttentionManager):
                raise RuntimeError(
                    "DSA MTP full group did not resolve to "
                    "FullAttentionManager: "
                    f"{type(mtp_full_manager).__name__}"
                )
            mtp_full_manager.use_eagle = True
            self.mtp_full_manager = mtp_full_manager

        dsa_config = get_ascend_config().dsa_offload_config
        if bool(use_eagle) != dsa_config.mtp_enabled:
            raise RuntimeError(
                "DSA coordinator Eagle mode disagrees with MTP config: "
                f"use_eagle={use_eagle}, mtp_enabled={dsa_config.mtp_enabled}"
            )
        if dsa_config.mtp_enabled != (self.mtp_full_manager is not None):
            raise RuntimeError(
                "DSA MTP config and finalized full-cache group disagree: "
                f"mtp_enabled={dsa_config.mtp_enabled}, "
                f"mtp_group_id={self.group_ids.mtp_full}"
            )
        if (
            self.group_ids.mtp_full is not None
            and len(
                kv_cache_config.kv_cache_groups[
                    self.group_ids.mtp_full
                ].layer_names
            )
            != dsa_config.mtp_cache_layer_count
        ):
            raise RuntimeError(
                "DSA MTP full-cache layer count disagrees with config: "
                f"configured={dsa_config.mtp_cache_layer_count}, "
                "group_layers="
                f"{len(kv_cache_config.kv_cache_groups[self.group_ids.mtp_full].layer_names)}"
            )
        self.request_cache_layout = DSARequestCachePlanner(
            block_size=scheduler_block_size,
            sparse_activation_tokens=(dsa_config.sparse_activation_tokens),
            prompt_budget_thresholds=(dsa_config.prompt_budget_thresholds),
            resident_budget_tokens=dsa_config.resident_budget_tokens,
            sparse_tail_block_count=dsa_config.sparse_tail_block_count,
        )

    def set_l3_direct_hbm_load_enabled(self, enabled: bool) -> None:
        """冻结 scheduler/worker 共用的 P1B direct-HBM capability。"""

        enabled = bool(enabled)
        if enabled and self.mtp_full_manager is not None:
            raise RuntimeError(
                "DSA P1B direct-HBM does not yet support MTP"
            )
        self.l3_direct_hbm_load_enabled = enabled

    def prepare_prefix_cache_hit(
        self,
        request: Request,
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """构造 L2、成对 target HBM 与 MTP 的本地命中候选。"""

        request_id = request.request_id
        self._pending_prefix_hits.pop(request_id, None)
        l2_blocks = self.dram_directory.find_longest_cache_hit(
            request.block_hashes,
            max_cache_hit_length,
        )
        l2_tokens = len(l2_blocks) * self.dram_block_size
        l2_tokens = (
            l2_tokens
            // self.scheduler_block_size
            * self.scheduler_block_size
        )
        l2_blocks = l2_blocks[
            : l2_tokens // self.dram_block_size
        ]
        if self.mtp_full_manager is not None and l2_blocks:
            # EAGLE/MTP replay needs the final cached block to be recomputed in
            # order to rebuild the drafting boundary state. L2 stores every
            # durable full block, so convert its raw bundle prefix into the
            # replayable prefix by dropping exactly one block.
            l2_blocks = l2_blocks[:-1]
            l2_tokens = len(l2_blocks) * self.dram_block_size

        max_local_tokens = min(
            int(max_cache_hit_length),
            len(request.block_hashes) * self.dram_block_size,
        )
        max_local_tokens = (
            max_local_tokens
            // self.scheduler_block_size
            * self.scheduler_block_size
        )

        # 两个 target HBM component 任一先 miss 就在较短处截断；MTP
        # 开启时再与 proposer full-cache 取公共前缀。该完整 HBM prefix
        # 不受 L2 长度约束，HBM-only 范围由首个 prefill forward 回填 L2。
        # 单 component source mask 留给后续优化阶段。
        hbm_indexer_blocks: tuple[KVCacheBlock, ...] = ()
        hbm_resident_blocks: tuple[KVCacheBlock, ...] = ()
        if max_local_tokens:
            component_hits: list[tuple[KVCacheBlock, ...]] = []
            for group_id, component_manager in (
                (self.group_ids.indexer, self.indexer_manager),
                (self.group_ids.resident_mla, self.resident_manager),
            ):
                component_spec = self.kv_cache_config.kv_cache_groups[
                    group_id
                ].kv_cache_spec
                component_block_size = int(component_spec.block_size)
                if component_block_size != self.dram_block_size:
                    raise RuntimeError(
                        "DSA P1A target HBM prefix requires target and "
                        "durable block sizes to match: "
                        f"group={group_id}, hbm={component_block_size}, "
                        f"dram={self.dram_block_size}"
                    )
                hit = component_manager.find_longest_cache_hit(
                    block_hashes=request.block_hashes,
                    max_length=max_local_tokens,
                    kv_cache_group_ids=[group_id],
                    block_pool=component_manager.block_pool,
                    kv_cache_spec=component_spec,
                    drop_eagle_block=False,
                    alignment_tokens=self.scheduler_block_size,
                )[0]
                component_hits.append(tuple(hit))
            hbm_block_count = min(
                *(len(hit) for hit in component_hits),
            )
            hbm_indexer_blocks = component_hits[0][:hbm_block_count]
            hbm_resident_blocks = component_hits[1][:hbm_block_count]

        mtp_blocks: tuple[KVCacheBlock, ...] = ()
        if hbm_indexer_blocks and self.mtp_full_manager is not None:
            mtp_group_id = self.group_ids.mtp_full
            assert mtp_group_id is not None
            mtp_spec = self.kv_cache_config.kv_cache_groups[
                mtp_group_id
            ].kv_cache_spec
            mtp_block_size = int(mtp_spec.block_size)
            if mtp_block_size != self.dram_block_size:
                raise RuntimeError(
                    "DSA MTP multi-tier prefix requires matching block sizes: "
                    f"mtp={mtp_block_size}, dram={self.dram_block_size}"
                )
            mtp_hashes = (
                request.block_hashes
                if mtp_block_size == self.hash_block_size
                else BlockHashListWithBlockSize(
                    request.block_hashes,
                    self.hash_block_size,
                    mtp_block_size,
                )
            )
            # 与基线 HybridKVCacheCoordinator 一致：EAGLE/MTP 为了在
            # prefix 边界重建 drafting hidden states，多检查一个块后 drop。
            target_hbm_tokens = (
                len(hbm_indexer_blocks) * self.dram_block_size
            )
            mtp_max_length = min(
                target_hbm_tokens + mtp_block_size,
                int(max_cache_hit_length),
            )
            mtp_hit = self.mtp_full_manager.find_longest_cache_hit(
                block_hashes=mtp_hashes,
                max_length=mtp_max_length,
                kv_cache_group_ids=[mtp_group_id],
                block_pool=self.mtp_full_manager.block_pool,
                kv_cache_spec=mtp_spec,
                drop_eagle_block=True,
                alignment_tokens=self.scheduler_block_size,
            )[0]
            hbm_block_count = min(
                len(hbm_indexer_blocks),
                len(mtp_hit),
            )
            hbm_indexer_blocks = hbm_indexer_blocks[:hbm_block_count]
            hbm_resident_blocks = hbm_resident_blocks[:hbm_block_count]
            mtp_blocks = tuple(mtp_hit[:hbm_block_count])

        final_tokens = max(
            l2_tokens,
            len(hbm_indexer_blocks) * self.dram_block_size,
        )

        hbm_hit_blocks: list[list[KVCacheBlock]] = [
            [] for _ in range(self.num_single_type_manager)
        ]
        hbm_hit_blocks[self.group_ids.indexer] = list(
            hbm_indexer_blocks
        )
        hbm_hit_blocks[self.group_ids.resident_mla] = list(
            hbm_resident_blocks
        )
        if self.group_ids.mtp_full is not None:
            hbm_hit_blocks[self.group_ids.mtp_full] = list(mtp_blocks)
        if final_tokens:
            self._pending_prefix_hits[request_id] = (
                DSAPrefixCacheHitPlan(
                    request_id=request_id,
                    num_computed_tokens=final_tokens,
                    dram_blocks=tuple(l2_blocks),
                    hbm_indexer_blocks=hbm_indexer_blocks,
                    hbm_resident_blocks=hbm_resident_blocks,
                    mtp_blocks=mtp_blocks,
                )
            )
        return tuple(hbm_hit_blocks), final_tokens

    def can_reset_prefix_candidates(self) -> bool:
        """未分配的lookup候选可丢弃，已分配的restore/load计划必须先消费完。"""
        return not (self._pending_prefix_restores or self._pending_external_prefix_loads)

    def clear_prefix_candidates(self) -> None:
        if not self.can_reset_prefix_candidates():
            raise RuntimeError("DSA reset still has allocated prefix restore/load plans")
        self._pending_prefix_hits.clear()

    def get_pending_prefix_hit(
        self,
        request_id: str,
    ) -> DSAPrefixCacheHitPlan | None:
        return self._pending_prefix_hits.get(request_id)

    def prepare_external_prefix_cache_hit(
        self,
        request: Request,
        *,
        local_computed_tokens: int,
        external_block_count: int,
    ) -> int:
        """把 Connector 的 L3-only 连续命中并入本地零引用候选。"""

        request_id = request.request_id
        local_computed_tokens = int(local_computed_tokens)
        external_block_count = int(external_block_count)
        if local_computed_tokens < 0 or (
            local_computed_tokens % self.dram_block_size
        ):
            raise ValueError(
                "DSA local prefix tokens must be non-negative and block aligned"
            )
        if external_block_count < 0:
            raise ValueError("DSA external prefix block count cannot be negative")
        local_plan = self._pending_prefix_hits.get(request_id)
        if local_computed_tokens:
            if (
                local_plan is None
                or local_plan.num_computed_tokens != local_computed_tokens
                or local_plan.external_computed_tokens
            ):
                raise RuntimeError(
                    "DSA local prefix plan changed before external lookup: "
                    f"request_id={request_id!r}, local={local_computed_tokens}, "
                    f"plan={local_plan}"
                )
        elif local_plan is not None:
            raise RuntimeError(
                "DSA found an unexpected local prefix plan for a zero local hit"
            )
        if self.mtp_full_manager is not None and external_block_count:
            # The lookup starts at the already replayable local boundary. Its
            # first remote block is therefore the one-block MTP lookahead that
            # was dropped locally; dropping the final remote hit applies the
            # Eagle boundary rule exactly once to the merged L1/L2/L3 prefix.
            external_block_count -= 1
        if not external_block_count:
            return 0
        logical_start = local_computed_tokens // self.dram_block_size
        logical_end = logical_start + external_block_count
        if logical_end > len(request.block_hashes):
            raise RuntimeError(
                "DSA external prefix exceeds request block hashes: "
                f"request_id={request_id!r}, end={logical_end}, "
                f"hashes={len(request.block_hashes)}"
            )
        external_hashes = tuple(
            bytes(block_hash)
            for block_hash in request.block_hashes[
                logical_start:logical_end
            ]
        )
        external_tokens = external_block_count * self.dram_block_size
        self._pending_prefix_hits[request_id] = DSAPrefixCacheHitPlan(
            request_id=request_id,
            num_computed_tokens=local_computed_tokens + external_tokens,
            dram_blocks=(
                local_plan.dram_blocks if local_plan is not None else ()
            ),
            hbm_indexer_blocks=(
                local_plan.hbm_indexer_blocks
                if local_plan is not None
                else ()
            ),
            hbm_resident_blocks=(
                local_plan.hbm_resident_blocks
                if local_plan is not None
                else ()
            ),
            mtp_blocks=(
                local_plan.mtp_blocks if local_plan is not None else ()
            ),
            external_computed_tokens=external_tokens,
            external_block_hashes=external_hashes,
        )
        return external_tokens

    def can_commit_prefix_cache_hit(
        self,
        request_id: str,
        *,
        local_computed_tokens: int,
        external_computed_tokens: int,
        required_dram_block_count: int,
    ) -> bool:
        plan = self._pending_prefix_hits.get(request_id)
        if (
            int(local_computed_tokens) % self.dram_block_size
            or int(external_computed_tokens) % self.dram_block_size
        ):
            raise ValueError("DSA prefix admission requires block-aligned tokens")
        total = int(local_computed_tokens) + int(external_computed_tokens)
        if total <= 0:
            return True
        if (
            plan is None
            or plan.num_computed_tokens != total
            or plan.local_computed_tokens != int(local_computed_tokens)
            or plan.external_computed_tokens
            != int(external_computed_tokens)
        ):
            raise RuntimeError(
                "DSA prefix plan disagrees with allocation admission: "
                f"request_id={request_id!r}, local={local_computed_tokens}, "
                f"external={external_computed_tokens}, plan={plan}"
            )
        external_blocks = (
            int(external_computed_tokens) // self.dram_block_size
        )
        prefix_blocks = (
            int(local_computed_tokens) // self.dram_block_size
            + external_blocks
        )
        required_dram_block_count = int(required_dram_block_count)
        if required_dram_block_count < prefix_blocks:
            raise ValueError(
                "DSA required DRAM width cannot be shorter than its prefix"
            )
        return self.dram_directory.can_attach_cached_prefix(
            plan.dram_blocks,
            new_block_count=(
                required_dram_block_count - len(plan.dram_blocks)
            ),
        )

    def commit_prefix_cache_hit(
        self,
        request_id: str,
        local_computed_tokens: int,
        external_computed_tokens: int = 0,
        required_dram_block_count: int | None = None,
    ) -> None:
        """联合 admission 成功后提交 L2 引用、load 与 restore 映射。"""

        local_computed_tokens = int(local_computed_tokens)
        external_computed_tokens = int(external_computed_tokens)
        if (
            local_computed_tokens % self.dram_block_size
            or external_computed_tokens % self.dram_block_size
        ):
            raise ValueError("DSA prefix commit requires block-aligned tokens")
        total_computed_tokens = (
            local_computed_tokens + external_computed_tokens
        )
        if total_computed_tokens <= 0:
            self._pending_prefix_hits.pop(request_id, None)
            return
        plan = self._pending_prefix_hits.pop(request_id, None)
        if (
            plan is None
            or plan.num_computed_tokens != total_computed_tokens
            or plan.local_computed_tokens != local_computed_tokens
            or plan.external_computed_tokens
            != external_computed_tokens
        ):
            raise RuntimeError(
                "DSA prefix hit plan is missing or changed before commit: "
                f"request_id={request_id!r}, local={local_computed_tokens}, "
                f"external={external_computed_tokens}, "
                f"plan={plan}"
            )
        local_block_count = (
            local_computed_tokens // self.dram_block_size
        )
        external_block_count = (
            external_computed_tokens // self.dram_block_size
        )
        block_count = total_computed_tokens // self.dram_block_size
        local_l2_block_count = plan.l2_block_count
        if required_dram_block_count is None:
            required_dram_block_count = block_count
        required_dram_block_count = int(required_dram_block_count)
        if required_dram_block_count < block_count:
            raise ValueError(
                "DSA committed DRAM width cannot be shorter than its prefix"
            )
        if not self.dram_directory.can_attach_cached_prefix(
            plan.dram_blocks,
            new_block_count=(
                required_dram_block_count - local_l2_block_count
            ),
        ):
            raise RuntimeError(
                "DSA DRAM prefix capacity changed after admission"
            )
        indexer_blocks = self.indexer_manager.req_to_blocks[request_id]
        resident_blocks = self.resident_manager.req_to_blocks[request_id]
        mtp_blocks = (
            self.mtp_full_manager.req_to_blocks[request_id]
            if self.mtp_full_manager is not None
            else []
        )
        if (
            local_l2_block_count > local_block_count
            or len(plan.external_block_hashes) != external_block_count
            or len(indexer_blocks) < block_count
            or len(resident_blocks) < block_count
            or (
                self.mtp_full_manager is not None
                and len(mtp_blocks) < block_count
            )
        ):
            raise RuntimeError(
                "DSA prefix restore block tables do not cover the common "
                f"hit: request_id={request_id!r}, hit={block_count}, "
                f"local_effective={local_block_count}, "
                f"local_dram={local_l2_block_count}, "
                f"external={len(plan.external_block_hashes)}, "
                f"indexer={len(indexer_blocks)}, "
                f"resident={len(resident_blocks)}, "
                f"mtp={len(mtp_blocks)}"
            )
        hbm_block_count = plan.hbm_block_count
        if hbm_block_count > local_block_count:
            raise RuntimeError(
                "DSA paired HBM prefix exceeds the committed local hit: "
                f"request_id={request_id!r}, hbm={hbm_block_count}, "
                f"local={local_block_count}"
            )
        if (
            tuple(indexer_blocks[:hbm_block_count])
            != plan.hbm_indexer_blocks
            or tuple(resident_blocks[:hbm_block_count])
            != plan.hbm_resident_blocks
        ):
            raise RuntimeError(
                "DSA paired HBM prefix blocks changed before commit"
            )
        if self.mtp_full_manager is not None:
            if len(plan.mtp_blocks) != hbm_block_count:
                raise RuntimeError(
                    "DSA MTP prefix plan does not cover the complete HBM "
                    f"triple: target={hbm_block_count}, "
                    f"mtp={len(plan.mtp_blocks)}"
                )
            if tuple(mtp_blocks[: len(plan.mtp_blocks)]) != plan.mtp_blocks:
                raise RuntimeError(
                    "DSA MTP cached-prefix blocks changed before commit"
                )

        self.dram_directory.attach_cached_prefix(
            request_id,
            plan.dram_blocks,
        )
        self.dram_directory.ensure_request_blocks(
            request_id,
            required_dram_block_count,
        )
        all_dram_block_ids = self.dram_directory.get_request_blocks(
            request_id
        )
        if len(all_dram_block_ids) != required_dram_block_count:
            raise RuntimeError(
                "DSA DRAM prefix allocation produced an incomplete row: "
                f"request_id={request_id!r}, "
                f"expected={required_dram_block_count}, "
                f"actual={len(all_dram_block_ids)}"
            )
        prefix_dram_block_ids = tuple(all_dram_block_ids[:block_count])
        if external_block_count:
            external_destinations = tuple(
                prefix_dram_block_ids[
                    local_block_count:block_count
                ]
            )
            self._pending_external_prefix_loads[request_id] = (
                DSAExternalPrefixLoadPlan(
                    request_id=request_id,
                    logical_block_start=local_block_count,
                    block_hashes=plan.external_block_hashes,
                    destination_dram_block_ids=external_destinations,
                    destination_indexer_hbm_block_ids=tuple(
                        block.block_id
                        for block in indexer_blocks[
                            local_block_count:block_count
                        ]
                    ),
                    destination_resident_hbm_block_ids=tuple(
                        block.block_id
                        for block in resident_blocks[
                            local_block_count:block_count
                        ]
                    ),
                    destination_mtp_hbm_block_ids=(
                        tuple(
                            block.block_id
                            for block in mtp_blocks[
                                local_block_count:block_count
                            ]
                        )
                        if self.mtp_full_manager is not None
                        else ()
                    ),
                )
            )
        restore_start = hbm_block_count
        direct_hbm_load = bool(
            getattr(self, "l3_direct_hbm_load_enabled", False)
        )
        if direct_hbm_load:
            # Experimental direct-HBM only restores the local L2-covered
            # range. L3-only blocks remain absent from L2 until the first
            # prefill forward dumps the complete HBM prefix.
            restore_end = local_l2_block_count
            restore_ready_count = local_l2_block_count
        else:
            # Stable path loads the L3 suffix into L2, then restores HBM.
            # When HBM extends beyond the original L2 hit there is an
            # unready L2 gap. The external suffix may be copied to HBM now,
            # but the contiguous durable watermark must stay at the old L2
            # boundary until the first forward backfills that gap.
            restore_end = block_count
            restore_ready_count = (
                block_count
                if local_l2_block_count == local_block_count
                else local_l2_block_count
            )
        if restore_end < restore_start:
            restore_end = restore_start
        restore_source_ids = tuple(
            prefix_dram_block_ids[restore_start:restore_end]
        )
        if restore_source_ids or restore_ready_count:
            logical_start = (
                restore_start
                if restore_source_ids
                else restore_ready_count
            )
            self._pending_prefix_restores[request_id] = DSADRAMBlockRestore(
                request_id=request_id,
                source_dram_block_ids=restore_source_ids,
                destination_indexer_block_ids=tuple(
                    block.block_id
                    for block in indexer_blocks[restore_start:restore_end]
                ),
                destination_resident_block_ids=tuple(
                    block.block_id
                    for block in resident_blocks[restore_start:restore_end]
                ),
                destination_mtp_block_ids=(
                    tuple(
                        block.block_id
                        for block in mtp_blocks[restore_start:restore_end]
                    )
                    if self.mtp_full_manager is not None
                    else ()
                ),
                logical_block_start=logical_start,
                ready_block_count=restore_ready_count,
            )

    def take_prefix_restore_plan(
        self,
        request_id: str,
    ) -> DSADRAMBlockRestore | None:
        return self._pending_prefix_restores.pop(request_id, None)

    def take_external_prefix_load_plan(
        self,
        request_id: str,
    ) -> DSAExternalPrefixLoadPlan | None:
        return self._pending_external_prefix_loads.pop(request_id, None)

    def cache_blocks(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> None:
        """发布可复用的 HBM dense-context hash。"""

        # Indexer 在请求全生命周期中始终保持完整上下文语义。
        self.indexer_manager.cache_blocks(
            request,
            num_computed_tokens,
            retention_interval=self.retention_interval,
        )

        # resident MLA 只有 PREFILL/DENSE 表与文本逻辑块一一对应；ENTER
        # 后的 budget/tail 是稀疏物理槽，绝不能发布 prefix hash。
        state = self.request_cache_layout.get_state(request.request_id)
        if state is None:
            raise RuntimeError(
                "DSA cannot cache target HBM blocks before committing "
                f"its layout: request_id={request.request_id!r}"
            )
        if state.stage in (
            DSARequestCacheStage.PREFILL,
            DSARequestCacheStage.DENSE_DECODE,
        ):
            self.resident_manager.cache_blocks(
                request,
                num_computed_tokens,
                retention_interval=self.retention_interval,
            )

        if self.mtp_full_manager is not None:
            aligned_tokens = (
                int(num_computed_tokens)
                // self.scheduler_block_size
                * self.scheduler_block_size
            )
            mtp_tokens_to_cache = aligned_tokens
            if aligned_tokens > 0:
                mtp_tokens_to_cache = min(
                    int(num_computed_tokens),
                    aligned_tokens + self.mtp_full_manager.block_size,
                )
            self.mtp_full_manager.cache_blocks(
                request,
                mtp_tokens_to_cache,
                retention_interval=self.retention_interval,
            )

    def get_num_blocks_to_allocate_by_group(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> tuple[int, ...]:
        requirements: list[int] = []
        for group_id, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                requirement = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    (),
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                requirement = manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[group_id],
                    total_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
            requirements.append(requirement)
        return tuple(requirements)

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # 保留 vLLM 公共接口的“总待分配块数”语义；真正 admission 使用
        # get_num_blocks_to_allocate_by_group + can_allocate。
        return sum(
            self.get_num_blocks_to_allocate_by_group(
                request_id=request_id,
                num_tokens=num_tokens,
                new_computed_blocks=new_computed_blocks,
                num_encoder_tokens=num_encoder_tokens,
                total_computed_tokens=total_computed_tokens,
                num_tokens_main_model=num_tokens_main_model,
                apply_admission_cap=apply_admission_cap,
            )
        )

    def can_allocate(
        self,
        requirements: Sequence[int],
        *,
        reserved_blocks: int = 0,
    ) -> bool:
        if len(requirements) != len(self.physical_block_pools):
            raise ValueError(
                "DSA component requirement count does not match physical "
                f"pools: requirements={len(requirements)}, "
                f"pools={len(self.physical_block_pools)}"
            )
        if reserved_blocks:
            raise RuntimeError("DSA split KV-cache does not support connector block reservations")
        return all(
            required <= pool.get_num_free_blocks()
            for required, pool in zip(
                requirements,
                self.physical_block_pools,
                strict=True,
            )
        )

    def can_admit_dense_request(
        self,
        *,
        request_id: str,
        num_tokens: int,
        total_computed_tokens: int = 0,
    ) -> bool:
        """检查一个 prefill 请求能否同时装入所有 full-context group。"""

        empty_blocks = tuple(() for _ in range(self.num_single_type_manager))
        requirements = self.get_num_blocks_to_allocate_by_group(
            request_id=request_id,
            num_tokens=min(int(num_tokens), self.max_model_len),
            new_computed_blocks=empty_blocks,
            num_encoder_tokens=0,
            total_computed_tokens=int(total_computed_tokens),
            num_tokens_main_model=min(
                int(num_tokens),
                self.max_model_len,
            ),
            apply_admission_cap=True,
        )
        return self.can_allocate(requirements)

    def get_request_cache_state(
        self,
        request_id: str,
    ) -> DSARequestCacheState | None:
        return self.request_cache_layout.get_state(request_id)

    def ensure_request_dram_blocks(
        self,
        request_id: str,
        logical_block_count: int,
    ) -> tuple[int, tuple[int, ...]]:
        """让 DRAM 目录覆盖本轮可持久化的 durable full-block 前缀。"""

        indexer_blocks = self.indexer_manager.req_to_blocks.get(
            request_id,
            (),
        )
        logical_block_count = int(logical_block_count)
        if logical_block_count > len(indexer_blocks):
            raise RuntimeError(
                "DSA durable DRAM prefix exceeds the committed HBM Indexer "
                "table: "
                f"request_id={request_id!r}, "
                f"durable_blocks={logical_block_count}, "
                f"indexer_blocks={len(indexer_blocks)}"
            )
        return self.dram_directory.ensure_request_blocks(
            request_id,
            logical_block_count,
        )

    def take_unpublished_dram_assignment(
        self,
        request_id: str,
    ) -> tuple[int, tuple[int, ...]]:
        return self.dram_directory.take_unpublished_assignment(request_id)

    def mark_request_dram_ready(
        self,
        request: Request,
        ready_block_count: int,
    ) -> None:
        self.dram_directory.mark_ready_prefix(
            request,
            ready_block_count,
        )

    def release_prefill_resident_blocks(
        self,
        request_id: str,
        *,
        num_computed_tokens: int,
    ) -> bool:
        """释放已 dump 的 dense-prefill MLA 满块，只保留不满尾块。

        当前首版只支持单 stream、同步 scheduler。数据面接通后，上一轮
        prefill 的 MLA 写入和 full-block dump 在同一 NPU stream 上有序
        完成，scheduler 收到输出后才能调用这里。若未来改为异步多流，
        必须先增加 event/readiness 协议，不能继续直接释放这些 HBM 块。
        """

        req_blocks = self.resident_manager.req_to_blocks.get(request_id)
        if not req_blocks:
            return False

        block_size = self.resident_manager.block_size
        tail_index = int(num_computed_tokens) // block_size if num_computed_tokens % block_size else None
        if num_computed_tokens < 0 or (tail_index is not None and tail_index >= len(req_blocks)):
            raise RuntimeError("DSA prefill tail is outside the computed resident block table")
        # Lookahead may allocate later blocks that contain no computed KV yet.
        preserved_tail = req_blocks[tail_index] if tail_index is not None else None
        releasable = [block for index, block in enumerate(req_blocks) if index != tail_index]
        self.resident_manager.req_to_blocks[request_id] = [preserved_tail] if preserved_tail is not None else []
        if releasable:
            self.resident_manager.block_pool.free_blocks(reversed(releasable))
        self.resident_manager.num_cached_block.pop(request_id, None)
        self.request_cache_layout.mark_prefill_resident_released(request_id)
        return True

    def free(self, request_id: str) -> None:
        self._pending_prefix_hits.pop(request_id, None)
        self._pending_prefix_restores.pop(request_id, None)
        self._pending_external_prefix_loads.pop(request_id, None)
        super().free(request_id)
        self.dram_directory.free_request(request_id)
        self.request_cache_layout.free(request_id)

    def get_num_common_prefix_blocks(
        self,
        running_request_id: str,
    ) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        del block_hashes, max_cache_hit_length
        if self.enable_caching:
            raise RuntimeError(
                "DSA prefix lookup requires the Request-aware "
                "prepare_prefix_cache_hit() path"
            )
        return tuple([] for _ in range(self.num_single_type_manager)), 0
