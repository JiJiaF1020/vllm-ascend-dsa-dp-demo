# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA target 两平面与可选 MTP full-cache 的请求级分配。

各 single-type manager 复用 vLLM 的 full-attention 块生命周期，但由
DSA coordinator 注入彼此独立的 BlockPool。顶层仍使用 vLLM 原生
``KVCacheManager``；Ascend patch 仅在其 coordinator
为 DSA 多物理池实现时，把 ``allocate_slots`` 中的标量容量判断替换为
component-wise 判断和生命周期分配。这样 scheduler 无论在 patch 前后
导入 manager 类，都不会产生一套平行的 manager 类型。

请求阶段由 ``request_cache_layout`` 以 plan/commit 协议统一规划；本文件只把
规划结果翻译成各 manager 的实际 block 分配、ENTER 收缩和 steady grow。
本地 prefix hit 仍走同一分配器：查询阶段的 L2、成对 target HBM 与 MTP
候选只有在所有 component admission 成功后才提交；HBM pair 可以独立于
L2 延伸，缺失的 L2 bundle 由首个 prefill forward 的既有 dump 回填。
MTP3 的 resident 表在 ENTER 时一次排成 ``budget + parity0 + parity1``；
保留下来的 dense 尾块按 logical-block parity 放入固定列，之后不再换址。
任何容量不足都必须发生在物理表修改前，不能让失败重试提前推进阶段。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm.logger import logger
from vllm.v1.core.kv_cache_manager import (
    KVCacheBlocks,
    KVCacheManager,
)
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.request import Request

from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCachePlan,
    DSARequestCacheStage,
)

if TYPE_CHECKING:
    from vllm_ascend.dsa_offload.kv_cache_coordinator import (
        DSAKVCacheCoordinator,
    )


class DSAIndexerKVCacheManager(FullAttentionManager):
    """管理完整上下文 Indexer dense plane 的 block table。"""


class DSAResidentMLAKVCacheManager(FullAttentionManager):
    """管理 dense-prefill / sparse-decode 共用的 MLA resident plane。"""


def _require_initial_runtime_contract(
    *,
    manager: KVCacheManager,
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    new_computed_blocks: KVCacheBlocks | None,
    num_new_computed_tokens: int,
    num_lookahead_tokens: int,
    num_external_computed_tokens: int,
    delay_cache_blocks: bool,
    num_encoder_tokens: int,
    reserved_blocks: int,
) -> tuple[Sequence[KVCacheBlock], ...]:
    """把首版支持矩阵在分配边界再次收紧，避免静默走错布局。"""

    if not 0 <= int(num_lookahead_tokens) <= 3:
        raise RuntimeError(
            "DSA compromise MTP supports at most three lookahead tokens: "
            f"got={num_lookahead_tokens}"
        )

    unsupported = {
        "delay_cache_blocks": delay_cache_blocks,
        "num_encoder_tokens": num_encoder_tokens,
        "reserved_blocks": reserved_blocks,
    }
    active = {name: value for name, value in unsupported.items() if bool(value)}
    if active:
        raise RuntimeError(f"DSA request cache layout received unsupported allocation features: {active}")

    if new_computed_blocks is None:
        blocks = manager.empty_kv_cache_blocks.blocks
    else:
        blocks = new_computed_blocks.blocks
    prefix_plan = coordinator.get_pending_prefix_hit(request.request_id)
    expected_local_tokens = (
        prefix_plan.local_computed_tokens
        if prefix_plan is not None
        else 0
    )
    expected_external_tokens = (
        prefix_plan.external_computed_tokens
        if prefix_plan is not None
        else 0
    )
    if (
        int(num_new_computed_tokens) != expected_local_tokens
        or int(num_external_computed_tokens)
        != expected_external_tokens
    ):
        raise RuntimeError(
            "DSA prefix-hit token count disagrees with its scheduler plan: "
            f"request_id={request.request_id!r}, "
            f"allocated_local={num_new_computed_tokens}, "
            f"allocated_external={num_external_computed_tokens}, "
            f"planned_local={expected_local_tokens}, "
            f"planned_external={expected_external_tokens}"
        )
    for group_id, group_blocks in enumerate(blocks):
        if group_id == coordinator.group_ids.indexer:
            expected = (
                prefix_plan.hbm_indexer_blocks
                if prefix_plan is not None
                else ()
            )
            if tuple(group_blocks) != expected:
                raise RuntimeError(
                    "DSA Indexer HBM prefix blocks disagree with the "
                    "pending hit plan"
                )
        elif group_id == coordinator.group_ids.resident_mla:
            expected = (
                prefix_plan.hbm_resident_blocks
                if prefix_plan is not None
                else ()
            )
            if tuple(group_blocks) != expected:
                raise RuntimeError(
                    "DSA resident HBM prefix blocks disagree with the "
                    "pending hit plan"
                )
        elif group_id == coordinator.group_ids.mtp_full:
            expected_mtp = (
                prefix_plan.mtp_blocks if prefix_plan is not None else ()
            )
            if tuple(group_blocks) != expected_mtp:
                raise RuntimeError(
                    "DSA MTP prefix blocks disagree with the pending hit plan"
                )
        elif group_blocks:
            raise RuntimeError(
                "DSA prefix hit contains blocks for an unknown cache group"
            )
    return blocks


def get_dsa_computed_blocks(
    manager: KVCacheManager,
    request: Request,
) -> tuple[KVCacheBlocks, int]:
    """Request-aware DSA lookup：联合 target HBM/L2 与可选 MTP。"""

    from vllm_ascend.dsa_offload.kv_cache_coordinator import (
        DSAKVCacheCoordinator,
    )

    coordinator = manager.coordinator
    if not isinstance(coordinator, DSAKVCacheCoordinator):
        raise RuntimeError(
            "get_dsa_computed_blocks requires DSAKVCacheCoordinator"
        )
    if not manager.enable_caching or request.skip_reading_prefix_cache:
        return manager.empty_kv_cache_blocks, 0

    max_cache_hit_length = request.num_tokens - 1
    computed_blocks, num_computed_tokens = (
        coordinator.prepare_prefix_cache_hit(
            request,
            max_cache_hit_length,
        )
    )
    prefix_plan = coordinator.get_pending_prefix_hit(request.request_id)
    logger.info(
        "DSA prefix lookup: request_id=%s hit=%s "
        "effective_tokens=%d hbm_pair_blocks=%d l2_blocks=%d "
        "hbm_only_blocks=%d "
        "mtp_blocks=%d "
        "request_tokens=%d dp_rank=%d",
        request.request_id,
        bool(num_computed_tokens),
        num_computed_tokens,
        (
            prefix_plan.hbm_block_count
            if prefix_plan is not None
            else 0
        ),
        len(prefix_plan.dram_blocks) if prefix_plan is not None else 0,
        (
            max(
                prefix_plan.hbm_block_count - prefix_plan.l2_block_count,
                0,
            )
            if prefix_plan is not None
            else 0
        ),
        len(prefix_plan.mtp_blocks) if prefix_plan is not None else 0,
        request.num_tokens,
        getattr(coordinator, "data_parallel_rank", 0),
    )
    if manager.log_stats:
        assert manager.prefix_cache_stats is not None
        manager.prefix_cache_stats.record(
            num_tokens=request.num_tokens,
            num_hits=num_computed_tokens,
            preempted=request.num_preemptions > 0,
        )
    return (
        manager.create_kv_cache_blocks(computed_blocks),
        num_computed_tokens,
    )


def _make_group_blocks(
    coordinator: DSAKVCacheCoordinator,
    *,
    indexer_blocks: list[KVCacheBlock],
    resident_blocks: list[KVCacheBlock],
    mtp_full_blocks: list[KVCacheBlock] | None = None,
) -> tuple[list[KVCacheBlock], ...]:
    blocks: list[list[KVCacheBlock]] = [[] for _ in range(coordinator.num_single_type_manager)]
    blocks[coordinator.group_ids.indexer] = indexer_blocks
    blocks[coordinator.group_ids.resident_mla] = resident_blocks
    if coordinator.group_ids.mtp_full is not None:
        if mtp_full_blocks is None:
            raise RuntimeError(
                "DSA MTP full group allocation did not publish its new blocks"
            )
        blocks[coordinator.group_ids.mtp_full] = mtp_full_blocks
    return tuple(blocks)


def _allocate_dense_plan(
    manager: KVCacheManager,
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
    computed_blocks: tuple[Sequence[KVCacheBlock], ...],
    num_new_computed_tokens: int,
    num_external_computed_tokens: int,
) -> KVCacheBlocks | None:
    total_prefix_tokens = (
        int(num_new_computed_tokens)
        + int(num_external_computed_tokens)
    )
    requirements = coordinator.get_num_blocks_to_allocate_by_group(
        request_id=request.request_id,
        num_tokens=plan.indexer_tokens_need_slot,
        new_computed_blocks=computed_blocks,
        num_encoder_tokens=0,
        total_computed_tokens=(
            request.num_computed_tokens + total_prefix_tokens
        ),
        num_tokens_main_model=plan.indexer_tokens_need_slot,
    )
    if not coordinator.can_allocate(requirements) or not (
        coordinator.can_commit_prefix_cache_hit(
            request.request_id,
            local_computed_tokens=num_new_computed_tokens,
            external_computed_tokens=num_external_computed_tokens,
            required_dram_block_count=(
                plan.indexer_tokens_need_slot
                // coordinator.dram_block_size
            ),
        )
    ):
        return None

    if total_prefix_tokens:
        coordinator.allocate_new_computed_blocks(
            request_id=request.request_id,
            new_computed_blocks=computed_blocks,
            # target HBM pair 直接 touch；其余 L2/L3 prefix 按已计算
            # 长度建账并由 allocate_new_blocks 分配 restore 目的块。
            # 若 HBM 比 L2 更长，L2 缺口在首个 forward 中回填。
            num_local_computed_tokens=total_prefix_tokens,
            num_external_computed_tokens=0,
        )

    new_blocks = coordinator.allocate_new_blocks(
        request.request_id,
        plan.indexer_tokens_need_slot,
        plan.indexer_tokens_need_slot,
    )
    coordinator.commit_prefix_cache_hit(
        request.request_id,
        num_new_computed_tokens,
        num_external_computed_tokens,
        plan.indexer_tokens_need_slot // coordinator.dram_block_size,
    )
    coordinator.request_cache_layout.commit(plan)
    return manager.create_kv_cache_blocks(new_blocks)


def _resident_blocks_to_preserve(
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
) -> tuple[tuple[int, KVCacheBlock], ...] | None:
    """在改表前定位未 durable 的已验证满块和部分尾块，不保留 lookahead。"""

    blocks = coordinator.resident_manager.req_to_blocks.get(request.request_id, [])
    block_size = coordinator.resident_manager.block_size
    computed = int(request.num_computed_tokens)
    full_count, partial = divmod(computed, block_size)
    state = coordinator.request_cache_layout.get_state(request.request_id)
    if state is not None and state.prefill_resident_released:
        if len(blocks) != int(bool(partial)):
            raise RuntimeError("DSA released prefill resident table lost its computed tail")
        preserved = ((full_count, blocks[0]),) if partial else ()
    else:
        first = full_count
        if plan.resident_tail_block_count > 1:
            first = coordinator.dram_directory.get_ready_block_count(request.request_id)
        end = full_count + int(bool(partial))
        if first > full_count or end > len(blocks) or end - first > plan.resident_tail_block_count:
            raise RuntimeError("DSA ENTER pending resident source exceeds its allocated fixed tail capacity")
        preserved = tuple((index, blocks[index]) for index in range(first, end))
    if any(block.is_null or block.ref_cnt < 1 for _, block in preserved):
        raise RuntimeError("DSA ENTER cannot repurpose a free or null resident tail")
    if len({block.block_id for _, block in preserved}) != len(preserved):
        raise RuntimeError("DSA ENTER pending resident sources alias one physical block")
    if any(index == full_count and block.ref_cnt != 1 for index, block in preserved):
        raise RuntimeError("DSA ENTER partial resident tail must be private")
    if any(block.ref_cnt > 1 for _, block in preserved):
        # A completed but not-yet-durable block may have been prefix-touched
        # by a peer. One more dense step dumps it without making it writable.
        return None
    return preserved


def _release_resident_blocks_for_enter(
    coordinator: DSAKVCacheCoordinator,
    request_id: str,
    *,
    preserved_blocks: tuple[tuple[int, KVCacheBlock], ...],
) -> None:
    resident_manager = coordinator.resident_manager
    req_blocks = resident_manager.req_to_blocks.get(request_id, [])
    preserved_ids = {block.block_id for _, block in preserved_blocks}
    releasable = [block for block in req_blocks if block.block_id not in preserved_ids]
    # A pending full block may already carry a dense HBM hash. Its bytes stay
    # live for prewrite dump, but its new writable parity role must not be hit.
    hashed_ids = {block.block_id for _, block in preserved_blocks if block.block_hash is not None}
    if hashed_ids:
        resident_manager.block_pool.evict_blocks(hashed_ids)
    resident_manager.req_to_blocks[request_id] = []
    if releasable:
        resident_manager.block_pool.free_blocks(reversed(releasable))
    resident_manager.num_cached_block.pop(request_id, None)


def _allocate_sparse_plan(
    manager: KVCacheManager,
    coordinator: DSAKVCacheCoordinator,
    request: Request,
    plan: DSARequestCachePlan,
) -> KVCacheBlocks | None:
    indexer_manager = coordinator.indexer_manager
    resident_manager = coordinator.resident_manager
    mtp_full_manager = coordinator.mtp_full_manager
    request_id = request.request_id
    empty: tuple[KVCacheBlock, ...] = ()

    indexer_blocks_needed = indexer_manager.get_num_blocks_to_allocate(
        request_id=request_id,
        num_tokens=plan.indexer_tokens_need_slot,
        new_computed_blocks=empty,
        total_computed_tokens=request.num_computed_tokens,
        num_tokens_main_model=plan.indexer_tokens_need_slot,
    )
    if indexer_blocks_needed > indexer_manager.block_pool.get_num_free_blocks():
        return None
    mtp_full_blocks_needed = 0
    if mtp_full_manager is not None:
        mtp_full_blocks_needed = (
            mtp_full_manager.get_num_blocks_to_allocate(
                request_id=request_id,
                num_tokens=plan.indexer_tokens_need_slot,
                new_computed_blocks=empty,
                total_computed_tokens=request.num_computed_tokens,
                num_tokens_main_model=plan.indexer_tokens_need_slot,
            )
        )
        if (
            mtp_full_blocks_needed
            > mtp_full_manager.block_pool.get_num_free_blocks()
        ):
            return None

    resident_new_blocks_needed = 0
    preserved_blocks: tuple[tuple[int, KVCacheBlock], ...] = ()
    releasable_resident_blocks = 0
    if plan.replace_resident_blocks:
        existing_resident_blocks = resident_manager.req_to_blocks.get(
            request_id,
            [],
        )
        preserved = _resident_blocks_to_preserve(coordinator, request, plan)
        if preserved is None:
            dense_plan = replace(
                plan,
                stage=DSARequestCacheStage.DENSE_DECODE,
                sparse_budget_tokens=0,
                resident_valid_tokens=-1,
                preserve_resident_tail_block=False,
            )
            return _allocate_dense_plan(
                manager, coordinator, request, dense_plan,
                tuple(() for _ in range(coordinator.num_single_type_manager)), 0, 0,
            )
        preserved_blocks = preserved
        preserved_ids = {block.block_id for _, block in preserved_blocks}
        releasable_resident_blocks = sum(
            block.ref_cnt == 1 and not block.is_null and block.block_id not in preserved_ids
            for block in existing_resident_blocks
        )
        budget_blocks = (
            plan.sparse_budget_tokens + resident_manager.block_size - 1
        ) // resident_manager.block_size
        target_resident_blocks = (
            budget_blocks + plan.resident_tail_block_count
        )
        resident_new_blocks_needed = target_resident_blocks - len(preserved_blocks)
        resident_available_after_release = (
            resident_manager.block_pool.get_num_free_blocks() + releasable_resident_blocks
        )
        if resident_new_blocks_needed > resident_available_after_release:
            return None
    else:
        expected_resident_blocks = (
            (
                plan.sparse_budget_tokens
                + resident_manager.block_size
                - 1
            )
            // resident_manager.block_size
            + plan.resident_tail_block_count
        )
        actual_resident_blocks = len(resident_manager.req_to_blocks.get(request_id, ()))
        if actual_resident_blocks != expected_resident_blocks:
            raise RuntimeError(
                "DSA steady sparse resident block table changed "
                f"unexpectedly: req_id={request_id}, "
                f"actual={actual_resident_blocks}, "
                f"expected={expected_resident_blocks}"
            )

    new_indexer_blocks = indexer_manager.allocate_new_blocks(
        request_id,
        plan.indexer_tokens_need_slot,
        plan.indexer_tokens_need_slot,
    )
    new_mtp_full_blocks: list[KVCacheBlock] | None = None
    if mtp_full_manager is not None:
        new_mtp_full_blocks = mtp_full_manager.allocate_new_blocks(
            request_id,
            plan.indexer_tokens_need_slot,
            plan.indexer_tokens_need_slot,
        )
        if len(new_mtp_full_blocks) != mtp_full_blocks_needed:
            raise RuntimeError(
                "DSA MTP full allocation disagreed with its capacity "
                f"precheck: req_id={request_id}, "
                f"allocated={len(new_mtp_full_blocks)}, "
                f"expected={mtp_full_blocks_needed}"
            )
    new_resident_blocks: list[KVCacheBlock] = []
    if plan.replace_resident_blocks:
        _release_resident_blocks_for_enter(
            coordinator,
            request_id,
            preserved_blocks=preserved_blocks,
        )
        budget_block_count = (
            plan.sparse_budget_tokens + resident_manager.block_size - 1
        ) // resident_manager.block_size
        resident_slots_to_allocate = (
            budget_block_count
            + plan.resident_tail_block_count
            - len(preserved_blocks)
        ) * resident_manager.block_size
        new_resident_blocks = resident_manager.allocate_new_blocks(
            request_id,
            resident_slots_to_allocate,
            resident_slots_to_allocate,
        )
        if len(new_resident_blocks) != resident_new_blocks_needed:
            raise RuntimeError(
                "DSA ENTER resident allocation disagreed with its "
                f"capacity precheck: req_id={request_id}, "
                f"allocated={len(new_resident_blocks)}, "
                f"expected={resident_new_blocks_needed}"
            )
        resident_blocks = resident_manager.req_to_blocks[request_id]
        budget_blocks = list(resident_blocks[:budget_block_count])
        # The vLLM free queue intentionally releases request blocks in reverse
        # eviction order. Once the pool wraps, a DSA ENTER allocation can
        # therefore switch from ascending to descending physical IDs even
        # though its logical sparse payload is identical. Canonicalize only
        # the hashless sparse-budget region; parity tail order remains fixed.
        budget_blocks.sort(key=lambda block: int(block.block_id))
        new_tail_blocks = iter(resident_blocks[budget_block_count:])
        tail_blocks: list[KVCacheBlock | None] = [
            None
        ] * plan.resident_tail_block_count
        for preserved_logical_block, preserved_tail in preserved_blocks:
            tail_blocks[
                preserved_logical_block
                % plan.resident_tail_block_count
            ] = preserved_tail
        for tail_index, tail_block in enumerate(tail_blocks):
            if tail_block is None:
                tail_blocks[tail_index] = next(new_tail_blocks)
        try:
            next(new_tail_blocks)
        except StopIteration:
            pass
        else:
            raise RuntimeError(
                "DSA ENTER allocated more tail blocks than its fixed layout "
                f"requires: req_id={request_id}"
            )
        resident_manager.req_to_blocks[request_id][:] = [
            *budget_blocks,
            *(block for block in tail_blocks if block is not None),
        ]

    coordinator.request_cache_layout.commit(plan)
    return manager.create_kv_cache_blocks(
        _make_group_blocks(
            coordinator,
            indexer_blocks=new_indexer_blocks,
            resident_blocks=new_resident_blocks,
            mtp_full_blocks=new_mtp_full_blocks,
        )
    )


def allocate_dsa_slots(
    manager: KVCacheManager,
    request: Request,
    num_new_tokens: int,
    num_new_computed_tokens: int = 0,
    new_computed_blocks: KVCacheBlocks | None = None,
    num_lookahead_tokens: int = 0,
    num_external_computed_tokens: int = 0,
    delay_cache_blocks: bool = False,
    num_encoder_tokens: int = 0,
    full_sequence_must_fit: bool = False,
    reserved_blocks: int = 0,
) -> KVCacheBlocks | None:
    """为 DSA 多物理池执行请求生命周期感知的分配。

    DENSE/PREFILL 同时扩展 target Indexer、target resident 和可选 MTP
    full-cache plane；ENTER 收缩 target resident 表并保留尚未 durable 的
    已验证满块及不满尾块；共享待 dump 满块先走一轮 DENSE 后再收缩。
    SPARSE 继续扩展完整上下文 Indexer 与 MTP full cache，
    target resident 的 budget+固定尾区物理表保持不变。所有阶段都先完成
    component-wise 容量预检，再修改 block table，最后 commit 请求账本。
    """

    from vllm_ascend.dsa_offload.kv_cache_coordinator import (
        DSAKVCacheCoordinator,
    )

    coordinator = manager.coordinator
    if not isinstance(coordinator, DSAKVCacheCoordinator):
        raise RuntimeError(f"allocate_dsa_slots requires DSAKVCacheCoordinator, got {type(coordinator).__name__}")

    if num_new_tokens <= 0:
        raise ValueError("DSA num_new_tokens must be greater than 0")

    new_computed_block_list = _require_initial_runtime_contract(
        manager=manager,
        coordinator=coordinator,
        request=request,
        new_computed_blocks=new_computed_blocks,
        num_new_computed_tokens=num_new_computed_tokens,
        num_lookahead_tokens=num_lookahead_tokens,
        num_external_computed_tokens=num_external_computed_tokens,
        delay_cache_blocks=delay_cache_blocks,
        num_encoder_tokens=num_encoder_tokens,
        reserved_blocks=reserved_blocks,
    )

    if full_sequence_must_fit:
        full_num_tokens = min(request.num_tokens, manager.max_model_len)
        full_requirements = coordinator.get_num_blocks_to_allocate_by_group(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=0,
            total_computed_tokens=(
                request.num_computed_tokens
                + num_new_computed_tokens
                + num_external_computed_tokens
            ),
            num_tokens_main_model=full_num_tokens,
            apply_admission_cap=True,
        )
        if not coordinator.can_allocate(full_requirements):
            return None

    plan = coordinator.request_cache_layout.plan(
        request,
        num_new_tokens=num_new_tokens,
        num_lookahead_tokens=num_lookahead_tokens,
        num_computed_tokens=(
            request.num_computed_tokens
            + num_new_computed_tokens
            + num_external_computed_tokens
        ),
        max_model_len=manager.max_model_len,
    )
    if plan.stage in (
        DSARequestCacheStage.PREFILL,
        DSARequestCacheStage.DENSE_DECODE,
    ):
        allocated = _allocate_dense_plan(
            manager,
            coordinator,
            request,
            plan,
            new_computed_block_list,
            num_new_computed_tokens,
            num_external_computed_tokens,
        )
    else:
        allocated = _allocate_sparse_plan(
            manager,
            coordinator,
            request,
            plan,
        )
    if allocated is not None and manager.enable_caching:
        num_tokens_to_cache = min(
            request.num_computed_tokens
            + num_new_computed_tokens
            + num_external_computed_tokens
            + num_new_tokens,
            request.num_tokens,
        )
        coordinator.cache_blocks(request, num_tokens_to_cache)
    return allocated
