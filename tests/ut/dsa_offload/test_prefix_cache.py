# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock

from vllm_ascend.dsa_offload.kv_cache_coordinator import (
    DSAKVCacheCoordinator,
    DSAPrefixCacheHitPlan,
)
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.scheduler_dram_pool import (
    DSASchedulerDRAMBlockDirectory,
)


class _HBMManager:
    def __init__(self, hits=()) -> None:
        self.hits = list(hits)
        self.block_pool = object()
        self.block_size = 128
        self.calls = []

    def find_longest_cache_hit(self, **kwargs):
        self.calls.append(kwargs)
        return (list(self.hits),)


def _bind_target_hbm(
    coordinator,
    *,
    indexer_hits=(),
    resident_hits=(),
) -> tuple[_HBMManager, _HBMManager]:
    indexer = _HBMManager(indexer_hits)
    resident = _HBMManager(resident_hits)
    coordinator.indexer_manager = indexer
    coordinator.resident_manager = resident
    groups = [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128)),
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128)),
    ]
    if coordinator.group_ids.mtp_full is not None:
        groups.append(
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=128)
            )
        )
    coordinator.kv_cache_config = SimpleNamespace(
        kv_cache_groups=groups,
    )
    return indexer, resident


def _request(request_id: str, block_count: int):
    return SimpleNamespace(
        request_id=request_id,
        block_hashes=[
            BlockHash(f"prefix-{index}".encode())
            for index in range(block_count)
        ],
    )


def _prime_dram_prefix(
    directory: DSASchedulerDRAMBlockDirectory,
    block_count: int,
):
    request = _request("source", block_count)
    directory.ensure_request_blocks(request.request_id, block_count)
    directory.mark_ready_prefix(request, block_count)
    directory.free_request(request.request_id)
    return request.block_hashes


def test_target_prefix_hit_uses_complete_dram_bundle_only() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=8,
        block_size=128,
        enable_caching=True,
    )
    hashes = _prime_dram_prefix(directory, 4)
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.dram_directory = directory
    coordinator.dram_block_size = 128
    coordinator.scheduler_block_size = 128
    coordinator.hash_block_size = 128
    coordinator.num_single_type_manager = 2
    coordinator.group_ids = SimpleNamespace(
        indexer=0,
        resident_mla=1,
        mtp_full=None,
    )
    coordinator.mtp_full_manager = None
    coordinator._pending_prefix_hits = {}
    _bind_target_hbm(coordinator)
    request = SimpleNamespace(request_id="target", block_hashes=hashes)

    groups, hit_tokens = coordinator.prepare_prefix_cache_hit(
        request,
        4 * 128,
    )

    assert hit_tokens == 4 * 128
    assert groups == ([], [])
    plan = coordinator.get_pending_prefix_hit(request.request_id)
    assert plan is not None
    assert len(plan.dram_blocks) == 4


def test_target_prefix_hit_reuses_paired_hbm_inside_l2_prefix() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=8,
        block_size=128,
        enable_caching=True,
    )
    hashes = _prime_dram_prefix(directory, 4)
    indexer_hits = tuple(KVCacheBlock(101 + index) for index in range(3))
    resident_hits = tuple(KVCacheBlock(201 + index) for index in range(2))
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.dram_directory = directory
    coordinator.dram_block_size = 128
    coordinator.scheduler_block_size = 128
    coordinator.hash_block_size = 128
    coordinator.num_single_type_manager = 2
    coordinator.group_ids = SimpleNamespace(
        indexer=0,
        resident_mla=1,
        mtp_full=None,
    )
    coordinator.mtp_full_manager = None
    coordinator._pending_prefix_hits = {}
    _bind_target_hbm(
        coordinator,
        indexer_hits=indexer_hits,
        resident_hits=resident_hits,
    )
    request = SimpleNamespace(request_id="target", block_hashes=hashes)

    groups, hit_tokens = coordinator.prepare_prefix_cache_hit(
        request,
        4 * 128,
    )

    assert hit_tokens == 4 * 128
    assert groups[0] == list(indexer_hits[:2])
    assert groups[1] == list(resident_hits)
    plan = coordinator.get_pending_prefix_hit("target")
    assert plan is not None
    assert len(plan.dram_blocks) == 4
    assert plan.hbm_block_count == 2
    assert plan.hbm_indexer_blocks == indexer_hits[:2]
    assert plan.hbm_resident_blocks == resident_hits


def test_hbm_pair_can_extend_beyond_complete_l2_backing() -> None:
    plan = DSAPrefixCacheHitPlan(
        request_id="target",
        num_computed_tokens=2 * 128,
        dram_blocks=(KVCacheBlock(10),),
        hbm_indexer_blocks=(KVCacheBlock(20), KVCacheBlock(21)),
        hbm_resident_blocks=(KVCacheBlock(30), KVCacheBlock(31)),
    )

    assert plan.l2_block_count == 1
    assert plan.hbm_block_count == 2


def test_hbm_pair_extends_effective_local_hit_beyond_l2() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=8,
        block_size=128,
        enable_caching=True,
    )
    hashes = _prime_dram_prefix(directory, 1)
    hashes.extend(
        BlockHash(f"hbm-only-{index}".encode())
        for index in range(1, 4)
    )
    indexer_hits = tuple(
        KVCacheBlock(101 + index) for index in range(3)
    )
    resident_hits = tuple(
        KVCacheBlock(201 + index) for index in range(3)
    )
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.dram_directory = directory
    coordinator.dram_block_size = 128
    coordinator.scheduler_block_size = 128
    coordinator.hash_block_size = 128
    coordinator.num_single_type_manager = 2
    coordinator.group_ids = SimpleNamespace(
        indexer=0,
        resident_mla=1,
        mtp_full=None,
    )
    coordinator.mtp_full_manager = None
    coordinator._pending_prefix_hits = {}
    _bind_target_hbm(
        coordinator,
        indexer_hits=indexer_hits,
        resident_hits=resident_hits,
    )
    request = SimpleNamespace(request_id="target", block_hashes=hashes)

    groups, hit_tokens = coordinator.prepare_prefix_cache_hit(
        request,
        4 * 128,
    )

    assert hit_tokens == 3 * 128
    assert groups[0] == list(indexer_hits)
    assert groups[1] == list(resident_hits)
    plan = coordinator.get_pending_prefix_hit("target")
    assert plan is not None
    assert plan.l2_block_count == 1
    assert plan.hbm_block_count == 3

    external_tokens = coordinator.prepare_external_prefix_cache_hit(
        request,
        local_computed_tokens=hit_tokens,
        external_block_count=1,
    )
    assert external_tokens == 128
    extended = coordinator.get_pending_prefix_hit("target")
    assert extended is not None
    assert extended.local_computed_tokens == 3 * 128
    assert extended.external_block_hashes == (bytes(hashes[3]),)


def test_pure_hbm_pair_is_visible_when_l2_has_no_hit() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=8,
        block_size=128,
        enable_caching=True,
    )
    indexer_hits = (KVCacheBlock(101), KVCacheBlock(102))
    resident_hits = (KVCacheBlock(201), KVCacheBlock(202))
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.dram_directory = directory
    coordinator.dram_block_size = 128
    coordinator.scheduler_block_size = 128
    coordinator.hash_block_size = 128
    coordinator.num_single_type_manager = 2
    coordinator.group_ids = SimpleNamespace(
        indexer=0,
        resident_mla=1,
        mtp_full=None,
    )
    coordinator.mtp_full_manager = None
    coordinator._pending_prefix_hits = {}
    _bind_target_hbm(
        coordinator,
        indexer_hits=indexer_hits,
        resident_hits=resident_hits,
    )
    request = _request("target", 3)

    groups, hit_tokens = coordinator.prepare_prefix_cache_hit(
        request,
        3 * 128,
    )

    assert hit_tokens == 2 * 128
    assert groups == (list(indexer_hits), list(resident_hits))
    plan = coordinator.get_pending_prefix_hit("target")
    assert plan is not None
    assert plan.l2_block_count == 0
    assert plan.hbm_block_count == 2


def test_external_prefix_extends_local_dram_candidate_by_hash_suffix() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=8,
        block_size=128,
        enable_caching=True,
    )
    hashes = _prime_dram_prefix(directory, 2)
    hashes.extend(
        [BlockHash(b"external-2"), BlockHash(b"external-3")]
    )
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.dram_directory = directory
    coordinator.dram_block_size = 128
    coordinator.scheduler_block_size = 128
    coordinator.hash_block_size = 128
    coordinator.num_single_type_manager = 2
    coordinator.group_ids = SimpleNamespace(
        indexer=0,
        resident_mla=1,
        mtp_full=None,
    )
    coordinator.mtp_full_manager = None
    coordinator._pending_prefix_hits = {}
    _bind_target_hbm(coordinator)
    request = SimpleNamespace(request_id="target", block_hashes=hashes)

    coordinator.prepare_prefix_cache_hit(request, 4 * 128)
    external_tokens = coordinator.prepare_external_prefix_cache_hit(
        request,
        local_computed_tokens=2 * 128,
        external_block_count=2,
    )

    assert external_tokens == 2 * 128
    plan = coordinator.get_pending_prefix_hit("target")
    assert plan is not None
    assert plan.local_computed_tokens == 2 * 128
    assert plan.external_computed_tokens == 2 * 128
    assert plan.external_block_hashes == (
        bytes(hashes[2]),
        bytes(hashes[3]),
    )


def test_dram_prefix_admission_counts_evictable_hits_and_new_suffix() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=8,
        block_size=128,
        enable_caching=True,
    )
    hashes = _prime_dram_prefix(directory, 4)
    hits = directory.find_longest_cache_hit(hashes, 4 * 128)

    assert directory.can_attach_cached_prefix(
        hits,
        new_block_count=4,
    )
    assert not directory.can_attach_cached_prefix(
        hits,
        new_block_count=5,
    )


def test_mtp_hbm_triple_and_l2_apply_one_boundary_drop() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=8,
        block_size=128,
        enable_caching=True,
    )
    hashes = _prime_dram_prefix(directory, 4)
    mtp_blocks = [KVCacheBlock(101 + index) for index in range(3)]

    class _MTPManager:
        def __init__(self) -> None:
            self.block_pool = object()
            self.calls = []

        def find_longest_cache_hit(self, **kwargs):
            self.calls.append(kwargs)
            # 模拟基线已经执行过 EAGLE drop 后的三个有效块。
            return (list(mtp_blocks),)

    mtp_manager = _MTPManager()
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.dram_directory = directory
    coordinator.dram_block_size = 128
    coordinator.scheduler_block_size = 128
    coordinator.hash_block_size = 128
    coordinator.num_single_type_manager = 3
    coordinator.group_ids = SimpleNamespace(
        indexer=0,
        resident_mla=1,
        mtp_full=2,
    )
    coordinator.mtp_full_manager = mtp_manager
    _bind_target_hbm(
        coordinator,
        indexer_hits=tuple(
            KVCacheBlock(201 + index) for index in range(4)
        ),
        resident_hits=tuple(
            KVCacheBlock(301 + index) for index in range(4)
        ),
    )
    coordinator._pending_prefix_hits = {}
    request = SimpleNamespace(request_id="target", block_hashes=hashes)

    groups, hit_tokens = coordinator.prepare_prefix_cache_hit(
        request,
        5 * 128,
    )

    assert hit_tokens == 3 * 128
    assert groups[0] == coordinator.indexer_manager.hits[:3]
    assert groups[1] == coordinator.resident_manager.hits[:3]
    assert groups[2] == mtp_blocks
    assert mtp_manager.calls[0]["drop_eagle_block"] is True
    assert mtp_manager.calls[0]["max_length"] == 5 * 128
    plan = coordinator.get_pending_prefix_hit(request.request_id)
    assert plan is not None
    assert len(plan.dram_blocks) == 3
    assert plan.mtp_blocks == tuple(mtp_blocks)


def test_mtp_complete_l2_bundle_replays_without_mtp_hbm_hit() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=8,
        block_size=128,
        enable_caching=True,
    )
    hashes = _prime_dram_prefix(directory, 4)
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.dram_directory = directory
    coordinator.dram_block_size = 128
    coordinator.scheduler_block_size = 128
    coordinator.hash_block_size = 128
    coordinator.num_single_type_manager = 3
    coordinator.group_ids = SimpleNamespace(
        indexer=0,
        resident_mla=1,
        mtp_full=2,
    )
    coordinator.mtp_full_manager = _HBMManager()
    _bind_target_hbm(coordinator)
    coordinator._pending_prefix_hits = {}
    request = SimpleNamespace(request_id="target", block_hashes=hashes)

    groups, hit_tokens = coordinator.prepare_prefix_cache_hit(
        request,
        4 * 128,
    )

    assert hit_tokens == 3 * 128
    assert groups == ([], [], [])
    plan = coordinator.get_pending_prefix_hit(request.request_id)
    assert plan is not None
    assert len(plan.dram_blocks) == 3


def test_mtp_external_suffix_drops_only_its_final_boundary_block() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=8,
        block_size=128,
        enable_caching=True,
    )
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.dram_directory = directory
    coordinator.dram_block_size = 128
    coordinator.mtp_full_manager = _HBMManager()
    coordinator._pending_prefix_hits = {}
    request = _request("target", 5)

    external_tokens = coordinator.prepare_external_prefix_cache_hit(
        request,
        local_computed_tokens=0,
        external_block_count=5,
    )

    assert external_tokens == 4 * 128
    plan = coordinator.get_pending_prefix_hit(request.request_id)
    assert plan is not None
    assert len(plan.external_block_hashes) == 4


def test_mtp_l2_l3_merged_prefix_drops_one_block_total() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=8,
        block_size=128,
        enable_caching=True,
    )
    hashes = list(_prime_dram_prefix(directory, 4))
    hashes.extend((BlockHash(b"prefix-4"), BlockHash(b"prefix-5")))
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.dram_directory = directory
    coordinator.dram_block_size = 128
    coordinator.scheduler_block_size = 128
    coordinator.hash_block_size = 128
    coordinator.num_single_type_manager = 3
    coordinator.group_ids = SimpleNamespace(
        indexer=0,
        resident_mla=1,
        mtp_full=2,
    )
    coordinator.mtp_full_manager = _HBMManager()
    _bind_target_hbm(coordinator)
    coordinator._pending_prefix_hits = {}
    request = SimpleNamespace(request_id="target", block_hashes=hashes)

    _, local_tokens = coordinator.prepare_prefix_cache_hit(
        request,
        6 * 128,
    )
    external_tokens = coordinator.prepare_external_prefix_cache_hit(
        request,
        local_computed_tokens=local_tokens,
        # L3 starts at logical block 3 and sees blocks 3, 4, 5.
        external_block_count=3,
    )

    assert local_tokens == 3 * 128
    assert external_tokens == 2 * 128
    plan = coordinator.get_pending_prefix_hit(request.request_id)
    assert plan is not None
    assert plan.num_computed_tokens == 5 * 128
    assert plan.external_block_hashes == tuple(
        bytes(block_hash) for block_hash in hashes[3:5]
    )


def test_target_hbm_hashes_only_dense_resident_blocks() -> None:
    class _CacheRecorder:
        def __init__(self) -> None:
            self.calls = []
            self.block_size = 128

        def cache_blocks(self, request, tokens, **kwargs) -> None:
            self.calls.append((request.request_id, tokens, kwargs))

    indexer = _CacheRecorder()
    resident = _CacheRecorder()
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.indexer_manager = indexer
    coordinator.resident_manager = resident
    coordinator.mtp_full_manager = None
    coordinator.scheduler_block_size = 128
    coordinator.retention_interval = None
    state = SimpleNamespace(stage=DSARequestCacheStage.PREFILL)
    coordinator.request_cache_layout = SimpleNamespace(
        get_state=lambda request_id: state,
    )
    request = SimpleNamespace(request_id="req")

    coordinator.cache_blocks(request, 256)
    assert len(indexer.calls) == 1
    assert len(resident.calls) == 1

    state.stage = DSARequestCacheStage.ENTER_SPARSE_DECODE
    coordinator.cache_blocks(request, 384)
    assert len(indexer.calls) == 2
    assert len(resident.calls) == 1

    mtp = _CacheRecorder()
    coordinator.mtp_full_manager = mtp
    coordinator.cache_blocks(request, 512)
    assert len(indexer.calls) == 3
    assert len(resident.calls) == 1
    assert len(mtp.calls) == 1
