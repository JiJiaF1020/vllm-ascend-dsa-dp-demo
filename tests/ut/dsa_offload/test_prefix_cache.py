# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock

from vllm_ascend.dsa_offload.kv_cache_coordinator import (
    DSAKVCacheCoordinator,
)
from vllm_ascend.dsa_offload.scheduler_dram_pool import (
    DSASchedulerDRAMBlockDirectory,
)


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


def test_mtp_effective_hit_limits_target_dram_prefix() -> None:
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
    coordinator.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            object(),
            object(),
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=128)
            ),
        ]
    )
    coordinator._pending_prefix_hits = {}
    request = SimpleNamespace(request_id="target", block_hashes=hashes)

    groups, hit_tokens = coordinator.prepare_prefix_cache_hit(
        request,
        5 * 128,
    )

    assert hit_tokens == 3 * 128
    assert groups[0] == []
    assert groups[1] == []
    assert groups[2] == mtp_blocks
    assert mtp_manager.calls[0]["drop_eagle_block"] is True
    assert mtp_manager.calls[0]["max_length"] == 5 * 128
    plan = coordinator.get_pending_prefix_hit(request.request_id)
    assert plan is not None
    assert len(plan.dram_blocks) == 3
    assert plan.mtp_blocks == tuple(mtp_blocks)
