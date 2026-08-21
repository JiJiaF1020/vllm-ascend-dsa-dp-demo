# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm_ascend.dsa_offload.scheduler_dram_pool import (
    DSASchedulerDRAMBlockDirectory,
    calculate_dram_capacity_plan,
    calculate_dram_usable_blocks,
)
from vllm_ascend.dsa_offload.worker_dram_store import DSAWorkerDRAMStore


def _cpu_arena(
    block_shape: tuple[int, ...],
    dtype: torch.dtype,
    capacity: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(
        (capacity, *block_shape),
        dtype=dtype,
        device=device,
    )


def _make_store() -> DSAWorkerDRAMStore:
    return DSAWorkerDRAMStore(
        usable_blocks=8,
        storage_rows=3,
        max_logical_blocks=6,
        device=torch.device("cpu"),
        arena_factory=_cpu_arena,
    )


def test_dram_capacity_uses_fractional_multiplier_with_ceil() -> None:
    assert calculate_dram_usable_blocks(7, 1.5) == 11


def test_scheduler_capacity_downsizes_but_keeps_full_indexer_mirror() -> None:
    plan = calculate_dram_capacity_plan(
        indexer_num_blocks=8,
        block_multiple=2.0,
        bytes_per_logical_block=100,
        tensor_parallel_size=2,
        # Per rank fits block 0 + 11 usable blocks.
        host_available_bytes=2 * 1200,
    )

    assert plan.minimum_usable_blocks == 8
    assert plan.desired_usable_blocks == 16
    assert plan.usable_blocks == 11
    assert plan.was_downsized
    assert plan.per_rank_reserved_bytes == 1200


def test_scheduler_capacity_rejects_less_than_indexer_mirror() -> None:
    with pytest.raises(RuntimeError, match="HBM Indexer capacity"):
        calculate_dram_capacity_plan(
            indexer_num_blocks=8,
            block_multiple=1.5,
            bytes_per_logical_block=100,
            tensor_parallel_size=2,
            # Per rank fits block 0 + 7 usable blocks only.
            host_available_bytes=2 * 800,
        )


def test_scheduler_capacity_clamps_configured_multiple_to_indexer_mirror() -> None:
    plan = calculate_dram_capacity_plan(
        indexer_num_blocks=8,
        block_multiple=0.5,
        bytes_per_logical_block=100,
        tensor_parallel_size=2,
        host_available_bytes=2 * 2000,
    )

    assert plan.desired_usable_blocks == 4
    assert plan.usable_blocks == 8
    assert plan.effective_block_multiple == 1.0


def test_layer_arenas_follow_resident_cache_block_shapes() -> None:
    store = _make_store()
    resident_nope = torch.empty(4, 128, 1, 512)
    resident_rope = torch.empty(4, 128, 1, 64)

    store.add_layer(
        layer_id=0,
        resident_nope_cache=resident_nope,
        resident_rope_cache=resident_rope,
    )
    arenas = store.get_layer_arenas(0)

    assert arenas.nope.shape == (9, 128, 1, 512)
    assert arenas.rope.shape == (9, 128, 1, 64)
    assert torch.count_nonzero(arenas.nope[0]).item() == 0
    assert torch.count_nonzero(arenas.rope[0]).item() == 0
    assert arenas.resident_source_bytes[0].data_ptr() == resident_nope.data_ptr()
    assert arenas.resident_source_bytes[1].data_ptr() == resident_rope.data_ptr()


def test_packed_c8_layer_uses_an_opaque_byte_arena() -> None:
    store = _make_store()
    resident_packed = torch.empty(4, 128, 1, 656, dtype=torch.int8)

    store.add_packed_layer(
        layer_id=0,
        resident_packed_cache=resident_packed,
    )
    arenas = store.get_layer_arenas(0)

    assert arenas.nope is None
    assert arenas.rope is None
    assert arenas.packed is not None
    assert arenas.packed.shape == (9, 128, 1, 656)
    assert arenas.packed.dtype == torch.int8
    assert torch.count_nonzero(arenas.packed[0]).item() == 0
    assert arenas.resident_source_bytes[0].data_ptr() == resident_packed.data_ptr()


def test_complete_layer_adds_indexer_byte_arenas_without_copy() -> None:
    resident_packed = torch.empty(4, 128, 1, 656, dtype=torch.int8)
    indexer_key = torch.empty(12, 128, 1, 128, dtype=torch.int8)
    indexer_scale = torch.empty(12, 128, 1, 1, dtype=torch.float32)
    expected_bytes = (
        resident_packed[0].numel() * resident_packed.element_size()
        + indexer_key[0].numel() * indexer_key.element_size()
        + indexer_scale[0].numel() * indexer_scale.element_size()
    )
    store = DSAWorkerDRAMStore(
        usable_blocks=8,
        storage_rows=3,
        max_logical_blocks=6,
        device=torch.device("cpu"),
        arena_factory=_cpu_arena,
        expected_bytes_per_logical_block=expected_bytes,
    )

    store.add_packed_layer(
        layer_id=0,
        resident_packed_cache=resident_packed,
        indexer_cache=(indexer_key, indexer_scale),
    )
    store.finalize_payload_layout()
    arenas = store.get_layer_arenas(0)

    assert tuple(tensor.shape for tensor in arenas.indexer_source_bytes) == (
        (12, 128 * 128),
        (12, 128 * 4),
    )
    assert arenas.indexer_source_bytes[0].data_ptr() == indexer_key.data_ptr()
    assert arenas.indexer_source_bytes[1].data_ptr() == indexer_scale.data_ptr()
    assert tuple(tensor.shape for tensor in arenas.indexer_dram_bytes) == (
        (9, 128 * 128),
        (9, 128 * 4),
    )


def test_scheduler_directory_assigns_once_and_reclaims_request() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=6,
        block_size=128,
    )
    assert directory.block_pool.num_gpu_blocks == 9
    assert directory.block_pool.null_block.block_id == 0

    start, first = directory.ensure_request_blocks("req-0", 3)
    assert start == 0
    assert len(first) == 3
    assert directory.num_free_blocks == 5
    assert directory.ensure_request_blocks("req-0", 3) == (3, ())
    start, suffix = directory.ensure_request_blocks("req-0", 5)
    assert start == 3
    assert len(suffix) == 2
    assert directory.get_request_blocks("req-0") == (*first, *suffix)

    assert directory.free_request("req-0") == (*first, *suffix)
    assert directory.num_free_blocks == 8


def test_worker_mirror_applies_scheduler_assignment_and_clears_row() -> None:
    store = _make_store()
    store.apply_assignment(
        pool_index=0,
        logical_block_start=0,
        physical_block_ids=(6, 7, 8),
    )
    store.apply_assignment(
        pool_index=1,
        logical_block_start=0,
        physical_block_ids=(5,),
    )
    resolved = store.resolve_blocks(
        pool_indices=np.array([0, 0, 1], dtype=np.intp),
        logical_block_indices=np.array([0, 2, 0], dtype=np.intp),
    )
    assert resolved.tolist() == [6, 8, 5]

    active = np.empty((2, 6), dtype=np.int32)
    store.gather_rows(
        pool_indices=np.array([1, 0], dtype=np.intp),
        output=active,
    )
    assert active[0, 0] == 5
    assert active[1, 0] == 6
    assert active[1, 2] == 8

    store.commit_durable_full_block_counts(
        pool_indices=np.array([0, 1], dtype=np.intp),
        full_block_counts=np.array([3, 1], dtype=np.int32),
    )
    durable = np.empty(2, dtype=np.int32)
    store.gather_durable_full_block_counts(
        pool_indices=np.array([1, 0], dtype=np.intp),
        output=durable,
    )
    assert durable.tolist() == [1, 3]

    store.release_pool_index(0)
    assert np.count_nonzero(store.logical_block_table[0]) == 0
    assert store.durable_full_block_counts[0] == 0
    assert store.logical_block_table[1, 0] != 0


def test_scheduler_dram_prefix_is_published_only_after_ready() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=6,
        block_size=128,
        enable_caching=True,
    )
    hashes = [BlockHash(f"hash-{i}".encode()) for i in range(3)]
    request = type(
        "RequestStub",
        (),
        {"request_id": "req-0", "block_hashes": hashes},
    )()
    _, assigned = directory.ensure_request_blocks("req-0", 3)

    assert directory.find_longest_cache_hit(hashes, 3 * 128) == ()
    directory.mark_ready_prefix(request, 2)
    assert tuple(
        block.block_id
        for block in directory.find_longest_cache_hit(
            hashes,
            3 * 128,
        )
    ) == assigned[:2]

    directory.free_request("req-0")
    cached = directory.find_longest_cache_hit(hashes, 3 * 128)
    assert tuple(block.block_id for block in cached) == assigned[:2]
    assert all(block.ref_cnt == 0 for block in cached)


def test_scheduler_dram_cached_prefix_is_shared_after_commit() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=8,
        max_logical_blocks=6,
        block_size=128,
        enable_caching=True,
    )
    hashes = [BlockHash(f"hash-{i}".encode()) for i in range(2)]
    request = type(
        "RequestStub",
        (),
        {"request_id": "source", "block_hashes": hashes},
    )()
    directory.ensure_request_blocks("source", 2)
    directory.mark_ready_prefix(request, 2)
    directory.free_request("source")
    hit = directory.find_longest_cache_hit(hashes, 2 * 128)

    # Lookup is side-effect free; attach is the admission commit boundary.
    assert all(block.ref_cnt == 0 for block in hit)
    directory.attach_cached_prefix("consumer", hit)
    assert all(block.ref_cnt == 1 for block in hit)
    start, assignment = directory.take_unpublished_assignment("consumer")
    assert start == 0
    assert assignment == tuple(block.block_id for block in hit)
    assert directory.take_unpublished_assignment("consumer") == (2, ())

    directory.free_request("consumer")
    assert all(block.ref_cnt == 0 for block in hit)


def test_scheduler_dram_lru_evicts_request_tail_before_prefix() -> None:
    directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=3,
        max_logical_blocks=3,
        block_size=128,
        enable_caching=True,
    )
    hashes = [BlockHash(f"hash-{i}".encode()) for i in range(3)]
    request = type(
        "RequestStub",
        (),
        {"request_id": "req", "block_hashes": hashes},
    )()
    directory.ensure_request_blocks("req", 3)
    directory.mark_ready_prefix(request, 3)
    directory.free_request("req")

    # All blocks share one release time; reverse free puts the logical tail
    # at the eviction front, preserving the longest useful prefix.
    directory.block_pool.get_new_blocks(1)
    assert len(directory.find_longest_cache_hit(hashes, 3 * 128)) == 2


def test_worker_mirror_allows_shared_prefix_block_ids() -> None:
    store = _make_store()
    store.apply_assignment(
        pool_index=0,
        logical_block_start=0,
        physical_block_ids=(6, 7),
    )
    store.apply_assignment(
        pool_index=1,
        logical_block_start=0,
        physical_block_ids=(6, 7),
    )
    assert store._block_active_refcounts[6:8].tolist() == [2, 2]

    store.release_pool_index(0)
    assert store._block_active_refcounts[6:8].tolist() == [1, 1]
    assert store.logical_block_table[1, :2].tolist() == [6, 7]
    store.release_pool_index(1)
    assert store._block_active_refcounts[6:8].tolist() == [0, 0]
