# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two DP-local coordinators with real vLLM BlockPools, no NPU required."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_ascend.dsa_offload.kv_cache_coordinator import DSABlockPoolView, DSAKVCacheCoordinator
from vllm_ascend.dsa_offload.kv_cache_manager import (
    DSAIndexerKVCacheManager,
    DSAResidentMLAKVCacheManager,
    allocate_dsa_slots,
    get_dsa_computed_blocks,
)
from vllm_ascend.dsa_offload.request_cache_layout import DSARequestCachePlanner, DSARequestCacheStage
from vllm_ascend.dsa_offload.scheduler_dram_pool import DSASchedulerDRAMBlockDirectory


def _make_dp(rank, *, mtp=False):
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=8, dtype=torch.float16)
    manager_types = [DSAIndexerKVCacheManager, DSAResidentMLAKVCacheManager]
    if mtp:
        manager_types.append(FullAttentionManager)
    pools = tuple(BlockPool(num_gpu_blocks=33, enable_caching=True, hash_block_size=128) for _ in manager_types)
    managers = tuple(
        cls(
            kv_cache_spec=spec,
            block_pool=pools[group],
            enable_caching=True,
            kv_cache_group_id=group,
            scheduler_block_size=128,
        )
        for group, cls in enumerate(manager_types)
    )
    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.data_parallel_rank = rank
    coordinator.max_model_len = 4096
    coordinator.enable_caching = True
    coordinator.scheduler_block_size = coordinator.hash_block_size = coordinator.dram_block_size = 128
    coordinator.retention_interval = None
    coordinator.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)] * len(managers))
    coordinator.group_ids = SimpleNamespace(indexer=0, resident_mla=1, mtp_full=2 if mtp else None)
    coordinator.num_single_type_manager = len(managers)
    coordinator.single_type_managers = managers
    coordinator.physical_block_pools = pools
    coordinator.indexer_manager, coordinator.resident_manager = managers[:2]
    coordinator.mtp_full_manager = managers[2] if mtp else None
    if mtp:
        coordinator.mtp_full_manager.use_eagle = True
    coordinator.dram_directory = DSASchedulerDRAMBlockDirectory(
        usable_blocks=32,
        max_logical_blocks=32,
        block_size=128,
        enable_caching=True,
    )
    coordinator.block_pool = DSABlockPoolView(pools, extra_prefix_cache_pools=(coordinator.dram_directory.block_pool,))
    coordinator.request_cache_layout = DSARequestCachePlanner(
        block_size=128,
        sparse_activation_tokens=256,
        prompt_budget_thresholds=(1024,),
        resident_budget_tokens=(256, 512),
        sparse_tail_block_count=2 if mtp else 1,
    )
    coordinator._pending_prefix_hits = {}
    coordinator._pending_prefix_restores = {}
    coordinator._pending_external_prefix_loads = {}
    coordinator.l3_direct_hbm_load_enabled = False
    manager = SimpleNamespace(
        coordinator=coordinator,
        max_model_len=4096,
        enable_caching=True,
        log_stats=False,
        empty_kv_cache_blocks=KVCacheBlocks(tuple([] for _ in managers)),
        create_kv_cache_blocks=KVCacheBlocks,
    )
    return manager, coordinator


def _request(request_id="same-id"):
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=513,
        num_prompt_tokens=513,
        num_computed_tokens=0,
        num_output_tokens=0,
        num_tokens_with_spec=513,
        skip_reading_prefix_cache=False,
        block_hashes=[BlockHash(f"shared-prefix-{i}".encode()) for i in range(4)],
    )


def _prime(coordinator, *, hbm=4, l2=4):
    req = _request("source")
    for group, manager in enumerate(coordinator.single_type_managers):
        blocks = manager.allocate_new_blocks(req.request_id, 513, 513)
        manager.block_pool.cache_full_blocks(req, blocks, 0, hbm, 128, group)
    if l2:
        coordinator.dram_directory.ensure_request_blocks(req.request_id, l2)
        coordinator.mark_request_dram_ready(req, l2)
    coordinator.free(req.request_id)


def test_same_request_id_hash_and_physical_ids_stay_dp_local():
    manager0, dp0 = _make_dp(0)
    manager1, dp1 = _make_dp(1)
    _prime(dp0)
    req0, req1 = _request(), _request()
    blocks0, hit0 = get_dsa_computed_blocks(manager0, req0)
    _, hit1 = get_dsa_computed_blocks(manager1, req1)
    assert hit0 == 512
    assert hit1 == 0

    _prime(dp1)
    blocks1, hit1 = get_dsa_computed_blocks(manager1, req1)
    assert hit1 == 512
    assert blocks0.get_block_ids() == blocks1.get_block_ids()
    assert blocks0.blocks[0][0] is not blocks1.blocks[0][0]
    assert allocate_dsa_slots(manager0, req0, 1, hit0, blocks0) is not None
    assert allocate_dsa_slots(manager1, req1, 1, hit1, blocks1) is not None
    peer_plan = dp1._pending_prefix_restores[req1.request_id]
    peer_ids = dp1.dram_directory.get_request_blocks(req1.request_id)
    dp0.free(req0.request_id)
    assert dp1.dram_directory.get_request_blocks(req1.request_id) == peer_ids
    assert dp1._pending_prefix_restores[req1.request_id] is peer_plan
    assert all(block.ref_cnt > 0 for block in blocks1.blocks[0])
    # Resetting DP0 must not invalidate DP1's live references or lookup index.
    assert dp0.block_pool.reset_prefix_cache()
    assert get_dsa_computed_blocks(manager0, _request("probe"))[1] == 0
    assert get_dsa_computed_blocks(manager1, _request("probe"))[1] == 512


@pytest.mark.parametrize(("hbm", "l2"), [(4, 0), (4, 2), (2, 4), (0, 4)])
def test_dp_local_prefix_commit_keeps_peer_cold_and_preserves_backfill_gap(hbm, l2):
    manager0, dp0 = _make_dp(0)
    manager1, dp1 = _make_dp(1)
    _prime(dp0, hbm=hbm, l2=l2)
    req = _request()
    blocks, hit = get_dsa_computed_blocks(manager0, req)
    assert hit == 512
    assert allocate_dsa_slots(manager0, req, 1, hit, blocks) is not None
    restore = dp0.take_prefix_restore_plan(req.request_id)
    if hbm == 4 and l2 == 0:
        assert restore is None
    else:
        assert restore.ready_block_count == l2
        assert len(restore.source_dram_block_ids) == max(l2 - hbm, 0)
    assert len(dp0.dram_directory.get_request_blocks(req.request_id)) >= 4
    assert get_dsa_computed_blocks(manager1, _request())[1] == 0
    assert dp1.dram_directory.get_request_blocks(req.request_id) == ()

    # Completion of the real first prefill publishes the missing full bundle.
    dp0.mark_request_dram_ready(req, 4)
    req.num_computed_tokens = 513
    req.num_output_tokens = 1
    req.num_tokens = req.num_tokens_with_spec = 514
    dp0.release_prefill_resident_blocks(req.request_id, preserve_tail_block=True)
    assert allocate_dsa_slots(manager0, req, 1) is not None
    assert dp0.get_request_cache_state(req.request_id).stage == DSARequestCacheStage.ENTER_SPARSE_DECODE
    assert len(dp0.resident_manager.req_to_blocks[req.request_id]) == 3
    # Sparse budget/tail blocks must never acquire dense prefix hashes.
    assert all(block.block_hash is None for block in dp0.resident_manager.req_to_blocks[req.request_id])
    assert dp1.get_request_cache_state(req.request_id) is None
    assert get_dsa_computed_blocks(manager1, _request())[1] == 0


@pytest.mark.parametrize(("hbm", "l2"), [(4, 0), (4, 2), (2, 4), (0, 4)])
def test_mtp_dp_local_triple_replay_and_dual_tail_remain_isolated(hbm, l2):
    manager0, dp0 = _make_dp(0, mtp=True)
    manager1, dp1 = _make_dp(1, mtp=True)
    _prime(dp0, hbm=hbm, l2=l2)
    req = _request()
    blocks, hit = get_dsa_computed_blocks(manager0, req)
    # The EAGLE boundary drops only one full block, local to the hit DP.
    assert hit == 384
    assert get_dsa_computed_blocks(manager1, _request())[1] == 0
    assert allocate_dsa_slots(manager0, req, 129, hit, blocks) is not None
    restore = dp0.take_prefix_restore_plan(req.request_id)
    if restore is not None:
        assert len(restore.destination_mtp_block_ids) == len(restore.source_dram_block_ids)
        assert restore.ready_block_count == max(l2 - 1, 0)
    dp0.mark_request_dram_ready(req, 4)
    req.num_computed_tokens = 513
    req.num_output_tokens = 1
    req.num_tokens = req.num_tokens_with_spec = 514
    dp0.release_prefill_resident_blocks(req.request_id, preserve_tail_block=True)
    assert allocate_dsa_slots(manager0, req, 1, num_lookahead_tokens=3) is not None
    assert len(dp0.resident_manager.req_to_blocks[req.request_id]) == 4
    assert len(dp0.mtp_full_manager.req_to_blocks[req.request_id]) == 5
    assert all(block.block_hash is None for block in dp0.resident_manager.req_to_blocks[req.request_id])
    assert dp1.get_request_cache_state(req.request_id) is None
    assert dp1.dram_directory.get_request_blocks(req.request_id) == ()
