# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

from vllm_ascend.dsa_offload.kv_cache_coordinator import (
    DSAKVCacheCoordinator,
    DSAPrefixCacheHitPlan,
)
from vllm_ascend.dsa_offload.kv_cache_manager import allocate_dsa_slots
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCachePlanner,
    DSARequestCacheStage,
)


@dataclass(frozen=True)
class _Block:
    block_id: int


class _Pool:
    def __init__(self, num_blocks: int, *, id_base: int) -> None:
        self._free = num_blocks
        self._next_id = id_base

    def get_num_free_blocks(self) -> int:
        return self._free

    def allocate(self, count: int) -> list[_Block]:
        if count > self._free:
            raise RuntimeError("test pool exhausted")
        blocks = [_Block(self._next_id + offset) for offset in range(count)]
        self._next_id += count
        self._free -= count
        return blocks

    def free_blocks(self, blocks) -> None:
        self._free += len(list(blocks))


class _SingleTypeManager:
    block_size = 128

    def __init__(self, pool: _Pool) -> None:
        self.block_pool = pool
        self.enable_caching = False
        self.req_to_blocks = defaultdict(list)
        self.num_cached_block: dict[str, int] = {}

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks,
        total_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del (
            new_computed_blocks,
            total_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        required = (num_tokens + self.block_size - 1) // self.block_size
        return max(required - len(self.req_to_blocks[request_id]), 0)

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[_Block]:
        del num_tokens_main_model
        count = self.get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            (),
            0,
            num_tokens,
        )
        blocks = self.block_pool.allocate(count)
        self.req_to_blocks[request_id].extend(blocks)
        return blocks

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        del num_local_computed_tokens, num_external_computed_tokens
        self.req_to_blocks[request_id].extend(new_computed_blocks)
        self.num_cached_block[request_id] = len(new_computed_blocks)

    def free(self, request_id: str) -> None:
        blocks = self.req_to_blocks.pop(request_id, [])
        self.block_pool.free_blocks(reversed(blocks))
        self.num_cached_block.pop(request_id, None)


class _TopManager:
    def __init__(
        self,
        coordinator: DSAKVCacheCoordinator,
        max_model_len: int = 16384,
    ) -> None:
        self.coordinator = coordinator
        self.max_model_len = max_model_len
        self.enable_caching = False
        self.log_stats = False
        self.empty_kv_cache_blocks = SimpleNamespace(
            blocks=tuple(
                () for _ in range(coordinator.num_single_type_manager)
            )
        )

    @staticmethod
    def create_kv_cache_blocks(blocks):
        return blocks


class _DRAMDirectory:
    def __init__(self) -> None:
        self.freed_request_ids: list[str] = []
        self.attached: dict[str, tuple[_Block, ...]] = {}
        self.request_blocks: dict[str, list[_Block]] = {}
        self._next_id = 9100

    def can_attach_cached_prefix(
        self,
        blocks: tuple[_Block, ...],
        *,
        new_block_count: int = 0,
    ) -> bool:
        del blocks, new_block_count
        return True

    def attach_cached_prefix(
        self,
        request_id: str,
        blocks: tuple[_Block, ...],
    ) -> None:
        self.attached[request_id] = blocks
        self.request_blocks[request_id] = list(blocks)

    def ensure_request_blocks(
        self,
        request_id: str,
        logical_block_count: int,
    ) -> tuple[int, tuple[int, ...]]:
        blocks = self.request_blocks.setdefault(request_id, [])
        start = len(blocks)
        allocated = [
            _Block(self._next_id + offset)
            for offset in range(logical_block_count - start)
        ]
        self._next_id += len(allocated)
        blocks.extend(allocated)
        return start, tuple(block.block_id for block in allocated)

    def get_request_blocks(self, request_id: str) -> tuple[int, ...]:
        return tuple(
            block.block_id for block in self.request_blocks[request_id]
        )

    def free_request(self, request_id: str) -> tuple[int, ...]:
        self.freed_request_ids.append(request_id)
        return ()


@dataclass
class _Request:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    num_tokens: int


def _make_manager(
    *,
    indexer_blocks: int = 128,
    resident_blocks: int = 64,
    mtp_full_blocks: int | None = None,
    sparse_tail_block_count: int = 1,
) -> tuple[_TopManager, DSAKVCacheCoordinator]:
    indexer_pool = _Pool(indexer_blocks, id_base=1000)
    resident_pool = _Pool(resident_blocks, id_base=2000)
    indexer_manager = _SingleTypeManager(indexer_pool)
    resident_manager = _SingleTypeManager(resident_pool)

    managers = [indexer_manager, resident_manager]
    pools = [indexer_pool, resident_pool]
    mtp_full_manager = None
    mtp_full_group_id = None
    if mtp_full_blocks is not None:
        mtp_full_pool = _Pool(mtp_full_blocks, id_base=3000)
        mtp_full_manager = _SingleTypeManager(mtp_full_pool)
        mtp_full_manager.use_eagle = True
        mtp_full_group_id = len(managers)
        managers.append(mtp_full_manager)
        pools.append(mtp_full_pool)

    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.max_model_len = 16384
    coordinator.enable_caching = False
    coordinator.dram_block_size = 128
    coordinator.num_single_type_manager = len(managers)
    coordinator.group_ids = SimpleNamespace(
        indexer=0,
        resident_mla=1,
        mtp_full=mtp_full_group_id,
    )
    coordinator.indexer_manager = indexer_manager
    coordinator.resident_manager = resident_manager
    coordinator.mtp_full_manager = mtp_full_manager
    coordinator.single_type_managers = tuple(managers)
    coordinator.physical_block_pools = tuple(pools)
    coordinator.dram_directory = _DRAMDirectory()
    coordinator._pending_prefix_hits = {}
    coordinator._pending_prefix_restores = {}
    coordinator._pending_external_prefix_loads = {}
    coordinator.request_cache_layout = DSARequestCachePlanner(
        block_size=128,
        sparse_activation_tokens=2048,
        prompt_budget_thresholds=(),
        resident_budget_tokens=(2048,),
        sparse_tail_block_count=sparse_tail_block_count,
    )
    return _TopManager(coordinator), coordinator


def test_dense_enter_sparse_keeps_indexer_full_and_resident_fixed() -> None:
    manager, coordinator = _make_manager()
    request = _Request("req", 3000, 0, 0, 3000)

    allocate_dsa_slots(manager, request, 3000)  # type: ignore[arg-type]
    assert len(coordinator.indexer_manager.req_to_blocks["req"]) == 24
    assert len(coordinator.resident_manager.req_to_blocks["req"]) == 24

    request.num_computed_tokens = 3000
    coordinator.release_prefill_resident_blocks(
        "req",
        preserve_tail_block=True,
    )
    preserved_tail = coordinator.resident_manager.req_to_blocks["req"][0]

    request.num_output_tokens = 1
    request.num_tokens = 3001
    allocate_dsa_slots(manager, request, 1)  # type: ignore[arg-type]
    state = coordinator.get_request_cache_state("req")
    assert state is not None
    assert state.stage == DSARequestCacheStage.ENTER_SPARSE_DECODE
    assert len(coordinator.indexer_manager.req_to_blocks["req"]) == 24
    assert len(coordinator.resident_manager.req_to_blocks["req"]) == 17
    assert coordinator.resident_manager.req_to_blocks["req"][-1] == preserved_tail

    resident_ids = tuple(block.block_id for block in coordinator.resident_manager.req_to_blocks["req"])
    request.num_computed_tokens = 3072
    request.num_output_tokens = 73
    request.num_tokens = 3073
    allocate_dsa_slots(manager, request, 1)  # type: ignore[arg-type]
    state = coordinator.get_request_cache_state("req")
    assert state is not None
    assert state.stage == DSARequestCacheStage.SPARSE_DECODE
    assert len(coordinator.indexer_manager.req_to_blocks["req"]) == 25
    assert tuple(block.block_id for block in coordinator.resident_manager.req_to_blocks["req"]) == resident_ids


def test_prefix_hit_allocates_target_destinations_and_commits_restore() -> None:
    manager, coordinator = _make_manager()
    request = _Request("prefix", 512, 0, 0, 512)
    dram_blocks = (_Block(9001), _Block(9002))
    coordinator._pending_prefix_hits[request.request_id] = (
        DSAPrefixCacheHitPlan(
            request_id=request.request_id,
            num_computed_tokens=256,
            dram_blocks=dram_blocks,  # type: ignore[arg-type]
        )
    )

    allocated = allocate_dsa_slots(
        manager,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        256,
        num_new_computed_tokens=256,
        new_computed_blocks=manager.empty_kv_cache_blocks,
    )

    assert allocated is not None
    assert len(coordinator.indexer_manager.req_to_blocks["prefix"]) == 4
    assert len(coordinator.resident_manager.req_to_blocks["prefix"]) == 4
    assert coordinator.dram_directory.attached["prefix"] == dram_blocks
    restore = coordinator.take_prefix_restore_plan("prefix")
    assert restore is not None
    assert restore.source_dram_block_ids == (9001, 9002)
    assert restore.destination_indexer_block_ids == (1000, 1001)
    assert restore.destination_resident_block_ids == (2000, 2001)
    assert restore.logical_block_start == 0
    assert restore.ready_block_count == 2


def test_paired_hbm_prefix_skips_covered_l2_restore_prefix() -> None:
    manager, coordinator = _make_manager()
    request = _Request("hbm-prefix", 512, 0, 0, 512)
    dram_blocks = (_Block(9001), _Block(9002))
    hbm_indexer = (_Block(110),)
    hbm_resident = (_Block(210),)
    coordinator._pending_prefix_hits[request.request_id] = (
        DSAPrefixCacheHitPlan(
            request_id=request.request_id,
            num_computed_tokens=256,
            dram_blocks=dram_blocks,  # type: ignore[arg-type]
            hbm_indexer_blocks=hbm_indexer,  # type: ignore[arg-type]
            hbm_resident_blocks=hbm_resident,  # type: ignore[arg-type]
        )
    )

    allocated = allocate_dsa_slots(
        manager,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        256,
        num_new_computed_tokens=256,
        new_computed_blocks=SimpleNamespace(
            blocks=(hbm_indexer, hbm_resident)
        ),
    )

    assert allocated is not None
    assert (
        coordinator.indexer_manager.req_to_blocks["hbm-prefix"][0]
        == hbm_indexer[0]
    )
    assert (
        coordinator.resident_manager.req_to_blocks["hbm-prefix"][0]
        == hbm_resident[0]
    )
    restore = coordinator.take_prefix_restore_plan("hbm-prefix")
    assert restore is not None
    assert restore.logical_block_start == 1
    assert restore.ready_block_count == 2
    assert restore.source_dram_block_ids == (9002,)
    assert restore.destination_indexer_block_ids == (1000,)
    assert restore.destination_resident_block_ids == (2000,)


def test_full_paired_hbm_prefix_publishes_ready_without_restore_copy() -> None:
    manager, coordinator = _make_manager()
    request = _Request("hbm-full", 512, 0, 0, 512)
    dram_blocks = (_Block(9001), _Block(9002))
    hbm_indexer = (_Block(110), _Block(111))
    hbm_resident = (_Block(210), _Block(211))
    coordinator._pending_prefix_hits[request.request_id] = (
        DSAPrefixCacheHitPlan(
            request_id=request.request_id,
            num_computed_tokens=256,
            dram_blocks=dram_blocks,  # type: ignore[arg-type]
            hbm_indexer_blocks=hbm_indexer,  # type: ignore[arg-type]
            hbm_resident_blocks=hbm_resident,  # type: ignore[arg-type]
        )
    )

    allocated = allocate_dsa_slots(
        manager,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        256,
        num_new_computed_tokens=256,
        new_computed_blocks=SimpleNamespace(
            blocks=(hbm_indexer, hbm_resident)
        ),
    )

    assert allocated is not None
    restore = coordinator.take_prefix_restore_plan("hbm-full")
    assert restore is not None
    assert restore.logical_block_start == 2
    assert restore.ready_block_count == 2
    assert restore.source_dram_block_ids == ()
    assert restore.destination_indexer_block_ids == ()
    assert restore.destination_resident_block_ids == ()


def test_hbm_only_suffix_allocates_l2_backfill_without_marking_ready() -> None:
    manager, coordinator = _make_manager()
    request = _Request("hbm-only", 512, 0, 0, 512)
    dram_blocks = (_Block(9001),)
    hbm_indexer = (_Block(110), _Block(111), _Block(112))
    hbm_resident = (_Block(210), _Block(211), _Block(212))
    coordinator._pending_prefix_hits[request.request_id] = (
        DSAPrefixCacheHitPlan(
            request_id=request.request_id,
            num_computed_tokens=384,
            dram_blocks=dram_blocks,  # type: ignore[arg-type]
            hbm_indexer_blocks=hbm_indexer,  # type: ignore[arg-type]
            hbm_resident_blocks=hbm_resident,  # type: ignore[arg-type]
        )
    )

    allocated = allocate_dsa_slots(
        manager,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        128,
        num_new_computed_tokens=384,
        new_computed_blocks=SimpleNamespace(
            blocks=(hbm_indexer, hbm_resident)
        ),
    )

    assert allocated is not None
    assert coordinator.dram_directory.attached["hbm-only"] == dram_blocks
    assert len(coordinator.dram_directory.request_blocks["hbm-only"]) == 4
    restore = coordinator.take_prefix_restore_plan("hbm-only")
    assert restore is not None
    assert restore.logical_block_start == 1
    assert restore.ready_block_count == 1
    assert restore.source_dram_block_ids == ()
    assert restore.destination_indexer_block_ids == ()
    assert restore.destination_resident_block_ids == ()


def test_hbm_only_gap_keeps_l2_watermark_before_l3_suffix() -> None:
    manager, coordinator = _make_manager()
    request = _Request("hbm-l3", 640, 0, 0, 640)
    local_dram = (_Block(9001),)
    hbm_indexer = (_Block(110), _Block(111), _Block(112))
    hbm_resident = (_Block(210), _Block(211), _Block(212))
    external_hash = b"external-hash"
    coordinator._pending_prefix_hits[request.request_id] = (
        DSAPrefixCacheHitPlan(
            request_id=request.request_id,
            num_computed_tokens=512,
            dram_blocks=local_dram,  # type: ignore[arg-type]
            hbm_indexer_blocks=hbm_indexer,  # type: ignore[arg-type]
            hbm_resident_blocks=hbm_resident,  # type: ignore[arg-type]
            external_computed_tokens=128,
            external_block_hashes=(external_hash,),
        )
    )

    allocated = allocate_dsa_slots(
        manager,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        128,
        num_new_computed_tokens=384,
        num_external_computed_tokens=128,
        new_computed_blocks=SimpleNamespace(
            blocks=(hbm_indexer, hbm_resident)
        ),
    )

    assert allocated is not None
    external = coordinator.take_external_prefix_load_plan("hbm-l3")
    assert external is not None
    assert external.logical_block_start == 3
    assert external.destination_dram_block_ids == (9102,)
    restore = coordinator.take_prefix_restore_plan("hbm-l3")
    assert restore is not None
    assert restore.logical_block_start == 3
    assert restore.ready_block_count == 1
    assert restore.source_dram_block_ids == (9102,)


def test_external_prefix_allocates_l2_suffix_and_one_combined_restore() -> None:
    manager, coordinator = _make_manager()
    request = _Request("external", 512, 0, 0, 512)
    local_dram = (_Block(9001),)
    external_hash = b"external-hash"
    coordinator._pending_prefix_hits[request.request_id] = (
        DSAPrefixCacheHitPlan(
            request_id=request.request_id,
            num_computed_tokens=256,
            dram_blocks=local_dram,  # type: ignore[arg-type]
            external_computed_tokens=128,
            external_block_hashes=(external_hash,),
        )
    )

    allocated = allocate_dsa_slots(
        manager,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        256,
        num_new_computed_tokens=128,
        num_external_computed_tokens=128,
        new_computed_blocks=manager.empty_kv_cache_blocks,
    )

    assert allocated is not None
    assert len(coordinator.indexer_manager.req_to_blocks["external"]) == 4
    assert len(coordinator.resident_manager.req_to_blocks["external"]) == 4
    external = coordinator.take_external_prefix_load_plan("external")
    assert external is not None
    assert external.logical_block_start == 1
    assert external.block_hashes == (external_hash,)
    assert external.destination_dram_block_ids == (9100,)
    restore = coordinator.take_prefix_restore_plan("external")
    assert restore is not None
    assert restore.source_dram_block_ids == (9001, 9100)
    assert restore.destination_indexer_block_ids == (1000, 1001)
    assert restore.destination_resident_block_ids == (2000, 2001)


def test_direct_hbm_external_prefix_restores_only_local_l2_prefix() -> None:
    manager, coordinator = _make_manager()
    coordinator.set_l3_direct_hbm_load_enabled(True)
    request = _Request("external-direct", 512, 0, 0, 512)
    local_dram = (_Block(9001),)
    external_hash = b"external-hash"
    coordinator._pending_prefix_hits[request.request_id] = (
        DSAPrefixCacheHitPlan(
            request_id=request.request_id,
            num_computed_tokens=256,
            dram_blocks=local_dram,  # type: ignore[arg-type]
            external_computed_tokens=128,
            external_block_hashes=(external_hash,),
        )
    )

    allocated = allocate_dsa_slots(
        manager,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        256,
        num_new_computed_tokens=128,
        num_external_computed_tokens=128,
        new_computed_blocks=manager.empty_kv_cache_blocks,
    )

    assert allocated is not None
    external = coordinator.take_external_prefix_load_plan(
        "external-direct"
    )
    assert external is not None
    assert external.destination_dram_block_ids == (9100,)
    assert external.destination_indexer_hbm_block_ids == (1001,)
    assert external.destination_resident_hbm_block_ids == (2001,)
    restore = coordinator.take_prefix_restore_plan("external-direct")
    assert restore is not None
    assert restore.logical_block_start == 0
    assert restore.ready_block_count == 1
    assert restore.source_dram_block_ids == (9001,)
    assert restore.destination_indexer_block_ids == (1000,)
    assert restore.destination_resident_block_ids == (2000,)


def test_direct_hbm_pure_l3_prefix_defers_l2_ready_to_first_forward() -> None:
    manager, coordinator = _make_manager()
    coordinator.set_l3_direct_hbm_load_enabled(True)
    request = _Request("external-only", 256, 0, 0, 256)
    coordinator._pending_prefix_hits[request.request_id] = (
        DSAPrefixCacheHitPlan(
            request_id=request.request_id,
            num_computed_tokens=128,
            dram_blocks=(),
            external_computed_tokens=128,
            external_block_hashes=(b"external-hash",),
        )
    )

    allocated = allocate_dsa_slots(
        manager,  # type: ignore[arg-type]
        request,  # type: ignore[arg-type]
        128,
        num_new_computed_tokens=0,
        num_external_computed_tokens=128,
        new_computed_blocks=manager.empty_kv_cache_blocks,
    )

    assert allocated is not None
    external = coordinator.take_external_prefix_load_plan("external-only")
    assert external is not None
    assert external.destination_indexer_hbm_block_ids == (1000,)
    assert external.destination_resident_hbm_block_ids == (2000,)
    assert coordinator.take_prefix_restore_plan("external-only") is None


def test_chunked_prefill_grows_both_dense_planes_incrementally() -> None:
    manager, coordinator = _make_manager()
    request = _Request("req", 3000, 0, 0, 3000)

    for computed, chunk_size, expected_blocks in (
        (0, 1024, 8),
        (1024, 1024, 16),
        (2048, 952, 24),
    ):
        request.num_computed_tokens = computed
        allocated = allocate_dsa_slots(  # type: ignore[arg-type]
            manager,
            request,
            chunk_size,
        )
        assert allocated is not None
        assert (
            len(coordinator.indexer_manager.req_to_blocks["req"])
            == expected_blocks
        )
        assert (
            len(coordinator.resident_manager.req_to_blocks["req"])
            == expected_blocks
        )
        state = coordinator.get_request_cache_state("req")
        assert state is not None
        assert state.stage == DSARequestCacheStage.PREFILL

    request.num_computed_tokens = request.num_prompt_tokens
    assert coordinator.request_cache_layout.should_release_resident_after_prefill(
        request
    )


def test_enter_capacity_failure_keeps_stage_and_resident_table() -> None:
    manager, coordinator = _make_manager(
        indexer_blocks=16,
        resident_blocks=32,
    )
    request = _Request("req", 2048, 0, 0, 2048)
    allocate_dsa_slots(manager, request, 2048)  # type: ignore[arg-type]
    resident_ids = tuple(block.block_id for block in coordinator.resident_manager.req_to_blocks["req"])

    request.num_computed_tokens = 2048
    request.num_output_tokens = 1
    request.num_tokens = 2049
    assert (
        allocate_dsa_slots(  # type: ignore[arg-type]
            manager,
            request,
            1,
        )
        is None
    )

    state = coordinator.get_request_cache_state("req")
    assert state is not None
    assert state.stage == DSARequestCacheStage.PREFILL
    assert tuple(block.block_id for block in coordinator.resident_manager.req_to_blocks["req"]) == resident_ids


def test_block_aligned_prefill_enter_allocates_a_new_tail_block() -> None:
    manager, coordinator = _make_manager()
    request = _Request("req", 2048, 0, 0, 2048)
    allocate_dsa_slots(manager, request, 2048)  # type: ignore[arg-type]

    request.num_computed_tokens = 2048
    assert coordinator.release_prefill_resident_blocks(
        "req",
        preserve_tail_block=False,
    )
    assert not coordinator.resident_manager.req_to_blocks["req"]

    request.num_output_tokens = 1
    request.num_tokens = 2049
    allocated = allocate_dsa_slots(  # type: ignore[arg-type]
        manager,
        request,
        1,
    )

    assert allocated is not None
    state = coordinator.get_request_cache_state("req")
    assert state is not None
    assert state.stage == DSARequestCacheStage.ENTER_SPARSE_DECODE
    assert state.resident_valid_tokens == 2049
    assert len(coordinator.resident_manager.req_to_blocks["req"]) == 17


def test_mtp_enter_installs_two_fixed_parity_tail_blocks() -> None:
    manager, coordinator = _make_manager(
        mtp_full_blocks=128,
        sparse_tail_block_count=2,
    )
    request = _Request("req", 3000, 0, 0, 3000)
    allocate_dsa_slots(manager, request, 3000)  # type: ignore[arg-type]

    request.num_computed_tokens = 3000
    coordinator.release_prefill_resident_blocks(
        "req",
        preserve_tail_block=True,
    )
    preserved_tail = coordinator.resident_manager.req_to_blocks["req"][0]

    request.num_output_tokens = 1
    request.num_tokens = 3001
    allocate_dsa_slots(  # type: ignore[arg-type]
        manager,
        request,
        4,
        num_lookahead_tokens=3,
    )

    resident_blocks = coordinator.resident_manager.req_to_blocks["req"]
    assert len(resident_blocks) == 18
    # logical block 23 is odd, therefore the preserved dense tail must occupy
    # the fixed parity-1 column after the 16 budget blocks.
    assert resident_blocks[17] == preserved_tail

    resident_ids = tuple(block.block_id for block in resident_blocks)
    request.num_computed_tokens = 3004
    request.num_output_tokens = 5
    request.num_tokens = 3005
    allocate_dsa_slots(  # type: ignore[arg-type]
        manager,
        request,
        4,
        num_lookahead_tokens=3,
    )
    assert tuple(
        block.block_id
        for block in coordinator.resident_manager.req_to_blocks["req"]
    ) == resident_ids


def test_mtp_full_plane_stays_dense_after_target_enters_sparse() -> None:
    manager, coordinator = _make_manager(
        mtp_full_blocks=128,
        sparse_tail_block_count=2,
    )
    mtp_manager = coordinator.mtp_full_manager
    assert mtp_manager is not None
    request = _Request("req", 3000, 0, 0, 3000)

    allocate_dsa_slots(manager, request, 3000)  # type: ignore[arg-type]
    assert len(coordinator.indexer_manager.req_to_blocks["req"]) == 24
    assert len(coordinator.resident_manager.req_to_blocks["req"]) == 24
    assert len(mtp_manager.req_to_blocks["req"]) == 24

    request.num_computed_tokens = 3000
    coordinator.release_prefill_resident_blocks(
        "req",
        preserve_tail_block=True,
    )

    request.num_output_tokens = 1
    request.num_tokens = 3001
    allocate_dsa_slots(  # type: ignore[arg-type]
        manager,
        request,
        4,
        num_lookahead_tokens=3,
    )
    resident_ids = tuple(
        block.block_id
        for block in coordinator.resident_manager.req_to_blocks["req"]
    )
    assert len(resident_ids) == 18
    assert len(coordinator.indexer_manager.req_to_blocks["req"]) == 24
    assert len(mtp_manager.req_to_blocks["req"]) == 24

    request.num_computed_tokens = 3072
    request.num_output_tokens = 73
    request.num_tokens = 3073
    allocated = allocate_dsa_slots(  # type: ignore[arg-type]
        manager,
        request,
        4,
        num_lookahead_tokens=3,
    )
    assert allocated is not None
    assert len(allocated[0]) == 1
    assert len(allocated[1]) == 0
    assert len(allocated[2]) == 1
    assert len(coordinator.indexer_manager.req_to_blocks["req"]) == 25
    assert len(mtp_manager.req_to_blocks["req"]) == 25
    assert tuple(
        block.block_id
        for block in coordinator.resident_manager.req_to_blocks["req"]
    ) == resident_ids


def test_dense_component_failure_does_not_allocate_or_commit() -> None:
    manager, coordinator = _make_manager(
        indexer_blocks=32,
        resident_blocks=1,
        mtp_full_blocks=32,
    )
    request = _Request("req", 256, 0, 0, 256)

    assert (
        allocate_dsa_slots(  # type: ignore[arg-type]
            manager,
            request,
            256,
        )
        is None
    )
    assert coordinator.request_cache_layout.get_state("req") is None
    assert not coordinator.indexer_manager.req_to_blocks["req"]
    assert not coordinator.resident_manager.req_to_blocks["req"]
    assert coordinator.mtp_full_manager is not None
    assert not coordinator.mtp_full_manager.req_to_blocks["req"]


def test_mtp_component_failure_does_not_allocate_or_commit() -> None:
    manager, coordinator = _make_manager(
        indexer_blocks=32,
        resident_blocks=32,
        mtp_full_blocks=1,
    )
    request = _Request("req", 256, 0, 0, 256)

    assert (
        allocate_dsa_slots(  # type: ignore[arg-type]
            manager,
            request,
            256,
        )
        is None
    )
    assert coordinator.request_cache_layout.get_state("req") is None
    assert not coordinator.indexer_manager.req_to_blocks["req"]
    assert not coordinator.resident_manager.req_to_blocks["req"]
    assert coordinator.mtp_full_manager is not None
    assert not coordinator.mtp_full_manager.req_to_blocks["req"]


def test_free_clears_all_physical_planes_and_request_cache_state() -> None:
    manager, coordinator = _make_manager(mtp_full_blocks=64)
    request = _Request("req", 512, 0, 0, 512)
    allocate_dsa_slots(manager, request, 512)  # type: ignore[arg-type]

    coordinator.free("req")

    assert "req" not in coordinator.indexer_manager.req_to_blocks
    assert "req" not in coordinator.resident_manager.req_to_blocks
    assert coordinator.mtp_full_manager is not None
    assert "req" not in coordinator.mtp_full_manager.req_to_blocks
    assert coordinator.request_cache_layout.get_state("req") is None
    assert coordinator.dram_directory.freed_request_ids == ["req"]
