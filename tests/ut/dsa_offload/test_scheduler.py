# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingType
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from vllm_ascend.dsa_offload.l3.metadata import DSAL3ConnectorMetadata
from vllm_ascend.dsa_offload.l3.scheduler import DSAL3StoreScheduler
from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)
from vllm_ascend.dsa_offload.scheduler import (
    DSAOffloadScheduler,
    _is_prefill_request,
    _resolve_dsa_l3_store_extender,
)
from vllm_ascend.dsa_offload.scheduler_output import (
    DSADRAMBlockAssignment,
    DSAOffloadSchedulerOutput,
    DSARequestCacheLayoutProjection,
    DSAResidentBlockTableReplacement,
)


@dataclass(eq=False)
class _Request:
    request_id: str
    num_prompt_tokens: int
    num_computed_tokens: int
    num_output_tokens: int
    num_tokens: int
    status: RequestStatus = RequestStatus.RUNNING
    num_output_placeholders: int = 0
    max_tokens: int = 128
    spec_token_ids: list[int] = field(default_factory=list)
    sampling_params: object | None = None

    @property
    def num_tokens_with_spec(self) -> int:
        return self.num_tokens + len(self.spec_token_ids)

    def is_finished(self) -> bool:
        return False


class _AdmissionGate:
    def __init__(self, can_admit: bool) -> None:
        self.can_admit = can_admit
        self.calls = 0

    def can_admit_dense_request(self, **kwargs) -> bool:
        del kwargs
        self.calls += 1
        return self.can_admit


def _make_scheduler() -> DSAOffloadScheduler:
    scheduler = object.__new__(DSAOffloadScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._inflight_prefills = set()
    scheduler.requests = {}
    scheduler.max_num_running_reqs = 4
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.max_model_len = 16384
    scheduler.block_size = 128
    scheduler.num_spec_tokens = 0
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=0,
        enable_chunked_prefill=False,
    )
    scheduler._pause_state = PauseState.UNPAUSED
    return scheduler


def test_prefill_detection_uses_uncomputed_prompt_tokens() -> None:
    partial = _Request("partial", 1000, 512, 1, 1000)
    boundary = _Request("boundary", 1000, 1000, 0, 1000)

    assert _is_prefill_request(partial)
    assert not _is_prefill_request(boundary)


def test_l3_config_and_scheduler_capability_must_match() -> None:
    def extender(**_kwargs) -> None:
        return None

    connector = SimpleNamespace(extend_dsa_store_jobs=extender)
    config = SimpleNamespace(kv_connector="DSAL3Connector")

    assert _resolve_dsa_l3_store_extender(config, connector) is extender
    assert _resolve_dsa_l3_store_extender(None, None) is None
    with pytest.raises(RuntimeError, match="configured=True, capable=False"):
        _resolve_dsa_l3_store_extender(config, None)
    with pytest.raises(RuntimeError, match="configured=False, capable=True"):
        _resolve_dsa_l3_store_extender(None, connector)


def test_pending_l3_store_keeps_zero_token_connector_progress_alive() -> None:
    scheduler = _make_scheduler()
    scheduler.finished_req_ids = set()
    scheduler.dsa_l3_enabled = True
    pending = True
    scheduler.connector = SimpleNamespace(
        has_pending_transfers=lambda: pending,
    )

    assert scheduler.has_requests()
    pending = False
    assert not scheduler.has_requests()


def test_external_reset_does_not_mutate_local_cache_while_store_is_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _make_scheduler()
    scheduler.dsa_l3_enabled = True
    pending = True
    scheduler.connector = SimpleNamespace(
        has_pending_transfers=lambda: pending,
    )
    base_calls: list[tuple[bool, bool]] = []

    def base_reset(
        _self,
        reset_running_requests=False,
        reset_connector=False,
    ) -> bool:
        base_calls.append((reset_running_requests, reset_connector))
        return True

    monkeypatch.setattr(Scheduler, "reset_prefix_cache", base_reset)

    assert not scheduler.reset_prefix_cache(reset_connector=True)
    assert not base_calls

    pending = False
    assert scheduler.reset_prefix_cache(reset_connector=True)
    assert base_calls == [(False, True)]


def test_withhold_decode_restores_original_running_order() -> None:
    scheduler = _make_scheduler()
    decode_a = _Request("decode-a", 1000, 1000, 1, 1001)
    prefill = _Request("prefill", 1000, 512, 0, 1000)
    decode_b = _Request("decode-b", 1000, 1000, 1, 1001)
    scheduler.running = [decode_a, prefill, decode_b]
    scheduler.requests = {request.request_id: request for request in scheduler.running}

    restore = scheduler._withhold_decode_running_for_prefill()

    assert restore is not None
    assert scheduler.running == [prefill]
    assert scheduler.max_num_running_reqs == 2
    restore()
    assert scheduler.running == [decode_a, prefill, decode_b]
    assert scheduler.max_num_running_reqs == 4


def test_waiting_prefill_only_blocks_decode_when_both_pools_can_admit() -> None:
    scheduler = _make_scheduler()
    decode = _Request("decode", 1000, 1000, 1, 1001)
    prefill = _Request(
        "prefill",
        4096,
        0,
        0,
        4096,
        status=RequestStatus.WAITING,
    )
    scheduler.running = [decode]
    scheduler.requests = {
        decode.request_id: decode,
        prefill.request_id: prefill,
    }
    scheduler.waiting.add_request(prefill)

    gate = _AdmissionGate(False)
    scheduler.dsa_coordinator = gate  # type: ignore[assignment]
    assert not scheduler._has_schedulable_waiting_prefill(8192)

    gate.can_admit = True
    assert scheduler._has_schedulable_waiting_prefill(8192)
    assert gate.calls == 2


def test_waiting_prefill_can_advance_one_chunk() -> None:
    scheduler = _make_scheduler()
    scheduler.scheduler_config.enable_chunked_prefill = True
    prefill = _Request(
        "prefill",
        4096,
        0,
        0,
        4096,
        status=RequestStatus.WAITING,
    )
    scheduler.requests = {prefill.request_id: prefill}
    scheduler.waiting.add_request(prefill)
    gate = _AdmissionGate(True)
    scheduler.dsa_coordinator = gate  # type: ignore[assignment]

    assert scheduler._has_schedulable_waiting_prefill(1024)
    assert gate.calls == 1


def test_decode_barrier_temporarily_hides_both_waiting_queues() -> None:
    scheduler = _make_scheduler()
    waiting = _Request(
        "waiting",
        1000,
        0,
        0,
        1000,
        status=RequestStatus.WAITING,
    )
    scheduler.waiting.add_request(waiting)
    original_waiting = scheduler.waiting
    original_skipped = scheduler.skipped_waiting

    restore = scheduler._withhold_waiting_for_decode()

    assert not scheduler.waiting
    assert not scheduler.skipped_waiting
    restore()
    assert scheduler.waiting is original_waiting
    assert scheduler.skipped_waiting is original_skipped


def test_ready_decode_detection_ignores_prefill_rows() -> None:
    scheduler = _make_scheduler()
    scheduler.running = [_Request("prefill", 1000, 512, 0, 1000)]
    assert not scheduler._has_ready_decode_work()

    scheduler.running.append(_Request("decode", 1000, 1000, 1, 1001))
    assert scheduler._has_ready_decode_work()


def test_compromise_mtp_rejects_non_greedy_request(monkeypatch) -> None:
    scheduler = _make_scheduler()
    scheduler.num_spec_tokens = 3
    request = _Request(
        "random",
        100,
        0,
        0,
        100,
        status=RequestStatus.WAITING,
        sampling_params=SimpleNamespace(sampling_type=SamplingType.RANDOM),
    )
    monkeypatch.setattr(
        Scheduler,
        "add_request",
        lambda self, req: None,
    )

    with pytest.raises(ValueError, match="temperature=0.0"):
        scheduler.add_request(request)  # type: ignore[arg-type]


def test_no_waiting_requests_use_the_upstream_schedule_fast_path(
    monkeypatch,
) -> None:
    scheduler = _make_scheduler()
    scheduler.running = [_Request("decode", 1000, 1000, 1, 1001)]
    marker = object()

    def fail_if_gate_is_evaluated() -> bool:
        raise AssertionError("steady decode should not evaluate the DSA phase gate")

    scheduler._has_running_prefill_work = fail_if_gate_is_evaluated  # type: ignore[method-assign]
    scheduler._attach_dsa_cache_layout = lambda output: output  # type: ignore[method-assign]
    monkeypatch.setattr(Scheduler, "schedule", lambda self: marker)

    assert scheduler.schedule() is marker


def test_inflight_prefill_skips_fast_path_and_withholds_decode(
    monkeypatch,
) -> None:
    scheduler = _make_scheduler()
    prefill = _Request("prefill", 4096, 1024, 0, 4096)
    decode = _Request("decode", 1000, 1000, 1, 1001)
    scheduler.running = [decode, prefill]
    scheduler.requests = {
        request.request_id: request for request in scheduler.running
    }
    scheduler._inflight_prefills.add(prefill)
    marker = object()
    observed_running: list[_Request] = []

    def schedule_prefill_only(self) -> object:
        observed_running.extend(self.running)
        return marker

    scheduler._attach_dsa_cache_layout = lambda output: output  # type: ignore[method-assign]
    monkeypatch.setattr(Scheduler, "schedule", schedule_prefill_only)

    assert scheduler.schedule() is marker
    assert observed_running == [prefill]
    assert scheduler.running == [decode, prefill]


def test_final_prefill_chunk_releases_resident_blocks_after_output(
    monkeypatch,
) -> None:
    scheduler = _make_scheduler()
    request = _Request("prefill", 1000, 512, 0, 1000)
    scheduler.requests = {request.request_id: request}
    scheduler.block_size = 128
    release_calls: list[tuple[str, bool]] = []
    scheduler.dsa_coordinator = SimpleNamespace(
        enable_caching=False,
        request_cache_layout=SimpleNamespace(
            should_release_resident_after_prefill=lambda req: (
                req.num_computed_tokens >= req.num_prompt_tokens
            )
        ),
        release_prefill_resident_blocks=lambda request_id, preserve_tail_block: (
            release_calls.append((request_id, preserve_tail_block))
        ),
    )
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request.request_id: 488}
    parent_result = {0: object()}
    monkeypatch.setattr(
        Scheduler,
        "update_from_output",
        lambda self, scheduler_output, model_runner_output: parent_result,
    )

    assert scheduler.update_from_output(output, object()) is parent_result
    assert not release_calls

    request.num_computed_tokens = request.num_prompt_tokens
    assert scheduler.update_from_output(output, object()) is parent_result
    assert release_calls == [(request.request_id, True)]


def test_dram_hash_is_published_before_parent_can_free_request(
    monkeypatch,
) -> None:
    scheduler = _make_scheduler()
    request = _Request("req", 256, 256, 0, 256)
    scheduler.requests = {request.request_id: request}
    calls: list[tuple[str, int]] = []
    scheduler.dsa_coordinator = SimpleNamespace(
        enable_caching=True,
        mark_request_dram_ready=lambda req, count: calls.append(
            (req.request_id, count)
        ),
        request_cache_layout=SimpleNamespace(
            should_release_resident_after_prefill=lambda req: False
        ),
    )
    base = SchedulerOutput.make_empty()
    base.num_scheduled_tokens = {request.request_id: 1}
    base.total_num_scheduled_tokens = 1
    output = DSAOffloadSchedulerOutput.from_base(
        base,
        dsa_cache_layout=DSARequestCacheLayoutProjection(
            request_ids=(request.request_id,),
            stages=(int(DSARequestCacheStage.PREFILL),),
            target_resident_budget_tokens=(2048,),
            sparse_budget_tokens=(0,),
            resident_valid_tokens=(-1,),
            resident_block_table_replacements=(),
            dram_ready_block_counts=(2,),
        ),
    )

    def parent_update(self, scheduler_output, model_runner_output):
        assert calls == [(request.request_id, 2)]
        return {0: object()}

    monkeypatch.setattr(Scheduler, "update_from_output", parent_update)

    scheduler.update_from_output(output, object())


def test_projection_carries_enter_resident_table_and_scalar_state() -> None:
    scheduler = _make_scheduler()
    # 模拟 hybrid group 的通用 scheduler LCM 为 256，而 DSA
    # Indexer/MLA/DRAM 的物理 block size 为 128。
    scheduler.block_size = 256
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {"dense": 1, "enter": 1}
    output.total_num_scheduled_tokens = 2
    states = {
        "dense": SimpleNamespace(
            stage=DSARequestCacheStage.DENSE_DECODE,
            target_resident_budget_tokens=2048,
            sparse_budget_tokens=0,
            resident_valid_tokens=-1,
        ),
        "enter": SimpleNamespace(
            stage=DSARequestCacheStage.ENTER_SPARSE_DECODE,
            target_resident_budget_tokens=4096,
            sparse_budget_tokens=4096,
            resident_valid_tokens=4097,
        ),
    }
    scheduler.requests = {
        # 模拟 super().schedule() 已把本轮 1 token 计入 computed。
        "dense": _Request("dense", 2048, 2049, 1, 2049),
        "enter": _Request("enter", 4096, 4097, 1, 4097),
    }
    durable_counts: dict[str, int] = {}
    unpublished = {
        "dense": (0, (701, 702)),
        "enter": (2, (703,)),
    }

    def ensure_dram_blocks(
        request_id: str,
        logical_block_count: int,
    ) -> tuple[int, tuple[int, ...]]:
        durable_counts[request_id] = logical_block_count
        return logical_block_count, ()

    scheduler.dsa_coordinator = SimpleNamespace(
        enable_caching=True,
        dram_block_size=128,
        get_request_cache_state=states.get,
        ensure_request_dram_blocks=ensure_dram_blocks,
        take_unpublished_dram_assignment=unpublished.pop,
        take_prefix_restore_plan=lambda request_id: None,
        resident_manager=SimpleNamespace(
            req_to_blocks={
                "enter": [
                    SimpleNamespace(block_id=301),
                    SimpleNamespace(block_id=302),
                ]
            }
        ),
    )

    projection = scheduler._build_dsa_cache_layout_projection(output)

    assert projection.request_ids == ("dense", "enter")
    assert projection.stages == (
        int(DSARequestCacheStage.DENSE_DECODE),
        int(DSARequestCacheStage.ENTER_SPARSE_DECODE),
    )
    assert projection.target_resident_budget_tokens == (2048, 4096)
    assert projection.resident_block_table_replacements == (
        DSAResidentBlockTableReplacement(
            request_id="enter",
            block_ids=(301, 302),
        ),
    )
    assert projection.dram_block_assignments == (
        DSADRAMBlockAssignment(
            request_id="dense",
            logical_block_start=0,
            physical_block_ids=(701, 702),
        ),
        DSADRAMBlockAssignment(
            request_id="enter",
            logical_block_start=2,
            physical_block_ids=(703,),
        ),
    )
    assert durable_counts == {"dense": 16, "enter": 32}
    assert projection.dram_ready_block_counts == (16, 32)
    assert projection.num_enter_rows == 1


def test_mtp_decode_dram_assignment_excludes_provisional_tokens() -> None:
    scheduler = _make_scheduler()
    scheduler.block_size = 256
    scheduler.num_spec_tokens = 3
    # forward 开始为 255；super().schedule() 已乐观加入本轮 Q=4。
    request = _Request("mtp", 128, 259, 1, 260)
    request.spec_token_ids = [11, 12, 13]
    scheduler.requests = {request.request_id: request}
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request.request_id: 4}
    output.total_num_scheduled_tokens = 4
    observed: list[int] = []
    scheduler.dsa_coordinator = SimpleNamespace(
        enable_caching=False,
        dram_block_size=128,
        get_request_cache_state=lambda request_id: SimpleNamespace(
            stage=DSARequestCacheStage.SPARSE_DECODE,
            target_resident_budget_tokens=8192,
            sparse_budget_tokens=8192,
            resident_valid_tokens=8319,
        ),
        ensure_request_dram_blocks=(
            lambda request_id, logical_block_count: (
                observed.append(logical_block_count) or (0, ())
            )
        ),
        take_unpublished_dram_assignment=lambda request_id: (0, ()),
        take_prefix_restore_plan=lambda request_id: None,
        resident_manager=SimpleNamespace(req_to_blocks={}),
    )

    scheduler._build_dsa_cache_layout_projection(output)

    # floor(255 / 128) == 1。若错误地把本轮 4 个 target query 算进去，
    # 会提前得到 floor(259 / 128) == 2 并为未验证尾块分配 DRAM id。
    assert observed == [1]


def test_l3_projection_carries_ready_watermark_without_local_prefix_cache() -> None:
    scheduler = _make_scheduler()
    scheduler.dsa_l3_enabled = True
    request = _Request("l3", 256, 256, 0, 256)
    scheduler.requests = {request.request_id: request}
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request.request_id: 256}
    output.total_num_scheduled_tokens = 256
    scheduler.dsa_coordinator = SimpleNamespace(
        enable_caching=False,
        dram_block_size=128,
        get_request_cache_state=lambda request_id: SimpleNamespace(
            stage=DSARequestCacheStage.PREFILL,
            target_resident_budget_tokens=6144,
            sparse_budget_tokens=0,
            resident_valid_tokens=-1,
        ),
        ensure_request_dram_blocks=lambda request_id, count: (count, ()),
        take_unpublished_dram_assignment=lambda request_id: (2, ()),
        resident_manager=SimpleNamespace(req_to_blocks={}),
    )

    projection = scheduler._build_dsa_cache_layout_projection(output)

    assert projection.dram_ready_block_counts == (2,)


def test_attach_projection_populates_real_l3_connector_metadata() -> None:
    scheduler = _make_scheduler()
    schema = SimpleNamespace(
        bytes_per_logical_block=12,
        has_mtp=False,
        block_keys=lambda block_hash: (
            f"indexer:{bytes(block_hash).hex()}",
            f"mla:{bytes(block_hash).hex()}",
        ),
    )
    store_scheduler = DSAL3StoreScheduler(
        schema,
        max_inflight_jobs=2,
    )
    connector = SimpleNamespace(
        extend_dsa_store_jobs=(
            lambda **kwargs: store_scheduler.extend_store_jobs(
                metadata=kwargs["scheduler_output"].kv_connector_metadata,
                projection=kwargs["projection"],
                coordinator=kwargs["coordinator"],
                requests=kwargs["requests"],
            )
        )
    )
    scheduler.connector = connector
    scheduler.dsa_l3_enabled = True
    scheduler._dsa_l3_extend_store_jobs = connector.extend_dsa_store_jobs
    request = _Request("l3", 256, 256, 0, 256)
    request.block_hashes = [
        BlockHash(b"hash-0"),
        BlockHash(b"hash-1"),
    ]
    scheduler.requests = {request.request_id: request}
    scheduler.dsa_coordinator = SimpleNamespace(
        enable_caching=False,
        dram_block_size=128,
        get_request_cache_state=lambda request_id: SimpleNamespace(
            stage=DSARequestCacheStage.PREFILL,
            target_resident_budget_tokens=6144,
            sparse_budget_tokens=0,
            resident_valid_tokens=-1,
        ),
        ensure_request_dram_blocks=lambda request_id, count: (0, (7, 8)),
        take_unpublished_dram_assignment=lambda request_id: (0, (7, 8)),
        take_prefix_restore_plan=lambda request_id: None,
        dram_directory=SimpleNamespace(
            get_request_blocks=lambda request_id: (7, 8),
        ),
        resident_manager=SimpleNamespace(req_to_blocks={}),
    )
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request.request_id: 256}
    output.total_num_scheduled_tokens = 256
    output.kv_connector_metadata = DSAL3ConnectorMetadata()

    wrapped = scheduler._attach_dsa_cache_layout(output)

    assert isinstance(wrapped, DSAOffloadSchedulerOutput)
    assert isinstance(
        wrapped.kv_connector_metadata,
        DSAL3ConnectorMetadata,
    )
    jobs = wrapped.kv_connector_metadata.store_jobs
    assert tuple(jobs) == (0,)
    assert len(jobs[0].blocks) == 2


def test_mtp_prefill_dram_assignment_includes_current_chunk() -> None:
    scheduler = _make_scheduler()
    scheduler.block_size = 256
    scheduler.num_spec_tokens = 3
    # forward 开始为 127；prefill 的 129-token chunk 已计入 computed。
    request = _Request("prefill", 384, 256, 0, 384)
    scheduler.requests = {request.request_id: request}
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = {request.request_id: 129}
    output.total_num_scheduled_tokens = 129
    observed: list[int] = []
    scheduler.dsa_coordinator = SimpleNamespace(
        enable_caching=False,
        dram_block_size=128,
        get_request_cache_state=lambda request_id: SimpleNamespace(
            stage=DSARequestCacheStage.PREFILL,
            target_resident_budget_tokens=8192,
            sparse_budget_tokens=0,
            resident_valid_tokens=-1,
        ),
        ensure_request_dram_blocks=(
            lambda request_id, logical_block_count: (
                observed.append(logical_block_count) or (0, ())
            )
        ),
        take_unpublished_dram_assignment=lambda request_id: (0, ()),
        take_prefix_restore_plan=lambda request_id: None,
        resident_manager=SimpleNamespace(req_to_blocks={}),
    )

    scheduler._build_dsa_cache_layout_projection(output)

    # Prefill 在 cache write 后 dump，因此本轮 127+129 个 token 对应的
    # 两个 full block 都可以在 forward 前获得 DRAM assignment。
    assert observed == [2]
