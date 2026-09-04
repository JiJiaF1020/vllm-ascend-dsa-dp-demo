# SPDX-License-Identifier: Apache-2.0
"""P6-B管理事务UT：真实BlockPool/L3 scheduler/原生pause Future，模拟无设备的DP停波。"""

import asyncio
from concurrent.futures import Future
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.distributed as dist
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine import core as core_module
from vllm.v1.request import RequestStatus

from tests.ut.dsa_offload.test_dp_mtp_l3 import _load, _plan, _planes, _SharedPayloadBackend
from tests.ut.dsa_offload.test_l3 import _make_mtp_schema, _make_mtp_store
from tests.ut.dsa_offload.test_scheduler import _make_scheduler, _Request
from vllm_ascend.dsa_offload import dp_control
from vllm_ascend.dsa_offload.dp_admin import DSADPCacheAdmin
from vllm_ascend.dsa_offload.kv_cache_coordinator import DSABlockPoolView, DSAKVCacheCoordinator
from vllm_ascend.dsa_offload.l3.backend_adapter import DSAL3BackendAdapter
from vllm_ascend.dsa_offload.l3.connector import DSAL3Connector
from vllm_ascend.dsa_offload.l3.scheduler import DSAL3StoreScheduler


class _Deployment:
    def __init__(self, monkeypatch):
        self.cores, self.calls = [], []
        self.store = dist.TCPStore(
            "127.0.0.1",
            0,
            world_size=1,
            is_master=True,
            wait_for_workers=False,
            timeout=timedelta(seconds=5),
            use_libuv=False,
        )
        self.pausing = asyncio.Event()
        self.io_done = asyncio.Event()
        self.io_done.set()
        self.failure = None
        self.driver = None
        monkeypatch.setattr(core_module.ParallelConfig, "has_unfinished_dp", lambda *args: False)
        for rank in range(2):
            core = object.__new__(core_module.DPEngineCoreProc)
            core.shutdown_state = core_module.EngineShutdownState.RUNNING
            core.dp_rank = rank
            core.dp_size = 2
            core.dp_group = object()
            core.dp_store = self.store
            core.engines_running = core.pending_pause = core.ignore_start_dp_wave = False
            core.batch_queue = None
            core._idle_state_callbacks = []
            scheduler = core.scheduler = _make_scheduler()
            scheduler.finished_req_ids = set()
            scheduler.dsa_l3_enabled = True
            coordinator = scheduler.dsa_coordinator = object.__new__(DSAKVCacheCoordinator)
            coordinator._pending_prefix_hits = {"unallocated": object()}
            coordinator._pending_prefix_restores = {}
            coordinator._pending_external_prefix_loads = {}
            pools = [BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=128) for _ in range(4)]
            view = DSABlockPoolView(pools[:3], extra_prefix_cache_pools=pools[3:])
            scheduler.kv_cache_manager = SimpleNamespace(
                block_pool=view,
                reset_prefix_cache=Mock(side_effect=view.reset_prefix_cache),
            )
            connector = scheduler.connector = object.__new__(DSAL3Connector)
            connector.data_parallel_size = 2
            connector.scheduler_handler = DSAL3StoreScheduler(_make_mtp_schema(), max_inflight_jobs=2)
            connector.supports_dsa_namespace_reset = False
            core._dsa_cache_admin_state = DSADPCacheAdmin(core)
            self.cores.append(core)

    def client(self, reverse=False):
        async def call(method, *args, engine):
            self.calls.append((engine, method, args))
            if method != "dsa_cache_admin":
                raise AssertionError(method)
            action = args[0]
            if self.failure == (engine, action):
                await asyncio.sleep(0)
                raise RuntimeError("injected admin failure")
            result = self.cores[engine]._dsa_cache_admin_state.execute(*args)
            if action == "pause" and all(core.pending_pause for core in self.cores):
                self.driver = asyncio.create_task(self._drain())
            if isinstance(result, Future):
                # 真正utility RPC的客户端取消不能取消服务器的Future。
                return await asyncio.shield(asyncio.wrap_future(result))
            return result

        return SimpleNamespace(
            core_engines=[1, 0] if reverse else [0, 1],
            _call_utility_async=call,
            vllm_config=SimpleNamespace(shutdown_timeout=300, parallel_config=SimpleNamespace(data_parallel_size=2)),
        )

    async def _drain(self):
        for core in self.cores:
            core.pending_pause = core.engines_running = False
            core.ignore_start_dp_wave = True
            if not core.has_work():
                core._notify_idle_state_callbacks()
        self.pausing.set()
        await self.io_done.wait()
        for core in self.cores:
            core.scheduler.connector.scheduler_handler._jobs.clear()
            if not core.has_work():
                core._notify_idle_state_callbacks()


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("already_paused", [False, True])
def test_reset_is_all_dp_and_preserves_existing_pause(monkeypatch, external, already_paused):
    async def run():
        deployment = _Deployment(monkeypatch)
        if already_paused:
            for core in deployment.cores:
                core.scheduler.set_pause_state(PauseState.PAUSED_NEW)
        # 统一epoch不能依赖各域原来的generation计数相等。
        deployment.cores[1].scheduler.connector.scheduler_handler._namespace_generation = 9
        assert await dp_control.call_dsa_dp_admin(deployment.client(), "reset_prefix_cache", False, external)
        schemas = [core.scheduler.connector.scheduler_handler.schema for core in deployment.cores]
        assert schemas[0] == schemas[1]
        assert (schemas[0].fingerprint != _make_mtp_schema().fingerprint) is external
        assert deployment.store.get("dsa/cache_admin/owner") == b""
        for core in deployment.cores:
            assert core.scheduler.pause_state == (PauseState.PAUSED_NEW if already_paused else PauseState.UNPAUSED)
            core.scheduler.kv_cache_manager.reset_prefix_cache.assert_called_once()
            assert not core.scheduler.dsa_coordinator._pending_prefix_hits
        actions = [args[0] for _, _, args in deployment.calls]
        assert actions.count("prepare") == actions.count("commit") == 2
        assert actions.count("resume") == (0 if already_paused else 2)

    asyncio.run(run())


def test_pause_wait_drains_l3_future_and_excludes_other_api_frontends(monkeypatch):
    async def run():
        deployment = _Deployment(monkeypatch)
        deployment.io_done.clear()
        deployment.cores[1].scheduler.connector.scheduler_handler._jobs[0] = object()
        first = asyncio.create_task(
            dp_control.call_dsa_dp_admin(deployment.client(), "reset_prefix_cache", False, True)
        )
        await asyncio.wait_for(deployment.pausing.wait(), 1)
        assert not first.done()
        assert not any(args[0] == "prepare" for _, _, args in deployment.calls)
        with pytest.raises(RuntimeError, match="busy or fail-closed"):
            await dp_control.call_dsa_dp_admin(deployment.client(reverse=True), "reset_prefix_cache", False, True)
        deployment.io_done.set()
        assert await asyncio.wait_for(first, 1)
        assert not any(core.scheduler.connector.has_pending_transfers() for core in deployment.cores)

    asyncio.run(run())


@pytest.mark.parametrize("action", ["prepare", "commit"])
def test_partial_failure_is_fail_closed_and_cannot_resume(monkeypatch, action):
    async def run():
        deployment = _Deployment(monkeypatch)
        deployment.failure = (1, action)
        client = deployment.client()
        with pytest.raises(RuntimeError, match="injected admin failure"):
            await dp_control.call_dsa_dp_admin(client, "reset_prefix_cache", False, True)
        assert deployment.store.get("dsa/cache_admin/owner")
        assert all(core._dsa_cache_admin_state.fatal_error for core in deployment.cores)
        assert not any(args[0] == "resume" for _, _, args in deployment.calls)
        with pytest.raises(RuntimeError, match="fail-closed"):
            await dp_control.call_dsa_dp_admin(client, "resume_scheduler")
        if action == "prepare":
            for core in deployment.cores:
                core.scheduler.kv_cache_manager.reset_prefix_cache.assert_not_called()

    asyncio.run(run())


def test_http_cancellation_does_not_abandon_started_reset(monkeypatch):
    async def run():
        deployment = _Deployment(monkeypatch)
        deployment.io_done.clear()
        deployment.cores[1].scheduler.connector.scheduler_handler._jobs[0] = object()
        client = deployment.client()
        caller = asyncio.create_task(dp_control.call_dsa_dp_admin(client, "reset_prefix_cache", False, True))
        await asyncio.wait_for(deployment.pausing.wait(), 1)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert client._dsa_admin_tasks
        deployment.io_done.set()
        await asyncio.wait_for(asyncio.gather(*client._dsa_admin_tasks), 1)
        assert deployment.store.get("dsa/cache_admin/owner") == b""
        assert all(core.scheduler.pause_state == PauseState.UNPAUSED for core in deployment.cores)

    asyncio.run(run())


def test_readonly_pool_preflight_catches_reference_without_partial_clear(monkeypatch):
    deployment = _Deployment(monkeypatch)
    core = deployment.cores[0]
    core.scheduler.set_pause_state(PauseState.PAUSED_NEW)
    pool = core.scheduler.kv_cache_manager.block_pool.prefix_cache_pools[-1]
    pool.get_new_blocks(1)
    admin = core._dsa_cache_admin_state
    token = "a" * 32
    assert admin.execute("acquire", token)
    with pytest.raises(RuntimeError, match="referenced blocks"):
        admin.execute("prepare", token, True)
    core.scheduler.kv_cache_manager.reset_prefix_cache.assert_not_called()
    assert core.scheduler.connector.scheduler_handler._namespace_generation == 0


def test_epoch_commit_is_idempotent_and_stale_owner_is_rejected(monkeypatch):
    deployment = _Deployment(monkeypatch)
    admin = deployment.cores[0]._dsa_cache_admin_state
    deployment.cores[0].scheduler.set_pause_state(PauseState.PAUSED_NEW)
    token = "b" * 32
    admin.execute("acquire", token)
    admin.execute("prepare", token, True)
    first = admin.execute("commit", token, True)
    assert admin.execute("commit", token, True) == first
    deployment.cores[0].scheduler.kv_cache_manager.reset_prefix_cache.assert_called_once()
    admin.execute("release", token)
    with pytest.raises(RuntimeError, match="does not own"):
        admin.execute("commit", token, True)


def test_manual_pause_reset_and_resume_preserve_queued_requests(monkeypatch):
    async def run():
        deployment = _Deployment(monkeypatch)
        client = deployment.client()
        assert await dp_control.call_dsa_dp_admin(client, "pause_scheduler", "wait", False) is None
        assert await dp_control.call_dsa_dp_admin(client, "is_scheduler_paused") is True
        scheduler = deployment.cores[0].scheduler
        request = _Request("queued", 4096, 0, 0, 4096, status=RequestStatus.WAITING)
        scheduler.waiting.add_request(request)
        scheduler.requests[request.request_id] = request
        assert await dp_control.call_dsa_dp_admin(client, "reset_prefix_cache", False, True)
        assert scheduler.pause_state == PauseState.PAUSED_NEW
        assert await dp_control.call_dsa_dp_admin(client, "resume_scheduler") is None
        assert await dp_control.call_dsa_dp_admin(client, "is_scheduler_paused") is False
        assert scheduler.waiting.peek_request() is request
        assert scheduler.get_num_unfinished_requests() == 1

    asyncio.run(run())


def test_partial_engine_list_is_rejected_before_lease(monkeypatch):
    deployment = _Deployment(monkeypatch)
    client = deployment.client()
    client.core_engines = [0]
    with pytest.raises(RuntimeError, match="all deployment participants"):
        asyncio.run(dp_control.call_dsa_dp_admin(client, "reset_prefix_cache", False, True))
    assert not deployment.calls


def test_admin_timeout_keeps_lease_and_does_not_resume(monkeypatch):
    original = dp_control._all

    async def short_pause(client, engines, method, *args, **kwargs):
        if args and args[0] == "pause":
            kwargs["timeout"] = 0.01
        return await original(client, engines, method, *args, **kwargs)

    monkeypatch.setattr(dp_control, "_all", short_pause)

    async def run():
        deployment = _Deployment(monkeypatch)
        deployment.io_done.clear()
        deployment.cores[1].scheduler.connector.scheduler_handler._jobs[0] = object()
        try:
            with pytest.raises(TimeoutError):
                await dp_control.call_dsa_dp_admin(deployment.client(), "reset_prefix_cache", False, True)
            assert deployment.store.get("dsa/cache_admin/owner")
            assert all(core._dsa_cache_admin_state.fatal_error for core in deployment.cores)
            assert not any(args[0] == "resume" for _, _, args in deployment.calls)
        finally:
            deployment.io_done.set()
            if deployment.driver is not None:
                await deployment.driver

    asyncio.run(run())


@pytest.mark.parametrize(
    "method,args",
    [
        ("pause_scheduler", ("abort", False)),
        ("pause_scheduler", ("wait", True)),
        ("sleep", (1,)),
        ("wake_up", ()),
        ("reset_prefix_cache", (True, True)),
    ],
)
def test_unsupported_management_is_rejected_before_acquiring_owner(monkeypatch, method, args):
    deployment = _Deployment(monkeypatch)
    with pytest.raises(ValueError):
        asyncio.run(dp_control.call_dsa_dp_admin(deployment.client(), method, *args))
    assert deployment.calls == []


@pytest.mark.parametrize("bad", [None, False, {}])
def test_invalid_status_cannot_be_reported_as_success(bad):
    with pytest.raises(RuntimeError, match="invalid participant status"):
        dp_control._check_statuses([bad])


def test_new_epoch_jobs_work_without_reinitializing_worker_schema(monkeypatch):
    async def run():
        deployment = _Deployment(monkeypatch)
        backend = _SharedPayloadBackend()
        source, destination = _make_mtp_store(), _make_mtp_store()
        writer = DSAL3BackendAdapter(backend=backend, schema=_make_mtp_schema())
        reader = DSAL3BackendAdapter(backend=backend, schema=_make_mtp_schema())
        writer.bind_store(source)
        reader.bind_store(destination)
        for marker, plane in enumerate(_planes(source), 21):
            plane[1].fill_(marker)
        assert await dp_control.call_dsa_dp_admin(deployment.client(), "reset_prefix_cache", False, True)
        owner = deployment.cores[0].scheduler.connector.scheduler_handler
        peer = deployment.cores[1].scheduler.connector.scheduler_handler
        job = next(iter(_plan(owner).store_jobs.values()))
        assert writer.schema.fingerprint != owner.schema.fingerprint
        assert writer.put_store_job(job=job, store=source) == owner.schema.bytes_per_logical_block
        assert not set(_make_mtp_schema().block_keys(b"shared")) & backend.payloads.keys()
        assert reader.get_load_job(job=_load(peer.schema), store=destination) == peer.schema.bytes_per_logical_block
        for left, right in zip(_planes(source), _planes(destination), strict=True):
            assert left[1].equal(right[2])
    asyncio.run(run())


@pytest.mark.parametrize("action", ["acquire", "pause", "prepare", "commit", "resume"])
def test_shutdown_rejects_new_management_without_mutating_state(monkeypatch, action):
    deployment = _Deployment(monkeypatch)
    core = deployment.cores[0]
    admin = core._dsa_cache_admin_state
    core.shutdown_state = core_module.EngineShutdownState.SHUTTING_DOWN
    assert admin.execute("status")["shutting_down"] is True
    with pytest.raises(RuntimeError, match="shutdown has started"):
        admin.execute(action, "a" * 32, False)
    core.scheduler.kv_cache_manager.reset_prefix_cache.assert_not_called()
    assert not deployment.store.check(["dsa/cache_admin/owner"])


def test_frontend_rejects_shutdown_participant_status(monkeypatch):
    deployment = _Deployment(monkeypatch)
    deployment.cores[1].shutdown_state = core_module.EngineShutdownState.SHUTTING_DOWN
    statuses = [core._dsa_cache_admin_state.status() for core in deployment.cores]
    with pytest.raises(RuntimeError, match="shutting down"):
        dp_control._check_statuses(statuses)
