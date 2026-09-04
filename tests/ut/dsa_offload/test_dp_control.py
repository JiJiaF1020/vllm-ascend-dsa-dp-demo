# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.dsa_offload.dp_control import _all, call_dsa_dp_admin


@pytest.mark.parametrize("second_result", [True, False, RuntimeError("DP1 failed"), None])
def test_admin_fanout_waits_for_all_dp_even_if_one_fails(second_result) -> None:
    async def run():
        calls = []
        first_finished = asyncio.Event()
        second_finished = asyncio.Event()

        async def call(method, *args, engine):
            calls.append((method, args, engine))
            if engine == b"dp0":
                first_finished.set()
                return True
            await first_finished.wait()
            second_finished.set()
            if isinstance(second_result, Exception):
                raise second_result
            return second_result

        client = SimpleNamespace(core_engines=[b"dp0", b"dp1"], _call_utility_async=call)
        if isinstance(second_result, Exception):
            with pytest.raises(RuntimeError, match="failed:.*1:"):
                await _all(client, client.core_engines, "reset_prefix_cache", False, False)
        else:
            assert await _all(client, client.core_engines, "reset_prefix_cache", False, False) == [True, second_result]
        assert second_finished.is_set()
        assert calls == [
            ("reset_prefix_cache", (False, False), b"dp0"),
            ("reset_prefix_cache", (False, False), b"dp1"),
        ]

    asyncio.run(run())


def test_local_reset_does_not_abandon_peer_after_first_rpc_exception() -> None:
    async def run():
        failed = asyncio.Event()
        peer_finished = asyncio.Event()

        async def call(method, *args, engine):
            if engine == b"dp0":
                failed.set()
                raise RuntimeError("DP0 failed")
            await failed.wait()
            peer_finished.set()
            return True

        client = SimpleNamespace(core_engines=[b"dp0", b"dp1"], _call_utility_async=call)
        with pytest.raises(RuntimeError, match="failed:.*0:"):
            await _all(client, client.core_engines, "reset_prefix_cache")
        assert peer_finished.is_set()

    asyncio.run(run())


def test_local_reset_requires_participants() -> None:
    with pytest.raises(RuntimeError, match="no EngineCore"):
        asyncio.run(call_dsa_dp_admin(SimpleNamespace(core_engines=[]), "reset_prefix_cache"))


@pytest.mark.parametrize(
    ("enabled", "method", "aggregate"),
    [
        (True, "reset_prefix_cache", True),
        (True, "pause_scheduler", True),
        (True, "resume_scheduler", True),
        (True, "is_scheduler_paused", True),
        (True, "sleep", True),
        (True, "wake_up", True),
        (False, "reset_prefix_cache", False),
        (True, "reset_mm_cache", False),
        (True, "reset_encoder_cache", False),
        (True, "profile", False),
        (True, "get_supported_tasks", False),
    ],
)
def test_dp_control_patch_only_intercepts_dsa_owned_management(monkeypatch, enabled, method, aggregate) -> None:
    from vllm_ascend.patch.platform import patch_dsa_data_parallel as patch_module

    calls = []

    async def original(client, method, *args):
        calls.append(("original", method, args))
        return "baseline-result"

    async def reset(client, method, *args):
        calls.append(("aggregate", args))
        return True

    monkeypatch.setattr(patch_module, "_ORIGINAL_CALL_UTILITY_ASYNC", original)
    monkeypatch.setattr(patch_module, "call_dsa_dp_admin", reset)
    client = SimpleNamespace(vllm_config=SimpleNamespace(additional_config={"dsa_sparse_config": {"enabled": enabled}}))
    result = asyncio.run(patch_module._call_utility_async(client, method, False, False))
    if aggregate:
        assert result is True
        assert calls == [("aggregate", (False, False))]
    else:
        assert result == "baseline-result"
        assert calls == [("original", method, (False, False))]


@pytest.mark.parametrize("api_server_count", [1, 2])
@pytest.mark.parametrize("method", ["reset_mm_cache", "reset_encoder_cache"])
def test_native_frontend_cache_reset_reaches_all_dp_without_dsa_admin(api_server_count, method) -> None:
    """Exercise the native cache utility called by API startup, without HTTP/NPU."""
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine.core_client import AsyncMPClient, DPLBAsyncMPClient

    from vllm_ascend.patch.platform import patch_dsa_data_parallel  # noqa: F401

    async def run():
        cores = [
            SimpleNamespace(
                scheduler=SimpleNamespace(
                    has_unfinished_requests=lambda: False,
                    reset_prefix_cache=Mock(),
                    reset_encoder_cache=Mock(),
                ),
                mm_receiver_cache=SimpleNamespace(clear_cache=Mock()),
                model_executor=SimpleNamespace(reset_mm_cache=Mock(), reset_encoder_cache=Mock()),
            )
            for _ in range(2)
        ]
        calls = []

        async def rpc(name, *args, engine):
            calls.append((name, engine))
            # Replace only the socket transport, not the reset implementations.
            await asyncio.sleep(0)
            return getattr(EngineCore, name)(cores[engine], *args)

        clients = []
        for _ in range(api_server_count):
            client = SimpleNamespace(
                core_engines=(0, 1),
                _call_utility_async=rpc,
                vllm_config=SimpleNamespace(
                    additional_config={"dsa_sparse_config": {"enabled": True}},
                    parallel_config=SimpleNamespace(data_parallel_size=2),
                ),
            )
            client.call_utility_async = MethodType(DPLBAsyncMPClient.call_utility_async, client)
            async_name = f"{method}_async"
            setattr(client, async_name, MethodType(getattr(AsyncMPClient, async_name), client))
            clients.append(client)

        await asyncio.gather(*(getattr(client, async_name)() for client in clients))
        assert sorted(calls) == sorted([(method, rank) for rank in (0, 1)] * api_server_count)
        assert all(not hasattr(client, "_dsa_admin_tasks") for client in clients)
        for core in cores:
            core.scheduler.reset_prefix_cache.assert_not_called()
            assert getattr(core.model_executor, method).call_count == api_server_count
            assert core.mm_receiver_cache.clear_cache.call_count == (
                api_server_count if method == "reset_mm_cache" else 0
            )
            assert core.scheduler.reset_encoder_cache.call_count == (
                api_server_count if method == "reset_encoder_cache" else 0
            )

    asyncio.run(run())


def _shutdown_cores(monkeypatch, timeout=300):
    from vllm_ascend.patch.platform import patch_dsa_data_parallel as patch_module

    class Store:
        def __init__(self):
            self.values = {}
            self.writes = []

        def set(self, key, value):
            self.writes.append(key)
            self.values[key] = value

        def check(self, keys):
            return all(key in self.values for key in keys)

        def get(self, key):
            return self.values[key].encode()

    store = Store()
    group = object()
    barriers = []
    monkeypatch.setattr(patch_module.dist, "barrier", lambda *, group: barriers.append(group))
    monkeypatch.setattr(patch_module.time, "sleep", lambda delay: None)
    monkeypatch.setattr(patch_module, "_ORIGINAL_HANDLE_SHUTDOWN", lambda core: core.has_work)
    config = SimpleNamespace(
        additional_config={"dsa_sparse_config": {"enabled": True}},
        kv_transfer_config=SimpleNamespace(kv_connector="DSAL3Connector"),
        shutdown_timeout=timeout,
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
    )
    cores = [SimpleNamespace(vllm_config=config, dp_rank=rank, dp_size=2, dp_store=store,
                             dp_group=group, has_work=False, engines_running=False,
                             model_executor=SimpleNamespace(collective_rpc=Mock(return_value=[True] * 4)),
                             ignore_start_dp_wave=False) for rank in (0, 1)]
    return patch_module, store, cores, barriers


def test_dp_l3_shutdown_closes_workers_after_drain_and_before_final_barrier(monkeypatch) -> None:
    from vllm.v1.engine import EngineCoreRequestType

    from vllm_ascend.dsa_offload.l3.connector import shutdown_dsa_l3_worker

    patch_module, store, cores, barriers = _shutdown_cores(monkeypatch)
    group = cores[0].dp_group
    cores[1].has_work = True  # Requests or write-behind still pending on DP1.
    assert patch_module._handle_shutdown(cores[0]) is True
    assert patch_module._handle_shutdown(cores[1]) is True
    assert patch_module._handle_shutdown(cores[0]) is True
    assert store.writes == ["dsa_l3/drained/0"]
    assert barriers == []
    for core in cores:
        core.model_executor.collective_rpc.assert_not_called()
    assert cores[0].ignore_start_dp_wave
    assert not cores[1].ignore_start_dp_wave
    patch_module.DPEngineCoreProc._handle_client_request(
        cores[0], EngineCoreRequestType.START_DP_WAVE, (1, 1)
    )
    assert not cores[0].engines_running
    cores[1].has_work = False
    assert patch_module._handle_shutdown(cores[1]) is True
    assert barriers == []  # DP0 has not reported closed resources yet.
    assert patch_module._handle_shutdown(cores[0]) is False
    assert patch_module._handle_shutdown(cores[1]) is False
    assert store.writes == [
        "dsa_l3/drained/0", "dsa_l3/drained/1", "dsa_l3/resources_closed/1", "dsa_l3/resources_closed/0",
    ]
    for core in cores:
        core.model_executor.collective_rpc.assert_called_once_with(shutdown_dsa_l3_worker, timeout=300)
    assert barriers == [group, group]


@pytest.mark.parametrize("enabled,connector", [(True, None), (False, "DSAL3Connector"), (True, "AscendStoreConnector")])
def test_non_dsa_l3_shutdown_does_not_touch_dp_store(monkeypatch, enabled, connector) -> None:
    from vllm_ascend.patch.platform import patch_dsa_data_parallel as patch_module

    monkeypatch.setattr(patch_module, "_ORIGINAL_HANDLE_SHUTDOWN", lambda core: False)
    core = SimpleNamespace(vllm_config=SimpleNamespace(
        additional_config={"dsa_sparse_config": {"enabled": enabled}},
        kv_transfer_config=SimpleNamespace(kv_connector=connector) if connector else None,
    ))
    assert patch_module._handle_shutdown(core) is False


@pytest.mark.parametrize("result", [None, [True] * 3, [True, True, True, False], TimeoutError("native close timeout")])
def test_resource_close_failure_is_not_reported_as_success_or_retried(monkeypatch, result):
    patch_module, store, cores, barriers = _shutdown_cores(monkeypatch)
    if isinstance(result, Exception):
        cores[1].model_executor.collective_rpc.side_effect = result
    else:
        cores[1].model_executor.collective_rpc.return_value = result
    assert patch_module._handle_shutdown(cores[0]) is True
    assert patch_module._handle_shutdown(cores[1]) is True
    assert store.get("dsa_l3/resources_closed/1") == b"failed"
    assert barriers == []
    assert patch_module._handle_shutdown(cores[0]) is False
    assert patch_module._handle_shutdown(cores[1]) is False
    assert store.get("dsa_l3/resources_closed/0") == b"closed"
    for core in cores:
        core.model_executor.collective_rpc.assert_called_once()


def test_abort_timeout_does_not_add_a_blocking_resource_rpc(monkeypatch):
    patch_module, store, cores, _ = _shutdown_cores(monkeypatch, timeout=0)
    assert patch_module._handle_shutdown(cores[0]) is True
    assert patch_module._handle_shutdown(cores[1]) is True
    assert patch_module._handle_shutdown(cores[0]) is False
    assert patch_module._handle_shutdown(cores[1]) is False
    for core in cores:
        core.model_executor.collective_rpc.assert_not_called()
        assert store.get(f"dsa_l3/resources_closed/{core.dp_rank}") == b"abort"


def test_early_close_obeys_native_shutdown_state_and_has_work(monkeypatch):
    from vllm.v1.engine.core import DPEngineCoreProc, EngineCoreProc, EngineShutdownState

    patch_module, store, fake_cores, _ = _shutdown_cores(monkeypatch)
    monkeypatch.setattr(patch_module, "_ORIGINAL_HANDLE_SHUTDOWN", EngineCoreProc._handle_shutdown)
    core = object.__new__(DPEngineCoreProc)
    core.__dict__.update({key: value for key, value in vars(fake_cores[0]).items() if key != "has_work"})
    pending = [False]
    core.scheduler = SimpleNamespace(has_requests=lambda: pending[0])
    core.batch_queue = None
    core.shutdown_state = EngineShutdownState.RUNNING
    assert patch_module._handle_shutdown(core) is True
    assert store.writes == []
    core.shutdown_state = EngineShutdownState.SHUTTING_DOWN
    pending[0] = True
    assert patch_module._handle_shutdown(core) is True
    core.model_executor.collective_rpc.assert_not_called()
    pending[0] = False
    store.set("dsa_l3/drained/1", "1")
    store.set("dsa_l3/resources_closed/1", "closed")
    assert patch_module._handle_shutdown(core) is False
    core.model_executor.collective_rpc.assert_called_once()


def test_worker_resource_rpc_uses_native_idempotent_connector_shutdown(monkeypatch):
    import cloudpickle
    from vllm.distributed.kv_transfer import kv_transfer_state

    from vllm_ascend.dsa_offload.l3 import connector as connector_module
    from vllm_ascend.dsa_offload.l3.connector import DSAL3Connector, shutdown_dsa_l3_worker

    events = []
    synchronize = Mock(side_effect=lambda: events.append("synchronize"))
    monkeypatch.setattr(connector_module.torch.npu, "current_stream", lambda: SimpleNamespace(synchronize=synchronize))
    connector = object.__new__(DSAL3Connector)
    connector._deferred_store_metadata = None
    connector.worker_handler = SimpleNamespace(shutdown=Mock(side_effect=lambda: events.append("close")))
    connector.scheduler_handler = None
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", connector)
    worker = SimpleNamespace(vllm_config=SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector="DSAL3Connector"),
    ))
    callback = cloudpickle.loads(cloudpickle.dumps(shutdown_dsa_l3_worker))
    assert callback(worker) is True
    assert kv_transfer_state.has_kv_transfer_group() is False
    assert callback(worker) is True
    kv_transfer_state.ensure_kv_transfer_shutdown()
    connector.worker_handler.shutdown.assert_called_once()
    synchronize.assert_called_once()
    assert events == ["synchronize", "close"]


@pytest.mark.parametrize(
    "wrong_config,deferred,sync_error", [(True, False, False), (False, True, False), (False, False, True)],
)
def test_worker_resource_rpc_guards_before_native_close(monkeypatch, wrong_config, deferred, sync_error):
    from vllm.distributed.kv_transfer import kv_transfer_state

    from vllm_ascend.dsa_offload.l3 import connector as connector_module
    from vllm_ascend.dsa_offload.l3.connector import DSAL3Connector, shutdown_dsa_l3_worker

    synchronize = Mock(side_effect=RuntimeError("device synchronization failed") if sync_error else None)
    monkeypatch.setattr(connector_module.torch.npu, "current_stream", lambda: SimpleNamespace(synchronize=synchronize))
    connector = object.__new__(DSAL3Connector)
    connector._deferred_store_metadata = object() if deferred else None
    connector.worker_handler = SimpleNamespace(shutdown=Mock())
    connector.scheduler_handler = None
    monkeypatch.setattr(kv_transfer_state, "_KV_CONNECTOR_AGENT", connector)
    worker = SimpleNamespace(vllm_config=SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector="Other" if wrong_config else "DSAL3Connector"),
    ))
    with pytest.raises(RuntimeError):
        shutdown_dsa_l3_worker(worker)
    connector.worker_handler.shutdown.assert_not_called()
    assert kv_transfer_state.get_kv_transfer_group() is connector
    assert synchronize.call_count == int(sync_error)


@pytest.mark.parametrize("fatal", [False, True])
def test_fatal_admin_rejects_new_requests_without_entering_scheduler(monkeypatch, fatal):
    from vllm_ascend.patch.platform import patch_dsa_data_parallel as patch_module

    calls = []
    core = SimpleNamespace(
        _dsa_cache_admin_state=SimpleNamespace(fatal_error="uncertain commit" if fatal else None),
        _send_error_outputs_to_client=lambda ids, index: calls.append(("error", ids, index)),
    )
    request = SimpleNamespace(request_id="queued", client_index=2)
    monkeypatch.setattr(patch_module, "_ORIGINAL_ADD_REQUEST", lambda self, req, wave: calls.append(("add", req, wave)))
    patch_module._add_request(core, request, 7)
    assert calls == ([("error", ["queued"], 2)] if fatal else [("add", request, 7)])
