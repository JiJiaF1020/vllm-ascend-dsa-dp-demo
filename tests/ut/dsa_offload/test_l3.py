# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.outputs import KVConnectorOutput

from examples.dsa_demo.L3_cache_sys_api_test.test_mooncake_tp import (
    _build_payload_layout,
    _resolve_io_roles,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
    GlobalTE,
)
from vllm_ascend.dsa_offload.kv_cache import (
    DSAIndexerKVSpec,
    DSAResidentMLAAttentionSpec,
    build_dsa_kv_cache_groups,
)
from vllm_ascend.dsa_offload.l3 import backend_adapter as l3_backend_adapter_module
from vllm_ascend.dsa_offload.l3 import connector as l3_connector_module
from vllm_ascend.dsa_offload.l3.backend_adapter import (
    DSAL3BackendAdapter,
    configure_dsa_mooncake_global_ub,
)
from vllm_ascend.dsa_offload.l3.connector import DSAL3Connector
from vllm_ascend.dsa_offload.l3.metadata import (
    DSAL3ConnectorMetadata,
    DSAL3JobResult,
    DSAL3StoreBlock,
    DSAL3StoreJob,
    DSAL3WorkerMetadata,
)
from vllm_ascend.dsa_offload.l3.scheduler import DSAL3StoreScheduler
from vllm_ascend.dsa_offload.l3.schema import DSAL3KeySchema
from vllm_ascend.dsa_offload.l3.worker import DSAL3StoreWorker
from vllm_ascend.dsa_offload.worker_dram_store import DSAWorkerDRAMStore


def test_mooncake_probe_glm52_payload_matches_dsa_worker_contract() -> None:
    sizes, groups = _build_payload_layout(
        "dsa_a5_glm52",
        payload_bytes=1,
        num_buffers=1,
    )

    assert len(sizes) == 120
    assert sum(sizes) == 6_904_320
    assert [(name, len(indices), sum(sizes[index] for index in indices)) for name, indices in groups] == [
        ("indexer", 42, 354_816),
        ("mla", 78, 6_549_504),
    ]


def test_mooncake_probe_supports_independent_producer_and_consumer() -> None:
    assert _resolve_io_roles(
        rank=0,
        world_size=2,
        producer_rank=1,
        consumer_rank=0,
    ) == (False, True, 1)
    assert _resolve_io_roles(
        rank=1,
        world_size=2,
        producer_rank=1,
        consumer_rank=0,
    ) == (True, False, 1)


def _cpu_arena(
    block_shape: tuple[int, ...],
    dtype: torch.dtype,
    capacity: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros((capacity, *block_shape), dtype=dtype, device=device)


def _make_store() -> DSAWorkerDRAMStore:
    store = DSAWorkerDRAMStore(
        usable_blocks=4,
        storage_rows=2,
        max_logical_blocks=4,
        device=torch.device("cpu"),
        arena_factory=_cpu_arena,
    )
    store.add_packed_layer(
        layer_id=0,
        resident_packed_cache=torch.empty(4, 128, 1, 656, dtype=torch.int8),
        indexer_cache=(
            torch.empty(8, 128, 1, 128, dtype=torch.int8),
            torch.empty(8, 128, 1, 1, dtype=torch.float32),
        ),
    )
    return store


def _make_schema() -> DSAL3KeySchema:
    return DSAL3KeySchema(
        fingerprint="f" * 32,
        block_size=128,
        indexer_layer_names=("indexer.0",),
        resident_layer_names=("mla.0",),
        indexer_plane_count=2,
        mla_plane_count=1,
        indexer_bytes_per_block=128 * 128 + 128 * 4,
        mla_bytes_per_block=128 * 656,
    )


def _make_job(job_id: int = 0, block_id: int = 1) -> DSAL3StoreJob:
    schema = _make_schema()
    block_hash = BlockHash(b"block-hash")
    indexer_key, mla_key = schema.block_keys(block_hash)
    return DSAL3StoreJob(
        job_id=job_id,
        blocks=(
            DSAL3StoreBlock(
                request_id="req",
                logical_block_index=0,
                source_l2_block_id=block_id,
                block_hash=block_hash,
                indexer_key=indexer_key,
                mla_key=mla_key,
            ),
        ),
    )


def _make_multi_block_job(
    block_ids: tuple[int, ...],
) -> DSAL3StoreJob:
    schema = _make_schema()
    blocks = []
    for logical_index, block_id in enumerate(block_ids):
        block_hash = BlockHash(f"block-hash-{logical_index}".encode())
        indexer_key, mla_key = schema.block_keys(block_hash)
        blocks.append(
            DSAL3StoreBlock(
                request_id="req",
                logical_block_index=logical_index,
                source_l2_block_id=block_id,
                block_hash=block_hash,
                indexer_key=indexer_key,
                mla_key=mla_key,
            )
        )
    return DSAL3StoreJob(job_id=0, blocks=tuple(blocks))


def test_l3_connector_metadata_is_pickle_safe() -> None:
    metadata = DSAL3ConnectorMetadata(
        store_jobs={0: _make_job()},
        jobs_to_flush={3, 5},
    )

    restored = pickle.loads(pickle.dumps(metadata))

    assert restored == metadata


def test_l3_schema_fingerprint_and_component_keys_are_stable() -> None:
    specs = {
        "model.layers.0.self_attn.indexer.k_cache": DSAIndexerKVSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        ),
        "model.layers.0.self_attn.attn": DSAResidentMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            sparse_head_dim=(512, 64, 0),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        ),
    }
    groups = build_dsa_kv_cache_groups(specs)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/models/glm",
            revision=None,
            quantization="ascend",
            hf_config=SimpleNamespace(
                architectures=("GlmMoeDsaForCausalLM",),
                model_type="glm_moe_dsa",
            ),
        )
    )
    kv_config = SimpleNamespace(kv_cache_groups=groups)

    left = DSAL3KeySchema.from_config(vllm_config, kv_config)
    right = DSAL3KeySchema.from_config(vllm_config, kv_config)
    assert left == right
    indexer_key, mla_key = left.block_keys(BlockHash(b"hash"))
    assert f":{left.fingerprint}:indexer:" in indexer_key
    assert f":{left.fingerprint}:mla:" in mla_key
    assert indexer_key != mla_key


def test_dsa_mooncake_config_requires_host_and_device_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASCEND_GLOBAL_RESOURCE_CONFIG", raising=False)
    monkeypatch.delenv("ASCEND_CONNECT_TIMEOUT", raising=False)
    monkeypatch.delenv("ASCEND_TRANSFER_TIMEOUT", raising=False)
    configure_dsa_mooncake_global_ub(4)
    raw = os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"]
    assert "ub_ctp:device" in raw
    assert "ub_ctp:host" in raw
    assert os.environ["ASCEND_CONNECT_TIMEOUT"] == "30000"
    assert os.environ["ASCEND_TRANSFER_TIMEOUT"] == "30000"

    monkeypatch.setenv(
        "ASCEND_GLOBAL_RESOURCE_CONFIG",
        '{"comm_resource_config.protocol_desc":["ub_ctp:device"]}',
    )
    with pytest.raises(ValueError, match="missing=.*ub_ctp:host"):
        configure_dsa_mooncake_global_ub(4)


@pytest.mark.parametrize(
    ("protocol", "expects_global_ub"),
    [("ascend", True), ("tcp", False)],
)
def test_dsa_mooncake_backend_routes_transport_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
    protocol: str,
    expects_global_ub: bool,
) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
        MooncakeBackend,
        MooncakeStoreConfig,
    )

    config = SimpleNamespace(protocol=protocol)
    monkeypatch.setattr(
        MooncakeStoreConfig,
        "load_from_env",
        staticmethod(lambda: config),
    )
    initialized: dict[str, Any] = {}

    def fake_init(self, parallel_config, *, config=None, **_kwargs):
        self.config = config
        initialized["parallel_config"] = parallel_config
        initialized["config"] = config

    monkeypatch.setattr(MooncakeBackend, "__init__", fake_init)
    global_ub_calls: list[int] = []
    monkeypatch.setattr(
        l3_backend_adapter_module,
        "configure_dsa_mooncake_global_ub",
        lambda channels: global_ub_calls.append(channels),
    )
    parallel_config = object()

    adapter = DSAL3BackendAdapter.create_mooncake(
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
        schema=_make_schema(),
        max_active_channels=6,
        max_descriptors_per_put=128,
    )

    assert adapter.backend.config is config
    assert initialized == {
        "parallel_config": parallel_config,
        "config": config,
    }
    assert global_ub_calls == ([6] if expects_global_ub else [])


class _FakeBackend:
    def __init__(self, result: list[int] | None = None) -> None:
        self.result = result or [0, 0]
        self.registered: tuple[list[int], list[int]] | None = None
        self.put_args: tuple[Any, Any, Any] | None = None
        self.put_calls: list[tuple[Any, Any, Any]] = []
        self.closed = False
        self.existing: list[int] | None = None

    def register_buffer(self, ptrs: list[int], sizes: list[int]) -> None:
        self.registered = (ptrs, sizes)

    def put(self, keys, addresses, sizes):
        self.put_args = (keys, addresses, sizes)
        self.put_calls.append(self.put_args)
        return self.result

    def exists(self, keys):
        return self.existing or [0] * len(keys)

    def close(self) -> None:
        self.closed = True


def test_backend_adapter_registers_and_puts_both_components() -> None:
    store = _make_store()
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    adapter.bind_store(store)

    transferred = adapter.put_store_job(job=_make_job(), store=store)
    assert backend.registered is not None
    assert len(backend.registered[0]) == 3
    assert backend.put_args is not None
    assert len(backend.put_args[0]) == 2
    assert transferred == 128 * 128 + 128 * 4 + 128 * 656


def test_backend_adapter_rejects_missing_per_key_result() -> None:
    store = _make_store()
    backend = _FakeBackend()
    backend.result = None
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    adapter.bind_store(store)

    with pytest.raises(RuntimeError, match="no per-key result"):
        adapter.put_store_job(job=_make_job(), store=store)


def test_backend_adapter_skips_components_already_in_l3() -> None:
    store = _make_store()
    backend = _FakeBackend(result=[0])
    backend.existing = [1, 0]
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    adapter.bind_store(store)

    transferred = adapter.put_store_job(job=_make_job(), store=store)

    assert backend.put_args is not None
    assert len(backend.put_args[0]) == 1
    assert backend.put_args[0][0].endswith(":mla:" + b"block-hash".hex())
    assert transferred == 128 * 656


def test_backend_adapter_splits_put_by_descriptor_budget() -> None:
    store = _make_store()
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(
        backend=backend,
        schema=_make_schema(),
        max_descriptors_per_put=3,
    )
    adapter.bind_store(store)

    transferred = adapter.put_store_job(
        job=_make_multi_block_job((1, 2)),
        store=store,
    )

    assert len(backend.put_calls) == 2
    assert [len(keys) for keys, _, _ in backend.put_calls] == [2, 2]
    assert transferred == 2 * (128 * 128 + 128 * 4 + 128 * 656)


def test_backend_adapter_isolates_component_above_descriptor_budget() -> None:
    store = _make_store()
    backend = _FakeBackend(result=[0])
    adapter = DSAL3BackendAdapter(
        backend=backend,
        schema=_make_schema(),
        max_descriptors_per_put=1,
    )
    adapter.bind_store(store)

    adapter.put_store_job(job=_make_job(), store=store)

    assert len(backend.put_calls) == 2
    assert [len(keys) for keys, _, _ in backend.put_calls] == [1, 1]


def test_scheduler_plans_rank0_store_and_fences_reused_l2_block() -> None:
    scheduler = DSAL3StoreScheduler(_make_schema(), max_inflight_jobs=2)
    request = SimpleNamespace(block_hashes=[BlockHash(b"hash")])
    requests = {"req": request}
    directory = SimpleNamespace(get_request_blocks=lambda _request_id: (3,))
    coordinator = SimpleNamespace(dram_directory=directory)
    projection = SimpleNamespace(
        request_ids=("req",),
        dram_ready_block_counts=(1,),
        dram_block_assignments=(),
    )
    metadata = scheduler.build_connector_meta()
    scheduler.extend_store_jobs(
        metadata=metadata,
        projection=projection,
        coordinator=coordinator,
        requests=requests,
    )
    assert tuple(metadata.store_jobs) == (0,)
    assert scheduler.has_pending_jobs()
    assert scheduler._planned_jobs == 1
    assert scheduler._planned_blocks == 1

    reused = scheduler.build_connector_meta()
    reused_projection = SimpleNamespace(
        request_ids=(),
        dram_ready_block_counts=(),
        dram_block_assignments=(SimpleNamespace(physical_block_ids=(3,)),),
    )
    scheduler.extend_store_jobs(
        metadata=reused,
        projection=reused_projection,
        coordinator=coordinator,
        requests=requests,
    )
    assert reused.jobs_to_flush == {0}

    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=DSAL3WorkerMetadata(results={0: DSAL3JobResult(job_id=0, success=True)})
        )
    )
    assert not scheduler.has_pending_jobs()
    assert scheduler._completed_jobs == 1
    assert scheduler._failed_jobs == 0
    after_completion = scheduler.build_connector_meta()
    scheduler.extend_store_jobs(
        metadata=after_completion,
        projection=projection,
        coordinator=coordinator,
        requests=requests,
    )
    assert not after_completion.store_jobs
    scheduler.shutdown()


def test_scheduler_shutdown_rejects_unacknowledged_store_job() -> None:
    scheduler = DSAL3StoreScheduler(_make_schema(), max_inflight_jobs=2)
    requests = {"req": SimpleNamespace(block_hashes=[BlockHash(b"hash")])}
    coordinator = SimpleNamespace(dram_directory=SimpleNamespace(get_request_blocks=lambda _request_id: (2,)))
    metadata = scheduler.build_connector_meta()
    scheduler.extend_store_jobs(
        metadata=metadata,
        projection=SimpleNamespace(
            request_ids=("req",),
            dram_ready_block_counts=(1,),
            dram_block_assignments=(),
        ),
        coordinator=coordinator,
        requests=requests,
    )

    with pytest.raises(RuntimeError, match="unacknowledged store jobs"):
        scheduler.shutdown()


def test_scheduler_retries_failed_incremental_store() -> None:
    scheduler = DSAL3StoreScheduler(_make_schema(), max_inflight_jobs=2)
    requests = {"req": SimpleNamespace(block_hashes=[BlockHash(b"hash")])}
    coordinator = SimpleNamespace(dram_directory=SimpleNamespace(get_request_blocks=lambda _request_id: (2,)))
    projection = SimpleNamespace(
        request_ids=("req",),
        dram_ready_block_counts=(1,),
        dram_block_assignments=(),
    )
    first = scheduler.build_connector_meta()
    scheduler.extend_store_jobs(
        metadata=first,
        projection=projection,
        coordinator=coordinator,
        requests=requests,
    )
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=DSAL3WorkerMetadata(
                results={
                    0: DSAL3JobResult(
                        job_id=0,
                        success=False,
                        error="failed",
                    )
                }
            )
        )
    )

    retry = scheduler.build_connector_meta()
    scheduler.extend_store_jobs(
        metadata=retry,
        projection=projection,
        coordinator=coordinator,
        requests=requests,
    )

    assert tuple(retry.store_jobs) == (1,)


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded = False
        self.synchronized = False

    def record(self) -> None:
        self.recorded = True

    def synchronize(self) -> None:
        assert self.recorded
        self.synchronized = True


def test_rank0_worker_waits_dump_event_and_reports_explicit_result() -> None:
    store = _make_store()
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    worker = DSAL3StoreWorker(
        tp_rank=0,
        device=torch.device("cpu"),
        backend=adapter,
        max_inflight_jobs=2,
        event_factory=_FakeEvent,
    )
    worker.bind_store(store)
    metadata = DSAL3ConnectorMetadata(store_jobs={0: _make_job()})
    worker.finish_step(metadata)
    worker.flush({0})
    assert worker._submitted_jobs == 1
    assert worker._finished_jobs == 1
    assert worker._failed_jobs == 0
    result = worker.build_worker_meta()
    assert result is not None
    assert result.results[0].success
    assert result.results[0].transferred_bytes == (128 * 128 + 128 * 4 + 128 * 656)
    worker.shutdown()


def test_nonproducer_worker_contributes_backend_without_binding_l2() -> None:
    store = _make_store()
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    worker = DSAL3StoreWorker(
        tp_rank=1,
        device=torch.device("cpu"),
        backend=adapter,
        max_inflight_jobs=2,
        event_factory=_FakeEvent,
    )

    worker.bind_store(store)
    worker.finish_step(DSAL3ConnectorMetadata())

    assert backend.registered is None
    assert worker.build_worker_meta() is None
    worker.shutdown()
    assert backend.closed


def test_worker_connector_initializes_backend_on_nonzero_tp_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"backend": "mooncake"},
        ),
    )
    monkeypatch.setattr(
        DSAL3KeySchema,
        "from_config",
        classmethod(lambda _cls, _vllm_config, _kv_cache_config: _make_schema()),
    )
    monkeypatch.setattr(
        DSAL3BackendAdapter,
        "create_mooncake",
        classmethod(lambda _cls, **_kwargs: adapter),
    )
    monkeypatch.setattr(
        l3_connector_module,
        "get_tensor_model_parallel_rank",
        lambda: 3,
    )
    monkeypatch.setattr(torch.npu, "current_device", lambda: 3)

    connector = DSAL3Connector(
        config,
        KVConnectorRole.WORKER,
        SimpleNamespace(),
    )

    assert connector.worker_handler is not None
    assert connector.worker_handler.tp_rank == 3
    assert connector.worker_handler.backend is adapter
    connector.shutdown()
    assert backend.closed


class _FakeTransferEngine:
    def __init__(self) -> None:
        self.registered: list[tuple[int, int]] = []
        self.unregistered: list[int] = []

    def register_memory(self, ptr: int, size: int) -> int:
        self.registered.append((ptr, size))
        return 0

    def unregister_memory(self, ptr: int) -> int:
        self.unregistered.append(ptr)
        return 0


def test_mooncake_global_te_supports_incremental_idempotent_registration() -> None:
    global_te = GlobalTE()
    engine = _FakeTransferEngine()
    global_te.transfer_engine = engine

    global_te.register_buffer([1000], [100])
    global_te.register_buffer([1000, 2000], [100, 200])
    assert engine.registered == [(1000, 100), (2000, 200)]
    with pytest.raises(ValueError, match="overlapping"):
        global_te.register_buffer([2050], [100])
    global_te.unregister_buffer([2000, 1000])
    assert engine.unregistered == [1000, 2000]
