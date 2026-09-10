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

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
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
from vllm_ascend.dsa_offload.l3 import schema as l3_schema_module
from vllm_ascend.dsa_offload.l3.backend_adapter import (
    DSAL3BackendAdapter,
    configure_dsa_mooncake_global_ub,
    configure_dsa_mooncake_tcp,
    validate_dsa_mooncake_direct_hbm,
)
from vllm_ascend.dsa_offload.l3.connector import DSAL3Connector
from vllm_ascend.dsa_offload.l3.metadata import (
    DSAL3ConnectorMetadata,
    DSAL3JobResult,
    DSAL3LoadBlock,
    DSAL3LoadJob,
    DSAL3StoreBlock,
    DSAL3StoreJob,
    DSAL3WorkerMetadata,
)
from vllm_ascend.dsa_offload.l3.scheduler import DSAL3StoreScheduler
from vllm_ascend.dsa_offload.l3.schema import DSAL3KeySchema
from vllm_ascend.dsa_offload.l3.worker import DSAL3StoreWorker
from vllm_ascend.dsa_offload.worker_dram_store import DSAWorkerDRAMStore


@pytest.mark.parametrize("protocol", ["tcp", "ascend"])
def test_direct_hbm_is_frozen_before_io(protocol: str) -> None:
    with pytest.raises(RuntimeError, match="is frozen"):
        validate_dsa_mooncake_direct_hbm(
            protocol=protocol,
            direct_hbm_load=True,
        )

    validate_dsa_mooncake_direct_hbm(
        protocol=protocol,
        direct_hbm_load=False,
    )


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


def _make_mtp_store() -> DSAWorkerDRAMStore:
    store = _make_store()
    store.add_mtp_layer(
        layer_id=0,
        mtp_full_cache=(
            torch.empty(4, 128, 1, 512, dtype=torch.bfloat16),
            torch.empty(4, 128, 1, 64, dtype=torch.bfloat16),
            torch.empty(4, 128, 1, 128, dtype=torch.bfloat16),
        ),
    )
    return store


def _make_mtp_schema() -> DSAL3KeySchema:
    base = _make_schema()
    return DSAL3KeySchema(
        fingerprint="m" * 32,
        block_size=base.block_size,
        indexer_layer_names=base.indexer_layer_names,
        resident_layer_names=base.resident_layer_names,
        indexer_plane_count=base.indexer_plane_count,
        mla_plane_count=base.mla_plane_count,
        indexer_bytes_per_block=base.indexer_bytes_per_block,
        mla_bytes_per_block=base.mla_bytes_per_block,
        mtp_layer_names=("mtp.0",),
        mtp_plane_count=3,
        mtp_bytes_per_block=128 * (512 + 64 + 128) * 2,
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


def _make_load_job(
    block_ids: tuple[int, ...] = (1,),
    *,
    logical_start: int = 0,
    direct_hbm: bool = False,
) -> DSAL3LoadJob:
    schema = _make_schema()
    blocks = []
    for offset, block_id in enumerate(block_ids):
        block_hash = BlockHash(f"load-hash-{offset}".encode())
        indexer_key, mla_key = schema.block_keys(block_hash)
        blocks.append(
            DSAL3LoadBlock(
                logical_block_index=logical_start + offset,
                destination_l2_block_id=block_id,
                destination_indexer_hbm_block_id=block_id,
                destination_resident_hbm_block_id=block_id,
                block_hash=block_hash,
                indexer_key=indexer_key,
                mla_key=mla_key,
            )
        )
    return DSAL3LoadJob(
        request_id="load-req",
        blocks=tuple(blocks),
        direct_hbm=direct_hbm,
    )


def test_l3_connector_metadata_is_pickle_safe() -> None:
    metadata = DSAL3ConnectorMetadata(
        store_jobs={0: _make_job()},
        jobs_to_flush={3, 5},
        load_jobs=(_make_load_job(),),
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
    model_config = SimpleNamespace(
        model="/models/glm",
        revision=None,
        quantization="ascend",
        hf_config=SimpleNamespace(
            architectures=("GlmMoeDsaForCausalLM",),
            model_type="glm_moe_dsa",
        ),
        compute_hash=lambda: "model-config-a",
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
    )
    kv_config = SimpleNamespace(kv_cache_groups=groups)

    left = DSAL3KeySchema.from_config(vllm_config, kv_config)
    right = DSAL3KeySchema.from_config(vllm_config, kv_config)
    assert left == right
    indexer_key, mla_key = left.block_keys(BlockHash(b"hash"))
    assert f":{left.fingerprint}:indexer:" in indexer_key
    assert f":{left.fingerprint}:mla:" in mla_key
    assert indexer_key != mla_key
    vllm_config.parallel_config.data_parallel_rank = 1
    vllm_config.parallel_config.data_parallel_size = 2
    assert DSAL3KeySchema.from_config(vllm_config, kv_config).block_keys(b"hash") == left.block_keys(b"hash")

    model_config.compute_hash = lambda: "model-config-b"
    changed_model = DSAL3KeySchema.from_config(vllm_config, kv_config)
    assert changed_model.fingerprint != left.fingerprint

    model_config.compute_hash = lambda: "model-config-a"
    vllm_config.parallel_config.tensor_parallel_size = 16
    changed_tp = DSAL3KeySchema.from_config(vllm_config, kv_config)
    assert changed_tp.fingerprint != left.fingerprint


@pytest.mark.parametrize("packed_c8", [False, True])
def test_l3_schema_adds_native_mtp_component(monkeypatch, packed_c8) -> None:
    specs = {
        "model.layers.0.self_attn.indexer.k_cache": DSAIndexerKVSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_sparse_li_c8=packed_c8,
        ),
        "model.layers.0.self_attn.attn": DSAResidentMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=656 if packed_c8 else 576,
            sparse_head_dim=(656, 0, 0) if packed_c8 else (512, 64, 0),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
            cache_sparse_sfa_c8=packed_c8,
        ),
        "model.layers.1.self_attn.attn": AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=704,
            sparse_head_dim=(512, 64, 128),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
            cache_sparse_sfa_c8=False,
            cache_sparse_li_c8=False,
        ),
    }
    groups = build_dsa_kv_cache_groups(specs)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/models/glm-mtp",
            revision=None,
            quantization="ascend",
            hf_config=SimpleNamespace(
                architectures=("GlmMoeDsaForCausalLM",),
                model_type="glm_moe_dsa",
            ),
            compute_hash=lambda: "mtp-model",
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=8),
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
            draft_model_config=SimpleNamespace(model="/models/glm-mtp", compute_hash=lambda: "draft-v1"),
        ),
    )

    schema = DSAL3KeySchema.from_config(
        vllm_config,
        SimpleNamespace(kv_cache_groups=groups),
    )

    assert schema.has_mtp
    assert schema.mtp_plane_count == 3
    assert schema.mtp_bytes_per_block == 128 * (512 + 64 + 128) * 2
    keys = schema.block_keys(BlockHash(b"hash"))
    assert len(keys) == 3
    assert f":{schema.fingerprint}:mtp:" in keys[2]
    vllm_config.parallel_config.data_parallel_rank = 1
    assert DSAL3KeySchema.from_config(vllm_config, SimpleNamespace(kv_cache_groups=groups)) == schema
    with monkeypatch.context() as scoped:
        scoped.setattr(l3_schema_module, "DSA_A5_MTP_ATTENTION_SEMANTICS", "old-per-query-tail")
        old_policy = DSAL3KeySchema.from_config(vllm_config, SimpleNamespace(kv_cache_groups=groups))
    assert (old_policy.fingerprint != schema.fingerprint) == packed_c8
    vllm_config.speculative_config.draft_model_config.compute_hash = lambda: "draft-v2"
    changed = DSAL3KeySchema.from_config(vllm_config, SimpleNamespace(kv_cache_groups=groups))
    assert changed.fingerprint != schema.fingerprint
    vllm_config.speculative_config = None
    with pytest.raises(ValueError, match="requires speculative model identity"):
        DSAL3KeySchema.from_config(vllm_config, SimpleNamespace(kv_cache_groups=groups))


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


def test_dsa_mooncake_tcp_enables_connection_pool_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MC_TCP_ENABLE_CONNECTION_POOL", raising=False)

    configure_dsa_mooncake_tcp()

    assert os.environ["MC_TCP_ENABLE_CONNECTION_POOL"] == "1"


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_dsa_mooncake_tcp_rejects_disabled_connection_pool(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("MC_TCP_ENABLE_CONNECTION_POOL", value)

    with pytest.raises(ValueError, match="requires MC_TCP_ENABLE_CONNECTION_POOL=1"):
        configure_dsa_mooncake_tcp()


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
    monkeypatch.delenv("MC_TCP_ENABLE_CONNECTION_POOL", raising=False)
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
    if protocol == "tcp":
        assert os.environ["MC_TCP_ENABLE_CONNECTION_POOL"] == "1"


def test_dsa_mooncake_lookup_reuses_scheduler_client_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
        MooncakeBackend,
        MooncakeStoreConfig,
    )

    config = SimpleNamespace(protocol="tcp")
    monkeypatch.setattr(
        MooncakeStoreConfig,
        "load_from_env",
        staticmethod(lambda: config),
    )
    scheduler_backend = _FakeBackend()
    observed: list[object] = []

    def create_scheduler_client(_cls, parallel_config):
        observed.append(parallel_config)
        return scheduler_backend

    monkeypatch.setattr(
        MooncakeBackend,
        "create_scheduler_client",
        classmethod(create_scheduler_client),
    )
    parallel_config = object()

    adapter = DSAL3BackendAdapter.create_mooncake(
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
        schema=_make_schema(),
        max_active_channels=6,
        max_descriptors_per_put=128,
        contribute_memory=False,
    )

    assert adapter.backend is scheduler_backend
    assert observed == [parallel_config]


class _FakeBackend:
    def __init__(self, result: list[int] | None = None) -> None:
        self.result = result or [0, 0]
        self.registered: tuple[list[int], list[int]] | None = None
        self.put_args: tuple[Any, Any, Any] | None = None
        self.put_calls: list[tuple[Any, Any, Any]] = []
        self.get_calls: list[tuple[Any, Any, Any]] = []
        self.exists_calls: list[list[str]] = []
        self.get_result: list[int] | None = None
        self.closed = False
        self.existing: list[int] | None = None
        self.stored_keys: set[str] = set()

    def register_buffer(self, ptrs: list[int], sizes: list[int]) -> None:
        self.registered = (ptrs, sizes)

    def put(self, keys, addresses, sizes):
        self.put_args = (keys, addresses, sizes)
        self.put_calls.append(self.put_args)
        if self.result is not None and len(self.result) == len(keys):
            self.stored_keys.update(key for key, result in zip(keys, self.result) if result >= 0)
        return self.result

    def exists(self, keys):
        self.exists_calls.append(list(keys))
        base = self.existing or [0] * len(keys)
        return [value if value < 0 else max(value, int(key in self.stored_keys)) for key, value in zip(keys, base)]

    def get(self, keys, addresses, sizes):
        self.get_calls.append((keys, addresses, sizes))
        return (
            self.get_result
            if self.get_result is not None
            else [0] * len(keys)
        )

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


def test_concurrent_put_success_is_not_published_before_all_components_are_readable() -> None:
    class CompetingBackend(_FakeBackend):
        def put(self, keys, addresses, sizes):
            self.put_calls.append((keys, addresses, sizes))
            # The peer owns these keys but has not completed the payload.
            return [0] * len(keys)

    store = _make_store()
    backend = CompetingBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    adapter.bind_store(store)
    with pytest.raises(RuntimeError, match="not yet readable"):
        adapter.put_store_job(job=_make_job(), store=store)
    assert len(backend.exists_calls) == 2
    backend.stored_keys.update(_make_job().blocks[0].component_keys)
    # Retry sees the winner's completed objects and avoids rewriting them.
    assert adapter.put_store_job(job=_make_job(), store=store) == 0
    assert len(backend.put_calls) == 1


def test_backend_adapter_round_trips_complete_mtp_bundle() -> None:
    store = _make_mtp_store()
    schema = _make_mtp_schema()
    backend = _FakeBackend(result=[0, 0, 0])
    adapter = DSAL3BackendAdapter(backend=backend, schema=schema)
    adapter.bind_store(store)
    block_hash = BlockHash(b"mtp-block")
    indexer_key, mla_key, mtp_key = schema.block_keys(block_hash)
    store_job = DSAL3StoreJob(
        job_id=0,
        blocks=(
            DSAL3StoreBlock(
                request_id="req",
                logical_block_index=0,
                source_l2_block_id=1,
                block_hash=block_hash,
                indexer_key=indexer_key,
                mla_key=mla_key,
                mtp_key=mtp_key,
            ),
        ),
    )

    put_bytes = adapter.put_store_job(job=store_job, store=store)
    assert backend.registered is not None
    assert len(backend.registered[0]) == 6
    assert backend.put_args is not None
    assert backend.put_args[0] == [indexer_key, mla_key, mtp_key]
    assert put_bytes == schema.bytes_per_logical_block

    load_job = DSAL3LoadJob(
        request_id="req",
        blocks=(
            DSAL3LoadBlock(
                logical_block_index=0,
                destination_l2_block_id=2,
                destination_indexer_hbm_block_id=1,
                destination_resident_hbm_block_id=1,
                destination_mtp_hbm_block_id=1,
                block_hash=block_hash,
                indexer_key=indexer_key,
                mla_key=mla_key,
                mtp_key=mtp_key,
            ),
        ),
    )
    load_bytes = adapter.get_load_job(job=load_job, store=store)
    assert len(backend.get_calls) == 1
    assert backend.get_calls[0][0] == [indexer_key, mla_key, mtp_key]
    assert load_bytes == schema.bytes_per_logical_block

    backend.existing = [1, 1, 1, 1, 1, 0]
    assert adapter.lookup_prefix_blocks((b"first", b"second")) == 1


def test_backend_adapter_direct_load_targets_split_hbm_views() -> None:
    store = _make_store()
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    adapter.bind_store(store, register_hbm=True)

    transferred = adapter.get_load_job(
        job=_make_load_job((1, 2), direct_hbm=True),
        store=store,
    )

    assert backend.registered is not None
    assert len(backend.registered[0]) == 6
    assert len(backend.get_calls) == 1
    keys, addresses, sizes = backend.get_calls[0]
    assert len(keys) == 4
    arenas = store.get_layer_arenas(0)
    assert addresses[0] == [
        int(plane[1].data_ptr()) for plane in arenas.indexer_source_bytes
    ]
    assert addresses[1] == [
        int(arenas.resident_source_bytes[0][1].data_ptr())
    ]
    assert sizes[0] == [128 * 128, 128 * 4]
    assert sizes[1] == [128 * 656]
    assert transferred == 2 * (128 * 128 + 128 * 4 + 128 * 656)


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


def test_backend_adapter_lookup_requires_both_components_per_block() -> None:
    backend = _FakeBackend()
    backend.existing = [1, 1, 1, 0, 1, 1]
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())

    hit_blocks = adapter.lookup_prefix_blocks(
        (b"hash-0", b"hash-1", b"hash-2")
    )

    assert hit_blocks == 1


def test_backend_adapter_loads_both_components_into_l2() -> None:
    store = _make_store()
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    adapter.bind_store(store)

    transferred = adapter.get_load_job(
        job=_make_load_job((1, 2)),
        store=store,
    )

    assert len(backend.get_calls) == 1
    assert len(backend.get_calls[0][0]) == 4
    assert transferred == 2 * (
        128 * 128 + 128 * 4 + 128 * 656
    )


def test_backend_adapter_rejects_incomplete_raw_get() -> None:
    store = _make_store()
    backend = _FakeBackend()

    def get_with_sizes(keys, _addresses, sizes):
        result = [sum(parts) for parts in sizes]
        result[-1] -= 1
        return result

    backend.get_with_sizes = get_with_sizes
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    adapter.bind_store(store)

    with pytest.raises(RuntimeError, match="incomplete payloads"):
        adapter.get_load_job(job=_make_load_job((1,)), store=store)


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


def test_fresh_l3_miss_invalidates_old_dedup_hints() -> None:
    schema = _make_schema()
    backend = _FakeBackend()
    backend.existing = [1, 1, 0, 0]
    adapter = DSAL3BackendAdapter(backend=backend, schema=schema)
    scheduler = DSAL3StoreScheduler(schema, max_inflight_jobs=2, lookup_backend=adapter, enable_load=True)
    scheduler.coordinator = SimpleNamespace(
        prepare_external_prefix_cache_hit=(
            lambda req, *, local_computed_tokens, external_block_count: external_block_count * 128
        ),
    )
    for block_hash in (b"a", b"b"):
        scheduler._remember_remote_ready_keys(schema.block_keys(block_hash))
    req = SimpleNamespace(num_tokens=257, block_hashes=[b"a", b"b"], request_id="req")
    assert scheduler.lookup_external_prefix(req, 0) == 128
    assert scheduler._remote_ready_keys == set(schema.block_keys(b"a"))


def test_scheduler_bounds_remote_ready_hints_to_l2_capacity() -> None:
    scheduler = DSAL3StoreScheduler(_make_schema(), max_inflight_jobs=2)
    scheduler._remote_ready_key_limit = 2

    scheduler._remember_remote_ready_keys(("indexer-a", "mla-a"))
    assert scheduler._remote_ready_keys == {"indexer-a", "mla-a"}

    scheduler._remember_remote_ready_keys(("indexer-b", "mla-b"))
    assert scheduler._remote_ready_keys == {"indexer-b", "mla-b"}


def test_scheduler_reset_rotates_remote_namespace() -> None:
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    scheduler = DSAL3StoreScheduler(
        _make_schema(),
        max_inflight_jobs=2,
        lookup_backend=adapter,
        enable_load=True,
    )
    scheduler.bind_coordinator(
        SimpleNamespace(
            dram_directory=SimpleNamespace(usable_blocks=8),
        )
    )
    old_key = scheduler.schema.block_keys(b"hash")[0]
    scheduler._remote_ready_keys.add(old_key)

    assert scheduler.reset_cache()

    new_key = scheduler.schema.block_keys(b"hash")[0]
    assert new_key != old_key
    assert adapter.schema is scheduler.schema
    assert not scheduler._remote_ready_keys
    scheduler.shutdown()


def test_scheduler_reset_rejects_pending_store() -> None:
    scheduler = DSAL3StoreScheduler(_make_schema(), max_inflight_jobs=2)
    scheduler._jobs[0] = _make_job()
    old_fingerprint = scheduler.schema.fingerprint

    assert not scheduler.reset_cache()
    assert scheduler.schema.fingerprint == old_fingerprint

    scheduler._jobs.clear()
    scheduler.shutdown()


def test_scheduler_queries_l3_suffix_and_builds_load_metadata() -> None:
    backend = _FakeBackend()
    backend.existing = [1, 1, 1, 1, 0, 0]
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    coordinator = SimpleNamespace()
    coordinator.prepare_external_prefix_cache_hit = (
        lambda request, *, local_computed_tokens, external_block_count: (
            external_block_count * 128
        )
    )
    coordinator.take_external_prefix_load_plan = lambda request_id: (
        SimpleNamespace(
            request_id=request_id,
            logical_block_start=1,
            block_hashes=(b"hash-1", b"hash-2"),
            destination_dram_block_ids=(2, 3),
            destination_indexer_hbm_block_ids=(12, 13),
            destination_resident_hbm_block_ids=(22, 23),
        )
    )
    scheduler = DSAL3StoreScheduler(
        _make_schema(),
        max_inflight_jobs=2,
        lookup_backend=adapter,
        enable_load=True,
    )
    scheduler.bind_coordinator(coordinator)
    request = SimpleNamespace(
        request_id="req",
        num_tokens=513,
        block_hashes=[
            BlockHash(b"hash-0"),
            BlockHash(b"hash-1"),
            BlockHash(b"hash-2"),
            BlockHash(b"hash-3"),
        ],
        skip_reading_prefix_cache=False,
    )

    external_tokens = scheduler.lookup_external_prefix(request, 128)
    scheduler.stage_external_load("req", external_tokens)
    metadata = scheduler.build_connector_meta()

    assert external_tokens == 256
    assert len(metadata.load_jobs) == 1
    assert metadata.load_jobs[0].request_id == "req"
    assert metadata.load_jobs[0].destination_l2_block_ids == (2, 3)
    assert metadata.load_jobs[0].destination_indexer_hbm_block_ids == (
        12,
        13,
    )
    assert metadata.load_jobs[0].destination_resident_hbm_block_ids == (
        22,
        23,
    )
    assert not metadata.load_jobs[0].direct_hbm
    assert not scheduler.build_connector_meta().load_jobs
    scheduler.shutdown()
    assert backend.closed


def test_scheduler_builds_mtp_component_load_metadata() -> None:
    schema = _make_mtp_schema()
    backend = _FakeBackend()
    backend.existing = [1, 1, 1]
    adapter = DSAL3BackendAdapter(backend=backend, schema=schema)
    coordinator = SimpleNamespace(
        dram_directory=SimpleNamespace(usable_blocks=8),
        prepare_external_prefix_cache_hit=(
            lambda request, *, local_computed_tokens, external_block_count: (
                external_block_count * 128
            )
        ),
        take_external_prefix_load_plan=lambda request_id: SimpleNamespace(
            request_id=request_id,
            logical_block_start=0,
            block_hashes=(b"hash",),
            destination_dram_block_ids=(2,),
            destination_indexer_hbm_block_ids=(12,),
            destination_resident_hbm_block_ids=(22,),
            destination_mtp_hbm_block_ids=(32,),
        ),
    )
    scheduler = DSAL3StoreScheduler(
        schema,
        max_inflight_jobs=2,
        lookup_backend=adapter,
        enable_load=True,
    )
    scheduler.bind_coordinator(coordinator)
    request = SimpleNamespace(
        request_id="req",
        num_tokens=129,
        block_hashes=[BlockHash(b"hash")],
        skip_reading_prefix_cache=False,
    )

    external_tokens = scheduler.lookup_external_prefix(request, 0)
    scheduler.stage_external_load("req", external_tokens)
    job = scheduler.build_connector_meta().load_jobs[0]

    assert external_tokens == 128
    assert job.destination_mtp_hbm_block_ids == (32,)
    assert job.blocks[0].mtp_key is not None
    scheduler.shutdown()


@pytest.mark.parametrize(
    ("existing", "expected_external_blocks"),
    [
        ([0, 0, 1, 1], 0),
        ([1, 1, 1, 1], 2),
    ],
)
def test_scheduler_l3_lookup_skips_l1_overlap(
    existing: list[int],
    expected_external_blocks: int,
) -> None:
    schema = _make_schema()
    backend = _FakeBackend()
    backend.existing = existing
    adapter = DSAL3BackendAdapter(backend=backend, schema=schema)
    observed: list[tuple[int, int]] = []

    def prepare_external(
        request,
        *,
        local_computed_tokens,
        external_block_count,
    ) -> int:
        del request
        observed.append((local_computed_tokens, external_block_count))
        return external_block_count * 128

    coordinator = SimpleNamespace(
        dram_directory=SimpleNamespace(usable_blocks=8),
        prepare_external_prefix_cache_hit=prepare_external,
    )
    scheduler = DSAL3StoreScheduler(
        schema,
        max_inflight_jobs=2,
        lookup_backend=adapter,
        enable_load=True,
    )
    scheduler.bind_coordinator(coordinator)
    hashes = [BlockHash(f"hash-{index}".encode()) for index in range(5)]
    request = SimpleNamespace(
        request_id="req",
        num_tokens=5 * 128 + 1,
        block_hashes=hashes,
        skip_reading_prefix_cache=False,
    )

    external_tokens = scheduler.lookup_external_prefix(
        request,
        3 * 128,
    )

    expected_keys = [
        key
        for block_hash in hashes[3:]
        for key in schema.block_keys(bytes(block_hash))
    ]
    assert backend.exists_calls == [expected_keys]
    assert observed == [(3 * 128, expected_external_blocks)]
    assert external_tokens == expected_external_blocks * 128
    scheduler.shutdown()


def test_scheduler_builds_direct_hbm_load_metadata() -> None:
    coordinator = SimpleNamespace(
        dram_directory=SimpleNamespace(usable_blocks=8),
        set_l3_direct_hbm_load_enabled=lambda enabled: None,
        take_external_prefix_load_plan=lambda request_id: SimpleNamespace(
            request_id=request_id,
            logical_block_start=0,
            block_hashes=(b"hash",),
            destination_dram_block_ids=(2,),
            destination_indexer_hbm_block_ids=(12,),
            destination_resident_hbm_block_ids=(22,),
        ),
    )
    scheduler = DSAL3StoreScheduler(
        _make_schema(),
        max_inflight_jobs=2,
        lookup_backend=DSAL3BackendAdapter(
            backend=_FakeBackend(),
            schema=_make_schema(),
        ),
        enable_load=True,
        direct_hbm_load=True,
    )
    scheduler.bind_coordinator(coordinator)

    scheduler.stage_external_load("req", 128)
    job = scheduler.build_connector_meta().load_jobs[0]

    assert job.direct_hbm
    assert job.destination_l2_block_ids == (2,)
    assert job.destination_indexer_hbm_block_ids == (12,)
    assert job.destination_resident_hbm_block_ids == (22,)
    scheduler.shutdown()


class _FakeEvent:
    def __init__(self) -> None:
        self.recorded = False
        self.synchronized = False

    def record(self) -> None:
        self.recorded = True

    def synchronize(self) -> None:
        assert self.recorded
        self.synchronized = True


class _RecordingConnectorWorker:
    def __init__(self) -> None:
        self.started = 0
        self.finished: list[DSAL3ConnectorMetadata] = []

    def start_step(self) -> None:
        self.started += 1

    def finish_step(self, metadata: DSAL3ConnectorMetadata) -> None:
        self.finished.append(metadata)


def _make_recording_worker_connector(
    *,
    has_mtp: bool,
) -> tuple[DSAL3Connector, _RecordingConnectorWorker]:
    connector = object.__new__(DSAL3Connector)
    worker = _RecordingConnectorWorker()
    connector.worker_handler = worker
    connector.scheduler_handler = None
    connector._connector_metadata = None
    connector._has_mtp_component = has_mtp
    connector._defer_store_until_wait = False
    connector._deferred_store_metadata = None
    return connector, worker


def test_mtp_connector_submits_store_only_after_proposer_finalize() -> None:
    connector, worker = _make_recording_worker_connector(has_mtp=True)
    metadata = DSAL3ConnectorMetadata(store_jobs={0: _make_job()})
    connector.bind_connector_metadata(metadata)

    connector.start_load_kv(SimpleNamespace(attn_metadata={}))
    connector.get_finished(set())

    assert worker.started == 1
    assert worker.finished == []
    assert connector._deferred_store_metadata is metadata

    connector.wait_for_save()

    assert worker.finished == [metadata]
    assert connector._deferred_store_metadata is None
    connector.wait_for_save()
    assert worker.finished == [metadata]


def test_mtp_connector_only_step_submits_store_immediately() -> None:
    connector, worker = _make_recording_worker_connector(has_mtp=True)
    metadata = DSAL3ConnectorMetadata(store_jobs={0: _make_job()})
    connector.bind_connector_metadata(metadata)

    connector.start_load_kv(SimpleNamespace(attn_metadata=None))
    connector.get_finished(set())

    assert worker.finished == [metadata]
    assert connector._deferred_store_metadata is None


def test_target_only_connector_keeps_immediate_store_submission() -> None:
    connector, worker = _make_recording_worker_connector(has_mtp=False)
    metadata = DSAL3ConnectorMetadata(store_jobs={0: _make_job()})
    connector.bind_connector_metadata(metadata)

    connector.start_load_kv(SimpleNamespace(attn_metadata={}))
    connector.get_finished(set())

    assert worker.finished == [metadata]
    assert connector._deferred_store_metadata is None


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


def test_nonproducer_load_worker_registers_and_restores_local_l2() -> None:
    store = _make_store()
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    worker = DSAL3StoreWorker(
        tp_rank=1,
        device=torch.device("cpu"),
        backend=adapter,
        max_inflight_jobs=2,
        enable_load=True,
        event_factory=_FakeEvent,
    )
    metadata = DSAL3ConnectorMetadata(
        load_jobs=(_make_load_job((1, 2)),),
    )

    worker.bind_store(store)
    worker.prepare_step(metadata)

    assert backend.registered is not None
    assert len(backend.get_calls) == 1
    assert worker._load_jobs == 1
    assert worker._load_blocks == 2
    worker.shutdown()


def test_nonproducer_direct_load_worker_registers_hbm() -> None:
    store = _make_store()
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    worker = DSAL3StoreWorker(
        tp_rank=1,
        device=torch.device("cpu"),
        backend=adapter,
        max_inflight_jobs=2,
        enable_load=True,
        direct_hbm_load=True,
        event_factory=_FakeEvent,
    )
    metadata = DSAL3ConnectorMetadata(
        load_jobs=(_make_load_job((1, 2), direct_hbm=True),),
    )

    worker.bind_store(store)
    worker.prepare_step(metadata)

    assert backend.registered is not None
    assert len(backend.registered[0]) == 6
    assert worker._load_jobs == 1
    assert worker._load_blocks == 2
    worker.shutdown()


def test_load_worker_fails_before_forward_on_any_component_error() -> None:
    store = _make_store()
    backend = _FakeBackend()
    backend.get_result = [0, -1]
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    worker = DSAL3StoreWorker(
        tp_rank=1,
        device=torch.device("cpu"),
        backend=adapter,
        max_inflight_jobs=2,
        enable_load=True,
        event_factory=_FakeEvent,
    )
    worker.bind_store(store)

    with pytest.raises(RuntimeError, match="get failed"):
        worker.prepare_step(
            DSAL3ConnectorMetadata(
                load_jobs=(_make_load_job(),),
            )
        )

    worker.shutdown()


def test_load_worker_rejects_peer_rank_failure_before_forward() -> None:
    store = _make_store()
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    worker = DSAL3StoreWorker(
        tp_rank=1,
        device=torch.device("cpu"),
        backend=adapter,
        max_inflight_jobs=2,
        enable_load=True,
        event_factory=_FakeEvent,
        load_consensus=lambda _local_success: False,
    )
    worker.bind_store(store)

    with pytest.raises(RuntimeError, match="peer TP rank"):
        worker.prepare_step(
            DSAL3ConnectorMetadata(
                load_jobs=(_make_load_job(),),
            )
        )

    assert worker._load_jobs == 0
    worker.shutdown()


def test_rank0_reuse_fence_error_joins_load_consensus() -> None:
    store = _make_store()
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    observed: list[bool] = []

    def consensus(local_success: bool) -> bool:
        observed.append(local_success)
        return False

    worker = DSAL3StoreWorker(
        tp_rank=0,
        device=torch.device("cpu"),
        backend=adapter,
        max_inflight_jobs=2,
        enable_load=True,
        event_factory=_FakeEvent,
        load_consensus=consensus,
    )
    worker.bind_store(store)

    with pytest.raises(RuntimeError, match="unknown job"):
        worker.prepare_step(
            DSAL3ConnectorMetadata(
                jobs_to_flush={99},
                load_jobs=(_make_load_job(),),
            )
        )

    assert observed == [False]
    assert not backend.get_calls
    worker.shutdown()


@pytest.mark.parametrize("dp_size", [1, 2])
def test_worker_connector_initializes_backend_on_nonzero_tp_rank(
    monkeypatch: pytest.MonkeyPatch,
    dp_size: int,
) -> None:
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_rank=dp_size - 1, data_parallel_size=dp_size),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_producer",
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
    requested_devices: list[str] = []

    def make_device(spec: str) -> torch.device:
        requested_devices.append(spec)
        return torch.device("cpu")

    # Validate NPU selection without requiring native device registration in this UT.
    monkeypatch.setattr(
        l3_connector_module,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 3), device=make_device),
    )

    connector = DSAL3Connector(
        config,
        KVConnectorRole.WORKER,
        SimpleNamespace(),
    )

    assert connector.worker_handler is not None
    assert connector.worker_handler.tp_rank == 3
    assert connector.worker_handler.backend is adapter
    assert not connector.worker_handler.is_store_rank
    assert requested_devices == ["npu:3"]
    assert connector.data_parallel_rank == dp_size - 1
    assert connector.data_parallel_size == dp_size
    assert connector.supports_dsa_namespace_reset == (dp_size == 1)
    connector.shutdown()
    assert backend.closed


@pytest.mark.parametrize("dp_size", [1, 2])
def test_scheduler_connector_uses_noncontributing_lookup_client(
    monkeypatch: pytest.MonkeyPatch,
    dp_size: int,
) -> None:
    backend = _FakeBackend()
    adapter = DSAL3BackendAdapter(backend=backend, schema=_make_schema())
    observed: list[bool] = []
    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        parallel_config=SimpleNamespace(data_parallel_rank=dp_size - 1, data_parallel_size=dp_size),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={"backend": "mooncake"},
        ),
    )
    monkeypatch.setattr(
        DSAL3KeySchema,
        "from_config",
        classmethod(
            lambda _cls, _vllm_config, _kv_cache_config: _make_schema()
        ),
    )

    def create_backend(_cls, **kwargs):
        observed.append(kwargs["contribute_memory"])
        return adapter

    monkeypatch.setattr(
        DSAL3BackendAdapter,
        "create_mooncake",
        classmethod(create_backend),
    )

    connector = DSAL3Connector(
        config,
        KVConnectorRole.SCHEDULER,
        SimpleNamespace(),
    )
    connector.bind_dsa_coordinator(SimpleNamespace())

    assert observed == [False]
    assert connector.scheduler_handler is not None
    assert connector.scheduler_handler.enable_load
    if dp_size > 1:
        assert not connector.supports_dsa_namespace_reset
        assert connector.reset_cache() is False
        assert connector.scheduler_handler._namespace_generation == 0
    connector.shutdown()
    assert backend.closed


def test_scheduler_connector_rejects_frozen_direct_hbm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config={
                "backend": "mooncake",
                "direct_hbm_load": True,
            },
        ),
    )
    monkeypatch.setattr(
        DSAL3KeySchema,
        "from_config",
        classmethod(
            lambda _cls, _vllm_config, _kv_cache_config: _make_schema()
        ),
    )
    monkeypatch.setattr(
        DSAL3BackendAdapter,
        "create_mooncake",
        classmethod(
            lambda _cls, **_kwargs: pytest.fail(
                "frozen direct-HBM must fail before backend creation"
            )
        ),
    )
    with pytest.raises(ValueError, match="is frozen"):
        DSAL3Connector(
            config,
            KVConnectorRole.SCHEDULER,
            SimpleNamespace(),
        )


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
