# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""共享后端、独立 DP job ID 与并发 writer 的控制面验证。"""

import ctypes
import threading
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.outputs import KVConnectorOutput

from tests.ut.dsa_offload.test_l3 import (
    _FakeBackend,
    _FakeEvent,
    _make_mtp_schema,
    _make_mtp_store,
    _make_schema,
    _make_store,
)
from vllm_ascend.dsa_offload.l3.backend_adapter import DSAL3BackendAdapter
from vllm_ascend.dsa_offload.l3.metadata import DSAL3LoadBlock, DSAL3LoadJob
from vllm_ascend.dsa_offload.l3.scheduler import DSAL3StoreScheduler
from vllm_ascend.dsa_offload.l3.worker import DSAL3StoreWorker


@pytest.mark.parametrize("mtp", [False, True])
def test_two_dp_writers_confirm_readability_before_publishing_ready_hints(mtp):
    claimed = threading.Event()
    finish = threading.Event()

    class SharedBackend(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.claimed = False
            self.payloads = {}
            self.lock = threading.Lock()

        def exists(self, keys):
            with self.lock:
                return [int(key in self.payloads) for key in keys]

        def put(self, keys, addresses, sizes):
            with self.lock:
                if self.claimed:
                    return [0] * len(keys)  # Mooncake duplicate-key semantics.
                self.claimed = True
            claimed.set()
            assert finish.wait(5), "test owner was not released"
            with self.lock:
                for key, pointers, lengths in zip(keys, addresses, sizes, strict=True):
                    self.payloads[key] = [ctypes.string_at(pointer, size) for pointer, size in zip(pointers, lengths)]
            return [0] * len(keys)

        def get(self, keys, addresses, sizes):
            for key, pointers, lengths in zip(keys, addresses, sizes, strict=True):
                for payload, pointer, length in zip(self.payloads[key], pointers, lengths, strict=True):
                    assert len(payload) == length
                    ctypes.memmove(pointer, payload, length)
            return [0] * len(keys)

    backend = SharedBackend()
    schema = _make_mtp_schema() if mtp else _make_schema()
    make_store = _make_mtp_store if mtp else _make_store

    def planes(store):
        arenas = store.get_layer_arenas(0)
        return (
            *arenas.indexer_dram_bytes,
            *arenas.resident_dram_bytes,
            *(store.get_mtp_layer_arenas(0).dram_bytes if mtp else ()),
        )

    schedulers = [DSAL3StoreScheduler(schema, max_inflight_jobs=2) for _ in range(2)]
    workers = []
    for _ in range(2):
        store = make_store()
        for plane in planes(store):
            plane[1].fill_(37)
        adapter = DSAL3BackendAdapter(backend=backend, schema=schema)
        worker = DSAL3StoreWorker(
            tp_rank=0,
            device=store.device,
            backend=adapter,
            max_inflight_jobs=2,
            event_factory=_FakeEvent,
        )
        worker.bind_store(store)
        workers.append(worker)
    projection = SimpleNamespace(request_ids=("req",), dram_ready_block_counts=(1,), dram_block_assignments=())
    coordinator = SimpleNamespace(dram_directory=SimpleNamespace(get_request_blocks=lambda req: (1,)))
    requests = {"req": SimpleNamespace(block_hashes=[BlockHash(b"shared")])}

    def plan(scheduler):
        metadata = scheduler.build_connector_meta()
        scheduler.extend_store_jobs(
            metadata=metadata, projection=projection, coordinator=coordinator, requests=requests
        )
        return metadata

    try:
        first, second = [plan(scheduler) for scheduler in schedulers]
        assert list(first.store_jobs) == list(second.store_jobs) == [0]
        workers[0].finish_step(first)
        assert claimed.wait(5)
        workers[1].finish_step(second)
        workers[1].flush({0})
        result1 = workers[1].build_worker_meta()
        assert result1.results[0].retryable and not result1.results[0].success
        schedulers[1].update_connector_output(KVConnectorOutput(kv_connector_worker_meta=result1))
        assert schedulers[1]._remote_ready_keys == set()
        assert len(schedulers[0]._jobs) == 1

        finish.set()
        workers[0].flush({0})
        result0 = workers[0].build_worker_meta()
        assert result0.results[0].success
        schedulers[0].update_connector_output(KVConnectorOutput(kv_connector_worker_meta=result0))
        retry = plan(schedulers[1])
        assert list(retry.store_jobs) == [1]
        workers[1].finish_step(retry)
        workers[1].flush({1})
        retry_result = workers[1].build_worker_meta()
        assert retry_result.results[1].success
        assert retry_result.results[1].transferred_bytes == 0
        schedulers[1].update_connector_output(KVConnectorOutput(kv_connector_worker_meta=retry_result))
        assert schedulers[0]._remote_ready_keys == schedulers[1]._remote_ready_keys == set(schema.block_keys(b"shared"))
        assert len(backend.payloads) == (3 if mtp else 2)
        # Independent consumer L2, with the same logical block but new storage.
        consumer = make_store()
        reader = DSAL3BackendAdapter(backend=backend, schema=schema)
        reader.bind_store(consumer)
        keys = schema.block_keys(b"shared")
        load = DSAL3LoadJob(
            request_id="consumer",
            blocks=(
                DSAL3LoadBlock(
                    logical_block_index=0,
                    destination_l2_block_id=2,
                    destination_indexer_hbm_block_id=1,
                    destination_resident_hbm_block_id=1,
                    destination_mtp_hbm_block_id=1 if mtp else None,
                    block_hash=b"shared",
                    indexer_key=keys[0],
                    mla_key=keys[1],
                    mtp_key=keys[2] if mtp else None,
                ),
            ),
        )
        assert reader.get_load_job(job=load, store=consumer) == schema.bytes_per_logical_block
        for plane in planes(consumer):
            assert torch.all(plane[2] == 37)
    finally:
        finish.set()
        for worker in workers:
            worker.shutdown()
