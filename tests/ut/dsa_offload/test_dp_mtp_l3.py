# SPDX-License-Identifier: Apache-2.0
"""P6-A跨域MTP L3合同：三组件可读性、proposer时序和局部TP失败隔离。

复用真实connector/scheduler/worker与CPU字节地址，仅替换网络及设备copy/event。
两个DP始终拥有独立runtime、job目录和L2地址，不从examples导入实现。
"""

import ctypes
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
    _make_recording_worker_connector,
)
from tests.ut.dsa_offload.test_prefix_cache import _bind_target_hbm, _HBMManager, _prime_dram_prefix, _request
from tests.ut.dsa_offload.test_runtime import _make_runtime
from vllm_ascend.dsa_offload import runtime as runtime_module
from vllm_ascend.dsa_offload.kv_cache_coordinator import DSAKVCacheCoordinator
from vllm_ascend.dsa_offload.l3.backend_adapter import DSAL3BackendAdapter
from vllm_ascend.dsa_offload.l3.metadata import DSAL3LoadBlock, DSAL3LoadJob
from vllm_ascend.dsa_offload.l3.scheduler import DSAL3StoreScheduler
from vllm_ascend.dsa_offload.l3.worker import DSAL3StoreWorker
from vllm_ascend.dsa_offload.runtime import DSAMTPFullCacheOffloadContext
from vllm_ascend.dsa_offload.scheduler_dram_pool import DSASchedulerDRAMBlockDirectory


class _SharedPayloadBackend(_FakeBackend):
    def __init__(self):
        super().__init__()
        self.payloads = {}
        self.mtp_pending = False
        self.short_mtp_get = False

    def exists(self, keys):
        return [int(key in self.payloads) for key in keys]

    def put(self, keys, addresses, sizes):
        self.put_calls.append((list(keys), addresses, sizes))
        for key, pointers, lengths in zip(keys, addresses, sizes, strict=True):
            if self.mtp_pending and ":mtp:" in key:
                continue  # 模拟对端已claim但尚不可读，API仍返回0。
            self.payloads.setdefault(
                key,
                tuple(ctypes.string_at(ptr, size) for ptr, size in zip(pointers, lengths, strict=True)),
            )
        return [0] * len(keys)

    def get_with_sizes(self, keys, addresses, sizes):
        self.get_calls.append((list(keys), addresses, sizes))
        result = []
        for key, pointers, lengths in zip(keys, addresses, sizes, strict=True):
            if key not in self.payloads:
                result.append(-1)
                continue
            for value, ptr, size in zip(self.payloads[key], pointers, lengths, strict=True):
                assert len(value) == size
                ctypes.memmove(ptr, value, size)
            result.append(sum(lengths) - int(self.short_mtp_get and ":mtp:" in key))
        return result


def test_mtp_dp_local_and_external_prefixes_drop_one_boundary_per_request():
    plans = []
    for local_blocks in (0, 4):
        coordinator = object.__new__(DSAKVCacheCoordinator)
        directory = DSASchedulerDRAMBlockDirectory(
            usable_blocks=8,
            max_logical_blocks=8,
            block_size=128,
            enable_caching=True,
        )
        if local_blocks:
            _prime_dram_prefix(directory, local_blocks)
        coordinator.dram_directory = directory
        coordinator.dram_block_size = coordinator.scheduler_block_size = coordinator.hash_block_size = 128
        coordinator.num_single_type_manager = 3
        coordinator.group_ids = SimpleNamespace(indexer=0, resident_mla=1, mtp_full=2)
        coordinator.mtp_full_manager = _HBMManager()
        coordinator._pending_prefix_hits = {}
        _bind_target_hbm(coordinator)
        request = _request("same-request-id", 6)
        _, local_tokens = coordinator.prepare_prefix_cache_hit(request, 6 * 128)
        assert local_tokens == max(0, local_blocks - 1) * 128
        external = coordinator.prepare_external_prefix_cache_hit(
            request,
            local_computed_tokens=local_tokens,
            external_block_count=6 - local_tokens // 128,
        )
        assert local_tokens + external == 5 * 128
        plans.append(coordinator.get_pending_prefix_hit(request.request_id))
    assert plans[0] is not plans[1]
    assert plans[0].num_computed_tokens == plans[1].num_computed_tokens == 5 * 128
    assert len(plans[0].external_block_hashes) == 5
    assert len(plans[1].external_block_hashes) == 2


def _plan(scheduler, block_hash=b"shared", request_id="req"):
    metadata = scheduler.build_connector_meta()
    scheduler.extend_store_jobs(
        metadata=metadata,
        projection=SimpleNamespace(request_ids=(request_id,), dram_ready_block_counts=(1,), dram_block_assignments=()),
        coordinator=SimpleNamespace(dram_directory=SimpleNamespace(get_request_blocks=lambda _: (1,))),
        requests={request_id: SimpleNamespace(block_hashes=[BlockHash(block_hash)])},
    )
    return metadata


def _load(schema, block_hash=b"shared"):
    keys = schema.block_keys(block_hash)
    return DSAL3LoadJob(
        request_id="consumer",
        blocks=(
            DSAL3LoadBlock(
                logical_block_index=0,
                destination_l2_block_id=2,
                destination_indexer_hbm_block_id=1,
                destination_resident_hbm_block_id=1,
                destination_mtp_hbm_block_id=1,
                block_hash=block_hash,
                indexer_key=keys[0],
                mla_key=keys[1],
                mtp_key=keys[2],
            ),
        ),
    )


def _planes(store):
    target = store.get_layer_arenas(0)
    return (*target.indexer_dram_bytes, *target.resident_dram_bytes, *store.get_mtp_layer_arenas(0).dram_bytes)


def test_partial_mtp_bundle_cannot_become_ready_and_retry_only_writes_missing_component():
    schema = _make_mtp_schema()
    backend = _SharedPayloadBackend()
    backend.mtp_pending = True
    writer_store = _make_mtp_store()
    for marker, plane in enumerate(_planes(writer_store), start=11):
        plane[1].fill_(marker)
    adapter = DSAL3BackendAdapter(backend=backend, schema=schema, max_descriptors_per_put=3)
    writer = DSAL3StoreWorker(
        tp_rank=0, device=writer_store.device, backend=adapter, max_inflight_jobs=2, event_factory=_FakeEvent
    )
    writer.bind_store(writer_store)
    scheduler = DSAL3StoreScheduler(schema, max_inflight_jobs=2)
    try:
        first = _plan(scheduler)
        writer.finish_step(first)
        writer.flush({0})
        outcome = writer.build_worker_meta()
        assert not outcome.results[0].success and outcome.results[0].retryable
        scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=outcome))
        assert not scheduler._remote_ready_keys
        assert len(backend.payloads) == 2
        assert adapter.lookup_prefix_blocks((b"shared",)) == 0

        backend.mtp_pending = False
        backend.put_calls.clear()
        retry = _plan(scheduler)
        writer.finish_step(retry)
        writer.flush({1})
        outcome = writer.build_worker_meta()
        assert outcome.results[1].success
        assert outcome.results[1].transferred_bytes == schema.mtp_bytes_per_block
        scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=outcome))
        assert scheduler._remote_ready_keys == set(schema.block_keys(b"shared"))
        assert backend.put_calls[0][0] == [schema.block_keys(b"shared")[2]]
        assert adapter.lookup_prefix_blocks((b"shared",)) == 1

        # 两个独立TP副本读入不同地址；单卡短读不得提交本域load成功状态。
        for tp_rank, short_read, peer_success in ((0, False, True), (1, True, False), (2, False, False)):
            consumer = _make_mtp_store()
            reader = DSAL3BackendAdapter(backend=backend, schema=schema, max_descriptors_per_put=3)
            votes = []

            def consensus(success, votes=votes, peer_success=peer_success):
                votes.append(success)
                return peer_success

            worker = DSAL3StoreWorker(
                tp_rank=tp_rank,
                device=consumer.device,
                backend=reader,
                max_inflight_jobs=2,
                enable_load=True,
                event_factory=_FakeEvent,
                load_consensus=consensus,
            )
            worker.bind_store(consumer)
            backend.short_mtp_get = short_read
            try:
                if peer_success:
                    worker.load_sync((_load(schema),))
                    for source, destination in zip(_planes(writer_store), _planes(consumer), strict=True):
                        assert torch.equal(source[1], destination[2])
                    assert worker._load_jobs == 1
                else:
                    with pytest.raises(RuntimeError, match="L3 load failed"):
                        worker.load_sync((_load(schema),))
                    assert worker._load_jobs == 0
                assert votes == [not short_read]
            finally:
                worker.shutdown()
        # 后端淘汰MTP一个组件后，再次查询不能沿用旧的完整bundle命中。
        del backend.payloads[schema.block_keys(b"shared")[2]]
        assert adapter.lookup_prefix_blocks((b"shared",)) == 0
    finally:
        writer.shutdown()


@pytest.mark.parametrize("prewrite", [False, True])
def test_each_dp_releases_its_own_mtp_store_after_real_proposer_copy(monkeypatch, prewrite):
    def copy_blocks(source, destination, source_ids, destination_ids):
        for src, dst in zip(source_ids.tolist(), destination_ids.tolist(), strict=True):
            if dst >= 0:
                destination[dst].copy_(source[src])

    monkeypatch.setattr(
        runtime_module.torch.ops._C_ascend, "kv_cache_full_block_dump_bytes", copy_blocks, raising=False
    )
    schema = _make_mtp_schema()
    backend = _SharedPayloadBackend()
    connectors, workers, runtimes, stores, metadata = [], [], [], [], []
    try:
        for rank in range(2):
            _, runtime, store = _make_runtime(
                packed_c8=True,
                resident_layer_count=1,
                selection_state_count=1,
                max_decode_query_len=4,
                enable_prefix_caching=True,
                mtp_durable_offload=True,
            )
            store.add_packed_layer(
                layer_id=0,
                resident_packed_cache=torch.zeros(4, 128, 1, 656, dtype=torch.int8),
                indexer_cache=(torch.zeros(4, 128, 1, 128, dtype=torch.int8), torch.zeros(4, 128, 1, 1)),
            )
            store.add_mtp_layer(
                layer_id=0,
                mtp_full_cache=tuple(torch.zeros(4, 128, 1, width, dtype=torch.bfloat16) for width in (512, 64, 128)),
            )
            runtime._forward_epoch = 7 + rank
            runtime.dump_job_count = runtime.dump_launch_count = 1
            runtime.dump_before_cache_write = prewrite
            runtime.dump_mtp_src_block_ids.cpu[0] = 2
            runtime.dump_mtp_src_block_ids.gpu[0] = 2
            runtime.dump_dst_block_ids.cpu[0] = 1
            runtime.dump_dst_block_ids.gpu[0] = 1
            for plane in store.get_mtp_layer_arenas(0).source_bytes:
                plane[2].fill_(41 + rank)

            # ready-event的record时点必须晚于本域的proposer copy与校验。
            class CheckedEvent(_FakeEvent):
                def __init__(self, runtime=runtime):
                    super().__init__()
                    self.runtime = runtime

                def record(self):
                    self.runtime.validate_mtp_dump_completion()
                    super().record()

            adapter = DSAL3BackendAdapter(backend=backend, schema=schema)
            worker = DSAL3StoreWorker(
                tp_rank=0, device=store.device, backend=adapter, max_inflight_jobs=2, event_factory=CheckedEvent
            )
            worker.bind_store(store)
            workers.append(worker)
            connector, _ = _make_recording_worker_connector(has_mtp=True)
            connector.worker_handler = worker
            plan = _plan(DSAL3StoreScheduler(schema, max_inflight_jobs=2), f"dp{rank}".encode())
            connector.bind_connector_metadata(plan)
            connector.start_load_kv(SimpleNamespace(attn_metadata={}))
            connector.get_finished(set())
            connectors.append(connector)
            runtimes.append(runtime)
            stores.append(store)
            metadata.append(plan)
        assert not backend.put_calls
        assert all(set(plan.store_jobs) == {0} for plan in metadata)
        with pytest.raises(RuntimeError, match="before every MTP"):
            runtimes[1].validate_mtp_dump_completion()
        with pytest.raises(RuntimeError, match="dummy forward"):
            connectors[1].start_load_kv(SimpleNamespace(skip_kv_transfer=True, attn_metadata={}))
        assert connectors[1]._deferred_store_metadata is metadata[1]

        for rank in range(2):
            DSAMTPFullCacheOffloadContext(layer_id=0, runtime=runtimes[rank]).dump_full_blocks(
                before_cache_write=prewrite
            )
            runtimes[rank].validate_mtp_dump_completion()
            connectors[rank].wait_for_save()
            workers[rank].flush({0})
            assert workers[rank].build_worker_meta().results[0].success
            mtp_key = schema.block_keys(f"dp{rank}".encode())[2]
            assert all(set(plane) == {41 + rank} for plane in backend.payloads[mtp_key])
            if rank == 0:
                assert connectors[1]._deferred_store_metadata is metadata[1]
                assert runtimes[1]._mtp_layer_dump_epochs == {}
                assert schema.block_keys(b"dp1")[2] not in backend.payloads
            connectors[rank].wait_for_save()  # 不重复提交相同job。
        assert len(backend.payloads) == 6
    finally:
        for worker in workers:
            worker.shutdown()
