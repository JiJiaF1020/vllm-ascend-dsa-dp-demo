# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA L3 all-rank pool contributor 与 TP-rank0 write-behind worker。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.dsa_offload.l3.backend_adapter import DSAL3BackendAdapter
from vllm_ascend.dsa_offload.l3.metadata import (
    DSAL3ConnectorMetadata,
    DSAL3JobResult,
    DSAL3StoreJob,
    DSAL3WorkerMetadata,
)
from vllm_ascend.dsa_offload.worker_dram_store import DSAWorkerDRAMStore

EventFactory = Callable[[], Any]


class DSAL3StoreWorker:
    """所有 TP rank 贡献 Mooncake segment，仅 TP0 执行副本 put。"""

    def __init__(
        self,
        *,
        tp_rank: int,
        device: torch.device,
        backend: DSAL3BackendAdapter,
        max_inflight_jobs: int,
        event_factory: EventFactory | None = None,
    ) -> None:
        self.tp_rank = int(tp_rank)
        self.device = torch.device(device)
        self.backend = backend
        self.max_inflight_jobs = int(max_inflight_jobs)
        if self.max_inflight_jobs <= 0:
            raise ValueError("DSA L3 max_inflight_jobs must be positive")
        self.store: DSAWorkerDRAMStore | None = None
        self._event_factory = event_factory
        self._available_events: list[Any] = []
        self._event_lock = threading.Lock()
        self._executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="dsa-l3-store",
            )
            if self.is_store_rank
            else None
        )
        self._futures: dict[int, Future[DSAL3JobResult]] = {}
        self._completed: dict[int, DSAL3JobResult] = {}
        self._submitted_jobs = 0
        self._submitted_blocks = 0
        self._finished_jobs = 0
        self._failed_jobs = 0
        self._transferred_bytes = 0

    @property
    def is_store_rank(self) -> bool:
        return self.tp_rank == 0

    def bind_store(self, store: DSAWorkerDRAMStore) -> None:
        if self.store is not None:
            raise RuntimeError("DSA L3 worker DRAM store was bound twice")
        self.store = store
        if not self.is_store_rank:
            return
        self.backend.bind_store(store)
        factory = self._event_factory or torch.npu.Event
        self._available_events = [factory() for _ in range(self.max_inflight_jobs)]

    def _take_event(self) -> Any:
        with self._event_lock:
            if not self._available_events:
                raise RuntimeError(
                    "DSA L3 store event pool is exhausted; lower store "
                    "production or increase max_inflight_l3_store_jobs"
                )
            return self._available_events.pop()

    def _return_event(self, event: Any) -> None:
        with self._event_lock:
            self._available_events.append(event)

    def _execute_store(
        self,
        job: DSAL3StoreJob,
        ready_event: Any,
    ) -> DSAL3JobResult:
        try:
            if self.device.type == "npu":
                torch.npu.set_device(self.device)
            ready_event.synchronize()
            if self.store is None:
                raise RuntimeError("DSA L3 worker is not bound")
            transferred = self.backend.put_store_job(
                job=job,
                store=self.store,
            )
            return DSAL3JobResult(
                job_id=job.job_id,
                success=True,
                transferred_bytes=transferred,
            )
        except Exception as exc:
            return DSAL3JobResult(
                job_id=job.job_id,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._return_event(ready_event)

    def _record_finished(
        self,
        job_id: int,
        result: DSAL3JobResult,
    ) -> None:
        self._completed[job_id] = result
        self._finished_jobs += 1
        self._transferred_bytes += int(result.transferred_bytes)
        if not result.success:
            self._failed_jobs += 1
        logger.info(
            "DSA L3 store completed: job=%d success=%s bytes=%d error=%s",
            job_id,
            result.success,
            result.transferred_bytes,
            result.error,
        )

    def _poll_finished(self) -> None:
        for job_id, future in tuple(self._futures.items()):
            if not future.done():
                continue
            self._record_finished(job_id, future.result())
            self._futures.pop(job_id)

    def _submit(self, job: DSAL3StoreJob) -> None:
        if job.job_id in self._futures or job.job_id in self._completed:
            raise RuntimeError(f"DSA L3 worker received duplicate job {job.job_id}")
        ready_event = self._take_event()
        # full-block dump 与 forward 使用同一当前 stream。后台线程只有在
        # event 完成后才读取 swapped arena，IO 不进入 eager/graph 数据面。
        ready_event.record()
        if self._executor is None:
            raise RuntimeError("DSA L3 non-producer rank received a store job")
        self._futures[job.job_id] = self._executor.submit(
            self._execute_store,
            job,
            ready_event,
        )
        self._submitted_jobs += 1
        self._submitted_blocks += len(job.blocks)
        logger.info(
            "DSA L3 store submitted: job=%d blocks=%d bytes_per_block=%d",
            job.job_id,
            len(job.blocks),
            self.backend.schema.bytes_per_logical_block,
        )

    def start_step(self) -> None:
        if self.is_store_rank:
            self._poll_finished()

    def finish_step(self, metadata: DSAL3ConnectorMetadata) -> None:
        if not self.is_store_rank:
            return
        if self.store is None:
            raise RuntimeError("DSA L3 worker has no bound DRAM store")
        self._poll_finished()
        for job in metadata.store_jobs.values():
            self._submit(job)
        self._poll_finished()

    def flush(self, job_ids: set[int]) -> None:
        if not self.is_store_rank:
            return
        for job_id in sorted(job_ids):
            future = self._futures.get(job_id)
            if future is None:
                if job_id in self._completed:
                    continue
                raise RuntimeError(f"DSA L3 reuse fence references unknown job {job_id}")
            self._record_finished(job_id, future.result())
            self._futures.pop(job_id)

    def build_worker_meta(self) -> DSAL3WorkerMetadata | None:
        if not self.is_store_rank:
            return None
        self._poll_finished()
        if not self._completed:
            return None
        results = self._completed
        self._completed = {}
        return DSAL3WorkerMetadata(results=results)

    def shutdown(self) -> None:
        try:
            if self.is_store_rank:
                self.flush(set(self._futures))
        finally:
            logger.info(
                "DSA L3 worker summary: tp_rank=%d submitted_jobs=%d "
                "submitted_blocks=%d finished_jobs=%d failed_jobs=%d "
                "transferred_bytes=%d pending_results=%d",
                self.tp_rank,
                self._submitted_jobs,
                self._submitted_blocks,
                self._finished_jobs,
                self._failed_jobs,
                self._transferred_bytes,
                len(self._completed),
            )
            try:
                self.backend.close()
            finally:
                if self._executor is not None:
                    self._executor.shutdown(
                        wait=True,
                        cancel_futures=False,
                    )
