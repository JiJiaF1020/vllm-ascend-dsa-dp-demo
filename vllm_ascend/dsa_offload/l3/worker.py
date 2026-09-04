# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 在每个 TP worker 中执行的 L3 加载与异步上传任务。

绑定本 rank 的 DRAM store 和 backend 后，所有 TP rank 按 scheduler 下发的
计划同步 load 到各自 L2，并在本 TP group 内汇总成功状态，再允许后续
assignment/restore 和 forward 继续。各 rank 的存储容量贡献由 backend
配置决定，与是否承担上传任务是两个独立职责。

上传仅由 TP0 的后台执行器处理：记录当前 NPU stream 的完成 event，后台
任务等待 L1 -> L2 dump 完成后才读取源 arena。主执行线程轮询 Future，
将 job 结果整理为 worker metadata；需要覆盖仍在使用的 L2 源块时，按
``jobs_to_flush`` 等待相关旧任务完成，关闭时排空任务并释放 backend。

本模块不分配 scheduler block ID，也不在后台线程中直接更新 scheduler
目录或 prefix hash。当前同步 load、TP0-only store 的边界不等于已支持
DP 下的共享 L3 组合，是否开放仍由配置门禁决定。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
import torch.distributed as dist
from vllm.logger import logger

from vllm_ascend.dsa_offload.l3.backend_adapter import DSAL3BackendAdapter, DSAL3StoreNotReady
from vllm_ascend.dsa_offload.l3.metadata import (
    DSAL3ConnectorMetadata,
    DSAL3JobResult,
    DSAL3LoadJob,
    DSAL3StoreJob,
    DSAL3WorkerMetadata,
)
from vllm_ascend.dsa_offload.worker_dram_store import DSAWorkerDRAMStore

EventFactory = Callable[[], Any]


class DSAL3StoreWorker:
    """所有 TP rank 贡献 segment/load，仅 TP0 执行副本 put。"""

    def __init__(
        self,
        *,
        tp_rank: int,
        device: torch.device,
        backend: DSAL3BackendAdapter,
        max_inflight_jobs: int,
        enable_load: bool = False,
        direct_hbm_load: bool = False,
        event_factory: EventFactory | None = None,
        load_consensus: Callable[[bool], bool] | None = None,
    ) -> None:
        self.tp_rank = int(tp_rank)
        self.device = torch.device(device)
        self.backend = backend
        self.enable_load = bool(enable_load)
        self.direct_hbm_load = bool(direct_hbm_load)
        if self.direct_hbm_load and not self.enable_load:
            raise ValueError("DSA direct-HBM requires load to be enabled")
        self.max_inflight_jobs = int(max_inflight_jobs)
        if self.max_inflight_jobs <= 0:
            raise ValueError("DSA L3 max_inflight_jobs must be positive")
        self.store: DSAWorkerDRAMStore | None = None
        self._event_factory = event_factory
        self._load_consensus = load_consensus
        self._load_status: torch.Tensor | None = None
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
        self._load_jobs = 0
        self._load_blocks = 0
        self._load_bytes = 0

    @property
    def is_store_rank(self) -> bool:
        return self.tp_rank == 0

    def bind_store(self, store: DSAWorkerDRAMStore) -> None:
        if self.store is not None:
            raise RuntimeError("DSA L3 worker DRAM store was bound twice")
        self.store = store
        if not self.is_store_rank and not self.enable_load:
            return
        self.backend.bind_store(
            store,
            register_hbm=self.direct_hbm_load,
        )
        if self.enable_load and self.device.type == "npu":
            self._load_status = torch.ones(
                1,
                dtype=torch.int32,
                device=self.device,
            )
        if not self.is_store_rank:
            return
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
                retryable=isinstance(exc, DSAL3StoreNotReady),
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

    def load_sync(
        self,
        load_jobs: tuple[DSAL3LoadJob, ...],
        *,
        prior_error: Exception | None = None,
    ) -> None:
        """在 DSA assignment/restore 之前同步恢复本 rank L2。"""

        if not load_jobs:
            return
        if not self.enable_load:
            raise RuntimeError("DSA L3 worker received load jobs while disabled")
        if self.store is None:
            raise RuntimeError("DSA L3 load worker has no bound DRAM store")
        if self.device.type == "npu":
            torch.npu.set_device(self.device)
        completed: list[tuple[DSAL3LoadJob, int]] = []
        local_error = prior_error
        if local_error is None:
            try:
                for job in load_jobs:
                    if bool(job.direct_hbm) != self.direct_hbm_load:
                        raise RuntimeError(
                            "DSA L3 load destination capability disagrees "
                            f"with worker config: job={job.direct_hbm}, "
                            f"worker={self.direct_hbm_load}"
                        )
                    transferred = self.backend.get_load_job(
                        job=job,
                        store=self.store,
                    )
                    completed.append((job, transferred))
            except Exception as exc:
                local_error = exc

        if not self._all_ranks_loaded(local_error is None):
            if local_error is not None:
                raise RuntimeError(
                    "DSA L3 load failed before forward: "
                    f"tp_rank={self.tp_rank}, error={local_error}"
                ) from local_error
            raise RuntimeError(
                "DSA L3 load failed on a peer TP rank before forward"
            )
        if local_error is not None:
            # Single-rank/CPU tests use the local result as their consensus.
            raise RuntimeError(
                "DSA L3 load failed before forward: "
                f"tp_rank={self.tp_rank}, error={local_error}"
            ) from local_error

        for job, transferred in completed:
            self._load_jobs += 1
            self._load_blocks += len(job.blocks)
            self._load_bytes += transferred
            logger.info(
                "DSA L3 load completed: tp_rank=%d request=%s "
                "blocks=%d bytes=%d destination=%s",
                self.tp_rank,
                job.request_id,
                len(job.blocks),
                transferred,
                "hbm" if job.direct_hbm else "l2",
            )

    def _all_ranks_loaded(self, local_success: bool) -> bool:
        if self._load_consensus is not None:
            return bool(self._load_consensus(local_success))
        if self.device.type != "npu":
            return local_success
        status = self._load_status
        if status is None:
            raise RuntimeError("DSA L3 load consensus buffer is not initialized")
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        if tp_group.world_size == 1:
            return local_success
        status.fill_(1 if local_success else 0)
        dist.all_reduce(
            status,
            op=dist.ReduceOp.SUM,
            group=tp_group.device_group,
        )
        return int(status.item()) == int(tp_group.world_size)

    def prepare_step(self, metadata: DSAL3ConnectorMetadata) -> None:
        """执行所有可能覆盖 L2 的 pre-update fence 与同步 load。"""

        flush_error: Exception | None = None
        try:
            self.flush(metadata.jobs_to_flush)
        except Exception as exc:
            flush_error = exc
        if metadata.load_jobs:
            # TP0 的 reuse fence 失败也必须参加 load success consensus，
            # 否则其他 rank 会在 all-reduce 中等待一个已经抛错的同伴。
            self.load_sync(
                metadata.load_jobs,
                prior_error=flush_error,
            )
        elif flush_error is not None:
            raise flush_error

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
                "transferred_bytes=%d pending_results=%d load_jobs=%d "
                "load_blocks=%d load_bytes=%d",
                self.tp_rank,
                self._submitted_jobs,
                self._submitted_blocks,
                self._finished_jobs,
                self._failed_jobs,
                self._transferred_bytes,
                len(self._completed),
                self._load_jobs,
                self._load_blocks,
                self._load_bytes,
            )
            try:
                self.backend.close()
            finally:
                if self._executor is not None:
                    self._executor.shutdown(
                        wait=True,
                        cancel_futures=False,
                    )
        logger.info("DSA L3 worker resource shutdown completed: tp_rank=%d", self.tp_rank)
