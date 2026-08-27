# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA L3 scheduler 侧 write-behind job 与 L2 reuse fence。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from vllm.logger import logger
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.dsa_offload.l3.metadata import (
    DSAL3ConnectorMetadata,
    DSAL3StoreBlock,
    DSAL3StoreJob,
    DSAL3WorkerMetadata,
)
from vllm_ascend.dsa_offload.l3.schema import DSAL3KeySchema


class DSAL3StoreScheduler:
    """scheduler 主线程持有的远端发布与 source-lifetime 状态。"""

    def __init__(
        self,
        schema: DSAL3KeySchema,
        max_inflight_jobs: int,
    ) -> None:
        self.schema = schema
        self.max_inflight_jobs = int(max_inflight_jobs)
        if self.max_inflight_jobs <= 0:
            raise ValueError("DSA L3 max_inflight_jobs must be positive")
        self._next_job_id = 0
        self._jobs: dict[int, DSAL3StoreJob] = {}
        self._pending_key_jobs: dict[str, int] = {}
        self._remote_ready_keys: set[str] = set()
        self._l2_block_jobs: dict[int, set[int]] = defaultdict(set)
        self._request_scanned_counts: dict[str, int] = {}
        self._logged_first_scan = False
        self._logged_first_durable_scan = False
        self._planner_calls = 0
        self._planned_jobs = 0
        self._planned_blocks = 0
        self._completed_jobs = 0
        self._failed_jobs = 0
        self._completed_bytes = 0

    def build_connector_meta(self) -> DSAL3ConnectorMetadata:
        return DSAL3ConnectorMetadata()

    def _new_job_id(self) -> int:
        job_id = self._next_job_id
        self._next_job_id += 1
        return job_id

    def extend_store_jobs(
        self,
        *,
        metadata: DSAL3ConnectorMetadata,
        projection: Any,
        coordinator: Any,
        requests: dict[str, Any],
    ) -> None:
        """在 DSA projection 完成后补充同轮 durable block store job。"""

        self._planner_calls += 1

        if projection.request_ids and not self._logged_first_scan:
            self._logged_first_scan = True
            logger.info(
                "DSA L3 store planner active: rows=%d ready_blocks=%s",
                len(projection.request_ids),
                tuple(projection.dram_ready_block_counts),
            )

        # assignment 对应的 physical ID 可能仍被旧 job 读取。该 job 必须
        # 在 worker 覆盖 assignment 前完成；当前轮新 job 在下面才登记，
        # 不会被误加入自己的 flush 集合。
        for assignment in projection.dram_block_assignments:
            for block_id in assignment.physical_block_ids:
                metadata.jobs_to_flush.update(self._l2_block_jobs.get(int(block_id), ()))

        if len(self._jobs) >= self.max_inflight_jobs:
            logger.debug(
                "DSA L3 store backpressure: inflight=%d limit=%d",
                len(self._jobs),
                self.max_inflight_jobs,
            )
            return

        planned_hashes: set[bytes] = set()
        blocks_to_store: list[DSAL3StoreBlock] = []
        for request_id, ready_count in zip(
            projection.request_ids,
            projection.dram_ready_block_counts,
            strict=True,
        ):
            request = requests.get(request_id)
            if request is None:
                raise RuntimeError(f"DSA L3 cannot plan a vanished request: request_id={request_id!r}")
            physical_ids = coordinator.dram_directory.get_request_blocks(request_id)
            ready_count = int(ready_count)
            if ready_count > 0 and not self._logged_first_durable_scan:
                self._logged_first_durable_scan = True
                logger.info(
                    "DSA L3 durable source visible: request=%s ready=%d l2_blocks=%d hashes=%d",
                    request_id,
                    ready_count,
                    len(physical_ids),
                    len(request.block_hashes),
                )
            scan_start = self._request_scanned_counts.get(request_id, 0)
            if ready_count < scan_start:
                raise RuntimeError(
                    "DSA L3 request durable watermark moved backwards: "
                    f"request_id={request_id!r}, scanned={scan_start}, "
                    f"ready={ready_count}"
                )
            if ready_count > len(physical_ids) or ready_count > len(request.block_hashes):
                raise RuntimeError(
                    "DSA L3 durable watermark exceeds its source/hash table: "
                    f"request_id={request_id!r}, ready={ready_count}, "
                    f"l2_blocks={len(physical_ids)}, "
                    f"hashes={len(request.block_hashes)}"
                )
            next_scan = scan_start
            for logical_index in range(scan_start, ready_count):
                block_hash = bytes(request.block_hashes[logical_index])
                if block_hash in planned_hashes:
                    break
                indexer_key, mla_key = self.schema.block_keys(block_hash)
                keys = (indexer_key, mla_key)
                if all(key in self._remote_ready_keys for key in keys):
                    next_scan = logical_index + 1
                    continue
                if any(key in self._pending_key_jobs for key in keys):
                    break
                planned_hashes.add(block_hash)
                blocks_to_store.append(
                    DSAL3StoreBlock(
                        request_id=request_id,
                        logical_block_index=logical_index,
                        source_l2_block_id=int(physical_ids[logical_index]),
                        block_hash=block_hash,
                        indexer_key=indexer_key,
                        mla_key=mla_key,
                    )
                )
                next_scan = logical_index + 1
            self._request_scanned_counts[request_id] = next_scan

        if not blocks_to_store:
            return
        job_id = self._new_job_id()
        job = DSAL3StoreJob(job_id=job_id, blocks=tuple(blocks_to_store))
        self._jobs[job_id] = job
        self._planned_jobs += 1
        self._planned_blocks += len(job.blocks)
        metadata.store_jobs[job_id] = job
        for block in job.blocks:
            self._pending_key_jobs[block.indexer_key] = job_id
            self._pending_key_jobs[block.mla_key] = job_id
            self._l2_block_jobs[block.source_l2_block_id].add(job_id)
        logger.info(
            "DSA L3 store planned: job=%d blocks=%d requests=%d bytes=%d",
            job_id,
            len(job.blocks),
            len({block.request_id for block in job.blocks}),
            len(job.blocks) * self.schema.bytes_per_logical_block,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        worker_meta = connector_output.kv_connector_worker_meta
        if worker_meta is None:
            return
        if not isinstance(worker_meta, DSAL3WorkerMetadata):
            raise TypeError(f"DSA L3 received unexpected worker metadata: {type(worker_meta).__name__}")
        for job_id, result in worker_meta.results.items():
            job = self._jobs.pop(job_id, None)
            if job is None:
                logger.debug("Ignoring stale DSA L3 result for job %d", job_id)
                continue
            for block in job.blocks:
                self._pending_key_jobs.pop(block.indexer_key, None)
                self._pending_key_jobs.pop(block.mla_key, None)
                pending = self._l2_block_jobs.get(block.source_l2_block_id)
                if pending is not None:
                    pending.discard(job_id)
                    if not pending:
                        self._l2_block_jobs.pop(
                            block.source_l2_block_id,
                            None,
                        )
                if result.success:
                    self._remote_ready_keys.update((block.indexer_key, block.mla_key))
            self._completed_jobs += 1
            self._completed_bytes += int(result.transferred_bytes)
            if not result.success:
                self._failed_jobs += 1
                for block in job.blocks:
                    scanned = self._request_scanned_counts.get(block.request_id)
                    if scanned is not None:
                        self._request_scanned_counts[block.request_id] = min(
                            scanned,
                            block.logical_block_index,
                        )
                logger.error(
                    "DSA L3 store job %d failed: %s",
                    job_id,
                    result.error,
                )

    def request_finished(self, request_id: str) -> None:
        self._request_scanned_counts.pop(request_id, None)

    def has_pending_jobs(self) -> bool:
        """让 EngineCore 在异步 store 完成前继续推进 connector。"""

        return bool(self._jobs)

    def shutdown(self) -> None:
        logger.info(
            "DSA L3 scheduler summary: planner_calls=%d planned_jobs=%d "
            "planned_blocks=%d completed_jobs=%d failed_jobs=%d "
            "completed_bytes=%d inflight_jobs=%d remote_ready_keys=%d",
            self._planner_calls,
            self._planned_jobs,
            self._planned_blocks,
            self._completed_jobs,
            self._failed_jobs,
            self._completed_bytes,
            len(self._jobs),
            len(self._remote_ready_keys),
        )
        if self._jobs:
            raise RuntimeError(
                f"DSA L3 scheduler shutdown with unacknowledged store jobs: job_ids={tuple(sorted(self._jobs))}"
            )
        self._jobs.clear()
        self._pending_key_jobs.clear()
        self._l2_block_jobs.clear()
        self._request_scanned_counts.clear()
