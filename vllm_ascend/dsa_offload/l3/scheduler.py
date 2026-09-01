# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA L3 scheduler 侧 prefix lookup、load plan 与 write-behind。"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import replace
from typing import Any

from vllm.logger import logger
from vllm.v1.outputs import KVConnectorOutput

from vllm_ascend.dsa_offload.l3.backend_adapter import (
    DSAL3BackendAdapter,
)
from vllm_ascend.dsa_offload.l3.metadata import (
    DSAL3ConnectorMetadata,
    DSAL3LoadBlock,
    DSAL3LoadJob,
    DSAL3StoreBlock,
    DSAL3StoreJob,
    DSAL3WorkerMetadata,
)
from vllm_ascend.dsa_offload.l3.schema import DSAL3KeySchema


class DSAL3StoreScheduler:
    """scheduler 主线程持有的远端查询、发布与 source-lifetime 状态。"""

    def __init__(
        self,
        schema: DSAL3KeySchema,
        max_inflight_jobs: int,
        *,
        lookup_backend: DSAL3BackendAdapter | None = None,
        enable_load: bool = False,
        direct_hbm_load: bool = False,
    ) -> None:
        self.schema = schema
        self._base_schema = schema
        self._namespace_generation = 0
        self.max_inflight_jobs = int(max_inflight_jobs)
        if self.max_inflight_jobs <= 0:
            raise ValueError("DSA L3 max_inflight_jobs must be positive")
        self.enable_load = bool(enable_load)
        self.direct_hbm_load = bool(direct_hbm_load)
        if self.direct_hbm_load and not self.enable_load:
            raise ValueError("DSA direct-HBM requires L3 load to be enabled")
        self.lookup_backend = lookup_backend
        if self.enable_load and self.lookup_backend is None:
            raise ValueError("DSA L3 load requires a scheduler lookup backend")
        self.coordinator: Any | None = None
        self._pending_load_jobs: list[DSAL3LoadJob] = []
        self._next_job_id = 0
        self._jobs: dict[int, DSAL3StoreJob] = {}
        self._pending_key_jobs: dict[str, int] = {}
        self._remote_ready_keys: set[str] = set()
        self._remote_ready_key_limit: int | None = None
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
        self._lookup_requests = 0
        self._lookup_query_blocks = 0
        self._lookup_hit_blocks = 0
        self._planned_load_jobs = 0
        self._planned_load_blocks = 0

    def build_connector_meta(self) -> DSAL3ConnectorMetadata:
        load_jobs = tuple(self._pending_load_jobs)
        self._pending_load_jobs.clear()
        return DSAL3ConnectorMetadata(load_jobs=load_jobs)

    def bind_coordinator(self, coordinator: Any) -> None:
        if self.coordinator is not None:
            raise RuntimeError("DSA L3 scheduler coordinator was bound twice")
        self.coordinator = coordinator
        configure_direct = getattr(
            coordinator,
            "set_l3_direct_hbm_load_enabled",
            None,
        )
        if not callable(configure_direct) and self.direct_hbm_load:
            raise RuntimeError(
                "DSA coordinator does not expose direct-HBM capability"
            )
        if callable(configure_direct):
            configure_direct(self.direct_hbm_load)
        usable_blocks = int(
            getattr(
                getattr(coordinator, "dram_directory", None),
                "usable_blocks",
                0,
            )
        )
        if usable_blocks > 0:
            # One logical block owns one key per active durable component. Bound
            # the hint cache to the local L2 capacity so a long-running
            # service cannot retain every key ever evicted by Mooncake.
            self._remote_ready_key_limit = usable_blocks * len(
                self.schema.active_components
            )

    def _remember_remote_ready_keys(self, keys: tuple[str, ...]) -> None:
        limit = self._remote_ready_key_limit
        if limit is not None:
            new_key_count = sum(
                key not in self._remote_ready_keys for key in keys
            )
            if len(self._remote_ready_keys) + new_key_count > limit:
                logger.info(
                    "DSA L3 remote-ready hint cache reached its bound; "
                    "dropping %d stale-capable keys before publishing %d",
                    len(self._remote_ready_keys),
                    len(keys),
                )
                self._remote_ready_keys.clear()
        self._remote_ready_keys.update(keys)

    def lookup_external_prefix(
        self,
        request: Any,
        local_computed_tokens: int,
    ) -> int:
        """查询最长本地命中之后的成对 L3 component suffix。

        ``local_computed_tokens`` 已由 coordinator 决议为
        ``max(HBM pair, complete L2)``。因此 L3 永远不会读取被 L1/L2
        覆盖的重叠区，只查询二者都未覆盖的连续后缀。
        """

        if not self.enable_load or bool(
            getattr(request, "skip_reading_prefix_cache", False)
        ):
            return 0
        if self.coordinator is None or self.lookup_backend is None:
            raise RuntimeError("DSA L3 lookup scheduler is not initialized")
        block_size = int(self.schema.block_size)
        local_computed_tokens = int(local_computed_tokens)
        if local_computed_tokens % block_size:
            raise RuntimeError("DSA L3 lookup local prefix is not block aligned")
        local_blocks = local_computed_tokens // block_size
        max_blocks = min(
            (int(request.num_tokens) - 1) // block_size,
            len(request.block_hashes),
        )
        if local_blocks >= max_blocks:
            return 0
        candidate_hashes = tuple(
            bytes(block_hash)
            for block_hash in request.block_hashes[
                local_blocks:max_blocks
            ]
        )
        self._lookup_requests += 1
        self._lookup_query_blocks += len(candidate_hashes)
        hit_blocks = self.lookup_backend.lookup_prefix_blocks(
            candidate_hashes
        )
        self._lookup_hit_blocks += hit_blocks
        external_tokens = self.coordinator.prepare_external_prefix_cache_hit(
            request,
            local_computed_tokens=local_computed_tokens,
            external_block_count=hit_blocks,
        )
        logger.info(
            "DSA L3 prefix lookup: request=%s local_blocks=%d "
            "queried_blocks=%d hit_blocks=%d effective_blocks=%d "
            "external_tokens=%d",
            request.request_id,
            local_blocks,
            len(candidate_hashes),
            hit_blocks,
            external_tokens // block_size,
            external_tokens,
        )
        return external_tokens

    def stage_external_load(
        self,
        request_id: str,
        num_external_tokens: int,
    ) -> None:
        if not num_external_tokens:
            return
        if not self.enable_load or self.coordinator is None:
            raise RuntimeError("DSA L3 external load is not enabled")
        plan = self.coordinator.take_external_prefix_load_plan(request_id)
        if plan is None:
            raise RuntimeError(
                "DSA L3 allocation did not publish its external load plan: "
                f"request_id={request_id!r}"
            )
        if (
            len(plan.block_hashes) * self.schema.block_size
            != int(num_external_tokens)
        ):
            raise RuntimeError(
                "DSA L3 external load token count changed after allocation"
            )
        blocks: list[DSAL3LoadBlock] = []
        mtp_destinations: tuple[int | None, ...]
        plan_mtp_destinations = tuple(
            getattr(plan, "destination_mtp_hbm_block_ids", ())
        )
        if plan_mtp_destinations:
            mtp_destinations = plan_mtp_destinations
        else:
            mtp_destinations = (None,) * len(plan.block_hashes)
        for offset, (
            block_hash,
            destination_l2,
            destination_indexer,
            destination_resident,
            destination_mtp,
        ) in enumerate(
            zip(
                plan.block_hashes,
                plan.destination_dram_block_ids,
                plan.destination_indexer_hbm_block_ids,
                plan.destination_resident_hbm_block_ids,
                mtp_destinations,
                strict=True,
            )
        ):
            component_keys = self.schema.block_keys(block_hash)
            if len(component_keys) != len(self.schema.active_components):
                raise RuntimeError("DSA L3 schema emitted an invalid key set")
            blocks.append(
                DSAL3LoadBlock(
                    logical_block_index=(
                        plan.logical_block_start + offset
                    ),
                    destination_l2_block_id=destination_l2,
                    destination_indexer_hbm_block_id=(
                        destination_indexer
                    ),
                    destination_resident_hbm_block_id=(
                        destination_resident
                    ),
                    block_hash=block_hash,
                    indexer_key=component_keys[0],
                    mla_key=component_keys[1],
                    destination_mtp_hbm_block_id=destination_mtp,
                    mtp_key=(
                        component_keys[2]
                        if self.schema.has_mtp
                        else None
                    ),
                )
            )
        self._pending_load_jobs.append(
            DSAL3LoadJob(
                request_id=request_id,
                blocks=tuple(blocks),
                direct_hbm=self.direct_hbm_load,
            )
        )
        self._planned_load_jobs += 1
        self._planned_load_blocks += len(blocks)
        logger.info(
            "DSA L3 load planned: request=%s logical_start=%d "
            "blocks=%d bytes_per_block=%d destination=%s",
            request_id,
            plan.logical_block_start,
            len(blocks),
            self.schema.bytes_per_logical_block,
            "hbm" if self.direct_hbm_load else "l2",
        )

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
                keys = self.schema.block_keys(block_hash)
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
                        indexer_key=keys[0],
                        mla_key=keys[1],
                        mtp_key=keys[2] if self.schema.has_mtp else None,
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
            for key in block.component_keys:
                self._pending_key_jobs[key] = job_id
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
            ready_keys: list[str] = []
            for block in job.blocks:
                for key in block.component_keys:
                    self._pending_key_jobs.pop(key, None)
                pending = self._l2_block_jobs.get(block.source_l2_block_id)
                if pending is not None:
                    pending.discard(job_id)
                    if not pending:
                        self._l2_block_jobs.pop(
                            block.source_l2_block_id,
                            None,
                        )
                if result.success:
                    ready_keys.extend(block.component_keys)
            if ready_keys:
                self._remember_remote_ready_keys(tuple(ready_keys))
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

    def reset_cache(self) -> bool:
        """Logically invalidate remote objects by rotating the key namespace."""

        if self._jobs or self._pending_load_jobs:
            logger.warning(
                "DSA L3 namespace reset rejected with pending transfers: "
                "store_jobs=%d load_jobs=%d",
                len(self._jobs),
                len(self._pending_load_jobs),
            )
            return False
        if self._pending_key_jobs or self._l2_block_jobs:
            raise RuntimeError(
                "DSA L3 reset found orphaned pending store indexes"
            )
        self._namespace_generation += 1
        fingerprint = hashlib.sha256(
            (
                f"{self._base_schema.fingerprint}:"
                f"{self._namespace_generation}"
            ).encode()
        ).hexdigest()[:32]
        self.schema = replace(
            self._base_schema,
            fingerprint=fingerprint,
        )
        if self.lookup_backend is not None:
            self.lookup_backend.schema = self.schema
        self._remote_ready_keys.clear()
        self._request_scanned_counts.clear()
        logger.info(
            "DSA L3 namespace reset: generation=%d fingerprint=%s",
            self._namespace_generation,
            fingerprint,
        )
        return True

    def has_pending_jobs(self) -> bool:
        """让 EngineCore 在异步 store 完成前继续推进 connector。"""

        return bool(self._jobs)

    def shutdown(self) -> None:
        logger.info(
            "DSA L3 scheduler summary: planner_calls=%d planned_jobs=%d "
            "planned_blocks=%d completed_jobs=%d failed_jobs=%d "
            "completed_bytes=%d inflight_jobs=%d remote_ready_keys=%d "
            "lookup_requests=%d lookup_query_blocks=%d "
            "lookup_hit_blocks=%d planned_load_jobs=%d "
            "planned_load_blocks=%d",
            self._planner_calls,
            self._planned_jobs,
            self._planned_blocks,
            self._completed_jobs,
            self._failed_jobs,
            self._completed_bytes,
            len(self._jobs),
            len(self._remote_ready_keys),
            self._lookup_requests,
            self._lookup_query_blocks,
            self._lookup_hit_blocks,
            self._planned_load_jobs,
            self._planned_load_blocks,
        )
        pending_job_ids = tuple(sorted(self._jobs))
        self._jobs.clear()
        self._pending_key_jobs.clear()
        self._l2_block_jobs.clear()
        self._request_scanned_counts.clear()
        self._pending_load_jobs.clear()
        try:
            if self.lookup_backend is not None:
                self.lookup_backend.close()
        finally:
            if pending_job_ids:
                raise RuntimeError(
                    "DSA L3 scheduler shutdown with unacknowledged store "
                    f"jobs: job_ids={pending_job_ids}"
                )
