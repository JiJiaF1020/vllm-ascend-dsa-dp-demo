# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 对 AscendStore backend 的严格结果与 buffer 合同。"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

from vllm.logger import logger

from vllm_ascend.dsa_offload.l3.metadata import DSAL3StoreJob
from vllm_ascend.dsa_offload.l3.schema import (
    DSAL3Component,
    DSAL3KeySchema,
)
from vllm_ascend.dsa_offload.worker_dram_store import DSAWorkerDRAMStore


def configure_dsa_mooncake_global_ub(max_active_channels: int) -> None:
    """在 TransferEngine 初始化前固化 DSA host+device endpoint 合同。"""

    max_active_channels = int(max_active_channels)
    if max_active_channels <= 0:
        raise ValueError("DSA L3 max_active_channels must be positive")
    # 与 vLLM-Ascend AscendStore 示例一致，同时保留显式用户覆盖。
    os.environ.setdefault("ASCEND_CONNECT_TIMEOUT", "30000")
    os.environ.setdefault("ASCEND_TRANSFER_TIMEOUT", "30000")
    required = {"ub_ctp:device", "ub_ctp:host"}
    raw = os.getenv("ASCEND_GLOBAL_RESOURCE_CONFIG")
    if raw:
        try:
            config = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("ASCEND_GLOBAL_RESOURCE_CONFIG is not valid JSON") from exc
        protocols = config.get("comm_resource_config.protocol_desc", ())
        if isinstance(protocols, str):
            protocols = (protocols,)
        missing = required - set(protocols)
        if missing:
            raise ValueError(
                "DSA swapped LocalDRAM requires Mooncake global_ub with "
                "both host and device endpoints; missing="
                f"{tuple(sorted(missing))}"
            )
        return
    os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] = json.dumps(
        {
            "comm_resource_config.protocol_desc": sorted(required),
            "comm_resource_config.max_active_channels": max_active_channels,
        },
        separators=(",", ":"),
    )


class DSAL3BackendAdapter:
    """把宽松 Backend API 收敛为 DSA 可判定成败的合同。"""

    def __init__(
        self,
        *,
        backend: Any,
        schema: DSAL3KeySchema,
        max_descriptors_per_put: int = 128,
    ) -> None:
        self.backend = backend
        self.schema = schema
        self.max_descriptors_per_put = int(max_descriptors_per_put)
        if self.max_descriptors_per_put <= 0:
            raise ValueError("DSA L3 max_descriptors_per_put must be positive")
        self._registered = False
        self._registered_regions: tuple[tuple[int, int], ...] = ()

    @classmethod
    def create_mooncake(
        cls,
        *,
        vllm_config: Any,
        schema: DSAL3KeySchema,
        max_active_channels: int,
        max_descriptors_per_put: int,
    ) -> DSAL3BackendAdapter:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
            MooncakeBackend,
            MooncakeStoreConfig,
        )

        config = MooncakeStoreConfig.load_from_env()
        if config.protocol == "ascend":
            configure_dsa_mooncake_global_ub(max_active_channels)
        backend = MooncakeBackend(
            vllm_config.parallel_config,
            config=config,
        )
        logger.info(
            "DSA L3 Mooncake backend created: protocol=%s store_owned_engine=%s",
            config.protocol,
            config.protocol == "tcp",
        )
        return cls(
            backend=backend,
            schema=schema,
            max_descriptors_per_put=max_descriptors_per_put,
        )

    def bind_store(self, store: DSAWorkerDRAMStore) -> None:
        if self._registered:
            raise RuntimeError("DSA L3 DRAM store was bound twice")
        indexer_bytes = store.get_l3_component_bytes_per_block(DSAL3Component.INDEXER)
        mla_bytes = store.get_l3_component_bytes_per_block(DSAL3Component.MLA)
        indexer_planes = store.get_l3_component_plane_count(DSAL3Component.INDEXER)
        mla_planes = store.get_l3_component_plane_count(DSAL3Component.MLA)
        if (
            indexer_bytes != self.schema.indexer_bytes_per_block
            or mla_bytes != self.schema.mla_bytes_per_block
            or indexer_planes != self.schema.indexer_plane_count
            or mla_planes != self.schema.mla_plane_count
        ):
            raise RuntimeError(
                "DSA L3 worker payload disagrees with scheduler schema: "
                "worker=("
                f"{indexer_planes},{mla_planes},"
                f"{indexer_bytes},{mla_bytes}), "
                "schema=("
                f"{self.schema.indexer_plane_count},"
                f"{self.schema.mla_plane_count},"
                f"{self.schema.indexer_bytes_per_block},"
                f"{self.schema.mla_bytes_per_block})"
            )
        regions = store.get_l3_registration_regions()
        self.backend.register_buffer(
            [pointer for pointer, _ in regions],
            [size for _, size in regions],
        )
        self._registered_regions = regions
        self._registered = True
        logger.info(
            "DSA L3 LocalDRAM registered: regions=%d bytes=%d indexer_planes=%d mla_planes=%d",
            len(regions),
            sum(size for _, size in regions),
            indexer_planes,
            mla_planes,
        )

    @staticmethod
    def _validate_result(
        result: Any,
        *,
        expected_count: int,
        operation: str,
    ) -> list[int]:
        if result is None or isinstance(result, (str, bytes)):
            raise RuntimeError(f"DSA L3 {operation} returned no per-key result")
        if not isinstance(result, Sequence):
            result = list(result)
        values = [int(value) for value in result]
        if len(values) != expected_count:
            raise RuntimeError(
                f"DSA L3 {operation} result count mismatch: actual={len(values)}, expected={expected_count}"
            )
        failed = [value for value in values if value < 0]
        if failed:
            raise RuntimeError(f"DSA L3 {operation} failed with codes={tuple(failed)}")
        return values

    def put_store_job(
        self,
        *,
        job: DSAL3StoreJob,
        store: DSAWorkerDRAMStore,
    ) -> int:
        if not self._registered:
            raise RuntimeError("DSA L3 backend is not bound to its DRAM store")
        block_ids = job.source_l2_block_ids
        indexer_addrs, indexer_sizes = store.build_l3_component_payload(
            component=DSAL3Component.INDEXER,
            physical_block_ids=block_ids,
        )
        mla_addrs, mla_sizes = store.build_l3_component_payload(
            component=DSAL3Component.MLA,
            physical_block_ids=block_ids,
        )
        keys: list[str] = []
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        for index, block in enumerate(job.blocks):
            keys.extend((block.indexer_key, block.mla_key))
            addresses.extend((indexer_addrs[index], mla_addrs[index]))
            sizes.extend((indexer_sizes[index], mla_sizes[index]))
        exists_result = self._validate_result(
            self.backend.exists(keys),
            expected_count=len(keys),
            operation="exists",
        )
        missing_indices = [index for index, exists in enumerate(exists_result) if exists == 0]
        if not missing_indices:
            logger.info(
                "DSA L3 store backend hit all keys: job=%d keys=%d",
                job.job_id,
                len(keys),
            )
            return 0
        missing_keys = [keys[index] for index in missing_indices]
        missing_addresses = [addresses[index] for index in missing_indices]
        missing_sizes = [sizes[index] for index in missing_indices]
        descriptor_counts = [len(parts) for parts in missing_addresses]
        batches: list[tuple[int, int]] = []
        batch_start = 0
        batch_descriptors = 0
        for index, descriptor_count in enumerate(descriptor_counts):
            if batch_descriptors and (batch_descriptors + descriptor_count > self.max_descriptors_per_put):
                batches.append((batch_start, index))
                batch_start = index
                batch_descriptors = 0
            if descriptor_count > self.max_descriptors_per_put:
                logger.warning_once(
                    "DSA L3 component exceeds the soft descriptor budget; "
                    "sending it as one Mooncake key and relying on the "
                    "Mooncake transport's internal chunking: "
                    "descriptors=%d budget=%d",
                    descriptor_count,
                    self.max_descriptors_per_put,
                )
                batches.append((index, index + 1))
                batch_start = index + 1
                batch_descriptors = 0
                continue
            batch_descriptors += descriptor_count
        if batch_start < len(missing_keys):
            batches.append((batch_start, len(missing_keys)))
        if len(batches) > 1:
            logger.info(
                "DSA L3 put split by descriptor budget: keys=%d descriptors=%d batches=%d budget=%d",
                len(missing_keys),
                sum(descriptor_counts),
                len(batches),
                self.max_descriptors_per_put,
            )
        logger.info(
            "DSA L3 store backend put: job=%d keys=%d missing=%d descriptors=%d batches=%d bytes=%d",
            job.job_id,
            len(keys),
            len(missing_keys),
            sum(descriptor_counts),
            len(batches),
            sum(sum(parts) for parts in missing_sizes),
        )
        for start, end in batches:
            result = self.backend.put(
                missing_keys[start:end],
                missing_addresses[start:end],
                missing_sizes[start:end],
            )
            self._validate_result(
                result,
                expected_count=end - start,
                operation="put",
            )
        transferred_bytes = sum(sum(parts) for parts in missing_sizes)
        logger.info(
            "DSA L3 store backend put finished: job=%d bytes=%d",
            job.job_id,
            transferred_bytes,
        )
        return transferred_bytes

    def close(self) -> None:
        try:
            if self._registered_regions:
                unregister = getattr(self.backend, "unregister_buffer", None)
                if callable(unregister):
                    unregister([pointer for pointer, _ in self._registered_regions])
                self._registered_regions = ()
                self._registered = False
        finally:
            close = getattr(self.backend, "close", None)
            if callable(close):
                close()
            else:
                store = getattr(self.backend, "store", None)
                if store is not None:
                    store.close()
