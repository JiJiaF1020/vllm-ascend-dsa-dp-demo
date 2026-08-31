# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 对 AscendStore backend 的严格结果与 buffer 合同。"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Sequence
from typing import Any

from vllm.logger import logger

from vllm_ascend.dsa_offload.l3.metadata import (
    DSAL3LoadJob,
    DSAL3StoreJob,
)
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


def configure_dsa_mooncake_tcp() -> None:
    """Require connection reuse before Mooncake constructs TCP transport."""

    name = "MC_TCP_ENABLE_CONNECTION_POOL"
    raw = os.getenv(name)
    if raw is None:
        os.environ[name] = "1"
        return
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return
    if value in {"0", "false", "no", "off"}:
        raise ValueError(
            "DSA L3 Store-owned TCP requires MC_TCP_ENABLE_CONNECTION_POOL=1; "
            "without connection reuse, multi-plane TP loadback exhausts the "
            "host ephemeral port range"
        )
    raise ValueError(
        f"{name} must be a boolean value, got {raw!r}"
    )


def validate_dsa_mooncake_direct_hbm(
    *,
    protocol: str,
    direct_hbm_load: bool,
) -> None:
    """冻结尚未满足 A5 L1/L2/L3 完整合同的 direct-HBM。"""

    if direct_hbm_load:
        raise RuntimeError(
            "DSA L3 direct-HBM is frozen in the current A5 multi-tier "
            "contract. Store-owned TCP cannot write NPU pointers, while "
            "the Ascend transport cannot yet provide the required "
            "topology-transparent LocalDRAM path. Keep "
            "direct_hbm_load=false and use L3->L2->HBM. "
            f"Configured Mooncake protocol={protocol!r}."
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
        self._io_lock = threading.Lock()

    @classmethod
    def create_mooncake(
        cls,
        *,
        vllm_config: Any,
        schema: DSAL3KeySchema,
        max_active_channels: int,
        max_descriptors_per_put: int,
        contribute_memory: bool = True,
        direct_hbm_load: bool = False,
    ) -> DSAL3BackendAdapter:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
            MooncakeBackend,
            MooncakeStoreConfig,
        )

        config = MooncakeStoreConfig.load_from_env()
        validate_dsa_mooncake_direct_hbm(
            protocol=config.protocol,
            direct_hbm_load=direct_hbm_load,
        )
        if config.protocol == "ascend":
            configure_dsa_mooncake_global_ub(max_active_channels)
        elif config.protocol == "tcp":
            configure_dsa_mooncake_tcp()
        backend = (
            MooncakeBackend(
                vllm_config.parallel_config,
                contribute_memory=True,
                config=config,
            )
            if contribute_memory
            else MooncakeBackend.create_scheduler_client(
                vllm_config.parallel_config
            )
        )
        logger.info(
            "DSA L3 Mooncake backend created: protocol=%s "
            "store_owned_engine=%s contribute_memory=%s",
            config.protocol,
            config.protocol == "tcp",
            contribute_memory,
        )
        return cls(
            backend=backend,
            schema=schema,
            max_descriptors_per_put=max_descriptors_per_put,
        )

    def bind_store(
        self,
        store: DSAWorkerDRAMStore,
        *,
        register_hbm: bool = False,
    ) -> None:
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
        regions = list(store.get_l3_registration_regions())
        if register_hbm:
            for component in DSAL3Component:
                hbm_addresses, hbm_sizes = (
                    store.build_l3_hbm_component_payload(
                        component=component,
                        physical_block_ids=(1,),
                    )
                )
                hbm_bytes = sum(hbm_sizes[0])
                expected = (
                    self.schema.indexer_bytes_per_block
                    if component is DSAL3Component.INDEXER
                    else self.schema.mla_bytes_per_block
                )
                if hbm_bytes != expected:
                    raise RuntimeError(
                        "DSA direct-HBM payload disagrees with schema: "
                        f"component={component.value}, hbm={hbm_bytes}, "
                        f"schema={expected}, addresses={len(hbm_addresses[0])}"
                    )
            regions.extend(store.get_l3_hbm_registration_regions())
        self.backend.register_buffer(
            [pointer for pointer, _ in regions],
            [size for _, size in regions],
        )
        self._registered_regions = tuple(regions)
        self._registered = True
        logger.info(
            "DSA L3 buffers registered: regions=%d bytes=%d "
            "indexer_planes=%d mla_planes=%d direct_hbm=%s",
            len(regions),
            sum(size for _, size in regions),
            indexer_planes,
            mla_planes,
            register_hbm,
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

    def lookup_prefix_blocks(
        self,
        block_hashes: Sequence[bytes],
    ) -> int:
        """返回 Indexer/MLA 均存在的最长连续 block 数。"""

        if not block_hashes:
            return 0
        keys: list[str] = []
        for block_hash in block_hashes:
            keys.extend(self.schema.block_keys(bytes(block_hash)))
        with self._io_lock:
            result = self.backend.exists(keys)
        exists = self._validate_result(
            result,
            expected_count=len(keys),
            operation="exists",
        )
        hit_blocks = 0
        for offset in range(0, len(exists), 2):
            if exists[offset] <= 0 or exists[offset + 1] <= 0:
                break
            hit_blocks += 1
        return hit_blocks

    def _descriptor_batches(
        self,
        addresses: Sequence[Sequence[int]],
    ) -> list[tuple[int, int]]:
        descriptor_counts = [len(parts) for parts in addresses]
        batches: list[tuple[int, int]] = []
        batch_start = 0
        batch_descriptors = 0
        for index, descriptor_count in enumerate(descriptor_counts):
            if batch_descriptors and (
                batch_descriptors + descriptor_count
                > self.max_descriptors_per_put
            ):
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
        if batch_start < len(addresses):
            batches.append((batch_start, len(addresses)))
        return batches

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
        with self._io_lock:
            result = self.backend.exists(keys)
        exists_result = self._validate_result(
            result,
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
        batches = self._descriptor_batches(missing_addresses)
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
            with self._io_lock:
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

    def get_load_job(
        self,
        *,
        job: DSAL3LoadJob,
        store: DSAWorkerDRAMStore,
    ) -> int:
        """同步恢复 L3-only suffix 到本 rank L2 或 split HBM。"""

        if not self._registered:
            raise RuntimeError("DSA L3 backend is not bound to its DRAM store")
        if job.direct_hbm:
            indexer_addrs, indexer_sizes = (
                store.build_l3_hbm_component_payload(
                    component=DSAL3Component.INDEXER,
                    physical_block_ids=(
                        job.destination_indexer_hbm_block_ids
                    ),
                )
            )
            mla_addrs, mla_sizes = store.build_l3_hbm_component_payload(
                component=DSAL3Component.MLA,
                physical_block_ids=(
                    job.destination_resident_hbm_block_ids
                ),
            )
        else:
            block_ids = job.destination_l2_block_ids
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
        batches = self._descriptor_batches(addresses)
        logger.info(
            "DSA L3 load backend get: request=%s blocks=%d keys=%d "
            "descriptors=%d batches=%d bytes=%d destination=%s",
            job.request_id,
            len(job.blocks),
            len(keys),
            sum(len(parts) for parts in addresses),
            len(batches),
            sum(sum(parts) for parts in sizes),
            "hbm" if job.direct_hbm else "l2",
        )
        for start, end in batches:
            with self._io_lock:
                get_with_sizes = getattr(
                    self.backend,
                    "get_with_sizes",
                    None,
                )
                result = (
                    get_with_sizes(
                        keys[start:end],
                        addresses[start:end],
                        sizes[start:end],
                    )
                    if callable(get_with_sizes)
                    else self.backend.get(
                        keys[start:end],
                        addresses[start:end],
                        sizes[start:end],
                    )
                )
            values = self._validate_result(
                result,
                expected_count=end - start,
                operation="get",
            )
            if callable(get_with_sizes):
                expected_bytes = [
                    sum(parts) for parts in sizes[start:end]
                ]
                incomplete = [
                    (index, actual, expected)
                    for index, (actual, expected) in enumerate(
                        zip(values, expected_bytes, strict=True),
                        start=start,
                    )
                    if actual != expected
                ]
                if incomplete:
                    raise RuntimeError(
                        "DSA L3 get returned incomplete payloads: "
                        f"{incomplete}"
                    )
        transferred_bytes = sum(sum(parts) for parts in sizes)
        logger.info(
            "DSA L3 load backend get finished: request=%s blocks=%d bytes=%d",
            job.request_id,
            len(job.blocks),
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
