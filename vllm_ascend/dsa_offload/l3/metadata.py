# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA L3 scheduler/worker 间的 pickle-safe job 元数据。"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)


@dataclass(frozen=True, slots=True)
class DSAL3StoreBlock:
    request_id: str
    logical_block_index: int
    source_l2_block_id: int
    block_hash: bytes
    indexer_key: str
    mla_key: str

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("DSA L3 store block requires a request ID")
        if self.logical_block_index < 0:
            raise ValueError("DSA L3 logical block index must be non-negative")
        if self.source_l2_block_id <= 0:
            raise ValueError("DSA L3 source L2 block ID must be positive")
        if not self.block_hash:
            raise ValueError("DSA L3 store block hash cannot be empty")
        if not self.indexer_key or not self.mla_key:
            raise ValueError("DSA L3 store block requires both component keys")


@dataclass(frozen=True, slots=True)
class DSAL3StoreJob:
    job_id: int
    blocks: tuple[DSAL3StoreBlock, ...]

    def __post_init__(self) -> None:
        if self.job_id < 0:
            raise ValueError("DSA L3 job ID must be non-negative")
        if not self.blocks:
            raise ValueError("DSA L3 store job cannot be empty")
        source_ids = tuple(block.source_l2_block_id for block in self.blocks)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("DSA L3 store job contains duplicate L2 blocks")

    @property
    def source_l2_block_ids(self) -> tuple[int, ...]:
        return tuple(block.source_l2_block_id for block in self.blocks)


@dataclass(slots=True)
class DSAL3ConnectorMetadata(KVConnectorMetadata):
    store_jobs: dict[int, DSAL3StoreJob] = field(default_factory=dict)
    jobs_to_flush: set[int] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class DSAL3JobResult:
    job_id: int
    success: bool
    transferred_bytes: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        if self.job_id < 0:
            raise ValueError("DSA L3 result job ID must be non-negative")
        if self.transferred_bytes < 0:
            raise ValueError("DSA L3 transferred bytes must be non-negative")
        if self.success and self.error is not None:
            raise ValueError("Successful DSA L3 result cannot contain an error")


@dataclass(slots=True)
class DSAL3WorkerMetadata(KVConnectorWorkerMetadata):
    results: dict[int, DSAL3JobResult] = field(default_factory=dict)

    def aggregate(
        self,
        other: KVConnectorWorkerMetadata,
    ) -> KVConnectorWorkerMetadata:
        if not isinstance(other, DSAL3WorkerMetadata):
            raise TypeError("DSA L3 worker metadata can only aggregate its own type")
        duplicate = self.results.keys() & other.results.keys()
        if duplicate:
            raise RuntimeError(
                f"DSA rank0-only store produced duplicate worker results: job_ids={tuple(sorted(duplicate))}"
            )
        return DSAL3WorkerMetadata(results={**self.results, **other.results})
