# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA L3 在 scheduler 与 worker 之间传递的类型化任务和完成结果。

Store/load block 描述请求、逻辑块、物理源或目标块以及 Indexer/MLA/可选
MTP 的内容 key；job 将多个 block 组成一次操作。Connector metadata 携带
待加载、待上传及覆盖 L2 前必须完成的旧 job 集合；worker metadata 携带
上传结果、实际传输字节数和错误，复用基线序列化及聚合接口返回 scheduler。
``retryable`` 标识并发 put 尚未确认可读等暂态；成功结果不能同时要求重试。
put 的字节数字表示提交的 payload 量，后端幂等返回时不保证等于物理网络流量。

这些对象只保存可序列化的值，不携带 tensor、原始指针、线程或通信句柄。
字段校验约束任务形状，不能替代 IO 完成确认。当前上传结果按 TP0-only
合同聚合，重复 job ID 会报错；job/block ID 不应解释为跨 DP 的全局编号。
"""

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
    mtp_key: str | None = None

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
        if self.mtp_key is not None and not self.mtp_key:
            raise ValueError("DSA L3 MTP store key cannot be empty")

    @property
    def component_keys(self) -> tuple[str, ...]:
        keys = [self.indexer_key, self.mla_key]
        if self.mtp_key is not None:
            keys.append(self.mtp_key)
        return tuple(keys)


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


@dataclass(frozen=True, slots=True)
class DSAL3LoadBlock:
    """一个远端 component pair 到本 rank L2 bundle 的恢复项。"""

    logical_block_index: int
    destination_l2_block_id: int
    destination_indexer_hbm_block_id: int
    destination_resident_hbm_block_id: int
    block_hash: bytes
    indexer_key: str
    mla_key: str
    destination_mtp_hbm_block_id: int | None = None
    mtp_key: str | None = None

    def __post_init__(self) -> None:
        if self.logical_block_index < 0:
            raise ValueError("DSA L3 load logical block index must be non-negative")
        if self.destination_l2_block_id <= 0:
            raise ValueError("DSA L3 load destination block ID must be positive")
        if (
            self.destination_indexer_hbm_block_id <= 0
            or self.destination_resident_hbm_block_id <= 0
        ):
            raise ValueError(
                "DSA L3 load HBM destination block IDs must be positive"
            )
        if not self.block_hash:
            raise ValueError("DSA L3 load block hash cannot be empty")
        if not self.indexer_key or not self.mla_key:
            raise ValueError("DSA L3 load block requires both component keys")
        if self.mtp_key is not None and not self.mtp_key:
            raise ValueError("DSA L3 MTP load key cannot be empty")
        if (self.destination_mtp_hbm_block_id is None) != (
            self.mtp_key is None
        ):
            raise ValueError(
                "DSA L3 MTP key and HBM destination must be both present or absent"
            )
        if (
            self.destination_mtp_hbm_block_id is not None
            and self.destination_mtp_hbm_block_id <= 0
        ):
            raise ValueError("DSA L3 MTP HBM destination must be positive")

    @property
    def component_keys(self) -> tuple[str, ...]:
        keys = [self.indexer_key, self.mla_key]
        if self.mtp_key is not None:
            keys.append(self.mtp_key)
        return tuple(keys)


@dataclass(frozen=True, slots=True)
class DSAL3LoadJob:
    """一个请求连续 L3-only suffix 的同步 all-rank load。"""

    request_id: str
    blocks: tuple[DSAL3LoadBlock, ...]
    direct_hbm: bool = False

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("DSA L3 load job requires a request ID")
        if not self.blocks:
            raise ValueError("DSA L3 load job cannot be empty")
        logical_indices = tuple(
            block.logical_block_index for block in self.blocks
        )
        expected = tuple(
            range(logical_indices[0], logical_indices[0] + len(self.blocks))
        )
        if logical_indices != expected:
            raise ValueError(
                "DSA L3 load job must contain one contiguous logical suffix"
            )
        destination_ids = tuple(
            block.destination_l2_block_id for block in self.blocks
        )
        if len(set(destination_ids)) != len(destination_ids):
            raise ValueError("DSA L3 load job contains duplicate L2 destinations")
        for name, hbm_ids in (
            (
                "Indexer",
                tuple(
                    block.destination_indexer_hbm_block_id
                    for block in self.blocks
                ),
            ),
            (
                "resident",
                tuple(
                    block.destination_resident_hbm_block_id
                    for block in self.blocks
                ),
            ),
        ):
            if len(set(hbm_ids)) != len(hbm_ids):
                raise ValueError(
                    f"DSA L3 load job contains duplicate {name} HBM "
                    "destinations"
                )
        mtp_ids = self.destination_mtp_hbm_block_ids
        if mtp_ids and len(set(mtp_ids)) != len(mtp_ids):
            raise ValueError(
                "DSA L3 load job contains duplicate MTP HBM destinations"
            )

    @property
    def destination_l2_block_ids(self) -> tuple[int, ...]:
        return tuple(
            block.destination_l2_block_id for block in self.blocks
        )

    @property
    def destination_indexer_hbm_block_ids(self) -> tuple[int, ...]:
        return tuple(
            block.destination_indexer_hbm_block_id
            for block in self.blocks
        )

    @property
    def destination_resident_hbm_block_ids(self) -> tuple[int, ...]:
        return tuple(
            block.destination_resident_hbm_block_id
            for block in self.blocks
        )

    @property
    def destination_mtp_hbm_block_ids(self) -> tuple[int, ...]:
        ids = tuple(
            block.destination_mtp_hbm_block_id for block in self.blocks
        )
        if all(block_id is None for block_id in ids):
            return ()
        if any(block_id is None for block_id in ids):
            raise RuntimeError(
                "DSA L3 load job mixes MTP and non-MTP destinations"
            )
        return tuple(int(block_id) for block_id in ids if block_id is not None)


@dataclass(slots=True)
class DSAL3ConnectorMetadata(KVConnectorMetadata):
    store_jobs: dict[int, DSAL3StoreJob] = field(default_factory=dict)
    jobs_to_flush: set[int] = field(default_factory=set)
    load_jobs: tuple[DSAL3LoadJob, ...] = ()


@dataclass(frozen=True, slots=True)
class DSAL3JobResult:
    job_id: int
    success: bool
    transferred_bytes: int = 0
    error: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        if self.job_id < 0:
            raise ValueError("DSA L3 result job ID must be non-negative")
        if self.transferred_bytes < 0:
            raise ValueError("DSA L3 transferred bytes must be non-negative")
        if self.success and self.error is not None:
            raise ValueError("Successful DSA L3 result cannot contain an error")
        if self.success and self.retryable:
            raise ValueError("Successful DSA L3 result cannot request a retry")


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
