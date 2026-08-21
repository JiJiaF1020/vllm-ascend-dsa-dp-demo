# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA scheduler 侧的单机 DRAM 容量合同与 block-id 目录。

EngineCore 在 HBM KV-cache 配置完成各 rank 收敛后，只观测一次单机可用
内存并生成 ``DSADRAMCapacityPlan``。``DSAKVCacheCoordinator`` 持有唯一
``DSASchedulerDRAMBlockDirectory``，在已提交 Indexer block table 的覆盖内按
durable full-block 前缀增量分配 DRAM block id；worker 只消费 assignment，
不在数据热路径分配。

目录采用组合而不是继承：底层分配、null block、refcount 和 free queue 直接
委托给 vLLM 原生 ``BlockPool``，DSA 只在外层增加 request→logical block
映射。只有未来出现无法由组合表达的介质事件或驱逐语义时，才考虑派生
``BlockPool``，避免当前复制或覆盖基线分配算法。

首版只承诺单机 TP，且 prefix cache、preemption、KV connector 仍关闭。
因此目录暂时使用 request-private 行，不包含 hash/refcount/LRU。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock


@dataclass(frozen=True, slots=True)
class DSADRAMCapacityPlan:
    """由 scheduler 生成并随 KVCacheConfig 下发的单机 DRAM 容量合同。"""

    host_available_bytes: int
    tensor_parallel_size: int
    per_rank_available_bytes: int
    bytes_per_logical_block: int
    minimum_usable_blocks: int
    desired_usable_blocks: int
    usable_blocks: int

    @property
    def arena_capacity(self) -> int:
        """包含空映射 block 0 的真实 arena 行数。"""

        return self.usable_blocks + 1

    @property
    def per_rank_reserved_bytes(self) -> int:
        return self.arena_capacity * self.bytes_per_logical_block

    @property
    def total_reserved_bytes(self) -> int:
        return self.per_rank_reserved_bytes * self.tensor_parallel_size

    @property
    def effective_block_multiple(self) -> float:
        return self.usable_blocks / self.minimum_usable_blocks

    @property
    def was_downsized(self) -> bool:
        return self.usable_blocks < self.desired_usable_blocks


def _read_cgroup_remaining_bytes() -> int | None:
    """读取常见 cgroup v1/v2 内存余量；读取失败时回退到宿主机值。"""

    candidates = (
        (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory.current"),
        ),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, usage_path in candidates:
        try:
            limit_text = limit_path.read_text(encoding="ascii").strip()
            if limit_text == "max":
                continue
            limit = int(limit_text)
            usage = int(usage_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        if limit <= 0:
            continue
        return max(limit - usage, 0)
    return None


def get_host_available_memory_bytes() -> int:
    """返回当前进程可使用的主机内存，兼顾容器 cgroup 上限。"""

    available = int(psutil.virtual_memory().available)
    cgroup_remaining = _read_cgroup_remaining_bytes()
    if cgroup_remaining is not None:
        available = min(available, cgroup_remaining)
    if available <= 0:
        raise RuntimeError(
            "DSA scheduler observed no available host memory for the DRAM pool"
        )
    return available


def calculate_dram_usable_blocks(
    indexer_num_blocks: int,
    block_multiple: float,
) -> int:
    """按浮点倍数向上取整，避免截断用户配置。"""

    indexer_num_blocks = int(indexer_num_blocks)
    block_multiple = float(block_multiple)
    if indexer_num_blocks <= 0:
        raise ValueError(
            "DSA DRAM sizing requires positive Indexer blocks, got "
            f"{indexer_num_blocks}"
        )
    if not math.isfinite(block_multiple) or block_multiple <= 0:
        raise ValueError(
            "DSA hot_cpu_block_multiple must be positive and finite, got "
            f"{block_multiple}"
        )
    return int(math.ceil(indexer_num_blocks * block_multiple))


def calculate_dram_capacity_plan(
    *,
    indexer_num_blocks: int,
    block_multiple: float,
    bytes_per_logical_block: int,
    tensor_parallel_size: int,
    host_available_bytes: int,
) -> DSADRAMCapacityPlan:
    """按完整 Indexer+MLA block 计算每 rank 的固定 DRAM 容量。"""

    minimum = int(indexer_num_blocks)
    desired = calculate_dram_usable_blocks(minimum, block_multiple)
    bytes_per_logical_block = int(bytes_per_logical_block)
    tensor_parallel_size = int(tensor_parallel_size)
    host_available_bytes = int(host_available_bytes)
    if bytes_per_logical_block <= 0:
        raise ValueError(
            "DSA DRAM bytes_per_logical_block must be positive, got "
            f"{bytes_per_logical_block}"
        )
    if tensor_parallel_size <= 0:
        raise ValueError(
            "DSA DRAM tensor_parallel_size must be positive, got "
            f"{tensor_parallel_size}"
        )
    if host_available_bytes <= 0:
        raise ValueError(
            "DSA DRAM host_available_bytes must be positive, got "
            f"{host_available_bytes}"
        )

    per_rank_available = host_available_bytes // tensor_parallel_size
    # 每个 arena 都保留 block 0 作为合法空映射，不能把这一行遗漏在
    # scheduler 的容量核算之外。
    max_usable = per_rank_available // bytes_per_logical_block - 1
    if max_usable < minimum:
        minimum_bytes_per_rank = (minimum + 1) * bytes_per_logical_block
        raise RuntimeError(
            "DSA host DRAM cannot hold one complete mirror of the per-rank "
            "HBM Indexer capacity: "
            f"host_available={host_available_bytes}, "
            f"tp_size={tensor_parallel_size}, "
            f"per_rank_available={per_rank_available}, "
            f"complete_block_bytes={bytes_per_logical_block}, "
            f"minimum_blocks={minimum}, "
            f"minimum_per_rank_bytes={minimum_bytes_per_rank}"
        )
    usable = max(minimum, min(desired, max_usable))
    if usable > np.iinfo(np.int32).max:
        raise RuntimeError(
            "DSA DRAM block capacity exceeds the int32 block-id ABI: "
            f"usable_blocks={usable}"
        )
    return DSADRAMCapacityPlan(
        host_available_bytes=host_available_bytes,
        tensor_parallel_size=tensor_parallel_size,
        per_rank_available_bytes=per_rank_available,
        bytes_per_logical_block=bytes_per_logical_block,
        minimum_usable_blocks=minimum,
        desired_usable_blocks=desired,
        usable_blocks=usable,
    )


class DSASchedulerDRAMBlockDirectory:
    """scheduler 侧请求逻辑块到全局 DRAM block id 的唯一账本。"""

    def __init__(
        self,
        *,
        usable_blocks: int,
        max_logical_blocks: int,
        block_size: int,
    ) -> None:
        if usable_blocks <= 0:
            raise ValueError("DSA DRAM directory requires positive capacity")
        if max_logical_blocks <= 0:
            raise ValueError("DSA DRAM directory requires positive row width")
        if usable_blocks > np.iinfo(np.int32).max:
            raise ValueError("DSA DRAM directory exceeds the int32 block-id ABI")
        if block_size <= 0:
            raise ValueError("DSA DRAM directory requires a positive block size")
        self.usable_blocks = int(usable_blocks)
        self.max_logical_blocks = int(max_logical_blocks)
        # vLLM BlockPool 会自动取出 block 0 作为 null block。传入
        # usable+1 后，真实 free queue 恰好仍有 usable 个有效 block。
        self.block_pool = BlockPool(
            num_gpu_blocks=self.usable_blocks + 1,
            enable_caching=False,
            hash_block_size=int(block_size),
        )
        if self.block_pool.null_block.block_id != 0:
            raise RuntimeError("DSA DRAM BlockPool did not reserve block 0")
        self._request_blocks: dict[str, list[KVCacheBlock]] = {}

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.get_num_free_blocks()

    def ensure_request_blocks(
        self,
        request_id: str,
        logical_block_count: int,
    ) -> tuple[int, tuple[int, ...]]:
        """把请求逻辑表扩展到指定长度，返回 ``(起点, 新 block IDs)``。"""

        logical_block_count = int(logical_block_count)
        if not 0 <= logical_block_count <= self.max_logical_blocks:
            raise ValueError(
                "DSA request DRAM width is outside directory capacity: "
                f"request_id={request_id!r}, required={logical_block_count}, "
                f"capacity={self.max_logical_blocks}"
            )
        blocks = self._request_blocks.setdefault(request_id, [])
        old_count = len(blocks)
        if logical_block_count < old_count:
            raise RuntimeError(
                "DSA request DRAM table cannot shrink before request release: "
                f"request_id={request_id!r}, old={old_count}, "
                f"new={logical_block_count}"
            )
        missing = logical_block_count - old_count
        if not missing:
            return old_count, ()
        free_blocks = self.block_pool.get_num_free_blocks()
        if missing > free_blocks:
            raise RuntimeError(
                "DSA scheduler DRAM block directory is exhausted: "
                f"request_id={request_id!r}, required={missing}, "
                f"free={free_blocks}, capacity={self.usable_blocks}"
            )
        allocated = self.block_pool.get_new_blocks(missing)
        blocks.extend(allocated)
        return old_count, tuple(block.block_id for block in allocated)

    def get_request_blocks(self, request_id: str) -> tuple[int, ...]:
        return tuple(
            block.block_id
            for block in self._request_blocks.get(request_id, ())
        )

    def free_request(self, request_id: str) -> tuple[int, ...]:
        released = self._request_blocks.pop(request_id, None)
        if not released:
            return ()
        released_ids = tuple(block.block_id for block in released)
        self.block_pool.free_blocks(reversed(released))
        return released_ids
