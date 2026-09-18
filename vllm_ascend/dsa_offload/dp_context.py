# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA 在 EngineCore 初始化期使用的 DP 上下文与节点 DRAM 预算协调。

每个 DP 域由自己的 EngineCore 持有基线已建立的 stateless CPU process
group。平台 patch 将该 group 与 rank 信息保存在本进程，稍后的 KV-cache
容量规划通过本模块读取；这里不创建或销毁第二套通信组。

在各域 L1 配置已确定、L2 arena 尚未分配时，收集节点身份、可用内存观测、
L1 Indexer 容量、bundle 大小和兼容性指纹。按宿主内核 boot ID 把各 DP 域
分组为共享 host DRAM 的节点范围；每个节点范围内取各观测的最小值作为该
节点的共同预算，再等分为该节点每 DP 的字节上限。跨节点时各节点独立预算，
各域实际 L2 块数仍由自己的 L1 比例决定。不以可重复的 hostname 作为身份。
探测失败也先参与同一次汇总，再让各域一致拒绝，不能只让一域提前退出。
同时校验共享 L3 的确定性哈希 seed 前置条件，避免相同签名却产生不同内容 key。

上下文属于单个 EngineCore 进程，不共享请求、block ID 或 prefix 目录，
也不参与每轮 forward。API 前端的跨 DP reset 结果聚合位于 ``dp_control``。
"""

from __future__ import annotations

import hashlib
import json
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import torch.distributed as dist

from vllm_ascend import envs

_HOST_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_MAX_PYTHON_HASH_SEED = (1 << 32) - 1


def require_dsa_shared_l3_hash_seed() -> str:
    """校验标准Python seed但不改写原文；vLLM用原字符串构造首块哈希根。"""

    seed = envs.PYTHONHASHSEED
    if (
        not isinstance(seed, str)
        or not seed
        or not seed.isascii()
        or not seed.isdecimal()
        or len(seed.lstrip("0")) > 10
        or int(seed.lstrip("0") or "0") > _MAX_PYTHON_HASH_SEED
    ):
        raise ValueError(
            "DSA DP shared L3 requires an explicit, identical PYTHONHASHSEED "
            "in [0, 4294967295] before launching Python (for example 114514); "
            "an unset seed randomizes vLLM's prefix hash root; Python random mode is not accepted for shared L3"
        )
    return seed


def _read_host_boot_id() -> str:
    """读取同宿主容器共享的内核启动标识；缺失时不回退hostname/machine-id。"""

    boot_id = UUID(_HOST_BOOT_ID_PATH.read_text(encoding="ascii").strip())
    if boot_id.int == 0:
        raise ValueError("host boot ID must not be zero")
    return str(boot_id)


@dataclass(frozen=True, slots=True)
class _NodeMemoryObservation:
    """已有EngineCore CPU group汇总的初始化信息，不携带设备资源。"""

    rank: int
    hostname: str
    boot_id: str | None
    probe_error: str | None
    available_bytes: int
    l1_indexer_blocks: int
    bundle_bytes: int
    tp_size: int
    signature: str


@dataclass(frozen=True, slots=True)
class DSAEngineDPContext:
    process_group: Any
    rank: int
    world_size: int
    local_rank: int


@dataclass(frozen=True, slots=True)
class DSANodeMemoryBudget:
    """本 rank 所在 host 节点的 DRAM 预算快照。

    多节点 DP 时 ``observed_available_bytes_by_dp`` 与
    ``l1_indexer_blocks_by_dp`` 只覆盖与本 rank 共享 host 的 DP 域
    （按全局 rank 升序），``data_parallel_rank``/``data_parallel_size``
    仍描述全局 DP 拓扑。
    """

    node_scope_id: str
    snapshot_id: str
    compatibility_signature: str
    data_parallel_rank: int
    data_parallel_size: int
    observed_available_bytes_by_dp: tuple[int, ...]
    l1_indexer_blocks_by_dp: tuple[int, ...]
    node_available_bytes: int
    per_dp_quota_bytes: int
    minimum_per_dp_quota_bytes: int


_ENGINE_DP_CONTEXT: DSAEngineDPContext | None = None


def install_dsa_engine_dp_context(
    *,
    process_group: Any,
    rank: int,
    world_size: int,
    local_rank: int,
) -> None:
    """在当前进程登记已有 EngineCore DP group，拒绝不一致的重复登记。"""

    global _ENGINE_DP_CONTEXT
    context = DSAEngineDPContext(
        process_group=process_group,
        rank=int(rank),
        world_size=int(world_size),
        local_rank=int(local_rank),
    )
    if context.world_size <= 1:
        raise ValueError("DSA EngineCore DP context requires world_size > 1")
    if not 0 <= context.rank < context.world_size:
        raise ValueError(
            f"DSA EngineCore DP rank is outside the process group: rank={context.rank}, world_size={context.world_size}"
        )
    if _ENGINE_DP_CONTEXT is not None:
        previous = _ENGINE_DP_CONTEXT
        if (
            previous.process_group is not process_group
            or previous.rank != context.rank
            or previous.world_size != context.world_size
            or previous.local_rank != context.local_rank
        ):
            raise RuntimeError("DSA EngineCore DP context was installed twice")
        return
    _ENGINE_DP_CONTEXT = context


def reset_dsa_engine_dp_context() -> None:
    """清除本进程的上下文引用，供测试及进程清理使用，不销毁通信组。"""

    global _ENGINE_DP_CONTEXT
    _ENGINE_DP_CONTEXT = None


def get_dsa_engine_dp_context() -> DSAEngineDPContext | None:
    return _ENGINE_DP_CONTEXT


def _snapshot_id(
    node_identity: str,
    observations: tuple[int, ...],
) -> tuple[str, str]:
    node_scope_id = hashlib.sha256(node_identity.encode("utf-8")).hexdigest()[:16]
    payload = json.dumps(
        {
            "node_scope_id": node_scope_id,
            "observed_available_bytes_by_dp": observations,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    snapshot_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return node_scope_id, snapshot_id


def coordinate_dsa_node_memory_budget(
    *,
    data_parallel_size: int,
    data_parallel_rank: int,
    observed_available_bytes: int,
    l1_indexer_blocks: int,
    complete_bundle_bytes: int,
    tensor_parallel_size: int,
    compatibility_signature: str,
) -> DSANodeMemoryBudget:
    """汇总各 DP 的内存观测，按 host 节点分组返回本节点的等额字节预算上限。

    全部 EngineCore 必须在任何 L2 arena 分配前参与。共享同一 host（boot ID
    一致）的 DP 域组成一个节点范围，取范围内观测最小值再等分给范围内各 DP；
    跨节点时各节点独立预算。等额只约束字节上限，不要求各 DP 的实际 L2 块数
    一致；本函数不分配或格式化 payload 内存。
    """

    dp_size = int(data_parallel_size)
    dp_rank = int(data_parallel_rank)
    observed = int(observed_available_bytes)
    local_l1_blocks = int(l1_indexer_blocks)
    bundle_bytes = int(complete_bundle_bytes)
    tp_size = int(tensor_parallel_size)
    signature = str(compatibility_signature)
    if observed <= 0:
        raise ValueError(f"DSA host-memory observation must be positive, got {observed}")
    if local_l1_blocks <= 0 or bundle_bytes <= 0 or tp_size <= 0:
        raise ValueError(
            "DSA DP capacity metadata must be positive: "
            f"l1_indexer_blocks={local_l1_blocks}, "
            f"complete_bundle_bytes={bundle_bytes}, tp_size={tp_size}"
        )
    if not signature:
        raise ValueError("DSA DP compatibility signature must not be empty")

    node_identity = socket.gethostname()
    if dp_size <= 1:
        observations = (observed,)
        l1_blocks_by_dp = (local_l1_blocks,)
        minimum_quota = (local_l1_blocks + 1) * bundle_bytes * tp_size
        node_scope_id, snapshot_id = _snapshot_id(
            node_identity,
            observations,
        )
        return DSANodeMemoryBudget(
            node_scope_id=node_scope_id,
            snapshot_id=snapshot_id,
            compatibility_signature=signature,
            data_parallel_rank=0,
            data_parallel_size=1,
            observed_available_bytes_by_dp=observations,
            l1_indexer_blocks_by_dp=l1_blocks_by_dp,
            node_available_bytes=observed,
            per_dp_quota_bytes=observed,
            minimum_per_dp_quota_bytes=minimum_quota,
        )

    context = _ENGINE_DP_CONTEXT
    if context is None:
        raise RuntimeError("DSA DP host-memory planning has no EngineCore DP context")
    if context.world_size != dp_size or context.rank != dp_rank:
        raise RuntimeError(
            "DSA EngineCore DP context disagrees with VllmConfig: "
            f"context=({context.rank}/{context.world_size}), "
            f"config=({dp_rank}/{dp_size})"
        )

    boot_id, probe_error = None, None
    try:
        boot_id = _read_host_boot_id()
    except (OSError, ValueError) as exc:
        probe_error = f"{type(exc).__name__}: {exc}"
    local_sample = _NodeMemoryObservation(
        rank=dp_rank,
        hostname=node_identity,
        boot_id=boot_id,
        probe_error=probe_error,
        available_bytes=observed,
        l1_indexer_blocks=local_l1_blocks,
        bundle_bytes=bundle_bytes,
        tp_size=tp_size,
        signature=signature,
    )
    gathered: list[_NodeMemoryObservation | None] = [None] * dp_size
    dist.all_gather_object(
        gathered,
        local_sample,
        group=context.process_group,
    )
    if any(not isinstance(sample, _NodeMemoryObservation) for sample in gathered):
        raise RuntimeError("DSA DP host-memory gather returned an invalid/empty observation; synchronize code versions")
    samples = [sample for sample in gathered if isinstance(sample, _NodeMemoryObservation)]
    samples.sort(key=lambda sample: sample.rank)
    ranks = tuple(sample.rank for sample in samples)
    if ranks != tuple(range(dp_size)):
        raise RuntimeError(
            f"DSA DP host-memory gather returned invalid ranks: ranks={ranks}, expected={tuple(range(dp_size))}"
        )

    if samples[dp_rank] != local_sample:
        raise RuntimeError("DSA DP host-memory gather disagrees with the local observation")
    failed_probes = {
        sample.rank: sample.probe_error or "missing boot ID"
        for sample in samples
        if sample.probe_error is not None or sample.boot_id is None
    }
    if failed_probes:
        raise RuntimeError(
            "DSA DP cannot verify the shared host-memory scope: "
            f"boot-ID probe failures={failed_probes}. Ensure {_HOST_BOOT_ID_PATH} "
            "is readable and exposes the host kernel boot identity in every EngineCore; "
            "hostname fallback is not allowed"
        )
    observations = tuple(sample.available_bytes for sample in samples)
    if any(value <= 0 for value in observations):
        raise RuntimeError(
            f"DSA DP host-memory gather returned a non-positive observation: observations={observations}"
        )
    bundle_sizes = {sample.bundle_bytes for sample in samples}
    tp_sizes = {sample.tp_size for sample in samples}
    if len(bundle_sizes) != 1 or len(tp_sizes) != 1:
        raise RuntimeError(
            "DSA DP ranks have incompatible L2 layouts: "
            f"bundle_bytes={sorted(bundle_sizes)}, "
            f"tp_sizes={sorted(tp_sizes)}"
        )
    signatures = {sample.signature for sample in samples}
    if len(signatures) != 1:
        raise RuntimeError(f"DSA DP ranks have incompatible configuration signatures: signatures={sorted(signatures)}")
    l1_blocks_by_dp = tuple(sample.l1_indexer_blocks for sample in samples)
    if any(value <= 0 for value in l1_blocks_by_dp):
        raise RuntimeError(f"DSA DP gather returned a non-positive L1 Indexer capacity: blocks={l1_blocks_by_dp}")
    shared_bundle_bytes = bundle_sizes.pop()
    shared_tp_size = tp_sizes.pop()
    shared_signature = signatures.pop()
    if shared_signature != signature:
        raise RuntimeError(
            f"DSA DP gathered signature does not match the local rank: local={signature}, gathered={shared_signature}"
        )

    # Ranks sharing one host boot ID draw from the same host DRAM; every
    # node scope budgets itself independently of the other nodes. All ranks
    # validate every node and reject together, so a node that cannot meet
    # its minimum quota never starts while its peers continue.
    local_boot_id = samples[dp_rank].boot_id
    assert local_boot_id is not None
    node_samples_by_scope: dict[str, tuple[_NodeMemoryObservation, ...]] = {}
    for sample in samples:
        assert sample.boot_id is not None
        scope_samples = node_samples_by_scope.setdefault(sample.boot_id, ())
        node_samples_by_scope[sample.boot_id] = (*scope_samples, sample)
    quota_failures: dict[str, str] = {}
    for scope_boot_id, scope_samples in node_samples_by_scope.items():
        scope_observations = tuple(sample.available_bytes for sample in scope_samples)
        scope_l1_blocks = tuple(sample.l1_indexer_blocks for sample in scope_samples)
        scope_hostnames = tuple(sample.hostname for sample in scope_samples)
        scope_available = min(scope_observations)
        scope_quota = scope_available // len(scope_samples)
        scope_minimum = (max(scope_l1_blocks) + 1) * shared_bundle_bytes * shared_tp_size
        if scope_quota <= 0:
            quota_failures[scope_boot_id] = (
                f"empty quota after equal partitioning: node_available={scope_available}, "
                f"node_dp_size={len(scope_samples)}, node_hostnames={scope_hostnames}"
            )
        elif scope_quota < scope_minimum:
            quota_failures[scope_boot_id] = (
                "quota cannot hold one complete mirror of the largest local L1 Indexer capacity: "
                f"per_dp_quota={scope_quota}, minimum_per_dp_quota={scope_minimum}, "
                f"l1_indexer_blocks={scope_l1_blocks}, node_hostnames={scope_hostnames}"
            )
    if quota_failures:
        raise RuntimeError(f"DSA DP host-memory budgeting failed: {quota_failures}")

    node_samples = node_samples_by_scope[local_boot_id]
    node_observations = tuple(sample.available_bytes for sample in node_samples)
    node_l1_blocks = tuple(sample.l1_indexer_blocks for sample in node_samples)
    node_available = min(node_observations)
    per_dp_quota = node_available // len(node_samples)
    minimum_quota = (max(node_l1_blocks) + 1) * shared_bundle_bytes * shared_tp_size
    node_scope_id, snapshot_id = _snapshot_id(
        local_boot_id,
        node_observations,
    )
    return DSANodeMemoryBudget(
        node_scope_id=node_scope_id,
        snapshot_id=snapshot_id,
        compatibility_signature=shared_signature,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        observed_available_bytes_by_dp=node_observations,
        l1_indexer_blocks_by_dp=node_l1_blocks,
        node_available_bytes=node_available,
        per_dp_quota_bytes=per_dp_quota,
        minimum_per_dp_quota_bytes=minimum_quota,
    )
