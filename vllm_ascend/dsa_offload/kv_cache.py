# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA Indexer/MLA 解耦后的 KV-cache 规格与物理容量规划。

vLLM v0.23 将 KV-cache 的职责拆成三层：

* ``KVCacheSpecRegistry`` 决定一种 spec 使用哪个 manager、能否与其他
  spec 归入同一组；
* ``KVCacheGroupSpec`` 表达共享同一张 block table 的逻辑层集合；
* ``KVCacheTensor`` 表达 worker 最终实际分配的字节数。

DSA 的 Indexer dense plane 与 MLA resident plane 在这三层都必须显式分离。
本模块只负责静态规格、分组、容量和最终 tensor 布局，不处理请求阶段转换，
也不在对象上动态追加 ``dsa_num_blocks`` 一类旁路字段。每个 plane 的真实
block 数统一由最终 ``KVCacheTensor.size / spec.page_size_bytes`` 推导。

L1 配置确定后，本模块还会定稿 L2 容量并附入类型化 KV-cache config。
DP 的共同节点预算通过 ``dp_context`` 协调，具体块容量计算复用
``scheduler_dram_pool``；各域等额的是字节预算上限，实际块数仍随本域
L1 容量和配置比例变化。

同机多 DP 的容量报告通过文件锁互斥输出，避免进程日志前缀的分段写入
交错。锁只覆盖报告，不参与容量计算、DP collective 或请求执行。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace

import torch
from filelock import FileLock, Timeout
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheTensor,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.dsa_offload.dp_context import (
    coordinate_dsa_node_memory_budget,
    require_dsa_shared_l3_hash_seed,
)
from vllm_ascend.dsa_offload.scheduler_dram_pool import (
    DSADRAMCapacityPlan,
    calculate_dram_capacity_plan,
    get_host_available_memory_bytes,
)

# 沿用 v0.19 DSA 原型的物理块数对齐。该对齐只作用于自动计算出来的
# base block 数；显式 num_gpu_blocks_override 仍由 vLLM 的标准入口接管。
DSA_KV_BLOCK_COUNT_ALIGNMENT = 128
DSA_CAPACITY_LOG_LOCK_TIMEOUT_SECONDS = 5


def _log_dsa_capacity_report(vllm_config: VllmConfig, message: str, *args: object) -> None:
    """仅序列化同机容量报告；日志锁不可用时降级输出，不阻止引擎启动。"""

    if int(getattr(vllm_config.parallel_config, "data_parallel_size", 1)) <= 1:
        logger.info_once(message, *args)
        return
    # 不含DP/rank/PID，保证各EngineCore争用同一把锁；不同用户独立。
    uid = getattr(os, "getuid", lambda: 0)()
    lock = FileLock(os.path.join(tempfile.gettempdir(), f"vllm-ascend-dsa-capacity-{uid}.lock"), mode=0o600)
    try:
        lock.acquire(timeout=DSA_CAPACITY_LOG_LOCK_TIMEOUT_SECONDS)
    except (OSError, Timeout) as exc:
        logger.warning("DSA capacity report lock unavailable; output may interleave: %s", exc)
        logger.info_once(message, *args)
        return
    try:
        # 标准StreamHandler在返回前flush；保留原生格式、进程前缀与info_once语义。
        logger.info_once(message, *args)
    finally:
        lock.release()


def _may_override_num_blocks(
    vllm_config: VllmConfig,
    num_blocks: int,
) -> int:
    override = vllm_config.cache_config.num_gpu_blocks_override
    return int(override) if override is not None else num_blocks


def dsa_mtp_cache_identity(vllm_config: VllmConfig) -> dict[str, object] | None:
    """描述产生MTP KV的模型身份，供DP启动校验和共享L3 schema共同使用。"""

    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is None:
        return None
    draft = speculative.draft_model_config
    compute_hash = getattr(draft, "compute_hash", None)
    return {
        "method": str(speculative.method),
        "num_speculative_tokens": int(speculative.num_speculative_tokens),
        "model": str(getattr(draft, "model", "")),
        "revision": str(getattr(draft, "revision", "") or ""),
        "config_hash": str(compute_hash()) if callable(compute_hash) else "",
    }


def build_dsa_dp_compatibility_signature(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    dsa_config: object,
) -> str:
    """Fingerprint DSA semantics that must agree across DP EngineCores."""

    model_config = vllm_config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    compute_hash = getattr(model_config, "compute_hash", None)
    model_config_hash = (
        str(compute_hash()) if callable(compute_hash) else ""
    )
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config
    compilation_config = vllm_config.compilation_config
    speculative_config = getattr(vllm_config, "speculative_config", None)
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)

    def group_identity(group: KVCacheGroupSpec) -> dict[str, object]:
        spec = group.kv_cache_spec
        sparse_head_dim = getattr(spec, "sparse_head_dim", None)
        return {
            "type": f"{type(spec).__module__}.{type(spec).__qualname__}",
            "dtype": str(getattr(spec, "dtype", "")),
            "block_size": int(spec.block_size),
            "page_size_bytes": int(spec.page_size_bytes),
            "num_kv_heads": int(getattr(spec, "num_kv_heads", 0)),
            "head_size": int(getattr(spec, "head_size", 0)),
            "sparse_head_dim": (
                tuple(int(value) for value in sparse_head_dim)
                if sparse_head_dim is not None
                else None
            ),
            "cache_sparse_sfa_c8": bool(
                getattr(spec, "cache_sparse_sfa_c8", False)
            ),
            "cache_sparse_li_c8": bool(
                getattr(spec, "cache_sparse_li_c8", False)
            ),
            "layers": tuple(group.layer_names),
        }

    descriptor = {
        "version": 1,
        "model": {
            "model": str(getattr(model_config, "model", "")),
            "revision": str(
                getattr(model_config, "revision", "") or ""
            ),
            "architecture": str(
                getattr(model_config, "architecture", "")
            ),
            "architectures": tuple(
                getattr(hf_config, "architectures", ()) or ()
            ),
            "model_type": str(getattr(hf_config, "model_type", "")),
            "quantization": str(
                getattr(model_config, "quantization", "") or ""
            ),
            "max_model_len": int(
                getattr(model_config, "max_model_len", 0) or 0
            ),
            "enforce_eager": bool(
                getattr(model_config, "enforce_eager", False)
            ),
            "config_hash": model_config_hash,
        },
        "parallel": {
            "tp": int(getattr(parallel_config, "tensor_parallel_size", 1)),
            "pp": int(getattr(parallel_config, "pipeline_parallel_size", 1)),
            "pcp": int(
                getattr(parallel_config, "prefill_context_parallel_size", 1)
            ),
            "dcp": int(
                getattr(parallel_config, "decode_context_parallel_size", 1)
            ),
            "dp": int(getattr(parallel_config, "data_parallel_size", 1)),
            "ep": bool(
                getattr(parallel_config, "enable_expert_parallel", False)
            ),
            "all2all_backend": str(
                getattr(parallel_config, "all2all_backend", "")
            ),
        },
        "scheduler": {
            "max_num_seqs": int(
                getattr(scheduler_config, "max_num_seqs", 0) or 0
            ),
            "max_num_batched_tokens": int(
                getattr(scheduler_config, "max_num_batched_tokens", 0) or 0
            ),
            "chunked_prefill": bool(
                getattr(scheduler_config, "enable_chunked_prefill", False)
            ),
            "reserve_full_isl": bool(
                getattr(scheduler_config, "scheduler_reserve_full_isl", False)
            ),
        },
        "cache": {
            "block_size": int(getattr(cache_config, "block_size", 0) or 0),
            "hash_block_size": int(
                getattr(cache_config, "hash_block_size", 0) or 0
            ),
            "prefix": bool(
                getattr(cache_config, "enable_prefix_caching", False)
            ),
        },
        "dsa": {
            "indexer_mla_block_ratio": int(
                getattr(dsa_config, "indexer_mla_block_ratio", 0)
            ),
            "sparse_activation_tokens": int(
                getattr(dsa_config, "sparse_activation_tokens", 0)
            ),
            "prompt_budget_thresholds": tuple(
                getattr(dsa_config, "prompt_budget_thresholds", ())
            ),
            "resident_budget_tokens": tuple(
                getattr(dsa_config, "resident_budget_tokens", ())
            ),
            "max_active_reqs": int(
                getattr(dsa_config, "max_active_reqs", 0)
            ),
            "hot_cpu_block_multiple": float(
                getattr(dsa_config, "hot_cpu_block_multiple", 0.0)
            ),
            "row_mode_graph": bool(
                getattr(dsa_config, "enable_row_mode_decode_graph", False)
            ),
            "mtp_tokens": int(
                getattr(dsa_config, "mtp_num_speculative_tokens", 0)
            ),
            "mtp_layers": int(
                getattr(dsa_config, "mtp_cache_layer_count", 0)
            ),
        },
        "speculative": {
            "method": str(getattr(speculative_config, "method", "")),
            "tokens": int(
                getattr(speculative_config, "num_speculative_tokens", 0)
                or 0
            ),
        },
        "connector": {
            "name": str(getattr(kv_transfer_config, "kv_connector", "")),
            "role": str(getattr(kv_transfer_config, "kv_role", "")),
        },
        "compilation": {
            "mode": str(getattr(compilation_config, "mode", "")),
            "cudagraph_mode": str(
                getattr(compilation_config, "cudagraph_mode", "")
            ),
            "capture_sizes": tuple(
                getattr(
                    compilation_config,
                    "cudagraph_capture_sizes",
                    (),
                )
                or ()
            ),
        },
        "groups": tuple(group_identity(group) for group in kv_cache_groups),
    }
    if speculative_config is not None:
        descriptor["speculative"]["cache_identity"] = dsa_mtp_cache_identity(vllm_config)
        descriptor["speculative"]["enforce_eager"] = bool(getattr(speculative_config, "enforce_eager", False))
        descriptor["speculative"]["draft_tp"] = int(
            getattr(speculative_config, "draft_tensor_parallel_size", None)
            or getattr(parallel_config, "tensor_parallel_size", 1)
        )
    if (
        getattr(kv_transfer_config, "kv_connector", None) == "DSAL3Connector"
        and int(getattr(parallel_config, "data_parallel_size", 1)) > 1
    ):
        hash_seed = require_dsa_shared_l3_hash_seed()
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
            MooncakeStoreConfig,
        )

        l3_config = MooncakeStoreConfig.load_from_env()
        descriptor["l3"] = {
            name: getattr(l3_config, name)
            for name in (
                "protocol", "metadata_server", "master_server_address",
                "global_segment_size", "local_buffer_size", "tcp_segment_port_base",
            )
        }
        descriptor["l3"]["hash_algorithm"] = str(getattr(cache_config, "prefix_caching_hash_algo", ""))
        descriptor["l3"]["python_hash_seed"] = hash_seed
    encoded = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


@dataclass(frozen=True, kw_only=True)
class DSAIndexerKVSpec(FullAttentionSpec):
    """完整驻留 HBM 的 Indexer dense plane。

    ``FullAttentionSpec`` 默认按 K+V 两个向量计算 page 大小，而 Indexer
    cache 每个 token 只有一个 key 向量，因此这里覆盖真实 page 字节数。
    继承 ``FullAttentionSpec`` 是为了复用 full-attention 的块生命周期
    语义；独立的 class identity 则阻止 registry 将它和 resident MLA
    自动合并成同一 KV-cache group。
    """

    cache_sparse_li_c8: bool = False
    c8_k_cache_dtype: torch.dtype = torch.float8_e4m3fn
    c8_k_scale_cache_dtype: torch.dtype = torch.float32
    c8_scale_dim: int = 1

    @classmethod
    def merge(cls, specs: list[DSAIndexerKVSpec]) -> DSAIndexerKVSpec:
        """合并同构 Indexer plane，同时保留 DSA/C8 扩展布局字段。

        ``FullAttentionSpec.merge`` 只重建基类字段；直接继承它会静默把
        ``cache_sparse_li_c8`` 等扩展字段恢复成默认值，导致 engine 侧容量
        规划与 worker 侧真实分配不一致。
        """

        assert specs, "Cannot merge an empty DSA Indexer spec list."
        assert all(type(spec) is cls for spec in specs), "All layers in a DSA Indexer group must use DSAIndexerKVSpec."
        assert all(spec == specs[0] for spec in specs[1:]), (
            "All layers in a DSA Indexer group must use the same cache layout."
        )
        return replace(specs[0])

    @property
    def real_page_size_bytes(self) -> int:
        token_bytes = self.head_size * get_dtype_size(self.dtype)
        if self.cache_sparse_li_c8:
            token_bytes = self.head_size * get_dtype_size(self.c8_k_cache_dtype) + self.c8_scale_dim * get_dtype_size(
                self.c8_k_scale_cache_dtype
            )
        return self.block_size * self.num_kv_heads * token_bytes


@dataclass(frozen=True, kw_only=True)
class DSAResidentMLAAttentionSpec(AscendMLAAttentionSpec):
    """仅保存 attention resident working set 的 MLA plane。

    该 spec 的 ``sparse_head_dim`` 必须把 Indexer 维度设为 0；Indexer
    cache 由 ``DSAIndexerKVSpec`` 独立描述。请求何时从 dense 布局收缩到
    resident budget 属于 scheduler/manager 生命周期语义，不放在 spec 中。
    """


@dataclass(frozen=True)
class DSAKVCacheGroupIds:
    """finalized KVCacheConfig 中 DSA/MTP 物理 plane 的稳定 group id。"""

    indexer: int
    resident_mla: int
    mtp_full: int | None = None


@dataclass
class DSAKVCacheConfig(KVCacheConfig):
    """携带 scheduler 统一 DRAM 容量合同的 DSA KV-cache 配置。"""

    dram_capacity: DSADRAMCapacityPlan | None = None


def is_dsa_indexer_spec(spec: KVCacheSpec) -> bool:
    return isinstance(spec, DSAIndexerKVSpec)


def is_dsa_resident_mla_spec(spec: KVCacheSpec) -> bool:
    return isinstance(spec, DSAResidentMLAAttentionSpec)


def is_dsa_mtp_full_spec(spec: KVCacheSpec) -> bool:
    """MTP 使用基线原生 BF16 MLA+Indexer full-cache spec。"""

    return type(spec) is AscendMLAAttentionSpec


def _validate_mtp_full_spec(
    spec: AscendMLAAttentionSpec,
    *,
    layer_name: str,
) -> None:
    sparse_head_dim = spec.sparse_head_dim
    valid_native_dims = (
        sparse_head_dim is not None
        and len(sparse_head_dim) == 3
        and all(int(dim) > 0 for dim in sparse_head_dim)
        and sum(sparse_head_dim) == spec.head_size
    )
    if (
        spec.dtype != torch.bfloat16
        or spec.cache_sparse_sfa_c8
        or spec.cache_sparse_li_c8
        or spec.scale_dim != 0
        or spec.compress_ratio != 1
        or spec.cache_dtype_str not in (None, "auto", "bfloat16")
        or not valid_native_dims
    ):
        raise RuntimeError(
            "DSA MTP full Cache must preserve the baseline three-entry BF16 "
            "MLA+Indexer layout: "
            f"layer={layer_name}, dtype={spec.dtype}, "
            f"sparse_head_dim={sparse_head_dim}, head_size={spec.head_size}, "
            f"scale_dim={spec.scale_dim}, compress_ratio={spec.compress_ratio}, "
            f"cache_dtype_str={spec.cache_dtype_str!r}, "
            f"sfa_c8={spec.cache_sparse_sfa_c8}, "
            f"li_c8={spec.cache_sparse_li_c8}"
        )


def has_dsa_split_kv_cache_specs(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> bool:
    return any(is_dsa_indexer_spec(spec) or is_dsa_resident_mla_spec(spec) for spec in kv_cache_specs.values())


def has_dsa_split_kv_cache_groups(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> bool:
    return any(
        is_dsa_indexer_spec(group.kv_cache_spec) or is_dsa_resident_mla_spec(group.kv_cache_spec)
        for group in kv_cache_groups
    )


def get_dsa_kv_cache_group_ids(
    kv_cache_config: KVCacheConfig,
) -> DSAKVCacheGroupIds:
    """解析并校验 Indexer/resident MLA 的最终 group id。

    group id 是 ``MultiGroupBlockTable``、scheduler block IDs 与 worker
    cache tensor 之间的共同索引。运行期只解析一次并保存在 model runner，
    避免热路径依赖“Indexer 固定是 group 0”之类的隐式顺序假设。
    """

    indexer_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if is_dsa_indexer_spec(group.kv_cache_spec)
    ]
    resident_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if is_dsa_resident_mla_spec(group.kv_cache_spec)
    ]
    mtp_group_ids = [
        group_id
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        if is_dsa_mtp_full_spec(group.kv_cache_spec)
    ]
    if (
        len(indexer_group_ids) != 1
        or len(resident_group_ids) != 1
        or len(mtp_group_ids) > 1
    ):
        raise RuntimeError(
            "DSA KV-cache config must contain exactly one Indexer group and "
            "one resident MLA group, plus at most one MTP full group: "
            f"indexer_group_ids={indexer_group_ids}, "
            f"resident_group_ids={resident_group_ids}, "
            f"mtp_group_ids={mtp_group_ids}"
        )
    return DSAKVCacheGroupIds(
        indexer=indexer_group_ids[0],
        resident_mla=resident_group_ids[0],
        mtp_full=mtp_group_ids[0] if mtp_group_ids else None,
    )


def _merge_group_specs(
    layer_specs: dict[str, KVCacheSpec],
) -> KVCacheSpec:
    specs = list(layer_specs.values())
    assert specs
    merged = type(specs[0]).merge(specs)
    assert type(merged) is type(specs[0])
    return merged


def build_dsa_kv_cache_groups(
    kv_cache_specs: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """构造 target 两平面与可选 MTP BF16 full-cache group。"""

    indexer_specs = {name: spec for name, spec in kv_cache_specs.items() if is_dsa_indexer_spec(spec)}
    resident_specs = {name: spec for name, spec in kv_cache_specs.items() if is_dsa_resident_mla_spec(spec)}
    mtp_specs = {
        name: spec
        for name, spec in kv_cache_specs.items()
        if is_dsa_mtp_full_spec(spec)
    }
    foreign_specs = {
        name: spec
        for name, spec in kv_cache_specs.items()
        if (
            not is_dsa_indexer_spec(spec)
            and not is_dsa_resident_mla_spec(spec)
            and not is_dsa_mtp_full_spec(spec)
        )
    }

    if not indexer_specs or not resident_specs or foreign_specs:
        raise RuntimeError(
            "DSA split KV-cache requires exactly one Indexer plane and one "
            "resident MLA plane, optionally followed by one native MTP "
            "full-cache plane: "
            f"indexer_layers={len(indexer_specs)}, "
            f"resident_layers={len(resident_specs)}, "
            f"mtp_layers={len(mtp_specs)}, "
            f"foreign_specs={tuple(sorted(type(spec).__name__ for spec in foreign_specs.values()))}"
        )
    if len(indexer_specs) > len(resident_specs):
        raise RuntimeError(
            "DSA split KV-cache cannot host more Indexer caches than resident "
            f"MLA layers: indexer_layers={len(indexer_specs)}, "
            f"resident_layers={len(resident_specs)}"
        )

    # GLM-5.2 共享 indexer 拓扑下 indexer 层是 resident 层的真子集（仅 full
    # 层建 indexer）。无论两个 plane 的层数是否相同，都要校验 Indexer
    # 下标是 resident 下标的子集，避免等基数但错位的映射被静默接受。
    from vllm.model_executor.models.utils import extract_layer_index

    resident_indices = {extract_layer_index(name) for name in resident_specs}
    orphan_indexer = sorted(name for name in indexer_specs if extract_layer_index(name) not in resident_indices)
    if orphan_indexer:
        raise RuntimeError(
            f"DSA Indexer cache has no matching resident MLA layer: orphan_indexer_layers={tuple(orphan_indexer)}"
        )

    for layer_name, spec in mtp_specs.items():
        _validate_mtp_full_spec(spec, layer_name=layer_name)

    groups = [
        KVCacheGroupSpec(
            layer_names=list(indexer_specs),
            kv_cache_spec=_merge_group_specs(indexer_specs),
        ),
        KVCacheGroupSpec(
            layer_names=list(resident_specs),
            kv_cache_spec=_merge_group_specs(resident_specs),
        ),
    ]
    if mtp_specs:
        groups.append(
            KVCacheGroupSpec(
                layer_names=list(mtp_specs),
                kv_cache_spec=_merge_group_specs(mtp_specs),
                is_eagle_group=True,
            )
        )
    return groups


def _get_dsa_ratio() -> int:
    from vllm_ascend.ascend_config import get_ascend_config

    ratio = int(get_ascend_config().dsa_offload_config.indexer_mla_block_ratio)
    if ratio <= 0:
        raise ValueError(f"DSA indexer_mla_block_ratio must be positive, got {ratio}")
    return ratio


def dsa_group_block_multiplier(group: KVCacheGroupSpec) -> int:
    """返回相对 resident base-N 的物理 block 倍数。"""

    if is_dsa_resident_mla_spec(group.kv_cache_spec):
        return 1
    if (
        is_dsa_indexer_spec(group.kv_cache_spec)
        or is_dsa_mtp_full_spec(group.kv_cache_spec)
    ):
        return _get_dsa_ratio()
    raise TypeError(
        "Unexpected KV-cache spec in DSA layout: "
        f"{type(group.kv_cache_spec).__name__}"
    )


def dsa_pool_bytes_per_base_block(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """返回一个 MLA base block 对应的总物理字节数。"""

    total = 0
    for group in kv_cache_groups:
        weight = dsa_group_block_multiplier(group)
        total += group.kv_cache_spec.page_size_bytes * len(group.layer_names) * weight
    if total <= 0:
        raise ValueError("DSA KV-cache groups contain no physical layers")
    return total


def build_dsa_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> DSAKVCacheConfig:
    """为 worker 构造 ratio 解耦的最终物理 tensor 布局。"""

    bytes_per_base_block = dsa_pool_bytes_per_base_block(kv_cache_groups)
    num_base_blocks = available_memory // bytes_per_base_block
    num_base_blocks = num_base_blocks // DSA_KV_BLOCK_COUNT_ALIGNMENT * DSA_KV_BLOCK_COUNT_ALIGNMENT
    num_base_blocks = _may_override_num_blocks(
        vllm_config,
        num_base_blocks,
    )
    if num_base_blocks <= 0:
        raise ValueError(
            "No KV-cache blocks fit in the DSA split layout: "
            f"available_memory={available_memory}, "
            f"bytes_per_base_block={bytes_per_base_block}"
        )

    tensors: list[KVCacheTensor] = []
    for group in kv_cache_groups:
        group_num_blocks = (
            num_base_blocks * dsa_group_block_multiplier(group)
        )
        for layer_name in group.layer_names:
            tensors.append(
                KVCacheTensor(
                    size=(group.kv_cache_spec.page_size_bytes * group_num_blocks),
                    shared_by=[layer_name],
                )
            )

    config = DSAKVCacheConfig(
        # vLLM v0.23 仍保留一个标量 num_blocks。DSA 将它定义为 MLA
        # base capacity；Indexer 的 ratio 容量只由最终 tensor size 表达。
        num_blocks=num_base_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=kv_cache_groups,
    )
    validate_dsa_kv_cache_config(config)
    return config


def dsa_dram_bytes_per_logical_block(
    kv_cache_groups: list[KVCacheGroupSpec],
    *,
    include_mtp: bool = True,
) -> int:
    """返回一个完整 DSA durable bundle 的每-rank 字节数。

    no-MTP bundle 由 target dense Indexer+MLA 组成；``include_mtp`` 时额外
    包含 proposer 原生 BF16 full-cache。MTP 仍不会参与 sparse decode 的
    token-wise swap；只有 prefix caching 或 L3 开启时，已验证满块才与 target
    payload 共用一个 L2 logical block。
    """

    bundle_groups = [
        group
        for group in kv_cache_groups
        if (
            is_dsa_indexer_spec(group.kv_cache_spec)
            or is_dsa_resident_mla_spec(group.kv_cache_spec)
            or (
                include_mtp
                and is_dsa_mtp_full_spec(group.kv_cache_spec)
            )
        )
    ]
    indexer_groups = [
        group
        for group in bundle_groups
        if is_dsa_indexer_spec(group.kv_cache_spec)
    ]
    resident_groups = [
        group
        for group in bundle_groups
        if is_dsa_resident_mla_spec(group.kv_cache_spec)
    ]
    mtp_groups = [
        group
        for group in bundle_groups
        if is_dsa_mtp_full_spec(group.kv_cache_spec)
    ]
    if len(indexer_groups) != 1 or len(resident_groups) != 1:
        raise RuntimeError(
            "DSA DRAM sizing requires exactly one target Indexer group and "
            "one target resident MLA group"
        )
    if len(mtp_groups) > 1:
        raise RuntimeError(
            "DSA DRAM sizing supports at most one MTP full-cache group"
        )
    indexer_block_size = int(indexer_groups[0].kv_cache_spec.block_size)
    resident_block_size = int(resident_groups[0].kv_cache_spec.block_size)
    if indexer_block_size != resident_block_size:
        raise RuntimeError(
            "DSA complete DRAM blocks require matching target Indexer/MLA "
            "block sizes: "
            f"indexer={indexer_block_size}, resident={resident_block_size}"
        )
    if mtp_groups:
        mtp_block_size = int(mtp_groups[0].kv_cache_spec.block_size)
        if mtp_block_size != indexer_block_size:
            raise RuntimeError(
                "DSA complete DRAM blocks require matching target/MTP block "
                "sizes: "
                f"target={indexer_block_size}, mtp={mtp_block_size}"
            )
    total = sum(
        int(group.kv_cache_spec.page_size_bytes)
        * len(group.layer_names)
        for group in bundle_groups
    )
    if total <= 0:
        raise RuntimeError("DSA complete DRAM block contains no payload")
    return total


def dsa_mtp_durable_offload_enabled(vllm_config: VllmConfig) -> bool:
    """Whether verified MTP full blocks need an L2/L3 durable mirror."""

    if bool(vllm_config.cache_config.enable_prefix_caching):
        return True
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    return bool(
        kv_transfer_config is not None
        and kv_transfer_config.kv_connector == "DSAL3Connector"
    )


def _dsa_l3_host_reservation_bytes(
    vllm_config: VllmConfig,
    tensor_parallel_size: int,
) -> int:
    """Return host bytes created after the DSA L2 capacity plan."""

    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    if (
        kv_transfer_config is None
        or kv_transfer_config.kv_connector != "DSAL3Connector"
    ):
        return 0
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
        MooncakeStoreConfig,
    )

    config = MooncakeStoreConfig.load_from_env()
    if config.protocol != "tcp":
        return 0
    global_segment_size = int(config.global_segment_size)
    local_buffer_size = int(config.local_buffer_size)
    if global_segment_size < 0 or local_buffer_size < 0:
        raise ValueError("Mooncake host reservation cannot be negative")
    per_rank = global_segment_size + local_buffer_size
    return per_rank * int(tensor_parallel_size)


def finalize_dsa_dram_capacity(
    vllm_config: VllmConfig,
    kv_cache_configs: list[KVCacheConfig],
    *,
    host_available_bytes: int | None = None,
) -> DSADRAMCapacityPlan | None:
    """在 EngineCore/scheduler 进程中为所有本机 TP worker 固化同一计划。"""

    dsa_configs = [
        config
        for config in kv_cache_configs
        if has_dsa_split_kv_cache_groups(config.kv_cache_groups)
    ]
    if not dsa_configs:
        return None
    if len(dsa_configs) != len(kv_cache_configs):
        raise RuntimeError(
            "DSA KV-cache layout cannot be mixed with non-DSA workers in "
            "one executor"
        )
    if not all(isinstance(config, DSAKVCacheConfig) for config in dsa_configs):
        raise RuntimeError(
            "DSA KV-cache configs lost their typed DRAM capacity field"
        )

    tp_size = int(vllm_config.parallel_config.tensor_parallel_size)
    if len(dsa_configs) != tp_size:
        raise RuntimeError(
            "Initial DSA scheduler-owned DRAM planning requires one local "
            "worker per TP rank: "
            f"worker_configs={len(dsa_configs)}, tp_size={tp_size}"
        )

    include_mtp = dsa_mtp_durable_offload_enabled(vllm_config)
    bundle_sizes = {
        dsa_dram_bytes_per_logical_block(
            config.kv_cache_groups,
            include_mtp=include_mtp,
        )
        for config in dsa_configs
    }
    if len(bundle_sizes) != 1:
        raise RuntimeError(
            "DSA TP workers have different complete DRAM block sizes: "
            f"sizes={sorted(bundle_sizes)}"
        )
    indexer_block_counts: set[int] = set()
    for config in dsa_configs:
        group_ids = get_dsa_kv_cache_group_ids(config)
        indexer_block_counts.add(
            get_dsa_group_num_blocks(
                config,
                config.kv_cache_groups[group_ids.indexer],
            )
        )
    if len(indexer_block_counts) != 1:
        raise RuntimeError(
            "DSA TP workers have different HBM Indexer capacities: "
            f"blocks={sorted(indexer_block_counts)}"
        )
    bundle_bytes = bundle_sizes.pop()
    indexer_blocks = indexer_block_counts.pop()

    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    compatibility_signature = build_dsa_dp_compatibility_signature(
        vllm_config,
        dsa_configs[0].kv_cache_groups,
        ascend_config.dsa_offload_config,
    )
    observed_available = (
        get_host_available_memory_bytes()
        if host_available_bytes is None
        else int(host_available_bytes)
    )
    parallel_config = vllm_config.parallel_config
    memory_budget = coordinate_dsa_node_memory_budget(
        data_parallel_size=int(
            getattr(parallel_config, "data_parallel_size", 1)
        ),
        data_parallel_rank=int(
            getattr(parallel_config, "data_parallel_rank", 0)
        ),
        observed_available_bytes=observed_available,
        l1_indexer_blocks=indexer_blocks,
        complete_bundle_bytes=bundle_bytes,
        tensor_parallel_size=tp_size,
        compatibility_signature=compatibility_signature,
    )
    l3_host_reservation = _dsa_l3_host_reservation_bytes(
        vllm_config,
        tp_size,
    )
    planning_available = (
        memory_budget.per_dp_quota_bytes - l3_host_reservation
    )
    if planning_available <= 0:
        raise RuntimeError(
            "DSA per-DP host DRAM quota is exhausted by the configured L3 "
            "client pools: "
            f"per_dp_quota={memory_budget.per_dp_quota_bytes}, "
            f"l3_reservation={l3_host_reservation}"
        )
    plan = calculate_dram_capacity_plan(
        indexer_num_blocks=indexer_blocks,
        block_multiple=(
            ascend_config.dsa_offload_config.hot_cpu_block_multiple
        ),
        bytes_per_logical_block=bundle_bytes,
        tensor_parallel_size=tp_size,
        host_available_bytes=planning_available,
    )
    plan = replace(
        plan,
        node_scope_id=memory_budget.node_scope_id,
        snapshot_id=memory_budget.snapshot_id,
        compatibility_signature=(
            memory_budget.compatibility_signature
        ),
        data_parallel_rank=memory_budget.data_parallel_rank,
        data_parallel_size=memory_budget.data_parallel_size,
        node_observed_available_bytes=(
            memory_budget.node_available_bytes
        ),
        per_dp_quota_bytes=memory_budget.per_dp_quota_bytes,
        observed_available_bytes_by_dp=(
            memory_budget.observed_available_bytes_by_dp
        ),
        l1_indexer_blocks_by_dp=(
            memory_budget.l1_indexer_blocks_by_dp
        ),
        minimum_per_dp_quota_bytes=(
            memory_budget.minimum_per_dp_quota_bytes
        ),
    )
    for config in dsa_configs:
        assert isinstance(config, DSAKVCacheConfig)
        config.dram_capacity = plan

    if plan.was_downsized:
        logger.warning_once(
            "DSA host DRAM cannot satisfy hot_cpu_block_multiple; using "
            "the largest complete durable-bundle pool that fits: "
            "desired_blocks=%d, selected_blocks=%d, effective_multiple=%.3f",
            plan.desired_usable_blocks,
            plan.usable_blocks,
            plan.effective_block_multiple,
        )
    elif plan.desired_usable_blocks < plan.minimum_usable_blocks:
        logger.warning_once(
            "DSA hot_cpu_block_multiple requested less than one complete "
            "HBM Indexer mirror; clamping DRAM usable blocks from %d to %d",
            plan.desired_usable_blocks,
            plan.usable_blocks,
        )

    mtp_bytes_per_block = sum(
        int(group.kv_cache_spec.page_size_bytes) * len(group.layer_names)
        for group in dsa_configs[0].kv_cache_groups
        if include_mtp and is_dsa_mtp_full_spec(group.kv_cache_spec)
    )
    bundle_description = (
        "target Indexer + MLA + MTP BF16"
        if mtp_bytes_per_block
        else "target Indexer + MLA"
    )
    _log_dsa_capacity_report(
        vllm_config,
        "\n"
        "================ DSA DRAM CAPACITY REPORT ====================\n"
        "  DP rank / size            : %s / %s\n"
        "  Node scope / snapshot     : %s / %s\n"
        "  Compatibility signature   : %s\n"
        "  Host observations by DP   : %s\n"
        "  L1 Indexer blocks by DP   : %s\n"
        "  Node conservative bytes   : %s bytes (%s GiB)\n"
        "  This-DP byte quota        : %s bytes (%s GiB)\n"
        "  Minimum common DP quota   : %s bytes (%s GiB)\n"
        "  L3 host reservation       : %s bytes (%s GiB)\n"
        "  This-DP planning bytes    : %s bytes (%s GiB)\n"
        "  Local TP workers          : %s\n"
        "  Per-rank available bytes  : %s bytes (%s GiB)\n"
        "  Complete block payload    : %s bytes (%s)\n"
        "  MTP durable payload       : %s\n"
        "  Minimum usable blocks     : %s (per-rank HBM Indexer mirror)\n"
        "  Desired usable blocks     : %s\n"
        "  Selected usable blocks    : %s (effective multiple %.3fx)\n"
        "  Per-rank reserved bytes   : %s bytes (%s GiB, includes block 0)\n"
        "  This-DP reserved bytes    : %s bytes (%s GiB)\n"
        "  Capacity downsized        : %s\n"
        "  Prefix cache              : %s\n"
        "=================================================================",
        f"{plan.data_parallel_rank:,}",
        f"{plan.data_parallel_size:,}",
        plan.node_scope_id,
        plan.snapshot_id,
        plan.compatibility_signature,
        tuple(
            f"{value:,}"
            for value in plan.observed_available_bytes_by_dp
        ),
        tuple(f"{value:,}" for value in plan.l1_indexer_blocks_by_dp),
        f"{plan.node_observed_available_bytes:,}",
        format_gib(plan.node_observed_available_bytes),
        f"{plan.per_dp_quota_bytes:,}",
        format_gib(plan.per_dp_quota_bytes),
        f"{plan.minimum_per_dp_quota_bytes:,}",
        format_gib(plan.minimum_per_dp_quota_bytes),
        f"{l3_host_reservation:,}",
        format_gib(l3_host_reservation),
        f"{plan.host_available_bytes:,}",
        format_gib(plan.host_available_bytes),
        f"{plan.tensor_parallel_size:,}",
        f"{plan.per_rank_available_bytes:,}",
        format_gib(plan.per_rank_available_bytes),
        f"{plan.bytes_per_logical_block:,}",
        bundle_description,
        (
            f"{mtp_bytes_per_block:,} bytes/block"
            if mtp_bytes_per_block
            else "disabled"
        ),
        f"{plan.minimum_usable_blocks:,}",
        f"{plan.desired_usable_blocks:,}",
        f"{plan.usable_blocks:,}",
        plan.effective_block_multiple,
        f"{plan.per_rank_reserved_bytes:,}",
        format_gib(plan.per_rank_reserved_bytes),
        f"{plan.total_reserved_bytes:,}",
        format_gib(plan.total_reserved_bytes),
        plan.was_downsized,
        (
            "enabled (complete durable bundle)"
            if vllm_config.cache_config.enable_prefix_caching
            else "disabled"
        ),
    )
    return plan


def dsa_max_memory_usage_bytes(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """计算至少容纳一个完整 prefill 请求所需的物理字节数。"""

    resident_groups = [group for group in kv_cache_groups if is_dsa_resident_mla_spec(group.kv_cache_spec)]
    if len(resident_groups) != 1:
        raise RuntimeError("DSA memory admission requires exactly one resident MLA group")
    resident_group = resident_groups[0]
    resident_page_size = resident_group.kv_cache_spec.page_size_bytes
    resident_bytes_per_layer = resident_group.kv_cache_spec.max_memory_usage_bytes(vllm_config)
    if resident_bytes_per_layer % resident_page_size != 0:
        raise RuntimeError("DSA resident max-memory requirement is not page aligned")

    # 一个完整 prefill 请求要求 MLA base plane 至少容纳完整上下文。
    # Indexer plane 虽有 ratio 倍容量，但不能拿多出来的 Indexer blocks
    # 抵偿缺失的 MLA blocks，因此 admission 必须按加权 base-block 成本
    # 检查，而不是简单相加两个 plane 各自的一份 max-memory。
    required_base_blocks = resident_bytes_per_layer // resident_page_size
    return required_base_blocks * dsa_pool_bytes_per_base_block(kv_cache_groups)


def _layer_tensor_sizes(
    kv_cache_config: KVCacheConfig,
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        for layer_name in tensor.shared_by:
            if layer_name in sizes:
                raise RuntimeError(f"DSA layer is backed by more than one KVCacheTensor: {layer_name}")
            sizes[layer_name] = int(tensor.size)
    return sizes


def get_dsa_group_num_blocks(
    kv_cache_config: KVCacheConfig,
    group: KVCacheGroupSpec,
) -> int:
    """从最终 tensor 大小反推 group 的真实物理 block 数。"""

    if not group.layer_names:
        return 0
    tensor_sizes = _layer_tensor_sizes(kv_cache_config)
    page_size = int(group.kv_cache_spec.page_size_bytes)
    block_counts: set[int] = set()
    for layer_name in group.layer_names:
        try:
            tensor_size = tensor_sizes[layer_name]
        except KeyError as exc:
            raise RuntimeError(f"DSA KV-cache tensor is missing for layer {layer_name}") from exc
        if tensor_size % page_size != 0:
            raise RuntimeError(
                "DSA KV-cache tensor is not page aligned: "
                f"layer={layer_name}, tensor_size={tensor_size}, "
                f"page_size={page_size}"
            )
        block_counts.add(tensor_size // page_size)
    if len(block_counts) != 1:
        raise RuntimeError(
            "Layers in one DSA KV-cache group have different capacities: "
            f"group_spec={type(group.kv_cache_spec).__name__}, "
            f"block_counts={sorted(block_counts)}"
        )
    return block_counts.pop()


def validate_dsa_kv_cache_config(
    kv_cache_config: KVCacheConfig,
) -> None:
    """校验最终物理 tensor 是否满足 target/MTP 容量契约。"""

    indexer_groups = [group for group in kv_cache_config.kv_cache_groups if is_dsa_indexer_spec(group.kv_cache_spec)]
    resident_groups = [
        group for group in kv_cache_config.kv_cache_groups if is_dsa_resident_mla_spec(group.kv_cache_spec)
    ]
    mtp_groups = [
        group
        for group in kv_cache_config.kv_cache_groups
        if is_dsa_mtp_full_spec(group.kv_cache_spec)
    ]
    if (
        len(indexer_groups) != 1
        or len(resident_groups) != 1
        or len(mtp_groups) > 1
    ):
        raise RuntimeError(
            "DSA KV-cache config must contain exactly one Indexer group and "
            "one resident MLA group, plus at most one MTP full group: "
            f"indexer={len(indexer_groups)}, "
            f"resident={len(resident_groups)}, mtp={len(mtp_groups)}"
        )

    indexer_blocks = get_dsa_group_num_blocks(kv_cache_config, indexer_groups[0])
    resident_blocks = get_dsa_group_num_blocks(kv_cache_config, resident_groups[0])
    ratio = _get_dsa_ratio()
    if resident_blocks != int(kv_cache_config.num_blocks):
        raise RuntimeError(
            "DSA resident MLA capacity must equal KVCacheConfig.num_blocks: "
            f"resident={resident_blocks}, "
            f"num_blocks={kv_cache_config.num_blocks}"
        )
    if indexer_blocks != resident_blocks * ratio:
        raise RuntimeError(
            "DSA Indexer/MLA capacity ratio mismatch: "
            f"indexer={indexer_blocks}, resident={resident_blocks}, "
            f"ratio={ratio}"
        )
    if mtp_groups:
        mtp_group = mtp_groups[0]
        _validate_mtp_full_spec(
            mtp_group.kv_cache_spec,
            layer_name=",".join(mtp_group.layer_names),
        )
        mtp_blocks = get_dsa_group_num_blocks(
            kv_cache_config,
            mtp_group,
        )
        if not mtp_group.is_eagle_group:
            raise RuntimeError(
                "DSA MTP full Cache group must retain is_eagle_group=True"
            )
        if mtp_blocks != resident_blocks * ratio:
            raise RuntimeError(
                "DSA MTP full/MLA capacity ratio mismatch: "
                f"mtp={mtp_blocks}, resident={resident_blocks}, "
                f"ratio={ratio}"
            )
        indexer_token_capacity = (
            indexer_blocks * indexer_groups[0].kv_cache_spec.block_size
        )
        mtp_token_capacity = mtp_blocks * mtp_group.kv_cache_spec.block_size
        if mtp_token_capacity != indexer_token_capacity:
            raise RuntimeError(
                "DSA MTP full cache must cover the same token range as the "
                "target Indexer: "
                f"mtp_tokens={mtp_token_capacity}, "
                f"indexer_tokens={indexer_token_capacity}"
            )


def get_dsa_kv_cache_binding_order(
    kv_cache_config: KVCacheConfig,
) -> list[str]:
    """返回 target 同层双 plane cache 的稳定绑定顺序。

    vLLM 的通用 ``bind_kv_cache`` 会把 cache 名先折叠为 transformer layer
    index；非 CUDA/XPU/CPU 平台若同一 layer index 对应多个 cache 名会直接
    拒绝。DSA 的 attention resident cache 与 Indexer cache 恰好必须共享
    transformer layer index，因此由 NPU model runner 使用此顺序逐个绑定。

    顺序定义为 layer index 优先、同层 resident MLA 在前、Indexer 在后。
    ``runner_kv_caches`` 当前只承担引用持有与清理职责，但仍固定顺序，避免
    后续消费者把字典插入顺序误当作 ABI。
    """

    # 延迟导入，避免 cache spec 注册阶段为了一个纯绑定辅助函数提前载入
    # 整个 model-executor utilities 模块。
    from vllm.model_executor.models.utils import extract_layer_index

    ordered_layers: list[tuple[int, int, str]] = []
    seen_layers: set[str] = set()
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if is_dsa_resident_mla_spec(spec):
            plane_order = 0
        elif is_dsa_indexer_spec(spec):
            plane_order = 1
        elif is_dsa_mtp_full_spec(spec):
            # MTP full Cache 由 vLLM-Ascend 基线通用绑定路径负责。
            continue
        else:
            raise RuntimeError(f"Unexpected KV-cache group in DSA binding: {type(spec).__name__}")
        for layer_name in group.layer_names:
            if layer_name in seen_layers:
                raise RuntimeError(f"DSA KV-cache layer appears in more than one group: {layer_name}")
            seen_layers.add(layer_name)
            ordered_layers.append(
                (
                    extract_layer_index(layer_name),
                    plane_order,
                    layer_name,
                )
            )

    if not ordered_layers:
        raise RuntimeError("DSA KV-cache binding received no cache layers")
    return [layer_name for _, _, layer_name in sorted(ordered_layers)]


def report_dsa_kv_cache_config(
    vllm_config: VllmConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    """按最终 tensor 大小输出一次确定性的 DSA HBM 容量报告。"""

    validate_dsa_kv_cache_config(kv_cache_config)
    indexer_group = next(group for group in kv_cache_config.kv_cache_groups if is_dsa_indexer_spec(group.kv_cache_spec))
    resident_group = next(
        group for group in kv_cache_config.kv_cache_groups if is_dsa_resident_mla_spec(group.kv_cache_spec)
    )
    mtp_group = next(
        (
            group
            for group in kv_cache_config.kv_cache_groups
            if is_dsa_mtp_full_spec(group.kv_cache_spec)
        ),
        None,
    )
    indexer_blocks = get_dsa_group_num_blocks(kv_cache_config, indexer_group)
    resident_blocks = get_dsa_group_num_blocks(kv_cache_config, resident_group)
    mtp_blocks = (
        get_dsa_group_num_blocks(kv_cache_config, mtp_group)
        if mtp_group is not None
        else 0
    )
    indexer_tokens = indexer_blocks * indexer_group.kv_cache_spec.block_size
    resident_tokens = resident_blocks * resident_group.kv_cache_spec.block_size
    mtp_tokens = (
        mtp_blocks * mtp_group.kv_cache_spec.block_size
        if mtp_group is not None
        else 0
    )
    max_model_len = int(vllm_config.model_config.max_model_len)
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    block_size = int(resident_group.kv_cache_spec.block_size)

    from vllm_ascend.ascend_config import get_ascend_config

    dsa_config = get_ascend_config().dsa_offload_config
    actual_mtp_layer_count = (
        len(mtp_group.layer_names) if mtp_group is not None else 0
    )
    if dsa_config.mtp_enabled != (mtp_group is not None) or (
        dsa_config.mtp_enabled
        and actual_mtp_layer_count != dsa_config.mtp_cache_layer_count
    ):
        raise RuntimeError(
            "DSA capacity report found an MTP config/cache-group mismatch: "
            f"mtp_enabled={dsa_config.mtp_enabled}, "
            f"configured_layers={dsa_config.mtp_cache_layer_count}, "
            f"physical_layers={actual_mtp_layer_count}"
        )
    resident_slots_per_request = (
        cdiv(dsa_config.max_resident_budget_tokens, block_size)
        * block_size
        + dsa_config.sparse_tail_block_count * block_size
    )
    decode_by_mla = resident_tokens // resident_slots_per_request
    indexer_blocks_per_request = cdiv(max_model_len, indexer_group.kv_cache_spec.block_size)
    decode_by_indexer = indexer_blocks // indexer_blocks_per_request
    decode_by_mtp = (
        mtp_blocks
        // cdiv(max_model_len, mtp_group.kv_cache_spec.block_size)
        if mtp_group is not None
        else max_num_seqs
    )
    configured_decode_limit = min(
        decode_by_mla,
        decode_by_indexer,
        decode_by_mtp,
        max_num_seqs,
    )
    dense_token_limits = [resident_tokens, indexer_tokens]
    if mtp_group is not None:
        dense_token_limits.append(mtp_tokens)
    total_bytes = sum(int(tensor.size) for tensor in kv_cache_config.kv_cache_tensors)
    layer_tensor_sizes = _layer_tensor_sizes(kv_cache_config)
    mtp_bytes = (
        sum(layer_tensor_sizes[layer_name] for layer_name in mtp_group.layer_names)
        if mtp_group is not None
        else 0
    )
    mtp_hbm_percent = (
        100.0 * mtp_bytes / total_bytes
        if total_bytes > 0
        else 0.0
    )
    if mtp_group is None:
        mtp_plane_summary = "disabled"
        mtp_bytes_summary = "disabled"
        mtp_limit_summary = "disabled"
    else:
        mtp_plane_summary = (
            f"{mtp_tokens:,} tokens "
            f"({mtp_blocks:,} blocks x "
            f"{mtp_group.kv_cache_spec.block_size:,} tokens)"
        )
        mtp_bytes_summary = (
            f"{mtp_bytes:,} bytes ({format_gib(mtp_bytes)} GiB, "
            f"{mtp_hbm_percent:.2f}% of KV HBM)"
        )
        mtp_limit_summary = (
            f"{decode_by_mtp:,} requests at max_model_len={max_model_len:,}"
        )

    _log_dsa_capacity_report(
        vllm_config,
        "\n"
        "================ DSA HBM CACHE CAPACITY REPORT ================\n"
        "  Split ratio             : indexer:mla = %d:1; base blocks = %s\n"
        "  Physical cache layers   : indexer=%s, resident=%s, mtp=%s\n"
        "  Allocated HBM KV bytes  : %s bytes (%s GiB)\n"
        "  MLA resident plane      : %s tokens (%s blocks x %s tokens)\n"
        "  Indexer dense plane     : %s tokens (%s blocks x %s tokens)\n"
        "  MTP full BF16 plane     : %s\n"
        "  MTP full BF16 bytes     : %s\n"
        "  Batched prefill limit   : %s tokens (all physical planes)\n"
        "  Sparse decode MLA limit : %s requests (%s resident slots/request)\n"
        "  Dense Indexer limit     : %s requests at max_model_len=%s\n"
        "  MTP full-cache limit    : %s\n"
        "  Configured decode limit : %s requests (max_num_seqs=%s)\n"
        "=================================================================",
        _get_dsa_ratio(),
        f"{resident_blocks:,}",
        f"{len(indexer_group.layer_names):,}",
        f"{len(resident_group.layer_names):,}",
        f"{actual_mtp_layer_count:,}",
        f"{total_bytes:,}",
        format_gib(total_bytes),
        f"{resident_tokens:,}",
        f"{resident_blocks:,}",
        f"{block_size:,}",
        f"{indexer_tokens:,}",
        f"{indexer_blocks:,}",
        f"{indexer_group.kv_cache_spec.block_size:,}",
        mtp_plane_summary,
        mtp_bytes_summary,
        f"{min(dense_token_limits):,}",
        f"{decode_by_mla:,}",
        f"{resident_slots_per_request:,}",
        f"{decode_by_indexer:,}",
        f"{max_model_len:,}",
        mtp_limit_summary,
        f"{configured_decode_limit:,}",
        f"{max_num_seqs:,}",
    )
