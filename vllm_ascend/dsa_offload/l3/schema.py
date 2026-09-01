# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA L3 component schema 与内容寻址 key。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.dsa_offload.kv_cache import (
    is_dsa_indexer_spec,
    is_dsa_mtp_full_spec,
    is_dsa_resident_mla_spec,
)


class DSAL3Component(StrEnum):
    INDEXER = "indexer"
    MLA = "mla"
    MTP = "mtp"


def _model_identity(vllm_config: Any) -> dict[str, Any]:
    model_config = vllm_config.model_config
    hf_config = model_config.hf_config
    compute_hash = getattr(model_config, "compute_hash", None)
    model_config_hash = (
        str(compute_hash()) if callable(compute_hash) else ""
    )
    parallel_config = getattr(vllm_config, "parallel_config", None)
    return {
        "model": str(getattr(model_config, "model", "")),
        "revision": str(getattr(model_config, "revision", "") or ""),
        "architectures": tuple(getattr(hf_config, "architectures", ()) or ()),
        "model_type": str(getattr(hf_config, "model_type", "")),
        "quantization": str(getattr(model_config, "quantization", "") or ""),
        "model_config_hash": model_config_hash,
        "tensor_parallel_size": int(
            getattr(parallel_config, "tensor_parallel_size", 1)
        ),
    }


def _cache_spec_identity(spec: Any) -> dict[str, Any]:
    """Return stable cache semantics not implied by page size alone."""

    sparse_head_dim = getattr(spec, "sparse_head_dim", None)
    return {
        "type": type(spec).__qualname__,
        "dtype": str(getattr(spec, "dtype", "")),
        "num_kv_heads": int(getattr(spec, "num_kv_heads", 0)),
        "head_size": int(getattr(spec, "head_size", 0)),
        "sparse_head_dim": (
            tuple(int(value) for value in sparse_head_dim)
            if sparse_head_dim is not None
            else None
        ),
        "scale_dim": int(getattr(spec, "scale_dim", 0)),
        "scale_dtype": str(getattr(spec, "scale_dtype", "")),
        "cache_sparse_sfa_c8": bool(
            getattr(spec, "cache_sparse_sfa_c8", False)
        ),
        "cache_sparse_li_c8": bool(
            getattr(spec, "cache_sparse_li_c8", False)
        ),
        "c8_k_cache_dtype": str(
            getattr(spec, "c8_k_cache_dtype", "")
        ),
        "c8_k_scale_cache_dtype": str(
            getattr(spec, "c8_k_scale_cache_dtype", "")
        ),
        "sfa_dcp_replicated_indexer_size": int(
            getattr(spec, "sfa_dcp_replicated_indexer_size", 1)
        ),
    }


@dataclass(frozen=True, slots=True)
class DSAL3KeySchema:
    """scheduler/worker 共用的远端对象 namespace。"""

    fingerprint: str
    block_size: int
    indexer_layer_names: tuple[str, ...]
    resident_layer_names: tuple[str, ...]
    indexer_plane_count: int
    mla_plane_count: int
    indexer_bytes_per_block: int
    mla_bytes_per_block: int
    mtp_layer_names: tuple[str, ...] = ()
    mtp_plane_count: int = 0
    mtp_bytes_per_block: int = 0

    @classmethod
    def from_config(
        cls,
        vllm_config: Any,
        kv_cache_config: KVCacheConfig,
    ) -> DSAL3KeySchema:
        indexer_groups = [
            group for group in kv_cache_config.kv_cache_groups if is_dsa_indexer_spec(group.kv_cache_spec)
        ]
        resident_groups = [
            group for group in kv_cache_config.kv_cache_groups if is_dsa_resident_mla_spec(group.kv_cache_spec)
        ]
        mtp_groups = [
            group
            for group in kv_cache_config.kv_cache_groups
            if is_dsa_mtp_full_spec(group.kv_cache_spec)
        ]
        if len(indexer_groups) != 1 or len(resident_groups) != 1:
            raise ValueError("DSA L3 schema requires exactly one Indexer and one resident MLA cache group")
        if len(mtp_groups) > 1:
            raise ValueError(
                "DSA L3 schema supports at most one MTP full-cache group"
            )
        indexer_group = indexer_groups[0]
        resident_group = resident_groups[0]
        mtp_group = mtp_groups[0] if mtp_groups else None
        indexer_spec = indexer_group.kv_cache_spec
        resident_spec = resident_group.kv_cache_spec
        block_size = int(indexer_spec.block_size)
        if block_size != int(resident_spec.block_size):
            raise ValueError(
                f"DSA L3 Indexer/MLA block sizes disagree: indexer={block_size}, resident={resident_spec.block_size}"
            )
        indexer_layers = tuple(indexer_group.layer_names)
        resident_layers = tuple(resident_group.layer_names)
        indexer_bytes = int(indexer_spec.page_size_bytes) * len(indexer_layers)
        mla_bytes = int(resident_spec.page_size_bytes) * len(resident_layers)
        indexer_plane_count = len(indexer_layers) * (
            2 if bool(getattr(indexer_spec, "cache_sparse_li_c8", False)) else 1
        )
        mla_plane_count = len(resident_layers) * (
            1 if bool(getattr(resident_spec, "cache_sparse_sfa_c8", False)) else 2
        )
        mtp_layers: tuple[str, ...] = ()
        mtp_plane_count = 0
        mtp_bytes = 0
        if mtp_group is not None:
            mtp_spec = mtp_group.kv_cache_spec
            if int(mtp_spec.block_size) != block_size:
                raise ValueError(
                    "DSA L3 target/MTP block sizes disagree: "
                    f"target={block_size}, mtp={mtp_spec.block_size}"
                )
            mtp_layers = tuple(mtp_group.layer_names)
            sparse_head_dim = getattr(mtp_spec, "sparse_head_dim", None)
            if not sparse_head_dim:
                raise ValueError(
                    "DSA L3 MTP full-cache spec must expose its native planes"
                )
            mtp_plane_count = len(mtp_layers) * len(sparse_head_dim)
            mtp_bytes = int(mtp_spec.page_size_bytes) * len(mtp_layers)
        descriptor = {
            "namespace": "vllm-ascend:dsa-l3:v1",
            "model": _model_identity(vllm_config),
            "block_size": block_size,
            "components": {
                DSAL3Component.INDEXER: {
                    "layers": indexer_layers,
                    "page_size_bytes": int(indexer_spec.page_size_bytes),
                    "plane_count": indexer_plane_count,
                    "spec": _cache_spec_identity(indexer_spec),
                },
                DSAL3Component.MLA: {
                    "layers": resident_layers,
                    "page_size_bytes": int(resident_spec.page_size_bytes),
                    "plane_count": mla_plane_count,
                    "spec": _cache_spec_identity(resident_spec),
                },
            },
        }
        if mtp_group is not None:
            descriptor["components"][DSAL3Component.MTP] = {
                "layers": mtp_layers,
                "page_size_bytes": int(mtp_group.kv_cache_spec.page_size_bytes),
                "plane_count": mtp_plane_count,
                "spec": _cache_spec_identity(mtp_group.kv_cache_spec),
            }
        encoded = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(encoded).hexdigest()[:32]
        return cls(
            fingerprint=fingerprint,
            block_size=block_size,
            indexer_layer_names=indexer_layers,
            resident_layer_names=resident_layers,
            indexer_plane_count=indexer_plane_count,
            mla_plane_count=mla_plane_count,
            indexer_bytes_per_block=indexer_bytes,
            mla_bytes_per_block=mla_bytes,
            mtp_layer_names=mtp_layers,
            mtp_plane_count=mtp_plane_count,
            mtp_bytes_per_block=mtp_bytes,
        )

    @property
    def bytes_per_logical_block(self) -> int:
        return (
            self.indexer_bytes_per_block
            + self.mla_bytes_per_block
            + self.mtp_bytes_per_block
        )

    @property
    def active_components(self) -> tuple[DSAL3Component, ...]:
        components = [DSAL3Component.INDEXER, DSAL3Component.MLA]
        if self.mtp_bytes_per_block:
            components.append(DSAL3Component.MTP)
        return tuple(components)

    @property
    def has_mtp(self) -> bool:
        return bool(self.mtp_bytes_per_block)

    def component_plane_count(self, component: DSAL3Component) -> int:
        if component is DSAL3Component.INDEXER:
            return self.indexer_plane_count
        if component is DSAL3Component.MLA:
            return self.mla_plane_count
        return self.mtp_plane_count

    def component_bytes_per_block(self, component: DSAL3Component) -> int:
        if component is DSAL3Component.INDEXER:
            return self.indexer_bytes_per_block
        if component is DSAL3Component.MLA:
            return self.mla_bytes_per_block
        return self.mtp_bytes_per_block

    def component_key(
        self,
        component: DSAL3Component,
        block_hash: bytes,
    ) -> str:
        raw_hash = bytes(block_hash)
        if not raw_hash:
            raise ValueError("DSA L3 block hash cannot be empty")
        return f"vllm-ascend:dsa-l3:v1:{self.fingerprint}:{component.value}:{raw_hash.hex()}"

    def block_keys(self, block_hash: bytes) -> tuple[str, ...]:
        return tuple(
            self.component_key(component, block_hash)
            for component in self.active_components
        )
