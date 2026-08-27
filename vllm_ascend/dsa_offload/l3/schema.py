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
    is_dsa_resident_mla_spec,
)


class DSAL3Component(StrEnum):
    INDEXER = "indexer"
    MLA = "mla"


def _model_identity(vllm_config: Any) -> dict[str, Any]:
    model_config = vllm_config.model_config
    hf_config = model_config.hf_config
    return {
        "model": str(getattr(model_config, "model", "")),
        "revision": str(getattr(model_config, "revision", "") or ""),
        "architectures": tuple(getattr(hf_config, "architectures", ()) or ()),
        "model_type": str(getattr(hf_config, "model_type", "")),
        "quantization": str(getattr(model_config, "quantization", "") or ""),
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
        if len(indexer_groups) != 1 or len(resident_groups) != 1:
            raise ValueError("DSA L3 schema requires exactly one Indexer and one resident MLA cache group")
        indexer_group = indexer_groups[0]
        resident_group = resident_groups[0]
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
        descriptor = {
            "namespace": "vllm-ascend:dsa-l3:v1",
            "model": _model_identity(vllm_config),
            "block_size": block_size,
            "components": {
                DSAL3Component.INDEXER: {
                    "layers": indexer_layers,
                    "page_size_bytes": int(indexer_spec.page_size_bytes),
                    "plane_count": indexer_plane_count,
                },
                DSAL3Component.MLA: {
                    "layers": resident_layers,
                    "page_size_bytes": int(resident_spec.page_size_bytes),
                    "plane_count": mla_plane_count,
                },
            },
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
        )

    @property
    def bytes_per_logical_block(self) -> int:
        return self.indexer_bytes_per_block + self.mla_bytes_per_block

    def component_key(
        self,
        component: DSAL3Component,
        block_hash: bytes,
    ) -> str:
        raw_hash = bytes(block_hash)
        if not raw_hash:
            raise ValueError("DSA L3 block hash cannot be empty")
        return f"vllm-ascend:dsa-l3:v1:{self.fingerprint}:{component.value}:{raw_hash.hex()}"

    def block_keys(self, block_hash: bytes) -> tuple[str, str]:
        return (
            self.component_key(DSAL3Component.INDEXER, block_hash),
            self.component_key(DSAL3Component.MLA, block_hash),
        )
