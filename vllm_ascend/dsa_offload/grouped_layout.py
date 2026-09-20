# SPDX-License-Identifier: Apache-2.0
"""Opt-in IndexShare KV layout. Logical block IDs and LIM state stay unchanged.

Physical storage is [B, 128, 1, 4*656]. Native PA operators see the *same*
allocation as [4*B, 128, 1, 656]. A physical token slot s in lane l becomes
4*s+l; a logical page b expands to pages 4*b+[0,1,2,3]. Neither conversion
changes the number of valid attention entries or the causal sequence lengths.
"""

from dataclasses import dataclass

import torch

from vllm_ascend.dsa_offload.contracts import DSA_A5_PACKED_KV_ROW_BYTES, DSA_REQUIRED_CACHE_BLOCK_SIZE

GROUP_SIZE = 4


def indexshare_groups(indexer_types: tuple[str, ...] | None) -> tuple[tuple[int, ...], ...]:
    if not indexer_types:
        raise ValueError("Grouped C8 KV requires explicit full/shared indexer_types")
    groups: list[list[int]] = []
    for layer, kind in enumerate(indexer_types):
        if kind == "full":
            groups.append([layer])
        elif kind == "shared" and groups:
            groups[-1].append(layer)
        else:
            raise ValueError(f"Invalid IndexShare topology at layer {layer}: {kind!r}")
    if any(len(group) not in (1, GROUP_SIZE) for group in groups):
        raise ValueError("Grouped C8 KV supports singleton or full+3 shared IndexShare groups only")
    if not any(len(group) == GROUP_SIZE for group in groups):
        raise ValueError("Grouped C8 KV requires at least one four-layer IndexShare group")
    return tuple(tuple(group) for group in groups)


def lane_slots(slots: torch.Tensor, lane: int) -> torch.Tensor:
    if not 0 <= lane < GROUP_SIZE:
        raise ValueError(f"Invalid grouped KV lane: {lane}")
    # Keep padding sentinels negative, including lanes 1..3.
    return torch.where(slots >= 0, slots * GROUP_SIZE + lane, slots)


def expanded_block_table(block_table: torch.Tensor) -> torch.Tensor:
    offsets = torch.arange(GROUP_SIZE, dtype=block_table.dtype, device=block_table.device)
    expanded = block_table.unsqueeze(-1) * GROUP_SIZE + offsets
    expanded = torch.where(block_table.unsqueeze(-1) >= 0, expanded, block_table.unsqueeze(-1))
    return expanded.flatten(-2).contiguous()


@dataclass(frozen=True)
class GroupedResidentCache:
    layer_ids: tuple[int, ...]
    packed: torch.Tensor

    @property
    def pa_view(self) -> torch.Tensor:
        return self.packed.view(-1, DSA_REQUIRED_CACHE_BLOCK_SIZE, 1, DSA_A5_PACKED_KV_ROW_BYTES)


def allocate_grouped_caches(kv_cache_config, layer_specs, indexer_types, allocate):
    """Allocate each group once, without first allocating four throwaway caches.

    Layer specs/capacity accounting remain per-layer. Only target resident MLA
    layers participate; singleton, Indexer and draft BF16 caches are untouched.
    """
    from vllm.model_executor.models.utils import extract_layer_index

    from vllm_ascend.dsa_offload.kv_cache import DSAResidentMLAAttentionSpec

    layers = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        for name in tensor.shared_by:
            spec = layer_specs[name]
            if not isinstance(spec, DSAResidentMLAAttentionSpec):
                continue
            if len(tensor.shared_by) != 1:
                raise ValueError("Grouped C8 KV cannot alias cache tensors across attention layers")
            if spec.block_size != DSA_REQUIRED_CACHE_BLOCK_SIZE or not spec.cache_sparse_sfa_c8:
                raise ValueError("Grouped C8 KV requires packed C8 and block_size=128")
            if spec.page_size_bytes != DSA_REQUIRED_CACHE_BLOCK_SIZE * DSA_A5_PACKED_KV_ROW_BYTES:
                raise ValueError("Grouped C8 KV requires 656 bytes per token per layer")
            if tensor.size % spec.page_size_bytes:
                raise ValueError("Grouped C8 KV allocation must be page aligned")
            layer = extract_layer_index(name)
            if layer in layers:
                raise ValueError(f"Multiple resident caches for grouped layer {layer}")
            layers[layer] = (name, tensor.size)
    if set(layers) != set(range(len(indexer_types))):
        raise ValueError("Grouped C8 KV requires all target layers on the worker (no PP/cache sharing)")
    result = {}
    for group in indexshare_groups(indexer_types):
        if len(group) == 1:
            continue
        sizes = {layers[layer][1] for layer in group}
        if len(sizes) != 1:
            raise ValueError(f"IndexShare group {group} has unequal resident cache capacities")
        raw = allocate(sizes.pop() * GROUP_SIZE, 2 * 1024 * 1024)
        packed = raw.view(torch.float8_e4m3fn).view(
            -1, DSA_REQUIRED_CACHE_BLOCK_SIZE, 1, GROUP_SIZE * DSA_A5_PACKED_KV_ROW_BYTES
        )
        cache = GroupedResidentCache(group, packed)
        for layer in group:
            result[layers[layer][0]] = cache
    return result
