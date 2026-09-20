# SPDX-License-Identifier: Apache-2.0
"""Separate grouped-C8 runtime path; original per-layer contexts are unchanged."""

from dataclasses import dataclass

import torch

from vllm_ascend.dsa_offload.grouped_layout import GroupedResidentCache, expanded_block_table, lane_slots
from vllm_ascend.dsa_offload.ops import dump_full_cache_plane_bytes
from vllm_ascend.dsa_offload.runtime import DSALayerOffloadContext


@dataclass(frozen=True, kw_only=True)
class DSAGroupedLayerOffloadContext(DSALayerOffloadContext):
    group: GroupedResidentCache
    group_dram: torch.Tensor
    lane: int

    def map_cache_slots(self, slots: torch.Tensor) -> torch.Tensor:
        return lane_slots(slots, self.lane)

    def _copy_or_defer_c8(
        self,
        *,
        resident_packed_cache,
        dram_packed_arena,
        resident_block_table,
        dram_block_table,
        outputs,
        num_query_rows,
    ):
        # All lanes share the owner's LIM state and destination slots. One
        # 2624B/token copy precedes attention for all four layers. Never replay
        # it on a follower: that would overwrite newly written follower KV.
        if self.lane == 0:
            torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_group4_c8_out(
                self.group.packed.view(torch.int8),
                self.group_dram,
                resident_block_table.contiguous(),
                dram_block_table.contiguous(),
                outputs.topk_index.contiguous(),
                outputs.topk_slots.contiguous(),
                outputs.miss_count.contiguous(),
            )
        return None

    def execute_paged_attention(
        self,
        *,
        query,
        query_rope,
        resident_cache,
        sparse_indices,
        scale_value,
        block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        sparse_mode,
    ):
        import torch_npu

        result = torch_npu.npu_kv_quant_sparse_flash_attention(
            query=torch.cat([query, query_rope], dim=-1).contiguous(),
            key=resident_cache[0],
            value=resident_cache[0],
            sparse_indices=lane_slots(sparse_indices, self.lane).contiguous(),
            scale_value=float(scale_value),
            sparse_block_size=1,
            block_table=expanded_block_table(block_table),
            actual_seq_lengths_query=actual_seq_lengths_query,
            # These are entry counts/causal lengths, NOT byte addresses. Do not
            # multiply them by four when translating PA indices.
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=sparse_mode,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            key_quant_mode=2,
            value_quant_mode=2,
            rope_head_dim=64,
        )
        return result[0] if isinstance(result, tuple) else result

    def execute_sparse_attention(
        self,
        *,
        query,
        resident_cache,
        selection,
        scale_value,
        resident_block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
    ):
        if selection.fused_copy_plan is not None or selection.resident_seq_lengths is None:
            raise RuntimeError("Grouped C8 KV requires split MTP LIM selection")
        return self.execute_paged_attention(
            query=query,
            query_rope=query_rope,
            resident_cache=resident_cache,
            sparse_indices=selection.sparse_indices,
            scale_value=scale_value,
            block_table=resident_block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=selection.resident_seq_lengths,
            sparse_mode=0,
        )

    def dump_full_blocks(self, *, resident_cache, before_cache_write: bool) -> None:
        runtime = self.runtime
        # Decode: dump ALL old lanes at the owner before any lane is overwritten.
        # Prefill: wait for the LAST lane to populate this chunk before dumping
        # the group. Full Indexer planes belong to the owner in either case.
        dump_lane = 0 if before_cache_write else len(self.group.layer_ids) - 1
        if self.lane != dump_lane or not runtime.dump_launch_count:
            return
        if before_cache_write != runtime.dump_before_cache_write:
            return
        claims = [
            runtime.claim_layer_dump(layer_id=layer, before_cache_write=before_cache_write)
            for layer in self.group.layer_ids
        ]
        if not any(claims):
            return
        if not all(claims):
            raise RuntimeError("Partially dumped grouped KV would publish stale lanes")
        arenas = runtime.dram_store.get_layer_arenas(self.group.layer_ids[0])
        count = runtime.dump_launch_count
        dst = runtime.dump_dst_block_ids.gpu[:count]
        for source, dram in zip(arenas.resident_source_bytes, arenas.resident_dram_bytes, strict=True):
            dump_full_cache_plane_bytes(
                source_bytes=source,
                dram_bytes=dram,
                src_block_ids=runtime.dump_src_block_ids.gpu[:count],
                dst_block_ids=dst,
            )
        for source, dram in zip(arenas.indexer_source_bytes, arenas.indexer_dram_bytes, strict=True):
            dump_full_cache_plane_bytes(
                source_bytes=source,
                dram_bytes=dram,
                src_block_ids=runtime.dump_indexer_src_block_ids.gpu[:count],
                dst_block_ids=dst,
            )
