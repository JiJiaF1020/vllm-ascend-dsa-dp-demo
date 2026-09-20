# SPDX-License-Identifier: Apache-2.0
"""CPU address/durability contracts; real kernels are covered by the A5 ST."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.dsa_offload.grouped_runtime as grouped_runtime
from vllm_ascend.dsa_offload.grouped_layout import (
    GroupedResidentCache,
    allocate_grouped_caches,
    expanded_block_table,
    indexshare_groups,
    lane_slots,
)
from vllm_ascend.dsa_offload.grouped_runtime import DSAGroupedLayerOffloadContext
from vllm_ascend.dsa_offload.ops import DSAOffloadSelectionOutput
from vllm_ascend.dsa_offload.worker_dram_store import DSAWorkerDRAMStore


def test_glm_indexshare_groups_preserve_first_two_layers_and_draft():
    topology = ("full", "full") + ("full", "shared", "shared", "shared") * 19
    groups = indexshare_groups(topology)
    assert groups[:3] == ((0,), (1,), (2, 3, 4, 5))
    assert groups[-1] == (74, 75, 76, 77)
    assert len(groups) == 21


@pytest.mark.parametrize("topology", [None, (), ("shared",), ("full", "shared"), ("full",), ("bad",)])
def test_invalid_topology_is_not_silently_regrouped(topology):
    with pytest.raises(ValueError):
        indexshare_groups(topology)


@pytest.mark.parametrize("lane", range(4))
@pytest.mark.parametrize("start", [0, 30, 62, 126, 127, 254, 255, 382])
def test_virtual_pa_addresses_equal_interleaved_storage(lane, start):
    # Non-identity physical pages; exercise boundaries within virtual and real
    # blocks, including q0/q1 in one tail and q2/q3 in the next.
    table = torch.tensor([[5, 2, 4, 1], [3, 6, 1, 4]], dtype=torch.int32)
    storage = torch.arange(7 * 128 * 4 * 656, dtype=torch.int32).remainder(251).to(torch.uint8)
    grouped = storage.view(7, 128, 4, 656)
    pa = storage.view(28, 128, 656)
    virtual_table = expanded_block_table(table)
    indices = torch.arange(start, start + 4)
    mapped = lane_slots(indices, lane)
    for row in range(2):
        expected = grouped[table[row, indices // 128].long(), indices % 128, lane]
        actual = pa[virtual_table[row, mapped // 128].long(), mapped % 128]
        assert torch.equal(actual, expected)
    slots = torch.tensor([-7, -1, 0, 127, 128, 255, 256])
    assert torch.equal(lane_slots(slots, lane)[:2], slots[:2])


def test_kv_write_changes_only_its_lane_and_preserves_padding():
    cache = GroupedResidentCache((2, 3, 4, 5), torch.zeros(3, 128, 1, 2624, dtype=torch.int8))
    assert cache.pa_view.data_ptr() == cache.packed.data_ptr()
    slots = torch.tensor([126, 127, 128, 129, -1])
    for lane in range(4):
        mapped = lane_slots(slots, lane)
        cache.pa_view.view(-1, 656)[mapped[mapped >= 0]] = lane + 1
    lanes = cache.packed.view(-1, 4, 656)
    for lane in range(4):
        assert torch.all(lanes[126:130, lane] == lane + 1)
    assert not lanes[:126].count_nonzero()
    assert not lanes[130:].count_nonzero()


def _contexts():
    packed = torch.arange(3 * 128 * 2624, dtype=torch.int32).remainder(127).to(torch.int8).view(3, 128, 1, 2624)
    group = GroupedResidentCache((0, 1, 2, 3), packed)
    store = DSAWorkerDRAMStore(
        usable_blocks=4,
        storage_rows=1,
        max_logical_blocks=4,
        device=torch.device("cpu"),
        arena_factory=lambda shape, dtype, capacity, device: torch.zeros(capacity, *shape, dtype=dtype),
        expected_bytes_per_logical_block=128 * (2624 + 128 + 4),
    )
    key = torch.ones(3, 128, 1, 128, dtype=torch.int8)
    scale = torch.ones(3, 128, 1, 1, dtype=torch.float32)
    dram = store.add_grouped_packed_layers(
        layer_ids=group.layer_ids, resident_packed_cache=packed, indexer_cache=(key, scale)
    )
    store.finalize_payload_layout()
    epochs = set()

    def claim(layer_id, before_cache_write):
        if layer_id in epochs:
            return False
        epochs.add(layer_id)
        return True

    runtime = SimpleNamespace(
        fused_copy_sfa=False,
        dram_store=store,
        dump_before_cache_write=True,
        dump_launch_count=2,
        claim_layer_dump=claim,
        dump_src_block_ids=SimpleNamespace(gpu=torch.tensor([1, 999], dtype=torch.int32)),
        dump_indexer_src_block_ids=SimpleNamespace(gpu=torch.tensor([2, 999], dtype=torch.int32)),
        dump_dst_block_ids=SimpleNamespace(gpu=torch.tensor([3, -1], dtype=torch.int32)),
    )
    contexts = [
        DSAGroupedLayerOffloadContext(
            layer_id=lane,
            indexer_cache=(key, scale) if lane == 0 else None,
            runtime=runtime,
            selection_state_id=0 if lane == 0 else None,
            selection_source_layer_id=None if lane == 0 else 0,
            packed_c8=True,
            group=group,
            group_dram=dram,
            lane=lane,
        )
        for lane in range(4)
    ]
    return group, dram, runtime, contexts


@pytest.mark.parametrize("before", [False, True])
def test_dump_occurs_once_at_correct_group_boundary(monkeypatch, before):
    group, dram, runtime, contexts = _contexts()
    runtime.dump_before_cache_write = before
    copied = []

    def dump(source_bytes, dram_bytes, src_block_ids, dst_block_ids):
        copied.append(source_bytes.data_ptr())
        valid = dst_block_ids >= 0
        dram_bytes[dst_block_ids[valid].long()] = source_bytes[src_block_ids[valid].long()]

    monkeypatch.setattr(grouped_runtime, "dump_full_cache_plane_bytes", dump)
    old = group.packed[1].clone()
    for lane, context in enumerate(contexts):
        context.dump_full_blocks(resident_cache=(group.pa_view,), before_cache_write=True)
        group.packed.view(3, 128, 4, 656)[1, :, lane] = lane + 10
        context.dump_full_blocks(resident_cache=(group.pa_view,), before_cache_write=False)
        assert len(copied) == (3 if before or lane == 3 else 0)
    assert torch.equal(dram[3], old if before else group.packed[1])
    assert not dram[:3].count_nonzero()
    contexts[0 if before else 3].dump_full_blocks(resident_cache=(group.pa_view,), before_cache_write=before)
    assert len(copied) == 3  # one grouped resident + key + scale
    owner = runtime.dram_store.get_layer_arenas(0)
    assert torch.all(owner.indexer_dram_bytes[0][3] == 1)
    for layer in (1, 2, 3):
        assert runtime.dram_store.get_layer_arenas(layer).resident_dram_bytes == ()


def test_one_copy_per_group_not_one_copy_per_layer(monkeypatch):
    group, dram, runtime, contexts = _contexts()
    calls = []
    monkeypatch.setattr(
        torch.ops._C_ascend,
        "npu_dsa_a5_kvcache_scatter_copy_group4_c8_out",
        lambda *args: calls.append(args),
        raising=False,
    )
    table = torch.tensor([[2, 1, 0]], dtype=torch.int32)
    outputs = SimpleNamespace(
        topk_index=torch.zeros(1, 1, 16384, dtype=torch.int32),
        topk_slots=torch.zeros(1, 1, 16384, dtype=torch.int32),
        miss_count=torch.tensor([0]),
    )
    for context in contexts:
        context._copy_or_defer_c8(
            resident_packed_cache=group.pa_view,
            dram_packed_arena=dram,
            resident_block_table=table,
            dram_block_table=table,
            outputs=outputs,
            num_query_rows=4,
        )
    assert len(calls) == 1
    assert calls[0][0].shape[-1] == 2624
    assert calls[0][0].data_ptr() == group.packed.data_ptr()


def test_grouped_fusion_defers_only_owner_and_followers_use_sfa(monkeypatch):
    group, dram, runtime, contexts = _contexts()
    runtime.fused_copy_sfa = True
    runtime._a5_attention_src_ids = torch.full((4, 1, 2048), -1, dtype=torch.int32)
    runtime._a5_per_query_miss_counts = torch.zeros(4, dtype=torch.int32)
    output = torch.empty(4, 8, 512, dtype=torch.bfloat16)
    runtime.get_fused_attention_output = lambda query: output
    copies, fused, sfa = [], [], []
    monkeypatch.setattr(torch.ops._C_ascend, "npu_dsa_a5_kvcache_scatter_copy_group4_c8_out",
                        lambda *args: copies.append(args), raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_dsa_a5_fused_copy_sfa_group4_c8_out",
                        lambda *args: fused.append(args), raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(
        npu_kv_quant_sparse_flash_attention=lambda **kwargs: sfa.append(kwargs) or output))
    table = torch.tensor([[2, 1, 0]], dtype=torch.int32)
    ids = torch.tensor([[[126, 127, 128, 129, -1]]], dtype=torch.int32).expand(4, 1, 5).contiguous()
    lengths = torch.tensor([4], dtype=torch.int32)
    outputs = SimpleNamespace(topk_index=ids[:1], topk_slots=ids[:1], miss_count=lengths, tail_info=None)
    plans = []
    for lane, context in enumerate(contexts):
        plan = context._copy_or_defer_c8(
            resident_packed_cache=group.pa_view, dram_packed_arena=dram,
            resident_block_table=table, dram_block_table=table, outputs=outputs, num_query_rows=4,
        )
        plans.append(plan)
        selection = DSAOffloadSelectionOutput(ids, None, lengths, plan)
        result = context.execute_sparse_attention(
            query=torch.zeros(4, 8, 512, dtype=torch.bfloat16),
            query_rope=torch.zeros(4, 8, 64, dtype=torch.bfloat16),
            resident_cache=(group.pa_view,), selection=selection, scale_value=0.1,
            resident_block_table=table, actual_seq_lengths_query=lengths, actual_seq_lengths_kv=lengths,
        )
        assert result is output
        assert len(fused) == 1 and len(sfa) == lane
        if lane:
            assert torch.equal(sfa[-1]["sparse_indices"], lane_slots(ids, lane))
    assert not copies
    assert plans[0] is not None and plans[1:] == [None, None, None]
    args = fused[0]
    assert len(args) == 17 and args[-1] is output
    assert args[1].shape[-1] == args[2].shape[-1] == 2624
    assert args[1].data_ptr() == group.packed.data_ptr() and args[2].data_ptr() == dram.data_ptr()
    assert args[3] is table and args[4] is table and args[5] is ids
    assert args[8] is lengths and args[9] is lengths
    assert torch.equal(args[13], lane_slots(ids, 0))
    assert torch.equal(args[14], expanded_block_table(table))
    bad_selection = DSAOffloadSelectionOutput(ids, None, lengths, plans[0])
    with pytest.raises(RuntimeError, match="Only the grouped owner"):
        contexts[1].execute_sparse_attention(
            query=None, query_rope=None, resident_cache=None, selection=bad_selection,
            scale_value=0.1, resident_block_table=table,
            actual_seq_lengths_query=lengths, actual_seq_lengths_kv=lengths,
        )
    missing_plan = DSAOffloadSelectionOutput(ids, None, lengths, None)
    with pytest.raises(RuntimeError, match="missing its deferred copy plan"):
        contexts[0].execute_sparse_attention(
            query=None, query_rope=None, resident_cache=None, selection=missing_plan,
            scale_value=0.1, resident_block_table=table,
            actual_seq_lengths_query=lengths, actual_seq_lengths_kv=lengths,
        )


@pytest.mark.parametrize("sparse_mode", [0, 3])
def test_attention_maps_only_addresses_not_lengths(monkeypatch, sparse_mode):
    group, _, _, contexts = _contexts()
    calls = []
    result = torch.empty(4, 8, 512)

    def qsfa(**kwargs):
        calls.append(kwargs)
        return (result,)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_kv_quant_sparse_flash_attention=qsfa))
    ids = torch.tensor([[[126, 127, 128, 129, -1]]], dtype=torch.int32).expand(4, 1, 5)
    seq = torch.tensor([130], dtype=torch.int32)
    table = torch.tensor([[2, 1, 0]], dtype=torch.int32)
    for lane, ctx in enumerate(contexts):
        assert (
            ctx.execute_paged_attention(
                query=torch.zeros(4, 8, 512),
                query_rope=torch.zeros(4, 8, 64),
                resident_cache=(group.pa_view,),
                sparse_indices=ids,
                scale_value=0.1,
                block_table=table,
                actual_seq_lengths_query=torch.tensor([4]),
                actual_seq_lengths_kv=seq,
                sparse_mode=sparse_mode,
            )
            is result
        )
        assert calls[-1]["actual_seq_lengths_kv"] is seq
        assert calls[-1]["sparse_mode"] == sparse_mode
        assert torch.equal(calls[-1]["sparse_indices"], lane_slots(ids, lane))
    assert torch.equal(ids[0, 0], torch.tensor([126, 127, 128, 129, -1]))


def test_allocation_preserves_total_payload_and_singletons(monkeypatch):
    class Spec:
        block_size = 128
        cache_sparse_sfa_c8 = True
        page_size_bytes = 128 * 656

    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.utils", SimpleNamespace(extract_layer_index=lambda s: int(s))
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.dsa_offload.kv_cache", SimpleNamespace(DSAResidentMLAAttentionSpec=Spec)
    )
    topology = ("full", "full", "full", "shared", "shared", "shared")
    specs = {str(i): Spec() for i in range(6)}
    allocated = []

    def allocate(size, alignment):
        allocated.append(size)
        return torch.empty(size, dtype=torch.int8)

    tensors = [SimpleNamespace(shared_by=[str(i)], size=3 * 128 * 656) for i in range(6)]
    mapping = allocate_grouped_caches(SimpleNamespace(kv_cache_tensors=tensors), specs, topology, allocate)
    assert set(mapping) == {"2", "3", "4", "5"}
    assert allocated == [4 * 3 * 128 * 656]
    assert len({id(value) for value in mapping.values()}) == 1
    assert mapping["2"].packed.shape == (3, 128, 1, 2624)
    assert mapping["2"].pa_view.shape == (12, 128, 1, 656)
    tensors[-1].size += 128 * 656
    with pytest.raises(ValueError, match="unequal"):
        allocate_grouped_caches(SimpleNamespace(kv_cache_tensors=tensors), specs, topology, allocate)


def test_runner_allocates_and_reshapes_group_without_per_layer_duplicates(monkeypatch):
    # Run the actual runner methods without importing the hardware-heavy model
    # stack. This catches integration errors (e.g. accidentally reshaping the
    # shared allocation again using a per-layer block count).
    class Spec:
        block_size = 128
        cache_sparse_sfa_c8 = True
        page_size_bytes = 128 * 656

    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.utils", SimpleNamespace(extract_layer_index=int))
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.dsa_offload.kv_cache", SimpleNamespace(DSAResidentMLAAttentionSpec=Spec)
    )
    specs = {str(i): Spec() for i in range(4)}
    names = list(specs)
    config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(shared_by=[name], size=2 * 128 * 656) for name in names],
        kv_cache_groups=[SimpleNamespace(layer_names=names)],
    )
    allocated = []

    def allocate(size, alignment):
        allocated.append(size)
        return torch.empty(size, dtype=torch.int8)

    runner = SimpleNamespace(
        dsa_grouped_caches={},
        shared_kv_cache_layers={},
        runner_only_attn_layers=set(),
        ascend_config=SimpleNamespace(
            dsa_offload_config=SimpleNamespace(
                enable_grouped_c8_kv=True,
                model_capabilities=SimpleNamespace(indexer_types=("full", "shared", "shared", "shared")),
            )
        ),
        _get_layer_kv_cache_specs=lambda _: specs,
        _allocate_int8_cache_tensor=allocate,
        _kv_cache_spec_attn_group_iterator=lambda: [
            SimpleNamespace(backend=None, kv_cache_spec=Spec(), layer_names=names)
        ],
    )
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner")
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "_allocate_kv_cache_tensors",
            "_reshape_kv_cache_tensors",
        )
    ]
    namespace = {
        "torch": torch,
        "allocate_grouped_caches": allocate_grouped_caches,
        "MambaSpec": type("MambaSpec", (), {}),
        "AttentionSpec": Spec,
        "DSAIndexerKVSpec": type("DSAIndexerKVSpec", (), {}),
    }
    for method in methods:
        for arg in method.args.args:
            arg.annotation = None
        method.returns = None
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    raw = namespace["_allocate_kv_cache_tensors"](runner, config)
    shaped = namespace["_reshape_kv_cache_tensors"](runner, config, raw)
    assert set(raw) == set(shaped) == set(names)
    assert allocated == [4 * 2 * 128 * 656]
    assert len({value[0].data_ptr() for value in shaped.values()}) == 1
    assert all(value[0].shape == (8, 128, 1, 656) for value in shaped.values())
