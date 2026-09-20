# SPDX-License-Identifier: Apache-2.0
"""A5 grouped-KV acceptance tests (requires the new Group4C8 custom op)."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.dsa_offload.grouped_layout import GroupedResidentCache, lane_slots
from vllm_ascend.dsa_offload.grouped_runtime import DSAGroupedLayerOffloadContext
from vllm_ascend.dsa_offload.ops import copy_full_cache_plane_bytes, require_dsa_offload_ops
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="requires Ascend950")


@pytest.fixture(scope="module", autouse=True)
def _require_ops():
    require_dsa_offload_ops(packed_c8=True, mtp=True, grouped_c8_kv=True)


def _swapped(shape):
    return torch_npu.empty_with_swapped_memory(shape, dtype=torch.int8, device="npu")


def test_grouped_dump_copy_eager_and_graph_with_padding():
    torch.manual_seed(19)
    source = torch.randint(-128, 128, (4, 128, 1, 2624), dtype=torch.int8, device="npu")
    dram = _swapped((4, 128, 1, 2624))
    hbm = torch.zeros_like(source)
    dump_src = torch.tensor([3, 1, 999], dtype=torch.int32, device="npu")
    dump_dst = torch.tensor([1, 2, -1], dtype=torch.int32, device="npu")
    htable = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32, device="npu")
    dtable = torch.tensor([[2, 1], [0, 0]], dtype=torch.int32, device="npu")
    src = torch.full((2, 1, 16384), -1, dtype=torch.int32, device="npu")
    dst = torch.full_like(src, -1)
    src[0, 0, :4] = torch.tensor([0, 127, 128, 129], dtype=torch.int32, device="npu")
    dst[0, 0, :4] = torch.tensor([0, 129, 126, 127], dtype=torch.int32, device="npu")
    counts = torch.tensor([4, 0], dtype=torch.int32, device="npu")

    def chain():
        copy_full_cache_plane_bytes(
            source_bytes=source.view(4, -1),
            destination_bytes=dram.view(4, -1),
            src_block_ids=dump_src,
            dst_block_ids=dump_dst,
        )
        torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_group4_c8_out(hbm, dram, htable, dtable, src, dst, counts)

    def check():
        torch.npu.synchronize()
        expected = torch.zeros_like(source.cpu())
        ref = source.cpu()
        expected[3, 0] = ref[1, 0]
        expected[1, 1] = ref[1, 127]
        expected[3, 126] = ref[3, 0]
        expected[3, 127] = ref[3, 1]
        assert torch.equal(hbm.cpu(), expected)

    chain()
    check()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        chain()
    for seed in (25, 31):
        torch.manual_seed(seed)
        source.copy_(torch.randint(-128, 128, source.shape, dtype=torch.int8, device="npu"))
        hbm.zero_()
        graph.replay()
        check()


@pytest.mark.parametrize("heads", [8, 16, 32, 64])
@pytest.mark.parametrize("mode", [0, 3])
def test_native_qsfa_virtual_pages_equal_original_cache(heads, mode):
    torch.manual_seed(123)
    blocks = 5
    original = []
    for lane in range(4):
        key = torch.randn(blocks, 128, 1, 512).clamp(-3, 3).to(torch.float8_e4m3fn).view(torch.int8)
        rope = torch.randn(blocks, 128, 1, 64).to(torch.bfloat16).view(torch.int8)
        scale = torch.rand(blocks, 128, 1, 4).add_(0.1).view(torch.int8)
        original.append(torch.cat([key, rope, scale], -1).to("npu"))
    packed = torch.cat(original, -1).view(torch.float8_e4m3fn)
    group = GroupedResidentCache((0, 1, 2, 3), packed)
    # q0/q1 attend through tokens 126/127; q2/q3 through 128/129.
    ids = torch.full((4, 1, 2048), -1, dtype=torch.int32, device="npu")
    for q in range(4):
        ids[q, 0, : 127 + q] = torch.arange(127 + q, dtype=torch.int32, device="npu")
    table = torch.tensor([[4, 2, 3, 1]], dtype=torch.int32, device="npu")
    q_lens = torch.tensor([4], dtype=torch.int32, device="npu")
    kv_lens = torch.tensor([130], dtype=torch.int32, device="npu")
    query = torch.randn(4, heads, 512, dtype=torch.bfloat16, device="npu")
    query_rope = torch.randn(4, heads, 64, dtype=torch.bfloat16, device="npu")
    contexts = [
        DSAGroupedLayerOffloadContext(
            layer_id=lane,
            indexer_cache=None,
            runtime=SimpleNamespace(),
            packed_c8=True,
            group=group,
            group_dram=packed.view(torch.int8),
            lane=lane,
        )
        for lane in range(4)
    ]

    def run_lane(lane):
        return contexts[lane].execute_paged_attention(
            query=query,
            query_rope=query_rope,
            resident_cache=(group.pa_view,),
            sparse_indices=ids,
            scale_value=576**-0.5,
            block_table=table,
            actual_seq_lengths_query=q_lens,
            actual_seq_lengths_kv=kv_lens,
            sparse_mode=mode,
        )

    def reference(lane):
        result = torch_npu.npu_kv_quant_sparse_flash_attention(
            query=torch.cat([query, query_rope], -1),
            key=original[lane].view(torch.float8_e4m3fn),
            value=original[lane].view(torch.float8_e4m3fn),
            sparse_indices=ids,
            scale_value=576**-0.5,
            sparse_block_size=1,
            block_table=table,
            actual_seq_lengths_query=q_lens,
            actual_seq_lengths_kv=kv_lens,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=mode,
            attention_mode=2,
            quant_scale_repo_mode=1,
            tile_size=128,
            key_quant_mode=2,
            value_quant_mode=2,
            rope_head_dim=64,
        )
        return result[0] if isinstance(result, tuple) else result

    for lane in range(4):
        expected = reference(lane)
        actual = run_lane(lane)
        torch.npu.synchronize()
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = [run_lane(lane) for lane in range(4)]
    for delta in (0, 32):
        # Same graph, new sequence/table/index tensors in the original buffers.
        kv_lens.fill_(130 + delta)
        table[0, :2] = torch.tensor([2, 4], dtype=torch.int32, device="npu")
        for q in range(4):
            ids[q, 0, : 127 + q] = torch.arange(delta, 127 + q + delta, dtype=torch.int32, device="npu")
        graph.replay()
        for lane in range(4):
            torch.testing.assert_close(outputs[lane], reference(lane), rtol=0, atol=0)


def test_native_packed_kv_scatter_writes_only_selected_lane():
    group = GroupedResidentCache((0, 1, 2, 3), torch.zeros(3, 128, 1, 2624, dtype=torch.int8, device="npu"))
    slots = torch.tensor([126, 127, 128, 129], dtype=torch.int64, device="npu")
    for lane in range(4):
        updates = torch.full((4, 656), lane + 1, dtype=torch.int8, device="npu")
        torch_npu.npu_scatter_nd_update_(group.pa_view.view(-1, 656), lane_slots(slots, lane).view(-1, 1), updates)
    actual = group.packed.cpu().view(-1, 4, 656)
    for lane in range(4):
        assert torch.all(actual[126:130, lane] == lane + 1)
    assert not actual[:126].count_nonzero()
    assert not actual[130:].count_nonzero()


def test_group_copy_microbenchmark(capsys):
    """Report, not assert, speed: dependent on actual DRAM/interconnect load."""
    count = 2048
    shape = (32, 128, 1, 656)
    singles = [(torch.zeros(shape, dtype=torch.int8, device="npu"), _swapped(shape)) for _ in range(4)]
    grouped = (torch.zeros(32, 128, 1, 2624, dtype=torch.int8, device="npu"), _swapped((32, 128, 1, 2624)))
    for _, dram in [*singles, grouped]:
        dram.zero_()
    table = torch.arange(32, dtype=torch.int32, device="npu").view(1, -1)
    src = torch.full((1, 1, 16384), -1, dtype=torch.int32, device="npu")
    dst = torch.full_like(src, -1)
    src[0, 0, :count] = torch.randperm(4096, device="npu")[:count].to(torch.int32)
    dst[0, 0, :count] = torch.arange(count, dtype=torch.int32, device="npu")
    counts = torch.tensor([count], dtype=torch.int32, device="npu")

    def measure(pairs, op):
        def run():
            for hbm, dram in pairs:
                op(hbm, dram, table, table, src, dst, counts)

        for _ in range(3):
            run()
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            run()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 20

    split_ms = measure(singles, torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out)
    group_ms = measure([grouped], torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_group4_c8_out)
    with capsys.disabled():
        print(f"[grouped-KV copy] 2048 tokens x 4 layers: old={split_ms:.4f}ms grouped={group_ms:.4f}ms")
