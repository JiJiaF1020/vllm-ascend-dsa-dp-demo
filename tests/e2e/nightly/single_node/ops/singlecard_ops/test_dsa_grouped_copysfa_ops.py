# SPDX-License-Identifier: Apache-2.0
"""Ascend950 Group4 CopySFA: real swapped DRAM, full writeback, SFA and graphs.

No model weights needed. These tests are device acceptance, not CPU emulation.
"""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.dsa_offload.grouped_layout import expanded_block_table, lane_slots
from vllm_ascend.dsa_offload.ops import copy_full_cache_plane_bytes, require_dsa_offload_ops
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

BLOCK = 128
ROW = 656
LANES = 4
TOPK = 2048
COPY_CAP = 16384
BUDGET = 8192
MAX_BUDGET = 16000

pytestmark = pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A5, reason="requires Ascend950")


@pytest.fixture(scope="module", autouse=True)
def _require_ops():
    require_dsa_offload_ops(packed_c8=True, mtp=True, fused_copy_sfa=True, grouped_c8_kv=True)


def _physical(table, tokens):
    return table[tokens // BLOCK].long() * BLOCK + tokens % BLOCK


def _packed_rows(rows, dtype, generator):
    latent = torch.randint(-4, 5, (rows, LANES, 512), generator=generator).float()
    rope = torch.rand(rows, LANES, 64, generator=generator).sub_(0.5).to(dtype)
    scale = torch.rand(rows, LANES, 4, generator=generator).mul_(0.06).add_(0.02)
    return torch.cat(
        (latent.to(torch.float8_e4m3fn).view(torch.int8), rope.view(torch.int8), scale.view(torch.int8)), -1
    ).reshape(-1, LANES * ROW)


def _make_case(count, routes=4, heads=64, dtype=torch.bfloat16, seed=19):
    generator = torch.Generator().manual_seed(seed)
    budget = max(BUDGET, count)
    blocks = (budget + 256) // BLOCK
    # Independent pages per request plus guards, with non-identity HBM/DRAM maps.
    physical_blocks = 2 * blocks + 2
    shape = (physical_blocks, BLOCK, 1, LANES * ROW)
    htable = torch.randperm(2 * blocks, generator=generator).int().view(2, blocks)
    dtable = torch.randperm(2 * blocks, generator=generator).int().view(2, blocks)
    initial = _packed_rows(physical_blocks * BLOCK, dtype, generator)
    dram_cpu = _packed_rows(physical_blocks * BLOCK, dtype, generator)
    src = torch.full((2, 1, COPY_CAP), -1, dtype=torch.int32)
    dst = torch.full_like(src, -1)
    src[0, 0, :count] = torch.randperm(budget, generator=generator)[:count].int()
    dst[0, 0, :count] = (torch.arange(count) + 127).remainder(budget).int()
    expected = initial.clone()
    expected[_physical(htable[0], dst[0, 0, :count].long())] = dram_cpu[_physical(dtable[0], src[0, 0, :count].long())]
    # First request is real, second is PAD. Every query has a valid prefix;
    # q0..q3 cross the 127/128 parity-tail boundary without multiplying lengths.
    slots = torch.full((routes + 1, 1, TOPK), -1, dtype=torch.int32)
    sources = torch.full_like(slots, -1)
    misses = torch.zeros(routes + 1, dtype=torch.int32)
    slot_source = torch.arange(budget + 256, dtype=torch.int32)
    slot_source[dst[0, 0, :count].long()] = src[0, 0, :count]
    for q in range(routes):
        selected = torch.cat((torch.arange(TOPK - 4), budget + torch.arange(123 + q, 127 + q)))
        missing = torch.isin(selected, dst[0, 0, :count])
        # Match LIM's miss-prefix/hit-suffix ABI, not an arbitrary permutation.
        slots[q, 0] = torch.cat((selected[missing], selected[~missing]))
        sources[q, 0] = slot_source[slots[q, 0].long()]
        misses[q] = missing.sum()
    query = torch.randn(routes + 1, heads, 576, generator=generator).to(dtype)
    golden = torch.zeros(LANES, routes + 1, heads, 512)
    for lane in range(LANES):
        for q in range(routes):
            raw = expected[_physical(htable[0], slots[q, 0].long())].reshape(-1, LANES, ROW)[:, lane]
            latent = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
            scale = raw[:, 640:].contiguous().view(torch.float32).repeat_interleave(128, -1)
            value = (latent * scale).to(dtype).float()
            rope = raw[:, 512:640].contiguous().view(dtype).float()
            key = torch.cat((value, rope), -1)
            golden[lane, q] = (query[q].float() @ key.T * 576**-0.5).softmax(-1) @ value
    dram = torch_npu.empty_with_swapped_memory(shape, dtype=torch.int8, device="npu")
    staging = dram_cpu.view(shape).to("npu")
    block_ids = torch.arange(physical_blocks, dtype=torch.int32, device="npu")
    copy_full_cache_plane_bytes(
        source_bytes=staging.view(physical_blocks, -1),
        destination_bytes=dram.view(physical_blocks, -1),
        src_block_ids=block_ids,
        dst_block_ids=block_ids,
    )
    torch.npu.synchronize()
    device = {
        name: tensor.to("npu")
        for name, tensor in dict(
            query=query,
            htable=htable,
            dtable=dtable,
            slots=slots,
            sources=sources,
            misses=misses,
            ends=torch.tensor([routes, routes + 1], dtype=torch.int32),
            resident=torch.tensor([budget + 256, 0], dtype=torch.int32),
            src=src,
            dst=dst,
            counts=torch.tensor([count, 0], dtype=torch.int32),
        ).items()
    }
    return SimpleNamespace(
        **device,
        hbm=initial.view(shape).to("npu"),
        dram=dram,
        initial=initial.view(shape).to("npu"),
        expected=expected.view(shape),
        golden=golden,
        routes=routes,
        out=torch.empty(routes + 1, heads, 512, dtype=dtype, device="npu"),
    )


def _fused(case):
    torch.ops._C_ascend.npu_dsa_a5_fused_copy_sfa_group4_c8_out(
        case.query,
        case.hbm,
        case.dram,
        case.htable,
        case.dtable,
        case.slots,
        case.sources,
        case.misses,
        case.ends,
        case.resident,
        case.src,
        case.dst,
        case.counts,
        lane_slots(case.slots, 0),
        expanded_block_table(case.htable),
        576**-0.5,
        case.out,
    )


def _attention(case, lane):
    kv = case.hbm.view(-1, BLOCK, 1, ROW).view(torch.float8_e4m3fn)
    out = torch_npu.npu_kv_quant_sparse_flash_attention(
        query=case.query,
        key=kv,
        value=kv,
        sparse_indices=lane_slots(case.slots, lane),
        scale_value=576**-0.5,
        sparse_block_size=1,
        block_table=expanded_block_table(case.htable),
        actual_seq_lengths_query=case.ends,
        actual_seq_lengths_kv=case.resident,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        key_quant_mode=2,
        value_quant_mode=2,
        rope_head_dim=64,
    )
    return out[0] if isinstance(out, tuple) else out


def _split(case):
    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_group4_c8_out(
        case.hbm,
        case.dram,
        case.htable,
        case.dtable,
        case.src,
        case.dst,
        case.counts,
    )
    return _attention(case, 0)


def _check(case, followers=True):
    torch.npu.synchronize()
    # Whole allocation, all four lanes, guards and PAD pages. Not merely topK.
    assert torch.equal(case.hbm.cpu(), case.expected)
    actual = case.out.cpu().float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, case.golden[0], atol=0.008, rtol=0.05)
    relative = (actual - case.golden[0]).flatten(1).norm(dim=1) / case.golden[0].flatten(1).norm(dim=1).clamp_min(1e-7)
    assert (relative <= 0.08).all()
    if followers:
        for lane in range(LANES):
            # No additional copy for followers. Native QSFA's PAD output is not
            # the oracle; compare real rows, plus the fused PAD against golden.
            native = _attention(case, lane).cpu().float()[: case.routes]
            torch.testing.assert_close(native, case.golden[lane, : case.routes], atol=0.008, rtol=0.05)
        torch.testing.assert_close(
            actual[: case.routes], _attention(case, 0).cpu().float()[: case.routes], atol=0.008, rtol=0.05
        )


@pytest.mark.parametrize("count", [0, 1, 127, 128, 129, 512, 1024, 1025, BUDGET, MAX_BUDGET])
def test_grouped_copysfa_union_copy_and_all_lanes(count):
    case = _make_case(count)
    _fused(case)
    _check(case)
    case.hbm.copy_(case.initial)
    split_output = _split(case)
    torch.npu.synchronize()
    assert torch.equal(case.hbm.cpu(), case.expected)
    torch.testing.assert_close(case.out[: case.routes], split_output[: case.routes], atol=0.008, rtol=0.05)


@pytest.mark.parametrize(
    "routes,heads,dtype",
    [
        (1, 8, torch.bfloat16),
        (2, 16, torch.bfloat16),
        (3, 32, torch.bfloat16),
        (4, 64, torch.float16),
        (4, 128, torch.bfloat16),
    ],
)
def test_grouped_copysfa_query_shapes(routes, heads, dtype):
    case = _make_case(512, routes=routes, heads=heads, dtype=dtype)
    _fused(case)
    _check(case)


def test_grouped_copysfa_dynamic_graph():
    case = _make_case(0)
    _fused(case)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _fused(case)
    owners = (
        case.query,
        case.hbm,
        case.dram,
        case.htable,
        case.dtable,
        case.slots,
        case.sources,
        case.misses,
        case.ends,
        case.resident,
        case.src,
        case.dst,
        case.counts,
        case.out,
    )
    pointers = tuple(t.data_ptr() for t in owners)
    for count, seed in ((1, 29), (1024, 31), (1025, 37), (BUDGET, 41), (0, 43)):
        reference = _make_case(count, seed=seed)
        # Every original input address is stable while contents/tables change.
        ids = torch.arange(case.dram.shape[0], dtype=torch.int32, device="npu")
        copy_full_cache_plane_bytes(
            source_bytes=reference.dram.view(len(ids), -1),
            destination_bytes=case.dram.view(len(ids), -1),
            src_block_ids=ids,
            dst_block_ids=ids,
        )
        for name in (
            "query",
            "htable",
            "dtable",
            "slots",
            "sources",
            "misses",
            "ends",
            "resident",
            "src",
            "dst",
            "counts",
        ):
            getattr(case, name).copy_(getattr(reference, name))
        case.expected, case.golden = reference.expected, reference.golden
        for _ in range(3):
            case.hbm.copy_(reference.initial)
            case.out.fill_(float("nan"))
            graph.replay()
            _check(case, followers=False)
            assert pointers == tuple(t.data_ptr() for t in owners)


def test_grouped_copysfa_microbenchmark(capsys):
    case = _make_case(512)

    def measure(run, graph_mode):
        for _ in range(3):
            run(case)
        if graph_mode:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run(case)
            launch = graph.replay
        else:
            launch = lambda: run(case)
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(30):
            launch()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) / 30

    for graph_mode in (False, True):
        split_ms, fused_ms = measure(_split, graph_mode), measure(_fused, graph_mode)
        with capsys.disabled():
            print(f"[Group4 CopySFA] graph={graph_mode} copy+owner-SFA: split={split_ms:.4f}ms fused={fused_ms:.4f}ms")
