# SPDX-License-Identifier: Apache-2.0
"""A5 packed-C8 DSA operator integration checks."""

import json

import pytest
import torch
import torch_npu

from vllm_ascend.dsa_offload.contracts import DSA_LIDU_CACHE_ROW_ALIGNMENT, DSA_LIDU_TOKEN_CAPACITY
from vllm_ascend.dsa_offload.ops import (
    quant_lightning_indexer_topk,
    require_dsa_offload_ops,
    sparse_flash_attention_for_offload_c8,
)
from vllm_ascend.utils import (
    AscendDeviceType,
    get_ascend_device_type,
)

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="packed-C8 DSA operators require Ascend A5",
)

_BLOCK_SIZE = 128
_PACKED_ROW_BYTES = 656
_TOPK = 2048
_COPY_CAPACITY = 16384
_ATTENTION_CAPACITY = _TOPK + _BLOCK_SIZE
_MTP_ATTENTION_CAPACITY = _TOPK + 2 * _BLOCK_SIZE


def _recycled_resident_block_ids() -> torch.Tensor:
    """Physical layout observed when the fifth 40K request reuses HBM."""

    return torch.cat(
        (
            torch.arange(294, 214, -1, dtype=torch.int32),
            torch.tensor([295], dtype=torch.int32),
        )
    )


def _fresh_resident_block_ids() -> torch.Tensor:
    return torch.cat(
        (
            torch.arange(327, 407, dtype=torch.int32),
            torch.tensor([326], dtype=torch.int32),
        )
    )


def _cache_row_width(max_model_len: int) -> int:
    raw_width = max_model_len + 1
    alignment = DSA_LIDU_CACHE_ROW_ALIGNMENT
    aligned_width = (raw_width + alignment - 1) // alignment * alignment
    return aligned_width if aligned_width - 1 <= DSA_LIDU_TOKEN_CAPACITY else raw_width


def _assert_exact_int_tensor(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = torch.nonzero(actual != expected, as_tuple=False)
    first_index = tuple(int(value) for value in mismatch[0].tolist())
    pytest.fail(
        "integer tensor mismatch: "
        f"first_index={first_index}, "
        f"actual={int(actual[first_index])}, "
        f"expected={int(expected[first_index])}, "
        f"mismatch_count={int(mismatch.shape[0])}"
    )


@pytest.fixture(scope="module", autouse=True)
def _load_a5_dsa_custom_ops() -> None:
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("packed-C8 DSA operators require Ascend A5")
    try:
        require_dsa_offload_ops(packed_c8=True, mtp=True)
    except RuntimeError as error:
        pytest.fail(str(error))


def _swapped_arena(shape: tuple[int, ...]) -> torch.Tensor:
    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        pytest.skip("torch_npu swapped-memory API is unavailable")
    return torch_npu.empty_with_swapped_memory(
        shape,
        dtype=torch.int8,
        device=torch.device(f"npu:{torch.npu.current_device()}"),
    )


def _read_swapped_arena(tensor: torch.Tensor) -> torch.Tensor:
    staging = torch.empty(
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    staging.fill_(1).mul_(tensor)
    torch.npu.synchronize()
    return staging.cpu()


def _write_swapped_arena(
    destination: torch.Tensor,
    source: torch.Tensor,
) -> None:
    destination.zero_()
    staging = source.to(destination.device)
    destination.add_(staging)
    torch.npu.synchronize()


def _make_unique_packed_rows(num_rows: int) -> torch.Tensor:
    """Encode the complete source token ID into every opaque packed row."""

    base = torch.arange(_PACKED_ROW_BYTES, dtype=torch.int16).remainder(251).sub(125).to(torch.int8)
    rows = base.repeat(num_rows, 1)
    token_ids = torch.arange(num_rows, dtype=torch.int32)
    rows[:, 0] = token_ids.to(torch.int8)
    rows[:, 1] = torch.div(token_ids, 256, rounding_mode="floor").to(torch.int8)
    rows[:, 2] = torch.div(token_ids, 65536, rounding_mode="floor").to(torch.int8)
    return rows


def test_packed_c8_full_block_dump_writes_only_active_rows() -> None:
    source = (
        torch.randint(
            -128,
            128,
            (2, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES),
            dtype=torch.int16,
        )
        .to(torch.int8)
        .to("npu")
    )
    destination = _swapped_arena((3, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES))
    destination.zero_()
    src_block_ids = torch.tensor([1, 999], dtype=torch.int32, device="npu")
    dst_block_ids = torch.tensor([2, -1], dtype=torch.int32, device="npu")

    torch.ops._C_ascend.kv_cache_full_block_dump_c8(
        source,
        destination,
        src_block_ids,
        dst_block_ids,
    )
    torch.npu.synchronize()

    actual = _read_swapped_arena(destination)
    expected_source = source[1].cpu()
    assert torch.equal(actual[2], expected_source)
    assert torch.count_nonzero(actual[:2]).item() == 0


def test_complete_indexer_byte_plane_dump_preserves_fp32_scale_bits() -> None:
    scale = torch.arange(
        2 * _BLOCK_SIZE,
        dtype=torch.float32,
        device="npu",
    ).view(2, _BLOCK_SIZE, 1, 1)
    source_bytes = scale.view(torch.int8).reshape(2, -1)
    destination = _swapped_arena((3, int(source_bytes.shape[1])))
    destination.zero_()
    src_block_ids = torch.tensor([1], dtype=torch.int32, device="npu")
    dst_block_ids = torch.tensor([2], dtype=torch.int32, device="npu")

    torch.ops._C_ascend.kv_cache_full_block_dump_bytes(
        source_bytes,
        destination,
        src_block_ids,
        dst_block_ids,
    )
    torch.npu.synchronize()

    actual = _read_swapped_arena(destination)
    assert torch.equal(actual[2], source_bytes[1].cpu())
    assert torch.count_nonzero(actual[:2]).item() == 0


def test_complete_indexer_byte_plane_dump_replays_fixed_metadata() -> None:
    key = torch.randn(
        (2, _BLOCK_SIZE, 1, 128),
        dtype=torch.bfloat16,
        device="npu",
    ).to(torch.float8_e4m3fn)
    source_bytes = key.view(torch.int8).reshape(2, -1)
    destination = _swapped_arena((3, int(source_bytes.shape[1])))
    destination.zero_()
    src_block_ids = torch.tensor([0], dtype=torch.int32, device="npu")
    dst_block_ids = torch.tensor([1], dtype=torch.int32, device="npu")
    stable_pointers = (
        source_bytes.data_ptr(),
        destination.data_ptr(),
        src_block_ids.data_ptr(),
        dst_block_ids.data_ptr(),
    )

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        torch.ops._C_ascend.kv_cache_full_block_dump_bytes(
            source_bytes,
            destination,
            src_block_ids,
            dst_block_ids,
        )

    graph.replay()
    torch.npu.synchronize()
    actual = _read_swapped_arena(destination)
    assert torch.equal(actual[1], source_bytes[0].cpu())

    _write_swapped_arena(destination, torch.zeros_like(actual))
    src_block_ids.fill_(1)
    dst_block_ids.fill_(2)
    graph.replay()
    torch.npu.synchronize()
    actual = _read_swapped_arena(destination)
    assert torch.equal(actual[2], source_bytes[1].cpu())

    # 图 PAD/steady 空转只允许读取 dst=-1；非法 src 不得被解引用。
    _write_swapped_arena(destination, torch.zeros_like(actual))
    src_block_ids.fill_(999)
    dst_block_ids.fill_(-1)
    graph.replay()
    torch.npu.synchronize()
    assert torch.count_nonzero(_read_swapped_arena(destination)).item() == 0
    assert stable_pointers == (
        source_bytes.data_ptr(),
        destination.data_ptr(),
        src_block_ids.data_ptr(),
        dst_block_ids.data_ptr(),
    )


def _make_fused_lidu_inputs(
    *,
    candidate_len: int,
    final_len: int,
    row_mode: int,
    batch: int = 1,
    seed: int = 7,
    heads: int = 32,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    blocks = (candidate_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    weights_storage = torch.randn(
        (batch, 128 + heads),
        dtype=torch.bfloat16,
        device="npu",
    )
    return {
        "weights": weights_storage[:, 128:],
        "query": torch.randn(
            (batch, heads, 128),
            dtype=torch.bfloat16,
            device="npu",
        ).to(torch.float8_e4m3fn),
        # Keep every row/head distinct so the batch=2 DENSE cases also catch
        # accidental reuse of the strided weights offset for this compact
        # scale tensor.
        "query_scale": torch.linspace(
            0.5,
            1.5,
            steps=batch * heads,
            dtype=torch.float32,
            device="npu",
        ).view(batch, heads),
        "query_ends": torch.arange(
            1,
            batch + 1,
            dtype=torch.int32,
            device="npu",
        ),
        "key": torch.randn((blocks, 128, 1, 128), dtype=torch.bfloat16, device="npu").to(torch.float8_e4m3fn),
        "key_scale": torch.ones((blocks, 128, 1), dtype=torch.float32, device="npu"),
        "block_table": torch.arange(
            blocks,
            dtype=torch.int32,
            device="npu",
        )
        .view(1, -1)
        .repeat(batch, 1),
        "candidate_lens": torch.full(
            (batch,),
            candidate_len,
            dtype=torch.int32,
            device="npu",
        ),
        "final_lens": torch.full(
            (batch,),
            final_len,
            dtype=torch.int32,
            device="npu",
        ),
        "row_modes": torch.full(
            (batch,),
            row_mode,
            dtype=torch.int32,
            device="npu",
        ),
        "req_entries": (
            torch.arange(batch, dtype=torch.int32, device="npu")
            if row_mode == 2
            else torch.full(
                (batch,),
                -1,
                dtype=torch.int32,
                device="npu",
            )
        ),
    }


def _make_disjoint_mtp_lidu_inputs(*, query_counts, max_candidate, heads, seed=7):
    """Independent request index blocks, with production-style block strides."""
    batch = len(query_counts)
    total_rows = sum(query_counts)
    inputs = _make_fused_lidu_inputs(
        candidate_len=max_candidate,
        final_len=max_candidate + 2 * _BLOCK_SIZE,
        row_mode=2,
        batch=total_rows,
        heads=heads,
        seed=seed,
    )
    capacity = max_candidate + 2 * _BLOCK_SIZE
    blocks = (capacity + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    key_storage = torch.randn(
        (batch * blocks, _BLOCK_SIZE + 2, 1, 128), dtype=torch.bfloat16, device="npu"
    ).to(torch.float8_e4m3fn)
    scale_storage = torch.ones((batch * blocks, _BLOCK_SIZE + 8, 1), dtype=torch.float32, device="npu")
    inputs["key"] = key_storage[:, :_BLOCK_SIZE]
    inputs["key_scale"] = scale_storage[:, :_BLOCK_SIZE]
    inputs["block_table"] = torch.arange(batch * blocks, dtype=torch.int32, device="npu").view(batch, blocks)
    for name in ("candidate_lens", "final_lens", "row_modes", "req_entries"):
        inputs[name] = inputs[name][:batch].clone()
    inputs["query_ends"] = torch.tensor(query_counts, dtype=torch.int32).cumsum(0, dtype=torch.int32).to("npu")
    return inputs


def _allocate_fused_lidu_outputs(
    batch: int = 1,
    total_query_rows: int | None = None,
) -> tuple[torch.Tensor, ...]:
    if total_query_rows is None:
        total_query_rows = batch
    attention_slots = torch.full(
        (total_query_rows, 1, _ATTENTION_CAPACITY),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    resident_seq_lengths = torch.full(
        (batch,),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    copy_src_ids = torch.full(
        (batch, 1, _COPY_CAPACITY),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    copy_dst_slots = torch.full_like(copy_src_ids, -77)
    copy_counts = torch.full(
        (batch,),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    return (
        attention_slots,
        resident_seq_lengths,
        copy_src_ids,
        copy_dst_slots,
        copy_counts,
    )


def _allocate_mtp_lidu_outputs(
    batch: int = 1,
    total_query_rows: int | None = None,
) -> tuple[torch.Tensor, ...]:
    if total_query_rows is None:
        total_query_rows = batch
    sparse_and_tail_slots = torch.full(
        (total_query_rows, 1, _MTP_ATTENTION_CAPACITY),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    sparse_and_tail_src_ids = torch.full_like(
        sparse_and_tail_slots,
        -77,
    )
    per_query_miss_counts = torch.full(
        (total_query_rows,),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    resident_seq_lengths = torch.full(
        (batch,),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    copy_src_ids = torch.full(
        (batch, 1, _COPY_CAPACITY),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    copy_dst_slots = torch.full_like(copy_src_ids, -77)
    copy_counts = torch.full(
        (batch,),
        -77,
        dtype=torch.int32,
        device="npu",
    )
    return (
        sparse_and_tail_slots,
        sparse_and_tail_src_ids,
        per_query_miss_counts,
        resident_seq_lengths,
        copy_src_ids,
        copy_dst_slots,
        copy_counts,
    )


def _launch_fused_lidu(
    inputs: dict[str, torch.Tensor],
    cache_slots: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    torch.ops._C_ascend.npu_dsa_a5_li_manage_nomtp_c8_out(
        inputs["weights"],
        inputs["query"],
        inputs["query_scale"],
        inputs["query_ends"],
        inputs["key"],
        inputs["key_scale"],
        inputs["block_table"],
        inputs["candidate_lens"],
        inputs["final_lens"],
        inputs["row_modes"],
        inputs["req_entries"],
        cache_slots,
        *outputs,
    )


def _launch_mtp_lim(
    inputs: dict[str, torch.Tensor],
    cache_slots: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    torch.ops._C_ascend.npu_dsa_a5_li_manage_c8_out(
        inputs["weights"],
        inputs["query"],
        inputs["query_scale"],
        inputs["query_ends"],
        inputs["key"],
        inputs["key_scale"],
        inputs["block_table"],
        inputs["candidate_lens"],
        inputs["final_lens"],
        inputs["row_modes"],
        inputs["req_entries"],
        cache_slots,
        *outputs,
    )


def _launch_packed_ksc(
    hbm: torch.Tensor,
    dram: torch.Tensor,
    block_table: torch.Tensor,
    outputs: tuple[torch.Tensor, ...],
) -> None:
    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
        hbm,
        dram,
        block_table,
        block_table,
        outputs[-3],
        outputs[-2],
        outputs[-1],
    )


def _native_topk(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return quant_lightning_indexer_topk(
        query=inputs["query"],
        key=inputs["key"],
        weights=inputs["weights"],
        query_dequant_scale=inputs["query_scale"],
        key_dequant_scale=inputs["key_scale"],
        actual_seq_lengths_query=inputs["query_ends"],
        candidate_lens=inputs["candidate_lens"],
        block_table=inputs["block_table"],
    )


def _native_mtp_topk_rows(inputs: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    """Independent R=1 LI calls: shared sparse prefix, per-query dense visibility."""
    rows = []
    start = 0
    for request, end in enumerate(inputs["query_ends"].cpu().tolist()):
        mode = int(inputs["row_modes"][request].cpu())
        final_len = int(inputs["final_lens"][request].cpu())
        candidate = int(inputs["candidate_lens"][request].cpu())
        for query_row in range(start, end):
            length = candidate if mode == 2 else final_len - (end - 1 - query_row)
            if mode == 0:
                rows.append(torch.empty(0, dtype=torch.int32))
            elif length <= _TOPK:
                rows.append(torch.arange(length, dtype=torch.int32))
            else:
                row_inputs = {
                    **inputs,
                    "query": inputs["query"][query_row : query_row + 1],
                    "weights": inputs["weights"][query_row : query_row + 1],
                    "query_scale": inputs["query_scale"][query_row : query_row + 1],
                    "query_ends": torch.ones(1, dtype=torch.int32, device="npu"),
                    "candidate_lens": torch.tensor([length], dtype=torch.int32, device="npu"),
                    "block_table": inputs["block_table"][request : request + 1],
                }
                rows.append(_native_topk(row_inputs).cpu().view(-1))
        start = end
    return rows


def _assert_mtp_selection(inputs, pool_before, pool_after, outputs, topk_rows):
    """Verify copy/cache semantics, then build QSFA indices from the independent LI oracle."""
    slots, sources, query_misses, resident, copy_src, copy_dst, counts = [x.cpu() for x in outputs]
    before, after = pool_before.cpu(), pool_after.cpu()
    expected_pool = before.clone()
    oracle_slots = torch.full_like(slots, -1)
    start = 0
    for request, end in enumerate(inputs["query_ends"].cpu().tolist()):
        mode = int(inputs["row_modes"][request].cpu())
        final_len = int(inputs["final_lens"][request].cpu())
        candidate = int(inputs["candidate_lens"][request].cpu())
        if mode == 2:
            entry = int(inputs["req_entries"][request].cpu())
            budget = abs(int(before[entry, -1]))
            first_fill = int(before[entry, -1]) < 0
            union = torch.unique(torch.cat(topk_rows[start:end])).long()
            misses = union[before[entry, union] < 0]
            count = int(counts[request])
            assert count == (budget if first_fill else misses.numel())
            copied = copy_src[request, 0, :count].long()
            destinations = copy_dst[request, 0, :count]
            assert torch.unique(copied).numel() == count
            assert torch.unique(destinations).numel() == count
            assert torch.all((copied >= 0) & (copied < candidate))
            assert torch.all((destinations >= 0) & (destinations < budget))
            if not first_fill:
                _assert_exact_int_tensor(torch.sort(copied).values, torch.sort(misses).values)
            old = before[entry, :-1]
            evicted = (old >= 0) & torch.isin(old, destinations)
            expected_pool[entry, :-1][evicted] = -1
            expected_pool[entry, copied] = destinations
            expected_pool[entry, -1] = budget
            assert torch.all(after[entry, union] >= 0)
            held = after[entry, :-1][after[entry, :-1] >= 0]
            _assert_exact_int_tensor(torch.sort(held).values, torch.arange(budget, dtype=torch.int32))
            assert resident[request] == budget + 2 * _BLOCK_SIZE
        else:
            assert counts[request] == 0
            assert resident[request] == (0 if mode == 0 else final_len)
        for row in range(start, end):
            expected_tokens = topk_rows[row]
            selected = expected_tokens.numel()
            actual_tokens = sources[row, 0, :selected].long()
            _assert_exact_int_tensor(torch.sort(actual_tokens).values, torch.sort(expected_tokens.long()).values)
            if mode == 2:
                m = int(query_misses[row])
                expected_misses = expected_tokens[before[entry, expected_tokens.long()] < 0]
                assert m == expected_misses.numel()
                _assert_exact_int_tensor(
                    torch.sort(actual_tokens[:m]).values, torch.sort(expected_misses.long()).values
                )
                assert torch.all(before[entry, actual_tokens[m:]] >= 0)
                _assert_exact_int_tensor(slots[row, 0, :selected], after[entry, actual_tokens])
                oracle_slots[row, 0, :selected] = after[entry, expected_tokens.long()]
                visible = final_len - (end - 1 - row)
                tail = torch.arange(candidate, visible, dtype=torch.int32)
                tail_slots = budget + tail.remainder(2 * _BLOCK_SIZE)
                _assert_exact_int_tensor(sources[row, 0, _TOPK : _TOPK + tail.numel()], tail)
                _assert_exact_int_tensor(slots[row, 0, _TOPK : _TOPK + tail.numel()], tail_slots)
                oracle_slots[row, 0, _TOPK : _TOPK + tail.numel()] = tail_slots
                selected = _TOPK + tail.numel()
            else:
                assert query_misses[row] == 0
                _assert_exact_int_tensor(slots[row, 0, :selected], actual_tokens.to(torch.int32))
                oracle_slots[row, 0, :selected] = expected_tokens
            assert torch.all(slots[row, 0, selected:] == -1)
            assert torch.all(sources[row, 0, selected:] == -1)
        start = end
    _assert_exact_int_tensor(after, expected_pool)
    return oracle_slots


@pytest.mark.parametrize("output_index", [0, 1])
def test_packed_c8_mtp_lim_rejects_old_2176_outputs(output_index) -> None:
    inputs = _make_fused_lidu_inputs(candidate_len=2048, final_len=0, row_mode=0)
    pool = torch.zeros((1, _cache_row_width(8192)), dtype=torch.int32, device="npu")
    outputs = list(_allocate_mtp_lidu_outputs())
    outputs[output_index] = torch.empty((1, 1, _ATTENTION_CAPACITY), dtype=torch.int32, device="npu")
    with pytest.raises(RuntimeError, match="2304"):
        _launch_mtp_lim(inputs, pool, tuple(outputs))


@pytest.mark.parametrize("invalid", ["source-dtype", "miss-dtype", "source-rows", "miss-rows", "source-stride"])
def test_packed_c8_mtp_lim_checks_new_output_contract(invalid) -> None:
    inputs = _make_fused_lidu_inputs(candidate_len=2048, final_len=0, row_mode=0)
    pool = torch.zeros((1, _cache_row_width(8192)), dtype=torch.int32, device="npu")
    outputs = list(_allocate_mtp_lidu_outputs())
    if invalid == "source-dtype":
        outputs[1] = outputs[1].to(torch.int64)
    elif invalid == "miss-dtype":
        outputs[2] = outputs[2].to(torch.int64)
    elif invalid == "source-rows":
        outputs[1] = torch.empty((2, 1, _MTP_ATTENTION_CAPACITY), dtype=torch.int32, device="npu")
    elif invalid == "miss-rows":
        outputs[2] = torch.empty(2, dtype=torch.int32, device="npu")
    else:
        outputs[1] = torch.empty((1, 1, 2 * _MTP_ATTENTION_CAPACITY), dtype=torch.int32, device="npu")[..., ::2]
    with pytest.raises(RuntimeError, match="int32|output shapes|contiguous"):
        _launch_mtp_lim(inputs, pool, tuple(outputs))


@pytest.mark.parametrize(
    "heads,groups,unaligned_weights", [(32, 1, True), (64, 1, True), (32, 7, False), (32, 7, True)]
)
def test_packed_c8_mtp_lim_mixed_2304_graph_replay(heads, groups, unaligned_weights) -> None:
    # T=70/head32 selects the four-query tile, including partial 1/2/3-row
    # tiles. Small mixed batches only exercise the two-query implementation.
    batch, total_rows, candidate, budget = 4 * groups, 10 * groups, 8320, 8192
    inputs = _make_fused_lidu_inputs(
        candidate_len=candidate, final_len=candidate + 256, row_mode=2, batch=total_rows, heads=heads
    )
    for name in ("block_table", "candidate_lens", "final_lens", "row_modes", "req_entries", "query_ends"):
        inputs[name] = inputs[name][:batch].clone()
    # Exercise the framework suffix-view ABI, including an unaligned row stride.
    if unaligned_weights:
        weight_storage = torch.empty((total_rows, heads + 1), dtype=torch.bfloat16, device="npu")
        weight_storage[:, :heads].copy_(inputs["weights"])
        inputs["weights"] = weight_storage[:, :heads]
    pool = torch.full((batch, _cache_row_width(candidate + 256)), -1, dtype=torch.int32, device="npu")
    outputs = _allocate_mtp_lidu_outputs(batch, total_rows)
    initial_ends = [10 * group + end for group in range(groups) for end in (1, 3, 6, 10)]
    inputs["query_ends"].copy_(torch.tensor(initial_ends, dtype=torch.int32, device="npu"))
    inputs["row_modes"].zero_()
    inputs["final_lens"].zero_()
    inputs["req_entries"].fill_(-1)
    pool[:, -1].zero_()
    _launch_mtp_lim(inputs, pool, outputs)
    torch.npu.synchronize()
    pointers = tuple(t.data_ptr() for t in (*inputs.values(), pool, *outputs))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        _launch_mtp_lim(inputs, pool, outputs)
    graph.replay()
    torch.npu.synchronize()
    assert outputs[-1].cpu().eq(0).all()

    for ends, modes, finals, entries, first_fill in (
        ([1, 3, 6, 10], [0, 1, 2, 2], [0, 16, candidate + 129, candidate + 256], [-1, -1, 2, 0], False),
        ([4, 7, 9, 10], [2, 1, 0, 2], [candidate + 129, 4096, 0, candidate + 256], [1, -1, -1, 0], True),
        ([4, 7, 9, 10], [2, 1, 0, 2], [candidate + 129, 4096, 0, candidate + 256], [1, -1, -1, 0], False),
        ([1, 3, 6, 10], [0, 2, 1, 2], [0, candidate + 130, 4096, candidate + 129], [-1, 1, -1, 0], False),
    ):
        ends = [10 * group + end for group in range(groups) for end in ends]
        entries = [4 * group + entry if entry >= 0 else -1 for group in range(groups) for entry in entries]
        modes = modes * groups
        finals = finals * groups
        metadata = (("query_ends", ends), ("row_modes", modes), ("final_lens", finals), ("req_entries", entries))
        for name, values in metadata:
            inputs[name].copy_(torch.tensor(values, dtype=torch.int32, device="npu"))
        initial = torch.full_like(pool, -1)
        initial[:, -1].zero_()
        for request, entry in enumerate(entries):
            if modes[request] == 2:
                initial[entry, :budget] = torch.arange(budget, dtype=torch.int32, device="npu")
                initial[entry, -1] = budget
        if first_fill:
            initial[entries[0]].fill_(-1)
            initial[entries[0], -1] = -budget
        topk = _native_mtp_topk_rows(inputs)
        pool.copy_(initial)
        _launch_mtp_lim(inputs, pool, outputs)
        torch.npu.synchronize()
        _assert_mtp_selection(inputs, initial, pool, outputs, topk)
        for _ in range(3):
            pool.copy_(initial)
            graph.replay()
            torch.npu.synchronize()
            _assert_mtp_selection(inputs, initial, pool, outputs, topk)
        resident_before = pool.clone()
        graph.replay()
        torch.npu.synchronize()
        _assert_mtp_selection(inputs, resident_before, pool, outputs, topk)
        assert outputs[-1].cpu().eq(0).all()
        assert pointers == tuple(t.data_ptr() for t in (*inputs.values(), pool, *outputs))


@pytest.mark.parametrize("dense_len", [1, 2048, 2049, 2176, 2177, 6144])
def test_packed_c8_fused_lidu_dense_matches_framework_semantics(
    dense_len: int,
) -> None:
    batch = 2
    inputs = _make_fused_lidu_inputs(
        candidate_len=dense_len,
        final_len=dense_len,
        row_mode=1,
        batch=batch,
    )
    assert inputs["weights"].stride() == (160, 1)
    assert not inputs["weights"].is_contiguous()
    expected_slots = (
        torch.arange(dense_len, dtype=torch.int32).view(1, 1, -1).repeat(batch, 1, 1)
        if dense_len <= _TOPK
        else _native_topk(inputs).cpu()
    )
    cache_slots = torch.full(
        (batch + 1, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    before = cache_slots.clone()
    outputs = _allocate_fused_lidu_outputs(batch)

    _launch_fused_lidu(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, resident_lengths, _, _, counts = (tensor.cpu() for tensor in outputs)
    selected_count = min(dense_len, _TOPK)
    _assert_exact_int_tensor(attention[:, :, :selected_count], expected_slots)
    assert torch.all(attention[:, :, selected_count:] == -1)
    assert resident_lengths.tolist() == [dense_len] * batch
    assert counts.tolist() == [0] * batch
    assert torch.equal(cache_slots, before)


def _pool_row_reference_hbm(source_bytes, pool_row, candidate, budget, dense_len, final_len):
    """CPU materialization deliberately bypasses KSC to isolate LIM/QSFA metadata effects."""
    span = budget + 2 * _BLOCK_SIZE
    assert budget > 0 and budget % _BLOCK_SIZE == 0
    assert source_bytes.ndim == 2 and source_bytes.shape[0] >= final_len
    assert source_bytes.shape[1] == _PACKED_ROW_BYTES
    assert 0 < dense_len <= span and candidate <= final_len <= candidate + 2 * _BLOCK_SIZE
    held = torch.nonzero(pool_row[:candidate] >= 0, as_tuple=False).flatten()
    slots = pool_row[held].long()
    _assert_exact_int_tensor(torch.sort(slots).values, torch.arange(budget))
    resident = torch.zeros((2, span, _PACKED_ROW_BYTES), dtype=torch.int8)
    # Even unused slots contain valid zero KV with positive scale, not poison.
    resident[:, :, 640:] = source_bytes[0, 640:]
    resident[0, :dense_len] = source_bytes[:dense_len]
    resident[1, slots] = source_bytes[held]
    tail = torch.arange(candidate, final_len)
    resident[1, budget + tail.remainder(2 * _BLOCK_SIZE)] = source_bytes[tail]
    return resident.view(2 * span // _BLOCK_SIZE, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES)


def _assert_pool_row_invariance(reports):
    """Metadata-row relocation must not alter the logical replacement policy."""
    assert reports, "pool-row regression produced no comparisons"
    fields = (
        "same_row_repeat", "same_logical_state_before", "same_resident_map_after",
        "same_selected_sets", "same_selected_order", "same_slots",
        "qsfa_bitwise_equal", "ordered_qsfa_bitwise_equal",
    )
    failures = []
    for report in reports:
        changed = [field for field in fields if report[field] is not True]
        if report["copy_counts"][0] != report["copy_counts"][1]:
            changed.append("copy_counts")
        if changed:
            failures.append({"step": report["step"], "changed": changed})
    assert not failures, f"pool-row relocation changed cache/attention behavior: {failures}"


def test_packed_c8_mtp_pool_row_relocation_diagnostic() -> None:
    """Strict single-card row-relocation regression, not an end-to-end gate.

    Exercise mixed DENSE/SPARSE, first-fill then two steady updates at the
    observed 70k geometry. Run 0a/1/0b with the same tensors and query sequence;
    only the sparse request's pool row changes. The logical victim policy and
    subsequent states must stay identical. KSC is bypassed by CPU byte materialization.
    """
    candidate, budget, dense_len, final_len = 73472, 12288, 7756, 73500
    batch, routes, steps = 2, 4, 3
    total_rows = batch * routes
    inputs = _make_fused_lidu_inputs(
        candidate_len=candidate, final_len=final_len, row_mode=2, batch=total_rows, heads=64, seed=83
    )
    for name in ("block_table", "candidate_lens", "final_lens", "row_modes", "req_entries"):
        inputs[name] = inputs[name][:batch].clone()
    inputs["row_modes"].copy_(torch.tensor([1, 2], dtype=torch.int32, device="npu"))
    inputs["final_lens"].copy_(torch.tensor([dense_len, final_len], dtype=torch.int32, device="npu"))
    inputs["candidate_lens"].copy_(torch.tensor([dense_len, candidate], dtype=torch.int32, device="npu"))
    inputs["query_ends"] = torch.tensor([routes, total_rows], dtype=torch.int32, device="npu")
    query_sequence = [
        torch.roll(inputs["query"].view(torch.int8), shifts=step, dims=0).view(torch.float8_e4m3fn).clone()
        for step in range(steps)
    ]
    pool = torch.full((2, _cache_row_width(100000)), -1, dtype=torch.int32, device="npu")
    outputs = _allocate_mtp_lidu_outputs(batch, total_rows)

    generator = torch.Generator().manual_seed(89)
    source_bytes = torch.zeros((final_len, _PACKED_ROW_BYTES), dtype=torch.int8)
    source_bytes[:, :512] = (
        torch.randint(-3, 4, (final_len, 512), dtype=torch.int8, generator=generator)
        .float().to(torch.float8_e4m3fn).view(torch.int8)
    )
    source_bytes[:, 512:640] = (
        (torch.randn((final_len, 64), generator=generator) * 0.1).to(torch.bfloat16).view(torch.int8)
    )
    source_bytes[:, 640:] = torch.full((final_len, 4), 0.05, dtype=torch.float32).view(torch.int8)
    query = (torch.randn((total_rows, 8, 576), generator=generator) * 0.1).to(torch.bfloat16).to("npu")
    span = budget + 2 * _BLOCK_SIZE
    blocks = span // _BLOCK_SIZE
    hbm = torch.zeros((batch * blocks, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES), dtype=torch.int8, device="npu")
    block_table = torch.arange(batch * blocks, dtype=torch.int32, device="npu").view(batch, blocks)
    canonical_slots = torch.empty_like(outputs[0])

    def attention(slots):
        result = sparse_flash_attention_for_offload_c8(
            query=query, packed_kv=hbm.view(torch.float8_e4m3fn), sparse_indices=slots,
            scale_value=576**-0.5, block_table=block_table,
            actual_seq_lengths_query=inputs["query_ends"], resident_seq_lengths=outputs[3],
        ).cpu()
        assert torch.isfinite(result).all()
        return result[routes:].clone()

    snapshots = {}
    for label, entry in (("row0-a", 0), ("row1", 1), ("row0-b", 0)):
        pool.fill_(-1)
        pool[:, -1].zero_()
        pool[entry, -1] = -budget
        inputs["req_entries"].copy_(torch.tensor([-1, entry], dtype=torch.int32, device="npu"))
        snapshots[label] = []
        for step in range(steps):
            inputs["query"].copy_(query_sequence[step])
            before = pool.cpu().clone()
            for output in outputs:
                output.fill_(-77)
            print(f"[mtp-pool-row] run={label} pool_row={entry} step={step} stage=lim", flush=True)
            _launch_mtp_lim(inputs, pool, outputs)
            torch.npu.synchronize()
            after = pool.cpu().clone()
            source_ids = outputs[1].cpu()
            slots = outputs[0].cpu()
            # The emitted source IDs validate copy/mapping/causal contracts,
            # NOT LI ranking against an independent golden. Cross-row ranking
            # differences are reported below instead of accepting them silently.
            selected = []
            for row in range(total_rows):
                ids = source_ids[row, 0, :_TOPK].long()
                limit = dense_len - (routes - 1 - row) if row < routes else candidate
                assert torch.all((ids >= 0) & (ids < limit))
                assert torch.unique(ids).numel() == _TOPK
                selected.append(ids)
            _assert_mtp_selection(inputs, before, after, outputs, selected)
            reference = _pool_row_reference_hbm(source_bytes, after[entry], candidate, budget, dense_len, final_len)
            hbm.copy_(reference.to("npu"))
            print(f"[mtp-pool-row] run={label} pool_row={entry} step={step} stage=qsfa", flush=True)
            raw_attention = attention(outputs[0])
            # Test-only order control: sort paired TopK by logical token ID,
            # keep causal tail and -1 suffix untouched. Never sort block IDs.
            ordered = slots.clone()
            order = source_ids[:, :, :_TOPK].argsort(dim=-1)
            ordered[:, :, :_TOPK] = slots[:, :, :_TOPK].gather(-1, order)
            canonical_slots.copy_(ordered.to("npu"))
            ordered_attention = attention(canonical_slots)
            snapshots[label].append({
                "before": before[entry], "after": after[entry],
                "sources": source_ids[routes:].clone(), "slots": slots[routes:].clone(),
                "copy_src": outputs[4].cpu()[1].clone(), "copy_dst": outputs[5].cpu()[1].clone(),
                "copy_count": int(outputs[6].cpu()[1]),
                "raw": raw_attention, "ordered": ordered_attention,
            })
    reports = []
    for step in range(steps):
        left, right, repeated = (snapshots[label][step] for label in ("row0-a", "row1", "row0-b"))
        same_row_repeat = all(torch.equal(left[key], repeated[key]) for key in (
            "before", "after", "sources", "slots", "copy_src", "copy_dst", "raw", "ordered"
        )) and left["copy_count"] == repeated["copy_count"]
        report = {
            "step": step, "phase": "first-fill" if step == 0 else "steady",
            "candidate": candidate, "budget": budget, "pool_width": int(pool.shape[1]),
            "same_row_repeat": same_row_repeat,
            "same_logical_state_before": torch.equal(left["before"], right["before"]),
            "same_resident_map_after": torch.equal(left["after"], right["after"]),
            "same_selected_sets": torch.equal(left["sources"].sort(-1).values, right["sources"].sort(-1).values),
            "same_selected_order": torch.equal(left["sources"], right["sources"]),
            "same_slots": torch.equal(left["slots"], right["slots"]),
            "copy_counts": [left["copy_count"], right["copy_count"]],
            "qsfa_bitwise_equal": torch.equal(left["raw"], right["raw"]),
            "qsfa_max_abs": float((left["raw"].float() - right["raw"].float()).abs().max()),
            "ordered_qsfa_bitwise_equal": torch.equal(left["ordered"], right["ordered"]),
            "ordered_qsfa_max_abs": float((left["ordered"].float() - right["ordered"].float()).abs().max()),
        }
        print("[mtp-pool-row] " + json.dumps(report, sort_keys=True), flush=True)
        reports.append(report)
    _assert_pool_row_invariance(reports)
    print("[mtp-pool-row] POOL_ROW_INVARIANCE_PASSED; synthetic LIM/QSFA, not full-model validation", flush=True)


def test_packed_c8_mtp_2304_copy_qsfa_chain_eager_and_graph() -> None:
    """Framework wiring check, not a closure of native QSFA's padding precision issue."""
    candidate, budget, routes, tail_capacity = 8320, 8192, 4, 256
    final_len = candidate + tail_capacity
    inputs = _make_fused_lidu_inputs(candidate_len=candidate, final_len=final_len, row_mode=2, batch=routes, seed=83)
    for name in ("block_table", "candidate_lens", "final_lens", "row_modes", "req_entries"):
        inputs[name] = inputs[name][:1]
    inputs["query_ends"] = torch.tensor([routes], dtype=torch.int32, device="npu")
    topk = _native_mtp_topk_rows(inputs)
    initial_pool = torch.full((2, _cache_row_width(final_len)), -1, dtype=torch.int32, device="npu")
    initial_pool[:, -1].zero_()
    initial_pool[0, :budget] = torch.arange(budget, dtype=torch.int32, device="npu")
    initial_pool[0, -1] = budget
    pool = initial_pool.clone()
    outputs = _allocate_mtp_lidu_outputs(1, routes)
    _launch_mtp_lim(inputs, pool, outputs)
    torch.npu.synchronize()
    oracle_slots = _assert_mtp_selection(inputs, initial_pool, pool, outputs, topk)
    count = int(outputs[-1].cpu()[0])
    assert count > 0
    sources = outputs[4][0, 0, :count].cpu().long()
    destinations = outputs[5][0, 0, :count].cpu().long()

    # Valid packed rows: 512 FP8 latent bytes, 128 BF16 RoPE bytes, four FP32 scales.
    generator = torch.Generator().manual_seed(89)
    source_bytes = torch.zeros((final_len, _PACKED_ROW_BYTES), dtype=torch.int8)
    source_bytes[:, :512] = (
        torch.randint(-3, 4, (final_len, 512), generator=generator).float().to(torch.float8_e4m3fn).view(torch.int8)
    )
    source_bytes[:, 512:640] = (
        (torch.randn((final_len, 64), generator=generator) * 0.1).to(torch.bfloat16).view(torch.int8)
    )
    source_bytes[:, 640:] = torch.full((final_len, 4), 0.05, dtype=torch.float32).view(torch.int8)
    dram_shape = (final_len // _BLOCK_SIZE, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES)
    dram = _swapped_arena(dram_shape)
    _write_swapped_arena(dram, source_bytes.view(dram_shape))
    hbm_blocks = (budget + tail_capacity) // _BLOCK_SIZE
    hbm_shape = (hbm_blocks, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES)
    initial_hbm = torch.full((budget + tail_capacity, _PACKED_ROW_BYTES), -91, dtype=torch.int8)
    initial_hbm[:budget] = source_bytes[:budget]
    tail_ids = torch.arange(candidate, final_len)
    initial_hbm[budget + tail_ids.remainder(tail_capacity)] = source_bytes[tail_ids]
    # Reset every copy target before BOTH eager and replay, not just before the first run.
    initial_hbm[destinations] = -91
    expected_hbm = initial_hbm.clone()
    expected_hbm[destinations] = source_bytes[sources]
    hbm = initial_hbm.view(hbm_shape).to("npu")
    hbm_table = torch.arange(hbm_blocks, dtype=torch.int32, device="npu").view(1, -1)
    dram_table = torch.arange(dram_shape[0], dtype=torch.int32, device="npu").view(1, -1)
    query = (torch.randn((routes, 8, 576), generator=generator) * 0.1).to(torch.bfloat16).to("npu")
    scale = 576**-0.5

    def chain():
        _launch_mtp_lim(inputs, pool, outputs)
        torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
            hbm, dram, hbm_table, dram_table, outputs[4], outputs[5], outputs[6]
        )
        return sparse_flash_attention_for_offload_c8(
            query=query,
            packed_kv=hbm.view(torch.float8_e4m3fn),
            sparse_indices=outputs[0],
            scale_value=scale,
            block_table=hbm_table,
            actual_seq_lengths_query=inputs["query_ends"],
            resident_seq_lengths=outputs[3],
        )

    # Separate golden data and independent LI-derived indices, not the tested slots.
    reference_kv = expected_hbm.view(hbm_shape).to("npu").view(torch.float8_e4m3fn)
    golden = torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query,
        key=reference_kv,
        value=reference_kv,
        sparse_indices=oracle_slots.to("npu"),
        scale_value=scale,
        key_quant_mode=2,
        value_quant_mode=2,
        block_table=hbm_table,
        actual_seq_lengths_query=inputs["query_ends"],
        actual_seq_lengths_kv=torch.tensor([budget + tail_capacity], dtype=torch.int32, device="npu"),
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    if isinstance(golden, tuple):
        golden = golden[0]
    pool.copy_(initial_pool)
    actual = chain()
    torch.npu.synchronize()
    _assert_mtp_selection(inputs, initial_pool, pool, outputs, topk)
    _assert_exact_int_tensor(hbm.cpu().view_as(expected_hbm), expected_hbm)
    torch.testing.assert_close(actual, golden, atol=0.002, rtol=0.02)
    pointers = tuple(t.data_ptr() for t in (pool, hbm, dram, hbm_table, *inputs.values(), *outputs))
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        graph_attention = chain()
    graph.replay()
    torch.npu.synchronize()
    assert outputs[-1].cpu().tolist() == [0]
    graph_output_pointer = graph_attention.data_ptr()
    for _ in range(3):
        pool.copy_(initial_pool)
        hbm.copy_(initial_hbm.view(hbm_shape).to("npu"))
        graph.replay()
        torch.npu.synchronize()
        _assert_mtp_selection(inputs, initial_pool, pool, outputs, topk)
        _assert_exact_int_tensor(hbm.cpu().view_as(expected_hbm), expected_hbm)
        torch.testing.assert_close(graph_attention, golden, atol=0.002, rtol=0.02)
        assert graph_attention.data_ptr() == graph_output_pointer
        assert pointers == tuple(t.data_ptr() for t in (pool, hbm, dram, hbm_table, *inputs.values(), *outputs))


def test_packed_c8_mtp_lim_dense_short_rows_are_causal() -> None:
    batch = 2
    queries_per_request = 4
    total_query_rows = batch * queries_per_request
    final_len = 16
    inputs = _make_fused_lidu_inputs(
        candidate_len=final_len,
        final_len=final_len,
        row_mode=1,
        batch=total_query_rows,
        seed=29,
    )
    inputs["query_ends"] = torch.tensor(
        [4, 8],
        dtype=torch.int32,
        device="npu",
    )
    for name in (
        "block_table",
        "candidate_lens",
        "final_lens",
        "row_modes",
        "req_entries",
    ):
        inputs[name] = inputs[name][:batch]
    cache_slots = torch.full(
        (batch + 1, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    before = cache_slots.clone()
    outputs = _allocate_mtp_lidu_outputs(
        batch,
        total_query_rows,
    )

    _launch_mtp_lim(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, _, _, resident_lengths, _, _, counts = (tensor.cpu() for tensor in outputs)
    for query_row in range(total_query_rows):
        local_query = query_row % queries_per_request
        visible_len = final_len - (queries_per_request - 1 - local_query)
        _assert_exact_int_tensor(
            attention[query_row, 0, :visible_len],
            torch.arange(visible_len, dtype=torch.int32),
        )
        assert torch.all(attention[query_row, 0, visible_len:] == -1)
    assert resident_lengths.tolist() == [final_len] * batch
    assert counts.tolist() == [0] * batch
    assert torch.equal(cache_slots, before)


def test_packed_c8_mtp_lim_dense_long_rows_are_causal() -> None:
    batch = 2
    queries_per_request = 4
    total_query_rows = batch * queries_per_request
    final_len = 2180
    inputs = _make_fused_lidu_inputs(
        candidate_len=final_len,
        final_len=final_len,
        row_mode=1,
        batch=total_query_rows,
        seed=31,
    )
    # Request-level tensors keep B rows while query/weight/scale keep T rows.
    inputs["query_ends"] = torch.tensor(
        [4, 8],
        dtype=torch.int32,
        device="npu",
    )
    inputs["block_table"] = inputs["block_table"][:batch]
    inputs["candidate_lens"] = inputs["candidate_lens"][:batch]
    inputs["final_lens"] = inputs["final_lens"][:batch]
    inputs["row_modes"] = inputs["row_modes"][:batch]
    inputs["req_entries"] = inputs["req_entries"][:batch]
    expected_topk = torch.stack(_native_mtp_topk_rows(inputs))
    cache_slots = torch.full(
        (batch + 1, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    before = cache_slots.clone()
    outputs = _allocate_mtp_lidu_outputs(
        batch,
        total_query_rows,
    )

    _launch_mtp_lim(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    (
        attention,
        _,
        _,
        resident_lengths,
        raw_topk,
        error_metadata,
        counts,
    ) = (tensor.cpu() for tensor in outputs)
    if any(count < 0 for count in counts.tolist()):
        diagnostics = []
        for request, count in enumerate(counts.tolist()):
            if count >= 0:
                continue
            raw = raw_topk[request, 0, :_TOPK]
            query_row = int(error_metadata[request, 0, 0])
            visible_len = int(error_metadata[request, 0, 1])
            invalid = torch.nonzero(
                (raw < 0) | (raw >= visible_len),
                as_tuple=False,
            ).view(-1)
            diagnostics.append(
                {
                    "request": request,
                    "query_row": query_row,
                    "visible_len": visible_len,
                    "invalid_count": int(invalid.numel()),
                    "first_invalid_indices": invalid[:16].tolist(),
                    "raw_prefix": raw[:16].tolist(),
                    "raw_suffix": raw[-16:].tolist(),
                    "valid_unique": int(torch.unique(raw[(raw >= 0) & (raw < visible_len)]).numel()),
                }
            )
        pytest.fail(f"MTP LIM dense selection error: {diagnostics}")
    for query_row in range(total_query_rows):
        # Score ordering is not an ABI guarantee; membership and multiplicity are.
        actual_row = torch.sort(attention[query_row, 0, :_TOPK]).values
        expected_row = torch.sort(expected_topk[query_row]).values
        if not torch.equal(actual_row, expected_row):
            mismatch = torch.nonzero(
                actual_row != expected_row,
                as_tuple=False,
            ).view(-1)
            invalid = torch.nonzero(
                (actual_row < 0) | (actual_row >= final_len),
                as_tuple=False,
            ).view(-1)
            pytest.fail(
                "MTP LIM dense top-k mismatch: "
                f"query_row={query_row}, "
                f"mismatch_count={int(mismatch.numel())}, "
                f"first_mismatch_indices={mismatch[:16].tolist()}, "
                f"actual_at_mismatch={actual_row[mismatch[:16]].tolist()}, "
                f"expected_at_mismatch={expected_row[mismatch[:16]].tolist()}, "
                f"invalid_count={int(invalid.numel())}, "
                f"invalid_indices={invalid[:16].tolist()}, "
                "same_multiset="
                f"{torch.equal(torch.sort(actual_row).values, torch.sort(expected_row).values)}"
            )
        assert torch.all(attention[query_row, 0, _TOPK:] == -1)
    assert resident_lengths.tolist() == [final_len] * batch
    assert counts.tolist() == [0] * batch
    assert torch.equal(cache_slots, before)


@pytest.mark.parametrize("source_len", [8192, 8320])
@pytest.mark.parametrize("visible_tail", [0, 1, 127, 128, 129, 255, 256])
def test_packed_c8_mtp_lim_uses_shared_prefix_and_dual_tails(source_len, visible_tail) -> None:
    budget = 8192
    queries_per_request = min(4, visible_tail + 1)
    final_len = source_len + visible_tail
    inputs = _make_fused_lidu_inputs(
        candidate_len=source_len,
        final_len=final_len,
        row_mode=2,
        batch=queries_per_request,
        seed=47,
    )
    inputs["query_ends"] = torch.tensor(
        [queries_per_request],
        dtype=torch.int32,
        device="npu",
    )
    inputs["block_table"] = inputs["block_table"][:1]
    inputs["candidate_lens"] = torch.tensor(
        [source_len],
        dtype=torch.int32,
        device="npu",
    )
    inputs["final_lens"] = torch.tensor(
        [final_len],
        dtype=torch.int32,
        device="npu",
    )
    inputs["row_modes"] = torch.tensor(
        [2],
        dtype=torch.int32,
        device="npu",
    )
    inputs["req_entries"] = torch.tensor(
        [0],
        dtype=torch.int32,
        device="npu",
    )

    expected_topk_rows: list[torch.Tensor] = []
    for query_row in range(queries_per_request):
        row_inputs = {
            "query": inputs["query"][query_row : query_row + 1],
            "key": inputs["key"],
            "weights": inputs["weights"][query_row : query_row + 1],
            "query_scale": inputs["query_scale"][query_row : query_row + 1],
            "key_scale": inputs["key_scale"],
            "query_ends": torch.tensor(
                [1],
                dtype=torch.int32,
                device="npu",
            ),
            "candidate_lens": torch.tensor(
                [source_len],
                dtype=torch.int32,
                device="npu",
            ),
            "block_table": inputs["block_table"],
        }
        expected_topk_rows.append(_native_topk(row_inputs).cpu().view(-1))

    cache_slots = torch.full(
        (2, _cache_row_width(final_len)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, -1] = -budget
    outputs = _allocate_mtp_lidu_outputs(
        batch=1,
        total_query_rows=queries_per_request,
    )

    _launch_mtp_lim(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, source_ids, query_misses, resident_lengths, _, _, counts = (tensor.cpu() for tensor in outputs)
    cache_row = cache_slots[0].cpu()
    for query_row, expected_tokens in enumerate(expected_topk_rows):
        actual_tokens = source_ids[query_row, 0, :_TOPK]
        _assert_exact_int_tensor(
            torch.sort(actual_tokens).values,
            torch.sort(expected_tokens).values,
        )
        _assert_exact_int_tensor(attention[query_row, 0, :_TOPK], cache_row[actual_tokens.long()])
        assert query_misses[query_row] == _TOPK

        visible_len = final_len - (queries_per_request - 1 - query_row)
        tail_tokens = torch.arange(
            source_len,
            visible_len,
            dtype=torch.int32,
        )
        expected_tail_slots = budget + tail_tokens.remainder(2 * _BLOCK_SIZE)
        tail_len = int(tail_tokens.numel())
        _assert_exact_int_tensor(
            attention[
                query_row,
                0,
                _TOPK : _TOPK + tail_len,
            ],
            expected_tail_slots,
        )
        _assert_exact_int_tensor(source_ids[query_row, 0, _TOPK : _TOPK + tail_len], tail_tokens)
        assert torch.all(attention[query_row, 0, _TOPK + tail_len :] == -1)
        assert torch.all(source_ids[query_row, 0, _TOPK + tail_len :] == -1)

    assert resident_lengths.tolist() == [budget + 2 * _BLOCK_SIZE]
    assert counts.tolist() == [budget]


@pytest.mark.parametrize("heads", [32, 64])
def test_packed_c8_mtp_lim_steady_fast_path_updates_union_and_dual_tails(
    heads: int,
) -> None:
    budget = 8192
    candidate_len = 8320
    final_len = candidate_len + 4
    queries_per_request = 4

    def make_request(seed: int) -> dict[str, torch.Tensor]:
        inputs = _make_fused_lidu_inputs(
            candidate_len=candidate_len,
            final_len=final_len,
            row_mode=2,
            batch=queries_per_request,
            seed=seed,
            heads=heads,
        )
        inputs["query_ends"] = torch.tensor(
            [queries_per_request],
            dtype=torch.int32,
            device="npu",
        )
        for name in (
            "block_table",
            "candidate_lens",
            "final_lens",
            "row_modes",
            "req_entries",
        ):
            inputs[name] = inputs[name][:1]
        return inputs

    cache_slots = torch.full(
        (2, _cache_row_width(final_len)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, -1] = -budget
    # The final pool element is metadata, deliberately leaving a logical
    # token capacity that is not a 128-token QuantLI score-row stride.
    assert (int(cache_slots.shape[1]) - 1) % _BLOCK_SIZE != 0

    # Establish the persistent request row through the correctness path. The
    # positive metadata left by this call is part of the fast-path contract.
    cold_outputs = _allocate_mtp_lidu_outputs(
        batch=1,
        total_query_rows=queries_per_request,
    )
    _launch_mtp_lim(make_request(seed=53), cache_slots, cold_outputs)
    torch.npu.synchronize()
    assert int(cache_slots[0, -1].cpu()) == budget

    steady_inputs = make_request(seed=59)
    expected_topk_rows = []
    for route in range(queries_per_request):
        row_inputs = {
            **steady_inputs,
            "query": steady_inputs["query"][route : route + 1],
            "weights": steady_inputs["weights"][route : route + 1],
            "query_scale": steady_inputs["query_scale"][route : route + 1],
            # candidate_lens is a durable prefix before every speculative
            # query. A batched sparse_mode=3 reference would instead treat
            # the queries as part of the key sequence and causally hide the
            # final 3/2/1 prefix tokens from its early rows.
            "query_ends": torch.tensor(
                [1],
                dtype=torch.int32,
                device="npu",
            ),
        }
        expected_topk_rows.append(_native_topk(row_inputs).cpu().view(-1))
    expected_topk = torch.stack(expected_topk_rows)
    pool_before = cache_slots[0].cpu().clone()
    expected_misses = torch.unique(expected_topk[pool_before[expected_topk.to(torch.long)] < 0])
    assert expected_misses.numel() > 0

    outputs = _allocate_mtp_lidu_outputs(
        batch=1,
        total_query_rows=queries_per_request,
    )
    _launch_mtp_lim(steady_inputs, cache_slots, outputs)
    torch.npu.synchronize()

    (
        attention,
        attention_src_ids,
        per_query_miss_counts,
        resident_lengths,
        source_ids,
        destination_slots,
        counts,
    ) = (tensor.cpu() for tensor in outputs)
    pool_after = cache_slots[0].cpu()
    copy_count = int(counts[0])
    assert copy_count == int(expected_misses.numel())
    actual_sources = source_ids[0, 0, :copy_count]
    actual_destinations = destination_slots[0, 0, :copy_count]
    _assert_exact_int_tensor(
        torch.sort(actual_sources).values,
        torch.sort(expected_misses).values,
    )
    assert torch.unique(actual_destinations).numel() == copy_count
    assert torch.all((actual_destinations >= 0) & (actual_destinations < budget))
    _assert_exact_int_tensor(
        pool_after[actual_sources.to(torch.long)],
        actual_destinations,
    )

    inverse_before = torch.full((budget,), -1, dtype=torch.int32)
    mapped_sources = torch.nonzero(
        pool_before[:candidate_len] >= 0,
        as_tuple=False,
    ).view(-1)
    inverse_before[pool_before[mapped_sources].to(torch.long)] = mapped_sources.to(torch.int32)
    victim_sources = inverse_before[actual_destinations.to(torch.long)]
    assert torch.all(victim_sources >= 0)
    assert torch.all(pool_after[victim_sources.to(torch.long)] == -1)

    for route in range(queries_per_request):
        expected_slots = pool_after[expected_topk[route].to(torch.long)]
        actual_slots = attention[route, 0, :_TOPK]
        actual_src_ids = attention_src_ids[route, 0, :_TOPK]
        route_miss_count = int(per_query_miss_counts[route])
        expected_route_misses = expected_topk[route][pool_before[expected_topk[route].to(torch.long)] < 0]
        assert route_miss_count == int(expected_route_misses.numel())
        _assert_exact_int_tensor(
            torch.sort(actual_src_ids[:route_miss_count]).values,
            torch.sort(expected_route_misses).values,
        )
        assert torch.all(pool_before[actual_src_ids[route_miss_count:].to(torch.long)] >= 0)
        _assert_exact_int_tensor(
            torch.sort(actual_src_ids).values,
            torch.sort(expected_topk[route]).values,
        )
        _assert_exact_int_tensor(
            actual_slots,
            pool_after[actual_src_ids.to(torch.long)],
        )
        # The optimized path deliberately publishes the miss destinations
        # before the compacted hit slots. Sparse attention consumes an
        # unordered resident-slot set, so preserving native TopK score order
        # would only add another 2048-entry reorder/round trip. Keep the test
        # strict about membership, multiplicity and cardinality instead.
        _assert_exact_int_tensor(
            torch.sort(actual_slots).values,
            torch.sort(expected_slots).values,
        )
        assert torch.unique(actual_slots).numel() == _TOPK
        visible_len = final_len - (queries_per_request - 1 - route)
        tail_tokens = torch.arange(
            candidate_len,
            visible_len,
            dtype=torch.int32,
        )
        expected_tail = budget + tail_tokens.remainder(2 * _BLOCK_SIZE)
        tail_len = int(tail_tokens.numel())
        _assert_exact_int_tensor(
            attention[route, 0, _TOPK : _TOPK + tail_len],
            expected_tail,
        )
        assert torch.all(attention[route, 0, _TOPK + tail_len :] == -1)

    assert resident_lengths.tolist() == [budget + 2 * _BLOCK_SIZE]
    assert int(pool_after[-1]) == budget

    expected_attention = outputs[0].clone()
    expected_resident_lengths = outputs[3].clone()
    pool_before_repeat = cache_slots.clone()
    for repeat_index in range(8):
        repeat_outputs = _allocate_mtp_lidu_outputs(
            batch=1,
            total_query_rows=queries_per_request,
        )
        _launch_mtp_lim(steady_inputs, cache_slots, repeat_outputs)
        torch.npu.synchronize()
        repeat_attention = repeat_outputs[0]
        for route in range(queries_per_request):
            _assert_exact_int_tensor(
                torch.sort(repeat_attention[route, 0]).values,
                torch.sort(expected_attention[route, 0]).values,
            )
        assert torch.equal(
            repeat_outputs[3],
            expected_resident_lengths,
        ), f"MTP steady resident lengths changed at repeat {repeat_index}"
        assert repeat_outputs[-1].cpu().tolist() == [0]
        assert torch.equal(cache_slots, pool_before_repeat)


@pytest.mark.parametrize("heads", [32, 64])
def test_packed_c8_mtp_lim_steady_fast_path_has_exact_monotonic_topk(
    heads: int,
) -> None:
    budget = 8192
    candidate_len = budget
    final_len = candidate_len + 4
    queries_per_request = 4
    inputs = _make_fused_lidu_inputs(
        candidate_len=candidate_len,
        final_len=final_len,
        row_mode=2,
        batch=queries_per_request,
        seed=67,
        heads=heads,
    )
    inputs["query_ends"] = torch.tensor(
        [queries_per_request],
        dtype=torch.int32,
        device="npu",
    )
    for name in (
        "block_table",
        "candidate_lens",
        "final_lens",
        "row_modes",
        "req_entries",
    ):
        inputs[name] = inputs[name][:1]

    # QK=128 and the head reduction is heads*128, both powers of two.
    # Scaling consecutive positive BF16 values by that factor therefore
    # produces 8192 unique, exactly ordered final BF16 scores without ties.
    inputs["query"] = torch.ones(
        tuple(inputs["query"].shape),
        dtype=torch.bfloat16,
        device="npu",
    ).to(torch.float8_e4m3fn)
    inputs["key"] = torch.ones(
        tuple(inputs["key"].shape),
        dtype=torch.bfloat16,
        device="npu",
    ).to(torch.float8_e4m3fn)
    inputs["weights"].fill_(1)
    inputs["query_scale"].fill_(1)
    score_bits = torch.arange(
        0x2000,
        0x2000 + candidate_len,
        dtype=torch.int32,
    ).to(torch.int16)
    desired_scores = score_bits.view(torch.bfloat16).to(torch.float32)
    key_scales = desired_scores / float(heads * 128)
    inputs["key_scale"].view(-1).copy_(key_scales.to(device="npu"))

    cache_slots = torch.full(
        (2, _cache_row_width(final_len)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, :candidate_len] = torch.arange(
        candidate_len,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[0, -1] = budget
    before = cache_slots.clone()
    outputs = _allocate_mtp_lidu_outputs(
        batch=1,
        total_query_rows=queries_per_request,
    )

    _launch_mtp_lim(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, _, _, resident_lengths, _, _, counts = (tensor.cpu() for tensor in outputs)
    expected_topk = torch.arange(
        candidate_len - _TOPK,
        candidate_len,
        dtype=torch.int32,
    )
    for route in range(queries_per_request):
        actual_topk = attention[route, 0, :_TOPK]
        sorted_actual = torch.sort(actual_topk).values
        if not torch.equal(sorted_actual, expected_topk):
            missing = expected_topk[~torch.isin(expected_topk, actual_topk)]
            extra = actual_topk[~torch.isin(actual_topk, expected_topk)]
            pytest.fail(
                "MTP LIM monotonic TopK mismatch: "
                f"heads={heads}, route={route}, "
                f"unique={int(torch.unique(actual_topk).numel())}, "
                f"actual_min={int(sorted_actual[0])}, "
                f"actual_max={int(sorted_actual[-1])}, "
                f"missing_count={int(missing.numel())}, "
                f"extra_count={int(extra.numel())}, "
                f"missing_prefix={missing[:16].tolist()}, "
                f"extra_prefix={extra[:16].tolist()}"
            )
        visible_len = final_len - (queries_per_request - 1 - route)
        tail_tokens = torch.arange(
            candidate_len,
            visible_len,
            dtype=torch.int32,
        )
        expected_tail = budget + tail_tokens.remainder(2 * _BLOCK_SIZE)
        tail_len = int(tail_tokens.numel())
        _assert_exact_int_tensor(
            attention[route, 0, _TOPK : _TOPK + tail_len],
            expected_tail,
        )
        assert torch.all(attention[route, 0, _TOPK + tail_len :] == -1)

    assert counts.tolist() == [0]
    assert resident_lengths.tolist() == [budget + 2 * _BLOCK_SIZE]
    assert torch.equal(cache_slots, before)


@pytest.mark.parametrize("budget", [6144, 10240, 12288])
def test_packed_c8_fused_lidu_sparse_first_fill_then_steady(
    budget: int,
) -> None:
    candidate_len = budget + _BLOCK_SIZE
    inputs = _make_fused_lidu_inputs(
        candidate_len=candidate_len,
        final_len=candidate_len + 1,
        row_mode=2,
        seed=19,
    )
    expected_topk = _native_topk(inputs).cpu()[0, 0]
    cache_slots = torch.full(
        (2, _cache_row_width(16384)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, -1] = -budget
    outputs = _allocate_fused_lidu_outputs()

    _launch_fused_lidu(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    pool = cache_slots.cpu()[0]
    attention, resident_lengths, source_ids, destination_slots, counts = (tensor.cpu() for tensor in outputs)
    assert counts.tolist() == [budget]
    assert resident_lengths.tolist() == [budget + 1]
    assert int(pool[-1]) == budget
    valid_sources = source_ids[0, 0, :budget]
    valid_destinations = destination_slots[0, 0, :budget]
    assert torch.unique(valid_sources).numel() == budget
    _assert_exact_int_tensor(
        valid_destinations,
        torch.arange(budget, dtype=torch.int32),
    )
    _assert_exact_int_tensor(
        pool[valid_sources.to(torch.int64)],
        valid_destinations,
    )
    _assert_exact_int_tensor(
        attention[0, 0, :_TOPK],
        pool[expected_topk.to(torch.int64)],
    )
    assert int(attention[0, 0, _TOPK]) == budget
    assert torch.all(attention[0, 0, _TOPK + 1 :] == -1)

    expected_attention = outputs[0].clone()
    expected_resident_lengths = outputs[1].clone()
    pool_before_repeat = cache_slots.clone()
    for repeat_index in range(8):
        repeat_outputs = _allocate_fused_lidu_outputs()
        _launch_fused_lidu(inputs, cache_slots, repeat_outputs)
        torch.npu.synchronize()
        assert torch.equal(
            repeat_outputs[0],
            expected_attention,
        ), f"steady attention changed at repeat {repeat_index}"
        assert torch.equal(
            repeat_outputs[1],
            expected_resident_lengths,
        ), f"steady resident lengths changed at repeat {repeat_index}"
        assert repeat_outputs[-1].cpu().tolist() == [0]
        assert torch.equal(cache_slots, pool_before_repeat)


def test_packed_c8_fused_lidu_mixed_dense_sparse_and_pad_rows() -> None:
    batch = 3
    dense_len = 2177
    sparse_candidate_len = 6272
    sparse_final_len = 6273
    inputs = _make_fused_lidu_inputs(
        candidate_len=sparse_candidate_len,
        final_len=sparse_final_len,
        row_mode=2,
        batch=batch,
        seed=23,
    )
    inputs["candidate_lens"].copy_(
        torch.tensor(
            [dense_len, sparse_candidate_len, _TOPK],
            dtype=torch.int32,
            device="npu",
        )
    )
    inputs["final_lens"].copy_(
        torch.tensor(
            [dense_len, sparse_final_len, 1],
            dtype=torch.int32,
            device="npu",
        )
    )
    inputs["row_modes"].copy_(torch.tensor([1, 2, 0], dtype=torch.int32, device="npu"))
    inputs["req_entries"].copy_(torch.tensor([-1, 0, -1], dtype=torch.int32, device="npu"))
    expected_topk = _native_topk(inputs).cpu()
    cache_slots = torch.full(
        (2, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, -1] = -6144
    outputs = _allocate_fused_lidu_outputs(batch)

    _launch_fused_lidu(inputs, cache_slots, outputs)
    torch.npu.synchronize()

    attention, resident_lengths, _, _, counts = (tensor.cpu() for tensor in outputs)
    _assert_exact_int_tensor(
        attention[0, :, :_TOPK],
        expected_topk[0],
    )
    assert resident_lengths.tolist() == [dense_len, 6145, 0]
    assert counts.tolist() == [0, 6144, 0]
    assert torch.all(attention[2] == -1)


def test_packed_c8_fused_lidu_graph_replay_changes_pool_state() -> None:
    inputs = _make_fused_lidu_inputs(
        candidate_len=6272,
        final_len=6273,
        row_mode=2,
        seed=31,
    )
    initial_pool = torch.full(
        (2, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    initial_pool[:, -1].zero_()
    initial_pool[0, -1] = -6144

    expected_pool = initial_pool.clone()
    expected_outputs = _allocate_fused_lidu_outputs()
    _launch_fused_lidu(inputs, expected_pool, expected_outputs)
    torch.npu.synchronize()
    expected_pool_cpu = expected_pool.cpu()
    expected_outputs_cpu = tuple(tensor.cpu() for tensor in expected_outputs)

    graph_pool_state = expected_pool.clone()
    graph_outputs = _allocate_fused_lidu_outputs()
    stable_pointers = (
        graph_pool_state.data_ptr(),
        *(tensor.data_ptr() for tensor in graph_outputs),
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        _launch_fused_lidu(inputs, graph_pool_state, graph_outputs)
    # A5 graph capture records the launch but does not guarantee one eager
    # execution. Replay explicitly before validating the resident steady state.
    graph.replay()
    torch.npu.synchronize()
    assert graph_outputs[-1].cpu().tolist() == [0]

    graph_pool_state.copy_(initial_pool)
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(graph_pool_state.cpu(), expected_pool_cpu)
    for actual, expected in zip(graph_outputs, expected_outputs_cpu):
        assert torch.equal(actual.cpu(), expected)
    assert stable_pointers == (
        graph_pool_state.data_ptr(),
        *(tensor.data_ptr() for tensor in graph_outputs),
    )


def test_packed_c8_fused_lidu_graph_replay_pad_row_is_noop() -> None:
    inputs = _make_fused_lidu_inputs(
        candidate_len=6272,
        final_len=6273,
        row_mode=2,
        seed=37,
    )
    cache_slots = torch.full(
        (2, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    outputs = _allocate_fused_lidu_outputs()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        _launch_fused_lidu(inputs, cache_slots, outputs)

    cache_slots.copy_(
        torch.arange(
            cache_slots.numel(),
            dtype=torch.int32,
            device="npu",
        ).reshape_as(cache_slots)
    )
    cache_before = cache_slots.clone()
    inputs["row_modes"].zero_()
    inputs["req_entries"].fill_(-1)
    for output in outputs:
        output.fill_(17)

    graph.replay()
    torch.npu.synchronize()

    attention, resident_lengths, _, _, counts = outputs
    assert torch.equal(cache_slots, cache_before)
    assert resident_lengths.cpu().tolist() == [0]
    assert counts.cpu().tolist() == [0]
    assert torch.all(attention == -1)


@pytest.mark.parametrize("routes", [1, 4])
def test_packed_c8_full_shared_next_full_graph_chain(routes) -> None:
    budget = 8192 if routes > 1 else 6144
    launch = _launch_mtp_lim if routes > 1 else _launch_fused_lidu
    allocate = _allocate_mtp_lidu_outputs if routes > 1 else _allocate_fused_lidu_outputs
    resident_index = 3 if routes > 1 else 1
    candidate_len = budget + _BLOCK_SIZE
    blocks = max(candidate_len, budget + 2 * _BLOCK_SIZE) // _BLOCK_SIZE
    full_inputs = (
        _make_fused_lidu_inputs(
            candidate_len=candidate_len,
            final_len=candidate_len + (130 if routes > 1 else 1),
            batch=routes,
            row_mode=2,
            seed=41,
        ),
        _make_fused_lidu_inputs(
            candidate_len=candidate_len,
            final_len=candidate_len + (130 if routes > 1 else 1),
            batch=routes,
            row_mode=2,
            seed=43,
        ),
    )
    if routes > 1:
        for inputs in full_inputs:
            for name in ("block_table", "candidate_lens", "final_lens", "row_modes", "req_entries"):
                inputs[name] = inputs[name][:1]
            inputs["query_ends"] = torch.tensor([routes], dtype=torch.int32, device="npu")
    initial_selection_states = torch.full(
        (2, 2, _cache_row_width(candidate_len + 256)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    initial_selection_states[:, :, -1].zero_()
    initial_selection_states[:, 0, -1] = -budget

    expected_states = initial_selection_states.clone()
    expected_outputs = (
        allocate(total_query_rows=routes),
        allocate(total_query_rows=routes),
    )
    for state_id in range(2):
        launch(
            full_inputs[state_id],
            expected_states[state_id],
            expected_outputs[state_id],
        )
    torch.npu.synchronize()
    assert [int(outputs[-1].cpu()[0]) for outputs in expected_outputs] == [
        budget,
        budget,
    ]
    assert not torch.equal(
        expected_outputs[0][-3][:, :, :budget],
        expected_outputs[1][-3][:, :, :budget],
    )

    block_table = torch.arange(
        blocks,
        dtype=torch.int32,
        device="npu",
    ).view(1, -1)
    dram_cpu_layers = []
    dram_layers = []
    hbm_layers = []
    for layer_id in range(3):
        dram_cpu = _make_unique_packed_rows(blocks * _BLOCK_SIZE)
        dram_cpu[:, 3] = layer_id + 1
        dram_cpu = dram_cpu.view(
            blocks,
            _BLOCK_SIZE,
            1,
            _PACKED_ROW_BYTES,
        )
        dram = _swapped_arena(tuple(dram_cpu.shape))
        _write_swapped_arena(dram, dram_cpu)
        dram_cpu_layers.append(dram_cpu)
        dram_layers.append(dram)
        hbm_layers.append(torch.zeros_like(dram_cpu, device="npu"))

    graph_states = initial_selection_states.clone()
    shared_scratch = allocate(total_query_rows=routes)
    shared_attention_snapshot = torch.empty_like(shared_scratch[0])
    shared_length_snapshot = torch.empty_like(shared_scratch[resident_index])
    stable_pointers = (
        graph_states.data_ptr(),
        *(tensor.data_ptr() for tensor in shared_scratch),
        shared_attention_snapshot.data_ptr(),
        shared_length_snapshot.data_ptr(),
        *(tensor.data_ptr() for tensor in hbm_layers),
    )
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        # full 0 produces one complete selection plan. shared 1 consumes the
        # same scratch but copies from its own DRAM arena into its own HBM.
        launch(
            full_inputs[0],
            graph_states[0],
            shared_scratch,
        )
        _launch_packed_ksc(
            hbm_layers[0],
            dram_layers[0],
            block_table,
            shared_scratch,
        )
        _launch_packed_ksc(
            hbm_layers[1],
            dram_layers[1],
            block_table,
            shared_scratch,
        )
        # Snapshot the attention metadata consumed by the shared SFA path so
        # the following full layer cannot hide an early scratch overwrite.
        shared_attention_snapshot.copy_(shared_scratch[0])
        shared_length_snapshot.copy_(shared_scratch[resident_index])
        # The next full layer owns a different compact selection state and may
        # overwrite the shared scratch only after the follower consumed it.
        launch(
            full_inputs[1],
            graph_states[1],
            shared_scratch,
        )
        _launch_packed_ksc(
            hbm_layers[2],
            dram_layers[2],
            block_table,
            shared_scratch,
        )

    graph_states.copy_(initial_selection_states)
    for hbm in hbm_layers:
        hbm.zero_()
    graph.replay()
    torch.npu.synchronize()

    assert torch.equal(graph_states.cpu(), expected_states.cpu())
    assert torch.equal(
        shared_attention_snapshot.cpu(),
        expected_outputs[0][0].cpu(),
    )
    assert torch.equal(
        shared_length_snapshot.cpu(),
        expected_outputs[0][resident_index].cpu(),
    )
    for actual, expected in zip(shared_scratch, expected_outputs[1]):
        assert torch.equal(actual.cpu(), expected.cpu())
    assert stable_pointers == (
        graph_states.data_ptr(),
        *(tensor.data_ptr() for tensor in shared_scratch),
        shared_attention_snapshot.data_ptr(),
        shared_length_snapshot.data_ptr(),
        *(tensor.data_ptr() for tensor in hbm_layers),
    )

    # Layers 0 and 1 consume full 0's plan; layer 2 consumes full 2's plan.
    source_plans = (expected_outputs[0], expected_outputs[0], expected_outputs[1])
    probe_indices = (0, budget // 2, budget - 1)
    for layer_id, outputs in enumerate(source_plans):
        source_ids = outputs[-3][0, 0, :budget].cpu().to(torch.int64)
        destination_slots = outputs[-2][0, 0, :budget].cpu().to(torch.int64)
        hbm_rows = hbm_layers[layer_id].cpu().view(-1, _PACKED_ROW_BYTES)
        dram_rows = dram_cpu_layers[layer_id].view(-1, _PACKED_ROW_BYTES)
        for probe_index in probe_indices:
            assert torch.equal(
                hbm_rows[destination_slots[probe_index]],
                dram_rows[source_ids[probe_index]],
            )


def test_packed_c8_fused_lidu_copy_plan_is_consumed_by_ksc() -> None:
    budget = 6144
    candidate_len = 6272
    blocks = candidate_len // _BLOCK_SIZE
    inputs = _make_fused_lidu_inputs(
        candidate_len=candidate_len,
        final_len=candidate_len + 1,
        row_mode=2,
        seed=37,
    )
    cache_slots = torch.full(
        (2, _cache_row_width(8192)),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    cache_slots[:, -1].zero_()
    cache_slots[0, -1] = -budget
    outputs = _allocate_fused_lidu_outputs()
    _launch_fused_lidu(inputs, cache_slots, outputs)

    dram_cpu = _make_unique_packed_rows(blocks * _BLOCK_SIZE).view(
        blocks,
        _BLOCK_SIZE,
        1,
        _PACKED_ROW_BYTES,
    )
    dram = _swapped_arena(tuple(dram_cpu.shape))
    _write_swapped_arena(dram, dram_cpu)
    hbm = torch.zeros_like(dram_cpu, device="npu")
    block_table = torch.arange(
        blocks,
        dtype=torch.int32,
        device="npu",
    ).view(1, -1)
    _launch_packed_ksc(hbm, dram, block_table, outputs)
    torch.npu.synchronize()

    source_ids = outputs[2][0, 0, :budget].cpu().to(torch.int64)
    destination_slots = outputs[3][0, 0, :budget].cpu().to(torch.int64)
    assert outputs[4].cpu().tolist() == [budget]
    actual_rows = hbm.cpu().view(-1, _PACKED_ROW_BYTES)[destination_slots]
    expected_rows = dram_cpu.view(-1, _PACKED_ROW_BYTES)[source_ids]
    assert torch.equal(actual_rows, expected_rows)


@pytest.mark.parametrize("counts", [(0, 0, 0), (0, 1, 3), (2, 127, 129), (1024, 65, 128)])
@pytest.mark.parametrize("shared_source", [False, True])
def test_packed_c8_ksc_ignores_only_scratch_after_copy_prefix(counts, shared_source) -> None:
    """Zero/mixed rows and queue tails must not read poisoned unused metadata."""
    batch, blocks_per_request, capacity = 3, 8, 1024
    physical_blocks = batch * blocks_per_request
    poison = -91
    dram_cpu = _make_unique_packed_rows(physical_blocks * _BLOCK_SIZE).view(
        physical_blocks, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES
    )
    dram = _swapped_arena(tuple(dram_cpu.shape))
    _write_swapped_arena(dram, dram_cpu)
    hbm = torch.full(
        (physical_blocks + 1, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES),
        poison,
        dtype=torch.int8,
        device="npu",
    )
    dram_table = torch.arange(physical_blocks, dtype=torch.int32).view(batch, -1)
    hbm_table = dram_table.flip(1).contiguous()
    if shared_source:
        # Local prefix reuse shares read-only DRAM, never writable resident KV.
        dram_table[:] = dram_table[0].clone()
    sources = torch.full((batch, 1, _COPY_CAPACITY), -1, dtype=torch.int32)
    destinations = torch.full_like(sources, -1)
    expected = torch.full(tuple(hbm.shape), poison, dtype=torch.int8)
    for row, count in enumerate(counts):
        tokens = torch.arange(count, dtype=torch.int32).mul(127).remainder(capacity)
        slots = torch.arange(count, dtype=torch.int32)
        sources[row, 0, :count] = tokens
        destinations[row, 0, :count] = slots
        for token, slot in zip(tokens.tolist(), slots.tolist()):
            expected[int(hbm_table[row, slot // _BLOCK_SIZE]), slot % _BLOCK_SIZE] = dram_cpu[
                int(dram_table[row, token // _BLOCK_SIZE]), token % _BLOCK_SIZE
            ]
    pointers = (hbm.data_ptr(), dram.data_ptr())
    args = (
        hbm,
        dram,
        hbm_table.to("npu"),
        dram_table.to("npu"),
        sources.to("npu"),
        destinations.to("npu"),
        torch.tensor(counts, dtype=torch.int32, device="npu"),
    )
    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(*args)
    torch.npu.synchronize()
    assert torch.equal(hbm.cpu(), expected)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle()):
        torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(*args)
    for _ in range(3):
        hbm.fill_(poison)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(hbm.cpu(), expected)
        assert pointers == (hbm.data_ptr(), dram.data_ptr())
    assert torch.equal(_read_swapped_arena(dram), dram_cpu)


def test_packed_c8_ksc_copies_one_opaque_row() -> None:
    hbm = torch.zeros(
        (17, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES),
        dtype=torch.int8,
        device="npu",
    )
    dram_cpu = torch.zeros(
        (16, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES),
        dtype=torch.int8,
    )
    expected = torch.arange(_PACKED_ROW_BYTES, dtype=torch.int16).remainder(256).sub(128).to(torch.int8)
    dram_cpu[1, 1, 0] = expected
    dram = _swapped_arena(tuple(dram_cpu.shape))
    _write_swapped_arena(dram, dram_cpu)
    hbm_table = torch.arange(17, dtype=torch.int32, device="npu").view(1, -1)
    dram_table = torch.arange(16, dtype=torch.int32, device="npu").view(1, -1)
    source_ids = torch.full(
        (1, 1, _COPY_CAPACITY),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    destination_slots = torch.full_like(source_ids, -1)
    source_ids[0, 0, 0] = 129
    destination_slots[0, 0, 0] = 3
    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
        hbm,
        dram,
        hbm_table,
        dram_table,
        source_ids,
        destination_slots,
        torch.tensor([1], dtype=torch.int32, device="npu"),
    )
    torch.npu.synchronize()

    assert torch.equal(hbm[0, 3, 0].cpu(), expected)


def _make_copysfa_chain_case(query_counts, modes, *, miss_count=300, first_fill=False, tail=64):
    """Real SIMT LIM inputs; attention golden uses original logical KV + native LI."""
    candidate, budget = 12288, 8192
    batch, total = len(query_counts), sum(query_counts)
    inputs = _make_disjoint_mtp_lidu_inputs(
        query_counts=query_counts,
        max_candidate=candidate,
        heads=32,
        seed=193,
    )
    inputs["row_modes"].copy_(torch.tensor(modes, dtype=torch.int32, device="npu"))
    inputs["final_lens"].fill_(candidate + tail)
    for b, mode in enumerate(modes):
        if mode == 0:
            inputs["candidate_lens"][b] = 0
            inputs["final_lens"][b] = 0
            inputs["req_entries"][b] = -1
    topk = _native_mtp_topk_rows(inputs)
    pool_cpu = torch.full((batch + 1, _cache_row_width(candidate + 256)), -1, dtype=torch.int32)
    pool_cpu[:, -1].zero_()
    cached = []
    begin = 0
    for b, (routes, mode) in enumerate(zip(query_counts, modes)):
        if mode == 2:
            union = torch.cat(topk[begin : begin + routes]).unique().long()
            missing = union[:miss_count]
            kept = union[miss_count:]
            fillers = torch.arange(candidate)[~torch.isin(torch.arange(candidate), union)]
            tokens = torch.cat((kept, fillers))[:budget]
            if first_fill and b == batch - 1:
                pool_cpu[b, -1] = -budget
                tokens = tokens[:0]
            else:
                pool_cpu[b, tokens] = torch.arange(budget, dtype=torch.int32)
                pool_cpu[b, -1] = budget
                assert not torch.isin(missing, tokens).any()
            cached.append(tokens)
        else:
            cached.append(torch.empty(0, dtype=torch.long))
        begin += routes
    pool_initial = pool_cpu.to("npu")
    pool = pool_initial.clone()
    outputs = _allocate_mtp_lidu_outputs(batch, total)
    _launch_mtp_lim(inputs, pool, outputs)
    torch.npu.synchronize()
    _assert_mtp_selection(inputs, pool_initial, pool, outputs, topk)

    generator = torch.Generator().manual_seed(197)
    blocks = (candidate + 256) // _BLOCK_SIZE + 1  # one guard block per request
    table = torch.stack([b * blocks + torch.randperm(blocks, generator=generator).int() for b in range(batch)])
    dram_table = torch.stack([b * blocks + torch.randperm(blocks, generator=generator).int() for b in range(batch)])
    logical = torch.zeros((batch, blocks * _BLOCK_SIZE, _PACKED_ROW_BYTES), dtype=torch.int8)
    logical[:, :, :512] = (
        torch.randint(
            -4,
            5,
            (batch, blocks * _BLOCK_SIZE, 512),
            generator=generator,
        )
        .float()
        .to(torch.float8_e4m3fn)
        .view(torch.int8)
    )
    logical[:, :, 512:640] = (
        torch.empty(batch, blocks * _BLOCK_SIZE, 64)
        .uniform_(
            -0.5,
            0.5,
            generator=generator,
        )
        .to(torch.bfloat16)
        .view(torch.int8)
    )
    logical[:, :, 640:] = (
        torch.empty(batch, blocks * _BLOCK_SIZE, 4)
        .uniform_(
            0.02,
            0.08,
            generator=generator,
        )
        .view(torch.int8)
    )
    initial = torch.full((batch * blocks * _BLOCK_SIZE, _PACKED_ROW_BYTES), -91, dtype=torch.int8)
    dram_cpu = torch.empty_like(initial)

    def physical(bt, b, ids):
        return bt[b, ids // _BLOCK_SIZE].long() * _BLOCK_SIZE + ids % _BLOCK_SIZE

    all_ids = torch.arange(blocks * _BLOCK_SIZE)
    for b, mode in enumerate(modes):
        dram_cpu[physical(dram_table, b, all_ids)] = logical[b]
        if mode == 1:
            ids = torch.arange(candidate + tail)
            initial[physical(table, b, ids)] = logical[b, ids]
        if mode == 2:
            tokens = cached[b]
            initial[physical(table, b, torch.arange(len(tokens)))] = logical[b, tokens]
            ids = torch.arange(candidate, candidate + tail)
            initial[physical(table, b, budget + ids % 256)] = logical[b, ids]
    expected = initial.clone()
    sources, destinations, counts = (tensor.cpu() for tensor in outputs[4:])
    for b, count in enumerate(counts.tolist()):
        ids, dst = sources[b, 0, :count].long(), destinations[b, 0, :count].long()
        initial[physical(table, b, dst)] = -91
        expected[physical(table, b, dst)] = logical[b, ids]
    query = torch.randn(total, 64, 576, generator=generator).bfloat16()
    golden = torch.zeros(total, 64, 512)
    begin = 0
    for b, (routes, mode) in enumerate(zip(query_counts, modes)):
        for row in range(begin, begin + routes):
            if mode == 0:
                continue
            ids = topk[row].long()
            if mode == 2:
                visible = candidate + tail - (begin + routes - 1 - row)
                ids = torch.cat((ids, torch.arange(candidate, visible)))
            raw = logical[b, ids].contiguous()
            latent = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
            scales = raw[:, 640:].contiguous().view(torch.float32).repeat_interleave(128, -1)
            value = (latent * scales).bfloat16().float()
            rope = raw[:, 512:640].contiguous().view(torch.bfloat16).float()
            key = torch.cat((value, rope), -1)
            golden[row] = (query[row].float() @ key.T * 576**-0.5).softmax(-1) @ value
        begin += routes
    shape = (batch * blocks, _BLOCK_SIZE, 1, _PACKED_ROW_BYTES)
    dram = _swapped_arena(shape)
    _write_swapped_arena(dram, dram_cpu.view(shape))
    tensors = dict(
        inputs=inputs,
        pool=pool,
        outputs=outputs,
        query=query.to("npu"),
        hbm=initial.view(shape).to("npu"),
        dram=dram,
        htable=table.to("npu"),
        dtable=dram_table.to("npu"),
        out=torch.empty(total, 64, 512, dtype=torch.bfloat16, device="npu"),
    )
    return dict(
        tensors=tensors,
        pool_initial=pool_initial,
        initial=tensors["hbm"].clone(),
        expected=expected.view(shape),
        golden=golden,
        topk=topk,
    )


def _launch_copysfa_chain(state):
    d = state["tensors"]
    _launch_mtp_lim(d["inputs"], d["pool"], d["outputs"])
    slots, sources, misses, resident, cs, cd, counts = d["outputs"]
    torch.ops._C_ascend.npu_dsa_a5_fused_copy_sfa_c8_out(
        d["query"],
        d["hbm"],
        d["dram"],
        d["htable"],
        d["dtable"],
        slots,
        sources,
        misses,
        d["inputs"]["query_ends"],
        resident,
        cs,
        cd,
        counts,
        576**-0.5,
        d["out"],
    )


def _check_copysfa_chain(state, reference):
    torch.npu.synchronize()
    d = state["tensors"]
    _assert_mtp_selection(d["inputs"], reference["pool_initial"], d["pool"], d["outputs"], reference["topk"])
    _assert_exact_int_tensor(d["hbm"].cpu(), reference["expected"])
    actual, golden = d["out"].cpu().float(), reference["golden"]
    assert torch.isfinite(actual).all()
    # Same strict fused tolerances as the operator repository. No SFA XFAIL.
    torch.testing.assert_close(actual, golden, atol=0.008, rtol=0.05)
    relative = (actual - golden).flatten(1).norm(dim=1) / golden.flatten(1).norm(dim=1).clamp_min(1e-7)
    assert (relative <= 0.08).all()


def _copysfa_reset(state, reference):
    d, src = state["tensors"], reference["tensors"]
    for name, tensor in d["inputs"].items():
        tensor.copy_(src["inputs"][name])
    for name in ("query", "htable", "dtable"):
        d[name].copy_(src[name])
    d["pool"].copy_(reference["pool_initial"])
    d["hbm"].copy_(reference["initial"])
    d["out"].fill_(float("nan"))


def test_packed_c8_copysfa_simt_chain_eager_and_dynamic_graph():
    require_dsa_offload_ops(packed_c8=True, mtp=True, fused_copy_sfa=True)
    # Every state has identical allocation shapes but a different device-side
    # lifecycle/query distribution. Independent physical blocks for each request.
    states = [
        _make_copysfa_chain_case([1, 2, 3, 4], [2, 2, 2, 2], miss_count=0),
        _make_copysfa_chain_case([1, 2, 3, 4], [2, 2, 2, 2], miss_count=300),
        _make_copysfa_chain_case([4, 1, 2, 3], [0, 1, 2, 2], miss_count=512, tail=129),
        _make_copysfa_chain_case([1, 2, 3, 4], [2, 2, 2, 2], miss_count=513),
        _make_copysfa_chain_case([4, 1, 2, 3], [0, 1, 2, 2], first_fill=True, tail=256),
    ]
    # Keep references immutable while overwriting the captured input buffers.
    active = _make_copysfa_chain_case([1, 2, 3, 4], [2, 2, 2, 2], miss_count=0)
    for reference in states:
        _copysfa_reset(active, reference)
        _launch_copysfa_chain(active)
        _check_copysfa_chain(active, reference)
    _copysfa_reset(active, states[0])
    _launch_copysfa_chain(active)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _launch_copysfa_chain(active)
    d = active["tensors"]
    owners = (*d["inputs"].values(), d["pool"], *d["outputs"], d["hbm"], d["out"])
    pointers = tuple(t.data_ptr() for t in owners)
    for reference in (*states[1:], states[1]):
        for _ in range(3):
            _copysfa_reset(active, reference)
            graph.replay()
            _check_copysfa_chain(active, reference)
            assert pointers == tuple(t.data_ptr() for t in owners)


def test_packed_c8_copysfa_workspace_graph_stream_isolation():
    require_dsa_offload_ops(packed_c8=True, mtp=True, fused_copy_sfa=True)
    states = [
        _make_copysfa_chain_case([4], [2], miss_count=300),
        _make_copysfa_chain_case([2, 4], [2, 2], miss_count=500),
        _make_copysfa_chain_case([4, 4, 4, 4], [2, 2, 2, 2], miss_count=300),
    ]
    capture_stream = torch.npu.Stream()
    capture_stream.wait_stream(torch.npu.current_stream())
    graphs = []
    for state in states[:2]:
        with torch.npu.stream(capture_stream):
            _copysfa_reset(state, state)
            _launch_copysfa_chain(state)
        capture_stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=capture_stream):
            _launch_copysfa_chain(state)
        graphs.append(graph)
    with torch.npu.stream(capture_stream):
        _copysfa_reset(states[2], states[2])
        _launch_copysfa_chain(states[2])  # grow eager workspace after graph capture
    torch.npu.synchronize()
    torch.npu.empty_cache()
    streams = [torch.npu.Stream(), torch.npu.Stream(), capture_stream]
    for _ in range(3):
        for index, (state, stream) in enumerate(zip(states, streams)):
            with torch.npu.stream(stream):
                _copysfa_reset(state, state)
                if index < len(graphs):
                    graphs[index].replay()
                else:
                    _launch_copysfa_chain(state)
        torch.npu.synchronize()
        for state in states:
            _check_copysfa_chain(state, state)


def test_packed_c8_ksc_honors_permuted_l2_and_hbm_block_tables() -> None:
    torch.manual_seed(1024)
    dram_logical_blocks = 325
    dram_physical_blocks = 326
    hbm_physical_blocks = 407
    copy_count = 10240
    dram_cpu = _make_unique_packed_rows(
        dram_physical_blocks * _BLOCK_SIZE
    ).view(
        dram_physical_blocks,
        _BLOCK_SIZE,
        1,
        _PACKED_ROW_BYTES,
    )
    dram = _swapped_arena(tuple(dram_cpu.shape))
    _write_swapped_arena(dram, dram_cpu)
    hbm = torch.full(
        (
            hbm_physical_blocks,
            _BLOCK_SIZE,
            1,
            _PACKED_ROW_BYTES,
        ),
        73,
        dtype=torch.int8,
        device="npu",
    )
    dram_table_cpu = torch.arange(
        1,
        dram_logical_blocks + 1,
        dtype=torch.int32,
    ).view(1, -1)
    hbm_table_cpu = _recycled_resident_block_ids().view(1, -1)
    source_tokens_cpu = torch.randperm(
        dram_logical_blocks * _BLOCK_SIZE,
        dtype=torch.int32,
    )[:copy_count]
    destination_slots_cpu = torch.arange(
        copy_count,
        dtype=torch.int32,
    )
    source_tokens = torch.full(
        (1, 1, _COPY_CAPACITY),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    destination_slots = torch.full_like(source_tokens, -1)
    source_tokens[0, 0, :copy_count] = source_tokens_cpu.to("npu")
    destination_slots[0, 0, :copy_count] = (
        destination_slots_cpu.to("npu")
    )

    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
        hbm,
        dram,
        hbm_table_cpu.to("npu"),
        dram_table_cpu.to("npu"),
        source_tokens,
        destination_slots,
        torch.tensor(
            [copy_count],
            dtype=torch.int32,
            device="npu",
        ),
    )
    torch.npu.synchronize()

    source_logical_blocks = source_tokens_cpu // _BLOCK_SIZE
    source_offsets = source_tokens_cpu % _BLOCK_SIZE
    source_physical_blocks = dram_table_cpu[
        0,
        source_logical_blocks.to(torch.int64),
    ].to(torch.int64)
    destination_logical_blocks = destination_slots_cpu // _BLOCK_SIZE
    destination_offsets = destination_slots_cpu % _BLOCK_SIZE
    destination_physical_blocks = hbm_table_cpu[
        0,
        destination_logical_blocks.to(torch.int64),
    ].to(torch.int64)
    actual = hbm.cpu()[
        destination_physical_blocks,
        destination_offsets.to(torch.int64),
        0,
    ]
    expected = dram_cpu[
        source_physical_blocks,
        source_offsets.to(torch.int64),
        0,
    ]
    assert torch.equal(actual, expected)


def test_packed_c8_qsfa_honors_permuted_block_table_and_is_repeatable() -> None:
    torch.manual_seed(1024)
    logical_blocks = 81
    physical_capacity = 1625
    block_table_width = 1024
    resident_seq_len = 10314
    query = torch.randn(
        (1, 64, 512 + 64),
        dtype=torch.bfloat16,
        device="npu",
    )
    key_nope = torch.randn(
        (logical_blocks, _BLOCK_SIZE, 1, 512),
        dtype=torch.float32,
    ).clamp_(-4.0, 4.0).to(torch.float8_e4m3fn).view(torch.int8)
    key_rope = torch.randn(
        (logical_blocks, _BLOCK_SIZE, 1, 64),
        dtype=torch.bfloat16,
    )
    scales = torch.rand(
        (logical_blocks, _BLOCK_SIZE, 1, 4),
        dtype=torch.float32,
    ).add_(0.01)
    logical_packed_kv = torch.cat(
        (
            key_nope,
            key_rope.contiguous().view(torch.int8),
            scales.contiguous().view(torch.int8),
        ),
        dim=-1,
    )
    identity_packed_kv = torch.full(
        (
            physical_capacity,
            _BLOCK_SIZE,
            1,
            _PACKED_ROW_BYTES,
        ),
        73,
        dtype=torch.int8,
    )
    fresh_block_ids = _fresh_resident_block_ids().to(torch.int64)
    identity_packed_kv[fresh_block_ids] = logical_packed_kv
    identity_packed_kv = identity_packed_kv.to("npu").view(
        torch.float8_e4m3fn
    )
    recycled_block_ids = _recycled_resident_block_ids().to(torch.int64)
    permuted_packed_kv = torch.full(
        (
            physical_capacity,
            _BLOCK_SIZE,
            1,
            _PACKED_ROW_BYTES,
        ),
        73,
        dtype=torch.int8,
    )
    permuted_packed_kv[recycled_block_ids] = logical_packed_kv
    permuted_packed_kv = permuted_packed_kv.to("npu").view(
        torch.float8_e4m3fn
    )
    sparse_indices = torch.randperm(
        resident_seq_len,
        dtype=torch.int32,
    )[:_ATTENTION_CAPACITY].view(1, 1, _ATTENTION_CAPACITY).to("npu")
    identity_block_table = torch.zeros(
        (1, block_table_width),
        dtype=torch.int32,
        device="npu",
    )
    identity_block_table[0, :logical_blocks] = fresh_block_ids.to(
        dtype=torch.int32,
        device="npu",
    )
    permuted_block_table = torch.zeros_like(identity_block_table)
    permuted_block_table[0, :logical_blocks] = recycled_block_ids.to(
        dtype=torch.int32,
        device="npu",
    )
    query_lengths = torch.tensor([1], dtype=torch.int32, device="npu")
    resident_lengths = torch.tensor(
        [resident_seq_len],
        dtype=torch.int32,
        device="npu",
    )

    def run_qsfa(
        packed_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        return sparse_flash_attention_for_offload_c8(
            query=query,
            packed_kv=packed_kv,
            sparse_indices=sparse_indices,
            scale_value=(512 + 64) ** -0.5,
            block_table=block_table,
            actual_seq_lengths_query=query_lengths,
            resident_seq_lengths=resident_lengths,
        ).cpu()

    expected = run_qsfa(identity_packed_kv, identity_block_table)
    permuted = run_qsfa(permuted_packed_kv, permuted_block_table)
    assert torch.isfinite(expected).all()
    assert torch.isfinite(permuted).all()
    assert torch.equal(permuted, expected), (
        "A5 DSA-shape QSFA changed when only physical block placement changed"
    )
    for repeat_index in range(8):
        actual = run_qsfa(permuted_packed_kv, permuted_block_table)
        if torch.equal(actual, expected):
            continue
        mismatch = torch.nonzero(actual != expected, as_tuple=False)
        first_index = tuple(int(value) for value in mismatch[0].tolist())
        pytest.fail(
            "A5 DSA-shape QSFA output is not bitwise repeatable: "
            f"repeat={repeat_index}, first_index={first_index}, "
            f"actual={float(actual[first_index])}, "
            f"expected={float(expected[first_index])}, "
            f"mismatch_count={int(mismatch.shape[0])}"
        )


def test_packed_c8_ksc_to_qsfa_recycled_blocks_are_ordered() -> None:
    torch.manual_seed(2048)
    dram_logical_blocks = 325
    dram_physical_blocks = 326
    resident_physical_capacity = 1625
    block_table_width = 1024
    budget = 10240
    resident_seq_len = 10314

    key_nope = torch.randn(
        (dram_logical_blocks, _BLOCK_SIZE, 1, 512),
        dtype=torch.float32,
    ).clamp_(-4.0, 4.0).to(torch.float8_e4m3fn).view(torch.int8)
    key_rope = torch.randn(
        (dram_logical_blocks, _BLOCK_SIZE, 1, 64),
        dtype=torch.bfloat16,
    )
    scales = torch.rand(
        (dram_logical_blocks, _BLOCK_SIZE, 1, 4),
        dtype=torch.float32,
    ).add_(0.01)
    logical_dram = torch.cat(
        (
            key_nope,
            key_rope.contiguous().view(torch.int8),
            scales.contiguous().view(torch.int8),
        ),
        dim=-1,
    )
    dram_cpu = torch.zeros(
        (
            dram_physical_blocks,
            _BLOCK_SIZE,
            1,
            _PACKED_ROW_BYTES,
        ),
        dtype=torch.int8,
    )
    dram_cpu[1:] = logical_dram
    dram = _swapped_arena(tuple(dram_cpu.shape))
    _write_swapped_arena(dram, dram_cpu)

    source_tokens_cpu = torch.randperm(
        dram_logical_blocks * _BLOCK_SIZE,
        dtype=torch.int32,
    )[:budget]
    destination_slots_cpu = torch.arange(budget, dtype=torch.int32)
    source_tokens = torch.full(
        (1, 1, _COPY_CAPACITY),
        -1,
        dtype=torch.int32,
        device="npu",
    )
    destination_slots = torch.full_like(source_tokens, -1)
    source_tokens[0, 0, :budget] = source_tokens_cpu.to("npu")
    destination_slots[0, 0, :budget] = destination_slots_cpu.to(
        "npu"
    )
    copy_counts = torch.tensor([budget], dtype=torch.int32, device="npu")
    dram_block_table = torch.zeros(
        (1, block_table_width),
        dtype=torch.int32,
        device="npu",
    )
    dram_block_table[0, :dram_logical_blocks] = torch.arange(
        1,
        dram_logical_blocks + 1,
        dtype=torch.int32,
        device="npu",
    )
    query = torch.randn(
        (1, 64, 512 + 64),
        dtype=torch.bfloat16,
        device="npu",
    )
    sparse_indices = torch.randperm(
        resident_seq_len,
        dtype=torch.int32,
    )[:_ATTENTION_CAPACITY].view(1, 1, _ATTENTION_CAPACITY).to("npu")
    query_lengths = torch.tensor([1], dtype=torch.int32, device="npu")
    resident_lengths = torch.tensor(
        [resident_seq_len],
        dtype=torch.int32,
        device="npu",
    )
    hbm_bytes = torch.empty(
        (
            resident_physical_capacity,
            _BLOCK_SIZE,
            1,
            _PACKED_ROW_BYTES,
        ),
        dtype=torch.int8,
        device="npu",
    )
    tail_block = logical_dram[-1].to("npu")

    def run_chain(
        resident_block_ids: torch.Tensor,
        stale_value: int,
    ) -> torch.Tensor:
        resident_block_ids = resident_block_ids.to(torch.int64)
        hbm_bytes.fill_(stale_value)
        hbm_bytes[int(resident_block_ids[-1])].copy_(tail_block)
        hbm_block_table = torch.zeros(
            (1, block_table_width),
            dtype=torch.int32,
            device="npu",
        )
        hbm_block_table[0, : resident_block_ids.numel()] = (
            resident_block_ids.to(dtype=torch.int32, device="npu")
        )
        torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(
            hbm_bytes,
            dram,
            hbm_block_table,
            dram_block_table,
            source_tokens,
            destination_slots,
            copy_counts,
        )
        # Deliberately do not synchronize between KSC and its QSFA consumer.
        return sparse_flash_attention_for_offload_c8(
            query=query,
            packed_kv=hbm_bytes.view(torch.float8_e4m3fn),
            sparse_indices=sparse_indices,
            scale_value=(512 + 64) ** -0.5,
            block_table=hbm_block_table,
            actual_seq_lengths_query=query_lengths,
            resident_seq_lengths=resident_lengths,
        ).cpu()

    fresh = run_chain(_fresh_resident_block_ids(), 17)
    recycled = run_chain(_recycled_resident_block_ids(), 73)
    assert torch.isfinite(fresh).all()
    assert torch.isfinite(recycled).all()
    assert torch.equal(recycled, fresh), (
        "KSC-to-QSFA changed output when only resident physical reuse changed"
    )
