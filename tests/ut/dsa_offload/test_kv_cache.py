# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.dsa_offload import kv_cache as dsa_kv_cache
from vllm_ascend.dsa_offload.kv_cache import (
    DSAIndexerKVSpec,
    DSAKVCacheConfig,
    DSAResidentMLAAttentionSpec,
    build_dsa_kv_cache_config,
    build_dsa_kv_cache_groups,
    dsa_dram_bytes_per_logical_block,
    dsa_max_memory_usage_bytes,
    dsa_pool_bytes_per_base_block,
    finalize_dsa_dram_capacity,
    get_dsa_group_num_blocks,
    get_dsa_kv_cache_binding_order,
    get_dsa_kv_cache_group_ids,
    validate_dsa_kv_cache_config,
)
from vllm_ascend.dsa_offload.kv_cache_coordinator import (
    DSABlockPoolView,
    DSAKVCacheCoordinator,
)
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _ascend_get_kv_cache_groups,
    _ascend_resolve_kv_cache_block_sizes,
)


def _make_specs(
    num_layers: int = 2,
) -> dict[str, DSAIndexerKVSpec | DSAResidentMLAAttentionSpec]:
    specs: dict[
        str,
        DSAIndexerKVSpec | DSAResidentMLAAttentionSpec,
    ] = {}
    for layer_idx in range(num_layers):
        specs[f"model.layers.{layer_idx}.self_attn.indexer.k_cache"] = DSAIndexerKVSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
        specs[f"model.layers.{layer_idx}.self_attn.attn"] = DSAResidentMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            sparse_head_dim=(512, 64, 0),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        )
    return specs


def _make_vllm_config(
    *,
    max_model_len: int = 32768,
    override: int | None = None,
    enable_prefix_caching: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=override,
            block_size=128,
            hash_block_size=None,
            enable_prefix_caching=enable_prefix_caching,
        ),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        kv_transfer_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            tensor_parallel_size=2,
        ),
    )


def _make_mtp_spec() -> AscendMLAAttentionSpec:
    return AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=704,
        sparse_head_dim=(512, 64, 128),
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        cache_sparse_sfa_c8=False,
        cache_sparse_li_c8=False,
    )


def test_indexer_spec_accounts_for_one_vector_per_token() -> None:
    spec = next(spec for spec in _make_specs(num_layers=1).values() if isinstance(spec, DSAIndexerKVSpec))

    assert spec.page_size_bytes == 128 * 1 * 128 * 2


def test_c8_indexer_spec_merge_preserves_split_cache_layout() -> None:
    specs = [
        DSAIndexerKVSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            cache_sparse_li_c8=True,
        )
        for _ in range(2)
    ]

    merged = DSAIndexerKVSpec.merge(specs)

    assert merged.cache_sparse_li_c8
    assert merged.c8_k_cache_dtype == torch.float8_e4m3fn
    assert merged.c8_k_scale_cache_dtype == torch.float32
    assert merged.page_size_bytes == 128 * (128 + 4)


def test_c8_resident_spec_merge_preserves_packed_cache_layout() -> None:
    specs = [
        DSAResidentMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=656,
            sparse_head_dim=(656, 0, 0),
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
            cache_sparse_sfa_c8=True,
            cache_sparse_li_c8=False,
        )
        for _ in range(2)
    ]

    merged = DSAResidentMLAAttentionSpec.merge(specs)

    assert type(merged) is DSAResidentMLAAttentionSpec
    assert merged.cache_sparse_sfa_c8
    assert not merged.cache_sparse_li_c8
    assert merged.sparse_head_dim == (656, 0, 0)
    assert merged.page_size_bytes == 128 * 656


def test_split_groups_keep_stable_plane_order() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs())

    assert len(groups) == 2
    assert isinstance(groups[0].kv_cache_spec, DSAIndexerKVSpec)
    assert isinstance(
        groups[1].kv_cache_spec,
        DSAResidentMLAAttentionSpec,
    )
    assert len(groups[0].layer_names) == len(groups[1].layer_names) == 2


def test_mtp_draft_layer_forms_native_bf16_full_cache_group() -> None:
    specs = _make_specs(num_layers=2)
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()

    groups = build_dsa_kv_cache_groups(specs)

    assert len(groups) == 3
    assert not any("layers.78" in name for name in groups[0].layer_names)
    assert not any("layers.78" in name for name in groups[1].layer_names)
    assert groups[2].layer_names == ["model.layers.78.self_attn.attn"]
    assert type(groups[2].kv_cache_spec) is AscendMLAAttentionSpec
    assert groups[2].kv_cache_spec.dtype == torch.bfloat16
    assert groups[2].kv_cache_spec.sparse_head_dim == (512, 64, 128)
    assert groups[2].is_eagle_group


@pytest.mark.parametrize(
    "mtp_spec",
    [
        replace(_make_mtp_spec(), cache_sparse_li_c8=True),
        replace(
            _make_mtp_spec(),
            head_size=576,
            sparse_head_dim=(512, 64, 0),
        ),
        replace(_make_mtp_spec(), compress_ratio=2),
        replace(_make_mtp_spec(), cache_dtype_str="fp8_ds_mla"),
    ],
)
def test_mtp_full_group_rejects_non_native_cache_layout(
    mtp_spec: AscendMLAAttentionSpec,
) -> None:
    specs = _make_specs(num_layers=1)
    specs["model.layers.78.self_attn.attn"] = mtp_spec

    with pytest.raises(RuntimeError, match="three-entry BF16"):
        build_dsa_kv_cache_groups(specs)


def test_engine_core_grouping_initializes_process_local_ascend_config(
    monkeypatch,
) -> None:
    initialized_with: list[object] = []
    vllm_config = object()
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.init_ascend_config",
        initialized_with.append,
    )

    groups = _ascend_get_kv_cache_groups(  # type: ignore[arg-type]
        vllm_config,
        _make_specs(),
    )

    assert initialized_with == [vllm_config]
    assert len(groups) == 2


def test_split_group_ids_follow_spec_identity_not_position() -> None:
    specs = _make_specs()
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()
    groups = list(reversed(build_dsa_kv_cache_groups(specs)))
    config = SimpleNamespace(kv_cache_groups=groups)

    group_ids = get_dsa_kv_cache_group_ids(config)  # type: ignore[arg-type]

    assert group_ids.indexer == 2
    assert group_ids.resident_mla == 1
    assert group_ids.mtp_full == 0


def test_split_cache_binding_orders_each_layer_resident_then_indexer() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs())
    config = SimpleNamespace(kv_cache_groups=groups)

    assert get_dsa_kv_cache_binding_order(config) == [
        "model.layers.0.self_attn.attn",
        "model.layers.0.self_attn.indexer.k_cache",
        "model.layers.1.self_attn.attn",
        "model.layers.1.self_attn.indexer.k_cache",
    ]


def test_split_cache_binding_leaves_mtp_full_group_to_baseline() -> None:
    specs = _make_specs(num_layers=1)
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()
    groups = build_dsa_kv_cache_groups(specs)
    config = SimpleNamespace(kv_cache_groups=groups)

    assert get_dsa_kv_cache_binding_order(config) == [
        "model.layers.0.self_attn.attn",
        "model.layers.0.self_attn.indexer.k_cache",
    ]


def test_split_cache_binding_rejects_duplicate_layer_name() -> None:
    groups = build_dsa_kv_cache_groups(_make_specs(num_layers=1))
    groups[1].layer_names.append(groups[0].layer_names[0])
    config = SimpleNamespace(kv_cache_groups=groups)

    with pytest.raises(RuntimeError, match="more than one group"):
        get_dsa_kv_cache_binding_order(config)


def test_ratio_is_expressed_by_final_tensor_sizes(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    bytes_per_base_block = dsa_pool_bytes_per_base_block(groups)

    config = build_dsa_kv_cache_config(
        _make_vllm_config(),
        groups,
        available_memory=bytes_per_base_block * 256,
    )

    assert config.num_blocks == 256
    assert get_dsa_group_num_blocks(config, groups[0]) == 768
    assert get_dsa_group_num_blocks(config, groups[1]) == 256
    assert not hasattr(groups[0], "dsa_num_blocks")
    assert not hasattr(groups[1], "dsa_num_blocks")


def test_current_dsa_groups_resolve_scheduler_and_hash_to_128(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 2)
    vllm_config = _make_vllm_config()
    groups = build_dsa_kv_cache_groups(_make_specs())
    config = build_dsa_kv_cache_config(
        vllm_config,
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 128,
    )

    assert _ascend_resolve_kv_cache_block_sizes(
        config,
        vllm_config,
    ) == (128, 128)
    prefix_config = _make_vllm_config(enable_prefix_caching=True)
    assert _ascend_resolve_kv_cache_block_sizes(
        config,
        prefix_config,
    ) == (128, 128)


def test_engine_core_finalizes_one_shared_dram_capacity_plan(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_ascend_config",
        lambda: SimpleNamespace(
            dsa_offload_config=SimpleNamespace(
                hot_cpu_block_multiple=1.5,
            )
        ),
    )
    groups = build_dsa_kv_cache_groups(_make_specs())
    hbm_bytes = dsa_pool_bytes_per_base_block(groups) * 128
    configs = [
        build_dsa_kv_cache_config(
            _make_vllm_config(),
            groups,
            available_memory=hbm_bytes,
        )
        for _ in range(2)
    ]
    bundle_bytes = dsa_dram_bytes_per_logical_block(groups)
    indexer_blocks = get_dsa_group_num_blocks(configs[0], groups[0])
    desired = int(indexer_blocks * 1.5)

    plan = finalize_dsa_dram_capacity(
        _make_vllm_config(),
        configs,
        host_available_bytes=(desired + 1) * bundle_bytes * 2,
    )

    assert plan is not None
    assert plan.usable_blocks == desired
    assert all(isinstance(config, DSAKVCacheConfig) for config in configs)
    assert all(config.dram_capacity is plan for config in configs)


def test_engine_core_reserves_l3_host_pools_before_dram_planning(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_ascend_config",
        lambda: SimpleNamespace(
            dsa_offload_config=SimpleNamespace(
                hot_cpu_block_multiple=1.5,
            )
        ),
    )
    groups = build_dsa_kv_cache_groups(_make_specs())
    hbm_bytes = dsa_pool_bytes_per_base_block(groups) * 128
    vllm_config = _make_vllm_config()
    vllm_config.kv_transfer_config = SimpleNamespace(
        kv_connector="DSAL3Connector"
    )
    configs = [
        build_dsa_kv_cache_config(
            vllm_config,
            groups,
            available_memory=hbm_bytes,
        )
        for _ in range(2)
    ]
    bundle_bytes = dsa_dram_bytes_per_logical_block(groups)
    indexer_blocks = get_dsa_group_num_blocks(configs[0], groups[0])
    desired = int(indexer_blocks * 1.5)
    l3_reservation = 4_096
    monkeypatch.setattr(
        dsa_kv_cache,
        "_dsa_l3_host_reservation_bytes",
        lambda _config, _tp_size: l3_reservation,
    )
    host_available = (
        (desired + 1) * bundle_bytes * 2 + l3_reservation
    )

    plan = finalize_dsa_dram_capacity(
        vllm_config,
        configs,
        host_available_bytes=host_available,
    )

    assert plan is not None
    assert plan.host_available_bytes == host_available - l3_reservation
    assert plan.usable_blocks == desired


def test_mtp_full_group_uses_dense_three_x_capacity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    specs = _make_specs()
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()
    groups = build_dsa_kv_cache_groups(specs)
    bytes_per_base_block = dsa_pool_bytes_per_base_block(groups)

    config = build_dsa_kv_cache_config(
        _make_vllm_config(),
        groups,
        available_memory=bytes_per_base_block * 256,
    )

    group_ids = get_dsa_kv_cache_group_ids(config)
    assert group_ids.indexer == 0
    assert group_ids.resident_mla == 1
    assert group_ids.mtp_full == 2
    assert config.num_blocks == 256
    assert get_dsa_group_num_blocks(config, groups[0]) == 768
    assert get_dsa_group_num_blocks(config, groups[1]) == 256
    assert get_dsa_group_num_blocks(config, groups[2]) == 768
    assert dsa_dram_bytes_per_logical_block(groups) == (
        groups[0].kv_cache_spec.page_size_bytes * len(groups[0].layer_names)
        + groups[1].kv_cache_spec.page_size_bytes
        * len(groups[1].layer_names)
    )


def test_mtp_full_group_requires_target_indexer_token_coverage(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    specs = _make_specs()
    specs["model.layers.78.self_attn.attn"] = replace(
        _make_mtp_spec(),
        block_size=64,
    )
    groups = build_dsa_kv_cache_groups(specs)

    with pytest.raises(RuntimeError, match="same token range"):
        build_dsa_kv_cache_config(
            _make_vllm_config(override=256),
            groups,
            available_memory=1,
        )


def test_num_gpu_blocks_override_keeps_base_block_semantics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())

    config = build_dsa_kv_cache_config(
        _make_vllm_config(override=320),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 256,
    )

    assert config.num_blocks == 320
    assert get_dsa_group_num_blocks(config, groups[0]) == 960
    assert get_dsa_group_num_blocks(config, groups[1]) == 320


def test_cross_rank_base_capacity_shrink_preserves_plane_ratio(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    specs = _make_specs()
    specs["model.layers.78.self_attn.attn"] = _make_mtp_spec()
    groups = build_dsa_kv_cache_groups(specs)
    config = build_dsa_kv_cache_config(
        _make_vllm_config(),
        groups,
        available_memory=dsa_pool_bytes_per_base_block(groups) * 256,
    )

    old_base_blocks = config.num_blocks
    config.num_blocks = 128
    for tensor in config.kv_cache_tensors:
        tensor.size = tensor.size // old_base_blocks * config.num_blocks

    validate_dsa_kv_cache_config(config)
    assert get_dsa_group_num_blocks(config, groups[0]) == 384
    assert get_dsa_group_num_blocks(config, groups[1]) == 128
    assert get_dsa_group_num_blocks(config, groups[2]) == 384


def test_prefill_admission_uses_weighted_base_block_cost(
    monkeypatch,
) -> None:
    monkeypatch.setattr(dsa_kv_cache, "_get_dsa_ratio", lambda: 3)
    groups = build_dsa_kv_cache_groups(_make_specs())
    vllm_config = _make_vllm_config(max_model_len=2048)

    assert dsa_max_memory_usage_bytes(
        vllm_config,
        groups,
    ) == 16 * dsa_pool_bytes_per_base_block(groups)


def test_split_groups_accept_indexer_subset_for_shared_topology() -> None:
    specs = _make_specs(num_layers=2)
    del specs["model.layers.1.self_attn.indexer.k_cache"]

    # GLM-5.2 共享 indexer 拓扑下 indexer 层是 resident 层的真子集，
    # 不再要求每层都有独立 indexer，分组应通过。
    groups = build_dsa_kv_cache_groups(specs)
    assert len(groups) == 2
    assert len(groups[0].layer_names) == 1  # 仅 layer 0 有 indexer
    assert len(groups[1].layer_names) == 2


def test_glm52_complete_dram_block_does_not_pad_shared_indexers() -> None:
    # 真实 GLM-5.2 拓扑：前两层 full，之后每四层一个 full，合计
    # 78 resident MLA + 21 full Indexer。shared follower 只复用 top-K，
    # 不应为 DRAM 完整块额外占一份空 Indexer payload。
    full_indexer_layers = {0, 1, *range(2, 78, 4)}
    assert len(full_indexer_layers) == 21
    specs = _make_specs(num_layers=78)
    for layer_index in range(78):
        if layer_index not in full_indexer_layers:
            del specs[
                f"model.layers.{layer_index}.self_attn.indexer.k_cache"
            ]

    groups = build_dsa_kv_cache_groups(specs)
    indexer_group, resident_group = groups
    expected = (
        21 * indexer_group.kv_cache_spec.page_size_bytes
        + 78 * resident_group.kv_cache_spec.page_size_bytes
    )

    assert len(indexer_group.layer_names) == 21
    assert len(resident_group.layer_names) == 78
    assert dsa_dram_bytes_per_logical_block(groups) == expected


def test_split_groups_reject_orphan_indexer_without_resident() -> None:
    # indexer 指向一个不存在 resident 层的 transformer 下标仍属硬错误。
    specs = _make_specs(num_layers=1)
    del specs["model.layers.0.self_attn.indexer.k_cache"]
    orphan = DSAIndexerKVSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    specs["model.layers.7.self_attn.indexer.k_cache"] = orphan

    with pytest.raises(RuntimeError, match="no matching resident MLA layer"):
        build_dsa_kv_cache_groups(specs)


def test_split_groups_reject_more_indexer_than_resident() -> None:
    specs = _make_specs(num_layers=1)
    specs["model.layers.1.self_attn.indexer.k_cache"] = DSAIndexerKVSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    # 保留 layer 0 resident，使入口先命中明确的数量合同。

    with pytest.raises(RuntimeError, match="more Indexer caches than resident"):
        build_dsa_kv_cache_groups(specs)


def test_block_pool_view_resets_every_physical_pool() -> None:
    class _Pool:
        num_gpu_blocks = 4

        def __init__(self, result: bool) -> None:
            self.result = result
            self.reset_calls = 0
            self.enable_caching = False

        def reset_prefix_cache(self) -> bool:
            self.reset_calls += 1
            return self.result

    first = _Pool(False)
    second = _Pool(True)
    dram = _Pool(True)
    view = DSABlockPoolView(  # type: ignore[list-item]
        [first, second],
        extra_prefix_cache_pools=[dram],
    )

    assert view.reset_prefix_cache() is False
    assert first.reset_calls == 1
    assert second.reset_calls == 1
    assert dram.reset_calls == 1


def test_component_admission_checks_each_physical_pool() -> None:
    class _Pool:
        def __init__(self, free_blocks: int) -> None:
            self.free_blocks = free_blocks

        def get_num_free_blocks(self) -> int:
            return self.free_blocks

    coordinator = object.__new__(DSAKVCacheCoordinator)
    coordinator.physical_block_pools = (  # type: ignore[attr-defined]
        _Pool(6),
        _Pool(2),
        _Pool(5),
    )

    assert coordinator.can_allocate((6, 2, 5))
    assert not coordinator.can_allocate((7, 1, 1))
    assert not coordinator.can_allocate((1, 3, 1))
    assert not coordinator.can_allocate((1, 1, 6))
    with pytest.raises(RuntimeError, match="reservations"):
        coordinator.can_allocate((1, 1, 1), reserved_blocks=1)
