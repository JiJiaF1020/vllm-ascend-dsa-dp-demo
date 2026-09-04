# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.config import CompilationMode, CUDAGraphMode

from vllm_ascend.dsa_offload.config import (
    DSAOffloadConfig,
    DSAOffloadTraceConfig,
)


def _vllm_config(
    *,
    architecture: str = "GlmMoeDsaForCausalLM",
    async_scheduling: bool | None = False,
    enable_chunked_prefill: bool = False,
    long_prefill_token_threshold: int = 0,
    scheduler_reserve_full_isl: bool = True,
    enable_prefix_caching: bool = False,
    enforce_eager: bool = True,
    block_size: int = 128,
    speculative_config=None,
    kv_transfer_config=None,
    shutdown_timeout: int = 0,
    decode_context_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    enable_expert_parallel: bool = True,
    data_parallel_backend: str = "mp",
    enable_elastic_ep: bool = False,
    data_parallel_external_lb: bool = False,
    data_parallel_hybrid_lb: bool = False,
    kv_cache_metrics: bool = False,
    enable_kv_cache_events: bool = False,
    compilation_mode: CompilationMode = CompilationMode.NONE,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    cudagraph_capture_sizes: list[int] | None = None,
):
    hf_text_config = SimpleNamespace(
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=64,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture=architecture,
            use_mla=True,
            enforce_eager=enforce_eager,
            hf_text_config=hf_text_config,
            hf_config=hf_text_config,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=16,
            async_scheduling=async_scheduling,
            enable_chunked_prefill=enable_chunked_prefill,
            long_prefill_token_threshold=long_prefill_token_threshold,
            scheduler_reserve_full_isl=scheduler_reserve_full_isl,
        ),
        cache_config=SimpleNamespace(
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=decode_context_parallel_size,
            prefill_context_parallel_size=prefill_context_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            data_parallel_backend=data_parallel_backend,
            enable_elastic_ep=enable_elastic_ep,
            data_parallel_external_lb=data_parallel_external_lb,
            data_parallel_hybrid_lb=data_parallel_hybrid_lb,
        ),
        observability_config=SimpleNamespace(
            kv_cache_metrics=kv_cache_metrics,
        ),
        kv_events_config=SimpleNamespace(
            enable_kv_cache_events=enable_kv_cache_events,
        ),
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
        shutdown_timeout=shutdown_timeout,
        compilation_config=SimpleNamespace(
            mode=compilation_mode,
            cudagraph_mode=cudagraph_mode,
            cudagraph_capture_sizes=([] if cudagraph_capture_sizes is None else cudagraph_capture_sizes),
        ),
    )


def _mtp_config(
    *,
    num_speculative_tokens: int = 3,
    method: str = "mtp",
    parallel_drafting: bool = False,
    disable_padded_drafter_batch: bool = False,
    num_nextn_predict_layers: int = 1,
    enforce_eager: bool = True,
    draft_tensor_parallel_size: int | None = None,
):
    return SimpleNamespace(
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        parallel_drafting=parallel_drafting,
        disable_padded_drafter_batch=disable_padded_drafter_batch,
        enforce_eager=enforce_eager,
        draft_tensor_parallel_size=draft_tensor_parallel_size,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_nextn_predict_layers=num_nextn_predict_layers,
            ),
        ),
    )


def test_disabled_config_uses_typed_defaults() -> None:
    config = DSAOffloadConfig.from_dict(None)

    assert config.enabled is False
    assert config.split_indexer_cache is False
    assert config.resident_budget_tokens == (6144, 10240, 12288)
    assert config.max_resident_budget_tokens == 12288


def test_enabled_config_keeps_fractional_dram_multiplier() -> None:
    config = DSAOffloadConfig.from_dict(
        {
            "enabled": True,
            "hot_cpu_block_multiple": 1.5,
        },
        vllm_config=_vllm_config(),
    )

    assert config.enabled is True
    assert config.split_indexer_cache is True
    assert config.hot_cpu_block_multiple == 1.5
    assert config.model_capabilities is not None
    assert config.model_capabilities.architecture == "GlmMoeDsaForCausalLM"


def test_single_node_dp_foundation_accepts_internal_ep() -> None:
    config = DSAOffloadConfig.from_dict(
        {"enabled": True},
        vllm_config=_vllm_config(data_parallel_size=2),
    )

    assert config.enabled


@pytest.mark.parametrize(
    ("config_updates", "message"),
    [
        (
            {
                "data_parallel_size": 2,
                "enable_expert_parallel": False,
            },
            "enable_expert_parallel=True",
        ),
        (
            {
                "data_parallel_size": 2,
                "data_parallel_backend": "ray",
            },
            "not Ray",
        ),
        (
            {
                "data_parallel_size": 2,
                "data_parallel_external_lb": True,
            },
            "internal load balancing or offline SPMD",
        ),
    ],
)
def test_single_node_dp_foundation_rejects_later_phase_features(
    config_updates,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        DSAOffloadConfig.from_dict(
            {"enabled": True},
            vllm_config=_vllm_config(**config_updates),
        )


@pytest.mark.parametrize("prefix", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_single_node_dp_accepts_fixed_mtp_with_eager_proposer(prefix, graph) -> None:
    config = DSAOffloadConfig.from_dict(
        {
            "enabled": True, "enable_row_mode_decode_graph": graph,
            "sparse_activation_tokens": 8192,
            "resident_budget_tokens": [8192, 10240, 12288],
        },
        vllm_config=_vllm_config(
            data_parallel_size=2, tensor_parallel_size=4,
            speculative_config=_mtp_config(draft_tensor_parallel_size=4),
            enable_prefix_caching=prefix, enable_chunked_prefill=True,
            enforce_eager=not graph,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY if graph else CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[4, 8, 16] if graph else [],
        ),
    )
    assert config.mtp_enabled
    assert config.sparse_tail_block_count == 2


@pytest.mark.parametrize(("spec", "message"), [
    (_mtp_config(enforce_eager=False), "enforce_eager=True"),
    (_mtp_config(draft_tensor_parallel_size=1), "same target and draft"),
])
def test_single_node_dp_mtp_rejects_unsupported_proposer_modes(spec, message) -> None:
    with pytest.raises(ValueError, match=message):
        DSAOffloadConfig.from_dict(
            {"enabled": True, "sparse_activation_tokens": 8192, "resident_budget_tokens": [8192, 10240, 12288]},
            vllm_config=_vllm_config(data_parallel_size=2, tensor_parallel_size=4, speculative_config=spec),
        )


@pytest.mark.parametrize("graph", [False, True])
def test_single_node_dp_accepts_local_prefix_with_chunked_prefill(graph) -> None:
    config = DSAOffloadConfig.from_dict(
        {"enabled": True, "enable_row_mode_decode_graph": graph},
        vllm_config=_vllm_config(
            data_parallel_size=2,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enforce_eager=not graph,
            compilation_mode=CompilationMode.VLLM_COMPILE if graph else CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY if graph else CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[1, 2, 4] if graph else [],
        ),
    )
    assert config.enabled
    assert not config.mtp_enabled


@pytest.mark.parametrize("prefix", [False, True])
def test_single_node_dp_accepts_l3_without_mtp(prefix, monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "114514")
    config = DSAOffloadConfig.from_dict(
        {"enabled": True},
        vllm_config=_vllm_config(
            data_parallel_size=2,
            enable_prefix_caching=prefix,
            shutdown_timeout=300,
            kv_transfer_config=SimpleNamespace(
                kv_connector="DSAL3Connector",
                kv_role="kv_producer",
                kv_connector_extra_config={"backend": "mooncake"},
            ),
        ),
    )
    assert config.enabled


@pytest.mark.parametrize("seed", [None, "", "random", "-1", "4294967296"])
def test_dp_shared_l3_rejects_missing_or_nondeterministic_hash_seed(monkeypatch, seed):
    if seed is None:
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    else:
        monkeypatch.setenv("PYTHONHASHSEED", seed)
    config = _vllm_config(
        data_parallel_size=2, shutdown_timeout=300,
        kv_transfer_config=SimpleNamespace(kv_connector="DSAL3Connector", kv_role="kv_producer",
                                           kv_connector_extra_config={"backend": "mooncake"}),
    )
    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        DSAOffloadConfig.from_dict({"enabled": True}, vllm_config=config)


def test_hash_seed_requirement_does_not_change_local_prefix_or_single_dp_l3(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    assert DSAOffloadConfig.from_dict(
        {"enabled": True}, vllm_config=_vllm_config(data_parallel_size=2, enable_prefix_caching=True),
    ).enabled
    config = _vllm_config(
        shutdown_timeout=300,
        kv_transfer_config=SimpleNamespace(kv_connector="DSAL3Connector", kv_role="kv_producer",
                                           kv_connector_extra_config={"backend": "mooncake"}),
    )
    assert DSAOffloadConfig.from_dict({"enabled": True}, vllm_config=config).enabled


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("role,prefix", [("kv_producer", False), ("kv_producer", True), ("kv_both", True)])
def test_single_node_dp_l3_accepts_mtp_data_path(monkeypatch, graph, role, prefix) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "114514")
    config = DSAOffloadConfig.from_dict(
        {
            "enabled": True, "enable_row_mode_decode_graph": graph,
            "sparse_activation_tokens": 8192, "resident_budget_tokens": [8192, 10240, 12288],
        },
        vllm_config=_vllm_config(
            data_parallel_size=2, tensor_parallel_size=4,
            speculative_config=_mtp_config(draft_tensor_parallel_size=4), enable_prefix_caching=prefix,
            enable_chunked_prefill=True, enforce_eager=not graph, shutdown_timeout=300,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY if graph else CUDAGraphMode.NONE,
            cudagraph_capture_sizes=[4, 8, 16] if graph else [],
            kv_transfer_config=SimpleNamespace(
                kv_connector="DSAL3Connector", kv_role=role, kv_load_failure_policy="fail",
                kv_connector_extra_config={"backend": "mooncake"},
            ),
        ),
    )
    assert config.mtp_enabled and config.sparse_tail_block_count == 2


def test_non_mtp_does_not_accept_the_mtp_only_8192_tier() -> None:
    with pytest.raises(ValueError, match="non-MTP resident budgets"):
        DSAOffloadConfig.from_dict(
            {
                "enabled": True,
                "resident_budget_tokens": [8192, 10240, 12288],
            },
            vllm_config=_vllm_config(),
        )


def test_declared_all_full_topology_still_validates_layer_count() -> None:
    vllm_config = _vllm_config()
    vllm_config.model_config.hf_text_config.indexer_types = ["full", "full"]
    vllm_config.model_config.hf_text_config.num_hidden_layers = 3

    with pytest.raises(ValueError, match="one indexer type per hidden layer"):
        DSAOffloadConfig.from_dict(
            {"enabled": True},
            vllm_config=vllm_config,
        )


def test_shared_topology_rejects_layer_without_preceding_full_source() -> None:
    vllm_config = _vllm_config()
    vllm_config.model_config.hf_text_config.indexer_types = ["shared", "full"]
    vllm_config.model_config.hf_text_config.num_hidden_layers = 2

    with pytest.raises(ValueError, match="follow a full source layer"):
        DSAOffloadConfig.from_dict(
            {"enabled": True},
            vllm_config=vllm_config,
        )


def test_enabled_config_rejects_non_split_layout() -> None:
    with pytest.raises(ValueError, match="always requires a split"):
        DSAOffloadConfig.from_dict(
            {
                "enabled": True,
                "split_indexer_cache": False,
            }
        )


@pytest.mark.parametrize(
    ("raw_config", "message"),
    [
        ({"unknown": 1}, "Unknown dsa_sparse_config keys"),
        (
            {
                "prompt_budget_thresholds": [32768, 65536],
                "resident_budget_tokens": [6144, 10240],
            },
            "exactly one more",
        ),
        (
            {"resident_budget_tokens": [6144, 7168, 12288]},
            "not supported",
        ),
        ({"max_active_reqs": 0}, "must be positive"),
        ({"hot_cpu_block_multiple": float("nan")}, "positive and finite"),
        ({"hot_cpu_block_multiple": float("inf")}, "positive and finite"),
    ],
)
def test_invalid_static_config_is_rejected(
    raw_config,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        DSAOffloadConfig.from_dict(raw_config)


@pytest.mark.parametrize(
    ("config_updates", "message"),
    [
        ({"async_scheduling": None}, "async_scheduling=False"),
        ({"async_scheduling": True}, "async_scheduling=False"),
        (
            {"speculative_config": _mtp_config(method="ngram")},
            "method='mtp'",
        ),
        (
            {
                "kv_transfer_config": SimpleNamespace(
                    kv_connector="AscendStoreConnector",
                    kv_role="kv_producer",
                    kv_connector_extra_config={},
                )
            },
            "validated DSAL3Connector",
        ),
        (
            {"decode_context_parallel_size": 2},
            "context parallelism",
        ),
        (
            {"prefill_context_parallel_size": 2},
            "context parallelism",
        ),
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"kv_cache_metrics": True}, "KV-cache metrics"),
        ({"enable_kv_cache_events": True}, "KV-cache events"),
    ],
)
def test_initial_runtime_support_matrix_is_enforced(
    config_updates,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DSAOffloadConfig.from_dict(
            {"enabled": True},
            vllm_config=_vllm_config(**config_updates),
        )


def test_dsa_l3_connector_is_allowed_for_mooncake_without_mtp() -> None:
    kv_transfer = SimpleNamespace(
        kv_connector="DSAL3Connector",
        kv_role="kv_producer",
        kv_connector_extra_config={"backend": "mooncake"},
    )

    config = DSAOffloadConfig.from_dict(
        {"enabled": True},
        vllm_config=_vllm_config(
            kv_transfer_config=kv_transfer,
            shutdown_timeout=300,
        ),
    )

    assert config.enabled


def test_dsa_l3_prefix_load_requires_kv_both_and_prefix_caching() -> None:
    kv_transfer = SimpleNamespace(
        kv_connector="DSAL3Connector",
        kv_role="kv_both",
        kv_load_failure_policy="fail",
        kv_connector_extra_config={"backend": "mooncake"},
    )

    config = DSAOffloadConfig.from_dict(
        {"enabled": True},
        vllm_config=_vllm_config(
            kv_transfer_config=kv_transfer,
            enable_prefix_caching=True,
            shutdown_timeout=300,
        ),
    )
    assert config.enabled

    with pytest.raises(ValueError, match="enable_prefix_caching=True"):
        DSAOffloadConfig.from_dict(
            {"enabled": True},
            vllm_config=_vllm_config(
                kv_transfer_config=kv_transfer,
                enable_prefix_caching=False,
                shutdown_timeout=300,
            ),
        )

    kv_transfer.kv_load_failure_policy = "recompute"
    with pytest.raises(ValueError, match="kv_load_failure_policy='fail'"):
        DSAOffloadConfig.from_dict(
            {"enabled": True},
            vllm_config=_vllm_config(
                kv_transfer_config=kv_transfer,
                enable_prefix_caching=True,
                shutdown_timeout=300,
            ),
        )


def test_dsa_l3_connector_requires_positive_shutdown_timeout() -> None:
    kv_transfer = SimpleNamespace(
        kv_connector="DSAL3Connector",
        kv_role="kv_producer",
        kv_connector_extra_config={"backend": "mooncake"},
    )

    with pytest.raises(ValueError, match="shutdown_timeout>0"):
        DSAOffloadConfig.from_dict(
            {"enabled": True},
            vllm_config=_vllm_config(
                kv_transfer_config=kv_transfer,
                shutdown_timeout=0,
            ),
        )


def test_dsa_l3_connector_accepts_compromise_mtp() -> None:
    kv_transfer = SimpleNamespace(
        kv_connector="DSAL3Connector",
        kv_role="kv_both",
        kv_load_failure_policy="fail",
        kv_connector_extra_config={"backend": "mooncake"},
    )

    config = DSAOffloadConfig.from_dict(
        {
            "enabled": True,
            "sparse_activation_tokens": 8192,
            "resident_budget_tokens": [8192, 10240, 12288],
        },
        vllm_config=_vllm_config(
            kv_transfer_config=kv_transfer,
            speculative_config=_mtp_config(),
            enable_prefix_caching=True,
            shutdown_timeout=300,
        ),
    )

    assert config.mtp_enabled


@pytest.mark.parametrize(
    "config_updates",
    [
        {"enable_chunked_prefill": True},
        {"long_prefill_token_threshold": 1024},
    ],
)
def test_chunked_prefill_modes_are_supported(config_updates) -> None:
    config = DSAOffloadConfig.from_dict(
        {"enabled": True},
        vllm_config=_vllm_config(**config_updates),
    )

    assert config.enabled


def test_local_dram_prefix_cache_is_supported() -> None:
    config = DSAOffloadConfig.from_dict(
        {"enabled": True},
        vllm_config=_vllm_config(enable_prefix_caching=True),
    )

    assert config.enabled


def test_compromise_mtp3_contract_is_accepted() -> None:
    config = DSAOffloadConfig.from_dict(
        {
            "enabled": True,
            "sparse_activation_tokens": 8192,
            "resident_budget_tokens": [8192, 10240, 12288],
        },
        vllm_config=_vllm_config(
            speculative_config=_mtp_config(),
            enable_prefix_caching=True,
        ),
    )

    assert config.mtp_enabled
    assert config.mtp_num_speculative_tokens == 3
    assert config.mtp_cache_layer_count == 1
    assert config.uniform_decode_query_len == 4
    assert config.sparse_tail_block_count == 2


def test_mtp_cache_layer_index_range_is_explicit() -> None:
    config = DSAOffloadConfig(
        enabled=True,
        mtp_num_speculative_tokens=3,
        mtp_cache_layer_count=1,
    )

    assert not config.is_mtp_cache_layer_index(77, 78)
    assert config.is_mtp_cache_layer_index(78, 78)
    assert not config.is_mtp_cache_layer_index(79, 78)


@pytest.mark.parametrize(
    ("speculative_config", "raw_config", "message"),
    [
        (
            _mtp_config(num_speculative_tokens=4),
            {
                "sparse_activation_tokens": 8192,
                "resident_budget_tokens": [8192, 10240, 12288],
            },
            "requires exactly num_speculative_tokens=3",
        ),
        (
            _mtp_config(),
            {},
            "resident budgets must be one of",
        ),
        (
            _mtp_config(),
            {
                "sparse_activation_tokens": 6144,
                "resident_budget_tokens": [8192, 10240, 12288],
            },
            "sparse_activation_tokens",
        ),
        (
            _mtp_config(parallel_drafting=True),
            {
                "sparse_activation_tokens": 8192,
                "resident_budget_tokens": [8192, 10240, 12288],
            },
            "parallel_drafting",
        ),
        (
            _mtp_config(num_nextn_predict_layers=0),
            {
                "sparse_activation_tokens": 8192,
                "resident_budget_tokens": [8192, 10240, 12288],
            },
            "at least one physical MTP cache layer",
        ),
    ],
)
def test_compromise_mtp_rejects_unsupported_contracts(
    speculative_config,
    raw_config,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DSAOffloadConfig.from_dict(
            {"enabled": True, **raw_config},
            vllm_config=_vllm_config(
                speculative_config=speculative_config,
            ),
        )


@pytest.mark.parametrize(
    "config_updates",
    [
        {
            "enable_chunked_prefill": True,
            "scheduler_reserve_full_isl": False,
        },
        {
            "long_prefill_token_threshold": 1024,
            "scheduler_reserve_full_isl": False,
        },
    ],
)
def test_chunked_prefill_requires_full_prompt_admission(
    config_updates,
) -> None:
    with pytest.raises(ValueError, match="scheduler_reserve_full_isl=True"):
        DSAOffloadConfig.from_dict(
            {"enabled": True},
            vllm_config=_vllm_config(**config_updates),
        )


def test_graph_config_rejects_enforce_eager() -> None:
    with pytest.raises(ValueError, match="enforce_eager=False"):
        DSAOffloadConfig.from_dict(
            {
                "enabled": True,
                "enable_row_mode_decode_graph": True,
            },
            vllm_config=_vllm_config(enforce_eager=True),
        )


def test_block_contract_is_checked_after_backend_refresh() -> None:
    vllm_config = _vllm_config(block_size=16)
    config = DSAOffloadConfig.from_dict(
        {"enabled": True},
        vllm_config=vllm_config,
    )

    with pytest.raises(ValueError, match="block_size=128"):
        config.validate_finalized_cache_contract(vllm_config)

    vllm_config.cache_config.block_size = 128
    config.validate_finalized_cache_contract(vllm_config)


def test_final_graph_contract_accepts_separate_full_decode_routine() -> None:
    vllm_config = _vllm_config(
        enforce_eager=False,
        compilation_mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[1, 2, 4],
    )
    config = DSAOffloadConfig.from_dict(
        {
            "enabled": True,
            "enable_row_mode_decode_graph": True,
        },
        vllm_config=vllm_config,
    )

    config.validate_finalized_graph_contract(
        vllm_config,
        phase="test",
        require_resolved_mode=True,
    )


def test_platform_graph_contract_allows_full_before_backend_resolution() -> None:
    vllm_config = _vllm_config(
        enforce_eager=False,
        compilation_mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=[1, 2, 4],
    )
    config = DSAOffloadConfig.from_dict(
        {
            "enabled": True,
            "enable_row_mode_decode_graph": True,
        },
        vllm_config=vllm_config,
    )

    # Ascend attention backend may still normalize exact FULL into a
    # separate decode routine. The stricter check runs after that resolution.
    config.validate_finalized_graph_contract(
        vllm_config,
        phase="platform",
    )


def test_final_graph_contract_rejects_empty_capture_sizes() -> None:
    vllm_config = _vllm_config(
        enforce_eager=False,
        compilation_mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[],
    )
    config = DSAOffloadConfig.from_dict(
        {
            "enabled": True,
            "enable_row_mode_decode_graph": True,
        },
        vllm_config=vllm_config,
    )

    with pytest.raises(ValueError, match="non-empty"):
        config.validate_finalized_graph_contract(
            vllm_config,
            phase="test",
            require_resolved_mode=True,
        )


@pytest.mark.parametrize(
    ("compilation_mode", "cudagraph_mode", "message"),
    [
        (
            CompilationMode.NONE,
            CUDAGraphMode.NONE,
            "mode=VLLM_COMPILE",
        ),
        (
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.PIECEWISE,
            "requires a cudagraph mode with FULL",
        ),
        (
            CompilationMode.VLLM_COMPILE,
            CUDAGraphMode.FULL,
            "separate FULL decode routine",
        ),
    ],
)
def test_final_graph_contract_rejects_incompatible_native_mode(
    compilation_mode: CompilationMode,
    cudagraph_mode: CUDAGraphMode,
    message: str,
) -> None:
    vllm_config = _vllm_config(
        enforce_eager=False,
        compilation_mode=compilation_mode,
        cudagraph_mode=cudagraph_mode,
        cudagraph_capture_sizes=[1, 2, 4],
    )
    config = DSAOffloadConfig.from_dict(
        {
            "enabled": True,
            "enable_row_mode_decode_graph": True,
        },
        vllm_config=vllm_config,
    )

    with pytest.raises(ValueError, match=message):
        config.validate_finalized_graph_contract(
            vllm_config,
            phase="test",
            require_resolved_mode=True,
        )


def test_disabled_dsa_graph_requires_true_eager_native_mode() -> None:
    vllm_config = _vllm_config(
        enforce_eager=False,
        compilation_mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[1, 2, 4],
    )
    config = DSAOffloadConfig.from_dict(
        {"enabled": True},
        vllm_config=vllm_config,
    )

    with pytest.raises(ValueError, match="requires true eager"):
        config.validate_finalized_graph_contract(
            vllm_config,
            phase="test",
        )


def test_disabled_dsa_graph_accepts_true_eager_native_mode() -> None:
    vllm_config = _vllm_config(
        enforce_eager=True,
        compilation_mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.NONE,
        cudagraph_capture_sizes=[],
    )
    config = DSAOffloadConfig.from_dict(
        {"enabled": True},
        vllm_config=vllm_config,
    )

    config.validate_finalized_graph_contract(
        vllm_config,
        phase="test",
    )


def test_trace_config_is_parsed_once_into_immutable_filters() -> None:
    trace = DSAOffloadTraceConfig.from_value(
        {
            "enabled": True,
            "points": ["first_sample"],
            "ranks": [0, 2],
        }
    )

    assert trace.enabled is True
    assert trace.points == frozenset({"first_sample"})
    assert trace.ranks == frozenset({0, 2})
