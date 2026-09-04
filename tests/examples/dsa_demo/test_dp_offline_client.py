# SPDX-License-Identifier: Apache-2.0
"""离线SPMD配置与验收，不加载模型/NPU；可用时使用真实vLLM配置类型解析。"""

import importlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/dsa_demo"))
    return importlib.import_module("offline_dp_closeout_test")


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("l3", [False, True])
def test_offline_builds_actual_spmd_kwargs_without_other_examples(module, mtp, l3, monkeypatch):
    args = module._parse_args(["--model", "test-model", "--mtp" if mtp else "--no-mtp",
                               "--l3" if l3 else "--no-l3"])
    # A user's edited single-prompt example must have no influence on this test.
    monkeypatch.setitem(sys.modules, "simple_prompt_test", None)
    kwargs = module._build_kwargs(args)
    module._validate_kwargs(args, kwargs)
    assert "data_parallel_size" not in kwargs
    assert kwargs["distributed_executor_backend"] == "mp"
    assert kwargs["tensor_parallel_size"] == 4 and kwargs["enable_expert_parallel"] is True
    assert kwargs["enable_prefix_caching"] is True and kwargs["async_scheduling"] is False
    assert kwargs["max_num_seqs"] == 2 and kwargs["max_model_len"] == 100000
    assert kwargs["max_num_batched_tokens"] == 8192
    assert kwargs["shutdown_timeout"] == 300
    dsa = kwargs["additional_config"]["dsa_sparse_config"]
    assert dsa["indexer_mla_block_ratio"] == args.indexer_mla_ratio
    assert dsa["hot_cpu_block_multiple"] == args.hot_cpu_block_multiple
    assert kwargs["compilation_config"]["cudagraph_capture_sizes"] == ([4, 8] if mtp else [1, 2])
    if mtp:
        assert kwargs["speculative_config"] == {"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": True}
    else:
        assert "speculative_config" not in kwargs
    if l3:
        connector = kwargs["kv_transfer_config"]
        assert (connector["kv_connector"], connector["kv_role"], connector["kv_load_failure_policy"]) == (
            "DSAL3Connector", "kv_both", "fail",
        )
        assert connector["kv_connector_extra_config"]["backend"] == "mooncake"
        assert connector["kv_connector_extra_config"]["direct_hbm_load"] is False
    else:
        assert "kv_transfer_config" not in kwargs


def test_missing_l3_kwargs_are_rejected_before_loading_model(module):
    args = module._parse_args([])
    kwargs = module._build_kwargs(args)
    kwargs.pop("kv_transfer_config")
    with pytest.raises(ValueError, match="ENABLE_L3=True.*lack"):
        module._validate_kwargs(args, kwargs)


@pytest.mark.parametrize("mtp", [False, True])
def test_actual_kwargs_parse_with_native_vllm_configs(module, mtp):
    pytest.importorskip("vllm")
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs

    kwargs = module._build_kwargs(module._parse_args(["--mtp" if mtp else "--no-mtp"]))
    # Match LLM.__init__'s real dict conversion, without constructing a model.
    kwargs["kv_transfer_config"] = KVTransferConfig(**kwargs["kv_transfer_config"])
    engine_args = EngineArgs(**kwargs)
    assert engine_args.kv_transfer_config.kv_connector == "DSAL3Connector"
    assert engine_args.kv_transfer_config.kv_role == "kv_both"
    assert engine_args.kv_transfer_config.kv_load_failure_policy == "fail"
    assert engine_args.compilation_config.cudagraph_mode.name == "FULL_DECODE_ONLY"


@pytest.mark.parametrize("extra", [
    ["--labels", "40k,40k"], ["--labels", "40k"], ["--max-tokens", "1"],
    ["--timeout", "nan"], ["--drain-timeout", "0"], ["--segment-base", "45001"],
])
def test_offline_rejects_invalid_settings_before_spawning(module, extra):
    with pytest.raises(SystemExit):
        module._parse_args(["--model", "model", *extra])


def test_offline_l3_requires_tcp_config(module, tmp_path):
    path = tmp_path / "mooncake.json"
    path.write_text('{"protocol":"ascend"}', encoding="utf-8")
    with pytest.raises(SystemExit):
        module._parse_args(["--model", "model", "--l3", "--mooncake-config", str(path)])
    path.write_text('{"protocol":"tcp"}', encoding="utf-8")
    assert module._parse_args(["--model", "model", "--l3", "--mooncake-config", str(path)]).l3


def test_offline_waits_for_model_and_io_before_local_reset(module, monkeypatch):
    quiet = {"busy": False, "pending_pause": False, "l3_pending": False, "fatal": None}
    core = SimpleNamespace(call_utility=Mock(side_effect=[{**quiet, "l3_pending": True}, quiet]))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    assert module._wait_idle(core, 5) == quiet
    assert core.call_utility.call_count == 2
    assert all(call.args == ("dsa_cache_admin", "status") for call in core.call_utility.call_args_list)
    core.call_utility.side_effect = [{**quiet, "fatal": "failed"}]
    with pytest.raises(RuntimeError, match="engine failed"):
        module._wait_idle(core, 5)


@pytest.mark.parametrize("cached,expect_hit,finish,count,valid", [
    (128, True, "length", 128, True), (0, False, "length", 128, True),
    (0, True, "length", 128, False), (None, False, "length", 128, False),
    (-1, False, "length", 128, False), (128, True, "stop", 128, False),
    (128, True, "length", 4, False),
])
def test_offline_checks_real_completion_and_cache_hit(module, cached, expect_hit, finish, count, valid):
    output = SimpleNamespace(
        num_cached_tokens=cached, prompt_token_ids=list(range(256)),
        outputs=[SimpleNamespace(token_ids=[1] * count, finish_reason=finish, text="ok")],
    )
    if valid:
        assert module._check_output(output, expect_hit=expect_hit, max_tokens=128)["cached_tokens"] == cached
    else:
        with pytest.raises(AssertionError):
            module._check_output(output, expect_hit=expect_hit, max_tokens=128)


@pytest.mark.parametrize("l3,broken", [(False, None), (True, None), (True, "connector"), (True, "namespace")])
def test_offline_worker_rank_phases_local_reset_and_shutdown(module, monkeypatch, tmp_path, l3, broken):
    args = module._parse_args(["--model", "model", "--no-mtp"])
    args.l3 = l3
    # Isolate child environment changes from the pytest process environment.
    monkeypatch.setattr(module.os, "environ", dict(os.environ))
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    core = SimpleNamespace(
        call_utility=Mock(return_value={
            "busy": False, "l3_pending": False, "pending_pause": False, "fatal": None, "dp_rank": 1, "dp_size": 2,
            "base_fingerprint": "base" if l3 and broken != "namespace" else None,
            "fingerprint": "base" if l3 and broken != "namespace" else None,
        }), shutdown=Mock(),
    )
    generated, resets = [], []

    class LLM:
        def __init__(self, **kwargs):
            assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "0"
            assert "data_parallel_size" not in kwargs
            assert os.environ["VLLM_DP_RANK"] == "1"
            assert os.environ["VLLM_DP_SIZE"] == "2"
            config = SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_size=2, data_parallel_rank=1, tensor_parallel_size=4),
                kv_transfer_config=(SimpleNamespace(**kwargs["kv_transfer_config"])
                                    if l3 and broken != "connector" else None),
                speculative_config=None,
                cache_config=SimpleNamespace(enable_prefix_caching=True),
                scheduler_config=SimpleNamespace(async_scheduling=False),
            )
            self.llm_engine = SimpleNamespace(engine_core=core, vllm_config=config)

        def generate(self, prompts, params, use_tqdm):
            phase_index = len(generated)
            generated.append(prompts)
            cached = 128 if phase_index == 1 or (phase_index == 2 and l3) else 0
            return [SimpleNamespace(
                request_id="test-request", prompt_token_ids=list(range(256)), num_cached_tokens=cached,
                outputs=[SimpleNamespace(token_ids=[1] * args.max_tokens, finish_reason="length", text="ok")],
            ) for _ in prompts]

        def reset_prefix_cache(self, *, reset_connector):
            resets.append(reset_connector)
            assert reset_connector is False  # No negative external-reset experiment in the GPU run.
            return True

        def collective_rpc(self, method, *, timeout):
            raise AssertionError("normal offline test must not install a diagnostic RPC")

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs))
    barrier = SimpleNamespace(wait=Mock(), abort=Mock())
    if broken:
        with pytest.raises(RuntimeError, match="requested L3"):
            module._worker(1, args, "run", 12345, [("8k", "short"), ("70k", "long")], barrier, tmp_path)
        assert generated == [] and resets == []
        barrier.abort.assert_called_once()
        core.shutdown.assert_called_once_with(timeout=300)
        assert "requested L3" in (tmp_path / "dp1-failure.txt").read_text(encoding="utf-8")
        return
    module._worker(1, args, "run", 12345, [("8k", "short"), ("70k", "long")], barrier, tmp_path)
    assert [prompts[0]["prompt"] for prompts in generated] == ["long", "long", "short"]
    assert [prompts[0]["cache_salt"] for prompts in generated] == ["run-70k", "run-70k", "run-8k"]
    assert resets == [False]
    assert not (tmp_path / "shutdown").exists()
    assert not (tmp_path / "shutdown-debug.zip").exists()
    assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "0"
    assert barrier.wait.call_count == 7
    barrier.abort.assert_not_called()
    core.shutdown.assert_called_once_with(timeout=300)
    assert len((tmp_path / "dp1.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_offline_no_arguments_selects_current_closeout_settings(module, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "stale-shell-model")
    args = module._parse_args([])
    assert args.model == "/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4"
    assert args.mode == "graph" and args.mtp is True and args.l3 is True
    assert args.labels == "8k,40k" and args.max_tokens == 128
    assert (args.max_num_seqs, args.max_model_len, args.chunk_size) == (2, 100000, 8192)
    assert (args.indexer_mla_ratio, args.hot_cpu_block_multiple) == (2, 1.0)
    assert args.visible_devices == "0,1,2,3,4,5,6,7"
    assert not hasattr(args, "shutdown_trace")


def test_offline_reads_user_section_and_keeps_optional_cli_overrides(module, monkeypatch):
    for name, value in {
        "MODEL_PATH": "edited-model", "RUN_MODE": "eager", "ENABLE_MTP": False, "ENABLE_L3": False,
        "PROMPT_LABELS": ("8k", "70k"), "MAX_NUM_SEQS": 4, "MAX_TOKENS": 64,
        "DSA_INDEXER_MLA_BLOCK_RATIO": 2, "DSA_HOT_CPU_BLOCK_MULTIPLE": 1.0,
    }.items():
        monkeypatch.setattr(module, name, value)
    args = module._parse_args([])
    assert (args.model, args.mode, args.mtp, args.l3) == ("edited-model", "eager", False, False)
    assert (args.labels, args.max_num_seqs, args.max_tokens) == ("8k,70k", 4, 64)
    assert (args.indexer_mla_ratio, args.hot_cpu_block_multiple) == (2, 1.0)
    override = module._parse_args(["--model", "cli-model", "--mtp", "--max-tokens", "128"])
    assert override.model == "cli-model" and override.mtp is True and override.max_tokens == 128


def test_preflight_model_and_dependency_failures_do_not_start_processes(module, monkeypatch, tmp_path):
    args = module._parse_args(["--model", str(tmp_path)])
    with pytest.raises(RuntimeError, match="containing config.json"):
        module._preflight(args)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None if name == "torch_npu" else object())
    with pytest.raises(RuntimeError, match="missing Python packages.*torch_npu"):
        module._preflight(args)


def test_dataset_loading_needs_no_other_demo(module, monkeypatch):
    monkeypatch.setitem(sys.modules, "novel_dataset", SimpleNamespace(chinese_5k="short", chinese_40k=["long"]))
    monkeypatch.setitem(sys.modules, "online_dp_foundation_test", None)
    assert module._load_cases(["5k", "40k"]) == [("5k", "short"), ("40k", "long")]


def test_preflight_master_failure_is_reported_before_model_load(module, monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    args = module._parse_args(["--model", str(tmp_path)])
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(module.socket, "create_connection", Mock(side_effect=ConnectionRefusedError()))
    with pytest.raises(RuntimeError, match="Master is unreachable.*no model was loaded"):
        module._preflight(args)


@pytest.mark.parametrize("override,port", [(None, 50088), ("127.0.0.1:50089", 50089), ("", None)])
def test_preflight_master_uses_backend_none_vs_empty_semantics(module, monkeypatch, tmp_path, override, port):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "mooncake.json"
    config.write_text('{"protocol":"tcp","master_server_address":"127.0.0.1:50088"}', encoding="utf-8")
    args = module._parse_args(["--model", str(tmp_path), "--mooncake-config", str(config)])
    if override is None:
        monkeypatch.delenv("MOONCAKE_MASTER", raising=False)
    else:
        monkeypatch.setenv("MOONCAKE_MASTER", override)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _: object())
    connect = Mock(return_value=nullcontext())
    monkeypatch.setattr(module.socket, "create_connection", connect)
    if port is None:
        with pytest.raises(ValueError, match="Master address"):
            module._preflight(args)
        connect.assert_not_called()
    else:
        module._preflight(args)
        connect.assert_called_once_with(("127.0.0.1", port), timeout=3)

def test_check_only_saves_real_kwargs_without_spawning(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, "CHECK_ONLY", True)
    monkeypatch.setattr(module, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(module, "_preflight", Mock())
    monkeypatch.setitem(sys.modules, "novel_dataset", SimpleNamespace(
        chinese_5k="a", chinese_8k="b", chinese_40k="c", chinese_70k="d",
    ))
    monkeypatch.setattr(module.os, "environ", dict(os.environ))
    spawn = Mock(side_effect=AssertionError("CHECK_ONLY must not spawn any process"))
    monkeypatch.setattr(module.mp, "get_context", spawn)
    module.main([])
    saved = json.loads(next(tmp_path.glob("*/llm-kwargs.json")).read_text(encoding="utf-8"))
    assert saved["kv_transfer_config"]["kv_connector"] == "DSAL3Connector"
    spawn.assert_not_called()


def test_offline_native_status_and_reset_rpc_keep_secure_serialization(module, monkeypatch):
    pytest.importorskip("vllm")
    from vllm.v1.engine.core_client import SyncMPClient
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    quiet = {"busy": False, "pending_pause": False, "l3_pending": False, "fatal": None}
    client = SimpleNamespace(
        core_engine=b"0", encoder=MsgpackEncoder(), input_socket=Mock(), utility_results={},
        ensure_alive=Mock(), free_pending_messages=Mock(),
    )
    for name in ("call_utility", "_send_input", "reset_prefix_cache"):
        setattr(client, name, MethodType(getattr(SyncMPClient, name), client))
    requests = []

    def send(message, **kwargs):
        _, call_id, method, args = MsgpackDecoder().decode(message[2:])
        requests.append((method, args))
        result = quiet if method == "dsa_cache_admin" else True
        frames = MsgpackEncoder().encode(UtilityResult(result))
        client.utility_results[call_id].set_result(MsgpackDecoder(UtilityResult).decode(frames).result)

    client.input_socket.send_multipart.side_effect = send
    assert module._wait_idle(client, 5) == quiet
    assert client.reset_prefix_cache(reset_connector=False) is True
    assert requests == [("dsa_cache_admin", ["status"]), ("reset_prefix_cache", [False, False])]
    assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "0"
