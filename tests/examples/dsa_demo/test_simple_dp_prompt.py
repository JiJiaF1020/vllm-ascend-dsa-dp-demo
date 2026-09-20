# SPDX-License-Identifier: Apache-2.0
"""DP入门脚本的配置、阶段和跨进程合同；不需要权重或NPU设备。"""

import copy
import importlib
import json
import os
import socket
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/dsa_demo"))
    monkeypatch.setitem(
        sys.modules,
        "novel_dataset",
        SimpleNamespace(
            chinese_8k=["8k"],
            chinese_5k=["5k"],
            chinese_70k=["70k"],
            chinese_40k=["40k"],
            chinese_10k=["10k"],
            chinese_64k=["64k"],
        ),
    )
    return importlib.import_module("simple_dp_prompt_test")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_grouped_c8_switch_is_forwarded_without_changing_other_dsa_settings(demo, enabled, fused):
    c = demo._settings()
    c.update(ENABLE_GROUPED_C8_KV=enabled, ENABLE_DSA_OFFLOAD=True, ENABLE_A5_PACKED_C8_DSA=True,
             ENABLE_MTP=True, ENABLE_FUSED_COPYSFA=fused, ENABLE_PREFIX_CACHING=False,
             ENABLE_L3_WRITE_BEHIND=False, ENABLE_L3_PREFIX_LOADBACK=False, REPLAY_ROUNDS=1,
             RESET_LOCAL_CACHE_BEFORE_LAST_REPLAY=False, SWAP_DP_ON_LAST_REPLAY=False)
    demo._validate(c)
    kwargs = demo._build_kwargs(c)
    dsa = kwargs["additional_config"]["dsa_sparse_config"]
    assert dsa["enable_grouped_c8_kv"] is enabled
    assert dsa["enable_fused_copy_sfa"] is fused
    assert dsa["resident_budget_tokens"] == c["DSA_MTP_RESIDENT_BUDGET_TOKENS"]
    if enabled:
        for conflict in ({"ENABLE_PREFIX_CACHING": True},
                         {"ENABLE_MTP": False}, {"ENABLE_DSA_OFFLOAD": False}):
            with pytest.raises(ValueError, match="Grouped C8 KV"):
                demo._validate(dict(c, **conflict))


def test_default_feature_stack_has_three_rounds_and_cross_dp_loadback(demo):
    c = demo._settings()
    demo._validate(c)
    assert c["PROMPTS"] is demo.PROMPTS
    assert c["RUN_MODE"] == "graph" and c["REPLAY_ROUNDS"] == 3
    assert c["PROMPTS"] == ["8k", "5k", "70k", "40k"] and c["PROMPT_DISTRIBUTION"] == "round_robin"
    assert c["SWAP_DP_ON_LAST_REPLAY"] is True
    assert c["MAX_NUM_SEQS"] == 4 and c["MAX_TOKENS"] == 128 and c["LOGPROBS"] == 5
    assert c["IGNORE_EOS"] is True and c["CHECK_ONLY"] is False
    assert all(
        c[key]
        for key in (
            "ENABLE_DSA_OFFLOAD",
            "ENABLE_CHUNKED_PREFILL",
            "ENABLE_PREFIX_CACHING",
            "ENABLE_EXPERT_PARALLEL",
            "ENABLE_A5_PACKED_C8_DSA",
            "ENABLE_MTP",
            "ENABLE_L3_WRITE_BEHIND",
            "ENABLE_L3_PREFIX_LOADBACK",
            "RESET_LOCAL_CACHE_BEFORE_LAST_REPLAY",
        )
    )
    kwargs = demo._build_kwargs(c)
    assert kwargs["kv_transfer_config"]["kv_role"] == "kv_both"
    assert kwargs["enforce_eager"] is False
    assert kwargs["compilation_config"]["cudagraph_capture_sizes"] == [4, 8, 16]
    assert kwargs["additional_config"]["dsa_sparse_config"]["enable_row_mode_decode_graph"] is True
    assert "trace_points" not in kwargs["additional_config"]["dsa_sparse_config"]
    assert [demo._round_phase(c, index) for index in range(3)] == [
        "seed",
        "local-replay-1",
        "after-local-reset",
    ]


@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize(
    "dsa,mtp,l3",
    [
        (False, False, "off"),
        (False, True, "off"),
        (True, False, "off"),
        (True, True, "off"),
        (True, False, "write"),
        (True, True, "both"),
    ],
)
def test_settings_build_isolated_spmd_kwargs(demo, mode, dsa, mtp, l3, monkeypatch):
    monkeypatch.setitem(sys.modules, "simple_prompt_test", None)
    monkeypatch.setitem(sys.modules, "offline_dp_closeout_test", None)
    c = demo._settings()
    c.update(
        RUN_MODE=mode,
        ENABLE_DSA_OFFLOAD=dsa,
        ENABLE_MTP=mtp,
        ENABLE_L3_WRITE_BEHIND=l3 != "off",
        ENABLE_L3_PREFIX_LOADBACK=l3 == "both",
        ENABLE_PREFIX_CACHING=l3 == "both",
    )
    demo._validate(c)
    kwargs = demo._build_kwargs(c)
    assert "data_parallel_size" not in kwargs
    assert kwargs["tensor_parallel_size"] == 4 and kwargs["async_scheduling"] is False
    assert kwargs["enforce_eager"] == (mode == "eager")
    assert ("dsa_sparse_config" in kwargs["additional_config"]) == dsa
    assert kwargs["additional_config"]["enable_sparse_sfa_c8"] is True
    assert (kwargs.get("speculative_config") is not None) == mtp
    if mtp:
        assert kwargs["speculative_config"]["enforce_eager"] is True
    if mode == "graph":
        assert kwargs["compilation_config"]["cudagraph_capture_sizes"] == ([4, 8, 16] if mtp else [1, 2, 4])
    else:
        assert "compilation_config" not in kwargs
    if l3 != "off":
        assert kwargs["kv_transfer_config"]["kv_role"] == ("kv_both" if l3 == "both" else "kv_producer")
        assert kwargs["kv_transfer_config"]["kv_connector_extra_config"]["direct_hbm_load"] is False
    else:
        assert "kv_transfer_config" not in kwargs


@pytest.mark.parametrize(
    "mtp,custom_budgets,expected",
    [
        (True, None, [8192, 10240, 12288]),
        (True, [8320, 12416, 16000], [8320, 12416, 16000]),
        (True, [16000, 16000, 16000], [16000, 16000, 16000]),
        (False, [16000, 16000, 16000], [6144, 10240, 12288]),
    ],
)
def test_resident_budget_configuration_preserves_defaults(demo, mtp, custom_budgets, expected):
    c = demo._settings()
    c.update(ENABLE_DSA_OFFLOAD=True, ENABLE_MTP=mtp)
    if custom_budgets is not None:
        c["DSA_MTP_RESIDENT_BUDGET_TOKENS"] = custom_budgets
    dsa = demo._build_kwargs(c)["additional_config"]["dsa_sparse_config"]
    assert dsa["resident_budget_tokens"] == expected
    assert dsa["sparse_activation_tokens"] == (8192 if mtp else 6144)


def test_single_domain_and_native_baseline_controls(demo):
    c = demo._settings()
    c.update(
        DATA_PARALLEL_SIZE=1,
        TENSOR_PARALLEL_SIZE=8,
        ENABLE_DSA_OFFLOAD=False,
        ENABLE_A5_PACKED_C8_DSA=False,
        ENABLE_EXPERT_PARALLEL=False,
        ENABLE_MTP=False,
        ENABLE_L3_WRITE_BEHIND=False,
        ENABLE_L3_PREFIX_LOADBACK=False,
        SWAP_DP_ON_LAST_REPLAY=False,
    )
    demo._validate(c)
    kwargs = demo._build_kwargs(c)
    assert "additional_config" not in kwargs and "kv_transfer_config" not in kwargs
    core = SimpleNamespace(call_utility=Mock(side_effect=AssertionError("DP1 cannot use DP administration")))
    c["ENABLE_DSA_OFFLOAD"] = True
    demo._wait_idle(core, c)
    core.call_utility.assert_not_called()
    c.update(ENABLE_L3_WRITE_BEHIND=True, ENABLE_EXPERT_PARALLEL=True)
    with pytest.raises(ValueError, match="DP1\\+L3"):
        demo._validate(c)


@pytest.mark.parametrize(
    "changes",
    [
        {"DATA_PARALLEL_SIZE": 1},
        {"DATA_PARALLEL_SIZE": 0},
        {"TENSOR_PARALLEL_SIZE": 0},
        {"VISIBLE_DEVICES": "0,1"},
        {"VISIBLE_DEVICES": "0,1,2,3,4,5,6,6"},
        {"RUN_MODE": "disabled"},
        {"ENABLE_DSA_OFFLOAD": "False"},
        {"ENABLE_EXPERT_PARALLEL": False},
        {"MAX_NUM_SEQS": 257},
        {"ENABLE_L3_WRITE_BEHIND": True, "ENABLE_DSA_OFFLOAD": False},
        {"ENABLE_L3_WRITE_BEHIND": False},
        {"ENABLE_PREFIX_CACHING": False},
        {"REPLAY_ROUNDS": 1},
        {"PROMPT_DISTRIBUTION": "invalid"},
        {"SWAP_DP_ON_LAST_REPLAY": "false"},
        {"SWAP_DP_ON_LAST_REPLAY": True, "REPLAY_ROUNDS": 2, "PROMPT_DISTRIBUTION": "replicate"},
        {"ENABLE_MTP": True, "MTP_NUM_SPECULATIVE_TOKENS": 2},
        {"ENABLE_MTP": True, "TEMPERATURE": 0.8},
        {"ENABLE_MTP": True, "ENABLE_A5_PACKED_C8_DSA": False},
        {"ENABLE_CHUNKED_PREFILL": False},
        {"GPU_MEMORY_UTILIZATION": 1.1},
        {"PROCESS_TIMEOUT_SECONDS": float("nan")},
        {"TEMPERATURE": float("nan")},
        {"GRAPH_CAPTURE_ROW_SIZES": [0]},
        {"LOGPROBS": -1},
        {"MAX_TOKENS": 100000},
        {"MOONCAKE_TCP_SEGMENT_PORT_BASE": 45001},
        {"DP_MASTER_PORT": 0.5},
        {"DSA_PROMPT_BUDGET_THRESHOLDS": [65536, 32768]},
        {"ENABLE_MTP": False, "DSA_RESIDENT_BUDGET_TOKENS": [1]},
        {"PROMPTS": []},
        {"PROMPTS": "not a list"},
        {"PROMPTS": [None]},
        {"PROMPTS": [""]},
        {"NATIVE_RUNTIME_ENV": {"VLLM_DP_RANK": "0"}},
        {"NATIVE_RUNTIME_ENV": {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}},
    ],
)
def test_bad_user_settings_fail_before_loading_model(demo, changes):
    c = demo._settings()
    c.update(changes)
    with pytest.raises(ValueError):
        demo._validate(c)


def test_user_prompts_distribution_and_rotation(demo):
    c = demo._settings()
    c["PROMPTS"] = ["custom-a", "custom-b", "custom-c", "custom-d", "custom-e"]
    original_prompts = c["PROMPTS"]
    demo._validate(c)
    assert c["PROMPTS"] is original_prompts
    c.update(
        REPLAY_ROUNDS=3, SWAP_DP_ON_LAST_REPLAY=True, PROMPT_DISTRIBUTION="round_robin"
    )
    assert demo._indices(c, 5, 0, 0) == [0, 2, 4]
    assert demo._indices(c, 5, 1, 0) == [1, 3]
    assert demo._indices(c, 5, 0, 2) == [1, 3]
    assert demo._indices(c, 5, 1, 2) == [0, 2, 4]
    c["PROMPTS"] = ["one"]
    with pytest.raises(ValueError, match="at least one prompt"):
        demo._validate(c)
    c.update(PROMPT_DISTRIBUTION="replicate", SWAP_DP_ON_LAST_REPLAY=False)
    demo._validate(c)
    assert c["PROMPTS"] == ["one"]
    assert demo._indices(c, 1, 1, 0) == [0]


def test_preflight_checks_only_enabled_backend(demo, monkeypatch, tmp_path):
    c = demo._settings()
    c.update(MODEL_PATH=str(tmp_path), ENABLE_L3_WRITE_BEHIND=False, ENABLE_L3_PREFIX_LOADBACK=False)
    with pytest.raises(ValueError, match="config.json"):
        demo._preflight(c)
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(demo.importlib.util, "find_spec", lambda _: object())
    connect = Mock()
    monkeypatch.setattr(demo.socket, "create_connection", connect)
    demo._preflight(c)
    connect.assert_not_called()
    c.update(ENABLE_L3_WRITE_BEHIND=True, MOONCAKE_CONFIG_PATH=tmp_path / "mooncake.json")
    c["MOONCAKE_CONFIG_PATH"].write_text('{"protocol":"ascend"}')
    with pytest.raises(ValueError, match="protocol=tcp"):
        demo._preflight(c)
    c["MOONCAKE_CONFIG_PATH"].write_text('{"protocol":"tcp","master_server_address":"127.0.0.1:50088"}')
    monkeypatch.delenv("MOONCAKE_MASTER", raising=False)
    connect.side_effect = ConnectionRefusedError("no master")
    with pytest.raises(ConnectionRefusedError):
        demo._preflight(c)


@pytest.mark.parametrize(
    "parent,child,expected",
    [
        (None, None, 50088),
        ("127.0.0.1:50089", None, 50089),
        ("127.0.0.1:50089", "127.0.0.1:50090", 50090),
        ("", None, None),
        ("127.0.0.1:50089", "", None),
    ],
)
def test_preflight_master_matches_effective_worker_environment(demo, monkeypatch, tmp_path, parent, child, expected):
    c = demo._settings()
    c.update(
        MODEL_PATH=str(tmp_path),
        MOONCAKE_CONFIG_PATH=tmp_path / "mooncake.json",
        NATIVE_RUNTIME_ENV={} if child is None else {"MOONCAKE_MASTER": child},
    )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    c["MOONCAKE_CONFIG_PATH"].write_text(
        '{"protocol":"tcp","master_server_address":"127.0.0.1:50088"}',
        encoding="utf-8",
    )
    if parent is None:
        monkeypatch.delenv("MOONCAKE_MASTER", raising=False)
    else:
        monkeypatch.setenv("MOONCAKE_MASTER", parent)
    monkeypatch.setattr(demo.importlib.util, "find_spec", lambda _: object())
    connect = Mock(return_value=nullcontext())
    monkeypatch.setattr(demo.socket, "create_connection", connect)
    if expected is None:
        with pytest.raises(ValueError, match="master address"):
            demo._preflight(c)
        connect.assert_not_called()
    else:
        demo._preflight(c)
        connect.assert_called_once_with(("127.0.0.1", expected), timeout=3)
    assert os.environ.get("MOONCAKE_MASTER") == parent


@pytest.mark.parametrize("mtp,l3", [(False, "off"), (True, "off"), (False, "write"), (True, "both")])
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_native_engine_args_parse(demo, mtp, l3, mode):
    pytest.importorskip("vllm")
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs

    c = demo._settings()
    c.update(
        RUN_MODE=mode,
        ENABLE_MTP=mtp,
        ENABLE_L3_WRITE_BEHIND=l3 != "off",
        ENABLE_L3_PREFIX_LOADBACK=l3 == "both",
        ENABLE_PREFIX_CACHING=l3 == "both",
    )
    kwargs = demo._build_kwargs(c)
    if "kv_transfer_config" in kwargs:
        kwargs["kv_transfer_config"] = KVTransferConfig(**kwargs["kv_transfer_config"])
    args = EngineArgs(**kwargs)
    assert args.tensor_parallel_size == 4 and args.async_scheduling is False
    assert args.enforce_eager == (mode == "eager")
    if mode == "graph":
        assert args.compilation_config.cudagraph_mode.name == "FULL_DECODE_ONLY"
    else:
        assert "compilation_config" not in kwargs
    if l3 != "off":
        assert args.kv_transfer_config.kv_connector == "DSAL3Connector"


def test_native_status_rpc_stays_on_secure_msgpack_path(demo, monkeypatch):
    pytest.importorskip("vllm")
    from vllm.v1.engine.core_client import SyncMPClient
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    client = SimpleNamespace(
        core_engine=b"0",
        encoder=MsgpackEncoder(),
        input_socket=Mock(),
        utility_results={},
        ensure_alive=Mock(),
        free_pending_messages=Mock(),
    )
    for name in ("call_utility", "_send_input"):
        setattr(client, name, MethodType(getattr(SyncMPClient, name), client))
    sent = []

    def send(message, **kwargs):
        _, call_id, method, args = MsgpackDecoder().decode(message[2:])
        sent.append((method, args))
        status = {"busy": False, "pending_pause": False, "l3_pending": False, "fingerprint": "schema", "fatal": None}
        wire = MsgpackEncoder().encode(UtilityResult(status))
        client.utility_results[call_id].set_result(MsgpackDecoder(UtilityResult).decode(wire).result)

    client.input_socket.send_multipart.side_effect = send
    c = demo._settings()
    c["ENABLE_DSA_OFFLOAD"] = False
    demo._wait_idle(client, c)
    assert sent == []
    c.update(ENABLE_DSA_OFFLOAD=True, ENABLE_L3_WRITE_BEHIND=True)
    demo._wait_idle(client, c)
    assert sent == [("dsa_cache_admin", ["status"])]


def test_check_only_does_not_spawn_or_change_runtime_environment(demo, monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "CHECK_ONLY", True)
    monkeypatch.setattr(demo, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(demo, "_preflight", Mock())
    monkeypatch.setenv("VLLM_DP_SIZE", "unchanged")
    spawn = Mock(side_effect=AssertionError("CHECK_ONLY cannot spawn"))
    monkeypatch.setattr(demo.mp, "get_context", spawn)
    demo.main()
    spawn.assert_not_called()
    assert os.environ["VLLM_DP_SIZE"] == "unchanged"
    kwargs = json.loads(next(tmp_path.glob("*/llm-kwargs.json")).read_text())
    assert "data_parallel_size" not in kwargs
    assert not list(tmp_path.glob("*/precision_compact"))


@pytest.mark.parametrize("failure", ["connector", "mtp", "generate"])
def test_worker_reports_original_failure_before_cleanup(demo, monkeypatch, tmp_path, failure):
    c = demo._settings()
    c.update(ENABLE_MTP=True, ENABLE_L3_WRITE_BEHIND=True, ENABLE_L3_PREFIX_LOADBACK=True, ENABLE_PREFIX_CACHING=True)
    monkeypatch.setattr(demo.os, "environ", dict(os.environ))
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2, data_parallel_rank=0, tensor_parallel_size=4),
        additional_config={"dsa_sparse_config": {"enabled": True}},
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        kv_transfer_config=None
        if failure == "connector"
        else SimpleNamespace(kv_connector="DSAL3Connector", kv_role="kv_both"),
        speculative_config=None if failure == "mtp" else SimpleNamespace(method="mtp", num_speculative_tokens=3),
    )
    core = SimpleNamespace(
        call_utility=Mock(
            return_value={"busy": False, "pending_pause": False, "l3_pending": False, "fingerprint": "ok"}
        ),
        shutdown=Mock(side_effect=RuntimeError("secondary cleanup failure")),
    )
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(vllm_config=config, engine_core=core),
        generate=Mock(side_effect=ValueError("primary generate failure")),
    )
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=lambda **_: llm, SamplingParams=SimpleNamespace))
    barrier = SimpleNamespace(wait=Mock(), abort=Mock())
    pattern = {
        "connector": "actual L3 connector",
        "mtp": "actual MTP settings",
        "generate": "primary generate failure",
    }[failure]
    with pytest.raises(ValueError if failure == "generate" else RuntimeError, match=pattern):
        demo._worker(0, c, ["one", "two"], 12345, barrier, tmp_path)
    assert pattern in (tmp_path / "dp0-failure.txt").read_text(encoding="utf-8")
    barrier.abort.assert_called_once()
    core.shutdown.assert_called_once_with(timeout=300)
    if failure != "generate":
        llm.generate.assert_not_called()


@pytest.mark.parametrize(
    "dsa,dp_size,l3,use_defaults",
    [
        (False, 1, False, False),
        (True, 1, False, False),
        (False, 2, False, False),
        (True, 2, False, False),
        (True, 2, True, False),
        (True, 2, True, True),
    ],
)
@pytest.mark.parametrize("mode", ["eager", "graph"])
def test_real_spawn_carries_settings_and_prints_results_without_npu(
    demo, tmp_path, dsa, dp_size, l3, use_defaults, mode
):
    # Real spawn, config transfer, files and stdout; a tiny fake vLLM replaces only
    # the model/runtime. Native config and RPC codecs are exercised separately above.
    fake = tmp_path / "fake_packages"
    fake.mkdir()
    for name in ("torch_npu", "vllm_ascend", "mooncake"):
        (fake / f"{name}.py").write_text("")
    (fake / "novel_dataset.py").write_text(
        "chinese_8k = ['8k']\nchinese_5k = ['5k']\nchinese_70k = ['70k']\nchinese_40k = ['40k']\n",
        encoding="utf-8",
    )
    (fake / "vllm.py").write_text(
        """
import json, os
from pathlib import Path
from types import SimpleNamespace as NS
SamplingParams = NS
class Core:
    def __init__(self, dsa): self.dsa = dsa
    def call_utility(self, *args):
        assert self.dsa and args == ('dsa_cache_admin', 'status')
        return dict(busy=False, pending_pause=False, l3_pending=False, fatal=None, fingerprint='schema')
    def shutdown(self, timeout): assert timeout == 300
class LLM:
    def __init__(self, **kwargs):
        assert 'data_parallel_size' not in kwargs
        rank = int(os.environ['VLLM_DP_RANK'])
        self.rank = rank
        self.kwargs = kwargs
        self.cached = False
        cfg = NS(parallel_config=NS(data_parallel_size=int(os.environ['VLLM_DP_SIZE']),
                 data_parallel_rank=rank, tensor_parallel_size=kwargs['tensor_parallel_size']),
                 kv_transfer_config=NS(**kwargs['kv_transfer_config']) if 'kv_transfer_config' in kwargs else None,
                 speculative_config=NS(**kwargs['speculative_config']) if 'speculative_config' in kwargs else None,
                 additional_config=kwargs['additional_config'],
                 cache_config=NS(enable_prefix_caching=kwargs['enable_prefix_caching']),
                 scheduler_config=NS(async_scheduling=kwargs['async_scheduling']))
        self.llm_engine = NS(vllm_config=cfg, engine_core=Core('dsa_sparse_config' in kwargs['additional_config']))
        payload = dict(rank=rank, kwargs=kwargs, seed=os.environ['PYTHONHASHSEED'],
                       devices=os.environ['ASCEND_RT_VISIBLE_DEVICES'],
                       insecure=os.environ.get('VLLM_ALLOW_INSECURE_SERIALIZATION'))
        (Path(__file__).parent / f'worker-{rank}.json').write_text(json.dumps(payload))
    def generate(self, inputs, params, use_tqdm):
        assert not use_tqdm
        assert params.ignore_eos is True
        with (Path(__file__).parent / f'inputs-{self.rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(inputs) + chr(10))
        results = [NS(request_id=f'{self.rank}-{i}', prompt_token_ids=[1]*256,
                      num_cached_tokens=128 if self.cached else 0,
                      outputs=[NS(token_ids=[7]*params.max_tokens, text=p['prompt'],
                                  finish_reason='length',
                                  logprobs=[{7+j:NS(logprob=-float(j)) for j in range(params.logprobs)}
                                            for _ in range(params.max_tokens)] if params.logprobs else None)])
                   for i,p in enumerate(inputs)]
        self.cached = True
        return results
    def reset_prefix_cache(self, reset_connector):
        assert reset_connector is False
        batches = (Path(__file__).parent / f'inputs-{self.rank}.jsonl').read_text().splitlines()
        with (Path(__file__).parent / f'resets-{self.rank}.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(reset_connector=reset_connector, after_batches=len(batches))))
            handle.write(chr(10))
        self.cached = False
        return True
""",
        encoding="utf-8",
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    overrides = (
        ""
        if use_defaults
        else f"""
demo.ENABLE_DSA_OFFLOAD = {dsa!r}
demo.ENABLE_MTP = {l3!r}
demo.ENABLE_L3_WRITE_BEHIND = {l3!r}
demo.ENABLE_L3_PREFIX_LOADBACK = {l3!r}
demo.DATA_PARALLEL_SIZE = {dp_size!r}
demo.TENSOR_PARALLEL_SIZE = {8 // dp_size!r}
demo.SWAP_DP_ON_LAST_REPLAY = {dp_size > 1!r}
demo.PROMPT_DISTRIBUTION = "round_robin"
demo.REPLAY_ROUNDS = 3
demo.PROMPTS = ["first", "second"]
"""
    )
    code = f"""
import sys
sys.path.insert(0, {str(Path(demo.__file__).parent)!r})
import simple_dp_prompt_test as demo
demo.RUN_MODE = {mode!r}
demo.MODEL_PATH = {str(model)!r}
demo.RESULT_DIR = {str(tmp_path / "results")!r}
demo.MOONCAKE_CONFIG_PATH = {str(tmp_path / "mooncake.json")!r}
{overrides}
demo.main()
"""
    with socket.socket() as master:
        master.bind(("127.0.0.1", 0))
        master.listen()
        address = f"127.0.0.1:{master.getsockname()[1]}"
        (tmp_path / "mooncake.json").write_text(json.dumps({"protocol": "tcp", "master_server_address": address}))
        env = {
            **os.environ,
            "PYTHONPATH": str(fake),
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "0",
            "MOONCAKE_MASTER": address,
        }
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=25
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "COMPLETED results=" in result.stdout
    assert "REPORT_READY" in result.stdout
    report = json.loads(
        next((tmp_path / "results").glob("*/precision_compact/precision_details.json")).read_text(encoding="utf-8")
    )
    assert report["settings"]["DATA_PARALLEL_SIZE"] == dp_size
    assert report["settings"]["RUN_MODE"] == mode
    assert not report["warnings"]
    assert len(report["records"]) == (12 if use_defaults else 6)
    assert len(report["comparisons"]) == (8 if use_defaults else 4)
    salts = set()
    for rank in range(dp_size):
        invocation = json.loads((fake / f"worker-{rank}.json").read_text())
        assert invocation["kwargs"]["enforce_eager"] == (mode == "eager")
        assert ("compilation_config" in invocation["kwargs"]) == (mode == "graph")
        assert invocation["rank"] == rank and invocation["seed"] == "114514"
        assert invocation["insecure"] == "0" and invocation["devices"] == "0,1,2,3,4,5,6,7"
        records = [
            json.loads(line) for line in next((tmp_path / "results").glob(f"*/dp{rank}.jsonl")).read_text().splitlines()
        ]
        expected_indices = (
            [rank, rank + 2] * 2 + [1 - rank, 3 - rank]
            if use_defaults
            else ([rank, rank, 1 - rank] if dp_size == 2 else [0, 1] * 3)
        )
        assert [r["prompt_index"] for r in records] == expected_indices
        assert [r["cached_tokens"] for r in records] == (
            [0, 0, 128, 128, 0, 0]
            if use_defaults
            else ([0, 128, 0] if dp_size == 2 else [0, 0, 128, 128, 0, 0])
        )
        assert all(r["dp_rank"] == rank and len(r["token_ids"]) == 128 for r in records)
        assert all(len(r["logprobs"]) == 128 and len(r["logprobs"][0]) == 5 for r in records)
        batches = [json.loads(line) for line in (fake / f"inputs-{rank}.jsonl").read_text().splitlines()]
        rank_salts = {request["cache_salt"] for batch in batches for request in batch}
        assert len(rank_salts) == 1 and {row["cache_salt"] for row in records} == rank_salts
        salts.update(rank_salts)
        resets = [json.loads(line) for line in (fake / f"resets-{rank}.jsonl").read_text().splitlines()]
        assert resets == [{"reset_connector": False, "after_batches": 2}]
        if use_defaults:
            prompts = ["8k", "5k", "70k", "40k"]
            assert [request["prompt"] for batch in batches for request in batch] == [
                prompts[index] for index in expected_indices
            ]
            assert [row["phase"] for row in records[::2]] == [
                "seed",
                "local-replay-1",
                "after-local-reset",
            ]
    assert len(salts) == 1
    if use_defaults:
        assert all(f"Output: '{prompt}'" in result.stdout for prompt in ("8k", "5k", "70k", "40k"))
        assert sum(pair["left"]["dp_rank"] != pair["right"]["dp_rank"] for pair in report["comparisons"]) == 4
        assert all(pair["left"]["cache_salt"] == pair["right"]["cache_salt"] for pair in report["comparisons"])
    else:
        assert "Output: 'first'" in result.stdout and "Output: 'second'" in result.stdout


# 摘要属于simple-DP的结果输出，和启动/输出门禁在同一文件维护，不依赖NPU。
def _row(rank, round_index, prompt_index, tokens=None):
    tokens = tokens if tokens is not None else [10, 11, 12]
    return {
        "dp_rank": rank,
        "round": round_index,
        "prompt_index": prompt_index,
        "request_id": f"req-{rank}-{round_index}-{prompt_index}",
        "prompt": f"private-input-{prompt_index}",
        "prompt_tokens": 4096 + prompt_index,
        "cached_tokens": 0 if round_index == 1 else 3968,
        "token_ids": tokens,
        "text": "输出文本",
        "finish_reason": "length",
        "logprobs": [{str(token): -0.1, "99": -4.0} for token in tokens],
    }


def _write(root, rows, **settings):
    dp_size = settings.get("DATA_PARALLEL_SIZE", 2)
    for rank in range(dp_size):
        (root / f"dp{rank}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if row["dp_rank"] == rank),
            encoding="utf-8",
        )
    (root / "settings.json").write_text(
        json.dumps(
            {
                "DATA_PARALLEL_SIZE": 2,
                "REPLAY_ROUNDS": 3,
                "PROMPTS": ["large prompt 0", "large prompt 1"],
                "PROMPT_DISTRIBUTION": "round_robin",
                "ENABLE_MTP": True,
                **settings,
            }
        ),
        encoding="utf-8",
    )
    for rank in range(dp_size):
        (root / f"dp{rank}-engine.json").write_text(
            json.dumps({"dp_rank": rank, "fingerprint": "same"}), encoding="utf-8"
        )


def _swapped_rows():
    return [
        _row(index if round_index < 3 else 1 - index, round_index, index)
        for round_index in (1, 2, 3)
        for index in (0, 1)
    ]


def test_pairs_by_prompt_not_file_order_after_dp_swap(demo, tmp_path):
    _write(tmp_path, list(reversed(_swapped_rows())))
    report = demo._summarize_results(tmp_path)
    assert not report["warnings"]
    assert len(report["records"]) == 6
    assert len(report["comparisons"]) == 4
    warm = [item for item in report["comparisons"] if item["left"]["round"] == 2]
    assert {(item["left"]["dp_rank"], item["right"]["dp_rank"]) for item in warm} == {(0, 1), (1, 0)}
    assert all(item["status"] == "tokens_match" for item in report["comparisons"])
    assert all(item["returned_topk_equal_before_feedback"] is True for item in report["comparisons"])
    assert all("prompt" not in row and "logprobs" not in row and "_logprobs" not in row for row in report["records"])
    assert "PROMPTS" not in report["settings"]


def test_first_divergence_does_not_treat_later_feedback_as_drift(demo, tmp_path):
    rows = _swapped_rows()
    right = next(row for row in rows if row["round"] == 3 and row["prompt_index"] == 0)
    right["token_ids"] = [10, 20, 21]
    right["logprobs"] = [{"10": -0.1, "99": -4.0}, {"11": -1.1, "20": -0.1, "99": -4.0}, {"99": -999.0}]
    _write(tmp_path, rows)
    result = demo._summarize_results(tmp_path)["comparisons"][0]
    assert result["status"] == "tokens_differ"
    assert result["first_difference_0based"] == 1
    assert (result["left_token"], result["right_token"]) == (11, 20)
    assert result["max_shared_logprob_delta_before_feedback"] == pytest.approx(1.0)
    after = next(row for row in result["logprob_window"] if row["position_0based"] == 2)
    assert after["same_generated_history"] is False


def test_length_and_missing_logprobs_are_reported(demo, tmp_path):
    rows = _swapped_rows()
    right = next(row for row in rows if row["round"] == 3 and row["prompt_index"] == 0)
    right["token_ids"] = [10, 11]
    right["finish_reason"] = "stop"
    del right["logprobs"]
    _write(tmp_path, rows)
    report = demo._summarize_results(tmp_path)
    result = report["comparisons"][0]
    assert result["first_difference_0based"] == 2
    assert result["right_token"] is None
    assert result["finish_reason_match"] is False
    assert result["max_shared_logprob_delta_before_feedback"] is None
    assert result["returned_topk_equal_before_feedback"] is None
    assert any("logprobs missing" in warning for warning in report["warnings"])


def test_prompt_change_is_not_numerical_mismatch(demo, tmp_path):
    rows = _swapped_rows()
    rows[-2]["prompt"] = "a different input"
    _write(tmp_path, rows)
    assert demo._summarize_results(tmp_path)["comparisons"][0]["status"] == "not_comparable"


def test_replicated_prompts_compare_same_rank(demo, tmp_path):
    rows = [_row(rank, round_index, index) for rank in (0, 1) for round_index in (1, 2, 3) for index in (0, 1)]
    _write(tmp_path, rows, PROMPT_DISTRIBUTION="replicate")
    report = demo._summarize_results(tmp_path)
    assert not report["warnings"]
    assert len(report["comparisons"]) == 8
    assert all(item["left"]["dp_rank"] == item["right"]["dp_rank"] for item in report["comparisons"])


def test_summary_preserves_legacy_diagnostic_conditions_without_runtime_switches(demo, tmp_path):
    rows = [_row(rank, round_index, 0) for rank in (0, 1) for round_index in (1, 2, 3, 4)]
    for row in rows:
        row["cache_salt"] = f"old-run-dp{row['dp_rank']}"
    _write(
        tmp_path, rows, REPLAY_ROUNDS=4, PROMPT_DISTRIBUTION="replicate",
        ISOLATE_DP_CACHE_SALT=True, TRACE_POOL_LIFECYCLE=True,
    )
    report = demo._summarize_results(tmp_path)
    for key in ("ISOLATE_DP_CACHE_SALT", "TRACE_POOL_LIFECYCLE"):
        assert key not in demo._settings()
        assert report["settings"][key] is True
    assert len(report["comparisons"]) == 6
    assert all(item["left"]["cache_salt"] == item["right"]["cache_salt"] for item in report["comparisons"])


@pytest.mark.parametrize("problem", ["duplicate", "rank", "json", "missing_file"])
def test_bad_input_is_not_silently_accepted(demo, tmp_path, problem):
    rows = _swapped_rows()
    if problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    _write(tmp_path, rows)
    if problem == "rank":
        row = _row(1, 1, 0)
        (tmp_path / "dp0.jsonl").write_text(json.dumps(row), encoding="utf-8")
    elif problem == "json":
        (tmp_path / "dp0.jsonl").write_text('{"incomplete":', encoding="utf-8")
    elif problem == "missing_file":
        (tmp_path / "dp1.jsonl").unlink()
    with pytest.raises((ValueError, OSError)):
        demo._summarize_results(tmp_path)


def test_incomplete_round_warns_and_nonfinite_is_valid_json(demo, tmp_path):
    rows = _swapped_rows()[:-1]
    rows[0]["logprobs"][0]["99"] = float("-inf")
    _write(tmp_path, rows)
    report = demo._summarize_results(tmp_path)
    assert any("missing or ambiguous" in warning for warning in report["warnings"])
    assert any("expected 1 record" in warning for warning in report["warnings"])
    assert any("non-finite" in warning for warning in report["warnings"])
    json.dumps(report, allow_nan=False)


def test_matching_tokens_still_preserve_probability_drift(demo, tmp_path):
    rows = _swapped_rows()
    rows[-2]["logprobs"][1]["11"] = -0.6
    _write(tmp_path, rows)
    result = demo._summarize_results(tmp_path)["comparisons"][0]
    assert result["status"] == "tokens_match"
    assert result["max_delta_position_0based"] == 1
    assert result["max_shared_logprob_delta_before_feedback"] == pytest.approx(0.5)


def test_summary_only_creates_small_reports_without_runtime_or_data_imports(demo, tmp_path):
    rows = _swapped_rows()
    for row in rows:
        row["prompt"] += " secret-long-input" * 10000
    _write(tmp_path, rows)
    originals = {name: (tmp_path / name).read_bytes() for name in ("dp0.jsonl", "dp1.jsonl", "settings.json")}
    script = tmp_path / "simple_dp_prompt_test.py"
    script.write_text(
        Path(demo.__file__)
        .read_text(encoding="utf-8")
        .replace('SUMMARY_ONLY_DIR = ""', f"SUMMARY_ONLY_DIR = {str(tmp_path)!r}", 1),
        encoding="utf-8",
    )
    for name in ("novel_dataset", "vllm", "torch_npu", "mooncake"):
        (tmp_path / f"{name}.py").write_text("raise AssertionError('summary must not import runtime or data')")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "REPORT_READY" in result.stdout
    assert "PASSED" not in result.stdout
    details = tmp_path / "precision_compact/precision_details.json"
    assert details.stat().st_size < sum(len(data) for data in originals.values()) // 10
    assert "secret-long-input" not in details.read_text(encoding="utf-8")
    for name, data in originals.items():
        assert (tmp_path / name).read_bytes() == data


def test_keeps_earliest_and_largest_probability_change_before_token_difference(demo, tmp_path):
    rows = _swapped_rows()
    for row in rows:
        row["token_ids"] = [10, 11, 12, 13, 14, 15]
        row["logprobs"] = [{str(token): -0.1, "99": -4.0} for token in row["token_ids"]]
    right = next(row for row in rows if row["round"] == 3 and row["prompt_index"] == 0)
    right["token_ids"][-1] = 16
    right["logprobs"][1]["11"] = -0.3
    right["logprobs"][2]["12"] = -0.8
    right["logprobs"][5] = {"15": -0.2, "16": -0.1, "99": -4.0}
    _write(tmp_path, rows)
    result = demo._summarize_results(tmp_path)["comparisons"][0]
    assert result["first_difference_0based"] == 5
    assert result["first_shared_logprob_change_0based"] == 1
    assert result["max_delta_position_0based"] == 2
    assert {1, 2, 5} <= {sample["position_0based"] for sample in result["logprob_window"]}
    assert result["returned_topk_equal_before_feedback"] is False


def test_zero_intersection_delta_does_not_claim_topk_sets_equal(demo, tmp_path):
    rows = _swapped_rows()
    rows[-2]["logprobs"][0] = {"10": -0.1, "98": -4.0}
    _write(tmp_path, rows)
    result = demo._summarize_results(tmp_path)["comparisons"][0]
    assert result["status"] == "tokens_match"
    assert result["max_shared_logprob_delta_before_feedback"] == 0
    assert result["first_shared_logprob_change_0based"] is None
    assert result["first_candidate_set_change_0based"] == 0
    assert result["returned_topk_equal_before_feedback"] is False


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("rounds", [1, 2, 3, 4])
@pytest.mark.parametrize("distribution", ["round_robin", "replicate"])
def test_summary_uses_actual_rank_count_and_rounds(demo, tmp_path, dp_size, rounds, distribution):
    rows = [
        _row(rank, round_index, index)
        for round_index in range(1, rounds + 1)
        for index in (0, 1)
        for rank in range(dp_size)
        if distribution == "replicate" or rank == index % dp_size
    ]
    _write(tmp_path, rows, DATA_PARALLEL_SIZE=dp_size, REPLAY_ROUNDS=rounds, PROMPT_DISTRIBUTION=distribution)
    report = demo._summarize_results(tmp_path)
    assert not report["warnings"]
    assert len(report["records"]) == len(rows)
    copies = dp_size if distribution == "replicate" else 1
    assert len(report["comparisons"]) == 2 * copies * (rounds - 1)
    if rounds > 1:
        assert report["comparisons"][0]["left"]["round"] == rounds - 1
        assert report["comparisons"][0]["right"]["round"] == rounds


def test_summary_only_does_not_validate_current_inference_settings(demo, monkeypatch, tmp_path):
    _write(tmp_path, _swapped_rows())
    monkeypatch.setattr(demo, "SUMMARY_ONLY_DIR", tmp_path)
    for name in ("_settings", "_validate", "_preflight"):
        monkeypatch.setattr(demo, name, Mock(side_effect=AssertionError("summary must not prepare inference")))
    monkeypatch.setattr(demo.mp, "get_context", Mock(side_effect=AssertionError("summary cannot spawn")))
    demo.main()
    assert (tmp_path / "precision_compact/precision_summary.txt").is_file()


def test_summary_without_settings_does_not_claim_run_is_complete(demo, tmp_path):
    _write(tmp_path, _swapped_rows())
    (tmp_path / "settings.json").unlink()
    report = demo._summarize_results(tmp_path)
    assert len(report["records"]) == 6 and len(report["comparisons"]) == 4
    assert any("completeness is not verified" in warning for warning in report["warnings"])
