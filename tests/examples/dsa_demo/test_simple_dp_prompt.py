# SPDX-License-Identifier: Apache-2.0
"""DP入门脚本的配置、阶段和跨进程合同；不需要权重或NPU设备。"""

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
        ),
    )
    return importlib.import_module("simple_dp_prompt_test")


def test_default_feature_stack_has_cold_local_and_cross_l3_rounds(demo):
    c = demo._settings()
    demo._validate(c)
    assert c["PROMPTS"] is demo.PROMPTS
    assert c["RUN_MODE"] == "graph" and c["REPLAY_ROUNDS"] == 3
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
            "SWAP_DP_ON_LAST_REPLAY",
        )
    )
    assert demo._build_kwargs(c)["kv_transfer_config"]["kv_role"] == "kv_both"


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
        {"PROMPT_DISTRIBUTION": "replicate"},
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
    c.update(REPLAY_ROUNDS=3, SWAP_DP_ON_LAST_REPLAY=True)
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
def test_native_engine_args_parse(demo, mtp, l3):
    pytest.importorskip("vllm")
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs

    c = demo._settings()
    c.update(
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
    assert args.compilation_config.cudagraph_mode.name == "FULL_DECODE_ONLY"
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
    "dsa,dp_size,l3", [(False, 1, False), (True, 1, False), (False, 2, False), (True, 2, False), (True, 2, True)]
)
def test_real_spawn_carries_settings_and_prints_results_without_npu(demo, tmp_path, dsa, dp_size, l3):
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
        if (dsa, dp_size, l3) == (True, 2, True)
        else f"""
demo.ENABLE_DSA_OFFLOAD = {dsa!r}
demo.ENABLE_MTP = {l3!r}
demo.ENABLE_L3_WRITE_BEHIND = {l3!r}
demo.ENABLE_L3_PREFIX_LOADBACK = {l3!r}
demo.DATA_PARALLEL_SIZE = {dp_size!r}
demo.TENSOR_PARALLEL_SIZE = {8 // dp_size!r}
demo.SWAP_DP_ON_LAST_REPLAY = {dp_size > 1!r}
"""
    )
    code = f"""
import sys
sys.path.insert(0, {str(Path(demo.__file__).parent)!r})
import simple_dp_prompt_test as demo
demo.MODEL_PATH = {str(model)!r}
demo.RESULT_DIR = {str(tmp_path / "results")!r}
demo.PROMPTS = ["first", "second"]
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
    salts = set()
    for rank in range(dp_size):
        invocation = json.loads((fake / f"worker-{rank}.json").read_text())
        assert invocation["rank"] == rank and invocation["seed"] == "114514"
        assert invocation["insecure"] == "0" and invocation["devices"] == "0,1,2,3,4,5,6,7"
        records = [
            json.loads(line) for line in next((tmp_path / "results").glob(f"*/dp{rank}.jsonl")).read_text().splitlines()
        ]
        assert [r["prompt_index"] for r in records] == ([rank, rank, 1 - rank] if dp_size == 2 else [0, 1, 0, 1, 0, 1])
        assert [r["cached_tokens"] for r in records] == ([0, 128, 0] if dp_size == 2 else [0, 0, 128, 128, 0, 0])
        assert all(r["dp_rank"] == rank and len(r["token_ids"]) == 128 for r in records)
        assert all(len(r["logprobs"]) == 128 and len(r["logprobs"][0]) == 5 for r in records)
        batches = [json.loads(line) for line in (fake / f"inputs-{rank}.jsonl").read_text().splitlines()]
        salts.update(request["cache_salt"] for batch in batches for request in batch)
    assert len(salts) == 1
    assert "Output: 'first'" in result.stdout and "Output: 'second'" in result.stdout
