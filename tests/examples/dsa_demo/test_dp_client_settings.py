# SPDX-License-Identifier: Apache-2.0
"""验证 DP 客户端从环境变量到真实请求辅助函数的配置传递，不加载模型或 NPU。"""

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CLIENTS = (
    ("online_dp_foundation_test", 32, 5),
    ("online_dp_prefix_test", 32, 16),
    ("online_dp_l3_test", 32, 7),
    ("online_dp_l3_accuracy_test", 64, 7),
)


@pytest.fixture
def fresh_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/dsa_demo"))
    # 配置在 import 时读取；隔离模块缓存，避免前一条用例的配置影响本次验证。
    saved = {name: sys.modules.pop(name) for name, _, _ in CLIENTS if name in sys.modules}
    for key in (
        "DSA_DP_MAX_TOKENS",
        "DSA_DP_API_BASE",
        "DSA_DP_API_KEY",
        "DSA_DP_MODEL_NAME",
        "DSA_DP_REQUEST_TIMEOUT_SECONDS",
        "DSA_DP_L3_LABELS",
        "DSA_DP_L3_STORE_WAIT_SECONDS",
        "DSA_DP_ACCURACY_LABELS",
        "DSA_DP_ACCURACY_RANKS",
    ):
        monkeypatch.delenv(key, raising=False)
    try:
        yield
    finally:
        for name, _, _ in CLIENTS:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


@pytest.mark.parametrize("name,default_tokens,request_count", CLIENTS)
@pytest.mark.parametrize("override", [None, "128"])
def test_environment_reaches_every_dp_request(
    fresh_imports, monkeypatch, tmp_path, capsys, name, default_tokens, request_count, override
):
    expected_tokens = default_tokens if override is None else int(override)
    if override is not None:
        monkeypatch.setenv("DSA_DP_MAX_TOKENS", override)
    monkeypatch.setenv("DSA_DP_API_BASE", "http://localhost:8035/v1")
    monkeypatch.setenv("DSA_DP_API_KEY", "test-key")
    monkeypatch.setenv("DSA_DP_MODEL_NAME", "test-model")
    monkeypatch.setenv("DSA_DP_REQUEST_TIMEOUT_SECONDS", "123")
    monkeypatch.setenv("DSA_DP_L3_STORE_WAIT_SECONDS", "0")
    module = importlib.import_module(name)
    assert expected_tokens == module.MAX_TOKENS
    requests = []
    constructors = []
    local = set()
    remote = set()
    shared_l3 = "l3" in name

    class Client:
        def __init__(self, **kwargs):
            constructors.append(kwargs)
            self.completions = self
            self.models = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def close(self):
            pass

        async def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="test-model")])

        async def create(self, **kwargs):
            requests.append(kwargs)
            assert kwargs["model"] == "test-model"
            assert kwargs["max_tokens"] == expected_tokens
            assert kwargs["extra_body"]["ignore_eos"] is True
            rank = kwargs.get("extra_headers", {}).get("X-data-parallel-rank", "0")
            salt = kwargs["extra_body"].get("cache_salt")
            cached = 256 if (rank, salt) in local or (shared_l3 and salt in remote) else 0
            local.add((rank, salt))
            remote.add(salt)
            payload = {
                "id": "response",
                "choices": [
                    {
                        "finish_reason": "length",
                        "text": "ok",
                        "token_ids": [100] * expected_tokens,
                        "logprobs": {"top_logprobs": [{"token_id:100": -0.1}] * expected_tokens},
                    }
                ],
                "usage": {
                    "prompt_tokens": 300,
                    "completion_tokens": expected_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
            return SimpleNamespace(
                model_dump=lambda: payload,
                choices=[SimpleNamespace(**payload["choices"][0])],
                usage=SimpleNamespace(**payload["usage"]),
            )

        async def post(self, url, params):
            assert url == "http://localhost:8035/reset_prefix_cache"
            assert params == {"reset_external": "false"}
            local.clear()
            return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    if hasattr(module, "httpx"):
        monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(
        module,
        "_load_cases",
        lambda: [SimpleNamespace(label=label, prompt=label) for label in ("5k", "8k", "40k", "70k")],
    )
    if hasattr(module, "RESULT_DIR"):
        monkeypatch.setattr(module, "RESULT_DIR", tmp_path)
    if name == "online_dp_l3_accuracy_test":
        asyncio.run(module._collect())
    else:
        asyncio.run(module._main())
    assert len(requests) == request_count
    assert constructors[0] == {
        "api_key": "test-key",
        "base_url": "http://localhost:8035/v1",
        "timeout": 123.0,
        "max_retries": 0,
    }
    assert f"max_tokens={expected_tokens}" in capsys.readouterr().out


@pytest.mark.parametrize("name,_,__", CLIENTS)
@pytest.mark.parametrize(
    "key,value",
    [
        ("DSA_DP_MAX_TOKENS", "1"),
        ("DSA_DP_MAX_TOKENS", "0"),
        ("DSA_DP_REQUEST_TIMEOUT_SECONDS", "nan"),
        ("DSA_DP_REQUEST_TIMEOUT_SECONDS", "0"),
        ("DSA_DP_API_BASE", "http://localhost:8035"),
    ],
)
def test_invalid_settings_fail_before_dataset_or_network(fresh_imports, monkeypatch, name, _, __, key, value):
    monkeypatch.setenv(key, value)
    module = importlib.import_module(name)

    def unexpected(*args, **kwargs):
        pytest.fail("invalid settings must be rejected before loading prompts or opening a client")

    monkeypatch.setattr(module, "_load_cases", unexpected)
    monkeypatch.setattr(module, "AsyncOpenAI", unexpected)
    with pytest.raises(ValueError, match=key):
        asyncio.run(module._collect() if name == "online_dp_l3_accuracy_test" else module._main())
