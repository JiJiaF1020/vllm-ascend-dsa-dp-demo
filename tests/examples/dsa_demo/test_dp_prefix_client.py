# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""在线示例脚本自身的协议与结果断言测试，不属于 DSA 框架 UT。"""

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def client_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples" / "dsa_demo"))
    return importlib.import_module("online_dp_prefix_test")


@pytest.mark.parametrize(
    ("cached", "expect_hit", "error"),
    [
        (None, False, None),
        (256, True, None),
        (384, True, None),
        (0, None, None),
        (384, None, None),
        (0, True, "expected this DP"),
        (256, False, "expected a cold DP"),
        (255, True, "invalid DSA"),
        (640, True, "invalid DSA"),
    ],
)
def test_prefix_client_checks_cached_usage(client_module, cached, expect_hit, error):
    response = {
        "choices": [{"finish_reason": "length"}],
        "usage": {"prompt_tokens": 513, "completion_tokens": 32},
    }
    if cached is not None:
        response["usage"]["prompt_tokens_details"] = {"cached_tokens": cached}
    if error:
        with pytest.raises(AssertionError, match=error):
            client_module._validate_usage(response, expect_hit=expect_hit, max_tokens=32)
    else:
        assert client_module._validate_usage(response, expect_hit=expect_hit, max_tokens=32) == (513, 32, cached or 0)


@pytest.mark.parametrize("ignore_dp_header", [False, True])
def test_prefix_client_matrix_detects_cross_dp_cache_leak(client_module, monkeypatch, tmp_path, ignore_dp_header):
    module = client_module
    calls = []
    cached_keys = set()

    class Client:
        def __init__(self, **kwargs):
            self.completions = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def create(self, **kwargs):
            calls.append(kwargs)
            rank = int(kwargs["extra_headers"]["X-data-parallel-rank"])
            salt = kwargs["extra_body"]["cache_salt"]
            key = (0 if ignore_dp_header else rank, salt)
            cached = 512 if key in cached_keys else 0
            cached_keys.add(key)
            payload = {
                "id": "reply",
                "choices": [{"finish_reason": "length", "token_ids": [1, 2]}],
                "usage": {
                    "prompt_tokens": 513,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
            return SimpleNamespace(model_dump=lambda: payload)

    async def model(client):
        return "test-model"

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    monkeypatch.setattr(module, "_resolve_model", model)
    monkeypatch.setattr(
        module,
        "_load_cases",
        lambda: [module.PromptCase(label, f"original-{label}") for label in ("5k", "8k", "40k", "70k")],
    )
    monkeypatch.setattr(module, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(module, "MAX_TOKENS", 2)
    if ignore_dp_header:
        with pytest.raises(AssertionError, match="isolated-cold/DP1/40k"):
            asyncio.run(module._main())
        assert len(calls) == 3
    else:
        asyncio.run(module._main())
        assert len(calls) == 16
        assert {call["extra_headers"]["X-data-parallel-rank"] for call in calls} == {"0", "1"}
    records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(calls)
    assert all(call["prompt"].startswith("original-") for call in calls)
