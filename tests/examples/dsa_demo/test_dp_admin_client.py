# SPDX-License-Identifier: Apache-2.0
"""管理客户端自测：两个旧salt均失效，新epoch跨域共享，暂停请求到resume后才完成。"""

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.parametrize(
    "broken,paused_reset,leave_paused",
    [(None, False, False), ("partial-epoch", False, False), ("pause-noop", False, False),
     (None, True, False), ("reset-resumes", True, False), (None, True, True)],
)
def test_admin_client_checks_all_domains_and_pause(monkeypatch, tmp_path, broken, paused_reset, leave_paused):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/dsa_demo"))
    module = importlib.import_module("online_dp_cache_admin_test")
    epoch = [0, 0]
    local, remote, calls = set(), set(), []
    paused = False
    resumed = asyncio.Event()
    resumed.set()
    original_sleep = asyncio.sleep

    async def short_sleep(_):
        await original_sleep(0)

    class Client:
        def __init__(self, **kwargs):
            self.completions = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def create(self, **kwargs):
            await resumed.wait()
            rank = int(kwargs["extra_headers"]["X-data-parallel-rank"])
            key = (epoch[rank], kwargs["extra_body"]["cache_salt"])
            hit = (rank, key) in local or key in remote
            local.add((rank, key))
            remote.add(key)
            payload = {
                "id": "response",
                "choices": [{"finish_reason": "length", "text": "ok"}],
                "usage": {
                    "prompt_tokens": 41672,
                    "completion_tokens": 128,
                    "prompt_tokens_details": {"cached_tokens": 41472 if hit else 0},
                },
            }
            return SimpleNamespace(model_dump=lambda: payload)

        async def post(self, url, params=None):
            nonlocal paused
            calls.append((url, params))
            if url.endswith("/reset_prefix_cache"):
                local.clear()
                if paused and broken == "reset-resumes":
                    paused = False
                    resumed.set()
                if params["reset_external"] == "true":
                    epoch[0] += 1
                    if broken != "partial-epoch":
                        epoch[1] += 1
            elif url.endswith("/pause"):
                assert params == {"mode": "wait", "clear_cache": "false"}
                paused = True
                if broken != "pause-noop":
                    resumed.clear()
            elif url.endswith("/resume"):
                paused = False
                resumed.set()
            else:
                raise AssertionError(url)
            return SimpleNamespace(raise_for_status=lambda: None)

        async def get(self, url):
            assert url.endswith("/is_paused")
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"is_paused": paused})

    async def model(_):
        return "model"

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "_resolve_model", model)
    monkeypatch.setattr(module, "_load_cases", lambda: [SimpleNamespace(label="40k", prompt="original")])
    monkeypatch.setattr(module, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(module, "MAX_TOKENS", 128)
    monkeypatch.setattr(module.asyncio, "sleep", short_sleep)
    if broken:
        message = {
            "partial-epoch": "expected a cold DP",
            "pause-noop": "request completed",
            "reset-resumes": "pause state|expected a cold DP",
        }[broken]
        with pytest.raises(AssertionError, match=message):
            asyncio.run(module._main(paused_reset=paused_reset, leave_paused=leave_paused))
    else:
        asyncio.run(module._main(paused_reset=paused_reset, leave_paused=leave_paused))
        records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
        assert len(records) == (9 if paused_reset else 8)
        assert len(remote) == (5 if paused_reset else 4)
        assert epoch == ([2, 2] if paused_reset else [1, 1])
        expected_resets = ["false", "true", "false"] + (["false", "true", "false"] if paused_reset else [])
        actual_resets = [params["reset_external"] for url, params in calls if url.endswith("/reset_prefix_cache")]
        assert actual_resets == expected_resets
    assert paused is leave_paused


def test_admin_script_defaults_and_optional_overrides(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/dsa_demo"))
    module = importlib.import_module("online_dp_cache_admin_test")
    args = module._parse_args([])
    assert args.paused_reset is True and args.leave_paused is False
    assert module.API_BASE == "http://127.0.0.1:8035/v1" and module.MAX_TOKENS == 128
    monkeypatch.setattr(module, "PAUSED_RESET", False)
    monkeypatch.setattr(module, "LEAVE_PAUSED", True)
    edited = module._parse_args([])
    assert edited.paused_reset is False and edited.leave_paused is True
    overridden = module._parse_args(["--paused-reset", "--no-leave-paused"])
    assert overridden.paused_reset is True and overridden.leave_paused is False


def test_admin_model_selection_uses_this_scripts_config(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/dsa_demo"))
    module = importlib.import_module("online_dp_cache_admin_test")
    monkeypatch.setattr(module, "MODEL_NAME", "edited-model")
    client = SimpleNamespace(models=SimpleNamespace(list=AsyncMock(return_value=SimpleNamespace(
        data=[SimpleNamespace(id="edited-model"), SimpleNamespace(id="other")],
    ))))
    assert asyncio.run(module._resolve_model(client)) == "edited-model"
