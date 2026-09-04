# SPDX-License-Identifier: Apache-2.0
"""共享 L3 在线示例的流程测试，独立于框架 UT。"""

import argparse
import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mtp,wrong_boundary", [(False, False), (True, False), (True, True)])
def test_script_preserves_l3_while_resetting_all_local_caches(monkeypatch, tmp_path, mtp, wrong_boundary):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/dsa_demo"))
    module = importlib.import_module("online_dp_l3_test")
    local = set()
    remote = set()
    reset_calls = []

    class Client:
        def __init__(self, **kwargs):
            self.completions = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def create(self, **kwargs):
            rank = kwargs["extra_headers"]["X-data-parallel-rank"]
            salt = kwargs["extra_body"]["cache_salt"]
            hit_tokens = 128 if mtp and not wrong_boundary else 256
            cached = hit_tokens if (rank, salt) in local or salt in remote else 0
            local.add((rank, salt))
            remote.add(salt)
            payload = {
                "id": "response",
                "choices": [{"finish_reason": "length", "text": "ok"}],
                "usage": {
                    "prompt_tokens": 300,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
            return SimpleNamespace(model_dump=lambda: payload)

        async def post(self, url, params):
            assert params == {"reset_external": "false"}
            reset_calls.append(url)
            local.clear()
            return SimpleNamespace(raise_for_status=lambda: None)

    async def resolve(client):
        return "model"

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "_resolve_model", resolve)
    monkeypatch.setattr(module, "_load_cases", lambda: [SimpleNamespace(label="40k", prompt="unchanged")])
    monkeypatch.setattr(module, "LABELS", ("40k",))
    monkeypatch.setattr(module, "STORE_WAIT_SECONDS", 0)
    monkeypatch.setattr(module, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(module, "API_BASE", "http://127.0.0.1:8035/v1")
    monkeypatch.setattr(module, "MAX_TOKENS", 2)
    if wrong_boundary:
        with pytest.raises(AssertionError, match="expected MTP cached_tokens=128"):
            asyncio.run(module._main(mtp=mtp))
        return
    asyncio.run(module._main(mtp=mtp))
    records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert len(records) == 7
    assert len(remote) == 2
    assert reset_calls == ["http://127.0.0.1:8035/reset_prefix_cache"] * 2
    assert records[1]["phase"] == "cross-load" and records[1]["dp_rank"] == 1
    assert records[-1]["phase"] == "concurrent-reset-load"


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("pairs", [[("8k", "70k")], [("40k", "40k")], [("8k", "70k"), ("70k", "8k")]])
def test_concurrent_pairs_swap_sources_and_force_external_reads(monkeypatch, tmp_path, mtp, pairs):
    _run_pair_test(monkeypatch, tmp_path, mtp=mtp, pairs=pairs)


def test_failed_pair_waits_for_peer_before_closing_and_never_resets(monkeypatch, tmp_path):
    _run_pair_test(monkeypatch, tmp_path, mtp=True, pairs=[("8k", "70k")], fail_seed=True)


def _run_pair_test(monkeypatch, tmp_path, *, mtp, pairs, fail_seed=False):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/dsa_demo"))
    module = importlib.import_module("online_dp_l3_test")
    local, remote = set(), set()
    sources = {}
    reset_calls = []
    active = max_active = 0
    prompt_lengths = {"8k": 7752, "40k": 41672, "70k": 73496}
    hit_lengths = (
        {"8k": 7552, "40k": 41472, "70k": 73344}
        if mtp
        else {
            "8k": 7680,
            "40k": 41600,
            "70k": 73472,
        }
    )

    class Client:
        def __init__(self, **kwargs):
            self.completions = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            assert active == 0
            return False

        async def create(self, **kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                rank = int(kwargs["extra_headers"]["X-data-parallel-rank"])
                request_id = kwargs["extra_headers"]["X-Request-Id"]
                await asyncio.sleep(0)
                if fail_seed and rank == 0:
                    raise RuntimeError("injected seed failure")
                await asyncio.sleep(0)
                salt = kwargs["extra_body"]["cache_salt"]
                label = kwargs["prompt"]
                source = "local" if (rank, salt) in local else "remote" if salt in remote else "cold"
                sources[request_id] = source
                local.add((rank, salt))
                remote.add(salt)
                assert kwargs["max_tokens"] == 128
                assert kwargs["extra_body"]["ignore_eos"] is True
                payload = {
                    "id": "response",
                    "choices": [{"finish_reason": "length", "text": "ok"}],
                    "usage": {
                        "prompt_tokens": prompt_lengths[label],
                        "completion_tokens": 128,
                        "prompt_tokens_details": {"cached_tokens": 0 if source == "cold" else hit_lengths[label]},
                    },
                }
                return SimpleNamespace(model_dump=lambda: payload)
            finally:
                active -= 1

        async def post(self, url, params):
            assert active == 0
            assert params == {"reset_external": "false"}
            reset_calls.append(url)
            local.clear()
            return SimpleNamespace(raise_for_status=lambda: None)

    async def resolve(client):
        return "model"

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "_resolve_model", resolve)
    monkeypatch.setattr(
        module, "_load_cases", lambda: [SimpleNamespace(label=label, prompt=label) for label in prompt_lengths]
    )
    monkeypatch.setattr(module, "LABELS", ("invalid-unused-serial-label",))
    monkeypatch.setattr(module, "STORE_WAIT_SECONDS", 0)
    monkeypatch.setattr(module, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(module, "API_BASE", "http://127.0.0.1:8035/v1")
    monkeypatch.setattr(module, "MAX_TOKENS", 128)
    if fail_seed:
        with pytest.raises(AssertionError, match="injected seed failure"):
            asyncio.run(module._main(mtp=mtp, concurrent_pairs=pairs))
        assert not reset_calls
    else:
        asyncio.run(module._main(mtp=mtp, concurrent_pairs=pairs))
        assert len(reset_calls) == len(pairs)
    assert max_active == 2
    records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    if fail_seed:
        assert len(records) == 1 and records[0]["dp_rank"] == 1
        return
    assert len(records) == len(pairs) * 8
    assert len({record["request_id"] for record in records}) == len(records)
    assert len({record["cache_salt"] for record in records}) == len(pairs) * 2
    for index, pair in enumerate(pairs):
        seeds = [record for record in records if record["phase"] == f"pair{index}-seed"]
        owners = {record["cache_salt"]: record["dp_rank"] for record in seeds}
        for phase, source, swapped in (
            ("seed", "cold", False),
            ("cross-load", "remote", True),
            ("local-replay", "local", True),
            ("reset-load", "remote", False),
        ):
            phase_records = [record for record in records if record["phase"] == f"pair{index}-{phase}"]
            assert len(phase_records) == 2
            for record in phase_records:
                owner = owners[record["cache_salt"]]
                assert record["label"] == pair[owner]
                assert record["dp_rank"] == (1 - owner if swapped else owner)
                assert sources[record["request_id"]] == source


@pytest.mark.parametrize("text", ["8k", "8k,", "8k,70k,40k", "8k,100k"])
def test_pair_parser_rejects_invalid_labels(monkeypatch, text):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/dsa_demo"))
    module = importlib.import_module("online_dp_l3_test")
    with pytest.raises(argparse.ArgumentTypeError, match="DP0_LABEL,DP1_LABEL"):
        module._parse_pair(text)
