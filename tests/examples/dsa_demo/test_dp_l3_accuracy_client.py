# SPDX-License-Identifier: Apache-2.0
"""DP精度示例的比较合同与流程自测；不依赖框架UT或NPU。"""

import argparse
import asyncio
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "examples/dsa_demo"))
    return importlib.import_module("online_dp_l3_accuracy_test")


def _record(phase="local-0"):
    return {
        "run_id": "run",
        "dp_rank": 0,
        "label": "40k",
        "model": "model",
        "salt": "salt",
        "prompt_sha256": "digest",
        "max_tokens": 3,
        "seed": 1024,
        "top_logprobs": 5,
        "phase": phase,
        "request_id": phase,
        "response": {
            "choices": [
                {
                    "finish_reason": "length",
                    "text": "abc",
                    "token_ids": [10, 11, 12],
                    "logprobs": {
                        "top_logprobs": [
                            {"token_id:10": -0.1, "token_id:20": -3.0},
                            {"token_id:11": -0.2, "token_id:21": -3.0},
                            {"token_id:12": -0.3, "token_id:22": -3.0},
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 300, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 256}},
        },
    }


def test_matched_tokens_report_logprob_drift_separately(module):
    left, right = _record(), _record("external-0")
    right["response"]["choices"][0]["logprobs"]["top_logprobs"][0]["token_id:10"] = -0.15
    result = module._compare("compare", left, right)
    assert result["status"] == "matched"
    assert result["first_logprob_drift"] == 0
    assert result["max_shared_topk_logprob_delta"] == pytest.approx(0.05)


def test_mismatch_only_compares_same_conditioning_history(module):
    left, right = _record(), _record("external-0")
    right["response"]["choices"][0]["token_ids"][1] = 21
    rows = right["response"]["choices"][0]["logprobs"]["top_logprobs"]
    rows[1] = {"token_id:11": -2.0, "token_id:21": -0.1}
    rows[2] = {"token_id:12": -100.0}  # Different history; must not inflate the reported drift.
    result = module._compare("compare", left, right)
    assert result["status"] == "mismatch"
    assert result["position"] == 1
    assert result["left_token"] == 11 and result["right_token"] == 21
    assert result["compared_logprob_rows"] == 2
    assert result["max_shared_topk_logprob_delta"] == pytest.approx(2.9)


@pytest.mark.parametrize("field,value", [("dp_rank", 1), ("salt", "different"), ("prompt_sha256", "different")])
def test_changed_identity_is_not_called_precision_failure(module, field, value):
    left, right = _record(), _record()
    right[field] = value
    assert module._compare("compare", left, right)["status"] == "incomparable"


def test_different_cached_lengths_are_incomparable(module):
    left, right = _record(), _record()
    right["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 128
    assert module._compare("compare", left, right)["status"] == "incomparable"


@pytest.mark.parametrize("broken", ["ids", "rows", "text_keys", "nan", "chosen"])
def test_invalid_trace_cannot_silently_pass(module, broken):
    record = _record()
    choice = record["response"]["choices"][0]
    if broken == "ids":
        choice.pop("token_ids")
    elif broken == "rows":
        choice["logprobs"]["top_logprobs"] = []
    elif broken == "text_keys":
        choice["logprobs"]["top_logprobs"][0] = {"a": -0.1}
    elif broken == "nan":
        choice["logprobs"]["top_logprobs"][0]["token_id:10"] = float("nan")
    else:
        choice["logprobs"]["top_logprobs"][0] = {"token_id:20": -0.1}
    with pytest.raises(ValueError):
        module._trace(record)


def test_incomplete_run_is_saved_as_incomplete(module, tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    assert not module._analyze(path)
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    assert summary[0]["comparisons"][0]["status"] == "incomplete"


@pytest.mark.parametrize("failure", ["mismatch", "invalid-evidence", "incomparable"])
def test_analysis_records_failures_without_losing_other_comparisons(module, tmp_path, failure):
    phases = ["seed", "local-0", "local-1", "external-0", "post-local-0", "external-1", "post-local-1"]
    records = [_record(phase) for phase in phases]
    candidate = records[3]["response"]
    if failure == "mismatch":
        candidate["choices"][0]["token_ids"][1] = 21
    elif failure == "invalid-evidence":
        candidate["choices"][0].pop("logprobs")
    else:
        candidate["usage"]["prompt_tokens_details"]["cached_tokens"] = 128
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    assert not module._analyze(path)
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    comparisons = {item["name"]: item for item in summary[0]["comparisons"]}
    assert len(comparisons) == len(module.COMPARISONS)
    assert comparisons["local-repeat"]["status"] == "matched"
    assert comparisons["local-vs-external-0"]["status"] == failure


def test_collection_controls_route_reset_and_compares_after_clients_close(module, monkeypatch, tmp_path):
    local, remote, requests, resets, closed = set(), set(), [], [], []

    class Client:
        def __init__(self, **kwargs):
            self.completions = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

        async def create(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            rank = kwargs["extra_headers"]["X-data-parallel-rank"]
            salt = kwargs["extra_body"]["cache_salt"]
            cached = 256 if (rank, salt) in local or salt in remote else 0
            local.add((rank, salt))
            remote.add(salt)
            payload = _record()["response"]
            if len(requests) == 1:
                # Cold calculation deliberately differs; it is evidence, not the warm reference.
                payload["choices"][0]["token_ids"][0] = 20
            payload["usage"]["prompt_tokens_details"]["cached_tokens"] = cached
            return SimpleNamespace(model_dump=lambda: payload)

        async def post(self, url, params):
            resets.append((url, params))
            local.clear()
            return SimpleNamespace(raise_for_status=lambda: None)

    async def resolve(client):
        return "model"

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "_resolve_model", resolve)
    monkeypatch.setattr(module, "_load_cases", lambda: [SimpleNamespace(label="40k", prompt="unchanged")])
    monkeypatch.setattr(module, "LABELS", ("40k",))
    monkeypatch.setattr(module, "DP_RANKS", (0,))
    monkeypatch.setattr(module, "MAX_TOKENS", 3)
    monkeypatch.setattr(module, "STORE_WAIT_SECONDS", 0)
    monkeypatch.setattr(module, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(module, "API_BASE", "http://127.0.0.1:8035/v1")
    path = asyncio.run(module._collect())
    assert len(closed) == 2
    assert module._analyze(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["phase"] for record in records] == [
        "seed",
        "local-0",
        "local-1",
        "external-0",
        "post-local-0",
        "external-1",
        "post-local-1",
    ]
    assert resets == [("http://127.0.0.1:8035/reset_prefix_cache", {"reset_external": "false"})] * 2
    assert len(remote) == 1
    assert all(request["extra_headers"]["X-data-parallel-rank"] == "0" for request in requests)
    assert all(request["temperature"] == 0 and request["extra_body"]["ignore_eos"] for request in requests)
    assert all(request["extra_body"]["seed"] == 1024 and request["logprobs"] == 5 for request in requests)


def _concurrent_records(module):
    records = []
    members = [{"dp_rank": rank, "label": "40k"} for rank in (0, 1)]
    for index, phase in enumerate(module.PHASES):
        for rank in (0, 1):
            record = _record(phase)
            record.update(
                dp_rank=rank,
                salt=f"salt-dp{rank}",
                request_id=f"dp{rank}-{phase}",
                execution_mode="concurrent",
                scenario="pair0-40k-40k",
                batch_id=f"run-pair0-{phase}",
                batch_members=members,
                batch_signature="pair-sig",
                started_offset_ms=index * 10 + rank,
                finished_offset_ms=index * 10 + 5 + rank,
            )
            if phase == "seed":
                record["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
            records.append(record)
    return records


def test_concurrent_analysis_reports_each_dp_without_cross_rank_comparison(module, tmp_path):
    records = _concurrent_records(module)
    for record in records:
        if record["dp_rank"] == 1:
            record["response"]["choices"][0]["token_ids"][0] = 20
    path = tmp_path / "concurrent.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    assert module._analyze(path)
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    assert {item["dp_rank"] for item in summary} == {0, 1}
    assert all(len(item["comparisons"]) == 7 for item in summary)
    for report in summary[0]["concurrent_rounds"]:
        assert report["status"] == "overlapped"
        assert report["client_overlap_ms"] == 4
        assert report["server_batch_alignment_verified"] is False


@pytest.mark.parametrize("problem", ["missing-rank", "no-overlap", "wrong-members", "wrong-batch", "invalid-time"])
def test_concurrent_coverage_cannot_pass_on_tokens_alone(module, tmp_path, problem):
    records = _concurrent_records(module)
    if problem == "missing-rank":
        records = [record for record in records if record["dp_rank"] == 0]
    elif problem == "no-overlap":
        records[1].update(started_offset_ms=6, finished_offset_ms=8)
    elif problem == "wrong-members":
        records[1]["batch_members"] = [{"dp_rank": 1, "label": "40k"}]
    elif problem == "wrong-batch":
        records[1]["batch_id"] = "other-wave"
    else:
        records[1]["finished_offset_ms"] = -1
    path = tmp_path / "concurrent.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    assert not module._analyze(path)
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    assert summary[0]["concurrent_rounds"][0]["status"] != "overlapped"


@pytest.mark.parametrize("field", ["execution_mode", "scenario", "batch_signature"])
def test_different_concurrent_conditions_are_incomparable(module, field):
    left, right = _record(), _record()
    left[field], right[field] = "first", "second"
    assert module._compare("compare", left, right)["status"] == "incomparable"


@pytest.mark.parametrize("value", ["40k", "0,1", "40k,40k,40k", "40k,"])
def test_invalid_concurrent_pair_is_rejected(module, value):
    with pytest.raises(argparse.ArgumentTypeError):
        module._parse_pair(value)


def _install_concurrent_client(module, monkeypatch, tmp_path, *, fail_phase=None):
    state = SimpleNamespace(
        active=0,
        max_active=0,
        requests=[],
        resets=[],
        closed=0,
        local=set(),
        remote=set(),
        gates={},
        entered={},
        peer_after_failure=False,
    )

    class Client:
        def __init__(self, **kwargs):
            self.completions = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            assert state.active == 0
            state.closed += 1

        async def create(self, **kwargs):
            wave = len(state.requests) // 2
            phase = module.PHASES[wave % len(module.PHASES)]
            state.requests.append(copy.deepcopy(kwargs))
            rank = kwargs["extra_headers"]["X-data-parallel-rank"]
            salt = kwargs["extra_body"]["cache_salt"]
            state.active += 1
            state.max_active = max(state.max_active, state.active)
            gate = state.gates.setdefault(wave, asyncio.Event())
            state.entered[wave] = state.entered.get(wave, 0) + 1
            if state.entered[wave] == 2:
                gate.set()
            try:
                # A serialized collector cannot pass this gate.
                await asyncio.wait_for(gate.wait(), timeout=1)
                if phase == fail_phase and rank == "0":
                    raise RuntimeError("DP0 request failed")
                await asyncio.sleep(0.005)
                if phase == fail_phase and rank == "1":
                    state.peer_after_failure = True
                cached = 256 if (rank, salt) in state.local or salt in state.remote else 0
                state.local.add((rank, salt))
                state.remote.add(salt)
                payload = _record()["response"]
                payload["usage"]["prompt_tokens_details"]["cached_tokens"] = cached
                return SimpleNamespace(model_dump=lambda: payload)
            finally:
                state.active -= 1

        async def post(self, url, params):
            assert state.active == 0
            state.resets.append((url, params))
            state.local.clear()
            return SimpleNamespace(raise_for_status=lambda: None)

    async def resolve(client):
        return "model"

    monkeypatch.setattr(module, "AsyncOpenAI", Client)
    monkeypatch.setattr(module.httpx, "AsyncClient", Client)
    monkeypatch.setattr(module, "_resolve_model", resolve)
    monkeypatch.setattr(
        module,
        "_load_cases",
        lambda: [SimpleNamespace(label=label, prompt=f"prompt-{label}") for label in ("40k", "8k", "70k")],
    )
    monkeypatch.setattr(module, "MAX_TOKENS", 3)
    monkeypatch.setattr(module, "STORE_WAIT_SECONDS", 0)
    monkeypatch.setattr(module, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(module, "API_BASE", "http://127.0.0.1:8035/v1")
    return state


@pytest.mark.parametrize("pairs", [[("40k", "40k")], [("40k", "40k"), ("8k", "70k")]])
def test_concurrent_collection_really_overlaps_and_keeps_phase_barriers(module, monkeypatch, tmp_path, pairs):
    state = _install_concurrent_client(module, monkeypatch, tmp_path)
    path = asyncio.run(module._collect(pairs))
    assert state.max_active == 2
    assert state.closed == 2
    assert len(state.requests) == 14 * len(pairs)
    assert len(state.resets) == 2 * len(pairs)
    assert len(state.remote) == 2 * len(pairs)
    assert all(params == {"reset_external": "false"} for _, params in state.resets)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for index, pair in enumerate(pairs):
        scenario = f"pair{index}-{pair[0]}-{pair[1]}"
        subset = [record for record in records if record["scenario"] == scenario]
        assert len(subset) == 14
        assert {(record["dp_rank"], record["label"]) for record in subset} == set(enumerate(pair))
        assert len({record["batch_signature"] for record in subset}) == 1
        for record in subset:
            request = next(
                item for item in state.requests if item["extra_headers"]["X-Request-Id"] == record["request_id"]
            )
            assert request["extra_headers"]["X-data-parallel-rank"] == str(record["dp_rank"])
            assert request["prompt"] == f"prompt-{record['label']}"
            assert request["extra_body"]["ignore_eos"]
    assert module._analyze(path)


def test_concurrent_failure_still_persists_successful_peer_before_closing(module, monkeypatch, tmp_path):
    state = _install_concurrent_client(module, monkeypatch, tmp_path, fail_phase="external-0")
    with pytest.raises(RuntimeError, match="concurrent round failed"):
        asyncio.run(module._collect([("40k", "40k")]))
    assert state.peer_after_failure
    assert state.closed == 2
    assert len(state.resets) == 1
    path = next(tmp_path.glob("*.jsonl"))
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 8
    external = {record["dp_rank"]: record for record in records if record["phase"] == "external-0"}
    assert "DP0 request failed" in external[0]["request_error"]
    assert external[1]["response"]["choices"][0]["token_ids"] == [10, 11, 12]
    assert not module._analyze(path)
