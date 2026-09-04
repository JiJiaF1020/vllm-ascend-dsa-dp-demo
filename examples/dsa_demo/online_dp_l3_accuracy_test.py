#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DP共享L3的受控精度门禁，不把冷算与热复用混在一起比较。

默认只测DP0/40k，所有请求串行、固定采样参数，另一DP无真实请求。
--concurrent-pair 40k,40k则每轮同时向DP0/DP1各发一个真实请求；两域
各用独立salt，先隔离同key写竞争。客户端记录在途重叠，不伪称已证明
服务端逐轮batch完全对齐；同一域只与相同并发请求组合下的本域结果比较。
先冷算留档，再比较两次本地热重放、两次清本地后的L3读回和各自的
本地再重放。相同token只是输出一致性，不等同于KV逐字节一致或完整
模型精度通过；logprobs漂移独立报告，不用任意放宽阈值掩盖token差异。

完整响应先落盘，比较在关闭HTTP客户端后进行。--analyze可离线重读
JSONL，无需启动服务。测试期间不得有其他流量；reset只清本地缓存。
当前首轮要求no-MTP、Mooncake TCP、kv_both和Local prefix开启。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path

import httpx
from online_dp_foundation_test import (
    API_BASE,
    API_KEY,
    REQUEST_TIMEOUT_SECONDS,
    _load_cases,
    _resolve_model,
    _validate_client_settings,
)
from online_dp_prefix_test import _validate_usage
from openai import AsyncOpenAI

LABELS = tuple(item.strip() for item in os.getenv("DSA_DP_ACCURACY_LABELS", "40k").split(","))
DP_RANKS = tuple(int(item) for item in os.getenv("DSA_DP_ACCURACY_RANKS", "0").split(","))
# 同一个客户端环境变量；精度入口有意使用比功能入口更长的默认输出。
MAX_TOKENS = int(os.getenv("DSA_DP_MAX_TOKENS", "64"))
STORE_WAIT_SECONDS = float(os.getenv("DSA_DP_L3_STORE_WAIT_SECONDS", "10"))
RESULT_DIR = Path(os.getenv("DSA_DP_ACCURACY_RESULT_DIR", str(Path(__file__).resolve().parent / "results/dp_accuracy")))
TOP_LOGPROBS = 5
SEED = 1024
PHASES = ("seed", "local-0", "local-1", "external-0", "post-local-0", "external-1", "post-local-1")
COMPARISONS = (
    ("local-repeat", "local-0", "local-1"),
    ("external-repeat", "external-0", "external-1"),
    ("local-vs-external-0", "local-1", "external-0"),
    ("local-vs-external-1", "local-1", "external-1"),
    ("external-to-local-0", "external-0", "post-local-0"),
    ("external-to-local-1", "external-1", "post-local-1"),
    ("post-local-repeat", "post-local-0", "post-local-1"),
)


def _trace(record: dict) -> tuple[list[int], list[dict[int, float]]]:
    choices = record["response"].get("choices") or []
    if len(choices) != 1:
        raise ValueError("expected exactly one completion")
    choice = choices[0]
    ids = choice.get("token_ids")
    if not isinstance(ids, list) or len(ids) != record["max_tokens"]:
        raise ValueError("missing/incomplete token_ids; require return_token_ids=True")
    if any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("invalid token_ids")
    rows = (choice.get("logprobs") or {}).get("top_logprobs")
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError("missing/incomplete top_logprobs")
    parsed = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not row:
            raise ValueError(f"missing top_logprobs at output position {index}")
        candidates = {}
        for key, value in row.items():
            if not key.startswith("token_id:"):
                raise ValueError("require return_tokens_as_token_ids=True for unambiguous logprobs")
            token = int(key.removeprefix("token_id:"))
            if token < 0 or not math.isfinite(float(value)):
                raise ValueError(f"invalid logprob at output position {index}")
            candidates[token] = float(value)
        if ids[index] not in candidates:
            raise ValueError(f"chosen token missing from top_logprobs at output position {index}")
        parsed.append(candidates)
    return ids, parsed


def _compare(name: str, left: dict, right: dict) -> dict:
    identity = ("dp_rank", "label", "model", "salt", "prompt_sha256", "max_tokens", "seed", "top_logprobs")
    for field in identity:
        if left[field] != right[field]:
            return {"name": name, "status": "incomparable", "reason": f"different {field}"}
    for field in ("execution_mode", "scenario", "batch_signature"):
        if left.get(field) != right.get(field):
            return {"name": name, "status": "incomparable", "reason": f"different {field}"}
    usages = [
        _validate_usage(item["response"], expect_hit=True, max_tokens=item["max_tokens"]) for item in (left, right)
    ]
    if usages[0] != usages[1]:
        return {
            "name": name,
            "status": "incomparable",
            "reason": "different prompt/completion/cached lengths",
            "left_usage": usages[0],
            "right_usage": usages[1],
        }
    left_ids, left_rows = _trace(left)
    right_ids, right_rows = _trace(right)
    common = next(
        (index for index, pair in enumerate(zip(left_ids, right_ids, strict=True)) if pair[0] != pair[1]), len(left_ids)
    )
    # Only compare distributions conditioned on the same generated history.
    compared_rows = min(common + 1, len(left_ids))
    max_delta = 0.0
    changed_sets = []
    first_drift = None
    for index in range(compared_rows):
        a, b = left_rows[index], right_rows[index]
        if a.keys() != b.keys():
            changed_sets.append(index)
        delta = max((abs(a[token] - b[token]) for token in a.keys() & b.keys()), default=0.0)
        max_delta = max(max_delta, delta)
        if first_drift is None and (a.keys() != b.keys() or delta > 1e-6):
            first_drift = index
    result = {
        "name": name,
        "status": "matched" if common == len(left_ids) else "mismatch",
        "left_request_id": left["request_id"],
        "right_request_id": right["request_id"],
        "common_prefix": common,
        "compared_logprob_rows": compared_rows,
        "max_shared_topk_logprob_delta": max_delta,
        "topk_set_changed_positions": changed_sets,
        "first_logprob_drift": first_drift,
    }
    if common < len(left_ids):
        result.update(
            position=common,
            left_token=left_ids[common],
            right_token=right_ids[common],
            left_topk=left_rows[common],
            right_topk=right_rows[common],
        )
    return result


def _concurrent_rounds(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    scenarios: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if record.get("execution_mode") == "concurrent":
            scenarios.setdefault((record["run_id"], record["scenario"]), []).append(record)
    reports = {}
    for key, items in scenarios.items():
        rounds = []
        for phase in PHASES:
            members = [item for item in items if item["phase"] == phase]
            report = {"phase": phase, "status": "invalid-coverage"}
            try:
                if len(members) != 2 or sorted(item["dp_rank"] for item in members) != [0, 1]:
                    raise ValueError("require exactly one response from each DP in every concurrent round")
                expected = sorted((item["dp_rank"], item["label"]) for item in members)
                for item in members:
                    if sorted((entry["dp_rank"], entry["label"]) for entry in item["batch_members"]) != expected:
                        raise ValueError("response membership differs from planned concurrent batch")
                    if item.get("request_error"):
                        raise ValueError(f"request failed: {item['request_error']}")
                    _validate_usage(item["response"], expect_hit=phase != "seed", max_tokens=item["max_tokens"])
                    _trace(item)
                if (members[0]["batch_id"], members[0]["batch_signature"]) != (
                    members[1]["batch_id"],
                    members[1]["batch_signature"],
                ):
                    raise ValueError("different batch identity within one concurrent round")
                starts = [float(item["started_offset_ms"]) for item in members]
                finishes = [float(item["finished_offset_ms"]) for item in members]
                if not all(math.isfinite(value) for value in (*starts, *finishes)):
                    raise ValueError("non-finite request timestamps")
                if any(start < 0 or end < start for start, end in zip(starts, finishes, strict=True)):
                    raise ValueError("invalid request time interval")
                overlap = min(finishes) - max(starts)
                report.update(
                    status="overlapped" if overlap > 0 else "no-client-overlap",
                    batch_id=members[0]["batch_id"],
                    request_ids=[item["request_id"] for item in members],
                    client_overlap_ms=round(max(0.0, overlap), 3),
                    start_skew_ms=round(max(starts) - min(starts), 3),
                    server_batch_alignment_verified=False,
                )
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                report["reason"] = str(exc)
            rounds.append(report)
        reports[key] = rounds
    return reports


def _analyze(result_path: Path) -> bool:
    groups: dict[tuple, dict[str, dict]] = {}
    records = []
    with result_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            records.append(record)
            key = (record["run_id"], record.get("scenario", "serial"), record["dp_rank"], record["label"])
            phases = groups.setdefault(key, {})
            if record["phase"] in phases:
                raise ValueError(f"duplicate phase in evidence: {key} {record['phase']}")
            phases[record["phase"]] = record
    if not groups:
        raise ValueError("empty evidence file")
    summary = []
    passed = True
    concurrent = _concurrent_rounds(records)
    for (run_id, scenario), rounds in concurrent.items():
        for report in rounds:
            passed = passed and report["status"] == "overlapped"
            print(
                f"[dsa-dp-accuracy] scenario={scenario} concurrency={json.dumps(report, ensure_ascii=False)}",
                flush=True,
            )
    expected = set(PHASES)
    for (run_id, scenario, rank, label), phases in sorted(groups.items()):
        missing = sorted(expected - phases.keys())
        if missing:
            comparisons = [{"name": "coverage", "status": "incomplete", "missing_phases": missing}]
        else:
            comparisons = []
            for name, a, b in COMPARISONS:
                try:
                    comparisons.append(_compare(name, phases[a], phases[b]))
                except (ValueError, KeyError, TypeError, AssertionError) as exc:
                    comparisons.append({"name": name, "status": "invalid-evidence", "reason": str(exc)})
        for comparison in comparisons:
            passed = passed and comparison["status"] == "matched"
            print(
                f"[dsa-dp-accuracy] scenario={scenario} dp={rank} label={label} "
                f"{json.dumps(comparison, ensure_ascii=False)}",
                flush=True,
            )
        summary.append(
            {
                "run_id": run_id,
                "scenario": scenario,
                "dp_rank": rank,
                "label": label,
                "comparisons": comparisons,
                "concurrent_rounds": concurrent.get((run_id, scenario), []),
            }
        )
    summary_path = result_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = "PASSED token replay consistency" if passed else "FAILED; inspect comparison categories"
    print(
        f"[dsa-dp-accuracy] {status}; not a KV bytewise/full model accuracy certificate; summary={summary_path}",
        flush=True,
    )
    return passed


async def _collect(concurrent_pairs: list[tuple[str, str]] | None = None) -> Path:
    _validate_client_settings(max_tokens=MAX_TOKENS)
    cases = {case.label: case for case in _load_cases()}
    if concurrent_pairs is None:
        if not DP_RANKS or len(set(DP_RANKS)) != len(DP_RANKS) or any(rank not in (0, 1) for rank in DP_RANKS):
            raise ValueError("DSA_DP_ACCURACY_RANKS must select distinct ranks from 0,1")
        if not LABELS or len(set(LABELS)) != len(LABELS) or any(label not in cases for label in LABELS):
            raise ValueError("DSA_DP_ACCURACY_LABELS must select distinct labels from 5k,8k,40k,70k")
        scenarios = [("serial", ((rank, label),)) for rank in DP_RANKS for label in LABELS]
        mode = "serial"
    else:
        if not concurrent_pairs or any(
            len(pair) != 2 or any(label not in cases for label in pair) for pair in concurrent_pairs
        ):
            raise ValueError("concurrent pairs must each contain one valid label for DP0 and DP1")
        scenarios = [
            (f"pair{index}-{pair[0]}-{pair[1]}", tuple(enumerate(pair))) for index, pair in enumerate(concurrent_pairs)
        ]
        mode = "concurrent"
    if not math.isfinite(STORE_WAIT_SECONDS) or STORE_WAIT_SECONDS < 0:
        raise ValueError("DSA_DP_L3_STORE_WAIT_SECONDS must be finite and non-negative")
    root = API_BASE.rstrip("/")
    if not root.endswith("/v1"):
        raise ValueError("DSA_DP_API_BASE must end in /v1")
    run_id = f"dsa-dp-accuracy-{uuid.uuid4().hex}"
    result_path = RESULT_DIR.resolve() / f"{run_id}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    async with (
        AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0) as client,
        httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, headers={"Authorization": f"Bearer {API_KEY}"}) as http,
    ):
        model = await _resolve_model(client)
        origin = time.perf_counter()
        print(
            f"[dsa-dp-accuracy] api={API_BASE} model={model} mode={mode} scenarios={scenarios} "
            f"max_tokens={MAX_TOKENS} results={result_path}",
            flush=True,
        )

        async def run(
            rank: int, label: str, phase: str, scenario: str, batch_members: list[dict], signature: str
        ) -> None:
            salt = f"{run_id}-{scenario}-dp{rank}-{label}"
            request_id = f"{salt}-{phase}"
            print(
                f"[dsa-dp-accuracy] sending phase={phase} dp={rank} label={label} request_id={request_id}", flush=True
            )
            started = time.perf_counter()
            record = {
                "run_id": run_id,
                "dp_rank": rank,
                "label": label,
                "phase": phase,
                "model": model,
                "salt": salt,
                "request_id": request_id,
                "seed": SEED,
                "prompt_sha256": hashlib.sha256(cases[label].prompt.encode("utf-8")).hexdigest(),
                "max_tokens": MAX_TOKENS,
                "top_logprobs": TOP_LOGPROBS,
                "execution_mode": mode,
                "scenario": scenario,
                "batch_id": f"{run_id}-{scenario}-{phase}",
                "batch_members": batch_members,
                "batch_signature": signature,
                "started_offset_ms": round((started - origin) * 1000, 3),
            }
            try:
                response = await client.completions.create(
                    model=model,
                    prompt=cases[label].prompt,
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                    logprobs=TOP_LOGPROBS,
                    stream=False,
                    extra_headers={"X-data-parallel-rank": str(rank), "X-Request-Id": request_id},
                    extra_body={
                        "seed": SEED,
                        "ignore_eos": True,
                        "cache_salt": salt,
                        "return_token_ids": True,
                        "return_tokens_as_token_ids": True,
                    },
                )
                record["response"] = response.model_dump()
            except Exception as exc:
                record["request_error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                finished = time.perf_counter()
                record["finished_offset_ms"] = round((finished - origin) * 1000, 3)
                record["latency_ms"] = round((finished - started) * 1000, 3)
                with result_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            payload = record["response"]
            prompt, completion, cached = _validate_usage(payload, expect_hit=phase != "seed", max_tokens=MAX_TOKENS)
            ids, _ = _trace(record)
            print(
                f"[dsa-dp-accuracy] phase={phase} dp={rank} label={label} prompt={prompt} "
                f"completion={completion} cached={cached} latency_ms={record['latency_ms']} "
                f"token_head={ids[:10]} request_id={request_id}",
                flush=True,
            )

        async def run_concurrent(phase, scenario, routes, members, signature):
            gate = asyncio.Event()
            remaining = len(routes)

            async def ready_then_run(rank: int, label: str) -> None:
                nonlocal remaining
                remaining -= 1
                if remaining == 0:
                    gate.set()
                await gate.wait()
                await run(rank, label, phase, scenario, members, signature)

            # Finish both requests (including a slow successful peer after an
            # exception) before resetting caches, changing phase, or closing HTTP.
            results = await asyncio.gather(
                *(ready_then_run(rank, label) for rank, label in routes), return_exceptions=True
            )
            failures = [
                f"DP{routes[index][0]}: {type(result).__name__}: {result}"
                for index, result in enumerate(results)
                if isinstance(result, BaseException)
            ]
            if failures:
                raise RuntimeError(f"concurrent round failed: {scenario}/{phase}: {failures}; evidence={result_path}")

        for scenario, routes in scenarios:
            members = [
                {
                    "dp_rank": rank,
                    "label": label,
                    "prompt_sha256": hashlib.sha256(cases[label].prompt.encode("utf-8")).hexdigest(),
                }
                for rank, label in routes
            ]
            signature = hashlib.sha256(
                json.dumps(
                    {
                        "model": model,
                        "members": members,
                        "max_tokens": MAX_TOKENS,
                        "seed": SEED,
                        "top_logprobs": TOP_LOGPROBS,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            for phase in PHASES:
                if phase.startswith("external-"):
                    await asyncio.sleep(STORE_WAIT_SECONDS)
                    reset = await http.post(root[:-3] + "/reset_prefix_cache", params={"reset_external": "false"})
                    reset.raise_for_status()
                    print(f"[dsa-dp-accuracy] local reset complete; scenario={scenario} next={phase}", flush=True)
                if mode == "serial":
                    rank, label = routes[0]
                    await run(rank, label, phase, scenario, members, signature)
                    continue
                await run_concurrent(phase, scenario, routes, members, signature)
    return result_path


def _parse_pair(value: str) -> tuple[str, str]:
    labels = tuple(item.strip() for item in value.split(","))
    if len(labels) != 2 or any(label not in ("5k", "8k", "40k", "70k") for label in labels):
        raise argparse.ArgumentTypeError("use DP0_LABEL,DP1_LABEL with labels from 5k,8k,40k,70k")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--analyze", type=Path, help="Analyze an existing JSONL without contacting a server")
    modes.add_argument(
        "--concurrent-pair",
        type=_parse_pair,
        action="append",
        metavar="DP0_LABEL,DP1_LABEL",
        help="Run one simultaneous request per DP in every phase; repeat for additional scenarios",
    )
    args = parser.parse_args()
    path = args.analyze if args.analyze is not None else asyncio.run(_collect(args.concurrent_pair))
    if not _analyze(path):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
