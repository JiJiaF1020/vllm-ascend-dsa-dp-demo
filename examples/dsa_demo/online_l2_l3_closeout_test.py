#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""在线收口 DSA LocalDRAM L2 与 Mooncake L3 prefix 生命周期。

服务端必须启用 DSA、prefix caching、L3 write-behind 与 L3 prefix
loadback。脚本依次执行：

1. 四条请求串行 seed；L3 已存在时复用，否则 cold 计算并写入；
2. 相同请求串行执行 Local-L2 A/B 自对照；
3. drain store、只清本地 cache 后，串行执行两次强制 L3 A/B 自对照；
4. 对 Local-L2/L3 差异做同介质重放归因；
5. 多轮四请求并发 Local-L2/L3 replay，覆盖 continuous batching、请求完成、
   row reuse 和重复 namespace lookup。

只修改下方“用户配置”，然后直接运行：

    python examples/dsa_demo/online_l2_l3_closeout_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

# =========================
# 用户配置
# =========================

API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
# None 表示服务只暴露一个模型时自动选择；多模型服务必须填写准确名称。
MODEL_NAME: str | None = None
# 必须与 serve_dsa.sh 一致；eager 下不接受任何 top-k 放宽。
SERVER_RUN_MODE = "graph"  # eager / graph

MAX_TOKENS = 32
# 5k prompt 为 4973 tokens，尾块已有 109/128 个 token。限制其生成长度，
# 避免测试阶段把非确定续写填成新的 durable 满块并持续产生新 L3 key。
MAX_TOKENS_BY_LABEL = {"5k": 8}
TEMPERATURE = 0.0
TOP_LOGPROBS = 5
REQUEST_TIMEOUT_SECONDS = 600.0

# 默认只清本地 HBM/L2 并复用现有 L3 namespace，因此反复运行不会整份重写
# 相同 prompt。设为 True 可获得隔离的 cold namespace，但旧对象只会等待
# Mooncake 淘汰，不会被该接口物理删除，容量会按测试轮次增长。
FRESH_EXTERNAL_NAMESPACE_AT_START = False

# 每轮先执行两次并发 Local-L2 对照，再清本地并执行一次并发 L3。
# 3 轮会产生 9 个 bsz=4 batch。
CONCURRENT_STRESS_CYCLES = 3
ROTATE_REQUEST_ORDER = True

# 允许首个分叉 token 互相落在对方 top-k，但跨介质分叉还必须先在
# Local-L2/Local-L2 或 L3/L3 自对照中出现，不能只靠放宽精度门禁。
ALLOW_REPLAY_TOPK_VARIANCE = True
PRINT_SEQUENTIAL_STREAM = True

RESULT_JSON = "/tmp/dsa-online-l2-l3-closeout.json"

# 写入服务端 request_id，便于从 EngineCore/Worker 日志关联每个阶段。
RUN_TAG = f"{int(time.time())}-{os.getpid()}"


@dataclass(frozen=True)
class PromptCase:
    label: str
    prompt: str


@dataclass
class CompletionTrace:
    phase: str
    label: str
    request_id: str
    request_order: int
    requested_max_tokens: int
    prompt_chars: int
    text: str
    finish_reason: str | None
    token_ids: list[int]
    token_texts: list[str]
    token_logprobs: list[float | None]
    top_logprobs: list[dict[str, float] | None]
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    first_token_ms: float | None

    def exact_fingerprint(self) -> tuple[Any, ...]:
        return (
            self.text,
            self.finish_reason,
            tuple(self.token_ids),
        )


@dataclass
class ComparisonReport:
    label: str
    exact_mismatch_labels: list[str]
    accepted_topk_labels: list[str]
    unexplained_labels: list[str]
    divergences: dict[str, DivergenceRecord]


@dataclass(frozen=True)
class DivergenceRecord:
    common_prefix: int
    left_token: int | None
    right_token: int | None

    def normalized_signature(self) -> tuple[int, tuple[int, ...]]:
        tokens = tuple(sorted(token for token in (self.left_token, self.right_token) if token is not None))
        return self.common_prefix, tokens


def _first_prompt(value: str | Sequence[str], *, name: str) -> str:
    if isinstance(value, str):
        prompt = value
    elif isinstance(value, Sequence) and value:
        prompt = value[0]
    else:
        raise ValueError(f"novel_dataset.{name} is empty or invalid")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"novel_dataset.{name} has no non-empty text")
    return prompt


def load_prompt_cases() -> list[PromptCase]:
    try:
        from novel_dataset import (
            chinese_5k,
            chinese_8k,
            chinese_40k,
            chinese_70k,
        )
    except ImportError as exc:
        raise RuntimeError(
            "online L2/L3 closeout requires novel_dataset.py on PYTHONPATH "
            "with chinese_5k/chinese_8k/chinese_40k/chinese_70k"
        ) from exc

    # 沿用此前混合长度 graph 的顺序，便于和离线日志逐请求对照。
    return [
        PromptCase("8k", _first_prompt(chinese_8k, name="chinese_8k")),
        PromptCase("5k", _first_prompt(chinese_5k, name="chinese_5k")),
        PromptCase("70k", _first_prompt(chinese_70k, name="chinese_70k")),
        PromptCase("40k", _first_prompt(chinese_40k, name="chinese_40k")),
    ]


def _server_root() -> str:
    return API_BASE.removesuffix("/v1").rstrip("/")


def _post_dev_api(path: str, *, timeout: float = 600.0) -> None:
    request = urllib.request.Request(
        f"{_server_root()}{path}",
        method="POST",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"dev API failed: path={path!r}, status={response.status}")


async def _run_blocking(function: Any, /, *args: Any, **kwargs: Any) -> None:
    await asyncio.to_thread(function, *args, **kwargs)


async def pause_and_reset(*, reset_external: bool) -> None:
    """等待请求和 write-behind 完成，再原子清理 prefix namespace。"""

    await _run_blocking(
        _post_dev_api,
        "/pause?mode=wait&clear_cache=false",
    )
    try:
        reset = "true" if reset_external else "false"
        await _run_blocking(
            _post_dev_api,
            f"/reset_prefix_cache?reset_external={reset}",
            timeout=60.0,
        )
    finally:
        await _run_blocking(
            _post_dev_api,
            "/resume",
            timeout=60.0,
        )
    scope = "local+external namespace" if reset_external else "local HBM/L2"
    print(f"[online-closeout] drained stores and reset {scope}")


async def drain_stores() -> None:
    await _run_blocking(
        _post_dev_api,
        "/pause?mode=wait&clear_cache=false",
    )
    await _run_blocking(_post_dev_api, "/resume", timeout=60.0)
    print("[online-closeout] pending requests/stores drained")


def _append_choice_metadata(
    choice: Any,
    token_ids: list[int],
    token_texts: list[str],
    token_logprobs: list[float | None],
    top_logprobs: list[dict[str, float] | None],
) -> None:
    payload = choice.model_dump() if hasattr(choice, "model_dump") else {}
    chunk_token_ids = payload.get("token_ids")
    if chunk_token_ids:
        token_ids.extend(int(token_id) for token_id in chunk_token_ids)

    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None:
        return
    tokens = getattr(logprobs, "tokens", None) or []
    values = getattr(logprobs, "token_logprobs", None) or []
    token_texts.extend(str(token) for token in tokens)
    token_logprobs.extend(None if value is None else float(value) for value in values)
    rows = getattr(logprobs, "top_logprobs", None) or []
    top_logprobs.extend(
        None if row is None else {str(token): float(value) for token, value in row.items()} for row in rows
    )


async def stream_completion(
    client: AsyncOpenAI,
    *,
    model_name: str,
    case: PromptCase,
    phase: str,
    request_order: int,
    print_stream: bool,
) -> CompletionTrace:
    started = time.perf_counter()
    requested_max_tokens = MAX_TOKENS_BY_LABEL.get(case.label, MAX_TOKENS)
    request_id = f"dsa-closeout-{RUN_TAG}-{phase}-{case.label}-{request_order}"
    first_token_ms: float | None = None
    text_parts: list[str] = []
    token_ids: list[int] = []
    token_texts: list[str] = []
    token_logprobs: list[float | None] = []
    top_logprobs: list[dict[str, float] | None] = []
    finish_reason: str | None = None
    final_usage = None

    stream = await client.completions.create(
        model=model_name,
        prompt=case.prompt,
        temperature=TEMPERATURE,
        max_tokens=requested_max_tokens,
        logprobs=TOP_LOGPROBS,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={
            "request_id": request_id,
            "seed": 1024,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        },
    )
    if print_stream:
        print(f"[{phase}:{case.label}]> ", end="", flush=True)
    async for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            final_usage = chunk.usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason is not None:
            finish_reason = str(choice.finish_reason)
        _append_choice_metadata(
            choice,
            token_ids,
            token_texts,
            token_logprobs,
            top_logprobs,
        )
        text = choice.text or ""
        if text:
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000.0
            text_parts.append(text)
            if print_stream:
                print(text, end="", flush=True)
    if print_stream:
        print()

    latency_ms = (time.perf_counter() - started) * 1000.0
    trace = CompletionTrace(
        phase=phase,
        label=case.label,
        request_id=request_id,
        request_order=request_order,
        requested_max_tokens=requested_max_tokens,
        prompt_chars=len(case.prompt),
        text="".join(text_parts),
        finish_reason=finish_reason,
        token_ids=token_ids,
        token_texts=token_texts,
        token_logprobs=token_logprobs,
        top_logprobs=top_logprobs,
        prompt_tokens=(None if final_usage is None else int(final_usage.prompt_tokens)),
        completion_tokens=(None if final_usage is None else int(final_usage.completion_tokens)),
        latency_ms=latency_ms,
        first_token_ms=first_token_ms,
    )
    print(
        "[online-closeout] result "
        f"phase={phase} label={case.label} request_id={request_id} "
        f"order={request_order} max_tokens={requested_max_tokens} "
        f"prompt_tokens={trace.prompt_tokens} "
        f"completion_tokens={trace.completion_tokens} "
        f"finish={trace.finish_reason!r} "
        f"ttft_ms={trace.first_token_ms} latency_ms={trace.latency_ms:.3f}"
    )
    print(f"[online-closeout] output label={case.label} token_ids={trace.token_ids} text={trace.text!r}")
    return trace


async def run_phase(
    client: AsyncOpenAI,
    *,
    model_name: str,
    cases: list[PromptCase],
    phase: str,
    concurrent: bool,
) -> dict[str, CompletionTrace]:
    if concurrent:
        traces = await asyncio.gather(
            *(
                stream_completion(
                    client,
                    model_name=model_name,
                    case=case,
                    phase=phase,
                    request_order=index,
                    print_stream=False,
                )
                for index, case in enumerate(cases)
            )
        )
    else:
        traces = []
        for index, case in enumerate(cases):
            traces.append(
                await stream_completion(
                    client,
                    model_name=model_name,
                    case=case,
                    phase=phase,
                    request_order=index,
                    print_stream=PRINT_SEQUENTIAL_STREAM,
                )
            )
    return {trace.label: trace for trace in traces}


def _token_id_from_logprob_key(value: str) -> int | None:
    prefix = "token_id:"
    if not value.startswith(prefix):
        return None
    try:
        return int(value.removeprefix(prefix))
    except ValueError:
        return None


def _candidate_logprobs(
    trace: CompletionTrace,
    position: int,
) -> dict[int, float]:
    if position >= len(trace.top_logprobs):
        return {}
    row = trace.top_logprobs[position]
    if row is None:
        return {}
    return {
        token_id: float(logprob)
        for token, logprob in row.items()
        if (token_id := _token_id_from_logprob_key(token)) is not None
    }


def _first_token_divergence(
    left: CompletionTrace,
    right: CompletionTrace,
) -> tuple[int, int | None, int | None]:
    common_prefix = 0
    for left_id, right_id in zip(
        left.token_ids,
        right.token_ids,
        strict=False,
    ):
        if left_id != right_id:
            break
        common_prefix += 1
    left_id = left.token_ids[common_prefix] if common_prefix < len(left.token_ids) else None
    right_id = right.token_ids[common_prefix] if common_prefix < len(right.token_ids) else None
    return common_prefix, left_id, right_id


def compare_phases(
    left: dict[str, CompletionTrace],
    right: dict[str, CompletionTrace],
    *,
    label: str,
    allow_topk_variance: bool,
) -> ComparisonReport:
    exact_mismatches: list[str] = []
    accepted_topk: list[str] = []
    unexplained: list[str] = []
    divergences: dict[str, DivergenceRecord] = {}
    if set(left) != set(right):
        raise AssertionError(f"{label} request labels differ: left={sorted(left)} right={sorted(right)}")
    for prompt_label in left:
        lhs = left[prompt_label]
        rhs = right[prompt_label]
        if lhs.exact_fingerprint() == rhs.exact_fingerprint():
            continue
        exact_mismatches.append(prompt_label)
        common_prefix, left_id, right_id = _first_token_divergence(
            lhs,
            rhs,
        )
        divergences[prompt_label] = DivergenceRecord(
            common_prefix=common_prefix,
            left_token=left_id,
            right_token=right_id,
        )
        left_candidates = _candidate_logprobs(lhs, common_prefix)
        right_candidates = _candidate_logprobs(rhs, common_prefix)
        topk_equivalent = (
            allow_topk_variance
            and ALLOW_REPLAY_TOPK_VARIANCE
            and SERVER_RUN_MODE == "graph"
            and common_prefix > 0
            and left_id is not None
            and right_id is not None
            and left_id in right_candidates
            and right_id in left_candidates
        )
        detail = (
            f"{label} mismatch prompt={prompt_label}: "
            f"left_request={lhs.request_id} right_request={rhs.request_id} "
            f"common_prefix={common_prefix} "
            f"left_token={left_id} right_token={right_id} "
            f"left_topk={sorted(left_candidates.items(), key=lambda row: -row[1])} "
            f"right_topk={sorted(right_candidates.items(), key=lambda row: -row[1])} "
            f"left_text={lhs.text!r} right_text={rhs.text!r} "
            f"left_ids={lhs.token_ids} right_ids={rhs.token_ids}"
        )
        print(f"[online-closeout] {detail}")
        if topk_equivalent:
            accepted_topk.append(prompt_label)
            print(f"[online-closeout] accepted controlled top-k variance prompt={prompt_label}")
        else:
            unexplained.append(prompt_label)

    if unexplained:
        status = "unexplained"
    else:
        status = "exact" if not exact_mismatches else "top-k-equivalent"
    print(f"[online-closeout] comparison {label}: {status}; mismatches={len(exact_mismatches)}/{len(left)}")
    return ComparisonReport(
        label=label,
        exact_mismatch_labels=exact_mismatches,
        accepted_topk_labels=accepted_topk,
        unexplained_labels=unexplained,
        divergences=divergences,
    )


def find_cross_variance_absent_from_replay_controls(
    controls: Sequence[ComparisonReport],
    cross_replay: ComparisonReport,
    *,
    label: str,
) -> list[tuple[str, int, int | None, int | None]]:
    observed_signatures = {
        (prompt_label, divergence.normalized_signature())
        for control in controls
        for prompt_label, divergence in control.divergences.items()
    }
    cross_only = [
        (
            prompt_label,
            divergence.common_prefix,
            divergence.left_token,
            divergence.right_token,
        )
        for prompt_label, divergence in cross_replay.divergences.items()
        if (
            prompt_label,
            divergence.normalized_signature(),
        )
        not in observed_signatures
    ]
    if cross_only:
        print(f"[online-closeout] {label} has cross-source-only divergences: {cross_only}")
    else:
        print(
            f"[online-closeout] {label} variance attribution passed; "
            f"control_signatures={sorted(observed_signatures)} "
            f"cross={cross_replay.divergences}"
        )
    return cross_only


def record_comparison_failures(
    failures: list[str],
    report: ComparisonReport,
) -> None:
    if report.unexplained_labels:
        failures.append(f"{report.label}: unexplained={report.unexplained_labels}")


def rotate_cases(cases: list[PromptCase], cycle: int) -> list[PromptCase]:
    if not ROTATE_REQUEST_ORDER or not cases:
        return list(cases)
    offset = cycle % len(cases)
    return [*cases[offset:], *cases[:offset]]


async def resolve_model_name(client: AsyncOpenAI) -> str:
    available = [model.id for model in (await client.models.list()).data]
    if MODEL_NAME is not None:
        if MODEL_NAME not in available:
            raise RuntimeError(f"model {MODEL_NAME!r} is not served; available={available}")
        return MODEL_NAME
    if len(available) != 1:
        raise RuntimeError(f"MODEL_NAME=None requires exactly one served model; available={available}")
    return available[0]


def write_result(
    *,
    model_name: str,
    phases: dict[str, dict[str, CompletionTrace]],
    comparisons: dict[str, ComparisonReport],
    diagnostic_failures: list[str],
    failure: str | None,
    drain_failure: str | None,
) -> Path:
    payload = {
        "api_base": API_BASE,
        "model": model_name,
        "server_run_mode": SERVER_RUN_MODE,
        "run_tag": RUN_TAG,
        "max_tokens": MAX_TOKENS,
        "max_tokens_by_label": MAX_TOKENS_BY_LABEL,
        "fresh_external_namespace_at_start": (FRESH_EXTERNAL_NAMESPACE_AT_START),
        "concurrent_stress_cycles": CONCURRENT_STRESS_CYCLES,
        "allow_replay_topk_variance": ALLOW_REPLAY_TOPK_VARIANCE,
        "phases": {
            phase: {label: asdict(trace) for label, trace in traces.items()} for phase, traces in phases.items()
        },
        "comparisons": {label: asdict(report) for label, report in comparisons.items()},
        "diagnostic_failures": diagnostic_failures,
        "status": "passed" if failure is None and drain_failure is None else "failed",
        "failure": failure,
        "drain_failure": drain_failure,
    }
    result_path = Path(RESULT_JSON)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_path


async def main_async() -> None:
    if SERVER_RUN_MODE not in {"eager", "graph"}:
        raise ValueError("SERVER_RUN_MODE must be eager or graph")
    if MAX_TOKENS <= 0 or any(value <= 0 for value in MAX_TOKENS_BY_LABEL.values()):
        raise ValueError("completion token limits must be positive")
    if CONCURRENT_STRESS_CYCLES < 0:
        raise ValueError("CONCURRENT_STRESS_CYCLES must be non-negative")
    cases = load_prompt_cases()
    unknown_overrides = set(MAX_TOKENS_BY_LABEL) - {case.label for case in cases}
    if unknown_overrides:
        raise ValueError(f"MAX_TOKENS_BY_LABEL has unknown labels: {unknown_overrides}")
    phases: dict[str, dict[str, CompletionTrace]] = {}
    comparisons: dict[str, ComparisonReport] = {}
    diagnostic_failures: list[str] = []
    failure: str | None = None
    drain_failure: str | None = None
    model_name = MODEL_NAME or "<unresolved>"

    async with AsyncOpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as client:
        model_name = await resolve_model_name(client)
        print(
            "[online-closeout] connected "
            f"api={API_BASE} model={model_name!r} "
            f"server_run_mode={SERVER_RUN_MODE} run_tag={RUN_TAG} "
            f"cases={[(case.label, len(case.prompt)) for case in cases]}"
        )

        try:
            await pause_and_reset(reset_external=FRESH_EXTERNAL_NAMESPACE_AT_START)

            phases["seed-sequential"] = await run_phase(
                client,
                model_name=model_name,
                cases=cases,
                phase="seed-sequential",
                concurrent=False,
            )
            phases["local-l2-sequential-a"] = await run_phase(
                client,
                model_name=model_name,
                cases=cases,
                phase="local-l2-sequential-a",
                concurrent=False,
            )
            phases["local-l2-sequential-b"] = await run_phase(
                client,
                model_name=model_name,
                cases=cases,
                phase="local-l2-sequential-b",
                concurrent=False,
            )
            comparisons["sequential-local-l2-control"] = compare_phases(
                phases["local-l2-sequential-a"],
                phases["local-l2-sequential-b"],
                label="sequential Local-L2 control",
                allow_topk_variance=True,
            )
            record_comparison_failures(
                diagnostic_failures,
                comparisons["sequential-local-l2-control"],
            )

            await pause_and_reset(reset_external=False)
            phases["l3-sequential-a"] = await run_phase(
                client,
                model_name=model_name,
                cases=cases,
                phase="l3-sequential-a",
                concurrent=False,
            )
            await pause_and_reset(reset_external=False)
            phases["l3-sequential-b"] = await run_phase(
                client,
                model_name=model_name,
                cases=cases,
                phase="l3-sequential-b",
                concurrent=False,
            )
            comparisons["sequential-l3-control"] = compare_phases(
                phases["l3-sequential-a"],
                phases["l3-sequential-b"],
                label="sequential L3 control",
                allow_topk_variance=True,
            )
            record_comparison_failures(
                diagnostic_failures,
                comparisons["sequential-l3-control"],
            )
            comparisons["sequential-local-l2-vs-l3"] = compare_phases(
                phases["local-l2-sequential-b"],
                phases["l3-sequential-a"],
                label="sequential Local-L2/L3",
                allow_topk_variance=True,
            )
            record_comparison_failures(
                diagnostic_failures,
                comparisons["sequential-local-l2-vs-l3"],
            )
            sequential_cross_only = find_cross_variance_absent_from_replay_controls(
                (
                    comparisons["sequential-local-l2-control"],
                    comparisons["sequential-l3-control"],
                ),
                comparisons["sequential-local-l2-vs-l3"],
                label="sequential Local-L2/L3",
            )
            if sequential_cross_only:
                diagnostic_failures.append(f"sequential Local-L2/L3 cross-source-only={sequential_cross_only}")

            for cycle in range(CONCURRENT_STRESS_CYCLES):
                cycle_cases = rotate_cases(cases, cycle)
                local_a_name = f"stress-{cycle + 1}-local-l2-a"
                local_b_name = f"stress-{cycle + 1}-local-l2-b"
                l3_name = f"stress-{cycle + 1}-l3"
                phases[local_a_name] = await run_phase(
                    client,
                    model_name=model_name,
                    cases=cycle_cases,
                    phase=local_a_name,
                    concurrent=True,
                )
                phases[local_b_name] = await run_phase(
                    client,
                    model_name=model_name,
                    cases=cycle_cases,
                    phase=local_b_name,
                    concurrent=True,
                )
                local_key = f"stress-{cycle + 1}-local-control"
                comparisons[local_key] = compare_phases(
                    phases[local_a_name],
                    phases[local_b_name],
                    label=f"stress cycle {cycle + 1} Local-L2 control",
                    allow_topk_variance=True,
                )
                record_comparison_failures(
                    diagnostic_failures,
                    comparisons[local_key],
                )

                await pause_and_reset(reset_external=False)
                phases[l3_name] = await run_phase(
                    client,
                    model_name=model_name,
                    cases=cycle_cases,
                    phase=l3_name,
                    concurrent=True,
                )
                l3_key = f"stress-{cycle + 1}-local-l2-vs-l3"
                comparisons[l3_key] = compare_phases(
                    phases[local_b_name],
                    phases[l3_name],
                    label=f"stress cycle {cycle + 1} Local-L2/L3",
                    allow_topk_variance=True,
                )
                record_comparison_failures(
                    diagnostic_failures,
                    comparisons[l3_key],
                )
                stress_cross_only = find_cross_variance_absent_from_replay_controls(
                    (comparisons[local_key],),
                    comparisons[l3_key],
                    label=(f"stress cycle {cycle + 1} Local-L2/L3"),
                )
                if stress_cross_only:
                    diagnostic_failures.append(f"stress cycle {cycle + 1} cross-source-only={stress_cross_only}")

            if diagnostic_failures:
                raise AssertionError(f"online closeout collected diagnostic failures: {diagnostic_failures}")
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                await drain_stores()
            except Exception as exc:
                drain_failure = f"{type(exc).__name__}: {exc}"
                if failure is None:
                    raise
                print(f"[online-closeout] drain failed after primary error: {drain_failure}")
            finally:
                result_path = write_result(
                    model_name=model_name,
                    phases=phases,
                    comparisons=comparisons,
                    diagnostic_failures=diagnostic_failures,
                    failure=failure,
                    drain_failure=drain_failure,
                )
                print(
                    "[online-closeout] result written "
                    f"status={'passed' if failure is None and drain_failure is None else 'failed'} "
                    f"path={result_path}"
                )

    print(f"[online-closeout] PASSED; result={RESULT_JSON}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
