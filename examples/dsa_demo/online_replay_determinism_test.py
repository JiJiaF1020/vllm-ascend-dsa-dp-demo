#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe repeated online completions without cache reset or L3 orchestration."""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

from openai import AsyncOpenAI

# =========================
# User configuration
# =========================

API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME: str | None = None
PROMPT_LABELS = ("5k", "40k")
REPEATS = 3
COMPARE_START_REPEAT = 0
MAX_TOKENS = 32
MAX_TOKENS_BY_LABEL = {"5k": 8, "40k": 10}
IGNORE_EOS = False
TOP_LOGPROBS = 5
REQUEST_TIMEOUT_SECONDS = 600.0
RUN_TAG = f"{int(time.time())}-{os.getpid()}"


@dataclass(frozen=True)
class PromptCase:
    label: str
    prompt: str


@dataclass(frozen=True)
class CompletionTrace:
    token_ids: tuple[int, ...]
    top_logprobs: tuple[dict[int, float], ...]
    text: str


def _first_prompt(value: str | Sequence[str], *, name: str) -> str:
    if isinstance(value, str):
        prompt = value
    elif isinstance(value, Sequence) and value:
        prompt = value[0]
    else:
        raise ValueError(f"novel_dataset.{name} is empty")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"novel_dataset.{name} has no text")
    return prompt


def load_cases() -> list[PromptCase]:
    try:
        from novel_dataset import chinese_5k, chinese_40k
    except ImportError as exc:
        raise RuntimeError(
            "online determinism probe requires novel_dataset.py on PYTHONPATH"
        ) from exc
    available = {
        "5k": _first_prompt(chinese_5k, name="chinese_5k"),
        "40k": _first_prompt(chinese_40k, name="chinese_40k"),
    }
    unknown = set(PROMPT_LABELS) - available.keys()
    if unknown:
        raise ValueError(f"unknown prompt labels: {tuple(sorted(unknown))}")
    return [PromptCase(label, available[label]) for label in PROMPT_LABELS]


def _token_id(value: str) -> int | None:
    prefix = "token_id:"
    if not value.startswith(prefix):
        return None
    try:
        return int(value.removeprefix(prefix))
    except ValueError:
        return None


async def resolve_model(client: AsyncOpenAI) -> str:
    models = [model.id for model in (await client.models.list()).data]
    if MODEL_NAME is not None:
        if MODEL_NAME not in models:
            raise RuntimeError(
                f"model {MODEL_NAME!r} is not served; available={models}"
            )
        return MODEL_NAME
    if len(models) != 1:
        raise RuntimeError(f"MODEL_NAME is required; available={models}")
    return models[0]


async def run_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    case: PromptCase,
    repeat: int,
) -> CompletionTrace:
    request_id = f"dsa-determinism-{RUN_TAG}-{case.label}-{repeat}"
    response = await client.completions.create(
        model=model,
        prompt=case.prompt,
        temperature=0.0,
        max_tokens=MAX_TOKENS_BY_LABEL.get(case.label, MAX_TOKENS),
        logprobs=TOP_LOGPROBS,
        stream=False,
        extra_body={
            "request_id": request_id,
            "seed": 1024,
            "ignore_eos": IGNORE_EOS,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        },
    )
    choice = response.choices[0]
    payload = response.model_dump()
    choice_payload = payload["choices"][0]
    token_ids = tuple(
        int(token) for token in choice_payload.get("token_ids") or ()
    )
    logprobs = getattr(choice, "logprobs", None)
    rows = getattr(logprobs, "top_logprobs", None) or ()
    top_logprobs: list[dict[int, float]] = []
    for row in rows:
        candidates: dict[int, float] = {}
        if row is not None:
            for token, value in row.items():
                token_id = _token_id(str(token))
                if token_id is not None:
                    candidates[token_id] = float(value)
        top_logprobs.append(candidates)
    trace = CompletionTrace(
        token_ids=token_ids,
        top_logprobs=tuple(top_logprobs),
        text=choice.text or "",
    )
    print(
        "[online-determinism] "
        f"label={case.label} repeat={repeat} request_id={request_id} "
        f"token_ids={list(trace.token_ids)} text={trace.text!r}"
    )
    return trace


def compare_traces(
    baseline: CompletionTrace,
    candidate: CompletionTrace,
    *,
    label: str,
    repeat: int,
) -> str | None:
    if baseline.token_ids == candidate.token_ids:
        return None
    common = 0
    for left, right in zip(
        baseline.token_ids,
        candidate.token_ids,
        strict=False,
    ):
        if left != right:
            break
        common += 1
    left = baseline.token_ids[common] if common < len(baseline.token_ids) else None
    right = candidate.token_ids[common] if common < len(candidate.token_ids) else None
    left_topk = (
        baseline.top_logprobs[common]
        if common < len(baseline.top_logprobs)
        else {}
    )
    right_topk = (
        candidate.top_logprobs[common]
        if common < len(candidate.top_logprobs)
        else {}
    )
    first_logprob_drift = None
    comparable_rows = min(
        common + 1,
        len(baseline.top_logprobs),
        len(candidate.top_logprobs),
    )
    for index in range(comparable_rows):
        left_row = baseline.top_logprobs[index]
        right_row = candidate.top_logprobs[index]
        keys = set(left_row) | set(right_row)
        if keys != set(left_row) or keys != set(right_row):
            first_logprob_drift = (
                index,
                "candidate_set",
                left_row,
                right_row,
            )
            break
        max_abs = max(
            (
                abs(left_row[token_id] - right_row[token_id])
                for token_id in keys
            ),
            default=0.0,
        )
        if max_abs > 1e-6:
            first_logprob_drift = (
                index,
                f"max_abs={max_abs:.9g}",
                left_row,
                right_row,
            )
            break
    detail = (
        f"label={label} repeat={repeat} common_prefix={common} "
        f"left_token={left} right_token={right} "
        f"left_topk={left_topk} right_topk={right_topk} "
        f"first_logprob_drift={first_logprob_drift}"
    )
    print(f"[online-determinism] mismatch {detail}")
    return detail


async def main_async() -> None:
    if REPEATS < 1:
        raise ValueError("REPEATS must be at least one")
    if REPEATS == 1 and COMPARE_START_REPEAT != 0:
        raise ValueError("single-run mode requires COMPARE_START_REPEAT=0")
    if REPEATS > 1 and not 0 <= COMPARE_START_REPEAT < REPEATS - 1:
        raise ValueError("COMPARE_START_REPEAT must leave two compared repeats")
    cases = load_cases()
    failures: list[str] = []
    async with AsyncOpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        timeout=REQUEST_TIMEOUT_SECONDS,
    ) as client:
        model = await resolve_model(client)
        print(
            f"[online-determinism] connected api={API_BASE} model={model!r} "
            f"repeats={REPEATS} compare_start={COMPARE_START_REPEAT}"
        )
        for case in cases:
            traces = [
                await run_completion(
                    client,
                    model=model,
                    case=case,
                    repeat=repeat,
                )
                for repeat in range(REPEATS)
            ]
            if REPEATS > 1:
                baseline = traces[COMPARE_START_REPEAT]
                for repeat in range(
                    COMPARE_START_REPEAT + 1,
                    REPEATS,
                ):
                    failure = compare_traces(
                        baseline,
                        traces[repeat],
                        label=case.label,
                        repeat=repeat,
                    )
                    if failure is not None:
                        failures.append(failure)
    if failures:
        raise AssertionError(
            f"online repeated completions are not deterministic: {failures}"
        )
    if REPEATS == 1:
        print("[online-determinism] COMPLETED (single run; no comparison)")
    else:
        print("[online-determinism] PASSED")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
