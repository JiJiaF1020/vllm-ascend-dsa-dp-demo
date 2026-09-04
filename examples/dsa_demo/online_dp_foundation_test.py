#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""单机 DP2 基础在线验证，同时提供各 DP 客户端共用的数据和连接设置。

混合请求后发送单个长请求，覆盖真实请求与空闲域 dummy 陪跑；本入口要求关闭
prefix/MTP/L3。DSA_DP_API_BASE/API_KEY/MODEL_NAME/REQUEST_TIMEOUT_SECONDS
和 DSA_DP_MAX_TOKENS 在导入时读取，环境变量需在启动 Python 前设置。
foundation/prefix/L3 功能客户端默认生成 32 tokens，精度客户端单独默认 64。
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

from openai import AsyncOpenAI

API_BASE = os.getenv("DSA_DP_API_BASE", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("DSA_DP_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("DSA_DP_MODEL_NAME")
MAX_TOKENS = int(os.getenv("DSA_DP_MAX_TOKENS", "32"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("DSA_DP_REQUEST_TIMEOUT_SECONDS", "900"))


def _validate_client_settings(*, max_tokens: int) -> None:
    if max_tokens < 2:
        raise ValueError("DSA_DP_MAX_TOKENS must be at least 2 to exercise decode")
    if not math.isfinite(REQUEST_TIMEOUT_SECONDS) or REQUEST_TIMEOUT_SECONDS <= 0:
        raise ValueError("DSA_DP_REQUEST_TIMEOUT_SECONDS must be finite and positive")
    if not API_BASE.rstrip("/").endswith("/v1"):
        raise ValueError("DSA_DP_API_BASE must end in /v1")


@dataclass(frozen=True, slots=True)
class PromptCase:
    label: str
    prompt: str


def _first_prompt(value: str | Sequence[str], *, name: str) -> str:
    if isinstance(value, str):
        prompt = value
    elif isinstance(value, Sequence) and value:
        prompt = value[0]
    else:
        raise ValueError(f"novel_dataset.{name} is empty")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"novel_dataset.{name} has no prompt")
    return prompt


def _load_cases() -> list[PromptCase]:
    try:
        from novel_dataset import (
            chinese_5k,
            chinese_8k,
            chinese_40k,
            chinese_70k,
        )
    except ImportError as exc:
        raise RuntimeError("online DP test requires novel_dataset.py on PYTHONPATH") from exc
    return [
        PromptCase("5k", _first_prompt(chinese_5k, name="chinese_5k")),
        PromptCase("8k", _first_prompt(chinese_8k, name="chinese_8k")),
        PromptCase("40k", _first_prompt(chinese_40k, name="chinese_40k")),
        PromptCase("70k", _first_prompt(chinese_70k, name="chinese_70k")),
    ]


async def _resolve_model(client: AsyncOpenAI) -> str:
    models = [item.id for item in (await client.models.list()).data]
    if MODEL_NAME:
        if MODEL_NAME not in models:
            raise RuntimeError(f"model {MODEL_NAME!r} is not served; available={models}")
        return MODEL_NAME
    if len(models) != 1:
        raise RuntimeError(f"DSA_DP_MODEL_NAME is required; available={models}")
    return models[0]


async def _run_case(
    client: AsyncOpenAI,
    *,
    model: str,
    phase: str,
    case: PromptCase,
) -> dict[str, object]:
    started = time.perf_counter()
    response = await client.completions.create(
        model=model,
        prompt=case.prompt,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        extra_body={"ignore_eos": True},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not response.choices:
        raise AssertionError(f"{phase}/{case.label} returned no choices")
    choice = response.choices[0]
    usage = response.usage
    prompt_tokens = int(usage.prompt_tokens) if usage is not None else 0
    completion_tokens = int(usage.completion_tokens) if usage is not None else 0
    if prompt_tokens <= 0 or completion_tokens != MAX_TOKENS:
        raise AssertionError(
            f"{phase}/{case.label} has invalid usage: "
            f"prompt={prompt_tokens}, completion={completion_tokens}, "
            f"expected_completion={MAX_TOKENS}"
        )
    result = {
        "phase": phase,
        "label": case.label,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": choice.finish_reason,
        "latency_ms": round(elapsed_ms, 3),
        "text_prefix": (choice.text or "")[:80],
    }
    print(f"[dsa-dp] {result}", flush=True)
    return result


async def _main() -> None:
    _validate_client_settings(max_tokens=MAX_TOKENS)
    cases = _load_cases()
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    try:
        model = await _resolve_model(client)
        print(
            f"[dsa-dp] connected api={API_BASE} model={model!r} max_tokens={MAX_TOKENS}",
            flush=True,
        )
        mixed = await asyncio.gather(
            *(
                _run_case(
                    client,
                    model=model,
                    phase="mixed",
                    case=case,
                )
                for case in cases
            )
        )
        single = await _run_case(
            client,
            model=model,
            phase="single-idle-peer",
            case=next(case for case in cases if case.label == "40k"),
        )
        print(
            f"[dsa-dp] PASSED mixed={len(mixed)} single={single['label']}",
            flush=True,
        )
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_main())
