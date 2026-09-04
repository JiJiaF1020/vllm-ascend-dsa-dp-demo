#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DP2 本地前缀复用的在线验证，使用显式 DP 路由和请求级命中断言。

服务需开启 DSA 与 prefix caching，L3 关闭，可选已适配的 MTP3。每次运行
使用独立 cache_salt，保持 prompt 文本不变，无需重置整个服务的缓存。
完整响应和 token 输出持久保存；本脚本验证路由、命中和执行完成，不以
不同 batch 的输出差异直接判定精度错误。MTP 验证建议生成至少 64 tokens。
共用环境变量由 online_dp_foundation_test 读取；DSA_DP_MAX_TOKENS 默认 32，
固定 ignore_eos=True。结果目录使用 DSA_DP_PREFIX_RESULT_DIR。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

from online_dp_foundation_test import (
    API_BASE,
    API_KEY,
    MAX_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    PromptCase,
    _load_cases,
    _resolve_model,
    _validate_client_settings,
)
from openai import AsyncOpenAI

RESULT_DIR = Path(os.getenv("DSA_DP_PREFIX_RESULT_DIR", str(Path(__file__).resolve().parent / "results" / "dp_prefix")))


def _validate_usage(response: dict, *, expect_hit: bool | None, max_tokens: int) -> tuple[int, int, int]:
    choices = response.get("choices") or []
    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    # v0.23 omits prompt_tokens_details when cached_tokens == 0.
    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = int(details.get("cached_tokens") or 0)
    if len(choices) != 1 or prompt_tokens <= 0 or completion_tokens != max_tokens:
        raise AssertionError(f"invalid completion/usage: choices={len(choices)} usage={usage}")
    if choices[0].get("finish_reason") != "length":
        raise AssertionError(f"expected length stop with ignore_eos: {choices[0].get('finish_reason')!r}")
    if cached_tokens < 0 or cached_tokens >= prompt_tokens or cached_tokens % 128:
        raise AssertionError(f"invalid DSA cached prefix: cached={cached_tokens} prompt={prompt_tokens}")
    if expect_hit and not cached_tokens:
        raise AssertionError(
            "expected this DP to reuse a prefix, got zero cached tokens; "
            "check enable_prefix_caching, --enable-prompt-tokens-details and server lookup logs"
        )
    if expect_hit is False and cached_tokens:
        raise AssertionError(
            f"expected a cold DP for this salt, got cached_tokens={cached_tokens}; "
            "check X-data-parallel-rank routing, cache salt and expected cache tiers"
        )
    return prompt_tokens, completion_tokens, cached_tokens


async def _run_case(
    client: AsyncOpenAI,
    *,
    model: str,
    case: PromptCase,
    dp_rank: int,
    phase: str,
    salt: str,
    expect_hit: bool | None,
    request_id: str,
    records: list[dict],
    result_path: Path,
    max_tokens: int,
) -> dict:
    started = time.perf_counter()
    response = await client.completions.create(
        model=model,
        prompt=case.prompt,
        temperature=0.0,
        max_tokens=max_tokens,
        extra_headers={"X-data-parallel-rank": str(dp_rank), "X-Request-Id": request_id},
        extra_body={"ignore_eos": True, "cache_salt": salt, "return_token_ids": True},
    )
    payload = response.model_dump()
    record = {
        "phase": phase,
        "dp_rank": dp_rank,
        "label": case.label,
        "request_id": request_id,
        "response_id": payload.get("id"),
        "expect_hit": expect_hit,
        "cache_salt": salt,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "usage": payload.get("usage"),
        "choices": payload.get("choices"),
    }
    # Persist before asserting so a failure still has its response evidence.
    with result_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    records.append(record)
    try:
        prompt, completion, cached = _validate_usage(payload, expect_hit=expect_hit, max_tokens=max_tokens)
    except AssertionError as exc:
        raise AssertionError(f"{phase}/DP{dp_rank}/{case.label}: {exc}; evidence={result_path}") from exc
    print(
        f"[dsa-dp-prefix] phase={phase} dp={dp_rank} label={case.label} "
        f"prompt_tokens={prompt} completion_tokens={completion} cached_tokens={cached} "
        f"latency_ms={record['latency_ms']} request_id={request_id}",
        flush=True,
    )
    return record


async def _main() -> None:
    _validate_client_settings(max_tokens=MAX_TOKENS)
    cases = {case.label: case for case in _load_cases()}
    run_id = f"dsa-dp-prefix-{uuid.uuid4().hex}"
    result_path = RESULT_DIR.resolve() / f"{run_id}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    async with AsyncOpenAI(
        api_key=API_KEY,
        base_url=API_BASE,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    ) as client:
        model = await _resolve_model(client)
        print(
            f"[dsa-dp-prefix] api={API_BASE} model={model!r} max_tokens={MAX_TOKENS} "
            f"ignore_eos=True results={result_path}",
            flush=True,
        )

        async def run(phase: str, label: str, rank: int, salt_group: str, hit: bool) -> None:
            await _run_case(
                client,
                model=model,
                case=cases[label],
                dp_rank=rank,
                phase=phase,
                salt=f"{run_id}-{salt_group}-{label}",
                expect_hit=hit,
                request_id=f"{run_id}-{phase}-dp{rank}-{label}",
                records=records,
                result_path=result_path,
                max_tokens=MAX_TOKENS,
            )

        # The same salted prompt is warm only on the DP that previously ran it.
        for rank in (0, 1):
            await run("isolated-cold", "40k", rank, "isolation", False)
            await run("isolated-warm", "40k", rank, "isolation", True)

        async def batch(items) -> None:
            results = await asyncio.gather(*(run(*item) for item in items), return_exceptions=True)
            failures = [str(result) for result in results if isinstance(result, BaseException)]
            if failures:
                raise AssertionError(f"DP prefix batch failed: {failures}; evidence={result_path}")

        routing = (("5k", 0), ("8k", 1), ("40k", 0), ("70k", 1))
        for phase, hit in (("mixed-cold", False), ("mixed-warm", True)):
            await batch((phase, label, rank, "mixed", hit) for label, rank in routing)

        # Warm short prefill/decode on DP1 overlaps a cold long prefill on DP0.
        await batch(
            (
                ("asymmetric", "70k", 0, "asymmetric", False),
                ("asymmetric", "40k", 1, "isolation", True),
            )
        )
        # Both domains re-enter with a cached request after peer idle work.
        for rank in (0, 1):
            await run("idle-reentry", "40k", rank, "isolation", True)

    print(
        f"[dsa-dp-prefix] PASSED requests={len(records)} "
        f"(routing/cache-hit/progress only; numerical equivalence not asserted) results={result_path}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(_main())
