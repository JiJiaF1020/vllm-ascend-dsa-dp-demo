#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""单机 DP2 共享 L3 验证：跨域首次读取、清本地后读回与并发同 key 请求。

服务开启 DSA + Local prefix + Mooncake TCP kv_both，可选P6-A固定MTP3。客户端只调用
基线显式 DP 路由和 local-only reset；不轮换 L3 namespace。等待时间只是
给异步上传留窗口，可读性仍以服务端 completed/get 与请求命中结果验收。
完整响应保存在持久 JSONL，数值一致性不在本脚本中宣称通过。
共用连接设置及 MAX_TOKENS 从 online_dp_foundation_test 导入，后者读取
DSA_DP_MAX_TOKENS（默认 32）；调用 prefix 的请求辅助函数时显式传入。
DSA_DP_L3_LABELS 默认 40k，STORE_WAIT_SECONDS 默认 10 秒，RESULT_DIR
用于持久结果目录；这三个设置均使用 DSA_DP_L3_ 环境变量前缀。
P6-A运行时加 --mtp，检查一次Eagle末块回退，建议 DSA_DP_MAX_TOKENS=128。
--concurrent-pair 8k,70k并发冷写后交换DP回载，再本地重放及清本地回载，共8请求。
此模式不使用DSA_DP_L3_LABELS；两份来源使用独立salt，即使长度相同也不混淆本地命中。
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import uuid
from pathlib import Path

import httpx
from online_dp_foundation_test import (
    API_BASE,
    API_KEY,
    MAX_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    _load_cases,
    _resolve_model,
    _validate_client_settings,
)
from online_dp_prefix_test import _run_case
from openai import AsyncOpenAI

LABELS = tuple(label.strip() for label in os.getenv("DSA_DP_L3_LABELS", "40k").split(",") if label.strip())
STORE_WAIT_SECONDS = float(os.getenv("DSA_DP_L3_STORE_WAIT_SECONDS", "10"))
RESULT_DIR = Path(os.getenv("DSA_DP_L3_RESULT_DIR", str(Path(__file__).resolve().parent / "results" / "dp_l3")))


def _parse_pair(value: str) -> tuple[str, str]:
    labels = tuple(item.strip() for item in value.split(","))
    if len(labels) != 2 or any(label not in ("5k", "8k", "40k", "70k") for label in labels):
        raise argparse.ArgumentTypeError("use DP0_LABEL,DP1_LABEL with labels from 5k,8k,40k,70k")
    return labels


async def _main(*, mtp: bool = False, concurrent_pairs: list[tuple[str, str]] | None = None):
    _validate_client_settings(max_tokens=MAX_TOKENS)
    if not math.isfinite(STORE_WAIT_SECONDS) or STORE_WAIT_SECONDS < 0:
        raise ValueError("DSA_DP_L3_STORE_WAIT_SECONDS must be finite and non-negative")
    cases = {case.label: case for case in _load_cases()}
    if concurrent_pairs is None:
        if not LABELS or len(set(LABELS)) != len(LABELS) or any(label not in cases for label in LABELS):
            raise ValueError("DSA_DP_L3_LABELS must contain distinct labels from 5k,8k,40k,70k")
    elif not concurrent_pairs or any(
        len(pair) != 2 or any(label not in cases for label in pair) for pair in concurrent_pairs
    ):
        raise ValueError("concurrent pairs require one valid label for each source DP")
    run_id = f"dsa-dp-l3-{uuid.uuid4().hex}"
    result_path = RESULT_DIR.resolve() / f"{run_id}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    root = API_BASE.rstrip("/")
    if not root.endswith("/v1"):
        raise ValueError("DSA_DP_API_BASE must end in /v1")
    root = root[:-3]
    async with (
        AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0) as client,
        httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, headers={"Authorization": f"Bearer {API_KEY}"}) as http,
    ):
        model = await _resolve_model(client)
        print(
            f"[dsa-dp-l3] api={API_BASE} model={model!r} "
            f"cases={concurrent_pairs if concurrent_pairs is not None else LABELS} max_tokens={MAX_TOKENS} "
            f"ignore_eos=True mtp_expected={mtp} store_wait_seconds={STORE_WAIT_SECONDS} results={result_path}",
            flush=True,
        )

        async def run(phase, label, rank, cohort, hit):
            record = await _run_case(
                client,
                model=model,
                case=cases[label],
                dp_rank=rank,
                phase=phase,
                salt=f"{run_id}-{cohort}-{label}",
                expect_hit=hit,
                request_id=f"{run_id}-{phase}-dp{rank}-{label}",
                records=records,
                result_path=result_path,
                max_tokens=MAX_TOKENS,
            )
            if mtp:
                usage = record["usage"]
                cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                expected = max(0, (int(usage["prompt_tokens"]) - 1) // 128 - 1) * 128
                if (hit or cached) and cached != expected:
                    raise AssertionError(
                        f"{phase}/DP{rank}/{label}: expected MTP cached_tokens={expected} after one boundary drop, "
                        f"got {cached}; check MTP server config and complete L3 triple; evidence={result_path}"
                    )

        async def wait_store():
            print(f"[dsa-dp-l3] allowing {STORE_WAIT_SECONDS}s for write-behind", flush=True)
            await asyncio.sleep(STORE_WAIT_SECONDS)

        async def reset_local():
            response = await http.post(root + "/reset_prefix_cache", params={"reset_external": "false"})
            response.raise_for_status()
            print("[dsa-dp-l3] local caches reset on both DP engines; L3 preserved", flush=True)

        async def concurrent(phase, items, hit):
            gate = asyncio.Event()
            ready = 0

            async def ready_then_run(label, rank, cohort):
                nonlocal ready
                ready += 1
                if ready == len(items):
                    gate.set()
                await gate.wait()
                await run(phase, label, rank, cohort, hit)

            print(
                f"[dsa-dp-l3] concurrent phase={phase} routes={[(rank, label) for label, rank, _ in items]}", flush=True
            )
            results = await asyncio.gather(
                *(ready_then_run(*item) for item in items),
                return_exceptions=True,
            )
            errors = [str(value) for value in results if isinstance(value, BaseException)]
            if errors:
                raise AssertionError(f"concurrent requests failed: phase={phase}: {errors}; evidence={result_path}")

        if concurrent_pairs is not None:
            for index, pair in enumerate(concurrent_pairs):
                scenario = f"pair{index}"
                seeds = [(label, rank, f"{scenario}-origin{rank}") for rank, label in enumerate(pair)]
                swapped = [(label, 1 - rank, cohort) for label, rank, cohort in seeds]
                await concurrent(f"{scenario}-seed", seeds, False)
                await wait_store()
                # Each target DP has only seen the other salt, including equal-length pairs.
                await concurrent(f"{scenario}-cross-load", swapped, True)
                await concurrent(f"{scenario}-local-replay", swapped, True)
                await wait_store()
                await reset_local()
                await concurrent(f"{scenario}-reset-load", seeds, True)
        else:
            for label in LABELS:
                await run("seed", label, 0, "cross", False)
                await wait_store()
                # DP1 has never seen this salt: any hit must be external.
                await run("cross-load", label, 1, "cross", True)
                await run("local-replay", label, 1, "cross", True)
                await wait_store()
                await reset_local()
                await run("reset-load", label, 0, "cross", True)

                # A peer may finish uploading first; the controlled UT covers put races.
                await concurrent("concurrent-seed", [(label, rank, "race") for rank in (0, 1)], None)
                await wait_store()
                await reset_local()
                await run("concurrent-reset-load", label, 1, "race", True)
    print(
        f"[dsa-dp-l3] PASSED requests={len(records)}; "
        f"verify server nonzero L3 get/L2 restore and readable store logs; results={result_path}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtp", action="store_true", help="Validate the P6-A MTP prefix boundary (one block rollback)")
    parser.add_argument(
        "--concurrent-pair",
        type=_parse_pair,
        action="append",
        metavar="DP0_LABEL,DP1_LABEL",
        help="Concurrent seed, cross-DP load with swapped sources, local replay, and reset/load (8 requests per pair)",
    )
    args = parser.parse_args()
    asyncio.run(_main(mtp=args.mtp, concurrent_pairs=args.concurrent_pair))
