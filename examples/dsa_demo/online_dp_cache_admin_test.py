#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P6-B在线管理闭环：local reset、统一external epoch、暂停期间排队与恢复。

修改顶部用户配置后直接运行：python online_dp_cache_admin_test.py。
连接已启动的P6-A DP2服务，默认128输出tokens，无其他流量。
两个不同salt分别在DP0/DP1播种，external reset后分别必须冷命中，再跨域验证
新namespace共享，避免一域重新写入后掩盖另一域未切换的情况。共8请求。
不比较数值精度，不删除Mooncake对象；管理错误时停止，不自动重试reset。
默认增加暂停/排队期间的local及external reset和跨域读回，共9请求，结束后恢复运行。
客户端不会启动或停止服务器。只有LEAVE_PAUSED=True才故意留下暂停状态，
暂停仍占用NPU/内存；需要用户随后正常停止服务。保留旧命令行开关兼容用法。
"""

from __future__ import annotations

import argparse
import asyncio
import math
import uuid
from pathlib import Path

import httpx
from online_dp_foundation_test import _load_cases
from online_dp_prefix_test import _run_case
from openai import AsyncOpenAI

# =========================
# 用户配置：不需要另外设置DSA_DP_*环境变量
# =========================
API_BASE = "http://127.0.0.1:8035/v1"
API_KEY = "EMPTY"
MODEL_NAME = "glm-5.2-dsa"  # 必须与serve脚本的SERVED_MODEL_NAME一致
MAX_TOKENS = 128  # 请求固定ignore_eos=True
REQUEST_TIMEOUT_SECONDS = 900.0
PAUSED_RESET = True  # 本轮九请求扩展；False为之前的八请求
LEAVE_PAUSED = False  # True只暂停，不关服；False测完恢复运行
RESULT_DIR = Path(__file__).resolve().parent / "results/dp_admin"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paused-reset", action=argparse.BooleanOptionalAction, default=PAUSED_RESET)
    parser.add_argument("--leave-paused", action=argparse.BooleanOptionalAction, default=LEAVE_PAUSED)
    return parser.parse_args(argv)


async def _resolve_model(client):
    # Read this script's explicit settings, not another client's environment defaults.
    models = [item.id for item in (await client.models.list()).data]
    if MODEL_NAME not in models:
        raise RuntimeError(f"MODEL_NAME={MODEL_NAME!r} is not served; available={models}")
    return MODEL_NAME


async def _main(*, paused_reset: bool | None = None, leave_paused: bool | None = None):
    paused_reset = PAUSED_RESET if paused_reset is None else paused_reset
    leave_paused = LEAVE_PAUSED if leave_paused is None else leave_paused
    if MAX_TOKENS < 2 or not math.isfinite(REQUEST_TIMEOUT_SECONDS) or REQUEST_TIMEOUT_SECONDS <= 0:
        raise ValueError("MAX_TOKENS must be >=2 and REQUEST_TIMEOUT_SECONDS finite and positive")
    if not API_BASE.rstrip("/").endswith("/v1"):
        raise ValueError("API_BASE must end in /v1")
    if type(paused_reset) is not bool or type(leave_paused) is not bool:
        raise ValueError("PAUSED_RESET and LEAVE_PAUSED must be True or False")
    case = next(case for case in _load_cases() if case.label == "40k")
    root = API_BASE.rstrip("/")[:-3]
    run_id = f"dsa-dp-admin-{uuid.uuid4().hex}"
    result_path = Path(RESULT_DIR) / f"{run_id}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    async with (
        AsyncOpenAI(api_key=API_KEY, base_url=API_BASE, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0) as client,
        httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, headers={"Authorization": f"Bearer {API_KEY}"}) as http,
    ):
        model = await _resolve_model(client)
        print(
            f"[dsa-dp-admin] api={API_BASE} model={model} max_tokens={MAX_TOKENS} "
            f"paused_reset={paused_reset} leave_paused={leave_paused} results={result_path}", flush=True,
        )

        async def run(phase, rank, source, hit):
            await _run_case(
                client,
                model=model,
                case=case,
                dp_rank=rank,
                phase=phase,
                salt=f"{run_id}-source{source}",
                expect_hit=hit,
                request_id=f"{run_id}-{phase}-dp{rank}",
                records=records,
                result_path=result_path,
                max_tokens=MAX_TOKENS,
            )

        async def reset(external, *, expect_paused=False):
            print(f"[dsa-dp-admin] reset_external={external}; waiting for all-DP drain/commit/resume", flush=True)
            response = await http.post(root + "/reset_prefix_cache", params={"reset_external": str(external).lower()})
            response.raise_for_status()
            state = await http.get(root + "/is_paused")
            state.raise_for_status()
            if state.json().get("is_paused") is not expect_paused:
                raise AssertionError(f"reset did not preserve the deployment pause state: expected={expect_paused}")

        await run("seed-a", 0, 0, False)
        seed_b = asyncio.create_task(run("seed-b", 1, 1, False))
        # 尝试在请求在途时发起reset；是否已admit以服务器pause日志为准。
        await asyncio.sleep(0.5)
        try:
            await reset(False)
            await seed_b
        except BaseException:
            seed_b.cancel()
            await asyncio.gather(seed_b, return_exceptions=True)
            raise
        await run("local-reset-load", 1, 0, True)
        await reset(True)
        await run("new-epoch-cold-a", 0, 0, False)
        await run("new-epoch-cold-b", 1, 1, False)
        # local reset也是新epoch写入排空点；两个域均必须从共享新namespace读取。
        await reset(False)
        await run("new-epoch-cross-a", 1, 0, True)
        await run("new-epoch-cross-b", 0, 1, True)

        paused = await http.post(root + "/pause", params={"mode": "wait", "clear_cache": "false"})
        paused.raise_for_status()
        queued = asyncio.create_task(run("queued-while-paused", 0, 0, not paused_reset))
        try:
            await asyncio.sleep(0.5)
            state = await http.get(root + "/is_paused")
            state.raise_for_status()
            if state.json().get("is_paused") is not True or queued.done():
                raise AssertionError("a request completed while the deployment should be paused")
            if paused_reset:
                for external in (False, True):
                    await reset(external, expect_paused=True)
                    if queued.done():
                        raise AssertionError("queued request completed during a paused reset")
                print("[dsa-dp-admin] paused local/external resets preserved pause and queued request", flush=True)
        finally:
            # 只恢复本测试显式创建的手工pause；事务fatal时服务端会拒绝恢复。
            try:
                resumed = await http.post(root + "/resume")
                resumed.raise_for_status()
            except BaseException:
                queued.cancel()
                await asyncio.gather(queued, return_exceptions=True)
                raise
            await queued
        if paused_reset:
            await reset(False)
            await run("paused-reset-cross-load", 1, 0, True)
        if leave_paused:
            paused = await http.post(root + "/pause", params={"mode": "wait", "clear_cache": "false"})
            paused.raise_for_status()
            state = await http.get(root + "/is_paused")
            state.raise_for_status()
            if state.json().get("is_paused") is not True:
                raise AssertionError("deployment did not stay paused for shutdown verification")
            print("[dsa-dp-admin] LEFT PAUSED intentionally; stop this test server to verify shutdown", flush=True)
        else:
            print("[dsa-dp-admin] service remains running; this client does not stop the server", flush=True)
    print(
        f"[dsa-dp-admin] PASSED requests={len(records)}; verify all-DP commit fingerprints; results={result_path}",
        flush=True,
    )


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(_main(paused_reset=args.paused_reset, leave_paused=args.leave_paused))
