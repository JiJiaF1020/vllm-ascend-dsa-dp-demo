#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""通过 OpenAI Chat Completions API 流式验证 DSA 在线服务。"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from openai import OpenAI

# =========================
# 用户配置
# =========================

API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "glm-5.1-dsa"

SYSTEM_PROMPT = "你是一个严谨、简洁的中文助手。"
LONG_CONTEXT_FILE: str | None = None

# 设为非 None 时，客户端改用 /v1/completions 做一次纯文本续写，
# 不套 chat template，也不进入下方的多轮对话循环。例如：
#
# from novel_dataset import chinese_40k
# CONTINUATION_PROMPT = chinese_40k[0]
CONTINUATION_PROMPT: str | None = None

MAX_TOKENS = 512
TEMPERATURE = 0.0
PRINT_REASONING = True
L3_PREFIX_REPLAY_ROUNDS = 1  # P1-R 验收时设为 3


def build_initial_messages() -> list[dict[str, str]]:
    system_content = SYSTEM_PROMPT
    if LONG_CONTEXT_FILE is not None:
        context_path = Path(LONG_CONTEXT_FILE).expanduser()
        context = context_path.read_text(encoding="utf-8")
        system_content += (
            "\n\n以下是本轮对话需要参考的长上下文：\n"
            f"{context}"
        )
        print(
            "[dsa-client] loaded context "
            f"path={context_path} chars={len(context)}"
        )
    return [{"role": "system", "content": system_content}]


def stream_assistant(
    client: OpenAI,
    messages: list[dict[str, str]],
) -> str:
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
    )

    content_parts: list[str] = []
    reasoning_started = False
    content_started = False
    final_usage = None

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            final_usage = chunk.usage
        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta
        reasoning = (
            getattr(delta, "reasoning", None)
            or getattr(delta, "reasoning_content", None)
        )
        content = getattr(delta, "content", None)

        if reasoning and PRINT_REASONING:
            if not reasoning_started:
                reasoning_started = True
                print("助手[思考]> ", end="", flush=True)
            print(reasoning, end="", flush=True)

        if content:
            if not content_started:
                if reasoning_started:
                    print()
                print("助手> ", end="", flush=True)
                content_started = True
            print(content, end="", flush=True)
            content_parts.append(content)

    if not content_started:
        print("助手> [未返回 content]", end="")
    print()
    if final_usage is not None:
        print(
            "[usage] "
            f"prompt={final_usage.prompt_tokens} "
            f"completion={final_usage.completion_tokens} "
            f"total={final_usage.total_tokens}"
        )
    return "".join(content_parts)


def stream_continuation(
    client: OpenAI,
    prompt: str,
) -> str:
    """通过 Completions API 流式续写原始文本。"""

    stream = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
    )

    text_parts: list[str] = []
    final_usage = None
    print("续写> ", end="", flush=True)
    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            final_usage = chunk.usage
        if not chunk.choices:
            continue
        text = chunk.choices[0].text
        if text:
            print(text, end="", flush=True)
            text_parts.append(text)

    print()
    if final_usage is not None:
        print(
            "[usage] "
            f"prompt={final_usage.prompt_tokens} "
            f"completion={final_usage.completion_tokens} "
            f"total={final_usage.total_tokens}"
        )
    return "".join(text_parts)


def _post_dev_api(path: str, *, timeout: float = 300.0) -> None:
    server_root = API_BASE.removesuffix("/v1").rstrip("/")
    request = urllib.request.Request(
        f"{server_root}{path}",
        method="POST",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(
                f"dev API failed: path={path!r}, status={response.status}"
            )


def drain_stores_and_reset_local_prefix_cache() -> None:
    """Drain async stores, clear local prefix, then resume generation."""

    _post_dev_api("/pause?mode=wait&clear_cache=false")
    try:
        _post_dev_api(
            "/reset_prefix_cache?reset_external=false",
            timeout=30,
        )
    finally:
        _post_dev_api("/resume", timeout=30)
    print(
        "[dsa-client] pending L3 stores drained; local HBM/L2 prefix reset; "
        "Mooncake objects retained"
    )


def main() -> None:
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    available_models = [model.id for model in client.models.list().data]
    if MODEL_NAME not in available_models:
        raise RuntimeError(
            f"model {MODEL_NAME!r} is not served; "
            f"available={available_models}"
        )

    if CONTINUATION_PROMPT is not None:
        print(
            "[dsa-client] raw continuation "
            f"chars={len(CONTINUATION_PROMPT)}"
        )
        if L3_PREFIX_REPLAY_ROUNDS not in (1, 3):
            raise ValueError(
                "L3_PREFIX_REPLAY_ROUNDS must be 1 or 3"
            )
        outputs: list[str] = []
        for round_index in range(L3_PREFIX_REPLAY_ROUNDS):
            print(
                f"[dsa-client] continuation round "
                f"{round_index + 1}/{L3_PREFIX_REPLAY_ROUNDS}"
            )
            outputs.append(
                stream_continuation(client, CONTINUATION_PROMPT)
            )
            if L3_PREFIX_REPLAY_ROUNDS == 3 and round_index == 1:
                drain_stores_and_reset_local_prefix_cache()
        if L3_PREFIX_REPLAY_ROUNDS == 3:
            if outputs[1] != outputs[2]:
                raise AssertionError(
                    "local-L2/L3 online replay text differs: "
                    f"local_l2={outputs[1]!r}, l3={outputs[2]!r}"
                )
            if outputs[0] != outputs[1]:
                print(
                    "[dsa-client] cold/prefix text differs; recorded as "
                    "full-prefill versus prefix-suffix numerical path "
                    "difference"
                )
            print(
                "[dsa-client] local-L2/L3 online replay consistency passed"
            )
        return

    messages = build_initial_messages()
    print("[dsa-client] 服务已连接。输入 /clear 清空历史，/quit 退出。")

    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[dsa-client] bye")
            return

        if not user_text:
            continue
        if user_text in {"/quit", "/exit"}:
            print("[dsa-client] bye")
            return
        if user_text == "/clear":
            messages = build_initial_messages()
            print("[dsa-client] 对话历史已清空")
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            assistant_text = stream_assistant(client, messages)
        except Exception:
            messages.pop()
            raise
        messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )


if __name__ == "__main__":
    main()
