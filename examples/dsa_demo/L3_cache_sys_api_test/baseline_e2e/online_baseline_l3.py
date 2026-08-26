#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Send two identical completion requests to a DSA-disabled L3 service."""

from __future__ import annotations

import time

from baseline_prompt import build_prompt
from openai import OpenAI

# =========================
# User configuration
# =========================
API_BASE = "http://127.0.0.1:8035/v1"
API_KEY = "EMPTY"
MODEL_NAME = "glm-5.2-baseline-l3"
PROMPT_REPETITIONS = 48
MAX_TOKENS = 1
REPLAY_DELAY_SECONDS = 10.0


def _token_ids(response) -> list[int] | None:
    payload = response.model_dump()
    value = payload["choices"][0].get("token_ids")
    return None if value is None else [int(token) for token in value]


def main() -> None:
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    available = [model.id for model in client.models.list().data]
    if MODEL_NAME not in available:
        raise RuntimeError(f"model {MODEL_NAME!r} is not served; available={available}")

    prompt = build_prompt(PROMPT_REPETITIONS)
    records: list[dict[str, object]] = []
    for round_index in range(2):
        if round_index:
            time.sleep(REPLAY_DELAY_SECONDS)
        start = time.perf_counter()
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            stream=False,
            extra_body={"return_token_ids": True},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        usage = None if response.usage is None else response.usage.model_dump()
        record = {
            "round": round_index + 1,
            "token_ids": _token_ids(response),
            "text": choice.text,
            "finish_reason": choice.finish_reason,
            "usage": usage,
            "latency_ms": round(elapsed_ms, 3),
        }
        records.append(record)
        print(f"[baseline-l3-online] result={record}")

    first_usage = records[0]["usage"]
    second_usage = records[1]["usage"]
    if not isinstance(first_usage, dict) or not isinstance(second_usage, dict):
        raise AssertionError(f"server did not return usage for both rounds: {records}")
    first_prompt_tokens = int(first_usage["prompt_tokens"])
    second_prompt_tokens = int(second_usage["prompt_tokens"])
    if first_prompt_tokens < 256 or first_prompt_tokens != second_prompt_tokens:
        raise AssertionError(
            "unexpected prompt token counts: "
            f"cold={first_prompt_tokens} warm={second_prompt_tokens}"
        )

    first_ids = records[0]["token_ids"]
    second_ids = records[1]["token_ids"]
    if first_ids is not None and second_ids is not None:
        if first_ids != second_ids:
            raise AssertionError(f"cold/warm token IDs differ: {records}")
    elif records[0]["text"] != records[1]["text"]:
        raise AssertionError(f"cold/warm text differs: {records}")
    print(
        "[baseline-l3-online] OUTPUT CONSISTENCY PASSED; L3 acceptance still "
        "requires successful put/get logs and a nonzero external hit"
    )


if __name__ == "__main__":
    main()
