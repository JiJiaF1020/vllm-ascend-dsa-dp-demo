#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared deterministic prompt for baseline external-L3 cold/warm tests."""

from __future__ import annotations


CONTEXT_PARAGRAPH = (
    "项目组正在验证一套长上下文推理系统。系统需要保存已经计算完成的上下文缓存，"
    "并在后续收到相同前缀时直接恢复这些缓存。为了确保测试结果可复核，每一段文字"
    "都描述相同的约束：数据必须完整、顺序必须稳定、不同请求之间不得发生串扰。"
)


def build_prompt(repetitions: int = 48) -> str:
    context = "\n".join(f"第{index + 1}段：{CONTEXT_PARAGRAPH}" for index in range(repetitions))
    return (
        f"{context}\n\n"
        "请根据以上内容，只回答下面问题的答案，不要解释："
        "系统要求不同请求之间不得发生什么？"
    )
