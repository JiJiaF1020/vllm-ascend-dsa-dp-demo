# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake 独立探针的布局与角色测试，不加载模型或连接后端服务。"""

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture
def probe_module(monkeypatch):
    # Load the actual probe by file path, independent of cwd or another "examples" package.
    path = Path(__file__).resolve().parents[3] / "examples/dsa_demo/L3_cache_sys_api_test/test_mooncake_tp.py"
    spec = importlib.util.spec_from_file_location("_dsa_mooncake_tp_probe_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_mooncake_probe_glm52_payload_matches_dsa_worker_contract(probe_module) -> None:
    sizes, groups = probe_module._build_payload_layout("dsa_a5_glm52", payload_bytes=1, num_buffers=1)
    assert len(sizes) == 120
    assert sum(sizes) == 6_904_320
    assert [(name, len(indices), sum(sizes[index] for index in indices)) for name, indices in groups] == [
        ("indexer", 42, 354_816),
        ("mla", 78, 6_549_504),
    ]


def test_mooncake_probe_glm52_mtp_payload_matches_three_component_bundle(probe_module) -> None:
    sizes, groups = probe_module._build_payload_layout("dsa_a5_glm52_mtp", payload_bytes=1, num_buffers=1)
    assert len(sizes) == 123
    assert sum(sizes) == 7_084_544
    assert [(name, len(indices), sum(sizes[index] for index in indices)) for name, indices in groups] == [
        ("indexer", 42, 354_816),
        ("mla", 78, 6_549_504),
        ("mtp", 3, 180_224),
    ]


def test_mooncake_probe_supports_independent_producer_and_consumer(probe_module) -> None:
    assert probe_module._resolve_io_roles(rank=0, world_size=2, producer_rank=1, consumer_rank=0) == (False, True, 1)
    assert probe_module._resolve_io_roles(rank=1, world_size=2, producer_rank=1, consumer_rank=0) == (True, False, 1)
