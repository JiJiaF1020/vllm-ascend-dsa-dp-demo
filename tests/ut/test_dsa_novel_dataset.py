# SPDX-License-Identifier: Apache-2.0
"""Dataset/import checks without loading vLLM or contacting a server."""

import ast
import importlib
import runpy
from collections.abc import Sequence
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parents[2] / "examples/dsa_demo"


def test_dataset_names_and_composition():
    data = runpy.run_path(str(DEMO / "novel_dataset.py"))
    for label in ("5k", "10k", "23k", "40k", "64k", "104k"):
        value = data[f"chinese_{label}"]
        assert isinstance(value, list) and len(value) == 1
        assert isinstance(value[0], str) and value[0].strip()
    assert data["chinese_104k"] == [data["chinese_40k"][0] + data["chinese_64k"][0]]


def test_all_direct_dataset_imports_exist():
    data = runpy.run_path(str(DEMO / "novel_dataset.py"))
    for path in DEMO.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module == "novel_dataset":
                for alias in node.names:
                    assert alias.name in data, (path, alias.name)


@pytest.mark.parametrize("script,loader", [
    ("online_dp_foundation_test.py", "_load_cases"),
    ("online_l2_l3_closeout_test.py", "load_prompt_cases"),
    ("online_replay_determinism_test.py", "load_cases"),
    ("offline_dp_closeout_test.py", "_load_cases"),
])
def test_real_dataset_loaders(monkeypatch, script, loader):
    monkeypatch.syspath_prepend(str(DEMO))
    path = DEMO / script
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (loader, "_first_prompt")]
    ns = {"importlib": importlib, "Sequence": Sequence, "PromptCase": lambda label, prompt: (label, prompt)}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PROMPT_LABELS" for t in node.targets
        ):
            ns["PROMPT_LABELS"] = ast.literal_eval(node.value)
    future = ast.parse("from __future__ import annotations").body
    exec(compile(ast.Module(body=future + nodes, type_ignores=[]), str(path), "exec"), ns)
    if script == "offline_dp_closeout_test.py":
        cases = ns[loader](ns["PROMPT_LABELS"])
    else:
        cases = ns[loader]()
    assert cases and all(prompt for _, prompt in cases)
    assert {label for label, _ in cases} <= {"5k", "10k", "23k", "40k", "64k", "104k"}
