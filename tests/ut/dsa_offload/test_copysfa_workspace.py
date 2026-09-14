# SPDX-License-Identifier: Apache-2.0
"""CPU workspace lifetime self-check; does not simulate NPU graph capture."""

import subprocess
import tempfile
from pathlib import Path


def test_copysfa_workspace_cache():
    source = Path(__file__).with_name("copysfa_workspace_host.cpp")
    with tempfile.TemporaryDirectory(prefix="copysfa-workspace-") as directory:
        binary = Path(directory) / "check"
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "-pthread", str(source), "-o", str(binary)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=60)
        assert "COPYSFA_WORKSPACE_CACHE_OK" in output.stdout
