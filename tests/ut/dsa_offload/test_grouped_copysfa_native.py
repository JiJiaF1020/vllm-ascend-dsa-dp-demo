# SPDX-License-Identifier: Apache-2.0
"""Host-only copy/address checks; not a substitute for A5 MIX kernel tests."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("packaged", [False, True])
def test_grouped_copysfa_tiling_include_in_source_and_flat_package(tmp_path, packaged):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required")
    attention = Path(__file__).resolve().parents[3] / "csrc/attention"
    names = ("vllm_a5_fused_copy_sfa_c8", "vllm_a5_fused_copy_sfa_group4_c8")
    for name in names:
        target = tmp_path / name
        if not packaged:
            target /= "op_kernel"
        target.mkdir(parents=True)
        shutil.copy2(attention / name / "op_kernel" / f"{name}_tiling.h", target)
    include_dir = tmp_path / names[1]
    if not packaged:
        include_dir /= "op_kernel"
    source = '#include "vllm_a5_fused_copy_sfa_group4_c8_tiling.h"\n'
    source += "#include <type_traits>\nstatic_assert(std::is_same_v<"
    source += "VllmA5FusedCopySfaGroup4C8TilingData, VllmA5FusedCopySfaC8TilingData>);\n"
    result = subprocess.run(
        [compiler, "-std=c++17", "-fsyntax-only", "-I", str(include_dir), "-x", "c++", "-"],
        input=source,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_grouped_copysfa_copy_bytes_and_virtual_table_width(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required")
    here = Path(__file__).parent
    kernel = here.parents[2] / "csrc/attention/vllm_a5_fused_copy_sfa_group4_c8/op_kernel"
    binary = tmp_path / "grouped-copysfa-copy"
    result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wno-sign-compare",
            "-Werror",
            "-I",
            str(here / "grouped_copy_host_stubs"),
            "-I",
            str(kernel),
            str(here / "grouped_copysfa_copy_host.cpp"),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GROUP4_COPYSFA_COPY_HOST_OK" in result.stdout
