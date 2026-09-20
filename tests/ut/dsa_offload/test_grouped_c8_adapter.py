# SPDX-License-Identifier: Apache-2.0
"""Compile the legacy and grouped copy adapters together without CANN."""

import shutil
import subprocess
import unittest
from pathlib import Path

import torch


class TestGroupedC8Adapter(unittest.TestCase):
    def test_legacy_and_grouped_symbols_compile_together(self):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required")
        root = Path(__file__).resolve().parents[3]
        torch_include = Path(torch.__file__).resolve().parent / "include"
        # Keep the real ATen types and TORCH_CHECK expressions. Only the NPU
        # launch is stubbed; this checks C++ names/signatures, not kernel execution.
        source = r"""
#include <ATen/ATen.h>
#include <type_traits>
#define EXEC_NPU_CMD(...) static_cast<void>(0)
#include "vllm_a5_kvcache_scatter_copy_c8/vllm_a5_kvcache_scatter_copy_c8_torch_adpt.h"
#include "vllm_a5_kvcache_scatter_copy_group4_c8/vllm_a5_kvcache_scatter_copy_group4_c8_torch_adpt.h"

using ScatterCopy = void (*)(at::Tensor, const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&,
                            const at::Tensor&, const at::Tensor&);
static_assert(std::is_same_v<
    decltype(&vllm_ascend::npu_dsa_a5_kvcache_scatter_copy_c8_out), ScatterCopy>);
static_assert(std::is_same_v<
    decltype(&vllm_ascend::npu_dsa_a5_kvcache_scatter_copy_group4_c8_out), ScatterCopy>);
"""
        result = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fsyntax-only",
                "-isystem",
                str(torch_include),
                "-I",
                str(root / "csrc/attention"),
                "-x",
                "c++",
                "-",
            ],
            input=source,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
