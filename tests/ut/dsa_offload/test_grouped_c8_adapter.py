# SPDX-License-Identifier: Apache-2.0
"""Compile the legacy and grouped copy adapters together without CANN."""

import shutil
import subprocess
import unittest
from pathlib import Path

import torch


class TestGroupedC8Adapter(unittest.TestCase):
    def test_fused_adapter_signatures_and_launcher_match(self):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required")
        root = Path(__file__).resolve().parents[3] / "csrc/attention"
        old = (root / "vllm_a5_fused_copy_sfa_c8/vllm_a5_fused_copy_sfa_c8_torch_adpt.h").read_text()
        new = (root / "vllm_a5_fused_copy_sfa_group4_c8/vllm_a5_fused_copy_sfa_group4_c8_torch_adpt.h").read_text()
        # The new launch macro must retain the known stream ordering/keepalive
        # implementation. Only its macro name differs; no CANN is executed here.
        macros = []
        bodies = []
        for source, prefix in ((old, "COPYSFA"), (new, "GROUP4_COPYSFA")):
            macro_name = f"EXEC_DSA_{prefix}_ORDERED_IMPL"
            macro = source[source.index(f"#define {macro_name}") : source.index("inline void check_")]
            macros.append(macro.replace(macro_name, "ORDERED_LAUNCH").strip())
            body = source[source.index("inline void check_") : source.index(f"#undef {macro_name}")]
            # Leave the launch call and all arguments intact: a stand-in checks
            # C++ parsing and tensor references without needing torch_npu headers.
            bodies.append(f"#define {macro_name}(...) static_cast<void>(keepalive)\n" + body)
        self.assertEqual(macros[0], macros[1])
        source = "#include <ATen/ATen.h>\n#include <cmath>\n#include <limits>\n#include <tuple>\n"
        source += "namespace vllm_ascend {\n" + "\n".join(bodies) + "\n}\n"
        source += "auto legacy = &vllm_ascend::npu_dsa_a5_fused_copy_sfa_c8_out;\n"
        source += "auto grouped = &vllm_ascend::npu_dsa_a5_fused_copy_sfa_group4_c8_out;\n"
        result = subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-fsyntax-only",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-isystem",
                str(Path(torch.__file__).resolve().parent / "include"),
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
