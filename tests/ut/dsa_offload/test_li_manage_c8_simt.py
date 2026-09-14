# SPDX-License-Identifier: Apache-2.0
"""Portable SIMT/scan contracts; does not simulate Ascend pipeline ordering."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestLIMSIMT(unittest.TestCase):
    def test_host_arithmetic(self):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required")
        kernel = Path(__file__).resolve().parents[3] / "csrc/attention/vllm_a5_li_manage_c8/op_kernel"
        for name, marker in (
            ("li_manage_c8_simt_host.cpp", "SIMT_POSTPROCESS_HOST_OK"),
            ("li_manage_c8_victim_scan_host.cpp", "LI_MTP_VICTIM_SCAN_HOST_OK"),
        ):
            with self.subTest(source=name), tempfile.TemporaryDirectory(prefix="lim-host-") as directory:
                executable = Path(directory) / "test"
                result = subprocess.run(
                    [
                        compiler,
                        "-std=c++17",
                        "-O2",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        "-I",
                        str(kernel),
                        str(Path(__file__).with_name(name)),
                        "-o",
                        str(executable),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(marker, result.stdout)

    def test_both_paths_seed_from_current_final_length(self):
        kernel = Path(__file__).resolve().parents[3] / "csrc/attention/vllm_a5_li_manage_c8/op_kernel"
        for name in ("vllm_a5_li_manage_c8_fast_union.h", "vllm_a5_li_manage_c8_manager.h"):
            with self.subTest(source=name):
                source = (kernel / name).read_text()
                calls = re.findall(r"HashVictimScanSeed\s*\(([^)]*)\)", source)
                self.assertEqual([arg.strip() for arg in calls], ["finalLen"])


if __name__ == "__main__":
    unittest.main()
