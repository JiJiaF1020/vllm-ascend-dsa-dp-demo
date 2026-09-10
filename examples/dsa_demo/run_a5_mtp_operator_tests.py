#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A5 MTP 算子升级门禁：直接 python 运行，不加载模型或启动 Mooncake。

默认先运行 CPU 框架 UT，再运行单卡 packed-C8 算子测试。开关直接在下方
修改。日志保存在脚本旁的 results；若把脚本复制到 draft，默认从已安装的
vllm_ascend 找源码仓库，也可以手工填写 PROJECT_ROOT。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# ------------------------- 常规门禁 -------------------------
PROJECT_ROOT = ""  # 留空自动定位；或填写 /home/.../vllm-ascend-v0.23。
RUN_FRAMEWORK_UT = True
RUN_NATIVE_OPS = True  # 需要 A5、新编译安装的 CANN 算子与 vllm_ascend_C。

# ------------------------- 可选破坏性负例 -------------------------
# 默认不跑。仅在允许设备侧 assertion 的独占测试环境打开。
# 每个负例单独启动进程，禁用 core 文件；不与正常算子测试共用 NPU context。
RUN_INVALID_PLAN_PROBES = False
INVALID_PLAN_TIMEOUT_SECONDS = 90

NATIVE_TEST = "tests/e2e/nightly/single_node/ops/singlecard_ops/test_dsa_a5_c8_ops.py"


def project_root() -> Path:
    if PROJECT_ROOT:
        root = Path(PROJECT_ROOT).expanduser().resolve()
    else:
        root = Path(__file__).resolve().parents[2]
        if not (root / NATIVE_TEST).is_file():
            spec = importlib.util.find_spec("vllm_ascend")
            if spec is None or spec.origin is None:
                raise RuntimeError("找不到 vllm_ascend 源码，请填写 PROJECT_ROOT")
            root = Path(spec.origin).resolve().parent.parent
    if not (root / NATIVE_TEST).is_file() or not (root / "tests/ut/dsa_offload").is_dir():
        raise RuntimeError(f"测试源码不完整：{root}；请同步项目或填写 PROJECT_ROOT")
    return root


def invalid_plan_probe(case: str) -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import torch
    import torch_npu

    from vllm_ascend.dsa_offload.ops import require_dsa_offload_ops

    require_dsa_offload_ops(packed_c8=True)
    device = torch.device("npu:0")
    hbm = torch.zeros((1, 128, 1, 656), dtype=torch.int8, device=device)
    dram = torch_npu.empty_with_swapped_memory(hbm.shape, dtype=torch.int8, device=device)
    table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    src = torch.zeros((1, 1, 16384), dtype=torch.int32, device=device)
    dst = torch.zeros_like(src)
    count = torch.tensor([-1 if case == "count" else 1], dtype=torch.int32, device=device)
    if case == "pair":
        src[0, 0, 0] = -1
    torch.npu.synchronize()
    print(f"[a5-mtp-test] INVALID_PLAN_READY case={case}", flush=True)
    # Device error details and the nonzero child exit are verified by the parent.
    torch.ops._C_ascend.npu_dsa_a5_kvcache_scatter_copy_c8_out(hbm, dram, table, table, src, dst, count)
    torch.npu.synchronize()
    print("[a5-mtp-test] INVALID_PLAN_WAS_ACCEPTED", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invalid-plan-probe", choices=("count", "pair"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.invalid_plan_probe:
        invalid_plan_probe(args.invalid_plan_probe)
        return 0

    if not any((RUN_FRAMEWORK_UT, RUN_NATIVE_OPS, RUN_INVALID_PLAN_PROBES)):
        raise ValueError("没有启用任何测试，请检查脚本顶部开关")
    root = project_root()
    result_dir = (
        Path(__file__).resolve().parent
        / "results"
        / "a5_mtp_ops"
        / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    )
    result_dir.mkdir(parents=True, exist_ok=False)
    stages = []
    if RUN_FRAMEWORK_UT:
        stages.append(("framework", ["-m", "pytest", "tests/ut/dsa_offload", "-q", "--tb=short"]))
    if RUN_NATIVE_OPS:
        stages.append(("native", ["-m", "pytest", NATIVE_TEST, "-vv", "-s", "--tb=short"]))
    print(f"[a5-mtp-test] project={root} python={sys.executable} results={result_dir}", flush=True)
    outcomes = []
    try:
        for name, command in stages:
            log_path = result_dir / f"{name}.log"
            xml_path = result_dir / f"{name}.xml"
            print(f"[a5-mtp-test] stage={name} log={log_path}", flush=True)
            with (
                log_path.open("w", encoding="utf-8") as log,
                subprocess.Popen(
                    [sys.executable, "-u", *command, f"--junitxml={xml_path}"],
                    cwd=root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                ) as process,
            ):
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            outcome = {"stage": name, "exit_code": code, "log": str(log_path)}
            outcomes.append(outcome)
            if code:
                print(f"[a5-mtp-test] FAILED stage={name}; 后续测试不执行", flush=True)
                return 1
            cases = list(ET.parse(xml_path).getroot().iter("testcase"))
            skipped = sum(case.find("skipped") is not None for case in cases)
            outcome.update(tests=len(cases), skipped=skipped)
            if not cases or (name == "native" and skipped):
                print(f"[a5-mtp-test] INCOMPLETE stage={name} tests={len(cases)} skipped={skipped}", flush=True)
                return 1
        if RUN_INVALID_PLAN_PROBES:
            for case, message in (("count", "reported an invalid row"), ("pair", "emitted an invalid copy pair")):
                log_path = result_dir / f"invalid-{case}.log"
                try:
                    with log_path.open("w", encoding="utf-8") as log:
                        result = subprocess.run(
                            [sys.executable, "-u", str(Path(__file__).resolve()), "--invalid-plan-probe", case],
                            cwd=root,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=INVALID_PLAN_TIMEOUT_SECONDS,
                        )
                except subprocess.TimeoutExpired:
                    outcomes.append({"stage": f"invalid-{case}", "timed_out": True, "log": str(log_path)})
                    print(f"[a5-mtp-test] FAILED invalid-{case} timeout; inspect {log_path}", flush=True)
                    return 1
                content = log_path.read_text(encoding="utf-8", errors="replace")
                rejected = (
                    result.returncode != 0
                    and "INVALID_PLAN_READY" in content
                    and message in content
                    and "INVALID_PLAN_WAS_ACCEPTED" not in content
                )
                outcomes.append(
                    {
                        "stage": f"invalid-{case}",
                        "exit_code": result.returncode,
                        "expected_rejection": rejected,
                        "log": str(log_path),
                    }
                )
                if not rejected:
                    print(f"[a5-mtp-test] FAILED invalid-{case}; inspect {log_path}", flush=True)
                    return 1
        print(f"[a5-mtp-test] PASSED enabled stages; results={result_dir}", flush=True)
        return 0
    finally:
        (result_dir / "summary.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
