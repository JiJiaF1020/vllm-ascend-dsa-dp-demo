#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""单机A5 DP2/TP4离线SPMD收口，复用官方每DP一个LLM客户端进程的方式。

spawn前统一hash seed，子进程首次导入vLLM前设置VLLM_DP_*。不把DP=2直接
传给两个独立LLM，否则它们可能各自创建部署。阶段间Barrier只协调测试，
不参与框架forward；查询已有管理status等待模型/IO排空，再清各域本地缓存。
可选MTP3/L3，验证cold、local、清本地后交换域读取；不比较数值精度。
数据来自novel_dataset；本文件独立构造LLM配置，不导入其他示例的配置或全局变量。
加载前检查文件/依赖/Master，加载后检查真实EngineCore配置和L3状态，再发请求。

修改下方“用户配置”后直接运行：python offline_dp_closeout_test.py。
默认即本轮Graph+MTP3+L3组合，不需要MODEL_PATH或DSA_DP_*环境变量。
旧命令行参数仍可覆盖配置，用于兼容之前的执行命令，不是必需入口。
本脚本只做正常推理/缓存回归，不安装退出探针、不启用pickle fallback。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import socket
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlsplit

# =========================
# 用户配置：日常只改这一段
# =========================
MODEL_PATH = "/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4"
RUN_MODE = "graph"  # eager / graph
VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"  # 本用例固定单机DP2/TP4，需要8张卡
ENABLE_MTP = True  # 固定MTP3，proposer eager
ENABLE_L3 = True  # 同时开启write-behind与prefix loadback；Master需已运行
# 常规小规模回归：六请求；完整矩阵用("5k", "8k", "40k", "70k")。
PROMPT_LABELS = ("8k", "40k")
MAX_TOKENS = 128  # 验收固定ignore_eos=True，跨越decode满块边界
MAX_NUM_SEQS = 2  # 每个DP域的上限，不是全机总数
MAX_MODEL_LEN = 100000
MAX_NUM_BATCHED_TOKENS = 8192
GPU_MEMORY_UTILIZATION = 0.90
# 对齐本次日志里已完成冷算的容量配置及在线配置，减少内存占用变量。
DSA_INDEXER_MLA_BLOCK_RATIO = 2
DSA_HOT_CPU_BLOCK_MULTIPLE = 1.0
MOONCAKE_CONFIG_PATH = (
    Path(__file__).resolve().parent / "L3_cache_sys_api_test/baseline_e2e/configs/mooncake-dsa-tcp.json"
)
MOONCAKE_GLOBAL_SEGMENT_SIZE = "4GB"  # 每个worker贡献4GB，8卡共32GB
MOONCAKE_TCP_SEGMENT_PORT_BASE = 45000
PROCESS_TIMEOUT_SECONDS = 1800
DRAIN_TIMEOUT_SECONDS = 300
RESULT_DIR = Path(__file__).resolve().parent / "results/dp_offline"
CHECK_ONLY = False  # True只做启动前检查并保存LLM参数，不加载模型、不创建NPU进程

# =========================
# 测试实现：通常无需修改
# =========================


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--mode", choices=("eager", "graph"), default=RUN_MODE)
    parser.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=ENABLE_MTP)
    parser.add_argument("--l3", action=argparse.BooleanOptionalAction, default=ENABLE_L3)
    parser.add_argument("--labels", default=",".join(PROMPT_LABELS))
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--chunk-size", type=int, default=MAX_NUM_BATCHED_TOKENS)
    parser.add_argument("--timeout", type=float, default=PROCESS_TIMEOUT_SECONDS)
    parser.add_argument("--drain-timeout", type=int, default=DRAIN_TIMEOUT_SECONDS)
    parser.add_argument("--mooncake-config", type=Path, default=MOONCAKE_CONFIG_PATH)
    parser.add_argument("--segment-base", type=int, default=MOONCAKE_TCP_SEGMENT_PORT_BASE)
    parser.add_argument("--results", type=Path, default=RESULT_DIR)
    parser.set_defaults(
        visible_devices=VISIBLE_DEVICES, max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        indexer_mla_ratio=DSA_INDEXER_MLA_BLOCK_RATIO, hot_cpu_block_multiple=DSA_HOT_CPU_BLOCK_MULTIPLE,
        global_segment_size=MOONCAKE_GLOBAL_SEGMENT_SIZE,
        check_only=CHECK_ONLY,
    )
    args = parser.parse_args(argv)
    args.mooncake_config = Path(args.mooncake_config)
    args.results = Path(args.results)
    if not isinstance(args.model, str) or not args.model.strip() or args.mode not in ("eager", "graph"):
        parser.error("set a nonempty MODEL_PATH and RUN_MODE=eager/graph")
    if type(args.mtp) is not bool or type(args.l3) is not bool:
        parser.error("ENABLE_MTP and ENABLE_L3 must be True or False")
    devices = args.visible_devices.split(",")
    if len(devices) != 8 or len(set(devices)) != 8 or not all(device.isdecimal() for device in devices):
        parser.error("VISIBLE_DEVICES must contain eight distinct numeric IDs for DP2/TP4")
    if args.max_num_seqs < 1 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("invalid MAX_NUM_SEQS or GPU_MEMORY_UTILIZATION")
    if args.indexer_mla_ratio < 1 or not 0 < args.hot_cpu_block_multiple < float("inf"):
        parser.error("invalid DSA HBM ratio or L2 block multiple")
    labels = tuple(args.labels.split(","))
    if len(labels) not in (2, 4) or len(set(labels)) != len(labels) or not set(labels) <= {"5k", "8k", "40k", "70k"}:
        parser.error("--labels must contain two or four distinct dataset labels")
    if args.max_tokens < 2 or args.chunk_size < 128 or args.max_model_len <= args.max_tokens:
        parser.error("invalid token/model/chunk capacity")
    if not 0 < args.drain_timeout < args.timeout < float("inf"):
        parser.error("require finite 0 < drain-timeout < timeout")
    if not 1 <= args.segment_base <= 45000:
        parser.error("segment-base must be in [1, 45000], as required by MooncakeStoreConfig")
    if args.l3 and not args.mooncake_config.is_file():
        parser.error(f"Mooncake config is missing: {args.mooncake_config}")
    if args.l3 and json.loads(args.mooncake_config.read_text(encoding="utf-8")).get("protocol") != "tcp":
        parser.error("offline DP L3 requires a Mooncake TCP config")
    return args


def _build_kwargs(args):
    graph = args.mode == "graph"
    kwargs = {
        "model": args.model, "tensor_parallel_size": 4, "pipeline_parallel_size": 1,
        "distributed_executor_backend": "mp", "enable_expert_parallel": True,
        "quantization": "ascend", "trust_remote_code": True, "seed": 1024,
        "max_num_seqs": args.max_num_seqs, "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.chunk_size, "gpu_memory_utilization": args.gpu_memory_utilization,
        "enable_chunked_prefill": True, "scheduler_reserve_full_isl": True,
        "enable_prefix_caching": True, "block_size": 128, "async_scheduling": False,
        "enforce_eager": not graph, "disable_log_stats": False,
        "shutdown_timeout": args.drain_timeout, "safetensors_load_strategy": "prefetch",
        "additional_config": {
            "enable_sparse_sfa_c8": True, "enable_sparse_li_c8": True,
            "dsa_sparse_config": {
                "enabled": True, "split_indexer_cache": True,
                "indexer_mla_block_ratio": args.indexer_mla_ratio,
                "hot_cpu_block_multiple": args.hot_cpu_block_multiple,
                "sparse_activation_tokens": 8192 if args.mtp else 6144,
                "prompt_budget_thresholds": [32768, 65536],
                "resident_budget_tokens": [8192 if args.mtp else 6144, 10240, 12288],
                "max_active_reqs": 256, "enable_row_mode_decode_graph": graph,
            },
        },
    }
    # Exactly the native offline SPMD convention: ParallelConfig reads VLLM_DP_*.
    if args.mtp:
        kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 3, "enforce_eager": True}
    if graph:
        query_len = 4 if args.mtp else 1
        rows = sorted({row for row in (1, 2, 4, 8, args.max_num_seqs) if row <= args.max_num_seqs})
        kwargs["compilation_config"] = {
            "mode": "VLLM_COMPILE", "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [row * query_len for row in rows],
        }
    if args.l3:
        kwargs["kv_transfer_config"] = {
            "kv_connector": "DSAL3Connector", "kv_role": "kv_both", "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "backend": "mooncake", "max_inflight_l3_store_jobs": 8,
                "max_active_channels": 8, "max_descriptors_per_l3_put": 128, "direct_hbm_load": False,
            },
        }
    return kwargs


def _preflight(args):
    if not (Path(args.model) / "config.json").is_file():
        raise RuntimeError(f"MODEL_PATH must be a local model directory containing config.json: {args.model}")
    packages = ("vllm", "vllm_ascend", "torch_npu") + (("mooncake",) if args.l3 else ())
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"missing Python packages before model startup: {missing}")
    if args.l3:
        config = json.loads(args.mooncake_config.read_text(encoding="utf-8"))
        # 后端只在环境变量未设置时使用文件值，显式空字符串也会覆盖配置。
        override = os.getenv("MOONCAKE_MASTER")
        address = override if override is not None else config.get("master_server_address", "")
        if not isinstance(address, str) or not address.strip():
            raise ValueError(f"invalid Mooncake Master address: {address!r}")
        endpoint = urlsplit(address if "://" in address else "//" + address)
        if not endpoint.hostname or not endpoint.port:
            raise ValueError(f"invalid Mooncake Master address: {address!r}")
        # This only checks the listener, not Mooncake data IO capability.
        try:
            with socket.create_connection((endpoint.hostname, endpoint.port), timeout=3):
                pass
        except OSError as exc:
            raise RuntimeError(f"Mooncake Master is unreachable at {address}; no model was loaded") from exc


def _validate_kwargs(args, kwargs):
    connector = kwargs.get("kv_transfer_config")
    if args.l3:
        if not isinstance(connector, dict) or (
            connector.get("kv_connector"), connector.get("kv_role"), connector.get("kv_load_failure_policy")
        ) != ("DSAL3Connector", "kv_both", "fail"):
            raise ValueError("ENABLE_L3=True but LLM kwargs lack the required DSAL3Connector configuration")
    elif connector is not None:
        raise ValueError("ENABLE_L3=False but LLM kwargs contain a connector")
    if "data_parallel_size" in kwargs or kwargs.get("tensor_parallel_size") != 4:
        raise ValueError("offline SPMD requires env-based DP identity and tensor_parallel_size=4")
    if ("speculative_config" in kwargs) != args.mtp:
        raise ValueError("ENABLE_MTP and the LLM kwargs disagree")


def _load_cases(labels):
    dataset = importlib.import_module("novel_dataset")
    cases = []
    for label in labels:
        value = getattr(dataset, f"chinese_{label}")
        prompt = value[0] if isinstance(value, (list, tuple)) and value else value
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"novel_dataset.chinese_{label} must contain a nonempty prompt")
        cases.append((label, prompt))
    return cases


def _check_engine(args, rank, llm, status):
    config = llm.llm_engine.vllm_config
    parallel = config.parallel_config
    connector = getattr(config, "kv_transfer_config", None)
    connector_name = getattr(connector, "kv_connector", None)
    if (parallel.data_parallel_size, parallel.data_parallel_rank, parallel.tensor_parallel_size) != (2, rank, 4):
        raise RuntimeError("actual engine topology differs from requested DP2/TP4")
    if (status["dp_rank"], status["dp_size"]) != (rank, 2):
        raise RuntimeError(f"incorrect offline scheduler identity: {status}")
    if args.l3:
        if connector_name != "DSAL3Connector" or getattr(connector, "kv_role", None) != "kv_both":
            raise RuntimeError(f"requested L3 but actual connector is {connector_name!r}; no requests were sent")
        if not status.get("fingerprint") or not status.get("base_fingerprint"):
            raise RuntimeError("requested L3 but scheduler has no active L3 namespace; no requests were sent")
    elif connector is not None:
        raise RuntimeError(f"unexpected connector in Local-only test: {connector_name!r}")
    speculative = getattr(config, "speculative_config", None)
    if (speculative is not None) != args.mtp or (args.mtp and (
        speculative.method != "mtp" or speculative.num_speculative_tokens != 3
    )):
        raise RuntimeError("actual MTP configuration differs from the test settings")
    if not config.cache_config.enable_prefix_caching or config.scheduler_config.async_scheduling:
        raise RuntimeError("offline closeout requires prefix caching and synchronous scheduling")
    return {"dp_rank": rank, "tp_size": 4, "connector": connector_name, "mtp": args.mtp,
            "fingerprint": status.get("fingerprint")}


def _wait_idle(core, timeout):
    deadline = time.monotonic() + timeout
    while True:
        status = core.call_utility("dsa_cache_admin", "status")
        if status.get("fatal"):
            raise RuntimeError(f"DSA offline engine failed: {status}")
        if not status["busy"] and not status["pending_pause"] and not status["l3_pending"]:
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(f"DSA offline drain timed out: {status}")
        time.sleep(0.1)


def _check_output(output, *, expect_hit, max_tokens):
    if len(output.outputs) != 1:
        raise AssertionError("offline request must have exactly one completion")
    completion = output.outputs[0]
    cached = output.num_cached_tokens
    if cached is None or not 0 <= cached <= len(output.prompt_token_ids) or (cached > 0) != expect_hit:
        raise AssertionError(f"offline cache hit mismatch: cached={cached}, expected_hit={expect_hit}")
    if len(completion.token_ids) != max_tokens or completion.finish_reason != "length":
        raise AssertionError(f"offline completion stopped early: {completion}")
    return {
        "prompt_tokens": len(output.prompt_token_ids), "cached_tokens": cached,
        "token_ids": list(completion.token_ids), "text": completion.text,
        "finish_reason": completion.finish_reason,
    }


def _worker(rank, args, run_id, port, cases, barrier, result_dir):
    os.environ.update({
        "VLLM_DP_SIZE": "2", "VLLM_DP_RANK": str(rank), "VLLM_DP_RANK_LOCAL": str(rank),
        "VLLM_DP_MASTER_IP": "127.0.0.1", "VLLM_DP_MASTER_PORT": str(port),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_OP_EXPANSION_MODE": "AIV", "HCCL_BUFFSIZE": "200",
        "OMP_NUM_THREADS": "10", "OMP_PROC_BIND": "false",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True", "VLLM_ASCEND_ENABLE_MLAPO": "1",
        "VLLM_LOGGING_LEVEL": "INFO",
    })
    if args.l3:
        os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        os.environ.update({
            "MOONCAKE_CONFIG_PATH": str(args.mooncake_config.resolve()),
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": args.global_segment_size,
            "MOONCAKE_TCP_SEGMENT_PORT_BASE": str(args.segment_base),
            "MC_TCP_ENABLE_CONNECTION_POOL": "1",
        })
    llm = None
    failed = False
    try:
        kwargs = _build_kwargs(args)
        _validate_kwargs(args, kwargs)
        from vllm import LLM, SamplingParams

        llm = LLM(**kwargs)
        core = llm.llm_engine.engine_core
        status = _wait_idle(core, args.drain_timeout)
        actual = _check_engine(args, rank, llm, status)
        (result_dir / f"dp{rank}-engine.json").write_text(json.dumps(actual, indent=2), encoding="utf-8")
        print(f"[dsa-dp-offline] actual engine: {actual}", flush=True)
        params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
        for phase in ("cold", "local", "reset-cross"):
            barrier.wait(timeout=args.timeout)
            # Negative external-reset coverage belongs in UT, not before a warm replay.
            if phase == "reset-cross":
                if not llm.reset_prefix_cache(reset_connector=False):
                    raise AssertionError("offline local reset refused after both domains drained")
            barrier.wait(timeout=args.timeout)
            selected = cases[(1 - rank if phase == "reset-cross" else rank)::2]
            prompts = [{"prompt": prompt, "cache_salt": f"{run_id}-{label}"} for label, prompt in selected]
            outputs = llm.generate(prompts, params, use_tqdm=False)
            if len(outputs) != len(selected):
                raise AssertionError("offline request/output count mismatch")
            for (label, _), output in zip(selected, outputs, strict=True):
                record = _check_output(output, expect_hit=phase == "local" or (phase == "reset-cross" and args.l3),
                                       max_tokens=args.max_tokens)
                record.update(phase=phase, label=label, dp_rank=rank, request_id=output.request_id)
                with (result_dir / f"dp{rank}.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[dsa-dp-offline] dp={rank} phase={phase} label={label} cached={record['cached_tokens']} "
                      f"tokens={len(record['token_ids'])}", flush=True)
            _wait_idle(core, args.drain_timeout)
        barrier.wait(timeout=args.timeout)
        print(f"[dsa-dp-offline] dp={rank} requesting native shutdown", flush=True)
    except BaseException:
        failed = True
        detail = traceback.format_exc()
        (result_dir / f"dp{rank}-failure.txt").write_text(detail, encoding="utf-8")
        print(f"[dsa-dp-offline] dp={rank} FAILED before cleanup:\n{detail}", flush=True)
        barrier.abort()
        raise
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=int(args.drain_timeout))
            except Exception:
                if not failed:
                    raise
                traceback.print_exc()  # Preserve the original test error if teardown also fails.


def main(argv=None):
    print(f"[dsa-dp-offline] script={Path(__file__).resolve()}", flush=True)
    args = _parse_args(argv)
    _preflight(args)
    cases = _load_cases(args.labels.split(","))
    # Python itself must see the same seed before spawning either client.
    os.environ["PYTHONHASHSEED"] = "114514"
    os.environ["ASCEND_VISIBLE_DEVICES"] = args.visible_devices
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.visible_devices
    run_id = f"dsa-dp-offline-{uuid.uuid4().hex}"
    result_dir = args.results.resolve() / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    kwargs = _build_kwargs(args)
    _validate_kwargs(args, kwargs)
    (result_dir / "llm-kwargs.json").write_text(json.dumps(kwargs, indent=2), encoding="utf-8")
    print(
        f"[dsa-dp-offline] model={args.model} mode={args.mode} dp=2 tp=4 devices={args.visible_devices} "
        f"mtp={args.mtp} l3={args.l3} max_tokens={args.max_tokens} max_num_seqs={args.max_num_seqs} "
        f"max_model_len={args.max_model_len} chunk_size={args.chunk_size} "
        f"hbm_ratio={args.indexer_mla_ratio} l2_multiple={args.hot_cpu_block_multiple} results={result_dir}",
        flush=True,
    )
    print(f"[dsa-dp-offline] kv_transfer_config={kwargs.get('kv_transfer_config')}", flush=True)
    if args.check_only:
        print("[dsa-dp-offline] PREFLIGHT PASSED; no model or NPU processes started", flush=True)
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    barrier = context.Barrier(2)
    processes = []
    try:
        for rank in (0, 1):
            process = context.Process(target=_worker, args=(rank, args, run_id, port, cases, barrier, result_dir))
            process.start()
            processes.append(process)
        deadline = time.monotonic() + args.timeout
        while any(process.is_alive() for process in processes):
            if any(process.exitcode not in (None, 0) for process in processes):
                raise RuntimeError(f"offline client failed: {[process.exitcode for process in processes]}")
            if time.monotonic() >= deadline:
                raise TimeoutError("offline deployment exceeded --timeout")
            time.sleep(0.2)
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(f"offline client failed: {[process.exitcode for process in processes]}")
    finally:
        # Only stop this script's own client processes, never enumerate other NPU users.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
    print(
        f"[dsa-dp-offline] INFERENCE/CACHE PASSED clients=2 phases=3 results={result_dir}; "
        "numerical equivalence and graceful worker exit are not asserted", flush=True,
    )


if __name__ == "__main__":
    main()
