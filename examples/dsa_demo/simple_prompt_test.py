#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSA 稀疏卸载最小离线冒烟脚本。

测试时只修改下方“用户配置”区，然后直接运行：

    python examples/dsa_demo/simple_prompt_test.py

``disabled`` 用于验证基线隔离，``cache-init`` 只验证双平面 KV cache
初始化，``eager`` 和 ``graph`` 分别验证 DSA eager 与 FULL decode graph。
默认短 prompt 主要覆盖 DENSE；验证真正的 sparse/ENTER 路径时，应换成
token 长度超过 ``DSA_SPARSE_ACTIVATION_TOKENS`` 的文本。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# =========================
# 用户配置
# =========================

MODEL_PATH = "/mnt/kv_dpc/weight/GLM-5.1-w4a8"
RUN_MODE = "eager"  # disabled / cache-init / eager / graph
PROMPTS = [
    "你好，请用一句话介绍你自己。",
    "请只回答数字：一加一等于多少？",
]

# 当前 L3 精度收缩测试在服务器侧使用 ``novel_dataset``：
#
#   Case A：5k 单请求 L3 graph
#       PROMPTS = chinese_5k
#       RUN_MODE = "graph"
#       MAX_NUM_SEQS = 1
#       MAX_TOKENS = 5
#       ENABLE_CHUNKED_PREFILL = True
#       ENABLE_PREFIX_CACHING = True
#       PREFIX_CACHE_REPLAY_ROUNDS = 3
#       ENABLE_L3_WRITE_BEHIND = True
#       ENABLE_L3_PREFIX_LOADBACK = True
#
#   Case B：混合 bsz=4 Local-L2 graph 稳定性对照
#       PROMPTS = chinese_8k + chinese_5k + chinese_70k + chinese_40k
#       RUN_MODE = "graph"
#       MAX_NUM_SEQS = 4
#       MAX_TOKENS = 5
#       ENABLE_CHUNKED_PREFILL = True
#       ENABLE_PREFIX_CACHING = True
#       PREFIX_CACHE_REPLAY_ROUNDS = 3
#       ENABLE_L3_WRITE_BEHIND = True
#       ENABLE_L3_PREFIX_LOADBACK = False
#
#   Case C：相同混合 bsz=4 的原生 HBM prefix graph 对照
#       PROMPTS = chinese_8k + chinese_5k + chinese_70k + chinese_40k
#       RUN_MODE = "disabled"
#       MAX_NUM_SEQS = 4
#       MAX_TOKENS = 5
#       ENABLE_CHUNKED_PREFILL = True
#       ENABLE_PREFIX_CACHING = True
#       PREFIX_CACHE_REPLAY_ROUNDS = 3
#       ENABLE_L3_WRITE_BEHIND = False
#       ENABLE_L3_PREFIX_LOADBACK = False
#
# 三组都保持 COMPARE_LAST_TWO_PREFIX_REPLAYS/STRICT_PREFIX_REPLAY_TOKEN_MATCH/
# ALLOW_MULTI_REQUEST_GRAPH_TOPK_VARIANCE 为 True。Case A 比较
# Local-L2/L3，Case B 比较 Local-L2/Local-L2。

TENSOR_PARALLEL_SIZE = 16
DATA_PARALLEL_SIZE = 1
MAX_NUM_SEQS = 2
MAX_MODEL_LEN = 8192
MAX_NUM_BATCHED_TOKENS = 8192
ENABLE_CHUNKED_PREFILL = False
ENABLE_PREFIX_CACHING = False
PREFIX_CACHE_REPLAY_ROUNDS = 1
PREFIX_REPLAY_LOGPROBS = 5
COMPARE_LAST_TWO_PREFIX_REPLAYS = True
STRICT_PREFIX_REPLAY_TOKEN_MATCH = True
ALLOW_MULTI_REQUEST_GRAPH_TOPK_VARIANCE = True
LOGGING_LEVEL = "INFO"
MAX_TOKENS = 32
GPU_MEMORY_UTILIZATION = 0.90
QUANTIZATION = "ascend"
ENABLE_EXPERT_PARALLEL = True
ENABLE_MTP = False
MTP_NUM_SPECULATIVE_TOKENS = 3
ENABLE_L3_WRITE_BEHIND = False
ENABLE_L3_PREFIX_LOADBACK = False
MOONCAKE_CONFIG_PATH = str(
    Path(__file__).resolve().parent / "L3_cache_sys_api_test/baseline_e2e/configs/mooncake-dsa-tcp.json"
)
MOONCAKE_GLOBAL_SEGMENT_SIZE = "4GB"
MOONCAKE_TCP_SEGMENT_PORT_BASE = "45000"
DSA_L3_MAX_INFLIGHT_STORE_JOBS = 8
DSA_L3_MAX_ACTIVE_CHANNELS = 8
DSA_L3_MAX_DESCRIPTORS_PER_PUT = 128
DSA_L3_SHUTDOWN_TIMEOUT_SECONDS = 300
DSA_L3_REPLAY_SETTLE_SECONDS = 5.0

# A5/950 上的 DSA 首版只支持 LI C8 与 SFA C8 同时开启；A3/910C 保持 False。
# disabled 模式也会保留这个 vLLM-Ascend 原生物理布局开关，便于用同一 C8
# cache 编码公平对照“原生 C8”与“DSA C8”。
ENABLE_A5_PACKED_C8_DSA = False

ENABLE_PROFILE = False
PROFILE_DIR = "/home/data/vllm_profile/dsa_smoke"
RESULT_JSON: str | None = None

DSA_SPARSE_ACTIVATION_TOKENS = 6144
DSA_PROMPT_BUDGET_THRESHOLDS = [32768, 65536]
DSA_RESIDENT_BUDGET_TOKENS = [6144, 10240, 12288]
DSA_INDEXER_MLA_BLOCK_RATIO = 3
DSA_MAX_ACTIVE_REQS = 256
DSA_HOT_CPU_BLOCK_MULTIPLE = 3.0
DSA_GRAPH_CAPTURE_SIZES = [1, 2, 4, 8]
DSA_TRACE_POINTS = {
    # 当前仅解析预留合同，尚无稳定日志 consumer；验收默认关闭。
    "enabled": False,
    "points": ["first_sample"],
    "ranks": [0],
}

NATIVE_RUNTIME_ENV_OVERRIDES = {
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_BUFFSIZE": "200",
    "OMP_NUM_THREADS": "10",
    "OMP_PROC_BIND": "false",
    "PYTHONHASHSEED": "114514",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "VLLM_ASCEND_ENABLE_MLAPO": "1",
    "VLLM_LOGGING_LEVEL": LOGGING_LEVEL,
}
if ENABLE_L3_WRITE_BEHIND:
    # Store-owned TCP does not consume global_ub. Avoid inheriting a stale
    # setting from an earlier AscendDirect probe in the same shell.
    os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
    NATIVE_RUNTIME_ENV_OVERRIDES.update(
        {
            "MOONCAKE_CONFIG_PATH": MOONCAKE_CONFIG_PATH,
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": MOONCAKE_GLOBAL_SEGMENT_SIZE,
            # 同一 Mooncake Master 上并行启动多个实例时必须使用不同 base，
            # 否则各实例的 host:base+rank segment identity 会冲突。
            "MOONCAKE_TCP_SEGMENT_PORT_BASE": (
                MOONCAKE_TCP_SEGMENT_PORT_BASE
            ),
            # DSA 的 120-plane block 会产生大量 TCP descriptor。连接池关闭
            # 时，每个 descriptor 都建立短连接，TP8 loadback 很快耗尽临时端口。
            "MC_TCP_ENABLE_CONNECTION_POOL": "1",
        }
    )

for key, value in NATIVE_RUNTIME_ENV_OVERRIDES.items():
    os.environ[key] = value

# 环境变量必须在首次导入 vLLM 前设置，平台插件会在 import/LLM 构造期间发现。
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig, ProfilerConfig  # noqa: E402


def build_dsa_config(enable_graph: bool) -> dict[str, Any]:
    sparse_activation_tokens = 8192 if ENABLE_MTP else DSA_SPARSE_ACTIVATION_TOKENS
    resident_budget_tokens = [8192, 10240, 12288] if ENABLE_MTP else DSA_RESIDENT_BUDGET_TOKENS
    return {
        "enabled": True,
        "split_indexer_cache": True,
        "indexer_mla_block_ratio": DSA_INDEXER_MLA_BLOCK_RATIO,
        "sparse_activation_tokens": sparse_activation_tokens,
        "prompt_budget_thresholds": DSA_PROMPT_BUDGET_THRESHOLDS,
        "resident_budget_tokens": resident_budget_tokens,
        "max_active_reqs": DSA_MAX_ACTIVE_REQS,
        "hot_cpu_block_multiple": DSA_HOT_CPU_BLOCK_MULTIPLE,
        "enable_row_mode_decode_graph": enable_graph,
        "trace_points": DSA_TRACE_POINTS,
    }


def build_llm_kwargs() -> dict[str, Any]:
    if ENABLE_MTP and not ENABLE_A5_PACKED_C8_DSA:
        raise ValueError("DSA compromise MTP3 requires ENABLE_A5_PACKED_C8_DSA=True")
    if ENABLE_MTP and MTP_NUM_SPECULATIVE_TOKENS != 3:
        raise ValueError("DSA compromise MTP requires MTP_NUM_SPECULATIVE_TOKENS=3")
    if ENABLE_L3_WRITE_BEHIND and ENABLE_MTP:
        raise ValueError("DSA L3 first version does not support MTP")
    if ENABLE_L3_WRITE_BEHIND and RUN_MODE == "disabled":
        raise ValueError("ENABLE_L3_WRITE_BEHIND requires DSA to be enabled")
    if ENABLE_L3_PREFIX_LOADBACK and not ENABLE_L3_WRITE_BEHIND:
        raise ValueError(
            "ENABLE_L3_PREFIX_LOADBACK requires ENABLE_L3_WRITE_BEHIND=True"
        )
    if ENABLE_L3_PREFIX_LOADBACK and not ENABLE_PREFIX_CACHING:
        raise ValueError(
            "ENABLE_L3_PREFIX_LOADBACK requires ENABLE_PREFIX_CACHING=True"
        )
    if ENABLE_L3_PREFIX_LOADBACK and PREFIX_CACHE_REPLAY_ROUNDS < 3:
        raise ValueError(
            "ENABLE_L3_PREFIX_LOADBACK requires three rounds: "
            "cold, local-L2 replay, and L3 replay"
        )
    graph_enabled = RUN_MODE == "graph"
    kwargs: dict[str, Any] = {
        "model": MODEL_PATH,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "pipeline_parallel_size": 1,
        "data_parallel_size": DATA_PARALLEL_SIZE,
        "quantization": QUANTIZATION,
        "seed": 1024,
        "enable_expert_parallel": ENABLE_EXPERT_PARALLEL,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "trust_remote_code": True,
        "enable_prefix_caching": ENABLE_PREFIX_CACHING,
        "enable_chunked_prefill": ENABLE_CHUNKED_PREFILL,
        "scheduler_reserve_full_isl": ENABLE_CHUNKED_PREFILL,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "block_size": 128,
        "async_scheduling": False,
        "enforce_eager": not graph_enabled,
        "disable_log_stats": False,
    }
    additional_config: dict[str, Any] = {}
    if ENABLE_A5_PACKED_C8_DSA:
        additional_config.update(
            {
                "enable_sparse_sfa_c8": ENABLE_A5_PACKED_C8_DSA,
                "enable_sparse_li_c8": ENABLE_A5_PACKED_C8_DSA,
            }
        )
    if RUN_MODE != "disabled":
        additional_config.update(
            {
                "dsa_sparse_config": build_dsa_config(graph_enabled),
            }
        )
    if additional_config:
        kwargs["additional_config"] = additional_config
    if ENABLE_MTP:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": MTP_NUM_SPECULATIVE_TOKENS,
        }
    if ENABLE_L3_WRITE_BEHIND:
        kwargs["shutdown_timeout"] = DSA_L3_SHUTDOWN_TIMEOUT_SECONDS
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="DSAL3Connector",
            kv_role=(
                "kv_both"
                if ENABLE_L3_PREFIX_LOADBACK
                else "kv_producer"
            ),
            kv_load_failure_policy="fail",
            kv_connector_extra_config={
                "backend": "mooncake",
                "max_inflight_l3_store_jobs": (DSA_L3_MAX_INFLIGHT_STORE_JOBS),
                "max_active_channels": DSA_L3_MAX_ACTIVE_CHANNELS,
                "max_descriptors_per_l3_put": (DSA_L3_MAX_DESCRIPTORS_PER_PUT),
            },
        )
    if graph_enabled:
        decode_query_len = 1 + MTP_NUM_SPECULATIVE_TOKENS if ENABLE_MTP else 1
        capture_sizes = sorted(
            {size * decode_query_len for size in (*DSA_GRAPH_CAPTURE_SIZES, MAX_NUM_SEQS) if size <= MAX_NUM_SEQS}
        )
        kwargs["compilation_config"] = {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": capture_sizes,
        }
    if ENABLE_PROFILE:
        profile_dir = Path(PROFILE_DIR).resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(profile_dir),
            torch_profiler_with_stack=True,
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=True,
        )
    return kwargs


def write_result(payload: dict[str, Any]) -> None:
    if RESULT_JSON is None:
        return
    path = Path(RESULT_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[dsa-smoke] wrote result: {path}")


def shutdown_l3_engine(llm: LLM) -> None:
    """离线 LLM finalizer 不传 timeout，DSA L3 需要显式 drain。"""

    if not ENABLE_L3_WRITE_BEHIND:
        return
    engine_core = llm.llm_engine.engine_core
    print(f"[dsa-smoke] draining pending L3 stores before engine teardown; timeout={DSA_L3_SHUTDOWN_TIMEOUT_SECONDS}s")
    engine_core.shutdown(timeout=DSA_L3_SHUTDOWN_TIMEOUT_SECONDS)


def _token_ids(outputs: list[Any]) -> list[tuple[int, ...]]:
    return [tuple(output.outputs[0].token_ids) for output in outputs]


def _top_logprobs(
    output: Any,
    position: int,
) -> list[tuple[int, float, int | None, str | None]]:
    logprobs = output.outputs[0].logprobs
    if logprobs is None or position >= len(logprobs):
        return []
    candidates = logprobs[position]
    if candidates is None:
        return []

    rows = [
        (
            int(token_id),
            float(candidate.logprob),
            candidate.rank,
            candidate.decoded_token,
        )
        for token_id, candidate in candidates.items()
    ]
    rows.sort(
        key=lambda row: (
            row[2] is None,
            row[2] if row[2] is not None else 0,
            -row[1],
        )
    )
    return rows


def _compare_replay_outputs(
    left_outputs: list[Any],
    right_outputs: list[Any],
    *,
    label: str,
) -> bool:
    if len(left_outputs) != len(right_outputs):
        raise AssertionError(f"{label} request count differs: left={len(left_outputs)} right={len(right_outputs)}")

    mismatch_count = 0
    unaccepted_mismatches: list[str] = []
    for request_index, (left, right) in enumerate(zip(left_outputs, right_outputs, strict=True)):
        if left.prompt != right.prompt:
            raise AssertionError(f"{label} output order changed at request {request_index}")
        left_ids = tuple(left.outputs[0].token_ids)
        right_ids = tuple(right.outputs[0].token_ids)
        if left_ids == right_ids:
            continue

        common_prefix = 0
        for left_token, right_token in zip(left_ids, right_ids):
            if left_token != right_token:
                break
            common_prefix += 1
        left_token = left_ids[common_prefix] if common_prefix < len(left_ids) else None
        right_token = right_ids[common_prefix] if common_prefix < len(right_ids) else None
        detail = (
            f"[dsa-smoke] {label} mismatch: "
            f"request={request_index} common_prefix={common_prefix} "
            f"left_token={left_token} right_token={right_token}"
        )
        mismatch_count += 1
        print(detail)
        left_top_logprobs = _top_logprobs(left, common_prefix)
        right_top_logprobs = _top_logprobs(right, common_prefix)
        print(
            "[dsa-smoke] left top-logprobs "
            "(token_id, logprob, rank, text): "
            f"{left_top_logprobs}"
        )
        print(
            "[dsa-smoke] right top-logprobs "
            "(token_id, logprob, rank, text): "
            f"{right_top_logprobs}"
        )

        topk_equivalent = (
            ALLOW_MULTI_REQUEST_GRAPH_TOPK_VARIANCE
            and RUN_MODE == "graph"
            and len(left_outputs) > 1
            and common_prefix > 0
            and left_token is not None
            and right_token is not None
            and left_token in {row[0] for row in right_top_logprobs}
            and right_token in {row[0] for row in left_top_logprobs}
        )
        if topk_equivalent:
            print(
                "[dsa-smoke] accepted multi-request graph numerical "
                "variance: first divergent tokens are present in both top-k"
            )
        else:
            unaccepted_mismatches.append(detail)

    if mismatch_count == 0:
        print(f"[dsa-smoke] {label} token consistency passed")
        return True
    if not unaccepted_mismatches:
        print(
            f"[dsa-smoke] {label} top-k numerical consistency passed: "
            f"requests={mismatch_count}/{len(left_outputs)}"
        )
        return True
    if STRICT_PREFIX_REPLAY_TOKEN_MATCH:
        raise AssertionError(
            f"{label} has {len(unaccepted_mismatches)}/"
            f"{len(left_outputs)} unexplained token mismatches: "
            f"{unaccepted_mismatches}"
        )
    print(
        f"[dsa-smoke] {label} unexplained token mismatch recorded without "
        f"failing: requests={len(unaccepted_mismatches)}/{len(left_outputs)}"
    )
    return False


def main() -> None:
    if RUN_MODE not in {"disabled", "cache-init", "eager", "graph"}:
        raise ValueError(f"RUN_MODE must be disabled/cache-init/eager/graph, got {RUN_MODE!r}")
    if len(PROMPTS) > MAX_NUM_SEQS:
        raise ValueError(f"PROMPTS has {len(PROMPTS)} rows but MAX_NUM_SEQS={MAX_NUM_SEQS}")
    if PREFIX_CACHE_REPLAY_ROUNDS <= 0:
        raise ValueError("PREFIX_CACHE_REPLAY_ROUNDS must be positive")
    if PREFIX_REPLAY_LOGPROBS < 0:
        raise ValueError("PREFIX_REPLAY_LOGPROBS must be non-negative")

    kwargs = build_llm_kwargs()
    print(
        "[dsa-smoke] "
        f"mode={RUN_MODE} model={MODEL_PATH!r} prompts={len(PROMPTS)} "
        f"max_model_len={MAX_MODEL_LEN} max_num_seqs={MAX_NUM_SEQS} "
        f"a5_packed_c8={ENABLE_A5_PACKED_C8_DSA} mtp={ENABLE_MTP} "
        f"l3_write_behind={ENABLE_L3_WRITE_BEHIND} "
        f"l3_prefix_loadback={ENABLE_L3_PREFIX_LOADBACK} "
        f"prefix_cache={ENABLE_PREFIX_CACHING} "
        f"replay_rounds={PREFIX_CACHE_REPLAY_ROUNDS}"
    )
    llm = LLM(**kwargs)

    if RUN_MODE == "cache-init":
        print(
            "[dsa-smoke] split Indexer/MLA cache initialized; "
            "verify exactly one capacity report in the complete process log"
        )
        write_result(
            {
                "mode": RUN_MODE,
                "model": MODEL_PATH,
                "status": "passed",
                "validated": ("LLM construction completed with DSA split cache enabled"),
            }
        )
        shutdown_l3_engine(llm)
        del llm
        return

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        min_tokens=0,
        ignore_eos=False,
        logprobs=PREFIX_REPLAY_LOGPROBS or None,
    )
    try:
        if ENABLE_PROFILE:
            llm.start_profile()
        output_rounds = []
        for round_index in range(PREFIX_CACHE_REPLAY_ROUNDS):
            output_rounds.append(
                llm.generate(
                    PROMPTS,
                    sampling_params,
                    use_tqdm=False,
                )
            )
            if (
                ENABLE_L3_PREFIX_LOADBACK
                and round_index + 2 == PREFIX_CACHE_REPLAY_ROUNDS
            ):
                # P1-W 不阻塞客户端响应。给后台 put 一个显式测试窗口，
                # 随后只清本地 HBM/L2 prefix，不重置 Connector/Mooncake。
                time.sleep(DSA_L3_REPLAY_SETTLE_SECONDS)
                if not llm.reset_prefix_cache(reset_connector=False):
                    raise RuntimeError(
                        "failed to reset local prefix cache before L3 replay"
                    )
                print(
                    "[dsa-smoke] local HBM/L2 prefix cache reset; "
                    "Mooncake objects retained for the next replay"
                )
    finally:
        if ENABLE_PROFILE:
            llm.stop_profile()

    records = []
    print("\nGenerated Outputs:\n" + "-" * 60)
    for round_index, outputs in enumerate(output_rounds):
        if PREFIX_CACHE_REPLAY_ROUNDS > 1:
            print(f"Prefix replay round {round_index + 1}:")
        for request_index, output in enumerate(outputs):
            completion = output.outputs[0]
            record = {
                "request_index": request_index,
                "prompt": output.prompt,
                "token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "text": completion.text,
            }
            if PREFIX_CACHE_REPLAY_ROUNDS > 1:
                record["round_index"] = round_index
            records.append(record)
            print(f"Req {request_index}: token_ids={record['token_ids']} finish_reason={record['finish_reason']!r}")
            print(f"Output: {record['text']!r}")
            print("-" * 60)

    try:
        if (
            COMPARE_LAST_TWO_PREFIX_REPLAYS
            and ENABLE_PREFIX_CACHING
            and PREFIX_CACHE_REPLAY_ROUNDS >= 3
        ):
            comparison_label = (
                "local-L2/L3 replay"
                if ENABLE_L3_PREFIX_LOADBACK
                else "local-L2/local-L2 replay"
            )
            _compare_replay_outputs(
                output_rounds[-2],
                output_rounds[-1],
                label=comparison_label,
            )
            if _token_ids(output_rounds[0]) != _token_ids(output_rounds[-2]):
                print(
                    "[dsa-smoke] cold/prefix token IDs differ; "
                    "recorded as full-prefill versus prefix-suffix numerical "
                    "path difference"
                )

        write_result(
            {
                "mode": RUN_MODE,
                "model": MODEL_PATH,
                "status": "passed",
                "outputs": records,
            }
        )
    finally:
        shutdown_l3_engine(llm)
        del llm


if __name__ == "__main__":
    main()
