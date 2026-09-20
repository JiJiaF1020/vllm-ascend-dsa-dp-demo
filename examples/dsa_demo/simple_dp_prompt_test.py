#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""同事入门用的单机DP离线脚本：修改顶部配置，然后直接python运行。

保留simple_prompt_test.py的配置/输出风格，但启动采用v0.23官方offline SPMD：
每个DP一个LLM客户端，在导入vLLM前设置该客户端的VLLM_DP_*，不重复创建DP部署。
不依赖其他demo的配置函数，不自动启动Mooncake，不安装退出探针。
完成后自动精简JSONL并输出跨轮精度证据，也可只整理已有结果而不启动推理。
默认三轮：首次生成、本地prefix复用、清本地并交换DP后从L3回载。
这是功能探索样例，不自动判定精度、缓存IO字节一致性或worker优雅退出。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import socket
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlsplit

# =============================================================================
# 0. 已有结果摘要：正常推理保持空字符串；每次推理结束会自动生成摘要
# =============================================================================
SUMMARY_ONLY_DIR = ""  # 填已有simple-dp-xxx目录后，只生成摘要，不加载数据/模型或连接Mooncake
# 只需python运行本脚本，无额外命令行参数；处理完旧结果后改回""才能重新推理。

ENABLE_PROFILE = False
PROFILE_DIR = "/home/vllm_test_new/"

# =============================================================================
# 1. 输入与采样：PROMPTS是整份输入，不是每个DP必须复制的一份
# =============================================================================
# 数据导入与PROMPTS由使用者直接维护；可替换为自己的数据或字符串列表。
if not SUMMARY_ONLY_DIR:
    from novel_dataset import chinese_5k, chinese_10k, chinese_40k, chinese_64k

    PROMPTS = chinese_64k * 8
else:
    PROMPTS = []  # 只分析已有JSONL时无需novel_dataset。
MAX_TOKENS = 256  # 默认观察MTP decode跨满块；只看短输出时可自行调小
TEMPERATURE = 0.0  # 0为greedy；本样例开启MTP时只允许0
IGNORE_EOS = True  # 忽略EOS，避免跨满块测试因提前停止而没有覆盖到目标路径
LOGPROBS = 5  # 保存逐token的top-5候选概率；只看文本/性能时可设0关闭
PRINT_TOKEN_IDS = True  # 终端同时打印token IDs与文本；长输出可关闭，JSONL仍保留

# =============================================================================
# 2. vLLM/vLLM-Ascend基线通用配置
# =============================================================================
MODEL_PATH = "/home/models/GLM-5.2-w4a4c8-mxfp4"  # 修改为本机权重目录
RUN_MODE = "graph"  # eager / graph；关DSA用下方ENABLE_DSA_OFFLOAD，不改成disabled
QUANTIZATION = "ascend"  # 按模型实际配置；非量化模型可改为None
TRUST_REMOTE_CODE = True
SEED = 1024
MAX_NUM_SEQS = 4  # 每个DP的并发上限；PROMPTS可以更多，多出来的由该域scheduler排队
MAX_MODEL_LEN = 100000  # prompt加输出的上下文上限，不是固定实际输入长度
MAX_NUM_BATCHED_TOKENS = 8192  # 每个DP每轮token预算，不是整机总预算
ENABLE_CHUNKED_PREFILL = True  # False时上面的token预算至少要覆盖MAX_MODEL_LEN
GPU_MEMORY_UTILIZATION = 0.78
ENABLE_EXPERT_PARALLEL = True  # MoE专家跨DP*TP布置；DSA+DP初版要求开启
ENABLE_A5_PACKED_C8_DSA = True  # A5/GLM W4A4C8原生LI/SFA布局开关，两者一起开
# 即使关DSA，也保留以上packed-C8开关做同布局基线对照；不要把它当作DSA总开关。
GRAPH_CAPTURE_ROW_SIZES = [1, 2, 4, 8,11]  # 请求行数，自动过滤大于MAX_NUM_SEQS的值
# 开MTP3时自动乘4，转换为target验算token数；只用于RUN_MODE="graph"。
# 本样例固定async_scheduling=False、PP=1、mp executor，不同时试验跨机/异步调度。

# =============================================================================
# 3. DP/TP：已验证A5八卡DP2*TP4*EP8；其他拓扑不等于已经验证
# =============================================================================
DATA_PARALLEL_SIZE = 2  # 可设1作无L3的单域对照；同时调整TP或可见卡数。DP1+L3沿用simple_prompt_test.py
TENSOR_PARALLEL_SIZE = 4
VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"  # 卡数必须等于DP*TP，脚本统一设置两种可见卡变量
PROMPT_DISTRIBUTION = "round_robin"  # 第i个输入交给DP(i%DPsize)
# replicate：每个DP都跑整份PROMPTS，可用于单个长prompt在不同域的对照。
# round_robin要求输入条数>=DPsize，不偷偷添加placeholder请求。
DP_MASTER_PORT = 0  # 0自动选择空闲端口；需要指定时填写未占用端口，IP固定本机127.0.0.1

# =============================================================================
# 4. DSA稀疏卸载：L2是工作缓存，关闭prefix cache也仍然需要它
# =============================================================================
ENABLE_DSA_OFFLOAD = True  # False为基线DP对照；此时必须同时关闭DSA L3开关
ENABLE_FUSED_COPYSFA = False
ENABLE_GROUPED_C8_KV = False  # 实验路径：四层IndexShare KV交错，MTP3+split；prefix/L3必须关闭
DSA_SPARSE_ACTIVATION_TOKENS = 6144  # 非MTP下超过该长度才进入稀疏路径
DSA_PROMPT_BUDGET_THRESHOLDS = [32768, 65536]
DSA_RESIDENT_BUDGET_TOKENS = [6144, 10240, 12288]  # 非MTP三个长度档的HBM驻留预算
DSA_MTP_RESIDENT_BUDGET_TOKENS = [8192, 10240, 12288]  # MTP: 8192..16000，128 对齐、非递减
# MTP3 activation 仍为 8192；驻留预算可独立修改，默认三档不变。
DSA_INDEXER_MLA_BLOCK_RATIO = 4  # Indexer/MLA块容量比例，不是DP之间的比例
DSA_HOT_CPU_BLOCK_MULTIPLE = 1.0  # 每域L2相对本域Indexer块数的倍数，仍受节点预算限制
DSA_MAX_ACTIVE_REQS = 256  # resident元数据行容量，必须>=MAX_NUM_SEQS

# =============================================================================
# 5. Prefix cache与重复推理：是否复用、复用几轮、是否清本地缓存
# =============================================================================
ENABLE_PREFIX_CACHING = False                # 关DSA时控制原生HBM prefix；开DSA时控制L1/L2 prefix
REPLAY_ROUNDS = 1                            # 默认：首次生成 -> 本地prefix复用 -> 清本地并交换DP后从L3回载
RESET_LOCAL_CACHE_BEFORE_LAST_REPLAY = False # 最后一轮前两域都只清本地HBM/L2，保留L3
SWAP_DP_ON_LAST_REPLAY = False               # 最后一轮循环交换各域输入，观察跨域复用；只适用于round_robin
# 本地prefix观察：ENABLE_PREFIX_CACHING=True、REPLAY_ROUNDS=2，reset/swap保持False。
# L3观察：ENABLE_L3_WRITE_BEHIND=True、ENABLE_L3_PREFIX_LOADBACK=True、
# ENABLE_PREFIX_CACHING=True、REPLAY_ROUNDS=3、RESET_LOCAL_CACHE_BEFORE_LAST_REPLAY=True。
# 再设SWAP_DP_ON_LAST_REPLAY=True可观察另一个DP读取同一内容。
# 仅打开L3开关不等于实际发生load；ENABLE_L3_PREFIX_LOADBACK=False时清本地后可重算。

# =============================================================================
# 6. MTP：固定3-token草稿；不改变target的RUN_MODE
# =============================================================================
ENABLE_MTP = True
ENABLE_MTP_GRAPH = True         # True：target graph时启用三轮合并图；卸载/baseline共用
MTP_NUM_SPECULATIVE_TOKENS = 3  # 本样例仅支持已适配的固定MTP3，且TEMPERATURE=0
# DSA下MTP层仍全量驻留，不做关键token swap；prefix开启后durable满块可下沉L2/L3。

# =============================================================================
# 7. DSA L3：当前只开放Mooncake Store-owned TCP，Master需手动提前启动
# =============================================================================
ENABLE_L3_WRITE_BEHIND = False     # 新durable满块L2->L3异步上传，允许独立于prefix开启
ENABLE_L3_PREFIX_LOADBACK = False  # True要求ENABLE_L3_WRITE_BEHIND和ENABLE_PREFIX_CACHING都为True
# 仅开启ENABLE_L3_WRITE_BEHIND为kv_producer；读写同时开启为kv_both。
# 当前读取固定L3->L2->HBM，不提供direct-HBM开关。
MOONCAKE_CONFIG_PATH = (
    Path(__file__).resolve().parent / "L3_cache_sys_api_test/baseline_e2e/configs/mooncake-dsa-tcp.json"
)
MOONCAKE_GLOBAL_SEGMENT_SIZE = "4GB"  # 每个worker贡献4GB，DP2*TP4合计32GB
MOONCAKE_TCP_SEGMENT_PORT_BASE = 45000  # 并行跑另一个实例时须换base，避免segment身份冲突
DSA_L3_MAX_INFLIGHT_STORE_JOBS = 8
DSA_L3_MAX_ACTIVE_CHANNELS = 8
DSA_L3_MAX_DESCRIPTORS_PER_PUT = 128

# =============================================================================
# 8. 运行环境与结果：一般不改；环境必须在子进程首次import vLLM前设置
# =============================================================================
LOGGING_LEVEL = "INFO"
NATIVE_RUNTIME_ENV = {
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_BUFFSIZE": "512",
    "OMP_NUM_THREADS": "10",
    "OMP_PROC_BIND": "false",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "VLLM_ASCEND_ENABLE_MLAPO": "1",
}
PROCESS_TIMEOUT_SECONDS = 1800  # 整个测试上限，异常只结束本脚本创建的客户端
SHUTDOWN_TIMEOUT_SECONDS = 300  # 正常排空上限；不是基线worker强杀窗口的替代参数
RESULT_DIR = Path(__file__).resolve().parent / "results/simple_dp"
CHECK_ONLY = False  # True只检查配置/文件/依赖/Master，保存参数；不创建LLM或NPU进程
# CHECK_ONLY不证明设备空闲、权重可完整加载或KV内存充足，这些仍由真实引擎初始化检查。

# =============================================================================
# 以下为启动和输出实现，日常无需修改。没有其他demo导入、退出探针或pickle放宽。
# =============================================================================


def _settings():
    return {name: value for name, value in globals().items() if name.isupper() and not name.startswith("_")}


def _validate(c):
    boolean_keys = [key for key in c if key.startswith("ENABLE_")] + [
        "IGNORE_EOS",
        "PRINT_TOKEN_IDS",
        "TRUST_REMOTE_CODE",
        "CHECK_ONLY",
        "RESET_LOCAL_CACHE_BEFORE_LAST_REPLAY",
        "SWAP_DP_ON_LAST_REPLAY",
    ]
    if any(type(c[key]) is not bool for key in boolean_keys):
        raise ValueError("boolean settings must be True or False")
    for key in (
        "DATA_PARALLEL_SIZE",
        "TENSOR_PARALLEL_SIZE",
        "MAX_NUM_SEQS",
        "MAX_MODEL_LEN",
        "MAX_NUM_BATCHED_TOKENS",
        "MAX_TOKENS",
        "REPLAY_ROUNDS",
        "DSA_MAX_ACTIVE_REQS",
        "DSA_INDEXER_MLA_BLOCK_RATIO",
        "DSA_L3_MAX_INFLIGHT_STORE_JOBS",
        "DSA_L3_MAX_ACTIVE_CHANNELS",
        "DSA_L3_MAX_DESCRIPTORS_PER_PUT",
    ):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    devices = c["VISIBLE_DEVICES"].split(",")
    if (
        len(devices) != c["DATA_PARALLEL_SIZE"] * c["TENSOR_PARALLEL_SIZE"]
        or len(set(devices)) != len(devices)
        or not all(x.isdecimal() for x in devices)
    ):
        raise ValueError("VISIBLE_DEVICES must contain DP*TP distinct numeric device IDs")
    if c["RUN_MODE"] not in ("eager", "graph") or c["PROMPT_DISTRIBUTION"] not in ("round_robin", "replicate"):
        raise ValueError("RUN_MODE=eager/graph; PROMPT_DISTRIBUTION=round_robin/replicate")
    if not 0 < c["GPU_MEMORY_UTILIZATION"] <= 1 or not 0 < c["DSA_HOT_CPU_BLOCK_MULTIPLE"] < float("inf"):
        raise ValueError("invalid HBM utilization or L2 multiple")
    if not 0 < c["SHUTDOWN_TIMEOUT_SECONDS"] < c["PROCESS_TIMEOUT_SECONDS"] < float("inf"):
        raise ValueError("require finite 0 < SHUTDOWN_TIMEOUT_SECONDS < PROCESS_TIMEOUT_SECONDS")
    if c["MAX_TOKENS"] >= c["MAX_MODEL_LEN"] or (
        not c["ENABLE_CHUNKED_PREFILL"] and c["MAX_NUM_BATCHED_TOKENS"] < c["MAX_MODEL_LEN"]
    ):
        raise ValueError(
            "output must fit MAX_MODEL_LEN; without chunked prefill the token budget must cover MAX_MODEL_LEN"
        )
    if (
        not math.isfinite(c["TEMPERATURE"])
        or c["TEMPERATURE"] < 0
        or type(c["LOGPROBS"]) is not int
        or c["LOGPROBS"] < 0
    ):
        raise ValueError("TEMPERATURE and LOGPROBS must be nonnegative")
    if not c["GRAPH_CAPTURE_ROW_SIZES"] or any(type(x) is not int or x < 1 for x in c["GRAPH_CAPTURE_ROW_SIZES"]):
        raise ValueError("GRAPH_CAPTURE_ROW_SIZES must contain positive request-row counts")
    if (
        type(c["DP_MASTER_PORT"]) is not int
        or type(c["MOONCAKE_TCP_SEGMENT_PORT_BASE"]) is not int
        or not 0 <= c["DP_MASTER_PORT"] <= 65535
        or not 1 <= c["MOONCAKE_TCP_SEGMENT_PORT_BASE"] <= 45000
    ):
        raise ValueError("invalid DP master port or Mooncake segment port base")
    reserved_env = {
        "ASCEND_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "PYTHONHASHSEED",
        "VLLM_ALLOW_INSECURE_SERIALIZATION",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
    }
    if not isinstance(c["NATIVE_RUNTIME_ENV"], dict) or any(
        not isinstance(key, str) or key in reserved_env or key.startswith("VLLM_DP_") for key in c["NATIVE_RUNTIME_ENV"]
    ):
        raise ValueError(
            "NATIVE_RUNTIME_ENV must not override device identity, DP identity, spawn or serialization settings"
        )
    if c.get("ENABLE_GROUPED_C8_KV", False) and (
        not c["ENABLE_DSA_OFFLOAD"] or not c["ENABLE_MTP"] or not c["ENABLE_A5_PACKED_C8_DSA"]
        or c.get("ENABLE_FUSED_COPYSFA", False) or c["ENABLE_PREFIX_CACHING"]
        or c["ENABLE_L3_WRITE_BEHIND"] or c["ENABLE_L3_PREFIX_LOADBACK"]
    ):
        raise ValueError("Grouped C8 KV requires packed-C8 DSA + MTP3 + split; disable prefix caching and L3")
    if c["ENABLE_DSA_OFFLOAD"]:
        thresholds = c["DSA_PROMPT_BUDGET_THRESHOLDS"]
        budget_key = "DSA_MTP_RESIDENT_BUDGET_TOKENS" if c["ENABLE_MTP"] else "DSA_RESIDENT_BUDGET_TOKENS"
        budgets = c[budget_key]
        if (
            not isinstance(thresholds, (list, tuple))
            or len(thresholds) != 2
            or any(type(x) is not int or x <= 0 for x in thresholds)
            or list(thresholds) != sorted(set(thresholds))
        ):
            raise ValueError("DSA_PROMPT_BUDGET_THRESHOLDS requires two increasing positive integers")
        if (
            not isinstance(budgets, (list, tuple))
            or len(budgets) != 3
            or any(type(x) is not int or x <= 0 for x in budgets)
        ):
            raise ValueError(f"{budget_key} requires three positive resident budgets")
        if not c["ENABLE_MTP"] and (
            type(c["DSA_SPARSE_ACTIVATION_TOKENS"]) is not int or c["DSA_SPARSE_ACTIVATION_TOKENS"] <= 0
        ):
            raise ValueError("invalid non-MTP DSA activation")
    if c["ENABLE_DSA_OFFLOAD"] and (not c["ENABLE_EXPERT_PARALLEL"] or c["DSA_MAX_ACTIVE_REQS"] < c["MAX_NUM_SEQS"]):
        raise ValueError("DSA DP requires expert parallel and enough DSA request rows")
    if c["ENABLE_MTP"] and (
        c["MTP_NUM_SPECULATIVE_TOKENS"] != 3 or c["TEMPERATURE"] != 0 or not c["ENABLE_A5_PACKED_C8_DSA"]
    ):
        raise ValueError("this example's MTP requires MTP3, greedy and A5 packed C8")
    if c["ENABLE_L3_WRITE_BEHIND"] and not c["ENABLE_DSA_OFFLOAD"]:
        raise ValueError("DSA L3 requires ENABLE_DSA_OFFLOAD=True; disable both L3 switches for a baseline run")
    if c["ENABLE_L3_WRITE_BEHIND"] and c["DATA_PARALLEL_SIZE"] == 1:
        raise ValueError("this example uses DP administration to drain L3; use simple_prompt_test.py for DP1+L3")
    if c["ENABLE_L3_PREFIX_LOADBACK"] and not (c["ENABLE_L3_WRITE_BEHIND"] and c["ENABLE_PREFIX_CACHING"]):
        raise ValueError("L3 loadback requires WRITE_BEHIND=True and PREFIX_CACHING=True")
    if (c["RESET_LOCAL_CACHE_BEFORE_LAST_REPLAY"] or c["SWAP_DP_ON_LAST_REPLAY"]) and c["REPLAY_ROUNDS"] < 2:
        raise ValueError("reset/swap on last replay requires REPLAY_ROUNDS>=2")
    if c["SWAP_DP_ON_LAST_REPLAY"] and (c["PROMPT_DISTRIBUTION"] != "round_robin" or c["DATA_PARALLEL_SIZE"] < 2):
        raise ValueError("SWAP_DP_ON_LAST_REPLAY requires round_robin distribution and DP>=2")
    prompts = c["PROMPTS"]
    if (
        not isinstance(prompts, (list, tuple))
        or not prompts
        or any(not isinstance(x, str) or not x.strip() for x in prompts)
    ):
        raise ValueError("PROMPTS must be a nonempty list of nonempty strings")
    if c["PROMPT_DISTRIBUTION"] == "round_robin" and len(prompts) < c["DATA_PARALLEL_SIZE"]:
        raise ValueError("round_robin needs at least one prompt per DP; use replicate for a single prompt")


def _build_kwargs(c):
    graph = c["RUN_MODE"] == "graph"
    kwargs = {
        "model": c["MODEL_PATH"],
        "tensor_parallel_size": c["TENSOR_PARALLEL_SIZE"],
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "enable_expert_parallel": c["ENABLE_EXPERT_PARALLEL"],
        "quantization": c["QUANTIZATION"],
        "trust_remote_code": c["TRUST_REMOTE_CODE"],
        "seed": c["SEED"],
        "max_num_seqs": c["MAX_NUM_SEQS"],
        "max_model_len": c["MAX_MODEL_LEN"],
        "max_num_batched_tokens": c["MAX_NUM_BATCHED_TOKENS"],
        "gpu_memory_utilization": c["GPU_MEMORY_UTILIZATION"],
        "enable_chunked_prefill": c["ENABLE_CHUNKED_PREFILL"],
        "scheduler_reserve_full_isl": c["ENABLE_CHUNKED_PREFILL"],
        "enable_prefix_caching": c["ENABLE_PREFIX_CACHING"],
        "block_size": 128,
        "async_scheduling": False,
        "enforce_eager": not graph,
        "disable_log_stats": False,
        "shutdown_timeout": c["SHUTDOWN_TIMEOUT_SECONDS"],
        "safetensors_load_strategy": "prefetch",
    }
    additional = {}
    if c["ENABLE_A5_PACKED_C8_DSA"]:
        additional.update(enable_sparse_li_c8=True, enable_sparse_sfa_c8=True)
    if c["ENABLE_DSA_OFFLOAD"]:
        additional["dsa_sparse_config"] = {
            "enabled": True,
            "enable_fused_copy_sfa": ENABLE_FUSED_COPYSFA,
            "enable_grouped_c8_kv": c.get("ENABLE_GROUPED_C8_KV", False),
            "split_indexer_cache": True,
            "indexer_mla_block_ratio": c["DSA_INDEXER_MLA_BLOCK_RATIO"],
            "hot_cpu_block_multiple": c["DSA_HOT_CPU_BLOCK_MULTIPLE"],
            "sparse_activation_tokens": 8192 if c["ENABLE_MTP"] else c["DSA_SPARSE_ACTIVATION_TOKENS"],
            "prompt_budget_thresholds": c["DSA_PROMPT_BUDGET_THRESHOLDS"],
            "resident_budget_tokens": (
                c["DSA_MTP_RESIDENT_BUDGET_TOKENS"] if c["ENABLE_MTP"] else c["DSA_RESIDENT_BUDGET_TOKENS"]
            ),
            "max_active_reqs": c["DSA_MAX_ACTIVE_REQS"],
            "enable_row_mode_decode_graph": graph,
        }
    if additional:
        kwargs["additional_config"] = additional
    if c["ENABLE_MTP"]:
        kwargs["speculative_config"] = {
            "method": "mtp", "num_speculative_tokens": 3,
            "enforce_eager": not (graph and c["ENABLE_MTP_GRAPH"]),
            "draft_tensor_parallel_size": c["TENSOR_PARALLEL_SIZE"],
        }
    if graph:
        rows = {x for x in c["GRAPH_CAPTURE_ROW_SIZES"] if x <= c["MAX_NUM_SEQS"]} | {c["MAX_NUM_SEQS"]}
        kwargs["compilation_config"] = {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": sorted(x * (4 if c["ENABLE_MTP"] else 1) for x in rows),
        }
    if c["ENABLE_L3_WRITE_BEHIND"]:
        kwargs["kv_transfer_config"] = {
            "kv_connector": "DSAL3Connector",
            "kv_role": "kv_both" if c["ENABLE_L3_PREFIX_LOADBACK"] else "kv_producer",
            "kv_load_failure_policy": "fail",
            "kv_connector_extra_config": {
                "backend": "mooncake",
                "direct_hbm_load": False,
                "max_inflight_l3_store_jobs": c["DSA_L3_MAX_INFLIGHT_STORE_JOBS"],
                "max_active_channels": c["DSA_L3_MAX_ACTIVE_CHANNELS"],
                "max_descriptors_per_l3_put": c["DSA_L3_MAX_DESCRIPTORS_PER_PUT"],
            },
        }
    # DP身份只由每个客户端的VLLM_DP_*传递；这里绝不能再次传data_parallel_size。
    return kwargs


def _preflight(c):
    if not (Path(c["MODEL_PATH"]) / "config.json").is_file():
        raise ValueError("MODEL_PATH must point to a local model directory containing config.json")
    packages = ["vllm", "vllm_ascend", "torch_npu"] + (["mooncake"] if c["ENABLE_L3_WRITE_BEHIND"] else [])
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"missing packages before model load: {missing}")
    if c["ENABLE_L3_WRITE_BEHIND"]:
        config = json.loads(Path(c["MOONCAKE_CONFIG_PATH"]).read_text(encoding="utf-8"))
        if config.get("protocol") != "tcp":
            raise ValueError("this DSA L3 example requires Mooncake protocol=tcp")
        # 与worker环境及MooncakeStoreConfig.from_file保持同一优先级；显式空值不能回退。
        override = (
            str(c["NATIVE_RUNTIME_ENV"]["MOONCAKE_MASTER"])
            if "MOONCAKE_MASTER" in c["NATIVE_RUNTIME_ENV"]
            else os.getenv("MOONCAKE_MASTER")
        )
        address = override if override is not None else config.get("master_server_address", "")
        if not isinstance(address, str) or not address.strip():
            raise ValueError(f"invalid Mooncake master address: {address!r}")
        endpoint = urlsplit(address if "://" in address else "//" + address)
        if not endpoint.hostname or not endpoint.port:
            raise ValueError(f"invalid Mooncake master address: {address!r}")
        with socket.create_connection((endpoint.hostname, endpoint.port), timeout=3):
            pass  # 只验证监听存在，不冒充数据IO测试；不启动/重启Master。


def _wait_idle(core, c):
    if not c["ENABLE_DSA_OFFLOAD"] or c["DATA_PARALLEL_SIZE"] == 1:
        return  # 基线和无L3的单域对照不调用DP专用管理接口。
    deadline = time.monotonic() + c["SHUTDOWN_TIMEOUT_SECONDS"]
    while True:
        status = core.call_utility("dsa_cache_admin", "status")
        if status.get("fatal"):
            raise RuntimeError(f"DSA engine failed: {status}")
        if not status["busy"] and not status["pending_pause"] and not status["l3_pending"]:
            if c["ENABLE_L3_WRITE_BEHIND"] and not status.get("fingerprint"):
                raise RuntimeError("L3 requested but scheduler has no active namespace")
            return status
        if time.monotonic() >= deadline:
            raise TimeoutError(f"DSA model/IO drain timed out: {status}")
        time.sleep(0.1)


def _indices(c, count, rank, round_index):
    if c["PROMPT_DISTRIBUTION"] == "replicate":
        return list(range(count))
    source = (
        (rank + 1) % c["DATA_PARALLEL_SIZE"]
        if c["SWAP_DP_ON_LAST_REPLAY"] and round_index == c["REPLAY_ROUNDS"] - 1
        else rank
    )
    return list(range(source, count, c["DATA_PARALLEL_SIZE"]))


def _round_phase(c, round_index):
    if round_index == 0:
        return "seed"
    if c["RESET_LOCAL_CACHE_BEFORE_LAST_REPLAY"] and round_index == c["REPLAY_ROUNDS"] - 1:
        return "after-local-reset"
    return f"local-replay-{round_index}" if c["ENABLE_PREFIX_CACHING"] else f"recompute-{round_index}"


def _worker(rank, c, prompts, port, barrier, run_dir):
    os.environ.update({str(key): str(value) for key, value in c["NATIVE_RUNTIME_ENV"].items()})
    os.environ.update(
        {
            "VLLM_DP_SIZE": str(c["DATA_PARALLEL_SIZE"]),
            "VLLM_DP_RANK": str(rank),
            "VLLM_DP_RANK_LOCAL": str(rank),
            "VLLM_DP_MASTER_IP": "127.0.0.1",
            "VLLM_DP_MASTER_PORT": str(port),
            "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_LOGGING_LEVEL": c["LOGGING_LEVEL"],
        }
    )
    if c["ENABLE_L3_WRITE_BEHIND"]:
        os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        os.environ.update(
            {
                "MOONCAKE_CONFIG_PATH": str(Path(c["MOONCAKE_CONFIG_PATH"]).resolve()),
                "MOONCAKE_GLOBAL_SEGMENT_SIZE": c["MOONCAKE_GLOBAL_SEGMENT_SIZE"],
                "MOONCAKE_TCP_SEGMENT_PORT_BASE": str(c["MOONCAKE_TCP_SEGMENT_PORT_BASE"]),
                "MC_TCP_ENABLE_CONNECTION_POOL": "1",
            }
        )
    llm, failed = None, False
    try:
        from vllm import LLM, SamplingParams

        kwargs = _build_kwargs(c)
        if c["ENABLE_PROFILE"]:
            from vllm.config import ProfilerConfig

            profile_dir = Path(c["PROFILE_DIR"]).resolve() / f"dp{rank}"
            profile_dir.mkdir(parents=True, exist_ok=True)
            kwargs["profiler_config"] = ProfilerConfig(
                profiler="torch",
                torch_profiler_dir=str(profile_dir),
                torch_profiler_with_stack=True,
                torch_profiler_record_shapes=True,
                torch_profiler_with_memory=True,
            )
        llm = LLM(**kwargs)
        
        actual = llm.llm_engine.vllm_config
        pc = actual.parallel_config
        kv = actual.kv_transfer_config
        if (pc.data_parallel_size, pc.data_parallel_rank, pc.tensor_parallel_size) != (
            c["DATA_PARALLEL_SIZE"],
            rank,
            c["TENSOR_PARALLEL_SIZE"],
        ):
            raise RuntimeError("actual engine topology differs from the requested offline SPMD topology")
        dsa_enabled = (actual.additional_config or {}).get("dsa_sparse_config", {}).get("enabled", False)
        if (
            dsa_enabled != c["ENABLE_DSA_OFFLOAD"]
            or actual.cache_config.enable_prefix_caching != c["ENABLE_PREFIX_CACHING"]
            or actual.scheduler_config.async_scheduling is not False
        ):
            raise RuntimeError("actual DSA/prefix/scheduling configuration differs from the requested settings")
        if c["ENABLE_L3_WRITE_BEHIND"]:
            if getattr(kv, "kv_connector", None) != "DSAL3Connector" or kv.kv_role != (
                "kv_both" if c["ENABLE_L3_PREFIX_LOADBACK"] else "kv_producer"
            ):
                raise RuntimeError("actual L3 connector/role differs from the requested settings")
        elif kv is not None:
            raise RuntimeError("unexpected KV connector in a non-L3 run")
        spec = actual.speculative_config
        if (spec is not None) != c["ENABLE_MTP"] or (
            spec is not None and (spec.method, spec.num_speculative_tokens) != ("mtp", 3)
        ):
            raise RuntimeError("actual MTP settings differ from requested settings")
        core = llm.llm_engine.engine_core
        status = _wait_idle(core, c) or {}
        engine_info = {
            "dp_rank": pc.data_parallel_rank,
            "dp_size": pc.data_parallel_size,
            "tp_size": pc.tensor_parallel_size,
            "dsa": dsa_enabled,
            "prefix": actual.cache_config.enable_prefix_caching,
            "mtp": spec is not None,
            "connector": getattr(kv, "kv_connector", None),
            "kv_role": getattr(kv, "kv_role", None),
            "fingerprint": status.get("fingerprint"),
        }
        (run_dir / f"dp{rank}-engine.json").write_text(json.dumps(engine_info, indent=2), encoding="utf-8")
        print(f"[simple-dp] actual engine: {engine_info}", flush=True)
        params = SamplingParams(
            temperature=c["TEMPERATURE"],
            max_tokens=c["MAX_TOKENS"],
            ignore_eos=c["IGNORE_EOS"],
            logprobs=c["LOGPROBS"] or None,
        )
        # 全域同一run salt且每轮不变：隔离旧实验缓存，保留本次跨DP内容共享。
        cache_salt = run_dir.name
        for round_index in range(c["REPLAY_ROUNDS"]):
            phase = _round_phase(c, round_index)
            barrier.wait(timeout=c["PROCESS_TIMEOUT_SECONDS"])
            if c["RESET_LOCAL_CACHE_BEFORE_LAST_REPLAY"] and round_index == c["REPLAY_ROUNDS"] - 1:
                if not llm.reset_prefix_cache(reset_connector=False):
                    raise RuntimeError("local-only prefix reset refused after both domains drained")
            barrier.wait(timeout=c["PROCESS_TIMEOUT_SECONDS"])
            selected = _indices(c, len(prompts), rank, round_index)
            inputs = [{"prompt": prompts[index], "cache_salt": cache_salt} for index in selected]
            
            if c["ENABLE_PROFILE"]:
                llm.start_profile()
            try:
                outputs = llm.generate(inputs, params, use_tqdm=False)
            finally:
                if c["ENABLE_PROFILE"]:
                    llm.stop_profile()
            
            if len(outputs) != len(selected):
                raise RuntimeError("request/output count mismatch")
            with (run_dir / f"dp{rank}.jsonl").open("a", encoding="utf-8") as handle:
                for index, output in zip(selected, outputs, strict=True):
                    completion = output.outputs[0]
                    record = {
                        "dp_rank": rank,
                        "round": round_index + 1,
                        "phase": phase,
                        "cache_salt": cache_salt,
                        "prompt_index": index,
                        "request_id": output.request_id,
                        "prompt": prompts[index],
                        "prompt_tokens": len(output.prompt_token_ids),
                        "cached_tokens": output.num_cached_tokens,
                        "token_ids": list(completion.token_ids),
                        "text": completion.text,
                        "finish_reason": completion.finish_reason,
                    }
                    if completion.logprobs:
                        record["logprobs"] = [
                            {str(token): float(value.logprob) for token, value in row.items()}
                            for row in completion.logprobs
                        ]
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(
                        f"[simple-dp] dp={rank} round={round_index + 1} phase={phase} prompt={index} "
                        f"prompt_tokens={record['prompt_tokens']} cached={record['cached_tokens']} "
                        f"output_tokens={len(completion.token_ids)}",
                        flush=True,
                    )
            _wait_idle(core, c)
        barrier.wait(timeout=c["PROCESS_TIMEOUT_SECONDS"])
    except BaseException:
        failed = True
        detail = traceback.format_exc()
        (run_dir / f"dp{rank}-failure.txt").write_text(detail, encoding="utf-8")
        print(f"[simple-dp] dp={rank} FAILED before cleanup:\n{detail}", flush=True)
        barrier.abort()
        raise
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=c["SHUTDOWN_TIMEOUT_SECONDS"])
            except Exception:
                if not failed:
                    raise
                traceback.print_exc()


# =============================================================================
# 结果摘要：纯标准库后处理，不改变推理、采样或缓存逻辑，不自动判定精度PASS/FAIL。
# =============================================================================
_SUMMARY_LOGPROB_CONTEXT = 1
_SUMMARY_SETTINGS = (
    "MODEL_PATH",
    "RUN_MODE",
    "DATA_PARALLEL_SIZE",
    "TENSOR_PARALLEL_SIZE",
    "MAX_NUM_SEQS",
    "MAX_MODEL_LEN",
    "MAX_NUM_BATCHED_TOKENS",
    "ENABLE_CHUNKED_PREFILL",
    "ENABLE_DSA_OFFLOAD",
    "ENABLE_GROUPED_C8_KV",
    "ENABLE_A5_PACKED_C8_DSA",
    "ENABLE_MTP",
    "ENABLE_MTP_GRAPH",
    "MTP_NUM_SPECULATIVE_TOKENS",
    "ENABLE_PREFIX_CACHING",
    "ENABLE_L3_WRITE_BEHIND",
    "ENABLE_L3_PREFIX_LOADBACK",
    "MAX_TOKENS",
    "IGNORE_EOS",
    "TEMPERATURE",
    "SEED",
    "LOGPROBS",
    "REPLAY_ROUNDS",
    "PROMPT_DISTRIBUTION",
    "RESET_LOCAL_CACHE_BEFORE_LAST_REPLAY",
    "SWAP_DP_ON_LAST_REPLAY",
    # 只保留旧结果的诊断条件说明，不再提供这些拉起开关。
    "ISOLATE_DP_CACHE_SALT",
    "TRACE_POOL_LIFECYCLE",
)
_SUMMARY_ENGINE_FIELDS = (
    "dp_rank",
    "dp_size",
    "tp_size",
    "dsa",
    "prefix",
    "mtp",
    "connector",
    "kv_role",
    "fingerprint",
)


def _summary_read_object(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _summary_probabilities(rows, location):
    if rows is None:
        return None, 0
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{location}: logprobs must be a list of token-to-logprob objects")
    cleaned, nonfinite = [], 0
    for row in rows:
        values = {}
        for token, value in row.items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"{location}: invalid logprob value for token {token}")
            if value is not None and not math.isfinite(value):
                value = None
                nonfinite += 1
            values[str(token)] = value
        cleaned.append(values)
    return cleaned, nonfinite


def _summary_read_records(root, ranks):
    records, identities = [], set()
    for rank in ranks:
        path = root / f"dp{rank}.jsonl"
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                location = f"{path.name}:{line_no}"
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{location}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{location}: expected a record object")
                for key, minimum in (("dp_rank", 0), ("round", 1), ("prompt_index", 0), ("prompt_tokens", 0)):
                    if type(row.get(key)) is not int or row[key] < minimum:
                        raise ValueError(f"{location}: invalid {key}")
                if row["dp_rank"] != rank:
                    raise ValueError(f"{location}: dp_rank does not match filename")
                if not isinstance(row.get("prompt"), str) or not isinstance(row.get("text"), str):
                    raise ValueError(f"{location}: original prompt and output text are required")
                tokens = row.get("token_ids")
                if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
                    raise ValueError(f"{location}: invalid token_ids")
                identity = (rank, row["round"], row["prompt_index"])
                if identity in identities:
                    raise ValueError(f"{location}: duplicate dp/round/prompt record {identity}")
                identities.add(identity)
                probabilities, nonfinite = _summary_probabilities(row.get("logprobs"), location)
                record = {
                    key: row.get(key)
                    for key in (
                        "dp_rank",
                        "round",
                        "phase",
                        "cache_salt",
                        "prompt_index",
                        "request_id",
                        "prompt_tokens",
                        "cached_tokens",
                        "finish_reason",
                        "token_ids",
                        "text",
                    )
                }
                record.update(
                    source=location,
                    prompt_sha256=hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
                    output_tokens=len(tokens),
                    logprob_rows=None if probabilities is None else len(probabilities),
                    nonfinite_logprobs=nonfinite,
                    _logprobs=probabilities,
                )
                records.append(record)
    if not records:
        raise ValueError("The input JSONL files contain no result records")
    return sorted(records, key=lambda record: (record["round"], record["prompt_index"], record["dp_rank"]))


def _summary_at(values, position):
    return values[position] if values is not None and 0 <= position < len(values) else None


def _summary_reference(record):
    return {
        key: record[key]
        for key in ("source", "dp_rank", "round", "phase", "cache_salt", "request_id", "cached_tokens", "finish_reason")
    }


def _summary_compare(left, right):
    result = {
        "prompt_index": left["prompt_index"],
        "left": _summary_reference(left),
        "right": _summary_reference(right),
    }
    if left["prompt_sha256"] != right["prompt_sha256"] or left["prompt_tokens"] != right["prompt_tokens"]:
        return {**result, "status": "not_comparable", "reason": "prompt content or token count changed"}
    lt, rt = left["token_ids"], right["token_ids"]
    common = 0
    while common < min(len(lt), len(rt)) and lt[common] == rt[common]:
        common += 1
    matched = common == len(lt) == len(rt)
    lp, rp = left["_logprobs"], right["_logprobs"]
    largest_delta, delta_position, delta_token = None, None, None
    first_probability_change, first_candidate_set_change = None, None
    probability_changes = []
    compared_rows, all_values_finite = 0, True
    # After the first different token, generated histories diverge. Those
    # later distributions must not be counted as same-input numerical drift.
    for position in range(min(common + 1, len(lt), len(rt))):
        lrow, rrow = _summary_at(lp, position), _summary_at(rp, position)
        if lrow is None or rrow is None:
            continue
        compared_rows += 1
        keys_match = lrow.keys() == rrow.keys()
        if not keys_match and first_candidate_set_change is None:
            first_candidate_set_change = position
        all_values_finite &= (
            bool(lrow) and bool(rrow) and all(value is not None for value in (*lrow.values(), *rrow.values()))
        )
        row_delta = None
        for token in sorted(lrow.keys() & rrow.keys()):
            if lrow[token] is None or rrow[token] is None:
                continue
            delta = abs(lrow[token] - rrow[token])
            row_delta = delta if row_delta is None else max(row_delta, delta)
            if largest_delta is None or delta > largest_delta:
                largest_delta, delta_position, delta_token = delta, position, token
        if row_delta is not None and row_delta > 0 and first_probability_change is None:
            first_probability_change = position
        if not keys_match or (row_delta is not None and row_delta > 0):
            probability_changes.append(
                {
                    "position_0based": position,
                    "candidate_keys_match": keys_match,
                    "max_shared_logprob_delta": row_delta,
                }
            )
    centers = {0, delta_position, first_probability_change, first_candidate_set_change}
    if not matched:
        centers.add(common)
    positions = {
        position
        for center in centers
        if center is not None
        for position in range(max(0, center - _SUMMARY_LOGPROB_CONTEXT), center + _SUMMARY_LOGPROB_CONTEXT + 1)
    }
    window = [
        {
            "position_0based": position,
            "same_generated_history": position <= common,
            "left_token": _summary_at(lt, position),
            "right_token": _summary_at(rt, position),
            "left_logprobs": _summary_at(lp, position),
            "right_logprobs": _summary_at(rp, position),
        }
        for position in sorted(positions)
        if position < max(len(lt), len(rt))
    ]
    result.update(
        status="tokens_match" if matched else "tokens_differ",
        common_prefix_tokens=common,
        first_difference_0based=None if matched else common,
        left_output_tokens=len(lt),
        right_output_tokens=len(rt),
        left_token=None if matched else _summary_at(lt, common),
        right_token=None if matched else _summary_at(rt, common),
        text_match=left["text"] == right["text"],
        finish_reason_match=left["finish_reason"] == right["finish_reason"],
        cached_tokens_match=left["cached_tokens"] == right["cached_tokens"],
        max_shared_logprob_delta_before_feedback=largest_delta,
        max_delta_position_0based=delta_position,
        max_delta_token=delta_token,
        first_shared_logprob_change_0based=first_probability_change,
        first_candidate_set_change_0based=first_candidate_set_change,
        compared_logprob_rows=compared_rows,
        returned_topk_equal_before_feedback=(
            not probability_changes
            if compared_rows == min(common + 1, len(lt), len(rt)) and compared_rows > 0 and all_values_finite
            else None
        ),
        same_history_probability_changes=probability_changes,
        logprob_window=window,
    )
    return result


def _summarize_results(root):
    root = Path(root).expanduser().resolve()
    settings_path = root / "settings.json"
    settings = _summary_read_object(settings_path) if settings_path.is_file() else {}
    warnings = []
    if not settings:
        warnings.append("settings.json missing; execution settings are not verified")
    dp_size = settings.get("DATA_PARALLEL_SIZE")
    if dp_size is None:
        ranks = sorted(int(path.stem[2:]) for path in root.glob("dp*.jsonl") if path.stem[2:].isdigit())
        warnings.append("DATA_PARALLEL_SIZE missing; DP files inferred, completeness is not verified")
    elif type(dp_size) is int and dp_size > 0:
        ranks = list(range(dp_size))
    else:
        raise ValueError("settings.json: DATA_PARALLEL_SIZE must be a positive integer")
    records = _summary_read_records(root, ranks)
    prompts = settings.get("PROMPTS")
    expected_count = len(prompts) if isinstance(prompts, list) else None
    rounds = settings.get("REPLAY_ROUNDS")
    if expected_count is not None and type(rounds) is int and rounds > 0:
        copies = len(ranks) if settings.get("PROMPT_DISTRIBUTION") == "replicate" else 1
        for round_index in range(1, rounds + 1):
            for prompt_index in range(expected_count):
                count = sum(row["round"] == round_index and row["prompt_index"] == prompt_index for row in records)
                if count != copies:
                    warnings.append(
                        f"prompt={prompt_index} round={round_index}: expected {copies} record(s), got {count}"
                    )
    engines = []
    for rank in ranks:
        path = root / f"dp{rank}-engine.json"
        if path.is_file():
            engine = _summary_read_object(path)
            engines.append({key: engine[key] for key in _SUMMARY_ENGINE_FIELDS if key in engine})
    fingerprints = {engine.get("fingerprint") for engine in engines if engine.get("fingerprint")}
    if len(fingerprints) > 1:
        warnings.append("DP engine fingerprints differ; do not assume cache compatibility")
    for row in records:
        if row["logprob_rows"] != row["output_tokens"]:
            warnings.append(f"{row['source']}: logprobs missing or row count differs from output tokens")
        if row["nonfinite_logprobs"]:
            warnings.append(f"{row['source']}: {row['nonfinite_logprobs']} non-finite logprobs were replaced by null")
        if not row["token_ids"]:
            warnings.append(f"{row['source']}: empty output")
    if rounds is None:
        rounds = max(row["round"] for row in records)
    if type(rounds) is not int or rounds < 1:
        raise ValueError("settings.json: REPLAY_ROUNDS must be a positive integer")
    # 优先显示最近两轮，再依次回看；单轮只输出记录，不虚构精度对照。
    compare_rounds = [(index, index + 1) for index in range(rounds - 1, 0, -1)]
    comparisons = []
    for index in sorted({row["prompt_index"] for row in records}):
        group = [row for row in records if row["prompt_index"] == index]
        for left_round, right_round in compare_rounds:
            left = {row["dp_rank"]: row for row in group if row["round"] == left_round}
            right = {row["dp_rank"]: row for row in group if row["round"] == right_round}
            if len(left) == len(right) == 1:
                pairs = [(next(iter(left.values())), next(iter(right.values())))]
            elif left and left.keys() == right.keys():
                pairs = [(left[rank], right[rank]) for rank in sorted(left)]
            else:
                warnings.append(f"prompt={index} rounds={left_round}/{right_round}: missing or ambiguous records")
                continue
            comparisons.extend(_summary_compare(a, b) for a, b in pairs)
    public_records = [{key: value for key, value in row.items() if not key.startswith("_")} for row in records]
    return {
        "schema": "simple-dp-precision-summary-v1",
        "run_dir": str(root),
        "note": "Evidence only, not an accuracy PASS/FAIL. Cold/prefix, DP placement and execution paths can differ. "
        "Result JSONL alone does not prove which cache transport executed. "
        "Post-divergence logprobs are context only.",
        "settings": {key: settings[key] for key in _SUMMARY_SETTINGS if key in settings},
        "engines": engines,
        "warnings": warnings,
        "records": public_records,
        "comparisons": comparisons,
    }


def _render_precision_summary(report):
    lines = [
        "[simple-dp-precision] 离线证据摘要（不是精度PASS/FAIL）",
        f"run_dir={report['run_dir']}",
        "settings=" + json.dumps(report["settings"], ensure_ascii=False),
        "engines=" + json.dumps(report["engines"], ensure_ascii=False),
        "",
        "记录：",
    ]
    for row in report["records"]:
        lines.append(
            f"dp={row['dp_rank']} round={row['round']} prompt={row['prompt_index']} "
            f"phase={row['phase']} "
            f"prompt_tokens={row['prompt_tokens']} cached={row['cached_tokens']} "
            f"output={row['output_tokens']} finish={row['finish_reason']}"
        )
    lines.extend(
        [
            "",
            "跨轮比较（position从0计数）：",
            "按最近相邻轮次优先对照；冷算/复用及实际L3路径请结合配置和服务端日志判断。",
        ]
    )
    for result in report["comparisons"]:
        a, b = result["left"], result["right"]
        lines.append(
            f"prompt={result['prompt_index']} r{a['round']}/dp{a['dp_rank']} -> "
            f"r{b['round']}/dp{b['dp_rank']}: {result['status']}"
        )
        lines.append(
            f"  phases={a['phase']}/{b['phase']} cached_tokens={a['cached_tokens']}/{b['cached_tokens']} "
            f"cache_salt={a['cache_salt']}/{b['cache_salt']}"
        )
        if result["status"] == "not_comparable":
            lines.append("  " + result["reason"])
            continue
        lines.append(
            f"  common_prefix={result['common_prefix_tokens']} first_diff={result['first_difference_0based']} "
            f"left_token={result['left_token']} right_token={result['right_token']} "
            f"lengths={result['left_output_tokens']}/{result['right_output_tokens']} "
            f"max_shared_logprob_delta={result['max_shared_logprob_delta_before_feedback']} "
            f"delta_position={result['max_delta_position_0based']}"
        )
        first = result["first_difference_0based"]
        centers = {
            first,
            result["max_delta_position_0based"],
            result["first_shared_logprob_change_0based"],
            result["first_candidate_set_change_0based"],
        }
        lines.append(
            f"  first_shared_logprob_change={result['first_shared_logprob_change_0based']} "
            f"first_candidate_set_change={result['first_candidate_set_change_0based']} "
            f"returned_topk_equal={result['returned_topk_equal_before_feedback']} "
            f"compared_logprob_rows={result['compared_logprob_rows']}"
        )
        if result["same_history_probability_changes"]:
            lines.append("  same_history_changes=" + json.dumps(result["same_history_probability_changes"]))
        for sample in result["logprob_window"]:
            if sample["position_0based"] in centers:
                lines.append(f"  probability_position={sample['position_0based']}")
                lines.append("  left_logprobs=" + json.dumps(sample["left_logprobs"], ensure_ascii=False))
                lines.append("  right_logprobs=" + json.dumps(sample["right_logprobs"], ensure_ascii=False))
    lines.extend(
        [
            "",
            *["WARNING: " + warning for warning in report["warnings"]],
            "",
            "完整输出文本/token IDs及首差异上下文在 precision_details.json；未保留完整prompt或全量logprobs。",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_precision_summary(root):
    """只写精简报告，不覆盖原始prompt、输出和全量logprobs。"""
    report = _summarize_results(root)
    output = Path(report["run_dir"]) / "precision_compact"
    output.mkdir(parents=True, exist_ok=True)
    details_path = output / "precision_details.json"
    text_path = output / "precision_summary.txt"
    details_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = _render_precision_summary(report)
    text_path.write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"[simple-dp-precision] REPORT_READY summary={text_path}", flush=True)
    print(f"[simple-dp-precision] details={details_path}; originals unchanged", flush=True)


def main():
    if SUMMARY_ONLY_DIR:
        _write_precision_summary(SUMMARY_ONLY_DIR)
        return
    c = _settings()
    _validate(c)
    prompts = list(c["PROMPTS"])
    _preflight(c)
    run_dir = Path(c["RESULT_DIR"]).resolve() / f"simple-dp-{uuid.uuid4().hex}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "settings.json").write_text(json.dumps(c, default=str, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "llm-kwargs.json").write_text(json.dumps(_build_kwargs(c), indent=2), encoding="utf-8")
    print(
        f"[simple-dp] script={Path(__file__).resolve()} dp={c['DATA_PARALLEL_SIZE']} tp={c['TENSOR_PARALLEL_SIZE']} "
        f"mode={c['RUN_MODE']} dsa={c['ENABLE_DSA_OFFLOAD']} prefix={c['ENABLE_PREFIX_CACHING']} "
        f"mtp={c['ENABLE_MTP']} l3_write={c['ENABLE_L3_WRITE_BEHIND']} l3_load={c['ENABLE_L3_PREFIX_LOADBACK']} "
        f"rounds={c['REPLAY_ROUNDS']} distribution={c['PROMPT_DISTRIBUTION']} results={run_dir}",
        flush=True,
    )
    if c["CHECK_ONLY"]:
        print("[simple-dp] CONFIG CHECK COMPLETED; no model loaded", flush=True)
        return
    os.environ.update(
        PYTHONHASHSEED="114514",
        ASCEND_VISIBLE_DEVICES=c["VISIBLE_DEVICES"],
        ASCEND_RT_VISIBLE_DEVICES=c["VISIBLE_DEVICES"],
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", c["DP_MASTER_PORT"]))
        port = sock.getsockname()[1]
    context = mp.get_context("spawn")
    barrier = context.Barrier(c["DATA_PARALLEL_SIZE"])
    processes = []
    try:
        for rank in range(c["DATA_PARALLEL_SIZE"]):
            process = context.Process(target=_worker, args=(rank, c, prompts, port, barrier, run_dir))
            process.start()
            processes.append(process)
        deadline = time.monotonic() + c["PROCESS_TIMEOUT_SECONDS"]
        while any(process.is_alive() for process in processes):
            if any(process.exitcode not in (None, 0) for process in processes):
                raise RuntimeError(f"offline client failed; inspect {run_dir}/dp*-failure.txt")
            if time.monotonic() >= deadline:
                raise TimeoutError("offline DP run exceeded PROCESS_TIMEOUT_SECONDS")
            time.sleep(0.2)
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError(f"offline client exit codes: {[process.exitcode for process in processes]}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
    # 主进程统一打印完整输出，避免两个DP的长文本在终端互相穿插。
    max_prompt_tokens = 0
    for rank in range(c["DATA_PARALLEL_SIZE"]):
        for line in (run_dir / f"dp{rank}.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            max_prompt_tokens = max(max_prompt_tokens, record["prompt_tokens"])
            print(
                f"\nDP {rank} / round {record['round']} / prompt {record['prompt_index']}: "
                f"cached={record['cached_tokens']} finish_reason={record['finish_reason']!r}"
            )
            if c["PRINT_TOKEN_IDS"]:
                print(f"token_ids={record['token_ids']}")
            print(f"Output: {record['text']!r}")
    if c["ENABLE_DSA_OFFLOAD"] and max_prompt_tokens <= (
        8192 if c["ENABLE_MTP"] else c["DSA_SPARSE_ACTIVATION_TOKENS"]
    ):
        print("[simple-dp] all prompts are below the sparse activation threshold; use longer inputs to test sparse IO")
    _write_precision_summary(run_dir)
    print(f"\n[simple-dp] COMPLETED results={run_dir}; accuracy and graceful worker exit are not asserted")


if __name__ == "__main__":
    main()
