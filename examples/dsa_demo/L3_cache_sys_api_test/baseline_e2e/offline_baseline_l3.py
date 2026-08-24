#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSA-disabled offline cold/warm smoke for AscendStoreConnector."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from baseline_prompt import build_prompt

# =========================
# User configuration
# =========================
MODEL_PATH = "/mnt/kv_dpc/weight/GLM-5.1-w4a4c8"
L3_BACKEND = "memcache"  # memcache / mooncake
TENSOR_PARALLEL_SIZE = 8
ASCEND_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
MOONCAKE_COMM_MODE = "hixlep"  # hixlep / auto / uboe
HIXLEP_CONFIG_PATH = Path("/etc/hixlep")
MAX_MODEL_LEN = 8192
MAX_NUM_BATCHED_TOKENS = 8192
GPU_MEMORY_UTILIZATION = 0.90
PROMPT_REPETITIONS = 48
MAX_TOKENS = 1
REPLAY_DELAY_SECONDS = 10.0

SCRIPT_DIR = Path(__file__).resolve().parent
MOONCAKE_CONFIG_PATH = SCRIPT_DIR / "configs/mooncake.json"
MMC_LOCAL_CONFIG_PATH = SCRIPT_DIR / "configs/mmc-local.conf"
MEMFABRIC_AICPU_CHECK = SCRIPT_DIR.parent / "test_memfabric_aicpu.py"


def configure_environment() -> tuple[str, int]:
    os.environ.update(
        {
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_BUFFSIZE": "200",
            "OMP_NUM_THREADS": "10",
            "OMP_PROC_BIND": "false",
            "PYTHONHASHSEED": "114514",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "VLLM_ASCEND_ENABLE_MLAPO": "1",
            "VLLM_LOGGING_LEVEL": "INFO",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "ASCEND_RT_VISIBLE_DEVICES": ASCEND_VISIBLE_DEVICES,
        }
    )
    if L3_BACKEND == "mooncake":
        os.environ["MOONCAKE_CONFIG_PATH"] = str(MOONCAKE_CONFIG_PATH)
        os.environ["HCCL_RDMA_TIMEOUT"] = "17"
        os.environ["ASCEND_CONNECT_TIMEOUT"] = "30000"
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = "30000"
        if MOONCAKE_COMM_MODE in ("hixlep", "auto"):
            os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
            os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        elif MOONCAKE_COMM_MODE == "uboe":
            os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
            os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] = (
                '{"comm_resource_config.protocol_desc":["uboe:device"]}'
            )
        else:
            raise ValueError(
                "MOONCAKE_COMM_MODE must be hixlep/auto/uboe, "
                f"got {MOONCAKE_COMM_MODE!r}"
            )
        os.environ.pop("MMC_LOCAL_CONFIG_PATH", None)
        return "127.0.0.1", 50088
    if L3_BACKEND == "memcache":
        os.environ["MMC_LOCAL_CONFIG_PATH"] = str(MMC_LOCAL_CONFIG_PATH)
        os.environ.pop("MOONCAKE_CONFIG_PATH", None)
        os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
        os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        return "127.0.0.1", 5000
    raise ValueError(f"L3_BACKEND must be mooncake or memcache, got {L3_BACKEND!r}")


def require_service(host: str, port: int) -> None:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"L3 control service is not reachable at {host}:{port}; "
            "start it from baseline_e2e first"
        ) from exc


def require_memcache_aicpu() -> None:
    if L3_BACKEND != "memcache":
        return
    subprocess.run(
        [sys.executable, str(MEMFABRIC_AICPU_CHECK), "--mode", "check"],
        check=True,
    )


def require_hixlep_config() -> None:
    if L3_BACKEND != "mooncake" or MOONCAKE_COMM_MODE != "hixlep":
        return
    visible_devices = [
        int(item.strip())
        for item in ASCEND_VISIBLE_DEVICES.split(",")
        if item.strip()
    ]
    invalid: list[str] = []
    for device_id in visible_devices:
        config_file = HIXLEP_CONFIG_PATH / f"ub_endpoint_npu_{device_id}.json"
        try:
            payload = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid.append(f"{config_file}: {exc}")
            continue
        if not isinstance(payload, dict) or not isinstance(
            payload.get("endpoint_list"), list
        ) or not payload["endpoint_list"]:
            invalid.append(f"{config_file}: missing non-empty endpoint_list")
    if invalid:
        raise RuntimeError(
            "A5 TP Mooncake requires valid per-device HiXLEP endpoint files; "
            f"invalid={invalid}. Run ../run_l3_cache_api_tests.sh "
            "hixlep-config before starting vLLM."
        )


def main() -> None:
    host, port = configure_environment()
    require_memcache_aicpu()
    require_service(host, port)
    require_hixlep_config()

    # Environment must be complete before the first vLLM import.
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    extra_config: dict[str, object] = {
        "lookup_rpc_port": "1",
        "backend": L3_BACKEND,
        "load_async": False,
        "use_layerwise": False,
    }
    if L3_BACKEND == "mooncake" and MOONCAKE_COMM_MODE == "hixlep":
        extra_config["ascend_local_comm_res_path"] = str(HIXLEP_CONFIG_PATH)
    kv_transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_role="kv_both",
        kv_load_failure_policy="fail",
        kv_connector_extra_config=extra_config,
    )
    backend_detail = (
        f"mooncake_comm_mode={MOONCAKE_COMM_MODE}"
        if L3_BACKEND == "mooncake"
        else f"memcache_config={MMC_LOCAL_CONFIG_PATH}"
    )
    print(
        "[baseline-l3-offline] DSA disabled "
        f"backend={L3_BACKEND} model={MODEL_PATH} {backend_detail}"
    )
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        quantization="ascend",
        safetensors_load_strategy="prefetch",
        seed=1024,
        enable_expert_parallel=True,
        max_num_seqs=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        trust_remote_code=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        block_size=128,
        async_scheduling=False,
        enforce_eager=True,
        disable_log_stats=False,
        kv_transfer_config=kv_transfer_config,
    )

    prompt = build_prompt(PROMPT_REPETITIONS)
    prompt_tokens = len(llm.get_tokenizer().encode(prompt))
    if prompt_tokens >= MAX_MODEL_LEN:
        raise ValueError(
            f"prompt_tokens={prompt_tokens} must be smaller than max_model_len={MAX_MODEL_LEN}"
        )
    if prompt_tokens < 256:
        raise ValueError(f"prompt is too short to exercise multiple cache blocks: {prompt_tokens}")
    print(
        f"[baseline-l3-offline] prompt_tokens={prompt_tokens} "
        f"replay_delay={REPLAY_DELAY_SECONDS}s"
    )

    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    records: list[dict[str, object]] = []
    for round_index in range(2):
        if round_index:
            time.sleep(REPLAY_DELAY_SECONDS)
        start = time.perf_counter()
        output = llm.generate([prompt], sampling, use_tqdm=False)[0].outputs[0]
        elapsed_ms = (time.perf_counter() - start) * 1000
        record = {
            "round": round_index + 1,
            "token_ids": list(output.token_ids),
            "text": output.text,
            "finish_reason": output.finish_reason,
            "latency_ms": round(elapsed_ms, 3),
        }
        records.append(record)
        print(f"[baseline-l3-offline] result={record}")

    if records[0]["token_ids"] != records[1]["token_ids"]:
        raise AssertionError(f"cold/warm token IDs differ: {records}")
    print(
        "[baseline-l3-offline] OUTPUT CONSISTENCY PASSED; L3 acceptance still "
        "requires successful put/get logs and a nonzero external hit"
    )
    del llm


if __name__ == "__main__":
    main()
