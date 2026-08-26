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
MODEL_PATH = "/models/GLM-5.2-w4a4c8-mxfp4"
L3_BACKEND = "mooncake"  # memcache / mooncake / none
ENABLE_PREFIX_CACHING = False
TENSOR_PARALLEL_SIZE = 8
ASCEND_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
MOONCAKE_COMM_MODE = "global_ub_device"
MOONCAKE_MAX_ACTIVE_CHANNELS = 8
HIXLEP_CONFIG_PATH = Path("/etc/hixlep")
MEMCACHE_LOCAL_EID_MAP_PATH: Path | None = Path("/tmp/memcache-local-eids.json")
MAX_MODEL_LEN = 8192
MAX_NUM_BATCHED_TOKENS = 8192
GPU_MEMORY_UTILIZATION = 0.90
PROMPT_REPETITIONS = 48
MAX_TOKENS = 1
TOP_LOGPROBS = 5
REPLAY_ROUNDS = 3
REPLAY_DELAY_SECONDS = 10.0
SHUTDOWN_TIMEOUT_SECONDS = 30

SCRIPT_DIR = Path(__file__).resolve().parent
MOONCAKE_CONFIG_PATH = SCRIPT_DIR / "configs/mooncake.json"
MMC_LOCAL_CONFIG_PATH = SCRIPT_DIR / "configs/mmc-local.conf"
MEMFABRIC_AICPU_CHECK = SCRIPT_DIR.parent / "test_memfabric_aicpu.py"


def configure_environment() -> tuple[str, int] | None:
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
            "VLLM_HOST_IP": "127.0.0.1",
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
        elif MOONCAKE_COMM_MODE in ("global_ub", "global_ub_device"):
            os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
            protocol_desc = ["ub_ctp:device"]
            if MOONCAKE_COMM_MODE == "global_ub":
                protocol_desc.append("ub_ctp:host")
            os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] = json.dumps(
                {
                    "comm_resource_config.protocol_desc": protocol_desc,
                    "comm_resource_config.max_active_channels": (
                        MOONCAKE_MAX_ACTIVE_CHANNELS
                    ),
                },
                separators=(",", ":"),
            )
        elif MOONCAKE_COMM_MODE == "uboe":
            os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
            os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] = (
                '{"comm_resource_config.protocol_desc":["uboe:device"]}'
            )
        else:
            raise ValueError(
                "MOONCAKE_COMM_MODE must be global_ub_device/global_ub/"
                "hixlep/auto/uboe, "
                f"got {MOONCAKE_COMM_MODE!r}"
            )
        os.environ.pop("MMC_LOCAL_CONFIG_PATH", None)
        return "127.0.0.1", 50088
    if L3_BACKEND == "memcache":
        os.environ["MMC_LOCAL_CONFIG_PATH"] = str(MMC_LOCAL_CONFIG_PATH)
        os.environ.pop("USE_LOCAL_EID", None)
        os.environ.pop("MOONCAKE_CONFIG_PATH", None)
        os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
        os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        return "127.0.0.1", 5000
    if L3_BACKEND == "none":
        os.environ.pop("USE_LOCAL_EID", None)
        os.environ.pop("MMC_LOCAL_CONFIG_PATH", None)
        os.environ.pop("MOONCAKE_CONFIG_PATH", None)
        os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
        os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        return None
    raise ValueError(
        f"L3_BACKEND must be mooncake, memcache or none, got {L3_BACKEND!r}"
    )


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


def require_memcache_local_eid_map() -> None:
    if L3_BACKEND != "memcache" or MEMCACHE_LOCAL_EID_MAP_PATH is None:
        return
    if not MEMCACHE_LOCAL_EID_MAP_PATH.is_file():
        raise FileNotFoundError(
            f"MemCache local EID map not found: {MEMCACHE_LOCAL_EID_MAP_PATH}. "
            "Create it from ../CURRENT_TEST_CHECKLIST.md before starting vLLM."
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


def summarize_first_token_logprobs(output) -> list[dict[str, object]]:
    if not output.logprobs:
        return []
    first_position = output.logprobs[0]
    return sorted(
        (
            {
                "token_id": int(token_id),
                "logprob": round(float(logprob.logprob), 7),
                "rank": logprob.rank,
                "decoded_token": logprob.decoded_token,
            }
            for token_id, logprob in first_position.items()
        ),
        key=lambda item: float(item["logprob"]),
        reverse=True,
    )


def output_signature(record: dict[str, object]) -> tuple[tuple[int, ...], str]:
    token_ids = tuple(int(token_id) for token_id in record["token_ids"])
    logprobs = json.dumps(
        record["first_token_logprobs"],
        ensure_ascii=False,
        sort_keys=True,
    )
    return token_ids, logprobs


def main() -> None:
    if REPLAY_ROUNDS < 2:
        raise ValueError("REPLAY_ROUNDS must be at least 2")
    service = configure_environment()
    if L3_BACKEND != "none" and ENABLE_PREFIX_CACHING:
        raise ValueError(
            "Disable HBM prefix caching while testing an external L3 backend; "
            "use L3_BACKEND='none' for the HBM-prefix control."
        )
    require_memcache_aicpu()
    require_memcache_local_eid_map()
    if service is not None:
        require_service(*service)
    require_hixlep_config()

    # Environment must be complete before the first vLLM import.
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kv_transfer_config = None
    if L3_BACKEND != "none":
        extra_config: dict[str, object] = {
            "lookup_rpc_port": "1",
            "backend": L3_BACKEND,
            "load_async": False,
            "use_layerwise": False,
            "log_transfer_summary": True,
        }
        if L3_BACKEND == "mooncake" and MOONCAKE_COMM_MODE == "hixlep":
            extra_config["ascend_local_comm_res_path"] = str(HIXLEP_CONFIG_PATH)
        if L3_BACKEND == "memcache" and MEMCACHE_LOCAL_EID_MAP_PATH is not None:
            extra_config["memcache_local_eid_map_path"] = str(
                MEMCACHE_LOCAL_EID_MAP_PATH
            )
        kv_transfer_config = KVTransferConfig(
            kv_connector="AscendStoreConnector",
            kv_role="kv_both",
            kv_load_failure_policy="fail",
            kv_connector_extra_config=extra_config,
        )
    if L3_BACKEND == "mooncake":
        backend_detail = f"mooncake_comm_mode={MOONCAKE_COMM_MODE}"
    elif L3_BACKEND == "memcache":
        backend_detail = (
            f"memcache_config={MMC_LOCAL_CONFIG_PATH} "
            f"local_eid_map={MEMCACHE_LOCAL_EID_MAP_PATH}"
        )
    else:
        backend_detail = f"hbm_prefix_caching={ENABLE_PREFIX_CACHING}"
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
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
        enable_chunked_prefill=False,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        block_size=128,
        async_scheduling=False,
        enforce_eager=True,
        disable_log_stats=False,
        shutdown_timeout=SHUTDOWN_TIMEOUT_SECONDS,
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

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=TOP_LOGPROBS,
    )
    records: list[dict[str, object]] = []
    for round_index in range(REPLAY_ROUNDS):
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
            "first_token_logprobs": summarize_first_token_logprobs(output),
        }
        records.append(record)
        print(f"[baseline-l3-offline] result={record}")

    signatures = [output_signature(record) for record in records]
    all_rounds_match = len(set(signatures)) == 1
    warm_replays_stable = len(set(signatures[1:])) == 1
    if not warm_replays_stable:
        raise AssertionError(
            "warm replay outputs are unstable across identical cache-hit requests: "
            f"{records}"
        )
    if not all_rounds_match:
        cache_replay_enabled = L3_BACKEND != "none" or ENABLE_PREFIX_CACHING
        if not cache_replay_enabled:
            raise AssertionError(f"full-recompute outputs differ: {records}")
        print(
            "[baseline-l3-offline] WARM REPLAY CONSISTENCY PASSED; cold/warm "
            "outputs differ. Track this as a model prefix-path precision issue; "
            "judge L3 separately from external-hit and put/get evidence."
        )
    else:
        print("[baseline-l3-offline] OUTPUT CONSISTENCY PASSED")
    if L3_BACKEND != "none":
        print(
            "[baseline-l3-offline] External L3 acceptance also requires "
            "successful put/get logs and a nonzero external hit"
        )
    del llm


if __name__ == "__main__":
    main()
