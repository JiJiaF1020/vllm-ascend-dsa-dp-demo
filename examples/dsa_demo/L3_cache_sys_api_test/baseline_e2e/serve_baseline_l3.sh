#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start DSA-disabled vLLM with the native AscendStoreConnector.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(dirname "$SCRIPT_DIR")

# =========================
# User configuration
# =========================
MODEL_PATH="/mnt/kv_dpc/weight/GLM-5.1-w4a4c8"
SERVED_MODEL_NAME="glm-5.1-baseline-l3"
L3_BACKEND="memcache"  # memcache / mooncake

HOST="0.0.0.0"
PORT="8035"
API_KEY="EMPTY"
TENSOR_PARALLEL_SIZE="8"
MAX_NUM_SEQS="1"
MAX_MODEL_LEN="8192"
MAX_NUM_BATCHED_TOKENS="8192"
GPU_MEMORY_UTILIZATION="0.90"
SAFETENSORS_LOAD_STRATEGY="prefetch"
ASCEND_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
MOONCAKE_COMM_MODE="hixlep"  # hixlep / auto / uboe
HIXLEP_CONFIG_PATH=${HIXLEP_CONFIG_PATH:-/etc/hixlep}

MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH:-$SCRIPT_DIR/configs/mooncake.json}
MMC_LOCAL_CONFIG_PATH=${MMC_LOCAL_CONFIG_PATH:-$SCRIPT_DIR/configs/mmc-local.conf}

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE="200"
export OMP_NUM_THREADS="10"
export OMP_PROC_BIND="false"
export PYTHONHASHSEED="114514"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_ENABLE_MLAPO="1"
export VLLM_LOGGING_LEVEL="INFO"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_VISIBLE_DEVICES"

case "$L3_BACKEND" in
    mooncake)
        export MOONCAKE_CONFIG_PATH
        export HCCL_RDMA_TIMEOUT="17"
        export ASCEND_CONNECT_TIMEOUT="30000"
        export ASCEND_TRANSFER_TIMEOUT="30000"
        case "$MOONCAKE_COMM_MODE" in
            hixlep)
                unset ASCEND_LOCAL_COMM_RES || true
                unset ASCEND_GLOBAL_RESOURCE_CONFIG || true
                ASCEND_RT_VISIBLE_DEVICES="$ASCEND_VISIBLE_DEVICES" \
                    HIXLEP_CONFIG_PATH="$HIXLEP_CONFIG_PATH" \
                    bash "$PARENT_DIR/run_l3_cache_api_tests.sh" hixlep-config
                ;;
            auto)
                unset ASCEND_LOCAL_COMM_RES || true
                unset ASCEND_GLOBAL_RESOURCE_CONFIG || true
                ;;
            uboe)
                unset ASCEND_LOCAL_COMM_RES || true
                export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["uboe:device"]}'
                ;;
            *)
                echo "MOONCAKE_COMM_MODE must be hixlep/auto/uboe, got: $MOONCAKE_COMM_MODE" >&2
                exit 2
                ;;
        esac
        unset MMC_LOCAL_CONFIG_PATH || true
        SERVICE_HOST="127.0.0.1"
        SERVICE_PORT="50088"
        bash "$PARENT_DIR/run_l3_cache_api_tests.sh" mooncake-runtime-import
        ;;
    memcache)
        export MMC_LOCAL_CONFIG_PATH
        unset MOONCAKE_CONFIG_PATH || true
        unset ASCEND_LOCAL_COMM_RES || true
        unset ASCEND_GLOBAL_RESOURCE_CONFIG || true
        SERVICE_HOST="127.0.0.1"
        SERVICE_PORT="5000"
        python "$PARENT_DIR/test_memfabric_aicpu.py" --mode check
        ;;
    *)
        echo "L3_BACKEND must be mooncake or memcache, got: $L3_BACKEND" >&2
        exit 2
        ;;
esac

if ! python - "$SERVICE_HOST" "$SERVICE_PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=1.0):
        pass
except OSError as exc:
    raise SystemExit(f"L3 control service is not reachable at {host}:{port}: {exc}")
PY
then
    echo "Start the matching L3 service before vLLM." >&2
    exit 2
fi

KV_TRANSFER_CONFIG=$(python - "$L3_BACKEND" "$MOONCAKE_COMM_MODE" "$HIXLEP_CONFIG_PATH" <<'PY'
import json
import sys

backend, mooncake_comm_mode, hixlep_config_path = sys.argv[1:]
extra_config = {
    "lookup_rpc_port": "1",
    "backend": backend,
    "load_async": False,
    "use_layerwise": False,
}
if backend == "mooncake" and mooncake_comm_mode == "hixlep":
    extra_config["ascend_local_comm_res_path"] = hixlep_config_path
print(json.dumps({
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "fail",
    "kv_connector_extra_config": extra_config,
}, separators=(",", ":")))
PY
)

echo "[baseline-l3] DSA disabled; no additional_config is passed"
echo "[baseline-l3] backend=$L3_BACKEND model=$MODEL_PATH"
echo "[baseline-l3] mooncake_comm_mode=$MOONCAKE_COMM_MODE"
echo "[baseline-l3] endpoint=http://127.0.0.1:$PORT/v1"
echo "[baseline-l3] kv_transfer_config=$KV_TRANSFER_CONFIG"

exec vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --quantization ascend \
    --safetensors-load-strategy "$SAFETENSORS_LOAD_STRATEGY" \
    --seed 1024 \
    --enable-expert-parallel \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --block-size 128 \
    --no-async-scheduling \
    --enforce-eager \
    --generation-config vllm \
    --kv-transfer-config "$KV_TRANSFER_CONFIG"
