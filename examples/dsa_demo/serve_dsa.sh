#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# GLM-5.x DSA 稀疏卸载在线服务最小启动脚本。
# 修改“用户配置”后直接执行：
#
#   bash examples/dsa_demo/serve_dsa.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# 用户配置
# =========================

MODEL_PATH="${MODEL_PATH:-/mnt/kv_dpc/weight/GLM-5.1-w4a8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.1-dsa}"
RUN_MODE="${RUN_MODE:-eager}"  # disabled / eager / graph

# A5/950 的 W4A4C8 模型设为 true；A3/910C 的 W4A8/BF16 保持 false。
# A5 DSA 只支持 LI C8 与 SFA C8 同时开启，不提供半开组合。
ENABLE_A5_PACKED_C8_DSA="${ENABLE_A5_PACKED_C8_DSA:-false}"
ENABLE_MTP="${ENABLE_MTP:-false}"
ENABLE_L3_WRITE_BEHIND="${ENABLE_L3_WRITE_BEHIND:-false}"
ENABLE_L3_PREFIX_LOADBACK="${ENABLE_L3_PREFIX_LOADBACK:-false}"
ENABLE_L3_DIRECT_HBM_LOAD="${ENABLE_L3_DIRECT_HBM_LOAD:-false}"
MOONCAKE_CONFIG_PATH="${SCRIPT_DIR}/L3_cache_sys_api_test/baseline_e2e/configs/mooncake-dsa-tcp.json"
MOONCAKE_GLOBAL_SEGMENT_SIZE="4GB"
MC_TCP_ENABLE_CONNECTION_POOL="${MC_TCP_ENABLE_CONNECTION_POOL:-1}"
# 同一 Mooncake Master 上的并行服务必须使用不同 base。
MOONCAKE_TCP_SEGMENT_PORT_BASE="${MOONCAKE_TCP_SEGMENT_PORT_BASE:-45000}"
DSA_L3_MAX_INFLIGHT_STORE_JOBS="8"
DSA_L3_MAX_ACTIVE_CHANNELS="8"
DSA_L3_MAX_DESCRIPTORS_PER_PUT="128"
DSA_L3_SHUTDOWN_TIMEOUT_SECONDS="300"
SAFETENSORS_LOAD_STRATEGY="prefetch"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-EMPTY}"

DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE_LOCAL="${DATA_PARALLEL_SIZE_LOCAL:-${DATA_PARALLEL_SIZE}}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-true}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
LOGGING_LEVEL="${LOGGING_LEVEL:-INFO}"

DSA_SPARSE_ACTIVATION_TOKENS="${DSA_SPARSE_ACTIVATION_TOKENS:-6144}"
DSA_INDEXER_MLA_BLOCK_RATIO="${DSA_INDEXER_MLA_BLOCK_RATIO:-3}"
DSA_MAX_ACTIVE_REQS="${DSA_MAX_ACTIVE_REQS:-256}"
DSA_HOT_CPU_BLOCK_MULTIPLE="${DSA_HOT_CPU_BLOCK_MULTIPLE:-3.0}"

export HCCL_OP_EXPANSION_MODE="AIV"
export HCCL_BUFFSIZE="512"
export OMP_NUM_THREADS="10"
export OMP_PROC_BIND="false"
export PYTHONHASHSEED="114514"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_ASCEND_ENABLE_MLAPO="1"
export VLLM_LOGGING_LEVEL="${LOGGING_LEVEL}"

case "${RUN_MODE}" in
    disabled)
        DSA_ENABLED="false"
        ENABLE_DSA_GRAPH="false"
        EXECUTION_ARGS=(
            --enforce-eager
        )
        ;;
    eager)
        DSA_ENABLED="true"
        ENABLE_DSA_GRAPH="false"
        EXECUTION_ARGS=(
            --enforce-eager
        )
        ;;
    graph)
        DSA_ENABLED="true"
        ENABLE_DSA_GRAPH="true"
        if [[ "${ENABLE_MTP}" == "true" ]]; then
            GRAPH_CAPTURE_SIZES='[4,8,16]'
        else
            GRAPH_CAPTURE_SIZES='[1,2,4]'
        fi
        EXECUTION_ARGS=(
            --no-enforce-eager
            --compilation-config
            "{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":${GRAPH_CAPTURE_SIZES}}"
        )
        ;;
    *)
        echo "RUN_MODE must be disabled, eager or graph, got: ${RUN_MODE}" >&2
        exit 2
        ;;
esac

case "${ENABLE_MTP}" in
    true)
        if [[ "${ENABLE_A5_PACKED_C8_DSA}" != "true" ]]; then
            echo "DSA MTP3 requires ENABLE_A5_PACKED_C8_DSA=true" >&2
            exit 2
        fi
        DSA_SPARSE_ACTIVATION_TOKENS="8192"
        DSA_RESIDENT_BUDGET_TOKENS='[8192,10240,12288]'
        SPECULATIVE_ARGS=(
            --speculative-config
            '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
        )
        ;;
    false)
        DSA_RESIDENT_BUDGET_TOKENS='[6144,10240,12288]'
        SPECULATIVE_ARGS=()
        ;;
    *)
        echo "ENABLE_MTP must be true or false, got: ${ENABLE_MTP}" >&2
        exit 2
        ;;
esac

case "${ENABLE_L3_PREFIX_LOADBACK}" in
    true)
        if [[ "${ENABLE_L3_WRITE_BEHIND}" != "true" ]]; then
            echo "ENABLE_L3_PREFIX_LOADBACK requires ENABLE_L3_WRITE_BEHIND=true" >&2
            exit 2
        fi
        ;;
    false)
        ;;
    *)
        echo "ENABLE_L3_PREFIX_LOADBACK must be true or false, got: ${ENABLE_L3_PREFIX_LOADBACK}" >&2
        exit 2
        ;;
esac

case "${ENABLE_L3_DIRECT_HBM_LOAD}" in
    true)
        echo "ENABLE_L3_DIRECT_HBM_LOAD is frozen; use L3->L2->HBM" >&2
        exit 2
        ;;
    false)
        ;;
    *)
        echo "ENABLE_L3_DIRECT_HBM_LOAD must be true or false, got: ${ENABLE_L3_DIRECT_HBM_LOAD}" >&2
        exit 2
        ;;
esac

case "${ENABLE_L3_WRITE_BEHIND}" in
    true)
        if [[ "${DSA_ENABLED}" != "true" ]]; then
            echo "ENABLE_L3_WRITE_BEHIND requires RUN_MODE=eager or graph" >&2
            exit 2
        fi
        if [[ ! -f "${MOONCAKE_CONFIG_PATH}" ]]; then
            echo "Mooncake config does not exist: ${MOONCAKE_CONFIG_PATH}" >&2
            exit 2
        fi
        if [[ "${ENABLE_L3_PREFIX_LOADBACK}" == "true" && "${ENABLE_PREFIX_CACHING}" != "true" ]]; then
            echo "ENABLE_L3_PREFIX_LOADBACK requires ENABLE_PREFIX_CACHING=true" >&2
            exit 2
        fi
        if [[ "${ENABLE_L3_PREFIX_LOADBACK}" == "true" ]]; then
            DSA_L3_KV_ROLE="kv_both"
            # 仅测试环境启用 vLLM 原生 cache reset endpoint。生产部署应关闭。
            export VLLM_SERVER_DEV_MODE=1
        else
            DSA_L3_KV_ROLE="kv_producer"
        fi
        export MOONCAKE_CONFIG_PATH
        export MOONCAKE_GLOBAL_SEGMENT_SIZE
        export MC_TCP_ENABLE_CONNECTION_POOL
        export MOONCAKE_TCP_SEGMENT_PORT_BASE
        unset ASCEND_GLOBAL_RESOURCE_CONFIG || true
        KV_TRANSFER_ARGS=(
            --shutdown-timeout
            "${DSA_L3_SHUTDOWN_TIMEOUT_SECONDS}"
            --kv-transfer-config
            "{\"kv_connector\":\"DSAL3Connector\",\"kv_role\":\"${DSA_L3_KV_ROLE}\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"backend\":\"mooncake\",\"max_inflight_l3_store_jobs\":${DSA_L3_MAX_INFLIGHT_STORE_JOBS},\"max_active_channels\":${DSA_L3_MAX_ACTIVE_CHANNELS},\"max_descriptors_per_l3_put\":${DSA_L3_MAX_DESCRIPTORS_PER_PUT},\"direct_hbm_load\":${ENABLE_L3_DIRECT_HBM_LOAD}}}"
        )
        ;;
    false)
        KV_TRANSFER_ARGS=()
        ;;
    *)
        echo "ENABLE_L3_WRITE_BEHIND must be true or false, got: ${ENABLE_L3_WRITE_BEHIND}" >&2
        exit 2
        ;;
esac

case "${ENABLE_A5_PACKED_C8_DSA}" in
    true)
        A5_PACKED_CONFIG_FIELDS='"enable_sparse_sfa_c8":true,"enable_sparse_li_c8":true'
        ;;
    false)
        A5_PACKED_CONFIG_FIELDS=""
        ;;
    *)
        echo "ENABLE_A5_PACKED_C8_DSA must be true or false, got: ${ENABLE_A5_PACKED_C8_DSA}" >&2
        exit 2
        ;;
esac

if [[ "${ENABLE_CHUNKED_PREFILL}" == "true" ]]; then
    PREFILL_ARGS=(
        --enable-chunked-prefill
        --scheduler-reserve-full-isl
    )
else
    PREFILL_ARGS=(
        --no-enable-chunked-prefill
    )
fi

case "${ENABLE_PREFIX_CACHING}" in
    true)
        PREFIX_CACHE_ARGS=(--enable-prefix-caching)
        ;;
    false)
        PREFIX_CACHE_ARGS=(--no-enable-prefix-caching)
        ;;
    *)
        echo "ENABLE_PREFIX_CACHING must be true or false, got: ${ENABLE_PREFIX_CACHING}" >&2
        exit 2
        ;;
esac

if [[ "${DSA_ENABLED}" == "true" && "${DATA_PARALLEL_SIZE}" -gt 1 ]]; then
    if [[ "${DATA_PARALLEL_SIZE_LOCAL}" != "${DATA_PARALLEL_SIZE}" ]]; then
        echo "Initial online DSA DP support requires all DP ranks on one node" >&2
        exit 2
    fi
fi

if [[ "${DSA_ENABLED}" == "true" ]]; then
    DSA_CONFIG_FIELD='"dsa_sparse_config":{"enabled":true,"split_indexer_cache":true,'
    DSA_CONFIG_FIELD+='"indexer_mla_block_ratio":'"${DSA_INDEXER_MLA_BLOCK_RATIO}"','
    DSA_CONFIG_FIELD+='"sparse_activation_tokens":'"${DSA_SPARSE_ACTIVATION_TOKENS}"','
    DSA_CONFIG_FIELD+='"prompt_budget_thresholds":[32768,65536],'
    DSA_CONFIG_FIELD+='"resident_budget_tokens":'"${DSA_RESIDENT_BUDGET_TOKENS}"','
    DSA_CONFIG_FIELD+='"max_active_reqs":'"${DSA_MAX_ACTIVE_REQS}"','
    DSA_CONFIG_FIELD+='"hot_cpu_block_multiple":'"${DSA_HOT_CPU_BLOCK_MULTIPLE}"','
    DSA_CONFIG_FIELD+='"enable_row_mode_decode_graph":'"${ENABLE_DSA_GRAPH}"'}'
else
    DSA_CONFIG_FIELD=""
fi

if [[ -n "${A5_PACKED_CONFIG_FIELDS}" && -n "${DSA_CONFIG_FIELD}" ]]; then
    ADDITIONAL_CONFIG="{${A5_PACKED_CONFIG_FIELDS},${DSA_CONFIG_FIELD}}"
elif [[ -n "${A5_PACKED_CONFIG_FIELDS}" ]]; then
    ADDITIONAL_CONFIG="{${A5_PACKED_CONFIG_FIELDS}}"
elif [[ -n "${DSA_CONFIG_FIELD}" ]]; then
    ADDITIONAL_CONFIG="{${DSA_CONFIG_FIELD}}"
else
    ADDITIONAL_CONFIG="{}"
fi

echo "[dsa-online] mode=${RUN_MODE} model=${MODEL_PATH}"
echo "[dsa-online] dsa_enabled=${DSA_ENABLED} prefix_cache=${ENABLE_PREFIX_CACHING}"
echo "[dsa-online] a5_packed_c8=${ENABLE_A5_PACKED_C8_DSA} mtp3=${ENABLE_MTP} safetensors=${SAFETENSORS_LOAD_STRATEGY}"
echo "[dsa-online] dp=${DATA_PARALLEL_SIZE} local_dp=${DATA_PARALLEL_SIZE_LOCAL} tp=${TENSOR_PARALLEL_SIZE} max_num_seqs=${MAX_NUM_SEQS} max_model_len=${MAX_MODEL_LEN}"
echo "[dsa-online] max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} chunked_prefill=${ENABLE_CHUNKED_PREFILL} gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
echo "[dsa-online] l3_write_behind=${ENABLE_L3_WRITE_BEHIND} mooncake_config=${MOONCAKE_CONFIG_PATH}"
echo "[dsa-online] l3_prefix_loadback=${ENABLE_L3_PREFIX_LOADBACK}"
echo "[dsa-online] l3_direct_hbm_load=${ENABLE_L3_DIRECT_HBM_LOAD}"
if [[ "${ENABLE_L3_WRITE_BEHIND}" == "true" ]]; then
    echo "[dsa-online] mooncake_tcp_connection_pool=${MC_TCP_ENABLE_CONNECTION_POOL}"
    echo "[dsa-online] mooncake_tcp_segment_port_base=${MOONCAKE_TCP_SEGMENT_PORT_BASE}"
fi
echo "[dsa-online] endpoint=http://${HOST}:${PORT}/v1"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --api-key "${API_KEY}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --pipeline-parallel-size 1 \
    --data-parallel-size "${DATA_PARALLEL_SIZE}" \
    --data-parallel-size-local "${DATA_PARALLEL_SIZE_LOCAL}" \
    --quantization ascend \
    --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}" \
    --seed 1024 \
    --enable-expert-parallel \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --trust-remote-code \
    "${PREFIX_CACHE_ARGS[@]}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --block-size 128 \
    --no-async-scheduling \
    --enable-prompt-tokens-details \
    --stream-interval 1 \
    --generation-config vllm \
    --reasoning-parser glm45 \
    --additional-config "${ADDITIONAL_CONFIG}" \
    "${KV_TRANSFER_ARGS[@]}" \
    "${SPECULATIVE_ARGS[@]}" \
    "${PREFILL_ARGS[@]}" \
    "${EXECUTION_ARGS[@]}"
