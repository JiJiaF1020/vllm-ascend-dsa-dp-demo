#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# P6-A单机八卡DP2/TP4/EP8：固定MTP3 + Local prefix + Mooncake TCP。
# 仅设置阶段默认值，实际vLLM参数统一由serve_dsa.sh生成；Master需提前启动。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/mnt/share/weights/GLM-5.2-w4a4c8-mxfp4}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2-dsa}"
export PORT="${PORT:-8035}"
export RUN_MODE="${RUN_MODE:-graph}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DATA_PARALLEL_SIZE=2 DATA_PARALLEL_SIZE_LOCAL=2 TENSOR_PARALLEL_SIZE=4
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export DSA_INDEXER_MLA_BLOCK_RATIO="${DSA_INDEXER_MLA_BLOCK_RATIO:-2}"
export DSA_HOT_CPU_BLOCK_MULTIPLE="${DSA_HOT_CPU_BLOCK_MULTIPLE:-1.0}"
export ENABLE_CHUNKED_PREFILL=true ENABLE_A5_PACKED_C8_DSA=true
export ENABLE_PREFIX_CACHING=true ENABLE_MTP=true
export ENABLE_L3_WRITE_BEHIND=true ENABLE_L3_PREFIX_LOADBACK=true
export ENABLE_L3_DIRECT_HBM_LOAD=false

echo "[dsa-p6a] DP2/TP4/EP8 MTP3 + Mooncake TCP; target=${RUN_MODE}, proposer=eager"
exec bash "${SCRIPT_DIR}/serve_dsa.sh"
