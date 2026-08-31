#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the standalone L3 backend API gates without starting vLLM.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
LOG_DIR=${LOG_DIR:-/tmp/dsa_l3_cache_probe_logs}

DEVICE_ID=${DEVICE_ID:-0}
DEVICE_PROTOCOL=${DEVICE_PROTOCOL:-}
DRAM_SIZE=${DRAM_SIZE:-1GB}
MAX_DRAM_SIZE=${MAX_DRAM_SIZE:-256GB}
WORLD_SIZE=${WORLD_SIZE:-256}

MOONCAKE_LOCAL_HOST=${MOONCAKE_LOCAL_HOST:-127.0.0.1}
MOONCAKE_TP_LOCAL_HOST=${MOONCAKE_TP_LOCAL_HOST:-auto}
MOONCAKE_MASTER_ADDRESS=${MOONCAKE_MASTER_ADDRESS:-127.0.0.1:50088}
HIXLEP_CONFIG_PATH=${HIXLEP_CONFIG_PATH:-/etc/hixlep}
MEMCACHE_TP_CONFIG_PATH=${MEMCACHE_TP_CONFIG_PATH:-$SCRIPT_DIR/baseline_e2e/configs/mmc-local.conf}
MEMCACHE_TP_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
MEMCACHE_TP_ACTIVE_DEVICES=${MEMCACHE_TP_ACTIVE_DEVICES:-0,1}
MEMCACHE_TP_LOCAL_EID_MAP=${MEMCACHE_TP_LOCAL_EID_MAP:-}
MEMCACHE_TP_LOCAL_EID_MAP_FILE=${MEMCACHE_TP_LOCAL_EID_MAP_FILE:-}
MOONCAKE_TP_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
MOONCAKE_TP_ACTIVE_DEVICES=${MOONCAKE_TP_ACTIVE_DEVICES:-$MOONCAKE_TP_VISIBLE_DEVICES}
MOONCAKE_TP_COMM_MODE=${MOONCAKE_TP_COMM_MODE:-global_ub_device}
MOONCAKE_TP_MAX_ACTIVE_CHANNELS=${MOONCAKE_TP_MAX_ACTIVE_CHANNELS:-8}
MOONCAKE_TP_BUFFER_MODE=${MOONCAKE_TP_BUFFER_MODE:-device}
MOONCAKE_TP_SOURCE_BUFFER_MODE=${MOONCAKE_TP_SOURCE_BUFFER_MODE:-}
MOONCAKE_TP_DESTINATION_BUFFER_MODE=${MOONCAKE_TP_DESTINATION_BUFFER_MODE:-}
MOONCAKE_TP_NUM_BUFFERS=${MOONCAKE_TP_NUM_BUFFERS:-1}
MOONCAKE_TP_PAYLOAD_BYTES=${MOONCAKE_TP_PAYLOAD_BYTES:-1048576}
MOONCAKE_TP_PAYLOAD_LAYOUT=${MOONCAKE_TP_PAYLOAD_LAYOUT:-uniform}
MOONCAKE_TP_SEGMENT_BYTES=${MOONCAKE_TP_SEGMENT_BYTES:-1073741824}
MOONCAKE_TP_LOCAL_BUFFER_BYTES=${MOONCAKE_TP_LOCAL_BUFFER_BYTES:-1073741824}
MOONCAKE_TP_ARENA_BLOCKS=${MOONCAKE_TP_ARENA_BLOCKS:-1}
MOONCAKE_TP_BLOCK_INDEX=${MOONCAKE_TP_BLOCK_INDEX:-0}
MOONCAKE_TP_IO_THREAD=${MOONCAKE_TP_IO_THREAD:-main}
MOONCAKE_TP_PLACEMENT_MODE=${MOONCAKE_TP_PLACEMENT_MODE:-preferred}
MOONCAKE_TP_FORCE_REMOTE_PUT=${MOONCAKE_TP_FORCE_REMOTE_PUT:-false}
MOONCAKE_TP_PRODUCER_RANK=${MOONCAKE_TP_PRODUCER_RANK:-}
MOONCAKE_TP_CONSUMER_DEVICE=${MOONCAKE_TP_CONSUMER_DEVICE:-}
MOONCAKE_TP_TARGET_DEVICE=${MOONCAKE_TP_TARGET_DEVICE:-}
MOONCAKE_STORE_TCP_DEVICES=${MOONCAKE_STORE_TCP_DEVICES:-0,1}
MOONCAKE_STORE_TCP_SEGMENT_BYTES=${MOONCAKE_STORE_TCP_SEGMENT_BYTES:-134217728}
MOONCAKE_STORE_TCP_LOCAL_BUFFER_BYTES=${MOONCAKE_STORE_TCP_LOCAL_BUFFER_BYTES:-67108864}
MOONCAKE_STORE_TCP_NUM_PLANES=${MOONCAKE_STORE_TCP_NUM_PLANES:-4}
MOONCAKE_STORE_TCP_PLANE_BYTES=${MOONCAKE_STORE_TCP_PLANE_BYTES:-262144}
MOONCAKE_STORE_TCP_TIMEOUT=${MOONCAKE_STORE_TCP_TIMEOUT:-90}
DSA_SWAPPED_COMPONENT=${DSA_SWAPPED_COMPONENT:-all}
DSA_SWAPPED_SCHEMA_MODE=${DSA_SWAPPED_SCHEMA_MODE:-exact}
DSA_SWAPPED_DESTINATION_MEMORY=${DSA_SWAPPED_DESTINATION_MEMORY:-swapped}
DSA_SWAPPED_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-$DEVICE_ID}}
DSA_SWAPPED_MAX_ACTIVE_CHANNELS=${DSA_SWAPPED_MAX_ACTIVE_CHANNELS:-1}
DSA_SWAPPED_MOONCAKE_COMM_MODE=${DSA_SWAPPED_MOONCAKE_COMM_MODE:-global_ub}
DSA_SWAPPED_MEMCACHE_LOCAL_EID=${DSA_SWAPPED_MEMCACHE_LOCAL_EID:-${USE_LOCAL_EID:-}}

mkdir -p "$LOG_DIR"

usage() {
    cat <<'EOF'
Usage:
  bash run_l3_cache_api_tests.sh <gate>

Gates:
  memcache-capabilities  Check installed packages and vLLM-Ascend APIs.
  memcache-aicpu-check   Verify the HYBM AICPU kernel required by A5 URMA.
  memcache-aicpu-provision  Run wheel self-provisioning when assets exist.
  memcache-host          Test MetaService, host_shm, H2G/G2H and bitwise IO.
  memcache-npu           Test A5 HBM L2G/G2L scatter-gather IO.
  memcache-gva           Test layerwise/GVA allocation, copy and lease APIs.
  memcache-tp            Test A5 LocalServices' cross-rank URMA payload IO.
  memcache-all           Run capabilities, host, NPU and GVA gates in order.
  ascend-store-memcache  Test the real vLLM-Ascend MemcacheBackend.
  ascend-store-mooncake  Test the real vLLM-Ascend MooncakeBackend.
  ascend-store-mooncake-tcp  Test Store-owned TCP in MooncakeBackend.
  mooncake-capabilities  Check the NPU wheel, Store and TransferEngine APIs.
  mooncake-runtime-import  Check the vLLM-like torch_npu import order.
  mooncake-imports       Isolate store/engine imports and preserve exit codes.
  mooncake-linkage       Print glibc and ldd results for both extension modules.
  mooncake-gdb-store     Capture the store import abort under GDB.
  mooncake-gdb-engine    Capture the engine import abort under GDB.
  mooncake-npu           Run NPU multi-buffer IO only if imports are stable.
  mooncake-tp            Test TP workers' cross-segment NPU round trip.
  mooncake-store-tcp-inspect  Inspect Store TCP APIs and vLLM protocol locks.
  mooncake-store-tcp-single   Store-owned TCP with one swapped L2 worker.
  mooncake-store-tcp-cross    Store-owned TCP ring across 2+ swapped L2 workers.
  mooncake-store-tcp-all      Run the three Store-owned TCP gates in order.
  dsa-swapped-mooncake   Test one exact A5 DSA block in swapped memory.
  dsa-swapped-memcache   Run the same swapped-memory schema on MemCache.
  hixlep-config          Validate A5 per-device LocalCommRes JSON files.
  hixlep-tooling         Inspect installed LocalCommRes generation tools.
  hcomm-logs             Collect recent endpoint/channel/UB plog excerpts.
  diagnostics            Collect package, environment, NPU and topology data.
  safe                   Run checks that do not load Mooncake native modules.

Environment overrides:
  PYTHON_BIN, LOG_DIR, DEVICE_ID, DEVICE_PROTOCOL, DRAM_SIZE,
  MAX_DRAM_SIZE, WORLD_SIZE, MEMCACHE_TP_ACTIVE_DEVICES,
  MEMCACHE_TP_LOCAL_EID_MAP, MEMCACHE_TP_LOCAL_EID_MAP_FILE,
  MOONCAKE_LOCAL_HOST, MOONCAKE_TP_LOCAL_HOST,
  MOONCAKE_MASTER_ADDRESS, HIXLEP_CONFIG_PATH, MOONCAKE_TP_COMM_MODE,
  MOONCAKE_TP_ACTIVE_DEVICES, MOONCAKE_TP_MAX_ACTIVE_CHANNELS,
  MOONCAKE_TP_BUFFER_MODE, MOONCAKE_TP_SOURCE_BUFFER_MODE,
  MOONCAKE_TP_DESTINATION_BUFFER_MODE, MOONCAKE_TP_NUM_BUFFERS,
  MOONCAKE_TP_PAYLOAD_LAYOUT, MOONCAKE_TP_SEGMENT_BYTES,
  MOONCAKE_TP_LOCAL_BUFFER_BYTES,
  MOONCAKE_TP_PAYLOAD_BYTES, MOONCAKE_TP_ARENA_BLOCKS,
  MOONCAKE_TP_BLOCK_INDEX, MOONCAKE_TP_IO_THREAD,
  MOONCAKE_TP_PLACEMENT_MODE,
  MOONCAKE_TP_FORCE_REMOTE_PUT,
  MOONCAKE_TP_PRODUCER_RANK, MOONCAKE_TP_CONSUMER_DEVICE,
  MOONCAKE_TP_TARGET_DEVICE,
  MOONCAKE_STORE_TCP_DEVICES, MOONCAKE_STORE_TCP_SEGMENT_BYTES,
  MOONCAKE_STORE_TCP_LOCAL_BUFFER_BYTES, MOONCAKE_STORE_TCP_NUM_PLANES,
  MOONCAKE_STORE_TCP_PLANE_BYTES, MOONCAKE_STORE_TCP_TIMEOUT,
  ASCEND_BUFFER_POOL,
  DSA_SWAPPED_COMPONENT, DSA_SWAPPED_SCHEMA_MODE, DSA_SWAPPED_DESTINATION_MEMORY,
  DSA_SWAPPED_VISIBLE_DEVICES,
  DSA_SWAPPED_MAX_ACTIVE_CHANNELS, DSA_SWAPPED_MOONCAKE_COMM_MODE,
  DSA_SWAPPED_MEMCACHE_LOCAL_EID

Examples:
  bash run_l3_cache_api_tests.sh memcache-host
  DEVICE_ID=0 DEVICE_PROTOCOL=device_uboe \
    bash run_l3_cache_api_tests.sh memcache-npu
EOF
}

maybe_source_cann() {
    if [[ -n ${ASCEND_HOME_PATH:-} ]]; then
        return
    fi
    local set_env=/usr/local/Ascend/ascend-toolkit/set_env.sh
    if [[ -f $set_env ]]; then
        echo "[l3-gate] sourcing $set_env"
        set +u
        # shellcheck disable=SC1091
        source "$set_env"
        set -u
    fi
}

require_device_protocol() {
    case "$DEVICE_PROTOCOL" in
        device_sdma|device_rdma|device_urma|device_uboe)
            ;;
        "")
            echo "[l3-gate] set DEVICE_PROTOCOL explicitly for an NPU gate." >&2
            echo "[l3-gate] A5 candidates are device_urma and device_uboe; choose from site configuration." >&2
            return 2
            ;;
        *)
            echo "[l3-gate] invalid NPU DEVICE_PROTOCOL=$DEVICE_PROTOCOL" >&2
            return 2
            ;;
    esac
}

run_logged() {
    local name=$1
    shift
    local log_file="$LOG_DIR/${name}.log"
    local start_marker="$LOG_DIR/${name}.start"

    echo "[l3-gate] command: $*"
    echo "[l3-gate] log: $log_file"
    : > "$start_marker"
    set +e
    "$@" 2>&1 | tee "$log_file"
    local command_status=${PIPESTATUS[0]}
    set -e
    echo "[l3-gate] exit_code=$command_status gate=$name"
    return "$command_status"
}

collect_recent_hcomm_logs() {
    local label=${1:-manual}
    local log_file="$LOG_DIR/${label}_hcomm.log"
    local start_marker="$LOG_DIR/${label}.start"
    local -a roots=(/root/ascend/log /usr/slog)
    local -a freshness=(-mmin -10)
    if [[ -f $start_marker ]]; then
        freshness=(-newer "$start_marker")
    fi

    {
        echo "[l3-gate] recent HComm/UB diagnostics label=$label"
        echo "[l3-gate] freshness_marker=$start_marker"
        for root in "${roots[@]}"; do
            if [[ ! -d $root ]]; then
                continue
            fi
            echo "[l3-gate] scan_root=$root"
            find "$root" -type f "${freshness[@]}" -print0 2>/dev/null \
                | xargs -0 -r grep -aHnE \
                    'Hcomm(Channel|Endpoint)|HcclCommInit|rank_table|AicpuTsUrmaChannel|DevUbCtpConnection|UbMemTransport|Socket|UBC_CTP|E19999|HCCL_E_' \
                    2>/dev/null || true
        done
    } | tail -n 2000 | tee "$log_file"
    echo "[l3-gate] HComm diagnostics log: $log_file"
}

memcache_capabilities() {
    run_logged memcache_capabilities \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_memcache.py" --mode capabilities
}

memcache_aicpu_check() {
    maybe_source_cann
    run_logged memcache_aicpu_check \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_memfabric_aicpu.py" --mode check
}

memcache_aicpu_provision() {
    maybe_source_cann
    run_logged memcache_aicpu_provision \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_memfabric_aicpu.py" --mode provision
}

memcache_host() {
    run_logged memcache_host \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_memcache.py" \
        --mode host \
        --protocol host_shm \
        --dram-size "$DRAM_SIZE" \
        --max-dram-size 4GB \
        --world-size 8
}

memcache_npu() {
    require_device_protocol
    memcache_aicpu_check
    maybe_source_cann
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
    run_logged memcache_npu \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_memcache.py" \
        --mode npu \
        --protocol "$DEVICE_PROTOCOL" \
        --device-id "$DEVICE_ID" \
        --dram-size "$DRAM_SIZE" \
        --max-dram-size "$MAX_DRAM_SIZE" \
        --world-size "$WORLD_SIZE"
}

memcache_gva() {
    require_device_protocol
    memcache_aicpu_check
    maybe_source_cann
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
    run_logged memcache_gva \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_memcache.py" \
        --mode gva \
        --protocol "$DEVICE_PROTOCOL" \
        --device-id "$DEVICE_ID" \
        --dram-size "$DRAM_SIZE" \
        --max-dram-size "$MAX_DRAM_SIZE" \
        --world-size "$WORLD_SIZE"
}

memcache_tp() {
    memcache_aicpu_check
    maybe_source_cann
    local status=0
    local -a args=(
        "$PYTHON_BIN" "$SCRIPT_DIR/test_memcache_tp.py"
        --config-path "$MEMCACHE_TP_CONFIG_PATH"
        --visible-devices "$MEMCACHE_TP_VISIBLE_DEVICES"
        --active-devices "$MEMCACHE_TP_ACTIVE_DEVICES"
    )
    if [[ -n $MEMCACHE_TP_LOCAL_EID_MAP ]]; then
        args+=(--local-eid-map "$MEMCACHE_TP_LOCAL_EID_MAP")
    fi
    if [[ -n $MEMCACHE_TP_LOCAL_EID_MAP_FILE ]]; then
        args+=(--local-eid-map-file "$MEMCACHE_TP_LOCAL_EID_MAP_FILE")
    fi
    run_logged memcache_tp "${args[@]}" || status=$?
    if ((status != 0)); then
        collect_recent_hcomm_logs memcache_tp
    fi
    return "$status"
}

ascend_store_memcache() {
    require_device_protocol
    memcache_aicpu_check
    maybe_source_cann
    run_logged ascend_store_memcache \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_ascend_store_backends.py" \
        --backend memcache \
        --device-id "$DEVICE_ID" \
        --visible-devices "${ASCEND_RT_VISIBLE_DEVICES:-0}" \
        --memcache-protocol "$DEVICE_PROTOCOL"
}

ascend_store_mooncake() {
    local protocol=${1:-ascend}
    local buffer_mode=device
    if [[ $protocol == tcp ]]; then
        buffer_mode=swapped
    fi
    mooncake_runtime_import
    maybe_source_cann
    run_logged "ascend_store_mooncake_${protocol}" \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_ascend_store_backends.py" \
        --backend mooncake \
        --device-id "$DEVICE_ID" \
        --visible-devices "${ASCEND_RT_VISIBLE_DEVICES:-0}" \
        --mooncake-protocol "$protocol" \
        --mooncake-buffer-mode "$buffer_mode"
}

mooncake_capabilities() {
    run_logged mooncake_capabilities \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_mooncake.py" --mode capabilities
}

mooncake_imports() {
    local log_file="$LOG_DIR/mooncake_imports.log"
    local failed=0
    local label
    local code
    local command_status
    local -a labels=(
        store_only
        engine_only
        store_engine
        engine_store
        acl_store
        acl_engine
        acl_init_store_finalize
        acl_init_engine_finalize
        torch_npu_store
        torch_npu_engine
        torch_npu_store_engine
    )
    local -a codes=(
        'import mooncake.store; print("store-only ok", flush=True)'
        'import mooncake.engine; print("engine-only ok", flush=True)'
        'import mooncake.store; import mooncake.engine; print("store-engine ok", flush=True)'
        'import mooncake.engine; import mooncake.store; print("engine-store ok", flush=True)'
        'import acl; import mooncake.store; print("acl-store ok", flush=True)'
        'import acl; import mooncake.engine; print("acl-engine ok", flush=True)'
        'import acl; assert acl.init() == 0; import mooncake.store; print("acl-init-store ok", flush=True); assert acl.finalize() == 0'
        'import acl; assert acl.init() == 0; import mooncake.engine; print("acl-init-engine ok", flush=True); assert acl.finalize() == 0'
        'import torch; import torch_npu; torch.npu.set_device(0); import mooncake.store; print("torch-npu-store ok", flush=True)'
        'import torch; import torch_npu; torch.npu.set_device(0); import mooncake.engine; print("torch-npu-engine ok", flush=True)'
        'import torch; import torch_npu; torch.npu.set_device(0); import mooncake.store; import mooncake.engine; print("torch-npu-store-engine ok", flush=True)'
    )

    : >"$log_file"
    set +e
    for index in "${!labels[@]}"; do
        label=${labels[$index]}
        code=${codes[$index]}
        echo "[l3-gate] mooncake import=$label" | tee -a "$log_file"
        (ulimit -c 0; exec "$PYTHON_BIN" -c "$code") 2>&1 | tee -a "$log_file"
        command_status=${PIPESTATUS[0]}
        echo "[l3-gate] import=$label exit_code=$command_status" | tee -a "$log_file"
        if ((command_status != 0)); then
            failed=1
        fi
    done
    set -e

    if ((failed != 0)); then
        echo "[l3-gate] Mooncake import gate failed; do not start master or run IO." \
            | tee -a "$log_file"
    fi
    return "$failed"
}

mooncake_runtime_import() {
    local visible_devices=${1:-${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-0}}}
    maybe_source_cann
    run_logged mooncake_runtime_import \
        env \
        ASCEND_RT_VISIBLE_DEVICES="$visible_devices" \
        MOONCAKE_PROBE_DEVICE_ID="$DEVICE_ID" \
        "$PYTHON_BIN" -c \
        'import os; import torch; import torch_npu; torch.npu.set_device(int(os.environ["MOONCAKE_PROBE_DEVICE_ID"])); import mooncake.store; import mooncake.engine; print("Mooncake vLLM-like runtime import passed", flush=True)'
}

mooncake_linkage() {
    local log_file="$LOG_DIR/mooncake_linkage.log"
    local store_so
    local engine_so

    : >"$log_file"
    {
        echo "[l3-gate] ldd version"
        # sed consumes the complete input. head can make ldd receive SIGPIPE,
        # which aborts this block when pipefail is enabled.
        ldd --version 2>&1 | sed -n '1p'
        "$PYTHON_BIN" - <<'PY'
import importlib.util
import platform

print("libc:", platform.libc_ver())
for name in ("mooncake.store", "mooncake.engine"):
    spec = importlib.util.find_spec(name)
    print(f"{name}: {None if spec is None else spec.origin}")
PY
    } 2>&1 | tee -a "$log_file"

    set +e
    store_so=$(
        "$PYTHON_BIN" -c \
            'import importlib.util; print(importlib.util.find_spec("mooncake.store").origin)'
    )
    engine_so=$(
        "$PYTHON_BIN" -c \
            'import importlib.util; print(importlib.util.find_spec("mooncake.engine").origin)'
    )
    set -e

    for extension in "$store_so" "$engine_so"; do
        if [[ -n $extension && -f $extension ]]; then
            if command -v sha256sum >/dev/null 2>&1; then
                sha256sum "$extension" | tee -a "$log_file"
            fi
            echo "[l3-gate] ldd $extension" | tee -a "$log_file"
            ldd "$extension" 2>&1 | tee -a "$log_file"
            if command -v readelf >/dev/null 2>&1; then
                echo "[l3-gate] readelf dynamic $extension" | tee -a "$log_file"
                readelf -d "$extension" 2>&1 \
                    | grep -E 'NEEDED|RPATH|RUNPATH' \
                    | tee -a "$log_file" || true
            fi
        else
            echo "[l3-gate] extension not found: ${extension:-<empty>}" \
                | tee -a "$log_file"
        fi
    done

    if grep -q "not found" "$log_file"; then
        echo "[l3-gate] unresolved shared libraries found" | tee -a "$log_file"
        return 1
    fi
}

mooncake_gdb() {
    local module=$1
    local gdb_status=0
    local log_file="$LOG_DIR/mooncake_gdb_${module}.log"
    local python_path
    if ! command -v gdb >/dev/null 2>&1; then
        echo "[l3-gate] gdb is not installed; cannot capture Mooncake native backtrace." >&2
        return 127
    fi
    python_path=$(command -v "$PYTHON_BIN")
    run_logged "mooncake_gdb_${module}" \
        gdb --quiet --batch \
        -ex "set pagination off" \
        -ex "set print thread-events off" \
        -ex "run" \
        -ex "bt full" \
        -ex "thread apply all bt" \
        -ex "info sharedlibrary" \
        --args "$python_path" -c \
        "import mooncake.${module}; print('${module} import completed', flush=True)" \
        || gdb_status=$?
    if grep -Eq 'received signal SIG(ABRT|SEGV|BUS)' "$log_file"; then
        echo "[l3-gate] Mooncake ${module} inferior received a fatal signal." \
            | tee -a "$log_file"
        return 1
    fi
    return "$gdb_status"
}

mooncake_npu() {
    if ! mooncake_runtime_import; then
        echo "[l3-gate] Mooncake NPU gate blocked by the vLLM-like runtime import."
        return 1
    fi
    maybe_source_cann
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
    run_logged mooncake_npu \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_mooncake.py" \
        --mode all \
        --device-id "$DEVICE_ID" \
        --local-host "$MOONCAKE_LOCAL_HOST" \
        --master-address "$MOONCAKE_MASTER_ADDRESS"
}

mooncake_tp() {
    case "$MOONCAKE_TP_COMM_MODE" in
        hixlep)
            ASCEND_RT_VISIBLE_DEVICES="$MOONCAKE_TP_VISIBLE_DEVICES" hixlep_config
            ;;
        global_ub|global_ub_device|global_ub_host|auto|tcp)
            ;;
        *)
            echo "MOONCAKE_TP_COMM_MODE must be global_ub/global_ub_device/global_ub_host/hixlep/auto/tcp, got: $MOONCAKE_TP_COMM_MODE" >&2
            return 2
            ;;
    esac
    mooncake_runtime_import "$MOONCAKE_TP_VISIBLE_DEVICES"
    maybe_source_cann
    local status=0
    local extra_args=(
        --buffer-mode "$MOONCAKE_TP_BUFFER_MODE"
        --num-buffers "$MOONCAKE_TP_NUM_BUFFERS"
        --payload-bytes "$MOONCAKE_TP_PAYLOAD_BYTES"
        --payload-layout "$MOONCAKE_TP_PAYLOAD_LAYOUT"
        --segment-bytes "$MOONCAKE_TP_SEGMENT_BYTES"
        --local-buffer-bytes "$MOONCAKE_TP_LOCAL_BUFFER_BYTES"
        --arena-blocks "$MOONCAKE_TP_ARENA_BLOCKS"
        --block-index "$MOONCAKE_TP_BLOCK_INDEX"
        --io-thread "$MOONCAKE_TP_IO_THREAD"
        --placement-mode "$MOONCAKE_TP_PLACEMENT_MODE"
    )
    if [[ $MOONCAKE_TP_FORCE_REMOTE_PUT == true ]]; then
        extra_args+=(--force-remote-put)
    fi
    if [[ -n $MOONCAKE_TP_PRODUCER_RANK ]]; then
        extra_args+=(--producer-rank "$MOONCAKE_TP_PRODUCER_RANK")
    fi
    if [[ -n $MOONCAKE_TP_CONSUMER_DEVICE ]]; then
        extra_args+=(--consumer-device "$MOONCAKE_TP_CONSUMER_DEVICE")
    fi
    if [[ -n $MOONCAKE_TP_TARGET_DEVICE ]]; then
        extra_args+=(--target-device "$MOONCAKE_TP_TARGET_DEVICE")
    fi
    if [[ -n $MOONCAKE_TP_SOURCE_BUFFER_MODE ]]; then
        extra_args+=(--source-buffer-mode "$MOONCAKE_TP_SOURCE_BUFFER_MODE")
    fi
    if [[ -n $MOONCAKE_TP_DESTINATION_BUFFER_MODE ]]; then
        extra_args+=(--destination-buffer-mode "$MOONCAKE_TP_DESTINATION_BUFFER_MODE")
    fi
    run_logged mooncake_tp \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_mooncake_tp.py" \
        --config-path "$HIXLEP_CONFIG_PATH" \
        --visible-devices "$MOONCAKE_TP_VISIBLE_DEVICES" \
        --active-devices "$MOONCAKE_TP_ACTIVE_DEVICES" \
        --comm-mode "$MOONCAKE_TP_COMM_MODE" \
        --max-active-channels "$MOONCAKE_TP_MAX_ACTIVE_CHANNELS" \
        --local-host "$MOONCAKE_TP_LOCAL_HOST" \
        --master-address "$MOONCAKE_MASTER_ADDRESS" \
        "${extra_args[@]}" \
        || status=$?
    if ((status != 0)); then
        collect_recent_hcomm_logs mooncake_tp
    fi
    return "$status"
}

mooncake_store_tcp() {
    local mode=$1
    maybe_source_cann
    run_logged "mooncake_store_tcp_${mode}" \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_mooncake_store_tcp.py" \
        --mode "$mode" \
        --devices "$MOONCAKE_STORE_TCP_DEVICES" \
        --local-host "$MOONCAKE_LOCAL_HOST" \
        --master-address "$MOONCAKE_MASTER_ADDRESS" \
        --segment-bytes "$MOONCAKE_STORE_TCP_SEGMENT_BYTES" \
        --local-buffer-bytes "$MOONCAKE_STORE_TCP_LOCAL_BUFFER_BYTES" \
        --num-planes "$MOONCAKE_STORE_TCP_NUM_PLANES" \
        --plane-bytes "$MOONCAKE_STORE_TCP_PLANE_BYTES" \
        --timeout "$MOONCAKE_STORE_TCP_TIMEOUT"
}

dsa_swapped_mooncake() {
    maybe_source_cann
    local status=0
    run_logged dsa_swapped_mooncake \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_dsa_swapped_l3_io.py" \
        --backend mooncake \
        --component "$DSA_SWAPPED_COMPONENT" \
        --schema-mode "$DSA_SWAPPED_SCHEMA_MODE" \
        --destination-memory "$DSA_SWAPPED_DESTINATION_MEMORY" \
        --device-id "$DEVICE_ID" \
        --visible-devices "$DSA_SWAPPED_VISIBLE_DEVICES" \
        --local-host "$MOONCAKE_LOCAL_HOST" \
        --mooncake-master "$MOONCAKE_MASTER_ADDRESS" \
        --mooncake-comm-mode "$DSA_SWAPPED_MOONCAKE_COMM_MODE" \
        --max-active-channels "$DSA_SWAPPED_MAX_ACTIVE_CHANNELS" \
        || status=$?
    if ((status != 0)); then
        collect_recent_hcomm_logs dsa_swapped_mooncake
    fi
    return "$status"
}

dsa_swapped_memcache() {
    memcache_aicpu_check
    maybe_source_cann
    local -a args=(
        "$PYTHON_BIN" "$SCRIPT_DIR/test_dsa_swapped_l3_io.py"
        --backend memcache
        --component "$DSA_SWAPPED_COMPONENT"
        --schema-mode "$DSA_SWAPPED_SCHEMA_MODE"
        --destination-memory "$DSA_SWAPPED_DESTINATION_MEMORY"
        --device-id "$DEVICE_ID"
        --visible-devices "$DSA_SWAPPED_VISIBLE_DEVICES"
        --memcache-config "$MEMCACHE_TP_CONFIG_PATH"
    )
    if [[ -n $DSA_SWAPPED_MEMCACHE_LOCAL_EID ]]; then
        args+=(--memcache-local-eid "$DSA_SWAPPED_MEMCACHE_LOCAL_EID")
    fi
    run_logged dsa_swapped_memcache "${args[@]}"
}

hixlep_config() {
    local visible_devices=${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
    run_logged hixlep_config \
        "$PYTHON_BIN" "$SCRIPT_DIR/check_hixlep.py" \
        --path "$HIXLEP_CONFIG_PATH" \
        --visible-devices "$visible_devices" \
        --require-placement device \
        --require-placement host
}

hixlep_tooling() {
    local log_file="$LOG_DIR/hixlep_tooling.log"
    {
        echo "[l3-gate] PATH=$PATH"
        for tool in hixl_tool lcrgen; do
            if command -v "$tool" >/dev/null 2>&1; then
                echo "[l3-gate] $tool=$(command -v "$tool")"
                "$tool" --help 2>&1 || true
                if [[ $tool == hixl_tool ]]; then
                    "$tool" local_comm_res --help 2>&1 || true
                fi
            else
                echo "[l3-gate] $tool=NOT_FOUND"
            fi
        done
        find /usr/local/Ascend -type f \
            \( -name hixl_tool -o -name lcrgen -o -name 'libhixl*.so*' \
            -o -name 'libadxl*.so*' \) -print 2>/dev/null || true
        ldconfig -p 2>/dev/null | grep -E 'lib(hixl|adxl)' || true
    } 2>&1 | tee "$log_file"
}

diagnostics() {
    local log_file="$LOG_DIR/diagnostics.log"
    {
        echo "[l3-gate] date=$(date --iso-8601=seconds)"
        echo "[l3-gate] python=$($PYTHON_BIN --version 2>&1)"
        "$PYTHON_BIN" -m pip show \
            memcache_hybrid memfabric_hybrid mooncake-transfer-engine-npu || true
        "$PYTHON_BIN" - <<'PY'
import os

try:
    import torch
    import torch_npu
    from torch_npu import version as torch_npu_version

    print(f"torch: {torch.__version__}")
    print(f"torch_npu: {torch_npu.__version__}")
    print(f"torch_npu.git_version: {torch_npu_version.git_version}")
    print(f"torch.npu.is_available: {torch.npu.is_available()}")
except Exception as exc:
    print(f"torch_npu runtime probe failed: {exc!r}")

for module_name in ("memcache_hybrid", "memfabric_hybrid"):
    try:
        module = __import__(module_name)
        print(f"{module_name}: {module.__file__}")
    except Exception as exc:
        print(f"{module_name}: import failed: {exc!r}")

for key in sorted(os.environ):
    if key.startswith(("ASCEND", "HCCL", "MMC", "MF", "PYTHON")):
        print(f"{key}={os.environ[key]}")
PY
        command -v npu-smi || true
        npu-smi info || true
        npu-smi info -t topo || true
    } 2>&1 | tee "$log_file"
}

gate=${1:-}
case "$gate" in
    memcache-capabilities)
        memcache_capabilities
        ;;
    memcache-aicpu-check)
        memcache_aicpu_check
        ;;
    memcache-aicpu-provision)
        memcache_aicpu_provision
        ;;
    memcache-host)
        memcache_host
        ;;
    memcache-npu)
        memcache_npu
        ;;
    memcache-gva)
        memcache_gva
        ;;
    memcache-tp)
        memcache_tp
        ;;
    memcache-all)
        memcache_capabilities
        memcache_host
        memcache_npu
        memcache_gva
        ;;
    ascend-store-memcache)
        ascend_store_memcache
        ;;
    ascend-store-mooncake)
        ascend_store_mooncake ascend
        ;;
    ascend-store-mooncake-tcp)
        ascend_store_mooncake tcp
        ;;
    mooncake-capabilities)
        mooncake_capabilities
        ;;
    mooncake-runtime-import)
        mooncake_runtime_import
        ;;
    mooncake-imports)
        mooncake_imports
        ;;
    mooncake-linkage)
        mooncake_linkage
        ;;
    mooncake-gdb-store)
        mooncake_gdb store
        ;;
    mooncake-gdb-engine)
        mooncake_gdb engine
        ;;
    mooncake-npu)
        mooncake_npu
        ;;
    mooncake-tp)
        mooncake_tp
        ;;
    mooncake-store-tcp-inspect)
        mooncake_store_tcp inspect
        ;;
    mooncake-store-tcp-single)
        mooncake_store_tcp single
        ;;
    mooncake-store-tcp-cross)
        mooncake_store_tcp cross
        ;;
    mooncake-store-tcp-all)
        mooncake_store_tcp inspect
        mooncake_store_tcp single
        mooncake_store_tcp cross
        ;;
    dsa-swapped-mooncake)
        dsa_swapped_mooncake
        ;;
    dsa-swapped-memcache)
        dsa_swapped_memcache
        ;;
    hixlep-config)
        hixlep_config
        ;;
    hixlep-tooling)
        hixlep_tooling
        ;;
    hcomm-logs)
        collect_recent_hcomm_logs manual
        ;;
    diagnostics)
        diagnostics
        ;;
    safe)
        memcache_capabilities
        memcache_host
        mooncake_linkage
        diagnostics
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
