#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start the native Mooncake Master in the foreground.

set -euo pipefail

RPC_PORT=${RPC_PORT:-50088}
METRICS_PORT=${METRICS_PORT:-9003}
CLIENT_TTL_SECONDS=${CLIENT_TTL_SECONDS:-120}
LEASE_TTL_MS=${LEASE_TTL_MS:-300000}

if [[ -z ${ASCEND_HOME_PATH:-} && -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi

MOONCAKE_MASTER_BIN=$(
    python - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("mooncake")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("mooncake package is not installed")
path = Path(next(iter(spec.submodule_search_locations))) / "mooncake_master"
if not path.is_file():
    raise SystemExit(f"native mooncake_master not found: {path}")
print(path)
PY
)

chmod u+x "$MOONCAKE_MASTER_BIN"
echo "[baseline-l3] starting Mooncake Master"
echo "[baseline-l3] binary=$MOONCAKE_MASTER_BIN"
echo "[baseline-l3] rpc=127.0.0.1:$RPC_PORT metrics=127.0.0.1:$METRICS_PORT"

exec "$MOONCAKE_MASTER_BIN" \
    --rpc_port="$RPC_PORT" \
    --metrics_port="$METRICS_PORT" \
    --eviction_high_watermark_ratio=0.9 \
    --eviction_ratio=0.1 \
    --default_kv_lease_ttl="$LEASE_TTL_MS" \
    --client_ttl="$CLIENT_TTL_SECONDS"
