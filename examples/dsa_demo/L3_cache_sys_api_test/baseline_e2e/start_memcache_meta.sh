#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start the external MemCache MetaService in the foreground.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MMC_META_CONFIG_PATH=${MMC_META_CONFIG_PATH:-$SCRIPT_DIR/configs/mmc-meta.conf}

if [[ ! -f $MMC_META_CONFIG_PATH ]]; then
    echo "MemCache meta config not found: $MMC_META_CONFIG_PATH" >&2
    exit 2
fi

if [[ -z ${ASCEND_HOME_PATH:-} && -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi

export MMC_META_CONFIG_PATH
echo "[baseline-l3] starting MemCache MetaService"
echo "[baseline-l3] config=$MMC_META_CONFIG_PATH"
echo "[baseline-l3] endpoint=tcp://127.0.0.1:5000 config_store=tcp://127.0.0.1:6000"

exec python -c 'from memcache_hybrid import MetaService; MetaService.main()'
