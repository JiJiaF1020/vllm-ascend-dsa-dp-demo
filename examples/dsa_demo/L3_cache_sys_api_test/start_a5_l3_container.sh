#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 在宿主机上启动用于 A5 L3/HiXLEP/HComm 验证的 vLLM-Ascend 容器。

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash start_a5_l3_container.sh <image> <container-name>

Optional environment variables:
  CONTAINER_HOSTNAME=dsa-host
  SHM_SIZE=256g
  CONTAINER_WORKDIR=/home
  L3_TOPOLOGY_MODE=auto  # auto, required, or off

The image owns its userspace libraries. This script intentionally never bind
mounts host /lib*, /usr/lib* or another distribution's glibc into the image.

Topology modes:
  auto      Mount every available L3/HComm topology file and report gaps.
  required  Require the complete L3 topology/tooling set before startup.
  off       Do not mount host L3/HComm topology files or optional tools.
EOF
}

if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi

IMAGE=$1
CONTAINER_NAME=$2
CONTAINER_HOSTNAME=${CONTAINER_HOSTNAME:-dsa-host}
SHM_SIZE=${SHM_SIZE:-256g}
CONTAINER_WORKDIR=${CONTAINER_WORKDIR:-/home}
L3_TOPOLOGY_MODE=${L3_TOPOLOGY_MODE:-auto}
CONTAINER_STARTED=0

case "$L3_TOPOLOGY_MODE" in
    auto|required|off) ;;
    *)
        echo "[a5-container] L3_TOPOLOGY_MODE must be auto, required, or off" >&2
        exit 2
        ;;
esac

cleanup_on_error() {
    local status=$?
    trap - ERR
    if [[ $CONTAINER_STARTED == 1 ]] && docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        echo "[a5-container] startup validation failed; removing $CONTAINER_NAME" >&2
        docker rm --force "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    exit "$status"
}

trap cleanup_on_error ERR

require_file() {
    local path=$1
    if [[ ! -f $path ]]; then
        echo "[a5-container] required file is missing: $path" >&2
        exit 2
    fi
}

require_dir() {
    local path=$1
    if [[ ! -d $path ]]; then
        echo "[a5-container] required directory is missing: $path" >&2
        exit 2
    fi
}

require_device() {
    local path=$1
    if [[ ! -c $path ]]; then
        echo "[a5-container] required character device is missing: $path" >&2
        exit 2
    fi
}

require_device_tree() {
    local path=$1
    if [[ ! -d $path ]]; then
        echo "[a5-container] required device directory is missing: $path" >&2
        exit 2
    fi
    if ! find "$path" -maxdepth 1 -type c -print -quit | grep -q .; then
        echo "[a5-container] device directory has no character devices: $path" >&2
        exit 2
    fi
}

resolve_command() {
    local name=$1
    local path
    path=$(command -v "$name" 2>/dev/null || true)
    if [[ -z $path || ! -f $path ]]; then
        echo "[a5-container] required command is missing: $name" >&2
        exit 2
    fi
    readlink -f "$path"
}

resolve_optional_command() {
    local name=$1
    local path
    path=$(command -v "$name" 2>/dev/null || true)
    if [[ -n $path && -f $path ]]; then
        readlink -f "$path"
    fi
}

require_dir /usr/local/Ascend/driver
require_dir /usr/local/dcmi

NPU_SMI=$(resolve_command npu-smi)
HCCN_TOOL=$(resolve_optional_command hccn_tool)
URMA_ADMIN=$(resolve_optional_command urma_admin)

topology_missing=()
if [[ $L3_TOPOLOGY_MODE != off ]]; then
    [[ -f /lib/route.conf ]] || topology_missing+=(/lib/route.conf)
    [[ -f /etc/hccn.conf ]] || topology_missing+=(/etc/hccn.conf)
    [[ -f /etc/hccl_rootinfo.json ]] || topology_missing+=(/etc/hccl_rootinfo.json)
    [[ -d /etc/hixlep ]] || topology_missing+=(/etc/hixlep)
    if [[ -d /etc/hixlep ]] &&
       ! find /etc/hixlep -maxdepth 1 -type f -name 'ub_endpoint_npu_*.json' -print -quit | grep -q .; then
        topology_missing+=(/etc/hixlep/ub_endpoint_npu_*.json)
    fi
    [[ -n $HCCN_TOOL ]] || topology_missing+=(hccn_tool)
    [[ -n $URMA_ADMIN ]] || topology_missing+=(urma_admin)
fi

if [[ $L3_TOPOLOGY_MODE == required && ${#topology_missing[@]} -ne 0 ]]; then
    printf '[a5-container] required L3 topology capability is missing: %s\n' \
        "${topology_missing[@]}" >&2
    exit 2
fi
if [[ $L3_TOPOLOGY_MODE == auto && ${#topology_missing[@]} -ne 0 ]]; then
    echo "[a5-container] warning: L3 topology is incomplete; ordinary A5/DSA tests can still run" >&2
    printf '[a5-container] optional capability missing: %s\n' \
        "${topology_missing[@]}" >&2
fi

for device_id in $(seq 0 7); do
    require_device "/dev/davinci${device_id}"
done
for device in /dev/davinci_manager /dev/hisi_hdc; do
    require_device "$device"
done
for device_dir in /dev/ummu /dev/uburma; do
    require_device_tree "$device_dir"
done

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[a5-container] Docker image is not available locally: $IMAGE" >&2
    exit 2
fi
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "[a5-container] container already exists: $CONTAINER_NAME" >&2
    exit 2
fi

docker_args=(
    run
    --name "$CONTAINER_NAME"
    --detach
    --interactive
    --tty
    --net host
    --pid host
    --shm-size "$SHM_SIZE"
    --hostname "$CONTAINER_HOSTNAME"
    --user root
    --workdir "$CONTAINER_WORKDIR"
    --privileged
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    --security-opt apparmor=unconfined
    --env "DSA_L3_TOPOLOGY_MODE=$L3_TOPOLOGY_MODE"
)

for device_id in $(seq 0 7); do
    docker_args+=(--device "/dev/davinci${device_id}:/dev/davinci${device_id}")
done
for device in /dev/davinci_manager /dev/hisi_hdc; do
    docker_args+=(--device "$device:$device")
done
for device_dir in /dev/ummu /dev/uburma; do
    docker_args+=(--device "$device_dir:$device_dir")
done
if [[ -c /dev/devmm_svm ]]; then
    docker_args+=(--device /dev/devmm_svm:/dev/devmm_svm)
fi

add_ro_file() {
    local source=$1
    local target=$2
    docker_args+=(--mount "type=bind,src=${source},dst=${target},readonly")
}

add_ro_dir() {
    local source=$1
    local target=$2
    docker_args+=(--mount "type=bind,src=${source},dst=${target},readonly")
}

add_rw_dir_if_present() {
    local source=$1
    local target=$2
    if [[ -d $source ]]; then
        docker_args+=(--mount "type=bind,src=${source},dst=${target}")
    fi
}

add_ro_dir /usr/local/Ascend/driver /usr/local/Ascend/driver
add_ro_dir /usr/local/dcmi /usr/local/dcmi
add_ro_file "$NPU_SMI" /usr/local/bin/npu-smi

if [[ $L3_TOPOLOGY_MODE != off ]]; then
    [[ -d /etc/hixlep ]] && add_ro_dir /etc/hixlep /etc/hixlep
    [[ -f /lib/route.conf ]] && add_ro_file /lib/route.conf /lib/route.conf
    [[ -f /etc/hccn.conf ]] && add_ro_file /etc/hccn.conf /etc/hccn.conf
    [[ -f /etc/hccl_rootinfo.json ]] &&
        add_ro_file /etc/hccl_rootinfo.json /etc/hccl_rootinfo.json
    [[ -n $HCCN_TOOL ]] && add_ro_file "$HCCN_TOOL" /usr/bin/hccn_tool
    [[ -n $URMA_ADMIN ]] && add_ro_file "$URMA_ADMIN" /usr/bin/urma_admin
fi

add_rw_dir_if_present /home /home
add_rw_dir_if_present /mnt /mnt
add_rw_dir_if_present /kvcache /kvcache
add_rw_dir_if_present /models /models
add_rw_dir_if_present /root/host /root/host
if [[ -d /var/log/npu ]]; then
    add_rw_dir_if_present /var/log/npu /usr/slog
fi

echo "[a5-container] keeping image-owned system libraries; host /usr/lib64 is not mounted"

echo "[a5-container] image=$IMAGE"
echo "[a5-container] name=$CONTAINER_NAME"
echo "[a5-container] l3_topology_mode=$L3_TOPOLOGY_MODE missing=${topology_missing[*]:-none}"

docker "${docker_args[@]}" "$IMAGE" sleep infinity
CONTAINER_STARTED=1

docker exec "$CONTAINER_NAME" bash -lc '
set -e
command -v npu-smi
test -d /dev/ummu
test -d /dev/uburma
find /dev/ummu -maxdepth 1 -type c -print -quit | grep -q .
find /dev/uburma -maxdepth 1 -type c -print -quit | grep -q .
for i in $(seq 0 7); do test -c /dev/davinci${i}; done

# A host openEuler /usr/lib64 mounted into an Ubuntu image leaves a linker
# script that references missing /lib64/libm*.so. Fail here instead of much
# later while Triton compiles npu_utils.so.
if [[ -f /usr/lib64/libm.so ]] &&
   grep -q "/lib64/libm.so.6" /usr/lib64/libm.so &&
   [[ ! -e /lib64/libm.so.6 ]]; then
    echo "[a5-container] incompatible glibc layout detected:" >&2
    echo "[a5-container] /usr/lib64/libm.so references missing /lib64/libm.so.6" >&2
    exit 1
fi

echo "[a5-container] A5 devices are visible"
topology_missing=()
[[ -f /lib/route.conf ]] || topology_missing+=(/lib/route.conf)
[[ -f /etc/hccn.conf ]] || topology_missing+=(/etc/hccn.conf)
[[ -f /etc/hccl_rootinfo.json ]] || topology_missing+=(/etc/hccl_rootinfo.json)
if [[ ! -d /etc/hixlep ]] ||
   ! find /etc/hixlep -maxdepth 1 -type f -name "ub_endpoint_npu_*.json" -print -quit 2>/dev/null | grep -q .; then
    topology_missing+=(/etc/hixlep/ub_endpoint_npu_*.json)
fi
command -v hccn_tool >/dev/null 2>&1 || topology_missing+=(hccn_tool)
command -v urma_admin >/dev/null 2>&1 || topology_missing+=(urma_admin)
if [[ ${#topology_missing[@]} -eq 0 ]]; then
    echo "[a5-container] L3 topology capability: complete"
else
    echo "[a5-container] L3 topology capability: incomplete (${topology_missing[*]})"
    if [[ ${DSA_L3_TOPOLOGY_MODE:-auto} == required ]]; then
        exit 1
    fi
fi
echo "[a5-container] os=$(sed -n "s/^PRETTY_NAME=//p" /etc/os-release | tr -d "\"")"
if command -v findmnt >/dev/null 2>&1 && [[ -e /usr/lib64 ]]; then
    echo "[a5-container] usr_lib64_mount=$(findmnt -T /usr/lib64 -n -o SOURCE,TARGET 2>/dev/null || true)"
fi
echo "[a5-container] libm=$(ldconfig -p 2>/dev/null | awk "/libm\\.so\\.6/{print \$NF; exit}")"
echo "[a5-container] toolkit=$(readlink -f /usr/local/Ascend/ascend-toolkit/latest 2>/dev/null || true)"
npu-smi info | sed -n "1,20p"
'

trap - ERR
echo "[a5-container] started successfully"
echo "[a5-container] enter with: docker exec -it $CONTAINER_NAME bash"
