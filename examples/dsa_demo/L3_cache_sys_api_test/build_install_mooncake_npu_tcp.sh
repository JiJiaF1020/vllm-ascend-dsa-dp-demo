#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build a Mooncake NPU wheel with both TCP and AscendDirect, then install it
# into an isolated target directory for non-destructive validation.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
MOONCAKE_REPO_URL=${MOONCAKE_REPO_URL:-https://github.com/kvcache-ai/Mooncake.git}
MOONCAKE_REF=${MOONCAKE_REF:-e9c61075720039bcfc5fffd19f847608402be3d0}
MOONCAKE_SOURCE_DIR=${MOONCAKE_SOURCE_DIR:-$HOME/mooncake-dsa-npu-tcp-src}
MOONCAKE_WORK_ROOT=${MOONCAKE_WORK_ROOT:-$HOME/.cache/dsa-mooncake-npu-tcp}
MOONCAKE_JOBS=${MOONCAKE_JOBS:-$(nproc)}
MOONCAKE_CLEAN_BUILD=${MOONCAKE_CLEAN_BUILD:-1}
YAML_CPP_REPO_URL=${YAML_CPP_REPO_URL:-https://github.com/jbeder/yaml-cpp.git}
YAML_CPP_REF=${YAML_CPP_REF:-0.8.0}
MSGPACK_REPO_URL=${MSGPACK_REPO_URL:-https://github.com/msgpack/msgpack-c.git}
MSGPACK_REF=${MSGPACK_REF:-cpp-7.0.0}
DEVICE_ID=${DEVICE_ID:-0}

PYTHON_VERSION=${PYTHON_VERSION:-}
BUILD_DIR_NAME=
BUILD_DIR=
BUILD_VENV=
WHEEL_DIR=
STAGE_DIR=
ACTIVATE_FILE=
NATIVE_DEPS_PREFIX=
NATIVE_DEPS_SOURCE_DIR=
NATIVE_DEPS_BUILD_DIR=
DEPENDENCY_PROBE_DIR=
BUILD_CONSTRAINTS=
MANAGED_MARKER=$MOONCAKE_SOURCE_DIR/.dsa_mooncake_managed

restore_upstream_manifest() {
    local backup=$MOONCAKE_SOURCE_DIR/mooncake-wheel/pyproject.toml.backup
    local manifest=$MOONCAKE_SOURCE_DIR/mooncake-wheel/pyproject.toml
    if [[ -f $backup ]]; then
        mv -f -- "$backup" "$manifest"
    fi
}
trap restore_upstream_manifest EXIT

usage() {
    cat <<'EOF'
Usage:
  bash build_install_mooncake_npu_tcp.sh <mode>

Modes:
  audit           Print the currently imported Mooncake package and runtime.
  check-deps      Check the native build dependencies without installing them.
  deps            Install system packages and private pinned source dependencies.
  build           Checkout/build the pinned TCP+AscendDirect NPU wheel.
  install         Install the newest built wheel into an isolated target dir.
  verify-tcp      Import the staged package and initialize TCP transport.
  verify-ascend   Import the staged package and initialize AscendDirect.
  verify-all      Run both transport initialization gates.
  all             audit + build + install + verify-all; dependencies must exist.
  bootstrap       deps + all; intended for the first build in a fresh container.
  print-env       Print the activation command for the staged package.

Important environment variables:
  PYTHON_BIN, DEVICE_ID, MOONCAKE_REPO_URL, MOONCAKE_REF,
  MOONCAKE_SOURCE_DIR, MOONCAKE_WORK_ROOT, MOONCAKE_JOBS,
  MOONCAKE_CLEAN_BUILD, YAML_CPP_REPO_URL, YAML_CPP_REF,
  MSGPACK_REPO_URL, MSGPACK_REF

The wheel install is non-destructive. It writes under
  $HOME/.cache/dsa-mooncake-npu-tcp
and does not replace the container's system wheel.

The deps/bootstrap modes install native development packages with apt, dnf,
or yum. Source-only dependencies are installed under MOONCAKE_WORK_ROOT.
EOF
}

source_cann_if_needed() {
    if [[ -n ${ASCEND_HOME_PATH:-} ]]; then
        return
    fi
    local candidates=(
        /usr/local/Ascend/cann/set_env.sh
        /usr/local/Ascend/cann-9.1.0/set_env.sh
        /usr/local/Ascend/ascend-toolkit/set_env.sh
        /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f $candidate ]]; then
            # shellcheck disable=SC1090
            source "$candidate"
            return
        fi
    done
    echo "[mooncake-build] CANN set_env.sh was not found" >&2
    return 1
}

require_tools() {
    local tool
    for tool in git cmake bash "$PYTHON_BIN"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            echo "[mooncake-build] missing required tool: $tool" >&2
            return 1
        fi
    done
}

resolve_work_root() {
    mkdir -p "$MOONCAKE_WORK_ROOT"
    MOONCAKE_WORK_ROOT=$(cd -- "$MOONCAKE_WORK_ROOT" && pwd)
    BUILD_VENV=$MOONCAKE_WORK_ROOT/build-venv
    WHEEL_DIR=$MOONCAKE_WORK_ROOT/wheels
    STAGE_DIR=$MOONCAKE_WORK_ROOT/site
    ACTIVATE_FILE=$MOONCAKE_WORK_ROOT/activate.sh
    NATIVE_DEPS_PREFIX=$MOONCAKE_WORK_ROOT/native
    NATIVE_DEPS_SOURCE_DIR=$MOONCAKE_WORK_ROOT/native-src
    NATIVE_DEPS_BUILD_DIR=$MOONCAKE_WORK_ROOT/native-build
    DEPENDENCY_PROBE_DIR=$MOONCAKE_WORK_ROOT/dependency-probe
    BUILD_CONSTRAINTS=$MOONCAKE_WORK_ROOT/build-constraints.txt
}

initialize_derived_paths() {
    if [[ -z $PYTHON_VERSION ]]; then
        PYTHON_VERSION=$(
            "$PYTHON_BIN" -c \
                'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
        )
    fi
    BUILD_DIR_NAME=.dsa-build-npu-tcp-py${PYTHON_VERSION}
    BUILD_DIR=$MOONCAKE_SOURCE_DIR/$BUILD_DIR_NAME
}

safe_remove_work_dir() {
    local target=$1
    case $target in
        "$MOONCAKE_WORK_ROOT"/*)
            rm -rf -- "$target"
            ;;
        *)
            echo "[mooncake-build] refusing to remove path outside work root: $target" >&2
            return 1
            ;;
    esac
}

run_as_root() {
    if [[ $(id -u) == 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        echo "[mooncake-build] root or sudo is required to install packages" >&2
        return 1
    fi
}

report_tmp_state() {
    echo "[mooncake-build] /tmp diagnostics:" >&2
    ls -ld /tmp >&2 || true
    df -Pk /tmp >&2 || true
    df -Pi /tmp >&2 || true
    mount | grep -E ' on /tmp( |$)' >&2 || true
}

prepare_apt_tmp() {
    if [[ -e /tmp && ! -d /tmp ]]; then
        echo "[mooncake-build] /tmp exists but is not a directory" >&2
        report_tmp_state
        return 1
    fi
    if [[ ! -d /tmp ]]; then
        run_as_root mkdir -p /tmp
    fi

    local mode
    mode=$(stat -c '%a' /tmp)
    if [[ $mode != 1777 ]]; then
        echo "[mooncake-build] repairing /tmp mode $mode -> 1777"
        if ! run_as_root chmod 1777 /tmp; then
            report_tmp_state
            return 1
        fi
    fi

    local probe
    if ! probe=$(run_as_root mktemp /tmp/dsa-apt-probe.XXXXXX); then
        echo "[mooncake-build] root cannot create a temporary file in /tmp" >&2
        report_tmp_state
        return 1
    fi
    run_as_root rm -f -- "$probe"

    if id _apt >/dev/null 2>&1 && command -v runuser >/dev/null 2>&1; then
        if ! run_as_root runuser -u _apt -- sh -c \
            'probe=$(mktemp /tmp/dsa-apt-user-probe.XXXXXX) && rm -f "$probe"'; then
            echo "[mooncake-build] the _apt user cannot write to /tmp" >&2
            report_tmp_state
            return 1
        fi
    fi
}

install_system_dependencies() {
    local packages=()
    if command -v apt-get >/dev/null 2>&1; then
        packages=(
            build-essential cmake ninja-build git wget curl pkg-config patchelf
            libgflags-dev libgoogle-glog-dev libjsoncpp-dev libibverbs-dev
            libnuma-dev libboost-dev libssl-dev libcurl4-openssl-dev
            libunwind-dev libzstd-dev libxxhash-dev
        )
        echo "[mooncake-build] installing Debian/Ubuntu native dependencies"
        prepare_apt_tmp
        run_as_root env TMPDIR=/tmp apt-get \
            -o APT::Update::Error-Mode=any update
        run_as_root env TMPDIR=/tmp DEBIAN_FRONTEND=noninteractive \
            apt-get install -y "${packages[@]}"
    else
        local package_manager=
        if command -v dnf >/dev/null 2>&1; then
            package_manager=dnf
        elif command -v yum >/dev/null 2>&1; then
            package_manager=yum
        else
            echo "[mooncake-build] no supported package manager found" >&2
            echo "[mooncake-build] expected apt-get, dnf, or yum" >&2
            return 1
        fi
        packages=(
            gcc gcc-c++ make cmake ninja-build git wget curl
            pkgconf-pkg-config patchelf gflags-devel glog-devel
            jsoncpp-devel libibverbs-devel numactl-devel boost-devel
            openssl-devel libcurl-devel libunwind-devel zstd-devel
            xxhash-devel
        )
        echo "[mooncake-build] installing openEuler/RHEL native dependencies"
        run_as_root "$package_manager" makecache -y || \
            run_as_root "$package_manager" makecache
        run_as_root "$package_manager" install -y "${packages[@]}"
    fi
}

checkout_private_dependency() {
    local repo_url=$1
    local ref=$2
    local source_dir=$3
    if [[ ! -d $source_dir/.git ]]; then
        mkdir -p "$(dirname -- "$source_dir")"
        git clone --filter=blob:none --no-checkout "$repo_url" "$source_dir"
    fi
    git -C "$source_dir" fetch --depth 1 origin "$ref"
    git -C "$source_dir" checkout --detach FETCH_HEAD
}

private_dependencies_ready() {
    [[ -f $NATIVE_DEPS_PREFIX/include/yaml-cpp/yaml.h ]] &&
        compgen -G "$NATIVE_DEPS_PREFIX/lib*/libyaml-cpp.*" >/dev/null &&
        [[ -f $NATIVE_DEPS_PREFIX/include/msgpack.hpp ]] &&
        [[ -f $NATIVE_DEPS_PREFIX/include/asio.hpp ]] &&
        find "$NATIVE_DEPS_PREFIX" -type f \
            -name 'yalantinglibsConfig.cmake' -print -quit | grep -q .
}

prepare_private_dependencies() {
    if private_dependencies_ready; then
        echo "[mooncake-build] private native dependencies are ready"
        return
    fi

    echo "[mooncake-build] building pinned private native dependencies"
    safe_remove_work_dir "$NATIVE_DEPS_PREFIX"
    safe_remove_work_dir "$NATIVE_DEPS_BUILD_DIR"
    mkdir -p "$NATIVE_DEPS_PREFIX" "$NATIVE_DEPS_BUILD_DIR"

    local yaml_source=$NATIVE_DEPS_SOURCE_DIR/yaml-cpp
    local yaml_build=$NATIVE_DEPS_BUILD_DIR/yaml-cpp
    checkout_private_dependency "$YAML_CPP_REPO_URL" "$YAML_CPP_REF" \
        "$yaml_source"
    cmake -S "$yaml_source" -B "$yaml_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="$NATIVE_DEPS_PREFIX" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DYAML_BUILD_SHARED_LIBS=OFF \
        -DYAML_CPP_BUILD_TESTS=OFF \
        -DYAML_CPP_BUILD_TOOLS=OFF \
        -DYAML_CPP_BUILD_CONTRIB=OFF
    cmake --build "$yaml_build" --parallel "$MOONCAKE_JOBS"
    cmake --install "$yaml_build"

    local msgpack_source=$NATIVE_DEPS_SOURCE_DIR/msgpack-c
    local msgpack_build=$NATIVE_DEPS_BUILD_DIR/msgpack-c
    checkout_private_dependency "$MSGPACK_REPO_URL" "$MSGPACK_REF" \
        "$msgpack_source"
    cmake -S "$msgpack_source" -B "$msgpack_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="$NATIVE_DEPS_PREFIX" \
        -DMSGPACK_BUILD_EXAMPLES=OFF \
        -DMSGPACK_BUILD_TESTS=OFF \
        -DMSGPACK_BUILD_DOCS=OFF \
        -DMSGPACK_USE_BOOST=OFF
    cmake --build "$msgpack_build" --parallel "$MOONCAKE_JOBS"
    cmake --install "$msgpack_build"

    local yalanting_source=$MOONCAKE_SOURCE_DIR/extern/yalantinglibs
    local yalanting_build=$NATIVE_DEPS_BUILD_DIR/yalantinglibs
    if [[ ! -f $yalanting_source/CMakeLists.txt ]]; then
        echo "[mooncake-build] yalantinglibs submodule is missing" >&2
        return 1
    fi
    cmake -S "$yalanting_source" -B "$yalanting_build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_INSTALL_PREFIX="$NATIVE_DEPS_PREFIX" \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_BENCHMARK=OFF \
        -DBUILD_UNIT_TESTS=OFF \
        -DINSTALL_THIRDPARTY=ON \
        -DINSTALL_INDEPENDENT_THIRDPARTY=ON \
        -DINSTALL_STANDALONE=ON \
        -DINSTALL_INDEPENDENT_STANDALONE=ON
    cmake --build "$yalanting_build" --parallel "$MOONCAKE_JOBS"
    cmake --install "$yalanting_build"

    private_dependencies_ready || {
        echo "[mooncake-build] private dependency installation is incomplete" >&2
        return 1
    }
}

dependency_prefix_path() {
    local value=$NATIVE_DEPS_PREFIX
    if [[ -n ${CMAKE_PREFIX_PATH:-} ]]; then
        value="$value;${CMAKE_PREFIX_PATH//:/;}"
    fi
    echo "$value"
}

activate_private_build_environment() {
    if [[ ${DSA_MOONCAKE_PRIVATE_BUILD_ENV_ACTIVE:-0} == 1 ]]; then
        return
    fi
    export CPLUS_INCLUDE_PATH="$NATIVE_DEPS_PREFIX/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    export C_INCLUDE_PATH="$NATIVE_DEPS_PREFIX/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
    export LIBRARY_PATH="$NATIVE_DEPS_PREFIX/lib:$NATIVE_DEPS_PREFIX/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LDFLAGS="-L$NATIVE_DEPS_PREFIX/lib -L$NATIVE_DEPS_PREFIX/lib64${LDFLAGS:+ $LDFLAGS}"
    DSA_MOONCAKE_PRIVATE_BUILD_ENV_ACTIVE=1
}

check_native_dependencies() {
    activate_private_build_environment
    safe_remove_work_dir "$DEPENDENCY_PROBE_DIR"
    mkdir -p "$DEPENDENCY_PROBE_DIR/source"
    cat >"$DEPENDENCY_PROBE_DIR/source/probe.cpp" <<'EOF'
#include <asio.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <json/json.h>
#include <msgpack.hpp>
#include <xxhash.h>
#include <yaml-cpp/yaml.h>

int main() { return 0; }
EOF
    cat >"$DEPENDENCY_PROBE_DIR/source/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(dsa_mooncake_dependency_probe CXX)
list(APPEND CMAKE_MODULE_PATH "$MOONCAKE_SOURCE_DIR/mooncake-common")
include(FindJsonCpp)
include(FindGLOG)
find_package(yaml-cpp REQUIRED)
set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
find_package(gflags REQUIRED)
find_package(yalantinglibs CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h)
find_library(IBVERBS_LIBRARY ibverbs)
find_library(NUMA_LIBRARY numa)
find_library(ZSTD_LIBRARY zstd)
find_path(XXHASH_INCLUDE_DIR xxhash.h)
find_library(XXHASH_LIBRARY NAMES xxhash libxxhash)
find_path(MSGPACK_INCLUDE_DIR msgpack.hpp)
find_path(ASIO_INCLUDE_DIR asio.hpp)
foreach(required_value
    IBVERBS_INCLUDE_DIR IBVERBS_LIBRARY NUMA_LIBRARY ZSTD_LIBRARY
    XXHASH_INCLUDE_DIR XXHASH_LIBRARY MSGPACK_INCLUDE_DIR ASIO_INCLUDE_DIR)
  if(NOT \${required_value})
    message(FATAL_ERROR "Required dependency is unavailable: \${required_value}")
  endif()
endforeach()
if(NOT TARGET JsonCpp::JsonCpp)
  message(FATAL_ERROR "JsonCpp target is unavailable")
endif()
if(NOT TARGET glog::glog)
  message(FATAL_ERROR "glog target is unavailable")
endif()
if(NOT TARGET gflags::gflags)
  message(FATAL_ERROR "gflags target is unavailable")
endif()
if(NOT TARGET yalantinglibs::yalantinglibs)
  message(FATAL_ERROR "yalantinglibs target is unavailable")
endif()
add_executable(mooncake_dependency_probe probe.cpp)
target_include_directories(mooncake_dependency_probe PRIVATE
  \${MSGPACK_INCLUDE_DIR} \${ASIO_INCLUDE_DIR} \${XXHASH_INCLUDE_DIR})
target_link_libraries(mooncake_dependency_probe PRIVATE
  yaml-cpp JsonCpp::JsonCpp glog::glog gflags::gflags
  yalantinglibs::yalantinglibs
  \${IBVERBS_LIBRARY} \${NUMA_LIBRARY} \${ZSTD_LIBRARY} \${XXHASH_LIBRARY})
message(STATUS "Mooncake native dependency probe passed")
EOF

    local prefix_path
    prefix_path=$(dependency_prefix_path)
    if ! cmake -S "$DEPENDENCY_PROBE_DIR/source" \
        -B "$DEPENDENCY_PROBE_DIR/build" \
        -DCMAKE_PREFIX_PATH="$prefix_path" \
        -DCMAKE_INCLUDE_PATH="$NATIVE_DEPS_PREFIX/include" \
        -DCMAKE_LIBRARY_PATH="$NATIVE_DEPS_PREFIX/lib;$NATIVE_DEPS_PREFIX/lib64"; then
        echo "[mooncake-build] native dependency preflight failed" >&2
        echo "[mooncake-build] run: bash $0 deps" >&2
        return 1
    fi
    if ! cmake --build "$DEPENDENCY_PROBE_DIR/build" --parallel 1; then
        echo "[mooncake-build] native dependency link probe failed" >&2
        echo "[mooncake-build] run: bash $0 deps" >&2
        return 1
    fi
}

prepare_all_dependencies() {
    install_system_dependencies
    require_tools
    prepare_source
    prepare_private_dependencies
    check_native_dependencies
}

audit_install() {
    "$PYTHON_BIN" - <<'PY'
import importlib.metadata as metadata
import importlib.util
import platform
import sys

print("[mooncake-build] python", sys.version.replace("\n", " "))
print("[mooncake-build] platform", platform.platform())
for distribution in (
    "mooncake-transfer-engine-npu",
    "mooncake-transfer-engine-non-cuda",
    "mooncake-transfer-engine",
    "torch",
    "torch-npu",
):
    try:
        value = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        value = "NOT_INSTALLED"
    print(f"[mooncake-build] distribution {distribution}={value}")
for module in ("mooncake", "mooncake.engine", "mooncake.store"):
    spec = importlib.util.find_spec(module)
    print(f"[mooncake-build] module {module}={None if spec is None else spec.origin}")
PY
}

prepare_source() {
    mkdir -p "$(dirname -- "$MOONCAKE_SOURCE_DIR")"
    if [[ ! -d $MOONCAKE_SOURCE_DIR/.git ]]; then
        echo "[mooncake-build] cloning $MOONCAKE_REPO_URL"
        git clone --filter=blob:none --no-checkout \
            "$MOONCAKE_REPO_URL" "$MOONCAKE_SOURCE_DIR"
        touch "$MANAGED_MARKER"
    fi

    if [[ -f $MANAGED_MARKER ]]; then
        git -C "$MOONCAKE_SOURCE_DIR" fetch --depth 1 origin "$MOONCAKE_REF"
        git -C "$MOONCAKE_SOURCE_DIR" checkout --detach FETCH_HEAD
    else
        local current
        current=$(git -C "$MOONCAKE_SOURCE_DIR" rev-parse HEAD)
        if [[ $current != "$MOONCAKE_REF" ]]; then
            echo "[mooncake-build] using caller-managed source at $current"
            echo "[mooncake-build] requested reference is $MOONCAKE_REF"
        fi
    fi

    if [[ -n $(git -C "$MOONCAKE_SOURCE_DIR" status --porcelain --untracked-files=no) ]]; then
        echo "[mooncake-build] tracked source changes detected; refusing to build" >&2
        git -C "$MOONCAKE_SOURCE_DIR" status --short
        return 1
    fi
    git -C "$MOONCAKE_SOURCE_DIR" submodule sync --recursive
    git -C "$MOONCAKE_SOURCE_DIR" submodule update --init --recursive
    echo "[mooncake-build] source=$(git -C "$MOONCAKE_SOURCE_DIR" rev-parse HEAD)"
}

prepare_build_venv() {
    if [[ -f $BUILD_VENV/pyvenv.cfg ]] &&
        grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' \
            "$BUILD_VENV/pyvenv.cfg"; then
        echo "[mooncake-build] recreating legacy system-site-packages build venv"
        safe_remove_work_dir "$BUILD_VENV"
    fi
    if [[ ! -x $BUILD_VENV/bin/python ]]; then
        "$PYTHON_BIN" -m venv "$BUILD_VENV"
    fi
    cat >"$BUILD_CONSTRAINTS" <<'EOF'
setuptools>=77,<81
EOF
    env -u PYTHONPATH PYTHONNOUSERSITE=1 \
        "$BUILD_VENV/bin/python" -m pip install --upgrade \
        --constraint "$BUILD_CONSTRAINTS" \
        'setuptools>=77,<81' 'build>=1,<2' wheel auditwheel
}

configure_and_build() {
    source_cann_if_needed
    prepare_source
    prepare_private_dependencies
    prepare_build_venv
    check_native_dependencies

    if [[ $MOONCAKE_CLEAN_BUILD == 1 && -d $BUILD_DIR ]]; then
        case $BUILD_DIR in
            "$MOONCAKE_SOURCE_DIR"/.dsa-build-*) rm -rf -- "$BUILD_DIR" ;;
            *)
                echo "[mooncake-build] refusing to clean unexpected build dir: $BUILD_DIR" >&2
                return 1
                ;;
        esac
    fi

    local generator_args=()
    if command -v ninja >/dev/null 2>&1; then
        generator_args=(-G Ninja)
    fi
    local prefix_path
    prefix_path=$(dependency_prefix_path)
    cmake -S "$MOONCAKE_SOURCE_DIR" -B "$BUILD_DIR" \
        "${generator_args[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_PREFIX_PATH="$prefix_path" \
        -DCMAKE_INCLUDE_PATH="$NATIVE_DEPS_PREFIX/include" \
        -DCMAKE_LIBRARY_PATH="$NATIVE_DEPS_PREFIX/lib;$NATIVE_DEPS_PREFIX/lib64" \
        -DPYTHON_EXECUTABLE="$BUILD_VENV/bin/python" \
        -DPython3_EXECUTABLE="$BUILD_VENV/bin/python" \
        -DUSE_CUDA=OFF \
        -DUSE_TCP=ON \
        -DUSE_ASCEND=OFF \
        -DUSE_ASCEND_DIRECT=ON \
        -DUSE_UBSHMEM=OFF \
        -DUSE_HTTP=ON \
        -DUSE_ETCD=OFF \
        -DSTORE_USE_ETCD=OFF \
        -DWITH_TE=ON \
        -DWITH_STORE=ON \
        -DWITH_STORE_GO=OFF \
        -DWITH_STORE_RUST=OFF \
        -DWITH_EP=OFF \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_UNIT_TESTS=OFF \
        -DBUILD_BENCHMARK=OFF
    cmake --build "$BUILD_DIR" --parallel "$MOONCAKE_JOBS"

    grep -E '^(USE_TCP|USE_ASCEND_DIRECT|USE_CUDA|WITH_STORE):' \
        "$BUILD_DIR/CMakeCache.txt"

    mkdir -p "$WHEEL_DIR"
    (
        cd "$MOONCAKE_SOURCE_DIR"
        env \
            PATH="$BUILD_VENV/bin:$PATH" \
            PYTHONPATH= \
            PYTHONNOUSERSITE=1 \
            CMAKE_PREFIX_PATH="$NATIVE_DEPS_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" \
            CMAKE_INCLUDE_PATH="$NATIVE_DEPS_PREFIX/include${CMAKE_INCLUDE_PATH:+:$CMAKE_INCLUDE_PATH}" \
            CMAKE_LIBRARY_PATH="$NATIVE_DEPS_PREFIX/lib:$NATIVE_DEPS_PREFIX/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}" \
            LD_LIBRARY_PATH="$NATIVE_DEPS_PREFIX/lib:$NATIVE_DEPS_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            PIP_CONSTRAINT="$BUILD_CONSTRAINTS" \
            PYTHON_VERSION="$PYTHON_VERSION" \
            OUTPUT_DIR="$WHEEL_DIR" \
            BUILD_DIR="$BUILD_DIR_NAME" \
            NPU_BUILD=1 \
            NON_CUDA_BUILD=0 \
            CU13_BUILD=0 \
            CI=false \
            FREE_BUILD_DIR=0 \
            bash scripts/build_wheel.sh
    )
    newest_wheel
}

newest_wheel() {
    local wheel
    wheel=$(find "$WHEEL_DIR" -maxdepth 1 -type f \
        -name 'mooncake_transfer_engine_npu-*.whl' \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    if [[ -z $wheel || ! -f $wheel ]]; then
        echo "[mooncake-build] no NPU wheel found under $WHEEL_DIR" >&2
        return 1
    fi
    echo "$wheel"
}

install_stage() {
    local wheel
    wheel=$(newest_wheel)
    safe_remove_work_dir "$STAGE_DIR"
    mkdir -p "$STAGE_DIR"
    "$PYTHON_BIN" -m pip install --no-deps --target "$STAGE_DIR" "$wheel"

    cat >"$ACTIVATE_FILE" <<EOF
#!/usr/bin/env bash
export DSA_MOONCAKE_CUSTOM_SITE="$STAGE_DIR"
export PYTHONPATH="$STAGE_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export PATH="$STAGE_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="$STAGE_DIR/mooncake:$NATIVE_DEPS_PREFIX/lib:$NATIVE_DEPS_PREFIX/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
    chmod +x "$ACTIVATE_FILE"
    echo "[mooncake-build] staged wheel=$wheel"
    echo "[mooncake-build] activate with: source $ACTIVATE_FILE"
}

verify_protocol() {
    local protocol=$1
    if [[ ! -d $STAGE_DIR/mooncake ]]; then
        echo "[mooncake-build] staged package is missing; run install first" >&2
        return 1
    fi
    env \
        PYTHONPATH="$STAGE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        LD_LIBRARY_PATH="$STAGE_DIR/mooncake:$NATIVE_DEPS_PREFIX/lib:$NATIVE_DEPS_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$PYTHON_BIN" "$SCRIPT_DIR/test_mooncake_transport_dispatch.py" \
        --protocol "$protocol" \
        --device-id "$DEVICE_ID" \
        --expected-site "$STAGE_DIR"
}

mode=${1:-}
if [[ -z $mode || $mode == -h || $mode == --help || $mode == help ]]; then
    usage
    exit 0
fi

resolve_work_root
initialize_derived_paths

case $mode in
    deps|bootstrap)
        command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
            echo "[mooncake-build] missing Python interpreter: $PYTHON_BIN" >&2
            exit 1
        }
        ;;
    audit|print-env)
        ;;
    *)
        require_tools
        ;;
esac

case $mode in
    audit)
        audit_install
        ;;
    check-deps)
        prepare_source
        prepare_private_dependencies
        check_native_dependencies
        ;;
    deps)
        prepare_all_dependencies
        ;;
    build)
        configure_and_build
        ;;
    install)
        install_stage
        ;;
    verify-tcp)
        verify_protocol tcp
        ;;
    verify-ascend)
        source_cann_if_needed
        verify_protocol ascend
        ;;
    verify-all)
        verify_protocol tcp
        source_cann_if_needed
        verify_protocol ascend
        ;;
    all)
        audit_install
        configure_and_build
        install_stage
        verify_protocol tcp
        source_cann_if_needed
        verify_protocol ascend
        ;;
    bootstrap)
        audit_install
        prepare_all_dependencies
        configure_and_build
        install_stage
        verify_protocol tcp
        source_cann_if_needed
        verify_protocol ascend
        ;;
    print-env)
        echo "source $ACTIVATE_FILE"
        ;;
    *)
        echo "[mooncake-build] unknown mode: $mode" >&2
        usage
        exit 2
        ;;
esac
