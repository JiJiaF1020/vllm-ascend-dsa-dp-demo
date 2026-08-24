#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inspect or provision the HYBM AICPU kernel required by A5 URMA."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import tarfile
from pathlib import Path


KERNEL_REL = Path("opp/vendors/cust/op_impl/aicpu/kernel")
CONFIG_REL = Path("opp/vendors/cust/op_impl/aicpu/config")
KERNEL_ARCHIVE = "cann-hybm-compat.tar.gz"
KERNEL_JSON = "libcann_hybm_kernel.json"
KERNEL_VERSION = "cann_hybm_kernel_version"
WHEEL_MARKER = ".hybm_aicpu_provision_wheel_only"


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _package_dir() -> Path | None:
    spec = importlib.util.find_spec("memfabric_hybrid")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _is_valid_json(path: Path) -> bool:
    try:
        with path.open(encoding="utf-8") as file:
            json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def _is_valid_kernel_archive(path: Path) -> bool:
    try:
        with tarfile.open(path, "r:gz") as archive:
            names = set(archive.getnames())
    except (OSError, tarfile.TarError):
        return False
    return "aicpu_kernels_device/libcann_hybm_kernel.so" in names


def _print_acl_info() -> None:
    try:
        import acl
    except Exception as exc:
        print(f"[memfabric-aicpu] acl import failed: {exc!r}")
        return

    for name in ("get_version", "get_soc_name"):
        function = getattr(acl, name, None)
        if function is None:
            print(f"[memfabric-aicpu] acl.{name}=MISSING")
            continue
        try:
            value = function()
        except Exception as exc:
            print(f"[memfabric-aicpu] acl.{name} failed: {exc!r}")
        else:
            print(f"[memfabric-aicpu] acl.{name}={value!r}")


def inspect() -> tuple[bool, bool]:
    ascend_home_raw = os.getenv("ASCEND_HOME_PATH", "")
    ascend_home = Path(ascend_home_raw).resolve() if ascend_home_raw else None
    package_dir = _package_dir()

    print(f"[memfabric-aicpu] memfabric_hybrid={_distribution_version('memfabric_hybrid')}")
    print(f"[memfabric-aicpu] ASCEND_HOME_PATH={ascend_home_raw or '<unset>'}")
    print(f"[memfabric-aicpu] package_dir={package_dir or '<not found>'}")
    _print_acl_info()

    auto_provision_files = (
        WHEEL_MARKER,
        "_provision.py",
        "VERSION",
        "_ops",
        "_hybm_src",
    )
    auto_provision_ready = package_dir is not None
    if package_dir is not None:
        for relative in auto_provision_files:
            path = package_dir / relative
            exists = path.exists()
            auto_provision_ready &= exists
            print(f"[memfabric-aicpu] wheel_asset {relative} exists={exists} path={path}")

    if ascend_home is None or not ascend_home.is_dir():
        print("[memfabric-aicpu] status=INVALID_ASCEND_HOME")
        return False, auto_provision_ready

    kernel_archive = ascend_home / KERNEL_REL / KERNEL_ARCHIVE
    kernel_json = ascend_home / CONFIG_REL / KERNEL_JSON
    kernel_version = ascend_home / CONFIG_REL / KERNEL_VERSION
    load_ini = ascend_home / "conf/ascend_package_load.ini"

    archive_valid = _is_valid_kernel_archive(kernel_archive)
    json_valid = _is_valid_json(kernel_json)
    version_present = kernel_version.is_file()
    for label, path, valid in (
        ("kernel_archive", kernel_archive, archive_valid),
        ("kernel_json", kernel_json, json_valid),
        ("kernel_version", kernel_version, version_present),
    ):
        print(f"[memfabric-aicpu] target {label} valid={valid} path={path}")

    ini_registered = False
    if load_ini.is_file():
        try:
            ini_registered = f"name:{KERNEL_ARCHIVE}" in load_ini.read_text(encoding="utf-8")
        except OSError:
            pass
    print(f"[memfabric-aicpu] ascend_package_load registered={ini_registered} path={load_ini}")

    installed = archive_valid and json_valid and version_present and ini_registered
    if installed:
        print("[memfabric-aicpu] status=READY")
    elif auto_provision_ready:
        print("[memfabric-aicpu] status=AUTO_PROVISION_AVAILABLE")
    else:
        print("[memfabric-aicpu] status=VERSION_MATCHED_RUN_REQUIRED")
    return installed, auto_provision_ready


def provision() -> int:
    installed, auto_provision_ready = inspect()
    if installed:
        return 0
    if not auto_provision_ready:
        raise RuntimeError(
            "the installed wheel has no self-provision assets; install "
            "memfabric_hybrid_aicpu_kernel.run built from the same native commit"
        )

    from memfabric_hybrid._provision import provision as provision_kernel

    print("[memfabric-aicpu] invoking wheel provision()")
    provision_kernel()
    installed, _ = inspect()
    if not installed:
        raise RuntimeError("wheel provision() returned but the AICPU kernel is still incomplete")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "provision"), default="check")
    args = parser.parse_args()
    if args.mode == "provision":
        return provision()
    installed, _ = inspect()
    return 0 if installed else 2


if __name__ == "__main__":
    raise SystemExit(main())
