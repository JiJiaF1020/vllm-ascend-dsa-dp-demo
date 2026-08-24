#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone MemCache v1.2 IO probe for Ascend A5.

This script intentionally does not import vLLM or vLLM-Ascend. It validates
the public memcache_hybrid API in three increasingly strict modes:

* capabilities: import/package/API inspection only;
* host: Host DRAM -> global pool -> Host DRAM scatter-gather round trip;
* npu: NPU HBM -> global pool -> NPU HBM scatter-gather round trip.
* gva: explicit global-address allocation, copy, publish and lease round trip.

By default the script starts an in-process MemCache MetaService in a spawned
child process. Pass --external-meta when a MetaService is already running.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
import uuid
from enum import IntEnum
from typing import Any


class MmcDirection(IntEnum):
    LOCAL_NPU_TO_GLOBAL = 0
    GLOBAL_TO_LOCAL_NPU = 1
    GLOBAL_TO_HOST = 2
    HOST_TO_GLOBAL = 3
    AUTO = 9


MEMCACHE_PROTOCOLS = (
    "host_rdma",
    "host_urma",
    "host_tcp",
    "host_shm",
    "device_sdma",
    "device_rdma",
    "device_urma",
    "device_uboe",
)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def print_capabilities() -> None:
    print("[memcache-probe] package versions")
    for name in ("memcache_hybrid", "memfabric_hybrid", "torch", "torch_npu"):
        print(f"  {name}={_distribution_version(name)}")

    try:
        import memcache_hybrid
        from memcache_hybrid import (
            DistributedObjectStore,
            LocalConfig,
            MetaConfig,
            MetaService,
        )
    except Exception:
        print("[memcache-probe] import failed")
        traceback.print_exc()
        raise

    print(f"[memcache-probe] module={memcache_hybrid.__file__}")
    classes = {
        "DistributedObjectStore": DistributedObjectStore,
        "LocalConfig": LocalConfig,
        "MetaConfig": MetaConfig,
        "MetaService": MetaService,
    }
    for name, cls in classes.items():
        print(f"  class {name}: {cls!r}")

    required_store_methods = (
        "setup",
        "init",
        "close",
        "get_local_service_id",
        "is_exist",
        "put",
        "get",
        "batch_is_exist",
        "batch_put_from_layers",
        "batch_get_into_layers",
        "batch_get_key_info",
        "batch_alloc",
        "batch_add_lease",
        "batch_remove_lease",
        "batch_copy",
        "batch_write_finish",
        "register_buffer",
        "unregister_buffer",
    )
    missing = [name for name in required_store_methods if not hasattr(DistributedObjectStore, name)]
    for name in required_store_methods:
        print(f"  API {name}: {hasattr(DistributedObjectStore, name)}")
    if missing:
        raise RuntimeError(f"missing required MemCache APIs: {missing}")


def _parse_tcp_url(url: str) -> tuple[str, int]:
    prefix = "tcp://"
    if not url.startswith(prefix):
        raise ValueError(f"expected tcp:// URL, got {url!r}")
    host, port_text = url[len(prefix) :].rsplit(":", 1)
    return host, int(port_text)


def _port_open(url: str, timeout: float = 0.2) -> bool:
    host, port = _parse_tcp_url(url)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(
    url: str,
    process: mp.Process,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _port_open(url):
            return
        if not process.is_alive():
            raise RuntimeError(f"MetaService exited before {url} became ready: exitcode={process.exitcode}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for MetaService at {url}")


def _run_meta_service(
    meta_url: str,
    config_store_url: str,
    metrics_url: str,
    log_level: str,
) -> None:
    from memcache_hybrid import MetaConfig, MetaService

    config = MetaConfig()
    config.meta_service_url = meta_url
    config.config_store_url = config_store_url
    config.metrics_url = metrics_url
    config.log_level = log_level
    if hasattr(config, "ha_enable"):
        config.ha_enable = False

    result = MetaService.setup(config)
    if result != 0:
        raise RuntimeError(f"MetaService.setup failed: result={result}")
    MetaService.main()


def start_meta_service(args: argparse.Namespace) -> mp.Process | None:
    if args.external_meta:
        if not _port_open(args.meta_url):
            raise RuntimeError(f"--external-meta was set but meta service is unreachable: {args.meta_url}")
        return None

    occupied = [url for url in (args.meta_url, args.config_store_url) if _port_open(url)]
    if occupied:
        raise RuntimeError(
            "refusing to start a private MetaService because ports are "
            f"already in use: {occupied}; use --external-meta if expected"
        )

    context = mp.get_context("spawn")
    process = context.Process(
        target=_run_meta_service,
        args=(
            args.meta_url,
            args.config_store_url,
            args.metrics_url,
            args.log_level,
        ),
        name="memcache-probe-meta",
    )
    process.start()
    _wait_for_port(args.meta_url, process, args.startup_timeout)
    print(f"[memcache-probe] MetaService ready pid={process.pid}")
    return process


def _set_config_attr(config: Any, name: str, value: Any) -> None:
    if not hasattr(config, name):
        raise RuntimeError(
            f"memcache_hybrid LocalConfig lacks {name!r}; the installed package does not match the v1.2 API"
        )
    setattr(config, name, value)


def create_store(args: argparse.Namespace):
    from memcache_hybrid import DistributedObjectStore, LocalConfig

    config = LocalConfig()
    values = {
        "meta_service_url": args.meta_url,
        "config_store_url": args.config_store_url,
        "log_level": args.log_level,
        "world_size": args.world_size,
        "protocol": args.protocol,
        "dram_size": args.dram_size,
        "hbm_size": "0",
        "max_dram_size": args.max_dram_size,
        "max_hbm_size": "0",
        "client_retry_milliseconds": args.client_retry_ms,
        "client_timeout_seconds": args.client_timeout_seconds,
    }
    for name, value in values.items():
        _set_config_attr(config, name, value)

    print("[memcache-probe] LocalConfig")
    for name, value in values.items():
        print(f"  {name}={value}")

    store = DistributedObjectStore()
    result = store.setup(config)
    if result != 0:
        raise RuntimeError(f"DistributedObjectStore.setup failed: {result}")
    result = store.init(args.device_id)
    if result != 0:
        raise RuntimeError(f"DistributedObjectStore.init failed: {result}")
    print(f"[memcache-probe] store initialized local_service_id={store.get_local_service_id()}")
    return store


def prepare_npu_device(device_id: int) -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    torch.npu.synchronize()
    print(f"[memcache-probe] NPU device ready device={device_id} name={torch.npu.get_device_name(device_id)}")


def _check_statuses(operation: str, statuses: Any, expected_count: int) -> None:
    values = list(statuses)
    print(f"[memcache-probe] {operation} statuses={values}")
    if len(values) != expected_count or any(int(value) != 0 for value in values):
        raise RuntimeError(f"{operation} failed: expected {expected_count} zeros, got {values}")


def _check_exists(store: Any, keys: list[str]) -> None:
    result = list(store.batch_is_exist(keys))
    print(f"[memcache-probe] exists={result}")
    if len(result) != len(keys) or any(int(value) != 1 for value in result):
        raise RuntimeError(f"unexpected batch_is_exist result: {result}")


def _make_key(prefix: str, index: int) -> str:
    return f"memcache-a5-probe-{prefix}-{os.getpid()}-{uuid.uuid4().hex}-{index}"


def run_basic_object_round_trip(store: Any) -> None:
    key = _make_key("bytes", 0)
    payload = b"memcache-a5-basic-object-round-trip"
    before = int(store.is_exist(key))
    if before != 0:
        raise RuntimeError(f"fresh key unexpectedly exists: key={key} state={before}")
    put_result = int(store.put(key, payload))
    if put_result != 0:
        raise RuntimeError(f"basic put failed: result={put_result}")
    after = int(store.is_exist(key))
    if after != 1:
        raise RuntimeError(f"basic key is not visible after put: state={after}")
    restored = store.get(key)
    if restored != payload:
        raise AssertionError(f"basic object mismatch: expected={payload!r}, actual={restored!r}")
    print("[memcache-probe] BASIC OBJECT ROUND TRIP PASSED")


def run_host_round_trip(store: Any) -> None:
    import torch

    sizes_by_key = ([4096, 6144, 8192], [3072, 5120, 9216])
    keys = [_make_key("host", index) for index in range(len(sizes_by_key))]
    source: list[list[torch.Tensor]] = []
    destination: list[list[torch.Tensor]] = []

    for key_index, sizes in enumerate(sizes_by_key):
        source_layers: list[torch.Tensor] = []
        destination_layers: list[torch.Tensor] = []
        for layer_index, size in enumerate(sizes):
            pattern = (torch.arange(size, dtype=torch.int64) + key_index * 37 + layer_index * 19) % 251
            source_layers.append(pattern.to(torch.uint8).contiguous())
            destination_layers.append(torch.zeros(size, dtype=torch.uint8))
        source.append(source_layers)
        destination.append(destination_layers)

    put_status = store.batch_put_from_layers(
        keys,
        [[tensor.data_ptr() for tensor in layers] for layers in source],
        [list(sizes) for sizes in sizes_by_key],
        int(MmcDirection.HOST_TO_GLOBAL),
    )
    _check_statuses("host put", put_status, len(keys))
    _check_exists(store, keys)

    get_status = store.batch_get_into_layers(
        keys,
        [[tensor.data_ptr() for tensor in layers] for layers in destination],
        [list(sizes) for sizes in sizes_by_key],
        int(MmcDirection.GLOBAL_TO_HOST),
    )
    _check_statuses("host get", get_status, len(keys))

    for key, src_layers, dst_layers in zip(keys, source, destination):
        for layer_index, (src, dst) in enumerate(zip(src_layers, dst_layers)):
            if not torch.equal(src, dst):
                raise AssertionError(f"host round trip mismatch: key={key} layer={layer_index}")
    print("[memcache-probe] HOST ROUND TRIP PASSED")


def _register_tensor(store: Any, tensor: Any) -> tuple[int, int]:
    pointer = int(tensor.data_ptr())
    size = int(tensor.numel() * tensor.element_size())
    result = store.register_buffer(pointer, size)
    if result != 0:
        raise RuntimeError(f"register_buffer failed: ptr=0x{pointer:x} size={size} ret={result}")
    return pointer, size


def run_npu_round_trip(store: Any, device_id: int) -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    num_keys = 2
    num_layers = 4
    # Two keys x four layers x 256 KiB = 2 MiB per backing tensor. This is
    # intentionally closer to the registration granularity used by Ascend
    # cache buffers than a tiny toy allocation.
    bytes_per_layer = 256 * 1024

    cpu_pattern = (
        torch.arange(
            num_keys * num_layers * bytes_per_layer,
            dtype=torch.int64,
        ).reshape(num_keys, num_layers, bytes_per_layer)
        % 251
    )
    source = cpu_pattern.to(torch.uint8).to(device).contiguous()
    destination = torch.zeros_like(source)
    torch.npu.synchronize()

    registered: list[tuple[int, int]] = []
    try:
        registered.append(_register_tensor(store, source))
        registered.append(_register_tensor(store, destination))

        keys = [_make_key("npu", index) for index in range(num_keys)]
        sizes = [bytes_per_layer] * num_layers
        put_status = store.batch_put_from_layers(
            keys,
            [[source[key_index, layer].data_ptr() for layer in range(num_layers)] for key_index in range(num_keys)],
            [sizes[:] for _ in range(num_keys)],
            int(MmcDirection.LOCAL_NPU_TO_GLOBAL),
        )
        _check_statuses("npu put", put_status, len(keys))
        _check_exists(store, keys)

        get_status = store.batch_get_into_layers(
            keys,
            [
                [destination[key_index, layer].data_ptr() for layer in range(num_layers)]
                for key_index in range(num_keys)
            ],
            [sizes[:] for _ in range(num_keys)],
            int(MmcDirection.GLOBAL_TO_LOCAL_NPU),
        )
        _check_statuses("npu get", get_status, len(keys))
        torch.npu.synchronize()

        if not torch.equal(source.cpu(), destination.cpu()):
            mismatch = torch.nonzero(source.cpu() != destination.cpu())
            first = mismatch[0].tolist() if mismatch.numel() else None
            raise AssertionError(f"NPU round trip mismatch: first={first}")
        print("[memcache-probe] NPU ROUND TRIP PASSED")
    finally:
        for pointer, size in reversed(registered):
            try:
                result = store.unregister_buffer(pointer, size)
                print(f"[memcache-probe] unregister_buffer ptr=0x{pointer:x} size={size} ret={result}")
            except Exception:
                traceback.print_exc()


def run_gva_round_trip(store: Any, device_id: int) -> None:
    """Exercise the GVA/lease APIs used by AscendStore layerwise mode."""

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    num_keys = 2
    bytes_per_key = 1024 * 1024
    keys = [_make_key("gva", index) for index in range(num_keys)]
    sizes = [bytes_per_key] * num_keys

    cpu_pattern = (
        torch.arange(
            num_keys * bytes_per_key,
            dtype=torch.int64,
        ).reshape(num_keys, bytes_per_key)
        % 251
    )
    source = cpu_pattern.to(torch.uint8).to(device).contiguous()
    destination = torch.zeros_like(source)
    torch.npu.synchronize()

    registered: list[tuple[int, int]] = []
    lease_added = False
    try:
        registered.append(_register_tensor(store, source))
        registered.append(_register_tensor(store, destination))

        gvas = [int(value) for value in store.batch_alloc(keys, sizes)]
        print(f"[memcache-probe] batch_alloc gvas={gvas}")
        if len(gvas) != num_keys or any(gva <= 0 for gva in gvas):
            raise RuntimeError(f"batch_alloc failed: gvas={gvas}")

        copy_result = int(
            store.batch_copy(
                gvas,
                [source[index].data_ptr() for index in range(num_keys)],
                sizes,
                int(MmcDirection.LOCAL_NPU_TO_GLOBAL),
            )
        )
        print(f"[memcache-probe] GVA write batch_copy ret={copy_result}")
        if copy_result != 0:
            raise RuntimeError(f"GVA write batch_copy failed: {copy_result}")

        finish_results = [
            int(value)
            for value in store.batch_write_finish(
                keys,
                [copy_result] * num_keys,
            )
        ]
        print(f"[memcache-probe] batch_write_finish={finish_results}")
        if len(finish_results) != num_keys or any(finish_results):
            raise RuntimeError(f"batch_write_finish failed: {finish_results}")
        _check_exists(store, keys)

        key_infos = list(store.batch_get_key_info(keys))
        if len(key_infos) != num_keys:
            raise RuntimeError(f"batch_get_key_info returned wrong count: expected={num_keys} actual={len(key_infos)}")
        load_gvas: list[int] = []
        for key, info in zip(keys, key_infos):
            info_size = int(info.size())
            info_gvas = [int(value) for value in info.gva_list()]
            print(
                "[memcache-probe] key_info "
                f"key={key} size={info_size} gvas={info_gvas} "
                f"types={list(info.type_list())} locations={list(info.loc_list())}"
            )
            if info_size <= 0 or not info_gvas or info_gvas[0] <= 0:
                raise RuntimeError(f"invalid KeyInfo for {key}: {info}")
            load_gvas.append(info_gvas[0])

        lease_results = [int(value) for value in store.batch_add_lease(keys, 5 * 60 * 1000)]
        print(f"[memcache-probe] batch_add_lease={lease_results}")
        if len(lease_results) != num_keys or any(lease_results):
            raise RuntimeError(f"batch_add_lease failed: {lease_results}")
        lease_added = True

        copy_result = int(
            store.batch_copy(
                load_gvas,
                [destination[index].data_ptr() for index in range(num_keys)],
                sizes,
                int(MmcDirection.GLOBAL_TO_LOCAL_NPU),
            )
        )
        print(f"[memcache-probe] GVA read batch_copy ret={copy_result}")
        if copy_result != 0:
            raise RuntimeError(f"GVA read batch_copy failed: {copy_result}")
        torch.npu.synchronize()

        if not torch.equal(source.cpu(), destination.cpu()):
            mismatch = torch.nonzero(source.cpu() != destination.cpu())
            first = mismatch[0].tolist() if mismatch.numel() else None
            raise AssertionError(f"GVA round trip mismatch: first={first}")
        print("[memcache-probe] GVA/LAYERWISE API ROUND TRIP PASSED")
    finally:
        if lease_added:
            try:
                result = int(store.batch_remove_lease(keys))
                print(f"[memcache-probe] batch_remove_lease ret={result}")
            except Exception:
                traceback.print_exc()
        for pointer, size in reversed(registered):
            try:
                result = store.unregister_buffer(pointer, size)
                print(f"[memcache-probe] unregister_buffer ptr=0x{pointer:x} size={size} ret={result}")
            except Exception:
                traceback.print_exc()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("capabilities", "host", "npu", "gva", "both", "all"),
        default="capabilities",
    )
    parser.add_argument(
        "--protocol",
        choices=MEMCACHE_PROTOCOLS,
        default="host_shm",
        help="MemFabric transport/data-operation protocol",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dram-size", default="1GB")
    parser.add_argument("--max-dram-size", default="256GB")
    parser.add_argument("--world-size", type=int, default=256)
    parser.add_argument("--meta-url", default="tcp://127.0.0.1:5000")
    parser.add_argument("--config-store-url", default="tcp://127.0.0.1:6000")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:18000")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--client-retry-ms", type=int, default=10000)
    parser.add_argument("--client-timeout-seconds", type=int, default=60)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument(
        "--external-meta",
        action="store_true",
        help="reuse a MetaService already listening on --meta-url",
    )
    parser.add_argument(
        "--strict-native-teardown",
        action="store_true",
        help=(
            "let Python run MemCache v1.2.0 native finalizers; this currently "
            "reproduces the upstream ResourceTracker exit-order crash"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print_capabilities()
    if args.mode == "capabilities":
        return 0

    if args.mode in {"npu", "gva", "both", "all"} and not args.protocol.startswith("device_"):
        raise ValueError("NPU mode requires a device protocol, for example --protocol device_uboe or device_urma")

    meta_process: mp.Process | None = None
    store = None
    exit_code = 1
    try:
        meta_process = start_meta_service(args)
        if args.mode in {"npu", "gva", "both", "all"}:
            prepare_npu_device(args.device_id)
        store = create_store(args)
        run_basic_object_round_trip(store)
        if args.mode in {"host", "both", "all"}:
            run_host_round_trip(store)
        if args.mode in {"npu", "both", "all"}:
            run_npu_round_trip(store, args.device_id)
        if args.mode in {"gva", "all"}:
            run_gva_round_trip(store, args.device_id)
        print("[memcache-probe] ALL REQUESTED TESTS PASSED")
        exit_code = 0
    except Exception:
        traceback.print_exc()
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                traceback.print_exc()
                exit_code = 1
            store = None
            gc.collect()
        if meta_process is not None:
            # MetaService exposes setup()/main() but no graceful Python stop.
            # SIGTERM enters its native signal cleanup and double-frees in the
            # v1.2.0 wheel. This private process owns no persistent data, so
            # kill it after the LocalService has closed.
            if meta_process.is_alive():
                meta_process.kill()
                meta_process.join(timeout=5.0)
            print(f"[memcache-probe] MetaService stopped exitcode={meta_process.exitcode}")

    if args.strict_native_teardown:
        print("[memcache-probe] strict native teardown enabled")
        return exit_code

    # MemCache v1.2.0 constructs a function-static ResourceTracker whose
    # constructor also registers an atexit callback. At process exit the
    # singleton can be destroyed before that callback accesses its mutex and
    # instance set, causing a use-after-destruction even after explicit
    # DistributedObjectStore.close(). All data-plane resources are already
    # closed above; bypass only the broken extension/global finalizer chain.
    print(
        "[memcache-probe] explicit cleanup complete; bypassing MemCache "
        "v1.2.0 native finalizers"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("[memcache-probe] FAILED", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
