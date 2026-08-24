#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone Mooncake A5 API probe matching vLLM-Ascend MooncakeBackend."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def print_capabilities() -> None:
    print(f"[mooncake-probe] distribution mooncake-transfer-engine-npu={_version('mooncake-transfer-engine-npu')}")
    # torch_npu has already initialized the device in main(). Keep Store before
    # TransferEngine to match vLLM-Ascend's backend import order.
    store_module = importlib.import_module("mooncake.store")
    engine_module = importlib.import_module("mooncake.engine")
    MooncakeDistributedStore = store_module.MooncakeDistributedStore
    ReplicateConfig = store_module.ReplicateConfig
    TransferEngine = engine_module.TransferEngine

    print(f"[mooncake-probe] engine={engine_module.__file__}")
    print(f"[mooncake-probe] store={store_module.__file__}")
    print(f"[mooncake-probe] master={shutil.which('mooncake_master')}")

    required_store_methods = (
        "setup",
        "close",
        "put",
        "get",
        "batch_is_exist",
        "batch_put_from_multi_buffers",
        "batch_get_into_multi_buffers",
    )
    required_engine_methods = (
        "initialize",
        "get_rpc_port",
        "get_engine",
        "register_memory",
        "unregister_memory",
    )
    for name in required_store_methods:
        present = hasattr(MooncakeDistributedStore, name)
        print(f"  store API {name}: {present}")
        if not present:
            raise RuntimeError(f"missing Mooncake store API: {name}")
    for name in required_engine_methods:
        present = hasattr(TransferEngine, name)
        print(f"  engine API {name}: {present}")
        if not present:
            raise RuntimeError(f"missing Mooncake engine API: {name}")
    print(f"  ReplicateConfig: {ReplicateConfig!r}")


def _parse_host_port(value: str) -> tuple[str, int]:
    host, port_text = value.rsplit(":", 1)
    return host, int(port_text)


def _port_open(address: str, timeout: float = 0.2) -> bool:
    host, port = _parse_host_port(address)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_master(
    address: str,
    process: subprocess.Popen | None,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _port_open(address):
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"mooncake_master exited before becoming ready: returncode={process.returncode}")
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for mooncake_master at {address}")


def start_master(args: argparse.Namespace) -> subprocess.Popen | None:
    if args.external_master:
        if not _port_open(args.master_address):
            raise RuntimeError(f"--external-master was set but the master is unreachable: {args.master_address}")
        return None

    if _port_open(args.master_address):
        raise RuntimeError(
            "master port is already occupied; use --external-master if expected: "
            f"{args.master_address}"
        )
    entrypoint = shutil.which("mooncake_master")
    if entrypoint is None:
        raise RuntimeError(
            "mooncake_master is not in PATH; the installed NPU wheel is missing its Store CLI entry point"
        )
    package_spec = importlib.util.find_spec("mooncake")
    native_binary = None
    if package_spec is not None and package_spec.submodule_search_locations:
        package_dir = next(iter(package_spec.submodule_search_locations))
        candidate = os.path.join(package_dir, "mooncake_master")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            native_binary = candidate
    binary = native_binary or entrypoint
    _, port = _parse_host_port(args.master_address)
    metrics_port = getattr(args, "metrics_port", 9003)
    command = [
        binary,
        f"--rpc_port={port}",
        f"--metrics_port={metrics_port}",
        "--eviction_high_watermark_ratio=0.9",
        "--eviction_ratio=0.1",
        f"--default_kv_lease_ttl={args.lease_ttl_ms}",
        f"--client_ttl={args.client_ttl_seconds}",
    ]
    print(f"[mooncake-probe] starting master: {command}")
    # The console entrypoint spawns a second native process. Start a dedicated
    # session and prefer the packaged native binary so cleanup can reap every
    # process that still owns the stdout pipe.
    process = subprocess.Popen(command, start_new_session=True)
    _wait_for_master(args.master_address, process, args.startup_timeout)
    print(f"[mooncake-probe] master ready pid={process.pid}")
    return process


def stop_master(process: subprocess.Popen) -> None:
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass

    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2.0)
    print(
        "[mooncake-probe] master process group stopped "
        f"pgid={process_group} returncode={process.returncode}"
    )


def _prepare_npu(device_id: int) -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    torch.npu.synchronize()
    print(f"[mooncake-probe] NPU ready device={device_id} name={torch.npu.get_device_name(device_id)}")


def create_store(args: argparse.Namespace):
    from mooncake.engine import TransferEngine
    from mooncake.store import MooncakeDistributedStore

    engine = TransferEngine()
    result = engine.initialize(
        args.local_host,
        "P2PHANDSHAKE",
        "ascend",
        args.device_name,
    )
    if result != 0:
        raise RuntimeError(f"TransferEngine.initialize failed: {result}")
    rpc_port = int(engine.get_rpc_port())
    local_segment = f"{args.local_host}:{rpc_port}"
    print(f"[mooncake-probe] TransferEngine ready local_segment={local_segment} device_name={args.device_name!r}")

    store = MooncakeDistributedStore()
    result = store.setup(
        local_hostname=local_segment,
        metadata_server="P2PHANDSHAKE",
        global_segment_size=args.global_segment_bytes,
        local_buffer_size=args.local_buffer_bytes,
        protocol="ascend",
        rdma_devices=args.device_name,
        master_server_addr=args.master_address,
        engine=engine.get_engine(),
    )
    if result != 0:
        raise RuntimeError(f"MooncakeDistributedStore.setup failed: {result}")
    print("[mooncake-probe] store initialized")
    return engine, store


def _key(prefix: str, index: int) -> str:
    return f"mooncake-a5-probe-{prefix}-{os.getpid()}-{uuid.uuid4().hex}-{index}"


def _wait_exists(store: Any, keys: list[str], timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    latest: list[int] = []
    while time.monotonic() < deadline:
        latest = [int(value) for value in store.batch_is_exist(keys)]
        if len(latest) == len(keys) and all(value == 1 for value in latest):
            print(f"[mooncake-probe] exists={latest}")
            return
        time.sleep(0.05)
    raise RuntimeError(f"keys did not become visible: keys={keys} states={latest}")


def run_basic_object_round_trip(store: Any) -> None:
    key = _key("bytes", 0)
    payload = b"mooncake-a5-basic-object-round-trip"
    result = int(store.put(key, payload))
    if result != 0:
        raise RuntimeError(f"basic put failed: {result}")
    _wait_exists(store, [key])
    restored = store.get(key)
    if restored != payload:
        raise AssertionError(f"basic object mismatch: expected={payload!r}, actual={restored!r}")
    print("[mooncake-probe] BASIC OBJECT ROUND TRIP PASSED")


def _register(engine: Any, tensor: Any) -> tuple[int, int]:
    pointer = int(tensor.data_ptr())
    size = int(tensor.numel() * tensor.element_size())
    result = int(engine.register_memory(pointer, size))
    if result != 0:
        raise RuntimeError(f"register_memory failed: ptr=0x{pointer:x} size={size} ret={result}")
    return pointer, size


def run_multi_buffer_round_trip(engine: Any, store: Any, device_id: int) -> None:
    import torch
    import torch_npu  # noqa: F401
    from mooncake.store import ReplicateConfig

    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    num_keys = 2
    num_layers = 4
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
        registered.append(_register(engine, source))
        registered.append(_register(engine, destination))
        keys = [_key("multi", index) for index in range(num_keys)]
        sizes = [[bytes_per_layer] * num_layers for _ in range(num_keys)]
        source_addresses = [
            [source[key_index, layer].data_ptr() for layer in range(num_layers)] for key_index in range(num_keys)
        ]
        destination_addresses = [
            [destination[key_index, layer].data_ptr() for layer in range(num_layers)] for key_index in range(num_keys)
        ]

        config = ReplicateConfig()
        config.prefer_alloc_in_same_node = True
        put_results = [
            int(value)
            for value in store.batch_put_from_multi_buffers(
                keys,
                source_addresses,
                sizes,
                config,
            )
        ]
        print(f"[mooncake-probe] batch_put results={put_results}")
        if len(put_results) != num_keys or any(value < 0 for value in put_results):
            raise RuntimeError(f"batch_put_from_multi_buffers failed: {put_results}")
        _wait_exists(store, keys)

        get_results = [
            int(value)
            for value in store.batch_get_into_multi_buffers(
                keys,
                destination_addresses,
                sizes,
            )
        ]
        print(f"[mooncake-probe] batch_get results={get_results}")
        if len(get_results) != num_keys or any(value < 0 for value in get_results):
            raise RuntimeError(f"batch_get_into_multi_buffers failed: {get_results}")
        torch.npu.synchronize()

        if not torch.equal(source.cpu(), destination.cpu()):
            mismatch = torch.nonzero(source.cpu() != destination.cpu())
            first = mismatch[0].tolist() if mismatch.numel() else None
            raise AssertionError(f"Mooncake multi-buffer mismatch: first={first}")
        print("[mooncake-probe] NPU MULTI-BUFFER ROUND TRIP PASSED")
    finally:
        for pointer, _ in reversed(registered):
            try:
                result = int(engine.unregister_memory(pointer))
                print(f"[mooncake-probe] unregister_memory ptr=0x{pointer:x} ret={result}")
            except Exception:
                traceback.print_exc()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("capabilities", "npu", "all"),
        default="capabilities",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--master-address", default="127.0.0.1:50088")
    parser.add_argument("--metrics-port", type=int, default=9003)
    parser.add_argument("--global-segment-bytes", type=int, default=1 << 30)
    parser.add_argument("--local-buffer-bytes", type=int, default=1 << 30)
    parser.add_argument("--lease-ttl-ms", type=int, default=300000)
    parser.add_argument("--client-ttl-seconds", type=int, default=120)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--external-master", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    # Mooncake's Ascend DSOs require CANN to have an owner with a balanced
    # process lifetime. This also matches a real vLLM worker, where torch_npu
    # initializes the device before the external KV backend is constructed.
    _prepare_npu(args.device_id)
    print_capabilities()
    if args.mode == "capabilities":
        return 0

    master_process: subprocess.Popen | None = None
    engine = None
    store = None
    try:
        master_process = start_master(args)
        engine, store = create_store(args)
        run_basic_object_round_trip(store)
        run_multi_buffer_round_trip(engine, store, args.device_id)
        print("[mooncake-probe] ALL REQUESTED TESTS PASSED")
        return 0
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                traceback.print_exc()
        del engine
        if master_process is not None:
            stop_master(master_process)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("[mooncake-probe] FAILED", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
