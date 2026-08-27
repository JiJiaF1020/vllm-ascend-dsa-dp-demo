#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""验证 Mooncake Store 自管 TCP engine 对 DSA swapped L2 的支持。"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
import uuid
from contextlib import suppress
from pathlib import Path
from threading import BrokenBarrierError
from typing import Any


def _parse_devices(raw: str) -> tuple[int, ...]:
    devices = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not devices:
        raise ValueError("at least one NPU device is required")
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate NPU devices: {devices}")
    return devices


def _free_ports(host: str, count: int) -> tuple[int, ...]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, 0))
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
        return tuple(ports)
    finally:
        for sock in sockets:
            sock.close()


def _as_int_list(result: Any, *, operation: str, count: int) -> list[int]:
    if result is None or isinstance(result, (str, bytes)):
        raise RuntimeError(f"Mooncake {operation} returned no per-key result")
    values = [int(value) for value in result]
    if len(values) != count:
        raise RuntimeError(
            f"Mooncake {operation} result count mismatch: "
            f"actual={len(values)}, expected={count}"
        )
    failed = [value for value in values if value < 0]
    if failed:
        raise RuntimeError(f"Mooncake {operation} failed: {values}")
    return values


def _fill_pattern(tensor: Any, seed: int) -> tuple[int, ...]:
    values = tuple((seed + index * 53) % 251 for index in range(4))
    width = int(tensor.numel())
    for index, value in enumerate(values):
        start = index * width // len(values)
        end = (index + 1) * width // len(values)
        tensor[start:end].fill_(value)
    return values


def _assert_pattern(tensor: Any, values: tuple[int, ...]) -> None:
    actual = tensor.cpu()
    width = int(actual.numel())
    for index, value in enumerate(values):
        start = index * width // len(values)
        end = (index + 1) * width // len(values)
        if not bool((actual[start:end] == value).all()):
            raise AssertionError(
                f"payload mismatch in quarter={index}: expected={value}"
            )


def _allocate_swapped_planes(
    *,
    torch: Any,
    torch_npu: Any,
    device_id: int,
    num_planes: int,
    plane_bytes: int,
    arena_blocks: int,
    block_index: int,
    seed: int,
) -> tuple[list[Any], list[Any], list[tuple[int, ...]]]:
    device = torch.device(f"npu:{device_id}")
    sources: list[Any] = []
    destinations: list[Any] = []
    patterns: list[tuple[int, ...]] = []
    for plane_index in range(num_planes):
        source = torch_npu.empty_with_swapped_memory(
            (arena_blocks, plane_bytes),
            dtype=torch.uint8,
            device=device,
        )
        destination = torch_npu.empty_with_swapped_memory(
            (arena_blocks, plane_bytes),
            dtype=torch.uint8,
            device=device,
        )
        patterns.append(
            _fill_pattern(
                source[block_index],
                seed + plane_index * 17,
            )
        )
        destination[block_index].fill_(255)
        sources.append(source)
        destinations.append(destination)
    torch.npu.synchronize()
    return sources, destinations, patterns


def _payload_columns(
    tensors: list[Any],
    *,
    block_index: int,
) -> tuple[list[int], list[int]]:
    addresses: list[int] = []
    sizes: list[int] = []
    for tensor in tensors:
        row_bytes = int(tensor.stride(0)) * tensor.element_size()
        addresses.append(int(tensor.data_ptr()) + block_index * row_bytes)
        sizes.append(int(tensor.shape[1]) * tensor.element_size())
    return addresses, sizes


def _register_arenas(store: Any, tensors: list[Any]) -> list[int]:
    registered: list[int] = []
    for tensor in tensors:
        pointer = int(tensor.data_ptr())
        size = int(tensor.numel()) * tensor.element_size()
        result = int(store.register_buffer(pointer, size))
        if result != 0:
            raise RuntimeError(
                f"Mooncake Store register_buffer failed: "
                f"ptr=0x{pointer:x}, bytes={size}, ret={result}"
            )
        registered.append(pointer)
    return registered


def _unregister_arenas(store: Any, pointers: list[int]) -> None:
    for pointer in reversed(pointers):
        result = int(store.unregister_buffer(pointer))
        if result != 0:
            raise RuntimeError(
                f"Mooncake Store unregister_buffer failed: "
                f"ptr=0x{pointer:x}, ret={result}"
            )


def _setup_store(
    *,
    local_endpoint: str,
    master_address: str,
    segment_bytes: int,
    local_buffer_bytes: int,
) -> Any:
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    result = int(
        store.setup(
            local_hostname=local_endpoint,
            metadata_server="P2PHANDSHAKE",
            global_segment_size=segment_bytes,
            local_buffer_size=local_buffer_bytes,
            protocol="tcp",
            rdma_devices="",
            master_server_addr=master_address,
            # Deliberately omit engine: Store must create and own its TCP TE.
        )
    )
    if result != 0:
        raise RuntimeError(
            f"Mooncake Store-owned TCP setup failed: "
            f"endpoint={local_endpoint}, ret={result}"
        )
    return store


def _wait_for_key(store: Any, key: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _as_int_list(
            store.batch_is_exist([key]),
            operation="exists",
            count=1,
        )
        if result[0] != 0:
            return
        time.sleep(0.05)
    raise TimeoutError(f"Mooncake key was not published within {timeout}s: {key}")


def _put(
    *,
    store: Any,
    key: str,
    addresses: list[int],
    sizes: list[int],
) -> list[int]:
    from mooncake.store import ReplicateConfig

    config = ReplicateConfig()
    config.preferred_segment = store.get_hostname()
    config.prefer_alloc_in_same_node = False
    return _as_int_list(
        store.batch_put_from_multi_buffers(
            [key],
            [addresses],
            [sizes],
            config,
        ),
        operation="put",
        count=1,
    )


def _get(
    *,
    store: Any,
    key: str,
    addresses: list[int],
    sizes: list[int],
) -> list[int]:
    result = _as_int_list(
        store.batch_get_into_multi_buffers(
            [key],
            [addresses],
            [sizes],
        ),
        operation="get",
        count=1,
    )
    expected_bytes = sum(sizes)
    if result[0] not in {0, expected_bytes}:
        raise RuntimeError(
            f"Mooncake get byte count mismatch: "
            f"actual={result[0]}, expected={expected_bytes}"
        )
    return result


def inspect_capabilities() -> None:
    import mooncake.store
    from mooncake.store import MooncakeDistributedStore

    print("[store-tcp] package inspection", flush=True)
    for distribution in (
        "mooncake-transfer-engine-npu",
        "mooncake-transfer-engine",
    ):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT_INSTALLED"
        print(f"  {distribution}={version}", flush=True)
    print(f"  store_module={Path(mooncake.store.__file__).resolve()}", flush=True)
    for name in (
        "setup",
        "register_buffer",
        "unregister_buffer",
        "batch_is_exist",
        "batch_put_from_multi_buffers",
        "batch_get_into_multi_buffers",
    ):
        print(
            f"  Store API {name}={hasattr(MooncakeDistributedStore, name)}",
            flush=True,
        )

    try:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
            MooncakeBackend,
        )
        from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import (
            GlobalTE,
        )

        backend_source = inspect.getsource(MooncakeBackend.__init__)
        te_source = inspect.getsource(GlobalTE.get_transfer_engine)
        rejects_tcp = 'protocol != "ascend"' in backend_source
        hardcodes_ascend = '"P2PHANDSHAKE", "ascend"' in te_source
        print(
            "  vLLM-Ascend rejects non-ascend protocol="
            f"{rejects_tcp}",
            flush=True,
        )
        print(
            "  vLLM-Ascend GlobalTE hardcodes ascend="
            f"{hardcodes_ascend}",
            flush=True,
        )
    except Exception as exc:
        print(f"  vLLM-Ascend inspection unavailable: {exc!r}", flush=True)


def _runtime_imports(device_id: int) -> tuple[Any, Any]:
    for key in (
        "ASCEND_GLOBAL_RESOURCE_CONFIG",
        "ASCEND_LOCAL_COMM_RES",
        "HCCL_INTRA_ROCE_ENABLE",
    ):
        os.environ.pop(key, None)
    import torch
    import torch_npu

    torch.npu.set_device(device_id)
    return torch, torch_npu


def run_single(args: argparse.Namespace) -> None:
    device_id = _parse_devices(args.devices)[0]
    torch, torch_npu = _runtime_imports(device_id)
    port = _free_ports(args.local_host, 1)[0]
    endpoint = f"{args.local_host}:{port}"
    store = None
    registered: list[int] = []
    try:
        store = _setup_store(
            local_endpoint=endpoint,
            master_address=args.master_address,
            segment_bytes=args.segment_bytes,
            local_buffer_bytes=args.local_buffer_bytes,
        )
        sources, destinations, patterns = _allocate_swapped_planes(
            torch=torch,
            torch_npu=torch_npu,
            device_id=device_id,
            num_planes=args.num_planes,
            plane_bytes=args.plane_bytes,
            arena_blocks=args.arena_blocks,
            block_index=args.block_index,
            seed=31,
        )
        registered = _register_arenas(store, sources + destinations)
        source_addrs, sizes = _payload_columns(
            sources,
            block_index=args.block_index,
        )
        destination_addrs, destination_sizes = _payload_columns(
            destinations,
            block_index=args.block_index,
        )
        key = f"dsa-store-tcp-single-{uuid.uuid4().hex}"
        put_result = _put(
            store=store,
            key=key,
            addresses=source_addrs,
            sizes=sizes,
        )
        _wait_for_key(store, key, args.timeout)
        get_result = _get(
            store=store,
            key=key,
            addresses=destination_addrs,
            sizes=destination_sizes,
        )
        for destination, pattern in zip(destinations, patterns, strict=True):
            _assert_pattern(destination[args.block_index], pattern)
        print(
            "[store-tcp] SINGLE PASSED "
            f"device={device_id} endpoint={endpoint} planes={args.num_planes} "
            f"bytes={sum(sizes)} put={put_result} get={get_result}",
            flush=True,
        )
    finally:
        if store is not None:
            if registered:
                _unregister_arenas(store, registered)
            store.close()


def _cross_worker(
    rank: int,
    world_size: int,
    device_id: int,
    port: int,
    args: argparse.Namespace,
    run_id: str,
    barrier: Any,
    result_queue: Any,
) -> None:
    store = None
    registered: list[int] = []
    result: dict[str, Any] = {"rank": rank, "device": device_id}
    try:
        torch, torch_npu = _runtime_imports(device_id)
        endpoint = f"{args.local_host}:{port}"
        store = _setup_store(
            local_endpoint=endpoint,
            master_address=args.master_address,
            segment_bytes=args.segment_bytes,
            local_buffer_bytes=args.local_buffer_bytes,
        )
        sources, destinations, _ = _allocate_swapped_planes(
            torch=torch,
            torch_npu=torch_npu,
            device_id=device_id,
            num_planes=args.num_planes,
            plane_bytes=args.plane_bytes,
            arena_blocks=args.arena_blocks,
            block_index=args.block_index,
            seed=97 + rank * 61,
        )
        registered = _register_arenas(store, sources + destinations)
        source_addrs, sizes = _payload_columns(
            sources,
            block_index=args.block_index,
        )
        destination_addrs, destination_sizes = _payload_columns(
            destinations,
            block_index=args.block_index,
        )
        barrier.wait(timeout=args.timeout)
        own_key = f"dsa-store-tcp-cross-{run_id}-rank{rank}"
        put_result = _put(
            store=store,
            key=own_key,
            addresses=source_addrs,
            sizes=sizes,
        )
        barrier.wait(timeout=args.timeout)
        peer_rank = (rank + 1) % world_size
        peer_key = f"dsa-store-tcp-cross-{run_id}-rank{peer_rank}"
        _wait_for_key(store, peer_key, args.timeout)
        get_result = _get(
            store=store,
            key=peer_key,
            addresses=destination_addrs,
            sizes=destination_sizes,
        )
        for plane_index, destination in enumerate(destinations):
            expected = tuple(
                (97 + peer_rank * 61 + plane_index * 17 + index * 53) % 251
                for index in range(4)
            )
            _assert_pattern(destination[args.block_index], expected)
        result.update(
            success=True,
            endpoint=endpoint,
            peer_rank=peer_rank,
            bytes=sum(sizes),
            put=put_result,
            get=get_result,
        )
    except (Exception, BrokenBarrierError) as exc:
        with suppress(Exception):
            barrier.abort()
        result.update(
            success=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    finally:
        try:
            if store is not None:
                if registered:
                    _unregister_arenas(store, registered)
                store.close()
        except Exception as exc:
            result.setdefault("teardown_error", f"{type(exc).__name__}: {exc}")
            result["success"] = False
        result_queue.put(result)


def run_cross(args: argparse.Namespace) -> None:
    devices = _parse_devices(args.devices)
    if len(devices) < 2:
        raise ValueError(
            f"cross mode requires at least two devices, got {devices}"
        )
    ports = _free_ports(args.local_host, len(devices))
    context = mp.get_context("spawn")
    barrier = context.Barrier(len(devices))
    result_queue = context.Queue()
    run_id = uuid.uuid4().hex
    processes = [
        context.Process(
            target=_cross_worker,
            args=(
                rank,
                len(devices),
                device_id,
                ports[rank],
                args,
                run_id,
                barrier,
                result_queue,
            ),
        )
        for rank, device_id in enumerate(devices)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=args.timeout * 3)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    results: list[dict[str, Any]] = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=2))
        except queue.Empty:
            break
    exitcodes = [process.exitcode for process in processes]
    for result in sorted(results, key=lambda item: int(item["rank"])):
        print(f"[store-tcp] cross result={result}", flush=True)
    failures = [result for result in results if not result.get("success")]
    if len(results) != len(processes) or failures or any(code != 0 for code in exitcodes):
        raise RuntimeError(
            f"Mooncake Store-owned TCP cross gate failed: "
            f"devices={devices}, exitcodes={exitcodes}, results={results}"
        )
    print(
        "[store-tcp] CROSS PASSED "
        f"devices={devices} planes={args.num_planes} "
        f"bytes={args.num_planes * args.plane_bytes}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("inspect", "single", "cross", "all"),
        default="inspect",
    )
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--master-address", default="127.0.0.1:50088")
    parser.add_argument("--segment-bytes", type=int, default=134217728)
    parser.add_argument("--local-buffer-bytes", type=int, default=67108864)
    parser.add_argument("--num-planes", type=int, default=4)
    parser.add_argument("--plane-bytes", type=int, default=262144)
    parser.add_argument("--arena-blocks", type=int, default=2)
    parser.add_argument("--block-index", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    if args.num_planes <= 0 or args.plane_bytes <= 0:
        parser.error("num-planes and plane-bytes must be positive")
    if args.arena_blocks <= 0 or not 0 <= args.block_index < args.arena_blocks:
        parser.error("block-index must be inside arena-blocks")
    if args.segment_bytes <= 0 or args.local_buffer_bytes < 0:
        parser.error("segment sizes are invalid")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.mode in {"inspect", "all"}:
        inspect_capabilities()
    if args.mode in {"single", "all"}:
        run_single(args)
    if args.mode in {"cross", "all"}:
        run_cross(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
