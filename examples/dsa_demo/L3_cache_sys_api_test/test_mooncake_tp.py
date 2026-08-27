#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise Mooncake cross-segment NPU IO across A5 TP worker processes."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import queue
import socket
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

_DSA_A5_GLM52_INDEXER_LAYERS = 21
_DSA_A5_GLM52_MLA_LAYERS = 78
_DSA_A5_INDEXER_CACHE_BYTES = 128 * 128
_DSA_A5_INDEXER_SCALE_BYTES = 128 * 4
_DSA_A5_MLA_PACKED_BYTES = 128 * 656


def _build_payload_layout(
    layout: str,
    *,
    payload_bytes: int,
    num_buffers: int,
) -> tuple[tuple[int, ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    """Build per-address sizes and per-key descriptor groups."""

    if layout == "uniform":
        if num_buffers <= 0:
            raise ValueError("num_buffers must be positive")
        base_size, remainder = divmod(payload_bytes, num_buffers)
        if base_size <= 0:
            raise ValueError(
                f"payload_bytes={payload_bytes} is too small for "
                f"num_buffers={num_buffers}"
            )
        sizes = tuple(
            base_size + (1 if index < remainder else 0)
            for index in range(num_buffers)
        )
        return sizes, (("payload", tuple(range(num_buffers))),)

    if layout == "dsa_a5_glm52":
        indexer_sizes = (
            _DSA_A5_INDEXER_CACHE_BYTES,
            _DSA_A5_INDEXER_SCALE_BYTES,
        ) * _DSA_A5_GLM52_INDEXER_LAYERS
        mla_sizes = (_DSA_A5_MLA_PACKED_BYTES,) * _DSA_A5_GLM52_MLA_LAYERS
        sizes = indexer_sizes + mla_sizes
        indexer_end = len(indexer_sizes)
        return sizes, (
            ("indexer", tuple(range(indexer_end))),
            ("mla", tuple(range(indexer_end, len(sizes)))),
        )

    raise ValueError(f"unsupported payload layout: {layout!r}")


def _parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError(f"invalid visible device list: {value!r}")
    return devices


def _resolve_io_roles(
    *,
    rank: int,
    world_size: int,
    producer_rank: int | None,
    consumer_rank: int | None,
) -> tuple[bool, bool, int]:
    """Return producer/consumer flags and the key's source rank."""

    if producer_rank is None:
        return True, True, (rank + 1) % world_size
    produces = rank == producer_rank
    consumes = produces if consumer_rank is None else rank == consumer_rank
    return produces, consumes, producer_rank


def _resolve_local_host(value: str) -> str:
    if value != "auto":
        return value
    hostname = socket.gethostname()
    try:
        addresses = socket.getaddrinfo(
            hostname,
            None,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        print(
            f"[mooncake-tp] hostname {hostname!r} is not resolvable "
            f"({exc}); using 127.0.0.1 for this single-node probe",
            flush=True,
        )
        return "127.0.0.1"
    candidates = {item[4][0] for item in addresses}
    candidates.discard("127.0.0.1")
    if not candidates:
        print(
            "[mooncake-tp] no non-loopback IPv4 address was discovered; "
            "using 127.0.0.1 for this single-node probe",
            flush=True,
        )
        return "127.0.0.1"
    return sorted(candidates)[0]


def _require_master_service(address: str) -> None:
    host, separator, raw_port = address.rpartition(":")
    if not separator or not host:
        raise ValueError(
            f"invalid Mooncake master address {address!r}; expected host:port"
        )
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(
            f"invalid Mooncake master port in {address!r}"
        ) from exc
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Mooncake Master is not reachable at {address}; start "
            "baseline_e2e/start_mooncake_master.sh in another terminal"
        ) from exc


def _wait_exists(
    store: Any,
    keys: list[str],
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    latest: list[int] = []
    while time.monotonic() < deadline:
        latest = [int(value) for value in store.batch_is_exist(keys)]
        if latest == [1] * len(keys):
            return
        time.sleep(0.05)
    raise RuntimeError(
        f"keys did not become visible: keys={keys} states={latest}"
    )


def _worker(
    rank: int,
    devices: list[int],
    visible_devices: list[int],
    comm_mode: str,
    max_active_channels: int,
    config_path: str,
    local_host: str,
    master_address: str,
    run_id: str,
    segment_bytes: int,
    local_buffer_bytes: int,
    source_buffer_mode: str,
    destination_buffer_mode: str,
    payload_layout: str,
    payload_buffer_sizes: tuple[int, ...],
    key_groups: tuple[tuple[str, tuple[int, ...]], ...],
    arena_blocks: int,
    block_index: int,
    io_thread: str,
    placement_mode: str,
    force_remote_put: bool,
    producer_rank: int | None,
    consumer_rank: int | None,
    target_rank: int | None,
    segments: Any,
    barrier: Any,
    result_queue: Any,
) -> None:
    device_id = devices[rank]
    stage = "configure_environment"
    engine = None
    store = None
    registered: list[tuple[int, int]] = []
    io_executor = None
    try:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
            str(device) for device in visible_devices
        )
        if comm_mode == "hixlep":
            endpoint_file = Path(config_path) / f"ub_endpoint_npu_{device_id}.json"
            endpoint_config = json.loads(endpoint_file.read_text(encoding="utf-8"))
            os.environ["ASCEND_LOCAL_COMM_RES"] = json.dumps(
                endpoint_config,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        elif comm_mode in {
            "global_ub",
            "global_ub_device",
            "global_ub_host",
        }:
            os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
            protocol_desc = []
            if comm_mode != "global_ub_host":
                protocol_desc.append("ub_ctp:device")
            if comm_mode != "global_ub_device":
                protocol_desc.append("ub_ctp:host")
            os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] = json.dumps(
                {
                    "comm_resource_config.protocol_desc": protocol_desc,
                    "comm_resource_config.max_active_channels": (
                        max_active_channels
                    ),
                },
                separators=(",", ":"),
            )
        elif comm_mode in {"auto", "tcp"}:
            os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
            os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        else:
            raise ValueError(f"unsupported communication mode: {comm_mode!r}")
        os.environ.setdefault("ASCEND_CONNECT_TIMEOUT", "30000")
        os.environ.setdefault("ASCEND_TRANSFER_TIMEOUT", "30000")

        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(device_id)

        from mooncake.engine import TransferEngine
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        transport_protocol = "tcp" if comm_mode == "tcp" else "ascend"
        stage = "transfer_engine_initialize"
        engine = TransferEngine()
        ret = int(
            engine.initialize(
                local_host,
                "P2PHANDSHAKE",
                transport_protocol,
                "",
            )
        )
        if ret != 0:
            raise RuntimeError(f"rank={rank} TransferEngine.initialize ret={ret}")
        local_segment = f"{local_host}:{int(engine.get_rpc_port())}"

        stage = "store_setup"
        store = MooncakeDistributedStore()
        ret = int(
            store.setup(
                local_hostname=local_segment,
                metadata_server="P2PHANDSHAKE",
                global_segment_size=segment_bytes,
                local_buffer_size=local_buffer_bytes,
                protocol=transport_protocol,
                rdma_devices="",
                master_server_addr=master_address,
                engine=engine.get_engine(),
            )
        )
        if ret != 0:
            raise RuntimeError(f"rank={rank} Mooncake store setup ret={ret}")

        stage = "register_buffers"
        device = torch.device(f"npu:{device_id}")
        produces, consumes, source_rank = _resolve_io_roles(
            rank=rank,
            world_size=len(devices),
            producer_rank=producer_rank,
            consumer_rank=consumer_rank,
        )
        participates_io = produces or consumes
        source_buffers = []
        destination_buffers = []
        source_payload_ptrs = []
        destination_payload_ptrs = []
        for index, size in enumerate(payload_buffer_sizes):
            expected = (rank * 17 + index) % 251
            if produces:
                if source_buffer_mode == "device":
                    source = torch.full(
                        (size,),
                        expected,
                        dtype=torch.uint8,
                        device=device,
                    )
                    source_pointer = int(source.data_ptr())
                elif source_buffer_mode == "swapped":
                    source = torch_npu.empty_with_swapped_memory(
                        (arena_blocks, size),
                        dtype=torch.uint8,
                        device=device,
                    )
                    source[block_index].fill_(expected)
                    source_pointer = (
                        int(source.data_ptr()) + block_index * size
                    )
                elif source_buffer_mode == "pinned":
                    source = torch.empty(
                        (arena_blocks, size),
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=True,
                    )
                    source[block_index].fill_(expected)
                    source_pointer = (
                        int(source.data_ptr()) + block_index * size
                    )
                else:
                    raise ValueError(
                        "unsupported source buffer mode: "
                        f"{source_buffer_mode!r}"
                    )
                source_buffers.append(source)
                source_payload_ptrs.append(source_pointer)
            if consumes:
                if destination_buffer_mode == "device":
                    destination = torch.zeros(
                        (size,),
                        dtype=torch.uint8,
                        device=device,
                    )
                    destination_pointer = int(destination.data_ptr())
                elif destination_buffer_mode == "swapped":
                    destination = torch_npu.empty_with_swapped_memory(
                        (1, size),
                        dtype=torch.uint8,
                        device=device,
                    )
                    destination[0].zero_()
                    destination_pointer = int(destination.data_ptr())
                elif destination_buffer_mode == "pinned":
                    destination = torch.zeros(
                        (1, size),
                        dtype=torch.uint8,
                        device="cpu",
                        pin_memory=True,
                    )
                    destination_pointer = int(destination.data_ptr())
                else:
                    raise ValueError(
                        "unsupported destination buffer mode: "
                        f"{destination_buffer_mode!r}"
                    )
                destination_buffers.append(destination)
                destination_payload_ptrs.append(destination_pointer)
        if produces and len(source_payload_ptrs) != len(payload_buffer_sizes):
            raise RuntimeError(
                "Mooncake probe did not allocate every source plane: "
                f"expected={len(payload_buffer_sizes)}, "
                f"actual={len(source_payload_ptrs)}"
            )
        if consumes and len(destination_payload_ptrs) != len(
            payload_buffer_sizes
        ):
            raise RuntimeError(
                "Mooncake probe did not allocate every destination plane: "
                f"expected={len(payload_buffer_sizes)}, "
                f"actual={len(destination_payload_ptrs)}"
            )
        torch.npu.synchronize()
        for tensor in (*source_buffers, *destination_buffers):
            pointer = int(tensor.data_ptr())
            size = int(tensor.numel() * tensor.element_size())
            ret = int(engine.register_memory(pointer, size))
            if ret != 0:
                raise RuntimeError(
                    f"rank={rank} register_memory ptr=0x{pointer:x} ret={ret}"
                )
            registered.append((pointer, size))

        segments[rank] = local_segment
        stage = "wait_for_peer_segments"
        barrier.wait(timeout=60.0)

        if participates_io and io_thread == "background":
            io_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mooncake-probe",
            )

        def run_io(callable_):
            if io_executor is None:
                return callable_()

            def thread_entry():
                torch.npu.set_device(device_id)
                return callable_()

            return io_executor.submit(thread_entry).result()

        own_keys = [
            f"mooncake-a5-tp-{run_id}-rank-{rank}-{suffix}"
            for suffix, _ in key_groups
        ]
        if produces:
            source_addresses_by_key = [
                [source_payload_ptrs[index] for index in indices]
                for _, indices in key_groups
            ]
        else:
            source_addresses_by_key = []
        if consumes:
            destination_addresses_by_key = [
                [destination_payload_ptrs[index] for index in indices]
                for _, indices in key_groups
            ]
        else:
            destination_addresses_by_key = []
        sizes_by_key = [
            [payload_buffer_sizes[index] for index in indices]
            for _, indices in key_groups
        ]
        should_put = produces
        if should_put:
            stage = "put"
            replicate_config = ReplicateConfig()
            replicate_config.prefer_alloc_in_same_node = True
            selected_target_rank: int | None = None
            if placement_mode == "preferred":
                selected_target_rank = (
                    target_rank
                    if target_rank is not None
                    else (
                        (rank + 1) % len(devices)
                        if force_remote_put
                        else rank
                    )
                )
                replicate_config.preferred_segment = str(
                    segments[selected_target_rank]
                )
            exists_before_put = run_io(
                lambda: [
                    int(value)
                    for value in store.batch_is_exist(own_keys)
                ]
            )
            if exists_before_put != [0] * len(own_keys):
                raise RuntimeError(
                    f"rank={rank} keys unexpectedly existed before put: "
                    f"{exists_before_put}"
                )
            put_result = run_io(
                lambda: [
                    int(value)
                    for value in store.batch_put_from_multi_buffers(
                        own_keys,
                        source_addresses_by_key,
                        sizes_by_key,
                        replicate_config,
                    )
                ]
            )
            if len(put_result) != len(own_keys) or any(
                value < 0 for value in put_result
            ):
                raise RuntimeError(
                    f"rank={rank} put placement={placement_mode} "
                    f"target_rank={selected_target_rank} failed: "
                    f"{put_result}"
                )
            _wait_exists(store, own_keys, timeout_seconds=30.0)
            print(
                f"[mooncake-tp] rank={rank} device={device_id} "
                f"local_segment={local_segment} "
                f"placement={placement_mode} "
                f"target_rank={selected_target_rank} "
                f"target_device="
                f"{devices[selected_target_rank] if selected_target_rank is not None else 'master'} "
                f"exists_before_put={exists_before_put} "
                f"put={put_result} source_buffer_mode={source_buffer_mode} "
                f"payload_layout={payload_layout} keys={len(own_keys)} "
                f"key_bytes={[sum(parts) for parts in sizes_by_key]} "
                f"num_buffers={len(payload_buffer_sizes)} "
                f"arena_blocks={arena_blocks} "
                f"block_index={block_index} io_thread={io_thread}",
                flush=True,
            )

        stage = "wait_for_peer_puts"
        barrier.wait(timeout=60.0)
        should_get = consumes
        if should_get:
            remote_keys = [
                f"mooncake-a5-tp-{run_id}-rank-{source_rank}-{suffix}"
                for suffix, _ in key_groups
            ]
            stage = "remote_get"
            get_result = run_io(
                lambda: [
                    int(value)
                    for value in store.batch_get_into_multi_buffers(
                        remote_keys,
                        destination_addresses_by_key,
                        sizes_by_key,
                    )
                ]
            )
            if len(get_result) != len(remote_keys) or any(
                value < 0 for value in get_result
            ):
                raise RuntimeError(
                    f"rank={rank} remote get from rank={source_rank} "
                    f"failed: {get_result}"
                )
            torch.npu.synchronize()
            for index, destination in enumerate(destination_buffers):
                expected = (source_rank * 17 + index) % 251
                actual = destination
                if destination_buffer_mode in {"swapped", "pinned"}:
                    destination = destination[0]
                    actual = destination
                if destination_buffer_mode == "swapped":
                    actual = torch.empty(
                        tuple(destination.shape),
                        dtype=destination.dtype,
                        device=device,
                    )
                    actual.fill_(1).mul_(destination)
                if not bool(torch.all(actual == expected).item()):
                    raise AssertionError(
                        f"rank={rank} remote payload mismatch from "
                        f"rank={source_rank} buffer={index}"
                    )
            print(
                f"[mooncake-tp] rank={rank} "
                f"remote_source_rank={source_rank} "
                f"remote_get={get_result} "
                f"payload_bytes={sum(payload_buffer_sizes)} "
                f"source_buffer_mode={source_buffer_mode} "
                f"destination_buffer_mode={destination_buffer_mode} "
                f"payload_layout={payload_layout} "
                f"keys={len(remote_keys)} "
                f"num_buffers={len(payload_buffer_sizes)} "
                f"arena_blocks={arena_blocks} block_index={block_index} "
                f"io_thread={io_thread}",
                flush=True,
            )
        stage = "wait_for_peer_gets"
        barrier.wait(timeout=60.0)
        result_queue.put((rank, True, "passed"))
    except Exception:
        with contextlib.suppress(Exception):
            barrier.abort()
        result_queue.put(
            (
                rank,
                False,
                f"stage={stage} device={device_id}\n{traceback.format_exc()}",
            )
        )
    finally:
        if io_executor is not None:
            io_executor.shutdown(wait=True, cancel_futures=False)
        if engine is not None:
            for pointer, _ in reversed(registered):
                try:
                    engine.unregister_memory(pointer)
                except Exception:
                    traceback.print_exc()
        if store is not None:
            try:
                store.close()
            except Exception:
                traceback.print_exc()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=Path("/etc/hixlep"))
    parser.add_argument("--visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--active-devices",
        help="Subset of visible devices that should create TP probe workers.",
    )
    parser.add_argument(
        "--comm-mode",
        choices=(
            "global_ub",
            "global_ub_device",
            "global_ub_host",
            "hixlep",
            "auto",
            "tcp",
        ),
        default="global_ub_device",
    )
    parser.add_argument("--max-active-channels", type=int, default=8)
    parser.add_argument("--local-host", default="auto")
    parser.add_argument("--master-address", default="127.0.0.1:50088")
    parser.add_argument("--payload-bytes", type=int, default=1 << 20)
    parser.add_argument("--segment-bytes", type=int, default=1 << 30)
    parser.add_argument("--local-buffer-bytes", type=int, default=1 << 30)
    parser.add_argument(
        "--buffer-mode",
        choices=("device", "swapped", "pinned"),
        default="device",
        help="Default for source and destination unless overridden.",
    )
    parser.add_argument(
        "--source-buffer-mode",
        choices=("device", "swapped", "pinned"),
    )
    parser.add_argument(
        "--destination-buffer-mode",
        choices=("device", "swapped", "pinned"),
    )
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument(
        "--payload-layout",
        choices=("uniform", "dsa_a5_glm52"),
        default="uniform",
        help=(
            "uniform splits payload-bytes evenly; dsa_a5_glm52 reproduces "
            "the real 42-plane Indexer plus 78-plane MLA two-key payload"
        ),
    )
    parser.add_argument("--arena-blocks", type=int, default=1)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument(
        "--io-thread",
        choices=("main", "background"),
        default="main",
    )
    parser.add_argument(
        "--placement-mode",
        choices=("preferred", "automatic"),
        default="preferred",
        help=(
            "preferred pins all keys to one selected segment; automatic "
            "matches MooncakeBackend when preferred_segment=false"
        ),
    )
    parser.add_argument("--force-remote-put", action="store_true")
    parser.add_argument(
        "--producer-rank",
        type=int,
        help="When set, only this active-rank index performs put.",
    )
    parser.add_argument(
        "--consumer-device",
        type=int,
        help=(
            "Physical NPU that performs get. By default the producer also "
            "performs get."
        ),
    )
    parser.add_argument(
        "--target-device",
        type=int,
        help=(
            "Physical NPU that receives preferred placement. The device "
            "must be present in active-devices."
        ),
    )
    parser.add_argument("--process-timeout", type=float, default=120.0)
    args = parser.parse_args()

    if args.max_active_channels <= 0:
        raise ValueError("max_active_channels must be positive")
    if args.segment_bytes <= 0 or args.local_buffer_bytes <= 0:
        raise ValueError("segment and local buffer sizes must be positive")
    if args.arena_blocks <= 0:
        raise ValueError("arena_blocks must be positive")
    if not 0 <= args.block_index < args.arena_blocks:
        raise ValueError("block_index must be inside the source arena")

    visible_devices = _parse_devices(args.visible_devices)
    devices = _parse_devices(args.active_devices or args.visible_devices)
    unavailable_devices = sorted(set(devices) - set(visible_devices))
    if unavailable_devices:
        raise ValueError(
            f"active devices are not visible: {unavailable_devices}; "
            f"visible={visible_devices}"
        )
    if len(devices) < 2:
        raise ValueError(
            "cross-segment TP probe requires at least two visible devices; "
            f"got {devices}"
        )
    if args.producer_rank is not None and not (
        0 <= args.producer_rank < len(devices)
    ):
        raise ValueError(
            f"producer_rank={args.producer_rank} is outside "
            f"[0, {len(devices)})"
        )
    consumer_rank: int | None = None
    if args.consumer_device is not None:
        if args.producer_rank is None:
            raise ValueError(
                "consumer_device requires an explicit producer_rank"
            )
        try:
            consumer_rank = devices.index(args.consumer_device)
        except ValueError as exc:
            raise ValueError(
                f"consumer_device={args.consumer_device} is not active: "
                f"active_devices={devices}"
            ) from exc
    if args.placement_mode == "automatic" and (
        args.force_remote_put or args.target_device is not None
    ):
        raise ValueError(
            "automatic placement cannot be combined with "
            "force_remote_put or target_device"
        )
    target_rank: int | None = None
    if args.target_device is not None:
        try:
            target_rank = devices.index(args.target_device)
        except ValueError as exc:
            raise ValueError(
                f"target_device={args.target_device} is not active: "
                f"active_devices={devices}"
            ) from exc
    buffer_sizes, key_groups = _build_payload_layout(
        args.payload_layout,
        payload_bytes=args.payload_bytes,
        num_buffers=args.num_buffers,
    )
    _require_master_service(args.master_address)
    local_host = _resolve_local_host(args.local_host)
    source_buffer_mode = args.source_buffer_mode or args.buffer_mode
    destination_buffer_mode = (
        args.destination_buffer_mode or args.buffer_mode
    )
    run_id = uuid.uuid4().hex
    print(
        f"[mooncake-tp] visible_devices={visible_devices} "
        f"active_devices={devices} comm_mode={args.comm_mode} "
        f"max_active_channels={args.max_active_channels} "
        f"source_buffer_mode={source_buffer_mode} "
        f"destination_buffer_mode={destination_buffer_mode} "
        f"payload_layout={args.payload_layout} "
        f"keys={len(key_groups)} "
        f"num_buffers={len(buffer_sizes)} "
        f"payload_bytes={sum(buffer_sizes)} "
        f"segment_bytes={args.segment_bytes} "
        f"local_buffer_bytes={args.local_buffer_bytes} "
        f"arena_blocks={args.arena_blocks} "
        f"block_index={args.block_index} "
        f"io_thread={args.io_thread} "
        f"placement_mode={args.placement_mode} "
        f"force_remote_put={args.force_remote_put} "
        f"producer_rank={args.producer_rank} "
        f"consumer_device={args.consumer_device} "
        f"target_device={args.target_device} "
        f"local_host={local_host} "
        f"master={args.master_address} config_path={args.config_path}",
        flush=True,
    )

    context = mp.get_context("spawn")
    manager = context.Manager()
    segments = manager.list([""] * len(devices))
    barrier = context.Barrier(len(devices))
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_worker,
            args=(
                rank,
                devices,
                visible_devices,
                args.comm_mode,
                args.max_active_channels,
                str(args.config_path),
                local_host,
                args.master_address,
                run_id,
                args.segment_bytes,
                args.local_buffer_bytes,
                source_buffer_mode,
                destination_buffer_mode,
                args.payload_layout,
                buffer_sizes,
                key_groups,
                args.arena_blocks,
                args.block_index,
                args.io_thread,
                args.placement_mode,
                args.force_remote_put,
                args.producer_rank,
                consumer_rank,
                target_rank,
                segments,
                barrier,
                result_queue,
            ),
            name=f"mooncake-tp-{rank}",
        )
        for rank in range(len(devices))
    ]
    for process in processes:
        process.start()

    results: dict[int, tuple[bool, str]] = {}
    deadline = time.monotonic() + args.process_timeout
    while len(results) < len(processes) and time.monotonic() < deadline:
        try:
            rank, passed, detail = result_queue.get(timeout=0.5)
            results[int(rank)] = (bool(passed), str(detail))
        except queue.Empty:
            if all(not process.is_alive() for process in processes):
                break

    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    failures: list[str] = []
    for rank, process in enumerate(processes):
        result = results.get(rank)
        if result is None:
            failures.append(
                f"rank={rank} produced no result exitcode={process.exitcode}"
            )
        elif not result[0]:
            failures.append(f"rank={rank} failed:\n{result[1]}")
        elif process.exitcode not in (0, None):
            failures.append(
                f"rank={rank} passed data check but exited {process.exitcode}"
            )

    if failures:
        for failure in failures:
            print(f"[mooncake-tp] ERROR {failure}", flush=True)
        print("[mooncake-tp] FAILED", flush=True)
        manager.shutdown()
        return 1
    print("[mooncake-tp] CROSS-SEGMENT ROUND TRIP PASSED", flush=True)
    manager.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
