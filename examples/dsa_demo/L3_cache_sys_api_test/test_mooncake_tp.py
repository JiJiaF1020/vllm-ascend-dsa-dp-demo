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
from pathlib import Path
from typing import Any


def _parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError(f"invalid visible device list: {value!r}")
    return devices


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


def _wait_exists(store: Any, key: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    latest: list[int] = []
    while time.monotonic() < deadline:
        latest = [int(value) for value in store.batch_is_exist([key])]
        if latest == [1]:
            return
        time.sleep(0.05)
    raise RuntimeError(f"key did not become visible: key={key} states={latest}")


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
    payload_bytes: int,
    segment_bytes: int,
    barrier: Any,
    result_queue: Any,
) -> None:
    device_id = devices[rank]
    stage = "configure_environment"
    engine = None
    store = None
    registered: list[tuple[int, int]] = []
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
        elif comm_mode in {"global_ub", "global_ub_device"}:
            os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
            protocol_desc = ["ub_ctp:device"]
            if comm_mode == "global_ub":
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
        elif comm_mode == "auto":
            os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
            os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        else:
            raise ValueError(f"unsupported communication mode: {comm_mode!r}")

        import torch
        import torch_npu  # noqa: F401

        torch.npu.set_device(device_id)

        from mooncake.engine import TransferEngine
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        stage = "transfer_engine_initialize"
        engine = TransferEngine()
        ret = int(
            engine.initialize(
                local_host,
                "P2PHANDSHAKE",
                "ascend",
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
                local_buffer_size=segment_bytes,
                protocol="ascend",
                rdma_devices="",
                master_server_addr=master_address,
                engine=engine.get_engine(),
            )
        )
        if ret != 0:
            raise RuntimeError(f"rank={rank} Mooncake store setup ret={ret}")

        stage = "register_npu_buffers"
        device = torch.device(f"npu:{device_id}")
        source = torch.full(
            (payload_bytes,),
            rank % 251,
            dtype=torch.uint8,
            device=device,
        )
        destination = torch.zeros_like(source)
        torch.npu.synchronize()
        for tensor in (source, destination):
            pointer = int(tensor.data_ptr())
            size = int(tensor.numel() * tensor.element_size())
            ret = int(engine.register_memory(pointer, size))
            if ret != 0:
                raise RuntimeError(
                    f"rank={rank} register_memory ptr=0x{pointer:x} ret={ret}"
                )
            registered.append((pointer, size))

        own_key = f"mooncake-a5-tp-{run_id}-rank-{rank}"
        stage = "local_put"
        replicate_config = ReplicateConfig()
        replicate_config.preferred_segment = local_segment
        replicate_config.prefer_alloc_in_same_node = True
        put_result = [
            int(value)
            for value in store.batch_put_from_multi_buffers(
                [own_key],
                [[int(source.data_ptr())]],
                [[payload_bytes]],
                replicate_config,
            )
        ]
        if len(put_result) != 1 or put_result[0] < 0:
            raise RuntimeError(f"rank={rank} local put failed: {put_result}")
        _wait_exists(store, own_key, timeout_seconds=30.0)
        print(
            f"[mooncake-tp] rank={rank} device={device_id} "
            f"local_segment={local_segment} local_put={put_result}",
            flush=True,
        )

        stage = "wait_for_peer_puts"
        barrier.wait(timeout=60.0)
        source_rank = (rank + 1) % len(devices)
        remote_key = f"mooncake-a5-tp-{run_id}-rank-{source_rank}"
        stage = "remote_get"
        get_result = [
            int(value)
            for value in store.batch_get_into_multi_buffers(
                [remote_key],
                [[int(destination.data_ptr())]],
                [[payload_bytes]],
            )
        ]
        if len(get_result) != 1 or get_result[0] < 0:
            raise RuntimeError(
                f"rank={rank} remote get from rank={source_rank} failed: "
                f"{get_result}"
            )
        torch.npu.synchronize()
        expected = source_rank % 251
        if not bool(torch.all(destination == expected).item()):
            raise AssertionError(
                f"rank={rank} remote payload mismatch from rank={source_rank}"
            )
        print(
            f"[mooncake-tp] rank={rank} remote_source_rank={source_rank} "
            f"remote_get={get_result} payload_bytes={payload_bytes}",
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
        choices=("global_ub", "global_ub_device", "hixlep", "auto"),
        default="global_ub_device",
    )
    parser.add_argument("--max-active-channels", type=int, default=8)
    parser.add_argument("--local-host", default="auto")
    parser.add_argument("--master-address", default="127.0.0.1:50088")
    parser.add_argument("--payload-bytes", type=int, default=1 << 20)
    parser.add_argument("--segment-bytes", type=int, default=1 << 30)
    parser.add_argument("--process-timeout", type=float, default=120.0)
    args = parser.parse_args()

    if args.max_active_channels <= 0:
        raise ValueError("max_active_channels must be positive")

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
    _require_master_service(args.master_address)
    local_host = _resolve_local_host(args.local_host)
    run_id = uuid.uuid4().hex
    print(
        f"[mooncake-tp] visible_devices={visible_devices} "
        f"active_devices={devices} comm_mode={args.comm_mode} "
        f"max_active_channels={args.max_active_channels} "
        f"local_host={local_host} "
        f"master={args.master_address} config_path={args.config_path}",
        flush=True,
    )

    context = mp.get_context("spawn")
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
                args.payload_bytes,
                args.segment_bytes,
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
        return 1
    print("[mooncake-tp] CROSS-SEGMENT ROUND TRIP PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
