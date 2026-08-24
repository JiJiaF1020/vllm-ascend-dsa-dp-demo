#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise MemCache cross-rank A5 device_urma IO without loading a model."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any


def _parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError(f"invalid device list: {value!r}")
    return devices


def _load_local_eid_map(
    raw_map: str | None,
    map_file: Path | None,
) -> dict[int, str]:
    if raw_map and map_file is not None:
        raise ValueError(
            "use only one of --local-eid-map and --local-eid-map-file"
        )
    if map_file is not None:
        path = map_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        raw_map = path.read_text(encoding="utf-8")
    if not raw_map:
        return {}

    payload = json.loads(raw_map)
    if not isinstance(payload, dict):
        raise ValueError("local EID map must be a JSON object")

    result: dict[int, str] = {}
    for raw_device, raw_eid in payload.items():
        try:
            device_id = int(raw_device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid EID map device: {raw_device!r}") from exc
        if isinstance(raw_eid, str):
            eid = raw_eid.strip().lower()
        else:
            raise ValueError(
                f"EID for device {device_id} must be a string, got {raw_eid!r}"
            )
        try:
            eid_bytes = bytes.fromhex(eid)
        except ValueError as exc:
            raise ValueError(
                f"EID for device {device_id} is not hexadecimal: {raw_eid!r}"
            ) from exc
        if len(eid_bytes) != 16 or len(eid) != 32:
            raise ValueError(
                f"EID for device {device_id} must contain exactly 32 hex digits"
            )
        if device_id in result:
            raise ValueError(f"duplicate EID map device: {device_id}")
        result[device_id] = eid
    return result


def _service_ready(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _send(connection: Any, message: dict[str, Any]) -> None:
    with contextlib.suppress(BrokenPipeError, EOFError, OSError):
        connection.send(message)


def _worker(
    connection: Any,
    rank: int,
    active_devices: list[int],
    visible_devices: list[int],
    config_path: str,
    run_id: str,
    payload_bytes: int,
    local_eid: str | None,
    barrier: Any,
) -> None:
    device_id = active_devices[rank]
    backend = None
    registered: list[tuple[int, int]] = []
    stage = "configure_environment"
    result: dict[str, Any] = {
        "rank": rank,
        "device": device_id,
        "ok": False,
    }
    try:
        visible_devices_value = ",".join(str(device) for device in visible_devices)
        os.environ["ASCEND_VISIBLE_DEVICES"] = visible_devices_value
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices_value
        os.environ["MMC_LOCAL_CONFIG_PATH"] = config_path
        os.environ["VLLM_HOST_IP"] = "127.0.0.1"
        if local_eid is None:
            os.environ.pop("USE_LOCAL_EID", None)
        else:
            os.environ["USE_LOCAL_EID"] = local_eid

        import torch
        import torch_npu  # noqa: F401
        from vllm.config import ParallelConfig

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
            MemcacheBackend,
        )

        torch.npu.set_device(device_id)
        torch.npu.synchronize()

        stage = "backend_initialize"
        print(
            f"[memcache-tp] rank={rank} device={device_id} "
            f"eid_source={'override' if local_eid else 'rootinfo'} "
            f"eid={local_eid or '-'}",
            flush=True,
        )
        backend = MemcacheBackend(
            ParallelConfig(),
            local_rank=device_id,
            init_bm=True,
        )

        stage = "register_npu_buffers"
        device = torch.device(f"npu:{device_id}")
        source = torch.full(
            (payload_bytes,),
            rank + 1,
            dtype=torch.uint8,
            device=device,
        )
        destination = torch.zeros_like(source)
        torch.npu.synchronize()
        for tensor in (source, destination):
            registered.append(
                (int(tensor.data_ptr()), int(tensor.numel() * tensor.element_size()))
            )
        backend.register_buffer(
            [pointer for pointer, _ in registered],
            [size for _, size in registered],
        )

        own_key = f"memcache-a5-tp-{run_id}-rank-{rank}"
        stage = "local_put"
        backend.put(
            [own_key],
            [[int(source.data_ptr())]],
            [[payload_bytes]],
        )
        deadline = time.monotonic() + 30.0
        states: list[int] = []
        while time.monotonic() < deadline:
            states = [int(value) for value in backend.exists([own_key])]
            if states == [1]:
                break
            time.sleep(0.05)
        if states != [1]:
            raise RuntimeError(f"local key did not become visible: {states}")
        print(
            f"[memcache-tp] rank={rank} device={device_id} local_put=passed",
            flush=True,
        )

        stage = "wait_for_peer_puts"
        barrier.wait(timeout=60.0)
        peer_rank = (rank + 1) % len(active_devices)
        peer_key = f"memcache-a5-tp-{run_id}-rank-{peer_rank}"
        stage = "remote_get"
        statuses_raw = backend.get(
            [peer_key],
            [[int(destination.data_ptr())]],
            [[payload_bytes]],
        )
        statuses = (
            None
            if statuses_raw is None
            else [int(value) for value in statuses_raw]
        )
        if statuses != [0]:
            raise RuntimeError(f"remote get returned statuses={statuses}")
        torch.npu.synchronize()
        expected = peer_rank + 1
        if not bool(torch.all(destination == expected).item()):
            raise AssertionError(
                f"remote payload mismatch: peer_rank={peer_rank} expected={expected}"
            )
        print(
            f"[memcache-tp] rank={rank} device={device_id} "
            f"peer_rank={peer_rank} remote_get=passed",
            flush=True,
        )
        result.update(ok=True, stage="complete", peer_rank=peer_rank)
    except Exception:
        with contextlib.suppress(Exception):
            barrier.abort()
        result.update(stage=stage, error=traceback.format_exc())
    finally:
        cleanup_errors: list[str] = []
        if backend is not None and backend.store is not None:
            for pointer, size in reversed(registered):
                try:
                    ret = int(backend.store.unregister_buffer(pointer, size))
                    if ret != 0:
                        cleanup_errors.append(
                            f"unregister ret={ret} ptr=0x{pointer:x}"
                        )
                except Exception as exc:
                    cleanup_errors.append(f"unregister failed: {exc!r}")
            try:
                backend.store.close()
            except Exception as exc:
                cleanup_errors.append(f"store.close failed: {exc!r}")
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            result["ok"] = False
        _send(connection, result)
        connection.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if result["ok"] else 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--active-devices", default="0,7")
    parser.add_argument("--meta-host", default="127.0.0.1")
    parser.add_argument("--meta-port", type=int, default=5000)
    parser.add_argument("--payload-bytes", type=int, default=1 << 20)
    parser.add_argument("--process-timeout", type=float, default=180.0)
    parser.add_argument(
        "--local-eid-map",
        default=os.getenv("MEMCACHE_TP_LOCAL_EID_MAP"),
        help="JSON object mapping physical NPU IDs to 32-digit fullmesh EIDs",
    )
    default_eid_map_file = os.getenv("MEMCACHE_TP_LOCAL_EID_MAP_FILE")
    parser.add_argument(
        "--local-eid-map-file",
        type=Path,
        default=Path(default_eid_map_file) if default_eid_map_file else None,
        help="path to the same JSON map; mutually exclusive with --local-eid-map",
    )
    args = parser.parse_args()

    config_path = args.config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not _service_ready(args.meta_host, args.meta_port):
        raise RuntimeError(
            f"MemCache MetaService is not reachable at "
            f"{args.meta_host}:{args.meta_port}"
        )

    visible_devices = _parse_devices(args.visible_devices)
    active_devices = _parse_devices(args.active_devices)
    if len(active_devices) < 2:
        raise ValueError("MemCache TP probe requires at least two active devices")
    unavailable = sorted(set(active_devices) - set(visible_devices))
    if unavailable:
        raise ValueError(
            f"active devices are not visible: {unavailable}; "
            f"visible={visible_devices}"
        )
    local_eid_map = _load_local_eid_map(
        args.local_eid_map,
        args.local_eid_map_file,
    )
    if local_eid_map:
        missing_eids = sorted(set(active_devices) - set(local_eid_map))
        if missing_eids:
            raise ValueError(
                f"local EID map does not cover active devices: {missing_eids}"
            )

    run_id = uuid.uuid4().hex
    print(
        f"[memcache-tp] visible_devices={visible_devices} "
        f"active_devices={active_devices} config={config_path} "
        f"eid_source={'override' if local_eid_map else 'rootinfo'}",
        flush=True,
    )
    context = mp.get_context("spawn")
    barrier = context.Barrier(len(active_devices))
    processes: list[mp.Process] = []
    parents: list[Any] = []
    for rank in range(len(active_devices)):
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_worker,
            args=(
                child,
                rank,
                active_devices,
                visible_devices,
                str(config_path),
                run_id,
                args.payload_bytes,
                local_eid_map.get(active_devices[rank]),
                barrier,
            ),
            name=f"memcache-tp-{rank}",
        )
        parents.append(parent)
        processes.append(process)
        process.start()
        child.close()

    deadline = time.monotonic() + args.process_timeout
    results: list[dict[str, Any]] = []
    for parent, process in zip(parents, processes):
        remaining = max(0.0, deadline - time.monotonic())
        if parent.poll(remaining):
            results.append(parent.recv())
        else:
            results.append(
                {
                    "ok": False,
                    "stage": "parent_timeout",
                    "error": f"process {process.name} timed out",
                }
            )
        parent.close()

    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)

    failures = [result for result in results if not result.get("ok")]
    if failures:
        for failure in failures:
            print(f"[memcache-tp] ERROR {failure}", flush=True)
        print("[memcache-tp] FAILED", flush=True)
        return 1
    print("[memcache-tp] CROSS-RANK ROUND TRIP PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
