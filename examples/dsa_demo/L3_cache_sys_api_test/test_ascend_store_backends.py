#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real A5 integration probe for vLLM-Ascend AscendStore backends.

This probe does not load a model. It starts a real metadata service, then uses
separate spawned processes for the scheduler client and the NPU worker backend.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import socket
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any


NUM_KEYS = 2
NUM_LAYERS = 3
BYTES_PER_LAYER = 128 * 1024


def _free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def _set_child_environment(config: dict[str, Any]) -> None:
    os.environ["VLLM_HOST_IP"] = "127.0.0.1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = config["visible_devices"]
    if config["backend"] == "memcache":
        os.environ["MMC_LOCAL_CONFIG_PATH"] = config["backend_config_path"]
    else:
        os.environ["MOONCAKE_CONFIG_PATH"] = config["backend_config_path"]


def _prepare_worker_device(device_id: int) -> None:
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    torch.npu.synchronize()


def _make_parallel_config():
    from vllm.config import ParallelConfig

    return ParallelConfig()


def _make_backend(config: dict[str, Any], scheduler: bool):
    parallel_config = _make_parallel_config()
    if config["backend"] == "memcache":
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
            MemcacheBackend,
        )

        if scheduler:
            return MemcacheBackend.create_scheduler_client(parallel_config)
        return MemcacheBackend(
            parallel_config,
            local_rank=config["device_id"],
            init_bm=True,
        )

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
        MooncakeBackend,
    )

    if scheduler:
        return MooncakeBackend.create_scheduler_client(parallel_config)
    return MooncakeBackend(parallel_config)


def _cleanup_backend(
    config: dict[str, Any],
    backend: Any,
    registered: list[tuple[int, int]],
) -> list[str]:
    errors: list[str] = []
    if backend is None:
        return errors

    try:
        if config["backend"] == "memcache":
            if backend.store is not None:
                for pointer, size in reversed(registered):
                    result = int(backend.store.unregister_buffer(pointer, size))
                    if result != 0:
                        errors.append(f"memcache unregister ret={result} ptr=0x{pointer:x}")
        else:
            from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

            engine = global_te.transfer_engine
            if engine is not None:
                for pointer, _ in reversed(registered):
                    result = int(engine.unregister_memory(pointer))
                    if result != 0:
                        errors.append(f"mooncake unregister ret={result} ptr=0x{pointer:x}")
    except Exception as exc:
        errors.append(f"buffer cleanup failed: {exc!r}")

    try:
        if backend.store is not None:
            backend.store.close()
    except Exception as exc:
        errors.append(f"store.close failed: {exc!r}")

    if config["backend"] == "mooncake":
        try:
            from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te

            global_te.transfer_engine = None
            global_te.is_register_buffer = False
        except Exception as exc:
            errors.append(f"transfer engine cleanup failed: {exc!r}")

    del backend
    gc.collect()
    return errors


def _send(connection, message: dict[str, Any]) -> None:
    try:
        connection.send(message)
    except (BrokenPipeError, EOFError, OSError):
        pass


def _worker_process(connection, config: dict[str, Any], keys: list[str]) -> None:
    backend = None
    registered: list[tuple[int, int]] = []
    final: dict[str, Any] = {
        "stage": "final",
        "ok": False,
        "backend": config["backend"],
    }
    try:
        _set_child_environment(config)
        _prepare_worker_device(config["device_id"])
        backend = _make_backend(config, scheduler=False)

        import torch

        total_bytes = NUM_KEYS * NUM_LAYERS * BYTES_PER_LAYER
        cpu_pattern = torch.arange(total_bytes, dtype=torch.int64) % 251
        device = torch.device(f"npu:{config['device_id']}")
        source = (
            cpu_pattern.to(torch.uint8)
            .reshape(NUM_KEYS, NUM_LAYERS, BYTES_PER_LAYER)
            .to(device)
            .contiguous()
        )
        destination = torch.zeros_like(source)
        torch.npu.synchronize()

        for tensor in (source, destination):
            registered.append((int(tensor.data_ptr()), int(tensor.numel() * tensor.element_size())))
        backend.register_buffer(
            [pointer for pointer, _ in registered],
            [size for _, size in registered],
        )

        sizes = [[BYTES_PER_LAYER] * NUM_LAYERS for _ in range(NUM_KEYS)]
        source_addresses = [
            [int(source[key_index, layer].data_ptr()) for layer in range(NUM_LAYERS)]
            for key_index in range(NUM_KEYS)
        ]
        destination_addresses = [
            [int(destination[key_index, layer].data_ptr()) for layer in range(NUM_LAYERS)]
            for key_index in range(NUM_KEYS)
        ]

        backend.put(keys, source_addresses, sizes)
        deadline = time.monotonic() + config["operation_timeout"]
        states: list[int] = []
        while time.monotonic() < deadline:
            states = [int(value) for value in backend.exists(keys)]
            if states == [1] * NUM_KEYS:
                break
            time.sleep(0.05)
        if states != [1] * NUM_KEYS:
            raise RuntimeError(f"worker keys did not become visible: {states}")

        _send(connection, {"stage": "put_ready", "ok": True, "states": states})
        if not connection.poll(config["operation_timeout"]):
            raise TimeoutError("timed out waiting for scheduler visibility result")
        command = connection.recv()
        if command != "scheduler_done":
            raise RuntimeError(f"unexpected parent command: {command!r}")

        get_result = backend.get(keys, destination_addresses, sizes)
        torch.npu.synchronize()
        statuses = None if get_result is None else [int(value) for value in get_result]
        if statuses != [0] * NUM_KEYS:
            raise RuntimeError(f"backend.get returned unexpected statuses: {statuses}")
        if not torch.equal(source.cpu(), destination.cpu()):
            mismatch = torch.nonzero(source.cpu() != destination.cpu())
            first = mismatch[0].tolist() if mismatch.numel() else None
            raise AssertionError(f"backend round trip mismatch: first={first}")

        final.update(
            ok=True,
            statuses=statuses,
            payload_bytes=total_bytes,
        )
    except Exception:
        final["error"] = traceback.format_exc()
    finally:
        cleanup_errors = _cleanup_backend(config, backend, registered)
        if cleanup_errors:
            final["ok"] = False
            final["cleanup_errors"] = cleanup_errors
        _send(connection, final)
        connection.close()
        sys.stdout.flush()
        sys.stderr.flush()
        if config["backend"] == "memcache":
            # MemCache v1.2.0 has a native ResourceTracker/atexit ordering
            # defect. Explicit backend cleanup is complete above; bypass only
            # the broken process-global finalizer in this probe child.
            os._exit(0 if final["ok"] else 1)


def _scheduler_process(connection, config: dict[str, Any], keys: list[str]) -> None:
    backend = None
    result: dict[str, Any] = {
        "stage": "scheduler",
        "ok": False,
        "backend": config["backend"],
    }
    try:
        _set_child_environment(config)
        if config["backend"] == "mooncake":
            _prepare_worker_device(config["device_id"])
        backend = _make_backend(config, scheduler=True)
        states = [int(value) for value in backend.exists(keys)]
        result.update(ok=states == [1] * NUM_KEYS, states=states)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        cleanup_errors = _cleanup_backend(config, backend, [])
        if cleanup_errors:
            result["ok"] = False
            result["cleanup_errors"] = cleanup_errors
        _send(connection, result)
        connection.close()
        sys.stdout.flush()
        sys.stderr.flush()
        if config["backend"] == "memcache":
            os._exit(0 if result["ok"] else 1)


def _receive(connection, process: mp.Process, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if connection.poll(0.1):
            return connection.recv()
        if not process.is_alive():
            raise RuntimeError(f"child exited before sending a result: exitcode={process.exitcode}")
    raise TimeoutError(f"timed out waiting for child pid={process.pid}")


def _join_or_kill(process: mp.Process, timeout: float = 10.0) -> None:
    process.join(timeout)
    if process.is_alive():
        process.kill()
        process.join(2.0)


def _write_memcache_config(path: Path, config: dict[str, Any]) -> None:
    text = "\n".join(
        (
            f"ock.mmc.meta_service_url = {config['meta_url']}",
            f"ock.mmc.local_service.config_store_url = {config['config_store_url']}",
            "ock.mmc.log_level = info",
            f"ock.mmc.local_service.world_size = {config['world_size']}",
            f"ock.mmc.local_service.protocol = {config['memcache_protocol']}",
            f"ock.mmc.local_service.hcom_url = tcp://127.0.0.1:{config['hcom_port']}",
            f"ock.mmc.local_service.dram.size = {config['dram_size']}",
            f"ock.mmc.local_service.max.dram.size = {config['max_dram_size']}",
            "ock.mmc.local_service.hbm.size = 0",
            "ock.mmc.local_service.max.hbm.size = 0",
            "",
        )
    )
    path.write_text(text, encoding="utf-8")


def _write_mooncake_config(path: Path, config: dict[str, Any]) -> None:
    payload = {
        "metadata_server": "P2PHANDSHAKE",
        "global_segment_size": config["mooncake_segment_bytes"],
        "local_buffer_size": config["mooncake_buffer_bytes"],
        "protocol": "ascend",
        "device_name": "",
        "master_server_address": config["master_address"],
        "preferred_segment": False,
        "prefer_alloc_in_same_node": True,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _start_service(config: dict[str, Any]):
    if config["backend"] == "memcache":
        from memcache_a5_probe import start_meta_service

        service_args = argparse.Namespace(
            external_meta=False,
            meta_url=config["meta_url"],
            config_store_url=config["config_store_url"],
            metrics_url=config["metrics_url"],
            log_level="info",
            startup_timeout=config["startup_timeout"],
        )
        return start_meta_service(service_args)

    from test_mooncake import start_master

    service_args = argparse.Namespace(
        external_master=False,
        master_address=config["master_address"],
        metrics_port=config["master_metrics_port"],
        lease_ttl_ms=300000,
        client_ttl_seconds=120,
        startup_timeout=config["startup_timeout"],
    )
    return start_master(service_args)


def _stop_service(backend: str, service: Any) -> None:
    if service is None:
        return
    if backend == "memcache":
        if service.is_alive():
            service.kill()
        service.join(timeout=5.0)
        print(f"[ascend-store-backend] MetaService stopped exitcode={service.exitcode}")
        return

    from test_mooncake import stop_master

    stop_master(service)


def run(config: dict[str, Any]) -> None:
    context = mp.get_context("spawn")
    keys = [f"ascend-store-{config['backend']}-{uuid.uuid4().hex}-{index}" for index in range(NUM_KEYS)]
    service = None
    worker = None
    scheduler = None

    with tempfile.TemporaryDirectory(prefix=f"ascend_store_{config['backend']}_") as directory:
        config_path = Path(directory) / ("mmc-local.conf" if config["backend"] == "memcache" else "mooncake.json")
        config["backend_config_path"] = str(config_path)
        if config["backend"] == "memcache":
            _write_memcache_config(config_path, config)
        else:
            _write_mooncake_config(config_path, config)

        try:
            service = _start_service(config)

            worker_parent, worker_child = context.Pipe(duplex=True)
            worker = context.Process(
                target=_worker_process,
                args=(worker_child, config, keys),
                name=f"ascend-store-{config['backend']}-worker",
            )
            worker.start()
            worker_child.close()

            worker_ready = _receive(worker_parent, worker, config["process_timeout"])
            print(f"[ascend-store-backend] worker_ready={worker_ready}")
            if worker_ready.get("stage") != "put_ready" or not worker_ready.get("ok"):
                raise RuntimeError(f"worker failed before scheduler query: {worker_ready}")

            scheduler_parent, scheduler_child = context.Pipe(duplex=False)
            scheduler = context.Process(
                target=_scheduler_process,
                args=(scheduler_child, config, keys),
                name=f"ascend-store-{config['backend']}-scheduler",
            )
            scheduler.start()
            scheduler_child.close()
            scheduler_result = _receive(scheduler_parent, scheduler, config["process_timeout"])
            print(f"[ascend-store-backend] scheduler_result={scheduler_result}")
            _join_or_kill(scheduler)

            worker_parent.send("scheduler_done")
            worker_result = _receive(worker_parent, worker, config["process_timeout"])
            print(f"[ascend-store-backend] worker_result={worker_result}")
            _join_or_kill(worker)

            if not scheduler_result.get("ok"):
                raise RuntimeError(f"scheduler backend did not see worker keys: {scheduler_result}")
            if not worker_result.get("ok"):
                raise RuntimeError(f"worker backend round trip failed: {worker_result}")
            if scheduler.exitcode != 0 or worker.exitcode != 0:
                raise RuntimeError(
                    f"child exit failure: scheduler={scheduler.exitcode} worker={worker.exitcode}"
                )
            print(
                "[ascend-store-backend] PASSED "
                f"backend={config['backend']} keys={NUM_KEYS} "
                f"payload_bytes={worker_result['payload_bytes']}"
            )
        finally:
            if worker is not None and worker.is_alive():
                worker.kill()
                worker.join(2.0)
            if scheduler is not None and scheduler.is_alive():
                scheduler.kill()
                scheduler.join(2.0)
            _stop_service(config["backend"], service)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("memcache", "mooncake"), required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--visible-devices", default="0")
    parser.add_argument("--memcache-protocol", default="device_urma")
    parser.add_argument("--dram-size", default="1GB")
    parser.add_argument("--max-dram-size", default="256GB")
    parser.add_argument("--world-size", type=int, default=256)
    parser.add_argument("--mooncake-segment-bytes", type=int, default=1 << 30)
    parser.add_argument("--mooncake-buffer-bytes", type=int, default=1 << 30)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--operation-timeout", type=float, default=60.0)
    parser.add_argument("--process-timeout", type=float, default=120.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = vars(args)
    meta_port, config_port, metrics_port, hcom_port, master_port, master_metrics_port = _free_ports(6)
    config.update(
        meta_url=f"tcp://127.0.0.1:{meta_port}",
        config_store_url=f"tcp://127.0.0.1:{config_port}",
        metrics_url=f"http://127.0.0.1:{metrics_port}",
        hcom_port=hcom_port,
        master_address=f"127.0.0.1:{master_port}",
        master_metrics_port=master_metrics_port,
    )
    print(f"[ascend-store-backend] config={config}")
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
