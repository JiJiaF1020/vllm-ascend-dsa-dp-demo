#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate exact A5 DSA swapped-memory component IO through an L3 backend.

This is the first DSA L2/L3 gate. It deliberately does not load a model. One
logical GLM-5.2 A5 block is represented by the real byte-level schema:

* 21 full Indexer layers, each with a 16,384-byte key plane and a 512-byte
  FP32-scale plane;
* 78 packed MLA layers, each with 128 * 656 bytes.

The probe always allocates source planes with
``torch_npu.empty_with_swapped_memory``. Destination planes can use swapped
memory or ordinary HBM, allowing an isolated L2->L3->L1 direct-load gate.
It registers ordinary HBM first and swapped arenas second, stores the two
logical components, loads them back, and checks every byte. Mooncake is the
primary path; MemCache uses the same schema as a fallback backend.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BLOCK_SIZE = 128
FULL_INDEXER_LAYERS = 21
TARGET_MLA_LAYERS = 78
INDEXER_KEY_BYTES_PER_BLOCK = BLOCK_SIZE * 128
INDEXER_SCALE_BYTES_PER_BLOCK = BLOCK_SIZE * 4
PACKED_MLA_BYTES_PER_BLOCK = BLOCK_SIZE * 656

EXPECTED_INDEXER_BYTES = 354_816
EXPECTED_MLA_BYTES = 6_549_504
EXPECTED_COMPLETE_BLOCK_BYTES = 6_904_320


@dataclass(frozen=True, slots=True)
class PlaneSpec:
    name: str
    component: str
    num_bytes: int
    seed: int


@dataclass(slots=True)
class PlaneBuffers:
    spec: PlaneSpec
    source: Any
    destination: Any
    expected: Any
    block_index: int


def build_a5_dsa_plane_specs(component: str) -> list[PlaneSpec]:
    specs: list[PlaneSpec] = []
    if component in {"all", "indexer"}:
        for layer in range(FULL_INDEXER_LAYERS):
            specs.append(
                PlaneSpec(
                    name=f"indexer.layer{layer}.key",
                    component="indexer",
                    num_bytes=INDEXER_KEY_BYTES_PER_BLOCK,
                    seed=11 + layer * 7,
                )
            )
            specs.append(
                PlaneSpec(
                    name=f"indexer.layer{layer}.scale",
                    component="indexer",
                    num_bytes=INDEXER_SCALE_BYTES_PER_BLOCK,
                    seed=97 + layer * 13,
                )
            )
    if component in {"all", "mla"}:
        for layer in range(TARGET_MLA_LAYERS):
            specs.append(
                PlaneSpec(
                    name=f"mla.layer{layer}.packed",
                    component="mla",
                    num_bytes=PACKED_MLA_BYTES_PER_BLOCK,
                    seed=173 + layer * 17,
                )
            )
    _validate_schema(specs, component)
    return specs


def select_schema_mode(
    specs: list[PlaneSpec],
    schema_mode: str,
) -> list[PlaneSpec]:
    if schema_mode == "exact":
        return specs
    if schema_mode != "smoke":
        raise ValueError(f"unsupported schema mode: {schema_mode!r}")
    selected: list[PlaneSpec] = []
    for component in ("indexer", "mla"):
        first = next(
            (spec for spec in specs if spec.component == component),
            None,
        )
        if first is not None:
            selected.append(first)
    return selected


def _validate_schema(specs: list[PlaneSpec], component: str) -> None:
    indexer_specs = [spec for spec in specs if spec.component == "indexer"]
    mla_specs = [spec for spec in specs if spec.component == "mla"]
    indexer_bytes = sum(spec.num_bytes for spec in indexer_specs)
    mla_bytes = sum(spec.num_bytes for spec in mla_specs)
    if component in {"all", "indexer"}:
        if len(indexer_specs) != FULL_INDEXER_LAYERS * 2:
            raise RuntimeError("DSA Indexer schema has an unexpected plane count")
        if indexer_bytes != EXPECTED_INDEXER_BYTES:
            raise RuntimeError(
                f"DSA Indexer schema byte count changed: actual={indexer_bytes}, expected={EXPECTED_INDEXER_BYTES}"
            )
    if component in {"all", "mla"}:
        if len(mla_specs) != TARGET_MLA_LAYERS:
            raise RuntimeError("DSA MLA schema has an unexpected plane count")
        if mla_bytes != EXPECTED_MLA_BYTES:
            raise RuntimeError(f"DSA MLA schema byte count changed: actual={mla_bytes}, expected={EXPECTED_MLA_BYTES}")
    if component == "all" and indexer_bytes + mla_bytes != EXPECTED_COMPLETE_BLOCK_BYTES:
        raise RuntimeError("DSA complete block byte count changed")


def _require_tcp_service(address: str, service_name: str) -> None:
    host, separator, raw_port = address.rpartition(":")
    if not separator or not host:
        raise ValueError(f"invalid {service_name} address {address!r}; expected host:port")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"invalid {service_name} port in {address!r}") from exc
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as exc:
        raise RuntimeError(f"{service_name} is not reachable at {address}") from exc


def _configure_mooncake_environment(args: argparse.Namespace) -> None:
    if args.mooncake_comm_mode not in {"global_ub_device", "global_ub"}:
        raise ValueError("this swapped-memory probe supports global_ub_device/global_ub")
    os.environ.pop("ASCEND_LOCAL_COMM_RES", None)
    protocol_desc = ["ub_ctp:device"]
    if args.mooncake_comm_mode == "global_ub":
        protocol_desc.append("ub_ctp:host")
    os.environ["ASCEND_GLOBAL_RESOURCE_CONFIG"] = json.dumps(
        {
            "comm_resource_config.protocol_desc": protocol_desc,
            "comm_resource_config.max_active_channels": args.max_active_channels,
        },
        separators=(",", ":"),
    )
    os.environ["HCCL_RDMA_TIMEOUT"] = os.getenv("HCCL_RDMA_TIMEOUT", "17")
    os.environ["ASCEND_CONNECT_TIMEOUT"] = os.getenv("ASCEND_CONNECT_TIMEOUT", "30000")
    os.environ["ASCEND_TRANSFER_TIMEOUT"] = os.getenv("ASCEND_TRANSFER_TIMEOUT", "30000")


def _configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    if args.visible_devices:
        os.environ["ASCEND_VISIBLE_DEVICES"] = args.visible_devices
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.visible_devices
    if args.backend == "mooncake":
        _require_tcp_service(args.mooncake_master, "Mooncake Master")
        _configure_mooncake_environment(args)
        os.environ.pop("MMC_LOCAL_CONFIG_PATH", None)
    else:
        config_path = args.memcache_config.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        _require_tcp_service(args.memcache_meta, "MemCache MetaService")
        os.environ["MMC_LOCAL_CONFIG_PATH"] = str(config_path)
        if args.memcache_local_eid:
            os.environ["USE_LOCAL_EID"] = args.memcache_local_eid
        else:
            os.environ.pop("USE_LOCAL_EID", None)
        os.environ.pop("ASCEND_GLOBAL_RESOURCE_CONFIG", None)
        os.environ.pop("ASCEND_LOCAL_COMM_RES", None)


def _make_pattern(torch: Any, num_bytes: int, seed: int, device: Any) -> Any:
    return torch.arange(num_bytes, dtype=torch.int32, device=device).add(seed).remainder(251).sub(125).to(torch.int8)


def _read_swapped_block(torch: Any, tensor: Any, block_index: int) -> Any:
    source = tensor[block_index]
    staging = torch.empty(tuple(source.shape), dtype=source.dtype, device=source.device)
    staging.fill_(1).mul_(source)
    return staging.cpu()


def _allocate_schema_buffers(
    torch: Any,
    torch_npu: Any,
    specs: list[PlaneSpec],
    device: Any,
    arena_blocks: int,
    block_index: int,
    destination_memory: str,
) -> list[PlaneBuffers]:
    buffers: list[PlaneBuffers] = []
    for spec in specs:
        source = torch_npu.empty_with_swapped_memory(
            (arena_blocks, spec.num_bytes),
            dtype=torch.int8,
            device=device,
        )
        destination = (
            torch_npu.empty_with_swapped_memory(
                (arena_blocks, spec.num_bytes),
                dtype=torch.int8,
                device=device,
            )
            if destination_memory == "swapped"
            else torch.empty(
                (arena_blocks, spec.num_bytes),
                dtype=torch.int8,
                device=device,
            )
        )
        source.zero_()
        destination.zero_()
        pattern = _make_pattern(torch, spec.num_bytes, spec.seed, device)
        source[block_index].add_(pattern)
        buffers.append(
            PlaneBuffers(
                spec=spec,
                source=source,
                destination=destination,
                expected=pattern.cpu(),
                block_index=block_index,
            )
        )
    torch.npu.synchronize()
    return buffers


def _component_payload(
    buffers: list[PlaneBuffers],
    component: str,
    use_destination: bool,
) -> tuple[list[int], list[int]]:
    selected = [item for item in buffers if item.spec.component == component]
    tensors = [item.destination if use_destination else item.source for item in selected]
    return (
        [
            int(tensor.data_ptr()) + item.block_index * item.spec.num_bytes
            for item, tensor in zip(selected, tensors, strict=True)
        ],
        [item.spec.num_bytes for item in selected],
    )


def _build_io_columns(
    buffers: list[PlaneBuffers],
    run_id: str,
    use_destination: bool,
) -> tuple[list[str], list[list[int]], list[list[int]]]:
    components = []
    for component in ("indexer", "mla"):
        if any(item.spec.component == component for item in buffers):
            components.append(component)
    keys = [f"vllm-ascend:dsa-l3-probe:v1:{run_id}:{item}" for item in components]
    addrs: list[list[int]] = []
    sizes: list[list[int]] = []
    for component in components:
        component_addrs, component_sizes = _component_payload(
            buffers,
            component,
            use_destination,
        )
        addrs.append(component_addrs)
        sizes.append(component_sizes)
    return keys, addrs, sizes


def _verify_buffers(torch: Any, buffers: list[PlaneBuffers]) -> dict[str, str]:
    digests = {"indexer": hashlib.sha256(), "mla": hashlib.sha256()}
    for item in buffers:
        actual = _read_swapped_block(
            torch,
            item.destination,
            item.block_index,
        )
        torch.npu.synchronize()
        if not torch.equal(actual, item.expected):
            mismatch = torch.nonzero(actual != item.expected, as_tuple=False)
            first = int(mismatch[0].item()) if mismatch.numel() else -1
            raise AssertionError(
                "DSA swapped L3 payload mismatch: "
                f"plane={item.spec.name}, first_byte={first}, "
                f"actual={int(actual[first]) if first >= 0 else None}, "
                f"expected={int(item.expected[first]) if first >= 0 else None}"
            )
        digests[item.spec.component].update(actual.numpy().tobytes())
    return {
        component: digest.hexdigest()
        for component, digest in digests.items()
        if any(item.spec.component == component for item in buffers)
    }


def _wait_exists(
    exists: Any,
    keys: list[str],
    timeout_seconds: float,
) -> list[int]:
    deadline = time.monotonic() + timeout_seconds
    latest: list[int] = []
    while time.monotonic() < deadline:
        latest = [int(value) for value in exists(keys)]
        if latest == [1] * len(keys):
            return latest
        time.sleep(0.05)
    raise RuntimeError(f"L3 keys did not become visible: states={latest}")


def _initialize_mooncake_engine(args: argparse.Namespace) -> Any:
    """Initialize HIXL before allocating the swapped arenas.

    vLLM-Ascend constructs its Mooncake backend before it registers KV-cache
    buffers. Keeping the probe in that order separates HComm initialization
    failures from the swapped-memory registration and IO capability under test.
    """
    from mooncake.engine import TransferEngine

    print(
        "[dsa-swapped-l3] stage=transfer_engine_initialize "
        f"max_active_channels={args.max_active_channels}",
        flush=True,
    )
    engine = TransferEngine()
    ret = int(
        engine.initialize(
            args.local_host,
            "P2PHANDSHAKE",
            "ascend",
            "",
        )
    )
    if ret != 0:
        raise RuntimeError(f"Mooncake TransferEngine.initialize ret={ret}")
    print("[dsa-swapped-l3] stage=transfer_engine_initialize passed", flush=True)
    return engine


def _run_mooncake(
    args: argparse.Namespace,
    torch: Any,
    buffers: list[PlaneBuffers],
    hbm_sentinels: list[Any],
    engine: Any,
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    store = None
    registered: list[tuple[int, int]] = []
    try:
        local_segment = f"{args.local_host}:{int(engine.get_rpc_port())}"
        store = MooncakeDistributedStore()
        ret = int(
            store.setup(
                local_hostname=local_segment,
                metadata_server="P2PHANDSHAKE",
                global_segment_size=args.segment_bytes,
                local_buffer_size=args.segment_bytes,
                protocol="ascend",
                rdma_devices="",
                master_server_addr=args.mooncake_master,
                engine=engine.get_engine(),
            )
        )
        if ret != 0:
            raise RuntimeError(f"Mooncake store.setup ret={ret}")

        # Deliberately register ordinary HBM first, then swapped-memory ranges.
        for tensor in hbm_sentinels:
            pointer = int(tensor.data_ptr())
            size = int(tensor.numel() * tensor.element_size())
            ret = int(engine.register_memory(pointer, size))
            if ret != 0:
                raise RuntimeError(f"Mooncake HBM register_memory ptr=0x{pointer:x} ret={ret}")
            registered.append((pointer, size))
        for item in buffers:
            for tensor in (item.source, item.destination):
                pointer = int(tensor.data_ptr())
                size = int(tensor.numel() * tensor.element_size())
                ret = int(engine.register_memory(pointer, size))
                if ret != 0:
                    raise RuntimeError(
                        "Mooncake swapped register_memory failed: "
                        f"plane={item.spec.name}, ptr=0x{pointer:x}, "
                        f"bytes={size}, ret={ret}"
                    )
                registered.append((pointer, size))

        run_id = uuid.uuid4().hex
        keys, source_addrs, sizes = _build_io_columns(
            buffers,
            run_id,
            use_destination=False,
        )
        destination_keys, destination_addrs, destination_sizes = _build_io_columns(
            buffers,
            run_id,
            use_destination=True,
        )
        if destination_keys != keys or destination_sizes != sizes:
            raise RuntimeError("DSA source/destination IO schema changed")

        config = ReplicateConfig()
        config.preferred_segment = local_segment
        config.prefer_alloc_in_same_node = True
        put_result = [
            int(value)
            for value in store.batch_put_from_multi_buffers(
                keys,
                source_addrs,
                sizes,
                config,
            )
        ]
        if len(put_result) != len(keys) or any(value < 0 for value in put_result):
            raise RuntimeError(f"Mooncake swapped put failed: {put_result}")
        _wait_exists(store.batch_is_exist, keys, args.io_timeout)
        get_result = [
            int(value)
            for value in store.batch_get_into_multi_buffers(
                keys,
                destination_addrs,
                destination_sizes,
            )
        ]
        if len(get_result) != len(keys) or any(value < 0 for value in get_result):
            raise RuntimeError(f"Mooncake swapped get failed: {get_result}")
        torch.npu.synchronize()
        return put_result, get_result, registered
    finally:
        for pointer, _ in reversed(registered):
            with contextlib.suppress(Exception):
                engine.unregister_memory(pointer)
        if store is not None:
            with contextlib.suppress(Exception):
                store.close()


def _run_memcache(
    args: argparse.Namespace,
    torch: Any,
    buffers: list[PlaneBuffers],
    hbm_sentinels: list[Any],
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    from vllm.config import ParallelConfig

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.memcache_backend import (
        MemcacheBackend,
    )

    backend = MemcacheBackend(
        ParallelConfig(),
        local_rank=args.device_id,
        init_bm=True,
    )
    registered: list[tuple[int, int]] = []
    try:
        tensors = [*hbm_sentinels]
        for item in buffers:
            tensors.extend((item.source, item.destination))
        registered = [
            (
                int(tensor.data_ptr()),
                int(tensor.numel() * tensor.element_size()),
            )
            for tensor in tensors
        ]
        backend.register_buffer(
            [pointer for pointer, _ in registered],
            [size for _, size in registered],
        )

        run_id = uuid.uuid4().hex
        keys, source_addrs, sizes = _build_io_columns(
            buffers,
            run_id,
            use_destination=False,
        )
        destination_keys, destination_addrs, destination_sizes = _build_io_columns(
            buffers,
            run_id,
            use_destination=True,
        )
        if destination_keys != keys or destination_sizes != sizes:
            raise RuntimeError("DSA source/destination IO schema changed")
        backend.put(keys, source_addrs, sizes)
        _wait_exists(backend.exists, keys, args.io_timeout)
        raw_result = backend.get(keys, destination_addrs, destination_sizes)
        get_result = [] if raw_result is None else [int(value) for value in raw_result]
        if get_result != [0] * len(keys):
            raise RuntimeError(f"MemCache swapped get failed: {get_result}")
        torch.npu.synchronize()
        # MemCache Backend currently does not return put statuses. Visibility is
        # the strictest available success signal until that contract is fixed.
        put_result = [0] * len(keys)
        return put_result, get_result, registered
    finally:
        if backend.store is not None:
            for pointer, size in reversed(registered):
                with contextlib.suppress(Exception):
                    backend.store.unregister_buffer(pointer, size)
            with contextlib.suppress(Exception):
                backend.store.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mooncake", "memcache"), default="mooncake")
    parser.add_argument("--component", choices=("all", "indexer", "mla"), default="all")
    parser.add_argument("--schema-mode", choices=("smoke", "exact"), default="exact")
    parser.add_argument(
        "--destination-memory",
        choices=("swapped", "hbm"),
        default="swapped",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--visible-devices", default=os.getenv("ASCEND_RT_VISIBLE_DEVICES", "0"))
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--mooncake-master", default="127.0.0.1:50088")
    parser.add_argument(
        "--mooncake-comm-mode",
        choices=("global_ub_device", "global_ub"),
        default="global_ub",
    )
    parser.add_argument("--max-active-channels", type=int, default=1)
    parser.add_argument("--segment-bytes", type=int, default=1 << 30)
    parser.add_argument("--arena-blocks", type=int, default=8)
    parser.add_argument("--block-index", type=int, default=1)
    parser.add_argument(
        "--memcache-config",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline_e2e/configs/mmc-local.conf",
    )
    parser.add_argument("--memcache-meta", default="127.0.0.1:5000")
    parser.add_argument("--memcache-local-eid", default=os.getenv("USE_LOCAL_EID"))
    parser.add_argument("--io-timeout", type=float, default=60.0)
    args = parser.parse_args()

    if args.device_id < 0:
        raise ValueError("device_id must be non-negative")
    if args.max_active_channels <= 0:
        raise ValueError("max_active_channels must be positive")
    if args.segment_bytes < EXPECTED_COMPLETE_BLOCK_BYTES:
        raise ValueError("segment_bytes is smaller than one complete DSA block")
    if args.arena_blocks <= 0:
        raise ValueError("arena_blocks must be positive")
    if not 0 <= args.block_index < args.arena_blocks:
        raise ValueError("block_index must be inside the swapped arena")

    _configure_environment(args)

    # Runtime imports must happen after communication environment setup.
    import torch
    import torch_npu

    if not hasattr(torch_npu, "empty_with_swapped_memory"):
        raise RuntimeError("torch_npu.empty_with_swapped_memory is unavailable")
    torch.npu.set_device(args.device_id)
    mooncake_engine = None
    if args.backend == "mooncake":
        mooncake_engine = _initialize_mooncake_engine(args)
    device = torch.device(f"npu:{args.device_id}")
    specs = select_schema_mode(
        build_a5_dsa_plane_specs(args.component),
        args.schema_mode,
    )
    buffers = _allocate_schema_buffers(
        torch,
        torch_npu,
        specs,
        device,
        args.arena_blocks,
        args.block_index,
        args.destination_memory,
    )
    hbm_sentinels = [
        torch.arange(4096, dtype=torch.int32, device=device).remainder(127).to(torch.int8),
        torch.zeros(4096, dtype=torch.int8, device=device),
    ]
    torch.npu.synchronize()

    component_counts = {
        component: sum(1 for spec in specs if spec.component == component) for component in ("indexer", "mla")
    }
    component_bytes = {
        component: sum(spec.num_bytes for spec in specs if spec.component == component)
        for component in ("indexer", "mla")
    }
    print(
        "[dsa-swapped-l3] schema "
        f"backend={args.backend} device={args.device_id} "
        f"component={args.component} schema_mode={args.schema_mode} "
        f"destination_memory={args.destination_memory} "
        f"arena_blocks={args.arena_blocks} block_index={args.block_index} "
        f"planes={component_counts} bytes={component_bytes} "
        f"total_bytes={sum(component_bytes.values())}",
        flush=True,
    )

    started = time.perf_counter()
    try:
        if args.backend == "mooncake":
            if mooncake_engine is None:
                raise RuntimeError("Mooncake engine was not initialized")
            put_result, get_result, registered = _run_mooncake(
                args,
                torch,
                buffers,
                hbm_sentinels,
                mooncake_engine,
            )
        else:
            put_result, get_result, registered = _run_memcache(
                args,
                torch,
                buffers,
                hbm_sentinels,
            )
        digests = _verify_buffers(torch, buffers)
    except Exception:
        print("[dsa-swapped-l3] FAILED", flush=True)
        traceback.print_exc()
        return 1

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(
        "[dsa-swapped-l3] result "
        f"put={put_result} get={get_result} registered_regions={len(registered)} "
        f"registered_bytes={sum(size for _, size in registered)} "
        f"digests={digests} elapsed_ms={elapsed_ms:.3f}",
        flush=True,
    )
    print("[dsa-swapped-l3] DSA SWAPPED L3 ROUND TRIP PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
