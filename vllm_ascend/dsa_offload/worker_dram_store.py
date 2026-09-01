# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA worker 侧固定容量 payload arena 与 scheduler 目录镜像。

容量计划与 physical block-id 真源位于 ``scheduler_dram_pool.py``。本模块只创建固定
地址的 swapped-memory payload arena，并把 scheduler 下发的 assignment
镜像为 ``resident pool row -> logical block -> physical block id``。

no-MTP 时 store 保存 target transformer payload；MTP prefix 开启后，同一个
logical block 还保存 proposer 原生 BF16 full-cache planes。MTP 在活跃 sparse
decode 中仍全量驻留 HBM，不做 token-wise swap；只有 scheduler 已确认前缀内的
完整块才进入该 arena，provisional tail 永不发布。

prefix cache 的 hash/refcount 真源仍在 scheduler；worker 镜像允许多个 active
请求行指向同一个只读 DRAM block，只维护用于校验 assignment/release 对称性的
active refcount。block 0 保留为空映射，有效物理 block id 从 1 开始；初始化后
arena 地址和容量均不可变化。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from vllm_ascend.dsa_offload.contracts import (
    DSA_A5_PACKED_KV_ROW_BYTES,
    DSA_DRAM_NULL_BLOCK_ID,
)
from vllm_ascend.dsa_offload.l3.schema import DSAL3Component
from vllm_ascend.dsa_offload.ops import cache_block_byte_view

ArenaFactory = Callable[
    [tuple[int, ...], torch.dtype, int, torch.device],
    torch.Tensor,
]


@dataclass(frozen=True)
class DSALayerDRAMArenas:
    """同层 target MLA payload 与可选完整 Indexer 字节 plane。"""

    nope: torch.Tensor | None = None
    rope: torch.Tensor | None = None
    packed: torch.Tensor | None = None
    resident_source_bytes: tuple[torch.Tensor, ...] = ()
    resident_dram_bytes: tuple[torch.Tensor, ...] = ()
    indexer_source_bytes: tuple[torch.Tensor, ...] = ()
    indexer_dram_bytes: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class DSAMTPLayerDRAMArenas:
    """一个 MTP full-cache 层的原生 HBM 与 swapped-memory byte planes。"""

    source_bytes: tuple[torch.Tensor, ...]
    dram_bytes: tuple[torch.Tensor, ...]


def _allocate_swapped_arena(
    block_shape: tuple[int, ...],
    dtype: torch.dtype,
    capacity: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError("DSA sparse offload requires torch_npu swapped-memory arenas") from exc

    arena = torch_npu.empty_with_swapped_memory(
        (int(capacity), *block_shape),
        dtype=dtype,
        device=device,
    )
    if not arena.is_contiguous():
        raise RuntimeError(
            "DSA swapped-memory arena must be contiguous, got "
            f"shape={tuple(arena.shape)}, stride={tuple(arena.stride())}"
        )
    return arena


class DSAWorkerDRAMStore:
    """持有 worker payload arena 和 scheduler 目录的稳定行镜像。"""

    def __init__(
        self,
        *,
        usable_blocks: int,
        storage_rows: int,
        max_logical_blocks: int,
        device: torch.device,
        arena_factory: ArenaFactory | None = None,
        expected_bytes_per_logical_block: int | None = None,
    ) -> None:
        if usable_blocks <= 0:
            raise ValueError("DSA DRAM store requires positive usable blocks")
        if usable_blocks > np.iinfo(np.int32).max:
            raise ValueError("DSA DRAM store exceeds the int32 block-id ABI")
        if storage_rows <= 0:
            raise ValueError("DSA DRAM store requires positive row capacity")
        if max_logical_blocks <= 0:
            raise ValueError("DSA DRAM store requires positive logical block width")

        self.usable_blocks = int(usable_blocks)
        self.arena_capacity = self.usable_blocks + 1
        self.storage_rows = int(storage_rows)
        self.max_logical_blocks = int(max_logical_blocks)
        self.device = torch.device(device)
        self._arena_factory = arena_factory or _allocate_swapped_arena
        self._layer_arenas: dict[int, DSALayerDRAMArenas] = {}
        self._mtp_layer_arenas: dict[int, DSAMTPLayerDRAMArenas] = {}
        self.expected_bytes_per_logical_block = (
            int(expected_bytes_per_logical_block)
            if expected_bytes_per_logical_block is not None
            else None
        )
        self._registered_bytes_per_logical_block = 0

        self.logical_block_table = np.full(
            (self.storage_rows, self.max_logical_blocks),
            DSA_DRAM_NULL_BLOCK_ID,
            dtype=np.int32,
        )
        # worker 不分配 block id。prefix hit 允许多个 active row 共享同一
        # 物理块，因此这里只验证引用增减，不再记录唯一 owner。
        self._block_active_refcounts = np.zeros(
            self.arena_capacity,
            dtype=np.int32,
        )
        # 每个 resident-pool 行已经持久化到 DRAM 的连续完整块前缀。
        # MTP provisional token 只会写 target HBM 双尾块；只有下一 outer
        # step scheduler 已确认的前缀才能推进该水位。
        self.durable_full_block_counts = np.zeros(
            self.storage_rows,
            dtype=np.int32,
        )
        self.table_version = 0

    def _allocate_payload_arena(
        self,
        *,
        block_shape: tuple[int, ...],
        dtype: torch.dtype,
        payload_name: str,
    ) -> torch.Tensor:
        try:
            return self._arena_factory(
                block_shape,
                dtype,
                self.arena_capacity,
                self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                "DSA worker failed to create its scheduler-planned DRAM "
                f"arena: payload={payload_name}, "
                f"usable_blocks={self.usable_blocks}, "
                f"arena_capacity={self.arena_capacity}, "
                f"block_shape={block_shape}, dtype={dtype}"
            ) from exc

    def add_layer(
        self,
        *,
        layer_id: int,
        resident_nope_cache: torch.Tensor,
        resident_rope_cache: torch.Tensor,
        indexer_cache: torch.Tensor | None = None,
    ) -> None:
        """按 resident cache 的真实 block shape 建立该层 arena。"""

        layer_id = int(layer_id)
        if layer_id in self._layer_arenas:
            raise RuntimeError(f"DSA DRAM layer {layer_id} was initialized twice")
        if resident_nope_cache.shape[0] <= 0 or resident_rope_cache.shape[0] <= 0:
            raise ValueError("DSA resident cache must contain physical blocks")
        nope = self._allocate_payload_arena(
            block_shape=tuple(resident_nope_cache.shape[1:]),
            dtype=resident_nope_cache.dtype,
            payload_name=f"resident_nope[layer={layer_id}]",
        )
        rope = self._allocate_payload_arena(
            block_shape=tuple(resident_rope_cache.shape[1:]),
            dtype=resident_rope_cache.dtype,
            payload_name=f"resident_rope[layer={layer_id}]",
        )
        # block 0 是空映射。只清这一块，不触碰大块有效 arena。
        nope[DSA_DRAM_NULL_BLOCK_ID].zero_()
        rope[DSA_DRAM_NULL_BLOCK_ID].zero_()
        self._layer_arenas[layer_id] = DSALayerDRAMArenas(
            nope=nope,
            rope=rope,
            resident_source_bytes=(
                cache_block_byte_view(
                    resident_nope_cache,
                    name=f"resident_nope_cache[layer={layer_id}]",
                ),
                cache_block_byte_view(
                    resident_rope_cache,
                    name=f"resident_rope_cache[layer={layer_id}]",
                ),
            ),
            resident_dram_bytes=(
                cache_block_byte_view(
                    nope,
                    name=f"resident_nope_dram[layer={layer_id}]",
                ),
                cache_block_byte_view(
                    rope,
                    name=f"resident_rope_dram[layer={layer_id}]",
                ),
            ),
            **self._create_indexer_byte_arenas(
                layer_id=layer_id,
                indexer_planes=(indexer_cache,)
                if indexer_cache is not None
                else (),
            ),
        )
        self._registered_bytes_per_logical_block += (
            int(resident_nope_cache[0].numel())
            * resident_nope_cache.element_size()
            + int(resident_rope_cache[0].numel())
            * resident_rope_cache.element_size()
        )

    def add_packed_layer(
        self,
        *,
        layer_id: int,
        resident_packed_cache: torch.Tensor,
        indexer_cache: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        """为 A5 packed C8 resident cache 建立 opaque-byte DRAM arena。"""

        layer_id = int(layer_id)
        if layer_id in self._layer_arenas:
            raise RuntimeError(f"DSA DRAM layer {layer_id} was initialized twice")
        if (
            resident_packed_cache.ndim != 4
            or resident_packed_cache.shape[0] <= 0
            or tuple(resident_packed_cache.shape[1:]) != (128, 1, DSA_A5_PACKED_KV_ROW_BYTES)
        ):
            raise ValueError(
                f"DSA packed resident cache must be [blocks,128,1,656], got {tuple(resident_packed_cache.shape)}"
            )
        if resident_packed_cache.dtype not in {
            torch.int8,
            torch.float8_e4m3fn,
        }:
            raise ValueError(
                f"DSA packed resident cache must use int8/float8 byte storage, got {resident_packed_cache.dtype}"
            )
        if not resident_packed_cache.is_contiguous():
            raise ValueError("DSA packed resident cache must be contiguous")
        packed = self._allocate_payload_arena(
            block_shape=tuple(resident_packed_cache.shape[1:]),
            dtype=torch.int8,
            payload_name=f"resident_packed[layer={layer_id}]",
        )
        expected_shape = (
            self.arena_capacity,
            *resident_packed_cache.shape[1:],
        )
        if packed.dtype != torch.int8 or tuple(packed.shape) != expected_shape or not packed.is_contiguous():
            raise RuntimeError(
                "DSA packed DRAM arena factory returned an invalid tensor: "
                f"shape={tuple(packed.shape)}, dtype={packed.dtype}, "
                f"contiguous={packed.is_contiguous()}"
            )
        packed[DSA_DRAM_NULL_BLOCK_ID].zero_()
        self._layer_arenas[layer_id] = DSALayerDRAMArenas(
            packed=packed,
            resident_source_bytes=(
                cache_block_byte_view(
                    resident_packed_cache,
                    name=f"resident_packed_cache[layer={layer_id}]",
                ),
            ),
            resident_dram_bytes=(
                cache_block_byte_view(
                    packed,
                    name=f"resident_packed_dram[layer={layer_id}]",
                ),
            ),
            **self._create_indexer_byte_arenas(
                layer_id=layer_id,
                indexer_planes=indexer_cache or (),
            ),
        )
        self._registered_bytes_per_logical_block += (
            int(resident_packed_cache[0].numel())
            * resident_packed_cache.element_size()
        )

    def _create_indexer_byte_arenas(
        self,
        *,
        layer_id: int,
        indexer_planes: tuple[torch.Tensor, ...],
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        """为完整 Indexer plane 建立固定地址 byte view 与 swapped arena。"""

        source_views: list[torch.Tensor] = []
        dram_views: list[torch.Tensor] = []
        for plane_index, plane in enumerate(indexer_planes):
            source = cache_block_byte_view(
                plane,
                name=(
                    f"indexer_cache[layer={layer_id},plane={plane_index}]"
                ),
            )
            dram = self._allocate_payload_arena(
                block_shape=(int(source.shape[1]),),
                dtype=torch.int8,
                payload_name=(
                    f"indexer[layer={layer_id},plane={plane_index}]"
                ),
            )
            expected_shape = (
                self.arena_capacity,
                int(source.shape[1]),
            )
            if (
                dram.dtype != torch.int8
                or tuple(dram.shape) != expected_shape
                or not dram.is_contiguous()
            ):
                raise RuntimeError(
                    "DSA Indexer DRAM arena factory returned an invalid "
                    f"tensor: layer={layer_id}, plane={plane_index}, "
                    f"shape={tuple(dram.shape)}, dtype={dram.dtype}, "
                    f"contiguous={dram.is_contiguous()}"
                )
            dram[DSA_DRAM_NULL_BLOCK_ID].zero_()
            source_views.append(source)
            dram_views.append(dram)
            self._registered_bytes_per_logical_block += int(
                source.shape[1]
            )
        return {
            "indexer_source_bytes": tuple(source_views),
            "indexer_dram_bytes": tuple(dram_views),
        }

    def add_mtp_layer(
        self,
        *,
        layer_id: int,
        mtp_full_cache: tuple[torch.Tensor, ...],
    ) -> None:
        """为原生 BF16 MTP full-cache 建立 opaque-byte DRAM planes。"""

        layer_id = int(layer_id)
        if layer_id in self._mtp_layer_arenas:
            raise RuntimeError(
                f"DSA MTP DRAM layer {layer_id} was initialized twice"
            )
        if len(mtp_full_cache) != 3 or any(
            plane.dtype != torch.bfloat16 for plane in mtp_full_cache
        ):
            raise ValueError(
                "DSA MTP full-cache must contain three native BF16 planes"
            )
        source_views: list[torch.Tensor] = []
        dram_views: list[torch.Tensor] = []
        for plane_index, plane in enumerate(mtp_full_cache):
            if plane.shape[0] <= 0:
                raise ValueError("DSA MTP cache must contain physical blocks")
            source = cache_block_byte_view(
                plane,
                name=f"mtp_cache[layer={layer_id},plane={plane_index}]",
            )
            dram = self._allocate_payload_arena(
                block_shape=(int(source.shape[1]),),
                dtype=torch.int8,
                payload_name=f"mtp[layer={layer_id},plane={plane_index}]",
            )
            expected_shape = (self.arena_capacity, int(source.shape[1]))
            if (
                dram.dtype != torch.int8
                or tuple(dram.shape) != expected_shape
                or not dram.is_contiguous()
            ):
                raise RuntimeError(
                    "DSA MTP DRAM arena factory returned an invalid tensor: "
                    f"layer={layer_id}, plane={plane_index}, "
                    f"shape={tuple(dram.shape)}, dtype={dram.dtype}, "
                    f"contiguous={dram.is_contiguous()}"
                )
            dram[DSA_DRAM_NULL_BLOCK_ID].zero_()
            source_views.append(source)
            dram_views.append(dram)
            self._registered_bytes_per_logical_block += int(source.shape[1])
        self._mtp_layer_arenas[layer_id] = DSAMTPLayerDRAMArenas(
            source_bytes=tuple(source_views),
            dram_bytes=tuple(dram_views),
        )

    def finalize_payload_layout(self) -> None:
        """在 worker 启动期确认实际 arena 与 scheduler 字节模型完全一致。"""

        expected = self.expected_bytes_per_logical_block
        if expected is None:
            return
        if self._registered_bytes_per_logical_block != expected:
            raise RuntimeError(
                "DSA worker complete DRAM block bytes disagree with the "
                "scheduler capacity plan: "
                f"registered={self._registered_bytes_per_logical_block}, "
                f"expected={expected}"
            )

    def get_layer_arenas(self, layer_id: int) -> DSALayerDRAMArenas:
        try:
            return self._layer_arenas[int(layer_id)]
        except KeyError as exc:
            raise RuntimeError(f"DSA DRAM arenas are not initialized for layer {layer_id}") from exc

    def get_mtp_layer_arenas(self, layer_id: int) -> DSAMTPLayerDRAMArenas:
        try:
            return self._mtp_layer_arenas[int(layer_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"DSA MTP DRAM arenas are not initialized for layer {layer_id}"
            ) from exc

    @property
    def l3_components(self) -> tuple[DSAL3Component, ...]:
        components = [DSAL3Component.INDEXER, DSAL3Component.MLA]
        if self._mtp_layer_arenas:
            components.append(DSAL3Component.MTP)
        return tuple(components)

    @property
    def mtp_layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._mtp_layer_arenas))

    def _l3_component_planes(
        self,
        component: DSAL3Component | str,
    ) -> tuple[torch.Tensor, ...]:
        """按稳定 layer/plane 顺序返回一个 L3 component 的 byte views。"""

        component = DSAL3Component(component)
        planes: list[torch.Tensor] = []
        if component is DSAL3Component.MTP:
            for layer_id in sorted(self._mtp_layer_arenas):
                planes.extend(self._mtp_layer_arenas[layer_id].dram_bytes)
        else:
            for layer_id in sorted(self._layer_arenas):
                arenas = self._layer_arenas[layer_id]
                if component is DSAL3Component.INDEXER:
                    planes.extend(arenas.indexer_dram_bytes)
                else:
                    planes.extend(arenas.resident_dram_bytes)
        if not planes:
            raise RuntimeError(
                f"DSA L3 component {component.value!r} has no DRAM planes"
            )
        for plane in planes:
            if (
                plane.dtype != torch.int8
                or plane.ndim != 2
                or int(plane.shape[0]) != self.arena_capacity
                or not plane.is_contiguous()
            ):
                raise RuntimeError(
                    "DSA L3 requires contiguous [arena, bytes] int8 views, "
                    f"got shape={tuple(plane.shape)}, dtype={plane.dtype}, "
                    f"contiguous={plane.is_contiguous()}"
                )
        return tuple(planes)

    def _l3_hbm_component_planes(
        self,
        component: DSAL3Component | str,
    ) -> tuple[torch.Tensor, ...]:
        """按 L3 schema 顺序返回 split HBM component byte views。"""

        component = DSAL3Component(component)
        planes: list[torch.Tensor] = []
        if component is DSAL3Component.MTP:
            for layer_id in sorted(self._mtp_layer_arenas):
                planes.extend(self._mtp_layer_arenas[layer_id].source_bytes)
        else:
            for layer_id in sorted(self._layer_arenas):
                arenas = self._layer_arenas[layer_id]
                if component is DSAL3Component.INDEXER:
                    planes.extend(arenas.indexer_source_bytes)
                else:
                    planes.extend(arenas.resident_source_bytes)
        if not planes:
            raise RuntimeError(
                f"DSA L3 component {component.value!r} has no HBM planes"
            )
        for plane in planes:
            if (
                plane.dtype != torch.int8
                or plane.ndim != 2
                or int(plane.shape[0]) <= 1
                or not plane.is_contiguous()
            ):
                raise RuntimeError(
                    "DSA L3 direct-HBM requires contiguous [blocks, bytes] "
                    f"int8 views, got shape={tuple(plane.shape)}, "
                    f"dtype={plane.dtype}, contiguous={plane.is_contiguous()}"
                )
        return tuple(planes)

    def get_l3_registration_regions(self) -> tuple[tuple[int, int], ...]:
        """返回初始化期需向 L3 backend 增量注册的完整 arena 区间。"""

        regions: list[tuple[int, int]] = []
        seen_ptrs: set[int] = set()
        for component in self.l3_components:
            for plane in self._l3_component_planes(component):
                pointer = int(plane.data_ptr())
                size = int(plane.numel()) * plane.element_size()
                if pointer in seen_ptrs:
                    continue
                seen_ptrs.add(pointer)
                regions.append((pointer, size))
        return tuple(regions)

    def get_l3_hbm_registration_regions(
        self,
    ) -> tuple[tuple[int, int], ...]:
        """返回 direct-HBM load 需要注册的底层 storage 区间。"""

        region_sizes: dict[int, int] = {}
        for component in self.l3_components:
            for plane in self._l3_hbm_component_planes(component):
                storage = plane.untyped_storage()
                pointer = int(storage.data_ptr())
                size = int(storage.nbytes())
                if pointer <= 0 or size <= 0:
                    raise RuntimeError(
                        "DSA HBM cache has an invalid storage region: "
                        f"ptr={pointer}, bytes={size}"
                    )
                region_sizes[pointer] = max(
                    region_sizes.get(pointer, 0),
                    size,
                )
        return tuple(region_sizes.items())

    def get_l3_component_bytes_per_block(
        self,
        component: DSAL3Component | str,
    ) -> int:
        return sum(
            int(plane.shape[1]) * plane.element_size()
            for plane in self._l3_component_planes(component)
        )

    def get_l3_component_plane_count(
        self,
        component: DSAL3Component | str,
    ) -> int:
        return len(self._l3_component_planes(component))

    def build_l3_component_payload(
        self,
        *,
        component: DSAL3Component | str,
        physical_block_ids: tuple[int, ...],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """把 L2 block IDs 映射为 Mooncake multi-buffer 地址/长度列。"""

        if not physical_block_ids:
            return [], []
        block_ids = np.asarray(physical_block_ids, dtype=np.int64)
        if (
            block_ids.ndim != 1
            or int(block_ids.min()) <= DSA_DRAM_NULL_BLOCK_ID
            or int(block_ids.max()) > self.usable_blocks
        ):
            raise ValueError(
                "DSA L3 payload contains an invalid L2 block ID: "
                f"ids={physical_block_ids}, capacity={self.usable_blocks}"
            )
        planes = self._l3_component_planes(component)
        plane_sizes = [
            int(plane.shape[1]) * plane.element_size()
            for plane in planes
        ]
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        for block_id in block_ids.tolist():
            addresses.append(
                [
                    int(plane.data_ptr())
                    + int(block_id) * int(plane.stride(0)) * plane.element_size()
                    for plane in planes
                ]
            )
            sizes.append(list(plane_sizes))
        return addresses, sizes

    def build_l3_hbm_component_payload(
        self,
        *,
        component: DSAL3Component | str,
        physical_block_ids: tuple[int, ...],
    ) -> tuple[list[list[int]], list[list[int]]]:
        """把 split HBM block IDs 映射为 Mooncake multi-buffer 列。"""

        if not physical_block_ids:
            return [], []
        block_ids = np.asarray(physical_block_ids, dtype=np.int64)
        planes = self._l3_hbm_component_planes(component)
        capacity = min(int(plane.shape[0]) for plane in planes)
        if (
            block_ids.ndim != 1
            or int(block_ids.min()) <= 0
            or int(block_ids.max()) >= capacity
        ):
            raise ValueError(
                "DSA L3 direct-HBM payload contains an invalid block ID: "
                f"component={DSAL3Component(component).value}, "
                f"ids={physical_block_ids}, capacity={capacity}"
            )
        plane_sizes = [
            int(plane.shape[1]) * plane.element_size()
            for plane in planes
        ]
        addresses: list[list[int]] = []
        sizes: list[list[int]] = []
        for block_id in block_ids.tolist():
            addresses.append(
                [
                    int(plane.data_ptr())
                    + int(block_id)
                    * int(plane.stride(0))
                    * plane.element_size()
                    for plane in planes
                ]
            )
            sizes.append(list(plane_sizes))
        return addresses, sizes

    def apply_assignment(
        self,
        *,
        pool_index: int,
        logical_block_start: int,
        physical_block_ids: tuple[int, ...],
    ) -> None:
        """把 scheduler 下发的连续 assignment 原址写入 worker 镜像。"""

        pool_index = int(pool_index)
        logical_block_start = int(logical_block_start)
        physical = np.asarray(physical_block_ids, dtype=np.int32)
        if not 0 <= pool_index < self.storage_rows:
            raise IndexError("DSA DRAM pool row is outside table capacity")
        logical_block_end = logical_block_start + int(physical.size)
        if (
            logical_block_start < 0
            or logical_block_end > self.max_logical_blocks
        ):
            raise IndexError(
                "DSA scheduler assignment exceeds worker logical width: "
                f"start={logical_block_start}, end={logical_block_end}, "
                f"capacity={self.max_logical_blocks}"
            )
        if physical.size == 0:
            return
        if int(physical.min()) <= 0 or int(physical.max()) > self.usable_blocks:
            raise IndexError(
                "DSA scheduler assignment contains an invalid physical "
                f"block id: min={int(physical.min())}, "
                f"max={int(physical.max())}, capacity={self.usable_blocks}"
            )
        if np.unique(physical).size != physical.size:
            raise RuntimeError("DSA scheduler assignment contains duplicate block IDs")

        destination = self.logical_block_table[
            pool_index,
            logical_block_start:logical_block_end,
        ]
        if np.array_equal(destination, physical):
            return
        if np.any(destination != DSA_DRAM_NULL_BLOCK_ID):
            raise RuntimeError(
                "DSA worker received a conflicting DRAM assignment for "
                f"pool row {pool_index}"
            )
        destination[:] = physical
        self._block_active_refcounts[physical] += 1
        self.table_version += 1

    def resolve_blocks(
        self,
        *,
        pool_indices: np.ndarray,
        logical_block_indices: np.ndarray,
    ) -> np.ndarray:
        """解析 scheduler 已分配的映射；worker 绝不在这里补分配。"""

        pool_indices = np.asarray(pool_indices, dtype=np.intp)
        logical_block_indices = np.asarray(
            logical_block_indices,
            dtype=np.intp,
        )
        if (
            pool_indices.ndim != 1
            or logical_block_indices.ndim != 1
            or pool_indices.shape != logical_block_indices.shape
        ):
            raise ValueError(
                "DSA DRAM resolution requires matching one-dimensional rows"
            )
        if pool_indices.size == 0:
            return np.empty(0, dtype=np.int32)
        if int(pool_indices.min()) < 0 or int(pool_indices.max()) >= self.storage_rows:
            raise IndexError("DSA DRAM pool row is outside table capacity")
        if int(logical_block_indices.min()) < 0 or int(logical_block_indices.max()) >= self.max_logical_blocks:
            raise IndexError("DSA logical block is outside DRAM table capacity")
        physical = self.logical_block_table[
            pool_indices,
            logical_block_indices,
        ].copy()
        missing = physical == DSA_DRAM_NULL_BLOCK_ID
        if np.any(missing):
            first = int(np.flatnonzero(missing)[0])
            raise RuntimeError(
                "DSA worker dump plan has no scheduler DRAM assignment: "
                f"pool_row={int(pool_indices[first])}, "
                f"logical_block={int(logical_block_indices[first])}"
            )
        return physical

    def release_pool_index(self, pool_index: int) -> None:
        """请求结束时整行回收，无逐 logical-block Python 调用。"""

        pool_index = int(pool_index)
        if not 0 <= pool_index < self.storage_rows:
            raise IndexError(f"DSA DRAM pool row {pool_index} is outside capacity")
        row = self.logical_block_table[pool_index]
        released = row[row != DSA_DRAM_NULL_BLOCK_ID]
        if released.size == 0:
            self.durable_full_block_counts[pool_index] = 0
            return
        released = np.unique(released)
        previous = self._block_active_refcounts[released]
        if np.any(previous <= 0):
            block_id = int(released[np.flatnonzero(previous <= 0)[0]])
            raise RuntimeError(
                "DSA worker DRAM active refcount underflow: "
                f"block_id={block_id}"
            )
        self._block_active_refcounts[released] -= 1
        row.fill(DSA_DRAM_NULL_BLOCK_ID)
        self.durable_full_block_counts[pool_index] = 0
        self.table_version += 1

    def gather_durable_full_block_counts(
        self,
        *,
        pool_indices: np.ndarray,
        output: np.ndarray,
    ) -> None:
        """按 active request 行序读取 durable full-block 水位。"""

        pool_indices = np.asarray(pool_indices, dtype=np.intp)
        if output.shape != pool_indices.shape:
            raise ValueError(
                "DSA durable watermark output shape mismatch: "
                f"output={output.shape}, rows={pool_indices.shape}"
            )
        np.take(
            self.durable_full_block_counts,
            pool_indices,
            out=output,
        )

    def commit_durable_full_block_counts(
        self,
        *,
        pool_indices: np.ndarray,
        full_block_counts: np.ndarray,
    ) -> None:
        """在 dump job 已完整规划后推进请求级 durable 水位。"""

        pool_indices = np.asarray(pool_indices, dtype=np.intp)
        full_block_counts = np.asarray(full_block_counts, dtype=np.int32)
        if pool_indices.shape != full_block_counts.shape:
            raise ValueError(
                "DSA durable watermark update requires matching rows"
            )
        previous = self.durable_full_block_counts[pool_indices]
        if np.any(full_block_counts < previous):
            raise RuntimeError(
                "DSA durable full-block watermark cannot move backwards"
            )
        self.durable_full_block_counts[pool_indices] = full_block_counts

    def gather_rows(
        self,
        *,
        pool_indices: np.ndarray,
        output: np.ndarray,
    ) -> None:
        """将稳定 pool 行批量投影到当前 InputBatch 行序。"""

        pool_indices = np.asarray(pool_indices, dtype=np.intp)
        if output.shape != (
            pool_indices.size,
            self.max_logical_blocks,
        ):
            raise ValueError(
                "DSA active DRAM table output shape mismatch: "
                f"output={output.shape}, rows={pool_indices.size}, "
                f"width={self.max_logical_blocks}"
            )
        np.take(
            self.logical_block_table,
            pool_indices,
            axis=0,
            out=output,
        )
