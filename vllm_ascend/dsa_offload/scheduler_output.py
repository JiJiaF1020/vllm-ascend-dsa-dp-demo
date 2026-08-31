# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA scheduler→worker 的最小类型化投影。

vLLM v0.23 的 ``SchedulerOutput`` 对 cached request 只传新增 block IDs，
无法表达 DSA ENTER step 对 resident MLA block table 的整表替换。这里使用
一个薄子类增加单个 DSA projection 字段：

* 不修改 vLLM；
* 不复制 ``SchedulerOutput`` 的调度或序列化逻辑；
* 所有基线字段仍保持原对象引用，只做一次固定字段数的浅包装；
* request 语义按 ``num_scheduled_tokens`` 的顺序列式传输，避免四份
  request-id→scalar 字典；
* 只有 ENTER 行携带 resident block IDs 全量快照。
* DRAM 目录只传本轮新增的连续 assignment 后缀，不重传完整逻辑表。
* prefix-hit 行额外携带一次性的 Indexer/MLA restore 列；每行 ready 水位
  只供 scheduler 在 worker 成功返回后发布 DRAM hash。

多进程执行器使用 pickle 传递 scheduler output，因此该类型化子类会随普通
``SchedulerOutput`` 一起跨进程恢复。worker 最终仍按自己的 ``InputBatch``
行序重排这些列。
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.dsa_offload.request_cache_layout import (
    DSARequestCacheStage,
)


@dataclass(frozen=True, slots=True)
class DSAResidentBlockTableReplacement:
    """一个 ENTER 请求需要覆盖到 worker 的 resident 全量 block IDs。"""

    request_id: str
    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.block_ids:
            raise ValueError("DSA resident block-table replacement cannot be empty")


@dataclass(frozen=True, slots=True)
class DSADRAMBlockAssignment:
    """scheduler 为一个请求新增的连续逻辑 DRAM block-id 后缀。"""

    request_id: str
    logical_block_start: int
    physical_block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.logical_block_start < 0:
            raise ValueError("DSA DRAM assignment start must be non-negative")
        if not self.physical_block_ids:
            raise ValueError("DSA DRAM assignment cannot be empty")
        if any(block_id <= 0 for block_id in self.physical_block_ids):
            raise ValueError(
                "DSA DRAM assignment block IDs must be greater than zero"
            )


@dataclass(frozen=True, slots=True)
class DSADRAMBlockRestore:
    """一个 prefix-hit 请求的 target DRAM->HBM 分平面恢复任务。

    ``ready_block_count`` 是恢复完成后 L2 的连续 durable 水位，不要求
    覆盖本次 copy range。HBM-only 本地前缀可能在 L2 中形成待回填缺口；
    此时可以先把更靠后的 L3 suffix 从 L2 恢复到 HBM，而 durable 水位
    仍停留在缺口之前，随后由首个 prefill forward 的 dump 补齐。
    """

    request_id: str
    source_dram_block_ids: tuple[int, ...]
    destination_indexer_block_ids: tuple[int, ...]
    destination_resident_block_ids: tuple[int, ...]
    logical_block_start: int = 0
    ready_block_count: int | None = None

    def __post_init__(self) -> None:
        lengths = (
            len(self.source_dram_block_ids),
            len(self.destination_indexer_block_ids),
            len(self.destination_resident_block_ids),
        )
        if len(set(lengths)) != 1:
            raise ValueError(
                "DSA DRAM restore planes must have one matching "
                f"length, got {lengths}"
            )
        if self.logical_block_start < 0:
            raise ValueError("DSA DRAM restore start must be non-negative")
        ready_block_count = (
            self.logical_block_start + lengths[0]
            if self.ready_block_count is None
            else int(self.ready_block_count)
        )
        if ready_block_count < 0:
            raise ValueError(
                "DSA DRAM restore ready watermark must be non-negative: "
                f"ready={ready_block_count}"
            )
        object.__setattr__(
            self,
            "ready_block_count",
            ready_block_count,
        )
        if any(
            block_id <= 0
            for block_ids in (
                self.source_dram_block_ids,
                self.destination_indexer_block_ids,
                self.destination_resident_block_ids,
            )
            for block_id in block_ids
        ):
            raise ValueError("DSA DRAM restore block IDs must be positive")


@dataclass(frozen=True, slots=True)
class DSARequestCacheLayoutProjection:
    """一次 model-forward 中 scheduled request 的 cache 布局列。"""

    request_ids: tuple[str, ...]
    stages: tuple[int, ...]
    target_resident_budget_tokens: tuple[int, ...]
    sparse_budget_tokens: tuple[int, ...]
    resident_valid_tokens: tuple[int, ...]
    resident_block_table_replacements: tuple[
        DSAResidentBlockTableReplacement,
        ...,
    ]
    dram_block_assignments: tuple[DSADRAMBlockAssignment, ...] = ()
    dram_ready_block_counts: tuple[int, ...] = ()
    dram_block_restores: tuple[DSADRAMBlockRestore, ...] = ()

    def __post_init__(self) -> None:
        row_count = len(self.request_ids)
        column_lengths = (
            len(self.stages),
            len(self.target_resident_budget_tokens),
            len(self.sparse_budget_tokens),
            len(self.resident_valid_tokens),
        )
        if any(length != row_count for length in column_lengths):
            raise ValueError(
                "DSA cache-layout projection columns have different lengths: "
                f"request_ids={row_count}, columns={column_lengths}"
            )
        if self.dram_ready_block_counts and len(
            self.dram_ready_block_counts
        ) != row_count:
            raise ValueError(
                "DSA DRAM ready column has a different length: "
                f"request_ids={row_count}, "
                f"ready={len(self.dram_ready_block_counts)}"
            )
        if any(count < 0 for count in self.dram_ready_block_counts):
            raise ValueError("DSA DRAM ready block counts must be non-negative")
        enter_rows = sum(stage == int(DSARequestCacheStage.ENTER_SPARSE_DECODE) for stage in self.stages)
        if len(self.resident_block_table_replacements) != enter_rows:
            raise ValueError(
                "DSA resident replacement count does not match ENTER rows: "
                f"replacements={len(self.resident_block_table_replacements)}, "
                f"enter_rows={enter_rows}"
            )
        scheduled_request_ids = set(self.request_ids)
        assignment_request_ids: set[str] = set()
        for assignment in self.dram_block_assignments:
            if assignment.request_id not in scheduled_request_ids:
                raise ValueError(
                    "DSA DRAM assignment belongs to an unscheduled request: "
                    f"request_id={assignment.request_id!r}"
                )
            if assignment.request_id in assignment_request_ids:
                raise ValueError(
                    "DSA projection contains duplicate DRAM assignments for "
                    f"request {assignment.request_id!r}"
                )
            assignment_request_ids.add(assignment.request_id)
        restore_request_ids: set[str] = set()
        for restore in self.dram_block_restores:
            if restore.request_id not in scheduled_request_ids:
                raise ValueError(
                    "DSA DRAM restore belongs to an unscheduled request: "
                    f"request_id={restore.request_id!r}"
                )
            if restore.request_id in restore_request_ids:
                raise ValueError(
                    "DSA projection contains duplicate DRAM restores for "
                    f"request {restore.request_id!r}"
                )
            restore_request_ids.add(restore.request_id)

    @property
    def num_rows(self) -> int:
        return len(self.request_ids)

    @property
    def num_enter_rows(self) -> int:
        return len(self.resident_block_table_replacements)


_BASE_SCHEDULER_OUTPUT_FIELD_NAMES = tuple(field.name for field in fields(SchedulerOutput))


@dataclass
class DSAOffloadSchedulerOutput(SchedulerOutput):
    """保持基线字段语义、仅增加 DSA cache-layout projection 的薄子类。"""

    dsa_cache_layout: DSARequestCacheLayoutProjection | None = None

    @classmethod
    def from_base(
        cls,
        output: SchedulerOutput,
        *,
        dsa_cache_layout: DSARequestCacheLayoutProjection,
    ) -> DSAOffloadSchedulerOutput:
        if isinstance(output, cls):
            output.dsa_cache_layout = dsa_cache_layout
            return output
        base_fields = {name: getattr(output, name) for name in _BASE_SCHEDULER_OUTPUT_FIELD_NAMES}
        return cls(
            **base_fields,
            dsa_cache_layout=dsa_cache_layout,
        )
