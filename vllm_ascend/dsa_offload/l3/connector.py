# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""复用 vLLM Connector 生命周期的 DSA L2/L3 双向入口。"""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.forward_context import ForwardContext
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from vllm_ascend.dsa_offload.l3.backend_adapter import DSAL3BackendAdapter
from vllm_ascend.dsa_offload.l3.metadata import (
    DSAL3ConnectorMetadata,
    DSAL3WorkerMetadata,
)
from vllm_ascend.dsa_offload.l3.scheduler import DSAL3StoreScheduler
from vllm_ascend.dsa_offload.l3.schema import DSAL3KeySchema
from vllm_ascend.dsa_offload.l3.worker import DSAL3StoreWorker
from vllm_ascend.dsa_offload.worker_dram_store import DSAWorkerDRAMStore


class DSAL3Connector(KVConnectorBase_V1, SupportsHMA):
    """DSA rank0 write-behind 与同步 all-rank prefix load。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        schema = DSAL3KeySchema.from_config(vllm_config, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        kv_role = str(self._kv_transfer_config.kv_role)
        self.enable_store = kv_role in ("kv_producer", "kv_both")
        self.enable_load = kv_role in ("kv_consumer", "kv_both")
        if not self.enable_store and not self.enable_load:
            raise ValueError(f"Unsupported DSA L3 kv_role={kv_role!r}")
        backend_name = str(extra.get("backend", "mooncake"))
        if backend_name != "mooncake":
            raise ValueError(f"DSA L3 first version supports backend='mooncake' only, got {backend_name!r}")
        max_inflight = int(extra.get("max_inflight_l3_store_jobs", 8))
        max_active_channels = int(extra.get("max_active_channels", 8))
        max_descriptors_per_put = int(
            extra.get("max_descriptors_per_l3_put", 128)
        )
        self.scheduler_handler: DSAL3StoreScheduler | None = None
        self.worker_handler: DSAL3StoreWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            logger.info(
                "DSA prefix tiers: target_hbm=disabled local_l2=%s l3=%s",
                bool(vllm_config.cache_config.enable_prefix_caching),
                self.enable_load,
            )
            lookup_backend = (
                DSAL3BackendAdapter.create_mooncake(
                    vllm_config=vllm_config,
                    schema=schema,
                    max_active_channels=max_active_channels,
                    max_descriptors_per_put=(
                        max_descriptors_per_put
                    ),
                    contribute_memory=False,
                )
                if self.enable_load
                else None
            )
            self.scheduler_handler = DSAL3StoreScheduler(
                schema,
                max_inflight_jobs=max_inflight,
                lookup_backend=lookup_backend,
                enable_load=self.enable_load,
            )
        elif role == KVConnectorRole.WORKER:
            tp_rank = get_tensor_model_parallel_rank()
            # 对齐 AscendStore 的 embedded-client 范式：每个 TP worker
            # 都挂载自己的 Mooncake segment，组成全局池。Indexer/MLA
            # payload 是 TP 副本，因此仍只允许 TP0 发起一次 put。
            backend = DSAL3BackendAdapter.create_mooncake(
                vllm_config=vllm_config,
                schema=schema,
                max_active_channels=max_active_channels,
                max_descriptors_per_put=max_descriptors_per_put,
            )
            logger.info(
                "DSA L3 worker backend active: tp_rank=%d "
                "contributes_segment=True store_producer=%s "
                "load_consumer=%s",
                tp_rank,
                self.enable_store and tp_rank == 0,
                self.enable_load,
            )
            self.worker_handler = DSAL3StoreWorker(
                tp_rank=tp_rank,
                device=torch.device(f"npu:{torch.npu.current_device()}"),
                backend=backend,
                max_inflight_jobs=max_inflight,
                enable_load=self.enable_load,
            )

    def bind_dsa_coordinator(self, coordinator: Any) -> None:
        if self.scheduler_handler is None:
            raise RuntimeError("DSA coordinator can only bind on scheduler")
        self.scheduler_handler.bind_coordinator(coordinator)

    def bind_dsa_dram_store(self, store: DSAWorkerDRAMStore) -> None:
        if self.worker_handler is None:
            raise RuntimeError("DSA L3 DRAM store can only bind on a worker")
        self.worker_handler.bind_store(store)

    def handle_preemptions(self, kv_connector_metadata: Any) -> None:
        if not isinstance(kv_connector_metadata, DSAL3ConnectorMetadata):
            raise TypeError("DSA L3 worker received invalid connector metadata")
        assert self.worker_handler is not None
        self.worker_handler.prepare_step(kv_connector_metadata)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del forward_context, kwargs
        assert self.worker_handler is not None
        self.worker_handler.start_step()

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        return

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str], set[str]]:
        del finished_req_ids
        assert self.worker_handler is not None
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, DSAL3ConnectorMetadata):
            raise TypeError("DSA L3 worker metadata is not bound")
        self.worker_handler.finish_step(metadata)
        return set(), set()

    def build_connector_worker_meta(self) -> DSAL3WorkerMetadata | None:
        assert self.worker_handler is not None
        return self.worker_handler.build_worker_meta()

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self.scheduler_handler is not None
        external_tokens = self.scheduler_handler.lookup_external_prefix(
            request,
            num_computed_tokens,
        )
        return external_tokens, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        del blocks
        assert self.scheduler_handler is not None
        self.scheduler_handler.stage_external_load(
            request.request_id,
            num_external_tokens,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> DSAL3ConnectorMetadata:
        del scheduler_output
        assert self.scheduler_handler is not None
        return self.scheduler_handler.build_connector_meta()

    def extend_dsa_store_jobs(
        self,
        *,
        scheduler_output: SchedulerOutput,
        projection: Any,
        coordinator: Any,
        requests: dict[str, Request],
    ) -> None:
        assert self.scheduler_handler is not None
        metadata = scheduler_output.kv_connector_metadata
        if not isinstance(metadata, DSAL3ConnectorMetadata):
            raise TypeError("DSA scheduler output is missing DSAL3ConnectorMetadata")
        self.scheduler_handler.extend_store_jobs(
            metadata=metadata,
            projection=projection,
            coordinator=coordinator,
            requests=requests,
        )

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        assert self.scheduler_handler is not None
        self.scheduler_handler.update_connector_output(connector_output)

    def has_pending_transfers(self) -> bool:
        """Scheduler-side progress hint for asynchronous write-behind."""

        if self.scheduler_handler is None:
            return False
        return self.scheduler_handler.has_pending_jobs()

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        assert self.scheduler_handler is not None
        self.scheduler_handler.request_finished(request.request_id)
        return False, None

    def request_finished_all_groups(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        assert self.scheduler_handler is not None
        self.scheduler_handler.request_finished(request.request_id)
        return False, None

    def get_finished_count(self) -> int | None:
        return 1

    def reset_cache(self) -> bool:
        if self.scheduler_handler is None:
            # reset_cache is scheduler-owned; worker payload is overwritten by
            # subsequent assignments and does not own the L3 key namespace.
            return True
        return self.scheduler_handler.reset_cache()

    def shutdown(self) -> None:
        if self.worker_handler is not None:
            self.worker_handler.shutdown()
        if self.scheduler_handler is not None:
            self.scheduler_handler.shutdown()
