# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""将 DSA 的 DP 初始化与前端管理操作挂接到 vLLM 基线入口。

EngineCore 侧在原生 ``_init_data_parallel`` 完成后，把已有 CPU DP group
交给 ``dsa_offload.dp_context``，供后续 L2 容量协调使用。API 前端侧仅对
DSA 的缓存管理utility转交 ``dp_control`` 协调全部DP，EngineCore的低频
参与者 ``dp_admin`` 使用已有TCPStore串行化多API前端操作；非DSA沿用基线。

DP L3正常退出在本域模型与IO排空后，通过已有DP Store公布drain；全部域就绪后
经原生collective_rpc提前关闭本域worker L3，再收集全域关闭结果并做最后一次Gloo
barrier，才交回原生进程退出。避免重资源关闭落入death-pipe之后的4秒退出窗口。
零timeout的abort保留原生快速路径；关闭失败显式报告，不修改通用worker退出策略。

本文件只负责入口选择和幂等 patch 注册；容量算法与结果判定留在 DSA
模块中。它不改变请求路由、Scheduler 主循环或每轮模型 collective 次序。
这里依赖 v0.23 的私有方法，升级基线时需要核对签名和调用时机。
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import torch.distributed as dist
from vllm.logger import logger
from vllm.v1.engine.core import DPEngineCoreProc
from vllm.v1.engine.core_client import DPLBAsyncMPClient

from vllm_ascend.dsa_offload import DSA_SPARSE_CONFIG_KEY
from vllm_ascend.dsa_offload.dp_admin import DSADPCacheAdmin
from vllm_ascend.dsa_offload.dp_context import (
    install_dsa_engine_dp_context,
)
from vllm_ascend.dsa_offload.dp_control import DSA_DP_ADMIN_METHODS, call_dsa_dp_admin


def _dsa_enabled(vllm_config) -> bool:
    additional_config = vllm_config.additional_config or {}
    raw_config = additional_config.get(DSA_SPARSE_CONFIG_KEY)
    return isinstance(raw_config, Mapping) and raw_config.get("enabled") is True


if not getattr(
    DPEngineCoreProc._init_data_parallel,
    "_vllm_ascend_dsa_dp_patch",
    False,
):
    _ORIGINAL_INIT_DATA_PARALLEL = DPEngineCoreProc._init_data_parallel

    def _init_data_parallel(self, vllm_config) -> None:
        _ORIGINAL_INIT_DATA_PARALLEL(self, vllm_config)
        if not _dsa_enabled(vllm_config):
            return
        parallel_config = vllm_config.parallel_config
        install_dsa_engine_dp_context(
            process_group=self.dp_group,
            rank=int(parallel_config.data_parallel_rank),
            world_size=int(parallel_config.data_parallel_size),
            local_rank=int(parallel_config.data_parallel_rank_local or 0),
        )
        self._dsa_cache_admin_state = DSADPCacheAdmin(self)

    _init_data_parallel._vllm_ascend_dsa_dp_patch = True
    DPEngineCoreProc._init_data_parallel = _init_data_parallel


if not getattr(DPLBAsyncMPClient.call_utility_async, "_vllm_ascend_dsa_dp_patch", False):
    _ORIGINAL_CALL_UTILITY_ASYNC = DPLBAsyncMPClient.call_utility_async

    async def _call_utility_async(self, method: str, *args):
        if method in DSA_DP_ADMIN_METHODS and _dsa_enabled(self.vllm_config):
            return await call_dsa_dp_admin(self, method, *args)
        return await _ORIGINAL_CALL_UTILITY_ASYNC(self, method, *args)

    _call_utility_async._vllm_ascend_dsa_dp_patch = True
    DPLBAsyncMPClient.call_utility_async = _call_utility_async


def _dsa_cache_admin(self, action: str, operation_id: str = "", reset_external: bool = False):
    admin = getattr(self, "_dsa_cache_admin_state", None)
    if admin is None:
        raise RuntimeError("DSA cache administration requires a DSA DP EngineCore")
    return admin.execute(action, operation_id, reset_external)


DPEngineCoreProc.dsa_cache_admin = _dsa_cache_admin


if not getattr(DPEngineCoreProc.add_request, "_vllm_ascend_dsa_admin_patch", False):
    _ORIGINAL_ADD_REQUEST = DPEngineCoreProc.add_request

    def _add_request(self, request, request_wave: int = 0):
        admin = getattr(self, "_dsa_cache_admin_state", None)
        if admin is not None and admin.fatal_error:
            self._send_error_outputs_to_client([request.request_id], request.client_index)
            return
        return _ORIGINAL_ADD_REQUEST(self, request, request_wave)

    _add_request._vllm_ascend_dsa_admin_patch = True
    DPEngineCoreProc.add_request = _add_request


if not getattr(DPEngineCoreProc._handle_shutdown, "_vllm_ascend_dsa_l3_drain_patch", False):
    _ORIGINAL_HANDLE_SHUTDOWN = DPEngineCoreProc._handle_shutdown

    def _handle_shutdown(self) -> bool:
        """各域本地 drain 后再共同释放 embedded L3 segment，不进入 forward 热路径。"""
        if not getattr(self, "_dsa_l3_local_drained", False):
            keep_stepping = _ORIGINAL_HANDLE_SHUTDOWN(self)
            if keep_stepping:
                return True
            config = self.vllm_config
            kv_transfer = config.kv_transfer_config
            if not _dsa_enabled(config) or getattr(kv_transfer, "kv_connector", None) != "DSAL3Connector":
                return False
            # 该 Store 与已有 stateless EngineCore DP group 同生命周期。
            # 先非阻塞公布 drain，防止与尚在模型 wave 中的同伴错配 collective。
            # 本域已拒绝 admission 且当前 wave 已结束；迟到的唤醒不能重启 dummy。
            self.ignore_start_dp_wave = True
            self.dp_store.set(f"dsa_l3/drained/{self.dp_rank}", "1")
            self._dsa_l3_local_drained = True
            logger.info("DSA DP L3 shutdown local drained: dp=%d/%d", self.dp_rank, self.dp_size)

        keys = [f"dsa_l3/drained/{rank}" for rank in range(self.dp_size)]
        if not self.dp_store.check(keys):
            time.sleep(0.01)
            return True

        if not getattr(self, "_dsa_l3_resource_close_attempted", False):
            from vllm_ascend.dsa_offload.l3.connector import shutdown_dsa_l3_worker

            self._dsa_l3_resource_close_attempted = True
            logger.info(
                "DSA DP L3 shutdown all drained: dp=%d/%d; segment teardown allowed", self.dp_rank, self.dp_size
            )
            timeout = float(self.vllm_config.shutdown_timeout)
            close_status = "abort"
            if timeout > 0:
                started = time.monotonic()
                try:
                    results = self.model_executor.collective_rpc(shutdown_dsa_l3_worker, timeout=timeout)
                    expected = self.vllm_config.parallel_config.tensor_parallel_size
                    if not isinstance(results, (list, tuple)) or len(results) != expected or any(
                        result is not True for result in results
                    ):
                        raise RuntimeError(f"incomplete worker L3 close acknowledgements: {results!r}")
                    close_status = "closed"
                    logger.info(
                        "DSA DP L3 worker resources closed: dp=%d workers=%d elapsed_ms=%.3f",
                        self.dp_rank, expected, (time.monotonic() - started) * 1000,
                    )
                except Exception:
                    close_status = "failed"
                    logger.exception(
                        "DSA DP L3 worker resource close failed: dp=%d; falling back to native termination",
                        self.dp_rank,
                    )
            else:
                logger.info("DSA DP L3 early resource close skipped for abort mode: dp=%d", self.dp_rank)
            self.dp_store.set(f"dsa_l3/resources_closed/{self.dp_rank}", close_status)

        close_keys = [f"dsa_l3/resources_closed/{rank}" for rank in range(self.dp_size)]
        if not self.dp_store.check(close_keys):
            time.sleep(0.01)
            return True
        statuses = [self.dp_store.get(key).decode() for key in close_keys]
        # 唯一的退出barrier后移到关闭回报之后，保护rank0持有的Store生命周期。
        dist.barrier(group=self.dp_group)
        if all(status == "closed" for status in statuses):
            logger.info("DSA DP L3 all worker resources closed: dp=%d; native process teardown", self.dp_rank)
        else:
            logger.warning("DSA DP L3 resource close not fully confirmed: dp=%d statuses=%s", self.dp_rank, statuses)
        return False

    _handle_shutdown._vllm_ascend_dsa_l3_drain_patch = True
    DPEngineCoreProc._handle_shutdown = _handle_shutdown
