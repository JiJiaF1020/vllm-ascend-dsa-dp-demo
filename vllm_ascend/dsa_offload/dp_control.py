# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API前端的单机internal-DP缓存管理协调，不参与推理热路径。

通过已有utility RPC调用dp_admin参与者，使用DP TCPStore租约排斥多API前端
并发管理。reset依次暂停新admission、排空模型/IO、全域prepare、统一epoch
commit、全结果确认及resume；原先已暂停的部署保持暂停。请求在暂停期间排队。

暂停开始后的超时、部分失败等不确定结果采用fail-closed：不释放租约、不自动
resume，要求重启整个部署。HTTP取消不取消已开始的操作。该协议不是进程故障
下可恢复的持久化事务；不做Mooncake物理删除，不开放offline单域external reset。
普通手工pause只开放wait/clear_cache=False；sleep/wake和keep/abort留待独立适配。
MM/encoder缓存不属于DSA prefix层，沿用原生utility；API启动会自动reset_mm_cache。
"""

from __future__ import annotations

import asyncio
import uuid

from vllm.logger import logger

DSA_DP_ADMIN_METHODS = frozenset(
    {
        "reset_prefix_cache",
        "pause_scheduler",
        "resume_scheduler",
        "is_scheduler_paused",
        "sleep",
        "wake_up",
    }
)
_RPC_TIMEOUT_SECONDS = 30.0


async def _all(client, engines, method, *args, timeout=_RPC_TIMEOUT_SECONDS):
    results = await asyncio.wait_for(
        asyncio.gather(
            *(client._call_utility_async(method, *args, engine=engine) for engine in engines),
            return_exceptions=True,
        ),
        timeout=timeout,
    )
    failed = {index: repr(result) for index, result in enumerate(results) if isinstance(result, BaseException)}
    if failed:
        raise RuntimeError(f"DSA DP admin {method} failed: {failed}")
    return results


def _check_statuses(statuses):
    required = {
        "dp_rank",
        "dp_size",
        "fatal",
        "base_fingerprint",
        "fingerprint",
        "pause_state",
        "busy",
        "pending_pause",
    }
    if not statuses or any(not isinstance(status, dict) or not required <= status.keys() for status in statuses):
        raise RuntimeError(f"DSA admin received invalid participant status: {statuses}")
    if {status["dp_rank"] for status in statuses} != set(range(len(statuses))):
        raise RuntimeError(f"DSA admin participant ranks disagree: {statuses}")
    if any(status["dp_size"] != len(statuses) for status in statuses):
        raise RuntimeError("DSA admin must reach every DP engine")
    if any(status["fatal"] for status in statuses):
        raise RuntimeError(f"DSA admin is fail-closed; restart the deployment: {statuses}")
    if any(status.get("shutting_down", False) for status in statuses):
        raise RuntimeError("DSA cache administration cannot run while an EngineCore is shutting down")
    for key in ("base_fingerprint", "fingerprint"):
        if len({status[key] for status in statuses}) != 1:
            raise RuntimeError(f"DSA DP admin {key} disagrees across engines: {statuses}")


async def _transaction(client, method, args):
    if method not in DSA_DP_ADMIN_METHODS:
        raise ValueError(f"Unknown DSA DP management method: {method}")
    engines = tuple(client.core_engines)
    if not engines:
        raise RuntimeError("DSA DP cache admin has no EngineCore participants")
    expected = int(client.vllm_config.parallel_config.data_parallel_size)
    if expected <= 1 or len(engines) != expected:
        raise RuntimeError("DSA DP cache admin requires all deployment participants, not a local subset")
    if method in ("sleep", "wake_up"):
        raise ValueError(f"DSA DP {method} is not supported by prefix cache administration")
    reset_external = False
    if method == "reset_prefix_cache":
        reset_running = args[0] if args else False
        reset_external = args[1] if len(args) > 1 else False
        if len(args) > 2 or type(reset_running) is not bool or type(reset_external) is not bool or reset_running:
            raise ValueError("DSA DP reset requires reset_running_requests=False and boolean reset_connector")
    if method == "pause_scheduler" and args != ("wait", False):
        raise ValueError("DSA DP manual pause requires mode='wait', clear_cache=False; reset caches separately")
    if method in ("resume_scheduler", "is_scheduler_paused") and args:
        raise ValueError(f"DSA DP {method} takes no arguments")
    if method == "is_scheduler_paused":
        statuses = await _all(client, engines, "dsa_cache_admin", "status")
        _check_statuses(statuses)
        states = {status["pause_state"] != "UNPAUSED" for status in statuses}
        if len(states) != 1 or any(status["pending_pause"] for status in statuses):
            raise RuntimeError("DSA DP pause is transitioning; wait for the management request to finish")
        return states.pop()

    operation_id = uuid.uuid4().hex
    owner_engine = engines[0]
    # 唯一仲裁在EngineCore的共享Store，不依赖某个API进程自己的asyncio.Lock。
    acquired = await asyncio.wait_for(
        client._call_utility_async("dsa_cache_admin", "acquire", operation_id, engine=owner_engine),
        timeout=_RPC_TIMEOUT_SECONDS,
    )
    if acquired is not True:
        raise RuntimeError("DSA DP admin lease was not acquired")
    critical = False
    try:
        statuses = await _all(client, engines, "dsa_cache_admin", "status")
        _check_statuses(statuses)
        paused = {status["pause_state"] != "UNPAUSED" for status in statuses}
        if len(paused) != 1 or any(status["pending_pause"] for status in statuses):
            raise RuntimeError("DSA admin requires a uniformly running or fully paused deployment")
        initially_paused = paused.pop()
        if initially_paused and any(status["busy"] for status in statuses):
            raise RuntimeError("DSA admin cannot operate on a partially drained pause")
        if method in ("reset_prefix_cache", "pause_scheduler"):
            if not initially_paused:
                critical = True
                pause_timeout = max(float(getattr(client.vllm_config, "shutdown_timeout", 0) or 300), 30.0)
                results = await _all(
                    client,
                    engines,
                    "dsa_cache_admin",
                    "pause",
                    operation_id,
                    timeout=pause_timeout,
                )
                if any(result is not None for result in results):
                    raise RuntimeError("DSA pause did not acknowledge all engines")
            if method == "reset_prefix_cache":
                prepared = await _all(client, engines, "dsa_cache_admin", "prepare", operation_id, reset_external)
                _check_statuses(prepared)
                critical = True
                committed = await _all(client, engines, "dsa_cache_admin", "commit", operation_id, reset_external)
                _check_statuses(committed)
                if any(status["committed"] != operation_id for status in committed):
                    raise RuntimeError("DSA reset has incomplete commit acknowledgements")
                if reset_external and committed[0]["fingerprint"] is not None:
                    if committed[0]["fingerprint"] == prepared[0]["fingerprint"]:
                        raise RuntimeError("DSA external reset did not rotate namespace")
                elif committed[0]["fingerprint"] != prepared[0]["fingerprint"]:
                    raise RuntimeError("DSA local-only reset unexpectedly changed the namespace")
                if not initially_paused:
                    results = await _all(client, engines, "dsa_cache_admin", "resume", operation_id)
                    if any(result is not None for result in results):
                        raise RuntimeError("DSA resume did not acknowledge all engines")
        elif method == "resume_scheduler":
            if initially_paused:
                critical = True
                results = await _all(client, engines, "dsa_cache_admin", "resume", operation_id)
                if any(result is not None for result in results):
                    raise RuntimeError("DSA resume did not acknowledge all engines")
        released = await asyncio.wait_for(
            client._call_utility_async("dsa_cache_admin", "release", operation_id, engine=owner_engine),
            timeout=_RPC_TIMEOUT_SECONDS,
        )
        if released is not True:
            raise RuntimeError("DSA DP admin lease release was not acknowledged")
        logger.info(
            "DSA DP cache admin finished: operation=%s method=%s external=%s", operation_id, method, reset_external
        )
        return True if method == "reset_prefix_cache" else None
    except BaseException:
        if critical:
            try:
                await _all(client, engines, "dsa_cache_admin", "fail", operation_id)
            except BaseException:
                logger.exception(
                    "DSA admin fail-closed notification incomplete; lease retained, restart all DP engines"
                )
            logger.error(
                "DSA DP cache admin uncertain: operation=%s; do not resume, restart all DP engines", operation_id
            )
        else:
            try:
                await asyncio.wait_for(
                    client._call_utility_async("dsa_cache_admin", "release", operation_id, engine=owner_engine),
                    timeout=_RPC_TIMEOUT_SECONDS,
                )
            except BaseException:
                logger.exception("DSA admin lease cleanup failed before mutation; restart before further management")
        raise


async def call_dsa_dp_admin(client, method: str, *args):
    """调用方取消等待时，让已经开始的有界管理操作继续完成而非丢在半途。"""
    tasks = getattr(client, "_dsa_admin_tasks", None)
    if tasks is None:
        tasks = client._dsa_admin_tasks = set()
    task = asyncio.create_task(_transaction(client, method, args))
    tasks.add(task)
    detached = False

    def report(result):
        tasks.discard(result)
        if not result.cancelled():
            error = result.exception()
            if detached and error is not None:
                logger.error("DSA admin failed after caller cancellation: %s", error)

    task.add_done_callback(report)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        detached = True
        raise
