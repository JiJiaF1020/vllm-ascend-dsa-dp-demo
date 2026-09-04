# SPDX-License-Identifier: Apache-2.0
"""EngineCore中的低频DP缓存管理参与者，与API侧dp_control配对使用。

使用已有DP TCPStore的compare_set在多个API前端之间持有独占管理租约；
复用基线pause(wait)/resume，让运行请求与connector任务排空后再prepare/commit。
只读预检检查本域引用与IO，外部reset使用前端提供的统一UUID epoch。失败保持
租约和fatal标记，禁止恢复；不是支持进程故障恢复的持久化分布式事务。

不创建线程/通信组，不进入每轮forward。新请求在正常管理期间由基线排队，
fatal后由EngineCore入口拒绝；旧L3对象不删除，仅在本部署逻辑失效。
"""

from __future__ import annotations

from typing import Any

from vllm.logger import logger
from vllm.v1.core.sched.interface import PauseState

_OWNER_KEY = "dsa/cache_admin/owner"


class DSADPCacheAdmin:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.fatal_error: str | None = None
        self._prepared: tuple[str, bool, str | None] | None = None
        self._committed: str | None = None

    def _owner(self) -> str:
        store = self.engine.dp_store
        return store.get(_OWNER_KEY).decode() if store.check([_OWNER_KEY]) else ""

    def _require_owner(self, operation_id: str) -> None:
        if (
            not isinstance(operation_id, str)
            or len(operation_id) != 32
            or any(ch not in "0123456789abcdef" for ch in operation_id)
        ):
            raise ValueError("DSA admin operation ID must be a 32-character lowercase hex token")
        if self._owner() != operation_id:
            raise RuntimeError("DSA admin operation does not own the deployment lease")

    def _connector(self):
        scheduler = self.engine.scheduler
        return scheduler.connector if scheduler.dsa_l3_enabled else None

    def status(self) -> dict[str, Any]:
        engine = self.engine
        connector = self._connector()
        handler = connector.scheduler_handler if connector is not None else None
        return {
            "dp_rank": int(engine.dp_rank),
            "dp_size": int(engine.dp_size),
            "pause_state": engine.scheduler.pause_state.name,
            "busy": bool(engine.has_work()),
            "pending_pause": bool(engine.pending_pause),
            "running": len(engine.scheduler.running),
            "waiting": len(engine.scheduler.waiting) + len(engine.scheduler.skipped_waiting),
            "l3_pending": bool(handler.has_pending_jobs()) if handler is not None else False,
            "base_fingerprint": handler._base_schema.fingerprint if handler is not None else None,
            "fingerprint": handler.schema.fingerprint if handler is not None else None,
            "committed": self._committed,
            "fatal": self.fatal_error,
            "shutting_down": not engine.is_running(),
        }

    def _require_quiescent(self, *, check_cache: bool = True) -> dict[str, Any]:
        status = self.status()
        if status["pause_state"] == PauseState.UNPAUSED.name or status["busy"] or status["pending_pause"]:
            raise RuntimeError(f"DSA cache admin requires paused/drained EngineCore: {status}")
        if check_cache and not self.engine.scheduler.can_reset_dsa_prefix_cache():
            raise RuntimeError(f"DSA cache admin has referenced blocks or pending IO: {status}")
        return status

    def execute(self, action: str, operation_id: str = "", reset_external: bool = False):
        if type(reset_external) is not bool:
            raise ValueError("DSA admin reset_external must be boolean")
        if action == "status":
            return self.status()
        if action not in ("fail", "release") and not self.engine.is_running():
            raise RuntimeError("DSA cache management is unavailable after EngineCore shutdown has started")
        if action == "acquire":
            if self.fatal_error:
                raise RuntimeError(f"DSA admin is fail-closed; restart the deployment: {self.fatal_error}")
            if (
                not isinstance(operation_id, str)
                or len(operation_id) != 32
                or any(ch not in "0123456789abcdef" for ch in operation_id)
            ):
                raise ValueError("invalid DSA admin operation ID")
            owner = self.engine.dp_store.compare_set(_OWNER_KEY, "", operation_id).decode()
            if owner != operation_id:
                raise RuntimeError(
                    f"DSA admin is busy or fail-closed (owner={owner}); do not run concurrent management"
                )
            return True
        self._require_owner(operation_id)
        if action == "fail":
            self.fatal_error = f"operation={operation_id}; uncertain management outcome; restart all DP engines"
            if self.engine.scheduler.pause_state == PauseState.UNPAUSED:
                self.engine.scheduler.set_pause_state(PauseState.PAUSED_NEW)
            self.engine.ignore_start_dp_wave = True
            logger.error("DSA DP cache admin fail-closed: dp=%d %s", self.engine.dp_rank, self.fatal_error)
            return True
        if self.fatal_error:
            raise RuntimeError(f"DSA admin is fail-closed; restart the deployment: {self.fatal_error}")
        if action == "release":
            if self.engine.dp_store.compare_set(_OWNER_KEY, operation_id, "").decode() != "":
                raise RuntimeError("DSA admin lease release lost ownership")
            return True
        if action == "pause":
            status = self.status()
            logger.info(
                "DSA DP cache admin pause: dp=%d operation=%s running=%d waiting=%d l3_pending=%s",
                self.engine.dp_rank,
                operation_id,
                status["running"],
                status["waiting"],
                status["l3_pending"],
            )
            return self.engine.pause_scheduler(mode="wait", clear_cache=False)
        if action == "prepare":
            status = self._require_quiescent()
            self._prepared = (operation_id, reset_external, status["fingerprint"])
            logger.info(
                "DSA DP cache admin prepared: dp=%d operation=%s waiting=%d",
                self.engine.dp_rank, operation_id, status["waiting"],
            )
            return status
        if action == "commit":
            status = self._require_quiescent()
            if self._committed == operation_id:
                return status
            if self._prepared != (operation_id, reset_external, status["fingerprint"]):
                raise RuntimeError("DSA cache admin commit has no matching preparation")
            if not self.engine.scheduler.commit_dsa_prefix_cache_reset():
                raise RuntimeError("DSA local prefix reset refused during commit")
            connector = self._connector()
            if reset_external and connector is not None and not connector.reset_cache_for_dp(operation_id):
                raise RuntimeError("DSA external namespace commit refused")
            self._committed = operation_id
            status = self.status()
            logger.info(
                "DSA DP cache admin committed: dp=%d operation=%s external=%s fingerprint=%s",
                self.engine.dp_rank,
                operation_id,
                reset_external,
                status["fingerprint"],
            )
            return status
        if action == "resume":
            self._require_quiescent(check_cache=False)
            logger.info("DSA DP cache admin resume: dp=%d operation=%s", self.engine.dp_rank, operation_id)
            return self.engine.resume_scheduler()
        raise ValueError(f"unknown DSA cache admin action: {action}")
