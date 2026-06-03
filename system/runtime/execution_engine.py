from __future__ import annotations

from collections.abc import Callable
from typing import Any

from system.agent.task import RuntimeTask

from .state_manager import RuntimeStateManager


# 保持你原有的执行器类型定义，如果你喜欢更通用也可以保留
RuntimeExecutor = Callable[[Any], Any]


class RuntimeExecutionEngine:
    """管理任务类型与执行器的映射，并驱动状态更新。"""

    def __init__(self, *, state_manager: RuntimeStateManager | None = None) -> None:
        self.state_manager = state_manager or RuntimeStateManager()
        self._executors: dict[str, RuntimeExecutor] = {}

    @staticmethod
    def _normalize_task_type(task_type: str | None) -> str:
        """规范化任务类型字符串，并拒绝空值。"""
        key = (task_type or "").strip().lower()
        if not key:
            raise ValueError("task_type must be a non-empty string")
        return key

    def register(self, task_type: str, executor: RuntimeExecutor) -> None:
        """注册一个任务类型对应的执行器。"""
        key = self._normalize_task_type(task_type)
        self._executors[key] = executor

    def execute(self, task: RuntimeTask, args: Any) -> Any:
        """
        执行指定任务并自动更新状态。
        流程：开始 → 成功，或 开始 → 失败。
        """
        key = self._normalize_task_type(task.task_type)
        executor = self._executors.get(key)
        if executor is None:
            raise KeyError(
                f"No executor for '{key}'. Available types: {list(self._executors)}"
            )

        self.state_manager.start(task)
        try:
            result = executor(args)
        except Exception as exc:
            self.state_manager.fail(task, exc)
            raise
        else:
            self.state_manager.succeed(task, result)
            return result
