from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from system.automation.executor import ExecutionResult, TaskExecutor
from system.core.planner import Plan


@dataclass
class ExecutorRequest:
    plan: Plan
    task_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedExecutor:
    def __init__(
        self,
        *,
        project_root: str = ".",
        executor: TaskExecutor | None = None,
    ) -> None:
        self._executor = executor or TaskExecutor(project_root=project_root)

    @property
    def raw_executor(self) -> TaskExecutor:
        return self._executor

    def execute(self, request: ExecutorRequest) -> ExecutionResult:
        return self._executor.run(request.plan)

    def run(
        self,
        plan: Plan,
        *,
        task_id: str = "",
        metadata: Dict[str, Any] | None = None,
    ) -> ExecutionResult:
        return self.execute(
            ExecutorRequest(
                plan=plan,
                task_id=str(task_id or ""),
                metadata=dict(metadata or {}),
            )
        )
