from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .task import RuntimeTask


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class AgentTaskState:
    task_id: str
    task_type: str
    intent: str
    goal: str
    entrypoint: str
    status: str = TaskStatus.PENDING
    source: str = "cli"
    started_at: float = 0.0
    finished_at: float = 0.0
    last_error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(
        cls,
        task: RuntimeTask,
        *,
        status: str = TaskStatus.PENDING,
    ) -> "AgentTaskState":
        return cls(
            task_id=str(task.task_id or "").strip(),
            task_type=str(task.task_type or "").strip(),
            intent=str(task.intent or "").strip(),
            goal=str(task.goal or "").strip(),
            entrypoint=str(task.entrypoint or "").strip(),
            status=str(status or TaskStatus.PENDING),
            source=str(task.source or "cli").strip() or "cli",
            metadata=dict(task.metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "intent": self.intent,
            "goal": self.goal,
            "entrypoint": self.entrypoint,
            "status": self.status,
            "source": self.source,
            "started_at": float(self.started_at or 0.0),
            "finished_at": float(self.finished_at or 0.0),
            "last_error": self.last_error,
            "metadata": dict(self.metadata or {}),
            "result": dict(self.result or {}),
        }
