from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RuntimeTask:
    task_id: str
    task_type: str
    intent: str
    goal: str
    entrypoint: str
    source: str = "cli"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        goal = str(self.goal or "").strip()
        if goal:
            return f"{self.task_type}:{goal}"
        return self.task_type
