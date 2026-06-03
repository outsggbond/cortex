from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from system.core.planner import Plan, Task, TaskPlanner


@dataclass
class PlannerRequest:
    goal: str
    intent: str = ""
    memories: List[str] = field(default_factory=list)
    feedback: str = ""
    max_candidates: int = 3
    previous_plan: Plan | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedPlanner:
    def __init__(self, planner: TaskPlanner | None = None) -> None:
        self._planner = planner or TaskPlanner()

    @property
    def raw_planner(self) -> TaskPlanner:
        return self._planner

    def plan(
        self,
        goal: str,
        model=None,
        memories: List[str] | None = None,
        feedback: str = "",
        max_candidates: int = 3,
        previous_plan: Plan | None = None,
    ) -> Plan:
        request = PlannerRequest(
            goal=str(goal or ""),
            memories=list(memories or []),
            feedback=str(feedback or ""),
            max_candidates=max_candidates,
            previous_plan=previous_plan,
        )
        return self.build_plan(request, model=model)

    def build_plan(self, request: PlannerRequest, *, model=None) -> Plan:
        return self._planner.plan(
            str(request.goal or ""),
            model=model,
            memories=list(request.memories or []),
            feedback=str(request.feedback or ""),
            max_candidates=max(1, int(request.max_candidates or 1)),
            previous_plan=request.previous_plan,
        )

    def parse_task(self, text: str) -> Task:
        return self._planner._parse_task(str(text or ""))

    def split_plan(self, text: str) -> Plan:
        return self._planner._plan_by_split(str(text or ""))

    def __getattr__(self, name: str):
        # Compatibility bridge while internal callers are migrated off TaskPlanner internals.
        return getattr(self._planner, name)
