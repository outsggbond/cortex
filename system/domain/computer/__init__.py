from __future__ import annotations

from typing import Any, Dict

from system.computer_use.runtime import build_computer_use_runtime, run_computer_use_runtime


class ComputerDomainService:
    def __init__(self, args: Any) -> None:
        self.runtime = build_computer_use_runtime(args)

    def run_goal(self, goal: str | None = None) -> Dict[str, Any]:
        resolved_goal = str(goal or getattr(self.runtime.config, "goal", "") or "").strip()
        if not resolved_goal:
            raise ValueError("goal is required")
        return dict(self.runtime.run(resolved_goal))


def build_computer_domain_service(args: Any) -> ComputerDomainService:
    return ComputerDomainService(args)


def run_computer_domain(*args, **kwargs):
    return run_computer_use_runtime(*args, **kwargs)


__all__ = [
    "ComputerDomainService",
    "build_computer_domain_service",
    "run_computer_domain",
]
