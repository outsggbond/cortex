# -*- coding: utf-8 -*-
"""Framework-agnostic computer-use adapter for API handlers."""

from __future__ import annotations

from typing import Any, Dict

from system.domain.computer import build_computer_domain_service


class ComputerUseAPIAdapter:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.service = build_computer_domain_service(args)

    def run_goal(self, goal: str | None = None) -> Dict[str, Any]:
        return self.service.run_goal(goal)


def create_computer_use_api(args: Any) -> ComputerUseAPIAdapter:
    return ComputerUseAPIAdapter(args)


__all__ = [
    "ComputerUseAPIAdapter",
    "create_computer_use_api",
]
