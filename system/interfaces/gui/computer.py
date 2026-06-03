# -*- coding: utf-8 -*-
"""GUI-facing computer-use session helper."""

from __future__ import annotations

from typing import Any, Dict, List

from system.interfaces.api import ComputerUseAPIAdapter


class ComputerUseGUISession:
    def __init__(self, args: Any) -> None:
        self._adapter = ComputerUseAPIAdapter(args)
        self._runs: List[Dict[str, Any]] = []

    def submit_goal(self, goal: str) -> Dict[str, Any]:
        resolved_goal = str(goal or "").strip()
        if not resolved_goal:
            raise ValueError("goal is required")
        result = self._adapter.run_goal(resolved_goal)
        record = {"goal": resolved_goal, "result": dict(result)}
        self._runs.append(record)
        return dict(result)

    def history(self) -> List[Dict[str, Any]]:
        return [
            {
                "goal": str(item.get("goal", "") or ""),
                "result": dict(item.get("result", {}) or {}),
            }
            for item in list(self._runs)
        ]

    def reset(self) -> None:
        self._runs = []


def create_computer_use_gui_session(args: Any) -> ComputerUseGUISession:
    return ComputerUseGUISession(args)


__all__ = [
    "ComputerUseGUISession",
    "create_computer_use_gui_session",
]
