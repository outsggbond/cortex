# -*- coding: utf-8 -*-
"""GUI-facing automation controller."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from system.interfaces.api import AutomationAPIAdapter


class AutomationGUIController:
    def __init__(self, args: Any) -> None:
        self._adapter = AutomationAPIAdapter(args)
        self._last_summary: Dict[str, Any] | None = None
        self._last_workflows: List[Dict[str, Any]] = []

    @property
    def last_summary(self) -> Dict[str, Any] | None:
        return None if self._last_summary is None else dict(self._last_summary)

    @property
    def last_workflows(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in list(self._last_workflows)]

    def list_workflows(self) -> List[Dict[str, Any]]:
        self._last_workflows = list(self._adapter.list_workflows())
        return self.last_workflows

    def scan(
        self,
        *,
        workflow_ids: Iterable[str] | None = None,
        force_run: bool = False,
    ) -> Dict[str, Any]:
        self._last_summary = self._adapter.scan(workflow_ids=workflow_ids, force_run=force_run)
        return self.last_summary or {}

    def run(self, workflow_ids: Iterable[str]) -> Dict[str, Any]:
        self._last_summary = self._adapter.run(workflow_ids)
        return self.last_summary or {}

    def loop(
        self,
        *,
        poll_s: float | None = None,
        max_cycles: int | None = None,
        workflow_ids: Iterable[str] | None = None,
    ) -> Dict[str, Any]:
        self._last_summary = self._adapter.loop(
            poll_s=poll_s,
            max_cycles=max_cycles,
            workflow_ids=workflow_ids,
        )
        return self.last_summary or {}

    def reset(self) -> None:
        self._last_summary = None
        self._last_workflows = []


def create_automation_gui_controller(args: Any) -> AutomationGUIController:
    return AutomationGUIController(args)


__all__ = [
    "AutomationGUIController",
    "create_automation_gui_controller",
]
