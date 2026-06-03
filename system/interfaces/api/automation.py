# -*- coding: utf-8 -*-
"""Framework-agnostic automation adapter for API handlers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from system.domain.automation import build_automation_domain_service


class AutomationAPIAdapter:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.service = build_automation_domain_service(args)

    def list_workflows(self) -> List[Dict[str, Any]]:
        return self.service.list_workflows()

    def scan(
        self,
        *,
        workflow_ids: Iterable[str] | None = None,
        force_run: bool = False,
    ) -> Dict[str, Any]:
        return self.service.scan(workflow_ids=workflow_ids, force_run=force_run)

    def run(self, workflow_ids: Iterable[str]) -> Dict[str, Any]:
        return self.service.run(workflow_ids)

    def loop(
        self,
        *,
        poll_s: float | None = None,
        max_cycles: int | None = None,
        workflow_ids: Iterable[str] | None = None,
    ) -> Dict[str, Any]:
        return self.service.loop(
            poll_s=poll_s,
            max_cycles=max_cycles,
            workflow_ids=workflow_ids,
        )


def create_automation_api(args: Any) -> AutomationAPIAdapter:
    return AutomationAPIAdapter(args)


__all__ = [
    "AutomationAPIAdapter",
    "create_automation_api",
]
