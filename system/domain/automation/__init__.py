from __future__ import annotations

from typing import Any, Dict, Iterable, List

from system.automation.runtime import build_automation_runtime, run_automation_runtime


def _coerce_workflow_ids(workflow_ids: Any) -> List[str]:
    if workflow_ids is None:
        return []
    if isinstance(workflow_ids, str):
        items = [workflow_ids]
    else:
        items = list(workflow_ids)
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


class AutomationDomainService:
    def __init__(self, args: Any) -> None:
        self.args = args
        self.runtime = build_automation_runtime(args)

    def list_workflows(self) -> List[Dict[str, Any]]:
        return list(self.runtime.list_workflows())

    def scan(
        self,
        *,
        workflow_ids: Iterable[str] | None = None,
        force_run: bool = False,
    ) -> Dict[str, Any]:
        return dict(
            self.runtime.scan_once(
                workflow_ids=_coerce_workflow_ids(workflow_ids),
                force_run=bool(force_run),
            )
        )

    def run(self, workflow_ids: Iterable[str]) -> Dict[str, Any]:
        ids = _coerce_workflow_ids(workflow_ids)
        if not ids:
            raise ValueError("workflow_ids is required")
        return dict(self.runtime.scan_once(workflow_ids=ids, force_run=True))

    def loop(
        self,
        *,
        poll_s: float | None = None,
        max_cycles: int | None = None,
        workflow_ids: Iterable[str] | None = None,
    ) -> Dict[str, Any]:
        return dict(
            self.runtime.run_loop(
                poll_s=float(
                    poll_s
                    if poll_s is not None
                    else getattr(self.args, "automation_poll_s", 30.0)
                ),
                max_cycles=max(
                    0,
                    int(
                        max_cycles
                        if max_cycles is not None
                        else getattr(self.args, "automation_max_cycles", 0)
                    ),
                ),
                workflow_ids=_coerce_workflow_ids(workflow_ids),
            )
        )


def build_automation_domain_service(args: Any) -> AutomationDomainService:
    return AutomationDomainService(args)


def run_automation_domain(*args, **kwargs):
    return run_automation_runtime(*args, **kwargs)


__all__ = [
    "AutomationDomainService",
    "build_automation_domain_service",
    "run_automation_domain",
]
