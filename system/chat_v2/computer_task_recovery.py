# -*- coding: utf-8 -*-
"""Checkpoint persistence for resumable computer-control tasks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from system.core.planner import Plan, Task


DEFAULT_COMPUTER_RECOVERY_PATH = "artifacts/audit/computer_task_recovery.json"
BROWSER_SESSION_REPAIR_TASKS = {
    "plugin:browser_dom_click",
    "plugin:browser_dom_type",
    "plugin:browser_dom_extract_text",
    "plugin:browser_dom_wait_text",
    "plugin:browser_dom_screenshot",
    "plugin:browser_dom_close",
}
DESKTOP_CONTEXT_REPAIR_TASKS = {
    "plugin:desktop_type_control",
    "plugin:desktop_type_text",
    "plugin:desktop_hotkey",
    "plugin:desktop_click_control",
    "plugin:desktop_click",
    "plugin:desktop_drag",
    "plugin:desktop_click_text",
    "plugin:desktop_ocr",
    "plugin:desktop_screenshot",
}
DESKTOP_CONTEXT_ANCHOR_TASKS = {
    "plugin:desktop_focus_window",
    "plugin:desktop_launch",
    "plugin:desktop_list_windows",
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _task_to_dict(step: Task) -> Dict[str, Any]:
    return {
        "name": _clean(step.name),
        "detail": _clean(step.detail),
        "priority": int(getattr(step, "priority", 1) or 1),
        "status": _clean(getattr(step, "status", "pending")) or "pending",
        "payload": dict(getattr(step, "payload", {}) or {}),
    }


def _task_from_dict(payload: Dict[str, Any]) -> Task:
    return Task(
        name=_clean((payload or {}).get("name")),
        detail=_clean((payload or {}).get("detail")),
        priority=int((payload or {}).get("priority", 1) or 1),
        status=_clean((payload or {}).get("status", "pending")) or "pending",
        payload=dict((payload or {}).get("payload", {}) or {}),
    )


def _clone_task(step: Task) -> Task:
    return Task(
        name=_clean(getattr(step, "name", "")),
        detail=_clean(getattr(step, "detail", "")),
        priority=int(getattr(step, "priority", 1) or 1),
        status=_clean(getattr(step, "status", "pending")) or "pending",
        payload=dict(getattr(step, "payload", {}) or {}),
    )


def _step_name(step: Task | None) -> str:
    return _clean(getattr(step, "name", "") if step is not None else "")


def _find_last_step_index(steps: Sequence[Task], before_index: int, names: set[str]) -> int:
    limit = min(max(0, int(before_index)), max(0, len(steps) - 1))
    for idx in range(limit, -1, -1):
        if _step_name(steps[idx]) in names:
            return idx
    return -1


def _has_goal_hint(payload: Dict[str, Any]) -> bool:
    node = payload if isinstance(payload, dict) else {}
    return bool(
        node.get("verify_change")
        or _clean(node.get("verify_text"))
        or _clean(node.get("verify_not_text"))
        or _clean(node.get("verify_window_title"))
    )


def _result_verified(item: Dict[str, Any]) -> bool:
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    if "verified" in result:
        return bool(result.get("verified", False))
    return bool(item.get("ok", False))


def _find_repair_anchor(step_results: Sequence[Dict[str, Any]], next_index: int) -> tuple[int, str]:
    limit = max(0, int(next_index))
    for idx in range(limit - 1, -1, -1):
        row = step_results[idx] if idx < len(step_results) else {}
        if bool(row.get("ok", False)) and _result_verified(row):
            return idx, "resume_from_last_verified_step"
    for idx in range(limit - 1, -1, -1):
        row = step_results[idx] if idx < len(step_results) else {}
        if bool(row.get("ok", False)):
            return idx, "resume_from_last_successful_step"
    return limit, "resume_remaining_steps"


def _apply_failure_repair_policy(
    *,
    steps: Sequence[Task],
    next_index: int,
    feedback_event: Dict[str, Any] | None = None,
) -> tuple[int, List[Task], str]:
    if not steps:
        return 0, [], "none"
    failed_index = max(0, min(int(next_index), len(steps) - 1))
    failed_step = steps[failed_index]
    failed_name = _step_name(failed_step)
    reason_code = _clean((feedback_event or {}).get("reason_code"))

    if failed_name in BROWSER_SESSION_REPAIR_TASKS:
        open_index = _find_last_step_index(steps, failed_index, {"plugin:browser_dom_open", "plugin:browser_open_url"})
        if 0 <= open_index < failed_index:
            return open_index, [_clone_task(step) for step in steps[open_index:]], "repair_browser_session"

    if failed_name in DESKTOP_CONTEXT_REPAIR_TASKS:
        anchor_index = _find_last_step_index(steps, failed_index, DESKTOP_CONTEXT_ANCHOR_TASKS)
        if 0 <= anchor_index < failed_index:
            return anchor_index, [_clone_task(step) for step in steps[anchor_index:]], "repair_desktop_context"

    if failed_name == "plugin:desktop_focus_window" or reason_code == "window_not_found":
        list_index = _find_last_step_index(steps, failed_index, {"plugin:desktop_list_windows"})
        if 0 <= list_index < failed_index:
            return list_index, [_clone_task(step) for step in steps[list_index:]], "repair_desktop_refresh_windows"
        synthetic = Task(name="plugin:desktop_list_windows", detail="list windows", payload={})
        repair_steps = [synthetic] + [_clone_task(step) for step in steps[failed_index:]]
        return failed_index, repair_steps, "repair_desktop_refresh_windows"

    return failed_index, [_clone_task(step) for step in steps[failed_index:]], "resume_remaining_steps"


def build_computer_recovery_payload(
    *,
    query: str,
    plan: Plan,
    execution: Any | None,
    feedback_event: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    steps = list(getattr(plan, "steps", []) or [])
    if len(steps) < 2:
        return None
    task_names = [_clean(getattr(step, "name", "")) for step in steps]
    if not any(name.startswith("plugin:browser_") or name.startswith("plugin:desktop_") for name in task_names):
        return None
    rows = list(getattr(execution, "step_results", []) if execution is not None else [])
    next_index = len(rows)
    for idx, row in enumerate(rows):
        if not bool(row.get("ok", False)):
            next_index = idx
            break
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        if _has_goal_hint(payload) and not _result_verified(row):
            next_index = idx
            break
    if next_index >= len(steps):
        completion_status = _clean((feedback_event or {}).get("task_completion_status"))
        if completion_status in {"completed", "completed_inferred"}:
            resume_available = False
        else:
            next_index = max(0, len(steps) - 1)
            resume_available = True
    else:
        resume_available = True
    completed_count = max(0, min(next_index, len(steps)))
    remaining = steps[next_index:] if resume_available else []
    repair_index, repair_strategy = _find_repair_anchor(rows, next_index) if resume_available else (next_index, "none")
    repair_index = max(0, min(int(repair_index), len(steps)))
    repair_steps = [_clone_task(step) for step in steps[repair_index:]] if resume_available else []
    if resume_available:
        policy_index, policy_steps, policy_strategy = _apply_failure_repair_policy(
            steps=steps,
            next_index=next_index,
            feedback_event=feedback_event,
        )
        if policy_steps and policy_index <= repair_index:
            repair_index = int(policy_index)
            repair_steps = list(policy_steps)
            repair_strategy = _clean(policy_strategy) or repair_strategy
    payload = {
        "version": 1,
        "ts": time.time(),
        "query": _clean(query),
        "status": _clean((feedback_event or {}).get("task_completion_status")) or _clean((feedback_event or {}).get("status")) or "unknown",
        "completion_rule": _clean((feedback_event or {}).get("task_completion_rule")),
        "resume_available": bool(resume_available and remaining),
        "resume_from_index": int(next_index),
        "completed_step_count": int(completed_count),
        "step_count": len(steps),
        "steps": [_task_to_dict(step) for step in steps],
        "remaining_steps": [_task_to_dict(step) for step in remaining],
        "repair_from_index": int(repair_index),
        "repair_strategy": _clean(repair_strategy),
        "repair_step_count": len(repair_steps),
        "repair_anchor_step": _task_to_dict(repair_steps[0]) if resume_available and repair_steps else {},
        "repair_steps": [_task_to_dict(step) for step in repair_steps],
        "suggested_command": "resume last computer task" if remaining else "",
    }
    return payload


def save_computer_recovery_payload(path: str, payload: Dict[str, Any] | None) -> bool:
    if not payload:
        return False
    node = Path(_clean(path) or DEFAULT_COMPUTER_RECOVERY_PATH)
    try:
        node.parent.mkdir(parents=True, exist_ok=True)
        node.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def load_computer_recovery_payload(path: str) -> Dict[str, Any] | None:
    node = Path(_clean(path) or DEFAULT_COMPUTER_RECOVERY_PATH)
    if not node.exists():
        return None
    try:
        payload = json.loads(node.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def build_resume_plan(path: str) -> Plan | None:
    payload = load_computer_recovery_payload(path)
    if not payload or not bool(payload.get("resume_available", False)):
        return None
    raw_steps = list(payload.get("repair_steps", []) or [])
    strategy = _clean(payload.get("repair_strategy"))
    if not raw_steps:
        raw_steps = list(payload.get("remaining_steps", []) or [])
        if not strategy:
            strategy = "resume_remaining_steps"
    steps = [_task_from_dict(item) for item in raw_steps if isinstance(item, dict)]
    if not steps:
        return None
    goal = _clean(payload.get("query")) or "resume computer task"
    return Plan(goal=f"{goal} [{strategy or 'resume'}]", steps=steps, levels=[[step] for step in steps])


__all__ = [
    "DEFAULT_COMPUTER_RECOVERY_PATH",
    "build_computer_recovery_payload",
    "build_resume_plan",
    "load_computer_recovery_payload",
    "save_computer_recovery_payload",
]
