# -*- coding: utf-8 -*-
"""Task-level completion evaluation for multi-step computer actions."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


_GOAL_VERIFICATION_KEYS = ("verify_change", "verify_text", "verify_not_text", "verify_window_title")
_GOAL_VERIFICATION_TYPES = {
    "screen_change",
    "ocr_contains_text",
    "ocr_not_contains_text",
    "window_title_match",
}
_TERMINAL_SIGNAL_TASKS = {
    "plugin:desktop_ocr",
    "plugin:desktop_click_text",
    "plugin:browser_dom_wait_text",
    "plugin:browser_dom_extract_text",
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _has_goal_hint(payload: Dict[str, Any]) -> bool:
    node = payload if isinstance(payload, dict) else {}
    return any(bool(_clean(node.get(key))) if key != "verify_change" else bool(node.get(key, False)) for key in _GOAL_VERIFICATION_KEYS)


def _result_verified(result: Dict[str, Any], *, fallback: bool = True) -> bool:
    node = result if isinstance(result, dict) else {}
    if "verified" in node:
        return bool(node.get("verified", False))
    return bool(fallback)


def assess_computer_task_completion(
    *,
    task_names: Sequence[str],
    step_results: Sequence[Dict[str, Any]],
    blocked: Sequence[str],
    invalid: Sequence[str],
    failed_steps: Sequence[str],
) -> Dict[str, Any]:
    names = [_clean(name) for name in list(task_names or []) if _clean(name)]
    rows = [item for item in list(step_results or []) if isinstance(item, dict)]
    step_count = len(names)
    result_count = len(rows)
    ok_rows = [item for item in rows if bool(item.get("ok", False))]
    verified_rows = [item for item in ok_rows if _result_verified(item.get("result") if isinstance(item.get("result"), dict) else {}, fallback=True)]
    explicit_goal_rows: List[Dict[str, Any]] = []
    terminal_signal_rows: List[Dict[str, Any]] = []
    for item in ok_rows:
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        verification = _clean(result.get("verification"))
        if _has_goal_hint(payload) or verification in _GOAL_VERIFICATION_TYPES:
            explicit_goal_rows.append(item)
        if _clean(item.get("name")) in _TERMINAL_SIGNAL_TASKS:
            terminal_signal_rows.append(item)

    terminal_row = ok_rows[-1] if ok_rows else {}
    terminal_result = terminal_row.get("result") if isinstance(terminal_row.get("result"), dict) else {}
    terminal_name = _clean(terminal_row.get("name"))
    terminal_signal = _clean(terminal_result.get("verification")) or ("terminal_text_signal" if terminal_name in _TERMINAL_SIGNAL_TASKS else "step_ok")
    terminal_verified = _result_verified(terminal_result, fallback=bool(terminal_row.get("ok", False)))

    explicit_goal_requested = bool(explicit_goal_rows)
    explicit_goal_verified = bool(explicit_goal_rows) and all(
        _result_verified(item.get("result") if isinstance(item.get("result"), dict) else {}, fallback=bool(item.get("ok", False)))
        for item in explicit_goal_rows
    )
    all_steps_recorded = bool(step_count > 0 and result_count >= step_count)
    all_steps_ok = bool(all_steps_recorded and len(ok_rows) >= step_count and not failed_steps)
    all_steps_verified = bool(all_steps_ok and len(verified_rows) >= step_count)
    terminal_text_present = bool(_clean(terminal_result.get("text")))
    has_terminal_signal = terminal_name in _TERMINAL_SIGNAL_TASKS or bool(terminal_signal_rows)

    completion_status = "incomplete"
    completion_success = False
    completion_rule = "no_goal_signal"
    completion_reason = "final goal was not verified"
    if blocked or invalid or failed_steps:
        completion_status = "failed"
        completion_rule = "step_failed"
        completion_reason = "one or more task steps failed or were blocked"
    elif not rows:
        completion_status = "failed"
        completion_rule = "no_step_results"
        completion_reason = "no step results were produced"
    elif explicit_goal_requested:
        if explicit_goal_verified and all_steps_ok:
            completion_status = "completed"
            completion_success = True
            completion_rule = "explicit_goal_verified"
            completion_reason = "explicit task verification passed"
        elif any(_result_verified(item.get("result") if isinstance(item.get("result"), dict) else {}, fallback=False) for item in explicit_goal_rows):
            completion_status = "partial"
            completion_rule = "explicit_goal_partial"
            completion_reason = "some explicit task verification passed, but not all"
        else:
            completion_status = "incomplete"
            completion_rule = "explicit_goal_unverified"
            completion_reason = "explicit task verification did not pass"
    elif has_terminal_signal and all_steps_ok:
        if terminal_verified and (terminal_text_present or terminal_name not in _TERMINAL_SIGNAL_TASKS):
            completion_status = "completed"
            completion_success = True
            completion_rule = "terminal_signal_verified"
            completion_reason = "terminal task signal was verified"
        elif terminal_verified:
            completion_status = "completed_inferred"
            completion_success = True
            completion_rule = "terminal_signal_inferred"
            completion_reason = "terminal task succeeded without a stronger goal signal"
        else:
            completion_status = "partial"
            completion_rule = "terminal_signal_unverified"
            completion_reason = "terminal task signal was not verified"
    elif all_steps_verified:
        completion_status = "completed_inferred"
        completion_success = True
        completion_rule = "all_steps_verified"
        completion_reason = "all task steps reported verified"
    elif all_steps_ok and verified_rows:
        completion_status = "partial"
        completion_rule = "partial_step_verification"
        completion_reason = "some steps were verified, but the task goal was not"
    elif all_steps_ok:
        completion_status = "incomplete"
        completion_rule = "no_verification"
        completion_reason = "steps ran without a clear completion signal"

    return {
        "task_step_count": step_count,
        "task_result_count": result_count,
        "task_ok_steps": len(ok_rows),
        "task_verified_steps": len(verified_rows),
        "task_explicit_goal_steps": len(explicit_goal_rows),
        "task_terminal_signal_steps": len(terminal_signal_rows),
        "task_completion_status": completion_status,
        "task_completion_success": bool(completion_success),
        "task_completion_rule": completion_rule,
        "task_completion_reason": completion_reason,
        "task_completion_signal": terminal_signal,
        "task_completion_task": terminal_name or (_clean(names[-1]) if names else ""),
        "task_explicit_verification_requested": bool(explicit_goal_requested),
        "task_terminal_verified": bool(terminal_verified),
    }


__all__ = ["assess_computer_task_completion"]
