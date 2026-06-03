# -*- coding: utf-8 -*-
"""Structured feedback and replay writeback for computer-control tasks."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from system.learning.dialogue.dialogue_evolver import DialogueFailureReflector
from .computer_task_completion import assess_computer_task_completion


DEFAULT_FEEDBACK_PATH = "artifacts/audit/computer_task_feedback.jsonl"
DEFAULT_STATS_PATH = "artifacts/audit/computer_task_stats.json"
DEFAULT_REFLECTIONS_PATH = "artifacts/memory/dialogue_reflections.jsonl"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _bool_env(name: str, default: bool) -> bool:
    raw = _clean(os.environ.get(name))
    if not raw:
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "on"}


def _bump(bucket: Dict[str, int], key: str) -> None:
    node = _clean(key) or "unknown"
    bucket[node] = int(bucket.get(node, 0)) + 1


def _is_computer_task(name: str) -> bool:
    return _clean(name).startswith("plugin:browser_") or _clean(name).startswith("plugin:desktop_")


def _task_names(steps: Sequence[Any]) -> List[str]:
    names: List[str] = []
    for step in list(steps or []):
        name = _clean(getattr(step, "name", ""))
        if name:
            names.append(name)
    return names


def _result_map(step_results: Sequence[Dict[str, Any]], names: Sequence[str]) -> List[Dict[str, Any]]:
    allowed = {name for name in list(names or []) if _clean(name)}
    out: List[Dict[str, Any]] = []
    for item in list(step_results or []):
        if not isinstance(item, dict):
            continue
        if _clean(item.get("name")) in allowed:
            out.append(item)
    return out


def _normalize_reason_code(status: str, reason: str) -> str:
    text = _clean(reason).lower()
    if status == "blocked":
        if "browser control is disabled" in text:
            return "browser_disabled"
        if "desktop control is disabled" in text:
            return "desktop_disabled"
        return "permission_blocked"
    if status == "invalid":
        return "validator_rejected"
    if status == "no_result":
        return "no_step_result"
    if status == "unverified":
        return "verification_failed"
    if status == "failed":
        if "http://" in text or "https://" in text:
            return "invalid_url"
        if "window not found" in text:
            return "window_not_found"
        if "sendkeys failed" in text:
            return "sendkeys_failed"
        if "windows only" in text or "currently implemented for windows only" in text:
            return "unsupported_platform"
        return "execution_failed"
    return "success"


def _primary_task(task_names: Sequence[str]) -> str:
    for name in list(task_names or []):
        if _is_computer_task(name):
            return _clean(name)
    return ""


def _completion_repaired_reply(event: Dict[str, Any]) -> str:
    status = _clean(event.get("task_completion_status"))
    rule = _clean(event.get("task_completion_rule"))
    task = _clean(event.get("task_completion_task"))
    if status == "partial":
        if rule == "explicit_goal_partial":
            return "Retry the task and satisfy every explicit verification condition before stopping."
        return "Retry the task and verify the final state before stopping."
    if status == "incomplete":
        if rule == "explicit_goal_unverified":
            return "Retry the task and make the explicit verification condition pass."
        if rule == "no_verification":
            return "Retry the task with `verify change`, `verify text`, or `verify window` so completion can be confirmed."
        return "Retry the task and verify the final state before stopping."
    if status == "failed":
        if task:
            return f"Retry the task after fixing the failure in {task}."
        return "Retry the task after fixing the failing step."
    return ""


def _repaired_reply(task_name: str, status: str, reason: str) -> str:
    task = _clean(task_name)
    detail = _clean(reason)
    if status == "unverified":
        if task == "plugin:browser_open_url":
            return "Open the URL again and verify the browser tab before continuing."
        if task == "plugin:desktop_focus_window":
            return "Verify the active window title, then retry focus window."
        return "Verify the target state, then retry the computer action."
    if status == "no_result":
        if task == "plugin:browser_open_url":
            return "Retry open url and confirm the browser opened the target page."
        if task == "plugin:desktop_focus_window":
            return "Run list windows, then retry focus window with the exact title."
        return "Retry the computer action and confirm the target result."
    if task == "plugin:browser_open_url":
        if "http://" in detail or "https://" in detail or "url" in detail.lower():
            return "Use a full https URL and retry open url."
        return "Check the URL and retry open url."
    if task == "plugin:browser_dom_open":
        return "Check the page URL and browser setup, then retry browser open."
    if task == "plugin:browser_dom_click":
        return "Check the CSS selector, then retry browser click."
    if task == "plugin:browser_dom_type":
        return "Check the selector and input text, then retry browser type."
    if task == "plugin:browser_dom_extract_text":
        return "Check the selector, then retry browser text extraction."
    if task == "plugin:browser_dom_wait_text":
        return "Check the expected page text, then retry browser wait text."
    if task == "plugin:browser_dom_screenshot":
        return "Check the browser session, then retry browser screenshot."
    if task == "plugin:desktop_focus_window":
        return "Run list windows, copy the exact title, then retry focus window."
    if task == "plugin:desktop_list_controls":
        return "Focus the target window, then retry list controls with the exact window title."
    if task == "plugin:desktop_click_control":
        return "List controls for the target window, copy the exact control title or type, then retry click control."
    if task == "plugin:desktop_type_control":
        return "Focus the target window, confirm the control title or type, then retry type control."
    if task == "plugin:desktop_launch":
        return "Use the executable name or full path, then retry launch."
    if task == "plugin:desktop_click":
        return "Check the target coordinates, then retry click."
    if task == "plugin:desktop_drag":
        return "Check the drag coordinates, then retry drag."
    if task == "plugin:desktop_ocr":
        return "Install an OCR backend or provide a clearer screenshot, then retry ocr."
    if task == "plugin:desktop_click_text":
        return "Check the target text on screen, then retry click text."
    if task == "plugin:desktop_type_text":
        return "Focus the target window, then retry type text."
    if task == "plugin:desktop_hotkey":
        return "Focus the target window, then retry the hotkey."
    if task == "plugin:desktop_screenshot":
        return "Use a writable path, then retry screenshot."
    if task == "plugin:desktop_list_windows":
        return "Retry list windows and confirm desktop access is available."
    return "Check the target and retry the computer action."


class ComputerTaskFeedbackRecorder:
    def __init__(
        self,
        *,
        feedback_path: str = "",
        stats_path: str = "",
        reflections_path: str = "",
        enable_reflections: bool | None = None,
    ) -> None:
        self.feedback_path = Path(
            _clean(feedback_path) or _clean(os.environ.get("COMPUTER_TASK_FEEDBACK_PATH")) or DEFAULT_FEEDBACK_PATH
        )
        self.stats_path = Path(
            _clean(stats_path) or _clean(os.environ.get("COMPUTER_TASK_STATS_PATH")) or DEFAULT_STATS_PATH
        )
        self.reflections_path = Path(
            _clean(reflections_path)
            or _clean(os.environ.get("COMPUTER_TASK_REFLECTIONS_PATH"))
            or DEFAULT_REFLECTIONS_PATH
        )
        if enable_reflections is None:
            self.enable_reflections = _bool_env("COMPUTER_TASK_REFLECTIONS_ENABLE", True)
        else:
            self.enable_reflections = bool(enable_reflections)
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.reflector = DialogueFailureReflector(path=self.reflections_path.as_posix())

    def build_event(
        self,
        *,
        query: str,
        reply: str,
        source: str,
        plan_steps: Sequence[Any],
        blocked: Sequence[str],
        invalid: Sequence[str],
        execution: Any | None,
        repaired: bool,
    ) -> Dict[str, Any] | None:
        task_names = [name for name in _task_names(plan_steps) if _is_computer_task(name)]
        if not task_names:
            return None
        primary = _primary_task(task_names)
        step_results = _result_map(getattr(execution, "step_results", []) if execution is not None else [], task_names)
        errors = list(getattr(execution, "errors", []) if execution is not None else [])
        failed_steps = list(getattr(execution, "failed", []) if execution is not None else [])
        completed_steps = list(getattr(execution, "completed", []) if execution is not None else [])

        reason = ""
        status = "success"
        if blocked:
            status = "blocked"
            reason = _clean(list(blocked)[0])
        elif invalid and not step_results:
            status = "invalid"
            reason = _clean(list(invalid)[0])
        elif errors:
            status = "failed"
            reason = _clean((errors[0] or {}).get("error"))
        elif any(not bool(item.get("ok", False)) for item in step_results):
            status = "failed"
            for item in step_results:
                if not bool(item.get("ok", False)):
                    reason = _clean(item.get("error")) or _clean(item.get("type"))
                    break
        elif not step_results:
            status = "no_result"
        else:
            verified_values = []
            for item in step_results:
                result = item.get("result") if isinstance(item.get("result"), dict) else {}
                if isinstance(result, dict) and "verified" in result:
                    verified_values.append(bool(result.get("verified", False)))
            if verified_values and not all(verified_values):
                status = "unverified"
                reason = "one or more verification checks failed"

        reason_code = _normalize_reason_code(status, reason)
        completion = assess_computer_task_completion(
            task_names=task_names,
            step_results=step_results,
            blocked=blocked,
            invalid=invalid,
            failed_steps=failed_steps,
        )
        task_success = bool(completion.get("task_completion_success", False))
        success = status == "success" and task_success
        reflection_candidate = self.enable_reflections and (
            status in {"failed", "unverified", "no_result"}
            or not task_success
        )
        repaired_reply = _repaired_reply(primary, status, reason)
        if not task_success:
            repaired_reply = _completion_repaired_reply(completion) or repaired_reply
        return {
            "ts": time.time(),
            "query": _clean(query),
            "reply": _clean(reply),
            "source": _clean(source),
            "task_names": task_names,
            "primary_task": primary,
            "status": status,
            "success": bool(success),
            "reason": reason,
            "reason_code": reason_code,
            "blocked_count": len(list(blocked or [])),
            "invalid_count": len(list(invalid or [])),
            "failed_count": len(failed_steps),
            "completed_count": len(completed_steps),
            "step_result_count": len(step_results),
            "verified_count": sum(
                1
                for item in step_results
                if bool((item.get("result") or {}).get("verified", False))
            ),
            **completion,
            "repaired_plan": bool(repaired),
            "dry_run": _bool_env("COMPUTER_CONTROL_DRY_RUN", False),
            "reflection_candidate": bool(reflection_candidate),
            "replay_repaired": repaired_reply if reflection_candidate else "",
        }

    def record(self, event: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(event or {})
        payload["ts"] = float(payload.get("ts", 0.0) or time.time())
        reflection = self._write_reflection(payload)
        payload["reflection_written"] = bool(reflection)
        self._append_jsonl(payload)
        return self._update_stats(payload)

    def _write_reflection(self, payload: Dict[str, Any]) -> Dict[str, Any] | None:
        if not bool(payload.get("reflection_candidate", False)):
            return None
        repaired = _clean(payload.get("replay_repaired"))
        if not repaired:
            return None
        return self.reflector.record(
            query=_clean(payload.get("query")),
            reply=_clean(payload.get("reply")),
            route="agent_computer_task",
            reason=_clean(payload.get("reason_code")) or _clean(payload.get("status")),
            context="computer_task",
            repaired=repaired,
        )

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        with self.feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _default_stats(self) -> Dict[str, Any]:
        return {
            "totals": {
                "events": 0,
                "successes": 0,
                "blocked": 0,
                "failed": 0,
                "unverified": 0,
                "invalid": 0,
                "no_result": 0,
                "reflections": 0,
                "task_completed": 0,
                "task_completed_inferred": 0,
                "task_partial": 0,
                "task_incomplete": 0,
                "task_failed": 0,
            },
            "by_status": {},
            "by_task": {},
            "by_reason_code": {},
            "by_source": {},
            "by_completion_status": {},
            "by_completion_rule": {},
            "last_event": {},
        }

    def _load_stats(self) -> Dict[str, Any]:
        if not self.stats_path.exists():
            return self._default_stats()
        try:
            payload = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_stats()
        if not isinstance(payload, dict):
            return self._default_stats()
        base = self._default_stats()
        base.update(payload)
        for key in ("totals", "by_status", "by_task", "by_reason_code", "by_source", "by_completion_status", "by_completion_rule"):
            if not isinstance(base.get(key), dict):
                base[key] = {}
        return base

    def _update_stats(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        stats = self._load_stats()
        totals = dict(stats.get("totals", {}) or {})
        status = _clean(payload.get("status")) or "unknown"
        primary = _clean(payload.get("primary_task")) or "unknown"
        reason_code = _clean(payload.get("reason_code")) or "unknown"
        source = _clean(payload.get("source")) or "unknown"
        completion_status = _clean(payload.get("task_completion_status")) or "unknown"
        completion_rule = _clean(payload.get("task_completion_rule")) or "unknown"

        totals["events"] = int(totals.get("events", 0)) + 1
        if bool(payload.get("success", False)):
            totals["successes"] = int(totals.get("successes", 0)) + 1
        if status in {"blocked", "failed", "unverified", "invalid", "no_result"}:
            totals[status] = int(totals.get(status, 0)) + 1
        if bool(payload.get("reflection_written", False)):
            totals["reflections"] = int(totals.get("reflections", 0)) + 1
        if completion_status in {"completed", "completed_inferred", "partial", "incomplete", "failed"}:
            totals[f"task_{completion_status}"] = int(totals.get(f"task_{completion_status}", 0)) + 1

        by_status = dict(stats.get("by_status", {}) or {})
        by_task = dict(stats.get("by_task", {}) or {})
        by_reason_code = dict(stats.get("by_reason_code", {}) or {})
        by_source = dict(stats.get("by_source", {}) or {})
        by_completion_status = dict(stats.get("by_completion_status", {}) or {})
        by_completion_rule = dict(stats.get("by_completion_rule", {}) or {})
        _bump(by_status, status)
        _bump(by_task, primary)
        _bump(by_reason_code, reason_code)
        _bump(by_source, source)
        _bump(by_completion_status, completion_status)
        _bump(by_completion_rule, completion_rule)

        stats["totals"] = totals
        stats["by_status"] = by_status
        stats["by_task"] = by_task
        stats["by_reason_code"] = by_reason_code
        stats["by_source"] = by_source
        stats["by_completion_status"] = by_completion_status
        stats["by_completion_rule"] = by_completion_rule
        stats["last_event"] = {
            "ts": float(payload.get("ts", 0.0) or 0.0),
            "status": status,
            "primary_task": primary,
            "reason_code": reason_code,
            "source": source,
            "success": bool(payload.get("success", False)),
            "task_completion_status": completion_status,
            "task_completion_rule": completion_rule,
            "reflection_written": bool(payload.get("reflection_written", False)),
        }
        self.stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        return stats


__all__ = ["ComputerTaskFeedbackRecorder"]
