# -*- coding: utf-8 -*-
"""Computer task helpers extracted from WorkspaceAgent."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from system.automation.executor import ExecutionResult
from system.core.planner import Plan

from ._agent_types import AgentOutcome
from .computer_task_recovery import (
    DEFAULT_COMPUTER_RECOVERY_PATH,
    build_computer_recovery_payload,
    build_resume_plan,
    load_computer_recovery_payload,
    save_computer_recovery_payload,
)
from .computer_task_templates import build_template_plan, parse_template_command
from .types import ChatRequest


def _special_computer_plan(agent, query: str) -> Tuple[Plan | None, str]:
    text = str(query or "").strip()
    if not text:
        return None, ""
    qq_plan = agent._special_qq_send_plan(text)
    if qq_plan is not None:
        return qq_plan, ""
    template_call = parse_template_command(text)
    if template_call is not None:
        name, args = template_call
        if not str(name or "").strip():
            return None, "Missing computer template name."
        try:
            return build_template_plan(name, args, goal=text), ""
        except Exception as exc:
            return None, str(exc)
    if agent._is_recovery_request(text):
        plan = build_resume_plan(str(agent.config.computer_recovery_path or DEFAULT_COMPUTER_RECOVERY_PATH))
        if plan is None:
            payload = load_computer_recovery_payload(str(agent.config.computer_recovery_path or DEFAULT_COMPUTER_RECOVERY_PATH))
            if payload is not None and not bool(payload.get("resume_available", False)):
                return None, "No resumable computer task is available. The latest task is already complete."
            return None, "No saved computer task recovery point found."
        return plan, ""
    return None, ""


def _special_qq_send_plan(agent, text: str) -> Plan | None:
    if not bool(agent.config.allow_desktop):
        return None
    parsed = agent._parse_qq_send_goal(text)
    if parsed is None:
        return None
    contact, message, open_requested = parsed
    args = {"contact": contact, "message": message}
    if open_requested:
        launch_target = agent._guess_qq_launch_target()
        if launch_target:
            args["launch_target"] = launch_target
    try:
        return build_template_plan("qq_send_message", args, goal=text)
    except Exception:
        return None


def _parse_qq_send_goal(agent, text: str) -> Tuple[str, str, bool] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    normalized = (
        raw.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    low = normalized.lower()
    if "qq" not in low:
        return None
    open_requested = ("打开qq" in low) or ("打开 qq" in low)
    body = re.sub(r"^\s*(?:请|帮我|麻烦|请帮我)\s*", "", normalized, flags=re.IGNORECASE)
    body = re.sub(r"^\s*打开\s*qq(?:\s*(?:并|后))?\s*", "", body, flags=re.IGNORECASE)
    body = re.sub(r"^\s*(?:在\s*qq\s*里|qq\s*里)\s*", "", body, flags=re.IGNORECASE)
    patterns = (
        r"给(?P<contact>.+?)发(?:送|消息|一\s*句|一句|一\s*条|一条|一\s*个|一个)?(?P<message>.+)$",
        r"给(?P<contact>.+?)说(?P<message>.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.IGNORECASE)
        if not match:
            continue
        contact = str(match.group("contact") or "").strip().strip(" ,，。:：;；")
        message = str(match.group("message") or "").strip()
        message = re.sub(r"^[\s,:：;；]+", "", message)
        message = agent._strip_wrapping_quotes(message)
        if contact and message:
            return contact, message, open_requested
    return None


def _strip_wrapping_quotes(agent, text: str) -> str:
    out = str(text or "").strip()
    pairs = {('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")}
    changed = True
    while changed and len(out) >= 2:
        changed = False
        for left, right in pairs:
            if out.startswith(left) and out.endswith(right):
                out = out[len(left) : len(out) - len(right)].strip()
                changed = True
                break
    return out


def _guess_qq_launch_target(agent) -> str:
    candidates: List[str] = []
    for env_name in ("QQ_EXECUTABLE", "QQ_PATH"):
        value = str(os.environ.get(env_name, "") or "").strip()
        if value:
            candidates.append(value)
    candidates.extend(
        [
            r"D:\qq\QQ.exe",
        ]
    )
    seen = set()
    for candidate in candidates:
        path_text = str(candidate or "").strip()
        if not path_text or path_text.lower() in seen:
            continue
        seen.add(path_text.lower())
        try:
            path = Path(path_text).expanduser()
        except Exception:
            continue
        if path.exists() and path.is_file():
            return str(path)
    return ""


def _save_computer_recovery(
    agent,
    query: str,
    plan: Plan,
    execution: ExecutionResult | None,
    outcome: AgentOutcome,
    feedback_event: Dict[str, Any] | None = None,
) -> None:
    if not bool(agent.config.enable_computer_recovery):
        return
    try:
        payload = build_computer_recovery_payload(
            query=query,
            plan=plan,
            execution=execution,
            feedback_event=feedback_event,
        )
        if payload is None:
            return
        saved = save_computer_recovery_payload(
            str(agent.config.computer_recovery_path or DEFAULT_COMPUTER_RECOVERY_PATH),
            payload,
        )
        if not bool(saved):
            outcome.metadata["computer_task_resume_save_failed"] = True
            outcome.metadata.pop("computer_task_resume_available", None)
            outcome.metadata.pop("computer_task_resume_from_index", None)
            outcome.metadata.pop("computer_task_repair_from_index", None)
            outcome.metadata.pop("computer_task_repair_strategy", None)
            outcome.metadata.pop("computer_task_resume_command", None)
            return
        outcome.metadata["computer_task_resume_available"] = bool(payload.get("resume_available", False))
        outcome.metadata["computer_task_resume_from_index"] = int(payload.get("resume_from_index", 0) or 0)
        outcome.metadata["computer_task_repair_from_index"] = int(payload.get("repair_from_index", 0) or 0)
        outcome.metadata["computer_task_repair_strategy"] = str(payload.get("repair_strategy", "") or "")
        outcome.metadata["computer_task_resume_save_failed"] = False
        if payload.get("resume_available", False):
            outcome.metadata["computer_task_resume_command"] = str(payload.get("suggested_command", "") or "")
    except Exception:
        return


def _record_computer_feedback(
    agent,
    *,
    req: ChatRequest,
    plan: Plan,
    execution: ExecutionResult | None,
    blocked: Sequence[str],
    invalid: Sequence[str],
    outcome: AgentOutcome,
    repaired: bool,
) -> Dict[str, Any] | None:
    if agent.computer_feedback is None:
        return None
    try:
        event = agent.computer_feedback.build_event(
            query=req.user_text,
            reply=outcome.text,
            source=outcome.source,
            plan_steps=list(plan.steps or []),
            blocked=list(blocked or []),
            invalid=list(invalid or []),
            execution=execution,
            repaired=repaired,
        )
        if event is not None:
            outcome.metadata["computer_task_completion"] = str(event.get("task_completion_status", "") or "")
            outcome.metadata["computer_task_completion_rule"] = str(event.get("task_completion_rule", "") or "")
            outcome.metadata["computer_task_success"] = bool(event.get("task_completion_success", False))
            agent.computer_feedback.record(event)
            return event
    except Exception:
        return None
    return None


def _is_template_request(agent, text: str) -> bool:
    return parse_template_command(text) is not None


def _is_recovery_request(agent, text: str) -> bool:
    low = str(text or "").strip().lower()
    return low in {
        "resume last computer task",
        "resume computer task",
        "continue last computer task",
        "continue computer task",
    }
