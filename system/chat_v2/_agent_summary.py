# -*- coding: utf-8 -*-
"""Summarization helpers extracted from WorkspaceAgent."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from system.automation.executor import ExecutionResult
from system.core.planner import Plan


def _compose_answer(
    agent,
    req,
    plan: Plan,
    execution: ExecutionResult,
    *,
    blocked: Sequence[str],
    invalid: Sequence[str],
    repaired: bool,
) -> Tuple[str, str]:
    if execution.failed or blocked or invalid:
        return agent._local_summary(req, execution, blocked=blocked, invalid=invalid, repaired=repaired), "agent"
    llm_text = agent._llm_summary(req, plan, execution, blocked=blocked, invalid=invalid, repaired=repaired)
    if llm_text:
        return llm_text, "agent_llm"
    return agent._local_summary(req, execution, blocked=blocked, invalid=invalid, repaired=repaired), "agent"


def _llm_summary(
    agent,
    req,
    plan: Plan,
    execution: ExecutionResult,
    *,
    blocked: Sequence[str],
    invalid: Sequence[str],
    repaired: bool,
) -> str:
    if agent.llm_client is None or not agent.llm_client.available():
        return ""
    findings = agent._findings_block(execution)
    if not findings:
        return ""
    low = str(req.user_text or "").lower()
    detail_line = "Answer in a grounded, moderately detailed way."
    if agent._is_project_inspection_request(str(req.user_text or ""), low):
        detail_line = (
            "Answer in a grounded, moderately detailed way. "
            "For project inspection questions, synthesize the architecture, entrypoints, docs, and tests into 4-8 compact bullets."
        )
    prompt = (
        "Answer the user's workspace question using only the verified findings below.\n"
        f"{detail_line} If the findings are insufficient, say so clearly.\n\n"
        f"User Question:\n{req.user_text}\n\n"
        f"Plan Steps:\n{agent._step_list(plan.steps)}\n\n"
        f"Verified Findings:\n{findings}\n"
    )
    if repaired:
        prompt += "\nThe agent repaired at least one path mismatch before answering.\n"
    if blocked:
        prompt += "\nBlocked Steps:\n" + "\n".join(f"- {item}" for item in list(blocked)[:3]) + "\n"
    if invalid:
        prompt += "\nRejected Steps:\n" + "\n".join(f"- {item}" for item in list(invalid)[:3]) + "\n"
    try:
        return str(agent.llm_client.generate(prompt, req.history) or "").strip()
    except Exception:
        return ""


def _local_summary(
    agent,
    req,
    execution: ExecutionResult,
    *,
    blocked: Sequence[str],
    invalid: Sequence[str],
    repaired: bool,
) -> str:
    text = str(req.user_text or "").strip()
    low = text.lower()
    if agent._is_project_inspection_request(text, low):
        summary = agent._project_inspection_summary(
            execution,
            blocked=blocked,
            invalid=invalid,
            repaired=repaired,
        )
        if summary:
            return summary
    return agent._template_summary(execution, blocked=blocked, invalid=invalid, repaired=repaired)


def _project_inspection_summary(
    agent,
    execution: ExecutionResult,
    *,
    blocked: Sequence[str],
    invalid: Sequence[str],
    repaired: bool,
) -> str:
    listings: Dict[str, List[str]] = {}
    file_searches: Dict[Tuple[str, str], List[str]] = {}
    for item in execution.step_results:
        if not bool(item.get("ok", False)):
            continue
        name = str(item.get("name", "") or "").strip()
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        if name == "list_dir":
            path = str(payload.get("path", ".") or ".").strip() or "."
            listings[path] = [str(x) for x in list(result.get("entries", []) or [])]
        elif name == "search_files":
            pattern = str(payload.get("pattern", "") or "").strip()
            path = str(payload.get("path", ".") or ".").strip() or "."
            file_searches[(pattern, path)] = [str(x) for x in list(result.get("hits", []) or [])]

    root_entries = listings.get(".", [])
    system_entries = listings.get("system", [])
    docs_entries = listings.get("docs", [])
    tests_entries = listings.get("tests", [])
    readme_hits = agent._project_search_hits(file_searches, "README")
    pyproject_hits = agent._project_search_hits(file_searches, "pyproject.toml")
    main_hits = agent._project_search_hits(file_searches, "main.py")
    app_runtime_hits = agent._project_search_hits(file_searches, "app_runtime.py", path="system")
    chat_runtime_hits = agent._project_search_hits(file_searches, "runtime.py", path="system/chat_v2")

    lines: List[str] = ["I inspected the project locally with read-only workspace actions."]
    if repaired:
        lines.append("A path mismatch was repaired before the final answer.")
    if root_entries:
        lines.append(f"Workspace root: {agent._preview_items(root_entries, limit=10)}")
    if readme_hits:
        lines.append(f"Top-level docs entrypoints: {agent._preview_items(readme_hits, limit=4)}")
    if pyproject_hits:
        lines.append(f"Packaging/config entrypoints: {agent._preview_items(pyproject_hits, limit=4)}")
    entry_candidates: List[str] = []
    entry_candidates.extend(main_hits[:2])
    entry_candidates.extend(app_runtime_hits[:2])
    entry_candidates.extend(chat_runtime_hits[:2])
    if entry_candidates:
        lines.append(f"Runtime entrypoint candidates: {agent._preview_items(entry_candidates, limit=6)}")
    if system_entries:
        lines.append(f"Core runtime modules under system/: {agent._preview_items(system_entries, limit=12)}")
    if docs_entries:
        lines.append(f"Docs currently available: {agent._preview_items(docs_entries, limit=8)}")
    if tests_entries:
        lines.append(f"Tests currently available: {agent._preview_items(tests_entries, limit=10)}")
    if blocked:
        lines.append("Blocked steps:")
        lines.extend(f"- {item}" for item in list(blocked)[:3])
    if invalid:
        lines.append("Rejected steps:")
        lines.extend(f"- {item}" for item in list(invalid)[:3])
    if execution.failed:
        lines.append("Failed steps:")
        lines.extend(f"- {item}" for item in execution.failed[:3])
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


def _template_summary(
    agent,
    execution: ExecutionResult,
    *,
    blocked: Sequence[str],
    invalid: Sequence[str],
    repaired: bool,
) -> str:
    lines: List[str] = []
    if execution.failed:
        lines.append("I inspected the workspace, but the agent loop did not fully complete.")
    else:
        lines.append("I inspected the workspace and verified the result with the current agent loop.")
    if repaired:
        lines.append("A path mismatch was repaired before the final answer.")

    findings = agent._structured_findings(execution)
    if findings:
        lines.extend(findings[:6])
    elif execution.completed:
        lines.append("Completed steps:")
        lines.extend(f"- {step}" for step in execution.completed[:4])
    else:
        lines.append("No verified workspace findings were produced.")

    if blocked:
        lines.append("Blocked steps:")
        lines.extend(f"- {item}" for item in list(blocked)[:3])
    if invalid:
        lines.append("Rejected steps:")
        lines.extend(f"- {item}" for item in list(invalid)[:3])
    if execution.failed:
        lines.append("Failed steps:")
        lines.extend(f"- {item}" for item in execution.failed[:3])
    return "\n".join(lines)


def _findings_block(agent, execution: ExecutionResult) -> str:
    return "\n".join(agent._structured_findings(execution))


def _structured_findings(agent, execution: ExecutionResult) -> List[str]:
    lines: List[str] = []
    for item in execution.step_results:
        if not bool(item.get("ok", False)):
            continue
        name = str(item.get("name", ""))
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        if name == "read_file":
            path = str(payload.get("path", "") or "").strip()
            content = str(result.get("output", "") or "").strip()
            if content:
                lines.append(f"Read {path or '<unknown>'}: {agent._snippet(content)}")
            continue
        if name == "list_dir":
            path = str(payload.get("path", "") or ".").strip() or "."
            entries = list(result.get("entries", []) or [])
            preview = ", ".join(str(x) for x in entries[:8]) if entries else "(empty)"
            lines.append(f"Listed {path}: {preview}")
            continue
        if name == "check_exists":
            path = str(payload.get("path", "") or "").strip()
            note = str(result.get("note", "") or "").strip()
            lines.append(f"Checked {path}: {note}")
            continue
        if name == "search_text":
            pattern = str(payload.get("pattern", "") or "").strip()
            path = str(payload.get("path", "") or ".").strip() or "."
            hits = list(result.get("hits", []) or [])
            preview = " | ".join(str(x) for x in hits[:5]) if hits else "no matches"
            lines.append(f"Searched text '{pattern}' in {path}: {preview}")
            continue
        if name == "search_files":
            pattern = str(payload.get("pattern", "") or "").strip()
            path = str(payload.get("path", "") or ".").strip() or "."
            hits = list(result.get("hits", []) or [])
            preview = ", ".join(str(x) for x in hits[:8]) if hits else "no matches"
            lines.append(f"Found files matching '{pattern}' in {path}: {preview}")
            continue
        if name == "plugin:browser_open_url":
            url = str(result.get("url", "") or payload.get("url", "")).strip()
            verified = bool(result.get("verified", False))
            lines.append(f"Opened URL {url}: verified={verified}")
            continue
        if name == "plugin:browser_dom_open":
            url = str(result.get("url", "") or "").strip()
            session = str(result.get("session", "") or "default").strip()
            lines.append(f"Opened browser session {session}: {url or '(no url)'}")
            continue
        if name == "plugin:browser_dom_click":
            selector = str(result.get("selector", "") or payload.get("selector", "")).strip()
            verified = bool(result.get("verified", False))
            lines.append(f"Clicked browser selector {selector}: verified={verified}")
            continue
        if name == "plugin:browser_dom_type":
            selector = str(result.get("selector", "") or payload.get("selector", "")).strip()
            preview = str(result.get("text_preview", "") or "").strip()
            lines.append(f"Typed '{preview}' into browser selector {selector}")
            continue
        if name == "plugin:browser_dom_extract_text":
            selector = str(result.get("selector", "") or payload.get("selector", "")).strip()
            text = str(result.get("text", "") or "").strip()
            lines.append(f"Extracted browser text from {selector}: {agent._snippet(text)}")
            continue
        if name == "plugin:browser_dom_wait_text":
            text = str(result.get("text", "") or payload.get("text", "")).strip()
            verified = bool(result.get("verified", False))
            lines.append(f"Waited for browser text '{text}': verified={verified}")
            continue
        if name == "plugin:browser_dom_screenshot":
            path = str(result.get("path", "") or "").strip()
            verified = bool(result.get("verified", False))
            lines.append(f"Saved browser screenshot {path}: verified={verified}")
            continue
        if name == "plugin:browser_dom_close":
            session = str(result.get("session", "") or "default").strip()
            verified = bool(result.get("verified", False))
            lines.append(f"Closed browser session {session}: verified={verified}")
            continue
        if name == "plugin:desktop_list_windows":
            windows = list(result.get("windows", []) or [])
            preview_items = [str((item or {}).get("title", "")).strip() for item in windows[:10] if str((item or {}).get("title", "")).strip()]
            preview = ", ".join(preview_items)
            extra = max(0, len(windows) - len(preview_items))
            suffix = f" (+{extra} more)" if extra > 0 else ""
            lines.append(f"Listed desktop windows: {preview or '(none)'}{suffix}")
            continue
        if name == "plugin:desktop_list_controls":
            controls = list(result.get("controls", []) or [])
            preview_items = []
            for item in controls[:8]:
                title = str((item or {}).get("title", "")).strip() or "(untitled)"
                control_type = str((item or {}).get("control_type", "")).strip() or "Control"
                preview_items.append(f"{title}/{control_type}")
            preview = ", ".join(preview_items)
            extra = max(0, len(controls) - len(preview_items))
            suffix = f" (+{extra} more)" if extra > 0 else ""
            lines.append(f"Listed window controls: {preview or '(none)'}{suffix}")
            continue
        if name == "plugin:desktop_focus_window":
            matched = str(result.get("matched_title", "") or payload.get("title", "")).strip()
            verified = bool(result.get("verified", False))
            lines.append(f"Focused window {matched}: verified={verified}")
            continue
        if name == "plugin:desktop_launch":
            target = str(result.get("target", "") or payload.get("target", "")).strip()
            pid = int(result.get("pid", 0) or 0)
            verified = bool(result.get("verified", False))
            lines.append(f"Launched {target}: pid={pid} verified={verified}")
            continue
        if name == "plugin:desktop_type_text":
            preview = str(result.get("text_preview", "") or "").strip()
            active_title = str(result.get("active_title", "") or "").strip()
            lines.append(f"Typed text '{preview}' into {active_title or 'active window'}")
            continue
        if name == "plugin:desktop_click_control":
            control_title = str(result.get("control_title", "") or payload.get("control", "")).strip()
            control_type = str(result.get("control_type", "") or payload.get("control_type", "")).strip()
            matched = str(result.get("matched_window", "") or payload.get("window", "")).strip()
            lines.append(f"Clicked control {control_title or '(untitled)'}{('/' + control_type) if control_type else ''} in {matched or 'window'}")
            continue
        if name == "plugin:desktop_type_control":
            control_title = str(result.get("control_title", "") or payload.get("control", "")).strip()
            control_type = str(result.get("control_type", "") or payload.get("control_type", "")).strip()
            matched = str(result.get("matched_window", "") or payload.get("window", "")).strip()
            preview = str(result.get("text_preview", "") or "").strip()
            lines.append(f"Typed text '{preview}' into control {control_title or '(untitled)'}{('/' + control_type) if control_type else ''} in {matched or 'window'}")
            continue
        if name == "plugin:desktop_hotkey":
            hotkey = str(result.get("hotkey", "") or "").strip()
            active_title = str(result.get("active_title", "") or "").strip()
            lines.append(f"Sent hotkey {hotkey} in {active_title or 'active window'}")
            continue
        if name == "plugin:desktop_screenshot":
            path = str(result.get("path", "") or "").strip()
            verified = bool(result.get("verified", False))
            lines.append(f"Saved screenshot {path}: verified={verified}")
            continue
        if name == "plugin:desktop_click":
            x = int(result.get("x", 0) or 0)
            y = int(result.get("y", 0) or 0)
            button = str(result.get("button", "") or "left").strip()
            clicks = int(result.get("clicks", 1) or 1)
            lines.append(f"Clicked {button} at {x},{y} x{clicks}")
            continue
        if name == "plugin:desktop_drag":
            x1 = int(result.get("x1", 0) or 0)
            y1 = int(result.get("y1", 0) or 0)
            x2 = int(result.get("x2", 0) or 0)
            y2 = int(result.get("y2", 0) or 0)
            lines.append(f"Dragged mouse from {x1},{y1} to {x2},{y2}")
            continue
        if name == "plugin:desktop_ocr":
            text = str(result.get("text", "") or "").strip()
            lines.append(f"OCR text: {agent._snippet(text)}")
            continue
        if name == "plugin:desktop_click_text":
            query = str(result.get("query", "") or payload.get("text", "")).strip()
            matched = str(result.get("matched_text", "") or "").strip()
            lines.append(f"Clicked OCR text '{query}': matched={matched or '(none)'}")
            continue
    return lines


def _preview_items(agent, items: Sequence[str], *, limit: int = 8) -> str:
    rows = [str(x).strip() for x in list(items or []) if str(x).strip()]
    if not rows:
        return "(none)"
    preview = rows[: max(1, int(limit))]
    tail = "" if len(rows) <= len(preview) else f" (+{len(rows) - len(preview)} more)"
    return ", ".join(preview) + tail


def _feedback_from_errors(agent, execution: ExecutionResult) -> str:
    parts: List[str] = []
    for err in list(getattr(execution, "errors", []) or [])[:4]:
        step = str(err.get("step", "") or err.get("name", "")).strip()
        msg = str(err.get("error", "") or err.get("type", "")).strip()
        chunk = ": ".join(part for part in (step, msg) if part)
        if chunk:
            parts.append(chunk)
    return "; ".join(parts)
