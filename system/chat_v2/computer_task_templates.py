# -*- coding: utf-8 -*-
"""Named templates for reusable computer-control flows."""

from __future__ import annotations

import re
import shlex
from typing import Any, Dict, List, Tuple

from system.core.planner import Plan, Task

from .computer_task_learning import LearnedComputerTaskStore


SEARCH_LABEL = "\u641c\u7d22"
SEND_LABEL = "\u53d1\u9001"
CHAT_INPUT_LABEL = "\u804a\u5929\u8f93\u5165\u533a"


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _decode_unicode_escapes(value: str) -> str:
    text = str(value or "")
    if "\\u" not in text and "\\U" not in text:
        return text

    def repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = raw[2:]
        try:
            return chr(int(digits, 16))
        except Exception:
            return raw

    text = re.sub(r"\\u[0-9a-fA-F]{4}", repl, text)
    text = re.sub(r"\\U[0-9a-fA-F]{8}", repl, text)
    return text


def _task(name: str, detail: str, payload: Dict[str, Any]) -> Task:
    return Task(name=name, detail=detail, payload=dict(payload))


def parse_template_command(text: str) -> Tuple[str, Dict[str, str]] | None:
    raw = _clean(text)
    low = raw.lower()
    if low.startswith("run template "):
        body = raw[13:].strip()
    elif low.startswith("template "):
        body = raw[9:].strip()
    else:
        return None
    if not body:
        return ("", {})
    try:
        tokens = shlex.split(body)
    except Exception:
        tokens = body.split()
    if not tokens:
        return ("", {})
    name = _clean(tokens[0]).lower()
    args: Dict[str, str] = {}
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = _clean(key).lower().replace("-", "_")
        value = _decode_unicode_escapes(_clean(value))
        if key:
            args[key] = value
    return (name, args)


def available_templates() -> List[str]:
    builtins = [
        "browser_capture_page",
        "browser_search",
        "desktop_find_text",
        "desktop_focus_capture",
        "desktop_window_controls",
        "qq_send_message",
    ]
    try:
        learned = LearnedComputerTaskStore().template_names()
    except Exception:
        learned = []
    return builtins + [name for name in learned if name not in builtins]


def build_template_plan(name: str, args: Dict[str, str], *, goal: str = "") -> Plan:
    key = _clean(name).lower()
    params = {str(k).strip().lower(): str(v).strip() for k, v in dict(args or {}).items() if str(k).strip()}
    if key == "browser_capture_page":
        url = _clean(params.get("url"))
        if not url:
            raise ValueError("template browser_capture_page requires url=<https://...>")
        steps = [
            _task("plugin:browser_dom_open", f"browser open {url}", {"url": url, "session": "default"}),
            _task("plugin:browser_dom_screenshot", "browser screenshot", {"session": "default"}),
        ]
    elif key == "browser_search":
        url = _clean(params.get("url"))
        query = _clean(params.get("query"))
        selector = _clean(params.get("selector")) or "#search"
        submit_selector = _clean(params.get("submit_selector"))
        wait_text = _clean(params.get("wait_text"))
        if not url or not query:
            raise ValueError("template browser_search requires url=<https://...> query=<text>")
        steps = [
            _task("plugin:browser_dom_open", f"browser open {url}", {"url": url, "session": "default"}),
            _task(
                "plugin:browser_dom_type",
                f'browser type "{query}" into "{selector}"',
                {"text": query, "selector": selector, "session": "default"},
            ),
        ]
        if submit_selector:
            steps.append(
                _task(
                    "plugin:browser_dom_click",
                    f'browser click "{submit_selector}"',
                    {"selector": submit_selector, "session": "default"},
                )
            )
        if wait_text:
            steps.append(
                _task(
                    "plugin:browser_dom_wait_text",
                    f'browser wait text "{wait_text}"',
                    {"text": wait_text, "session": "default"},
                )
            )
    elif key == "desktop_find_text":
        text_value = _clean(params.get("text"))
        if not text_value:
            raise ValueError("template desktop_find_text requires text=<screen text>")
        steps = [
            _task("plugin:desktop_ocr", "ocr screen", {}),
            _task(
                "plugin:desktop_click_text",
                f"click text {text_value}",
                {"text": text_value, "verify_change": True},
            ),
            _task("plugin:desktop_ocr", "ocr screen", {}),
        ]
    elif key == "desktop_focus_capture":
        title = _clean(params.get("title"))
        if not title:
            raise ValueError("template desktop_focus_capture requires title=<window title>")
        steps = [
            _task(
                "plugin:desktop_focus_window",
                f"focus window {title}",
                {"title": title, "verify_window_title": title},
            ),
            _task("plugin:desktop_screenshot", "screenshot", {}),
        ]
    elif key == "desktop_window_controls":
        title = _clean(params.get("title") or params.get("window"))
        control = _clean(params.get("control"))
        control_type = _clean(params.get("control_type"))
        if not title:
            raise ValueError("template desktop_window_controls requires title=<window title>")
        payload = {"window": title}
        if control:
            payload["control"] = control
        if control_type:
            payload["control_type"] = control_type
        steps = [
            _task(
                "plugin:desktop_focus_window",
                f"focus window {title}",
                {"title": title, "verify_window_title": title},
            ),
            _task(
                "plugin:desktop_list_controls",
                f"list controls in window {title}",
                payload,
            ),
        ]
    elif key == "qq_send_message":
        contact = _clean(params.get("contact"))
        message = _clean(params.get("message"))
        main_window = _clean(params.get("window")) or "QQ"
        launch_target = _clean(params.get("launch_target"))
        if not contact or not message:
            raise ValueError("template qq_send_message requires contact=<name> message=<text>")
        steps = []
        if launch_target:
            steps.append(
                _task(
                    "plugin:desktop_launch",
                    f"launch {launch_target}",
                    {"target": launch_target},
                )
            )
        steps.extend(
            [
                _task(
                    "plugin:desktop_focus_window",
                    f"focus window {main_window}",
                    {"title": main_window, "verify_window_title": main_window},
                ),
                _task(
                    "plugin:desktop_type_control",
                    f'type "{contact}" into control {SEARCH_LABEL} in window {main_window} control type Edit clear first',
                    {
                        "window": main_window,
                        "control": SEARCH_LABEL,
                        "control_type": "Edit",
                        "text": contact,
                        "clear_first": True,
                        "verify_change": True,
                        "fallback_to_search_region": True,
                        "fallback_to_ocr": True,
                        "ocr_text": SEARCH_LABEL,
                    },
                ),
                _task(
                    "plugin:desktop_click_control",
                    f"click control {contact} in window {main_window} control type ListItem index 0",
                    {
                        "window": main_window,
                        "control": contact,
                        "control_type": "ListItem",
                        "index": 0,
                        "verify_change": True,
                        "fallback_to_search_result": True,
                    },
                ),
                _task(
                    "plugin:desktop_click_control",
                    f"click control {CHAT_INPUT_LABEL} in window {main_window} control type Group",
                    {
                        "window": main_window,
                        "control": CHAT_INPUT_LABEL,
                        "control_type": "Group",
                        "fallback_to_message_input": True,
                        "prefer_uia_message_input": True,
                        "anchor_control": SEND_LABEL,
                        "anchor_control_type": "Button",
                    },
                ),
                _task(
                    "plugin:desktop_type_text",
                    f'type "{message}"',
                    {
                        "text": message,
                        "verify_change": True,
                    },
                ),
                _task(
                    "plugin:desktop_click_control",
                    f"click control {SEND_LABEL} in window {main_window} control type Button",
                    {
                        "window": main_window,
                        "control": SEND_LABEL,
                        "control_type": "Button",
                        "verify_change": True,
                    },
                ),
            ]
        )
    else:
        learned_plan = None
        try:
            learned_plan = LearnedComputerTaskStore().build_plan_for_template(key, goal=goal or key)
        except Exception:
            learned_plan = None
        if learned_plan is not None:
            return learned_plan
        known = ", ".join(available_templates())
        raise ValueError(f"unknown computer template: {name}. available: {known}")
    return Plan(goal=goal or key, steps=steps, levels=[[step] for step in steps])
