# -*- coding: utf-8 -*-
"""Persistent learned computer-task plans for local deterministic replay."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from system.core.planner import Plan, Task


DEFAULT_COMPUTER_LEARNING_PATH = "artifacts/memory/computer_task_learnings.json"

_COMPUTER_TASK_NAMES = {
    "plugin:browser_open_url",
    "plugin:browser_dom_open",
    "plugin:browser_dom_click",
    "plugin:browser_dom_type",
    "plugin:browser_dom_extract_text",
    "plugin:browser_dom_wait_text",
    "plugin:browser_dom_screenshot",
    "plugin:browser_dom_close",
    "plugin:desktop_launch",
    "plugin:desktop_list_windows",
    "plugin:desktop_list_controls",
    "plugin:desktop_focus_window",
    "plugin:desktop_click_control",
    "plugin:desktop_type_control",
    "plugin:desktop_type_text",
    "plugin:desktop_hotkey",
    "plugin:desktop_screenshot",
    "plugin:desktop_click",
    "plugin:desktop_drag",
    "plugin:desktop_ocr",
    "plugin:desktop_click_text",
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _learning_path(path: str = "") -> Path:
    raw = _clean(path) or _clean(os.environ.get("COMPUTER_TASK_LEARNING_PATH")) or DEFAULT_COMPUTER_LEARNING_PATH
    return Path(raw)


def normalize_computer_query(text: str) -> str:
    return re.sub(r"\s+", " ", _clean(text)).strip().lower()


def _task_to_dict(step: Task) -> Dict[str, Any]:
    return {
        "name": _clean(getattr(step, "name", "")),
        "detail": _clean(getattr(step, "detail", "")),
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


def _result_verified(item: Dict[str, Any]) -> bool:
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    if "verified" in result:
        return bool(result.get("verified", False))
    return bool(item.get("ok", False))


class LearnedComputerTaskStore:
    def __init__(self, path: str = "") -> None:
        self.path = _learning_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.items = []
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self.items = []
            return
        if isinstance(payload, dict):
            rows = list(payload.get("items", []) or [])
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
        self.items = [row for row in rows if isinstance(row, dict)]

    def _save(self) -> None:
        payload = {"version": 1, "items": self.items}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def template_names(self) -> List[str]:
        names = []
        for item in self.items:
            name = _clean(item.get("template_name"))
            if name:
                names.append(name)
        return sorted(set(names))

    def find_by_query(self, query: str, *, min_successes: int = 1) -> Dict[str, Any] | None:
        query_norm = normalize_computer_query(query)
        if not query_norm:
            return None
        for item in self.items:
            if _clean(item.get("query_norm")) != query_norm:
                continue
            if int(item.get("success_count", 0) or 0) < max(1, int(min_successes)):
                continue
            return dict(item)
        return None

    def find_by_template(self, template_name: str) -> Dict[str, Any] | None:
        key = _clean(template_name)
        if not key:
            return None
        for item in self.items:
            if _clean(item.get("template_name")) == key:
                return dict(item)
        return None

    def build_plan_for_query(self, query: str, *, min_successes: int = 1, goal: str = "") -> Plan | None:
        item = self.find_by_query(query, min_successes=min_successes)
        if item is None:
            return None
        return self._build_plan(item, goal=goal or query)

    def build_plan_for_template(self, template_name: str, *, goal: str = "") -> Plan | None:
        item = self.find_by_template(template_name)
        if item is None:
            return None
        return self._build_plan(item, goal=goal or _clean(item.get("query")) or template_name)

    def remember_success(self, *, query: str, plan: Plan, execution: Any, source: str = "") -> Dict[str, Any] | None:
        query_norm = normalize_computer_query(query)
        steps = list(getattr(plan, "steps", []) or [])
        if not query_norm or not steps:
            return None
        if not all(_clean(getattr(step, "name", "")) in _COMPUTER_TASK_NAMES for step in steps):
            return None
        rows = list(getattr(execution, "step_results", []) if execution is not None else [])
        if len(rows) < len(steps):
            return None
        for idx, step in enumerate(steps):
            row = rows[idx] if idx < len(rows) and isinstance(rows[idx], dict) else {}
            if _clean(row.get("name")) != _clean(getattr(step, "name", "")):
                return None
            if not bool(row.get("ok", False)) or not _result_verified(row):
                return None

        template_name = "learned_" + hashlib.sha1(query_norm.encode("utf-8")).hexdigest()[:12]
        now = time.time()
        for item in self.items:
            if _clean(item.get("query_norm")) != query_norm:
                continue
            item["query"] = _clean(query) or _clean(item.get("query"))
            item["template_name"] = template_name
            item["success_count"] = int(item.get("success_count", 0) or 0) + 1
            item["last_success_ts"] = now
            item["plan_steps"] = [_task_to_dict(step) for step in steps]
            item["source"] = _clean(source) or _clean(item.get("source"))
            self._save()
            return dict(item)

        record = {
            "template_name": template_name,
            "query": _clean(query),
            "query_norm": query_norm,
            "success_count": 1,
            "last_success_ts": now,
            "plan_steps": [_task_to_dict(step) for step in steps],
            "source": _clean(source),
        }
        self.items.append(record)
        self._save()
        return dict(record)

    def _build_plan(self, item: Dict[str, Any], *, goal: str) -> Plan | None:
        raw_steps = list((item or {}).get("plan_steps", []) or [])
        steps = [_task_from_dict(step) for step in raw_steps if isinstance(step, dict)]
        if not steps:
            return None
        return Plan(goal=_clean(goal), steps=steps, levels=[[step] for step in steps])


__all__ = [
    "DEFAULT_COMPUTER_LEARNING_PATH",
    "LearnedComputerTaskStore",
    "normalize_computer_query",
]
