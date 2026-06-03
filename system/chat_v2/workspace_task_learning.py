# -*- coding: utf-8 -*-
"""Persistent learned workspace-task plans for local deterministic replay."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from system.core.planner import Plan, Task


DEFAULT_WORKSPACE_LEARNING_PATH = "artifacts/memory/workspace_task_learnings.json"

_WORKSPACE_TASK_NAMES = {
    "read_file",
    "list_dir",
    "check_exists",
    "search_text",
    "search_files",
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _learning_path(path: str = "") -> Path:
    raw = _clean(path) or _clean(os.environ.get("WORKSPACE_TASK_LEARNING_PATH")) or DEFAULT_WORKSPACE_LEARNING_PATH
    return Path(raw)


def normalize_workspace_query(text: str) -> str:
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


class LearnedWorkspaceTaskStore:
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

    def find_by_query(self, query: str, *, min_successes: int = 1) -> Dict[str, Any] | None:
        query_norm = normalize_workspace_query(query)
        if not query_norm:
            return None
        for item in self.items:
            if _clean(item.get("query_norm")) != query_norm:
                continue
            if int(item.get("success_count", 0) or 0) < max(1, int(min_successes)):
                continue
            return dict(item)
        return None

    def build_plan_for_query(self, query: str, *, min_successes: int = 1, goal: str = "") -> Plan | None:
        item = self.find_by_query(query, min_successes=min_successes)
        if item is None:
            return None
        return self._build_plan(item, goal=goal or query)

    def remember_success(self, *, query: str, plan: Plan, execution: Any, source: str = "") -> Dict[str, Any] | None:
        query_norm = normalize_workspace_query(query)
        steps = list(getattr(plan, "steps", []) or [])
        if not query_norm or not steps:
            return None
        if not all(_clean(getattr(step, "name", "")) in _WORKSPACE_TASK_NAMES for step in steps):
            return None
        rows = list(getattr(execution, "step_results", []) if execution is not None else [])
        if len(rows) < len(steps):
            return None
        for idx, step in enumerate(steps):
            row = rows[idx] if idx < len(rows) and isinstance(rows[idx], dict) else {}
            if _clean(row.get("name")) != _clean(getattr(step, "name", "")):
                return None
            if not bool(row.get("ok", False)):
                return None

        template_name = "workspace_" + hashlib.sha1(query_norm.encode("utf-8")).hexdigest()[:12]
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
    "DEFAULT_WORKSPACE_LEARNING_PATH",
    "LearnedWorkspaceTaskStore",
    "normalize_workspace_query",
]
