# -*- coding: utf-8 -*-
"""Thread context tracking — follow-up detection, context labeling, and conflict detection.

Extracted from ChatPipeline (pipeline.py) to improve cohesion.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from system.l_utils.text_utils import contains_cjk
from system.l_utils.tokens import code_tokens, match_key, normalize_structured_token, path_tokens, salient_keys, time_tokens

from .intents import ChatIntent, classify_intent
from .router import RouteKind

# Regex patterns
_PATH_TOKEN_RE = re.compile(r"(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+(?:\.[A-Za-z0-9_]+)?")
_CODE_TOKEN_RE = re.compile(r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+\b(?:\([^)]*\))?")
_TIME_TOKEN_RE = re.compile(
    r"\b\d+\s*(?:ms|msec|s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|day|days)\b"
    r"|\d+\s*(?:分钟|小时|天)"
)

_CRITICAL_LABEL_KEYS = {
    "target path", "priority", "constraint", "runtime", "error", "goal",
}
_CRITICAL_LABEL_KEYS_CJK = {
    "目标路径", "优先级", "约束", "运行环境", "报错", "目标",
}


class ThreadContextTracker:
    """Tracks multi-turn conversation context: follow-ups, clarification state, thread conflicts."""

    def __init__(self) -> None:
        pass

    # -- context labeling ---------------------------------------------------------

    def default_context_label(self, route_kind: str, *, cjk: bool) -> str:
        kind = str(route_kind or "").strip()
        if kind == str(RouteKind.WORKSPACE_ACTION.value):
            return "目标路径" if cjk else "Target path"
        if kind == str(RouteKind.PROJECT_QA.value):
            return "关注点" if cjk else "Focus"
        if kind == str(RouteKind.REASONING.value):
            return "补充信息" if cjk else "Detail"
        if kind == str(RouteKind.FACTUAL_QA.value):
            return "上下文" if cjk else "Context"
        return "补充信息" if cjk else "Additional context"

    def context_label(self, route_kind: str, original_query: str, answer: str) -> str:
        text = f"{original_query} {answer}".strip()
        low = text.lower()
        cjk = contains_cjk(text)
        kind = str(route_kind or "").strip()
        if kind == str(RouteKind.WORKSPACE_ACTION.value):
            if re.search(r"(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+(?:\.[A-Za-z0-9_]+)?", str(answer or "").strip()):
                return "目标路径" if cjk else "Target path"
            return "细节" if cjk else "Detail"
        if kind == str(RouteKind.PROJECT_QA.value):
            return "关注点" if cjk else "Focus"
        if kind == str(RouteKind.REASONING.value):
            if any(t in low for t in ("minute", "minutes", "hour", "hours", "day", "days", "window", "deadline", "time", "timeout", "分钟", "小时", "窗口", "时限")):
                return "约束" if cjk else "Constraint"
            if any(t in low for t in ("rollback", "safety", "risk", "speed", "cost", "priority", "回滚", "安全", "风险", "速度", "成本", "优先")):
                return "优先级" if cjk else "Priority"
            if any(t in low for t in ("goal", "outcome", "target", "目标", "结果")):
                return "目标" if cjk else "Goal"
            return "补充信息" if cjk else "Detail"
        if kind == str(RouteKind.FACTUAL_QA.value):
            if any(t in low for t in ("python", "javascript", "typescript", "node", "java", "go", "rust", "c#", "php")):
                return "运行环境" if cjk else "Runtime"
            if any(t in low for t in ("file", "string", "http", "response", "api", "文件", "字符串", "响应")):
                return "上下文" if cjk else "Context"
            if any(t in low for t in ("error", "traceback", "exception", "报错", "异常")):
                return "报错" if cjk else "Error"
            return "上下文" if cjk else "Context"
        return self.default_context_label(kind, cjk=cjk)

    # -- assistant turn helpers ---------------------------------------------------

    def last_assistant_turn(self, history: Sequence[Any]) -> Any | None:
        turns = list(history or [])
        if not turns:
            return None
        last = turns[-1]
        if str(getattr(last, "role", "") or "").strip() == "assistant":
            return last
        for turn in reversed(turns):
            if str(getattr(turn, "role", "") or "").strip() == "assistant":
                return turn
        return None

    # -- thread context normalization --------------------------------------------

    def normalize_thread_context(self, raw: Any) -> Dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        original_query = str(raw.get("original_query", "") or "").strip()
        route_kind = str(raw.get("route_kind", "") or "").strip()
        if not original_query:
            return None
        items: List[Dict[str, str]] = []
        seen: set[str] = set()
        for row in list(raw.get("items", []) or []):
            if not isinstance(row, dict):
                continue
            value = str(row.get("value", "") or "").strip()
            if not value:
                continue
            key = re.sub(r"\s+", " ", value).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            label = str(row.get("label", "") or "").strip()
            if not label:
                label = self.context_label(route_kind, original_query, value)
            items.append({"label": label, "value": value})
        return {"original_query": original_query, "route_kind": route_kind, "items": items}

    def assistant_thread_context(self, history: Sequence[Any]) -> Dict[str, Any] | None:
        last = self.last_assistant_turn(history)
        if last is None:
            return None
        meta = dict(getattr(last, "metadata", {}) or {})
        thread = self.normalize_thread_context(meta.get("thread_context"))
        if thread is not None:
            return thread
        if str(getattr(last, "source", "") or "").strip() == "clarify":
            raw = {
                "original_query": str(meta.get("original_query", "") or "").strip(),
                "route_kind": str(meta.get("route_kind", "") or "").strip(),
                "items": list(meta.get("items", []) or []),
            }
            return self.normalize_thread_context(raw)
        return None

    def pending_clarification(self, history: Sequence[Any]) -> Dict[str, Any] | None:
        last = self.last_assistant_turn(history)
        if last is None:
            return None
        if str(getattr(last, "source", "") or "").strip() != "clarify":
            return None
        thread = self.assistant_thread_context(history)
        if thread is None:
            return None
        meta = dict(getattr(last, "metadata", {}) or {})
        meta["thread_context"] = thread
        meta["original_query"] = str(thread.get("original_query", "") or "").strip()
        meta["route_kind"] = str(thread.get("route_kind", "") or "").strip()
        return meta

    # -- task / context detection -------------------------------------------------

    def looks_like_new_task(self, text: str) -> bool:
        cur = str(text or "").strip()
        if not cur:
            return False
        if classify_intent(cur) != ChatIntent.TASK:
            return True
        low = cur.lower()
        if "?" in cur or "？" in cur:
            return True
        if re.match(r"^(?:how|what|why|where|who|which|can|could|should|would|do|does|did)\b", low):
            return True
        if re.match(r"^(?:read|open|write|run|search|find|list|show|inspect|grep|pytest)\b", low):
            return True
        if re.match(r"^(?:读取|打开|写入|运行|搜索|查找|列出)\b", cur):
            return True
        return False

    def looks_like_context_update(self, text: str) -> bool:
        cur = str(text or "").strip()
        if not cur:
            return False
        compact = re.sub(r"\s+", "", cur).lower()
        if compact in {"ok", "okay", "thanks", "thankyou", "thx", "gotit", "cool", "understood", "好", "好的", "谢谢", "收到", "明白", "知道了"}:
            return False
        if self.looks_like_new_task(cur):
            return False
        low = cur.lower()
        if len(cur) <= 80:
            return True
        if re.match(r"^(?:also|and|plus|with|without|from|for|using|priority|focus|target|it'?s|its|under|around)\b", low):
            return True
        if re.match(r"^(?:补充|另外|还有|优先|路径|文件|从|用|回滚|风险|时限)", cur):
            return True
        return False

    # -- thread context mutation --------------------------------------------------

    def append_thread_context(self, thread: Dict[str, Any], answer: str) -> Dict[str, Any]:
        base = self.normalize_thread_context(thread) or {}
        original_query = str(base.get("original_query", "") or "").strip()
        route_kind = str(base.get("route_kind", "") or "").strip()
        items = list(base.get("items", []) or [])
        value = str(answer or "").strip()
        if not original_query or not value:
            return dict(base)
        key = re.sub(r"\s+", " ", value).strip().lower()
        seen = {re.sub(r"\s+", " ", str(item.get("value", "") or "")).strip().lower() for item in items}
        if key in seen:
            return {"original_query": original_query, "route_kind": route_kind, "items": items}
        label = self.context_label(route_kind, original_query, value)
        items.append({"label": label, "value": value})
        return {"original_query": original_query, "route_kind": route_kind, "items": items}

    def compose_thread_query(self, thread: Dict[str, Any]) -> str:
        node = self.normalize_thread_context(thread) or {}
        original_query = str(node.get("original_query", "") or "").strip()
        if not original_query:
            return ""
        parts = [original_query]
        for item in list(node.get("items", []) or []):
            label = str(item.get("label", "") or "").strip()
            value = str(item.get("value", "") or "").strip()
            if not value:
                continue
            parts.append(f"{label}: {value}" if label else value)
        return "\n".join(parts).strip()

    def expand_followup_query(self, history: Sequence[Any], user_text: str) -> Tuple[str, Dict[str, Any] | None]:
        thread = self.assistant_thread_context(history)
        if thread is None:
            return str(user_text or "").strip(), None
        pending = self.pending_clarification(history)
        answer = str(user_text or "").strip()
        if not answer:
            return answer, thread
        if self.looks_like_new_task(answer):
            return answer, None
        if pending is None and not self.looks_like_context_update(answer):
            return answer, None
        updated = self.append_thread_context(thread, answer)
        merged = self.compose_thread_query(updated)
        if not merged:
            return answer, updated
        out = dict(updated)
        out["followup_answer"] = answer
        out["resolved_query"] = merged
        if pending is not None:
            out["pending"] = True
        return merged, out

    def attach_thread_context(
        self, metadata: Dict[str, Any] | None, thread_context: Dict[str, Any] | None,
        *, resolved_query: str = "",
    ) -> Dict[str, Any]:
        out = dict(metadata or {})
        node = self.normalize_thread_context(thread_context)
        if node is None:
            return out
        thread = dict(node)
        if resolved_query:
            thread["resolved_query"] = str(resolved_query or "").strip()
        out["thread_context"] = thread
        return out

    # -- thread context validation ------------------------------------------------

    def is_critical_context_label(self, label: str) -> bool:
        cur = str(label or "").strip().lower()
        if not cur:
            return False
        if cur in _CRITICAL_LABEL_KEYS:
            return True
        return any(mark in str(label or "") for mark in _CRITICAL_LABEL_KEYS_CJK)

    def thread_context_issue(
        self, answer: str, route_kind_str: str, thread_context: Dict[str, Any] | None,
        *, _workspace_action: str = "",
    ) -> Dict[str, str] | None:
        node = self.normalize_thread_context(thread_context)
        if node is None:
            return None
        if not _workspace_action:
            _workspace_action = str(RouteKind.WORKSPACE_ACTION.value)
        answer_paths = path_tokens(answer)
        answer_codes = code_tokens(answer)
        answer_times = time_tokens(answer)
        answer_keys = salient_keys(answer)

        for item in list(node.get("items", []) or []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "") or "").strip()
            value = str(item.get("value", "") or "").strip()
            if not value:
                continue
            value_paths = path_tokens(value)
            value_codes = code_tokens(value)
            value_times = time_tokens(value)
            value_keys = salient_keys(value)

            if value_paths:
                if answer_paths and not (answer_paths & value_paths):
                    return {"kind": "thread_path_conflict", "label": label, "value": value}
                if route_kind_str == _workspace_action and not answer_paths:
                    return {"kind": "thread_path_missing", "label": label, "value": value}
            if value_codes and answer_codes and not (answer_codes & value_codes):
                return {"kind": "thread_code_conflict", "label": label, "value": value}
            if value_times:
                if answer_times and not (answer_times & value_times):
                    return {"kind": "thread_time_conflict", "label": label, "value": value}
                if route_kind_str == str(RouteKind.REASONING.value) and not answer_times:
                    return {"kind": "thread_time_missing", "label": label, "value": value}
            if self.is_critical_context_label(label) and value_keys:
                if len(value_keys) <= 4 and not (answer_keys & value_keys):
                    return {"kind": "thread_context_missing", "label": label, "value": value}
        return None
