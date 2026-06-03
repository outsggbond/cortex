# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class _CompiledRoute:
    name: str
    patterns: List[re.Pattern[str]]
    response: str


_DEFAULT_ROUTE_DATA = [
    {
        "name": "greeting",
        "patterns": [
            r"\b(hi|hello|hey)\b",
            r"^(\u4f60\u597d|\u60a8\u597d|\u65e9\u4e0a\u597d|\u4e2d\u5348\u597d|\u4e0b\u5348\u597d|\u665a\u4e0a\u597d)[!\uff01\u3002?\uff1f]?$",
        ],
        "response": "\u4f60\u597d\uff0c\u6211\u662f Tina\u3002\u6709\u4ec0\u4e48\u6211\u53ef\u4ee5\u5e2e\u4f60\uff1f",
    },
    {
        "name": "identity",
        "patterns": [
            r"\u4f60.*(\u540d\u5b57|\u53eb.*\u4ec0\u4e48|\u662f\u8c01|\u600e\u4e48\u79f0\u547c)",
            r"(what'?s your name|who are you)",
        ],
        "response": "\u6211\u662f Tina\uff0c\u4e00\u4e2a\u5b66\u4e60\u578b AI \u52a9\u624b\u3002",
    },
    {
        "name": "capability",
        "patterns": [
            r"\u4f60.*(\u4f1a\u4ec0\u4e48|\u80fd\u505a\u4ec0\u4e48|\u529f\u80fd|\u64c5\u957f)",
            r"(what can you do|capabilities)",
        ],
        "response": "\u6211\u662f Tina\uff0c\u53ef\u4ee5\u966a\u4f60\u804a\u5929\uff0c\u4e5f\u80fd\u5e2e\u4f60\u5199\u4ee3\u7801\u3001\u6392\u67e5\u62a5\u9519\u3001\u6574\u7406\u6b65\u9aa4\u3002",
    },
    {
        "name": "thanks",
        "patterns": [
            r"(\u8c22\u8c22|\u611f\u8c22|\u8f9b\u82e6\u4e86)",
            r"\b(thanks|thank you|thx)\b",
        ],
        "response": "\u4e0d\u5ba2\u6c14\uff0c\u6211\u662f Tina\u3002\u4f60\u53ef\u4ee5\u7ee7\u7eed\u8bf4\u9700\u6c42\u3002",
    },
    {
        "name": "hostile",
        "patterns": [
            r"(\u50bb\u903c|\u5783\u573e|\u5e9f\u7269)",
            r"\b(stupid|idiot|trash)\b",
        ],
        "response": "\u6211\u662f Tina\u3002\u4f60\u76f4\u63a5\u8bf4\u95ee\u9898\uff0c\u6211\u7ed9\u4f60\u53ef\u6267\u884c\u7684\u6b65\u9aa4\u3002",
    },
]

_ROUTE_PATH = Path(os.environ.get("INTENT_ROUTE_FILE", "config/intent_routes.json"))
_CACHE: List[_CompiledRoute] = []
_CACHE_MTIME: float = -1.0


def _compile_routes(route_data: list[dict]) -> List[_CompiledRoute]:
    compiled: List[_CompiledRoute] = []
    for item in route_data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip() or "route"
        response = str(item.get("response", "")).strip()
        patterns_raw = item.get("patterns", [])
        if not response or not isinstance(patterns_raw, list):
            continue
        patterns: List[re.Pattern[str]] = []
        for p in patterns_raw:
            text = str(p).strip()
            if not text:
                continue
            try:
                patterns.append(re.compile(text, re.IGNORECASE))
            except re.error:
                continue
        if patterns:
            compiled.append(_CompiledRoute(name=name, patterns=patterns, response=response))
    return compiled


def _load_route_data() -> list[dict]:
    if _ROUTE_PATH.exists():
        try:
            raw = json.loads(_ROUTE_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return raw
        except Exception:
            pass
    return list(_DEFAULT_ROUTE_DATA)


def _get_routes() -> List[_CompiledRoute]:
    global _CACHE, _CACHE_MTIME
    mtime = -1.0
    if _ROUTE_PATH.exists():
        try:
            mtime = float(_ROUTE_PATH.stat().st_mtime)
        except Exception:
            mtime = -1.0
    if _CACHE and mtime == _CACHE_MTIME:
        return _CACHE
    route_data = _load_route_data()
    _CACHE = _compile_routes(route_data)
    _CACHE_MTIME = mtime
    return _CACHE


def route_intent(message: str) -> Optional[str]:
    text = (message or "").strip()
    if not text:
        return None
    for route in _get_routes():
        for pattern in route.patterns:
            if pattern.search(text):
                return route.response
    return None
