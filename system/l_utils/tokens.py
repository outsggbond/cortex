# -*- coding: utf-8 -*-
"""Token extraction utilities — pure functions for parsing structured tokens from text.

Extracted from system/chat_v2/pipeline.py to eliminate the "god class" coupling.
These functions have zero dependencies and belong in the L1 infrastructure layer.
"""

from __future__ import annotations

import re
from typing import List

# Pre-compiled regex patterns (shared with pipeline.py constants)
_PATH_TOKEN_RE = re.compile(r"(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+(?:\.[A-Za-z0-9_]+)?")
_CODE_TOKEN_RE = re.compile(r"\b[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+\b(?:\([^)]*\))?")
_TIME_TOKEN_RE = re.compile(
    r"\b\d+\s*(?:ms|msec|s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|day|days)\b"
    r"|\d+\s*(?:分钟|小时|天)"
)
_TOKEN_RE = re.compile(r"[a-z0-9_./\\-]+|[一-鿿]")
_EN_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "how", "i", "if", "in", "is", "it", "its", "me", "my", "of", "on",
    "or", "our", "that", "the", "their", "them", "they", "this", "to",
    "up", "use", "using", "we", "what", "when", "where", "which", "who",
    "why", "with", "you", "your",
}
_CJK_STOPWORDS: set[str] = set("的了在是和就都而及与着或一个我你他她它们")


def _contains_cjk(text: str) -> bool:
    import re
    return re.search(r"[一-鿿]", str(text or "")) is not None


def normalize_structured_token(token: str) -> str:
    """Normalize a raw token: strip, lowercase, slash-normalize."""
    node = str(token or "").strip("`'\"()[]{}<> ,;:").lower()
    return node.replace("\\", "/")


def path_tokens(text: str) -> set[str]:
    """Extract file-path-like tokens from text."""
    out: set[str] = set()
    for token in _PATH_TOKEN_RE.findall(str(text or "")):
        node = normalize_structured_token(token)
        if not node:
            continue
        if "/" not in node and not re.match(r"^[a-z]:", node):
            tail = node.rsplit(".", 1)[-1]
            if "_" in tail or len(tail) > 5:
                continue
        if "/" in node or re.match(r"^[a-z]:", node) or re.search(r"\.[a-z0-9_]{1,10}$", node):
            out.add(node)
    return out


def code_tokens(text: str) -> set[str]:
    """Extract code identifier tokens (dotted names) from text."""
    out: set[str] = set()
    for token in _CODE_TOKEN_RE.findall(str(text or "").lower()):
        node = re.sub(r"\([^)]*\)$", "", str(token or "").strip())
        if "." in node:
            out.add(node)
    return out


def time_tokens(text: str) -> set[str]:
    """Extract time/duration tokens from text, normalizing units."""
    out: set[str] = set()
    for token in _TIME_TOKEN_RE.findall(str(text or "").lower()):
        node = re.sub(r"\s+", " ", str(token or "").strip())
        if node:
            node = re.sub(r"\bmins?\b", "minute", node)
            node = re.sub(r"\bminutes\b", "minute", node)
            node = re.sub(r"\bhours?\b", "hour", node)
            node = re.sub(r"\bhrs?\b", "hour", node)
            node = re.sub(r"\bsecs?\b", "second", node)
            node = re.sub(r"\bseconds\b", "second", node)
            node = re.sub(r"\bdays\b", "day", node)
            out.add(node)
    return out


def match_key(token: str) -> str:
    """Normalize a token for salient-key matching."""
    cur = str(token or "").lower().strip()
    if not cur:
        return ""
    if any(ch in cur for ch in "./\\"):
        return normalize_structured_token(cur)
    if _contains_cjk(cur):
        return cur
    if len(cur) <= 5:
        return cur
    return cur[:6]


def salient_tokens(text: str) -> List[str]:
    """Extract meaningful tokens (non-stopword, non-trivial length)."""
    out: List[str] = []
    for token in _TOKEN_RE.findall(str(text or "").lower()):
        cur = str(token or "").strip("`'\"()[]{}<> .,;:!?")
        if not cur:
            continue
        if _contains_cjk(cur):
            if len(cur) == 1 and cur in _CJK_STOPWORDS:
                continue
            out.append(cur)
            continue
        if len(cur) <= 1:
            continue
        if cur in _EN_STOPWORDS:
            continue
        out.append(cur)
    return out


def salient_keys(text: str) -> set[str]:
    """Convert text to a set of salient match keys."""
    out = {match_key(token) for token in salient_tokens(text)}
    out.discard("")
    return out


__all__ = [
    "contains_cjk",
    "normalize_structured_token",
    "path_tokens",
    "code_tokens",
    "time_tokens",
    "match_key",
    "salient_tokens",
    "salient_keys",
]

# Module-level alias for backwards compatibility
contains_cjk = _contains_cjk
