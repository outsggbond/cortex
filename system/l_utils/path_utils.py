# -*- coding: utf-8 -*-
"""Path extraction utilities — pure functions for detecting and resolving file paths from text.

Extracted from system/chat_v2/agent.py to eliminate the "god class" coupling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

# Regex patterns (shared with agent.py constants)
_PATH_TOKEN_RE = re.compile(r"(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+(?:\.[A-Za-z0-9_]+)?")
_URL_RE = re.compile(r"https?://[^\s`\"'>)]+", flags=re.IGNORECASE)

# CJK constants for search patterns
_ZH_SEARCH = "搜"
_ZH_FIND = "找"
_ZH_SEARCH_SHORT = "搜"
_ZH_FIND_SHORT = "找"
_ZH_IN = "在"
_ZH_TO = "到"


def normalize_path(value: str) -> str:
    """Strip, clean, normalize slashes in a path-like string."""
    return str(value or "").strip().strip(",;:()[]{}<>").replace("\\", "/")


def looks_like_path(value: str) -> bool:
    """Heuristic: does the string look like a file path?"""
    text = str(value or "").strip()
    if not text:
        return False
    if "/" in text or "\\" in text:
        return True
    return "." in text and any(ch.isalpha() for ch in text)


def looks_like_filename(value: str) -> bool:
    """Check if value looks like a filename (has extension)."""
    text = str(value or "").strip()
    return looks_like_path(text) or bool(re.search(r"\.[A-Za-z0-9_]+$", text))


def resolve_path(path: str, project_root: str = ".") -> Path:
    """Resolve a path relative to project root."""
    base = Path(str(project_root or ".")).resolve()
    candidate = Path(str(path or "").replace("\\", "/"))
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def extract_url(text: str) -> str:
    """Extract the first URL from a string."""
    match = _URL_RE.search(str(text or ""))
    if not match:
        return ""
    return str(match.group(0) or "").strip()


def extract_paths(text: str, project_root: str = ".") -> List[str]:
    """Extract file-path-like tokens from query text.

    Scans backtick, double-quote, and single-quote delimiters first,
    then falls back to bare-path regex.
    """
    hits: List[str] = []
    seen: set[str] = set()
    raw_text = str(text or "")

    # Quoted paths
    for pat in (r"`([^`]+)`", r'"([^"]+)"', r"'([^']+)'"):
        for raw in re.findall(pat, raw_text):
            candidate = normalize_path(raw)
            if looks_like_path(candidate) and candidate not in seen:
                seen.add(candidate)
                hits.append(candidate)

    # Bare regex paths
    for raw in _PATH_TOKEN_RE.findall(raw_text):
        candidate = normalize_path(raw)
        if not looks_like_path(candidate):
            continue
        if candidate in seen:
            continue
        # Exclude common command words that look like paths
        if candidate.lower() in {"read", "write", "append", "list", "find", "search", "run", "tests"}:
            continue
        seen.add(candidate)
        hits.append(candidate)

    return hits


def pick_path(text: str, project_root: str = ".") -> str:
    """Pick the most relevant path from extracted paths.

    Prefers existing paths, then shortest paths.
    """
    paths = extract_paths(text, project_root)
    if not paths:
        return ""
    existing = [p for p in paths if resolve_path(p, project_root).exists()]
    if existing:
        existing.sort(key=len)
        return existing[0]
    paths.sort(key=len)
    return paths[0]


def extract_search_pattern(text: str) -> str | None:
    """Extract a search pattern from natural language text.

    Handles: quoted patterns, "search X in Y", "find X", "where X",
    and CJK equivalents (搜索/查找 in 目录).
    """
    raw_text = str(text or "").strip()

    # Quoted patterns
    for pat in (r'"([^"]+)"', r"'([^']+)'", r"`([^`]+)`"):
        matches = [m.strip() for m in re.findall(pat, raw_text) if str(m or "").strip()]
        if matches:
            for candidate in matches:
                if not looks_like_path(candidate):
                    return candidate
            return matches[0]

    # English patterns
    en_patterns = [
        r"(?:search|find|grep|look for)\s+(.+?)(?:\s+in\s+(.+))?$",
    ]
    for pat in en_patterns:
        match = re.search(pat, raw_text, flags=re.IGNORECASE)
        if match:
            candidate = str(match.group(1) or "").strip().strip(" ?!.,;:")
            if candidate:
                return candidate

    # CJK patterns
    cjk_patterns = [
        rf"(?:{_ZH_SEARCH}|{_ZH_FIND}|{_ZH_SEARCH_SHORT}|{_ZH_FIND_SHORT})\s*(.+?)(?:\s*(?:{_ZH_IN}|{_ZH_TO})\s*(.+))?$",
    ]
    for pat in cjk_patterns:
        match = re.search(pat, raw_text, flags=re.IGNORECASE)
        if match:
            candidate = str(match.group(1) or "").strip().strip(" ?!.,;:")
            if candidate:
                return candidate

    # "where X" pattern
    low = raw_text.lower()
    if " where " in f" {low} ":
        parts = re.split(r"\bwhere\b", raw_text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            candidate = parts[1].strip().strip(" ?!.,;:")
            if candidate:
                return candidate

    return None


__all__ = [
    "normalize_path",
    "looks_like_path",
    "looks_like_filename",
    "resolve_path",
    "extract_url",
    "extract_paths",
    "pick_path",
    "extract_search_pattern",
]
