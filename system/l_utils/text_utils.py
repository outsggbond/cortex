# -*- coding: utf-8 -*-
"""Text utility functions — shared helpers extracted from god classes.

Extracted from system/chat_v2/pipeline.py and system/chat_v2/agent.py.
These are pure functions with zero domain dependencies.
"""

from __future__ import annotations

import re


def contains_cjk(text: str) -> bool:
    """Return True if the text contains CJK (Chinese/Japanese/Korean) characters."""
    return bool(re.search(r"[一-鿿]", str(text or "")))


def snippet(text: str, limit: int = 320) -> str:
    """Truncate text to *limit* characters with ellipsis."""
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def is_valid_reply(text: str, min_len: int = 2) -> bool:
    """Check if a reply text meets minimum quality standards."""
    clean = str(text or "").strip()
    return len(clean) >= min_len


def is_valid_rag_reply(text: str, min_len: int = 4) -> bool:
    """Check if a RAG-generated reply meets minimum length."""
    return len(str(text or "").strip()) >= min_len


__all__ = [
    "contains_cjk",
    "snippet",
    "is_valid_reply",
    "is_valid_rag_reply",
]
