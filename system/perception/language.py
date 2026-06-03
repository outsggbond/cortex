# -*- coding: utf-8 -*-
"""Language detection & CJK text support."""

from __future__ import annotations

import re
from typing import Optional


def detect_language(text: str) -> str:
    """Detect the primary language of a text string.

    Returns 'zh' (Chinese), 'ja' (Japanese), 'ko' (Korean),
    'en' (English), 'mixed', or 'other'.
    """
    if not text or not text.strip():
        return "en"

    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    hiragana = sum(1 for ch in text if "぀" <= ch <= "ゟ")
    katakana = sum(1 for ch in text if "゠" <= ch <= "ヿ")
    hangul = sum(1 for ch in text if "가" <= ch <= "힯")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())

    total = max(1, len(text))

    if cjk / total > 0.2:
        if hiragana + katakana > cjk * 0.3:
            return "ja"
        return "zh"
    if (hiragana + katakana) / total > 0.1:
        return "ja"
    if hangul / total > 0.15:
        return "ko"
    if latin / total > 0.7:
        return "en"
    if cjk / total > 0.05:
        return "mixed"
    return "other"


def has_cjk(text: str) -> bool:
    """Check if text contains any CJK characters."""
    return bool(re.search(r"[一-鿿぀-ゟ゠-ヿ가-힯]", text))


def tokenize_chinese(text: str) -> list[str]:
    """Tokenize Chinese text using jieba (if available)."""
    try:
        import jieba
        return [t.strip() for t in jieba.cut(text) if t.strip()]
    except ImportError:
        # Fallback: character-level for CJK
        tokens = []
        for ch in text:
            if "一" <= ch <= "鿿":
                tokens.append(ch)
            elif not tokens:
                tokens.append(ch)
            else:
                tokens[-1] += ch
        return [t for t in tokens if t.strip()]


def to_pinyin(text: str) -> str:
    """Convert Chinese text to pinyin (if pypinyin available)."""
    try:
        from pypinyin import lazy_pinyin
        return " ".join(lazy_pinyin(text))
    except ImportError:
        return text
