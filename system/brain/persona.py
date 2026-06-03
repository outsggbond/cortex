# -*- coding: utf-8 -*-
"""Persona helpers to keep assistant identity/style consistent."""

from __future__ import annotations

import os
import re


PERSONA_NAME = (os.environ.get("ASSISTANT_PERSONA_NAME", "Tina") or "Tina").strip() or "Tina"


def persona_identity_reply() -> str:
    return f"\u6211\u662f {PERSONA_NAME}\uff0c\u4e00\u4e2a\u5b66\u4e60\u578b AI \u52a9\u624b\u3002"


def persona_capability_reply() -> str:
    return (
        f"\u6211\u662f {PERSONA_NAME}\uff0c\u53ef\u4ee5\u966a\u4f60\u804a\u5929\uff0c"
        "\u4e5f\u80fd\u5e2e\u4f60\u5199\u4ee3\u7801\u3001\u6392\u67e5\u62a5\u9519\u3001\u6574\u7406\u6b65\u9aa4\u3002"
    )


def persona_prompt_reply() -> str:
    return (
        f"\u6211\u662f {PERSONA_NAME}\u3002"
        "\u4f60\u76f4\u63a5\u8bf4\u76ee\u6807\u3001\u62a5\u9519\u548c\u9650\u5236\u6761\u4ef6\uff0c"
        "\u6211\u7ed9\u4f60\u5177\u4f53\u6b65\u9aa4\u3002"
    )


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def is_identity_query(text: str) -> bool:
    compact = _compact(text)
    if not compact:
        return False
    return bool(
        re.search(
            r"(\u4f60.*(\u53eb|\u540d\u5b57)|\u4f60\u7684\u540d\u5b57|\u600e\u4e48\u79f0\u547c|\u4f60\u662f\u8c01|what'?syourname|whoareyou)",
            compact.lower(),
        )
    )


def is_capability_query(text: str) -> bool:
    compact = _compact(text).lower()
    if not compact:
        return False
    return bool(
        re.search(
            r"(\u4f60.*(\u4f1a\u4ec0\u4e48|\u80fd\u505a\u4ec0\u4e48|\u80fd\u5e72\u4ec0\u4e48|\u529f\u80fd|\u64c5\u957f)|whatcanyoudo|capabilit(y|ies))",
            compact,
        )
    )


def persona_rule_reply(text: str) -> str:
    cur = str(text or "").strip()
    if not cur:
        return ""
    compact = _compact(cur)
    low_compact = compact.lower()
    if is_identity_query(cur):
        return persona_identity_reply()
    if is_capability_query(cur):
        return persona_capability_reply()
    if low_compact in {"hi", "hello", "hey"}:
        return f"\u4f60\u597d\uff0c\u6211\u662f {PERSONA_NAME}\u3002\u6709\u4ec0\u4e48\u6211\u53ef\u4ee5\u5e2e\u4f60\uff1f"
    if any(k in compact for k in ("\u65e9\u4e0a\u597d", "\u65e9\u5b89", "\u4e0a\u5348\u597d")):
        return f"\u65e9\u4e0a\u597d\uff0c\u6211\u662f {PERSONA_NAME}\u3002\u4eca\u5929\u60f3\u5148\u5904\u7406\u4ec0\u4e48\uff1f"
    if any(k in compact for k in ("\u4e2d\u5348\u597d", "\u5348\u5b89", "\u5348\u597d")):
        return f"\u4e2d\u5348\u597d\uff0c\u6211\u662f {PERSONA_NAME}\u3002\u9700\u8981\u6211\u5e2e\u4f60\u5904\u7406\u4ec0\u4e48\uff1f"
    if any(k in compact for k in ("\u4e0b\u5348\u597d",)):
        return f"\u4e0b\u5348\u597d\uff0c\u6211\u662f {PERSONA_NAME}\u3002\u73b0\u5728\u8981\u63a8\u8fdb\u54ea\u4ef6\u4e8b\uff1f"
    if any(k in compact for k in ("\u665a\u4e0a\u597d", "\u665a\u5b89")):
        return f"\u665a\u4e0a\u597d\uff0c\u6211\u662f {PERSONA_NAME}\u3002\u6709\u4ec0\u4e48\u6211\u53ef\u4ee5\u5e2e\u4f60\u7684\u5417\uff1f"
    if any(k in compact for k in ("\u4f60\u597d", "\u60a8\u597d", "\u54c8\u55bd", "\u55e8")):
        return f"\u4f60\u597d\uff0c\u6211\u662f {PERSONA_NAME}\u3002\u5f88\u9ad8\u5174\u8ba4\u8bc6\u4f60\u3002"
    return ""


def normalize_persona_reply(user_text: str, reply: str) -> str:
    user_text = str(user_text or "").strip()
    text = str(reply or "").strip()
    if is_identity_query(user_text):
        return persona_identity_reply()
    if is_capability_query(user_text):
        return persona_capability_reply()
    if not text:
        return persona_prompt_reply()

    text = re.sub(r"\btina\b", "Tina", text, flags=re.IGNORECASE)

    # Known low-signal generic fallback.
    low_text = _compact(text).lower()
    if "icanhelprightnow:shareyourgoal,currenterror,andconstraints" in low_text:
        return persona_prompt_reply()
    if "tellmeyourgoalandconstraints,andiwillprovideanexecutableplan" in low_text:
        return persona_prompt_reply()

    # Block romantic/persona-drift responses from polluted memory.
    romantic_markers = (
        "\u5fc3\u91cc\u53ea\u88c5\u5f97\u4e0b\u4f60\u4e00\u4e2a",
        "\u8ba8\u538c\uff0c\u8fd9\u4e48\u5feb\u5c31\u5fd8\u5566",
        "\u6c38\u8fdc\u662f\u4f60\u7684",
        "\u6211\u7231\u7684\u662f\u4f60\u7684\u7075\u9b42",
        "\u5934\u53f7\u7c89\u4e1d",
        "\u6700\u559c\u6b22\u4f60",
        "\u8865\u7ed9\u7ad9",
    )
    if any(marker in text for marker in romantic_markers):
        return persona_capability_reply()

    # Normalize old assistant naming to Tina.
    if any(marker in text for marker in ("\u5c0f\u52a9", "\u672c\u5730\u667a\u80fd\u52a9\u624b", "\u79bb\u7ebf\u52a9\u624b")):
        return persona_identity_reply()

    return text
