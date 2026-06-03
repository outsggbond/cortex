# -*- coding: utf-8 -*-
"""Intent classification and deterministic policy replies for v2 chat."""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from system.brain.persona import (
    persona_capability_reply,
    persona_identity_reply,
)


class ChatIntent(str, Enum):
    IDENTITY = "identity"
    CAPABILITY = "capability"
    GREETING = "greeting"
    TASK = "task"


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())


def _looks_like_explicit_task(text: str) -> bool:
    raw = str(text or "").strip()
    compact = _compact(raw).lower()
    if not compact:
        return False
    explicit_prefixes = (
        "template",
        "runtemplate",
        "resumelastcomputertask",
        "resumecomputertask",
        "continuelastcomputertask",
        "continuecomputertask",
        "browseropen",
        "browserclick",
        "browsertype",
        "browsertext",
        "browserextract",
        "browserwaittext",
        "browserscreenshot",
        "browserclose",
        "openurl",
        "clicktext",
        "clickcontrol",
        "listwindows",
        "listcontrols",
        "focuswindow",
        "launch",
        "hotkey",
        "screenshot",
        "ocrscreen",
        "read",
        "write",
        "append",
        "touch",
        "list",
        "exists",
        "mkdir",
        "copy",
        "move",
        "rename",
        "delete",
        "remove",
        "search",
        "find",
        "run",
        "chmod",
        "rollback",
    )
    if compact.startswith(explicit_prefixes):
        return True
    explicit_fragments = (
        "thenbrowser",
        "thenclick",
        "thenfocuswindow",
        "thenlistcontrols",
        "thentype",
        "intocontrol",
        "inwindow",
        "verifywindow",
        "verifytext",
        "verifychange",
        "retry",
        "retries",
        "controltype",
    )
    return any(fragment in compact for fragment in explicit_fragments)


def _looks_like_natural_language_computer_goal(text: str) -> bool:
    compact = _compact(text).lower()
    if not compact:
        return False
    action_tokens = (
        "send",
        "message",
        "open",
        "launch",
        "search",
        "find",
        "type",
        "input",
        "click",
        "\u53d1\u9001",
        "\u53d1\u6d88\u606f",
        "\u53d1\u4e00\u53e5",
        "\u53d1\u4e2a",
        "\u53d1\u6761",
        "\u53d1\u7ed9",
        "\u6253\u5f00",
        "\u641c\u7d22",
        "\u67e5\u627e",
        "\u8f93\u5165",
        "\u70b9\u51fb",
        "\u70b9\u5f00",
    )
    object_tokens = (
        "qq",
        "wechat",
        "wecom",
        "feishu",
        "contact",
        "chat",
        "message",
        "window",
        "desktop",
        "\u8054\u7cfb\u4eba",
        "\u804a\u5929",
        "\u6d88\u606f",
        "\u5bf9\u8bdd",
        "\u7a97\u53e3",
        "\u684c\u9762",
    )
    return any(token in compact for token in action_tokens) and any(token in compact for token in object_tokens)


def classify_intent(text: str) -> ChatIntent:
    compact = _compact(text).lower()
    if not compact:
        return ChatIntent.TASK
    if _looks_like_explicit_task(text):
        return ChatIntent.TASK
    if _looks_like_natural_language_computer_goal(text):
        return ChatIntent.TASK
    if re.search(
        r"(\u4f60.*(\u53eb|\u540d\u5b57)|\u4f60\u7684\u540d\u5b57|\u600e\u4e48\u79f0\u547c|\u4f60\u662f\u8c01|what'?syourname|whoareyou)",
        compact,
    ):
        return ChatIntent.IDENTITY
    if re.search(
        r"(\u4f60.*(\u4f1a\u4ec0\u4e48|\u80fd\u505a\u4ec0\u4e48|\u80fd\u5e72\u4ec0\u4e48|\u529f\u80fd|\u64c5\u957f)|whatcanyoudo|capabilit(y|ies))",
        compact,
    ):
        return ChatIntent.CAPABILITY
    if compact in {"hi", "hello", "hey"}:
        return ChatIntent.GREETING
    if any(
        key in compact
        for key in (
            "\u4f60\u597d",
            "\u60a8\u597d",
            "\u54c8\u55bd",
            "\u55e8",
            "\u65e9\u4e0a\u597d",
            "\u65e9\u5b89",
            "\u4e0a\u5348\u597d",
            "\u4e2d\u5348\u597d",
            "\u5348\u5b89",
            "\u5348\u597d",
            "\u4e0b\u5348\u597d",
            "\u665a\u4e0a\u597d",
            "\u665a\u5b89",
        )
    ):
        return ChatIntent.GREETING
    return ChatIntent.TASK


def policy_reply(intent: ChatIntent, text: str) -> Optional[str]:
    if intent == ChatIntent.IDENTITY:
        return persona_identity_reply()
    if intent == ChatIntent.CAPABILITY:
        return persona_capability_reply()
    if intent == ChatIntent.GREETING:
        compact = _compact(text)
        if any(k in compact for k in ("\u65e9\u4e0a\u597d", "\u65e9\u5b89", "\u4e0a\u5348\u597d")):
            return "\u65e9\u4e0a\u597d\uff0c\u6211\u662f Tina\u3002\u4eca\u5929\u60f3\u5148\u5904\u7406\u4ec0\u4e48\uff1f"
        if any(k in compact for k in ("\u4e2d\u5348\u597d", "\u5348\u5b89", "\u5348\u597d")):
            return "\u4e2d\u5348\u597d\uff0c\u6211\u662f Tina\u3002\u9700\u8981\u6211\u5e2e\u4f60\u5904\u7406\u4ec0\u4e48\uff1f"
        if any(k in compact for k in ("\u4e0b\u5348\u597d",)):
            return "\u4e0b\u5348\u597d\uff0c\u6211\u662f Tina\u3002\u73b0\u5728\u8981\u63a8\u8fdb\u54ea\u4ef6\u4e8b\uff1f"
        if any(k in compact for k in ("\u665a\u4e0a\u597d", "\u665a\u5b89")):
            return "\u665a\u4e0a\u597d\uff0c\u6211\u662f Tina\u3002\u6709\u4ec0\u4e48\u6211\u53ef\u4ee5\u5e2e\u4f60\u7684\u5417\uff1f"
        return "\u4f60\u597d\uff0c\u6211\u662f Tina\u3002\u6709\u4ec0\u4e48\u6211\u53ef\u4ee5\u5e2e\u4f60\uff1f"
    return None
