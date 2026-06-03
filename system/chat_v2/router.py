# -*- coding: utf-8 -*-
"""Lightweight route classifier for chat runtime v2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from .intents import ChatIntent


class RouteKind(str, Enum):
    POLICY = "policy"
    WORKSPACE_ACTION = "workspace_action"
    PROJECT_QA = "project_qa"
    REASONING = "reasoning"
    FACTUAL_QA = "factual_qa"
    GENERAL = "general"


@dataclass(frozen=True)
class RouteProfile:
    kind: RouteKind
    prefers_agent: bool = False
    prefers_reasoning: bool = False
    prefers_rag: bool = False
    prefers_memory: bool = True


_PATH_TOKEN_RE = re.compile(r"(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+(?:\.[A-Za-z0-9_]+)?")
_QUESTION_PATTERNS = (
    r"\?$",
    r"\bhow\b",
    r"\bwhat\b",
    r"\bwhere\b",
    r"\bwhich\b",
    r"\bwho\b",
    r"\u600e\u4e48",
    r"\u4ec0\u4e48",
    r"\u54ea\u91cc",
    r"\u54ea\u4e2a",
    r"\u4e3a\u4ec0\u4e48",
    r"\u5982\u4f55",
)
_REASONING_PATTERNS = (
    r"\bwhy\b",
    r"\breason\b",
    r"\bcause\b",
    r"\broot cause\b",
    r"\bshould\b",
    r"\btrade[\s-]?off\b",
    r"\bplan\b",
    r"\bstrategy\b",
    r"\bapproach\b",
    r"\broadmap\b",
    r"\brisk\b",
    r"\brollback\b",
    r"\bconstraint\b",
    r"\bcompare\b",
    r"\bchoose\b",
    r"\bbetter\b",
    r"\bsafer\b",
    r"\u4e3a\u4ec0\u4e48",
    r"\u539f\u56e0",
    r"\u5e94\u8be5",
    r"\u8ba1\u5212",
    r"\u7b56\u7565",
    r"\u65b9\u6848",
    r"\u53d6\u820d",
    r"\u6743\u8861",
    r"\u56de\u6eda",
    r"\u98ce\u9669",
    r"\u7ea6\u675f",
    r"\u6bd4\u8f83",
    r"\u5bf9\u6bd4",
)
_WORKSPACE_ACTION_PATTERNS = (
    r"\btemplate\b",
    r"\bresume\b",
    r"\bcontinue\b",
    r"\bread\b",
    r"\bopen\b",
    r"\bbrowse\b",
    r"\bvisit\b",
    r"\bnavigate\b",
    r"\blist\b",
    r"\bshow\b",
    r"\binspect\b",
    r"\bsearch\b",
    r"\bfind\b",
    r"\bgrep\b",
    r"\bwrite\b",
    r"\bedit\b",
    r"\bmodify\b",
    r"\bcreate\b",
    r"\bupdate\b",
    r"\brun\b",
    r"\bexecute\b",
    r"\bfocus\b",
    r"\blaunch\b",
    r"\bclick\b",
    r"\bdouble click\b",
    r"\bdrag\b",
    r"\bocr\b",
    r"\bscreenshot\b",
    r"\bhotkey\b",
    r"\btype\b",
    r"\binput\b",
    r"\bselector\b",
    r"\bdom\b",
    r"\bpytest\b",
    r"\btest\b",
    r"\u8bfb\u53d6",
    r"\u6253\u5f00",
    r"\u8bbf\u95ee",
    r"\u6d4f\u89c8",
    r"\u5217\u51fa",
    r"\u67e5\u770b",
    r"\u641c\u7d22",
    r"\u67e5\u627e",
    r"\u5199\u5165",
    r"\u4fee\u6539",
    r"\u521b\u5efa",
    r"\u66f4\u65b0",
    r"\u8fd0\u884c",
    r"\u6267\u884c",
    r"\u6d4b\u8bd5",
    r"\u622a\u56fe",
    r"\u805a\u7126",
    r"\u542f\u52a8",
    r"\u8f93\u5165",
    r"\u70b9\u51fb",
    r"\u62d6\u62fd",
    r"\u8bc6\u522b",
)
_WORKSPACE_OBJECT_PATTERNS = (
    r"\bcomputer task\b",
    r"\bdesktop task\b",
    r"\bbrowser task\b",
    r"\bfile\b",
    r"\bfolder\b",
    r"\bdirectory\b",
    r"\brepo\b",
    r"\brepository\b",
    r"\bworkspace\b",
    r"\bcodebase\b",
    r"\bsource\b",
    r"\bwindows?\b",
    r"\bbrowsers?\b",
    r"\btabs?\b",
    r"\bapps?\b",
    r"\bapplications?\b",
    r"\burls?\b",
    r"\bselectors?\b",
    r"\bdom\b",
    r"\bscreen\b",
    r"\bdesktop\b",
    r"\u6587\u4ef6",
    r"\u76ee\u5f55",
    r"\u4ed3\u5e93",
    r"\u9879\u76ee",
    r"\u4ee3\u7801",
    r"\u7a97\u53e3",
    r"\u6d4f\u89c8\u5668",
    r"\u6807\u7b7e\u9875",
    r"\u5e94\u7528",
    r"\u7f51\u9875",
    r"\u5c4f\u5e55",
    r"\u684c\u9762",
)
_PROJECT_QA_PATTERNS = (
    r"\bproject\b",
    r"\brepo\b",
    r"\brepository\b",
    r"\barchitecture\b",
    r"\bmodule\b",
    r"\bpackage\b",
    r"\bconfig\b",
    r"\bsetting\b",
    r"\bdocs?\b",
    r"\breadme\b",
    r"\bworkflow\b",
    r"\bruntime\b",
    r"\benv(?:ironment)?\b",
    r"\u9879\u76ee",
    r"\u4ed3\u5e93",
    r"\u67b6\u6784",
    r"\u914d\u7f6e",
    r"\u6587\u6863",
    r"\u8bf4\u660e",
    r"\u6a21\u5757",
    r"\u8fd0\u884c",
    r"\u73af\u5883",
)
_COMPUTER_GOAL_ACTION_PATTERNS = (
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
_COMPUTER_GOAL_OBJECT_PATTERNS = (
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
    "\u641c\u7d22",
    "\u53d1\u9001",
)


def _has_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        haystack = text.lower() if "\\b" in pattern else text
        if re.search(pattern, haystack, flags=re.IGNORECASE):
            return True
    return False


def _looks_like_workspace_action(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    if low.startswith("template ") or low.startswith("run template "):
        return True
    if low in {
        "resume last computer task",
        "resume computer task",
        "continue last computer task",
        "continue computer task",
    }:
        return True
    if _looks_like_natural_language_computer_goal(text):
        return True
    has_action = _has_pattern(text, _WORKSPACE_ACTION_PATTERNS)
    if not has_action:
        if re.match(r"^(?:read|write|append|list|find|search|run|pytest|cat)\b", low):
            return True
        return False
    if _contains_path_hint(text):
        return True
    if _has_pattern(text, _WORKSPACE_OBJECT_PATTERNS):
        return True
    if re.search(r"--[a-z0-9][a-z0-9-]*", low):
        return True
    if "`" in text:
        return True
    return False


def _looks_like_natural_language_computer_goal(text: str) -> bool:
    if not text:
        return False
    if not _has_pattern(text, _COMPUTER_GOAL_ACTION_PATTERNS):
        return False
    return _has_pattern(text, _COMPUTER_GOAL_OBJECT_PATTERNS)


def _contains_path_hint(text: str) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    for token in _PATH_TOKEN_RE.findall(raw):
        node = str(token or "").strip("`'\"()[]{}<> ,;:")
        if not node:
            continue
        if "/" in node or "\\" in node:
            return True
        if re.match(r"^[A-Za-z]:", node):
            return True
        if node.startswith(".") and len(node) > 1:
            return True
        if re.search(r"\.[A-Za-z0-9_]{1,8}$", node):
            return True
    return False


def _looks_like_reasoning(text: str) -> bool:
    return _has_pattern(text, _REASONING_PATTERNS)


def _looks_like_question(text: str) -> bool:
    return _has_pattern(text, _QUESTION_PATTERNS)


def _looks_like_project_qa(text: str) -> bool:
    if not text:
        return False
    if _has_pattern(text, _PROJECT_QA_PATTERNS):
        return True
    return _looks_like_question(text) and _has_pattern(text, _WORKSPACE_OBJECT_PATTERNS)


def classify_route(text: str, intent: ChatIntent) -> RouteProfile:
    if intent != ChatIntent.TASK:
        return RouteProfile(kind=RouteKind.POLICY, prefers_memory=False)

    query = str(text or "").strip()
    if not query:
        return RouteProfile(kind=RouteKind.GENERAL)

    if _looks_like_workspace_action(query):
        return RouteProfile(
            kind=RouteKind.WORKSPACE_ACTION,
            prefers_agent=True,
            prefers_rag=False,
            prefers_memory=False,
        )

    if _looks_like_reasoning(query):
        return RouteProfile(
            kind=RouteKind.REASONING,
            prefers_reasoning=True,
            prefers_rag=True,
            prefers_memory=True,
        )

    if _looks_like_project_qa(query):
        return RouteProfile(
            kind=RouteKind.PROJECT_QA,
            prefers_rag=True,
            prefers_memory=True,
        )

    if _looks_like_question(query):
        return RouteProfile(
            kind=RouteKind.FACTUAL_QA,
            prefers_rag=True,
            prefers_memory=True,
        )

    return RouteProfile(
        kind=RouteKind.GENERAL,
        prefers_rag=True,
        prefers_memory=True,
    )


__all__ = ["RouteKind", "RouteProfile", "classify_route"]
