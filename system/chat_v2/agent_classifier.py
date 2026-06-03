# -*- coding: utf-8 -*-
"""Request classification — intent heuristics for the workspace agent.

Extracted from WorkspaceAgent (agent.py) to improve cohesion.
All methods are pure keyword/pattern matchers with zero mutable state.
"""

from __future__ import annotations

import re
from typing import List

from system.l_utils.path_utils import extract_url, looks_like_filename, looks_like_path

from .intents import ChatIntent

# Shared CJK constants (moved from agent.py)
ZH_PROJECT = "项目"
ZH_REPO = "仓库"
ZH_CODE = "代码"
ZH_DIR = "目录"
ZH_FILE = "文件"
ZH_TEST = "测试"
ZH_SEARCH = "搜"
ZH_FIND = "找"
ZH_SEARCH_SHORT = "搜"
ZH_FIND_SHORT = "找"
ZH_READ = "读"
ZH_READ_FULL = "读取"
ZH_OPEN = "打开"
ZH_CONTENT = "内容"
ZH_INSIDE = "里面"
ZH_LIST = "列出"
ZH_STRUCTURE = "结构"
ZH_WHAT_FILES = "有哪些文件"
ZH_EXISTS = "存在"
ZH_HAS = "有没有"
ZH_HAS_Q = "是否有"
ZH_WHERE = "在哪"
ZH_DEFINE = "定义"
ZH_WRITE = "写"
ZH_WRITE_FULL = "写入"
ZH_APPEND = "追加"
ZH_CREATE = "创建"
ZH_UPDATE = "更新"
ZH_MODIFY = "修改"
ZH_EDIT = "编辑"
ZH_RUN = "运行"
ZH_EXEC = "执行"
ZH_SCRIPT = "脚本"

COMPUTER_GOAL_ACTION_PATTERNS = (
    "send", "message", "open", "launch", "search", "find", "type", "input", "click",
    "发送", "发消息", "发一句", "发个",
    "发条", "发给", "打开", "搜索", "查找",
    "输入", "点击", "点开",
)
COMPUTER_GOAL_OBJECT_PATTERNS = (
    "qq", "wechat", "wecom", "feishu", "contact", "chat", "message",
    "window", "desktop", "联系人", "聊天", "消息",
    "对话", "窗口", "桌面",
)


class RequestClassifier:
    """Pure keyword/pattern-based classification of user requests."""

    def is_read_request(self, text: str) -> bool:
        low = str(text or "").lower()
        return any(t in low or t in text for t in (
            "read", "open", "show", "inspect", "look at", "cat ", "review",
            ZH_LOOK, ZH_LOOK2, ZH_READ, ZH_READ_FULL, ZH_OPEN, ZH_CONTENT, ZH_INSIDE,
        ))

    def is_list_request(self, text: str) -> bool:
        low = str(text or "").lower()
        return any(t in low or t in text for t in (
            "list", "ls", "tree", "structure", "files", "directories",
            ZH_DIR, ZH_LIST, ZH_STRUCTURE, ZH_WHAT_FILES,
        ))

    def is_exists_request(self, text: str) -> bool:
        low = str(text or "").lower()
        return any(t in low or t in text for t in (
            "exists", "exist", "has ", "present", ZH_EXISTS, ZH_HAS, ZH_HAS_Q,
        ))

    def is_search_request(self, text: str) -> bool:
        low = str(text or "").lower()
        return any(t in low or t in text for t in (
            "search", "find", "grep", "look for", "where",
            ZH_SEARCH, ZH_FIND, ZH_SEARCH_SHORT, ZH_FIND_SHORT, ZH_WHERE, ZH_DEFINE,
        ))

    def is_write_request(self, text: str) -> bool:
        low = str(text or "").lower()
        return any(t in low or t in text for t in (
            "write", "append", "create", "update", "modify", "edit", "save",
            ZH_WRITE, ZH_WRITE_FULL, ZH_APPEND, ZH_CREATE, ZH_UPDATE, ZH_MODIFY, ZH_EDIT,
        ))

    def is_browser_request(self, text: str) -> bool:
        if extract_url(text):
            return True
        low = str(text or "").lower()
        return any(t in low or t in text for t in (
            "browser ", "open url", "browse", "visit", "navigate", "selector",
            "dom", "网页", "浏览器", "访问",
        ))

    def is_desktop_request(self, text: str) -> bool:
        low = str(text or "").lower()
        return any(t in low or t in text for t in (
            "click ", "double click ", "drag ", "ocr ", "click text ",
            "click control ", "list windows", "list controls", "focus window",
            "launch ", "hotkey ", "type ", "screenshot", "window", "control",
            "browser", "app ", "窗口", "聚焦", "启动",
            "截图", "输入", "press ", "key ", "scroll", "wait ",
            "open ", "close window", "minimize", "maximize",
        ))

    def is_exec_request(self, text: str) -> bool:
        low = str(text or "").lower()
        return any(t in low or t in text for t in (
            "run", "execute", "pytest", "test", "script",
            ZH_RUN, ZH_EXEC, ZH_TEST, ZH_SCRIPT,
        ))

    def is_project_inspection_request(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        low = raw.lower()
        project_terms = ("project", "repo", "repository", "workspace", "codebase", ZH_PROJECT, ZH_REPO, ZH_CODE)
        inspection_terms = (
            "inspect", "check", "review", "analyze", "summarize", "overview",
            "architecture", "structure", "module", "entrypoint", "runtime flow",
            "检查", "查看", "看看", "分析",
            "梳理", "总结", "概览", "架构",
            "结构", "模块", "入口", "流程",
        )
        has_project = any(t in low or t in raw for t in project_terms)
        has_inspection = any(t in low or t in raw for t in inspection_terms)
        if has_project and has_inspection:
            return True
        return has_project and any(t in low or t in raw for t in ("what files", "目录", "文件", "readme", "docs", "tests"))

    def looks_like_natural_language_computer_goal(self, text: str) -> bool:
        raw = str(text or "").strip()
        if not raw:
            return False
        low = raw.lower()
        has_action = any(t in low or t in raw for t in COMPUTER_GOAL_ACTION_PATTERNS)
        if not has_action:
            return False
        return any(t in low or t in raw for t in COMPUTER_GOAL_OBJECT_PATTERNS)

    def should_handle(self, text: str, intent: ChatIntent, agent_config, extract_paths_fn, extract_search_pattern_fn) -> bool:
        """Full should_handle logic extracted from WorkspaceAgent."""
        if not getattr(agent_config, "enabled", False) or intent != ChatIntent.TASK:
            return False
        raw = str(text or "").strip()
        if not raw:
            return False
        low = raw.lower()

        has_path = bool(extract_paths_fn(raw))
        if self.is_list_request(raw):
            return True
        if self.is_exists_request(raw) and has_path:
            return True
        if self.is_search_request(raw):
            return has_path or extract_search_pattern_fn(raw) is not None
        if self.is_read_request(raw) and has_path:
            return True
        if self.is_write_request(raw) and has_path:
            return True
        if self.is_browser_request(raw) or self.is_desktop_request(raw):
            return True
        if self.looks_like_natural_language_computer_goal(raw):
            return True
        if self.is_exec_request(raw):
            return has_path or "pytest" in low or "tests" in low or ZH_TEST in raw
        workspace_terms = ("repo", "repository", "project", "workspace", "codebase", ZH_PROJECT, ZH_REPO, ZH_CODE, ZH_DIR, ZH_FILE)
        return has_path or any(t in raw or t in low for t in workspace_terms)
