# -*- coding: utf-8 -*-
"""Programmatic API adapters built on top of runtime/domain layers."""

from .automation import AutomationAPIAdapter, create_automation_api
from .chat import APIChatResult, APIChatTurn, ChatAPIAdapter, create_chat_api
from .computer import ComputerUseAPIAdapter, create_computer_use_api
from .http import (
    HTTPAPIRequest,
    HTTPAPIResponse,
    HTTPAPIShell,
    build_http_api_handler,
    create_http_api_shell,
    run_http_api_server,
)

__all__ = [
    "AutomationAPIAdapter",
    "APIChatResult",
    "APIChatTurn",
    "ChatAPIAdapter",
    "ComputerUseAPIAdapter",
    "HTTPAPIRequest",
    "HTTPAPIResponse",
    "HTTPAPIShell",
    "build_http_api_handler",
    "create_automation_api",
    "create_chat_api",
    "create_computer_use_api",
    "create_http_api_shell",
    "run_http_api_server",
]
