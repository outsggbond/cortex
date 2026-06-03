# -*- coding: utf-8 -*-
"""GUI-facing adapters with stateful session helpers."""

from .automation import AutomationGUIController, create_automation_gui_controller
from .chat import ChatGUISession, GUIChatMessage, create_chat_gui_session
from .computer import ComputerUseGUISession, create_computer_use_gui_session
from .shell import DesktopWebGUIShell, GUIShellSnapshot, create_desktop_web_gui_shell

__all__ = [
    "AutomationGUIController",
    "ChatGUISession",
    "ComputerUseGUISession",
    "DesktopWebGUIShell",
    "GUIShellSnapshot",
    "GUIChatMessage",
    "create_automation_gui_controller",
    "create_chat_gui_session",
    "create_computer_use_gui_session",
    "create_desktop_web_gui_shell",
]
