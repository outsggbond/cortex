# -*- coding: utf-8 -*-
"""Unified desktop/Web GUI shell built on top of GUI adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .automation import AutomationGUIController, create_automation_gui_controller
from .chat import ChatGUISession, create_chat_gui_session
from .computer import ComputerUseGUISession, create_computer_use_gui_session


@dataclass
class GUIShellSnapshot:
    chat_transcript: List[Dict[str, Any]] = field(default_factory=list)
    computer_runs: List[Dict[str, Any]] = field(default_factory=list)
    automation_workflows: List[Dict[str, Any]] = field(default_factory=list)
    automation_summary: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_transcript": [dict(item or {}) for item in list(self.chat_transcript)],
            "computer_runs": [dict(item or {}) for item in list(self.computer_runs)],
            "automation_workflows": [dict(item or {}) for item in list(self.automation_workflows)],
            "automation_summary": None if self.automation_summary is None else dict(self.automation_summary),
        }


class DesktopWebGUIShell:
    def __init__(
        self,
        args: Any,
        *,
        chat_session: ChatGUISession | None = None,
        computer_session: ComputerUseGUISession | None = None,
        automation_controller: AutomationGUIController | None = None,
    ) -> None:
        self.args = args
        self._chat_session = chat_session
        self._computer_session = computer_session
        self._automation_controller = automation_controller

    @property
    def chat_session(self) -> ChatGUISession:
        if self._chat_session is None:
            self._chat_session = create_chat_gui_session(self.args)
        return self._chat_session

    @property
    def computer_session(self) -> ComputerUseGUISession:
        if self._computer_session is None:
            self._computer_session = create_computer_use_gui_session(self.args)
        return self._computer_session

    @property
    def automation_controller(self) -> AutomationGUIController:
        if self._automation_controller is None:
            self._automation_controller = create_automation_gui_controller(self.args)
        return self._automation_controller

    def submit_chat(self, user_text: str) -> Dict[str, Any]:
        return self.chat_session.submit(user_text).to_dict()

    def chat_transcript(self) -> List[Dict[str, Any]]:
        if self._chat_session is None:
            return []
        return [item.to_dict() for item in self._chat_session.transcript()]

    def run_computer_goal(self, goal: str) -> Dict[str, Any]:
        return dict(self.computer_session.submit_goal(goal))

    def computer_history(self) -> List[Dict[str, Any]]:
        if self._computer_session is None:
            return []
        return self._computer_session.history()

    def list_workflows(self) -> List[Dict[str, Any]]:
        return self.automation_controller.list_workflows()

    def scan_automation(
        self,
        *,
        workflow_ids: Iterable[str] | None = None,
        force_run: bool = False,
    ) -> Dict[str, Any]:
        return self.automation_controller.scan(
            workflow_ids=workflow_ids,
            force_run=force_run,
        )

    def run_automation(self, workflow_ids: Iterable[str]) -> Dict[str, Any]:
        return self.automation_controller.run(workflow_ids)

    def loop_automation(
        self,
        *,
        poll_s: float | None = None,
        max_cycles: int | None = None,
        workflow_ids: Iterable[str] | None = None,
    ) -> Dict[str, Any]:
        return self.automation_controller.loop(
            poll_s=poll_s,
            max_cycles=max_cycles,
            workflow_ids=workflow_ids,
        )

    def snapshot(self) -> GUIShellSnapshot:
        controller = self._automation_controller
        return GUIShellSnapshot(
            chat_transcript=self.chat_transcript(),
            computer_runs=self.computer_history(),
            automation_workflows=[] if controller is None else controller.last_workflows,
            automation_summary=None if controller is None else controller.last_summary,
        )

    def reset(
        self,
        *,
        chat: bool = False,
        computer: bool = False,
        automation: bool = False,
        all_sections: bool = False,
    ) -> GUIShellSnapshot:
        reset_all = bool(all_sections or not any((chat, computer, automation)))
        if (reset_all or chat) and self._chat_session is not None:
            self._chat_session.reset()
        if (reset_all or computer) and self._computer_session is not None:
            self._computer_session.reset()
        if (reset_all or automation) and self._automation_controller is not None:
            self._automation_controller.reset()
        return self.snapshot()


def create_desktop_web_gui_shell(
    args: Any,
    *,
    chat_session: ChatGUISession | None = None,
    computer_session: ComputerUseGUISession | None = None,
    automation_controller: AutomationGUIController | None = None,
) -> DesktopWebGUIShell:
    return DesktopWebGUIShell(
        args,
        chat_session=chat_session,
        computer_session=computer_session,
        automation_controller=automation_controller,
    )


__all__ = [
    "DesktopWebGUIShell",
    "GUIShellSnapshot",
    "create_desktop_web_gui_shell",
]
