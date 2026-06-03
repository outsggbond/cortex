# -*- coding: utf-8 -*-
"""Stateful chat adapter for GUI shells."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from system.interfaces.api import APIChatResult, APIChatTurn, ChatAPIAdapter


@dataclass
class GUIChatMessage:
    role: str
    text: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": str(self.role or ""),
            "text": str(self.text or ""),
            "source": str(self.source or ""),
            "metadata": dict(self.metadata or {}),
        }


class ChatGUISession:
    def __init__(self, args: Any) -> None:
        self._adapter = ChatAPIAdapter(args)
        self._history: List[APIChatTurn] = []

    def submit(self, user_text: str) -> APIChatResult:
        result = self._adapter.respond(user_text, history=self._history)
        self._history = list(result.history)
        return result

    def transcript(self) -> List[GUIChatMessage]:
        return [
            GUIChatMessage(
                role=str(turn.role or ""),
                text=str(turn.text or ""),
                source=str(turn.source or ""),
                metadata=dict(turn.metadata or {}),
            )
            for turn in list(self._history)
        ]

    def reset(self) -> None:
        self._history = []


def create_chat_gui_session(args: Any) -> ChatGUISession:
    return ChatGUISession(args)


__all__ = [
    "ChatGUISession",
    "GUIChatMessage",
    "create_chat_gui_session",
]
