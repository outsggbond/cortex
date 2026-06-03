# -*- coding: utf-8 -*-
"""Framework-agnostic chat adapter for API handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from system.domain.chat import DEFAULT_CHAT_HISTORY_LIMIT, build_chat_domain_service


@dataclass
class APIChatTurn:
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


@dataclass
class APIChatResult:
    text: str
    source: str
    intent: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[APIChatTurn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": str(self.text or ""),
            "source": str(self.source or ""),
            "intent": str(self.intent or ""),
            "metadata": dict(self.metadata or {}),
            "history": [turn.to_dict() for turn in list(self.history)],
        }


def _coerce_turn(item: Any) -> APIChatTurn:
    if isinstance(item, APIChatTurn):
        return APIChatTurn(
            role=str(item.role or ""),
            text=str(item.text or ""),
            source=str(item.source or ""),
            metadata=dict(item.metadata or {}),
        )
    if isinstance(item, dict):
        return APIChatTurn(
            role=str(item.get("role", "") or ""),
            text=str(item.get("text", "") or ""),
            source=str(item.get("source", "") or ""),
            metadata=dict(item.get("metadata", {}) or {}),
        )
    return APIChatTurn(
        role=str(getattr(item, "role", "") or ""),
        text=str(getattr(item, "text", "") or ""),
        source=str(getattr(item, "source", "") or ""),
        metadata=dict(getattr(item, "metadata", {}) or {}),
    )


def _coerce_history(history: Sequence[Any] | None) -> List[APIChatTurn]:
    out: List[APIChatTurn] = []
    for item in list(history or []):
        turn = _coerce_turn(item)
        if not str(turn.role or "").strip():
            continue
        if not str(turn.text or "").strip():
            continue
        out.append(turn)
    return out


class ChatAPIAdapter:
    def __init__(self, args: Any) -> None:
        self.service = build_chat_domain_service(args)

    def respond(
        self,
        user_text: str,
        history: Sequence[Any] | None = None,
        *,
        history_limit: int = DEFAULT_CHAT_HISTORY_LIMIT,
    ) -> APIChatResult:
        text = str(user_text or "").strip()
        if not text:
            raise ValueError("user_text is required")

        turns = _coerce_history(history)
        request_history = [turn.to_dict() for turn in turns]
        result = self.service.respond(text, request_history, history_limit=history_limit)
        return APIChatResult(
            text=str(getattr(result, "text", "") or ""),
            source=str(getattr(result, "source", "") or ""),
            intent=str(getattr(result, "intent", "") or ""),
            metadata=dict(getattr(result, "metadata", {}) or {}),
            history=[_coerce_turn(item) for item in list(getattr(result, "history", []) or [])],
        )


def create_chat_api(args: Any) -> ChatAPIAdapter:
    return ChatAPIAdapter(args)


__all__ = [
    "APIChatResult",
    "APIChatTurn",
    "ChatAPIAdapter",
    "create_chat_api",
]
