from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from system.chat_v2 import run_chat_v2
from system.chat_v2.pipeline import build_request
from system.chat_v2.runtime import (
    DEFAULT_CHAT_HISTORY_LIMIT,
    append_chat_history,
    build_chat_runtime_components,
)


@dataclass
class ChatDomainResult:
    text: str
    source: str
    intent: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": str(self.text or ""),
            "source": str(self.source or ""),
            "intent": str(self.intent or ""),
            "metadata": dict(self.metadata or {}),
            "history": [dict(item or {}) for item in list(self.history)],
        }


def _coerce_history_turn(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        payload = dict(item)
    else:
        payload = {
            "role": getattr(item, "role", ""),
            "text": getattr(item, "text", ""),
            "source": getattr(item, "source", ""),
            "metadata": getattr(item, "metadata", {}),
        }
    return {
        "role": str(payload.get("role", "") or ""),
        "text": str(payload.get("text", "") or ""),
        "source": str(payload.get("source", "") or ""),
        "metadata": dict(payload.get("metadata", {}) or {}),
    }


class ChatDomainService:
    def __init__(self, args: Any) -> None:
        self.components = build_chat_runtime_components(args)
        self.pipeline = self.components.pipeline

    def respond(
        self,
        user_text: str,
        history: Sequence[Any] | None = None,
        *,
        history_limit: int = DEFAULT_CHAT_HISTORY_LIMIT,
    ) -> ChatDomainResult:
        text = str(user_text or "").strip()
        if not text:
            raise ValueError("user_text is required")

        request_history: List[Dict[str, Any]] = []
        for item in list(history or []):
            turn = _coerce_history_turn(item)
            if not turn["role"] or not turn["text"]:
                continue
            request_history.append(turn)

        response = self.pipeline.respond(build_request(text, request_history))
        updated_history = append_chat_history(
            request_history,
            text,
            response,
            limit=history_limit,
        )
        return ChatDomainResult(
            text=str(getattr(response, "text", "") or ""),
            source=str(getattr(response, "source", "") or ""),
            intent=str(getattr(response, "intent", "") or ""),
            metadata=dict(getattr(response, "metadata", {}) or {}),
            history=[_coerce_history_turn(item) for item in updated_history],
        )


def build_chat_domain_service(args: Any) -> ChatDomainService:
    return ChatDomainService(args)


def run_chat_domain(*args, **kwargs):
    return run_chat_v2(*args, **kwargs)


__all__ = [
    "ChatDomainResult",
    "ChatDomainService",
    "DEFAULT_CHAT_HISTORY_LIMIT",
    "build_chat_domain_service",
    "run_chat_domain",
]
