# -*- coding: utf-8 -*-
"""Shared datatypes for chat runtime v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ChatTurn:
    role: str
    text: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatRequest:
    user_text: str
    history: List[ChatTurn] = field(default_factory=list)


@dataclass
class ChatResponse:
    text: str
    source: str
    intent: str
    metadata: Dict[str, Any] = field(default_factory=dict)
