# -*- coding: utf-8 -*-
"""Short-term sliding-window memory."""

from __future__ import annotations

from typing import List


class ShortTermMemory:
    """Fixed-capacity FIFO buffer for recent conversation turns."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = max(1, int(capacity))
        self._items: List[str] = []

    def add(self, text: str) -> None:
        if not text:
            return
        self._items.append(str(text))
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity :]

    def items(self) -> List[str]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
