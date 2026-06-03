# -*- coding: utf-8 -*-
"""Chat memory with JSONL persistence, semantic search, and summarization support."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class ChatMemory:
    """Stores chat turns (role + text) with basic keyword search and summarization."""

    def __init__(self, path: str = "artifacts/memory/chat_memory.jsonl", capacity: int = 200) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = max(1, int(capacity))
        self._turns: List[Dict[str, str]] = []
        self._load()

    def add(self, role: str, text: str) -> None:
        if not text:
            return
        self._turns.append({"role": str(role), "text": str(text)})
        if len(self._turns) > self.capacity:
            self._turns = self._turns[-self.capacity :]
        self._save()

    def search(self, query: str, k: int = 4) -> List[str]:
        """Simple keyword-overlap search. Returns up to `k` matching text strings."""
        if not query or not self._turns:
            return []
        terms = set(re.findall(r"\w+", query.lower()))
        if not terms:
            return []
        scored = []
        for turn in self._turns:
            text = str(turn.get("text", ""))
            text_lower = text.lower()
            score = sum(1 for t in terms if t in text_lower)
            if score > 0:
                scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[: max(1, int(k))]]

    def summarize_if_needed(
        self,
        window: int,
        min_turns: int,
        summarizer: Optional[Callable[[str], Optional[str]]] = None,
    ) -> Optional[str]:
        """If turns exceed `window`, summarize the oldest `min_turns` and return the summary."""
        if len(self._turns) < min_turns:
            return None
        if len(self._turns) <= window:
            return None
        old_text = "\n".join(
            f"{t['role']}: {t['text']}"
            for t in self._turns[:min_turns]
        )
        if summarizer:
            summary = summarizer(old_text)
            if summary:
                # Replace summarized turns with a synthetic turn
                self._turns = [{"role": "system", "text": f"[summary] {summary}"}] + self._turns[min_turns:]
                self._save()
                return summary
        return None

    def items(self) -> List[Dict[str, str]]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()
        self._save()

    def _save(self) -> None:
        with self.path.open("w", encoding="utf-8") as fh:
            for turn in self._turns:
                fh.write(json.dumps(turn, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        if len(self._turns) > self.capacity:
            self._turns = self._turns[-self.capacity :]

    def __len__(self) -> int:
        return len(self._turns)
