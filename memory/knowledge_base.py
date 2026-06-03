# -*- coding: utf-8 -*-
"""JSON-backed knowledge base for structured facts (subject, predicate, object)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


class KnowledgeBase:
    """Simple file-backed triple store."""

    def __init__(self, path: str = "artifacts/memory/knowledge.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._facts: List[Tuple[str, str, str]] = []
        self._load()

    def add_fact(self, subject: str, predicate: str, obj: str) -> None:
        fact = (str(subject), str(predicate), str(obj))
        if fact not in self._facts:
            self._facts.append(fact)

    def query(self, subject: str = "", predicate: str = "", obj: str = "") -> List[Tuple[str, str, str]]:
        results = self._facts
        if subject:
            results = [f for f in results if f[0] == subject]
        if predicate:
            results = [f for f in results if f[1] == predicate]
        if obj:
            results = [f for f in results if f[2] == obj]
        return results

    def all_facts(self) -> List[Tuple[str, str, str]]:
        return list(self._facts)

    def save(self) -> None:
        payload = [{"s": s, "p": p, "o": o} for s, p, o in self._facts]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._facts = [(item["s"], item["p"], item["o"]) for item in data if isinstance(item, dict)]
        except (json.JSONDecodeError, KeyError):
            self._facts = []

    def clear(self) -> None:
        self._facts.clear()
        self.save()

    def __len__(self) -> int:
        return len(self._facts)
