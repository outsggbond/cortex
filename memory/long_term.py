# -*- coding: utf-8 -*-
"""Long-term JSONL-backed memory store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


class LongTermMemory:
    """Append-only JSONL file for persistent long-term dialogue storage."""

    def __init__(self, path: str = "artifacts/memory/long_term.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def add(self, text: str) -> None:
        if not text:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"text": str(text)}, ensure_ascii=False) + "\n")

    def items(self, limit: int = 50) -> List[str]:
        results: List[str] = []
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        results.append(str(obj.get("text", "")))
                    except json.JSONDecodeError:
                        results.append(line)
        except FileNotFoundError:
            pass
        limit = max(1, int(limit))
        if len(results) > limit:
            results = results[-limit:]
        return results

    def clear(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def __len__(self) -> int:
        return len(self.items(limit=10_000_000))
