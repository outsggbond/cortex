# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path
from typing import List


def _read_text(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_file(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    text = _read_text(path)
    if ext in (".yml", ".yaml"):
        try:
            import yaml
        except Exception:
            return [line.strip() for line in text.splitlines() if line.strip()]
        data = yaml.safe_load(text) or {}
        lines: List[str] = []
        if isinstance(data, dict):
            convs = data.get("conversations", [])
            for item in convs:
                if isinstance(item, list):
                    for utter in item:
                        if isinstance(utter, str) and utter.strip():
                            lines.append(utter.strip())
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str) and item.strip():
                    lines.append(item.strip())
        return lines
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_training_lines(path: str) -> List[str]:
    if os.path.isdir(path):
        all_lines: List[str] = []
        for p in sorted(Path(path).rglob("*")):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in (".yml", ".yaml", ".txt"):
                all_lines.extend(_load_file(str(p)))
        return all_lines
    return _load_file(path)
