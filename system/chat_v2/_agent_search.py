# -*- coding: utf-8 -*-
"""Project search helper extracted from WorkspaceAgent."""

from __future__ import annotations

from typing import Dict, List, Tuple


def _project_search_hits(
    agent,
    rows: Dict[Tuple[str, str], List[str]],
    pattern: str,
    *,
    path: str | None = None,
) -> List[str]:
    target_pattern = str(pattern or "").strip().lower()
    target_path = None if path is None else str(path or "").strip().lower()
    for (row_pattern, row_path), hits in rows.items():
        if str(row_pattern or "").strip().lower() != target_pattern:
            continue
        if target_path is not None and str(row_path or "").strip().lower() != target_path:
            continue
        return [str(x) for x in hits]
    return []
