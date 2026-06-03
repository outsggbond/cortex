from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from system.evaluation._artifact_utils import _read_json, _safe_float


def _load_registry(path: Path) -> Dict[str, Any]:
    raw = _read_json(path)
    artifacts = raw.get("artifacts", [])
    archives = raw.get("archives", [])
    rows = [x for x in artifacts if isinstance(x, dict)] if isinstance(artifacts, list) else []
    archives_rows = [x for x in archives if isinstance(x, dict)] if isinstance(archives, list) else []
    return {
        "schema_version": 1,
        "updated_at": _safe_float(raw.get("updated_at", 0.0), 0.0),
        "active_id": str(raw.get("active_id", "")).strip(),
        "artifacts": rows,
        "archives": archives_rows,
    }


def _find_artifact_idx(rows: List[Dict[str, Any]], *, real_path: str, adapter_path: str) -> int:
    for i, row in enumerate(rows):
        rpath = str(row.get("real_path", "")).strip()
        apath = str(row.get("adapter_path", "")).strip()
        if (real_path and rpath == real_path) or (adapter_path and apath == adapter_path):
            return i
    return -1
