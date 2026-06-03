from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _to_vector(values: Sequence[Any], dim: int) -> List[float]:
    if values is None:
        values = []
    elif hasattr(values, 'tolist'):
        values = values.tolist()
    elif not isinstance(values, (list, tuple)):
        values = list(values)
    vector = [float(x) for x in list(values)[:dim]]
    if len(vector) < dim:
        vector.extend([0.0] * (dim - len(vector)))
    return vector


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        fa = float(a)
        fb = float(b)
        dot += fa * fb
        left_norm += fa * fa
        right_norm += fb * fb
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


class VectorIndex:
    """Small JSON-backed vector index used as the repo default fallback."""

    def __init__(self, dim: int, max_items: int = 8000) -> None:
        self.dim = max(1, int(dim))
        self.max_items = max(1, int(max_items))
        self._items: List[Dict[str, Any]] = []

    def add(self, vector: Sequence[Any], *, id: str, metadata: Dict[str, Any] | None = None) -> None:
        row = {
            "id": str(id),
            "vector": _to_vector(vector, self.dim),
            "metadata": dict(metadata or {}),
        }
        self._items.append(row)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]

    def search(
        self,
        vector: Sequence[Any],
        *,
        k: int = 4,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        query = _to_vector(vector, self.dim)
        rows: List[Tuple[str, float, Dict[str, Any]]] = []
        for item in self._items:
            score = _cosine_similarity(query, item.get("vector", []))
            if score < float(threshold):
                continue
            rows.append(
                (
                    str(item.get("id", "")),
                    float(score),
                    dict(item.get("metadata", {}) or {}),
                )
            )
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[: max(1, int(k))]

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "max_items": self.max_items,
            "items": self._items,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def load(self, path: str) -> None:
        source = Path(path)
        if not source.exists():
            return
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        self.dim = max(1, int(payload.get("dim", self.dim) or self.dim))
        self.max_items = max(1, int(payload.get("max_items", self.max_items) or self.max_items))
        items = list(payload.get("items", []) or [])
        normalized: List[Dict[str, Any]] = []
        for item in items[-self.max_items :]:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "id": str(item.get("id", "")),
                    "vector": _to_vector(item.get("vector", []), self.dim),
                    "metadata": dict(item.get("metadata", {}) or {}),
                }
            )
        self._items = normalized

    def size(self) -> int:
        return len(self._items)
