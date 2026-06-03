# -*- coding: utf-8 -*-
"""Optimized memory module for perception graph — nearest-neighbor search over node features."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class OptimizedMemoryModule:
    """Memory-backed vector store for perception nodes with cosine-similarity search."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = max(1, int(dim))
        self._features: List[np.ndarray] = []
        self._ids: List[str] = []

    def add(self, node_id: str, feature: np.ndarray) -> None:
        vec = np.asarray(feature, dtype=np.float32).ravel()
        if vec.shape[0] != self.dim:
            # Simple resize via padding/truncation
            padded = np.zeros(self.dim, dtype=np.float32)
            n = min(len(vec), self.dim)
            padded[:n] = vec[:n]
            vec = padded
        self._ids.append(str(node_id))
        self._features.append(vec)

    def find_nearest_neighbors(
        self,
        query: np.ndarray,
        k: int = 5,
        nprobe: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return (id, cosine_similarity) for the k nearest neighbors."""
        if not self._features:
            return []
        q = np.asarray(query, dtype=np.float32).ravel()
        if q.shape[0] != self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            n = min(len(q), self.dim)
            padded[:n] = q[:n]
            q = padded
        # Cosine similarity
        feats = np.stack(self._features, axis=0)  # (N, dim)
        q_norm = np.linalg.norm(q) + 1e-10
        feats_norm = np.linalg.norm(feats, axis=1) + 1e-10
        sims = np.dot(feats, q) / (feats_norm * q_norm)
        # Top-k
        k = min(max(1, int(k)), len(self._ids))
        top_indices = np.argsort(sims)[::-1][:k]
        return [(self._ids[i], float(sims[i])) for i in top_indices]

    def size(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._features.clear()
        self._ids.clear()
