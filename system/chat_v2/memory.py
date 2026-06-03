# -*- coding: utf-8 -*-
"""Curated retrieval memory for chat runtime v2."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

from system.core.embeddings import cosine, encode_text


logger = logging.getLogger(__name__)


def _normalize_query(text: str) -> str:
    cur = re.sub(r"\s+", "", str(text or "").strip()).lower()
    cur = re.sub(r"[!！?？。．，,~～…]+$", "", cur)
    return cur


def _tokenize(text: str) -> List[str]:
    t = str(text or "").strip().lower()
    if not t:
        return []
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", t)


def _jaccard(a: str, b: str) -> float:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / max(1, len(sa | sb)))


@dataclass
class MemoryHit:
    user: str
    assistant: str
    score: float


@dataclass
class MemoryEntry:
    user: str
    assistant: str
    normalized: str
    token_set: FrozenSet[str]
    vector: object | None = field(default=None, repr=False)


def _token_overlap(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0
    return float(inter / max(1, min(len(a), len(b))))


def _substring_score(a: str, b: str) -> float:
    left = str(a or "").strip()
    right = str(b or "").strip()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if len(short) < 4:
        return 0.0
    if short in long:
        ratio = float(len(short) / max(1, len(long)))
        return min(0.96, 0.76 + 0.18 * ratio)
    return 0.0


class CuratedMemoryStore:
    """Read-only retrieval store built from curated dialogue data."""

    def __init__(self, path: str, min_similarity: float = 0.58, top_k: int = 3) -> None:
        self.path = Path(path)
        self.min_similarity = float(min_similarity)
        self.top_k = max(1, int(top_k))
        self.entries: List[MemoryEntry] = []
        self._exact: Dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        self.entries = []
        self._exact = {}
        if not self.path.exists():
            logger.info("v2 memory not found: %s", self.path)
            return
        bad = 0
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except Exception:
                    bad += 1
                    continue
                if not isinstance(item, dict):
                    bad += 1
                    continue
                user = str(item.get("user", "") or item.get("prompt", "") or "").strip()
                assistant = str(item.get("assistant", "") or item.get("response", "") or "").strip()
                if not user or not assistant:
                    bad += 1
                    continue
                key = _normalize_query(user)
                token_set = frozenset(_tokenize(user))
                entry = MemoryEntry(
                    user=user,
                    assistant=assistant,
                    normalized=key,
                    token_set=token_set,
                )
                self.entries.append(entry)
                if key and key not in self._exact:
                    self._exact[key] = entry
        except Exception as e:
            logger.warning("v2 memory load failed: %s", e)
            self.entries = []
            self._exact = {}
            return
        logger.info("v2 memory loaded. entries=%d bad=%d path=%s", len(self.entries), bad, self.path)

    def _semantic_score(self, query_vec, entry: MemoryEntry) -> float:
        if query_vec is None:
            return 0.0
        if entry.vector is None:
            try:
                entry.vector = encode_text(entry.user)
            except Exception as exc:
                logger.debug("v2 memory embedding failed for %s: %s", entry.user[:80], exc)
                entry.vector = False
        if entry.vector is False or entry.vector is None:
            return 0.0
        try:
            return max(0.0, float(cosine(query_vec, entry.vector)))
        except Exception:
            return 0.0

    def search_many(self, query: str, top_k: int | None = None) -> List[MemoryHit]:
        q = str(query or "").strip()
        if not q:
            return []
        limit = max(1, int(top_k or self.top_k))
        normalized_query = _normalize_query(q)
        exact = self._exact.get(normalized_query)
        if exact:
            return [MemoryHit(user=exact.user, assistant=exact.assistant, score=1.0)]

        query_tokens = frozenset(_tokenize(q))
        try:
            query_vec = encode_text(q)
        except Exception as exc:
            logger.debug("v2 memory query embedding failed: %s", exc)
            query_vec = None

        ranked: List[tuple[float, float, float, MemoryEntry]] = []
        semantic_floor = min(0.95, max(0.32, self.min_similarity + 0.12))
        lexical_floor = max(0.24, self.min_similarity * 0.72)
        for entry in self.entries:
            lexical = max(
                _jaccard(q, entry.user),
                _token_overlap(query_tokens, entry.token_set),
                _substring_score(normalized_query, entry.normalized),
            )
            semantic = self._semantic_score(query_vec, entry)
            score = max(lexical, 0.62 * lexical + 0.38 * semantic)
            if score < self.min_similarity and lexical < lexical_floor and semantic < semantic_floor:
                continue
            ranked.append((score, lexical, semantic, entry))
        if not ranked:
            return []

        ranked.sort(key=lambda item: (item[0], item[1], item[2], -len(item[3].user)), reverse=True)
        hits: List[MemoryHit] = []
        seen_pairs = set()
        for score, _lexical, _semantic, entry in ranked:
            key = (entry.user, entry.assistant)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            hits.append(MemoryHit(user=entry.user, assistant=entry.assistant, score=round(float(score), 4)))
            if len(hits) >= limit:
                break
        return hits

    def search(self, query: str) -> Optional[MemoryHit]:
        hits = self.search_many(query, top_k=1)
        return hits[0] if hits else None
