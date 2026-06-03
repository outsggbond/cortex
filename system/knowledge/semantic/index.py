from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    tokens: List[str] = []
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            import jieba  # type: ignore

        tokens = [t.strip() for t in jieba.cut(text) if t and t.strip()]
    except Exception:
        tokens = []
    if not tokens:
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    return tokens


def _jaccard_tokens(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union <= 0:
        return 0.0
    return float(inter / union)


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    sample_ids: Set[int] = field(default_factory=set)
    terminal_ids: Set[int] = field(default_factory=set)


class TokenTrie:
    def __init__(self, min_prefix_len: int = 2):
        self.root = TrieNode()
        self.min_prefix_len = max(1, int(min_prefix_len))

    def clear(self) -> None:
        self.root = TrieNode()

    def insert(self, token: str, sample_id: int) -> None:
        token = (token or "").strip()
        if not token:
            return
        node = self.root
        for ch in token:
            node = node.children.setdefault(ch, TrieNode())
            node.sample_ids.add(sample_id)
        node.terminal_ids.add(sample_id)

    def _walk(self, text: str) -> Optional[TrieNode]:
        if not text:
            return None
        node = self.root
        for ch in text:
            nxt = node.children.get(ch)
            if nxt is None:
                return None
            node = nxt
        return node

    def search_token(self, token: str) -> Set[int]:
        node = self._walk((token or "").strip())
        if node is None:
            return set()
        return set(node.terminal_ids)

    def search_prefix(self, prefix: str) -> Set[int]:
        node = self._walk((prefix or "").strip())
        if node is None:
            return set()
        return set(node.sample_ids)

    def retrieve_for_tokens(self, tokens: List[str]) -> Set[int]:
        out: Set[int] = set()
        for token in tokens:
            tok = (token or "").strip()
            if not tok:
                continue
            out.update(self.search_token(tok))
            if len(tok) >= self.min_prefix_len + 1:
                limit = min(len(tok) - 1, self.min_prefix_len + 2)
                for n in range(self.min_prefix_len, limit + 1):
                    out.update(self.search_prefix(tok[:n]))
        return out


class SemanticGraph:
    def __init__(self, window_size: int = 3):
        self.window_size = max(1, int(window_size))
        self.token_to_samples: Dict[str, Set[int]] = defaultdict(set)
        self.edges: Dict[str, Dict[str, float]] = defaultdict(dict)

    def clear(self) -> None:
        self.token_to_samples.clear()
        self.edges.clear()

    def add_sample(self, tokens: List[str], sample_id: int) -> None:
        clean = [t for t in tokens if t]
        if not clean:
            return
        for t in set(clean):
            self.token_to_samples[t].add(sample_id)
        win = self.window_size
        n = len(clean)
        for i in range(n):
            a = clean[i]
            for j in range(i + 1, min(n, i + win + 1)):
                b = clean[j]
                if not a or not b or a == b:
                    continue
                dist = float(j - i)
                w = 1.0 / max(1.0, dist)
                self.edges[a][b] = float(self.edges[a].get(b, 0.0)) + w
                self.edges[b][a] = float(self.edges[b].get(a, 0.0)) + w

    def candidate_ids(
        self,
        query_tokens: List[str],
        depth: int = 1,
        max_neighbors: int = 8,
        min_edge_weight: float = 0.05,
    ) -> Tuple[Set[int], Set[str]]:
        seeds = [t for t in query_tokens if t]
        expanded = self.expand_tokens(
            seeds,
            depth=depth,
            max_neighbors=max_neighbors,
            min_edge_weight=min_edge_weight,
        )
        ids: Set[int] = set()
        for token in expanded:
            ids.update(self.token_to_samples.get(token, set()))
        return ids, expanded

    def expand_tokens(
        self,
        tokens: List[str],
        depth: int = 1,
        max_neighbors: int = 8,
        min_edge_weight: float = 0.05,
    ) -> Set[str]:
        roots = [t for t in tokens if t]
        if not roots:
            return set()
        depth = max(0, int(depth))
        max_neighbors = max(1, int(max_neighbors))
        out: Set[str] = set(roots)
        if depth <= 0:
            return out
        q: deque[Tuple[str, int]] = deque((t, 0) for t in roots)
        seen: Set[str] = set(roots)
        while q:
            token, d = q.popleft()
            if d >= depth:
                continue
            neighbors = self.edges.get(token, {})
            if not neighbors:
                continue
            ranked = sorted(neighbors.items(), key=lambda kv: kv[1], reverse=True)
            used = 0
            for nxt, w in ranked:
                if used >= max_neighbors:
                    break
                if float(w) < float(min_edge_weight):
                    continue
                used += 1
                out.add(nxt)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
        return out

    def score_overlap(self, query_tokens: List[str], candidate_tokens: List[str], expanded_tokens: Set[str]) -> float:
        qt = [t for t in query_tokens if t]
        ct = [t for t in candidate_tokens if t]
        if not qt or not ct:
            return 0.0
        direct = _jaccard_tokens(qt, ct)
        if not expanded_tokens:
            return direct
        expanded = _jaccard_tokens(expanded_tokens, ct)
        return 0.6 * direct + 0.4 * expanded


class VectorReranker:
    def __init__(self, max_cache: int = 4096):
        self.max_cache = max(0, int(max_cache))
        self._cache: Dict[str, np.ndarray] = {}

    def clear(self) -> None:
        self._cache.clear()

    def _embed(self, text: str) -> Optional[np.ndarray]:
        key = (text or "").strip()
        if not key:
            return None
        if self.max_cache > 0:
            vec = self._cache.get(key)
            if vec is not None:
                return vec
        try:
            from system.core.embeddings import encode_text

            arr = encode_text(key)
            vec = np.asarray(arr, dtype="float32")
        except Exception:
            return None
        if self.max_cache > 0:
            if len(self._cache) >= self.max_cache:
                self._cache.clear()
            self._cache[key] = vec
        return vec

    def similarity(self, query: str, text: str) -> float:
        v1 = self._embed(query)
        v2 = self._embed(text)
        if v1 is None or v2 is None:
            return 0.0
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        if denom <= 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)


@dataclass
class HybridRecord:
    record_id: int
    intent: str
    user: str
    assistant: str
    tokens: Tuple[str, ...]


@dataclass
class HybridHit:
    record_id: int
    intent: str
    user: str
    response: str
    score: float
    lexical_score: float
    graph_score: float
    vector_score: float


class HybridSemanticIndex:
    def __init__(
        self,
        index_path: str = "",
        trace_path: str = "",
        auto_persist: Optional[bool] = None,
    ) -> None:
        self.trie = TokenTrie(min_prefix_len=int(os.environ.get("DIALOGUE_TRIE_MIN_PREFIX", "2")))
        self.graph = SemanticGraph(window_size=int(os.environ.get("DIALOGUE_GRAPH_WINDOW", "3")))
        self.reranker = VectorReranker(max_cache=int(os.environ.get("DIALOGUE_VECTOR_CACHE", "4096")))
        self.records: Dict[int, HybridRecord] = {}
        self.record_vectors: Dict[int, List[float]] = {}
        self.intent_to_ids: Dict[str, List[int]] = defaultdict(list)
        self._next_id = 0
        self.max_candidates = max(50, int(os.environ.get("DIALOGUE_INDEX_MAX_CANDIDATES", "800")))
        self.graph_depth = max(0, int(os.environ.get("DIALOGUE_GRAPH_DEPTH", "1")))
        self.graph_neighbors = max(1, int(os.environ.get("DIALOGUE_GRAPH_MAX_NEIGHBORS", "8")))
        self.graph_min_edge = float(os.environ.get("DIALOGUE_GRAPH_MIN_EDGE", "0.05"))
        self.min_candidates_for_intent = max(1, int(os.environ.get("DIALOGUE_INTENT_MIN_CANDIDATES", "6")))
        self.fallback_candidates = max(20, int(os.environ.get("DIALOGUE_FALLBACK_CANDIDATES", "240")))
        self.weight_lex = float(os.environ.get("DIALOGUE_SCORE_LEX", "0.40"))
        self.weight_graph = float(os.environ.get("DIALOGUE_SCORE_GRAPH", "0.25"))
        self.weight_vec = float(os.environ.get("DIALOGUE_SCORE_VEC", "0.35"))
        self.intent_bonus = float(os.environ.get("DIALOGUE_SCORE_INTENT_BONUS", "0.05"))

        self.index_path = Path(index_path or os.environ.get("DIALOGUE_HYBRID_INDEX_PATH", "artifacts/memory/hybrid_semantic_index.json"))
        self.trace_path = Path(trace_path or os.environ.get("DIALOGUE_REASON_TRACE_PATH", "artifacts/audit/dialogue_reason_trace.jsonl"))
        self.trace_enabled = os.environ.get("DIALOGUE_TRACE_ENABLE", "1") != "0"
        if auto_persist is None:
            self.auto_persist = os.environ.get("DIALOGUE_HYBRID_PERSIST", "1") != "0"
        else:
            self.auto_persist = bool(auto_persist)
        self.save_every = max(1, int(os.environ.get("DIALOGUE_HYBRID_SAVE_EVERY", "20")))
        self._pending_since_save = 0
        self.load()

    def clear(self, reset_persistence_counter: bool = True) -> None:
        self.trie.clear()
        self.graph.clear()
        self.reranker.clear()
        self.records.clear()
        self.record_vectors.clear()
        self.intent_to_ids.clear()
        self._next_id = 0
        if reset_persistence_counter:
            self._pending_since_save = 0

    def build(self, records: List[Tuple[str, str, str]]) -> None:
        self.clear()
        for intent, user, assistant in records:
            self.add_record(intent=intent, user=user, assistant=assistant, persist=False)
        logger.info(
            "Hybrid semantic index built. records=%d intents=%d",
            len(self.records),
            len(self.intent_to_ids),
        )
        if self.auto_persist:
            self.save()

    def add_record(self, intent: str, user: str, assistant: str, persist: bool = True) -> int:
        u = (user or "").strip()
        a = (assistant or "").strip()
        if not u or not a:
            return -1
        key = str(intent or "default")
        rid = self._next_id
        self._next_id += 1
        tokens = tuple(_tokenize(u))
        rec = HybridRecord(
            record_id=rid,
            intent=key,
            user=u,
            assistant=a,
            tokens=tokens,
        )
        vec = self._embed_text(u)
        self._insert_record(rec, vec=vec)
        if persist:
            self._pending_since_save += 1
            if self.auto_persist and self._pending_since_save >= self.save_every:
                self.save()
        return rid

    def _insert_record(self, rec: HybridRecord, vec: Optional[List[float]] = None) -> None:
        self.records[rec.record_id] = rec
        self.intent_to_ids[rec.intent].append(rec.record_id)
        for token in set(rec.tokens):
            self.trie.insert(token, rec.record_id)
        self.graph.add_sample(list(rec.tokens), rec.record_id)
        if vec is not None:
            self.record_vectors[rec.record_id] = vec

    @staticmethod
    def _embed_text(text: str) -> Optional[List[float]]:
        try:
            from system.core.embeddings import encode_text

            vec = encode_text(text)
            arr = np.asarray(vec, dtype="float32")
            return arr.tolist()
        except Exception:
            return None

    def save(self) -> bool:
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "ts": time.time(),
                "next_id": int(self._next_id),
                "records": [
                    {
                        "record_id": int(rec.record_id),
                        "intent": rec.intent,
                        "user": rec.user,
                        "assistant": rec.assistant,
                        "tokens": list(rec.tokens),
                    }
                    for rec in self.records.values()
                ],
                "vectors": {str(rid): vec for rid, vec in self.record_vectors.items()},
            }
            tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self.index_path)
            bak_path = self.index_path.with_suffix(self.index_path.suffix + ".bak")
            try:
                bak_tmp = bak_path.with_suffix(bak_path.suffix + ".tmp")
                bak_tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                bak_tmp.replace(bak_path)
            except Exception:
                pass
            self._pending_since_save = 0
            return True
        except Exception as e:
            logger.warning("Hybrid semantic index save failed: %s", e)
            return False

    def load(self) -> bool:
        if not self.index_path.exists():
            return False
        source_path = self.index_path
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as e:
            size = -1
            try:
                size = int(self.index_path.stat().st_size)
            except Exception:
                size = -1
            bak_path = self.index_path.with_suffix(self.index_path.suffix + ".bak")
            if bak_path.exists():
                try:
                    raw = json.loads(bak_path.read_text(encoding="utf-8"))
                    source_path = bak_path
                    logger.warning(
                        "Hybrid semantic index load fallback: path=%s size=%d err=%s -> using backup=%s",
                        self.index_path,
                        size,
                        e,
                        bak_path,
                    )
                except Exception:
                    logger.warning(
                        "Hybrid semantic index load failed: path=%s size=%d err=%s",
                        self.index_path,
                        size,
                        e,
                    )
                    return False
            else:
                logger.warning(
                    "Hybrid semantic index load failed: path=%s size=%d err=%s",
                    self.index_path,
                    size,
                    e,
                )
                return False
        try:
            records = raw.get("records", []) if isinstance(raw, dict) else []
            vectors = raw.get("vectors", {}) if isinstance(raw, dict) else {}
            self.clear(reset_persistence_counter=True)
            for item in records:
                if not isinstance(item, dict):
                    continue
                rid = int(item.get("record_id", -1))
                intent = str(item.get("intent", "default"))
                user = str(item.get("user", "")).strip()
                assistant = str(item.get("assistant", "")).strip()
                tokens_raw = item.get("tokens", [])
                if rid < 0 or not user or not assistant:
                    continue
                tokens = tuple(str(t).strip() for t in tokens_raw if str(t).strip()) if isinstance(tokens_raw, list) else tuple(_tokenize(user))
                rec = HybridRecord(
                    record_id=rid,
                    intent=intent,
                    user=user,
                    assistant=assistant,
                    tokens=tokens,
                )
                vec = None
                if isinstance(vectors, dict):
                    vv = vectors.get(str(rid))
                    if isinstance(vv, list):
                        try:
                            vec = [float(x) for x in vv]
                        except Exception:
                            vec = None
                self._insert_record(rec, vec=vec)
                self._next_id = max(self._next_id, rid + 1)
            next_id = raw.get("next_id") if isinstance(raw, dict) else None
            if isinstance(next_id, int):
                self._next_id = max(self._next_id, int(next_id))
            if source_path != self.index_path:
                try:
                    self.save()
                except Exception:
                    pass
            logger.info("Hybrid semantic index loaded. records=%d", len(self.records))
            return True
        except Exception as e:
            logger.warning("Hybrid semantic index restore failed: %s", e)
            self.clear(reset_persistence_counter=True)
            return False

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        if denom <= 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def _vector_similarity(self, query: str, record_id: int, record_user: str, query_vec: Optional[np.ndarray]) -> float:
        if query_vec is not None:
            vv = self.record_vectors.get(record_id)
            if isinstance(vv, list) and vv:
                try:
                    rv = np.asarray(vv, dtype="float32")
                    return self._cosine(query_vec, rv)
                except Exception:
                    pass
        return self.reranker.similarity(query, record_user)

    def query(
        self,
        query_text: str,
        intent: Optional[str],
        top_k: int = 5,
        use_embeddings: bool = True,
        transformer_similarity: Optional[Callable[[str, str], float]] = None,
        trace_meta: Optional[Dict[str, Any]] = None,
    ) -> List[HybridHit]:
        query = (query_text or "").strip()
        if not query:
            return []
        q_tokens = _tokenize(query)
        if not self.records:
            return []

        trie_ids = self.trie.retrieve_for_tokens(q_tokens) if q_tokens else set()
        graph_ids, expanded_tokens = self.graph.candidate_ids(
            q_tokens,
            depth=self.graph_depth,
            max_neighbors=self.graph_neighbors,
            min_edge_weight=self.graph_min_edge,
        )
        candidate_ids = set(trie_ids) | set(graph_ids)

        chosen_intent = str(intent or "").strip()
        if chosen_intent and chosen_intent not in {"default", "other"}:
            scoped = set(self.intent_to_ids.get(chosen_intent, []))
            if scoped:
                if candidate_ids:
                    intersect = candidate_ids & scoped
                    if len(intersect) >= self.min_candidates_for_intent:
                        candidate_ids = intersect
                    else:
                        candidate_ids = intersect | set(list(scoped)[-self.min_candidates_for_intent :])
                else:
                    candidate_ids = set(list(scoped)[-self.fallback_candidates :])

        if not candidate_ids:
            candidate_ids = set(list(self.records.keys())[-self.fallback_candidates :])

        original_candidate_size = len(candidate_ids)
        if len(candidate_ids) > self.max_candidates:
            ranked = sorted(
                list(candidate_ids),
                key=lambda rid: self._quick_overlap(q_tokens, self.records[rid].tokens),
                reverse=True,
            )
            candidate_ids = set(ranked[: self.max_candidates])

        query_vec: Optional[np.ndarray] = None
        if use_embeddings:
            try:
                from system.core.embeddings import encode_text

                query_vec = np.asarray(encode_text(query), dtype="float32")
            except Exception:
                query_vec = None

        hits: List[HybridHit] = []
        for rid in candidate_ids:
            rec = self.records.get(rid)
            if rec is None:
                continue
            lex = _jaccard_tokens(q_tokens, rec.tokens)
            graph_score = self.graph.score_overlap(q_tokens, list(rec.tokens), expanded_tokens)
            vec = 0.0
            if use_embeddings:
                vec = max(vec, self._vector_similarity(query, rid, rec.user, query_vec))
            if transformer_similarity is not None:
                try:
                    vec = max(vec, float(transformer_similarity(query, rec.user)))
                except Exception:
                    vec = max(vec, 0.0)
            final = (self.weight_lex * lex) + (self.weight_graph * graph_score) + (self.weight_vec * vec)
            if chosen_intent and rec.intent == chosen_intent and chosen_intent not in {"default", "other"}:
                final += self.intent_bonus
            hits.append(
                HybridHit(
                    record_id=rid,
                    intent=rec.intent,
                    user=rec.user,
                    response=rec.assistant,
                    score=float(final),
                    lexical_score=float(lex),
                    graph_score=float(graph_score),
                    vector_score=float(vec),
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        out = hits[: max(1, int(top_k))]
        self._write_trace(
            query=query,
            intent=chosen_intent,
            query_tokens=q_tokens,
            trie_count=len(trie_ids),
            graph_count=len(graph_ids),
            candidate_count_before_limit=original_candidate_size,
            candidate_count_after_limit=len(candidate_ids),
            hits=out,
            extra=trace_meta or {},
        )
        return out

    def _write_trace(
        self,
        query: str,
        intent: str,
        query_tokens: List[str],
        trie_count: int,
        graph_count: int,
        candidate_count_before_limit: int,
        candidate_count_after_limit: int,
        hits: List[HybridHit],
        extra: Dict[str, Any],
    ) -> None:
        if not self.trace_enabled:
            return
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ts": time.time(),
                "query": query,
                "intent": intent or "",
                "query_tokens": query_tokens[:32],
                "candidates": {
                    "trie": int(trie_count),
                    "graph": int(graph_count),
                    "before_limit": int(candidate_count_before_limit),
                    "after_limit": int(candidate_count_after_limit),
                },
                "top_hits": [
                    {
                        "record_id": int(h.record_id),
                        "intent": h.intent,
                        "score": float(h.score),
                        "lexical_score": float(h.lexical_score),
                        "graph_score": float(h.graph_score),
                        "vector_score": float(h.vector_score),
                        "user": h.user[:240],
                        "response": h.response[:240],
                    }
                    for h in hits[:10]
                ],
                "extra": extra or {},
            }
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Hybrid semantic trace write failed", exc_info=True)

    @staticmethod
    def _quick_overlap(query_tokens: List[str], candidate_tokens: Iterable[str]) -> float:
        if not query_tokens:
            return 0.0
        q = set(query_tokens)
        c = set(candidate_tokens)
        if not c:
            return 0.0
        return float(len(q & c))
