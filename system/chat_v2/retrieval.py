# -*- coding: utf-8 -*-
"""Retrieval manager — memory and RAG context retrieval, formatting, and augmentation.

Extracted from ChatPipeline (pipeline.py) to improve cohesion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .memory import CuratedMemoryStore
from .router import RouteKind, RouteProfile


class RetrievalManager:
    """Manages memory and RAG retrieval for the chat pipeline."""

    def __init__(
        self,
        memory_store: CuratedMemoryStore,
        rag_store: Any | None = None,
        *,
        memory_topk: int = 3,
        memory_max_chars: int = 900,
        rag_topk: int = 4,
        rag_min_sim: float = 0.2,
        rag_max_chars: int = 1200,
        enable_memory_retrieval: bool = True,
        enable_rag: bool = False,
    ) -> None:
        self.memory_store = memory_store
        self.rag_store = rag_store
        self.memory_topk = max(1, int(memory_topk))
        self.memory_max_chars = max(120, int(memory_max_chars))
        self.rag_topk = max(1, int(rag_topk))
        self.rag_min_sim = float(rag_min_sim)
        self.rag_max_chars = max(120, int(rag_max_chars))
        self.enable_memory_retrieval = bool(enable_memory_retrieval)
        self.enable_rag = bool(enable_rag)

    # -- memory retrieval --------------------------------------------------------

    def memory_score_threshold(self, route: RouteProfile | None) -> float:
        if route is None:
            return 0.0
        if route.kind in {RouteKind.PROJECT_QA, RouteKind.REASONING, RouteKind.WORKSPACE_ACTION}:
            return 0.68
        if route.kind == RouteKind.FACTUAL_QA:
            return 0.78
        return 0.74

    def filter_memory_hits(self, memory_hits: Sequence[Any], route: RouteProfile | None) -> List[Any]:
        hits = list(memory_hits or [])
        threshold = float(self.memory_score_threshold(route))
        if threshold <= 0.0:
            return hits
        filtered = []
        for hit in hits:
            try:
                score = float(getattr(hit, "score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if score >= threshold:
                filtered.append(hit)
        return filtered

    def memory_hits(self, query: str, route: RouteProfile | None = None) -> List[Any]:
        if not bool(self.enable_memory_retrieval):
            return []
        try:
            hits = list(self.memory_store.search_many(query, top_k=self.memory_topk))
        except Exception:
            return []
        return self.filter_memory_hits(hits, route)

    def memory_block(self, memory_hits: List[Any]) -> str:
        if not memory_hits:
            return ""
        parts: List[str] = []
        total = 0
        limit = self.memory_max_chars
        for idx, hit in enumerate(memory_hits, 1):
            user = str(getattr(hit, "user", "") or "").strip()
            assistant = str(getattr(hit, "assistant", "") or "").strip()
            score = float(getattr(hit, "score", 0.0) or 0.0)
            if not user or not assistant:
                continue
            block = f"[{idx}] Similar dialogue (score={score:.2f})\nUser: {user}\nAssistant: {assistant}"
            if total + len(block) > limit:
                remain = limit - total
                if remain < 80:
                    break
                block = block[:remain].rstrip()
            if not block:
                break
            parts.append(block)
            total += len(block)
            if total >= limit:
                break
        return "\n\n".join(parts)

    # -- RAG retrieval -----------------------------------------------------------

    def rag_context(self, query: str) -> Optional[Dict[str, Any]]:
        if not bool(self.enable_rag) or self.rag_store is None:
            return None
        try:
            return self.rag_store.build_context(
                query,
                k=self.rag_topk,
                min_sim=self.rag_min_sim,
                max_chars=self.rag_max_chars,
            )
        except Exception:
            return None

    def rag_block(self, rag_context: Optional[Dict[str, Any]]) -> str:
        if not rag_context:
            return ""
        snippets = list(rag_context.get("snippets", []) or [])
        if not snippets:
            return ""
        parts: List[str] = []
        for idx, item in enumerate(snippets, 1):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "") or "").strip()
            path_str = str(item.get("path", "") or "").strip()
            if not text:
                continue
            label = f"[{idx}]"
            if path_str:
                label = f"[{idx}] {path_str}"
            parts.append(f"{label}\n{text}")
        return "\n\n".join(parts)

    def rag_reply(self, query: str, rag_context: Optional[Dict[str, Any]]) -> str:
        if not rag_context:
            return ""
        snippets = list(rag_context.get("snippets", []) or [])
        if not snippets:
            return ""
        first = snippets[0] if isinstance(snippets[0], dict) else {}
        text = str(first.get("text", "") or "").strip()
        path_str = str(first.get("path", "") or "").strip()
        if not text:
            return ""
        if path_str:
            return f"According to retrieved project context from {path_str}, {text}"
        return f"According to retrieved project context, {text}"

    def has_rag_evidence(self, rag_context: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(rag_context, dict):
            return False
        for item in list(rag_context.get("snippets", []) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("text", "") or "").strip():
                return True
        return False

    def rag_snippet_count(self, rag_context: Optional[Dict[str, Any]]) -> int:
        if not isinstance(rag_context, dict):
            return 0
        return int(len(list(rag_context.get("snippets", []) or [])))

    # -- route guidance ----------------------------------------------------------

    @staticmethod
    def route_guidance(route: RouteProfile) -> str:
        kind = route.kind
        if kind == RouteKind.WORKSPACE_ACTION:
            return (
                "The user is asking about the workspace. Answer only from grounded evidence, "
                "and do not invent files, commands, or test results."
            )
        if kind == RouteKind.PROJECT_QA:
            return (
                "The user wants project-specific guidance. Prefer retrieved project context, "
                "and if the docs or evidence are incomplete, say so clearly."
            )
        if kind == RouteKind.REASONING:
            return (
                "The user wants a decision or explanation. Give the conclusion first, "
                "then the key reason or tradeoff, and note missing evidence briefly."
            )
        if kind == RouteKind.FACTUAL_QA:
            return "Answer directly and practically. Prefer grounded evidence over generic advice."
        return "Answer concisely and stay grounded in the available evidence."

    # -- query augmentation ------------------------------------------------------

    def augment_query_with_context(
        self, query: str, memory_hits: List[Any], rag_context: Optional[Dict[str, Any]], route: RouteProfile,
    ) -> str:
        memory_block = self.memory_block(memory_hits)
        rag_block = self.rag_block(rag_context)
        guidance = self.route_guidance(route)
        if not memory_block and not rag_block:
            if not guidance:
                return query
            return f"{guidance}\n\nUser Question: {query}"
        parts: List[str] = []
        if memory_block:
            parts.append("Retrieved Dialogue Memory:\n" + memory_block)
        if rag_block:
            parts.append("Retrieved Project Context:\n" + rag_block)
        context_block = "\n\n".join(parts)
        return (
            f"{guidance}\n"
            "Use the retrieved dialogue memory and project context only when they are relevant.\n"
            "Prefer retrieved evidence over guesswork. If the evidence is insufficient, say so briefly.\n\n"
            f"{context_block}\n\n"
            f"User Question: {query}"
        )
