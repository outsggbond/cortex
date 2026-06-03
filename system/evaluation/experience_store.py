from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from memory.vector_index import VectorIndex
from system.core.embeddings import encode_text, embedding_available
from system.core.state_encoder import StateEncoder
from system.brain.world_model import WorldState


logger = logging.getLogger(__name__)


class ExperienceStore:
    def __init__(self, path: str = "artifacts/memory/experience_store.jsonl", capacity: int = 2000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self._items: List[Dict[str, Any]] = []
        self._index: VectorIndex | None = None
        self._dim: int | None = None
        self._encoder = StateEncoder()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        for line in lines[-self.capacity :]:
            try:
                self._items.append(json.loads(line))
            except Exception:
                continue
        self._build_index()

    def _build_index(self) -> None:
        self._index = None
        self._dim = None
        for idx, item in enumerate(self._items):
            goal = item.get("goal", "")
            before = item.get("before_state")
            if before:
                state = WorldState.from_dict(before)
                vec = self._encoder.encode(state, goal)
            else:
                text = item.get("state_text", "")
                if not text and not goal:
                    continue
                if embedding_available():
                    try:
                        vec = encode_text(f"{goal} | {text}", require_model=True)
                    except Exception:
                        vec = self._encoder.encode_text(text, goal)
                else:
                    vec = self._encoder.encode_text(text, goal)
            if self._dim is None:
                self._dim = int(vec.shape[0])
                self._index = VectorIndex(self._dim)
            if vec.shape[0] != self._dim or self._index is None:
                continue
            self._index.add(vec, id=str(idx))

    def add(
        self,
        state_text: str,
        goal: str,
        plan_steps: List[str],
        success: bool,
        score: float,
        meta: Optional[Dict[str, Any]] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        action_sig: str = "",
    ) -> None:
        entry = {
            "ts": time.time(),
            "state_text": state_text,
            "goal": goal,
            "plan_steps": plan_steps,
            "success": bool(success),
            "score": float(score),
            "before_state": before_state or {},
            "after_state": after_state or {},
            "action_sig": action_sig,
            "meta": meta or {},
        }
        self._items.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if len(self._items) > self.capacity:
            self._items = self._items[-self.capacity :]
            self._rebuild_file()
            self._build_index()
        else:
            if self._index is not None:
                try:
                    if before_state:
                        state = WorldState.from_dict(before_state)
                        vec = self._encoder.encode(state, goal)
                    elif embedding_available():
                        vec = encode_text(f"{goal} | {state_text}", require_model=True)
                    else:
                        vec = self._encoder.encode_text(state_text, goal)
                    if self._dim is None:
                        self._dim = int(vec.shape[0])
                        self._index = VectorIndex(self._dim)
                    if self._index is not None and vec.shape[0] == self._dim:
                        self._index.add(vec, id=str(len(self._items) - 1))
                except Exception:
                    logger.debug("experience_store: add index failed", exc_info=True)

    def retrieve_by_state(self, state: WorldState, goal: str, k: int = 3, success_only: bool = True) -> List[Dict[str, Any]]:
        if state is None:
            return []
        if self._index is not None:
            try:
                qv = self._encoder.encode(state, goal)
                hits = self._index.search(qv, k=k * 3)
                out: List[Dict[str, Any]] = []
                for hit in hits:
                    idx = hit[0] if isinstance(hit, (list, tuple)) and len(hit) >= 1 else None
                    if idx is None:
                        continue
                    try:
                        item = self._items[int(idx)]
                    except Exception:
                        continue
                    if success_only and not item.get("success", False):
                        continue
                    out.append(item)
                    if len(out) >= k:
                        break
                return out
            except Exception:
                logger.debug("experience_store: retrieve_by_state index failed", exc_info=True)
        # Fallback: lexical overlap
        out: List[Dict[str, Any]] = []
        q = (goal or "").lower()
        for item in reversed(self._items):
            if success_only and not item.get("success", False):
                continue
            text = (item.get("state_text", "") + " " + item.get("goal", "")).lower()
            if not text:
                continue
            if any(tok in text for tok in q.split()[:5]):
                out.append(item)
            if len(out) >= k:
                break
        return out

    def retrieve(self, query: str, k: int = 3, success_only: bool = True) -> List[Dict[str, Any]]:
        if not query:
            return []
        if self._index is not None:
            try:
                qv = encode_text(query, require_model=True) if embedding_available() else self._encoder.encode_text(query)
                hits = self._index.search(qv, k=k * 3)
                out: List[Dict[str, Any]] = []
                for hit in hits:
                    idx = hit[0] if isinstance(hit, (list, tuple)) and len(hit) >= 1 else None
                    if idx is None:
                        continue
                    try:
                        item = self._items[int(idx)]
                    except Exception:
                        continue
                    if success_only and not item.get("success", False):
                        continue
                    out.append(item)
                    if len(out) >= k:
                        break
                return out
            except Exception:
                logger.debug("experience_store: retrieve index failed", exc_info=True)
        return self.retrieve_by_state(WorldState(goal=""), query, k=k, success_only=success_only)

    def _rebuild_file(self) -> None:
        self.path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in self._items) + "\n",
            encoding="utf-8",
        )
