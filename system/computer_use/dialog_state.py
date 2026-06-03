from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from system.core.embeddings import encode_text, cosine


@dataclass
class DialogState:
    state: str  # "task" | "smalltalk" | "clarify"
    reason: str
    prompt: str = ""


_DEFAULT_CLARIFY = "能具体一点吗？你希望我做什么？"


def _read_json_dict(path: Path) -> Dict[str, List[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if not isinstance(v, list):
            continue
        items = [str(x).strip() for x in v if str(x).strip()]
        if items:
            out[k] = items
    return out


def _examples_from_dialogue_memory(limit: int = 200) -> List[str]:
    path = Path("artifacts/memory/dialogue_patterns.json")
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    examples: List[str] = []
    if isinstance(data, dict):
        for pattern in data.values():
            pairs = pattern.get("pairs") if isinstance(pattern, dict) else None
            if not isinstance(pairs, list):
                continue
            for pair in pairs:
                if not isinstance(pair, list) or not pair:
                    continue
                text = str(pair[0]).strip()
                if text:
                    examples.append(text)
                if len(examples) >= limit:
                    return examples
    return examples


def _examples_from_experience_store(limit: int = 200) -> List[str]:
    path = Path("artifacts/memory/experience_store.jsonl")
    if not path.exists():
        return []
    examples: List[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if str(item.get("state_text", "")) == "on_the_fly_dialogue":
            continue
        meta = item.get("meta")
        if isinstance(meta, dict) and meta.get("kind") == "chat":
            continue
        goal = str(item.get("goal", "")).strip()
        if goal:
            examples.append(goal)
        if len(examples) >= limit:
            break
    return examples


def _examples_from_train_file(limit: int = 200) -> List[str]:
    path = Path("train.txt")
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
    examples: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        item = None
        try:
            item = json.loads(line)
        except Exception:
            item = None
        if isinstance(item, dict):
            user = str(item.get("user", "")).strip()
            if user:
                examples.append(user)
        else:
            examples.append(line)
        if len(examples) >= limit:
            break
    return examples


class IntentRouter:
    def __init__(self) -> None:
        self.min_score = float(os.environ.get("INTENT_MIN_SCORE", "0.42"))
        self.min_delta = float(os.environ.get("INTENT_MIN_DELTA", "0.04"))
        self.max_examples = int(os.environ.get("INTENT_MAX_EXAMPLES", "200"))
        self._examples: Dict[str, List[str]] = {}
        self._centroids: Dict[str, np.ndarray] = {}
        self._load_examples()
        self._build_centroids()

    def _load_examples(self) -> None:
        sources: List[str] = []
        env_paths = os.environ.get("INTENT_EXAMPLES", "").strip()
        if env_paths:
            sources.extend([p for p in env_paths.split(os.pathsep) if p])
        sources.extend([
            "artifacts/memory/intent_examples.json",
            "config/intent_examples.json",
        ])
        merged: Dict[str, List[str]] = {}
        for raw in sources:
            p = Path(raw)
            if not p.exists():
                continue
            data = _read_json_dict(p)
            for k, items in data.items():
                merged.setdefault(k, []).extend(items)
        if merged:
            self._examples = merged
            return

        # Auto-seed from existing memories (no hardcoded keywords)
        smalltalk = _examples_from_dialogue_memory(limit=self.max_examples)
        task = _examples_from_experience_store(limit=self.max_examples)
        if not smalltalk:
            smalltalk = _examples_from_train_file(limit=self.max_examples)
        if smalltalk:
            self._examples["smalltalk"] = smalltalk
        if task:
            self._examples["task"] = task

    def _build_centroids(self) -> None:
        self._centroids = {}
        for label, items in self._examples.items():
            if not items:
                continue
            vecs = []
            for text in items[: self.max_examples]:
                try:
                    vecs.append(encode_text(text))
                except Exception:
                    continue
            if not vecs:
                continue
            centroid = np.mean(vecs, axis=0)
            norm = float(np.linalg.norm(centroid) + 1e-8)
            self._centroids[label] = centroid / norm

    def classify(self, text: str) -> str:
        if not text:
            return "clarify"
        if not self._centroids:
            return "clarify" if len(text) <= 2 else "task"
        try:
            vec = encode_text(text)
        except Exception:
            return "clarify" if len(text) <= 2 else "task"
        scores: Dict[str, float] = {}
        for label, centroid in self._centroids.items():
            scores[label] = cosine(vec, centroid)
        if not scores:
            return "clarify" if len(text) <= 2 else "task"
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else None
        if best_score < self.min_score:
            return "clarify" if len(text) <= 2 else "task"
        if second_score is not None and (best_score - second_score) < self.min_delta:
            return "clarify"
        return best_label


_ROUTER: IntentRouter | None = None


def _get_router() -> IntentRouter:
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = IntentRouter()
    return _ROUTER


def route_dialog_state(message: str) -> DialogState:
    text = (message or "").strip()
    if not text:
        return DialogState("clarify", "empty", _DEFAULT_CLARIFY)
    intent = _get_router().classify(text)
    if intent == "smalltalk":
        return DialogState("smalltalk", "similarity")
    if intent == "task":
        return DialogState("task", "similarity")
    return DialogState("clarify", "low_conf", _DEFAULT_CLARIFY)
