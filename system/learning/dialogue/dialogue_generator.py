# -*- coding: utf-8 -*-
"""Lightweight dialogue learner/generator used by main.py.

This module keeps an intent-keyed memory of (user, assistant) pairs,
retrieves similar examples at inference time, and ranks candidates with
simple quality gates. It is intentionally local/offline and dependency-light.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from system.knowledge.semantic.index import HybridSemanticIndex
from system.brain.persona import normalize_persona_reply, persona_prompt_reply, persona_rule_reply


logger = logging.getLogger(__name__)


class TransformerSentenceEncoder:
    """Optional transformer sentence encoder for semantic similarity."""

    def __init__(self, model_path: str = "", device: str = "") -> None:
        self.model_path = (model_path or os.environ.get("DIALOGUE_TRANSFORMER_MODEL", "")).strip()
        self.device = (device or os.environ.get("DIALOGUE_TRANSFORMER_DEVICE", "")).strip()
        self.local_only = os.environ.get("DIALOGUE_TRANSFORMER_LOCAL_ONLY", "1") != "0"
        self.max_tokens = max(16, int(os.environ.get("DIALOGUE_TRANSFORMER_MAX_TOKENS", "256")))
        self.max_cache = max(128, int(os.environ.get("DIALOGUE_TRANSFORMER_MAX_CACHE", "2048")))
        self._available = False
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._cache: Dict[str, Any] = {}
        if self.model_path:
            self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=self.local_only)
            model = AutoModel.from_pretrained(self.model_path, local_files_only=self.local_only)
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self.device = device
            self._available = True
            logger.info("Dialogue transformer encoder ready: %s (%s)", self.model_path, self.device)
        except Exception as e:
            self._available = False
            logger.warning("Dialogue transformer encoder disabled: %s", e)

    def available(self) -> bool:
        return self._available and self._model is not None and self._tokenizer is not None

    def encode(self, text: str):
        if not self.available():
            return None
        text = (text or "").strip()
        if not text:
            return None
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_tokens,
            )
            assert self._torch is not None
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                out = self._model(**inputs)
                hidden = out.last_hidden_state
                mask = inputs.get("attention_mask")
                if mask is None:
                    pooled = hidden.mean(dim=1)
                else:
                    m = mask.unsqueeze(-1).to(hidden.dtype)
                    denom = m.sum(dim=1).clamp(min=1e-6)
                    pooled = (hidden * m).sum(dim=1) / denom
                vec = pooled.squeeze(0)
                vec = vec / (vec.norm(p=2) + 1e-8)
                arr = vec.detach().cpu().numpy()
        except Exception:
            return None
        if len(self._cache) >= self.max_cache:
            self._cache.clear()
        self._cache[text] = arr
        return arr

    def similarity(self, a: str, b: str) -> float:
        va = self.encode(a)
        vb = self.encode(b)
        if va is None or vb is None:
            return 0.0
        try:
            import numpy as np

            denom = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
            return float(np.dot(va, vb) / denom)
        except Exception:
            return 0.0


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


def _normalize_user_query(text: str) -> str:
    cur = str(text or "").strip()
    if not cur:
        return ""
    if re.search(r"[\u4e00-\u9fff]", cur):
        out = cur
        out = re.sub(r"[!！?？。．，,~～…\s]+$", "", out)
        out = re.sub(r"[啊呀哦噢呢嘛吧啦哈呗哟]+$", "", out)
        out = re.sub(r"[!！?？。．，,~～…\s]+$", "", out)
        if out:
            return out
    return cur


def _rule_based_reply(text: str) -> str:
    return persona_rule_reply(text)

def _jaccard(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter / union) if union > 0 else 0.0


@dataclass
class DialoguePattern:
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    _pair_set: Set[Tuple[str, str]] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        cleaned: List[Tuple[str, str]] = []
        for user, assistant in self.pairs:
            u = str(user or "").strip()
            a = str(assistant or "").strip()
            if not u or not a:
                continue
            key = (u, a)
            if key in self._pair_set:
                continue
            self._pair_set.add(key)
            cleaned.append(key)
        self.pairs = cleaned

    def add_pair(self, user: str, assistant: str) -> bool:
        u = str(user or "").strip()
        a = str(assistant or "").strip()
        if not u or not a:
            return False
        key = (u, a)
        if key in self._pair_set:
            return False
        self._pair_set.add(key)
        self.pairs.append(key)
        return True


class DialogueMemory:
    def __init__(self, path: str = "artifacts/memory/dialogue_patterns.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.patterns: Dict[str, DialoguePattern] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("dialogue memory load failed: %s", e)
            return
        if not isinstance(raw, dict):
            return
        patterns: Dict[str, DialoguePattern] = {}
        for intent, item in raw.items():
            if not isinstance(item, dict):
                continue
            pairs_raw = item.get("pairs", [])
            pairs: List[Tuple[str, str]] = []
            if isinstance(pairs_raw, list):
                for p in pairs_raw:
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        continue
                    u = str(p[0]).strip()
                    a = str(p[1]).strip()
                    if u and a:
                        pairs.append((u, a))
            if pairs:
                patterns[str(intent)] = DialoguePattern(pairs=pairs)
        self.patterns = patterns

    def save(self) -> None:
        data: Dict[str, Dict[str, Any]] = {}
        for intent, pattern in self.patterns.items():
            data[intent] = {"pairs": pattern.pairs}
        try:
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("dialogue memory save failed: %s", e)

    def update(self, intent: str, user_input: str, response: str) -> bool:
        key = str(intent or "default")
        if key not in self.patterns:
            self.patterns[key] = DialoguePattern()
        return self.patterns[key].add_pair(user_input, response)


class ResponseRanker:
    BAD_MARKERS = (
        "need more information",
        "provide more details",
        "still learning",
        "not sure",
        "cannot answer this",
        "please clarify",
        "insufficient context",
        "missing constraints",
        "missing requirements",
        "tell me your goal and constraints",
        "executable plan",
        "i can help right now: share your goal",
    )


    @staticmethod
    def rank_responses(candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        merged: Dict[str, float] = {}
        for resp, base in candidates:
            text = (resp or "").strip()
            if not text:
                continue
            score = float(base)
            if any(m in text for m in ResponseRanker.BAD_MARKERS):
                score -= 0.5
            if len(text) < 4:
                score -= 0.2
            if len(text) > 120:
                score -= 0.1
            old = merged.get(text)
            if old is None or score > old:
                merged[text] = score
        return sorted(merged.items(), key=lambda x: x[1], reverse=True)


class IntentClassifier:
    """Similarity-based intent classifier with optional transformer semantics."""

    def __init__(
        self,
        memory: DialogueMemory,
        transformer_encoder: Optional[TransformerSentenceEncoder] = None,
    ) -> None:
        self._memory = memory
        self._transformer = transformer_encoder
        self._min_sim = float(os.environ.get("DIALOGUE_INTENT_MIN_SIM", "0.42"))
        self._samples_per_intent = max(1, int(os.environ.get("DIALOGUE_INTENT_SAMPLES", "12")))

    def _semantic_similarity(self, a: str, b: str) -> float:
        sim = _jaccard(a, b)
        if self._transformer is not None:
            sim = max(sim, self._transformer.similarity(a, b))
        return sim

    def classify(self, user_input: str) -> str:
        text = (user_input or "").strip()
        if not text:
            return "default"
        patterns = self._memory.patterns
        if not patterns:
            return "default"

        best_intent = "default"
        best_score = 0.0
        for intent, pattern in patterns.items():
            if intent in {"default", "other"}:
                continue
            if not pattern.pairs:
                continue
            local_best = 0.0
            for sample, _ in pattern.pairs[-self._samples_per_intent :]:
                score = self._semantic_similarity(text, sample)
                if score > local_best:
                    local_best = score
            if local_best > best_score:
                best_score = local_best
                best_intent = intent

        if best_score >= self._min_sim:
            return best_intent
        return "default"


class DialogueGenerator:
    def __init__(
        self,
        nanobrain=None,
        experience_store=None,
        chat_memory=None,
        rule_miner=None,
        transformer_model: str = "",
        transformer_device: str = "",
        memory_path: str = "artifacts/memory/dialogue_patterns.json",
    ):
        self.nb = nanobrain
        self.exp = experience_store
        self.chat = chat_memory
        self.miner = rule_miner
        self.memory = DialogueMemory(path=memory_path)
        self._transformer_encoder = TransformerSentenceEncoder(
            model_path=transformer_model,
            device=transformer_device,
        )
        self.intent_classifier = IntentClassifier(
            self.memory,
            transformer_encoder=self._transformer_encoder,
        )
        self.response_ranker = ResponseRanker()
        self._use_hybrid_index = os.environ.get("DIALOGUE_HYBRID_INDEX", "1") != "0"
        hybrid_index_path = os.environ.get("DIALOGUE_HYBRID_INDEX_PATH", "").strip()
        if not hybrid_index_path:
            hybrid_index_path = str(Path(memory_path).with_name("hybrid_semantic_index.json"))
        trace_path = os.environ.get("DIALOGUE_REASON_TRACE_PATH", "").strip()
        if not trace_path:
            trace_path = str(Path(memory_path).with_name("dialogue_reason_trace.jsonl"))
        self._hybrid_index = (
            HybridSemanticIndex(index_path=hybrid_index_path, trace_path=trace_path)
            if self._use_hybrid_index
            else None
        )
        self._use_embeddings = os.environ.get("DIALOGUE_EMBEDDINGS", "1") != "0"
        self._embed_cache: Dict[str, Any] = {}
        self._sim_min = float(os.environ.get("DIALOGUE_SIM_MIN", "0.50"))
        self._sim_gap = float(os.environ.get("DIALOGUE_SIM_GAP", "0.05"))
        self._retrieve_min_sim = float(os.environ.get("DIALOGUE_RETRIEVE_MIN_SIM", "0.35"))
        self._max_pairs_for_retrieval = max(100, int(os.environ.get("DIALOGUE_MAX_RETRIEVE_PAIRS", "1500")))
        self._max_pairs_per_intent = int(os.environ.get("DIALOGUE_MAX_PAIRS_PER_INTENT", "4000"))
        self._save_every = max(1, int(os.environ.get("DIALOGUE_SAVE_EVERY", "10")))
        self._pending_since_save = 0
        self._rebuild_hybrid_index()

    def _rebuild_hybrid_index(self) -> None:
        if self._hybrid_index is None:
            return
        records: List[Tuple[str, str, str]] = []
        for intent, pattern in self.memory.patterns.items():
            for user, assistant in pattern.pairs:
                records.append((intent, user, assistant))
        self._hybrid_index.build(records)

    def learn_sample(self, user_input: str, response: str, intent: Optional[str] = None, persist: bool = False) -> bool:
        u = (user_input or "").strip()
        a = normalize_persona_reply(u, response)
        if not u or not a:
            return False
        if intent is None:
            try:
                from system.computer_use.dialog_state import route_dialog_state

                intent = route_dialog_state(u).state
            except Exception:
                intent = "default"
        key = str(intent or "default")
        added = self.memory.update(key, u, a)
        if not added:
            return False
        if self._hybrid_index is not None:
            self._hybrid_index.add_record(intent=key, user=u, assistant=a, persist=False)
        pattern = self.memory.patterns.get(key)
        truncated = False
        if pattern and len(pattern.pairs) > self._max_pairs_per_intent:
            pattern.pairs = pattern.pairs[-self._max_pairs_per_intent :]
            truncated = True
        if truncated and self._hybrid_index is not None:
            self._rebuild_hybrid_index()
        self._pending_since_save += 1
        if persist or self._pending_since_save >= self._save_every:
            self.memory.save()
            if self._hybrid_index is not None:
                self._hybrid_index.save()
            self._pending_since_save = 0
        return True

    def flush(self) -> None:
        try:
            self.memory.save()
        except Exception:
            pass
        if self._hybrid_index is not None:
            try:
                self._hybrid_index.save()
            except Exception:
                pass

    def _embed(self, text: str):
        if not self._use_embeddings or not text:
            return None
        if text in self._embed_cache:
            return self._embed_cache[text]
        try:
            from system.core.embeddings import encode_text

            vec = encode_text(text)
        except Exception:
            return None
        self._embed_cache[text] = vec
        return vec

    def _embedding_similarity(self, a: str, b: str) -> float:
        v1 = self._embed(a)
        v2 = self._embed(b)
        if v1 is None or v2 is None:
            return 0.0
        try:
            from system.core.embeddings import cosine

            return float(cosine(v1, v2))
        except Exception:
            return 0.0

    def _transformer_similarity(self, a: str, b: str) -> float:
        return self._transformer_encoder.similarity(a, b)

    def _semantic_similarity(self, a: str, b: str) -> float:
        sim = _jaccard(a, b)
        if sim < 0.9:
            sim = max(sim, self._transformer_similarity(a, b))
        if self._use_embeddings and sim < 0.9:
            sim = max(sim, self._embedding_similarity(a, b))
        return sim

    def _iter_pairs(self, intent: str | None) -> List[Tuple[str, str]]:
        if intent and intent in self.memory.patterns and intent not in {"default", "other"}:
            pairs = list(self.memory.patterns[intent].pairs)
            if len(pairs) > self._max_pairs_for_retrieval:
                return pairs[-self._max_pairs_for_retrieval :]
            return pairs
        out: List[Tuple[str, str]] = []
        for pattern in self.memory.patterns.values():
            out.extend(pattern.pairs)
        if len(out) > self._max_pairs_for_retrieval:
            return out[-self._max_pairs_for_retrieval :]
        return out

    def _retrieve_similar(self, user_input: str, intent: str | None) -> List[Tuple[str, float]]:
        if self._hybrid_index is not None:
            hits = self._hybrid_index.query(
                query_text=user_input,
                intent=intent,
                top_k=5,
                use_embeddings=self._use_embeddings,
                transformer_similarity=self._transformer_similarity,
                trace_meta={"component": "dialogue_generator"},
            )
            if hits:
                return [(h.response, float(h.score)) for h in hits]
        pairs = self._iter_pairs(intent)
        if not pairs:
            return []
        scored: List[Tuple[str, float]] = []
        for inp, resp in pairs:
            sim = self._semantic_similarity(user_input, inp)
            if sim >= self._retrieve_min_sim:
                scored.append((resp, float(sim)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:5]

    def _fallback_response(self) -> str:
        return persona_prompt_reply()

    def generate_response(self, user_input: str, intent: str | None = None) -> str:
        text = (user_input or "").strip()
        if not text:
            return self._fallback_response()
        norm_text = _normalize_user_query(text)
        query = norm_text or text
        ruled = _rule_based_reply(query)
        if ruled:
            return ruled
        chosen_intent = intent or self.intent_classifier.classify(query)
        similar = self._retrieve_similar(query, chosen_intent)
        response_votes: Dict[str, int] = {}
        if similar:
            # Hybrid index often returns duplicate response texts; collapse first so
            # margin gating uses distinct candidates instead of identical duplicates.
            dedup: Dict[str, float] = {}
            for resp, score in similar:
                r = str(resp or "").strip()
                if not r:
                    continue
                response_votes[r] = int(response_votes.get(r, 0)) + 1
                prev = dedup.get(r)
                s = float(score)
                if prev is None or s > prev:
                    dedup[r] = s
            similar = sorted(
                dedup.items(),
                key=lambda x: (x[1], float(response_votes.get(x[0], 0))),
                reverse=True,
            )
        if similar:
            top = float(similar[0][1])
            second = float(similar[1][1]) if len(similar) > 1 else 0.0
            top_votes = int(response_votes.get(str(similar[0][0]), 0))
            second_votes = int(response_votes.get(str(similar[1][0]), 0)) if len(similar) > 1 else 0
            if top < self._sim_min:
                similar = []
            elif len(similar) > 1 and (top - second) < self._sim_gap and top_votes < second_votes:
                similar = []
        candidates: List[Tuple[str, float]] = []
        for resp, sim in similar[:3]:
            candidates.append((resp, 0.65 + 0.35 * sim))

        if self.nb is not None:
            try:
                nb_resp = self.nb.generate(query, context={"intent": chosen_intent})
            except Exception:
                nb_resp = ""
            if nb_resp:
                candidates.append((str(nb_resp), 0.80))

        candidates.append((self._fallback_response(), 0.55))
        ranked = self.response_ranker.rank_responses(candidates)
        best = ranked[0][0] if ranked else self._fallback_response()
        return normalize_persona_reply(user_input, best)

    def train(
        self,
        dialogue_data: List[Dict[str, str]],
        epochs: int = 10,
        learning_rate: float = 0.01,
    ) -> Dict[str, Any]:
        del learning_rate  # kept for API compatibility
        total = 0
        for _ in range(max(1, int(epochs))):
            for sample in dialogue_data:
                user_input = str(sample.get("user", "")).strip()
                response = str(sample.get("assistant", "")).strip()
                if not user_input or not response:
                    continue
                if self.learn_sample(user_input=user_input, response=response, persist=False):
                    total += 1
        self.memory.save()
        self._pending_since_save = 0
        return {
            "total_samples": len(dialogue_data),
            "epochs": max(1, int(epochs)),
            "patterns_learned": total,
            "templates_created": len(self.memory.patterns),
            "loss_history": [],
        }

    def evaluate(self, test_data: List[Dict[str, str]]) -> Dict[str, Any]:
        total = len(test_data)
        if total <= 0:
            return {
                "accuracy": 0.0,
                "strict_accuracy": 0.0,
                "avg_bleu": 0.0,
                "total_samples": 0,
                "correct": 0,
                "strict_correct": 0,
                "threshold": 0.6,
            }
        threshold = 0.6
        soft_correct = 0
        strict_correct = 0
        sims: List[float] = []
        for sample in test_data:
            user_input = str(sample.get("user", ""))
            expected = str(sample.get("assistant", ""))
            generated = self.generate_response(user_input)
            if generated == expected:
                strict_correct += 1
            sim = _jaccard(generated, expected)
            sims.append(sim)
            if sim >= threshold:
                soft_correct += 1
        return {
            "accuracy": float(soft_correct / total),
            "strict_accuracy": float(strict_correct / total),
            "avg_bleu": float(sum(sims) / max(1, len(sims))),
            "total_samples": total,
            "correct": soft_correct,
            "strict_correct": strict_correct,
            "threshold": threshold,
        }

