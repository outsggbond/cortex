from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from system.evaluation.response_guard import should_learn_from_chat


BUCKETS: Tuple[str, ...] = ("safety", "reasoning", "qa", "smalltalk")

SAFETY_KEYWORDS = (
    "违法",
    "非法",
    "危险",
    "攻击",
    "自杀",
    "炸弹",
    "武器",
    "毒品",
    "hack",
    "phishing",
    "exploit",
    "weapon",
    "malware",
    "ransomware",
)

REASONING_KEYWORDS = (
    "为什么",
    "原因",
    "推理",
    "分析",
    "步骤",
    "计划",
    "如何",
    "调试",
    "证明",
    "design",
    "debug",
    "reason",
    "optimiz",
    "strategy",
)

QA_KEYWORDS = (
    "?",
    "？",
    "什么",
    "谁",
    "哪里",
    "何时",
    "why",
    "what",
    "who",
    "when",
    "where",
)

INTENT_TO_BUCKET = {
    "safety": "safety",
    "secure": "safety",
    "security": "safety",
    "reason": "reasoning",
    "analysis": "reasoning",
    "debug": "reasoning",
    "qa": "qa",
    "question": "qa",
    "smalltalk": "smalltalk",
    "greeting": "smalltalk",
    "chitchat": "smalltalk",
}

_TEXT_TOKEN_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _tokenize_text(text: str) -> List[str]:
    cur = _norm_text(text).lower()
    tokens = [str(x) for x in _TEXT_TOKEN_RE.findall(cur) if str(x).strip()]
    if tokens:
        return tokens
    compact = re.sub(r"\s+", "", cur)
    if len(compact) <= 1:
        return [compact] if compact else []
    return [compact[i : i + 2] for i in range(len(compact) - 1)]


def _char_ngrams(text: str, *, n: int = 3) -> set[str]:
    cur = re.sub(r"\s+", "", _norm_text(text).lower())
    k = max(1, int(n))
    if not cur:
        return set()
    if len(cur) <= k:
        return {cur}
    return {cur[i : i + k] for i in range(0, len(cur) - k + 1)}


def _set_jaccard(a: set[str], b: set[str]) -> float:
    if (not a) or (not b):
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(max(1, union))


def _text_similarity(a: str, b: str) -> float:
    na = _norm_text(a).lower()
    nb = _norm_text(b).lower()
    if (not na) or (not nb):
        return 0.0
    tok = _set_jaccard(set(_tokenize_text(na)), set(_tokenize_text(nb)))
    gram = _set_jaccard(_char_ngrams(na, n=3), _char_ngrams(nb, n=3))
    seq = float(difflib.SequenceMatcher(a=na, b=nb).ratio())
    score = 0.45 * float(tok) + 0.35 * float(gram) + 0.20 * float(seq)
    return max(0.0, min(1.0, float(score)))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                yield item
    except Exception:
        return


def _contains_any(text: str, keywords: Tuple[str, ...] | List[str]) -> bool:
    t = str(text or "").lower()
    return any(k in t for k in keywords)


def _intent_bucket(intent: str) -> str:
    it = str(intent or "").strip().lower()
    if not it:
        return ""
    for k, v in INTENT_TO_BUCKET.items():
        if k in it:
            return v
    return ""


def _infer_rule_bucket(prompt: str, response: str, intent: str = "", source: str = "") -> Tuple[str, bool]:
    q = str(prompt or "").lower()
    a = str(response or "").lower()
    intent_l = str(intent or "").lower()
    source_l = str(source or "").lower()

    ib = _intent_bucket(intent_l)
    if ib == "safety":
        return "safety", True
    if ib == "reasoning":
        return "reasoning", True
    if ib == "qa":
        return "qa", True

    if _contains_any(q, SAFETY_KEYWORDS) or _contains_any(a, SAFETY_KEYWORDS):
        return "safety", True

    if _contains_any(q, REASONING_KEYWORDS) or _contains_any(a, REASONING_KEYWORDS):
        return "reasoning", True

    if _contains_any(q, QA_KEYWORDS):
        return "qa", True

    if "smalltalk" in intent_l or "greeting" in intent_l:
        return "smalltalk", True
    if source_l == "patterns" and len(q) <= 20 and len(a) <= 120:
        return "smalltalk", False
    return "smalltalk", False


def infer_bucket(prompt: str, response: str, intent: str = "", source: str = "") -> str:
    bucket, _ = _infer_rule_bucket(prompt, response, intent=intent, source=source)
    return bucket


class TinyBucketClassifier:
    """Small NB-like classifier used as a rebucketing fallback."""

    def __init__(self, *, alpha: float = 1.0, min_docs: int = 12) -> None:
        self.alpha = max(1e-6, float(alpha))
        self.min_docs = max(1, int(min_docs))
        self.word_counts: Dict[str, Counter[str]] = {b: Counter() for b in BUCKETS}
        self.token_totals: Dict[str, float] = {b: 0.0 for b in BUCKETS}
        self.doc_counts: Dict[str, int] = {b: 0 for b in BUCKETS}
        self.total_docs: int = 0
        self.vocab: set[str] = set()
        self.fitted: bool = False

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        t = str(text or "").lower()
        tokens = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]", t)
        if "?" in t or "？" in t:
            tokens.append("__question__")
        return tokens

    def _row_text(self, row: Dict[str, Any]) -> str:
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        intent = str(row.get("intent", "")).strip().lower()
        source = str(row.get("source", "")).strip().lower()
        return f"{prompt}\n{response}\nintent:{intent}\nsource:{source}"

    def fit(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            bucket = str(row.get("bucket", "")).strip().lower()
            if bucket not in BUCKETS:
                continue
            tokens = self._tokenize(self._row_text(row))
            if not tokens:
                continue
            score = float(row.get("score", 0.8))
            weight = max(0.25, min(3.0, score))
            if bool(row.get("hard_sample", False)):
                weight = min(3.5, weight * 1.2)
            self.doc_counts[bucket] += 1
            self.total_docs += 1
            for tok in tokens:
                self.word_counts[bucket][tok] += weight
                self.token_totals[bucket] += weight
                self.vocab.add(tok)

        active = sum(1 for b in BUCKETS if self.doc_counts[b] > 0)
        self.fitted = bool(self.total_docs >= self.min_docs and active >= 2 and len(self.vocab) > 4)

    def predict(
        self,
        *,
        prompt: str,
        response: str,
        intent: str = "",
        source: str = "",
    ) -> Tuple[str, float]:
        if not self.fitted:
            return "smalltalk", 0.0

        text = f"{prompt}\n{response}\nintent:{intent}\nsource:{source}"
        tokens = self._tokenize(text)
        if not tokens:
            return "smalltalk", 0.0

        vocab_size = max(1, len(self.vocab))
        denom_docs = float(self.total_docs) + self.alpha * float(len(BUCKETS))
        log_scores: Dict[str, float] = {}
        for bucket in BUCKETS:
            prior = (float(self.doc_counts[bucket]) + self.alpha) / denom_docs
            logp = math.log(prior)
            denom = float(self.token_totals[bucket]) + self.alpha * float(vocab_size)
            for tok in tokens:
                c = float(self.word_counts[bucket].get(tok, 0.0))
                logp += math.log((c + self.alpha) / denom)
            log_scores[bucket] = logp

        pred = max(log_scores.items(), key=lambda x: x[1])[0]
        max_log = max(log_scores.values())
        exps = {k: math.exp(v - max_log) for k, v in log_scores.items()}
        z = sum(exps.values()) + 1e-8
        conf = float(exps[pred] / z)
        return pred, conf


@dataclass
class DialogueDatasetBuildConfig:
    patterns_path: str = "artifacts/memory/dialogue_patterns.json"
    chat_memory_path: str = "artifacts/memory/chat_memory.jsonl"
    reflections_path: str = "artifacts/memory/dialogue_reflections.jsonl"
    output_train_path: str = "artifacts/memory/adapter_train.jsonl"
    output_eval_path: str = "artifacts/memory/adapter_eval.jsonl"
    output_manifest_path: str = "artifacts/memory/adapter_dataset_manifest.json"
    min_quality: float = 0.45
    min_user_chars: int = 2
    min_assistant_chars: int = 4
    max_samples: int = 20000
    eval_ratio: float = 0.1
    seed: int = 7
    include_patterns: bool = True
    include_chat_memory: bool = True
    include_reflections: bool = True
    hard_min_ratio: float = 0.30
    hard_max_ratio: float = 0.65
    max_per_prompt: int = 6
    max_per_response: int = 4
    strip_templatey_responses: bool = True
    prevent_cross_split_prompt_leakage: bool = True
    diversity_penalty_enabled: bool = True
    diversity_similarity_threshold: float = 0.82
    diversity_penalty_strength: float = 0.20
    diversity_reference_cap: int = 12

    # Hybrid bucket classifier
    bucket_classifier_enabled: bool = True
    bucket_classifier_min_conf: float = 0.62
    bucket_classifier_min_docs: int = 12
    bucket_classifier_alpha: float = 1.0
    bucket_classifier_force_override: bool = False


class DialogueDatasetBuilder:
    TEMPLATEY_MARKERS = (
        "need more information",
        "provide more details",
        "still learning",
        "not sure",
        "我需要更多信息",
        "请补充更多信息",
        "我不太确定",
        "我还在学习中",
    )

    def __init__(self, config: Optional[DialogueDatasetBuildConfig] = None) -> None:
        self.cfg = config or DialogueDatasetBuildConfig()

    def build(self) -> Dict[str, Any]:
        samples: List[Dict[str, Any]] = []
        if self.cfg.include_patterns:
            samples.extend(self._from_patterns(Path(self.cfg.patterns_path)))
        if self.cfg.include_chat_memory:
            samples.extend(self._from_chat_memory(Path(self.cfg.chat_memory_path)))
        if self.cfg.include_reflections:
            samples.extend(self._from_reflections(Path(self.cfg.reflections_path)))

        samples = self._rebucket_with_classifier(samples)
        samples = self._filter_and_dedup(samples)
        samples = self._apply_hard_weighting(samples, int(self.cfg.max_samples))
        train_samples, eval_samples = self._split_train_eval(samples)

        self._write_jsonl(Path(self.cfg.output_train_path), train_samples)
        self._write_jsonl(Path(self.cfg.output_eval_path), eval_samples)
        manifest = self._build_manifest(train_samples, eval_samples)
        self._write_manifest(Path(self.cfg.output_manifest_path), manifest)
        return manifest

    def _split_train_eval(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        pool = list(samples or [])
        if not pool:
            return [], []
        rng = random.Random(int(self.cfg.seed))
        rng.shuffle(pool)
        eval_n = 0
        if pool and float(self.cfg.eval_ratio) > 0:
            eval_n = int(round(len(pool) * float(self.cfg.eval_ratio)))
            eval_n = min(max(1, eval_n), max(1, len(pool) - 1))
        if eval_n <= 0:
            return pool, []
        if not bool(self.cfg.prevent_cross_split_prompt_leakage):
            eval_samples = list(pool[:eval_n])
            train_samples = list(pool[eval_n:])
            return train_samples, eval_samples

        groups: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for row in pool:
            key = _norm_text(str(row.get("prompt", "")))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(row)
        group_keys = list(order)
        rng.shuffle(group_keys)

        selected: List[str] = []
        selected_count = 0
        for key in group_keys:
            block = groups.get(key, [])
            if not block:
                continue
            if selected_count >= eval_n:
                break
            selected.append(key)
            selected_count += len(block)

        while len(selected) > 1:
            tail = selected[-1]
            tail_n = len(groups.get(tail, []))
            cur_dist = abs(int(selected_count) - int(eval_n))
            alt_dist = abs(int(selected_count - tail_n) - int(eval_n))
            if alt_dist <= cur_dist:
                selected.pop()
                selected_count -= tail_n
                continue
            break

        selected_set = set(selected)
        eval_samples = [row for key in selected for row in list(groups.get(key, []) or [])]
        train_samples = [row for key, rows in groups.items() if key not in selected_set for row in list(rows or [])]

        if (not train_samples) and eval_samples:
            move_key = selected[-1] if selected else ""
            if move_key:
                moved_rows = list(groups.get(move_key, []) or [])
                eval_samples = [row for row in eval_samples if row not in moved_rows]
                train_samples.extend(moved_rows)
        if (not eval_samples) and train_samples:
            by_size = sorted(
                [(key, len(list(groups.get(key, []) or []))) for key in groups.keys()],
                key=lambda x: x[1],
            )
            pick = by_size[0][0] if by_size else ""
            if pick:
                moved_rows = list(groups.get(pick, []) or [])
                eval_samples.extend(moved_rows)
                train_samples = [row for row in train_samples if row not in moved_rows]
        if not train_samples:
            train_samples = list(pool[eval_n:]) if len(pool) > eval_n else list(pool[1:])
        if not eval_samples:
            eval_samples = list(pool[:eval_n]) if eval_n > 0 else []
        return train_samples, eval_samples

    def _from_patterns(self, path: Path) -> List[Dict[str, Any]]:
        raw = _read_json(path)
        out: List[Dict[str, Any]] = []
        if not isinstance(raw, dict):
            return out
        for intent, node in raw.items():
            if not isinstance(node, dict):
                continue
            pairs = node.get("pairs")
            if not isinstance(pairs, list):
                continue
            for pair in pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                user = _norm_text(str(pair[0]))
                assistant = _norm_text(str(pair[1]))
                bucket = infer_bucket(user, assistant, intent=str(intent), source="patterns")
                out.append(
                    {
                        "prompt": user,
                        "response": assistant,
                        "source": "patterns",
                        "intent": str(intent),
                        "bucket": bucket,
                        "score": 0.75,
                        "hard_sample": False,
                    }
                )
        return out

    def _from_chat_memory(self, path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        pending_user = ""
        for item in _iter_jsonl(path):
            role = str(item.get("role", "")).strip().lower()
            text = _norm_text(str(item.get("text", "")))
            ts = item.get("timestamp")
            if role == "user":
                pending_user = text
                continue
            if role != "assistant":
                continue
            if not pending_user or not text:
                continue
            bucket = infer_bucket(pending_user, text, source="chat_memory")
            out.append(
                {
                    "prompt": pending_user,
                    "response": text,
                    "source": "chat_memory",
                    "intent": "",
                    "bucket": bucket,
                    "score": 0.60,
                    "ts": ts,
                    "hard_sample": False,
                }
            )
            pending_user = ""
        return out

    def _from_reflections(self, path: Path) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in _iter_jsonl(path):
            query = _norm_text(str(item.get("query", "")))
            repaired = _norm_text(str(item.get("repaired", "")))
            expected = _norm_text(str(item.get("expected", "")))
            reply = _norm_text(str(item.get("reply", "")))
            target = repaired or expected or ""
            if not target:
                continue
            quality = item.get("quality")
            try:
                q = float(quality)
            except Exception:
                q = 0.5
            q = max(0.0, min(1.0, q))
            reason = str(item.get("reason", "")).strip()
            tags = item.get("tags") if isinstance(item.get("tags"), list) else []
            bucket = infer_bucket(query, target, intent=str(item.get("context", "")), source="reflections")
            out.append(
                {
                    "prompt": query,
                    "response": target,
                    "source": "reflections",
                    "intent": str(item.get("context", "")),
                    "bucket": bucket,
                    "score": max(q, 0.70 if repaired else 0.55),
                    "hard_sample": True,
                    "meta": {
                        "reason": reason,
                        "tags": tags,
                        "original_reply": reply,
                        "suggested_action": str(item.get("suggested_action", "")),
                    },
                }
            )
        return out

    def _rebucket_with_classifier(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.cfg.bucket_classifier_enabled:
            return samples
        if len(samples) < int(self.cfg.bucket_classifier_min_docs):
            return samples

        clf = TinyBucketClassifier(
            alpha=float(self.cfg.bucket_classifier_alpha),
            min_docs=int(self.cfg.bucket_classifier_min_docs),
        )
        clf.fit(samples)
        if not clf.fitted:
            return samples

        out: List[Dict[str, Any]] = []
        conf_thr = float(self.cfg.bucket_classifier_min_conf)
        force = bool(self.cfg.bucket_classifier_force_override)
        for sample in samples:
            prompt = str(sample.get("prompt", "")).strip()
            response = str(sample.get("response", "")).strip()
            intent = str(sample.get("intent", "")).strip()
            source = str(sample.get("source", "")).strip()
            rule_bucket, rule_strong = _infer_rule_bucket(prompt, response, intent=intent, source=source)
            pred, conf = clf.predict(prompt=prompt, response=response, intent=intent, source=source)

            final_bucket = rule_bucket
            if conf >= conf_thr:
                if force:
                    final_bucket = pred
                elif rule_bucket == "smalltalk" and not rule_strong:
                    final_bucket = pred
                elif rule_bucket == "qa" and pred in {"reasoning", "safety"} and conf >= min(0.95, conf_thr + 0.10):
                    final_bucket = pred

            row = dict(sample)
            row["bucket"] = final_bucket
            row["bucket_rule"] = rule_bucket
            row["bucket_classifier"] = pred
            row["bucket_confidence"] = round(float(conf), 6)
            out.append(row)
        return out

    def _is_templatey_response(self, text: str) -> bool:
        if not self.cfg.strip_templatey_responses:
            return False
        t = str(text or "").strip().lower()
        if not t:
            return True
        if any(marker in t for marker in self.TEMPLATEY_MARKERS):
            return True
        return False

    def _filter_and_dedup(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
        prompt_counts: Counter[str] = Counter()
        response_counts: Counter[str] = Counter()

        for sample in samples:
            user = _norm_text(str(sample.get("prompt", "")))
            assistant = _norm_text(str(sample.get("response", "")))
            if len(user) < int(self.cfg.min_user_chars) or len(assistant) < int(self.cfg.min_assistant_chars):
                continue
            if not should_learn_from_chat(user, assistant):
                continue
            if self._is_templatey_response(assistant):
                continue
            score = float(sample.get("score", 0.0))
            if score < float(self.cfg.min_quality):
                continue
            key = (user, assistant)
            cur = best_by_pair.get(key)
            if cur is None or float(cur.get("score", 0.0)) < score:
                entry = dict(sample)
                entry["prompt"] = user
                entry["response"] = assistant
                entry["score"] = score
                entry["bucket"] = str(
                    sample.get("bucket")
                    or infer_bucket(user, assistant, str(sample.get("intent", "")), str(sample.get("source", "")))
                )
                entry["hard_sample"] = bool(sample.get("hard_sample", False))
                best_by_pair[key] = entry

        diversity_enabled = bool(self.cfg.diversity_penalty_enabled)
        sim_threshold = max(0.0, min(1.0, float(self.cfg.diversity_similarity_threshold)))
        penalty_strength = max(0.0, min(1.0, float(self.cfg.diversity_penalty_strength)))
        reference_cap = max(1, int(self.cfg.diversity_reference_cap))

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in list(best_by_pair.values()):
            prompt = str(item.get("prompt", ""))
            grouped.setdefault(prompt, []).append(item)

        ranked: List[Dict[str, Any]] = []
        for prompt, rows in grouped.items():
            ordered = sorted(rows, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            reference_responses: List[str] = []
            for item in ordered:
                response = str(item.get("response", ""))
                max_similarity = 0.0
                if diversity_enabled and reference_responses:
                    for ref in reference_responses[:reference_cap]:
                        sim = _text_similarity(response, ref)
                        if sim > max_similarity:
                            max_similarity = float(sim)
                penalty = 0.0
                if diversity_enabled and max_similarity >= sim_threshold:
                    denom = max(1e-6, 1.0 - float(sim_threshold))
                    penalty = float(penalty_strength) * max(0.0, float(max_similarity - sim_threshold)) / float(denom)
                score = float(item.get("score", 0.0))
                row = dict(item)
                row["diversity_max_similarity"] = round(float(max_similarity), 6)
                row["diversity_penalty"] = round(float(penalty), 6)
                row["selection_score"] = round(float(score) - float(penalty), 6)
                ranked.append(row)
                reference_responses.append(response)

        out: List[Dict[str, Any]] = []
        max_per_prompt = max(1, int(self.cfg.max_per_prompt))
        max_per_response = max(1, int(self.cfg.max_per_response))
        for item in sorted(
            ranked,
            key=lambda x: (
                float(x.get("selection_score", x.get("score", 0.0)) or 0.0),
                float(x.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        ):
            p = str(item.get("prompt", ""))
            r = str(item.get("response", ""))
            if prompt_counts[p] >= max_per_prompt:
                continue
            if response_counts[r] >= max_per_response:
                continue
            prompt_counts[p] += 1
            response_counts[r] += 1
            out.append(item)
        return out

    def _apply_hard_weighting(self, samples: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
        if max_samples <= 0 or len(samples) <= max_samples:
            return samples
        hard = [s for s in samples if bool(s.get("hard_sample", False))]
        normal = [s for s in samples if not bool(s.get("hard_sample", False))]
        target = int(max_samples)
        if target <= 0:
            return []
        min_hard = max(0, int(round(target * float(self.cfg.hard_min_ratio))))
        max_hard = max(min_hard, int(round(target * float(self.cfg.hard_max_ratio))))
        hard_n = min(len(hard), max_hard)
        if hard_n < min_hard:
            hard_n = min(len(hard), min_hard)
        normal_n = max(0, target - hard_n)
        rng = random.Random(int(self.cfg.seed))

        def _weighted_pick(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
            if n <= 0 or not rows:
                return []
            if len(rows) <= n:
                picked = list(rows)
                rng.shuffle(picked)
                return picked
            scored = sorted(rows, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            head = scored[: max(n, int(n * 2))]
            rng.shuffle(head)
            return head[:n]

        pick_h = _weighted_pick(hard, hard_n)
        pick_n = _weighted_pick(normal, normal_n)
        out = pick_h + pick_n
        if len(out) < target:
            remain = [s for s in samples if s not in out]
            need = target - len(out)
            out.extend(_weighted_pick(remain, need))
        rng.shuffle(out)
        return out[:target]

    @staticmethod
    def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _build_manifest(train_rows: List[Dict[str, Any]], eval_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_rows = train_rows + eval_rows
        src_counter = Counter()
        bucket_counter = Counter()
        hard_count = 0
        score_vals: List[float] = []
        diversity_penalties: List[float] = []
        near_duplicate_samples = 0
        rebucketed = 0
        for row in all_rows:
            src_counter[str(row.get("source", ""))] += 1
            bucket_counter[str(row.get("bucket", ""))] += 1
            if bool(row.get("hard_sample", False)):
                hard_count += 1
            score_vals.append(float(row.get("score", 0.0)))
            penalty = float(row.get("diversity_penalty", 0.0) or 0.0)
            diversity_penalties.append(float(penalty))
            if penalty > 1e-9:
                near_duplicate_samples += 1
            if str(row.get("bucket_rule", "")) and str(row.get("bucket_rule")) != str(row.get("bucket", "")):
                rebucketed += 1
        avg_score = float(sum(score_vals) / max(1, len(score_vals))) if score_vals else 0.0
        avg_diversity_penalty = (
            float(sum(diversity_penalties) / max(1, len(diversity_penalties))) if diversity_penalties else 0.0
        )
        train_prompt = {_norm_text(str(row.get("prompt", ""))) for row in train_rows if str(row.get("prompt", "")).strip()}
        eval_prompt = {_norm_text(str(row.get("prompt", ""))) for row in eval_rows if str(row.get("prompt", "")).strip()}
        train_pair = {
            (_norm_text(str(row.get("prompt", ""))), _norm_text(str(row.get("response", ""))))
            for row in train_rows
            if str(row.get("prompt", "")).strip() and str(row.get("response", "")).strip()
        }
        eval_pair = {
            (_norm_text(str(row.get("prompt", ""))), _norm_text(str(row.get("response", ""))))
            for row in eval_rows
            if str(row.get("prompt", "")).strip() and str(row.get("response", "")).strip()
        }
        overlap_prompt = train_prompt & eval_prompt
        overlap_pair = train_pair & eval_pair
        prompt_overlap_ratio = (
            float(len(overlap_prompt)) / float(max(1, min(len(train_prompt), len(eval_prompt))))
            if train_prompt and eval_prompt
            else 0.0
        )
        pair_overlap_ratio = (
            float(len(overlap_pair)) / float(max(1, min(len(train_pair), len(eval_pair))))
            if train_pair and eval_pair
            else 0.0
        )
        return {
            "version": 3,
            "ts": time.time(),
            "train_samples": len(train_rows),
            "eval_samples": len(eval_rows),
            "total_samples": len(all_rows),
            "avg_score": round(avg_score, 6),
            "hard_samples": int(hard_count),
            "hard_ratio": round(float(hard_count / max(1, len(all_rows))), 6),
            "rebucketed_samples": int(rebucketed),
            "near_duplicate_samples": int(near_duplicate_samples),
            "avg_diversity_penalty": round(float(avg_diversity_penalty), 6),
            "cross_split_prompt_overlap_count": int(len(overlap_prompt)),
            "cross_split_prompt_overlap_ratio": round(float(prompt_overlap_ratio), 6),
            "cross_split_pair_overlap_count": int(len(overlap_pair)),
            "cross_split_pair_overlap_ratio": round(float(pair_overlap_ratio), 6),
            "by_source": dict(src_counter),
            "by_bucket": dict(bucket_counter),
        }


def build_dataset(config: Optional[DialogueDatasetBuildConfig] = None) -> Dict[str, Any]:
    return DialogueDatasetBuilder(config=config).build()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build adapter training dataset from dialogue logs")
    parser.add_argument("--patterns", type=str, default="artifacts/memory/dialogue_patterns.json")
    parser.add_argument("--chat-memory", type=str, default="artifacts/memory/chat_memory.jsonl")
    parser.add_argument("--reflections", type=str, default="artifacts/memory/dialogue_reflections.jsonl")
    parser.add_argument("--train-out", type=str, default="artifacts/memory/adapter_train.jsonl")
    parser.add_argument("--eval-out", type=str, default="artifacts/memory/adapter_eval.jsonl")
    parser.add_argument("--manifest-out", type=str, default="artifacts/memory/adapter_dataset_manifest.json")
    parser.add_argument("--min-quality", type=float, default=0.45)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hard-min-ratio", type=float, default=0.30)
    parser.add_argument("--hard-max-ratio", type=float, default=0.65)
    parser.add_argument("--max-per-prompt", type=int, default=6)
    parser.add_argument("--max-per-response", type=int, default=4)
    parser.add_argument("--keep-templatey", action="store_true")
    parser.add_argument("--allow-cross-split-prompt", action="store_true")
    parser.add_argument("--disable-diversity-penalty", action="store_true")
    parser.add_argument("--diversity-similarity-threshold", type=float, default=0.82)
    parser.add_argument("--diversity-penalty-strength", type=float, default=0.20)
    parser.add_argument("--diversity-reference-cap", type=int, default=12)
    parser.add_argument("--no-bucket-classifier", action="store_true")
    parser.add_argument("--bucket-classifier-min-conf", type=float, default=0.62)
    parser.add_argument("--bucket-classifier-min-docs", type=int, default=12)
    parser.add_argument("--bucket-classifier-alpha", type=float, default=1.0)
    parser.add_argument("--bucket-classifier-force-override", action="store_true")
    parser.add_argument("--no-patterns", action="store_true")
    parser.add_argument("--no-chat-memory", action="store_true")
    parser.add_argument("--no-reflections", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DialogueDatasetBuildConfig(
        patterns_path=args.patterns,
        chat_memory_path=args.chat_memory,
        reflections_path=args.reflections,
        output_train_path=args.train_out,
        output_eval_path=args.eval_out,
        output_manifest_path=args.manifest_out,
        min_quality=float(args.min_quality),
        max_samples=int(args.max_samples),
        eval_ratio=float(args.eval_ratio),
        seed=int(args.seed),
        include_patterns=not bool(args.no_patterns),
        include_chat_memory=not bool(args.no_chat_memory),
        include_reflections=not bool(args.no_reflections),
        hard_min_ratio=float(args.hard_min_ratio),
        hard_max_ratio=float(args.hard_max_ratio),
        max_per_prompt=int(args.max_per_prompt),
        max_per_response=int(args.max_per_response),
        strip_templatey_responses=not bool(args.keep_templatey),
        prevent_cross_split_prompt_leakage=not bool(args.allow_cross_split_prompt),
        diversity_penalty_enabled=not bool(args.disable_diversity_penalty),
        diversity_similarity_threshold=float(args.diversity_similarity_threshold),
        diversity_penalty_strength=float(args.diversity_penalty_strength),
        diversity_reference_cap=int(args.diversity_reference_cap),
        bucket_classifier_enabled=not bool(args.no_bucket_classifier),
        bucket_classifier_min_conf=float(args.bucket_classifier_min_conf),
        bucket_classifier_min_docs=int(args.bucket_classifier_min_docs),
        bucket_classifier_alpha=float(args.bucket_classifier_alpha),
        bucket_classifier_force_override=bool(args.bucket_classifier_force_override),
    )
    manifest = build_dataset(cfg)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
