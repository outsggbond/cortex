# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os
import re
from pathlib import Path

from system.core.embeddings import encode_text, cosine, embedding_available


@dataclass
class FAQPair:
    question: str
    answer: str
    source: str = ""
    score: float = 0.0


def _read_text(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _parse_pairs_from_yaml(text: str, source: str) -> List[FAQPair]:
    try:
        import yaml
    except Exception:
        return []
    data = yaml.safe_load(text) or {}
    pairs: List[FAQPair] = []

    def add_pair(q, a):
        q = (q or "").strip()
        a = (a or "").strip()
        if q and a:
            pairs.append(FAQPair(q, a, source))

    if isinstance(data, dict):
        convs = data.get("conversations", [])
        for conv in convs:
            if isinstance(conv, list):
                for i in range(0, len(conv) - 1, 2):
                    add_pair(conv[i], conv[i + 1])
            elif isinstance(conv, dict):
                q = conv.get("q") or conv.get("question") or ""
                a = conv.get("a") or conv.get("answer") or ""
                add_pair(q, a)
    elif isinstance(data, list):
        # If list of strings, pair sequentially; if list of dicts, use q/a.
        if data and all(isinstance(x, str) for x in data):
            for i in range(0, len(data) - 1, 2):
                add_pair(data[i], data[i + 1])
        else:
            for item in data:
                if isinstance(item, dict):
                    q = item.get("q") or item.get("question") or ""
                    a = item.get("a") or item.get("answer") or ""
                    add_pair(q, a)
    return pairs


def _parse_pairs_from_txt(text: str, source: str) -> List[FAQPair]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pairs: List[FAQPair] = []
    q = None
    q_markers = ("q:", "问:", "问题:")
    a_markers = ("a:", "答:", "回答:")
    has_markers = any(ln.lower().startswith(q_markers + a_markers) for ln in lines)

    def add_pair(qv, av):
        qv = (qv or "").strip()
        av = (av or "").strip()
        if qv and av:
            pairs.append(FAQPair(qv, av, source))

    if has_markers:
        for ln in lines:
            lower = ln.lower()
            if lower.startswith(q_markers):
                q = ln.split(":", 1)[1].strip() if ":" in ln else ln[2:].strip()
                continue
            if lower.startswith(a_markers):
                a = ln.split(":", 1)[1].strip() if ":" in ln else ln[2:].strip()
                if q:
                    add_pair(q, a)
                    q = None
                continue
        return pairs

    # Fallback: pair sequentially
    for i in range(0, len(lines) - 1, 2):
        add_pair(lines[i], lines[i + 1])
    return pairs


def _load_faq_file(path: str) -> List[FAQPair]:
    text = _read_text(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yml", ".yaml"):
        return _parse_pairs_from_yaml(text, path)
    return _parse_pairs_from_txt(text, path)


def load_faq_pairs(path: str) -> List[FAQPair]:
    if not path:
        return []
    if os.path.isdir(path):
        pairs: List[FAQPair] = []
        for p in sorted(Path(path).rglob("*")):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in (".yml", ".yaml", ".txt"):
                pairs.extend(_load_faq_file(str(p)))
        return pairs
    if not os.path.isfile(path):
        return []
    return _load_faq_file(path)


def guess_faq_path(explicit: str = "") -> str:
    if explicit and (os.path.isfile(explicit) or os.path.isdir(explicit)):
        return explicit
    if os.path.isfile("train.txt") and os.path.getsize("train.txt") > 0:
        return "train.txt"
    if os.path.isdir("chinese"):
        return "chinese"
    if os.path.isfile("chinese/ai.yml"):
        return "chinese/ai.yml"
    return ""


class FAQRetriever:
    def __init__(self):
        self.pairs: List[FAQPair] = []
        self.q_vecs: List = []
        self.q_pinyin: List[str] = []
        self._pinyin_available = False
        self._lazy_pinyin = None
        self._embeddings_available = False

    def _to_pinyin(self, text: str) -> str:
        if not self._pinyin_available or not text:
            return ""
        if re.search(r"[A-Za-z]", text) and not re.search(r"[\u4e00-\u9fff]", text):
            return re.sub(r"[^a-z0-9]+", "", text.lower())
        return "".join(self._lazy_pinyin(text))

    def load(self, path: str) -> int:
        self.pairs = load_faq_pairs(path)
        self.q_vecs = []
        self._embeddings_available = False
        if self.pairs and embedding_available():
            try:
                self.q_vecs = [encode_text(p.question, require_model=True) for p in self.pairs]
                self._embeddings_available = True
            except Exception:
                self.q_vecs = []
                self._embeddings_available = False
        self.q_pinyin = []
        try:
            from pypinyin import lazy_pinyin

            self._lazy_pinyin = lazy_pinyin
            self._pinyin_available = True
            self.q_pinyin = [self._to_pinyin(p.question) for p in self.pairs]
        except Exception:
            self._pinyin_available = False
            self.q_pinyin = []
        return len(self.pairs)

    def _normalize(self, text: str) -> str:
        t = text.strip().lower()
        t = re.sub(r"[\W_]+", "", t, flags=re.UNICODE)
        return t

    def _char_ngrams(self, text: str, n: int = 2) -> set:
        t = re.sub(r"\s+", "", text)
        if len(t) <= n:
            return {t} if t else set()
        return {t[i : i + n] for i in range(len(t) - n + 1)}

    def _lex_sim(self, a: str, b: str) -> float:
        a_set = self._char_ngrams(a, n=2)
        b_set = self._char_ngrams(b, n=2)
        if not a_set or not b_set:
            return 0.0
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        return inter / union if union else 0.0

    def _pinyin_sim(self, a: str, b: str) -> float:
        if not self._pinyin_available:
            return 0.0
        a = re.sub(r"[^a-z0-9]+", "", a.lower())
        b = re.sub(r"[^a-z0-9]+", "", b.lower())
        if not a or not b:
            return 0.0
        a_set = self._char_ngrams(a, n=2)
        b_set = self._char_ngrams(b, n=2)
        if not a_set or not b_set:
            return 0.0
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        return inter / union if union else 0.0

    def search(self, query: str, min_score: Optional[float] = None, min_lex: Optional[float] = None) -> Optional[FAQPair]:
        if not self.pairs:
            return None
        if min_score is None:
            try:
                min_score = float(os.environ.get("FAQ_MIN_SCORE", "0.78"))
            except Exception:
                min_score = 0.78
        if min_lex is None:
            try:
                min_lex = float(os.environ.get("FAQ_MIN_LEX", "0.22"))
            except Exception:
                min_lex = 0.22
        qn = self._normalize(query)
        # exact match shortcut
        for p in self.pairs:
            if self._normalize(p.question) == qn and qn:
                return FAQPair(p.question, p.answer, p.source, 1.0)
        qv = None
        if self._embeddings_available:
            try:
                qv = encode_text(query, require_model=True)
            except Exception:
                qv = None
        qpy = self._to_pinyin(query)
        best_idx = -1
        best_score = -1.0
        best_lex = 0.0
        best_py = 0.0
        best_emb = 0.0
        total = len(self.pairs)
        for i in range(total):
            v = self.q_vecs[i] if (self._embeddings_available and i < len(self.q_vecs)) else None
            if qv is None:
                sc = 0.0
            else:
                sc = cosine(qv, v) if v is not None else 0.0
            lex = self._lex_sim(query, self.pairs[i].question)
            py = self._pinyin_sim(qpy, self.q_pinyin[i]) if qpy and i < len(self.q_pinyin) else 0.0
            if self._embeddings_available:
                score = (0.8 * sc) + (0.15 * lex) + (0.05 * py)
            else:
                score = max(lex, py)
            if score > best_score:
                best_score = score
                best_idx = i
                best_lex = lex
                best_py = py
                best_emb = sc
        if best_idx < 0:
            return None
        emb_ok = best_emb >= min_score
        lex_ok = best_lex >= min_lex
        py_ok = best_py >= min_lex
        if self._embeddings_available:
            if not (emb_ok or lex_ok or py_ok):
                return None
        else:
            if not (lex_ok or py_ok):
                return None
        p = self.pairs[best_idx]
        return FAQPair(p.question, p.answer, p.source, best_score)
