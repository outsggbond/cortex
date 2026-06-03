from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Tuple


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.findall(r"[a-z0-9_]+|[一-鿿]+", text.lower()) if t]


def _normalize_text(text: str) -> str:
    return " ".join(_tokenize(_to_text(text)))


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = {x for x in a if x}
    sb = {x for x in b if x}
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


_NEGATION_PATTERNS = (
    r"\bnot\b",
    r"\bnever\b",
    r"\bcannot\b",
    r"\bcan\s+not\b",
    r"\bcan't\b",
    r"\bmust\s+not\b",
    r"\bshould\s+not\b",
    r"\bno\b",
    r"\bwithout\b",
    r"不能",
    r"不可",
    r"不得",
    r"不应",
    r"不应该",
    r"不要",
    r"无法",
    r"没法",
)


def _is_negated(text: str) -> bool:
    raw = _to_text(text)
    if not raw:
        return False
    low = raw.lower()
    for pat in _NEGATION_PATTERNS:
        target = low if "\\b" in pat else raw
        if re.search(pat, target):
            return True
    return False


def _strip_negation(text: str) -> str:
    out = _to_text(text)
    if not out:
        return out
    low = out.lower()
    for pat in _NEGATION_PATTERNS:
        if "\\b" in pat:
            low = re.sub(pat, " ", low)
    out = low
    for pat in _NEGATION_PATTERNS:
        if "\\b" not in pat:
            out = re.sub(pat, " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _clean_modal_phrase(text: str) -> str:
    out = _to_text(text)
    if not out:
        return ""
    out = re.sub(r"^[\s,.;:!?，。；：！？、]+", "", out)
    out = re.sub(r"[\s,.;:!?，。；：！？、]+$", "", out)
    return _normalize_text(out)


def _extract_modals(text: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    raw = _to_text(text)
    if not raw:
        return out
    low = raw.lower()

    en_patterns = (
        (1, r"\bmust\s+([a-z0-9_ \-]{2,100})"),
        (-1, r"\bcannot\s+([a-z0-9_ \-]{2,100})"),
        (-1, r"\bcan\s+not\s+([a-z0-9_ \-]{2,100})"),
        (-1, r"\bcan't\s+([a-z0-9_ \-]{2,100})"),
        (-1, r"\bmust\s+not\s+([a-z0-9_ \-]{2,100})"),
        (1, r"\bshould\s+([a-z0-9_ \-]{2,100})"),
        (-1, r"\bshould\s+not\s+([a-z0-9_ \-]{2,100})"),
    )
    zh_patterns = (
        (1, r"必须([^，。；：！？,.!?]{1,40})"),
        (-1, r"不能([^，。；：！？,.!?]{1,40})"),
        (-1, r"不得([^，。；：！？,.!?]{1,40})"),
        (1, r"应该([^，。；：！？,.!?]{1,40})"),
        (-1, r"不应该([^，。；：！？,.!?]{1,40})"),
    )

    for polarity, pat in en_patterns:
        for m in re.finditer(pat, low):
            phrase = _clean_modal_phrase(m.group(1))
            if phrase:
                out.append((int(polarity), phrase))
    for polarity, pat in zh_patterns:
        for m in re.finditer(pat, raw):
            phrase = _clean_modal_phrase(m.group(1))
            if phrase:
                out.append((int(polarity), phrase))
    return out


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _parse_evidence_kv(evidence: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    raw = _to_text(evidence)
    if not raw:
        return out
    for part in raw.replace(";", ",").split(","):
        node = part.strip()
        if "=" not in node:
            continue
        k, v = node.split("=", 1)
        key = _to_text(k).lower()
        if not key:
            continue
        out[key] = _to_text(v)
    return out


class RuleFeedbackMemory:
    def __init__(self, path: str = "artifacts/audit/reasoning_rule_feedback.json") -> None:
        self.path = Path(str(path or "artifacts/audit/reasoning_rule_feedback.json"))
        self.rules: Dict[str, Dict[str, float]] = {}
        self.updated_at: float = 0.0

    @staticmethod
    def _sanitize_counts(wins: Any, total: Any) -> Tuple[float, float]:
        try:
            total_value = max(0.0, float(total))
        except Exception:
            total_value = 0.0
        try:
            wins_value = max(0.0, float(wins))
        except Exception:
            wins_value = 0.0
        if wins_value > total_value:
            wins_value = total_value
        return float(wins_value), float(total_value)

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        node = raw.get("rules")
        if not isinstance(node, dict):
            return
        loaded: Dict[str, Dict[str, float]] = {}
        for rule, row in node.items():
            key = _to_text(rule)
            if not key or not isinstance(row, dict):
                continue
            wins, total = self._sanitize_counts(row.get("wins", 0.0), row.get("total", 0.0))
            loaded[key] = {"wins": float(wins), "total": float(total)}
        self.rules = loaded
        try:
            self.updated_at = float(raw.get("updated_at", 0.0) or 0.0)
        except Exception:
            self.updated_at = 0.0

    def prior(self, rule: str) -> float:
        key = _to_text(rule)
        if not key:
            return 0.5
        row = self.rules.get(key)
        if not row:
            return 0.5
        wins, total = self._sanitize_counts(row.get("wins", 0.0), row.get("total", 0.0))
        return _clamp((float(wins) + 1.0) / (float(total) + 2.0), 0.0, 1.0)

    def total(self, rule: str) -> float:
        key = _to_text(rule)
        if not key:
            return 0.0
        row = self.rules.get(key, {})
        try:
            return max(0.0, float(row.get("total", 0.0) or 0.0))
        except Exception:
            return 0.0

    def update(self, rule: str, success: bool, *, weight: float = 1.0) -> Dict[str, float]:
        key = _to_text(rule)
        if not key:
            return {}
        w = max(0.05, float(weight))
        prev = self.rules.get(key, {})
        wins, total = self._sanitize_counts(prev.get("wins", 0.0), prev.get("total", 0.0))
        total += float(w)
        if bool(success):
            wins += float(w)
        wins = min(wins, total)
        self.rules[key] = {"wins": float(wins), "total": float(total)}
        self.updated_at = float(time.time())
        return {
            "rule": key,
            "wins": float(wins),
            "total": float(total),
            "score": float(self.prior(key)),
        }

    def top_rules(self, *, limit: int = 5, min_total: float = 1.0) -> List[Dict[str, Any]]:
        rows: List[Tuple[float, float, str]] = []
        for rule, node in self.rules.items():
            total = 0.0
            try:
                total = max(0.0, float(node.get("total", 0.0) or 0.0))
            except Exception:
                total = 0.0
            if total < float(min_total):
                continue
            rows.append((float(self.prior(rule)), float(total), rule))
        rows.sort(key=lambda x: (float(x[0]), float(x[1])), reverse=True)
        out: List[Dict[str, Any]] = []
        for score, total, rule in rows[: max(0, int(limit))]:
            out.append({"rule": str(rule), "score": float(score), "total": float(total)})
        return out

    def save(self) -> bool:
        payload = {
            "version": 1,
            "updated_at": float(time.time()),
            "rules": {
                str(rule): {
                    "wins": float(max(0.0, float(node.get("wins", 0.0) or 0.0))),
                    "total": float(max(0.0, float(node.get("total", 0.0) or 0.0))),
                    "score": float(self.prior(rule)),
                }
                for rule, node in self.rules.items()
                if _to_text(rule)
            },
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.updated_at = float(payload["updated_at"])
            return True
        except Exception:
            return False
