# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Callable, Dict, List


logger = logging.getLogger(__name__)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower()) if t]


def _normalize(text: str) -> str:
    return " ".join(_tokenize(_to_text(text)))


_NEGATION_PATTERNS = (
    r"\bnot\b",
    r"\bnever\b",
    r"\bcannot\b",
    r"\bcan\s+not\b",
    r"\bcan't\b",
    r"\bmust\s+not\b",
    r"\bshould\s+not\b",
    r"\bwithout\b",
    r"不能",
    r"不可",
    r"不得",
    r"不应",
    r"不应该",
    r"无法",
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
    raw = _to_text(text)
    if not raw:
        return ""
    low = raw.lower()
    for pat in _NEGATION_PATTERNS:
        if "\\b" in pat:
            low = re.sub(pat, " ", low)
    out = low
    for pat in _NEGATION_PATTERNS:
        if "\\b" not in pat:
            out = re.sub(pat, " ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _contradict(a: str, b: str) -> bool:
    if not a or not b:
        return False
    pa = _is_negated(a)
    pb = _is_negated(b)
    if pa == pb:
        return False
    na = _normalize(_strip_negation(a))
    nb = _normalize(_strip_negation(b))
    if not na or not nb:
        return False
    if na == nb:
        return True
    return _jaccard(na.split(), nb.split()) >= 0.65


def _ctx_message(ctx: Dict[str, Any]) -> str:
    return _to_text((ctx or {}).get("message"))


def _clean_phrase(text: str) -> str:
    out = _to_text(text)
    out = re.sub(r"^[\s,.;:!?，。；：！？、]+", "", out)
    out = re.sub(r"[\s,.;:!?，。；：！？、]+$", "", out)
    return out


@dataclass
class Rule:
    name: str
    when: Callable[[dict], bool]
    then: Callable[[dict], dict]
    weight: float = 1.0
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class TraceStep:
    rule: str
    evidence: str
    conclusion: str


class NeuroSymbolicEngine:
    def __init__(self):
        self._rules: List[Rule] = []
        self._dynamic_rules: List[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)

    def add_dynamic_rule(self, rule: Rule) -> None:
        self._dynamic_rules.append(rule)

    def make_contains_rule(self, name: str, keyword: str, conclusion: str, weight: float = 0.6, evidence: str = "") -> Rule:
        key = _to_text(keyword)
        key_low = key.lower()
        return Rule(
            name=name,
            when=lambda ctx, k=key, kl=key_low: (k and k in _ctx_message(ctx)) or (kl and kl in _ctx_message(ctx).lower()),
            then=lambda ctx, v=conclusion, e=evidence: {"conclusion": v, "evidence": e or "keyword_match"},
            weight=weight,
            meta={"type": "contains", "keyword": key, "conclusion": _to_text(conclusion)},
        )

    def make_static_rule(self, name: str, conclusion: str, weight: float = 0.5) -> Rule:
        return Rule(
            name=name,
            when=lambda ctx: True,
            then=lambda ctx, v=conclusion: {"conclusion": v, "evidence": "static"},
            weight=weight,
            meta={"type": "static", "conclusion": _to_text(conclusion)},
        )

    def make_and_rule(self, name: str, keywords: List[str], conclusion: str, weight: float = 0.6) -> Rule:
        keys = [_to_text(k) for k in (keywords or []) if _to_text(k)]
        keys_low = [k.lower() for k in keys]

        def _match(ctx: Dict[str, Any], ks: List[str], ksl: List[str]) -> bool:
            msg = _ctx_message(ctx)
            low = msg.lower()
            for k, kl in zip(ks, ksl):
                if (k not in msg) and (kl not in low):
                    return False
            return bool(ks)

        return Rule(
            name=name,
            when=lambda ctx, ks=keys, ksl=keys_low: _match(ctx, ks, ksl),
            then=lambda ctx, v=conclusion: {"conclusion": v, "evidence": "and_rule"},
            weight=weight,
            meta={"type": "and", "keyword": "|".join(keys), "conclusion": _to_text(conclusion)},
        )

    def export_dynamic_rules(self) -> List[dict]:
        return [{"name": r.name, "weight": r.weight, "meta": r.meta} for r in self._dynamic_rules]

    def _aggregate_conclusions(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        vote_map: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            text = _to_text(row.get("conclusion"))
            if not text:
                continue
            key = _normalize(_strip_negation(text))
            if not key:
                continue
            score = float(row.get("score", 0.5) or 0.5)
            node = vote_map.get(key)
            if node is None:
                vote_map[key] = {
                    "key": key,
                    "conclusion": text,
                    "vote_score": float(score),
                    "count": 1,
                    "negated": bool(_is_negated(text)),
                }
            else:
                prev_avg = float(node.get("vote_score", 0.0)) / max(1, int(node.get("count", 1)))
                node["vote_score"] = float(node.get("vote_score", 0.0)) + float(score)
                node["count"] = int(node.get("count", 0)) + 1
                if float(score) >= prev_avg:
                    node["conclusion"] = text
                    node["negated"] = bool(_is_negated(text))

        ranked = sorted(vote_map.values(), key=lambda x: float(x.get("vote_score", 0.0)), reverse=True)
        best = ranked[0] if ranked else {}
        total_vote = sum(float(x.get("vote_score", 0.0)) for x in ranked) or 1.0

        conflicts: List[Dict[str, Any]] = []
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                a = _to_text(ranked[i].get("conclusion"))
                b = _to_text(ranked[j].get("conclusion"))
                if _contradict(a, b):
                    conflicts.append(
                        {
                            "a": a,
                            "b": b,
                            "a_score": float(ranked[i].get("vote_score", 0.0)),
                            "b_score": float(ranked[j].get("vote_score", 0.0)),
                        }
                    )

        consensus = float(best.get("vote_score", 0.0)) / float(total_vote)
        if conflicts:
            consensus = max(0.0, consensus - min(0.4, 0.1 * float(len(conflicts))))
        return {
            "best_conclusion": _to_text(best.get("conclusion")),
            "best_score": float(best.get("vote_score", 0.0)),
            "consensus_score": float(consensus),
            "conflicts": conflicts,
            "ranked_conclusions": ranked[:8],
        }

    def infer(self, context: dict) -> Dict[str, object]:
        conclusions: List[dict] = []
        trace: List[TraceStep] = []
        for rule in self._rules + self._dynamic_rules:
            try:
                if rule.weight >= 0.2 and rule.when(context):
                    result = rule.then(context) or {}
                    conclusion_text = _to_text(result.get("conclusion"))
                    evidence_text = _to_text(result.get("evidence")) or "matched"
                    score = max(0.0, float(rule.weight))
                    if conclusion_text:
                        conclusions.append(
                            {
                                "rule": str(rule.name),
                                "conclusion": conclusion_text,
                                "evidence": evidence_text,
                                "score": float(score),
                            }
                        )
                    trace.append(
                        TraceStep(
                            rule=rule.name,
                            evidence=evidence_text,
                            conclusion=conclusion_text or "ok",
                        )
                    )
            except Exception:
                logger.debug("neurosymbolic: rule execution failed", exc_info=True)
        agg = self._aggregate_conclusions(conclusions)
        return {"conclusions": conclusions, "trace": trace, **agg}

    def _add_dynamic_contains(self, *, prefix: str, cond: str, cons: str, evidence: str) -> bool:
        key = _clean_phrase(cond)
        val = _clean_phrase(cons)
        if not key or not val:
            return False
        safe = re.sub(r"[^a-zA-Z0-9_\u4e00-\u9fff]", "_", key)[:12] or "rule"
        self.add_dynamic_rule(
            self.make_contains_rule(
                name=f"dyn_{prefix}_{safe}",
                keyword=key,
                conclusion=val,
                weight=0.6,
                evidence=evidence,
            )
        )
        return True

    def auto_expand_rules(self, message: str) -> int:
        created = 0
        text = _to_text(message)
        if not text:
            return 0

        seen: set[str] = set()

        def _dedup_key(a: str, b: str) -> str:
            return f"{_normalize(a)}=>{_normalize(b)}"

        for m in re.finditer(r"\bif\s+(.{1,120}?)\s+then\s+(.{1,180})", text, flags=re.IGNORECASE):
            cond, cons = _clean_phrase(m.group(1)), _clean_phrase(m.group(2))
            k = _dedup_key(cond, cons)
            if k in seen:
                continue
            if self._add_dynamic_contains(prefix="if", cond=cond, cons=cons, evidence="dynamic rule from 'if...then'"):
                seen.add(k)
                created += 1

        for m in re.finditer(r"\bwhen\s+(.{1,120}?)\s+then\s+(.{1,180})", text, flags=re.IGNORECASE):
            cond, cons = _clean_phrase(m.group(1)), _clean_phrase(m.group(2))
            k = _dedup_key(cond, cons)
            if k in seen:
                continue
            if self._add_dynamic_contains(prefix="when", cond=cond, cons=cons, evidence="dynamic rule from 'when...then'"):
                seen.add(k)
                created += 1

        for m in re.finditer(r"如果(.{1,80}?)那么(.{1,120})", text):
            cond, cons = _clean_phrase(m.group(1)), _clean_phrase(m.group(2))
            k = _dedup_key(cond, cons)
            if k in seen:
                continue
            if self._add_dynamic_contains(prefix="if", cond=cond, cons=cons, evidence="dynamic rule from '如果...那么'"):
                seen.add(k)
                created += 1

        for m in re.finditer(r"若(.{1,80}?)则(.{1,120})", text):
            cond, cons = _clean_phrase(m.group(1)), _clean_phrase(m.group(2))
            k = _dedup_key(cond, cons)
            if k in seen:
                continue
            if self._add_dynamic_contains(prefix="when", cond=cond, cons=cons, evidence="dynamic rule from '若...则'"):
                seen.add(k)
                created += 1

        return created

    def reinforce(self, trace: List[TraceStep], reward: float) -> None:
        name_to_rule = {r.name: r for r in (self._rules + self._dynamic_rules)}
        for t in trace:
            r = name_to_rule.get(t.rule)
            if not r:
                continue
            r.weight = max(0.0, min(1.5, r.weight + 0.1 * reward))


def build_default_engine() -> NeuroSymbolicEngine:
    engine = NeuroSymbolicEngine()

    engine.add_rule(
        Rule(
            name="goal_split",
            when=lambda ctx: ("goal" in _ctx_message(ctx).lower()) or ("目标" in _ctx_message(ctx)),
            then=lambda ctx: {
                "conclusion": "decompose goal into verifiable steps",
                "evidence": "contains goal/目标",
            },
        )
    )

    engine.add_rule(
        Rule(
            name="constraint_check",
            when=lambda ctx: (
                ("must" in _ctx_message(ctx).lower() and "cannot" in _ctx_message(ctx).lower())
                or ("must" in _ctx_message(ctx).lower() and "must not" in _ctx_message(ctx).lower())
                or ("必须" in _ctx_message(ctx) and ("不能" in _ctx_message(ctx) or "不得" in _ctx_message(ctx)))
            ),
            then=lambda ctx: {
                "conclusion": "constraint conflict detected, needs clarification",
                "evidence": "contains must/cannot or 必须/不能",
            },
        )
    )

    engine.add_rule(
        Rule(
            name="why_how",
            when=lambda ctx: (
                ("why" in _ctx_message(ctx).lower())
                or ("how" in _ctx_message(ctx).lower())
                or ("为什么" in _ctx_message(ctx))
                or ("如何" in _ctx_message(ctx))
            ),
            then=lambda ctx: {
                "conclusion": "explain rationale first, then give steps",
                "evidence": "question type",
            },
        )
    )

    return engine
