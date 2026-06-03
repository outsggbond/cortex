from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from system.brain._reasoning_feedback import (
    _clamp,
    _extract_modals,
    _is_negated,
    _jaccard,
    _normalize_text,
    _strip_negation,
    _to_text,
    _tokenize,
)
from system.brain._reasoning_profile import ReasoningStep


class ForwardChainer:
    def run(
        self,
        facts: Dict[str, str],
        rules: List[Callable[[Dict[str, str]], Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for rule in rules:
            try:
                res = rule(facts)
                if not res:
                    continue
                if isinstance(res, str):
                    results.append({"conclusion": res, "score": 0.5, "rule": "string_rule", "evidence": "raw_text"})
                    continue
                if isinstance(res, dict):
                    results.append(
                        {
                            "rule": _to_text(res.get("rule")) or "low_rule",
                            "score": float(res.get("score", 0.5) or 0.5),
                            "evidence": _to_text(res.get("evidence")) or "rule_fired",
                            "conclusion": _to_text(res.get("conclusion")),
                        }
                    )
            except Exception:
                continue
        return results


class BackwardChainer:
    _STOPWORDS = {
        "the",
        "a",
        "an",
        "to",
        "for",
        "and",
        "or",
        "of",
        "in",
        "on",
        "with",
        "by",
        "is",
        "are",
    }

    def alignment_score(self, goal: str, candidate: str, facts: Dict[str, str]) -> float:
        g = _normalize_text(goal)
        c = _normalize_text(candidate)
        if not g or not c:
            return 0.0
        gt = [x for x in g.split() if x not in self._STOPWORDS]
        ct = [x for x in c.split() if x not in self._STOPWORDS]
        base = _jaccard(gt, ct)
        if base > 0.0:
            return float(base)
        for value in facts.values():
            vt = _tokenize(_to_text(value))
            if _jaccard(gt, vt) >= 0.35 and _jaccard(ct, vt) >= 0.20:
                return 0.35
        return 0.0

    def verify(self, goal: str, facts: Dict[str, str], candidate: str = "") -> bool:
        if not _to_text(goal):
            return bool(_to_text(candidate))
        return bool(self.alignment_score(goal, candidate, facts) >= 0.20)


class ConsistencyChecker:
    def contradiction(self, a: str, b: str) -> bool:
        ta = _normalize_text(a)
        tb = _normalize_text(b)
        if not ta or not tb:
            return False
        pa = _is_negated(a)
        pb = _is_negated(b)
        if pa == pb:
            return False
        sa = _normalize_text(_strip_negation(a))
        sb = _normalize_text(_strip_negation(b))
        if sa and sb and sa == sb:
            return True
        if sa and sb and _jaccard(sa.split(), sb.split()) >= 0.65:
            return True
        return False

    def analyze(self, steps: List[ReasoningStep], message: str = "") -> Dict[str, Any]:
        support = 0
        contradictions = 0
        conflicts: List[Tuple[int, int]] = []
        for i in range(len(steps)):
            for j in range(i + 1, len(steps)):
                a = steps[i].conclusion
                b = steps[j].conclusion
                if not a or not b:
                    continue
                if self.contradiction(a, b):
                    contradictions += 1
                    conflicts.append((i, j))
                    continue
                sim = _jaccard(_tokenize(a), _tokenize(b))
                if sim >= 0.45:
                    support += 1

        modal_pairs = _extract_modals(message)
        modal_conflicts = 0
        modal_seen: Dict[str, int] = {}
        for polarity, phrase in modal_pairs:
            prev = modal_seen.get(phrase)
            if prev is None:
                modal_seen[phrase] = polarity
                continue
            if int(prev) != int(polarity):
                modal_conflicts += 1
        # Heuristic low-level conflict rules should also influence final caution mode.
        modal_conflicts += sum(1 for s in steps if _to_text(getattr(s, "rule", "")) == "constraint_conflict")

        total_edges = max(1, support + contradictions)
        consistency_score = _clamp((float(support) - float(contradictions)) / float(total_edges) * 0.5 + 0.5, 0.0, 1.0)
        if modal_conflicts > 0:
            consistency_score = _clamp(consistency_score - min(0.5, 0.2 * float(modal_conflicts)), 0.0, 1.0)
        return {
            "support_edges": int(support),
            "contradiction_edges": int(contradictions),
            "modal_conflicts": int(modal_conflicts),
            "consistency_score": float(consistency_score),
            "conflicts": conflicts,
        }
