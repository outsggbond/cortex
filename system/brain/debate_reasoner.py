from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from system.brain.neurosymbolic import NeuroSymbolicEngine, TraceStep, build_default_engine


@dataclass
class DebateConfig:
    rounds: int = 2
    consistency_weight: float = 0.7
    relevance_weight: float = 0.3


def _text_relevance(a: str, b: str) -> float:
    sa = {x for x in str(a).lower().split() if x}
    sb = {x for x in str(b).lower().split() if x}
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


class DebateReasoner:
    """Two-agent debate scaffold with neurosymbolic consistency scoring."""

    def __init__(
        self,
        *,
        proposer_fn: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        challenger_fn: Optional[Callable[[str, Dict[str, Any]], str]] = None,
        engine: Optional[NeuroSymbolicEngine] = None,
        config: Optional[DebateConfig] = None,
    ):
        self.proposer_fn = proposer_fn
        self.challenger_fn = challenger_fn
        self.engine = engine if engine is not None else build_default_engine()
        self.config = config if config is not None else DebateConfig()

    def _fallback_proposer(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate a proposal using the neurosymbolic engine as a knowledge source."""
        _ = context
        infer = self.engine.infer({"message": str(prompt)})
        trace = infer.get("trace", [])
        if isinstance(trace, list) and trace:
            steps = []
            for t in trace[:5]:
                if hasattr(t, "conclusion") and hasattr(t, "evidence"):
                    conclusion = str(getattr(t, "conclusion", "") or "").strip()
                    evidence = str(getattr(t, "evidence", "") or "").strip()
                    if conclusion:
                        line = f"- {conclusion}"
                        if evidence:
                            line += f"  (evidence: {evidence})"
                        steps.append(line)
            if steps:
                return (
                    f"Claim regarding '{prompt}':\n"
                    + "\n".join(steps)
                    + f"\n\nConclusion: Based on the rules above, {prompt} should be analyzed stepwise."
                )
        # Minimal fallback
        return f"Claim regarding '{prompt}': Analyze step by step, grounded in available evidence."

    def _fallback_challenger(self, prompt: str, context: Dict[str, Any]) -> str:
        """Challenge the proposal by checking for contradictions and weak evidence."""
        _ = context
        infer = self.engine.infer({"message": f"critique: {prompt}"})
        trace = infer.get("trace", [])
        if isinstance(trace, list) and trace:
            critiques = []
            for t in trace[:5]:
                if hasattr(t, "conclusion"):
                    conclusion = str(getattr(t, "conclusion", "") or "").strip()
                    if conclusion:
                        critiques.append(f"- Potential issue: {conclusion}")
            if critiques:
                return (
                    f"Challenger review of '{prompt}':\n"
                    + "\n".join(critiques)
                    + "\n\nVerdict: Re-verify assumptions before accepting the claim."
                )
        return f"Challenger review of '{prompt}': Verify assumptions and reject inconsistent claims."

    def _score_argument(self, prompt: str, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        infer = self.engine.infer({"message": str(text), **dict(context or {})})
        trace = infer.get("trace", [])
        if not isinstance(trace, list):
            trace = []
        trace_steps: List[TraceStep] = [x for x in trace if isinstance(x, TraceStep)]
        consistency = 1.0 if trace_steps else 0.3
        relevance = _text_relevance(prompt, text)
        w_cons = max(0.0, float(self.config.consistency_weight))
        w_rel = max(0.0, float(self.config.relevance_weight))
        score = float(w_cons * consistency + w_rel * relevance)
        return {
            "score": float(score),
            "consistency": float(consistency),
            "relevance": float(relevance),
            "trace": [
                {
                    "rule": str(t.rule),
                    "evidence": str(t.evidence),
                    "conclusion": str(t.conclusion),
                }
                for t in trace_steps
            ],
        }

    def debate(self, prompt: str, *, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(context or {})
        rounds = max(1, int(self.config.rounds))
        proposer = self.proposer_fn if callable(self.proposer_fn) else self._fallback_proposer
        challenger = self.challenger_fn if callable(self.challenger_fn) else self._fallback_challenger
        history: List[Dict[str, Any]] = []

        latest_claim = str(prompt)
        for r in range(rounds):
            p_text = str(proposer(latest_claim, ctx) or "").strip()
            c_text = str(challenger(latest_claim, ctx) or "").strip()
            p_score = self._score_argument(prompt, p_text, ctx)
            c_score = self._score_argument(prompt, c_text, ctx)
            winner = "proposer" if float(p_score.get("score", 0.0)) >= float(c_score.get("score", 0.0)) else "challenger"
            turn = {
                "round": int(r + 1),
                "proposer": {"text": p_text, **p_score},
                "challenger": {"text": c_text, **c_score},
                "winner": str(winner),
            }
            history.append(turn)
            latest_claim = p_text if winner == "proposer" else c_text

        final = history[-1] if history else {}
        final_winner = str(final.get("winner", "proposer") or "proposer")
        final_text = str(((final.get(final_winner, {}) or {}).get("text", "")) if isinstance(final, dict) else "")
        return {
            "ok": True,
            "prompt": str(prompt),
            "rounds": history,
            "winner": str(final_winner),
            "answer": str(final_text),
        }

