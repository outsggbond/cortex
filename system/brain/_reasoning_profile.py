from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict

from system.brain._reasoning_feedback import _clamp, _safe_float


@dataclass
class ReasoningStep:
    level: str
    rule: str
    score: float
    evidence: str
    conclusion: str


@dataclass
class ReasoningProfile:
    goal_align_weight: float = 0.30
    grounding_weight: float = 0.20
    rule_prior_scale: float = 1.00

    contradiction_penalty_scale: float = 0.08
    modal_penalty_scale: float = 0.10
    contradiction_penalty_cap: float = 0.35

    unsupported_penalty_scale: float = 0.25
    unsupported_penalty_cap: float = 0.18
    unsupported_unseen_ratio_trigger: float = 0.70
    unsupported_min_tokens: int = 4
    goal_miss_penalty: float = 0.08
    goal_miss_align_threshold: float = 0.10
    goal_restate_penalty: float = 0.25

    goal_restate_switch_gap: float = 0.06

    ambiguity_close_gap: float = 0.08
    ambiguity_collect_gap: float = 0.04
    ambiguity_collect_similarity: float = 0.35
    ambiguity_conflict_penalty: float = 0.08
    ambiguity_divergent_penalty: float = 0.05

    confidence_goal_base: float = 0.55
    confidence_no_candidate_base: float = 0.35
    confidence_consistency_weight: float = 0.25
    confidence_align_weight: float = 0.20
    proceed_confidence_threshold: float = 0.70
    goal_align_action_threshold: float = 0.20
    calibration_overconfidence_threshold: float = 0.88
    calibration_evidence_low_threshold: float = 0.20
    calibration_penalty: float = 0.08
    calibration_goal_detail_evidence_floor: float = 0.14
    calibration_consistency_floor: float = 0.35

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any] | None) -> "ReasoningProfile":
        src = raw or {}
        base = cls()
        return cls(
            goal_align_weight=_clamp(_safe_float(src.get("goal_align_weight", base.goal_align_weight), base.goal_align_weight), 0.0, 1.0),
            grounding_weight=_clamp(_safe_float(src.get("grounding_weight", base.grounding_weight), base.grounding_weight), 0.0, 1.0),
            rule_prior_scale=_clamp(_safe_float(src.get("rule_prior_scale", base.rule_prior_scale), base.rule_prior_scale), 0.0, 2.0),
            contradiction_penalty_scale=_clamp(
                _safe_float(src.get("contradiction_penalty_scale", base.contradiction_penalty_scale), base.contradiction_penalty_scale),
                0.0,
                1.0,
            ),
            modal_penalty_scale=_clamp(_safe_float(src.get("modal_penalty_scale", base.modal_penalty_scale), base.modal_penalty_scale), 0.0, 1.0),
            contradiction_penalty_cap=_clamp(
                _safe_float(src.get("contradiction_penalty_cap", base.contradiction_penalty_cap), base.contradiction_penalty_cap),
                0.0,
                1.0,
            ),
            unsupported_penalty_scale=_clamp(
                _safe_float(src.get("unsupported_penalty_scale", base.unsupported_penalty_scale), base.unsupported_penalty_scale),
                0.0,
                1.0,
            ),
            unsupported_penalty_cap=_clamp(
                _safe_float(src.get("unsupported_penalty_cap", base.unsupported_penalty_cap), base.unsupported_penalty_cap),
                0.0,
                1.0,
            ),
            unsupported_unseen_ratio_trigger=_clamp(
                _safe_float(src.get("unsupported_unseen_ratio_trigger", base.unsupported_unseen_ratio_trigger), base.unsupported_unseen_ratio_trigger),
                0.0,
                1.0,
            ),
            unsupported_min_tokens=max(1, int(_safe_float(src.get("unsupported_min_tokens", base.unsupported_min_tokens), base.unsupported_min_tokens))),
            goal_miss_penalty=_clamp(_safe_float(src.get("goal_miss_penalty", base.goal_miss_penalty), base.goal_miss_penalty), 0.0, 1.0),
            goal_miss_align_threshold=_clamp(
                _safe_float(src.get("goal_miss_align_threshold", base.goal_miss_align_threshold), base.goal_miss_align_threshold),
                0.0,
                1.0,
            ),
            goal_restate_penalty=_clamp(
                _safe_float(src.get("goal_restate_penalty", base.goal_restate_penalty), base.goal_restate_penalty),
                0.0,
                1.0,
            ),
            goal_restate_switch_gap=_clamp(
                _safe_float(src.get("goal_restate_switch_gap", base.goal_restate_switch_gap), base.goal_restate_switch_gap),
                0.0,
                0.5,
            ),
            ambiguity_close_gap=_clamp(
                _safe_float(src.get("ambiguity_close_gap", base.ambiguity_close_gap), base.ambiguity_close_gap),
                0.0,
                0.5,
            ),
            ambiguity_collect_gap=_clamp(
                _safe_float(src.get("ambiguity_collect_gap", base.ambiguity_collect_gap), base.ambiguity_collect_gap),
                0.0,
                0.5,
            ),
            ambiguity_collect_similarity=_clamp(
                _safe_float(src.get("ambiguity_collect_similarity", base.ambiguity_collect_similarity), base.ambiguity_collect_similarity),
                0.0,
                1.0,
            ),
            ambiguity_conflict_penalty=_clamp(
                _safe_float(src.get("ambiguity_conflict_penalty", base.ambiguity_conflict_penalty), base.ambiguity_conflict_penalty),
                0.0,
                1.0,
            ),
            ambiguity_divergent_penalty=_clamp(
                _safe_float(src.get("ambiguity_divergent_penalty", base.ambiguity_divergent_penalty), base.ambiguity_divergent_penalty),
                0.0,
                1.0,
            ),
            confidence_goal_base=_clamp(
                _safe_float(src.get("confidence_goal_base", base.confidence_goal_base), base.confidence_goal_base),
                0.0,
                1.0,
            ),
            confidence_no_candidate_base=_clamp(
                _safe_float(src.get("confidence_no_candidate_base", base.confidence_no_candidate_base), base.confidence_no_candidate_base),
                0.0,
                1.0,
            ),
            confidence_consistency_weight=_clamp(
                _safe_float(src.get("confidence_consistency_weight", base.confidence_consistency_weight), base.confidence_consistency_weight),
                0.0,
                1.0,
            ),
            confidence_align_weight=_clamp(
                _safe_float(src.get("confidence_align_weight", base.confidence_align_weight), base.confidence_align_weight),
                0.0,
                1.0,
            ),
            proceed_confidence_threshold=_clamp(
                _safe_float(src.get("proceed_confidence_threshold", base.proceed_confidence_threshold), base.proceed_confidence_threshold),
                0.0,
                1.0,
            ),
            goal_align_action_threshold=_clamp(
                _safe_float(src.get("goal_align_action_threshold", base.goal_align_action_threshold), base.goal_align_action_threshold),
                0.0,
                1.0,
            ),
            calibration_overconfidence_threshold=_clamp(
                _safe_float(
                    src.get("calibration_overconfidence_threshold", base.calibration_overconfidence_threshold),
                    base.calibration_overconfidence_threshold,
                ),
                0.0,
                1.0,
            ),
            calibration_evidence_low_threshold=_clamp(
                _safe_float(
                    src.get("calibration_evidence_low_threshold", base.calibration_evidence_low_threshold),
                    base.calibration_evidence_low_threshold,
                ),
                0.0,
                1.0,
            ),
            calibration_penalty=_clamp(
                _safe_float(src.get("calibration_penalty", base.calibration_penalty), base.calibration_penalty),
                0.0,
                1.0,
            ),
            calibration_goal_detail_evidence_floor=_clamp(
                _safe_float(
                    src.get("calibration_goal_detail_evidence_floor", base.calibration_goal_detail_evidence_floor),
                    base.calibration_goal_detail_evidence_floor,
                ),
                0.0,
                1.0,
            ),
            calibration_consistency_floor=_clamp(
                _safe_float(src.get("calibration_consistency_floor", base.calibration_consistency_floor), base.calibration_consistency_floor),
                0.0,
                1.0,
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_align_weight": float(self.goal_align_weight),
            "grounding_weight": float(self.grounding_weight),
            "rule_prior_scale": float(self.rule_prior_scale),
            "contradiction_penalty_scale": float(self.contradiction_penalty_scale),
            "modal_penalty_scale": float(self.modal_penalty_scale),
            "contradiction_penalty_cap": float(self.contradiction_penalty_cap),
            "unsupported_penalty_scale": float(self.unsupported_penalty_scale),
            "unsupported_penalty_cap": float(self.unsupported_penalty_cap),
            "unsupported_unseen_ratio_trigger": float(self.unsupported_unseen_ratio_trigger),
            "unsupported_min_tokens": int(self.unsupported_min_tokens),
            "goal_miss_penalty": float(self.goal_miss_penalty),
            "goal_miss_align_threshold": float(self.goal_miss_align_threshold),
            "goal_restate_penalty": float(self.goal_restate_penalty),
            "goal_restate_switch_gap": float(self.goal_restate_switch_gap),
            "ambiguity_close_gap": float(self.ambiguity_close_gap),
            "ambiguity_collect_gap": float(self.ambiguity_collect_gap),
            "ambiguity_collect_similarity": float(self.ambiguity_collect_similarity),
            "ambiguity_conflict_penalty": float(self.ambiguity_conflict_penalty),
            "ambiguity_divergent_penalty": float(self.ambiguity_divergent_penalty),
            "confidence_goal_base": float(self.confidence_goal_base),
            "confidence_no_candidate_base": float(self.confidence_no_candidate_base),
            "confidence_consistency_weight": float(self.confidence_consistency_weight),
            "confidence_align_weight": float(self.confidence_align_weight),
            "proceed_confidence_threshold": float(self.proceed_confidence_threshold),
            "goal_align_action_threshold": float(self.goal_align_action_threshold),
            "calibration_overconfidence_threshold": float(self.calibration_overconfidence_threshold),
            "calibration_evidence_low_threshold": float(self.calibration_evidence_low_threshold),
            "calibration_penalty": float(self.calibration_penalty),
            "calibration_goal_detail_evidence_floor": float(self.calibration_goal_detail_evidence_floor),
            "calibration_consistency_floor": float(self.calibration_consistency_floor),
        }


def load_reasoning_profile(path: str) -> ReasoningProfile:
    p = Path(str(path or "").strip())
    if not str(p):
        return ReasoningProfile()
    if not p.exists():
        return ReasoningProfile()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ReasoningProfile()
    if not isinstance(raw, dict):
        return ReasoningProfile()
    return ReasoningProfile.from_mapping(raw)
