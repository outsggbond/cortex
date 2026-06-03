from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from system.brain._reasoning_chains import BackwardChainer, ConsistencyChecker, ForwardChainer
from system.brain._reasoning_feedback import (
    RuleFeedbackMemory,
    _clamp,
    _jaccard,
    _normalize_text,
    _parse_evidence_kv,
    _to_text,
    _tokenize,
)
from system.brain._reasoning_profile import (
    ReasoningProfile,
    ReasoningStep,
    load_reasoning_profile,
)


class HierarchicalReasoner:
    def __init__(
        self,
        feedback_path: str = "artifacts/audit/reasoning_rule_feedback.json",
        *,
        profile_path: str = "",
        profile_overrides: Dict[str, Any] | None = None,
        load_feedback: bool = True,
    ):
        self.forward = ForwardChainer()
        self.backward = BackwardChainer()
        self.consistency = ConsistencyChecker()
        self.profile_path = _to_text(profile_path)
        base_profile = load_reasoning_profile(self.profile_path) if self.profile_path else ReasoningProfile()
        if isinstance(profile_overrides, dict) and profile_overrides:
            merged = dict(base_profile.to_dict())
            merged.update(profile_overrides)
            self.profile = ReasoningProfile.from_mapping(merged)
        else:
            self.profile = base_profile
        self.rule_feedback = RuleFeedbackMemory(path=feedback_path)
        if bool(load_feedback):
            self.rule_feedback.load()

    def profile_snapshot(self) -> Dict[str, Any]:
        return {
            "path": str(self.profile_path or ""),
            "profile": self.profile.to_dict(),
        }

    def _rule_prior_bonus(self, rule: str) -> float:
        prior = float(self.rule_feedback.prior(rule))
        bonus = 0.24 * float(self.profile.rule_prior_scale) * (float(prior) - 0.5)
        if float(self.rule_feedback.total(rule)) < 3.0:
            bonus *= 0.5
        return float(bonus)

    def _support_vocab(self, facts: Dict[str, str], goal: str) -> set[str]:
        vocab: set[str] = set()
        if goal:
            vocab.update(_tokenize(goal))
        for key, value in (facts or {}).items():
            if key is None:
                continue
            vocab.update(_tokenize(_to_text(key)))
            vocab.update(_tokenize(_to_text(value)))
        return vocab

    def _candidate_grounding_score(self, conclusion: str, facts: Dict[str, str]) -> float:
        fact_blob = " ".join(_to_text(v) for v in (facts or {}).values())
        return float(_jaccard(_tokenize(_to_text(conclusion)), _tokenize(fact_blob)))

    def _candidate_quality(
        self,
        *,
        step: ReasoningStep,
        goal: str,
        facts: Dict[str, str],
        contradiction_edges: int,
        modal_conflicts: int,
    ) -> float:
        base = _clamp(float(step.score), 0.0, 1.0)
        conclusion = _to_text(step.conclusion)
        if not conclusion:
            return -1e9

        grounding = self._candidate_grounding_score(conclusion, facts)
        goal_align = self.backward.alignment_score(goal, conclusion, facts) if goal else 0.0

        support_vocab = self._support_vocab(facts, goal)
        cand_tokens = _tokenize(conclusion)
        unseen_ratio = 1.0
        if cand_tokens:
            unseen = sum(1 for t in cand_tokens if t not in support_vocab)
            unseen_ratio = float(unseen) / float(max(1, len(cand_tokens)))

        contradiction_penalty = min(
            float(self.profile.contradiction_penalty_cap),
            float(self.profile.contradiction_penalty_scale) * float(contradiction_edges)
            + float(self.profile.modal_penalty_scale) * float(modal_conflicts),
        )
        unsupported_penalty = 0.0
        prior_penalty = 0.0
        if _to_text(step.rule) in {"goal_restate"}:
            # Keep goal hints useful, but prevent them from overwhelming concrete low-level evidence.
            prior_penalty += float(self.profile.goal_restate_penalty)
        if len(cand_tokens) >= int(self.profile.unsupported_min_tokens) and unseen_ratio >= float(
            self.profile.unsupported_unseen_ratio_trigger
        ):
            unsupported_penalty += min(
                float(self.profile.unsupported_penalty_cap),
                float(self.profile.unsupported_penalty_scale) * float(unseen_ratio),
            )
        if goal and goal_align < float(self.profile.goal_miss_align_threshold):
            unsupported_penalty += float(self.profile.goal_miss_penalty)
        rule_prior_bonus = self._rule_prior_bonus(_to_text(step.rule))

        quality = (
            float(base)
            + float(self.profile.goal_align_weight) * float(goal_align)
            + float(self.profile.grounding_weight) * float(grounding)
            + float(rule_prior_bonus)
            - float(contradiction_penalty)
            - float(unsupported_penalty)
            - float(prior_penalty)
        )
        return float(quality)

    def _rank_candidates(
        self,
        steps: List[ReasoningStep],
        *,
        goal: str,
        facts: Dict[str, str],
        report: Dict[str, Any],
    ) -> List[Tuple[float, ReasoningStep]]:
        contradiction_edges = int(report.get("contradiction_edges", 0) or 0)
        modal_conflicts = int(report.get("modal_conflicts", 0) or 0)
        scored: List[Tuple[float, ReasoningStep]] = []
        for s in steps:
            q = self._candidate_quality(
                step=s,
                goal=goal,
                facts=facts,
                contradiction_edges=contradiction_edges,
                modal_conflicts=modal_conflicts,
            )
            scored.append((float(q), s))
        scored.sort(key=lambda x: float(x[0]), reverse=True)
        return scored

    def _heuristic_low_claims(self, facts: Dict[str, str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        goal = _to_text(facts.get("goal"))
        msg = _to_text(facts.get("message"))
        if goal:
            out.append(
                {
                    "rule": "goal_restate",
                    "score": 0.45,
                    "evidence": "goal_from_context",
                    "conclusion": goal,
                }
            )
        low_msg = msg.lower()
        if ("why" in low_msg) or ("because" in low_msg) or ("为什么" in msg) or ("因为" in msg):
            out.append(
                {
                    "rule": "causal_explain",
                    "score": 0.35,
                    "evidence": "causal_pattern",
                    "conclusion": "explain causal chain before final instruction",
                }
            )
        if (
            ("must" in low_msg and "cannot" in low_msg)
            or ("must" in low_msg and "must not" in low_msg)
            or ("must" in low_msg and "can't" in low_msg)
            or ("should" in low_msg and "should not" in low_msg)
            or ("必须" in msg and "不能" in msg)
            or ("必须" in msg and "不得" in msg)
            or ("应该" in msg and "不应该" in msg)
        ):
            out.append(
                {
                    "rule": "constraint_conflict",
                    "score": 0.65,
                    "evidence": "message_contains_conflicting_modals",
                    "conclusion": "resolve conflicting constraints before execution",
                }
            )
        return out

    def _dedup_steps(self, steps: List[ReasoningStep]) -> List[ReasoningStep]:
        best: Dict[str, ReasoningStep] = {}
        for s in steps:
            key = _normalize_text(s.conclusion) or f"{s.rule}:{_normalize_text(s.evidence)}"
            if not key:
                continue
            prev = best.get(key)
            if prev is None or float(s.score) > float(prev.score):
                best[key] = s
        return list(best.values())

    def _select_best_candidate(
        self,
        steps: List[ReasoningStep],
        *,
        goal: str,
        facts: Dict[str, str],
        report: Dict[str, Any],
        ranked: List[Tuple[float, ReasoningStep]] | None = None,
    ) -> ReasoningStep | None:
        if not steps:
            return None
        ranked_rows = (
            list(ranked or [])
            if ranked is not None
            else self._rank_candidates(
                steps,
                goal=str(goal or ""),
                facts=facts,
                report=report,
            )
        )
        if not ranked_rows:
            return None
        best = ranked_rows[0][1]
        if len(ranked_rows) >= 2:
            q0, s0 = ranked_rows[0]
            q1, s1 = ranked_rows[1]
            if (
                _to_text(s0.rule) in {"goal_restate"}
                and _to_text(s1.rule) not in {"goal_restate"}
                and float(q1) >= float(q0) - float(self.profile.goal_restate_switch_gap)
            ):
                best = s1
        return best

    def _ambiguity_signal(self, ranked: List[Tuple[float, ReasoningStep]]) -> Dict[str, Any]:
        if len(ranked) < 2:
            return {
                "gap": 1.0,
                "similarity": 1.0,
                "contradict": False,
                "close_gap": False,
            }
        q0, s0 = ranked[0]
        q1, s1 = ranked[1]
        gap = max(0.0, float(q0) - float(q1))
        similarity = _jaccard(_tokenize(_to_text(s0.conclusion)), _tokenize(_to_text(s1.conclusion)))
        contradict = bool(self.consistency.contradiction(_to_text(s0.conclusion), _to_text(s1.conclusion)))
        close_gap = bool(gap <= float(self.profile.ambiguity_close_gap))
        return {
            "gap": float(gap),
            "similarity": float(similarity),
            "contradict": bool(contradict),
            "close_gap": bool(close_gap),
            "rule_a": _to_text(s0.rule),
            "rule_b": _to_text(s1.rule),
        }

    def _final_action(
        self,
        *,
        has_goal: bool,
        candidate: str,
        selected_rule: str,
        consistency_score: float,
        align_score: float,
        contradiction_edges: int,
        modal_conflicts: int,
        ambiguity_gap: float,
        ambiguity_similarity: float,
        ambiguity_conflict: bool,
    ) -> Tuple[str, float]:
        base = float(self.profile.confidence_no_candidate_base) if not candidate else float(self.profile.confidence_goal_base)
        confidence = (
            float(base)
            + float(self.profile.confidence_consistency_weight) * float(consistency_score)
            + float(self.profile.confidence_align_weight) * float(align_score)
        )
        confidence -= min(
            float(self.profile.contradiction_penalty_cap),
            float(self.profile.contradiction_penalty_scale) * float(contradiction_edges)
            + float(self.profile.modal_penalty_scale) * float(modal_conflicts),
        )
        if bool(ambiguity_conflict) and float(ambiguity_gap) <= float(self.profile.ambiguity_close_gap):
            confidence -= float(self.profile.ambiguity_conflict_penalty)
        elif (
            float(ambiguity_gap) <= float(self.profile.ambiguity_collect_gap)
            and float(ambiguity_similarity) < float(self.profile.ambiguity_collect_similarity)
        ):
            confidence -= float(self.profile.ambiguity_divergent_penalty)
        confidence = _clamp(confidence, 0.0, 1.0)
        if modal_conflicts > 0 or contradiction_edges > 0:
            return "clarify_constraints", confidence
        if bool(ambiguity_conflict) and float(ambiguity_gap) <= float(self.profile.ambiguity_close_gap):
            return "clarify_constraints", confidence
        if (
            has_goal
            and float(ambiguity_gap) <= float(self.profile.ambiguity_collect_gap)
            and float(ambiguity_similarity) < float(self.profile.ambiguity_collect_similarity)
        ):
            return "collect_goal_details", confidence
        if has_goal and _to_text(selected_rule) in {"goal_restate"}:
            return "collect_goal_details", confidence
        if has_goal and align_score < float(self.profile.goal_align_action_threshold):
            return "collect_goal_details", confidence
        if confidence >= float(self.profile.proceed_confidence_threshold):
            return "proceed", confidence
        return "proceed_with_caution", confidence

    def _calibrate_decision(
        self,
        *,
        action: str,
        confidence: float,
        has_goal: bool,
        align_score: float,
        grounding_score: float,
        consistency_score: float,
    ) -> Tuple[str, float, Dict[str, Any]]:
        evidence = 0.55 * float(align_score) + 0.45 * float(grounding_score)
        overconfident = bool(
            float(confidence) >= float(self.profile.calibration_overconfidence_threshold)
            and float(evidence) <= float(self.profile.calibration_evidence_low_threshold)
        )
        calibrated = float(confidence)
        out_action = str(action or "")
        if overconfident:
            calibrated = _clamp(float(calibrated) - float(self.profile.calibration_penalty), 0.0, 1.0)
        if out_action == "proceed" and calibrated < float(self.profile.proceed_confidence_threshold):
            out_action = "proceed_with_caution"
        if has_goal and float(evidence) <= float(self.profile.calibration_goal_detail_evidence_floor):
            if out_action in {"proceed", "proceed_with_caution"}:
                out_action = "collect_goal_details"
        if out_action == "proceed_with_caution" and float(consistency_score) < float(self.profile.calibration_consistency_floor):
            out_action = "clarify_constraints"
        details = {
            "evidence": float(evidence),
            "overconfident": bool(overconfident),
            "adjusted": bool(abs(float(calibrated) - float(confidence)) > 1e-9 or out_action != action),
        }
        return out_action, float(calibrated), details

    def reason(self, facts: Dict[str, str], low_rules: List[Callable]) -> List[ReasoningStep]:
        steps: List[ReasoningStep] = []
        low = self.forward.run(facts, low_rules) + self._heuristic_low_claims(facts)
        for r in low:
            conclusion = _to_text(r.get("conclusion"))
            if not conclusion:
                continue
            steps.append(
                ReasoningStep(
                    level="low",
                    rule=_to_text(r.get("rule")) or "low_rule",
                    score=float(r.get("score", 0.5)),
                    evidence=_to_text(r.get("evidence")) or "rule_fired",
                    conclusion=conclusion,
                )
            )

        steps = self._dedup_steps(steps)
        low_steps = list(steps)
        goal = _to_text(facts.get("goal"))
        report = self.consistency.analyze(low_steps, message=_to_text(facts.get("message")))
        ranked = self._rank_candidates(low_steps, goal=goal, facts=facts, report=report)
        best = self._select_best_candidate(low_steps, goal=goal, facts=facts, report=report, ranked=ranked)
        ambiguity = self._ambiguity_signal(ranked)

        if low_steps:
            steps.append(
                ReasoningStep(
                    level="mid",
                    rule="consistency_check",
                    score=float(report.get("consistency_score", 0.5)),
                    evidence=f"support={int(report.get('support_edges', 0))}, conflict={int(report.get('contradiction_edges', 0))}, modal_conflict={int(report.get('modal_conflicts', 0))}",
                    conclusion="consistent_reasoning"
                    if float(report.get("consistency_score", 0.0)) >= 0.5
                    else "reasoning_conflict_detected",
                )
            )
        if ranked:
            top = ranked[:3]
            rank_text = ",".join(
                (
                    f"{_to_text(s.rule) or 'candidate'}:{float(q):.3f}"
                    f"|prior={float(self.rule_feedback.prior(_to_text(s.rule))):.3f}"
                )
                for q, s in top
            )
            prior_avg = (
                sum(float(self.rule_feedback.prior(_to_text(s.rule))) for _, s in top)
                / float(max(1, len(top)))
            )
            steps.append(
                ReasoningStep(
                    level="mid",
                    rule="rule_reliability",
                    score=_clamp(float(prior_avg), 0.0, 1.0),
                    evidence=(
                        "top="
                        + ",".join(
                            (
                                f"{_to_text(s.rule) or 'candidate'}:"
                                f"{float(self.rule_feedback.prior(_to_text(s.rule))):.3f}/"
                                f"{float(self.rule_feedback.total(_to_text(s.rule))):.1f}"
                            )
                            for _, s in top
                        )
                    ),
                    conclusion="rule_feedback_applied",
                )
            )
            steps.append(
                ReasoningStep(
                    level="mid",
                    rule="candidate_ranking",
                    score=_clamp(float(top[0][0]), 0.0, 1.0),
                    evidence=f"top={rank_text}",
                    conclusion=_to_text(top[0][1].conclusion),
                )
            )
            if len(ranked) >= 2:
                steps.append(
                    ReasoningStep(
                        level="mid",
                        rule="candidate_ambiguity",
                        score=_clamp(1.0 - float(ambiguity.get("gap", 1.0) or 1.0), 0.0, 1.0),
                        evidence=(
                            f"gap={float(ambiguity.get('gap', 1.0) or 1.0):.3f},"
                            f"similarity={float(ambiguity.get('similarity', 1.0) or 1.0):.3f},"
                            f"contradict={int(bool(ambiguity.get('contradict', False)))}"
                        ),
                        conclusion=(
                            "ambiguous_top_candidates"
                            if bool(ambiguity.get("close_gap", False))
                            else "top_candidate_stable"
                        ),
                    )
                )
        if best is not None:
            steps.append(
                ReasoningStep(
                    level="mid",
                    rule="candidate_select",
                    score=_clamp(float(best.score), 0.0, 1.0),
                    evidence=f"selected_from={len(low_steps)}",
                    conclusion=best.conclusion,
                )
            )
        if int(report.get("modal_conflicts", 0)) > 0:
            steps.append(
                ReasoningStep(
                    level="mid",
                    rule="self_critique_constraints",
                    score=0.72,
                    evidence="modal_conflict_detected",
                    conclusion="ask for constraint clarification before irreversible actions",
                )
            )

        candidate = best.conclusion if best is not None else ""
        selected_rule = _to_text(best.rule) if best is not None else ""
        align = self.backward.alignment_score(goal, candidate, facts) if goal and candidate else 0.0
        if goal:
            if self.backward.verify(goal, facts, candidate) and (selected_rule not in {"goal_restate"}):
                steps.append(
                    ReasoningStep(
                        level="high",
                        rule="goal_alignment",
                        score=max(0.55, _clamp(float(align), 0.0, 1.0)),
                        evidence="candidate_supports_goal",
                        conclusion="goal_aligned_strategy",
                    )
                )
            else:
                steps.append(
                    ReasoningStep(
                        level="high",
                        rule="goal_gap",
                        score=_clamp(float(align), 0.2, 0.6),
                        evidence="goal_not_fully_supported",
                        conclusion="need_more_evidence_or_constraints",
                    )
                )
        elif candidate:
            steps.append(
                ReasoningStep(
                    level="high",
                    rule="fallback_finalize",
                    score=0.55,
                    evidence="no_explicit_goal",
                    conclusion=candidate,
                )
            )

        consistency_score = float(report.get("consistency_score", 0.0) or 0.0)
        contradiction_edges = int(report.get("contradiction_edges", 0) or 0)
        modal_conflicts = int(report.get("modal_conflicts", 0) or 0)
        grounding_score = self._candidate_grounding_score(candidate, facts) if candidate else 0.0
        action, confidence = self._final_action(
            has_goal=bool(goal),
            candidate=candidate,
            selected_rule=selected_rule,
            consistency_score=consistency_score,
            align_score=float(align),
            contradiction_edges=contradiction_edges,
            modal_conflicts=modal_conflicts,
            ambiguity_gap=float(ambiguity.get("gap", 1.0) or 1.0),
            ambiguity_similarity=float(ambiguity.get("similarity", 1.0) or 1.0),
            ambiguity_conflict=bool(ambiguity.get("contradict", False)),
        )
        calibrated_action, calibrated_confidence, calib = self._calibrate_decision(
            action=action,
            confidence=float(confidence),
            has_goal=bool(goal),
            align_score=float(align),
            grounding_score=float(grounding_score),
            consistency_score=float(consistency_score),
        )
        if bool(calib.get("adjusted", False)):
            steps.append(
                ReasoningStep(
                    level="high",
                    rule="confidence_calibration",
                    score=_clamp(float(calibrated_confidence), 0.0, 1.0),
                    evidence=(
                        f"from={action},to={calibrated_action},overconfident={int(bool(calib.get('overconfident', False)))},"
                        f"evidence={float(calib.get('evidence', 0.0) or 0.0):.3f}"
                    ),
                    conclusion="calibrated_decision",
                )
            )
        action = str(calibrated_action)
        confidence = float(calibrated_confidence)

        final_conclusion = candidate or ("need_more_evidence_or_constraints" if goal else "")
        if final_conclusion:
            steps.append(
                ReasoningStep(
                    level="high",
                    rule="final_decision",
                    score=float(confidence),
                    evidence=(
                        f"action={action}, consistency={consistency_score:.3f}, align={float(align):.3f}, "
                        f"grounding={float(grounding_score):.3f}, "
                        f"contradiction={contradiction_edges}, modal_conflict={modal_conflicts}, "
                        f"ambiguity_gap={float(ambiguity.get('gap', 1.0) or 1.0):.3f}, "
                        f"ambiguity_conflict={int(bool(ambiguity.get('contradict', False)))}"
                    ),
                    conclusion=final_conclusion,
                )
            )

        if action in {"clarify_constraints", "collect_goal_details"}:
            steps.append(
                ReasoningStep(
                    level="high",
                    rule="request_clarification",
                    score=max(0.55, float(confidence)),
                    evidence=f"action={action}",
                    conclusion="ask follow-up question before irreversible operation",
                )
            )
        return steps

    def feedback_snapshot(self, *, limit: int = 5) -> Dict[str, Any]:
        return {
            "path": self.rule_feedback.path.as_posix(),
            "rule_count": int(len(self.rule_feedback.rules)),
            "top_rules": self.rule_feedback.top_rules(limit=max(0, int(limit))),
            "updated_at": float(self.rule_feedback.updated_at or 0.0),
        }

    def learn_from_feedback(
        self,
        steps: List[ReasoningStep],
        *,
        success: bool,
        save: bool = True,
    ) -> Dict[str, Any]:
        low_steps = [
            s
            for s in list(steps or [])
            if _to_text(getattr(s, "level", "")) == "low" and _to_text(getattr(s, "rule", ""))
        ]
        if not low_steps:
            return {
                "ok": True,
                "updated_rules": 0,
                "saved": False,
                "path": self.rule_feedback.path.as_posix(),
            }

        selected_key = ""
        for s in reversed(list(steps or [])):
            if _to_text(getattr(s, "rule", "")) != "candidate_select":
                continue
            selected_key = _normalize_text(_to_text(getattr(s, "conclusion", "")))
            if selected_key:
                break

        selected_rules: List[str] = []
        for s in low_steps:
            if selected_key and _normalize_text(_to_text(s.conclusion)) == selected_key:
                selected_rules.append(_to_text(s.rule))

        if not selected_rules:
            best = max(low_steps, key=lambda x: float(getattr(x, "score", 0.0) or 0.0))
            selected_rules.append(_to_text(best.rule))

        touched: Dict[str, Dict[str, float]] = {}
        for rule in selected_rules:
            if not rule or rule in touched:
                continue
            row = self.rule_feedback.update(rule, bool(success), weight=1.0)
            if row:
                touched[rule] = row

        if bool(success):
            for s in low_steps:
                rule = _to_text(s.rule)
                if not rule or rule in touched:
                    continue
                sim = 0.0
                if selected_key:
                    sim = _jaccard(_tokenize(selected_key), _tokenize(_to_text(s.conclusion)))
                if sim < 0.35:
                    continue
                row = self.rule_feedback.update(rule, True, weight=0.25)
                if row:
                    touched[rule] = row

        saved = bool(self.rule_feedback.save()) if bool(save) else False
        return {
            "ok": True,
            "success": bool(success),
            "updated_rules": int(len(touched)),
            "rules": [
                {
                    "rule": str(rule),
                    "score": float(row.get("score", 0.5)),
                    "wins": float(row.get("wins", 0.0)),
                    "total": float(row.get("total", 0.0)),
                }
                for rule, row in touched.items()
            ],
            "saved": bool(saved),
            "path": self.rule_feedback.path.as_posix(),
        }

    def summarize(self, steps: List[ReasoningStep]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "best_conclusion": "",
            "confidence": 0.0,
            "action": "",
            "requires_clarification": False,
        }
        if not steps:
            return summary

        top = max(steps, key=lambda s: float(getattr(s, "score", 0.0) or 0.0))
        summary["best_conclusion"] = _to_text(top.conclusion)
        summary["confidence"] = float(_clamp(float(top.score or 0.0), 0.0, 1.0))

        for s in reversed(steps):
            if s.rule != "final_decision":
                continue
            summary["best_conclusion"] = _to_text(s.conclusion) or summary["best_conclusion"]
            summary["confidence"] = float(_clamp(float(s.score or 0.0), 0.0, 1.0))
            kv = _parse_evidence_kv(s.evidence)
            summary["action"] = _to_text(kv.get("action"))
            break

        for s in steps:
            if s.rule in {"self_critique_constraints", "request_clarification"}:
                summary["requires_clarification"] = True
                break
        return summary


# Re-export public names for backward compatibility
__all__ = [
    "BackwardChainer",
    "ConsistencyChecker",
    "ForwardChainer",
    "HierarchicalReasoner",
    "ReasoningProfile",
    "ReasoningStep",
    "RuleFeedbackMemory",
    "load_reasoning_profile",
]
