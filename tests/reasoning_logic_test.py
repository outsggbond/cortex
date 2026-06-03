from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.brain.neurosymbolic import Rule, NeuroSymbolicEngine
from system.brain.reasoning import HierarchicalReasoner


def test_reasoner_detects_modal_conflict() -> None:
    reasoner = HierarchicalReasoner()
    facts = {
        "message": "You must deploy now, but you cannot deploy now.",
        "goal": "deploy service safely",
    }
    low_rules = [
        lambda f: {
            "rule": "candidate_a",
            "score": 0.6,
            "evidence": "manual",
            "conclusion": "deploy immediately",
        }
    ]
    steps = reasoner.reason(facts, low_rules)
    rules = {s.rule for s in steps}
    assert "self_critique_constraints" in rules
    assert "consistency_check" in rules


def test_reasoner_goal_alignment() -> None:
    reasoner = HierarchicalReasoner()
    facts = {
        "message": "Need to prepare a project report.",
        "goal": "prepare project report",
    }
    low_rules = [
        lambda f: {
            "rule": "candidate_report",
            "score": 0.8,
            "evidence": "direct_match",
            "conclusion": "prepare project report with checklist",
        }
    ]
    steps = reasoner.reason(facts, low_rules)
    assert any(s.rule == "goal_alignment" for s in steps)


def test_reasoner_chinese_conflict_summary() -> None:
    reasoner = HierarchicalReasoner()
    facts = {
        "message": "你必须现在发布，但你不能现在发布。",
        "goal": "安全发布服务",
    }
    low_rules = [
        lambda f: {
            "rule": "candidate_cn",
            "score": 0.75,
            "evidence": "manual_cn",
            "conclusion": "现在发布服务",
        }
    ]
    steps = reasoner.reason(facts, low_rules)
    assert any(s.rule == "self_critique_constraints" for s in steps)
    assert any(s.rule == "final_decision" for s in steps)
    meta = reasoner.summarize(steps)
    assert bool(meta.get("requires_clarification")) is True
    assert str(meta.get("action", "")) in {"clarify_constraints", "collect_goal_details", "proceed_with_caution"}


def test_reasoner_prefers_goal_aligned_candidate_even_if_raw_score_lower() -> None:
    reasoner = HierarchicalReasoner()
    facts = {
        "message": "Need a safe migration preparation plan.",
        "goal": "prepare database backup before migration",
    }
    low_rules = [
        lambda f: {
            "rule": "off_goal_high_score",
            "score": 0.92,
            "evidence": "manual_high",
            "conclusion": "restart web server and clear cache",
        },
        lambda f: {
            "rule": "on_goal_mid_score",
            "score": 0.72,
            "evidence": "manual_mid",
            "conclusion": "prepare database backup and verify checksum before migration",
        },
    ]
    steps = reasoner.reason(facts, low_rules)
    meta = reasoner.summarize(steps)
    best = str(meta.get("best_conclusion", "")).lower()
    assert "backup" in best
    assert any(s.rule == "candidate_ranking" for s in steps)
    assert any(s.rule == "rule_reliability" for s in steps)


def test_reasoner_feedback_learning_persists_rule_prior() -> None:
    with tempfile.TemporaryDirectory() as td:
        feedback_path = Path(td) / "reasoning_feedback.json"
        reasoner = HierarchicalReasoner(feedback_path=feedback_path.as_posix(), load_feedback=True)
        facts = {
            "message": "Need a safe migration preparation plan.",
            "goal": "prepare database backup before migration",
        }
        low_rules = [
            lambda f: {
                "rule": "off_goal_high_score",
                "score": 0.92,
                "evidence": "manual_high",
                "conclusion": "restart web server and clear cache",
            },
            lambda f: {
                "rule": "on_goal_mid_score",
                "score": 0.72,
                "evidence": "manual_mid",
                "conclusion": "prepare database backup and verify checksum before migration",
            },
        ]
        before = float(reasoner.rule_feedback.prior("on_goal_mid_score"))
        steps = reasoner.reason(facts, low_rules)
        out = reasoner.learn_from_feedback(steps, success=True, save=True)
        after = float(reasoner.rule_feedback.prior("on_goal_mid_score"))
        assert bool(out.get("updated_rules", 0)) is True
        assert after > before
        assert feedback_path.exists()

        loaded = HierarchicalReasoner(feedback_path=feedback_path.as_posix(), load_feedback=True)
        reloaded_prior = float(loaded.rule_feedback.prior("on_goal_mid_score"))
        assert reloaded_prior >= after - 1e-6


def test_reasoner_collects_goal_details_on_close_divergent_candidates() -> None:
    reasoner = HierarchicalReasoner()
    facts = {
        "message": "Need latency improvement plan.",
        "goal": "reduce api latency safely",
    }
    low_rules = [
        lambda f: {
            "rule": "db_tune_path",
            "score": 0.74,
            "evidence": "manual",
            "conclusion": "reduce api latency via index coverage and query rewrite",
        },
        lambda f: {
            "rule": "pool_tune_path",
            "score": 0.73,
            "evidence": "manual",
            "conclusion": "reduce api latency using connection pooling and keepalive tuning",
        },
    ]
    steps = reasoner.reason(facts, low_rules)
    meta = reasoner.summarize(steps)
    assert str(meta.get("action", "")) == "collect_goal_details"
    assert bool(meta.get("requires_clarification")) is True
    assert any(s.rule == "candidate_ambiguity" for s in steps)


def test_reasoner_profile_override_controls_proceed_threshold() -> None:
    reasoner = HierarchicalReasoner(
        profile_overrides={
            "proceed_confidence_threshold": 0.96,
            "goal_align_action_threshold": 0.95,
            "goal_align_weight": 0.30,
            "grounding_weight": 0.20,
        }
    )
    facts = {
        "message": "Need migration preparation immediately.",
        "goal": "prepare database backup before migration",
    }
    low_rules = [
        lambda f: {
            "rule": "on_goal_high",
            "score": 0.88,
            "evidence": "manual",
            "conclusion": "prepare database backup and checksum before migration",
        }
    ]
    steps = reasoner.reason(facts, low_rules)
    meta = reasoner.summarize(steps)
    assert str(meta.get("action", "")) == "collect_goal_details"
    snap = reasoner.profile_snapshot()
    prof = snap.get("profile", {})
    assert isinstance(prof, dict)
    assert float(prof.get("proceed_confidence_threshold", 0.0)) >= 0.95


def test_reasoner_confidence_calibration_reduces_overconfidence() -> None:
    reasoner = HierarchicalReasoner(
        profile_overrides={
            "proceed_confidence_threshold": 0.60,
            "calibration_overconfidence_threshold": 0.55,
            "calibration_evidence_low_threshold": 0.90,
            "calibration_penalty": 0.30,
        }
    )
    facts = {
        "message": "Do something quickly.",
        "goal": "",
    }
    low_rules = [
        lambda f: {
            "rule": "speculative_high",
            "score": 0.92,
            "evidence": "manual",
            "conclusion": "rewrite whole stack immediately",
        }
    ]
    steps = reasoner.reason(facts, low_rules)
    meta = reasoner.summarize(steps)
    assert str(meta.get("action", "")) == "proceed_with_caution"
    assert any(s.rule == "confidence_calibration" for s in steps)


def test_neurosym_aggregation_conflict_and_best_conclusion() -> None:
    engine = NeuroSymbolicEngine()
    engine.add_rule(
        Rule(
            name="r1",
            when=lambda ctx: True,
            then=lambda ctx: {"conclusion": "deploy now", "evidence": "rule_1"},
            weight=0.9,
        )
    )
    engine.add_rule(
        Rule(
            name="r2",
            when=lambda ctx: True,
            then=lambda ctx: {"conclusion": "deploy now", "evidence": "rule_2"},
            weight=0.8,
        )
    )
    engine.add_rule(
        Rule(
            name="r3",
            when=lambda ctx: True,
            then=lambda ctx: {"conclusion": "must not deploy now", "evidence": "rule_3"},
            weight=0.6,
        )
    )

    out = engine.infer({"message": "release check"})
    assert isinstance(out, dict)
    assert str(out.get("best_conclusion", "")).strip() == "deploy now"
    assert float(out.get("best_score", 0.0)) > 1.0
    conflicts = out.get("conflicts", [])
    assert isinstance(conflicts, list)
    assert len(conflicts) >= 1


def test_neurosym_auto_expand_chinese_if_then_rule() -> None:
    engine = NeuroSymbolicEngine()
    created = engine.auto_expand_rules("如果网络断开那么启用离线缓存")
    assert created >= 1
    out = engine.infer({"message": "网络断开，进入降级流程"})
    conclusions = out.get("conclusions", [])
    assert isinstance(conclusions, list)
    assert any("离线缓存" in str(row.get("conclusion", "")) for row in conclusions)


def main() -> None:
    test_reasoner_detects_modal_conflict()
    test_reasoner_goal_alignment()
    test_reasoner_chinese_conflict_summary()
    test_reasoner_prefers_goal_aligned_candidate_even_if_raw_score_lower()
    test_reasoner_feedback_learning_persists_rule_prior()
    test_reasoner_collects_goal_details_on_close_divergent_candidates()
    test_reasoner_profile_override_controls_proceed_threshold()
    test_reasoner_confidence_calibration_reduces_overconfidence()
    test_neurosym_aggregation_conflict_and_best_conclusion()
    test_neurosym_auto_expand_chinese_if_then_rule()
    print("reasoning_logic_ok")


if __name__ == "__main__":
    main()
