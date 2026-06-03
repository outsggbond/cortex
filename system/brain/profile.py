from __future__ import annotations

import itertools
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence

from system.brain.reasoning import HierarchicalReasoner, ReasoningProfile, _clamp, load_reasoning_profile
from system.brain.reasoning_benchmark import (
    adversarial_reasoning_cases,
    compare_reasoning_reports,
    ReasoningBenchmarkCase,
    default_reasoning_cases,
    evaluate_reasoning_cases,
    gate_reasoning_report,
    robustness_reasoning_cases,
)


def _mean(values: Sequence[float]) -> float:
    data = [float(x) for x in values]
    if not data:
        return 0.0
    return float(sum(data) / float(max(1, len(data))))


def _profile_key(row: Dict[str, Any]) -> str:
    return "|".join(f"{k}={row.get(k)}" for k in sorted(row.keys()))


def _unique_profiles(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = _profile_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_profile_candidates(base: ReasoningProfile, *, max_candidates: int = 48) -> List[Dict[str, Any]]:
    base_row = base.to_dict()
    candidates: List[Dict[str, Any]] = [dict(base_row)]

    g_vals = sorted(
        {
            round(float(base.goal_align_weight), 4),
            round(_clamp(float(base.goal_align_weight) - 0.05, 0.0, 1.0), 4),
            round(_clamp(float(base.goal_align_weight) + 0.05, 0.0, 1.0), 4),
        }
    )
    gr_vals = sorted(
        {
            round(float(base.grounding_weight), 4),
            round(_clamp(float(base.grounding_weight) - 0.05, 0.0, 1.0), 4),
            round(_clamp(float(base.grounding_weight) + 0.05, 0.0, 1.0), 4),
        }
    )
    p_vals = sorted(
        {
            round(float(base.proceed_confidence_threshold), 4),
            round(_clamp(float(base.proceed_confidence_threshold) - 0.03, 0.0, 1.0), 4),
            round(_clamp(float(base.proceed_confidence_threshold) + 0.03, 0.0, 1.0), 4),
        }
    )
    for goal_w, grounding_w, proceed_t in itertools.product(g_vals, gr_vals, p_vals):
        row = dict(base_row)
        row["goal_align_weight"] = float(goal_w)
        row["grounding_weight"] = float(grounding_w)
        row["proceed_confidence_threshold"] = float(proceed_t)
        candidates.append(row)

    gap_vals = sorted(
        {
            round(float(base.ambiguity_collect_gap), 4),
            round(_clamp(float(base.ambiguity_collect_gap) - 0.01, 0.0, 0.5), 4),
            round(_clamp(float(base.ambiguity_collect_gap) + 0.01, 0.0, 0.5), 4),
        }
    )
    sim_vals = sorted(
        {
            round(float(base.ambiguity_collect_similarity), 4),
            round(_clamp(float(base.ambiguity_collect_similarity) - 0.05, 0.0, 1.0), 4),
            round(_clamp(float(base.ambiguity_collect_similarity) + 0.05, 0.0, 1.0), 4),
        }
    )
    switch_vals = sorted(
        {
            round(float(base.goal_restate_switch_gap), 4),
            round(_clamp(float(base.goal_restate_switch_gap) - 0.02, 0.0, 0.5), 4),
            round(_clamp(float(base.goal_restate_switch_gap) + 0.02, 0.0, 0.5), 4),
        }
    )
    for gap, sim, switch_gap in itertools.product(gap_vals, sim_vals, switch_vals):
        row = dict(base_row)
        row["ambiguity_collect_gap"] = float(gap)
        row["ambiguity_collect_similarity"] = float(sim)
        row["goal_restate_switch_gap"] = float(switch_gap)
        candidates.append(row)

    contra_vals = sorted(
        {
            round(float(base.contradiction_penalty_scale), 4),
            round(_clamp(float(base.contradiction_penalty_scale) - 0.02, 0.0, 1.0), 4),
            round(_clamp(float(base.contradiction_penalty_scale) + 0.02, 0.0, 1.0), 4),
        }
    )
    modal_vals = sorted(
        {
            round(float(base.modal_penalty_scale), 4),
            round(_clamp(float(base.modal_penalty_scale) - 0.03, 0.0, 1.0), 4),
            round(_clamp(float(base.modal_penalty_scale) + 0.03, 0.0, 1.0), 4),
        }
    )
    for contra, modal in itertools.product(contra_vals, modal_vals):
        row = dict(base_row)
        row["contradiction_penalty_scale"] = float(contra)
        row["modal_penalty_scale"] = float(modal)
        candidates.append(row)

    uniq = _unique_profiles(candidates)
    return uniq[: max(1, int(max_candidates))]


_FAILURE_SIGNAL_DELTAS: Dict[str, Dict[str, float]] = {
    "action_mismatch": {
        "proceed_confidence_threshold": 0.04,
        "goal_align_action_threshold": 0.03,
        "contradiction_penalty_scale": 0.03,
    },
    "clarification_mismatch": {
        "proceed_confidence_threshold": 0.05,
        "goal_align_action_threshold": 0.04,
        "calibration_consistency_floor": 0.05,
    },
    "missing_expected_phrase": {
        "goal_align_weight": 0.08,
        "goal_miss_penalty": 0.06,
        "unsupported_penalty_scale": 0.05,
    },
    "rule_missing": {
        "rule_prior_scale": 0.12,
        "grounding_weight": 0.04,
    },
    "ranking_missing": {
        "goal_align_weight": 0.05,
        "grounding_weight": 0.05,
        "goal_restate_penalty": 0.05,
    },
    "category_alignment_low": {
        "goal_align_weight": 0.08,
        "goal_miss_penalty": 0.06,
        "goal_align_action_threshold": 0.03,
    },
    "category_safety_low": {
        "contradiction_penalty_scale": 0.05,
        "modal_penalty_scale": 0.05,
        "proceed_confidence_threshold": 0.04,
    },
    "category_planning_low": {
        "grounding_weight": 0.05,
        "goal_align_weight": 0.04,
    },
    "category_causal_low": {
        "grounding_weight": 0.06,
        "confidence_no_candidate_base": 0.04,
    },
    "category_consistency_low": {
        "contradiction_penalty_scale": 0.05,
        "calibration_consistency_floor": 0.06,
    },
    "category_ambiguity_low": {
        "ambiguity_collect_gap": 0.02,
        "ambiguity_collect_similarity": 0.07,
        "ambiguity_divergent_penalty": 0.05,
    },
}


def _apply_profile_deltas(
    base_row: Dict[str, Any],
    deltas: Dict[str, float],
    *,
    scale: float,
) -> Dict[str, Any]:
    row = dict(base_row)
    for key, delta in (deltas or {}).items():
        current = float(row.get(key, 0.0) or 0.0)
        row[key] = float(current) + float(delta) * float(scale)
    return ReasoningProfile.from_mapping(row).to_dict()


def _load_failure_guided_state(path: str) -> Dict[str, Any]:
    target = Path(str(path or "").strip() or "artifacts/audit/reasoning_failure_guided_state.json")
    if not target.exists():
        return {"signals": {}, "updated_at": 0.0}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {"signals": {}, "updated_at": 0.0}
    if not isinstance(raw, dict):
        return {"signals": {}, "updated_at": 0.0}
    signals = raw.get("signals", {})
    if not isinstance(signals, dict):
        signals = {}
    return {
        "signals": dict(signals),
        "updated_at": float(raw.get("updated_at", 0.0) or 0.0),
    }


def _save_failure_guided_state(path: str, state: Dict[str, Any]) -> None:
    target = Path(str(path or "").strip() or "artifacts/audit/reasoning_failure_guided_state.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state or {})
    payload["updated_at"] = float(time.time())
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


def _signal_entry_weight(runtime_weight: float, entry: Dict[str, Any]) -> float:
    if runtime_weight <= 0.0:
        return 0.0
    node = dict(entry or {})
    uses = max(0, int(node.get("uses", 0) or 0))
    wins = max(0, int(node.get("wins", 0) or 0))
    win_rate = float(wins) / float(max(1, uses))
    ema_gain = float(node.get("ema_gain", 0.0) or 0.0)
    factor = 1.0 + 0.6 * float(ema_gain) + 0.4 * (float(win_rate) - 0.5)
    factor = _clamp(float(factor), 0.35, 1.80)
    return float(runtime_weight) * float(factor)


def _failure_signal_weights(
    report: Dict[str, Any],
    *,
    min_category_rate: float,
    required_categories: Sequence[str],
) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    failure_breakdown = report.get("failure_breakdown", {})
    if isinstance(failure_breakdown, dict):
        for raw_name, raw_count in failure_breakdown.items():
            name = str(raw_name).strip().lower()
            count = float(raw_count or 0.0)
            if (not name) or count <= 0.0:
                continue
            weights[name] = float(weights.get(name, 0.0) + count)

    by_category = report.get("by_category", {})
    if not isinstance(by_category, dict):
        by_category = {}
    categories = [str(x).strip().lower() for x in list(required_categories or []) if str(x).strip()]
    if not categories:
        categories = [str(k).strip().lower() for k in by_category.keys() if str(k).strip()]
    for cat in categories:
        node = by_category.get(cat, {})
        if not isinstance(node, dict):
            continue
        cat_rate = float(node.get("pass_rate", 0.0) or 0.0)
        if cat_rate < float(min_category_rate):
            severity = float(min_category_rate) - float(cat_rate)
            signal = f"category_{cat}_low"
            weights[signal] = float(weights.get(signal, 0.0) + max(0.25, severity * 4.0))
    return weights


def build_failure_guided_candidates(
    base: ReasoningProfile,
    report: Dict[str, Any],
    *,
    min_category_rate: float,
    required_categories: Sequence[str],
    state_signals: Dict[str, Any] | None = None,
    max_variants: int = 16,
) -> Dict[str, Any]:
    max_rows = max(0, int(max_variants))
    if max_rows <= 0:
        return {"signals": [], "variants": []}
    base_row = base.to_dict()
    runtime_weights = _failure_signal_weights(
        report,
        min_category_rate=float(min_category_rate),
        required_categories=list(required_categories or []),
    )
    state_map = dict(state_signals or {})
    weights: Dict[str, float] = {}
    for name, runtime_weight in runtime_weights.items():
        entry = state_map.get(str(name), {})
        if not isinstance(entry, dict):
            entry = {}
        weights[str(name)] = _signal_entry_weight(float(runtime_weight), entry)
    ranked = sorted(
        [(str(name), float(weight)) for name, weight in weights.items() if float(weight) > 0.0],
        key=lambda item: item[1],
        reverse=True,
    )
    variants: List[Dict[str, Any]] = []
    scales = (0.7, 1.0, 1.3)
    for signal, _weight in ranked:
        deltas = _FAILURE_SIGNAL_DELTAS.get(str(signal), {})
        if not deltas:
            continue
        for scale in scales:
            variants.append(
                {
                    "profile": _apply_profile_deltas(base_row, deltas, scale=float(scale)),
                    "signals": [str(signal)],
                    "scale": float(scale),
                    "source": "failure_guided",
                }
            )
            if len(variants) >= max_rows:
                break
        if len(variants) >= max_rows:
            break

    # Blend top two signals when available to capture interaction failures.
    if len(variants) < max_rows and len(ranked) >= 2:
        first = _FAILURE_SIGNAL_DELTAS.get(str(ranked[0][0]), {})
        second = _FAILURE_SIGNAL_DELTAS.get(str(ranked[1][0]), {})
        merged: Dict[str, float] = {}
        for key, value in first.items():
            merged[key] = float(merged.get(key, 0.0) + float(value))
        for key, value in second.items():
            merged[key] = float(merged.get(key, 0.0) + float(value))
        if merged:
            variants.append(
                {
                    "profile": _apply_profile_deltas(base_row, merged, scale=0.85),
                    "signals": [str(ranked[0][0]), str(ranked[1][0])],
                    "scale": 0.85,
                    "source": "failure_guided_blend",
                }
            )

    uniq_items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in variants:
        if not isinstance(item, dict):
            continue
        profile = item.get("profile", {})
        if not isinstance(profile, dict):
            continue
        key = _profile_key(ReasoningProfile.from_mapping(profile).to_dict())
        if key in seen:
            continue
        seen.add(key)
        uniq_items.append(item)
        if len(uniq_items) >= max_rows:
            break
    return {
        "signals": [
            {
                "name": str(name),
                "weight": float(weight),
                "runtime_weight": float(runtime_weights.get(str(name), 0.0) or 0.0),
            }
            for name, weight in ranked[: min(8, len(ranked))]
        ],
        "variants": uniq_items,
    }


def _report_metrics(report: Dict[str, Any]) -> Dict[str, float]:
    by_category = report.get("by_category", {})
    cat_rates = []
    if isinstance(by_category, dict):
        for node in by_category.values():
            if isinstance(node, dict):
                cat_rates.append(float(node.get("pass_rate", 0.0) or 0.0))
    category_mean = _mean(cat_rates)
    suite_mean = _mean(
        [
            float(node.get("pass_rate", 0.0) or 0.0)
            for node in list((report.get("by_suite", {}) or {}).values())
            if isinstance(node, dict)
        ]
    )
    difficulty_mean = _mean(
        [
            float(node.get("pass_rate", 0.0) or 0.0)
            for node in list((report.get("by_difficulty", {}) or {}).values())
            if isinstance(node, dict)
        ]
    )
    weighted_pass_rate = float(report.get("weighted_pass_rate", report.get("pass_rate", 0.0)) or 0.0)
    hard_case_pass_rate = float(report.get("hard_case_pass_rate", 0.0) or 0.0)
    calibration = report.get("calibration", {})
    if not isinstance(calibration, dict):
        calibration = {}
    ece = float(calibration.get("ece", 1.0) or 1.0)
    brier = float(calibration.get("brier", 1.0) or 1.0)
    case_rows = list(report.get("cases", []) or [])
    clarify_rate = (
        float(sum(1 for c in case_rows if bool(c.get("requires_clarification", False))))
        / float(max(1, len(case_rows)))
        if case_rows
        else 0.0
    )
    pass_rate = float(report.get("pass_rate", 0.0) or 0.0)
    score = (
        2.0 * float(pass_rate)
        + 1.2 * float(weighted_pass_rate)
        + 0.8 * float(hard_case_pass_rate)
        + float(category_mean)
        + 0.3 * float(difficulty_mean)
        + 0.5 * float(suite_mean)
        - 0.05 * float(clarify_rate)
        - 0.6 * float(ece)
        - 0.4 * float(brier)
    )
    return {
        "pass_rate": float(pass_rate),
        "weighted_pass_rate": float(weighted_pass_rate),
        "hard_case_pass_rate": float(hard_case_pass_rate),
        "category_mean": float(category_mean),
        "difficulty_mean": float(difficulty_mean),
        "suite_mean": float(suite_mean),
        "ece": float(ece),
        "brier": float(brier),
        "clarify_rate": float(clarify_rate),
        "score": float(score),
    }


def _baseline_bonus(compare: Dict[str, Any]) -> float:
    payload = dict(compare or {})
    comparable = int(payload.get("comparable_cases", 0) or 0)
    if comparable <= 0:
        return 0.0
    delta = float(payload.get("delta_pass_rate", 0.0) or 0.0)
    win_rate = float(payload.get("discordant_win_rate", 0.0) or 0.0)
    win_lb = float(payload.get("discordant_win_rate_ci95_lower", 0.0) or 0.0)
    pval = float(payload.get("mcnemar_pvalue", 1.0) or 1.0)
    discordant = int(payload.get("discordant_cases", 0) or 0)
    confidence_weight = _clamp(float(comparable) / 12.0, 0.2, 1.0)
    conflict_weight = _clamp(float(discordant) / float(max(1, comparable)), 0.2, 1.0)
    bonus = (
        1.4 * float(delta)
        + 0.45 * (float(win_rate) - 0.5)
        + 0.35 * (float(win_lb) - 0.5)
        - 0.25 * max(0.0, float(pval) - 0.05)
    )
    return float(bonus) * float(confidence_weight) * float(conflict_weight)


def _update_failure_guided_state(
    state: Dict[str, Any],
    *,
    observed_signals: Sequence[str],
    selected_signals: Sequence[str],
    score_gain: float,
    gate_passed: bool,
) -> Dict[str, Any]:
    payload = dict(state or {})
    signals = payload.get("signals", {})
    if not isinstance(signals, dict):
        signals = {}
    now_ts = float(time.time())
    observed = [str(x).strip().lower() for x in list(observed_signals or []) if str(x).strip()]
    selected = [str(x).strip().lower() for x in list(selected_signals or []) if str(x).strip()]
    selected_set = set(selected)
    for signal in observed:
        node = signals.get(signal, {})
        if not isinstance(node, dict):
            node = {}
        node["seen"] = int(node.get("seen", 0) or 0) + 1
        node["last_seen_ts"] = float(now_ts)
        signals[signal] = node
    improved = bool(float(score_gain) > 1e-9)
    for signal in selected_set:
        node = signals.get(signal, {})
        if not isinstance(node, dict):
            node = {}
        uses = int(node.get("uses", 0) or 0) + 1
        wins = int(node.get("wins", 0) or 0) + (1 if (improved and bool(gate_passed)) else 0)
        ema = float(node.get("ema_gain", 0.0) or 0.0)
        ema = 0.8 * float(ema) + 0.2 * float(score_gain)
        node.update(
            {
                "uses": int(uses),
                "wins": int(wins),
                "ema_gain": float(ema),
                "last_selected_ts": float(now_ts),
            }
        )
        signals[signal] = node
    payload["signals"] = signals
    payload["updated_at"] = float(now_ts)
    return payload


def tune_reasoning_profile(
    *,
    profile_path: str = "",
    feedback_path: str = "artifacts/audit/reasoning_rule_feedback.json",
    max_candidates: int = 48,
    min_pass_rate: float = 0.90,
    min_category_rate: float = 0.75,
    min_weighted_pass_rate: float = 0.0,
    min_pass_rate_ci95_lower: float = 0.0,
    min_category_rate_ci95_lower: float = 0.0,
    required_categories: Sequence[str] = ("alignment", "safety", "planning", "causal", "consistency", "ambiguity"),
    min_suite_rate: float = 0.0,
    min_suite_rate_ci95_lower: float = 0.0,
    required_suites: Sequence[str] = (),
    min_difficulty_rate: float = 0.0,
    min_difficulty_rate_ci95_lower: float = 0.0,
    required_difficulties: Sequence[str] = (),
    min_hard_case_rate: float = 0.0,
    min_hard_case_rate_ci95_lower: float = 0.0,
    max_ece: float = 0.0,
    max_brier: float = 0.0,
    with_robustness: bool = False,
    with_adversarial: bool = False,
    extra_cases: Sequence[ReasoningBenchmarkCase] = (),
    failure_guided: bool = True,
    failure_guided_max_variants: int = 16,
    failure_guided_state_path: str = "artifacts/audit/reasoning_failure_guided_state.json",
    persist_failure_guided_state: bool = True,
    use_feedback: bool = False,
    baseline_report: Dict[str, Any] | None = None,
    baseline_profile_path: str = "",
    baseline_feedback_path: str = "",
    min_delta_pass_rate: float = 0.0,
    max_mcnemar_pvalue: float = 0.0,
    min_discordant_win_rate: float = 0.0,
    min_discordant_win_rate_ci95_lower: float = 0.0,
    min_comparable_cases: int = 0,
) -> Dict[str, Any]:
    base_profile = load_reasoning_profile(profile_path) if str(profile_path or "").strip() else ReasoningProfile()
    base_candidates = build_profile_candidates(base_profile, max_candidates=max(1, int(max_candidates)))
    core_cases = list(default_reasoning_cases())
    benchmark_cases = list(core_cases)
    if bool(with_robustness):
        benchmark_cases.extend(robustness_reasoning_cases(core_cases))
    if bool(with_adversarial):
        benchmark_cases.extend(adversarial_reasoning_cases(core_cases))
    if extra_cases:
        benchmark_cases.extend(list(extra_cases))
    baseline_profile_path_raw = str(baseline_profile_path or "").strip()
    baseline_feedback_path_raw = str(baseline_feedback_path or "").strip() or str(feedback_path)
    baseline_gate_requested = (
        float(min_delta_pass_rate) != 0.0
        or float(max_mcnemar_pvalue) > 0.0
        or float(min_discordant_win_rate) > 0.0
        or float(min_discordant_win_rate_ci95_lower) > 0.0
        or int(min_comparable_cases) > 0
    )
    baseline_source = ""
    baseline_for_gate: Dict[str, Any] | None = None
    if isinstance(baseline_report, dict) and bool(baseline_report):
        baseline_for_gate = dict(baseline_report)
        baseline_source = "report:in_memory"
    elif baseline_profile_path_raw:
        baseline_reasoner_for_gate = HierarchicalReasoner(
            feedback_path=str(baseline_feedback_path_raw),
            profile_path=str(baseline_profile_path_raw),
            load_feedback=bool(use_feedback),
        )
        baseline_for_gate = dict(
            evaluate_reasoning_cases(
                benchmark_cases,
                reasoner=baseline_reasoner_for_gate,
                learn_feedback=False,
                save_feedback=False,
            )
        )
        baseline_source = f"profile:{baseline_profile_path_raw}"
    if baseline_gate_requested and (baseline_for_gate is None):
        raise ValueError(
            "baseline-relative tuning gates require baseline_report or baseline_profile_path"
        )
    state_path = str(failure_guided_state_path or "").strip() or "artifacts/audit/reasoning_failure_guided_state.json"
    failure_guided_meta: Dict[str, Any] = {
        "enabled": bool(failure_guided),
        "max_variants": max(0, int(failure_guided_max_variants)),
        "variants_added": 0,
        "signals": [],
        "state_path": str(state_path),
        "state_loaded": False,
        "state_persisted": False,
        "state_signal_count": 0,
        "selected_source": "",
        "selected_signals": [],
        "score_gain": 0.0,
    }
    baseline_meta: Dict[str, Any] = {
        "enabled": baseline_for_gate is not None,
        "source": str(baseline_source),
        "gate_active": bool(baseline_gate_requested),
        "min_delta_pass_rate": float(min_delta_pass_rate),
        "max_mcnemar_pvalue": float(max_mcnemar_pvalue),
        "min_discordant_win_rate": float(min_discordant_win_rate),
        "min_discordant_win_rate_ci95_lower": float(min_discordant_win_rate_ci95_lower),
        "min_comparable_cases": int(min_comparable_cases),
    }
    candidate_limit = max(1, int(max_candidates)) + (
        max(0, int(failure_guided_max_variants)) if bool(failure_guided) else 0
    )
    candidate_items: List[Dict[str, Any]] = []
    seen_profile_keys: set[str] = set()

    def _append_candidate(
        profile_row: Dict[str, Any],
        *,
        source: str,
        signals: Sequence[str] | None = None,
        scale: float = 1.0,
    ) -> bool:
        if len(candidate_items) >= int(candidate_limit):
            return False
        normalized = ReasoningProfile.from_mapping(profile_row).to_dict()
        key = _profile_key(normalized)
        if key in seen_profile_keys:
            return False
        seen_profile_keys.add(key)
        clean_signals = [str(x).strip().lower() for x in list(signals or []) if str(x).strip()]
        candidate_items.append(
            {
                "profile": normalized,
                "source": str(source or "grid"),
                "signals": clean_signals,
                "scale": float(scale),
            }
        )
        return True

    for row in base_candidates:
        _append_candidate(dict(row), source="grid", signals=(), scale=1.0)

    baseline_metrics: Dict[str, float] = {}
    failure_baseline_report: Dict[str, Any] = {}
    observed_signal_names: List[str] = []
    failure_guided_state: Dict[str, Any] = {"signals": {}, "updated_at": 0.0}
    if bool(failure_guided):
        failure_guided_state = _load_failure_guided_state(state_path)
        state_signals = failure_guided_state.get("signals", {})
        if not isinstance(state_signals, dict):
            state_signals = {}
        failure_guided_meta["state_loaded"] = bool(state_signals)
        baseline_reasoner = HierarchicalReasoner(
            feedback_path=str(feedback_path),
            profile_overrides=base_profile.to_dict(),
            load_feedback=bool(use_feedback),
        )
        failure_baseline_report = dict(
            evaluate_reasoning_cases(
                benchmark_cases,
                reasoner=baseline_reasoner,
                learn_feedback=False,
                save_feedback=False,
            )
        )
        baseline_metrics = _report_metrics(failure_baseline_report)
        guided = build_failure_guided_candidates(
            base_profile,
            failure_baseline_report,
            min_category_rate=float(min_category_rate),
            required_categories=list(required_categories or []),
            state_signals=state_signals,
            max_variants=max(0, int(failure_guided_max_variants)),
        )
        guided_items = list(guided.get("variants", []) or [])
        added_count = 0
        for guided_item in guided_items:
            if not isinstance(guided_item, dict):
                continue
            profile_row = guided_item.get("profile", {})
            if not isinstance(profile_row, dict):
                continue
            added = _append_candidate(
                profile_row,
                source=str(guided_item.get("source", "failure_guided") or "failure_guided"),
                signals=list(guided_item.get("signals", []) or []),
                scale=float(guided_item.get("scale", 1.0) or 1.0),
            )
            if added:
                added_count += 1
            if len(candidate_items) >= int(candidate_limit):
                break
        observed_signal_names = [
            str(node.get("name", "")).strip().lower()
            for node in list(guided.get("signals", []) or [])
            if isinstance(node, dict) and str(node.get("name", "")).strip()
        ]
        failure_breakdown = failure_baseline_report.get("failure_breakdown", {})
        if not isinstance(failure_breakdown, dict):
            failure_breakdown = {}
        failure_guided_meta.update(
            {
                "baseline_pass_rate": float(baseline_metrics.get("pass_rate", 0.0) or 0.0),
                "baseline_weighted_pass_rate": float(baseline_metrics.get("weighted_pass_rate", 0.0) or 0.0),
                "baseline_hard_case_pass_rate": float(baseline_metrics.get("hard_case_pass_rate", 0.0) or 0.0),
                "baseline_score": float(baseline_metrics.get("score", 0.0) or 0.0),
                "baseline_failure_breakdown": dict(failure_breakdown),
                "variants_added": int(added_count),
                "signals": list(guided.get("signals", []) or []),
            }
        )
    rows: List[Dict[str, Any]] = []

    for idx, candidate_item in enumerate(candidate_items):
        profile_row = dict(candidate_item.get("profile", {}) or {})
        reasoner = HierarchicalReasoner(
            feedback_path=str(feedback_path),
            profile_overrides=dict(profile_row),
            load_feedback=bool(use_feedback),
        )
        report = evaluate_reasoning_cases(
            benchmark_cases,
            reasoner=reasoner,
            learn_feedback=False,
            save_feedback=False,
        )
        gate_ok, reasons = gate_reasoning_report(
            report,
            min_pass_rate=float(min_pass_rate),
            min_category_rate=float(min_category_rate),
            min_weighted_pass_rate=(float(min_weighted_pass_rate) if float(min_weighted_pass_rate) > 0.0 else None),
            min_pass_rate_ci95_lower=(
                float(min_pass_rate_ci95_lower) if float(min_pass_rate_ci95_lower) > 0.0 else None
            ),
            min_category_rate_ci95_lower=(
                float(min_category_rate_ci95_lower) if float(min_category_rate_ci95_lower) > 0.0 else None
            ),
            required_categories=list(required_categories),
            min_suite_rate=(float(min_suite_rate) if float(min_suite_rate) > 0.0 else None),
            min_suite_rate_ci95_lower=(
                float(min_suite_rate_ci95_lower) if float(min_suite_rate_ci95_lower) > 0.0 else None
            ),
            required_suites=list(required_suites or []),
            min_difficulty_rate=(float(min_difficulty_rate) if float(min_difficulty_rate) > 0.0 else None),
            min_difficulty_rate_ci95_lower=(
                float(min_difficulty_rate_ci95_lower) if float(min_difficulty_rate_ci95_lower) > 0.0 else None
            ),
            required_difficulties=list(required_difficulties or []),
            min_hard_case_rate=(float(min_hard_case_rate) if float(min_hard_case_rate) > 0.0 else None),
            min_hard_case_rate_ci95_lower=(
                float(min_hard_case_rate_ci95_lower) if float(min_hard_case_rate_ci95_lower) > 0.0 else None
            ),
            max_ece=(float(max_ece) if float(max_ece) > 0.0 else None),
            max_brier=(float(max_brier) if float(max_brier) > 0.0 else None),
            baseline_report=baseline_for_gate,
            min_delta_pass_rate=(float(min_delta_pass_rate) if float(min_delta_pass_rate) != 0.0 else None),
            max_mcnemar_pvalue=(float(max_mcnemar_pvalue) if float(max_mcnemar_pvalue) > 0.0 else None),
            min_discordant_win_rate=(
                float(min_discordant_win_rate) if float(min_discordant_win_rate) > 0.0 else None
            ),
            min_discordant_win_rate_ci95_lower=(
                float(min_discordant_win_rate_ci95_lower)
                if float(min_discordant_win_rate_ci95_lower) > 0.0
                else None
            ),
            min_comparable_cases=(int(min_comparable_cases) if int(min_comparable_cases) > 0 else None),
        )
        metrics = _report_metrics(report)
        baseline_compare: Dict[str, Any] = {}
        if baseline_for_gate is not None:
            baseline_compare = compare_reasoning_reports(report, baseline_for_gate)
        baseline_bonus = _baseline_bonus(baseline_compare) if baseline_compare else 0.0
        rows.append(
            {
                "idx": int(idx),
                "profile": dict(profile_row),
                "pass_rate": float(metrics.get("pass_rate", 0.0) or 0.0),
                "weighted_pass_rate": float(metrics.get("weighted_pass_rate", 0.0) or 0.0),
                "hard_case_pass_rate": float(metrics.get("hard_case_pass_rate", 0.0) or 0.0),
                "total_cases": int(report.get("total_cases", 0) or 0),
                "passed_cases": int(report.get("passed_cases", 0) or 0),
                "category_mean": float(metrics.get("category_mean", 0.0) or 0.0),
                "difficulty_mean": float(metrics.get("difficulty_mean", 0.0) or 0.0),
                "suite_mean": float(metrics.get("suite_mean", 0.0) or 0.0),
                "ece": float(metrics.get("ece", 0.0) or 0.0),
                "brier": float(metrics.get("brier", 0.0) or 0.0),
                "clarify_rate": float(metrics.get("clarify_rate", 0.0) or 0.0),
                "gate_passed": bool(gate_ok),
                "gate_reasons": list(reasons),
                "score_raw": float(metrics.get("score", 0.0) or 0.0),
                "score_baseline_bonus": float(baseline_bonus),
                "score": float(metrics.get("score", 0.0) or 0.0) + float(baseline_bonus),
                "baseline_compare": dict(baseline_compare),
                "baseline_comparable_cases": int(baseline_compare.get("comparable_cases", 0) or 0)
                if baseline_compare
                else 0,
                "baseline_delta_pass_rate": float(baseline_compare.get("delta_pass_rate", 0.0) or 0.0)
                if baseline_compare
                else 0.0,
                "baseline_mcnemar_pvalue": float(baseline_compare.get("mcnemar_pvalue", 1.0) or 1.0)
                if baseline_compare
                else 1.0,
                "baseline_discordant_win_rate": float(baseline_compare.get("discordant_win_rate", 0.0) or 0.0)
                if baseline_compare
                else 0.0,
                "baseline_discordant_win_rate_ci95_lower": float(
                    baseline_compare.get("discordant_win_rate_ci95_lower", 0.0) or 0.0
                )
                if baseline_compare
                else 0.0,
                "candidate_source": str(candidate_item.get("source", "grid") or "grid"),
                "candidate_signals": [
                    str(x).strip().lower()
                    for x in list(candidate_item.get("signals", []) or [])
                    if str(x).strip()
                ],
                "candidate_scale": float(candidate_item.get("scale", 1.0) or 1.0),
            }
        )

    rows.sort(
        key=lambda r: (
            1 if bool(r.get("gate_passed", False)) else 0,
            float(r.get("score", 0.0) or 0.0),
            float(r.get("weighted_pass_rate", 0.0) or 0.0),
            float(r.get("hard_case_pass_rate", 0.0) or 0.0),
            float(r.get("pass_rate", 0.0) or 0.0),
            float(r.get("category_mean", 0.0) or 0.0),
            float(r.get("difficulty_mean", 0.0) or 0.0),
            float(r.get("suite_mean", 0.0) or 0.0),
            float(r.get("baseline_delta_pass_rate", 0.0) or 0.0),
            float(r.get("baseline_discordant_win_rate_ci95_lower", 0.0) or 0.0),
            -float(r.get("baseline_mcnemar_pvalue", 1.0) or 1.0),
            -float(r.get("ece", 0.0) or 0.0),
            -float(r.get("brier", 0.0) or 0.0),
        ),
        reverse=True,
    )
    best = rows[0] if rows else {"profile": base_profile.to_dict(), "gate_passed": False, "gate_reasons": ["no_candidates"]}
    if bool(failure_guided):
        selected_signals = [str(x).strip().lower() for x in list(best.get("candidate_signals", []) or []) if str(x).strip()]
        selected_source = str(best.get("candidate_source", "") or "")
        score_gain = float(best.get("score", 0.0) or 0.0) - float(baseline_metrics.get("score", 0.0) or 0.0)
        failure_guided_meta.update(
            {
                "selected_source": selected_source,
                "selected_signals": list(selected_signals),
                "score_gain": float(score_gain),
            }
        )
        next_state = _update_failure_guided_state(
            failure_guided_state,
            observed_signals=observed_signal_names,
            selected_signals=selected_signals,
            score_gain=float(score_gain),
            gate_passed=bool(best.get("gate_passed", False)),
        )
        signal_map = next_state.get("signals", {})
        if not isinstance(signal_map, dict):
            signal_map = {}
        failure_guided_meta["state_signal_count"] = int(len(signal_map))
        if bool(persist_failure_guided_state):
            try:
                _save_failure_guided_state(state_path, next_state)
                failure_guided_meta["state_persisted"] = True
            except Exception as exc:
                failure_guided_meta["state_persisted"] = False
                failure_guided_meta["state_persist_error"] = f"{type(exc).__name__}: {exc}"
    return {
        "ok": True,
        "candidates_evaluated": int(len(rows)),
        "best_profile": dict(best.get("profile", base_profile.to_dict()) or base_profile.to_dict()),
        "failure_guided": failure_guided_meta,
        "baseline": baseline_meta,
        "best_metrics": {
            "pass_rate": float(best.get("pass_rate", 0.0) or 0.0),
            "weighted_pass_rate": float(best.get("weighted_pass_rate", 0.0) or 0.0),
            "hard_case_pass_rate": float(best.get("hard_case_pass_rate", 0.0) or 0.0),
            "category_mean": float(best.get("category_mean", 0.0) or 0.0),
            "difficulty_mean": float(best.get("difficulty_mean", 0.0) or 0.0),
            "suite_mean": float(best.get("suite_mean", 0.0) or 0.0),
            "ece": float(best.get("ece", 0.0) or 0.0),
            "brier": float(best.get("brier", 0.0) or 0.0),
            "clarify_rate": float(best.get("clarify_rate", 0.0) or 0.0),
            "score": float(best.get("score", 0.0) or 0.0),
            "score_raw": float(best.get("score_raw", 0.0) or 0.0),
            "score_baseline_bonus": float(best.get("score_baseline_bonus", 0.0) or 0.0),
            "gate_passed": bool(best.get("gate_passed", False)),
            "gate_reasons": list(best.get("gate_reasons", []) or []),
            "baseline_comparable_cases": int(best.get("baseline_comparable_cases", 0) or 0),
            "baseline_delta_pass_rate": float(best.get("baseline_delta_pass_rate", 0.0) or 0.0),
            "baseline_mcnemar_pvalue": float(best.get("baseline_mcnemar_pvalue", 1.0) or 1.0),
            "baseline_discordant_win_rate": float(best.get("baseline_discordant_win_rate", 0.0) or 0.0),
            "baseline_discordant_win_rate_ci95_lower": float(
                best.get("baseline_discordant_win_rate_ci95_lower", 0.0) or 0.0
            ),
            "baseline_compare": dict(best.get("baseline_compare", {}) or {}),
        },
        "top_candidates": rows[: min(8, len(rows))],
    }
