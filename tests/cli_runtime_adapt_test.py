from __future__ import annotations

import contextlib
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.main_cli import build_parser, parse_args


def _assert_parse_fails(argv: list[str]) -> None:
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            parse_args(argv)
    except SystemExit as exc:
        assert int(exc.code or 0) == 2
        return
    raise AssertionError(f"expected parse failure for argv={argv!r}")


def test_runtime_adapt_cli_flags_parse() -> None:
    args = parse_args(
        [
            "--runtime-adapt",
            "--runtime-adapt-sample-interval-s",
            "7",
            "--runtime-adapt-cooldown-s",
            "9",
            "--runtime-adapt-pressure-high",
            "0.9",
            "--runtime-adapt-pressure-low",
            "0.5",
            "--runtime-adapt-heavy-route-min-score",
            "0.66",
            "--runtime-adapt-heavy-route-max-pressure",
            "0.72",
            "--runtime-adapt-bandit",
            "--runtime-adapt-bandit-explore",
            "0.88",
            "--runtime-adapt-bandit-auto-explore",
            "--runtime-adapt-bandit-explore-min",
            "0.25",
            "--runtime-adapt-bandit-explore-max",
            "1.4",
            "--runtime-adapt-bandit-explore-step-up",
            "0.05",
            "--runtime-adapt-bandit-explore-step-down",
            "0.03",
            "--runtime-adapt-bandit-contextual-bias",
            "--runtime-adapt-bandit-context-pressure-weight",
            "0.42",
            "--runtime-adapt-bandit-context-score-weight",
            "0.28",
            "--runtime-adapt-bandit-context-bias-cap",
            "0.18",
            "--runtime-adapt-bandit-context-cold-start",
            "--runtime-adapt-bandit-context-cold-start-gain",
            "0.55",
            "--runtime-adapt-bandit-hardware-guard",
            "--runtime-adapt-bandit-hardware-strict",
            "--runtime-adapt-bandit-hardware-mismatch-count-decay",
            "0.5",
            "--runtime-adapt-bandit-hardware-mismatch-value-decay",
            "0.6",
            "--runtime-adapt-bandit-hardware-mismatch-explore-boost",
            "0.3",
            "--runtime-adapt-bandit-time-decay",
            "--runtime-adapt-bandit-time-decay-half-life-s",
            "2400",
            "--runtime-adapt-bandit-time-decay-min-factor",
            "0.22",
            "--runtime-adapt-bandit-time-decay-cooldown-s",
            "30",
            "--runtime-adapt-bandit-arm-cooldown",
            "--runtime-adapt-bandit-arm-cooldown-reward-threshold",
            "-0.7",
            "--runtime-adapt-bandit-arm-cooldown-trigger-streak",
            "3",
            "--runtime-adapt-bandit-arm-cooldown-s",
            "50",
            "--runtime-adapt-bandit-all-cooled-safe-fallback",
            "--runtime-adapt-bandit-arm-risk-aware",
            "--runtime-adapt-bandit-arm-risk-alpha",
            "0.33",
            "--runtime-adapt-bandit-arm-risk-weight",
            "0.55",
            "--runtime-adapt-bandit-arm-risk-neg-weight",
            "0.44",
            "--runtime-adapt-bandit-arm-risk-cap",
            "0.66",
            "--runtime-adapt-bandit-safety-guard",
            "--runtime-adapt-bandit-safety-pressure-gate",
            "0.91",
            "--runtime-adapt-bandit-safety-score-gate",
            "0.29",
            "--runtime-adapt-bandit-safety-max-multiplier",
            "1.0",
            "--runtime-adapt-bandit-uncertainty-guard",
            "--runtime-adapt-bandit-uncertainty-var-gate",
            "0.31",
            "--runtime-adapt-bandit-uncertainty-min-samples",
            "7",
            "--runtime-adapt-bandit-uncertainty-pressure-gate",
            "0.58",
            "--runtime-adapt-bandit-uncertainty-score-gate",
            "0.39",
            "--runtime-adapt-bandit-uncertainty-max-multiplier",
            "0.99",
            "--runtime-adapt-bandit-regime-split",
            "--runtime-adapt-bandit-regime-pressure-gate",
            "0.79",
            "--runtime-adapt-bandit-regime-score-gate",
            "0.43",
            "--runtime-adapt-bandit-shock-guard",
            "--runtime-adapt-bandit-shock-reward-threshold",
            "-0.88",
            "--runtime-adapt-bandit-shock-trigger-count",
            "3",
            "--runtime-adapt-bandit-shock-window-s",
            "28",
            "--runtime-adapt-bandit-shock-cooldown-s",
            "55",
            "--runtime-adapt-bandit-shock-max-multiplier",
            "0.95",
            "--runtime-adapt-bandit-spike-guard",
            "--runtime-adapt-bandit-spike-pressure-delta-threshold",
            "0.19",
            "--runtime-adapt-bandit-spike-score-drop-threshold",
            "0.23",
            "--runtime-adapt-bandit-spike-min-pressure",
            "0.57",
            "--runtime-adapt-bandit-spike-cooldown-s",
            "38",
            "--runtime-adapt-bandit-spike-max-multiplier",
            "0.98",
            "--runtime-adapt-state-path",
            "artifacts/audit/runtime_state_test.json",
            "--runtime-adapt-load-state",
            "--runtime-adapt-save-interval-s",
            "13",
            "--runtime-adapt-dynamic-interval",
            "--runtime-adapt-interval-min-s",
            "2.5",
            "--runtime-adapt-interval-max-s",
            "21",
            "--runtime-adapt-interval-high-mul",
            "0.6",
            "--runtime-adapt-interval-low-mul",
            "1.4",
            "--runtime-adapt-interval-throttle-mul",
            "0.5",
            "--runtime-adapt-threshold-auto",
            "--runtime-adapt-threshold-min-high",
            "0.69",
            "--runtime-adapt-threshold-max-high",
            "0.91",
            "--runtime-adapt-threshold-gap",
            "0.2",
            "--runtime-adapt-threshold-step-up",
            "0.004",
            "--runtime-adapt-threshold-step-down",
            "0.012",
            "--runtime-transformer-hot-tune",
            "--runtime-planner-hot-tune",
            "--runtime-planner-depth-min",
            "2",
            "--runtime-planner-depth-max",
            "7",
            "--runtime-planner-beam-min",
            "2",
            "--runtime-planner-beam-max",
            "11",
        ]
    )
    assert bool(args.runtime_adapt_enabled) is True
    assert float(args.runtime_adapt_sample_interval_s) == 7.0
    assert float(args.runtime_adapt_cooldown_s) == 9.0
    assert float(args.runtime_adapt_pressure_high) == 0.9
    assert float(args.runtime_adapt_pressure_low) == 0.5
    assert float(args.runtime_adapt_heavy_route_min_score) == 0.66
    assert float(args.runtime_adapt_heavy_route_max_pressure) == 0.72
    assert bool(args.runtime_adapt_bandit_enabled) is True
    assert float(args.runtime_adapt_bandit_explore) == 0.88
    assert bool(args.runtime_adapt_bandit_auto_explore) is True
    assert float(args.runtime_adapt_bandit_explore_min) == 0.25
    assert float(args.runtime_adapt_bandit_explore_max) == 1.4
    assert float(args.runtime_adapt_bandit_explore_step_up) == 0.05
    assert float(args.runtime_adapt_bandit_explore_step_down) == 0.03
    assert bool(args.runtime_adapt_bandit_contextual_bias) is True
    assert float(args.runtime_adapt_bandit_context_pressure_weight) == 0.42
    assert float(args.runtime_adapt_bandit_context_score_weight) == 0.28
    assert float(args.runtime_adapt_bandit_context_bias_cap) == 0.18
    assert bool(args.runtime_adapt_bandit_context_cold_start) is True
    assert float(args.runtime_adapt_bandit_context_cold_start_gain) == 0.55
    assert bool(args.runtime_adapt_bandit_hardware_guard) is True
    assert bool(args.runtime_adapt_bandit_hardware_strict) is True
    assert float(args.runtime_adapt_bandit_hardware_mismatch_count_decay) == 0.5
    assert float(args.runtime_adapt_bandit_hardware_mismatch_value_decay) == 0.6
    assert float(args.runtime_adapt_bandit_hardware_mismatch_explore_boost) == 0.3
    assert bool(args.runtime_adapt_bandit_time_decay) is True
    assert float(args.runtime_adapt_bandit_time_decay_half_life_s) == 2400.0
    assert float(args.runtime_adapt_bandit_time_decay_min_factor) == 0.22
    assert float(args.runtime_adapt_bandit_time_decay_cooldown_s) == 30.0
    assert bool(args.runtime_adapt_bandit_arm_cooldown) is True
    assert float(args.runtime_adapt_bandit_arm_cooldown_reward_threshold) == -0.7
    assert int(args.runtime_adapt_bandit_arm_cooldown_trigger_streak) == 3
    assert float(args.runtime_adapt_bandit_arm_cooldown_s) == 50.0
    assert bool(args.runtime_adapt_bandit_all_cooled_safe_fallback) is True
    assert bool(args.runtime_adapt_bandit_arm_risk_aware) is True
    assert float(args.runtime_adapt_bandit_arm_risk_alpha) == 0.33
    assert float(args.runtime_adapt_bandit_arm_risk_weight) == 0.55
    assert float(args.runtime_adapt_bandit_arm_risk_neg_weight) == 0.44
    assert float(args.runtime_adapt_bandit_arm_risk_cap) == 0.66
    assert bool(args.runtime_adapt_bandit_safety_guard) is True
    assert float(args.runtime_adapt_bandit_safety_pressure_gate) == 0.91
    assert float(args.runtime_adapt_bandit_safety_score_gate) == 0.29
    assert float(args.runtime_adapt_bandit_safety_max_multiplier) == 1.0
    assert bool(args.runtime_adapt_bandit_uncertainty_guard) is True
    assert float(args.runtime_adapt_bandit_uncertainty_var_gate) == 0.31
    assert int(args.runtime_adapt_bandit_uncertainty_min_samples) == 7
    assert float(args.runtime_adapt_bandit_uncertainty_pressure_gate) == 0.58
    assert float(args.runtime_adapt_bandit_uncertainty_score_gate) == 0.39
    assert float(args.runtime_adapt_bandit_uncertainty_max_multiplier) == 0.99
    assert bool(args.runtime_adapt_bandit_regime_split) is True
    assert float(args.runtime_adapt_bandit_regime_pressure_gate) == 0.79
    assert float(args.runtime_adapt_bandit_regime_score_gate) == 0.43
    assert bool(args.runtime_adapt_bandit_shock_guard) is True
    assert float(args.runtime_adapt_bandit_shock_reward_threshold) == -0.88
    assert int(args.runtime_adapt_bandit_shock_trigger_count) == 3
    assert float(args.runtime_adapt_bandit_shock_window_s) == 28.0
    assert float(args.runtime_adapt_bandit_shock_cooldown_s) == 55.0
    assert float(args.runtime_adapt_bandit_shock_max_multiplier) == 0.95
    assert bool(args.runtime_adapt_bandit_spike_guard) is True
    assert float(args.runtime_adapt_bandit_spike_pressure_delta_threshold) == 0.19
    assert float(args.runtime_adapt_bandit_spike_score_drop_threshold) == 0.23
    assert float(args.runtime_adapt_bandit_spike_min_pressure) == 0.57
    assert float(args.runtime_adapt_bandit_spike_cooldown_s) == 38.0
    assert float(args.runtime_adapt_bandit_spike_max_multiplier) == 0.98
    assert str(args.runtime_adapt_state_path) == "artifacts/audit/runtime_state_test.json"
    assert bool(args.runtime_adapt_load_state) is True
    assert float(args.runtime_adapt_save_interval_s) == 13.0
    assert bool(args.runtime_adapt_dynamic_interval) is True
    assert float(args.runtime_adapt_interval_min_s) == 2.5
    assert float(args.runtime_adapt_interval_max_s) == 21.0
    assert float(args.runtime_adapt_interval_high_mul) == 0.6
    assert float(args.runtime_adapt_interval_low_mul) == 1.4
    assert float(args.runtime_adapt_interval_throttle_mul) == 0.5
    assert bool(args.runtime_adapt_threshold_auto) is True
    assert float(args.runtime_adapt_threshold_min_high) == 0.69
    assert float(args.runtime_adapt_threshold_max_high) == 0.91
    assert float(args.runtime_adapt_threshold_gap) == 0.2
    assert float(args.runtime_adapt_threshold_step_up) == 0.004
    assert float(args.runtime_adapt_threshold_step_down) == 0.012
    assert bool(args.runtime_transformer_hot_tune) is True
    assert bool(args.runtime_planner_hot_tune) is True
    assert int(args.runtime_planner_depth_min) == 2
    assert int(args.runtime_planner_depth_max) == 7
    assert int(args.runtime_planner_beam_min) == 2
    assert int(args.runtime_planner_beam_max) == 11

    args2 = parse_args(
        [
            "--runtime-adapt-no-load-state",
            "--runtime-adapt-no-dynamic-interval",
            "--runtime-adapt-no-threshold-auto",
            "--runtime-adapt-bandit-no-auto-explore",
            "--runtime-adapt-bandit-no-contextual-bias",
            "--runtime-adapt-bandit-no-context-cold-start",
            "--runtime-adapt-bandit-no-hardware-guard",
            "--runtime-adapt-bandit-no-hardware-strict",
            "--runtime-adapt-bandit-no-time-decay",
            "--runtime-adapt-bandit-no-arm-cooldown",
            "--runtime-adapt-bandit-no-all-cooled-safe-fallback",
            "--runtime-adapt-bandit-no-arm-risk-aware",
            "--runtime-adapt-bandit-no-safety-guard",
            "--runtime-adapt-bandit-no-uncertainty-guard",
            "--runtime-adapt-bandit-no-regime-split",
            "--runtime-adapt-bandit-no-shock-guard",
            "--runtime-adapt-bandit-no-spike-guard",
        ]
    )
    assert bool(args2.runtime_adapt_load_state) is False
    assert bool(args2.runtime_adapt_dynamic_interval) is False
    assert bool(args2.runtime_adapt_threshold_auto) is False
    assert bool(args2.runtime_adapt_bandit_auto_explore) is False
    assert bool(args2.runtime_adapt_bandit_contextual_bias) is False
    assert bool(args2.runtime_adapt_bandit_context_cold_start) is False
    assert bool(args2.runtime_adapt_bandit_hardware_guard) is False
    assert bool(args2.runtime_adapt_bandit_hardware_strict) is False
    assert bool(args2.runtime_adapt_bandit_time_decay) is False
    assert bool(args2.runtime_adapt_bandit_arm_cooldown) is False
    assert bool(args2.runtime_adapt_bandit_all_cooled_safe_fallback) is False
    assert bool(args2.runtime_adapt_bandit_arm_risk_aware) is False
    assert bool(args2.runtime_adapt_bandit_safety_guard) is False
    assert bool(args2.runtime_adapt_bandit_uncertainty_guard) is False
    assert bool(args2.runtime_adapt_bandit_regime_split) is False
    assert bool(args2.runtime_adapt_bandit_shock_guard) is False
    assert bool(args2.runtime_adapt_bandit_spike_guard) is False


def test_federated_passthrough_requires_separator() -> None:
    _assert_parse_fails(["--federated-train", "--fed-rounds", "3"])


def test_federated_passthrough_after_separator_is_preserved() -> None:
    args = parse_args(["--federated-train", "--", "--fed-rounds", "3", "--fed-num-clients", "2"])
    assert bool(args.federated_train) is True
    assert list(args.federated_forward_args) == ["--fed-rounds", "3", "--fed-num-clients", "2"]


def test_unknown_args_not_silenced_by_federated_env() -> None:
    prev = os.environ.get("FEDERATED_AUTO_TRAIN")
    os.environ["FEDERATED_AUTO_TRAIN"] = "1"
    try:
        _assert_parse_fails(["--not-a-real-flag"])
    finally:
        if prev is None:
            os.environ.pop("FEDERATED_AUTO_TRAIN", None)
        else:
            os.environ["FEDERATED_AUTO_TRAIN"] = prev


def test_cli_help_exposes_domain_groups() -> None:
    help_text = build_parser().format_help()
    for title in ("chat:", "computer_use:", "automation:", "artifact:", "federated:"):
        assert title in help_text


def main() -> None:
    test_runtime_adapt_cli_flags_parse()
    test_federated_passthrough_requires_separator()
    test_federated_passthrough_after_separator_is_preserved()
    test_unknown_args_not_silenced_by_federated_env()
    test_cli_help_exposes_domain_groups()
    print("cli_runtime_adapt_ok")


if __name__ == "__main__":
    main()
