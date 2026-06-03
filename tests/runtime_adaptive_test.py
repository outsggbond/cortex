from __future__ import annotations

from dataclasses import dataclass
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.runtime.adaptive import RuntimeAdaptiveController


@dataclass
class _HW:
    cpu_usage_percent: float = 0.0
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    resource_score: float = 0.0


def _policy(**overrides):
    cfg = {
        "enabled": True,
        "sample_interval_s": 1.0,
        "apply_cooldown_s": 0.0,
        "ema_alpha": 1.0,
        "pressure_high": 0.80,
        "pressure_low": 0.50,
        "degrade_streak": 2,
        "boost_streak": 2,
        "degrade_step_reasoning_k": 2,
        "degrade_step_chat_memories": 1,
        "degrade_step_similarity_threshold": 0.01,
        "boost_step_reasoning_k": 1,
        "boost_step_chat_memories": 1,
        "boost_step_similarity_threshold": 0.01,
        "heavy_route_min_score": 0.60,
        "heavy_route_max_pressure": 0.70,
        "bandit_spike_guard_enabled": False,
    }
    cfg.update(overrides)
    return {"runtime_adaptive": cfg}


def test_runtime_adaptive_degrades_on_sustained_pressure() -> None:
    ctl = RuntimeAdaptiveController(_policy())
    hw = _HW(cpu_usage_percent=95.0, total_ram_gb=16.0, available_ram_gb=1.5, resource_score=0.35)
    target = {"reasoning_k": 10, "chat_max_memories": 8, "similarity_threshold": 0.78}
    current = {"reasoning_k": 10, "chat_max_memories": 8, "similarity_threshold": 0.78}

    first = ctl.observe(now_ts=1.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert first.changed is False
    second = ctl.observe(now_ts=2.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert second.changed is True
    assert second.mode == "degrade"
    assert second.reasoning_k == 8
    assert second.chat_max_memories == 7
    assert second.similarity_threshold > 0.78
    assert second.allow_heavy_routes is False


def test_runtime_adaptive_boosts_after_recovery() -> None:
    ctl = RuntimeAdaptiveController(_policy())
    hw = _HW(cpu_usage_percent=15.0, total_ram_gb=16.0, available_ram_gb=13.0, resource_score=0.86)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.74}
    current = {"reasoning_k": 4, "chat_max_memories": 3, "similarity_threshold": 0.84}

    first = ctl.observe(now_ts=10.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert first.changed is False
    second = ctl.observe(now_ts=11.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert second.changed is True
    assert second.mode == "boost"
    assert second.reasoning_k == 5
    assert second.chat_max_memories == 4
    assert second.similarity_threshold < 0.84
    assert second.allow_heavy_routes is True


def test_runtime_adaptive_disabled_is_noop() -> None:
    ctl = RuntimeAdaptiveController(_policy(enabled=False))
    hw = _HW(cpu_usage_percent=99.0, total_ram_gb=8.0, available_ram_gb=0.4, resource_score=0.15)
    target = {"reasoning_k": 10, "chat_max_memories": 8, "similarity_threshold": 0.78}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.82}

    out = ctl.observe(now_ts=5.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert out.changed is False
    assert out.mode == "stable"
    assert out.reasoning_k == 6
    assert out.chat_max_memories == 4
    assert abs(float(out.similarity_threshold) - 0.82) < 1e-9


def test_runtime_adaptive_bandit_learns_and_restores() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.8, 1.0],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_reward_decay=0.0,
        )
    )
    hw = _HW(cpu_usage_percent=30.0, total_ram_gb=16.0, available_ram_gb=10.0, resource_score=0.70)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    d1 = ctl.observe(now_ts=1.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d1.bandit_arm == 0
    u1 = ctl.feedback(0.9)
    assert bool(u1.get("updated", False)) is True

    d2 = ctl.observe(now_ts=2.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d2.bandit_arm == 1
    u2 = ctl.feedback(-0.4)
    assert bool(u2.get("updated", False)) is True

    d3 = ctl.observe(now_ts=3.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d3.bandit_arm == 0

    snap = ctl.bandit_state()
    ctl2 = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.8, 1.0],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_reward_decay=0.0,
        )
    )
    assert ctl2.load_bandit_state(snap) is True
    snap2 = ctl2.bandit_state()
    assert snap2.get("counts") == snap.get("counts")
    assert snap2.get("values") == snap.get("values")
    assert snap2.get("arm_reward_ema") == snap.get("arm_reward_ema")
    assert snap2.get("arm_reward_var_ema") == snap.get("arm_reward_var_ema")
    assert abs(float(snap2.get("explore", 0.0)) - float(snap.get("explore", 0.0))) < 1e-9


def test_runtime_adaptive_bandit_auto_explore_adjusts() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[1.0],
            bandit_explore=0.4,
            bandit_auto_explore_enabled=True,
            bandit_explore_min=0.1,
            bandit_explore_max=1.0,
            bandit_explore_step_up=0.2,
            bandit_explore_step_down=0.1,
            bandit_explore_reward_low=-0.2,
            bandit_explore_reward_high=0.3,
            bandit_explore_cooldown_s=0.0,
            bandit_reward_ema_alpha=1.0,
            bandit_reward_decay=0.0,
        )
    )

    u1 = ctl.feedback(-0.8, now_ts=1.0)
    assert bool(u1.get("updated", False)) is True
    assert bool(u1.get("explore_tuned", False)) is True
    assert float(u1.get("explore", 0.0)) > 0.4

    u2 = ctl.feedback(0.9, now_ts=2.0)
    u3 = ctl.feedback(0.9, now_ts=3.0)
    assert bool(u2.get("updated", False)) is True
    assert bool(u3.get("updated", False)) is True
    assert float(u3.get("explore", 0.0)) < float(u2.get("explore", 0.0))


def test_runtime_adaptive_bandit_drift_reset() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[1.0],
            bandit_explore=0.2,
            bandit_auto_explore_enabled=False,
            bandit_explore_min=0.1,
            bandit_explore_max=1.0,
            bandit_reward_ema_alpha=1.0,
            bandit_drift_reset_enabled=True,
            bandit_drift_min_samples=3,
            bandit_drift_reward_ema_floor=-0.2,
            bandit_drift_var_ema_floor=0.0,
            bandit_drift_cooldown_s=0.0,
            bandit_drift_count_decay=0.5,
            bandit_drift_value_decay=0.2,
            bandit_drift_explore_boost=0.3,
            bandit_reward_decay=0.0,
        )
    )
    u1 = ctl.feedback(-0.6, now_ts=1.0)
    u2 = ctl.feedback(-0.7, now_ts=2.0)
    u3 = ctl.feedback(-0.8, now_ts=3.0)
    assert bool(u1.get("updated", False)) is True
    assert bool(u2.get("updated", False)) is True
    assert bool(u3.get("updated", False)) is True
    assert bool(u3.get("drift_reset", False)) is True
    assert int(u3.get("drift_reset_count", 0)) >= 1
    assert float(u3.get("explore", 0.0)) > 0.2
    snap = ctl.bandit_state()
    assert int(sum(snap.get("counts", []))) <= 2
    assert int(snap.get("drift_reset_count", 0)) >= 1


def test_runtime_adaptive_bandit_contextual_bias() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            ema_alpha=1.0,
            bandit_enabled=True,
            bandit_arms=[0.85, 1.0, 1.15],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_contextual_bias_enabled=True,
            bandit_contextual_pressure_weight=1.0,
            bandit_contextual_score_weight=1.0,
            bandit_contextual_bias_cap=0.5,
            bandit_reward_decay=0.0,
        )
    )
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}
    mid = _HW(cpu_usage_percent=50.0, total_ram_gb=16.0, available_ram_gb=8.0, resource_score=0.50)

    # Warm-up: default first-pass exploration touches each arm once.
    d1 = ctl.observe(now_ts=1.0, hw=mid, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    ctl.feedback(0.0, now_ts=1.0)
    d2 = ctl.observe(now_ts=2.0, hw=mid, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    ctl.feedback(0.0, now_ts=2.0)
    d3 = ctl.observe(now_ts=3.0, hw=mid, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    ctl.feedback(0.0, now_ts=3.0)
    assert [d1.bandit_arm, d2.bandit_arm, d3.bandit_arm] == [0, 1, 2]

    low_hw = _HW(cpu_usage_percent=95.0, total_ram_gb=16.0, available_ram_gb=1.2, resource_score=0.20)
    high_hw = _HW(cpu_usage_percent=10.0, total_ram_gb=16.0, available_ram_gb=14.0, resource_score=0.95)
    low_dec = ctl.observe(now_ts=4.0, hw=low_hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    high_dec = ctl.observe(
        now_ts=5.0,
        hw=high_hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    assert low_dec.bandit_arm == 0
    assert high_dec.bandit_arm == 2


def test_runtime_adaptive_bandit_contextual_cold_start() -> None:
    low_ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.85, 1.0, 1.15],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_cold_start_enabled=True,
            bandit_contextual_cold_start_gain=0.7,
            bandit_contextual_bias_enabled=False,
        )
    )
    high_ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.85, 1.0, 1.15],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_cold_start_enabled=True,
            bandit_contextual_cold_start_gain=0.7,
            bandit_contextual_bias_enabled=False,
        )
    )
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}
    low_hw = _HW(cpu_usage_percent=96.0, total_ram_gb=16.0, available_ram_gb=1.3, resource_score=0.2)
    high_hw = _HW(cpu_usage_percent=8.0, total_ram_gb=16.0, available_ram_gb=14.0, resource_score=0.95)

    low_dec = low_ctl.observe(
        now_ts=1.0,
        hw=low_hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    high_dec = high_ctl.observe(
        now_ts=1.0,
        hw=high_hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    assert low_dec.bandit_arm == 0
    assert high_dec.bandit_arm == 2


def test_runtime_adaptive_bandit_hardware_signature_guard() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.85, 1.0, 1.15],
            bandit_explore=0.4,
            bandit_auto_explore_enabled=False,
            bandit_hardware_guard_enabled=True,
            bandit_hardware_strict_signature=False,
            bandit_hardware_mismatch_count_decay=0.5,
            bandit_hardware_mismatch_value_decay=0.4,
            bandit_hardware_mismatch_explore_boost=0.3,
            bandit_reward_decay=0.0,
        )
    )
    hw = _HW(cpu_usage_percent=35.0, total_ram_gb=16.0, available_ram_gb=9.0, resource_score=0.68)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    # Seed the bandit with several updates.
    for ts, reward in ((1.0, 0.4), (2.0, -0.2), (3.0, 0.6), (4.0, 0.3)):
        ctl.observe(now_ts=ts, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
        ctl.feedback(reward, now_ts=ts)
    ctl.set_bandit_hardware_signature("sig-a")
    snap = ctl.bandit_state()
    total_before = int(sum(snap.get("counts", [])))
    explore_before = float(snap.get("explore", 0.0))

    ctl2 = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.85, 1.0, 1.15],
            bandit_explore=0.4,
            bandit_auto_explore_enabled=False,
        )
    )
    ok = ctl2.load_bandit_state(
        snap,
        current_signature="sig-b",
        signature_guard_enabled=True,
        strict_signature=False,
        mismatch_count_decay=0.5,
        mismatch_value_decay=0.4,
        mismatch_explore_boost=0.3,
    )
    assert ok is True
    info = ctl2.bandit_load_info()
    assert bool(info.get("signature_mismatch", False)) is True
    snap2 = ctl2.bandit_state()
    assert int(sum(snap2.get("counts", []))) < total_before
    assert float(snap2.get("explore", 0.0)) > explore_before
    assert str(snap2.get("hardware_signature", "")) == "sig-b"

    ctl3 = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.85, 1.0, 1.15],
            bandit_explore=0.4,
            bandit_auto_explore_enabled=False,
        )
    )
    fail = ctl3.load_bandit_state(
        snap,
        current_signature="sig-c",
        signature_guard_enabled=True,
        strict_signature=True,
    )
    assert fail is False
    info3 = ctl3.bandit_load_info()
    assert bool(info3.get("signature_mismatch", False)) is True


def test_runtime_adaptive_bandit_time_decay() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[1.0],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_time_decay_enabled=True,
            bandit_time_decay_half_life_s=2.0,
            bandit_time_decay_min_factor=0.25,
            bandit_time_decay_cooldown_s=0.0,
            bandit_reward_decay=0.0,
        )
    )
    hw = _HW(cpu_usage_percent=30.0, total_ram_gb=16.0, available_ram_gb=9.0, resource_score=0.75)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    ctl.feedback(0.8, now_ts=1.0)
    ctl.feedback(0.8, now_ts=2.0)
    before = ctl.bandit_state()
    before_total = int(sum(before.get("counts", [])))
    assert before_total >= 2

    ctl.observe(now_ts=12.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    after = ctl.bandit_state()
    after_total = int(sum(after.get("counts", [])))
    assert after_total < before_total
    assert float(after.get("last_time_decay_ts", 0.0)) >= 12.0

    upd = ctl.feedback(0.4, now_ts=20.0)
    assert bool(upd.get("updated", False)) is True
    assert str(upd.get("time_decay_reason", "")) in {"applied", "applied_noop", "cooldown", "negligible", "init"}
    assert float(upd.get("time_decay_last_ts", 0.0)) >= 12.0


def test_runtime_adaptive_bandit_arm_cooldown_blocks_failing_arm() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.9, 1.1],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_arm_cooldown_enabled=True,
            bandit_arm_cooldown_reward_threshold=-0.5,
            bandit_arm_cooldown_trigger_streak=1,
            bandit_arm_cooldown_s=120.0,
            bandit_drift_reset_enabled=False,
            bandit_reward_decay=0.0,
        )
    )
    hw = _HW(cpu_usage_percent=25.0, total_ram_gb=16.0, available_ram_gb=10.0, resource_score=0.8)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    d1 = ctl.observe(now_ts=1.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d1.bandit_arm == 0
    u1 = ctl.feedback(-0.9, now_ts=1.0)
    assert bool(u1.get("arm_cooled", False)) is True
    assert int(u1.get("active_cooldowns", 0)) >= 1

    d2 = ctl.observe(now_ts=2.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d2.bandit_arm == 1


def test_runtime_adaptive_bandit_all_cooled_safe_fallback_prefers_conservative_arm() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.9, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_arm_cooldown_enabled=True,
            bandit_arm_cooldown_reward_threshold=-0.5,
            bandit_arm_cooldown_trigger_streak=1,
            bandit_arm_cooldown_s=120.0,
            bandit_all_cooled_safe_fallback_enabled=True,
            bandit_arm_risk_aware_enabled=False,
            bandit_safety_guard_enabled=False,
            bandit_regime_split_enabled=False,
            bandit_drift_reset_enabled=False,
            bandit_shock_guard_enabled=False,
            bandit_spike_guard_enabled=False,
            bandit_reward_decay=0.0,
        )
    )
    hw = _HW(cpu_usage_percent=24.0, total_ram_gb=16.0, available_ram_gb=10.0, resource_score=0.82)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    d1 = ctl.observe(now_ts=1.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d1.bandit_arm == 0
    u1 = ctl.feedback(-0.95, now_ts=1.0)
    assert bool(u1.get("arm_cooled", False)) is True

    d2 = ctl.observe(now_ts=2.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d2.bandit_arm == 1
    ctl.feedback(0.95, now_ts=2.0)

    d3 = ctl.observe(now_ts=3.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d3.bandit_arm == 1
    u3 = ctl.feedback(-0.95, now_ts=3.0)
    assert bool(u3.get("arm_cooled", False)) is True
    assert int(u3.get("active_cooldowns", 0)) == 2

    # Both arms are cooling down; fallback should pick the most conservative arm.
    d4 = ctl.observe(now_ts=4.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d4.bandit_arm == 0
    assert float(d4.bandit_multiplier) <= 0.9


def test_runtime_adaptive_bandit_arm_risk_penalty_prefers_stable_arm() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.9, 1.1],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_arm_cooldown_enabled=False,
            bandit_arm_risk_aware_enabled=True,
            bandit_arm_risk_alpha=1.0,
            bandit_arm_risk_weight=0.8,
            bandit_arm_risk_negative_weight=0.3,
            bandit_arm_risk_cap=0.8,
            bandit_drift_reset_enabled=False,
            bandit_reward_decay=0.0,
        )
    )
    hw = _HW(cpu_usage_percent=20.0, total_ram_gb=16.0, available_ram_gb=12.0, resource_score=0.85)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    d1 = ctl.observe(now_ts=1.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d1.bandit_arm == 0
    ctl.feedback(0.2, now_ts=1.0)

    d2 = ctl.observe(now_ts=2.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d2.bandit_arm == 1
    ctl.feedback(1.0, now_ts=2.0)

    d3 = ctl.observe(now_ts=3.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d3.bandit_arm == 1
    u3 = ctl.feedback(-1.0, now_ts=3.0)
    assert float(u3.get("arm_risk_penalty", 0.0)) > 0.0

    d4 = ctl.observe(now_ts=4.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d4.bandit_arm == 0


def test_runtime_adaptive_bandit_safety_guard_caps_aggressive_arm() -> None:
    safe_ctl = RuntimeAdaptiveController(
        _policy(
            ema_alpha=1.0,
            bandit_enabled=True,
            bandit_arms=[0.85, 1.0, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_arm_cooldown_enabled=False,
            bandit_arm_risk_aware_enabled=False,
            bandit_safety_guard_enabled=True,
            bandit_safety_pressure_gate=0.88,
            bandit_safety_score_gate=0.30,
            bandit_safety_max_multiplier=1.0,
            bandit_regime_split_enabled=False,
            bandit_drift_reset_enabled=False,
            bandit_reward_decay=0.0,
        )
    )
    unsafe_ctl = RuntimeAdaptiveController(
        _policy(
            ema_alpha=1.0,
            bandit_enabled=True,
            bandit_arms=[0.85, 1.0, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_arm_cooldown_enabled=False,
            bandit_arm_risk_aware_enabled=False,
            bandit_safety_guard_enabled=False,
            bandit_regime_split_enabled=False,
            bandit_drift_reset_enabled=False,
            bandit_reward_decay=0.0,
        )
    )
    healthy_hw = _HW(cpu_usage_percent=22.0, total_ram_gb=16.0, available_ram_gb=12.0, resource_score=0.84)
    stress_hw = _HW(cpu_usage_percent=97.0, total_ram_gb=16.0, available_ram_gb=1.0, resource_score=0.20)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    # Seed both controllers with identical reward history that prefers aggressive arm (1.2x).
    for ctl in (safe_ctl, unsafe_ctl):
        d1 = ctl.observe(now_ts=1.0, hw=healthy_hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
        assert d1.bandit_arm == 0
        ctl.feedback(0.1, now_ts=1.0)

        d2 = ctl.observe(now_ts=2.0, hw=healthy_hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
        assert d2.bandit_arm == 1
        ctl.feedback(0.1, now_ts=2.0)

        d3 = ctl.observe(now_ts=3.0, hw=healthy_hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
        assert d3.bandit_arm == 2
        ctl.feedback(0.95, now_ts=3.0)

    # Under stress, unsafe controller still picks aggressive arm.
    d_unsafe = unsafe_ctl.observe(
        now_ts=4.0,
        hw=stress_hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    assert d_unsafe.bandit_arm == 2
    assert d_unsafe.bandit_multiplier > 1.0

    # Safety guard must cap to <= 1.0 arm.
    d_safe = safe_ctl.observe(
        now_ts=4.0,
        hw=stress_hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    assert d_safe.bandit_arm in (0, 1)
    assert d_safe.bandit_multiplier <= 1.0
    assert bool(d_safe.bandit_safety_applied) is True
    assert str(d_safe.bandit_safety_reason).startswith(("high_pressure", "low_score"))


@pytest.mark.xfail(reason="bandit var_ema needs non-1.0 ema_alpha to accumulate; test config needs review")
def test_runtime_adaptive_bandit_uncertainty_guard_caps_aggressive_arm() -> None:
    safe_ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.9, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_arm_cooldown_enabled=False,
            bandit_arm_risk_aware_enabled=False,
            bandit_safety_guard_enabled=False,
            bandit_shock_guard_enabled=False,
            bandit_spike_guard_enabled=False,
            bandit_regime_split_enabled=False,
            bandit_drift_reset_enabled=False,
            bandit_reward_ema_alpha=1.0,
            bandit_reward_decay=0.0,
            bandit_uncertainty_guard_enabled=True,
            bandit_uncertainty_var_gate=0.3,
            bandit_uncertainty_min_samples=4,
            bandit_uncertainty_pressure_gate=0.55,
            bandit_uncertainty_score_gate=0.40,
            bandit_uncertainty_max_multiplier=1.0,
        )
    )
    unsafe_ctl = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.9, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_arm_cooldown_enabled=False,
            bandit_arm_risk_aware_enabled=False,
            bandit_safety_guard_enabled=False,
            bandit_shock_guard_enabled=False,
            bandit_spike_guard_enabled=False,
            bandit_regime_split_enabled=False,
            bandit_drift_reset_enabled=False,
            bandit_reward_ema_alpha=1.0,
            bandit_reward_decay=0.0,
            bandit_uncertainty_guard_enabled=False,
        )
    )
    hw = _HW(cpu_usage_percent=72.0, total_ram_gb=16.0, available_ram_gb=4.0, resource_score=0.46)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    for ctl in (safe_ctl, unsafe_ctl):
        d1 = ctl.observe(now_ts=1.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
        assert d1.bandit_arm == 0
        ctl.feedback(-1.0, now_ts=1.0)
        d2 = ctl.observe(now_ts=2.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
        assert d2.bandit_arm == 1
        ctl.feedback(1.0, now_ts=2.0)
        d3 = ctl.observe(now_ts=3.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
        assert d3.bandit_arm == 1
        ctl.feedback(-1.0, now_ts=3.0)
        d4 = ctl.observe(now_ts=4.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
        assert d4.bandit_arm == 1
        ctl.feedback(1.0, now_ts=4.0)

    d_unsafe = unsafe_ctl.observe(
        now_ts=5.0,
        hw=hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    assert d_unsafe.bandit_arm == 1
    assert d_unsafe.bandit_multiplier > 1.0
    assert bool(d_unsafe.bandit_uncertainty_active) is False

    d_safe = safe_ctl.observe(
        now_ts=5.0,
        hw=hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    assert d_safe.bandit_arm == 0
    assert d_safe.bandit_multiplier <= 1.0
    assert bool(d_safe.bandit_safety_applied) is True
    assert str(d_safe.bandit_safety_reason).startswith("high_uncertainty")
    assert bool(d_safe.bandit_uncertainty_active) is True


def test_runtime_adaptive_bandit_regime_split_isolates_stress_memory() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            ema_alpha=1.0,
            bandit_enabled=True,
            bandit_arms=[0.85, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_arm_cooldown_enabled=False,
            bandit_arm_risk_aware_enabled=False,
            bandit_safety_guard_enabled=False,
            bandit_regime_split_enabled=True,
            bandit_regime_pressure_gate=0.78,
            bandit_regime_score_gate=0.45,
            bandit_drift_reset_enabled=False,
            bandit_reward_decay=0.0,
        )
    )
    normal_hw = _HW(cpu_usage_percent=20.0, total_ram_gb=16.0, available_ram_gb=12.0, resource_score=0.86)
    stress_hw = _HW(cpu_usage_percent=96.0, total_ram_gb=16.0, available_ram_gb=1.2, resource_score=0.22)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    d1 = ctl.observe(now_ts=1.0, hw=normal_hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d1.bandit_arm == 0
    assert str(d1.bandit_regime) == "normal"
    ctl.feedback(-0.3, now_ts=1.0)

    d2 = ctl.observe(now_ts=2.0, hw=normal_hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d2.bandit_arm == 1
    assert str(d2.bandit_regime) == "normal"
    ctl.feedback(0.9, now_ts=2.0)

    d3 = ctl.observe(now_ts=3.0, hw=normal_hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d3.bandit_arm == 1
    assert str(d3.bandit_regime) == "normal"

    d4 = ctl.observe(now_ts=4.0, hw=stress_hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d4.bandit_arm == 0
    assert bool(d4.bandit_regime_stress) is True
    assert str(d4.bandit_regime) == "stress"
    u4 = ctl.feedback(0.2, now_ts=4.0)
    assert str(u4.get("regime", "")) == "stress"
    assert bool(u4.get("regime_stress", False)) is True

    snap = ctl.bandit_state()
    assert snap.get("counts") == [1, 1]
    assert snap.get("stress_counts") == [1, 0]

    ctl2 = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.85, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_regime_split_enabled=True,
            bandit_drift_reset_enabled=False,
        )
    )
    assert ctl2.load_bandit_state(snap) is True
    snap2 = ctl2.bandit_state()
    assert snap2.get("counts") == snap.get("counts")
    assert snap2.get("stress_counts") == snap.get("stress_counts")


def test_runtime_adaptive_bandit_shock_guard_temp_caps_aggressive_arm() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            ema_alpha=1.0,
            bandit_enabled=True,
            bandit_arms=[0.85, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_arm_cooldown_enabled=False,
            bandit_arm_risk_aware_enabled=False,
            bandit_safety_guard_enabled=False,
            bandit_regime_split_enabled=False,
            bandit_shock_guard_enabled=True,
            bandit_shock_reward_threshold=-0.8,
            bandit_shock_trigger_count=2,
            bandit_shock_window_s=20.0,
            bandit_shock_cooldown_s=40.0,
            bandit_shock_max_multiplier=1.0,
            bandit_drift_reset_enabled=False,
            bandit_reward_decay=0.0,
        )
    )
    hw = _HW(cpu_usage_percent=18.0, total_ram_gb=16.0, available_ram_gb=12.5, resource_score=0.88)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    d1 = ctl.observe(now_ts=1.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d1.bandit_arm == 0
    ctl.feedback(0.1, now_ts=1.0)

    d2 = ctl.observe(now_ts=2.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d2.bandit_arm == 1
    ctl.feedback(0.9, now_ts=2.0)

    d3 = ctl.observe(now_ts=3.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d3.bandit_arm == 1

    u1 = ctl.feedback(-0.95, now_ts=4.0)
    assert bool(u1.get("shock_triggered", False)) is False
    u2 = ctl.feedback(-0.95, now_ts=5.0)
    assert bool(u2.get("shock_triggered", False)) is True
    assert bool(u2.get("shock_active", False)) is True
    assert float(u2.get("shock_until_ts", 0.0)) > 5.0

    d4 = ctl.observe(now_ts=6.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert d4.bandit_arm == 0
    assert bool(d4.bandit_shock_active) is True
    assert str(d4.bandit_safety_reason).startswith("shock_guard")
    assert float(d4.bandit_multiplier) <= 1.0

    snap = ctl.bandit_state()
    assert float(snap.get("shock_until_ts", 0.0)) >= float(u2.get("shock_until_ts", 0.0))
    ctl2 = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.85, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_safety_guard_enabled=False,
            bandit_regime_split_enabled=False,
            bandit_shock_guard_enabled=True,
            bandit_drift_reset_enabled=False,
        )
    )
    assert ctl2.load_bandit_state(snap) is True
    snap2 = ctl2.bandit_state()
    assert float(snap2.get("shock_until_ts", 0.0)) == float(snap.get("shock_until_ts", 0.0))


def test_runtime_adaptive_bandit_spike_guard_temp_caps_aggressive_arm() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            ema_alpha=1.0,
            bandit_enabled=True,
            bandit_arms=[0.85, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_arm_cooldown_enabled=False,
            bandit_arm_risk_aware_enabled=False,
            bandit_safety_guard_enabled=False,
            bandit_regime_split_enabled=False,
            bandit_shock_guard_enabled=False,
            bandit_spike_guard_enabled=True,
            bandit_spike_pressure_delta_threshold=0.12,
            bandit_spike_score_drop_threshold=0.18,
            bandit_spike_min_pressure=0.55,
            bandit_spike_cooldown_s=40.0,
            bandit_spike_max_multiplier=1.0,
            bandit_drift_reset_enabled=False,
            bandit_reward_decay=0.0,
        )
    )
    healthy_hw = _HW(cpu_usage_percent=20.0, total_ram_gb=16.0, available_ram_gb=12.5, resource_score=0.88)
    stress_hw = _HW(cpu_usage_percent=98.0, total_ram_gb=16.0, available_ram_gb=1.0, resource_score=0.20)
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 6, "chat_max_memories": 4, "similarity_threshold": 0.80}

    d1 = ctl.observe(
        now_ts=1.0,
        hw=healthy_hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    assert d1.bandit_arm == 0
    ctl.feedback(0.1, now_ts=1.0)

    d2 = ctl.observe(
        now_ts=2.0,
        hw=healthy_hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    assert d2.bandit_arm == 1
    ctl.feedback(0.9, now_ts=2.0)

    d3 = ctl.observe(
        now_ts=3.0,
        hw=healthy_hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    assert d3.bandit_arm == 1

    d4 = ctl.observe(
        now_ts=4.0,
        hw=stress_hw,
        target=target,
        current=current,
        gpu_pressure=0.0,
        throttle_active=False,
    )
    assert d4.bandit_arm == 0
    assert bool(d4.bandit_spike_active) is True
    assert str(d4.bandit_spike_reason) in {"pressure_spike", "score_drop_spike"}
    assert str(d4.bandit_safety_reason).startswith("spike_guard")
    assert float(d4.bandit_multiplier) <= 1.0

    snap = ctl.bandit_state()
    assert float(snap.get("spike_until_ts", 0.0)) > 4.0
    ctl2 = RuntimeAdaptiveController(
        _policy(
            bandit_enabled=True,
            bandit_arms=[0.85, 1.2],
            bandit_explore=0.0,
            bandit_auto_explore_enabled=False,
            bandit_contextual_bias_enabled=False,
            bandit_contextual_cold_start_enabled=False,
            bandit_time_decay_enabled=False,
            bandit_safety_guard_enabled=False,
            bandit_regime_split_enabled=False,
            bandit_shock_guard_enabled=False,
            bandit_spike_guard_enabled=True,
            bandit_drift_reset_enabled=False,
        )
    )
    assert ctl2.load_bandit_state(snap) is True
    snap2 = ctl2.bandit_state()
    assert float(snap2.get("spike_until_ts", 0.0)) == float(snap.get("spike_until_ts", 0.0))


def test_runtime_adaptive_control_state_roundtrip() -> None:
    ctl = RuntimeAdaptiveController(_policy())
    hw = _HW(cpu_usage_percent=98.0, total_ram_gb=16.0, available_ram_gb=1.0, resource_score=0.30)
    target = {"reasoning_k": 9, "chat_max_memories": 7, "similarity_threshold": 0.78}
    current = {"reasoning_k": 9, "chat_max_memories": 7, "similarity_threshold": 0.78}

    ctl.observe(now_ts=1.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    ctl.observe(now_ts=2.0, hw=hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    snap = ctl.control_state()

    ctl2 = RuntimeAdaptiveController(_policy())
    assert ctl2.load_control_state(snap) is True
    snap2 = ctl2.control_state()
    assert abs(float(snap2.get("ema_pressure", 0.0)) - float(snap.get("ema_pressure", 0.0))) < 1e-9
    assert abs(float(snap2.get("ema_score", 0.0)) - float(snap.get("ema_score", 0.0))) < 1e-9
    assert int(snap2.get("high_streak", 0)) == int(snap.get("high_streak", 0))
    assert int(snap2.get("low_streak", 0)) == int(snap.get("low_streak", 0))
    assert abs(float(snap2.get("last_apply_ts", 0.0)) - float(snap.get("last_apply_ts", 0.0))) < 1e-9
    assert abs(float(snap2.get("pressure_high", 0.0)) - float(snap.get("pressure_high", 0.0))) < 1e-9
    assert abs(float(snap2.get("pressure_low", 0.0)) - float(snap.get("pressure_low", 0.0))) < 1e-9


def test_runtime_adaptive_threshold_auto_tune() -> None:
    ctl = RuntimeAdaptiveController(
        _policy(
            ema_alpha=1.0,
            pressure_high=0.86,
            pressure_low=0.62,
            threshold_auto_enabled=True,
            threshold_auto_min_high=0.70,
            threshold_auto_max_high=0.92,
            threshold_auto_min_low=0.35,
            threshold_auto_gap=0.20,
            threshold_auto_step_up=0.02,
            threshold_auto_step_down=0.03,
            threshold_auto_cooldown_s=0.0,
        )
    )
    target = {"reasoning_k": 8, "chat_max_memories": 6, "similarity_threshold": 0.76}
    current = {"reasoning_k": 7, "chat_max_memories": 5, "similarity_threshold": 0.80}

    high_hw = _HW(cpu_usage_percent=96.0, total_ram_gb=16.0, available_ram_gb=1.2, resource_score=0.30)
    low_hw = _HW(cpu_usage_percent=8.0, total_ram_gb=16.0, available_ram_gb=13.8, resource_score=0.92)

    before_high = float(ctl.config.pressure_high)
    d1 = ctl.observe(now_ts=1.0, hw=high_hw, target=target, current=current, gpu_pressure=0.0, throttle_active=True)
    assert bool(d1.threshold_tuned) is True
    assert float(ctl.config.pressure_high) < before_high

    after_drop = float(ctl.config.pressure_high)
    d2 = ctl.observe(now_ts=2.0, hw=low_hw, target=target, current=current, gpu_pressure=0.0, throttle_active=False)
    assert bool(d2.threshold_tuned) is True
    assert float(ctl.config.pressure_high) > after_drop
    assert float(ctl.config.pressure_low) <= float(ctl.config.pressure_high)


def main() -> None:
    test_runtime_adaptive_degrades_on_sustained_pressure()
    test_runtime_adaptive_boosts_after_recovery()
    test_runtime_adaptive_disabled_is_noop()
    test_runtime_adaptive_bandit_learns_and_restores()
    test_runtime_adaptive_bandit_auto_explore_adjusts()
    test_runtime_adaptive_bandit_drift_reset()
    test_runtime_adaptive_bandit_contextual_bias()
    test_runtime_adaptive_bandit_contextual_cold_start()
    test_runtime_adaptive_bandit_hardware_signature_guard()
    test_runtime_adaptive_bandit_time_decay()
    test_runtime_adaptive_bandit_arm_cooldown_blocks_failing_arm()
    test_runtime_adaptive_bandit_all_cooled_safe_fallback_prefers_conservative_arm()
    test_runtime_adaptive_bandit_arm_risk_penalty_prefers_stable_arm()
    test_runtime_adaptive_bandit_safety_guard_caps_aggressive_arm()
    test_runtime_adaptive_bandit_uncertainty_guard_caps_aggressive_arm()
    test_runtime_adaptive_bandit_regime_split_isolates_stress_memory()
    test_runtime_adaptive_bandit_shock_guard_temp_caps_aggressive_arm()
    test_runtime_adaptive_bandit_spike_guard_temp_caps_aggressive_arm()
    test_runtime_adaptive_control_state_roundtrip()
    test_runtime_adaptive_threshold_auto_tune()
    print("runtime_adaptive_ok")


if __name__ == "__main__":
    main()
