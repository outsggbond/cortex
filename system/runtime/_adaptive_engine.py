from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from system.runtime._adaptive_guards import (
    _clamp,
    ControllerConfig,
    ControllerState,
    Guard,
    ShockGuard,
    SpikeGuard,
    ThrottleGuard,
    HighPressureGuard,
    LowScoreGuard,
    UncertaintyGuard,
)


# ---------------------------------------------------------------------------
# 纯逻辑引擎
# ---------------------------------------------------------------------------
class AdaptiveEngine:
    """无状态机：所有计算依赖 Config + State，结果直接修改 State"""

    def __init__(self, config: ControllerConfig, state: ControllerState):
        self.config = config
        self.state = state

    # ---------- 压力/分数 ----------
    @staticmethod
    def _pressure(hw: Any, gpu_pressure: float) -> tuple[float, float]:
        cpu = _clamp(getattr(hw, "cpu_usage_percent", 0.0) / 100.0, 0.0, 1.0)
        total_ram = max(0.0, getattr(hw, "total_ram_gb", 0.0))
        avail_ram = max(0.0, getattr(hw, "available_ram_gb", 0.0))
        ram = _clamp(1.0 - (avail_ram / total_ram), 0.0, 1.0) if total_ram > 0.0 else 0.0
        gpu = _clamp(gpu_pressure, 0.0, 1.0)
        pressure = max(cpu, ram, gpu)
        score = _clamp(getattr(hw, "resource_score", 0.0), 0.0, 1.0)
        if score <= 0.0:
            score = _clamp(1.0 - pressure, 0.0, 1.0)
        return pressure, score

    # ---------- 阈值自动校准 ----------
    def autocalibrate_thresholds(self, now_ts: float, throttle_active: bool) -> Dict[str, Any]:
        c = self.config
        s = self.state
        out = {
            "changed": False,
            "reason": "disabled",
            "pressure_high": c.pressure_high,
            "pressure_low": c.pressure_low,
        }
        if not c.threshold_auto_enabled:
            return out
        if now_ts - s.last_threshold_tune_ts < c.threshold_auto_cooldown_s:
            out["reason"] = "cooldown"
            return out

        prev_high = c.pressure_high
        prev_low = c.pressure_low
        next_high = prev_high
        reason = "steady"
        if throttle_active or s.ema_pressure >= prev_high:
            next_high = prev_high - c.threshold_auto_step_down
            reason = "high_pressure"
        elif s.ema_pressure <= prev_low and s.ema_score >= max(0.1, c.heavy_route_min_score * 0.7):
            next_high = prev_high + c.threshold_auto_step_up
            reason = "low_pressure"

        next_high = _clamp(next_high, c.threshold_auto_min_high, c.threshold_auto_max_high)
        low_cap = max(0.05, next_high - 0.03)
        low_floor = min(c.threshold_auto_min_low, low_cap)
        gap = max(0.03, min(0.6, c.threshold_auto_gap))
        next_low = _clamp(next_high - gap, low_floor, low_cap)
        changed = abs(next_high - prev_high) > 1e-9 or abs(next_low - prev_low) > 1e-9
        if changed:
            # 直接修改 config（因为是 pydantic 模型，字段可写）
            c.pressure_high = next_high
            c.pressure_low = next_low
            s.last_threshold_tune_ts = now_ts
        out.update(changed=changed, reason=reason, pressure_high=c.pressure_high, pressure_low=c.pressure_low)
        return out

    # ---------- 上下文偏置 / 风险 ----------
    def _contextual_arm_bonus(self, arm_multiplier: float, pressure: float, score: float) -> float:
        c = self.config
        if not c.bandit_contextual_bias_enabled:
            return 0.0
        headroom = c.bandit_contextual_score_weight * _clamp(score, 0.0, 1.0) - c.bandit_contextual_pressure_weight * _clamp(pressure, 0.0, 1.0)
        direction = arm_multiplier - 1.0
        if abs(direction) <= 1e-12:
            return 0.0
        return _clamp(direction * headroom, -c.bandit_contextual_bias_cap, c.bandit_contextual_bias_cap)

    def _contextual_target_multiplier(self, pressure: float, score: float) -> float:
        arms = self.config.bandit_arms
        if not arms:
            return 1.0
        low = min(arms)
        high = max(arms)
        headroom = _clamp(score - pressure, -1.0, 1.0)
        return _clamp(1.0 + headroom * self.config.bandit_contextual_cold_start_gain, low, high)

    def _arm_risk_penalty(self, arm: int) -> float:
        c = self.config
        s = self.state
        if not c.bandit_arm_risk_aware_enabled or len(c.bandit_arms) == 0:
            return 0.0
        idx = max(0, min(len(c.bandit_arms) - 1, arm))
        risk_var = math.sqrt(max(0.0, s.bandit_arm_reward_var_ema[idx]))
        risk_neg = max(0.0, -s.bandit_arm_reward_ema[idx])
        raw = c.bandit_arm_risk_weight * risk_var + c.bandit_arm_risk_negative_weight * risk_neg
        return _clamp(raw, 0.0, c.bandit_arm_risk_cap)

    # ---------- Bandit 辅助 ----------
    def _active_bandit_indices(self, now_ts: float) -> np.ndarray:
        s = self.state
        arms = self.config.bandit_arms
        if len(arms) == 0:
            return np.array([], dtype=np.int32)
        cooldown = s.bandit_cooldown_until_ts
        active = np.where(cooldown <= now_ts)[0]
        if len(active) > 0:
            return active
        if self.config.bandit_all_cooled_safe_fallback_enabled:
            safe_mul = min(arms)
            safe = np.where(np.array(arms) <= safe_mul + 1e-12)[0]
            if len(safe) > 0:
                return safe
        return np.arange(len(arms))

    def _is_stress_regime(self, pressure: float, score: float, throttle_active: bool) -> bool:
        c = self.config
        if not c.bandit_regime_split_enabled:
            return False
        if throttle_active:
            return True
        return pressure >= c.bandit_regime_pressure_gate or score <= c.bandit_regime_score_gate

    def _regime_arrays(self, stress: bool) -> Tuple[np.ndarray, np.ndarray]:
        s = self.state
        return (s.bandit_stress_counts, s.bandit_stress_values) if stress else (s.bandit_counts, s.bandit_values)

    # ---------- 时间衰减 ----------
    def apply_time_decay(self, now_ts: float) -> Dict[str, Any]:
        c = self.config
        s = self.state
        out = {
            "changed": False,
            "reason": "disabled",
            "factor": 1.0,
            "elapsed_s": 0.0,
            "last_decay_ts": s.last_bandit_time_decay_ts,
        }
        if not c.bandit_time_decay_enabled or len(c.bandit_arms) == 0:
            return out
        if s.last_bandit_time_decay_ts <= 0.0:
            s.last_bandit_time_decay_ts = now_ts
            out.update(reason="init", last_decay_ts=now_ts)
            return out
        elapsed = max(0.0, now_ts - s.last_bandit_time_decay_ts)
        out["elapsed_s"] = elapsed
        if elapsed < c.bandit_time_decay_cooldown_s:
            out["reason"] = "cooldown"
            return out
        half_life = max(1.0, c.bandit_time_decay_half_life_s)
        factor = _clamp(math.pow(2.0, -elapsed / half_life), c.bandit_time_decay_min_factor, 1.0)
        out["factor"] = factor
        if factor >= 0.999999:
            out["reason"] = "negligible"
            return out

        prev_counts = s.bandit_counts.copy()
        prev_values = s.bandit_values.copy()
        # 向量化衰减
        s.bandit_counts = np.maximum(0, np.rint(s.bandit_counts * factor).astype(np.int32))
        s.bandit_values *= factor
        s.bandit_stress_counts = np.maximum(0, np.rint(s.bandit_stress_counts * factor).astype(np.int32))
        s.bandit_stress_values *= factor
        s.bandit_bad_streak = np.maximum(0, np.rint(s.bandit_bad_streak * factor).astype(np.int32))
        s.bandit_arm_reward_ema *= factor
        s.bandit_arm_reward_var_ema = np.maximum(0.0, s.bandit_arm_reward_var_ema * factor)
        s.bandit_reward_ema = _clamp(s.bandit_reward_ema * factor, -1.0, 1.0)
        s.bandit_reward_var_ema = max(0.0, s.bandit_reward_var_ema * factor)
        s.last_bandit_time_decay_ts = now_ts
        out["last_decay_ts"] = now_ts
        out["changed"] = (
            not np.array_equal(prev_counts, s.bandit_counts)
            or not np.allclose(prev_values, s.bandit_values)
        )
        out["reason"] = "applied" if out["changed"] else "applied_noop"
        return out

    # ---------- Arm cooldown ----------
    def apply_arm_cooldown(self, arm: int, reward: float, now_ts: float) -> Dict[str, Any]:
        c = self.config
        s = self.state
        idx = max(0, min(len(c.bandit_arms) - 1, arm)) if c.bandit_arms else 0
        out = {
            "changed": False,
            "reason": "disabled",
            "arm": idx,
            "cooldown_until_ts": s.bandit_cooldown_until_ts[idx] if len(s.bandit_cooldown_until_ts) > 0 else 0.0,
            "bad_streak": s.bandit_bad_streak[idx] if len(s.bandit_bad_streak) > 0 else 0,
            "active_cooldowns": int(np.sum(s.bandit_cooldown_until_ts > now_ts)),
        }
        if not c.bandit_arm_cooldown_enabled or not c.bandit_arms:
            return out
        if c.bandit_arm_cooldown_s <= 0.0:
            out["reason"] = "cooldown_disabled"
            return out
        if reward <= c.bandit_arm_cooldown_reward_threshold:
            s.bandit_bad_streak[idx] += 1
            out["reason"] = "below_threshold"
        else:
            s.bandit_bad_streak[idx] = 0
            out["reason"] = "recovered"
        out["bad_streak"] = int(s.bandit_bad_streak[idx])
        if s.bandit_bad_streak[idx] >= c.bandit_arm_cooldown_trigger_streak:
            s.bandit_cooldown_until_ts[idx] = max(s.bandit_cooldown_until_ts[idx], now_ts + c.bandit_arm_cooldown_s)
            s.bandit_bad_streak[idx] = 0
            out["changed"] = True
            out["reason"] = "arm_cooled"
            out["cooldown_until_ts"] = float(s.bandit_cooldown_until_ts[idx])
            out["bad_streak"] = 0
        out["active_cooldowns"] = int(np.sum(s.bandit_cooldown_until_ts > now_ts))
        return out

    # ---------- Shock / Spike 更新 ----------
    def update_shock_guard(self, reward: float, now_ts: float) -> Dict[str, Any]:
        c = self.config
        s = self.state
        out = {
            "changed": False,
            "triggered": False,
            "active": now_ts < s.bandit_shock_until_ts,
            "reason": "disabled",
            "until_ts": s.bandit_shock_until_ts,
            "event_count": len(s.bandit_shock_events_ts),
        }
        if not c.bandit_shock_guard_enabled:
            s.bandit_shock_events_ts = []
            s.bandit_shock_until_ts = 0.0
            out.update(active=False, until_ts=0.0, event_count=0)
            return out
        window = max(0.0, c.bandit_shock_window_s)
        if window <= 0.0:
            s.bandit_shock_events_ts = []
        else:
            s.bandit_shock_events_ts = [ts for ts in s.bandit_shock_events_ts if now_ts - ts <= window]
        if reward <= c.bandit_shock_reward_threshold:
            s.bandit_shock_events_ts.append(now_ts)
            out["reason"] = "catastrophic_reward"
        else:
            out["reason"] = "reward_ok"
        if len(s.bandit_shock_events_ts) >= c.bandit_shock_trigger_count:
            s.bandit_shock_until_ts = max(s.bandit_shock_until_ts, now_ts + c.bandit_shock_cooldown_s)
            s.bandit_shock_events_ts = []
            out["changed"] = True
            out["triggered"] = True
            out["reason"] = "shock_triggered"
        out["active"] = now_ts < s.bandit_shock_until_ts
        out["until_ts"] = s.bandit_shock_until_ts
        out["event_count"] = len(s.bandit_shock_events_ts)
        return out

    def update_spike_guard(self, pressure: float, score: float, now_ts: float) -> Dict[str, Any]:
        c = self.config
        s = self.state
        out = {
            "changed": False,
            "triggered": False,
            "active": now_ts < s.bandit_spike_until_ts,
            "reason": "disabled",
            "until_ts": s.bandit_spike_until_ts,
            "pressure_jump": 0.0,
            "score_drop": 0.0,
        }
        if not c.bandit_spike_guard_enabled:
            s.bandit_spike_until_ts = 0.0
            out.update(active=False, until_ts=0.0)
            return out
        p = _clamp(pressure, 0.0, 1.0)
        base_p = p if s.ema_pressure <= 0.0 else _clamp(s.ema_pressure, 0.0, 1.0)
        base_s = score if s.ema_score <= 0.0 else _clamp(s.ema_score, 0.0, 1.0)
        pressure_jump = max(0.0, p - base_p)
        score_drop = max(0.0, base_s - score)
        out.update(pressure_jump=pressure_jump, score_drop=score_drop)
        triggered = False
        reason = "stable"
        if p < c.bandit_spike_min_pressure:
            reason = "below_min_pressure"
        elif pressure_jump >= c.bandit_spike_pressure_delta_threshold:
            reason = "pressure_spike"
            triggered = True
        elif score_drop >= c.bandit_spike_score_drop_threshold:
            reason = "score_drop_spike"
            triggered = True
        if triggered:
            s.bandit_spike_until_ts = max(s.bandit_spike_until_ts, now_ts + c.bandit_spike_cooldown_s)
            out["changed"] = True
            out["triggered"] = True
        out["reason"] = reason
        out["active"] = now_ts < s.bandit_spike_until_ts
        out["until_ts"] = s.bandit_spike_until_ts
        return out

    # ---------- 安全守卫组合 ----------
    def apply_safety_guard(
        self,
        arm: int,
        pressure: float,
        score: float,
        throttle_active: bool,
        now_ts: float,
    ) -> Dict[str, Any]:
        c = self.config
        s = self.state
        guards: List[Guard] = [
            ShockGuard(),
            SpikeGuard(),
            ThrottleGuard(),
            HighPressureGuard(),
            LowScoreGuard(),
            UncertaintyGuard(),
        ]
        triggered = None
        # 按优先级找到第一个触发的守卫（防御性最强）
        for guard in guards:
            result = guard.evaluate(c, s, pressure, score, throttle_active, now_ts)
            if result is not None:
                triggered = result
                break

        idx = max(0, min(len(c.bandit_arms) - 1, arm)) if c.bandit_arms else 0
        out = {
            "arm": idx,
            "changed": False,
            "triggered": False,
            "reason": "safe",
            "multiplier": c.bandit_arms[idx] if c.bandit_arms else 1.0,
        }
        if triggered is None:
            return out
        if not c.bandit_safety_guard_enabled and not (
            "shock" in triggered["reason"] or "spike" in triggered["reason"]
        ):
            return out

        out["triggered"] = True
        max_multiplier = triggered["max_multiplier"]
        active = self._active_bandit_indices(now_ts)
        if len(active) == 0:
            out["reason"] = triggered["reason"] + ":no_active"
            return out

        safe_candidates = np.array([i for i in active if c.bandit_arms[i] <= max_multiplier + 1e-12])
        if len(safe_candidates) == 0:
            safe_candidates = np.array(active)
            out["reason"] = triggered["reason"] + ":no_cap_match"
        else:
            out["reason"] = triggered["reason"]

        if idx in safe_candidates:
            out["reason"] += ":current_safe"
            return out

        # 在安全候选臂中选择最佳
        regime_stress = s.active_regime == "stress"
        counts, values = self._regime_arrays(regime_stress)
        safe_total = max(1, int(np.sum(counts[safe_candidates])))
        best_idx = int(safe_candidates[0])
        best_ucb = -1e18
        for i in safe_candidates:
            n = max(1, int(counts[i]))
            q = values[i]
            ucb = q + c.bandit_explore * math.sqrt(math.log(max(1, safe_total) + 1.0) / n)
            ucb += self._contextual_arm_bonus(c.bandit_arms[int(i)], pressure, score)
            ucb -= self._arm_risk_penalty(int(i))
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = int(i)
        out["arm"] = best_idx
        out["multiplier"] = c.bandit_arms[best_idx]
        out["changed"] = best_idx != idx
        return out

    # ---------- 选择臂 ----------
    def choose_bandit_arm(self, pressure: float, score: float, now_ts: float, stress_mode: bool) -> int:
        c = self.config
        s = self.state
        if not c.bandit_enabled or not c.bandit_arms:
            s.active_arm = 0
            s.active_regime = "normal"
            return 0
        counts, values = self._regime_arrays(stress_mode)
        s.active_regime = "stress" if stress_mode else "normal"
        active = self._active_bandit_indices(now_ts)
        untried = active[counts[active] <= 0]  # 向量化判断
        if len(untried) > 0:
            if c.bandit_contextual_cold_start_enabled:
                target = self._contextual_target_multiplier(pressure, score)
                best_idx = int(untried[np.argmin(np.abs(np.array(c.bandit_arms)[untried] - target))])
                s.active_arm = best_idx
                return best_idx
            s.active_arm = int(untried[0])
            return int(untried[0])

        total = max(1, int(np.sum(counts[active])))
        best_idx = int(active[0])
        best_ucb = -1e18
        for i in active:
            n = max(1, int(counts[i]))
            q = values[i]
            ucb = q + c.bandit_explore * math.sqrt(math.log(total + 1.0) / n)
            ucb += self._contextual_arm_bonus(c.bandit_arms[int(i)], pressure, score)
            ucb -= self._arm_risk_penalty(int(i))
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = int(i)
        s.active_arm = best_idx
        return best_idx

    # ---------- 反馈 ----------
    def feedback(self, reward: float, now_ts: float) -> Dict[str, Any]:
        c = self.config
        s = self.state
        if not c.bandit_enabled or not c.bandit_arms:
            return {"updated": False}

        time_decay = self.apply_time_decay(now_ts)
        regime_stress = s.active_regime == "stress"
        counts, values = self._regime_arrays(regime_stress)
        arm = max(0, min(len(c.bandit_arms) - 1, s.active_arm))
        r = _clamp(reward, -1.0, 1.0)

        # 更新计数与价值
        prev_count = int(counts[arm])
        prev_value = float(values[arm])
        if c.bandit_reward_decay > 0.0:
            prev_value *= (1.0 - c.bandit_reward_decay)
        new_count = prev_count + 1
        new_value = prev_value + (r - prev_value) / float(new_count)
        counts[arm] = new_count
        values[arm] = new_value

        # 臂级 EMA 风险
        arm_alpha = max(1e-12, c.bandit_arm_risk_alpha)
        prev_arm_ema = s.bandit_arm_reward_ema[arm]
        prev_arm_var = s.bandit_arm_reward_var_ema[arm]
        if prev_count <= 0:
            s.bandit_arm_reward_ema[arm] = r
            s.bandit_arm_reward_var_ema[arm] = 0.0
        else:
            diff = r - prev_arm_ema
            s.bandit_arm_reward_ema[arm] = _clamp((1.0 - arm_alpha) * prev_arm_ema + arm_alpha * r, -1.0, 1.0)
            s.bandit_arm_reward_var_ema[arm] = max(0.0, (1.0 - arm_alpha) * prev_arm_var + arm_alpha * diff ** 2)

        risk = self._arm_risk_penalty(arm)

        # 全局 EMA
        alpha = max(1e-12, c.bandit_reward_ema_alpha)
        total_samples = int(np.sum(s.bandit_counts) + np.sum(s.bandit_stress_counts))
        if total_samples <= 1:
            s.bandit_reward_ema = r
            s.bandit_reward_var_ema = 0.0
        else:
            diff = r - s.bandit_reward_ema
            s.bandit_reward_ema = (1.0 - alpha) * s.bandit_reward_ema + alpha * r
            s.bandit_reward_var_ema = (1.0 - alpha) * s.bandit_reward_var_ema + alpha * diff ** 2

        shock = self.update_shock_guard(r, now_ts)
        arm_cooldown = self.apply_arm_cooldown(arm, r, now_ts)
        tune = self._tune_bandit_explore(r, now_ts)
        drift = self._maybe_drift_reset(now_ts)

        return {
            "updated": True,
            "arm": arm,
            "regime": s.active_regime,
            "regime_stress": regime_stress,
            "multiplier": c.bandit_arms[arm],
            "reward": r,
            "count": new_count,
            "value": new_value,
            "explore": c.bandit_explore,
            "explore_tuned": tune["changed"],
            "explore_tune_reason": tune["reason"],
            "reward_ema": s.bandit_reward_ema,
            "reward_var_ema": s.bandit_reward_var_ema,
            "arm_reward_ema": float(s.bandit_arm_reward_ema[arm]),
            "arm_reward_var_ema": float(s.bandit_arm_reward_var_ema[arm]),
            "arm_risk_penalty": risk,
            "shock_active": shock["active"],
            "shock_triggered": shock["triggered"],
            "shock_reason": shock["reason"],
            "shock_until_ts": shock["until_ts"],
            "shock_event_count": shock["event_count"],
            "spike_active": now_ts < s.bandit_spike_until_ts,
            "spike_until_ts": s.bandit_spike_until_ts,
            "drift_reset": drift["changed"],
            "drift_reason": drift["reason"],
            "drift_reset_count": drift["reset_count"],
            "time_decay_applied": time_decay["changed"],
            "time_decay_reason": time_decay["reason"],
            "time_decay_factor": time_decay["factor"],
            "time_decay_elapsed_s": time_decay["elapsed_s"],
            "time_decay_last_ts": time_decay["last_decay_ts"],
            "arm_cooled": arm_cooldown["changed"],
            "arm_cooldown_reason": arm_cooldown["reason"],
            "arm_cooldown_until_ts": arm_cooldown["cooldown_until_ts"],
            "arm_bad_streak": arm_cooldown["bad_streak"],
            "active_cooldowns": arm_cooldown["active_cooldowns"],
        }

    def _tune_bandit_explore(self, reward: float, now_ts: float) -> Dict[str, Any]:
        c = self.config
        s = self.state
        out = {
            "changed": False,
            "explore": c.bandit_explore,
            "reward_ema": s.bandit_reward_ema,
            "reward_var_ema": s.bandit_reward_var_ema,
            "reason": "disabled",
        }
        if not c.bandit_auto_explore_enabled:
            return out
        if now_ts - s.last_bandit_explore_tune_ts < c.bandit_explore_cooldown_s:
            out["reason"] = "cooldown"
            return out
        prev = c.bandit_explore
        nxt = prev
        reason = "steady"
        if reward <= c.bandit_explore_reward_low:
            nxt = prev + c.bandit_explore_step_up
            reason = "reward_low"
        elif reward >= c.bandit_explore_reward_high and s.bandit_reward_var_ema <= 0.12:
            nxt = prev - c.bandit_explore_step_down
            reason = "reward_high_stable"
        elif s.bandit_reward_var_ema > 0.35:
            nxt = prev + 0.5 * c.bandit_explore_step_up
            reason = "reward_volatile"
        nxt = _clamp(nxt, c.bandit_explore_min, c.bandit_explore_max)
        changed = abs(nxt - prev) > 1e-9
        if changed:
            c.bandit_explore = nxt
            s.last_bandit_explore_tune_ts = now_ts
        out.update(changed=changed, explore=c.bandit_explore, reason=reason)
        return out

    def _maybe_drift_reset(self, now_ts: float) -> Dict[str, Any]:
        c = self.config
        s = self.state
        total = int(np.sum(s.bandit_counts) + np.sum(s.bandit_stress_counts))
        out = {
            "changed": False,
            "reason": "disabled",
            "reset_count": s.bandit_drift_reset_count,
            "total_samples": total,
            "last_reset_ts": s.last_bandit_drift_reset_ts,
        }
        if not c.bandit_drift_reset_enabled:
            return out
        if total < c.bandit_drift_min_samples:
            out["reason"] = "insufficient_samples"
            return out
        if now_ts - s.last_bandit_drift_reset_ts < c.bandit_drift_cooldown_s:
            out["reason"] = "cooldown"
            return out

        reward_bad = s.bandit_reward_ema <= c.bandit_drift_reward_ema_floor
        var_high = s.bandit_reward_var_ema >= c.bandit_drift_var_ema_floor
        explore_high = c.bandit_explore >= 0.8 * c.bandit_explore_max
        if not (reward_bad and (var_high or explore_high)):
            out["reason"] = "healthy"
            return out

        # 漂移重置（向量化）
        s.bandit_counts = np.maximum(0, np.rint(s.bandit_counts * c.bandit_drift_count_decay).astype(np.int32))
        s.bandit_values *= c.bandit_drift_value_decay
        s.bandit_stress_counts = np.maximum(0, np.rint(s.bandit_stress_counts * c.bandit_drift_count_decay).astype(np.int32))
        s.bandit_stress_values *= c.bandit_drift_value_decay
        s.bandit_arm_reward_ema = np.clip(s.bandit_arm_reward_ema * c.bandit_drift_value_decay, -1.0, 1.0)
        s.bandit_arm_reward_var_ema = np.maximum(0.0, s.bandit_arm_reward_var_ema * c.bandit_drift_value_decay)
        c.bandit_explore = _clamp(c.bandit_explore + c.bandit_drift_explore_boost, c.bandit_explore_min, c.bandit_explore_max)
        if s.active_arm >= len(c.bandit_arms):
            s.active_arm = 0
        s.bandit_drift_reset_count += 1
        s.last_bandit_drift_reset_ts = now_ts
        out.update(changed=True, reason="drift_reset", reset_count=s.bandit_drift_reset_count,
                   total_samples=int(np.sum(s.bandit_counts) + np.sum(s.bandit_stress_counts)),
                   last_reset_ts=now_ts)
        return out
