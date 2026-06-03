from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from system.runtime._adaptive_guards import (
    _clamp,
    ControllerConfig,
    ControllerState,
    RuntimeAdaptiveDecision,
)
from system.runtime._adaptive_engine import AdaptiveEngine

# 重新导出所有公开名称，保持原有 import 路径可用
__all__ = [
    "_clamp",
    "ControllerConfig",
    "ControllerState",
    "RuntimeAdaptiveDecision",
    "Guard",
    "ShockGuard",
    "SpikeGuard",
    "ThrottleGuard",
    "HighPressureGuard",
    "LowScoreGuard",
    "UncertaintyGuard",
    "AdaptiveEngine",
    "RuntimeAdaptiveController",
]

# 从子模块引入守卫类以便重新导出
from system.runtime._adaptive_guards import (  # noqa: E402
    Guard,
    ShockGuard,
    SpikeGuard,
    ThrottleGuard,
    HighPressureGuard,
    LowScoreGuard,
    UncertaintyGuard,
)


# ---------------------------------------------------------------------------
# 外观控制器（保持原有公开 API）
# ---------------------------------------------------------------------------
class RuntimeAdaptiveController:
    def __init__(self, policy: Dict[str, Any] | None, *, min_reasoning_k: int = 2, min_chat_max_memories: int = 2):
        raw = (policy or {}).get("runtime_adaptive", {})
        cfg_dict = raw if isinstance(raw, dict) else {}
        # Pydantic 自动校验 + 转换
        self.config = ControllerConfig(**cfg_dict)
        # 部分参数需要基于其它参数重新 clamp（保持原有逻辑）
        c = self.config
        c.pressure_high = _clamp(c.pressure_high, c.threshold_auto_min_high, c.threshold_auto_max_high)
        c.pressure_low = _clamp(c.pressure_low, c.threshold_auto_min_low, c.pressure_high)
        if c.pressure_low > c.pressure_high - 0.03:
            c.pressure_low = max(0.05, c.pressure_high - 0.03)
        c.bandit_explore = _clamp(c.bandit_explore, c.bandit_explore_min, c.bandit_explore_max)

        # 初始化状态
        n_arms = len(c.bandit_arms) if c.bandit_arms else 1
        self.state = ControllerState(
            bandit_counts=np.zeros(n_arms, dtype=np.int32),
            bandit_values=np.zeros(n_arms, dtype=np.float64),
            bandit_stress_counts=np.zeros(n_arms, dtype=np.int32),
            bandit_stress_values=np.zeros(n_arms, dtype=np.float64),
            bandit_bad_streak=np.zeros(n_arms, dtype=np.int32),
            bandit_cooldown_until_ts=np.zeros(n_arms, dtype=np.float64),
            bandit_arm_reward_ema=np.zeros(n_arms, dtype=np.float64),
            bandit_arm_reward_var_ema=np.zeros(n_arms, dtype=np.float64),
            bandit_explore=c.bandit_explore,
        )

        self.engine = AdaptiveEngine(self.config, self.state)

        # 为了保持原来接口中的 min_ 参数，如果 policy 没指定，则使用传入值
        if cfg_dict.get("min_reasoning_k") is None:
            self.config.min_reasoning_k = min_reasoning_k
        if cfg_dict.get("min_chat_max_memories") is None:
            self.config.min_chat_max_memories = min_chat_max_memories

    # ---------- 公开接口 ----------
    def control_state(self) -> Dict[str, Any]:
        s = self.state
        c = self.config
        return {
            "ema_pressure": s.ema_pressure,
            "ema_score": s.ema_score,
            "high_streak": s.high_streak,
            "low_streak": s.low_streak,
            "last_apply_ts": s.last_apply_ts,
            "pressure_high": c.pressure_high,
            "pressure_low": c.pressure_low,
            "last_threshold_tune_ts": s.last_threshold_tune_ts,
        }

    def load_control_state(self, state: Dict[str, Any]) -> bool:
        if not isinstance(state, dict):
            return False
        s = self.state
        c = self.config
        s.ema_pressure = _clamp(state.get("ema_pressure", s.ema_pressure), 0.0, 1.0)
        s.ema_score = _clamp(state.get("ema_score", s.ema_score), 0.0, 1.0)
        s.high_streak = max(0, state.get("high_streak", s.high_streak))
        s.low_streak = max(0, state.get("low_streak", s.low_streak))
        s.last_apply_ts = max(0.0, state.get("last_apply_ts", s.last_apply_ts))
        c.pressure_high = _clamp(state.get("pressure_high", c.pressure_high),
                                 c.threshold_auto_min_high, c.threshold_auto_max_high)
        low_cap = max(0.05, c.pressure_high - 0.03)
        low_floor = min(c.threshold_auto_min_low, low_cap)
        c.pressure_low = _clamp(state.get("pressure_low", c.pressure_low), low_floor, low_cap)
        s.last_threshold_tune_ts = max(0.0, state.get("last_threshold_tune_ts", s.last_threshold_tune_ts))
        return True

    def set_bandit_hardware_signature(self, signature: str) -> None:
        self.state.bandit_hardware_signature = str(signature or "")

    def bandit_load_info(self) -> Dict[str, Any]:
        return dict(self._bandit_last_load_info)

    def bandit_state(self) -> Dict[str, Any]:
        s = self.state
        c = self.config
        return {
            "enabled": c.bandit_enabled,
            "arms": list(c.bandit_arms),
            "counts": s.bandit_counts.tolist(),
            "values": s.bandit_values.tolist(),
            "stress_counts": s.bandit_stress_counts.tolist(),
            "stress_values": s.bandit_stress_values.tolist(),
            "active_arm": s.active_arm,
            "active_regime": s.active_regime,
            "shock_until_ts": s.bandit_shock_until_ts,
            "spike_until_ts": s.bandit_spike_until_ts,
            "shock_events_ts": list(s.bandit_shock_events_ts),
            "explore": c.bandit_explore,
            "reward_ema": s.bandit_reward_ema,
            "reward_var_ema": s.bandit_reward_var_ema,
            "last_explore_tune_ts": s.last_bandit_explore_tune_ts,
            "last_time_decay_ts": s.last_bandit_time_decay_ts,
            "bad_streak": s.bandit_bad_streak.tolist(),
            "cooldown_until_ts": s.bandit_cooldown_until_ts.tolist(),
            "arm_reward_ema": s.bandit_arm_reward_ema.tolist(),
            "arm_reward_var_ema": s.bandit_arm_reward_var_ema.tolist(),
            "drift_reset_count": s.bandit_drift_reset_count,
            "last_drift_reset_ts": s.last_bandit_drift_reset_ts,
            "hardware_signature": s.bandit_hardware_signature,
        }

    def load_bandit_state(
        self,
        state: Dict[str, Any],
        *,
        current_signature: str = "",
        signature_guard_enabled: bool | None = None,
        strict_signature: bool | None = None,
        mismatch_count_decay: float | None = None,
        mismatch_value_decay: float | None = None,
        mismatch_explore_boost: float | None = None,
    ) -> bool:
        # 保持原接口行为，使用 numpy 优化
        c = self.config
        s = self.state
        if not isinstance(state, dict) or not c.bandit_arms:
            self._bandit_last_load_info = {"loaded": False, "signature_mismatch": False, "saved_signature": "",
                                           "current_signature": current_signature, "reason": "invalid_state"}
            return False
        counts = state.get("counts")
        values = state.get("values")
        if not isinstance(counts, list) or not isinstance(values, list):
            self._bandit_last_load_info = {"loaded": False, "signature_mismatch": False, "saved_signature": "",
                                           "current_signature": current_signature, "reason": "invalid_arrays"}
            return False
        if len(counts) != len(c.bandit_arms) or len(values) != len(c.bandit_arms):
            self._bandit_last_load_info = {"loaded": False, "signature_mismatch": False, "saved_signature": "",
                                           "current_signature": current_signature, "reason": "arms_mismatch"}
            return False

        out_counts = np.maximum(0, np.array(counts, dtype=np.int32))
        out_values = np.array(values, dtype=np.float64)
        out_bad_streak = np.zeros(len(c.bandit_arms), dtype=np.int32)
        out_cooldown = np.zeros(len(c.bandit_arms), dtype=np.float64)
        out_arm_reward_ema = np.zeros(len(c.bandit_arms), dtype=np.float64)
        out_arm_reward_var_ema = np.zeros(len(c.bandit_arms), dtype=np.float64)
        out_stress_counts = np.zeros(len(c.bandit_arms), dtype=np.int32)
        out_stress_values = np.zeros(len(c.bandit_arms), dtype=np.float64)

        for key, default_arr in [
            ("bad_streak", out_bad_streak),
            ("cooldown_until_ts", out_cooldown),
            ("arm_reward_ema", out_arm_reward_ema),
            ("arm_reward_var_ema", out_arm_reward_var_ema),
            ("stress_counts", out_stress_counts),
            ("stress_values", out_stress_values),
        ]:
            val = state.get(key)
            if isinstance(val, list) and len(val) == len(c.bandit_arms):
                if key.endswith("_values") or key == "arm_reward_ema" or key == "arm_reward_var_ema":
                    default_arr[...] = np.array(val, dtype=np.float64)
                else:
                    default_arr[...] = np.maximum(0, np.array(val, dtype=np.int32))

        out_shock_until = max(0.0, state.get("shock_until_ts", s.bandit_shock_until_ts))
        out_spike_until = max(0.0, state.get("spike_until_ts", s.bandit_spike_until_ts))
        shock_events = state.get("shock_events_ts")
        out_shock_events = list(shock_events) if isinstance(shock_events, list) else []
        if c.bandit_shock_window_s > 0.0:
            now_ts = time.time()
            out_shock_events = [ts for ts in out_shock_events if now_ts - ts <= c.bandit_shock_window_s]

        guard = c.bandit_hardware_guard_enabled if signature_guard_enabled is None else signature_guard_enabled
        strict = c.bandit_hardware_strict_signature if strict_signature is None else strict_signature
        count_decay = c.bandit_hardware_mismatch_count_decay if mismatch_count_decay is None else _clamp(mismatch_count_decay, 0.0, 1.0)
        value_decay = c.bandit_hardware_mismatch_value_decay if mismatch_value_decay is None else _clamp(mismatch_value_decay, 0.0, 1.0)
        explore_boost = c.bandit_hardware_mismatch_explore_boost if mismatch_explore_boost is None else max(0.0, mismatch_explore_boost)

        saved_sig = state.get("hardware_signature", "") or ""
        current_sig = current_signature or ""
        mismatch = bool(guard and saved_sig and current_sig and saved_sig != current_sig)

        if mismatch and strict:
            self._bandit_last_load_info = {"loaded": False, "signature_mismatch": True,
                                           "saved_signature": saved_sig, "current_signature": current_sig,
                                           "reason": "signature_mismatch_strict"}
            return False
        if mismatch:
            out_counts = np.maximum(0, np.rint(out_counts * count_decay).astype(np.int32))
            out_values *= value_decay
            out_stress_counts = np.maximum(0, np.rint(out_stress_counts * count_decay).astype(np.int32))
            out_stress_values *= value_decay
            out_bad_streak = np.maximum(0, np.rint(out_bad_streak * count_decay).astype(np.int32))
            out_cooldown[...] = 0.0
            out_arm_reward_ema = np.clip(out_arm_reward_ema * value_decay, -1.0, 1.0)
            out_arm_reward_var_ema = np.maximum(0.0, out_arm_reward_var_ema * value_decay)
            out_shock_until = 0.0
            out_spike_until = 0.0
            out_shock_events = []

        s.bandit_counts = out_counts
        s.bandit_values = out_values
        s.bandit_stress_counts = out_stress_counts
        s.bandit_stress_values = out_stress_values
        s.bandit_bad_streak = out_bad_streak
        s.bandit_cooldown_until_ts = out_cooldown
        s.bandit_arm_reward_ema = out_arm_reward_ema
        s.bandit_arm_reward_var_ema = out_arm_reward_var_ema
        s.bandit_shock_until_ts = out_shock_until
        s.bandit_spike_until_ts = out_spike_until
        s.bandit_shock_events_ts = out_shock_events

        s.active_arm = max(0, min(len(c.bandit_arms) - 1, state.get("active_arm", s.active_arm)))
        regime = str(state.get("active_regime", s.active_regime) or "").lower()
        s.active_regime = "stress" if regime == "stress" else "normal"

        c.bandit_explore = _clamp(state.get("explore", c.bandit_explore), c.bandit_explore_min, c.bandit_explore_max)
        if mismatch and explore_boost > 0.0:
            c.bandit_explore = _clamp(c.bandit_explore + explore_boost, c.bandit_explore_min, c.bandit_explore_max)

        s.bandit_reward_ema = _clamp(state.get("reward_ema", s.bandit_reward_ema), -1.0, 1.0)
        s.bandit_reward_var_ema = max(0.0, state.get("reward_var_ema", s.bandit_reward_var_ema))
        s.last_bandit_explore_tune_ts = max(0.0, state.get("last_explore_tune_ts", s.last_bandit_explore_tune_ts))
        s.last_bandit_time_decay_ts = max(0.0, state.get("last_time_decay_ts", s.last_bandit_time_decay_ts))
        s.bandit_drift_reset_count = max(0, state.get("drift_reset_count", s.bandit_drift_reset_count))
        s.last_bandit_drift_reset_ts = max(0.0, state.get("last_drift_reset_ts", s.last_bandit_drift_reset_ts))

        s.bandit_hardware_signature = current_sig if current_sig else saved_sig

        self._bandit_last_load_info = {
            "loaded": True,
            "signature_mismatch": mismatch,
            "saved_signature": saved_sig,
            "current_signature": current_sig,
            "reason": "loaded_with_decay" if mismatch else "loaded",
        }
        return True

    def feedback(self, reward: float, *, now_ts: float | None = None) -> Dict[str, Any]:
        ts = time.time() if now_ts is None else now_ts
        return self.engine.feedback(reward, ts)

    def observe(
        self,
        *,
        now_ts: float,
        hw: Any,
        target: Dict[str, Any],
        current: Dict[str, Any],
        gpu_pressure: float = 0.0,
        throttle_active: bool = False,
    ) -> RuntimeAdaptiveDecision:
        c = self.config
        s = self.state
        cur_k = max(1, current.get("reasoning_k", c.min_reasoning_k))
        cur_mem = max(1, current.get("chat_max_memories", c.min_chat_max_memories))
        cur_sim = _clamp(current.get("similarity_threshold", c.similarity_min), c.similarity_min, c.similarity_max)
        tgt_k = max(c.min_reasoning_k, target.get("reasoning_k", cur_k))
        tgt_mem = max(c.min_chat_max_memories, target.get("chat_max_memories", cur_mem))
        tgt_sim = _clamp(target.get("similarity_threshold", cur_sim), c.similarity_min, c.similarity_max)

        pressure, score = self.engine._pressure(hw, gpu_pressure)
        self.engine.apply_time_decay(now_ts)

        bandit_pressure = _clamp((s.ema_pressure + pressure) * 0.5, 0.0, 1.0) if s.ema_pressure > 0.0 else pressure
        bandit_score = _clamp((s.ema_score + score) * 0.5, 0.0, 1.0) if s.ema_score > 0.0 else score
        regime_pressure = max(bandit_pressure, pressure)
        regime_score = min(bandit_score, score)
        stress_mode = self.engine._is_stress_regime(regime_pressure, regime_score, throttle_active)

        spike = self.engine.update_spike_guard(pressure, score, now_ts)
        spike_active = spike["active"]
        shock_active = now_ts < s.bandit_shock_until_ts

        bandit_arm = self.engine.choose_bandit_arm(bandit_pressure, bandit_score, now_ts, stress_mode)
        safety = self.engine.apply_safety_guard(
            arm=bandit_arm,
            pressure=pressure,
            score=score,
            throttle_active=throttle_active,
            now_ts=now_ts,
        )
        if safety.get("changed", False):
            bandit_arm = safety["arm"]
            s.active_arm = bandit_arm

        bandit_mul = c.bandit_arms[bandit_arm] if c.bandit_arms else 1.0
        tgt_k = max(c.min_reasoning_k, int(round(tgt_k * bandit_mul)))
        tgt_mem = max(c.min_chat_max_memories, int(round(tgt_mem * bandit_mul)))
        tgt_sim = _clamp(tgt_sim + (1.0 - bandit_mul) * 0.02, c.similarity_min, c.similarity_max)

        alpha = c.ema_alpha
        s.ema_pressure = pressure if s.ema_pressure <= 0.0 else (1.0 - alpha) * s.ema_pressure + alpha * pressure
        s.ema_score = score if s.ema_score <= 0.0 else (1.0 - alpha) * s.ema_score + alpha * score

        high_now = throttle_active or s.ema_pressure >= c.pressure_high
        low_now = (not throttle_active) and s.ema_pressure <= c.pressure_low
        s.high_streak = s.high_streak + 1 if high_now else 0
        s.low_streak = s.low_streak + 1 if low_now else 0

        out_k, out_mem, out_sim = cur_k, cur_mem, cur_sim
        mode, reason = "stable", "steady_state"
        changed = False
        cooldown_ok = (now_ts - s.last_apply_ts) >= c.apply_cooldown_s
        if c.enabled and cooldown_ok and s.high_streak >= c.degrade_streak:
            mode, reason = "degrade", "high_pressure"
            out_k = max(c.min_reasoning_k, min(tgt_k, cur_k - c.degrade_step_reasoning_k))
            out_mem = max(c.min_chat_max_memories, min(tgt_mem, cur_mem - c.degrade_step_chat_memories))
            out_sim = _clamp(cur_sim + c.degrade_step_similarity, c.similarity_min, c.similarity_max)
        elif c.enabled and cooldown_ok and s.low_streak >= c.boost_streak:
            mode, reason = "boost", "low_pressure_recover"
            out_k = min(tgt_k, cur_k + c.boost_step_reasoning_k)
            out_mem = min(tgt_mem, cur_mem + c.boost_step_chat_memories)
            out_sim = max(tgt_sim, cur_sim - c.boost_step_similarity) if cur_sim > tgt_sim else min(tgt_sim, cur_sim + c.boost_step_similarity)

        changed = out_k != cur_k or out_mem != cur_mem or abs(out_sim - cur_sim) > 1e-9
        if changed:
            s.last_apply_ts = now_ts

        arm_bias = bandit_mul - 1.0
        allow_heavy = (
            not throttle_active
            and s.ema_score >= _clamp(c.heavy_route_min_score - 0.05 * arm_bias, 0.0, 1.0)
            and s.ema_pressure <= _clamp(c.heavy_route_max_pressure + 0.05 * arm_bias, 0.0, 1.0)
        )
        threshold_tune = self.engine.autocalibrate_thresholds(now_ts, throttle_active)

        return RuntimeAdaptiveDecision(
            changed=changed,
            mode=mode,
            reason=reason,
            pressure=_clamp(s.ema_pressure, 0.0, 1.0),
            resource_score=_clamp(s.ema_score, 0.0, 1.0),
            allow_heavy_routes=allow_heavy,
            reasoning_k=out_k,
            chat_max_memories=out_mem,
            similarity_threshold=out_sim,
            high_streak=s.high_streak,
            low_streak=s.low_streak,
            bandit_arm=bandit_arm,
            bandit_multiplier=bandit_mul,
            bandit_enabled=c.bandit_enabled,
            bandit_regime=s.active_regime,
            bandit_regime_stress=s.active_regime == "stress",
            bandit_safety_applied=safety.get("triggered", False),
            bandit_safety_reason=safety.get("reason", ""),
            bandit_uncertainty_active="uncertainty" in safety.get("reason", ""),
            bandit_shock_active=shock_active,
            bandit_shock_reason=safety.get("reason", "") if shock_active else "",
            bandit_shock_until_ts=s.bandit_shock_until_ts,
            bandit_spike_active=spike_active,
            bandit_spike_reason=spike.get("reason", "") if spike_active else "",
            bandit_spike_until_ts=s.bandit_spike_until_ts,
            pressure_high=c.pressure_high,
            pressure_low=c.pressure_low,
            threshold_tuned=threshold_tune.get("changed", False),
        )
