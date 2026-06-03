from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


# ---------------------------------------------------------------------------
# 不可变的配置（Pydantic 自动校验、转换）
# ---------------------------------------------------------------------------
class ControllerConfig(BaseModel):
    """所有静态配置参数，带类型校验与范围约束"""
    enabled: bool = True
    sample_interval_s: float = Field(default=12.0, ge=1.0)
    apply_cooldown_s: float = Field(default=20.0, ge=0.0)
    ema_alpha: float = Field(default=0.35, ge=0.05, le=1.0)
    pressure_high: float = Field(default=0.82, ge=0.2, le=1.0)
    pressure_low: float = Field(default=0.55, ge=0.05)

    threshold_auto_enabled: bool = True
    threshold_auto_min_high: float = Field(default=0.68, ge=0.2, le=1.0)
    threshold_auto_max_high: float = Field(default=0.92, ge=0.2, le=1.0)
    threshold_auto_min_low: float = Field(default=0.35, ge=0.05, le=1.0)
    threshold_auto_gap: float = Field(default=0.22, ge=0.05, le=0.6)
    threshold_auto_step_up: float = Field(default=0.003, ge=0.0, le=0.05)
    threshold_auto_step_down: float = Field(default=0.01, ge=0.0, le=0.1)
    threshold_auto_cooldown_s: float = Field(default=4.0, ge=0.0)

    degrade_streak: int = Field(default=2, ge=1)
    boost_streak: int = Field(default=3, ge=1)
    degrade_step_reasoning_k: int = Field(default=2, ge=1)
    degrade_step_chat_memories: int = Field(default=1, ge=1)
    degrade_step_similarity: float = Field(default=0.01, ge=0.0)
    boost_step_reasoning_k: int = Field(default=1, ge=1)
    boost_step_chat_memories: int = Field(default=1, ge=1)
    boost_step_similarity: float = Field(default=0.005, ge=0.0)

    similarity_min: float = Field(default=0.65, ge=0.1, le=0.99)
    similarity_max: float = Field(default=0.95, ge=0.1, le=0.999)
    heavy_route_min_score: float = Field(default=0.62, ge=0.0, le=1.0)
    heavy_route_max_pressure: float = Field(default=0.70, ge=0.0, le=1.0)
    min_reasoning_k: int = Field(default=2, ge=1)
    min_chat_max_memories: int = Field(default=2, ge=1)

    # Bandit 通用
    bandit_enabled: bool = True
    bandit_arms: List[float] = Field(default=[0.85, 1.0, 1.15])
    bandit_explore: float = Field(default=0.75, ge=0.0)
    bandit_reward_decay: float = Field(default=0.0, ge=0.0, le=0.5)
    bandit_auto_explore_enabled: bool = True
    bandit_explore_min: float = Field(default=0.2, ge=0.0)
    bandit_explore_max: float = Field(default=1.6)
    bandit_explore_step_up: float = Field(default=0.03, ge=0.0, le=0.25)
    bandit_explore_step_down: float = Field(default=0.02, ge=0.0, le=0.25)
    bandit_explore_reward_low: float = Field(default=-0.15, ge=-1.0, le=1.0)
    bandit_explore_reward_high: float = Field(default=0.25, ge=-1.0, le=1.0)
    bandit_explore_cooldown_s: float = Field(default=4.0, ge=0.0)
    bandit_reward_ema_alpha: float = Field(default=0.2, ge=0.01, le=1.0)
    bandit_contextual_bias_enabled: bool = True
    bandit_contextual_pressure_weight: float = Field(default=0.35, ge=0.0, le=2.0)
    bandit_contextual_score_weight: float = Field(default=0.25, ge=0.0, le=2.0)
    bandit_contextual_bias_cap: float = Field(default=0.2, ge=0.0, le=1.0)
    bandit_contextual_cold_start_enabled: bool = True
    bandit_contextual_cold_start_gain: float = Field(default=0.4, ge=0.0, le=2.0)
    bandit_hardware_guard_enabled: bool = True
    bandit_hardware_strict_signature: bool = False
    bandit_hardware_mismatch_count_decay: float = Field(default=0.45, ge=0.0, le=1.0)
    bandit_hardware_mismatch_value_decay: float = Field(default=0.55, ge=0.0, le=1.0)
    bandit_hardware_mismatch_explore_boost: float = Field(default=0.25, ge=0.0)
    bandit_time_decay_enabled: bool = True
    bandit_time_decay_half_life_s: float = Field(default=1800.0, ge=1.0)
    bandit_time_decay_min_factor: float = Field(default=0.15, ge=0.0, le=1.0)
    bandit_time_decay_cooldown_s: float = Field(default=20.0, ge=0.0)
    bandit_arm_cooldown_enabled: bool = True
    bandit_arm_cooldown_reward_threshold: float = Field(default=-0.82, ge=-1.0, le=1.0)
    bandit_arm_cooldown_trigger_streak: int = Field(default=2, ge=1)
    bandit_arm_cooldown_s: float = Field(default=45.0, ge=0.0)
    bandit_all_cooled_safe_fallback_enabled: bool = True
    bandit_arm_risk_aware_enabled: bool = True
    bandit_arm_risk_alpha: float = Field(default=0.25, ge=0.01, le=1.0)
    bandit_arm_risk_weight: float = Field(default=0.45, ge=0.0, le=2.0)
    bandit_arm_risk_negative_weight: float = Field(default=0.35, ge=0.0, le=2.0)
    bandit_arm_risk_cap: float = Field(default=0.5, ge=0.0, le=1.0)
    bandit_regime_split_enabled: bool = True
    bandit_regime_pressure_gate: float = Field(default=0.78, ge=0.0, le=1.0)
    bandit_regime_score_gate: float = Field(default=0.45, ge=0.0, le=1.0)
    bandit_safety_guard_enabled: bool = True
    bandit_safety_pressure_gate: float = Field(default=0.90, ge=0.0, le=1.0)
    bandit_safety_score_gate: float = Field(default=0.30, ge=0.0, le=1.0)
    bandit_safety_max_multiplier: float = Field(default=1.0, ge=0.0)
    bandit_uncertainty_guard_enabled: bool = True
    bandit_uncertainty_var_gate: float = Field(default=0.32, ge=0.0)
    bandit_uncertainty_min_samples: int = Field(default=6, ge=1)
    bandit_uncertainty_pressure_gate: float = Field(default=0.62, ge=0.0, le=1.0)
    bandit_uncertainty_score_gate: float = Field(default=0.42, ge=0.0, le=1.0)
    bandit_uncertainty_max_multiplier: float = Field(default=1.0, ge=0.0)
    bandit_shock_guard_enabled: bool = True
    bandit_shock_reward_threshold: float = Field(default=-0.9, ge=-1.0, le=1.0)
    bandit_shock_trigger_count: int = Field(default=2, ge=1)
    bandit_shock_window_s: float = Field(default=25.0, ge=0.0)
    bandit_shock_cooldown_s: float = Field(default=45.0, ge=0.0)
    bandit_shock_max_multiplier: float = Field(default=1.0, ge=0.0)
    bandit_spike_guard_enabled: bool = True
    bandit_spike_pressure_delta_threshold: float = Field(default=0.18, ge=0.0, le=1.0)
    bandit_spike_score_drop_threshold: float = Field(default=0.22, ge=0.0, le=1.0)
    bandit_spike_min_pressure: float = Field(default=0.55, ge=0.0, le=1.0)
    bandit_spike_cooldown_s: float = Field(default=30.0, ge=0.0)
    bandit_spike_max_multiplier: float = Field(default=1.0, ge=0.0)
    bandit_drift_reset_enabled: bool = True
    bandit_drift_min_samples: int = Field(default=24, ge=1)
    bandit_drift_reward_ema_floor: float = Field(default=-0.22, ge=-1.0, le=1.0)
    bandit_drift_var_ema_floor: float = Field(default=0.30, ge=0.0)
    bandit_drift_cooldown_s: float = Field(default=90.0, ge=0.0)
    bandit_drift_count_decay: float = Field(default=0.5, ge=0.0, le=1.0)
    bandit_drift_value_decay: float = Field(default=0.35, ge=0.0, le=1.0)
    bandit_drift_explore_boost: float = Field(default=0.2, ge=0.0)

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# 运行时状态（纯数据，可序列化/反序列化）
# ---------------------------------------------------------------------------
@dataclass
class ControllerState:
    ema_pressure: float = 0.0
    ema_score: float = 0.0
    high_streak: int = 0
    low_streak: int = 0
    last_apply_ts: float = 0.0
    last_threshold_tune_ts: float = 0.0

    # Bandit 状态（使用 numpy 数组以获得向量化性能）
    bandit_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    bandit_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    bandit_stress_counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    bandit_stress_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    bandit_bad_streak: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    bandit_cooldown_until_ts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    bandit_arm_reward_ema: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    bandit_arm_reward_var_ema: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    active_arm: int = 0
    active_regime: str = "normal"
    bandit_shock_until_ts: float = 0.0
    bandit_spike_until_ts: float = 0.0
    bandit_shock_events_ts: List[float] = field(default_factory=list)
    bandit_reward_ema: float = 0.0
    bandit_reward_var_ema: float = 0.0
    last_bandit_explore_tune_ts: float = 0.0
    last_bandit_time_decay_ts: float = 0.0
    bandit_drift_reset_count: int = 0
    last_bandit_drift_reset_ts: float = 0.0
    bandit_hardware_signature: str = ""
    bandit_explore: float = 0.0   # 会被 config 初始化


# ---------------------------------------------------------------------------
# 不变的决策结构（保持原接口）
# ---------------------------------------------------------------------------
@dataclass
class RuntimeAdaptiveDecision:
    changed: bool
    mode: str
    reason: str
    pressure: float
    resource_score: float
    allow_heavy_routes: bool
    reasoning_k: int
    chat_max_memories: int
    similarity_threshold: float
    high_streak: int
    low_streak: int
    bandit_arm: int = -1
    bandit_multiplier: float = 1.0
    bandit_enabled: bool = False
    bandit_regime: str = "normal"
    bandit_regime_stress: bool = False
    bandit_safety_applied: bool = False
    bandit_safety_reason: str = ""
    bandit_uncertainty_active: bool = False
    bandit_shock_active: bool = False
    bandit_shock_reason: str = ""
    bandit_shock_until_ts: float = 0.0
    bandit_spike_active: bool = False
    bandit_spike_reason: str = ""
    bandit_spike_until_ts: float = 0.0
    pressure_high: float = 0.0
    pressure_low: float = 0.0
    threshold_tuned: bool = False


# ---------------------------------------------------------------------------
# 守卫抽象与具体实现（职责链模式）
# ---------------------------------------------------------------------------
class Guard(ABC):
    """安全守卫基类：返回触发的限制信息（若未触发则返回 None）"""

    @abstractmethod
    def evaluate(
        self,
        config: ControllerConfig,
        state: ControllerState,
        pressure: float,
        score: float,
        throttle_active: bool,
        now_ts: float,
    ) -> Optional[Dict[str, Any]]:
        """
        返回 None 表示该守卫未触发。
        触发时必须返回：
            reason: str
            max_multiplier: float     # 允许的最大倍数上限
            active: bool
        """
        ...


class ShockGuard(Guard):
    def evaluate(self, config, state, pressure, score, throttle_active, now_ts):
        if not config.bandit_shock_guard_enabled:
            return None
        if now_ts < state.bandit_shock_until_ts:
            return {
                "reason": "shock_guard",
                "max_multiplier": config.bandit_shock_max_multiplier,
                "active": True,
            }
        return None


class SpikeGuard(Guard):
    def evaluate(self, config, state, pressure, score, throttle_active, now_ts):
        if not config.bandit_spike_guard_enabled:
            return None
        if now_ts < state.bandit_spike_until_ts:
            return {
                "reason": "spike_guard",
                "max_multiplier": config.bandit_spike_max_multiplier,
                "active": True,
            }
        return None


class ThrottleGuard(Guard):
    def evaluate(self, config, state, pressure, score, throttle_active, now_ts):
        if throttle_active:
            return {
                "reason": "throttle",
                "max_multiplier": config.bandit_safety_max_multiplier,
                "active": True,
            }
        return None


class HighPressureGuard(Guard):
    def evaluate(self, config, state, pressure, score, throttle_active, now_ts):
        if pressure >= config.bandit_safety_pressure_gate:
            return {
                "reason": "high_pressure",
                "max_multiplier": config.bandit_safety_max_multiplier,
                "active": True,
            }
        return None


class LowScoreGuard(Guard):
    def evaluate(self, config, state, pressure, score, throttle_active, now_ts):
        if score <= config.bandit_safety_score_gate:
            return {
                "reason": "low_score",
                "max_multiplier": config.bandit_safety_max_multiplier,
                "active": True,
            }
        return None


class UncertaintyGuard(Guard):
    def evaluate(self, config, state, pressure, score, throttle_active, now_ts):
        if not config.bandit_uncertainty_guard_enabled:
            return None
        total_samples = int(np.sum(state.bandit_counts) + np.sum(state.bandit_stress_counts))
        if (
            total_samples >= config.bandit_uncertainty_min_samples
            and state.bandit_reward_var_ema >= config.bandit_uncertainty_var_gate
            and (
                pressure >= config.bandit_uncertainty_pressure_gate
                or score <= config.bandit_uncertainty_score_gate
            )
        ):
            return {
                "reason": "high_uncertainty",
                "max_multiplier": config.bandit_uncertainty_max_multiplier,
                "active": True,
            }
        return None
