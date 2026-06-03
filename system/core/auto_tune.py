from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "weights": {"cpu": 0.4, "ram": 0.4, "gpu": 0.2},
    "caps": {"cpu_cores": 32, "ram_gb": 64.0, "vram_gb": 16.0},
    "pressure_weight": 0.5,
    "level_thresholds": {"high": 0.75, "mid": 0.5},
    "feature_dim": {"min": 96, "max": 384, "step": 32},
    "reasoning_k": {"min": 4, "max": 12},
    "chat_max_memories": {"min": 3, "max": 10},
    "similarity_threshold": {"min": 0.72, "max": 0.88},
    "checkpoint_interval": {"min": 5, "max": 20},
    "temporal_decay": {"half_life_s": 1800},
    "index_compression": {"enabled": True, "after_s": 3600, "dim": 64},
    "prefetch": {"bytes": 16384},
    "throttle": {"gpu_mem_ratio": 0.9, "cooldown_s": 30, "min_reasoning_k": 2, "min_chat_max_memories": 2},
    "runtime_adaptive": {
        "enabled": True,
        "sample_interval_s": 12.0,
        "apply_cooldown_s": 20.0,
        "ema_alpha": 0.35,
        "pressure_high": 0.82,
        "pressure_low": 0.55,
        "degrade_streak": 2,
        "boost_streak": 3,
        "degrade_step_reasoning_k": 2,
        "degrade_step_chat_memories": 1,
        "degrade_step_similarity_threshold": 0.01,
        "boost_step_reasoning_k": 1,
        "boost_step_chat_memories": 1,
        "boost_step_similarity_threshold": 0.005,
        "similarity_threshold_min": 0.65,
        "similarity_threshold_max": 0.95,
        "min_reasoning_k": 2,
        "min_chat_max_memories": 2,
        "heavy_route_min_score": 0.62,
        "heavy_route_max_pressure": 0.70,
        "transformer_hot_tune": True,
        "planner_hot_tune": True,
        "planner_depth_min": 2,
        "planner_depth_max": 6,
        "planner_beam_min": 2,
        "planner_beam_max": 10,
        "transformer_token_floor": 48,
        "transformer_token_ceiling": 512,
        "transformer_input_floor": 256,
        "transformer_input_ceiling": 4096,
        "heartbeat_enabled": True,
        "heartbeat_interval_s": 6.0,
        "bandit_enabled": True,
        "bandit_arms": [0.85, 1.0, 1.15],
        "bandit_explore": 0.75,
        "bandit_auto_explore_enabled": True,
        "bandit_explore_min": 0.2,
        "bandit_explore_max": 1.6,
        "bandit_explore_step_up": 0.03,
        "bandit_explore_step_down": 0.02,
        "bandit_explore_reward_low": -0.15,
        "bandit_explore_reward_high": 0.25,
        "bandit_explore_cooldown_s": 4.0,
        "bandit_reward_ema_alpha": 0.2,
        "bandit_contextual_bias_enabled": True,
        "bandit_contextual_pressure_weight": 0.35,
        "bandit_contextual_score_weight": 0.25,
        "bandit_contextual_bias_cap": 0.2,
        "bandit_contextual_cold_start_enabled": True,
        "bandit_contextual_cold_start_gain": 0.4,
        "bandit_hardware_guard_enabled": True,
        "bandit_hardware_strict_signature": False,
        "bandit_hardware_mismatch_count_decay": 0.45,
        "bandit_hardware_mismatch_value_decay": 0.55,
        "bandit_hardware_mismatch_explore_boost": 0.25,
        "bandit_time_decay_enabled": True,
        "bandit_time_decay_half_life_s": 1800.0,
        "bandit_time_decay_min_factor": 0.15,
        "bandit_time_decay_cooldown_s": 20.0,
        "bandit_arm_cooldown_enabled": True,
        "bandit_arm_cooldown_reward_threshold": -0.82,
        "bandit_arm_cooldown_trigger_streak": 2,
        "bandit_arm_cooldown_s": 45.0,
        "bandit_all_cooled_safe_fallback_enabled": True,
        "bandit_arm_risk_aware_enabled": True,
        "bandit_arm_risk_alpha": 0.25,
        "bandit_arm_risk_weight": 0.45,
        "bandit_arm_risk_negative_weight": 0.35,
        "bandit_arm_risk_cap": 0.5,
        "bandit_safety_guard_enabled": True,
        "bandit_safety_pressure_gate": 0.90,
        "bandit_safety_score_gate": 0.30,
        "bandit_safety_max_multiplier": 1.0,
        "bandit_regime_split_enabled": True,
        "bandit_regime_pressure_gate": 0.78,
        "bandit_regime_score_gate": 0.45,
        "bandit_shock_guard_enabled": True,
        "bandit_shock_reward_threshold": -0.9,
        "bandit_shock_trigger_count": 2,
        "bandit_shock_window_s": 25.0,
        "bandit_shock_cooldown_s": 45.0,
        "bandit_shock_max_multiplier": 1.0,
        "bandit_spike_guard_enabled": True,
        "bandit_spike_pressure_delta_threshold": 0.18,
        "bandit_spike_score_drop_threshold": 0.22,
        "bandit_spike_min_pressure": 0.55,
        "bandit_spike_cooldown_s": 30.0,
        "bandit_spike_max_multiplier": 1.0,
        "bandit_drift_reset_enabled": True,
        "bandit_drift_min_samples": 24,
        "bandit_drift_reward_ema_floor": -0.22,
        "bandit_drift_var_ema_floor": 0.30,
        "bandit_drift_cooldown_s": 90.0,
        "bandit_drift_count_decay": 0.5,
        "bandit_drift_value_decay": 0.35,
        "bandit_drift_explore_boost": 0.2,
        "bandit_reward_decay": 0.0,
        "state_path": "artifacts/audit/runtime_adapt_state.json",
        "load_state": True,
        "save_state_interval_s": 12.0,
        "dynamic_interval_enabled": True,
        "dynamic_interval_min_s": 2.0,
        "dynamic_interval_max_s": 24.0,
        "dynamic_interval_high_mul": 0.55,
        "dynamic_interval_low_mul": 1.45,
        "dynamic_interval_throttle_mul": 0.45,
        "threshold_auto_enabled": True,
        "threshold_auto_min_high": 0.68,
        "threshold_auto_max_high": 0.92,
        "threshold_auto_min_low": 0.35,
        "threshold_auto_gap": 0.22,
        "threshold_auto_step_up": 0.003,
        "threshold_auto_step_down": 0.01,
        "threshold_auto_cooldown_s": 4.0,
        "health_emit_interval_s": 30.0,
        "feedback_ema_alpha": 0.12,
        "events_path": "artifacts/audit/runtime_adapt_events.jsonl",
        "bandit_state_path": "artifacts/audit/runtime_adapt_bandit_state.json",
    },
    "perception_models": {
        "image_tagger": "",
        "audio_tagger": "",
        "auto_extract": True,
        "confidence_threshold": 0.75,
        "top_k": 3,
    },
    "semantic_refiner": {
        "enabled": True,
        "mode": "hybrid",
        "min_co_count": 2.0,
        "min_cluster_size": 3,
        "max_cluster_size": 6,
        "min_density": 0.3,
        "min_avg_weight": 1.0,
        "decay": 0.9,
        "max_new_tokens": 3,
        "cooldown_s": 120.0,
        "min_resource_score": 0.6,
        "abstract_prefix": "abs::",
        "generation_interval": 20,
        "retain_ratio": 0.4,
        "kill_ratio": 0.2,
        "mutate_ratio": 0.4,
        "max_population": 50,
        "merge_overlap": 0.6,
        "decay_on_kill": 0.3,
        "generalization_cap": 30,
        "age_half_life_s": 3600.0,
        "fitness_weights": {"alpha": 1.0, "beta": 0.8, "gamma": 0.6, "delta": 0.4, "epsilon": 0.5}
    },
    "abstraction_competition": {
        "enabled": True,
        "max_candidates": 4,
        "max_combo": 2,
        "max_sets": 6,
        "min_score": 0.2
    },
    "cross_modal_aliases": {
        "cat": ["猫", "喵", "cat", "meow"],
    },
    "transformer": {
        "max_new_tokens": {"min": 64, "max": 256, "step": 8},
        "max_input_tokens": {"min": 384, "max": 2048, "step": 128},
        "temperature": {"min": 0.2, "max": 0.8},
        "top_p": {"min": 0.85, "max": 0.98},
        "top_k": {"min": 20, "max": 80},
        "repetition_penalty": {"min": 1.02, "max": 1.12},
        "internal_thought_tokens": {"min": 0, "max": 128, "step": 8},
        "deliberate_min_score": 0.7,
        "self_consistency": {"enabled": True, "min_score": 0.7, "min_overlap_chars": 18},
        "deterministic": True,
    },
    "gnn": {
        "embed_dim": {"min": 48, "max": 128, "step": 16},
        "context_size": {"min": 6, "max": 12},
        "top_k": {"min": 4, "max": 12},
        "temperature": {"min": 0.1, "max": 0.6},
        "deterministic": True,
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_policy(path: str | Path = "config/auto_tune.json") -> Dict[str, Any]:
    policy = copy.deepcopy(DEFAULT_POLICY)
    if not path:
        return policy
    policy_path = Path(path)
    if not policy_path.exists():
        return policy
    try:
        data = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception:
        return policy
    if isinstance(data, dict):
        _deep_update(policy, data)
    return policy


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _norm(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return _clamp(value / cap, 0.0, 1.0)


def compute_resource_score(hw, policy: Dict[str, Any]) -> float:
    caps = policy.get("caps", {})
    weights = policy.get("weights", {})
    cpu_cores = float(getattr(hw, "cpu_cores", 0) or 0)
    ram_avail = float(getattr(hw, "available_ram_gb", 0) or 0)
    ram_total = float(getattr(hw, "total_ram_gb", 0) or 0)
    gpu_available = bool(getattr(hw, "gpu_available", False))
    vram = float(getattr(hw, "gpu_vram_gb", 0) or 0) if gpu_available else 0.0

    cpu_score = _norm(cpu_cores, float(caps.get("cpu_cores", 1)))
    ram_base = ram_avail if ram_avail > 0 else ram_total
    ram_score = _norm(ram_base, float(caps.get("ram_gb", 1)))
    gpu_score = _norm(vram, float(caps.get("vram_gb", 1)))

    base = (
        cpu_score * float(weights.get("cpu", 0.4))
        + ram_score * float(weights.get("ram", 0.4))
        + gpu_score * float(weights.get("gpu", 0.2))
    )
    if base <= 0:
        base = 0.5

    cpu_usage = float(getattr(hw, "cpu_usage_percent", 0) or 0)
    cpu_pressure = _clamp(cpu_usage / 100.0, 0.0, 1.0) if cpu_usage > 0 else 0.0
    if ram_total > 0:
        ram_pressure = 1.0 - _clamp(ram_avail / ram_total, 0.0, 1.0)
    else:
        ram_pressure = 0.0
    pressure = max(cpu_pressure, ram_pressure)
    score = base * (1.0 - float(policy.get("pressure_weight", 0.5)) * pressure)
    return _clamp(score, 0.05, 1.0)


def classify_resource_level(score: float, policy: Dict[str, Any]) -> str:
    thresholds = policy.get("level_thresholds", {})
    high = float(thresholds.get("high", 0.75))
    mid = float(thresholds.get("mid", 0.5))
    if score >= high:
        return "high"
    if score >= mid:
        return "mid"
    return "low"


def scale_int(range_cfg: Dict[str, Any], score: float, invert: bool = False) -> int:
    if not range_cfg:
        return 0
    lo = int(range_cfg.get("min", 0))
    hi = int(range_cfg.get("max", lo))
    step = int(range_cfg.get("step", 1)) or 1
    s = 1.0 - score if invert else score
    val = lo + (hi - lo) * s
    val = int(round(val / step) * step)
    return int(_clamp(float(val), float(lo), float(hi)))


def scale_float(range_cfg: Dict[str, Any], score: float, invert: bool = False) -> float:
    if not range_cfg:
        return 0.0
    lo = float(range_cfg.get("min", 0.0))
    hi = float(range_cfg.get("max", lo))
    s = 1.0 - score if invert else score
    val = lo + (hi - lo) * s
    step = range_cfg.get("step")
    if step:
        step = float(step)
        if step > 0:
            val = round(val / step) * step
    return float(_clamp(val, min(lo, hi), max(lo, hi)))


def tune_strategy(policy: Dict[str, Any], hw) -> Dict[str, Any]:
    score = compute_resource_score(hw, policy)
    return {
        "resource_score": score,
        "resource_level": classify_resource_level(score, policy),
        "feature_dim": scale_int(policy.get("feature_dim", {}), score),
        "reasoning_k": scale_int(policy.get("reasoning_k", {}), score),
        "chat_max_memories": scale_int(policy.get("chat_max_memories", {}), score),
        "similarity_threshold": scale_float(policy.get("similarity_threshold", {}), score, invert=True),
        "checkpoint_interval": scale_int(policy.get("checkpoint_interval", {}), score, invert=True),
    }


def tune_transformer(policy: Dict[str, Any], hw) -> Dict[str, Any]:
    score = compute_resource_score(hw, policy)
    tcfg = policy.get("transformer", {})
    self_consistency_cfg = tcfg.get("self_consistency", {}) if isinstance(tcfg.get("self_consistency"), dict) else {}
    self_consistency = bool(self_consistency_cfg.get("enabled", False))
    sc_min = float(self_consistency_cfg.get("min_score", 0.7))
    if score < sc_min:
        self_consistency = False
    return {
        "resource_score": score,
        "max_new_tokens": scale_int(tcfg.get("max_new_tokens", {}), score),
        "max_input_tokens": scale_int(tcfg.get("max_input_tokens", {}), score),
        "temperature": scale_float(tcfg.get("temperature", {}), score),
        "top_p": scale_float(tcfg.get("top_p", {}), score),
        "top_k": scale_int(tcfg.get("top_k", {}), score),
        "repetition_penalty": scale_float(tcfg.get("repetition_penalty", {}), score),
        "deterministic": bool(tcfg.get("deterministic", False)),
        "internal_thought_tokens": scale_int(tcfg.get("internal_thought_tokens", {}), score),
        "deliberate_min_score": float(tcfg.get("deliberate_min_score", 0.7)),
        "self_consistency": self_consistency,
        "self_consistency_min_overlap": int(self_consistency_cfg.get("min_overlap_chars", 18)),
    }


def tune_gnn(policy: Dict[str, Any], hw) -> Dict[str, Any]:
    score = compute_resource_score(hw, policy)
    gcfg = policy.get("gnn", {})
    return {
        "embed_dim": scale_int(gcfg.get("embed_dim", {}), score),
        "context_size": scale_int(gcfg.get("context_size", {}), score),
        "top_k": scale_int(gcfg.get("top_k", {}), score),
        "temperature": scale_float(gcfg.get("temperature", {}), score),
        "deterministic": bool(gcfg.get("deterministic", False)),
    }
