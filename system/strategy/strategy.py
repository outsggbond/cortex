from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from system.core.auto_tune import load_policy, tune_strategy

from config.system_config import SystemConfig
from system.core.hardware import HardwareProfile


@dataclass
class Strategy:
    reasoning_k: int
    similarity_threshold: float
    enable_persistence: bool
    checkpoint_interval: int
    feature_dim: int
    chat_max_memories: int
    mode: str


def build_strategy(cfg: SystemConfig, hw: HardwareProfile, policy_path: Optional[str] = None) -> Strategy:
    if not getattr(cfg, "auto_tune", True):
        return Strategy(
            reasoning_k=cfg.reasoning_k,
            similarity_threshold=cfg.similarity_threshold,
            enable_persistence=cfg.enable_persistence,
            checkpoint_interval=cfg.checkpoint_interval,
            feature_dim=cfg.feature_dim,
            chat_max_memories=cfg.chat_max_memories,
            mode="static",
        )

    policy = load_policy(policy_path or getattr(cfg, "auto_tune_policy_path", "config/auto_tune.json"))
    if not policy.get("enabled", True):
        return Strategy(
            reasoning_k=cfg.reasoning_k,
            similarity_threshold=cfg.similarity_threshold,
            enable_persistence=cfg.enable_persistence,
            checkpoint_interval=cfg.checkpoint_interval,
            feature_dim=cfg.feature_dim,
            chat_max_memories=cfg.chat_max_memories,
            mode="static",
        )

    tuned = tune_strategy(policy, hw)
    mode = tuned.get("resource_level", "auto")
    score = tuned.get("resource_score", 0.0)
    return Strategy(
        reasoning_k=int(tuned.get("reasoning_k") or cfg.reasoning_k),
        similarity_threshold=float(tuned.get("similarity_threshold") or cfg.similarity_threshold),
        enable_persistence=True,
        checkpoint_interval=int(tuned.get("checkpoint_interval") or cfg.checkpoint_interval),
        feature_dim=int(tuned.get("feature_dim") or cfg.feature_dim),
        chat_max_memories=int(tuned.get("chat_max_memories") or cfg.chat_max_memories),
        mode=f"{mode}:{score:.2f}",
    )


def apply_strategy(cfg: SystemConfig, st: Strategy) -> None:
    cfg.reasoning_k = st.reasoning_k
    cfg.similarity_threshold = st.similarity_threshold
    cfg.enable_persistence = st.enable_persistence
    cfg.checkpoint_interval = st.checkpoint_interval
    cfg.feature_dim = st.feature_dim
    cfg.chat_max_memories = st.chat_max_memories


def apply_runtime_strategy(cfg: SystemConfig, st: Strategy) -> None:
    """Apply only safe runtime knobs (no feature_dim changes)."""
    cfg.reasoning_k = st.reasoning_k
    cfg.similarity_threshold = st.similarity_threshold
    cfg.enable_persistence = st.enable_persistence
    cfg.checkpoint_interval = st.checkpoint_interval
    cfg.chat_max_memories = st.chat_max_memories


def apply_throttle(cfg: SystemConfig, min_reasoning_k: int = 2, min_chat_max_memories: int = 2) -> None:
    cfg.reasoning_k = max(int(min_reasoning_k), int(cfg.reasoning_k) - 2)
    cfg.chat_max_memories = max(int(min_chat_max_memories), int(cfg.chat_max_memories) - 1)
