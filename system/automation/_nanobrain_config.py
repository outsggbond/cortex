from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class NanoBrainConfig:
    base_gain: float = 1.0
    decay: float = 0.85
    threshold: float = 0.08
    tau: float = 0.04
    gravity: float = 1.0
    fluidity: float = 1.0
    weight_decay: float = 0.0
    logic_weight: float = 1.0
    emotion_weight: float = 0.9
    cross_weight: float = 1.0
    empathy: float = 0.7
    pmi_beta: float = 0.2
    max_delta_pairs: int = 5000
    profile_max_tokens: int = 200
    delta_path: str = "artifacts/memory/synapse_delta.json"
    profile_path: str = "artifacts/memory/user_profile.json"
    batch_size: int = 10
    batch_interval: float = 5.0
    queue_maxsize: int = 200
    damping_eta: float = 0.1
    damping_alert: float = 0.4
    max_iters: int = 6
    epsilon: float = 1e-3
    top_k: int = 8
    max_vocab: int = 20000
    max_tokens_per_doc: int = 64
    max_neighbors: int = 48
    pmi_min: float = 0.0
    pmi_scale: float = 1.0
    learning_rate: float = 0.05
    critical_gain: float = 2.0
    default_gain: float = 1.0
    critical_keywords: List[str] = field(default_factory=list)
    special_nodes: List[str] = field(default_factory=lambda: ["UNK", "CONFIRM", "REJECT", "ALARM"])
    logic_nodes: List[str] = field(default_factory=list)
    emotion_nodes: List[str] = field(default_factory=list)
    cross_synapses: List[Dict[str, Any]] = field(default_factory=list)
    emotion_lexicon: Dict[str, List[str]] = field(default_factory=dict)
    min_edge_weight: float = 1e-4
    path_depth: int = 3
    intensity_low: float = 0.4
    intensity_high: float = 0.7
