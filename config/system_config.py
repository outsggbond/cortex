import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("multimodal")


@dataclass
class SystemConfig:
    # runtime
    feature_dim: int = 128
    use_gpu: Optional[bool] = None  # None => auto detect
    max_workers: int = 4
    seed: int = 42
    use_local_models: bool = False
    local_text_model_path: str = ""
    local_image_model_path: str = ""
    auto_tune: bool = True
    auto_tune_policy_path: str = "config/auto_tune.json"
    generative_only: bool = True
    deterministic_generation: bool = True
    temporal_decay_half_life_s: float = 0.0
    index_compress_after_s: float = 0.0
    index_compress_dim: int = 0
    prefetch_bytes: int = 0
    perception_auto_extract: bool = False
    perception_image_tagger: str = ""
    perception_audio_tagger: str = ""
    perception_conf_threshold: float = 0.75
    perception_top_k: int = 3

    # graph / memory
    similarity_threshold: float = 0.78
    reasoning_k: int = 5
    enable_persistence: bool = True
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"

    # evaluation
    eval_top_k: int = 5
    eval_rank_scores: Dict[str, float] = field(
        default_factory=lambda: {"1": 1.0, "2-3": 0.7, "4-5": 0.4}
    )
    chat_max_memories: int = 4

