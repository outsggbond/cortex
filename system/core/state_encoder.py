from __future__ import annotations

import hashlib
import os
import re
import logging
from typing import List, TYPE_CHECKING

import numpy as np

from system.core.embeddings import encode_text, embedding_available
from system.perception.encoders import project_to_dim

if TYPE_CHECKING:
    from system.brain.world_model import WorldState


logger = logging.getLogger(__name__)


# =========================
# ✅ 低冲突 hash（64bit + 双hash）
# =========================
def _hash_to_index(token: str, dim: int) -> int:
    h = hashlib.sha256(token.encode("utf-8")).digest()
    h1 = int.from_bytes(h[:8], "little")
    h2 = int.from_bytes(h[8:16], "little")
    return (h1 ^ h2) % dim  # XOR混合，降低冲突


# =========================
# Tokenize
# =========================
def _tokenize(text: str) -> List[str]:
    if not text:
        return []

    text = text.lower()

    if re.search(r"[\u4e00-\u9fff]", text):
        cleaned = re.sub(r"\s+", "", text)
        return (
            [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
            if len(cleaned) > 1
            else [cleaned] if cleaned else []
        )

    return [t for t in re.split(r"[^a-z0-9_]+", text) if t]


# =========================
# Encoder
# =========================
class StateEncoder:
    def __init__(self, dim: int | None = None, seed: int = 42):
        self.dim = int(dim or os.environ.get("STATE_VEC_DIM", "256"))
        self.seed = seed
        self.numeric_dims = 8

    # =========================
    # ✅ 数值归一（统一策略）
    # =========================
    def _normalize_numeric(self, nums: List[float]) -> np.ndarray:
        nums = np.array(nums, dtype="float32")

        # log scaling（适配动态系统）
        nums = np.log1p(nums)

        # normalize to [0,1]
        nums = nums / (np.max(nums) + 1e-6)

        return np.clip(nums, 0.0, 1.0)

    # =========================
    # 主编码
    # =========================
    def encode(self, state: WorldState, goal: str = "") -> np.ndarray:
        dim = self.dim
        vec = np.zeros(dim, dtype="float32")

        # =========================
        # ✅ numeric（统一归一）
        # =========================
        nums = [
            len(state.errors),
            len(state.missing_paths),
            len(state.perm_paths),
            state.added_count,
            state.modified_count,
            state.removed_count,
            state.total_files,
            1 if state.errors else 0,
        ]

        norm_nums = self._normalize_numeric(nums)
        vec[: self.numeric_dims] = norm_nums[: self.numeric_dims]

        # =========================
        # ✅ token + 权重 + 频次
        # =========================
        token_freq = {}

        def add_tokens(items, base_weight):
            for x in items:
                for t in _tokenize(str(x)):
                    token_freq[t] = token_freq.get(t, 0.0) + base_weight

        # 语义权重设计（关键）
        add_tokens(state.error_signatures, 3.0)
        add_tokens(state.error_types, 2.5)
        add_tokens(state.errors, 2.0)

        add_tokens(state.missing_paths, 1.5)
        add_tokens(state.perm_paths, 1.2)

        add_tokens([goal], 2.5)

        for abs_item in getattr(state, "active_abstractions", []) or []:
            if isinstance(abs_item, dict):
                tok = abs_item.get("token")
                if tok:
                    add_tokens([tok], 2.5)

                for m in (abs_item.get("members") or [])[:6]:
                    add_tokens([m], 1.5)

        # =========================
        # ✅ TF-like scaling（表达力提升）
        # =========================
        for tok, freq in list(token_freq.items())[:300]:
            idx = self.numeric_dims + _hash_to_index(tok, dim - self.numeric_dims)

            # sqrt压缩（避免高频爆炸）
            weight = np.sqrt(freq)

            vec[idx] += weight

        # normalize hashed
        hashed = vec[self.numeric_dims :]
        norm = np.linalg.norm(hashed) + 1e-8
        if norm > 0:
            vec[self.numeric_dims :] = hashed / norm

        # =========================
        # ✅ embedding 自适应融合
        # =========================
        if embedding_available():
            try:
                text = f"{goal} | {' '.join(state.errors[:3])}"
                emb = encode_text(text, require_model=True)
                emb = project_to_dim(np.array(emb, dtype="float32"), dim, self.seed)

                # embedding质量评估
                emb_norm = np.linalg.norm(emb)

                if emb_norm > 0.2:
                    alpha = 0.4
                elif emb_norm > 0.05:
                    alpha = 0.2
                else:
                    alpha = 0.05  # 几乎不用

                vec = (1 - alpha) * vec + alpha * emb

            except Exception:
                logger.debug("embedding blend failed", exc_info=True)

        return vec.astype("float32")

    # =========================
    # fallback文本编码
    # =========================
    def encode_text(self, state_text: str, goal: str = "") -> np.ndarray:
        dim = self.dim
        vec = np.zeros(dim, dtype="float32")

        tokens = _tokenize(state_text) + _tokenize(goal)

        token_freq = {}
        for t in tokens:
            token_freq[t] = token_freq.get(t, 0.0) + 1.0

        for tok, freq in list(token_freq.items())[:300]:
            idx = _hash_to_index(tok, dim)
            vec[idx] += np.sqrt(freq)

        norm = np.linalg.norm(vec) + 1e-8
        return (vec / norm).astype("float32")
