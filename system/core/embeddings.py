from __future__ import annotations

import functools
import os
import re
import numpy as np

_LOAD_ERROR = ""


# =========================
# Model Loading
# =========================
@functools.lru_cache(maxsize=1)
def _load_model():
    global _LOAD_ERROR
    _LOAD_ERROR = ""

    if os.environ.get("DISABLE_EMBEDDINGS_MODEL", "0") == "1":
        _LOAD_ERROR = "disabled"
        return None

    model_name = os.environ.get("EMBEDDING_MODEL", "").strip()
    if not model_name:
        _LOAD_ERROR = "model_name_not_set"
        return None

    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception as e:
        _LOAD_ERROR = str(e)
        return None


# =========================
# Dimension (改进1 ✅)
# =========================
def get_dim() -> int:
    model = _load_model()
    if model:
        try:
            return model.get_sentence_embedding_dimension()
        except Exception:
            pass
    return 384  # fallback 默认


# =========================
# Utils
# =========================
def _contains_cjk(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


# =========================
# Status (改进3 ✅)
# =========================
def embedding_status() -> dict:
    model_name = os.environ.get("EMBEDDING_MODEL", "").strip()
    disabled = os.environ.get("DISABLE_EMBEDDINGS_MODEL", "0") == "1"

    if disabled:
        return {
            "available": False,
            "mode": "disabled",
            "reason": "disabled"
        }

    if not model_name:
        return {
            "available": False,
            "mode": "fallback",
            "reason": "model_name_not_set",
            "dim": get_dim()
        }

    model = _load_model()

    if model is None:
        return {
            "available": False,
            "mode": "fallback",
            "reason": _LOAD_ERROR or "load_failed",
            "model": model_name,
            "dim": get_dim()
        }

    return {
        "available": True,
        "mode": "model",
        "model": model_name,
        "dim": get_dim()
    }


def embedding_available() -> bool:
    return embedding_status()["available"]


# =========================
# Fallback (改进2 ✅ 强化版)
# =========================
def _hash_fallback(text: str) -> np.ndarray:
    dim = get_dim()
    vec = np.zeros(dim, dtype="float32")

    if _contains_cjk(text):
        cleaned = re.sub(r"\s+", "", text)
        tokens = (
            [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]
            if len(cleaned) > 1
            else [cleaned] if cleaned else []
        )
    else:
        tokens = text.lower().split()

    for i, token in enumerate(tokens):
        h = hash(token)
        idx = h % dim
        value = ((h % 1000) / 1000.0)
        vec[idx] += value

    # normalize
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


# =========================
# Encode
# =========================
def encode_text(text: str, require_model: bool | None = None) -> np.ndarray:
    model = _load_model()

    if require_model is None:
        require_model = os.environ.get("EMBEDDING_REQUIRE", "0") == "1"

    if model is None:
        if require_model:
            raise RuntimeError(
                "Embedding model not available. Set EMBEDDING_MODEL or disable EMBEDDING_REQUIRE."
            )
        return _hash_fallback(text)

    emb = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return np.array(emb[0], dtype="float32")


# =========================
# Similarity
# =========================
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# =========================
# Clustering
# =========================
def cluster_embeddings(
    vectors: list[np.ndarray],
    threshold: float = 0.8
) -> list[np.ndarray]:

    clusters: list[np.ndarray] = []

    for v in vectors:
        if not clusters:
            clusters.append(v)
            continue

        sims = [cosine(v, c) for c in clusters]
        best = max(sims)

        if best >= threshold:
            idx = sims.index(best)
            clusters[idx] = (clusters[idx] + v) / 2.0
        else:
            clusters.append(v)

    return clusters
