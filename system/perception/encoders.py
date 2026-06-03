from __future__ import annotations

import hashlib
from typing import Optional
import numpy as np


def _hash_to_index(text: str, dim: int) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % dim


def encode_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype="float32")
    for token in text.lower().split():
        idx = _hash_to_index(token, dim)
        vec[idx] += 1.0
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


def encode_image(image: np.ndarray, dim: int) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] >= 3:
        rgb = image[:, :, :3].astype("float32")
    elif image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1).astype("float32")
    else:
        rgb = image.astype("float32")
    mean = rgb.mean(axis=(0, 1))
    vec = np.zeros(dim, dtype="float32")
    vec[:3] = mean[:3] / 255.0
    return vec


def encode_audio(audio: np.ndarray, dim: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(dim, dtype="float32")
    a = audio.astype("float32")
    energy = float(np.mean(a * a))
    mean = float(np.mean(a))
    std = float(np.std(a))
    vec = np.zeros(dim, dtype="float32")
    vec[:3] = np.array([energy, mean, std], dtype="float32")
    return vec


def encode_video(video: np.ndarray, dim: int) -> np.ndarray:
    if video.size == 0:
        return np.zeros(dim, dtype="float32")
    # sample up to 5 frames
    frames = video
    if video.ndim >= 4:
        step = max(1, len(video) // 5)
        frames = video[::step]
    feats = []
    for frame in frames:
        feats.append(encode_image(frame, dim))
    return np.mean(np.stack(feats), axis=0)


def project_to_dim(vec: np.ndarray, dim: int, seed: int = 42) -> np.ndarray:
    if vec.shape[0] == dim:
        return vec.astype("float32")
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((vec.shape[0], dim)).astype("float32")
    out = vec.astype("float32") @ proj
    norm = np.linalg.norm(out) + 1e-8
    return (out / norm).astype("float32")


def ensure_float_array(x: Optional[np.ndarray]) -> np.ndarray:
    if x is None:
        return np.array([], dtype="float32")
    return x.astype("float32")
