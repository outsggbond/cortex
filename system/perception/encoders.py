# -*- coding: utf-8 -*-
"""Multi-modal encoders — real implementations with graceful fallback.

Text:  sentence-transformers (preferred) → CLIP text → hash fallback
Image: CLIP vision (preferred) → raw features → color fallback
Audio: torchaudio MFCC (preferred) → numpy FFT → energy fallback
Video: sampled frames → image encoder → temporal average
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy-loaded model handles (singletons, loaded once) ──────────────────────

_clip_model = None
_clip_processor = None
_text_embedder = None
_TEXT_EMBEDDER_AVAILABLE = False


def _get_clip():
    """Load CLIP model once (ViT-B/32, ~340 MB)."""
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor
    try:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logger.info("CLIP ViT-B/32 loaded for multi-modal encoding")
        return _clip_model, _clip_processor
    except Exception as e:
        logger.debug("CLIP not available: %s — using fallback encoders", e)
        _clip_model = False
        _clip_processor = False
        return None, None


def _get_text_embedder():
    """Load sentence-transformers for text encoding."""
    global _text_embedder, _TEXT_EMBEDDER_AVAILABLE
    if _text_embedder is not None:
        return _text_embedder
    try:
        from sentence_transformers import SentenceTransformer
        _text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        _TEXT_EMBEDDER_AVAILABLE = True
        logger.info("SentenceTransformer all-MiniLM-L6-v2 loaded")
        return _text_embedder
    except Exception as e:
        logger.debug("SentenceTransformer not available: %s", e)
        _text_embedder = False
        return None


# ── Hash fallback (deterministic, no model needed) ───────────────────────────

def _hash_to_index(text: str, dim: int) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % dim


# ── Text ─────────────────────────────────────────────────────────────────────

def encode_text(text: str, dim: int = 384) -> np.ndarray:
    """Encode text to a fixed-dim vector.

    Uses sentence-transformers (384d) → projects to dim.
    Falls back to hash-based sparse encoding.
    """
    if not text or not text.strip():
        return np.zeros(dim, dtype="float32")

    # Try sentence-transformers
    embedder = _get_text_embedder()
    if embedder is not None:
        try:
            vec = embedder.encode([text], show_progress_bar=False)[0]
            return project_to_dim(vec.astype("float32"), dim)
        except Exception:
            pass

    # Try CLIP text encoder
    clip_model, clip_proc = _get_clip()
    if clip_model is not False and clip_model is not None:
        try:
            import torch
            inputs = clip_proc(text=[text], return_tensors="pt", truncation=True)
            with torch.no_grad():
                vec = clip_model.get_text_features(**inputs).numpy()[0]
            return project_to_dim(vec.astype("float32"), dim)
        except Exception:
            pass

    # Fallback: hash-based sparse encoding
    vec = np.zeros(dim, dtype="float32")
    for token in text.lower().split():
        idx = _hash_to_index(token, dim)
        vec[idx] += 1.0
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


# ── Image ────────────────────────────────────────────────────────────────────

def encode_image(image: np.ndarray, dim: int = 512) -> np.ndarray:
    """Encode an image (H, W, C) to a fixed-dim feature vector.

    Uses CLIP ViT-B/32 (512d) → projects to dim.
    Falls back to raw statistics if CLIP unavailable.
    """
    if image is None or image.size == 0:
        return np.zeros(dim, dtype="float32")

    # Ensure (H, W, C) format
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]

    # Try CLIP
    clip_model, clip_proc = _get_clip()
    if clip_model is not False and clip_model is not None:
        try:
            import torch
            from PIL import Image
            if image.dtype != np.uint8:
                img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                img_uint8 = image
            pil_img = Image.fromarray(img_uint8)
            inputs = clip_proc(images=pil_img, return_tensors="pt")
            with torch.no_grad():
                vec = clip_model.get_image_features(**inputs).numpy()[0]
            return project_to_dim(vec.astype("float32"), dim)
        except Exception as e:
            logger.debug("CLIP image encoding failed: %s", e)

    # Fallback: multi-level image statistics
    vec = np.zeros(dim, dtype="float32")
    rgb = image.astype("float32")
    # Channel means (3)
    vec[0:3] = rgb.mean(axis=(0, 1))[:3] / 255.0
    # Channel stds (3)
    vec[3:6] = rgb.std(axis=(0, 1))[:3] / 255.0
    # Spatial statistics (sampled)
    h, w = rgb.shape[0], rgb.shape[1]
    regions = 4
    for i in range(regions):
        for j in range(regions):
            rh = slice(i * h // regions, (i + 1) * h // regions)
            rw = slice(j * w // regions, (j + 1) * w // regions)
            idx = 6 + (i * regions + j) * 3
            if idx + 3 <= dim:
                vec[idx:idx + 3] = rgb[rh, rw].mean(axis=(0, 1))[:3] / 255.0
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


# ── Audio ────────────────────────────────────────────────────────────────────

def encode_audio(audio: np.ndarray, dim: int = 128) -> np.ndarray:
    """Encode audio waveform to a fixed-dim feature vector.

    Uses torchaudio MFCC (if available) → projects to dim.
    Falls back to FFT-based spectral features → basic statistics.
    """
    if audio is None or audio.size == 0:
        return np.zeros(dim, dtype="float32")

    a = audio.astype("float32")

    # Try torchaudio MFCC
    try:
        import torch
        import torchaudio
        waveform = torch.from_numpy(a).unsqueeze(0)  # (1, samples)
        # Resample to 16kHz if needed (assume input is 16kHz)
        mfcc = torchaudio.functional.mfcc(
            waveform,
            sample_rate=16000,
            n_mfcc=min(40, dim),
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 80},
        )  # (1, n_mfcc, time)
        vec = mfcc.mean(dim=-1).squeeze().numpy()  # (n_mfcc,)
        return project_to_dim(vec.astype("float32"), dim)
    except Exception as e:
        logger.debug("torchaudio MFCC failed: %s", e)

    # Try numpy FFT-based features
    try:
        n_fft = min(512, len(a))
        spec = np.abs(np.fft.rfft(a, n=n_fft))
        mel_bins = min(32, dim // 3)
        vec = np.zeros(dim, dtype="float32")
        # Low-frequency spectral energy
        vec[:mel_bins] = spec[:mel_bins] / (np.max(spec) + 1e-8)
        # Time-domain features
        vec[mel_bins:mel_bins + 3] = [
            float(np.mean(a * a)),  # energy
            float(np.mean(np.abs(a))),  # mean amplitude
            float(np.std(a)),  # std
        ]
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm
    except Exception:
        pass

    # Ultimate fallback: basic statistics
    vec = np.zeros(dim, dtype="float32")
    vec[0] = float(np.mean(a * a))
    vec[1] = float(np.mean(a))
    vec[2] = float(np.std(a))
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


# ── Video ────────────────────────────────────────────────────────────────────

def encode_video(video: np.ndarray, dim: int = 512) -> np.ndarray:
    """Encode video (N, H, W, C) by sampling frames and averaging."""
    if video is None or video.size == 0:
        return np.zeros(dim, dtype="float32")

    # Sample up to 8 frames
    frames = video
    if video.ndim >= 4:
        step = max(1, len(video) // 8)
        frames = video[::step][:8]

    feats = []
    for frame in frames:
        feats.append(encode_image(frame, dim))
    if not feats:
        return np.zeros(dim, dtype="float32")
    return np.mean(np.stack(feats), axis=0).astype("float32")


# ── Utility ──────────────────────────────────────────────────────────────────

def project_to_dim(vec: np.ndarray, dim: int, seed: int = 42) -> np.ndarray:
    """Project vector to target dimension (random projection if needed)."""
    if vec.shape[0] == dim:
        return vec.astype("float32")
    rng = np.random.default_rng(seed)
    if vec.shape[0] < dim:
        # Pad with zeros
        out = np.zeros(dim, dtype="float32")
        out[:vec.shape[0]] = vec
        return out
    # Random projection
    proj = rng.standard_normal((vec.shape[0], dim)).astype("float32")
    out = vec.astype("float32") @ proj
    norm = np.linalg.norm(out) + 1e-8
    return (out / norm).astype("float32")


def ensure_float_array(x: Optional[np.ndarray]) -> np.ndarray:
    if x is None:
        return np.array([], dtype="float32")
    return x.astype("float32")
