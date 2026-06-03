# -*- coding: utf-8 -*-
"""Image understanding — describe, classify, and answer questions about images.

Uses CLIP (ViT-B/32) for classification and BLIP/BLIP-2 for captioning
when available, with graceful fallback to basic image statistics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy model handles ───────────────────────────────────────────────────────

_blip_model = None
_blip_processor = None


def _get_blip():
    """Load BLIP image captioning model (lazy, singleton)."""
    global _blip_model, _blip_processor
    if _blip_model is not None:
        return _blip_model, _blip_processor
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        logger.info("BLIP image captioning model loaded")
        return _blip_model, _blip_processor
    except Exception as e:
        logger.debug("BLIP not available: %s", e)
        _blip_model = False
        _blip_processor = False
        return None, None


# ── Public API ───────────────────────────────────────────────────────────────

def load_image(path: str) -> Optional[np.ndarray]:
    """Load an image file as a numpy array (H, W, C) in RGB."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        logger.warning("Failed to load image %s: %s", path, e)
        return None


def describe_image(image: np.ndarray, *, max_length: int = 64) -> Optional[str]:
    """Generate a natural-language caption for an image using BLIP.

    Falls back to basic statistics if BLIP is unavailable.
    """
    if image is None or image.size == 0:
        return None

    # Try BLIP
    model, processor = _get_blip()
    if model is not False and model is not None:
        try:
            import torch
            from PIL import Image
            if image.dtype != np.uint8:
                img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                img_uint8 = image
            pil_img = Image.fromarray(img_uint8)
            inputs = processor(pil_img, return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs, max_length=max_length)
            caption = processor.decode(out[0], skip_special_tokens=True)
            return str(caption).strip()
        except Exception as e:
            logger.debug("BLIP captioning failed: %s", e)

    # Fallback: basic image description
    return _basic_image_description(image)


def analyze_image(image: np.ndarray) -> Dict[str, Any]:
    """Comprehensive image analysis: caption, statistics, dominant colors.

    Returns a dict suitable for passing to the chat pipeline as context.
    """
    result: Dict[str, Any] = {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "size_bytes": int(image.nbytes),
    }

    # Basic statistics
    rgb = image.astype("float32")
    result["mean_rgb"] = [round(float(rgb[:, :, i].mean()), 1) for i in range(3)]
    result["std_rgb"] = [round(float(rgb[:, :, i].std()), 1) for i in range(3)]

    # Dominant color
    try:
        from PIL import Image
        pil_img = Image.fromarray(
            image if image.dtype == np.uint8 else (image * 255).clip(0, 255).astype(np.uint8)
        )
        # Quantize to find dominant colors
        q = pil_img.quantize(colors=5, method=2)
        palette = q.getpalette()[:15]  # 5 colors × 3
        dominant = []
        for i in range(0, len(palette), 3):
            dominant.append(tuple(palette[i:i + 3]))
        result["dominant_colors"] = dominant
    except Exception:
        pass

    # Caption
    caption = describe_image(image)
    if caption:
        result["caption"] = caption

    # Resolution category
    h, w = image.shape[0], image.shape[1]
    pixels = h * w
    if pixels < 50000:
        result["resolution"] = "tiny"
    elif pixels < 500000:
        result["resolution"] = "medium"
    elif pixels < 2000000:
        result["resolution"] = "large"
    else:
        result["resolution"] = "very_large"

    # Aspect ratio
    result["aspect_ratio"] = round(w / max(1, h), 2)
    if result["aspect_ratio"] > 2.0:
        result["orientation"] = "wide"
    elif result["aspect_ratio"] < 0.5:
        result["orientation"] = "tall"
    else:
        result["orientation"] = "square"

    return result


def classify_image(image: np.ndarray, labels: List[str]) -> List[tuple[str, float]]:
    """Zero-shot classify an image against a list of text labels using CLIP.

    Returns sorted list of (label, score) tuples.
    """
    if not labels:
        return []

    from system.perception.encoders import encode_image, encode_text

    img_vec = encode_image(image, dim=512)
    results = []
    for label in labels:
        txt_vec = encode_text(label, dim=512)
        sim = float(np.dot(img_vec, txt_vec) / (np.linalg.norm(img_vec) * np.linalg.norm(txt_vec) + 1e-8))
        results.append((label, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _basic_image_description(image: np.ndarray) -> str:
    """Generate a basic description from image statistics (no model needed)."""
    h, w = image.shape[0], image.shape[1]
    rgb = image.astype("float32")
    means = rgb.mean(axis=(0, 1))
    r, g, b = means[0], means[1], means[2]

    # Guess at image type from color balance
    if r > g + 20 and r > b + 20:
        tone = "warm/reddish"
    elif b > r + 20 and b > g + 20:
        tone = "cool/bluish"
    elif g > r + 10 and g > b + 10:
        tone = "green-toned"
    else:
        tone = "neutral-toned"

    brightness = "bright" if means.mean() > 150 else ("dark" if means.mean() < 80 else "moderately bright")

    orientation = "landscape" if w > h else ("portrait" if h > w else "square")

    return (
        f"A {brightness}, {tone} {orientation} image "
        f"({w}x{h} pixels)"
    )


def load_and_analyze(path: str) -> Optional[Dict[str, Any]]:
    """Load an image from disk and return full analysis. One-shot convenience."""
    image = load_image(path)
    if image is None:
        return None
    result = analyze_image(image)
    result["path"] = str(path)
    return result
