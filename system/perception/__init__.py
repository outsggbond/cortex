# -*- coding: utf-8 -*-
"""Multi-modal perception — ASR, TTS, image analysis, encoders, language detection.

This package provides:
- Speech-to-text (ASR) via Whisper with Wav2Vec2 fallback
- Text-to-speech (TTS) cross-platform: SAPI5, pyttsx3, Piper, macOS say
- Image analysis: captioning (BLIP), classification (CLIP), statistics
- Multi-modal encoders: text, image, audio, video → fixed-dim vectors
- Language detection: CJK-aware with jieba tokenization + pinyin
- Perception graph system: multi-modal input → graph nodes → reasoning

All heavy model loading is lazy — importing this package is cheap.
"""

from __future__ import annotations

# ── ASR (speech-to-text) ───────────────────────────────────────────────────────
from system.perception.asr import (
    load_audio,
    transcribe,
    transcribe_file,
    transcribe_microphone,
    record_microphone,
    asr_available,
)

# ── TTS (text-to-speech) ───────────────────────────────────────────────────────
from system.perception.tts import (
    speak,
    speak_response,
    synthesize_to_file,
    tts_available,
    tts_engine_name,
)

# ── Image analysis ─────────────────────────────────────────────────────────────
from system.perception.image_analyzer import (
    load_image,
    describe_image,
    analyze_image,
    classify_image,
    load_and_analyze,
)

# ── Multi-modal encoders ───────────────────────────────────────────────────────
from system.perception.encoders import (
    encode_text,
    encode_image,
    encode_audio,
    encode_video,
    project_to_dim,
    ensure_float_array,
)

# ── Language detection & CJK support ───────────────────────────────────────────
from system.perception.language import (
    detect_language,
    has_cjk,
    tokenize_chinese,
    to_pinyin,
)

# ── Perception system (graph-based multi-modal reasoning) ──────────────────────
from system.perception.system import OptimizedPerceptionSystem

# ── Supporting classes ─────────────────────────────────────────────────────────
from system.perception.collector import InputCollector
from system.perception.data_pipeline import DataPipeline
from system.perception.local_models import LocalTextEncoder, LocalImageEncoder
from system.perception.reasoning import AdaptiveReasoningModule
from system.perception.resource_scheduler import ResourceScheduler

__all__ = [
    # ASR
    "load_audio",
    "transcribe",
    "transcribe_file",
    "transcribe_microphone",
    "record_microphone",
    "asr_available",
    # TTS
    "speak",
    "speak_response",
    "synthesize_to_file",
    "tts_available",
    "tts_engine_name",
    # Image
    "load_image",
    "describe_image",
    "analyze_image",
    "classify_image",
    "load_and_analyze",
    # Encoders
    "encode_text",
    "encode_image",
    "encode_audio",
    "encode_video",
    "project_to_dim",
    "ensure_float_array",
    # Language
    "detect_language",
    "has_cjk",
    "tokenize_chinese",
    "to_pinyin",
    # System
    "OptimizedPerceptionSystem",
    "InputCollector",
    "ResourceScheduler",
    "AdaptiveReasoningModule",
    "DataPipeline",
    "LocalTextEncoder",
    "LocalImageEncoder",
]
