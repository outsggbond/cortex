# -*- coding: utf-8 -*-
"""Speech-to-text (ASR) using Whisper with fallback to Wav2Vec2.

Whisper-tiny (~150 MB) handles 99 languages including Chinese.
Wav2Vec2 is ~315 MB but English-only.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy model handles ───────────────────────────────────────────────────────

_whisper_model = None
_whisper_loaded = False


def _get_whisper(model_size: str = "tiny"):
    """Load OpenAI Whisper model (lazy, singleton)."""
    global _whisper_model, _whisper_loaded
    if _whisper_loaded:
        return _whisper_model
    _whisper_loaded = True
    try:
        import whisper
        _whisper_model = whisper.load_model(model_size)
        logger.info("Whisper %s loaded for ASR", model_size)
        return _whisper_model
    except ImportError:
        logger.debug("openai-whisper not installed — try: pip install openai-whisper")
    except Exception as e:
        logger.debug("Whisper load failed: %s", e)
    return None


# ── Public API ───────────────────────────────────────────────────────────────

def load_audio(path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Load an audio file (WAV/MP3/etc) as float32 mono waveform at target_sr.

    Returns (samples,) array normalized to [-1, 1], or None on failure.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Audio file not found: %s", path)
        return None

    suffix = p.suffix.lower()

    # Try soundfile first (handles many formats)
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)  # mono
        if sr != target_sr:
            # Simple resample via linear interpolation
            import scipy.signal
            data = scipy.signal.resample(data, int(len(data) * target_sr / sr))
        return data.astype("float32")
    except ImportError:
        pass
    except Exception as e:
        logger.debug("soundfile load failed: %s", e)

    # WAV via standard library
    if suffix == ".wav":
        try:
            with wave.open(path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                data = np.frombuffer(frames, dtype=np.int16).astype("float32")
                if wf.getnchannels() > 1:
                    data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
                data = data / 32768.0
                if wf.getframerate() != target_sr:
                    try:
                        import scipy.signal
                        data = scipy.signal.resample(
                            data, int(len(data) * target_sr / wf.getframerate())
                        )
                    except ImportError:
                        pass
                return data.astype("float32")
        except Exception as e:
            logger.warning("WAV load failed: %s", e)
            return None

    # Fallback: try scipy.io.wavfile
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            data = data.astype("float32") / 32768.0
        elif data.dtype == np.int32:
            data = data.astype("float32") / 2147483648.0
        else:
            data = data.astype("float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            import scipy.signal
            data = scipy.signal.resample(data, int(len(data) * target_sr / sr))
        return data.astype("float32")
    except Exception as e:
        logger.warning("scipy wavfile load failed: %s", e)

    return None


def transcribe(audio: np.ndarray, sample_rate: int = 16000, language: str = "") -> Optional[str]:
    """Transcribe audio waveform to text using Whisper.

    Args:
        audio: float32 waveform, shape (samples,), normalized to [-1, 1]
        sample_rate: sample rate in Hz (default 16000)
        language: optional language code (e.g. 'zh', 'en') for better accuracy

    Returns transcribed text, or None on failure.
    """
    if audio is None or audio.size == 0:
        return None

    # Ensure float32
    audio = audio.astype("float32")

    # Try Whisper
    model = _get_whisper("tiny")
    if model is not None:
        try:
            # Whisper expects audio in [-1, 1] at 16kHz
            if sample_rate != 16000:
                try:
                    import scipy.signal
                    audio = scipy.signal.resample(
                        audio, int(len(audio) * 16000 / sample_rate)
                    )
                except ImportError:
                    pass
                sample_rate = 16000

            options = {}
            if language:
                options["language"] = language

            result = model.transcribe(audio, **options)
            text = str(result.get("text", "") or "").strip()
            if text:
                return text
        except Exception as e:
            logger.debug("Whisper transcription failed: %s", e)

    # Fallback: try Wav2Vec2 via transformers
    try:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )
        w2v_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import scipy.signal
            audio = scipy.signal.resample(
                audio, int(len(audio) * 16000 / sample_rate)
            )

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = w2v_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(predicted_ids)[0]
        if text and text.strip():
            return text.strip()
    except Exception as e:
        logger.debug("Wav2Vec2 fallback failed: %s", e)

    return None


def transcribe_file(path: str, language: str = "") -> Optional[str]:
    """One-shot: load an audio file and transcribe it."""
    audio = load_audio(path)
    if audio is None:
        return None
    return transcribe(audio, language=language)
