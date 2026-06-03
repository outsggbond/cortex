# -*- coding: utf-8 -*-
"""Text-to-speech (TTS) — cross-platform with multiple backends.

Windows: SAPI5 (built-in, no install needed)
Cross-platform: pyttsx3 (offline), Piper TTS (high quality)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Backend priority ─────────────────────────────────────────────────────────

_engine = None
_engine_name = ""


def _get_engine():
    """Lazy-load the best available TTS engine."""
    global _engine, _engine_name
    if _engine is not None:
        return _engine

    # 1. SAPI5 (Windows built-in — zero install)
    if sys.platform == "win32":
        try:
            import win32com.client
            _engine = win32com.client.Dispatch("SAPI.SpVoice")
            _engine_name = "sapi5"
            logger.info("TTS: SAPI5 ready")
            return _engine
        except ImportError:
            logger.debug("SAPI5 requires pywin32: pip install pywin32")
        except Exception as e:
            logger.debug("SAPI5 init failed: %s", e)

    # 2. pyttsx3 (cross-platform offline)
    try:
        import pyttsx3
        _engine = pyttsx3.init()
        _engine_name = "pyttsx3"
        logger.info("TTS: pyttsx3 ready")
        return _engine
    except ImportError:
        pass
    except Exception as e:
        logger.debug("pyttsx3 init failed: %s", e)

    # 3. Piper TTS (high quality, offline)
    try:
        piper_path = os.environ.get("PIPER_PATH", "piper")
        model_path = os.environ.get("PIPER_MODEL", "")
        if model_path:
            # Verify piper binary is callable
            import shutil
            if shutil.which(piper_path) or os.path.isfile(piper_path):
                _engine = piper_path  # store path for later use
                _engine_name = "piper"
                logger.info("TTS: Piper TTS ready (%s)", piper_path)
                return piper_path
            else:
                logger.debug("Piper TTS: binary not found at %s", piper_path)
        else:
            logger.debug("Piper TTS: PIPER_MODEL not set, skipping")
    except Exception as e:
        logger.debug("Piper TTS init failed: %s", e)

    # 4. say (macOS built-in)
    if sys.platform == "darwin":
        _engine_name = "say"
        return None

    _engine_name = "none"
    return None


# ── Public API ───────────────────────────────────────────────────────────────

def speak(text: str, *, rate: int = 180, volume: float = 1.0) -> bool:
    """Speak text aloud using the best available TTS engine.

    Returns True if speech was produced, False otherwise.
    """
    text = str(text or "").strip()
    if not text:
        return False

    engine = _get_engine()

    if _engine_name == "sapi5" and engine is not None:
        try:
            engine.Rate = rate
            engine.Volume = int(max(0, min(100, volume * 100)))
            engine.Speak(text)
            return True
        except Exception as e:
            logger.debug("SAPI5 speak failed: %s", e)

    if _engine_name == "pyttsx3" and engine is not None:
        try:
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as e:
            logger.debug("pyttsx3 speak failed: %s", e)

    if _engine_name == "piper":
        # Piper TTS — write to temp WAV, then play
        try:
            piper_path = os.environ.get("PIPER_PATH", "piper")
            model_path = os.environ.get("PIPER_MODEL", "")
            if model_path:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name
                cmd = [piper_path, "--model", model_path, "--output_file", wav_path]
                proc = subprocess.run(
                    cmd, input=text.encode("utf-8"),
                    capture_output=True, timeout=30,
                )
                if proc.returncode == 0:
                    _play_wav(wav_path)
                    return True
        except Exception as e:
            logger.debug("Piper TTS failed: %s", e)

    if _engine_name == "say":
        try:
            subprocess.run(["say", text], timeout=30)
            return True
        except Exception as e:
            logger.debug("say failed: %s", e)

    logger.debug("TTS: no engine available")
    return False


def speak_response(text: str, *, enabled: bool = True, **kwargs) -> bool:
    """Convenience: speak if enabled."""
    if not enabled:
        return False
    return speak(text, **kwargs)


def available() -> bool:
    """Check if any TTS engine is available."""
    _get_engine()
    return _engine_name != "none"


def engine_name() -> str:
    """Return the name of the active TTS engine."""
    _get_engine()
    return _engine_name


def tts_available() -> bool:
    """Alias for available()."""
    return available()


def tts_engine_name() -> str:
    """Alias for engine_name()."""
    return engine_name()


def synthesize_to_file(text: str, output_path: str, *, rate: int = 180, volume: float = 1.0) -> bool:
    """Synthesize speech to a WAV file without playing it.

    Uses pyttsx3's save_to_file if available, otherwise falls back to Piper.
    Returns True on success.
    """
    text = str(text or "").strip()
    if not text:
        return False

    _get_engine()

    if _engine_name == "pyttsx3" and _engine is not None:
        try:
            _engine.setProperty("rate", rate)
            _engine.setProperty("volume", volume)
            _engine.save_to_file(text, output_path)
            _engine.runAndWait()
            return True
        except Exception as e:
            logger.debug("pyttsx3 save_to_file failed: %s", e)

    if _engine_name == "piper":
        try:
            piper_path = os.environ.get("PIPER_PATH", "piper")
            model_path = os.environ.get("PIPER_MODEL", "")
            if model_path:
                cmd = [piper_path, "--model", model_path, "--output_file", output_path]
                proc = subprocess.run(
                    cmd, input=text.encode("utf-8"),
                    capture_output=True, timeout=30,
                )
                return proc.returncode == 0
        except Exception as e:
            logger.debug("Piper synthesize failed: %s", e)

    if _engine_name == "sapi5":
        # SAPI5 doesn't support save-to-file directly; use Piper fallback if available
        logger.debug("SAPI5: save_to_file not supported, use pyttsx3 or Piper")

    logger.debug("TTS synthesize_to_file: no suitable engine available")
    return False


# ── Internal ─────────────────────────────────────────────────────────────────

def _play_wav(path: str) -> None:
    """Play a WAV file cross-platform."""
    if sys.platform == "win32":
        try:
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)
        except Exception:
            pass
    elif sys.platform == "darwin":
        subprocess.run(["afplay", path], timeout=30)
    else:
        try:
            subprocess.run(["aplay", path], timeout=30)
        except Exception:
            subprocess.run(["paplay", path], timeout=30)
