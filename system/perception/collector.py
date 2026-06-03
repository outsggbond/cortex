from __future__ import annotations

import os
import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np


logger = logging.getLogger(__name__)


class InputCollector:
    def __init__(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        video_path: Optional[str] = None,
    ):
        self.text = text
        self.image_path = image_path
        self.audio_path = audio_path
        self.video_path = video_path

    def _load_image(self) -> Optional[np.ndarray]:
        if not self.image_path:
            return None
        try:
            from PIL import Image

            img = Image.open(self.image_path).convert("RGB")
            return np.array(img)
        except Exception:
            return None

    def _load_audio(self) -> Optional[np.ndarray]:
        if not self.audio_path:
            return None
        try:
            import wave

            with wave.open(self.audio_path, "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                data = np.frombuffer(frames, dtype=np.int16).astype("float32")
                if wf.getnchannels() > 1:
                    data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
                return data / 32768.0
        except Exception:
            return None

    def _load_video(self) -> Optional[np.ndarray]:
        if not self.video_path:
            return None
        path = self.video_path
        if path.endswith(".npy") and os.path.exists(path):
            try:
                return np.load(path)
            except Exception:
                return None
        # Try to decode common video files using OpenCV
        try:
            import cv2

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return None
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            if not frames:
                return None
            return np.stack(frames)
        except Exception:
            return None

    def _safe_text(self) -> Optional[str]:
        if self.text is not None:
            return self.text
        try:
            if os.isatty(0):
                return input("Enter text: ").strip()
        except Exception:
            logger.debug("collector: stdin text read failed", exc_info=True)
        return None

    def collect(self, modes: Tuple[str, ...]) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for m in modes:
            if m == "text":
                data["text"] = self._safe_text()
            elif m == "image":
                data["image"] = self._load_image()
            elif m == "audio":
                data["audio"] = self._load_audio()
            elif m == "video":
                data["video"] = self._load_video()
        return data
