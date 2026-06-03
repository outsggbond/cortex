from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


# =========================
# Result
# =========================
@dataclass
class TagResult:
    label: str
    score: float


# =========================
# Label Loader
# =========================
def _load_labels(model_path: Path) -> List[str]:
    candidates = [
        model_path.with_suffix(model_path.suffix + ".labels.txt"),
        model_path.with_suffix(".labels.txt"),
        model_path.with_suffix(".classes.txt"),
        model_path.with_suffix(".labels.json"),
        model_path.parent / "labels.txt",
        model_path.parent / "labels.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return [str(x) for x in data]
                if isinstance(data, dict):
                    ordered = [None] * (max(int(k) for k in data.keys()) + 1)
                    for k, v in data.items():
                        ordered[int(k)] = str(v)
                    return [x for x in ordered if x is not None]
            else:
                lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
                return [ln for ln in lines if ln]
        except Exception as e:
            print("[Label Load Error]:", e)
    return []


# =========================
# Model Loader (升级1 ✅)
# =========================
class ModelLoader:
    def __init__(self, path: str, use_gpu: bool = False):
        self.path = Path(path) if path else None
        self.use_gpu = bool(use_gpu)

        self.backend = ""
        self.session = None

        self.input_name = None
        self.input_shape = None
        self.output_shape = None

        self._load()

    def _load(self):
        if not self.path or not self.path.exists():
            print("[ModelLoader] model path not found")
            return

        suffix = self.path.suffix.lower()

        # ONNX
        if suffix == ".onnx":
            try:
                import onnxruntime as ort

                providers = ["CPUExecutionProvider"]
                if self.use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

                sess = ort.InferenceSession(str(self.path), providers=providers)

                self.session = sess
                self.backend = "onnx"

                inputs = sess.get_inputs()
                outputs = sess.get_outputs()

                if inputs:
                    self.input_name = inputs[0].name
                    self.input_shape = inputs[0].shape

                if outputs:
                    self.output_shape = outputs[0].shape

            except Exception as e:
                print("[ModelLoader] ONNX load failed:", e)

        # TorchScript
        elif suffix in {".pt", ".pth", ".jit", ".ts"}:
            try:
                import torch

                self.session = torch.jit.load(str(self.path))
                self.session.eval()

                if self.use_gpu and torch.cuda.is_available():
                    self.session.to("cuda")

                self.backend = "torchscript"

            except Exception as e:
                print("[ModelLoader] Torch load failed:", e)

    def available(self) -> bool:
        return self.session is not None

    def predict(self, input_array: np.ndarray) -> Optional[np.ndarray]:
        if not self.available():
            return None

        try:
            if self.backend == "onnx":
                if not self.input_name:
                    return None
                out = self.session.run(None, {self.input_name: input_array})
                return np.asarray(out[0]) if out else None

            if self.backend == "torchscript":
                import torch

                inp = torch.from_numpy(input_array)

                if self.use_gpu and torch.cuda.is_available():
                    inp = inp.cuda()

                with torch.no_grad():
                    out = self.session(inp)

                if hasattr(out, "detach"):
                    return out.detach().cpu().numpy()

                return np.asarray(out)

        except Exception as e:
            print("[Predict Error]:", e)

        return None


# =========================
# Image Tagger
# =========================
class ImageTagger:
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
        top_k: int = 3,
        threshold: float = 0.75,
    ):
        self.model = ModelLoader(model_path, use_gpu=use_gpu)
        self.labels = _load_labels(Path(model_path)) if model_path else []
        self.top_k = int(top_k)
        self.threshold = float(threshold)

    def available(self) -> bool:
        return self.model.available()

    # 升级2 ✅ 动态输入尺寸
    def _get_input_size(self):
        shape = self.model.input_shape
        if shape and len(shape) >= 4:
            h = shape[-2] if isinstance(shape[-2], int) else 224
            w = shape[-1] if isinstance(shape[-1], int) else 224
            return h, w
        return 224, 224

    # 升级3 ✅ 更智能 preprocess
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        from PIL import Image

        h, w = self._get_input_size()

        img = Image.fromarray(image.astype("uint8"))
        img = img.resize((w, h))

        arr = np.asarray(img).astype("float32") / 255.0

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        arr = arr[:, :, :3]

        mean = np.array([0.485, 0.456, 0.406], dtype="float32")
        std = np.array([0.229, 0.224, 0.225], dtype="float32")

        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        arr = arr[None]

        return arr.astype("float32")

    # 升级4 ✅ 自动 sigmoid / softmax
    def _postprocess(self, logits: np.ndarray):
        logits = logits.reshape(-1)

        if np.max(logits) > 1.5:
            exp = np.exp(logits - np.max(logits))
            probs = exp / (np.sum(exp) + 1e-9)
        else:
            probs = 1 / (1 + np.exp(-logits))

        return probs

    def tag(self, image: np.ndarray) -> List[TagResult]:
        if not self.available() or image is None:
            return []

        inp = self._preprocess(image)
        out = self.model.predict(inp)

        if out is None:
            return []

        probs = self._postprocess(out)

        idxs = np.argsort(-probs)[: max(1, self.top_k)]

        results = []
        for idx in idxs:
            score = float(probs[idx])
            if score < self.threshold:
                continue

            label = self.labels[idx] if idx < len(self.labels) else str(idx)
            results.append(TagResult(label=label, score=score))

        return results


# =========================
# Audio Tagger（同步升级）
# =========================
class AudioTagger:
    def __init__(
        self,
        model_path: str,
        use_gpu: bool = False,
        top_k: int = 3,
        threshold: float = 0.75,
    ):
        self.model = ModelLoader(model_path, use_gpu=use_gpu)
        self.labels = _load_labels(Path(model_path)) if model_path else []
        self.top_k = int(top_k)
        self.threshold = float(threshold)

    def available(self) -> bool:
        return self.model.available()

    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        return audio.astype("float32")[None, :]

    def _postprocess(self, logits: np.ndarray):
        logits = logits.reshape(-1)

        if np.max(logits) > 1.5:
            exp = np.exp(logits - np.max(logits))
            return exp / (np.sum(exp) + 1e-9)
        else:
            return 1 / (1 + np.exp(-logits))

    def tag(self, audio: np.ndarray) -> List[TagResult]:
        if not self.available() or audio is None:
            return []

        inp = self._preprocess(audio)
        out = self.model.predict(inp)

        if out is None:
            return []

        probs = self._postprocess(out)

        idxs = np.argsort(-probs)[: max(1, self.top_k)]

        results = []
        for idx in idxs:
            score = float(probs[idx])
            if score < self.threshold:
                continue

            label = self.labels[idx] if idx < len(self.labels) else str(idx)
            results.append(TagResult(label=label, score=score))

        return results
