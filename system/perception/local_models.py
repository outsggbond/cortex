from __future__ import annotations

from typing import Optional
import logging
import numpy as np

from system.perception.encoders import project_to_dim


logger = logging.getLogger(__name__)


class LocalTextEncoder:
    def __init__(self, model_path: str, dim: int, seed: int = 42):
        self.model_path = model_path
        self.dim = dim
        self.seed = seed
        self._model = None
        self._backend = None
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            try:
                self._model = SentenceTransformer(self.model_path, local_files_only=True)
            except TypeError:
                self._model = SentenceTransformer(self.model_path)
            self._backend = "sentence-transformers"
            return
        except Exception:
            logger.debug("local_models: sentence-transformers load failed", exc_info=True)
        try:
            from transformers import AutoTokenizer, AutoModel

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModel.from_pretrained(self.model_path, local_files_only=True)
            self._backend = "transformers"
        except Exception:
            self._model = None
            self._backend = None

    def available(self) -> bool:
        return self._model is not None

    def encode(self, text: str) -> np.ndarray:
        if not self._model:
            raise RuntimeError("LocalTextEncoder not available")
        if self._backend == "sentence-transformers":
            emb = self._model.encode([text])[0]
            return project_to_dim(np.array(emb), self.dim, self.seed)
        # transformers backend
        import torch

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            out = self._model(**inputs)
            pooled = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return project_to_dim(np.array(pooled), self.dim, self.seed)


class LocalImageEncoder:
    def __init__(self, model_path: str, dim: int, seed: int = 42):
        self.model_path = model_path
        self.dim = dim
        self.seed = seed
        self._model = None
        self._processor = None
        self._load()

    def _load(self) -> None:
        try:
            from transformers import CLIPProcessor, CLIPModel

            self._processor = CLIPProcessor.from_pretrained(self.model_path, local_files_only=True)
            self._model = CLIPModel.from_pretrained(self.model_path, local_files_only=True)
        except Exception:
            self._model = None
            self._processor = None

    def available(self) -> bool:
        return self._model is not None and self._processor is not None

    def encode(self, image_np: np.ndarray) -> np.ndarray:
        if not self._model or not self._processor:
            raise RuntimeError("LocalImageEncoder not available")
        from PIL import Image
        import torch

        img = Image.fromarray(image_np.astype("uint8"))
        inputs = self._processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = self._model.get_image_features(**inputs).squeeze(0).cpu().numpy()
        return project_to_dim(np.array(out), self.dim, self.seed)
