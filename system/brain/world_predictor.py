from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from system.core.state_encoder import StateEncoder
if TYPE_CHECKING:
    from system.brain.world_model import WorldState


@dataclass
class DeltaTarget:
    errors: float = 0.0
    missing: float = 0.0
    perm: float = 0.0
    added: float = 0.0
    modified: float = 0.0
    removed: float = 0.0


@dataclass
class ModelBuffer:
    dim: int
    l2: float
    window: int
    min_samples: int
    X: List[np.ndarray] = field(default_factory=list)
    Y: List[np.ndarray] = field(default_factory=list)
    weights: np.ndarray = field(default_factory=lambda: np.zeros((6, 1), dtype="float32"))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype="float32"))
    trained: bool = False
    last_mse: float = 0.0

    def add_sample(self, x: np.ndarray, y: np.ndarray) -> None:
        self.X.append(x.astype("float32"))
        self.Y.append(y.astype("float32"))
        if len(self.X) > self.window:
            self.X.pop(0)
            self.Y.pop(0)

    def fit(self) -> None:
        if len(self.X) < self.min_samples:
            self.trained = False
            return
        X = np.stack(self.X, axis=0)
        Y = np.stack(self.Y, axis=0)
        ones = np.ones((X.shape[0], 1), dtype="float32")
        X_aug = np.concatenate([X, ones], axis=1)  # n x (dim+1)
        XtX = X_aug.T @ X_aug
        # L2 on weights only (exclude bias)
        for i in range(self.dim):
            XtX[i, i] += self.l2
        try:
            W = np.linalg.solve(XtX, X_aug.T @ Y)  # (dim+1) x 6
            self.weights = W[: self.dim].T.astype("float32")
            self.bias = W[self.dim].T.astype("float32")
            self.trained = True
            pred = X @ self.weights.T + self.bias
            self.last_mse = float(np.mean((pred - Y) ** 2))
        except Exception:
            self.trained = False

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.zeros(6, dtype="float32")
        return (self.weights @ x + self.bias).astype("float32")

    def confidence(self) -> float:
        if not self.trained:
            return 0.0
        n = len(self.X)
        size_factor = min(1.0, n / float(max(1, self.min_samples * 2)))
        mse_factor = 1.0 / (1.0 + float(self.last_mse))
        return max(0.0, min(1.0, size_factor * mse_factor))


class WorldPredictor:
    def __init__(self, dim: int | None = None, lr: float = 0.05):
        self.encoder = StateEncoder(dim=dim)
        self.lr = float(os.environ.get("WORLD_PRED_LR", str(lr)))
        self.dim = self.encoder.dim
        self.window = int(os.environ.get("WORLD_PRED_WINDOW", "200"))
        self.min_samples = int(os.environ.get("WORLD_PRED_MIN", "8"))
        self.l2 = float(os.environ.get("WORLD_PRED_L2", "0.1"))
        self.persist_window = os.environ.get("WORLD_PRED_PERSIST_WINDOW", "0") == "1"
        self.global_model = ModelBuffer(self.dim, self.l2, self.window, self.min_samples)
        self.action_models: Dict[str, ModelBuffer] = {}

    def update(self, before: WorldState, after: WorldState, action_sig: str = "") -> None:
        x = self._encode(before, action_sig)
        y = self._target_from_states(before, after)
        self.global_model.add_sample(x, y)
        self.global_model.fit()
        if action_sig:
            model = self.action_models.get(action_sig)
            if model is None:
                model = ModelBuffer(self.dim, self.l2, self.window, self.min_samples)
                self.action_models[action_sig] = model
            model.add_sample(x, y)
            model.fit()

    def predict(self, state: WorldState, action_sig: str = "") -> Tuple[Dict[str, float], float]:
        x = self._encode(state, action_sig)
        model = None
        if action_sig and action_sig in self.action_models and self.action_models[action_sig].trained:
            model = self.action_models[action_sig]
        elif self.global_model.trained:
            model = self.global_model
        if model is None:
            return {}, 0.0
        pred = model.predict(x)
        conf = model.confidence()
        return {
            "errors": float(pred[0]),
            "missing": float(pred[1]),
            "perm": float(pred[2]),
            "added": float(pred[3]),
            "modified": float(pred[4]),
            "removed": float(pred[5]),
        }, conf

    def save(self, path: str = "artifacts/memory/world_predictor.json") -> None:
        data = {
            "dim": int(self.dim),
            "lr": float(self.lr),
            "window": int(self.window),
            "min_samples": int(self.min_samples),
            "l2": float(self.l2),
            "global": self._dump_model(self.global_model),
            "actions": {k: self._dump_model(v) for k, v in self.action_models.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str = "artifacts/memory/world_predictor.json") -> bool:
        p = Path(path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.dim = int(data.get("dim", self.dim))
            self.window = int(data.get("window", self.window))
            self.min_samples = int(data.get("min_samples", self.min_samples))
            self.l2 = float(data.get("l2", self.l2))
            self.global_model = self._load_model(data.get("global", {}))
            self.action_models = {}
            for k, v in (data.get("actions", {}) or {}).items():
                self.action_models[k] = self._load_model(v)
            return True
        except Exception:
            return False

    def _encode(self, state: WorldState, action_sig: str) -> np.ndarray:
        vec = self.encoder.encode(state, state.goal)
        if action_sig:
            # action-conditional feature: hash into a few dims
            for salt in ("a", "b", "c"):
                idx = self._hash_to_index(f"{action_sig}:{salt}") % self.dim
                vec[idx] += 0.2
        return vec.astype("float32")

    def _dump_model(self, model: ModelBuffer) -> Dict[str, Any]:
        data = {
            "weights": model.weights.tolist(),
            "bias": model.bias.tolist(),
            "trained": model.trained,
            "last_mse": model.last_mse,
            "count": len(model.X),
        }
        if self.persist_window:
            data["X"] = [x.tolist() for x in model.X]
            data["Y"] = [y.tolist() for y in model.Y]
        return data

    def _load_model(self, data: Dict[str, Any]) -> ModelBuffer:
        model = ModelBuffer(self.dim, self.l2, self.window, self.min_samples)
        try:
            model.weights = np.array(data.get("weights", []), dtype="float32")
            model.bias = np.array(data.get("bias", []), dtype="float32")
            model.trained = bool(data.get("trained", False))
            model.last_mse = float(data.get("last_mse", 0.0))
            if model.weights.shape != (6, self.dim):
                model.weights = np.zeros((6, self.dim), dtype="float32")
                model.bias = np.zeros(6, dtype="float32")
                model.trained = False
            if self.persist_window:
                X = data.get("X", []) or []
                Y = data.get("Y", []) or []
                model.X = [np.array(x, dtype="float32") for x in X][-self.window :]
                model.Y = [np.array(y, dtype="float32") for y in Y][-self.window :]
            return model
        except Exception:
            return model

    @staticmethod
    def _hash_to_index(text: str) -> int:
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    @staticmethod
    def _target_from_states(before: WorldState, after: WorldState) -> np.ndarray:
        return np.array(
            [
                len(after.errors) - len(before.errors),
                len(after.missing_paths) - len(before.missing_paths),
                len(after.perm_paths) - len(before.perm_paths),
                after.added_count - before.added_count,
                after.modified_count - before.modified_count,
                after.removed_count - before.removed_count,
            ],
            dtype="float32",
        )
