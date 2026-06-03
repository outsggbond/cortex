from __future__ import annotations

import json
import logging
import math
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy import sparse
except Exception:  # pragma: no cover - handled at runtime
    sparse = None

from system.automation._nanobrain_config import NanoBrainConfig
from system.automation._nanobrain_tokenizer import (
    _doc_tokens,
    _extract_action_tokens,
    _extract_path_tokens,
    _hash_choice,
    _is_path_token,
    _load_config,
    _load_templates,
    _read_jsonl,
    _read_text,
    tokenize,
)


logger = logging.getLogger(__name__)


class NanoBrain:
    def __init__(
        self,
        vocab: List[str],
        w_logic: "sparse.csr_matrix",
        w_emotion: "sparse.csr_matrix",
        w_cross: "sparse.csr_matrix",
        neighbors: List[List[Tuple[int, float]]],
        token_meta: Dict[str, Dict[str, float]],
        logic_nodes: List[str],
        emotion_nodes: List[str],
        config: NanoBrainConfig,
        templates: Dict[str, Any],
        cache_dir: Path,
    ) -> None:
        self.vocab = vocab
        self.index = {t: i for i, t in enumerate(vocab)}
        self.W_logic = w_logic
        self.W_emotion = w_emotion
        self.W_cross = w_cross
        self.neighbors = neighbors
        self.token_meta = token_meta
        self.logic_nodes = set(logic_nodes)
        self.emotion_nodes = set(emotion_nodes)
        self.logic_idx = {self.index[t] for t in self.logic_nodes if t in self.index}
        self.emotion_idx = {self.index[t] for t in self.emotion_nodes if t in self.index}
        self.config = config
        self.templates = templates
        self.cache_dir = cache_dir
        self.delta_path = self._resolve_path(self.config.delta_path, "synapse_delta.json")
        self.profile_path = self._resolve_path(self.config.profile_path, "user_profile.json")
        self.counts: Dict[str, int] = {}
        self.pair_counts: Dict[str, int] = {}
        self.total_obs: int = 0
        self.delta_weights: Dict[str, float] = {}
        self.user_profile: Dict[str, Any] = {"token_counts": {}, "emotion_counts": {}, "total": 0}
        self._persist_lock = threading.Lock()
        self._persist_inflight = False
        self._persist_dirty = False
        self._queue = queue.Queue(maxsize=int(self.config.queue_maxsize))
        self._worker_started = False
        self._hot_update_message = ""
        self._damped_keys = [
            "empathy",
            "threshold",
            "tau",
            "base_gain",
            "decay",
            "fluidity",
            "gravity",
            "logic_weight",
            "emotion_weight",
            "cross_weight",
        ]
        self._active_params: Dict[str, float] = {}
        self._target_params: Dict[str, float] = {}
        self._init_damping()

    @staticmethod
    def _resolve_path(path_value: str, default_name: str) -> Path:
        raw = path_value or default_name
        p = Path(raw)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p

    def _start_worker(self) -> None:
        if self._worker_started:
            return

        def worker() -> None:
            batch: List[Tuple[str, List[str]]] = []
            last_flush = time.time()
            while True:
                interval = float(self.config.batch_interval)
                target = max(1, int(self.config.batch_size))
                timeout = max(0.2, interval / 2.0)
                try:
                    item = self._queue.get(timeout=timeout)
                    if item is not None:
                        batch.append(item)
                except queue.Empty:
                    pass
                now = time.time()
                should_flush = False
                if batch and len(batch) >= target:
                    should_flush = True
                elif batch and (now - last_flush) >= interval:
                    should_flush = True
                if should_flush:
                    try:
                        self._process_batch(batch)
                    except Exception:
                        logger.debug("NanoBrain batch update failed", exc_info=True)
                    batch = []
                    last_flush = now

        self._worker_started = True
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    @classmethod
    def load_or_build(
        cls,
        exp_path: str | Path = "artifacts/memory/experience_store.jsonl",
        cache_dir: str | Path = "memory",
        config_path: str | Path = "config/nanobrain.json",
        template_path: str | Path = "config/nanobrain_templates.json",
    ) -> "NanoBrain":
        if sparse is None:
            raise RuntimeError("NanoBrain requires scipy. Please install scipy.")
        cfg = _load_config(Path(config_path))
        templates = _load_templates(Path(template_path))
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        graph_path = cache_dir / "nanobrain_graph_logic.npz"
        graph_emotion = cache_dir / "nanobrain_graph_emotion.npz"
        graph_cross = cache_dir / "nanobrain_graph_cross.npz"
        vocab_path = cache_dir / "nanobrain_vocab.json"
        meta_path = cache_dir / "nanobrain_meta.json"
        exp = Path(exp_path)
        config_mtime = 0.0
        try:
            config_mtime = max(Path(config_path).stat().st_mtime, Path(template_path).stat().st_mtime)
        except Exception:
            config_mtime = 0.0
        exp_mtime = exp.stat().st_mtime if exp.exists() else 0.0
        cache_mtime = min(
            graph_path.stat().st_mtime if graph_path.exists() else 0.0,
            graph_emotion.stat().st_mtime if graph_emotion.exists() else 0.0,
            graph_cross.stat().st_mtime if graph_cross.exists() else 0.0,
            vocab_path.stat().st_mtime if vocab_path.exists() else 0.0,
            meta_path.stat().st_mtime if meta_path.exists() else 0.0,
        )
        if graph_path.exists() and graph_emotion.exists() and graph_cross.exists() and vocab_path.exists() and meta_path.exists():
            if cache_mtime >= max(exp_mtime, config_mtime):
                try:
                    vocab = json.loads(_read_text(vocab_path))
                    meta = json.loads(_read_text(meta_path))
                    w_logic = sparse.load_npz(graph_path)
                    w_emotion = sparse.load_npz(graph_emotion)
                    w_cross = sparse.load_npz(graph_cross)
                    combined = w_logic + w_emotion + w_cross
                    neighbors = cls._build_neighbors(combined, cfg.max_neighbors)
                    logic_nodes = meta.get("logic_nodes") or []
                    emotion_nodes = meta.get("emotion_nodes") or []
                    token_meta = meta.get("token_meta") or {}
                    obj = cls(
                        vocab,
                        w_logic,
                        w_emotion,
                        w_cross,
                        neighbors,
                        token_meta,
                        logic_nodes,
                        emotion_nodes,
                        cfg,
                        templates,
                        cache_dir,
                    )
                    obj._load_delta(cache_mtime=cache_mtime)
                    obj._load_profile()
                    obj._start_worker()
                    return obj
                except Exception:
                    logger.info("NanoBrain cache invalid, rebuilding graph.")
        items = _read_jsonl(exp)
        vocab, w_logic, w_emotion, w_cross, neighbors, meta = cls._build_from_items(items, cfg)
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        sparse.save_npz(graph_path, w_logic)
        sparse.save_npz(graph_emotion, w_emotion)
        sparse.save_npz(graph_cross, w_cross)
        obj = cls(
            vocab,
            w_logic,
            w_emotion,
            w_cross,
            neighbors,
            meta.get("token_meta") or {},
            meta.get("logic_nodes") or [],
            meta.get("emotion_nodes") or [],
            cfg,
            templates,
            cache_dir,
        )
        obj._load_delta(cache_mtime=cache_mtime)
        obj._load_profile()
        obj._start_worker()
        return obj

    @staticmethod
    def _build_neighbors(matrix: "sparse.csr_matrix", max_neighbors: int) -> List[List[Tuple[int, float]]]:
        neighbors: List[List[Tuple[int, float]]] = []
        csr = matrix.tocsr()
        for i in range(csr.shape[0]):
            row = csr.getrow(i)
            cols = row.indices
            data = row.data
            pairs = sorted(zip(cols.tolist(), data.tolist()), key=lambda x: x[1], reverse=True)
            if max_neighbors > 0:
                pairs = pairs[:max_neighbors]
            neighbors.append(pairs)
        return neighbors

    @staticmethod
    def _build_from_items(
        items: List[Dict[str, Any]],
        cfg: NanoBrainConfig,
    ) -> Tuple[
        List[str],
        "sparse.csr_matrix",
        "sparse.csr_matrix",
        "sparse.csr_matrix",
        List[List[Tuple[int, float]]],
        Dict[str, Any],
    ]:
        token_counts: Dict[str, int] = {}
        docs: List[List[str]] = []
        token_meta: Dict[str, Dict[str, float]] = {}
        for item in items:
            tokens = _doc_tokens(item, cfg.max_tokens_per_doc)
            if not tokens:
                continue
            uniq = list(dict.fromkeys(tokens))
            docs.append(uniq)
            for t in uniq:
                token_counts[t] = token_counts.get(t, 0) + 1
            for t in _extract_action_tokens(item):
                token_meta.setdefault(t, {}).setdefault("action", 0.0)
                token_meta[t]["action"] += 1.0
            for t in item.get("error_types") or []:
                if not isinstance(t, str):
                    continue
                token_meta.setdefault(t, {}).setdefault("risk", 0.0)
                token_meta[t]["risk"] += 1.0
            for p in _extract_path_tokens(item):
                name = Path(str(p)).name
                if not name:
                    continue
                token_meta.setdefault(name, {}).setdefault("target", 0.0)
                token_meta[name]["target"] += 1.0

        vocab = sorted(token_counts.keys(), key=lambda k: (-token_counts[k], k))
        for special in cfg.special_nodes:
            if special not in vocab:
                vocab.append(special)
        for node in cfg.logic_nodes:
            if node and node not in vocab:
                vocab.append(node)
        for node in cfg.emotion_nodes:
            if node and node not in vocab:
                vocab.append(node)
        vocab = vocab[: cfg.max_vocab]
        index = {t: i for i, t in enumerate(vocab)}

        logic_nodes = set([t for t in cfg.logic_nodes if t in index])
        emotion_nodes = set([t for t in cfg.emotion_nodes if t in index])
        if not logic_nodes:
            logic_nodes = set(vocab)
        if emotion_nodes:
            logic_nodes = logic_nodes.difference(emotion_nodes)

        pair_counts: Dict[Tuple[int, int], int] = {}
        for doc in docs:
            filtered = [t for t in doc if t in index]
            if len(filtered) > cfg.max_tokens_per_doc:
                filtered = filtered[: cfg.max_tokens_per_doc]
            for i in range(len(filtered)):
                for j in range(i + 1, len(filtered)):
                    a = index[filtered[i]]
                    b = index[filtered[j]]
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1

        rows: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(vocab))}
        total_docs = max(1, len(docs))
        for (a, b), c in pair_counts.items():
            pa = token_counts.get(vocab[a], 1) / total_docs
            pb = token_counts.get(vocab[b], 1) / total_docs
            pxy = c / total_docs
            if pa <= 0 or pb <= 0 or pxy <= 0:
                continue
            pmi = math.log(pxy / (pa * pb))
            if pmi <= cfg.pmi_min:
                continue
            w = pmi * cfg.pmi_scale
            if w < cfg.min_edge_weight:
                continue
            rows[a].append((b, w))
            rows[b].append((a, w))

        logic_idx: List[int] = []
        logic_col: List[int] = []
        logic_data: List[float] = []
        emo_idx: List[int] = []
        emo_col: List[int] = []
        emo_data: List[float] = []
        cross_idx: List[int] = []
        cross_col: List[int] = []
        cross_data: List[float] = []

        for i in range(len(vocab)):
            for j, w in rows.get(i, []):
                src = vocab[i]
                dst = vocab[j]
                if src in emotion_nodes and dst in emotion_nodes:
                    emo_idx.append(i)
                    emo_col.append(j)
                    emo_data.append(float(w))
                elif src in logic_nodes and dst in logic_nodes:
                    logic_idx.append(i)
                    logic_col.append(j)
                    logic_data.append(float(w))
                else:
                    cross_idx.append(i)
                    cross_col.append(j)
                    cross_data.append(float(w))

        for syn in cfg.cross_synapses:
            if not isinstance(syn, dict):
                continue
            src = syn.get("src")
            dst = syn.get("dst")
            if not src or not dst:
                continue
            if src not in index or dst not in index:
                continue
            weight = syn.get("weight", 1.0)
            try:
                weight = float(weight)
            except Exception:
                weight = 1.0
            cross_idx.append(index[src])
            cross_col.append(index[dst])
            cross_data.append(weight)

        w_logic = sparse.csr_matrix((logic_data, (logic_idx, logic_col)), shape=(len(vocab), len(vocab)))
        w_emotion = sparse.csr_matrix((emo_data, (emo_idx, emo_col)), shape=(len(vocab), len(vocab)))
        w_cross = sparse.csr_matrix((cross_data, (cross_idx, cross_col)), shape=(len(vocab), len(vocab)))

        combined = w_logic + w_emotion + w_cross
        neighbors = NanoBrain._build_neighbors(combined, cfg.max_neighbors)
        meta = {
            "token_meta": token_meta,
            "logic_nodes": sorted(list(logic_nodes)),
            "emotion_nodes": sorted(list(emotion_nodes)),
        }
        return vocab, w_logic, w_emotion, w_cross, neighbors, meta

    def _gain(self, text: str) -> float:
        low = text.lower()
        base = float(self._param("base_gain"))
        for kw in self.config.critical_keywords:
            if kw and kw.lower() in low:
                return base * float(self.config.critical_gain)
        return base * float(self.config.default_gain)

    def _indices(self, tokens: Iterable[str]) -> List[int]:
        idxs: List[int] = []
        unk = self.index.get("UNK")
        for t in tokens:
            if t in self.index:
                idxs.append(self.index[t])
            elif unk is not None:
                idxs.append(unk)
        return idxs

    def _category(self, idx: int) -> str:
        if idx in self.emotion_idx:
            return "emotion"
        if idx in self.logic_idx:
            return "logic"
        return "logic"

    def _category_token(self, token: str) -> str:
        if token in self.emotion_nodes:
            return "emotion"
        return "logic"

    def _refresh_indices(self) -> None:
        self.logic_idx = {self.index[t] for t in self.logic_nodes if t in self.index}
        self.emotion_idx = {self.index[t] for t in self.emotion_nodes if t in self.index}

    def _init_damping(self) -> None:
        self._active_params = {}
        self._target_params = {}
        for key in self._damped_keys:
            val = getattr(self.config, key, None)
            if isinstance(val, (int, float)):
                fval = float(val)
                self._active_params[key] = fval
                self._target_params[key] = fval

    def _update_target_params(self) -> None:
        for key in self._damped_keys:
            val = getattr(self.config, key, None)
            if isinstance(val, (int, float)):
                self._target_params[key] = float(val)
                if key not in self._active_params:
                    self._active_params[key] = float(val)

    def _param(self, key: str) -> float:
        if key in self._active_params:
            return float(self._active_params[key])
        try:
            return float(getattr(self.config, key))
        except Exception:
            return 0.0

    def _apply_damping(self) -> None:
        eta = float(self.config.damping_eta)
        if eta <= 0:
            for key in self._damped_keys:
                if key in self._target_params:
                    self._active_params[key] = float(self._target_params[key])
            return
        for key in self._damped_keys:
            if key not in self._target_params:
                continue
            cur = float(self._active_params.get(key, self._target_params[key]))
            tgt = float(self._target_params[key])
            if math.isclose(cur, tgt, rel_tol=0.0, abs_tol=1e-4):
                self._active_params[key] = tgt
                continue
            cur = cur + eta * (tgt - cur)
            if key == "empathy":
                cur = max(0.0, min(1.0, cur))
            self._active_params[key] = cur

    @staticmethod
    def _infer_mode(old_vals: Dict[str, float], new_vals: Dict[str, float]) -> str:
        empathy_delta = float(new_vals.get("empathy", 0.0)) - float(old_vals.get("empathy", 0.0))
        threshold_delta = float(new_vals.get("threshold", 0.0)) - float(old_vals.get("threshold", 0.0))
        if empathy_delta <= -0.15 or threshold_delta >= 0.15:
            return "cool"
        if empathy_delta >= 0.15 or threshold_delta <= -0.15:
            return "warm"
        return "steady"

    def _propagate(self, A: np.ndarray, matrix: "sparse.csr_matrix") -> np.ndarray:
        if matrix is None:
            return A
        decay = float(self._param("decay")) * float(self._param("fluidity"))
        gravity = float(self._param("gravity"))
        tau = float(self._param("tau"))
        cur = A
        for _ in range(max(1, int(self.config.max_iters))):
            nxt = matrix.transpose().dot(cur)
            nxt = np.asarray(nxt).reshape(-1) * decay * gravity
            nxt = np.maximum(0.0, nxt - tau)
            if nxt.max() > 0:
                nxt = nxt / float(nxt.max())
            if np.linalg.norm(nxt - cur) < float(self.config.epsilon):
                cur = nxt
                break
            cur = nxt
        return cur

    def _load_delta(self, cache_mtime: float = 0.0) -> None:
        if not self.delta_path.exists():
            return
        try:
            data = json.loads(_read_text(self.delta_path))
        except Exception:
            return
        counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
        pairs = data.get("pairs") if isinstance(data.get("pairs"), dict) else {}
        deltas = data.get("delta_weights") if isinstance(data.get("delta_weights"), dict) else {}
        total = data.get("total", 0)
        try:
            self.total_obs = int(total)
        except Exception:
            self.total_obs = 0
        self.counts = {str(k): int(v) for k, v in counts.items() if str(k)}
        self.pair_counts = {str(k): int(v) for k, v in pairs.items() if str(k)}
        self.delta_weights = {str(k): float(v) for k, v in deltas.items() if str(k)}
        try:
            delta_mtime = self.delta_path.stat().st_mtime
        except Exception:
            delta_mtime = 0.0
        if not self.delta_weights:
            return
        if delta_mtime <= cache_mtime:
            return
        mat_logic = self.W_logic.tolil()
        mat_emotion = self.W_emotion.tolil()
        mat_cross = self.W_cross.tolil()
        for key, delta in self.delta_weights.items():
            if "|" not in key:
                continue
            a, b = key.split("|", 1)
            if a not in self.index or b not in self.index:
                continue
            i = self.index[a]
            j = self.index[b]
            cat_a = self._category_token(a)
            cat_b = self._category_token(b)
            if cat_a == "emotion" and cat_b == "emotion":
                mat = mat_emotion
            elif cat_a == "logic" and cat_b == "logic":
                mat = mat_logic
            else:
                mat = mat_cross
            mat[i, j] = float(mat[i, j]) + float(delta)
            mat[j, i] = float(mat[j, i]) + float(delta)
        self.W_logic = mat_logic.tocsr()
        self.W_emotion = mat_emotion.tocsr()
        self.W_cross = mat_cross.tocsr()
        combined = self.W_logic + self.W_emotion + self.W_cross
        self.neighbors = self._build_neighbors(combined, self.config.max_neighbors)

    def _save_delta(self) -> None:
        data = {
            "total": self.total_obs,
            "counts": self.counts,
            "pairs": self.pair_counts,
            "delta_weights": self.delta_weights,
            "ts": time.time(),
        }
        try:
            self.delta_path.parent.mkdir(parents=True, exist_ok=True)
            self.delta_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("NanoBrain delta save failed", exc_info=True)

    def _load_profile(self) -> None:
        if not self.profile_path.exists():
            return
        try:
            data = json.loads(_read_text(self.profile_path))
        except Exception:
            return
        token_counts = data.get("token_counts") if isinstance(data.get("token_counts"), dict) else {}
        emotion_counts = data.get("emotion_counts") if isinstance(data.get("emotion_counts"), dict) else {}
        total = data.get("total", 0)
        self.user_profile = {
            "token_counts": {str(k): int(v) for k, v in token_counts.items() if str(k)},
            "emotion_counts": {str(k): int(v) for k, v in emotion_counts.items() if str(k)},
            "total": int(total) if str(total).isdigit() else 0,
        }

    def _save_profile(self) -> None:
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            self.profile_path.write_text(json.dumps(self.user_profile, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("NanoBrain profile save failed", exc_info=True)

    def _persist_async(self) -> None:
        if self._persist_inflight:
            self._persist_dirty = True
            return

        def worker() -> None:
            try:
                while True:
                    with self._persist_lock:
                        self._save()
                        self._save_delta()
                    if self._persist_dirty:
                        self._persist_dirty = False
                        continue
                    break
            finally:
                self._persist_inflight = False

        self._persist_inflight = True
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def refresh_config(self) -> Optional[str]:
        try:
            new_cfg = _load_config(Path("config/nanobrain.json"))
        except Exception:
            return None
        with self._persist_lock:
            old = {k: float(self._active_params.get(k, getattr(self.config, k, 0.0))) for k in self._damped_keys}
            self.config = new_cfg
            self._update_target_params()
            try:
                self._queue.maxsize = int(self.config.queue_maxsize)
            except Exception:
                pass
        new = {k: float(self._target_params.get(k, old.get(k, 0.0))) for k in self._damped_keys}
        max_delta = 0.0
        for key, val in new.items():
            max_delta = max(max_delta, abs(float(val) - float(old.get(key, 0.0))))
        if max_delta >= float(self.config.damping_alert):
            mode = self._infer_mode(old, new)
            msg = f"[System] Cognitive shift detected; smoothing toward '{mode}' profile."
        else:
            msg = (
                f"[System] Params hot-reloaded: "
                f"empathy {old.get('empathy', 0.0):.2f}->{new.get('empathy', 0.0):.2f}, "
                f"threshold {old.get('threshold', 0.0):.2f}->{new.get('threshold', 0.0):.2f}, "
                f"tau {old.get('tau', 0.0):.2f}->{new.get('tau', 0.0):.2f}"
            )
        self._hot_update_message = msg
        return msg

    def consume_hot_update_message(self) -> str:
        msg = self._hot_update_message
        self._hot_update_message = ""
        return msg

    def observe_user_message(self, message: str) -> None:
        self._apply_damping()
        tokens = tokenize(message, max_tokens=int(self.config.profile_max_tokens or self.config.max_tokens_per_doc))
        if not tokens:
            return
        profile = self.user_profile or {"token_counts": {}, "emotion_counts": {}, "total": 0}
        token_counts = profile.get("token_counts", {})
        emotion_counts = profile.get("emotion_counts", {})
        total = int(profile.get("total", 0) or 0) + 1
        for t in tokens:
            token_counts[t] = int(token_counts.get(t, 0)) + 1
        lexicon = self.config.emotion_lexicon or {}
        for emotion, words in lexicon.items():
            if not isinstance(words, list):
                continue
            for w in words:
                if w in tokens:
                    emotion_counts[emotion] = int(emotion_counts.get(emotion, 0)) + 1
        profile["token_counts"] = token_counts
        profile["emotion_counts"] = emotion_counts
        profile["total"] = total
        self.user_profile = profile
        if float(self._param("empathy")) >= 0.7:
            top_tokens = sorted(token_counts.items(), key=lambda kv: kv[1], reverse=True)[:6]
            for tok, _cnt in top_tokens:
                if tok in self.index and tok not in self.logic_nodes:
                    self.emotion_nodes.add(tok)
            self._refresh_indices()
        self._save_profile()

    def _user_emotion_bias(self) -> List[str]:
        profile = self.user_profile or {}
        emotion_counts = profile.get("emotion_counts", {})
        if not emotion_counts:
            return []
        top = sorted(emotion_counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
        return [k for k, _v in top if k in self.emotion_nodes]

    def _experience_tokens(self, goal: str, plan_steps: List[str]) -> List[str]:
        text = goal or ""
        if plan_steps:
            text += " " + " ".join([s for s in plan_steps if s])
        return tokenize(text, max_tokens=self.config.max_tokens_per_doc)

    def on_experience_gained(self, goal: str, plan_steps: List[str], success: bool = True) -> None:
        if not success:
            return
        item = (goal, plan_steps)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.debug("NanoBrain update queue full; dropping experience")

    def _trim_pairs(self) -> None:
        limit = int(self.config.max_delta_pairs)
        if limit <= 0:
            return
        if len(self.pair_counts) <= limit:
            return
        items = sorted(self.pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        self.pair_counts = {k: v for k, v in items}
        if self.delta_weights:
            keep = set(self.pair_counts.keys())
            self.delta_weights = {k: v for k, v in self.delta_weights.items() if k in keep}

    def _apply_weight_decay(self) -> None:
        decay = float(self.config.weight_decay)
        if decay <= 0:
            return
        decay = min(0.5, max(0.0, decay))
        self.W_logic = self.W_logic.multiply(1.0 - decay).tocsr()
        self.W_emotion = self.W_emotion.multiply(1.0 - decay).tocsr()
        self.W_cross = self.W_cross.multiply(1.0 - decay).tocsr()

    def _apply_pmi_updates(self, pair_keys: Iterable[str]) -> None:
        beta = float(self.config.pmi_beta)
        if beta <= 0:
            return
        mat_logic = self.W_logic.tolil()
        mat_emotion = self.W_emotion.tolil()
        mat_cross = self.W_cross.tolil()
        for key in pair_keys:
            if "|" not in key:
                continue
            a, b = key.split("|", 1)
            if a not in self.index or b not in self.index:
                continue
            ci = max(1, int(self.counts.get(a, 1)))
            cj = max(1, int(self.counts.get(b, 1)))
            cij = max(1, int(self.pair_counts.get(key, 1)))
            target = (float(self.total_obs) * float(cij)) / (float(ci) * float(cj))
            if target <= 0:
                continue
            ia = self.index[a]
            ib = self.index[b]
            cat_a = self._category_token(a)
            cat_b = self._category_token(b)
            if cat_a == "emotion" and cat_b == "emotion":
                mat = mat_emotion
            elif cat_a == "logic" and cat_b == "logic":
                mat = mat_logic
            else:
                mat = mat_cross
            cur = float(mat[ia, ib]) if mat[ia, ib] is not None else 0.0
            delta = beta * (target - cur)
            if abs(delta) < 1e-6:
                continue
            new_val = cur + delta
            if new_val < float(self.config.min_edge_weight):
                new_val = 0.0
            mat[ia, ib] = new_val
            mat[ib, ia] = new_val
            self.delta_weights[key] = float(self.delta_weights.get(key, 0.0)) + float(delta)
        self.W_logic = mat_logic.tocsr()
        self.W_emotion = mat_emotion.tocsr()
        self.W_cross = mat_cross.tocsr()
        self.W_logic.eliminate_zeros()
        self.W_emotion.eliminate_zeros()
        self.W_cross.eliminate_zeros()

    def _process_batch(self, batch: List[Tuple[str, List[str]]]) -> None:
        if not batch:
            return
        touched_pairs: set[str] = set()
        with self._persist_lock:
            self._apply_weight_decay()
            for goal, steps in batch:
                tokens = self._experience_tokens(goal, steps)
                if not tokens:
                    continue
                uniq = list(dict.fromkeys([t for t in tokens if t in self.index]))
                if not uniq:
                    continue
                self.total_obs = int(self.total_obs) + 1
                for t in uniq:
                    self.counts[t] = int(self.counts.get(t, 0)) + 1
                for i in range(len(uniq)):
                    for j in range(i + 1, len(uniq)):
                        a, b = uniq[i], uniq[j]
                        key = f"{a}|{b}" if a <= b else f"{b}|{a}"
                        self.pair_counts[key] = int(self.pair_counts.get(key, 0)) + 1
                        touched_pairs.add(key)
            self._trim_pairs()
            self._apply_pmi_updates(touched_pairs)
            for goal, steps in batch:
                self.learn_from_success(goal, steps, score=1.0, save=False, apply_decay=False)
            combined = self.W_logic + self.W_emotion + self.W_cross
            self.neighbors = self._build_neighbors(combined, self.config.max_neighbors)
        self._persist_async()

    def stimulate(self, text: str, extra_tokens: Optional[List[str]] = None) -> Tuple[np.ndarray, List[int]]:
        self._apply_damping()
        tokens = tokenize(text, max_tokens=self.config.max_tokens_per_doc)
        if extra_tokens:
            tokens = tokens + extra_tokens
        idxs = self._indices(tokens)
        if not idxs:
            return np.zeros(len(self.vocab), dtype=np.float32), []
        A = np.zeros(len(self.vocab), dtype=np.float32)
        gain = self._gain(text)
        for i in idxs:
            A[i] = max(A[i], gain)
        A_logic = self._propagate(A, self.W_logic)
        cross_seed = self.W_cross.transpose().dot(A_logic) if self.W_cross is not None else A_logic
        cross_seed = np.asarray(cross_seed).reshape(-1)
        empathy = float(self._param("empathy"))
        A_emotion = self._propagate(A + empathy * cross_seed, self.W_emotion)
        logic_w = float(self._param("logic_weight"))
        emo_w = float(self._param("emotion_weight"))
        cross_w = float(self._param("cross_weight"))
        combined = (logic_w * A_logic) + (emo_w * A_emotion) + (cross_w * cross_seed)
        if combined.max() > 0:
            combined = combined / float(combined.max())
        return combined, idxs

    def _top_tokens(self, activation: np.ndarray, exclude: Optional[Iterable[int]] = None) -> List[Tuple[str, float]]:
        if activation.size == 0:
            return []
        exclude_set = set(exclude or [])
        pairs: List[Tuple[int, float]] = []
        for i, val in enumerate(activation.tolist()):
            if i in exclude_set:
                continue
            if val <= float(self._param("threshold")):
                continue
            pairs.append((i, float(val)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top = pairs[: max(1, int(self.config.top_k))]
        return [(self.vocab[i], v) for i, v in top]

    def _intensity(self, activation: np.ndarray) -> float:
        if activation.size == 0:
            return 0.0
        vals = sorted([float(v) for v in activation.tolist() if v > 0.0], reverse=True)
        if not vals:
            return 0.0
        k = min(6, len(vals))
        return float(sum(vals[:k]) / k)

    def _template(self, role: str, intensity: float, tone: str = "neutral") -> str:
        group = self.templates.get(role)
        if isinstance(group, dict):
            if tone and tone in group:
                items = group.get(tone) or []
                return _hash_choice([str(x) for x in items if str(x).strip()], f"{role}:{tone}:{intensity:.3f}")
            low = float(self.config.intensity_low)
            high = float(self.config.intensity_high)
            if intensity >= high:
                items = group.get("high") or group.get("default") or []
            elif intensity <= low:
                items = group.get("low") or group.get("default") or []
            else:
                items = group.get("mid") or group.get("default") or []
        elif isinstance(group, list):
            items = group
        else:
            items = []
        return _hash_choice([str(x) for x in items if str(x).strip()], f"{role}:{intensity:.3f}")

    def _tone_mode(self) -> str:
        profile = self.user_profile or {}
        token_counts = profile.get("token_counts", {})
        total = sum(int(v) for v in token_counts.values() if isinstance(v, int))
        if total < 6:
            return "neutral"
        prof_terms = self.config.emotion_lexicon.get("professional", []) if isinstance(self.config.emotion_lexicon, dict) else []
        casual_terms = self.config.emotion_lexicon.get("casual", []) if isinstance(self.config.emotion_lexicon, dict) else []
        prof_hits = sum(int(token_counts.get(t, 0)) for t in prof_terms)
        casual_hits = sum(int(token_counts.get(t, 0)) for t in casual_terms)
        if prof_hits / max(1, total) >= 0.08:
            return "professional"
        if casual_hits / max(1, total) >= 0.08:
            return "casual"
        return "neutral"

    def _role_for(self, token: str) -> str:
        if token in self.emotion_nodes:
            return "emotion"
        meta = self.token_meta.get(token) or {}
        if not meta:
            if _is_path_token(token):
                return "target"
            return "concept"
        best = max(meta.items(), key=lambda kv: kv[1])
        return best[0]

    def _build_path(self, src_idxs: List[int], target_idx: int) -> Optional[List[str]]:
        if not self.neighbors:
            return None
        import heapq

        max_depth = max(1, int(self.config.path_depth))
        eps = 1e-6
        heap: List[Tuple[float, int, List[int], int]] = []
        for s in src_idxs:
            heapq.heappush(heap, (0.0, s, [s], 0))
        seen: Dict[int, float] = {}
        while heap:
            cost, node, path, depth = heapq.heappop(heap)
            if node == target_idx:
                return [self.vocab[i] for i in path]
            if depth >= max_depth:
                continue
            if node in seen and seen[node] <= cost:
                continue
            seen[node] = cost
            for nxt, w in self.neighbors[node]:
                if nxt in path:
                    continue
                edge_cost = 1.0 / (float(w) + eps)
                heapq.heappush(heap, (cost + edge_cost, nxt, path + [nxt], depth + 1))
        return None

    def generate(self, message: str, memories: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None) -> str:
        context = context or {}
        mem_text = " ".join([m for m in (memories or []) if m])
        extra_tokens: List[str] = []
        if "faq_answer" in context and isinstance(context["faq_answer"], str):
            extra_tokens = tokenize(context["faq_answer"], max_tokens=24)
        bias = self._user_emotion_bias()
        if bias:
            extra_tokens = extra_tokens + bias
        activation, idxs = self.stimulate(message + " " + mem_text, extra_tokens=extra_tokens)
        intensity = self._intensity(activation)
        tone = self._tone_mode()
        top = self._top_tokens(activation, exclude=idxs)
        if not top:
            template = self._template("clarify", intensity, tone=tone)
            return template or "Need more detail."
        groups: Dict[str, List[str]] = {"action": [], "risk": [], "target": [], "concept": [], "emotion": []}
        for token, _score in top:
            role = self._role_for(token)
            if role not in groups:
                role = "concept"
            groups[role].append(token)
        lines: List[str] = []
        for role in ("target", "action", "risk", "concept"):
            items = groups.get(role) or []
            if not items:
                continue
            template = self._template(role, intensity, tone=tone)
            if template:
                lines.append(template.format(items=", ".join(items[:4]), intensity=intensity))
        emotion_items = groups.get("emotion") or []
        if emotion_items:
            prefix = self._template("emotion_prefix", intensity, tone=tone)
            suffix = self._template("emotion_suffix", intensity, tone=tone)
            if prefix:
                lines.insert(0, prefix.format(items=", ".join(emotion_items[:3]), intensity=intensity))
            if suffix:
                lines.append(suffix.format(items=", ".join(emotion_items[:3]), intensity=intensity))
        if idxs and top:
            target_idx = self.index.get(top[0][0])
            if target_idx is not None:
                path = self._build_path(idxs, target_idx)
                if path:
                    template = self._template("path", intensity, tone=tone)
                    if template:
                        lines.append(template.format(path=" -> ".join(path[:6]), intensity=intensity))
        return " ".join([ln for ln in lines if ln]).strip()

    def summarize(self, turns: List[Dict[str, str]]) -> str:
        text = " ".join([t.get("text", "") for t in turns if t.get("text")])
        activation, idxs = self.stimulate(text)
        intensity = self._intensity(activation)
        tone = self._tone_mode()
        top = self._top_tokens(activation, exclude=idxs)
        if not top:
            template = self._template("clarify", intensity, tone=tone)
            return template or "Need more detail."
        items = ", ".join([t for t, _ in top[:6]])
        template = self._template("concept", intensity, tone=tone) or "Summary: {items}"
        return template.format(items=items, intensity=intensity)

    def learn_from_success(
        self,
        goal: str,
        plan_steps: List[str],
        score: float = 1.0,
        save: bool = True,
        apply_decay: bool = True,
    ) -> None:
        if sparse is None:
            return
        if self.W_logic.shape[0] == 0:
            return
        if apply_decay:
            decay = float(self.config.weight_decay)
            if decay > 0:
                decay = min(0.5, max(0.0, decay))
                self.W_logic = self.W_logic.multiply(1.0 - decay).tocsr()
                self.W_emotion = self.W_emotion.multiply(1.0 - decay).tocsr()
                self.W_cross = self.W_cross.multiply(1.0 - decay).tocsr()
        src_act, _ = self.stimulate(goal)
        tgt_act, _ = self.stimulate(" ".join(plan_steps))
        src_idx = np.argsort(-src_act)[: min(12, len(self.vocab))]
        tgt_idx = np.argsort(-tgt_act)[: min(12, len(self.vocab))]
        lr = float(self.config.learning_rate) * max(0.1, float(score))
        mat_logic = self.W_logic.tolil()
        mat_emotion = self.W_emotion.tolil()
        mat_cross = self.W_cross.tolil()
        for i in src_idx:
            if src_act[i] <= 0:
                continue
            for j in tgt_idx:
                if tgt_act[j] <= 0:
                    continue
                ci = self._category(i)
                cj = self._category(j)
                delta = lr * float(src_act[i] * tgt_act[j])
                if ci == "emotion" and cj == "emotion":
                    mat_emotion[i, j] = float(mat_emotion[i, j]) + delta
                elif ci == "logic" and cj == "logic":
                    mat_logic[i, j] = float(mat_logic[i, j]) + delta
                else:
                    mat_cross[i, j] = float(mat_cross[i, j]) + delta
        self.W_logic = mat_logic.tocsr()
        self.W_emotion = mat_emotion.tocsr()
        self.W_cross = mat_cross.tocsr()
        combined = self.W_logic + self.W_emotion + self.W_cross
        self.neighbors = self._build_neighbors(combined, self.config.max_neighbors)
        if save:
            self._save()

    def _save(self) -> None:
        try:
            graph_path = self.cache_dir / "nanobrain_graph_logic.npz"
            graph_emotion = self.cache_dir / "nanobrain_graph_emotion.npz"
            graph_cross = self.cache_dir / "nanobrain_graph_cross.npz"
            vocab_path = self.cache_dir / "nanobrain_vocab.json"
            meta_path = self.cache_dir / "nanobrain_meta.json"
            sparse.save_npz(graph_path, self.W_logic)
            sparse.save_npz(graph_emotion, self.W_emotion)
            sparse.save_npz(graph_cross, self.W_cross)
            vocab_path.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")
            meta_path.write_text(
                json.dumps(
                    {
                        "token_meta": self.token_meta,
                        "logic_nodes": sorted(list(self.logic_nodes)),
                        "emotion_nodes": sorted(list(self.emotion_nodes)),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("NanoBrain save failed", exc_info=True)
