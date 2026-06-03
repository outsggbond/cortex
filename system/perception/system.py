from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import numpy as np

import os
import pickle
import re
from pathlib import Path

from config.system_config import SystemConfig, logger
from core.types import Node, Edge, PerceptionGraph
from memory.vector_index import VectorIndex
from system.perception.encoders import (
    encode_audio,
    encode_image,
    encode_text,
    encode_video,
    ensure_float_array,
    project_to_dim,
)
from system.perception.local_models import LocalImageEncoder, LocalTextEncoder
from system.core.model_loader import ImageTagger, AudioTagger


class OptimizedPerceptionSystem:
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.graph = PerceptionGraph()
        self.index = VectorIndex(self.config.feature_dim)
        self.cycle_count = 0
        self.performance_stats: Dict[str, List[float]] = {"cycle_time": []}
        self._text_encoder = None
        self._image_encoder = None
        self._image_tagger = None
        self._audio_tagger = None
        self._tag_top_k = int(getattr(self.config, "perception_top_k", 3) or 3)
        self._tag_threshold = float(getattr(self.config, "perception_conf_threshold", 0.75) or 0.75)
        self._temporal_half_life_s = float(getattr(self.config, "temporal_decay_half_life_s", 0.0) or 0.0)
        self._compress_after_s = float(getattr(self.config, "index_compress_after_s", 0.0) or 0.0)
        self._compress_dim = int(getattr(self.config, "index_compress_dim", 0) or 0)
        self._last_compress_ts = 0.0
        self._init_local_encoders()
        self._init_taggers()
        logger.info(
            f"System init | dim={self.config.feature_dim} | use_gpu={self.config.use_gpu}"
        )

    def _init_local_encoders(self) -> None:
        if not self.config.use_local_models:
            return
        if self.config.local_text_model_path:
            enc = LocalTextEncoder(
                self.config.local_text_model_path,
                self.config.feature_dim,
                self.config.seed,
            )
            if enc.available():
                self._text_encoder = enc
        if self.config.local_image_model_path:
            enc = LocalImageEncoder(
                self.config.local_image_model_path,
                self.config.feature_dim,
                self.config.seed,
            )
            if enc.available():
                self._image_encoder = enc

    def _init_taggers(self) -> None:
        if not getattr(self.config, "perception_auto_extract", False):
            return
        img_path = str(getattr(self.config, "perception_image_tagger", "") or "")
        if img_path:
            try:
                tagger = ImageTagger(
                    img_path,
                    use_gpu=bool(self.config.use_gpu),
                    top_k=self._tag_top_k,
                    threshold=self._tag_threshold,
                )
                if tagger.available():
                    self._image_tagger = tagger
            except Exception:
                self._image_tagger = None
        aud_path = str(getattr(self.config, "perception_audio_tagger", "") or "")
        if aud_path:
            try:
                tagger = AudioTagger(
                    aud_path,
                    use_gpu=bool(self.config.use_gpu),
                    top_k=self._tag_top_k,
                    threshold=self._tag_threshold,
                )
                if tagger.available():
                    self._audio_tagger = tagger
            except Exception:
                self._audio_tagger = None

    def run_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        self.cycle_count += 1
        start = time.time()
        new_nodes = self._create_nodes_from_input(input_data)
        for node in new_nodes:
            self.graph.add_node(node)
            if not node.metadata.get("dummy"):
                self.index.add(node.feature, metadata=node.metadata, id=node.id)

        edges_added = self._link_similar_nodes(new_nodes)
        modal_tokens = self._collect_modal_tokens(new_nodes)
        self._maybe_compress_old_nodes()
        result = {
            "nodes_created": len([n for n in new_nodes if not n.metadata.get("dummy")]),
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "new_edges": edges_added,
            "modal_tokens": modal_tokens,
        }
        result["cycle_time"] = time.time() - start
        self.performance_stats["cycle_time"].append(result["cycle_time"])
        if self.config.enable_persistence and self.cycle_count % self.config.checkpoint_interval == 0:
            self.save_checkpoint(self.config.checkpoint_dir, f"cycle_{self.cycle_count}")
        return result

    def evaluate_cycle(self, result: Dict[str, Any]) -> float:
        if not result:
            return 0.0
        score = 0.0
        if result.get("new_edges", 0) > 0:
            score += 0.5
        if result.get("nodes_created", 0) > 0:
            score += 0.3
        if result.get("cycle_time", 0) < 1.0:
            score += 0.2
        return min(score, 1.0)

    def evolve(self, feedback_score: float) -> Dict[str, float]:
        """Simple, safe evolution: adjust thresholds based on feedback."""
        before = self.config.similarity_threshold
        if feedback_score > 0.7:
            self.config.similarity_threshold = min(0.95, before + 0.02)
        elif feedback_score < 0.4:
            self.config.similarity_threshold = max(0.5, before - 0.02)
        return {
            "similarity_threshold": self.config.similarity_threshold,
        }

    def retrieve_memories(self, message: str, k: int | None = None) -> list[str]:
        q_feat = encode_text(message, self.config.feature_dim)
        neighbors = self.index.search(q_feat, k=k or self.config.reasoning_k)
        now = time.time()
        texts = []
        scored = []
        for item in neighbors:
            nid = item[0]
            sim = float(item[1]) if len(item) > 1 else 0.0
            node = self.graph.nodes.get(nid)
            if node and node.type == "text" and not node.metadata.get("dummy"):
                score = self._apply_temporal_decay(sim, node.timestamp, now)
                scored.append((score, node.metadata.get("content", "")))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _score, content in scored[: max(1, int(k or self.config.reasoning_k))]:
            texts.append(content)
        return texts

    def chat(self, message: str) -> str:
        """Fallback chat using retrieval over stored text nodes."""
        texts = self.retrieve_memories(message, k=self.config.reasoning_k)
        if not texts:
            return "我现在还没有足够的上下文。可以多输入一些信息让我学习。"
        max_mem = max(1, int(getattr(self.config, "chat_max_memories", 4)))
        snippet = "；".join([t[:50] for t in texts if t][:max_mem])[:200]
        return f"我记得这些内容：{snippet}。你想继续聊哪一部分？"

    def get_statistics(self) -> Dict[str, Any]:
        avg_cycle = (
            float(np.mean(self.performance_stats["cycle_time"]))
            if self.performance_stats["cycle_time"]
            else 0.0
        )
        return {
            "total_cycles": self.cycle_count,
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "avg_cycle_time": avg_cycle,
            "index_size": self.index.size,
        }

    def _create_nodes_from_input(self, input_data: Dict[str, Any]) -> List[Node]:
        nodes: List[Node] = []
        base_id = f"{self.cycle_count}_{len(self.graph.nodes)}"
        processors = {
            "text": self._process_text,
            "image": self._process_image,
            "audio": self._process_audio,
            "video": self._process_video,
        }
        for modality, processor in processors.items():
            data = input_data.get(modality)
            node = processor(data, base_id)
            nodes.append(node)
        return nodes

    def _process_text(self, text: Optional[str], base_id: str) -> Node:
        if text is None or not str(text).strip():
            return self._dummy_node("text", base_id)
        text_str = str(text)
        if self._text_encoder is not None:
            try:
                feature = self._text_encoder.encode(text_str)
            except Exception:
                feature = encode_text(text_str, self.config.feature_dim)
        else:
            feature = encode_text(text_str, self.config.feature_dim)
        tokens = self._wrap_tokens(self._tokenize_text(text_str))
        return Node(
            node_id=f"text_{base_id}",
            node_type="text",
            feature=feature,
            metadata={"length": len(text_str), "content": text_str[:500], "tokens": tokens},
        )

    def _process_image(self, image: Optional[np.ndarray], base_id: str) -> Node:
        image, tokens = self._normalize_modal_input(image)
        auto_tokens = self._auto_tag_image(image) if image is not None else []
        tokens = self._merge_tokens(tokens, auto_tokens)
        if image is None:
            return self._dummy_node("image", base_id)
        img = ensure_float_array(image)
        if self._image_encoder is not None:
            try:
                feature = self._image_encoder.encode(img.astype("uint8"))
            except Exception:
                feature = encode_image(img, self.config.feature_dim)
        else:
            feature = encode_image(img, self.config.feature_dim)
        return Node(
            node_id=f"image_{base_id}",
            node_type="image",
            feature=feature,
            metadata={"shape": list(image.shape), "tokens": tokens},
        )

    def _process_audio(self, audio: Optional[np.ndarray], base_id: str) -> Node:
        audio, tokens = self._normalize_modal_input(audio)
        auto_tokens = self._auto_tag_audio(audio) if audio is not None else []
        tokens = self._merge_tokens(tokens, auto_tokens)
        if audio is None:
            return self._dummy_node("audio", base_id)
        feature = encode_audio(ensure_float_array(audio), self.config.feature_dim)
        return Node(
            node_id=f"audio_{base_id}",
            node_type="audio",
            feature=feature,
            metadata={"length": int(audio.size), "tokens": tokens},
        )

    def _process_video(self, video: Optional[np.ndarray], base_id: str) -> Node:
        video, tokens = self._normalize_modal_input(video)
        tokens = self._merge_tokens(tokens, [])
        if video is None:
            return self._dummy_node("video", base_id)
        feature = encode_video(ensure_float_array(video), self.config.feature_dim)
        return Node(
            node_id=f"video_{base_id}",
            node_type="video",
            feature=feature,
            metadata={"shape": list(video.shape), "tokens": tokens},
        )

    def _dummy_node(self, modality: str, base_id: str) -> Node:
        return Node(
            node_id=f"{modality}_{base_id}_dummy",
            node_type=modality,
            feature=np.zeros(self.config.feature_dim, dtype="float32"),
            metadata={"dummy": True},
        )

    def _link_similar_nodes(self, new_nodes: List[Node]) -> int:
        edges_added = 0
        now = time.time()
        for node in new_nodes:
            if node.metadata.get("dummy"):
                continue
            neighbors = self.index.search(node.feature, k=self.config.reasoning_k)
            for item in neighbors:
                nid = item[0]
                sim = float(item[1]) if len(item) > 1 else 0.0
                if nid == node.id:
                    continue
                sim = self._apply_temporal_decay(sim, self.graph.nodes.get(nid).timestamp if nid in self.graph.nodes else now, now)
                if sim < self.config.similarity_threshold:
                    continue
                if self.graph.has_direct_edge(node.id, nid):
                    continue
                self.graph.add_edge(Edge(node.id, nid, "similar", float(sim)))
                edges_added += 1
        return edges_added

    def _apply_temporal_decay(self, sim: float, ts: float, now: float) -> float:
        half_life = self._temporal_half_life_s
        if not half_life or half_life <= 0:
            return sim
        age = max(0.0, now - float(ts or now))
        decay = 0.5 ** (age / half_life)
        return float(sim) * float(decay)

    def _tokenize_text(self, text: str) -> List[str]:
        if not text:
            return []
        parts = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text)
        tokens: List[str] = []
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r"[\u4e00-\u9fff]+", part):
                tokens.append(part)
                if len(part) > 1:
                    tokens.extend(list(part))
            else:
                tokens.append(part.lower())
        return list(dict.fromkeys(tokens))

    def _tokens_from_value(self, value: Any) -> List[str]:
        tokens: List[str] = []
        if value is None:
            return tokens
        if isinstance(value, (list, tuple, set)):
            for v in value:
                tokens.extend(self._tokens_from_value(v))
            return tokens
        text = str(value).strip()
        if text:
            tokens.extend(self._tokenize_text(text))
        return tokens

    def _normalize_modal_input(self, value: Any) -> tuple[Optional[np.ndarray], List[str]]:
        tokens: List[str] = []
        if isinstance(value, dict):
            data = value.get("data")
            if data is None:
                data = value.get("value") or value.get("array") or value.get("image") or value.get("audio") or value.get("video")
            token_fields = ("labels", "tags", "caption", "text", "transcript", "description")
            for key in token_fields:
                if key in value:
                    tokens.extend(self._tokens_from_value(value.get(key)))
            return data, list(dict.fromkeys(tokens))
        return value, tokens

    def _collect_modal_tokens(self, nodes: List[Node]) -> Dict[str, List]:
        modal_tokens: Dict[str, List] = {}
        for node in nodes:
            if node.metadata.get("dummy"):
                continue
            tokens = node.metadata.get("tokens") or []
            if not tokens:
                continue
            modal_tokens.setdefault(node.type, [])
            modal_tokens[node.type].extend(tokens)
        # dedupe per modality
        for k, v in list(modal_tokens.items()):
            seen = {}
            for item in v:
                if isinstance(item, dict):
                    tok = item.get("token")
                    score = float(item.get("score", 1.0))
                elif isinstance(item, (tuple, list)):
                    tok = item[0] if item else None
                    score = float(item[1]) if len(item) > 1 else 1.0
                else:
                    tok = str(item)
                    score = 1.0
                if not tok:
                    continue
                prev = seen.get(tok)
                if prev is None or score > prev:
                    seen[tok] = score
            modal_tokens[k] = [{"token": t, "score": s} for t, s in seen.items()]
        return modal_tokens

    def _wrap_tokens(self, tokens: List[str]) -> List[dict]:
        return [{"token": t, "score": 1.0} for t in tokens if t]

    def _merge_tokens(self, manual_tokens: List[str] | List[dict], auto_tokens: List[dict]) -> List[dict]:
        merged: Dict[str, float] = {}
        for item in manual_tokens:
            if isinstance(item, dict):
                tok = item.get("token")
                score = float(item.get("score", 1.0))
            elif isinstance(item, (tuple, list)):
                tok = item[0] if item else None
                score = float(item[1]) if len(item) > 1 else 1.0
            else:
                tok = str(item)
                score = 1.0
            if not tok:
                continue
            merged[tok] = max(merged.get(tok, 0.0), score)
        for item in auto_tokens:
            tok = item.get("token") if isinstance(item, dict) else None
            if not tok:
                continue
            score = float(item.get("score", 1.0))
            merged[tok] = max(merged.get(tok, 0.0), score)
        return [{"token": t, "score": s} for t, s in merged.items()]

    def _auto_tag_image(self, image: Optional[np.ndarray]) -> List[dict]:
        if image is None or self._image_tagger is None:
            return []
        try:
            results = self._image_tagger.tag(image)
        except Exception:
            return []
        return [{"token": r.label, "score": float(r.score)} for r in results if r.label]

    def _auto_tag_audio(self, audio: Optional[np.ndarray]) -> List[dict]:
        if audio is None or self._audio_tagger is None:
            return []
        try:
            results = self._audio_tagger.tag(audio)
        except Exception:
            return []
        return [{"token": r.label, "score": float(r.score)} for r in results if r.label]

    def _maybe_compress_old_nodes(self) -> None:
        if self._compress_after_s <= 0 or self._compress_dim <= 0:
            return
        if self._compress_dim >= int(self.config.feature_dim):
            return
        now = time.time()
        if now - self._last_compress_ts < max(5.0, self._compress_after_s / 4.0):
            return
        self._last_compress_ts = now
        for node in self.graph.nodes.values():
            if node.metadata.get("dummy"):
                continue
            if node.metadata.get("compressed_dim"):
                continue
            age = now - node.timestamp
            if age < self._compress_after_s:
                continue
            try:
                compressed = project_to_dim(node.feature, self._compress_dim, seed=self.config.seed)
            except Exception:
                continue
            node.feature = compressed
            node.metadata["compressed_dim"] = int(self._compress_dim)
            self.index.update(node.id, compressed, metadata=node.metadata)

    def save_checkpoint(self, base_dir: str, name: str) -> None:
        path = Path(base_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        # save graph
        graph_data = {
            "nodes": [
                (nid, n.type, n.feature, n.metadata, n.timestamp)
                for nid, n in self.graph.nodes.items()
            ],
            "edges": [
                (e.from_node, e.to_node, e.relation, e.weight, e.metadata, e.timestamp)
                for e in self.graph.edges.values()
            ],
            "cycle_count": self.cycle_count,
            "performance_stats": self.performance_stats,
        }
        with open(path / "graph.pkl", "wb") as f:
            pickle.dump(graph_data, f)
        # save index
        self.index.save(str(path / "index.npz"))
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        cp = Path(path)
        graph_file = cp / "graph.pkl"
        index_file = cp / "index.npz"
        if graph_file.exists():
            with open(graph_file, "rb") as f:
                data = pickle.load(f)
            self.graph = PerceptionGraph()
            for nid, ntype, feat, meta, ts in data.get("nodes", []):
                node = Node(nid, ntype, np.array(feat), meta)
                node.timestamp = ts
                self.graph.add_node(node)
            for fn, tn, rel, w, meta, ts in data.get("edges", []):
                edge = Edge(fn, tn, rel, w, meta)
                edge.timestamp = ts
                self.graph.add_edge(edge)
            self.cycle_count = data.get("cycle_count", 0)
            self.performance_stats = data.get("performance_stats", {"cycle_time": []})
        if index_file.exists():
            self.index = VectorIndex.load(str(index_file))
        logger.info(f"Checkpoint loaded: {cp}")
