# -*- coding: utf-8 -*-
"""Core graph types for the perception system — Node, Edge, PerceptionGraph."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class Node:
    """A single perception node (text/image/audio/video)."""

    def __init__(
        self,
        id: str = "",
        type: str = "",
        features: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = id or str(uuid.uuid4())
        self.type = type
        self.features = features if features is not None else np.array([], dtype=np.float32)
        self.metadata = dict(metadata or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "features": self.features.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Node:
        return cls(
            id=str(d.get("id", "")),
            type=str(d.get("type", "")),
            features=np.array(d.get("features", []), dtype=np.float32),
            metadata=dict(d.get("metadata", {}) or {}),
        )


class Edge:
    """A directed weighted edge between two perception nodes."""

    def __init__(
        self,
        source: str,
        target: str,
        relation: str = "",
        weight: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.source = source
        self.target = target
        self.relation = relation
        self.weight = float(weight)
        self.metadata = dict(metadata or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Edge:
        return cls(
            source=str(d.get("source", "")),
            target=str(d.get("target", "")),
            relation=str(d.get("relation", "")),
            weight=float(d.get("weight", 0.0)),
            metadata=dict(d.get("metadata", {}) or {}),
        )


class PerceptionGraph:
    """A multi-modal perception graph — nodes + edges with serialization."""

    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []
        self._adj_in: Dict[str, List[int]] = defaultdict(list)
        self._adj_out: Dict[str, List[int]] = defaultdict(list)

    # -- nodes ----------------------------------------------------------------

    @property
    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    @property
    def edges(self) -> List[Edge]:
        return list(self._edges)

    def add_node(self, node: Node) -> None:
        self._nodes[node.id] = node

    def get_node(self, nid: str) -> Optional[Node]:
        return self._nodes.get(nid)

    def remove_node(self, nid: str) -> None:
        self._nodes.pop(nid, None)
        # also clean edges referencing this node
        self._edges = [e for e in self._edges if e.source != nid and e.target != nid]
        self._rebuild_adj()

    def node_count(self) -> int:
        return len(self._nodes)

    # -- edges ----------------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        idx = len(self._edges)
        self._edges.append(edge)
        self._adj_out[edge.source].append(idx)
        self._adj_in[edge.target].append(idx)

    def remove_edge(self, idx: int) -> None:
        if 0 <= idx < len(self._edges):
            e = self._edges[idx]
            self._edges.pop(idx)
            self._rebuild_adj()

    def edge_count(self) -> int:
        return len(self._edges)

    def neighbors(self, nid: str, direction: str = "out") -> List[str]:
        ids: List[str] = []
        if direction == "out":
            for idx in self._adj_out.get(nid, []):
                if idx < len(self._edges):
                    ids.append(self._edges[idx].target)
        elif direction == "in":
            for idx in self._adj_in.get(nid, []):
                if idx < len(self._edges):
                    ids.append(self._edges[idx].source)
        else:
            ids = self.neighbors(nid, "out") + self.neighbors(nid, "in")
        return ids

    # -- adjacency ------------------------------------------------------------

    def _rebuild_adj(self) -> None:
        self._adj_in = defaultdict(list)
        self._adj_out = defaultdict(list)
        for i, e in enumerate(self._edges):
            self._adj_out[e.source].append(i)
            self._adj_in[e.target].append(i)

    # -- serialization --------------------------------------------------------

    def save(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def load(self, path: str) -> None:
        source = Path(path)
        if not source.exists():
            return
        data = json.loads(source.read_text(encoding="utf-8"))
        self._nodes = {}
        self._edges = []
        for nd in data.get("nodes", []):
            n = Node.from_dict(nd)
            self._nodes[n.id] = n
        for ed in data.get("edges", []):
            self._edges.append(Edge.from_dict(ed))
        self._rebuild_adj()

    def __repr__(self) -> str:
        return f"PerceptionGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
