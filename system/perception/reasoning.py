import numpy as np
from typing import List, Optional, Dict, Tuple
from core.types import Node, PerceptionGraph, Edge
from memory.memory_module import OptimizedMemoryModule


class AdaptiveReasoningModule:
    def __init__(self, graph: PerceptionGraph, memory: OptimizedMemoryModule, similarity_threshold: float, reasoning_k: int, nprobe: int = 10, multi_hop_alpha: float = 0.45, max_hops: int = 3, max_paths: int = 20, multi_hop_threshold: float = 0.6, max_new_edges_per_node: int = 3):
        self.graph = graph
        self.memory = memory
        self.threshold = similarity_threshold
        self.reasoning_k = reasoning_k
        self.nprobe = nprobe
        self.multi_hop_alpha = float(multi_hop_alpha)
        self.default_max_hops = int(max_hops)
        self.default_max_paths = int(max_paths)
        self.multi_hop_threshold = float(multi_hop_threshold)
        self.max_new_edges_per_node = int(max_new_edges_per_node)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a_norm, b_norm))

    def infer(self, graph: PerceptionGraph, new_nodes: List[Node], gnn_embeddings: Optional[Dict[str, 'torch.Tensor']] = None, node_id_to_index: Optional[Dict] = None, use_gnn: bool = True) -> int:
        if not new_nodes:
            return 0
        node_ids_to_check = {node.id for node in new_nodes}
        for node in new_nodes:
            neighbors = self.memory.find_nearest_neighbors(node.feature, k=self.reasoning_k, nprobe=self.nprobe)
            for nid, _ in neighbors:
                if nid in graph.nodes:
                    node_ids_to_check.add(nid)
            for edge in graph.find_edges_by_node(node.id):
                other_id = edge.target if edge.source == node.id else edge.source
                node_ids_to_check.add(other_id)

        nodes_list = [graph.nodes[nid] for nid in node_ids_to_check if nid in graph.nodes]
        new_edges = 0
        for i, node_i in enumerate(nodes_list):
            for j, node_j in enumerate(nodes_list[i + 1:], i + 1):
                if node_i.id not in {n.id for n in new_nodes} and node_j.id not in {n.id for n in new_nodes}:
                    continue
                similarity = self.cosine_similarity(node_i.feature, node_j.feature)
                if similarity > self.threshold:
                    if not graph.has_direct_edge(node_i.id, node_j.id, relation="semantic_related_deep"):
                        graph.add_edge(Edge(node_i.id, node_j.id, "semantic_related_deep", float(similarity)))
                        new_edges += 1
        return new_edges

    def multi_hop_reasoning(self, start_node: Node, max_hops: Optional[int] = None, max_paths: Optional[int] = None, score_threshold: Optional[float] = None, gnn_embeddings: Optional[Dict[str, 'torch.Tensor']] = None, node_id_to_index: Optional[Dict] = None) -> List[Tuple[List[str], float]]:
        """Perform multi-hop path-based reasoning from `start_node` to other nodes in the graph.
        Returns list of (path, score) where path is list of node ids and score is the path score.
        """
        # resolve defaults if None
        if max_hops is None:
            max_hops = self.default_max_hops
        if max_paths is None:
            max_paths = self.default_max_paths
        if score_threshold is None:
            score_threshold = self.multi_hop_threshold
        results = []
        # search candidate end nodes among memory neighbors and graph neighbors
        neighbors = self.memory.find_nearest_neighbors(start_node.feature, k=self.reasoning_k, nprobe=self.nprobe)
        candidate_ids = {nid for nid, _ in neighbors}
        # also include direct neighbors in graph
        for e in self.graph.find_edges_by_node(start_node.id):
            candidate_ids.add(e.source)
            candidate_ids.add(e.target)

        # limit candidate ids to avoid explosion (preserve order from memory neighbors)
        candidate_list = list(candidate_ids)
        # neighbors returned from memory are ordered; ensure we take top-K
        if neighbors:
            ordered = [nid for nid, _ in neighbors if nid in candidate_list]
            # append any remaining candidate_ids not in ordered
            for nid in candidate_list:
                if nid not in ordered:
                    ordered.append(nid)
            candidate_list = ordered[:min(len(ordered), self.reasoning_k)]
        else:
            candidate_list = candidate_list[:self.reasoning_k]

        for cand in candidate_list:
            if cand == start_node.id or cand not in self.graph.nodes:
                continue
            paths = self.graph.find_paths(start_node.id, cand, max_hops, max_paths)
            for path in paths:
                # compute path score as mean cosine similarity of consecutive nodes
                scores = []
                valid = True
                for i in range(len(path) - 1):
                    n1 = self.graph.nodes.get(path[i])
                    n2 = self.graph.nodes.get(path[i + 1])
                    if n1 is None or n2 is None:
                        valid = False
                        break
                    sc = self.cosine_similarity(n1.feature, n2.feature)
                    scores.append(sc)
                if not valid or not scores:
                    continue
                feature_score = float(sum(scores) / len(scores))
                gnn_score = None
                try:
                    if gnn_embeddings is not None and node_id_to_index is not None:
                        emb_scores = []
                        import torch
                        for i in range(len(path) - 1):
                            a_id = path[i]
                            b_id = path[i + 1]
                            a_idx = node_id_to_index.get(a_id)
                            b_idx = node_id_to_index.get(b_id)
                            if a_idx is None or b_idx is None:
                                continue
                            a_type, a_i = a_idx
                            b_type, b_i = b_idx
                            if a_type in gnn_embeddings and b_type in gnn_embeddings:
                                try:
                                    a_emb = gnn_embeddings[a_type][a_i]
                                    b_emb = gnn_embeddings[b_type][b_i]
                                    if isinstance(a_emb, torch.Tensor):
                                        score = float(torch.cosine_similarity(a_emb.unsqueeze(0), b_emb.unsqueeze(0)).cpu().numpy())
                                    else:
                                        import numpy as _np
                                        a_e = _np.asarray(a_emb)
                                        b_e = _np.asarray(b_emb)
                                        score = float(np.dot(a_e / (np.linalg.norm(a_e) + 1e-8), b_e / (np.linalg.norm(b_e) + 1e-8)))
                                    emb_scores.append(score)
                                except Exception:
                                    continue
                        if emb_scores:
                            gnn_score = float(sum(emb_scores) / len(emb_scores))
                except Exception:
                    gnn_score = None

                # combine scores: more weight to GNN when available
                if gnn_score is not None:
                    alpha = self.multi_hop_alpha
                    path_score = alpha * feature_score + (1 - alpha) * gnn_score
                else:
                    path_score = feature_score

                # filter out trivial/duplicated consecutive nodes (identical features)
                too_similar = False
                for i in range(len(path) - 1):
                    n1 = self.graph.nodes.get(path[i])
                    n2 = self.graph.nodes.get(path[i + 1])
                    if n1 is None or n2 is None:
                        too_similar = True
                        break
                    # if features are almost identical, consider this path noisy
                    if np.linalg.norm(n1.feature - n2.feature) < 1e-6:
                        too_similar = True
                        break
                if too_similar:
                    continue

                if path_score >= score_threshold:
                    results.append((path, float(path_score)))

        # sort and keep top-k paths — but deduplicate by end node, keep best path per end
        results = sorted(results, key=lambda x: x[1], reverse=True)
        best_by_end: Dict[str, Tuple[List[str], float]] = {}
        for path, score in results:
            end = path[-1]
            if end == start_node.id:
                continue
            prev = best_by_end.get(end)
            if prev is None or score > prev[1]:
                best_by_end[end] = (path, score)
        results = list(best_by_end.values())
        # final sorting and trim to max_paths
        results = sorted(results, key=lambda x: x[1], reverse=True)
        if len(results) > max_paths:
            results = results[:max_paths]

        # add inferred edges but limit number of new edges per start node to avoid flooding
        added = 0
        for path, score in results:
            if added >= self.max_new_edges_per_node:
                break
            end_id = path[-1]
            if not self.graph.has_direct_edge(start_node.id, end_id, relation="multi_hop_inferred"):
                meta = {'path': path, 'path_score': float(score)}
                self.graph.add_edge(Edge(start_node.id, end_id, "multi_hop_inferred", float(score), metadata=meta))
                added += 1

        return results
