from __future__ import annotations

import argparse
from typing import List

from system.knowledge.graph.kb import Neo4jKnowledgeBase
from system.knowledge.graph.query import Neo4jQuery
from system.l_utils.graphviz_export import export_trace_graph


def export_neighbor_graph(center: str, neighbors: List[str], out_path: str) -> str:
    try:
        from graphviz import Digraph
    except Exception:
        return ""
    g = Digraph("neo4j")
    g.attr(rankdir="LR")
    g.node("center", center, shape="oval")
    for i, n in enumerate(neighbors):
        node_id = f"n{i}"
        g.node(node_id, n, shape="box")
        g.edge("center", node_id)
    g.render(out_path, format="png", cleanup=True)
    return out_path + ".png"


def export_graph_edges(edges: List[tuple[str, str, str]], out_path: str) -> str:
    try:
        from graphviz import Digraph
    except Exception:
        return ""
    g = Digraph("neo4j_full")
    g.attr(rankdir="LR")
    for s, p, o in edges:
        g.node(s, s, shape="oval")
        g.node(o, o, shape="box")
        g.edge(s, o, label=p)
    g.render(out_path, format="png", cleanup=True)
    return out_path + ".png"


def main():
    p = argparse.ArgumentParser(description="Neo4j CLI")
    p.add_argument("--uri", required=True)
    p.add_argument("--user", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--neighbors", type=str, default="")
    p.add_argument("--path-start", type=str, default="")
    p.add_argument("--path-end", type=str, default="")
    p.add_argument("--relation", type=str, default="")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--export", type=str, default="")
    p.add_argument("--export-graph", type=str, default="")
    p.add_argument("--edge-limit", type=int, default=50)
    args = p.parse_args()

    kb = Neo4jKnowledgeBase(args.uri, args.user, args.password)
    if not kb.available():
        print("Neo4j driver unavailable")
        return
    query = Neo4jQuery(kb)
    if args.neighbors:
        neighbors = query.find_neighbors(args.neighbors, limit=args.limit)
        print(neighbors)
        if args.export:
            out = export_neighbor_graph(args.neighbors, neighbors, args.export)
            print(f"graph: {out}")
    if args.path_start and args.path_end:
        paths = query.find_paths(args.path_start, args.path_end, max_hops=3)
        print(paths)
    if args.neighbors and args.relation:
        rels = query.find_by_relation(args.neighbors, args.relation, limit=args.limit)
        print(rels)
    if args.export_graph:
        edges = query.sample_edges(limit=args.edge_limit)
        out = export_graph_edges(edges, args.export_graph)
        print(f"graph: {out}")
    kb.close()


if __name__ == "__main__":
    main()
