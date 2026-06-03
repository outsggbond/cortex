from __future__ import annotations

from system.knowledge.graph.kb import Neo4jKnowledgeBase
from system.knowledge.graph.query import Neo4jQuery


def main():
    uri = input("neo4j uri: ").strip()
    user = input("neo4j user: ").strip()
    password = input("neo4j password: ").strip()
    kb = Neo4jKnowledgeBase(uri, user, password)
    if not kb.available():
        print("Neo4j driver unavailable")
        return
    q = Neo4jQuery(kb)
    print("Commands: neighbors <name> [limit], relation <name> <rel>, path <start> <end> [hops], quit")
    while True:
        line = input("> ").strip()
        if not line or line.lower() in {"quit", "exit"}:
            break
        parts = line.split()
        if not parts:
            continue
        cmd = parts[0].lower()
        try:
            if cmd == "neighbors" and len(parts) >= 2:
                name = parts[1]
                limit = int(parts[2]) if len(parts) >= 3 else 5
                print(q.find_neighbors(name, limit=limit))
            elif cmd == "relation" and len(parts) >= 3:
                name = parts[1]
                rel = parts[2]
                print(q.find_by_relation(name, rel, limit=5))
            elif cmd == "path" and len(parts) >= 3:
                start = parts[1]
                end = parts[2]
                hops = int(parts[3]) if len(parts) >= 4 else 3
                print(q.find_paths(start, end, max_hops=hops))
            else:
                print("unknown command")
        except Exception as e:
            print(f"error: {e}")
    kb.close()


if __name__ == "__main__":
    main()
