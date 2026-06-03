from __future__ import annotations

from typing import List

from system.knowledge.graph.kb import Neo4jKnowledgeBase


class Neo4jQuery:
    def __init__(self, kb: Neo4jKnowledgeBase):
        self.kb = kb

    def find_neighbors(self, name: str, limit: int = 5) -> List[str]:
        if not self.kb or not self.kb.available():
            return []
        with self.kb._driver.session() as session:
            result = session.run(
                "MATCH (a:Entity {name:$n})-[r]->(b:Entity) RETURN b.name AS name LIMIT $l",
                n=name,
                l=limit,
            )
            return [record["name"] for record in result]

    def find_paths(self, start: str, end: str, max_hops: int = 3) -> List[List[str]]:
        if not self.kb or not self.kb.available():
            return []
        with self.kb._driver.session() as session:
            result = session.run(
                "MATCH p=(a:Entity {name:$s})-[*1..$h]->(b:Entity {name:$e}) "
                "RETURN [n IN nodes(p) | n.name] AS path LIMIT 10",
                s=start,
                e=end,
                h=max_hops,
            )
            return [record["path"] for record in result]

    def find_by_relation(self, name: str, rel: str, limit: int = 5) -> List[str]:
        if not self.kb or not self.kb.available():
            return []
        with self.kb._driver.session() as session:
            result = session.run(
                "MATCH (a:Entity {name:$n})-[r {type:$r}]->(b:Entity) "
                "RETURN b.name AS name LIMIT $l",
                n=name,
                r=rel,
                l=limit,
            )
            return [record["name"] for record in result]

    def sample_edges(self, limit: int = 50) -> List[tuple[str, str, str]]:
        if not self.kb or not self.kb.available():
            return []
        with self.kb._driver.session() as session:
            result = session.run(
                "MATCH (a:Entity)-[r:REL]->(b:Entity) RETURN a.name AS s, r.type AS p, b.name AS o LIMIT $l",
                l=limit,
            )
            return [(record["s"], record["p"], record["o"]) for record in result]
