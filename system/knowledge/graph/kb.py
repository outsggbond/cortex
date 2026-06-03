from __future__ import annotations

from typing import Optional


class Neo4jKnowledgeBase:
    def __init__(self, uri: str, user: str, password: str):
        try:
            from neo4j import GraphDatabase
        except Exception:
            GraphDatabase = None
        self._driver = None
        if GraphDatabase is not None:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def available(self) -> bool:
        return self._driver is not None

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def add_fact(self, subject: str, predicate: str, obj: str) -> None:
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                "MERGE (a:Entity {name:$s}) "
                "MERGE (b:Entity {name:$o}) "
                "MERGE (a)-[r:REL {type:$p}]->(b)",
                s=subject,
                o=obj,
                p=predicate,
            )

    def add_rule(self, name: str, weight: float) -> None:
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                "MERGE (r:Rule {name:$n}) SET r.weight=$w",
                n=name,
                w=weight,
            )

    def add_task(self, name: str, status: str) -> None:
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                "MERGE (t:Task {name:$n}) SET t.status=$s",
                n=name,
                s=status,
            )

    def add_trace(self, rules: list[str], score: float) -> None:
        if not self._driver:
            return
        import uuid

        trace_id = str(uuid.uuid4())
        with self._driver.session() as session:
            session.run(
                "CREATE (tr:Trace {id:$id, score:$s})",
                id=trace_id,
                s=score,
            )
            session.run(
                "MATCH (tr:Trace {id:$id}) "
                "UNWIND $rules AS r "
                "MERGE (ru:Rule {name:r}) "
                "MERGE (tr)-[:USES]->(ru)",
                id=trace_id,
                rules=rules,
            )

    def add_rule_weight(self, name: str, weight: float, score: float) -> None:
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run(
                "MERGE (ru:Rule {name:$n}) "
                "CREATE (rw:RuleWeight {weight:$w, score:$s, ts:timestamp()}) "
                "MERGE (ru)-[:HAS_WEIGHT]->(rw)",
                n=name,
                w=weight,
                s=score,
            )
