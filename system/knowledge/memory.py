from __future__ import annotations

import re
from typing import List, Tuple

from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.knowledge_base import KnowledgeBase
from memory.chat_memory import ChatMemory
from system.knowledge.graph.kb import Neo4jKnowledgeBase


class MemoryManager:
    def __init__(
        self,
        short_capacity: int = 32,
        long_path: str = "artifacts/memory/long_term.jsonl",
        kb_path: str = "artifacts/memory/knowledge.json",
        chat_path: str = "artifacts/memory/chat_memory.jsonl",
        chat_capacity: int = 200,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None,
    ):
        self.short = ShortTermMemory(capacity=short_capacity)
        self.long = LongTermMemory(path=long_path)
        self.kb = KnowledgeBase(path=kb_path)
        self.chat = ChatMemory(path=chat_path, capacity=chat_capacity)
        self.neo4j = None
        if neo4j_uri and neo4j_user and neo4j_password:
            self.neo4j = Neo4jKnowledgeBase(neo4j_uri, neo4j_user, neo4j_password)

    def ingest(self, text: str) -> None:
        if not text:
            return
        self.short.add(text)
        self.long.add(text)
        for s, p, o in self._extract_facts(text):
            self.kb.add_fact(s, p, o)
            if self.neo4j and self.neo4j.available():
                self.neo4j.add_fact(s, p, o)
        self.kb.save()

    def add_chat_turn(self, role: str, text: str) -> None:
        self.chat.add(role, text)

    def record_rule(self, name: str, weight: float) -> None:
        if self.neo4j and self.neo4j.available():
            self.neo4j.add_rule(name, weight)

    def record_rule_weight(self, name: str, weight: float, score: float) -> None:
        if self.neo4j and self.neo4j.available():
            self.neo4j.add_rule_weight(name, weight, score)

    def record_task(self, name: str, status: str) -> None:
        if self.neo4j and self.neo4j.available():
            self.neo4j.add_task(name, status)

    def record_trace(self, rules: list[str], score: float) -> None:
        if self.neo4j and self.neo4j.available():
            self.neo4j.add_trace(rules, score)

    def close(self) -> None:
        if self.neo4j:
            self.neo4j.close()

    def retrieve(self, limit: int = 10) -> List[str]:
        items = self.short.items()
        if len(items) < limit:
            items = self.long.items(limit=limit)
        return items[-limit:]

    def retrieve_relevant(self, query: str, limit: int = 10) -> List[str]:
        hits = self.chat.search(query, k=limit)
        if hits:
            return hits[:limit]
        return self.retrieve(limit=limit)

    def summarize_if_needed(self, window: int, min_turns: int, summarizer) -> str | None:
        return self.chat.summarize_if_needed(window=window, min_turns=min_turns, summarizer=summarizer)

    def _extract_facts(self, text: str) -> List[Tuple[str, str, str]]:
        facts: List[Tuple[str, str, str]] = []
        # simple pattern: "X 是 Y"
        m = re.findall(r"(\S+)\s*是\s*(\S+)", text)
        for s, o in m:
            facts.append((s, "is", o))
        # pattern: "X 包含 Y"
        m = re.findall(r"(\S+)\s*包含\s*(\S+)", text)
        for s, o in m:
            facts.append((s, "contains", o))
        return facts
