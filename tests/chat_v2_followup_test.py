from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.agent import AgentOutcome
from system.chat_v2.llm import NoopLLMClient
from system.chat_v2.memory import CuratedMemoryStore, MemoryHit
from system.chat_v2.pipeline import ChatPipeline, PipelineConfig, build_request


def _empty_memory(path: Path) -> CuratedMemoryStore:
    path.write_text("", encoding="utf-8")
    return CuratedMemoryStore(path=str(path))


def _history_from_turn(user_text: str, out) -> list[dict]:
    return [
        {"role": "user", "text": user_text},
        {
            "role": "assistant",
            "text": out.text,
            "source": out.source,
            "metadata": dict(getattr(out, "metadata", {}) or {}),
        },
    ]


def _append_turn(history: list[dict], user_text: str, out) -> list[dict]:
    rows = list(history)
    rows.append({"role": "user", "text": user_text})
    rows.append(
        {
            "role": "assistant",
            "text": out.text,
            "source": out.source,
            "metadata": dict(getattr(out, "metadata", {}) or {}),
        }
    )
    return rows


class ConditionalRagStore:
    def __init__(self) -> None:
        self.queries = []

    def build_context(self, query: str, k: int = 4, min_sim: float = 0.2, max_chars: int = 1200):
        del k, min_sim, max_chars
        self.queries.append(str(query))
        low = str(query).lower()
        if "runtime flow" in low:
            return {
                "snippets": [
                    {
                        "text": "Runtime flow routes a request through policy, retrieval, reasoning, and agent execution.",
                        "path": "docs/runtime.md",
                        "score": 0.94,
                    }
                ]
            }
        return {"snippets": []}


class ConditionalAgent:
    def __init__(self) -> None:
        self.queries = []

    def handle(self, req, intent):
        del intent
        self.queries.append(str(req.user_text))
        if "config/app.yaml" in str(req.user_text):
            return AgentOutcome(handled=True, text="agent inspected config/app.yaml", source="agent")
        return None


class ConditionalReasoner:
    def __init__(self) -> None:
        self.queries = []

    def handle(self, req, intent, *, memory_hits=None, rag_context=None):
        del intent, memory_hits, rag_context
        self.queries.append(str(req.user_text))
        low = str(req.user_text).lower()

        class _Outcome:
            def __init__(self, text: str, source: str, metadata: dict) -> None:
                self.handled = True
                self.text = text
                self.source = source
                self.metadata = metadata

        if "rollback" in low and "5 minute" in low:
            return _Outcome(
                "Rollback first. Keep the window under 5 minutes.",
                "reasoning",
                {
                    "action": "proceed_with_caution",
                    "followup": "rollback window",
                    "requires_clarification": False,
                },
            )
        if "rollback" in low:
            return _Outcome(
                "Keep rollback ready.",
                "reasoning",
                {
                    "action": "proceed_with_caution",
                    "followup": "whether speed, risk, cost, or rollback matters most",
                    "requires_clarification": False,
                },
            )
        return _Outcome(
            "Need more information.",
            "reasoning",
            {
                "action": "clarify_constraints",
                "followup": "which constraint wins",
                "requires_clarification": True,
            },
        )


class ConditionalMemoryStore:
    def __init__(self) -> None:
        self.queries = []

    def search_many(self, query: str, top_k: int = 3):
        del top_k
        self.queries.append(str(query))
        low = str(query).lower()
        if "json" in low and "file" in low:
            return [MemoryHit(user="json from file", assistant="Use json.load(file).", score=0.96)]
        return []


def test_project_followup_turn_reuses_clarify_context() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = _empty_memory(Path(td) / "memory.jsonl")
        rag = ConditionalRagStore()
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            rag_store=rag,
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=True),
        )
        first = pipeline.respond(build_request("What does this project do?", []))
        assert first.source == "clarify"
        history = _history_from_turn("What does this project do?", first)
        second = pipeline.respond(build_request("runtime flow", history))
        assert second.source == "rag"
        assert "runtime flow" in second.text.lower() or "retrieval" in second.text.lower()
        assert len(rag.queries) >= 2
        assert "focus: runtime flow" in rag.queries[-1].lower()


def test_workspace_followup_turn_injects_missing_path() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = _empty_memory(Path(td) / "memory.jsonl")
        agent = ConditionalAgent()
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=False),
            agent_runner=agent,
        )
        first = pipeline.respond(build_request("open the file", []))
        assert first.source == "clarify"
        history = _history_from_turn("open the file", first)
        second = pipeline.respond(build_request("config/app.yaml", history))
        assert second.source == "agent"
        assert "config/app.yaml" in second.text.lower()
        assert len(agent.queries) >= 2
        assert "target path: config/app.yaml" in agent.queries[-1].lower()


def test_reasoning_followup_turn_reuses_priority_answer() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = _empty_memory(Path(td) / "memory.jsonl")
        reasoner = ConditionalReasoner()
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=False),
            reasoning_runner=reasoner,
        )
        first = pipeline.respond(build_request("We must deploy now, but we cannot deploy now. What should we do?", []))
        assert first.source == "clarify"
        history = _history_from_turn("We must deploy now, but we cannot deploy now. What should we do?", first)
        second = pipeline.respond(build_request("rollback wins", history))
        assert second.source == "reasoning"
        assert "rollback" in second.text.lower()
        assert len(reasoner.queries) >= 2
        assert "priority: rollback wins" in reasoner.queries[-1].lower()


def test_reasoning_followup_accumulates_multiple_constraints() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = _empty_memory(Path(td) / "memory.jsonl")
        reasoner = ConditionalReasoner()
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=False),
            reasoning_runner=reasoner,
        )
        first = pipeline.respond(build_request("We must deploy now, but we cannot deploy now. What should we do?", []))
        assert first.source == "clarify"
        history = _history_from_turn("We must deploy now, but we cannot deploy now. What should we do?", first)
        second = pipeline.respond(build_request("rollback wins", history))
        assert second.source == "reasoning"
        history = _append_turn(history, "rollback wins", second)
        third = pipeline.respond(build_request("5 minute window", history))
        assert third.source == "reasoning"
        assert "5 minutes" in third.text.lower()
        assert len(reasoner.queries) >= 3
        last_query = reasoner.queries[-1].lower()
        assert "priority: rollback wins" in last_query
        assert "constraint: 5 minute window" in last_query


def test_factual_followup_turn_reuses_added_context() -> None:
    memory = ConditionalMemoryStore()
    pipeline = ChatPipeline(
        memory_store=memory,
        llm_client=NoopLLMClient(),
        config=PipelineConfig(enable_llm=True, enable_memory_retrieval=True, enable_rag=False),
    )
    first = pipeline.respond(build_request("How do I parse json?", []))
    assert first.source == "clarify"
    history = _history_from_turn("How do I parse json?", first)
    second = pipeline.respond(build_request("from a file", history))
    assert second.source == "memory"
    assert "json.load" in second.text.lower()
    assert len(memory.queries) >= 2
    assert "context: from a file" in memory.queries[-1].lower()


def main() -> None:
    test_project_followup_turn_reuses_clarify_context()
    test_workspace_followup_turn_injects_missing_path()
    test_reasoning_followup_turn_reuses_priority_answer()
    test_reasoning_followup_accumulates_multiple_constraints()
    test_factual_followup_turn_reuses_added_context()
    print("chat_v2_followup_ok")


if __name__ == "__main__":
    main()
