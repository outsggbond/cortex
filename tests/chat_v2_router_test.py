from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.agent import AgentOutcome
from system.chat_v2.intents import ChatIntent
from system.chat_v2.llm import NoopLLMClient
from system.chat_v2.memory import CuratedMemoryStore
from system.chat_v2.pipeline import ChatPipeline, PipelineConfig, build_request
from system.chat_v2.router import RouteKind, classify_route


class SpyAgent:
    def __init__(self, text: str = "agent handled") -> None:
        self.text = text
        self.calls = []

    def handle(self, req, intent):
        self.calls.append({"query": str(req.user_text), "intent": str(intent)})
        return AgentOutcome(handled=True, text=self.text, source="agent")


class FakeRagStore:
    def __init__(self, text: str, path: str = "docs/overview.md") -> None:
        self.text = text
        self.path = path

    def build_context(self, query: str, k: int = 4, min_sim: float = 0.2, max_chars: int = 1200):
        del query, k, min_sim, max_chars
        return {
            "snippets": [
                {
                    "text": self.text,
                    "path": self.path,
                    "score": 0.92,
                }
            ]
        }


def _memory_store(path: Path) -> CuratedMemoryStore:
    path.write_text("", encoding="utf-8")
    return CuratedMemoryStore(path=str(path))


def test_route_classifier_distinguishes_core_task_types() -> None:
    assert classify_route("read docs/README.md", ChatIntent.TASK).kind == RouteKind.WORKSPACE_ACTION
    assert classify_route("template browser_capture_page url=https://example.com", ChatIntent.TASK).kind == RouteKind.WORKSPACE_ACTION
    assert classify_route("resume last computer task", ChatIntent.TASK).kind == RouteKind.WORKSPACE_ACTION
    assert classify_route("打开QQ给盛哥发一句晚安", ChatIntent.TASK).kind == RouteKind.WORKSPACE_ACTION
    assert classify_route("What does this project do?", ChatIntent.TASK).kind == RouteKind.PROJECT_QA
    assert classify_route("Should we migrate now or prepare backup first?", ChatIntent.TASK).kind == RouteKind.REASONING
    assert classify_route("How do I parse json?", ChatIntent.TASK).kind == RouteKind.FACTUAL_QA


def test_project_question_prefers_rag_over_agent() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory = _memory_store(root / "memory.jsonl")
        rag = FakeRagStore("The project is a modular chat runtime with memory, RAG, and agent routing.")
        agent = SpyAgent()
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            rag_store=rag,
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=True),
            agent_runner=agent,
        )
        out = pipeline.respond(build_request("What does this project do?", []))
        assert out.source == "rag"
        assert "modular chat runtime" in out.text.lower()
        assert agent.calls == []


def test_project_inspection_prefers_agent_when_query_is_action_oriented() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory = _memory_store(root / "memory.jsonl")
        rag = FakeRagStore("The project is a modular chat runtime with memory, RAG, and agent routing.")
        agent = SpyAgent(text="agent inspected project locally")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            rag_store=rag,
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=True),
            agent_runner=agent,
        )
        out = pipeline.respond(build_request("检查项目架构", []))
        assert out.source == "agent"
        assert "project locally" in out.text.lower()
        assert len(agent.calls) == 1


def test_workspace_action_still_prefers_agent() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory = _memory_store(root / "memory.jsonl")
        agent = SpyAgent(text="agent inspected README.md")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False),
            agent_runner=agent,
        )
        out = pipeline.respond(build_request("read README.md", []))
        assert out.source == "agent"
        assert "readme" in out.text.lower()
        assert len(agent.calls) == 1


def test_explicit_desktop_command_with_greeting_text_still_prefers_agent() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory = _memory_store(root / "memory.jsonl")
        agent = SpyAgent(text="agent sent the desktop command")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False),
            agent_runner=agent,
        )
        query = 'type "晚安" into control 发送 in window 盛哥 control type Group clear first'
        out = pipeline.respond(build_request(query, []))
        assert out.source == "agent"
        assert "desktop command" in out.text.lower()
        assert len(agent.calls) == 1


def main() -> None:
    test_route_classifier_distinguishes_core_task_types()
    test_project_question_prefers_rag_over_agent()
    test_project_inspection_prefers_agent_when_query_is_action_oriented()
    test_workspace_action_still_prefers_agent()
    test_explicit_desktop_command_with_greeting_text_still_prefers_agent()
    print("chat_v2_router_ok")


if __name__ == "__main__":
    main()
