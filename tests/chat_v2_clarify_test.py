from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.llm import NoopLLMClient
from system.chat_v2.memory import CuratedMemoryStore
from system.chat_v2.pipeline import ChatPipeline, PipelineConfig, build_request


class FakeReasoner:
    def handle(self, req, intent, *, memory_hits=None, rag_context=None):
        del req, intent, memory_hits, rag_context

        class _Outcome:
            handled = True
            text = "Need more information."
            source = "reasoning"
            metadata = {
                "action": "clarify_constraints",
                "followup": "which constraint wins",
                "requires_clarification": True,
            }

        return _Outcome()


def _memory_store(path: Path) -> CuratedMemoryStore:
    path.write_text("", encoding="utf-8")
    return CuratedMemoryStore(path=str(path))


def test_project_question_without_evidence_asks_scoped_clarification() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = _memory_store(Path(td) / "memory.jsonl")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=False),
        )
        out = pipeline.respond(build_request("What does this project do?", []))
        assert out.source == "clarify"
        text = out.text.lower()
        assert "architecture" in text or "runtime flow" in text or "setup" in text


def test_workspace_action_without_path_asks_for_target_path() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = _memory_store(Path(td) / "memory.jsonl")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=False),
        )
        out = pipeline.respond(build_request("open the file", []))
        assert out.source == "clarify"
        text = out.text.lower()
        assert "file" in text or "path" in text


def test_reasoning_with_missing_signal_turns_into_targeted_question() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = _memory_store(Path(td) / "memory.jsonl")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=False),
            reasoning_runner=FakeReasoner(),
        )
        out = pipeline.respond(build_request("We must deploy now, but we cannot deploy now. What should we do?", []))
        assert out.source == "clarify"
        text = out.text.lower()
        assert "deadline" in text or "rollback" in text or "constraint" in text


def main() -> None:
    test_project_question_without_evidence_asks_scoped_clarification()
    test_workspace_action_without_path_asks_for_target_path()
    test_reasoning_with_missing_signal_turns_into_targeted_question()
    print("chat_v2_clarify_ok")


if __name__ == "__main__":
    main()
