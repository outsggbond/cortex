from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.llm import BaseLLMClient
from system.chat_v2.memory import CuratedMemoryStore
from system.chat_v2.pipeline import ChatPipeline, PipelineConfig, build_request


class FakeRagStore:
    def __init__(self, text: str, path: str = "docs/runtime.md") -> None:
        self.text = text
        self.path = path

    def build_context(self, query: str, k: int = 4, min_sim: float = 0.2, max_chars: int = 1200):
        del query, k, min_sim, max_chars
        return {
            "snippets": [
                {
                    "text": self.text,
                    "path": self.path,
                    "score": 0.93,
                }
            ]
        }


class FixedLLM(BaseLLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def available(self) -> bool:
        return True

    def generate(self, user_text: str, history) -> str:
        del user_text, history
        return self.reply


def _empty_memory(path: Path) -> CuratedMemoryStore:
    path.write_text("", encoding="utf-8")
    return CuratedMemoryStore(path=str(path))


def test_validation_feedback_records_override_event_and_stats() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        feedback_path = root / "validation_feedback.jsonl"
        stats_path = root / "validation_stats.json"
        memory = _empty_memory(root / "memory.jsonl")
        rag = FakeRagStore("Runtime config lives in config/app.yaml.")
        llm = FixedLLM("Use settings/runtime.yaml for the runtime config.")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=llm,
            rag_store=rag,
            config=PipelineConfig(
                enable_llm=True,
                enable_memory_retrieval=False,
                enable_rag=True,
                enable_validation_feedback=True,
                validation_feedback_path=str(feedback_path),
                validation_stats_path=str(stats_path),
            ),
        )
        out = pipeline.respond(build_request("Where is the runtime config?", []))
        assert out.source == "rag"
        assert feedback_path.exists()
        assert stats_path.exists()

        rows = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 1
        row = rows[0]
        assert str(row.get("validation", "")) == "evidence_path_conflict"
        assert str(row.get("candidate_source", "")) == "llm_rag"
        assert str(row.get("final_source", "")) == "rag"
        assert str(row.get("validation_override", "")) == "rag"

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        assert int(stats.get("totals", {}).get("events", 0)) == 1
        assert int(stats.get("totals", {}).get("overrides", 0)) == 1
        assert int(stats.get("by_validation", {}).get("evidence_path_conflict", 0)) == 1
        assert int(stats.get("by_final_source", {}).get("rag", 0)) == 1


def test_validation_feedback_records_pass_event() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        feedback_path = root / "validation_feedback.jsonl"
        stats_path = root / "validation_stats.json"
        memory = _empty_memory(root / "memory.jsonl")
        rag = FakeRagStore("Project docs say to use json.load(file) for JSON parsing.")
        llm = FixedLLM("Use json.load(file) to parse JSON from disk.")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=llm,
            rag_store=rag,
            config=PipelineConfig(
                enable_llm=True,
                enable_memory_retrieval=False,
                enable_rag=True,
                enable_validation_feedback=True,
                validation_feedback_path=str(feedback_path),
                validation_stats_path=str(stats_path),
            ),
        )
        out = pipeline.respond(build_request("How do I parse JSON from disk?", []))
        assert out.source == "llm_rag"

        rows = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 1
        row = rows[0]
        assert str(row.get("validation", "")) == "passed"
        assert str(row.get("candidate_source", "")) == "llm_rag"
        assert str(row.get("final_source", "")) == "llm_rag"
        assert str(row.get("validation_override", "")) == "pass"

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        assert int(stats.get("totals", {}).get("events", 0)) == 1
        assert int(stats.get("totals", {}).get("passed", 0)) == 1
        assert int(stats.get("totals", {}).get("overrides", 0)) == 0
        assert int(stats.get("by_validation", {}).get("passed", 0)) == 1


def main() -> None:
    test_validation_feedback_records_override_event_and_stats()
    test_validation_feedback_records_pass_event()
    print("chat_v2_validation_feedback_ok")


if __name__ == "__main__":
    main()
