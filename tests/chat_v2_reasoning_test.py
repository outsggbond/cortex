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
from system.chat_v2.reasoner import ChatReasoner


class FakeRagStore:
    def __init__(self, text: str, path: str = "docs/release.md") -> None:
        self.text = text
        self.path = path

    def build_context(self, query: str, k: int = 4, min_sim: float = 0.2, max_chars: int = 1200):
        del query, k, min_sim, max_chars
        return {
            "snippets": [
                {
                    "text": self.text,
                    "path": self.path,
                    "score": 0.91,
                }
            ]
        }


def _memory_store(path: Path, rows: list[str] | None = None) -> CuratedMemoryStore:
    content = "\n".join(list(rows or []))
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")
    return CuratedMemoryStore(path=str(path), min_similarity=0.40, top_k=3)


def test_reasoner_handles_conflicting_constraints() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory = _memory_store(root / "memory.jsonl")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=True),
            reasoning_runner=ChatReasoner(),
        )
        out = pipeline.respond(build_request("We must deploy now, but we cannot deploy now. What should we do?", []))
        assert out.source == "reasoning"
        text = out.text.lower()
        assert "conflict" in text or "safer direction" in text
        assert "pin down" in text or "constraints" in text


def test_reasoner_uses_memory_and_rag_evidence() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory = _memory_store(
            root / "memory.jsonl",
            rows=[
                '{"user":"safe migration rollout","assistant":"prepare database backup and verify checksum before migration"}',
                '{"user":"release safety review","assistant":"keep rollback steps ready before a production rollout"}',
            ],
        )
        rag = FakeRagStore("Release docs require backup verification before database migration.")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            rag_store=rag,
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=True, enable_rag=True),
            reasoning_runner=ChatReasoner(),
        )
        out = pipeline.respond(build_request("Should we migrate now or prepare backup first?", []))
        assert out.source == "reasoning_memory_rag"
        text = out.text.lower()
        assert "backup" in text
        assert "retrieved project context" in text or "dialogue memory" in text


def main() -> None:
    test_reasoner_handles_conflicting_constraints()
    test_reasoner_uses_memory_and_rag_evidence()
    print("chat_v2_reasoning_ok")


if __name__ == "__main__":
    main()
