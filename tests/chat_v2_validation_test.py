from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.llm import BaseLLMClient, NoopLLMClient
from system.chat_v2.memory import CuratedMemoryStore, MemoryHit
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
                    "score": 0.92,
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


class ConflictingReasoner:
    def handle(self, req, intent, *, memory_hits=None, rag_context=None):
        del req, intent, memory_hits, rag_context

        class _Outcome:
            handled = True
            text = "Choose speed. Use a 30 minute rollout window."
            source = "reasoning"
            metadata = {
                "action": "proceed_with_caution",
                "followup": "rollback window",
                "requires_clarification": False,
            }

        return _Outcome()


class LowConfidenceMemoryStore:
    def search_many(self, query: str, top_k: int = 3):
        del top_k
        if "桂林" not in str(query):
            return []
        return [
            MemoryHit(
                user="哪个最好",
                assistant="这取决于你的具体需求。让我帮你分析一下优缺点。",
                score=0.62,
            )
        ]


def _memory_store(path: Path, rows: list[str]) -> CuratedMemoryStore:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return CuratedMemoryStore(path=str(path), min_similarity=0.35, top_k=3)


def test_llm_answer_conflicting_with_rag_falls_back_to_rag() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory_path = Path(td) / "memory.jsonl"
        memory_path.write_text("", encoding="utf-8")
        memory = CuratedMemoryStore(path=str(memory_path))
        rag = FakeRagStore("Runtime config lives in config/app.yaml.")
        llm = FixedLLM("Use settings/runtime.yaml for the runtime config.")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=llm,
            rag_store=rag,
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=True),
        )
        out = pipeline.respond(build_request("Where is the runtime config?", []))
        assert out.source == "rag"
        assert "config/app.yaml" in out.text
        assert str(out.metadata.get("validation", "")) == "evidence_path_conflict"


def test_llm_answer_conflicting_with_memory_falls_back_to_memory() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory = _memory_store(
            Path(td) / "memory.jsonl",
            rows=[
                '{"user":"parse json from file","assistant":"Use json.load(file) to parse JSON from disk."}',
            ],
        )
        llm = FixedLLM("Use yaml.safe_load(config_text) for that input.")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=llm,
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=True, enable_rag=False),
        )
        out = pipeline.respond(build_request("How do I parse JSON from a file?", []))
        assert out.source == "memory"
        assert "json.load" in out.text
        assert str(out.metadata.get("validation", "")) == "evidence_code_conflict"


def test_reasoning_answer_conflicting_with_thread_context_asks_to_clarify() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory_path = Path(td) / "memory.jsonl"
        memory_path.write_text("", encoding="utf-8")
        memory = CuratedMemoryStore(path=str(memory_path))
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=False),
            reasoning_runner=ConflictingReasoner(),
        )
        history = [
            {"role": "user", "text": "We must deploy now, but we cannot deploy now. What should we do?"},
            {
                "role": "assistant",
                "text": "Keep rollback ready.",
                "source": "reasoning",
                "metadata": {
                    "thread_context": {
                        "original_query": "We must deploy now, but we cannot deploy now. What should we do?",
                        "route_kind": "reasoning",
                        "items": [
                            {"label": "Priority", "value": "rollback wins"},
                        ],
                    }
                },
            },
        ]
        out = pipeline.respond(build_request("5 minute window", history))
        assert out.source == "clarify"
        text = out.text.lower()
        assert "5 minute" in text or "constraint" in text or "priority" in text
        assert str(out.metadata.get("validation", "")).startswith("thread_")


def test_code_only_llm_answer_is_kept_for_code_request() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory_path = Path(td) / "memory.jsonl"
        memory_path.write_text("", encoding="utf-8")
        memory = CuratedMemoryStore(path=str(memory_path))
        llm = FixedLLM(
            '#include <iostream>\n'
            'using namespace std;\n\n'
            'int main() {\n'
            '    cout << "Hello, World!" << endl;\n'
            '    return 0;\n'
            '}'
        )
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=llm,
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=False),
        )
        out = pipeline.respond(build_request("给我写一个C++版本的hello world", []))
        assert out.source == "llm"
        assert "Hello, World!" in out.text


def test_low_confidence_memory_does_not_override_factual_llm() -> None:
    llm = FixedLLM("桂林位于中国广西壮族自治区东北部。")
    pipeline = ChatPipeline(
        memory_store=LowConfidenceMemoryStore(),
        llm_client=llm,
        config=PipelineConfig(enable_llm=True, enable_memory_retrieval=True, enable_rag=False),
    )
    out = pipeline.respond(build_request("桂林在哪个地方？", []))
    assert out.source == "llm"
    assert "广西" in out.text


def main() -> None:
    test_llm_answer_conflicting_with_rag_falls_back_to_rag()
    test_llm_answer_conflicting_with_memory_falls_back_to_memory()
    test_reasoning_answer_conflicting_with_thread_context_asks_to_clarify()
    test_code_only_llm_answer_is_kept_for_code_request()
    test_low_confidence_memory_does_not_override_factual_llm()
    print("chat_v2_validation_ok")


if __name__ == "__main__":
    main()
