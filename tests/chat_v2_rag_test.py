from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.llm import BaseLLMClient, NoopLLMClient
from system.chat_v2.memory import CuratedMemoryStore
from system.chat_v2.pipeline import ChatPipeline, PipelineConfig, build_request


class FakeRagStore:
    def __init__(self, text: str, path: str = "docs/cloud.md") -> None:
        self.text = text
        self.path = path
        self.calls = []

    def build_context(self, query: str, k: int = 4, min_sim: float = 0.2, max_chars: int = 1200):
        self.calls.append(
            {
                "query": query,
                "k": int(k),
                "min_sim": float(min_sim),
                "max_chars": int(max_chars),
            }
        )
        return {
            "snippets": [
                {
                    "text": self.text,
                    "path": self.path,
                    "score": 0.93,
                }
            ]
        }


class CaptureLLM(BaseLLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts = []

    def available(self) -> bool:
        return True

    def generate(self, user_text: str, history) -> str:
        self.prompts.append(str(user_text))
        return self.reply


def _empty_memory_store() -> CuratedMemoryStore:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        return CuratedMemoryStore(path=str(path))


def test_llm_receives_rag_augmented_query() -> None:
    memory = _empty_memory_store()
    rag = FakeRagStore("Project docs say to call json.load(file) for JSON parsing.")
    llm = CaptureLLM("Use json.load(file) to parse JSON from disk.")
    pipeline = ChatPipeline(
        memory_store=memory,
        llm_client=llm,
        rag_store=rag,
        config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=True),
    )
    out = pipeline.respond(build_request("How do I read json?", []))
    assert out.source == "llm_rag"
    assert "json.load" in out.text.lower()
    assert llm.prompts and "Retrieved Project Context" in llm.prompts[0]
    assert "docs/cloud.md" in llm.prompts[0]
    assert rag.calls and str(rag.calls[0]["query"]) == "How do I read json?"


def test_llm_receives_memory_and_rag_context() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory_path = Path(td) / "curated.jsonl"
        memory_path.write_text(
            "\n".join(
                [
                    '{"user":"python json怎么读取","assistant":"可以用 json.load(file) 读取 JSON 文件。"}',
                    '{"user":"python json解析文件","assistant":"常见做法是先 import json，再调用 json.load。"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        memory = CuratedMemoryStore(path=str(memory_path), min_similarity=0.4, top_k=2)
        rag = FakeRagStore("Project docs mention config/app.yaml for runtime settings.")
        llm = CaptureLLM("结合已有记忆和文档，优先用 json.load(file) 读取。")
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=llm,
            rag_store=rag,
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=True, enable_rag=True),
        )
        out = pipeline.respond(build_request("python json 文件怎么读？", []))
        assert out.source == "llm_memory_rag"
        assert "json.load" in out.text.lower()
        assert llm.prompts
        assert "Retrieved Dialogue Memory" in llm.prompts[0]
        assert "Retrieved Project Context" in llm.prompts[0]
        assert "python json怎么读取" in llm.prompts[0]
        assert "config/app.yaml" in llm.prompts[0]


def test_rag_fallback_without_llm() -> None:
    memory = _empty_memory_store()
    rag = FakeRagStore("Config path is config/app.yaml and logs go to artifacts/audit.")
    pipeline = ChatPipeline(
        memory_store=memory,
        llm_client=NoopLLMClient(),
        rag_store=rag,
        config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=True),
    )
    out = pipeline.respond(build_request("Where is the config?", []))
    assert out.source == "rag"
    assert "config/app.yaml" in out.text


def main() -> None:
    test_llm_receives_rag_augmented_query()
    test_llm_receives_memory_and_rag_context()
    test_rag_fallback_without_llm()
    print("chat_v2_rag_ok")


if __name__ == "__main__":
    main()
