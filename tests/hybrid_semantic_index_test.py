import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.learning.dialogue.dialogue_generator import DialogueGenerator
from system.knowledge.semantic.index import HybridSemanticIndex


def _test_hybrid_index_basic() -> None:
    idx = HybridSemanticIndex(auto_persist=False)
    idx.build(
        [
            ("study", "ml intro roadmap", "start with linear algebra and probability"),
            ("food", "tomato egg recipe", "fry egg first then add tomato"),
            ("code", "asyncio concurrency", "use event loop and coroutines"),
        ]
    )

    hits = idx.query(
        query_text="how to start ml intro",
        intent="study",
        top_k=1,
        use_embeddings=False,
        transformer_similarity=None,
    )
    assert hits, "hybrid index should return at least one hit"
    assert "linear algebra" in hits[0].response, f"unexpected top response: {hits[0].response}"


def _test_hybrid_index_trie_prefix() -> None:
    idx = HybridSemanticIndex(auto_persist=False)
    idx.build(
        [
            ("code", "asyncio concurrency", "use event loop and coroutines"),
            ("other", "tomato egg recipe", "fry egg first then add tomato"),
        ]
    )
    hits = idx.query(
        query_text="async code pattern",
        intent="code",
        top_k=1,
        use_embeddings=False,
        transformer_similarity=None,
    )
    assert hits, "prefix retrieval should return at least one hit"
    assert "event loop" in hits[0].response, f"unexpected top response: {hits[0].response}"


def _test_persistence_and_trace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "hybrid_index.json"
        trace_path = Path(tmp) / "reason_trace.jsonl"

        idx = HybridSemanticIndex(
            index_path=str(index_path),
            trace_path=str(trace_path),
            auto_persist=False,
        )
        idx.build(
            [
                ("study", "ml intro roadmap", "start with linear algebra and probability"),
                ("code", "asyncio concurrency", "use event loop and coroutines"),
            ]
        )
        assert idx.save(), "hybrid index save should succeed"
        assert index_path.exists(), "hybrid index file should exist"

        idx2 = HybridSemanticIndex(
            index_path=str(index_path),
            trace_path=str(trace_path),
            auto_persist=False,
        )
        hits = idx2.query(
            query_text="ml intro",
            intent="study",
            top_k=1,
            use_embeddings=False,
            transformer_similarity=None,
            trace_meta={"from_test": True},
        )
        assert hits, "loaded index should return hits"
        assert "linear algebra" in hits[0].response, f"unexpected loaded hit: {hits[0].response}"
        assert trace_path.exists(), "trace file should exist after query"
        lines = [ln for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert lines, "trace file should contain at least one line"
        payload = json.loads(lines[-1])
        assert payload.get("query") == "ml intro"


def _test_dialogue_generator_integration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mem_path = str(Path(tmp) / "dialogue_patterns.json")
        gen = DialogueGenerator(memory_path=mem_path)
        ok1 = gen.learn_sample(
            user_input="ml intro roadmap",
            response="start with linear algebra and probability",
            intent="study",
            persist=True,
        )
        ok2 = gen.learn_sample(
            user_input="tomato egg recipe",
            response="fry egg first then add tomato",
            intent="food",
            persist=True,
        )
        assert ok1 and ok2
        gen.flush()

        index_path = Path(mem_path).with_name("hybrid_semantic_index.json")
        assert index_path.exists(), "dialogue generator should persist hybrid index"

        gen2 = DialogueGenerator(memory_path=mem_path)
        reply = gen2.generate_response("how to start ml intro", intent="study")
        assert "linear algebra" in reply, f"unexpected dialogue reply: {reply}"


def main() -> None:
    os.environ["DIALOGUE_HYBRID_INDEX"] = "1"
    os.environ["DIALOGUE_EMBEDDINGS"] = "0"
    os.environ["DIALOGUE_SIM_MIN"] = "0.2"
    os.environ["DIALOGUE_TRACE_ENABLE"] = "1"
    _test_hybrid_index_basic()
    _test_hybrid_index_trie_prefix()
    _test_persistence_and_trace()
    _test_dialogue_generator_integration()
    print("hybrid_semantic_index_ok")


if __name__ == "__main__":
    main()
