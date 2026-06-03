import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.learning.dialogue.dialogue_evolver import (
    DialogueFailureReflector,
    DialogueRouteSelector,
    DialogueSelfPlay,
    score_response_quality,
)
from system.learning.dialogue.dialogue_generator import DialogueGenerator


def _test_route_selector_ordering() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "route_stats.json"
        trace_path = Path(tmp) / "route_trace.jsonl"
        selector = DialogueRouteSelector(
            stats_path=str(stats_path),
            trace_path=str(trace_path),
            enabled=True,
            auto_persist=False,
        )
        selector.record("route_a", success=False, quality=0.05, context="chat")
        selector.record("route_a", success=False, quality=0.10, context="chat")
        selector.record("route_b", success=True, quality=0.85, context="chat")
        selector.record("route_b", success=True, quality=0.90, context="chat")
        order = selector.choose_order(["route_a", "route_b"], context="chat")
        assert order and order[0] == "route_b", f"unexpected route order: {order}"
        board = selector.leaderboard(context="chat", limit=2, routes=["route_a", "route_b"])
        assert board and board[0].get("route") == "route_b", f"unexpected leaderboard: {board}"
        required_keys = {"route", "score", "attempts", "success_rate", "avg_quality", "fail_streak"}
        assert required_keys.issubset(set(board[0].keys())), f"leaderboard keys missing: {board[0].keys()}"
        assert selector.save(), "selector save should succeed"
        assert stats_path.exists(), "route stats file should exist"
        assert trace_path.exists(), "route trace file should exist"


def _test_selfplay_reflection_and_writeback() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mem_path = str(Path(tmp) / "dialogue_patterns.json")
        reflect_path = Path(tmp) / "dialogue_reflections.jsonl"
        selfplay_log = Path(tmp) / "dialogue_selfplay.jsonl"
        route_stats = Path(tmp) / "route_stats.json"
        route_trace = Path(tmp) / "route_trace.jsonl"

        gen = DialogueGenerator(memory_path=mem_path)
        gen.learn_sample(
            user_input="hello there",
            response="hello, what can I help with?",
            intent="smalltalk",
            persist=True,
        )
        gen.learn_sample(
            user_input="how to start ml",
            response="start with linear algebra and probability",
            intent="study",
            persist=True,
        )

        selector = DialogueRouteSelector(
            stats_path=str(route_stats),
            trace_path=str(route_trace),
            enabled=True,
            auto_persist=False,
        )
        reflector = DialogueFailureReflector(path=str(reflect_path))
        selfplay = DialogueSelfPlay(
            log_path=str(selfplay_log),
            enabled=True,
            every_turns=1,
            batch_size=2,
            seed=3,
        )

        def _bad_runner(route: str, prompt: str, intent: str | None) -> str:
            del route, prompt, intent
            return "idk"

        summary = selfplay.run(
            turn_id=1,
            dialogue_gen=gen,
            available_routes=["bad_route"],
            selector=selector,
            run_route=_bad_runner,
            reflector=reflector,
        )
        assert summary.get("runs", 0) > 0, f"unexpected self-play summary: {summary}"
        assert summary.get("failed", 0) > 0, "expected failed self-play samples"
        assert summary.get("writeback", 0) > 0, "expected failed samples writeback"
        gen.flush()
        assert reflect_path.exists(), "reflection log should exist"
        assert selfplay_log.exists(), "self-play log should exist"
        payload = json.loads(selfplay_log.read_text(encoding="utf-8").splitlines()[-1])
        assert int(payload.get("failed", 0)) >= 1
        persisted = json.loads(Path(mem_path).read_text(encoding="utf-8"))
        total_pairs = sum(len((v or {}).get("pairs", [])) for v in persisted.values())
        assert total_pairs >= 2, f"expected self-play memory to remain persisted, pairs={total_pairs}"
        assert any(
            "start with linear algebra and probability" in json.dumps(v, ensure_ascii=False)
            for v in persisted.values()
        ), "expected corrected answer to remain available in dialogue memory"


def _test_quality_score_bounds() -> None:
    q0 = score_response_quality("hi", "")
    q1 = score_response_quality("hi", "hello there")
    q2 = score_response_quality("how to start ml", "start with linear algebra", expected="start with linear algebra and probability")
    assert 0.0 <= q0 <= 1.0
    assert 0.0 <= q1 <= 1.0
    assert 0.0 <= q2 <= 1.0
    assert q2 >= q1 or q2 >= 0.4


def main() -> None:
    os.environ["DIALOGUE_EMBEDDINGS"] = "0"
    os.environ["DIALOGUE_HYBRID_INDEX"] = "1"
    os.environ["DIALOGUE_TRACE_ENABLE"] = "0"
    os.environ["DIALOGUE_ROUTE_SELECT"] = "1"
    os.environ["DIALOGUE_SELFPLAY_ENABLE"] = "1"
    _test_route_selector_ordering()
    _test_selfplay_reflection_and_writeback()
    _test_quality_score_bounds()
    print("dialogue_evolver_ok")


if __name__ == "__main__":
    main()
