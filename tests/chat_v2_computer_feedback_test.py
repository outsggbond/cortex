from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.build_replay_pool import ReplayBuildConfig, build_replay_pool
from system.chat_v2.agent import AgentConfig, WorkspaceAgent
from system.chat_v2.llm import NoopLLMClient
from system.chat_v2.memory import CuratedMemoryStore
from system.chat_v2.pipeline import ChatPipeline, PipelineConfig, build_request


def _memory_store(path: Path) -> CuratedMemoryStore:
    path.write_text("", encoding="utf-8")
    return CuratedMemoryStore(path=str(path))


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _pipeline(
    root: Path,
    *,
    allow_browser: bool = False,
    allow_desktop: bool = False,
    feedback_path: Path,
    stats_path: Path,
    reflections_path: Path,
) -> ChatPipeline:
    memory = _memory_store(root / "memory.jsonl")
    agent = WorkspaceAgent(
        AgentConfig(
            enabled=True,
            project_root=str(root),
            allow_browser=bool(allow_browser),
            allow_desktop=bool(allow_desktop),
            enable_computer_feedback=True,
            computer_feedback_path=feedback_path.as_posix(),
            computer_feedback_stats_path=stats_path.as_posix(),
            computer_reflections_path=reflections_path.as_posix(),
            replan_max=0,
        ),
        llm_client=NoopLLMClient(),
    )
    return ChatPipeline(
        memory_store=memory,
        llm_client=NoopLLMClient(),
        config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False),
        agent_runner=agent,
    )


def test_computer_feedback_records_success_event() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            pipeline = _pipeline(
                root,
                allow_browser=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
            )
            out = pipeline.respond(build_request("open url https://example.com", []))
            assert out.source == "agent"
            rows = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert rows
            event = rows[-1]
            assert event["status"] == "success"
            assert event["primary_task"] == "plugin:browser_open_url"
            assert event["success"] is True
            assert event["reflection_written"] is False
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            assert int(stats["totals"]["events"]) == 1
            assert int(stats["totals"]["successes"]) == 1
            assert int(stats["totals"]["reflections"]) == 0
            assert not reflections_path.exists()
    finally:
        _restore_env(snapshot)


def test_computer_feedback_failure_writes_reflection_and_replay_sample() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            replay_path = root / "adapter_replay.jsonl"
            manifest_path = root / "adapter_replay_manifest.json"
            pipeline = _pipeline(
                root,
                allow_desktop=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
            )
            query = "focus window definitely-not-a-real-window-title-xyz"
            out = pipeline.respond(build_request(query, []))
            assert out.source == "agent"
            rows = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert rows
            event = rows[-1]
            assert event["status"] == "failed"
            assert event["reason_code"] == "window_not_found"
            assert event["reflection_written"] is True
            assert reflections_path.exists()
            reflection_rows = [
                json.loads(line) for line in reflections_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            assert reflection_rows
            reflection = reflection_rows[-1]
            assert reflection["query"] == query
            assert "retry the task" in str(reflection.get("repaired", "")).lower()

            manifest = build_replay_pool(
                ReplayBuildConfig(
                    chat_memory_path=(root / "chat_memory.jsonl").as_posix(),
                    reflections_path=reflections_path.as_posix(),
                    output_path=replay_path.as_posix(),
                    manifest_path=manifest_path.as_posix(),
                    include_chat_memory=False,
                    include_reflections=True,
                    max_samples=10,
                    min_score=0.5,
                    seed=3,
                )
            )
            replay_rows = [json.loads(line) for line in replay_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert int(manifest.get("total_samples", 0)) == len(replay_rows)
            assert any(str(row.get("prompt", "")) == query for row in replay_rows)
    finally:
        _restore_env(snapshot)


def test_computer_feedback_marks_multistep_task_completed() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            pipeline = _pipeline(
                root,
                allow_desktop=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
            )
            query = 'click 10 20 verify text "Dry Run Text" then ocr screen'
            out = pipeline.respond(build_request(query, []))
            assert out.source == "agent"
            assert out.metadata["computer_task_completion"] == "completed"
            assert out.metadata["computer_task_success"] is True
            rows = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            event = rows[-1]
            assert event["status"] == "success"
            assert event["task_completion_status"] == "completed"
            assert event["task_completion_rule"] == "explicit_goal_verified"
            assert event["task_step_count"] == 2
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            assert int(stats["totals"]["task_completed"]) == 1
            assert int(stats["by_completion_status"]["completed"]) == 1
            assert not reflections_path.exists()
    finally:
        _restore_env(snapshot)


def test_computer_feedback_marks_multistep_task_incomplete_and_reflects() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            pipeline = _pipeline(
                root,
                allow_desktop=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
            )
            query = 'click 10 20 verify text "Missing Text" retries 0 then ocr screen'
            out = pipeline.respond(build_request(query, []))
            assert out.source == "agent"
            assert out.metadata["computer_task_completion"] == "incomplete"
            assert out.metadata["computer_task_success"] is False
            rows = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            event = rows[-1]
            assert event["status"] == "unverified"
            assert event["task_completion_status"] == "incomplete"
            assert event["task_completion_rule"] == "explicit_goal_unverified"
            assert event["reflection_written"] is True
            reflection_rows = [json.loads(line) for line in reflections_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert reflection_rows
            assert "verification condition" in str(reflection_rows[-1].get("repaired", "")).lower() or "verify" in str(reflection_rows[-1].get("repaired", "")).lower()
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            assert int(stats["totals"]["task_incomplete"]) == 1
            assert int(stats["by_completion_status"]["incomplete"]) == 1
    finally:
        _restore_env(snapshot)


def main() -> None:
    test_computer_feedback_records_success_event()
    test_computer_feedback_failure_writes_reflection_and_replay_sample()
    test_computer_feedback_marks_multistep_task_completed()
    test_computer_feedback_marks_multistep_task_incomplete_and_reflects()
    print("chat_v2_computer_feedback_ok")


if __name__ == "__main__":
    main()
