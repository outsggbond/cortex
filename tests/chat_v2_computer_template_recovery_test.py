from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import system.chat_v2.agent as agent_module
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
    recovery_path: Path,
) -> ChatPipeline:
    memory = _memory_store(root / "memory.jsonl")
    agent = WorkspaceAgent(
        AgentConfig(
            enabled=True,
            project_root=str(root),
            allow_browser=bool(allow_browser),
            allow_desktop=bool(allow_desktop),
            enable_computer_feedback=True,
            enable_computer_recovery=True,
            computer_feedback_path=feedback_path.as_posix(),
            computer_feedback_stats_path=stats_path.as_posix(),
            computer_reflections_path=reflections_path.as_posix(),
            computer_recovery_path=recovery_path.as_posix(),
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


def test_browser_template_runs_and_writes_non_resumable_checkpoint() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "computer_trace.jsonl"
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            recovery_path = root / "computer_recovery.json"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            pipeline = _pipeline(
                root,
                allow_browser=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
                recovery_path=recovery_path,
            )
            out = pipeline.respond(build_request("template browser_capture_page url=https://example.com", []))
            assert out.source == "agent"
            assert "Opened browser session default" in out.text
            assert "Saved browser screenshot" in out.text
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert [str(row.get("plugin", "")) for row in rows] == ["browser_dom_open", "browser_dom_screenshot"]
            payload = json.loads(recovery_path.read_text(encoding="utf-8"))
            assert payload["resume_available"] is False
            assert payload["status"] in {"completed", "completed_inferred", "success"}
    finally:
        _restore_env(snapshot)


def test_desktop_window_controls_template_runs_and_writes_non_resumable_checkpoint() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "computer_trace.jsonl"
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            recovery_path = root / "computer_recovery.json"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            pipeline = _pipeline(
                root,
                allow_desktop=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
                recovery_path=recovery_path,
            )
            out = pipeline.respond(build_request('template desktop_window_controls title="QQ"', []))
            assert out.source == "agent"
            assert "Focused window QQ" in out.text
            assert "Listed window controls" in out.text
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert [str(row.get("plugin", "")) for row in rows] == ["desktop_focus_window", "desktop_list_controls"]
            payload = json.loads(recovery_path.read_text(encoding="utf-8"))
            assert payload["resume_available"] is False
            assert payload["status"] in {"completed", "completed_inferred", "success"}
    finally:
        _restore_env(snapshot)


def test_incomplete_task_writes_resumable_checkpoint() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "computer_trace.jsonl"
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            recovery_path = root / "computer_recovery.json"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            pipeline = _pipeline(
                root,
                allow_desktop=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
                recovery_path=recovery_path,
            )
            query = 'click 10 20 verify text "Dry Run Text" then click 20 30 verify text "Missing Text" retries 0 then ocr screen'
            out = pipeline.respond(build_request(query, []))
            assert out.source == "agent"
            assert out.metadata["computer_task_resume_available"] is True
            assert out.metadata["computer_task_resume_command"] == "resume last computer task"
            assert out.metadata["computer_task_repair_from_index"] == 0
            assert out.metadata["computer_task_repair_strategy"] == "resume_from_last_verified_step"
            payload = json.loads(recovery_path.read_text(encoding="utf-8"))
            assert payload["resume_available"] is True
            assert payload["resume_from_index"] == 1
            assert payload["repair_from_index"] == 0
            assert payload["repair_strategy"] == "resume_from_last_verified_step"
            assert payload["completed_step_count"] == 1
            assert payload["step_count"] == 3
            remaining = [str(item.get("name", "")) for item in payload["remaining_steps"]]
            assert remaining == ["plugin:desktop_click", "plugin:desktop_ocr"]
            repair_steps = [str(item.get("name", "")) for item in payload["repair_steps"]]
            assert repair_steps == ["plugin:desktop_click", "plugin:desktop_click", "plugin:desktop_ocr"]
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert [str(row.get("plugin", "")) for row in rows] == ["desktop_click", "desktop_click", "desktop_ocr"]
    finally:
        _restore_env(snapshot)


def test_browser_failure_prefers_session_repair_anchor() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "computer_trace.jsonl"
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            recovery_path = root / "computer_recovery.json"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            pipeline = _pipeline(
                root,
                allow_browser=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
                recovery_path=recovery_path,
            )
            query = 'browser open https://example.com then browser click "#login" then browser type "hello" into "#search" verify text "Missing Text" retries 0 then browser screenshot'
            out = pipeline.respond(build_request(query, []))
            assert out.source == "agent"
            assert out.metadata["computer_task_resume_available"] is True
            assert out.metadata["computer_task_repair_from_index"] == 0
            assert out.metadata["computer_task_repair_strategy"] == "repair_browser_session"
            payload = json.loads(recovery_path.read_text(encoding="utf-8"))
            assert payload["resume_from_index"] == 2
            assert payload["repair_from_index"] == 0
            assert payload["repair_strategy"] == "repair_browser_session"
            assert [str(item.get("name", "")) for item in payload["remaining_steps"]] == [
                "plugin:browser_dom_type",
                "plugin:browser_dom_screenshot",
            ]
            assert [str(item.get("name", "")) for item in payload["repair_steps"]] == [
                "plugin:browser_dom_open",
                "plugin:browser_dom_click",
                "plugin:browser_dom_type",
                "plugin:browser_dom_screenshot",
            ]
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert [str(row.get("plugin", "")) for row in rows] == [
                "browser_dom_open",
                "browser_dom_click",
                "browser_dom_type",
                "browser_dom_screenshot",
            ]
    finally:
        _restore_env(snapshot)


def test_window_failure_inserts_list_windows_repair_step() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            recovery_path = root / "computer_recovery.json"
            pipeline = _pipeline(
                root,
                allow_desktop=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
                recovery_path=recovery_path,
            )
            query = "focus window definitely-not-a-real-window-title-xyz then screenshot"
            out = pipeline.respond(build_request(query, []))
            assert out.source == "agent"
            assert out.metadata["computer_task_resume_available"] is True
            assert out.metadata["computer_task_repair_from_index"] == 0
            assert out.metadata["computer_task_repair_strategy"] == "repair_desktop_refresh_windows"
            payload = json.loads(recovery_path.read_text(encoding="utf-8"))
            assert payload["resume_from_index"] == 0
            assert payload["repair_from_index"] == 0
            assert payload["repair_strategy"] == "repair_desktop_refresh_windows"
            assert [str(item.get("name", "")) for item in payload["remaining_steps"]] == [
                "plugin:desktop_focus_window",
                "plugin:desktop_screenshot",
            ]
            assert [str(item.get("name", "")) for item in payload["repair_steps"]] == [
                "plugin:desktop_list_windows",
                "plugin:desktop_focus_window",
                "plugin:desktop_screenshot",
            ]
    finally:
        _restore_env(snapshot)


def test_resume_last_computer_task_replays_from_verified_anchor() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "computer_trace.jsonl"
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            recovery_path = root / "computer_recovery.json"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            recovery_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "query": "desktop workflow",
                        "resume_available": True,
                        "resume_from_index": 2,
                        "repair_from_index": 1,
                        "repair_strategy": "resume_from_last_verified_step",
                        "completed_step_count": 2,
                        "step_count": 3,
                        "repair_steps": [
                            {
                                "name": "plugin:desktop_focus_window",
                                "detail": "focus window Notepad",
                                "priority": 1,
                                "status": "pending",
                                "payload": {"title": "Notepad", "verify_window_title": "Notepad"},
                            },
                            {
                                "name": "plugin:desktop_ocr",
                                "detail": "ocr screen",
                                "priority": 1,
                                "status": "pending",
                                "payload": {},
                            },
                        ],
                        "remaining_steps": [
                            {
                                "name": "plugin:desktop_ocr",
                                "detail": "ocr screen",
                                "priority": 1,
                                "status": "pending",
                                "payload": {},
                            }
                        ],
                        "suggested_command": "resume last computer task",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            pipeline = _pipeline(
                root,
                allow_desktop=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
                recovery_path=recovery_path,
            )
            out = pipeline.respond(build_request("resume last computer task", []))
            assert out.source == "agent"
            assert "Focused window Notepad" in out.text
            assert "OCR text: Dry Run Text" in out.text
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert [str(row.get("plugin", "")) for row in rows] == ["desktop_focus_window", "desktop_ocr"]
    finally:
        _restore_env(snapshot)


def test_resume_last_computer_task_keeps_old_remaining_step_checkpoint_compatible() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "computer_trace.jsonl"
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            recovery_path = root / "computer_recovery.json"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            recovery_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "query": "desktop workflow",
                        "resume_available": True,
                        "resume_from_index": 2,
                        "completed_step_count": 2,
                        "step_count": 3,
                        "remaining_steps": [
                            {
                                "name": "plugin:desktop_ocr",
                                "detail": "ocr screen",
                                "priority": 1,
                                "status": "pending",
                                "payload": {},
                            }
                        ],
                        "suggested_command": "resume last computer task",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            pipeline = _pipeline(
                root,
                allow_desktop=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
                recovery_path=recovery_path,
            )
            out = pipeline.respond(build_request("resume last computer task", []))
            assert out.source == "agent"
            assert "OCR text: Dry Run Text" in out.text
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert [str(row.get("plugin", "")) for row in rows] == ["desktop_ocr"]
    finally:
        _restore_env(snapshot)


def test_failed_recovery_save_does_not_expose_resume_metadata() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    original_save = agent_module.save_computer_recovery_payload
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            feedback_path = root / "computer_feedback.jsonl"
            stats_path = root / "computer_stats.json"
            reflections_path = root / "dialogue_reflections.jsonl"
            recovery_path = root / "computer_recovery.json"
            pipeline = _pipeline(
                root,
                allow_desktop=True,
                feedback_path=feedback_path,
                stats_path=stats_path,
                reflections_path=reflections_path,
                recovery_path=recovery_path,
            )
            agent_module.save_computer_recovery_payload = lambda path, payload: False
            out = pipeline.respond(
                build_request('click 10 20 verify text "Dry Run Text" then click 20 30 verify text "Missing Text" retries 0 then ocr screen', [])
            )
            assert out.source == "agent"
            assert out.metadata["computer_task_resume_save_failed"] is True
            assert "computer_task_resume_available" not in out.metadata
            assert "computer_task_resume_command" not in out.metadata
            assert not recovery_path.exists()
    finally:
        agent_module.save_computer_recovery_payload = original_save
        _restore_env(snapshot)


def main() -> None:
    test_browser_template_runs_and_writes_non_resumable_checkpoint()
    test_desktop_window_controls_template_runs_and_writes_non_resumable_checkpoint()
    test_incomplete_task_writes_resumable_checkpoint()
    test_browser_failure_prefers_session_repair_anchor()
    test_window_failure_inserts_list_windows_repair_step()
    test_resume_last_computer_task_replays_from_verified_anchor()
    test_resume_last_computer_task_keeps_old_remaining_step_checkpoint_compatible()
    test_failed_recovery_save_does_not_expose_resume_metadata()
    print("chat_v2_computer_template_recovery_ok")


if __name__ == "__main__":
    main()
