from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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


def _pipeline(root: Path, *, allow_browser: bool = False, allow_desktop: bool = False) -> ChatPipeline:
    memory = _memory_store(root / "memory.jsonl")
    agent = WorkspaceAgent(
        AgentConfig(
            enabled=True,
            project_root=str(root),
            allow_browser=bool(allow_browser),
            allow_desktop=bool(allow_desktop),
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


def test_agent_blocks_browser_control_by_default() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pipeline = _pipeline(root)
            out = pipeline.respond(build_request("open url https://example.com", []))
            assert out.source == "agent_blocked"
            assert "--v2-agent-allow-browser" in out.text
    finally:
        _restore_env(snapshot)


def test_agent_runs_browser_plugin_in_dry_run_and_records_trace() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "computer_trace.jsonl"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            pipeline = _pipeline(root, allow_browser=True)
            out = pipeline.respond(build_request("open url https://example.com", []))
            assert out.source == "agent"
            assert "Opened URL https://example.com" in out.text
            assert trace_path.exists()
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert rows and rows[-1]["plugin"] == "browser_open_url"
            assert rows[-1]["result"]["verified"] is True
    finally:
        _restore_env(snapshot)


def test_agent_runs_desktop_plugin_in_dry_run_and_records_trace() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "desktop_trace.jsonl"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            pipeline = _pipeline(root, allow_desktop=True)
            out = pipeline.respond(build_request("list windows", []))
            assert out.source == "agent"
            assert "Listed desktop windows" in out.text
            assert trace_path.exists()
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert rows and rows[-1]["plugin"] == "desktop_list_windows"
            assert rows[-1]["result"]["verified"] is True
    finally:
        _restore_env(snapshot)


def test_agent_runs_browser_dom_sequence_in_dry_run() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "browser_dom_trace.jsonl"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            pipeline = _pipeline(root, allow_browser=True)
            out = pipeline.respond(
                build_request(
                    'browser open https://example.com then browser click "#login" then browser type "hello" into "#search" then browser screenshot',
                    [],
                )
            )
            assert out.source == "agent"
            assert "Opened browser session default" in out.text
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            plugins = [str(row.get("plugin", "")) for row in rows]
            assert plugins == [
                "browser_dom_open",
                "browser_dom_click",
                "browser_dom_type",
                "browser_dom_screenshot",
            ]
            assert all(bool((row.get("result") or {}).get("verified", False)) for row in rows)
    finally:
        _restore_env(snapshot)


def test_agent_runs_desktop_click_drag_ocr_sequence_in_dry_run() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "desktop_automation_trace.jsonl"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            pipeline = _pipeline(root, allow_desktop=True)
            out = pipeline.respond(build_request("click 120 220 then drag 120 220 -> 320 220 then ocr screen", []))
            assert out.source == "agent"
            assert "Clicked left at 120,220" in out.text
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            plugins = [str(row.get("plugin", "")) for row in rows]
            assert plugins == ["desktop_click", "desktop_drag", "desktop_ocr"]
            assert rows[-1]["result"]["text"] == "Dry Run Text"
    finally:
        _restore_env(snapshot)


def test_agent_runs_desktop_uia_control_sequence_in_dry_run() -> None:
    keys = ["EXEC_SNAPSHOT", "COMPUTER_CONTROL_DRY_RUN", "COMPUTER_CONTROL_TRACE_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "desktop_uia_trace.jsonl"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            pipeline = _pipeline(root, allow_desktop=True)
            out = pipeline.respond(
                build_request(
                    'list controls in window QQ then click control 搜索 in window QQ control type Edit then type "盛哥" into control 搜索 in window QQ control type Edit clear first',
                    [],
                )
            )
            assert out.source == "agent"
            assert "Listed window controls" in out.text
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            plugins = [str(row.get("plugin", "")) for row in rows]
            assert plugins == [
                "desktop_list_controls",
                "desktop_click_control",
                "desktop_type_control",
            ]
            assert all(bool((row.get("result") or {}).get("verified", False)) for row in rows)
    finally:
        _restore_env(snapshot)


def main() -> None:
    test_agent_blocks_browser_control_by_default()
    test_agent_runs_browser_plugin_in_dry_run_and_records_trace()
    test_agent_runs_desktop_plugin_in_dry_run_and_records_trace()
    test_agent_runs_browser_dom_sequence_in_dry_run()
    test_agent_runs_desktop_click_drag_ocr_sequence_in_dry_run()
    test_agent_runs_desktop_uia_control_sequence_in_dry_run()
    print("chat_v2_computer_agent_ok")


if __name__ == "__main__":
    main()
