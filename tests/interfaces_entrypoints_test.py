from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import system.app_runtime as app_runtime
from system.interfaces.api import (
    AutomationAPIAdapter,
    ChatAPIAdapter,
    ComputerUseAPIAdapter,
    HTTPAPIRequest,
    HTTPAPIShell,
)
from system.interfaces.cli import run_cli_entrypoint
from system.interfaces.gui import (
    AutomationGUIController,
    ChatGUISession,
    ComputerUseGUISession,
    DesktopWebGUIShell,
)
from system.main_cli import parse_args


@contextmanager
def _patched_attr(obj, name: str, value) -> Iterator[None]:
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def _write_memory(path: Path) -> None:
    path.write_text(
        '{"user":"how to read json in python","assistant":"Use import json and json.load(file)."}\n',
        encoding="utf-8",
    )


def _build_chat_args(memory_path: str):
    return parse_args(
        [
            "--chat",
            "--runtime-arch",
            "v2",
            "--v2-memory-path",
            memory_path,
            "--rag-disable",
        ]
    )


def _build_automation_args(root: Path, config_path: Path, state_path: Path, runs_path: Path):
    return SimpleNamespace(
        automation_project_root=root.as_posix(),
        automation_config=config_path.as_posix(),
        automation_state=state_path.as_posix(),
        automation_runs=runs_path.as_posix(),
        automation_poll_s=30.0,
        automation_max_cycles=0,
    )


def _build_gui_shell_args(
    memory_path: str,
    root: Path,
    config_path: Path,
    state_path: Path,
    runs_path: Path,
):
    args = _build_chat_args(memory_path)
    args.automation_project_root = root.as_posix()
    args.automation_config = config_path.as_posix()
    args.automation_state = state_path.as_posix()
    args.automation_runs = runs_path.as_posix()
    args.automation_poll_s = 30.0
    args.automation_max_cycles = 0
    return args


def test_chat_api_adapter_responds_with_history() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory_path = Path(td) / "dialogue_curated.jsonl"
        _write_memory(memory_path)
        adapter = ChatAPIAdapter(_build_chat_args(memory_path.as_posix()))
        result = adapter.respond("how to read json in python")

    assert result.source == "memory"
    assert "json.load" in result.text.lower()
    assert [turn.role for turn in result.history[-2:]] == ["user", "assistant"]
    assert result.history[-1].source == "memory"


def test_chat_gui_session_tracks_transcript_and_reset() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory_path = Path(td) / "dialogue_curated.jsonl"
        _write_memory(memory_path)
        session = ChatGUISession(_build_chat_args(memory_path.as_posix()))
        reply = session.submit("how to read json in python")
        transcript = session.transcript()
        session.reset()

    assert reply.source == "memory"
    assert [item.role for item in transcript] == ["user", "assistant"]
    assert "json.load" in transcript[-1].text.lower()
    assert session.transcript() == []


def test_automation_api_adapter_lists_and_runs_workflow() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        config_path = root / "config" / "automation.json"
        state_path = root / "artifacts" / "audit" / "automation_state.json"
        runs_path = root / "artifacts" / "audit" / "automation_runs.jsonl"
        out_path = root / "artifacts" / "audit" / "automation" / "hello.txt"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            '{"workflows":[{"id":"write_hello","enabled":true,"trigger":{"type":"manual"},"steps":[{"name":"write_file","path":"artifacts/audit/automation/hello.txt","content":"hello interface\\n"}]}]}\n',
            encoding="utf-8",
        )
        adapter = AutomationAPIAdapter(_build_automation_args(root, config_path, state_path, runs_path))
        workflows = adapter.list_workflows()
        summary = adapter.run(["write_hello"])
        content = out_path.read_text(encoding="utf-8")

    assert workflows[0]["id"] == "write_hello"
    assert summary["ok"] is True
    assert content == "hello interface\n"


def test_automation_gui_controller_tracks_last_state_and_reset() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        config_path = root / "config" / "automation.json"
        state_path = root / "artifacts" / "audit" / "automation_state.json"
        runs_path = root / "artifacts" / "audit" / "automation_runs.jsonl"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            '{"workflows":[{"id":"write_hello","enabled":true,"trigger":{"type":"manual"},"steps":[{"name":"write_file","path":"artifacts/audit/automation/hello.txt","content":"hello gui\\n"}]}]}\n',
            encoding="utf-8",
        )
        controller = AutomationGUIController(_build_automation_args(root, config_path, state_path, runs_path))
        listed = controller.list_workflows()
        summary = controller.run(["write_hello"])
        content = (root / "artifacts" / "audit" / "automation" / "hello.txt").read_text(encoding="utf-8")
        controller.reset()

    assert listed[0]["id"] == "write_hello"
    assert summary["ok"] is True
    assert content == "hello gui\n"
    assert controller.last_summary is None
    assert controller.last_workflows == []


def test_computer_api_adapter_runs_goal_via_runtime_builder() -> None:
    calls: list[str] = []

    class _FakeService:
        def run_goal(self, goal: str | None = None):
            calls.append(goal)
            return {"ok": True, "goal": goal}

    import system.interfaces.api.computer as computer_api

    with _patched_attr(computer_api, "build_computer_domain_service", lambda _args: _FakeService()):
        adapter = ComputerUseAPIAdapter(SimpleNamespace())
        result = adapter.run_goal("open QQ")

    assert result["ok"] is True
    assert calls == ["open QQ"]


def test_computer_gui_session_tracks_history_and_reset() -> None:
    calls: list[str] = []

    class _FakeService:
        def run_goal(self, goal: str | None = None):
            calls.append(goal)
            return {"ok": True, "goal": goal, "status": "done"}

    import system.interfaces.api.computer as computer_api

    with _patched_attr(computer_api, "build_computer_domain_service", lambda _args: _FakeService()):
        session = ComputerUseGUISession(SimpleNamespace())
        result = session.submit_goal("open QQ")
        history = session.history()
        session.reset()

    assert result["status"] == "done"
    assert history == [{"goal": "open QQ", "result": {"ok": True, "goal": "open QQ", "status": "done"}}]
    assert calls == ["open QQ"]
    assert session.history() == []


def test_http_api_shell_routes_chat_and_errors() -> None:
    with tempfile.TemporaryDirectory() as td:
        memory_path = Path(td) / "dialogue_curated.jsonl"
        _write_memory(memory_path)
        shell = HTTPAPIShell(_build_chat_args(memory_path.as_posix()))
        ok_response = shell.handle(
            HTTPAPIRequest(
                method="POST",
                path="/api/chat/respond",
                body={"user_text": "how to read json in python"},
            )
        )
        bad_method = shell.handle(HTTPAPIRequest(method="GET", path="/api/chat/respond"))
        missing = shell.handle(HTTPAPIRequest(method="GET", path="/missing"))

    assert ok_response.status_code == 200
    assert ok_response.body["source"] == "memory"
    assert ok_response.body["history"][-1]["role"] == "assistant"
    assert bad_method.status_code == 405
    assert bad_method.body["ok"] is False
    assert missing.status_code == 404
    assert missing.body["ok"] is False


def test_http_api_shell_routes_automation_and_injected_computer_adapter() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        config_path = root / "config" / "automation.json"
        state_path = root / "artifacts" / "audit" / "automation_state.json"
        runs_path = root / "artifacts" / "audit" / "automation_runs.jsonl"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            '{"workflows":[{"id":"write_hello","enabled":true,"trigger":{"type":"manual"},"steps":[{"name":"write_file","path":"artifacts/audit/automation/hello.txt","content":"hello http\\n"}]}]}\n',
            encoding="utf-8",
        )

        class _FakeComputerAdapter:
            def run_goal(self, goal: str | None = None):
                return {"ok": True, "goal": goal, "status": "done"}

        shell = HTTPAPIShell(
            _build_automation_args(root, config_path, state_path, runs_path),
            computer_adapter=_FakeComputerAdapter(),
        )
        list_response = shell.handle(HTTPAPIRequest(method="GET", path="/api/automation/workflows"))
        run_response = shell.handle(
            HTTPAPIRequest(
                method="POST",
                path="/api/automation/run",
                body={"workflow_ids": ["write_hello"]},
            )
        )
        computer_response = shell.handle(
            HTTPAPIRequest(
                method="POST",
                path="/api/computer/run",
                body={"goal": "open QQ"},
            )
        )
        content = (root / "artifacts" / "audit" / "automation" / "hello.txt").read_text(encoding="utf-8")

    assert list_response.status_code == 200
    assert list_response.body["workflows"][0]["id"] == "write_hello"
    assert run_response.status_code == 200
    assert run_response.body["ok"] is True
    assert content == "hello http\n"
    assert computer_response.body["status"] == "done"


def test_desktop_web_gui_shell_aggregates_chat_and_automation_state() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory_path = root / "dialogue_curated.jsonl"
        config_path = root / "config" / "automation.json"
        state_path = root / "artifacts" / "audit" / "automation_state.json"
        runs_path = root / "artifacts" / "audit" / "automation_runs.jsonl"
        _write_memory(memory_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            '{"workflows":[{"id":"write_hello","enabled":true,"trigger":{"type":"manual"},"steps":[{"name":"write_file","path":"artifacts/audit/automation/hello.txt","content":"hello gui shell\\n"}]}]}\n',
            encoding="utf-8",
        )
        shell = DesktopWebGUIShell(
            _build_gui_shell_args(
                memory_path.as_posix(),
                root,
                config_path,
                state_path,
                runs_path,
            )
        )
        chat_result = shell.submit_chat("how to read json in python")
        listed = shell.list_workflows()
        summary = shell.run_automation(["write_hello"])
        snapshot = shell.snapshot().to_dict()
        reset_snapshot = shell.reset(all_sections=True).to_dict()
        content = (root / "artifacts" / "audit" / "automation" / "hello.txt").read_text(encoding="utf-8")

    assert chat_result["source"] == "memory"
    assert listed[0]["id"] == "write_hello"
    assert summary["ok"] is True
    assert snapshot["chat_transcript"][-1]["role"] == "assistant"
    assert snapshot["automation_workflows"][0]["id"] == "write_hello"
    assert content == "hello gui shell\n"
    assert reset_snapshot["chat_transcript"] == []
    assert reset_snapshot["automation_workflows"] == []


def test_desktop_web_gui_shell_uses_injected_computer_session() -> None:
    calls: list[str] = []
    rows: list[dict] = []

    class _FakeComputerSession:
        def submit_goal(self, goal: str):
            calls.append(goal)
            row = {"goal": goal, "result": {"ok": True, "goal": goal, "status": "done"}}
            rows.append(row)
            return row["result"]

        def history(self):
            return list(rows)

        def reset(self):
            rows.clear()

    shell = DesktopWebGUIShell(SimpleNamespace(), computer_session=_FakeComputerSession())
    result = shell.run_computer_goal("open QQ")
    snapshot = shell.snapshot().to_dict()
    reset_snapshot = shell.reset(computer=True).to_dict()

    assert result["status"] == "done"
    assert calls == ["open QQ"]
    assert snapshot["computer_runs"][0]["goal"] == "open QQ"
    assert reset_snapshot["computer_runs"] == []


def test_cli_interface_entrypoint_delegates_to_app_runtime() -> None:
    calls: list[str] = []

    def _record() -> None:
        calls.append("runtime")

    with _patched_attr(app_runtime, "main", _record):
        run_cli_entrypoint()

    assert calls == ["runtime"]


def main() -> None:
    test_chat_api_adapter_responds_with_history()
    test_chat_gui_session_tracks_transcript_and_reset()
    test_automation_api_adapter_lists_and_runs_workflow()
    test_automation_gui_controller_tracks_last_state_and_reset()
    test_computer_api_adapter_runs_goal_via_runtime_builder()
    test_computer_gui_session_tracks_history_and_reset()
    test_http_api_shell_routes_chat_and_errors()
    test_http_api_shell_routes_automation_and_injected_computer_adapter()
    test_desktop_web_gui_shell_aggregates_chat_and_automation_state()
    test_desktop_web_gui_shell_uses_injected_computer_session()
    test_cli_interface_entrypoint_delegates_to_app_runtime()
    print("interfaces_entrypoints_ok")


if __name__ == "__main__":
    main()
