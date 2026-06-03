from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from system.computer_use.llm import BaseComputerUseLLMClient
from system.computer_use.llm import OpenAIChatComputerUseClient, build_computer_use_llm_client
from system.computer_use.runtime import ComputerUseConfig, ComputerUseRuntime


class ScriptedComputerUseLLM(BaseComputerUseLLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def available(self) -> bool:
        return True

    def decide(
        self,
        *,
        prompt: str,
        image_paths,
        system_prompt: str = "",
    ) -> str:
        index = len(self.calls)
        self.calls.append(
            {
                "prompt": prompt,
                "image_paths": list(image_paths or []),
                "system_prompt": system_prompt,
            }
        )
        if index >= len(self.responses):
            raise RuntimeError("no scripted response left")
        return self.responses[index]


def _trace_rows(result: dict[str, object]) -> list[dict[str, object]]:
    trace_path = Path(str(result["trace_path"]))
    return [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summary_row(result: dict[str, object]) -> dict[str, object]:
    summary_path = Path(str(result["summary_path"]))
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _fake_observation(root: Path, **overrides) -> dict[str, object]:
    screen_path = root / "fake_screen.png"
    grid_path = root / "fake_grid.png"
    screen_path.write_bytes(b"fake-screen")
    grid_path.write_bytes(b"fake-grid")
    data: dict[str, object] = {
        "step_index": 1,
        "screen_path": screen_path.as_posix(),
        "grid_path": grid_path.as_posix(),
        "screen_sha1": "fake-sha1",
        "image_width": 1280,
        "image_height": 720,
        "active_window": {"title": "Browser", "process_name": "browser.exe", "pid": 1, "rect": [0, 0, 1280, 720]},
        "open_windows": [{"title": "Browser", "process_name": "browser.exe", "pid": 1}],
        "controls_window": "Browser",
        "controls": [{"title": "Search", "control_type": "Edit", "automation_id": "search-box", "rect": [50, 50, 280, 92]}],
        "controls_error": "",
        "ocr_text": "Example page",
        "ocr_engine": "dry-run",
        "ocr_items": [{"text": "Example page", "conf": 99.0, "left": 20, "top": 20, "width": 140, "height": 30, "block": "A1"}],
        "nonempty_blocks": [{"block": "A1", "text": "Example page", "item_count": 1}],
        "grid_cells": [{"id": "A1", "center_x": 40, "center_y": 35, "left": 0, "top": 0, "right": 80, "bottom": 70}],
        "surface_mode": "uia",
        "special_drawn_ui_likely": False,
        "targeting_hint": "prefer_control_actions",
    }
    data.update(overrides)
    return data


def test_computer_use_runtime_runs_scripted_dry_run_flow() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "focus_window", "title": "QQ", "reason": "focus target app"}),
                json.dumps(
                    {
                        "type": "type_control",
                        "window": "QQ",
                        "control": "Search",
                        "control_type": "Edit",
                        "text": "sheng ge",
                        "clear_first": True,
                        "reason": "search the contact",
                    }
                ),
                json.dumps(
                    {
                        "type": "click_control",
                        "window": "QQ",
                        "control": "Send",
                        "control_type": "Button",
                        "reason": "submit the message",
                    }
                ),
                json.dumps({"type": "done", "summary": "message sent in dry-run"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Open QQ and send a message",
                dry_run=True,
                target_window="QQ",
                max_steps=6,
                trace_root="artifacts/audit/computer_use_test",
            ),
            llm_client=llm,
        )
        result = runtime.run()
        assert result["ok"] is True
        assert result["status"] == "done"
        assert Path(str(result["latest_screen_path"])).exists()
        assert Path(str(result["latest_grid_path"])).exists()
        rows = _trace_rows(result)
        step_rows = [row for row in rows if row.get("kind") == "step"]
        assert [row["action"]["type"] for row in step_rows] == [
            "focus_window",
            "type_control",
            "click_control",
        ]
        assert all(bool(row["execution"]["verified"]) for row in step_rows)
        assert llm.calls
        first_images = list(llm.calls[0]["image_paths"])
        assert len(first_images) == 2
        assert all(Path(str(path)).exists() for path in first_images)
        summary = _summary_row(result)
        assert summary["status"] == "done"
        assert summary["verified_step_count"] == 3
        assert summary["failed_step_count"] == 0
        assert float(summary["verification_rate"]) == 1.0
        assert summary["action_counts"] == {
            "focus_window": 1,
            "type_control": 1,
            "click_control": 1,
        }
        assert summary["latest_controls_window"] == "QQ"
        assert summary["summary"] == "message sent in dry-run"
        assert len(summary["recent_steps"]) == 3


def test_computer_use_runtime_supports_grid_block_actions() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "click_block", "block": "A1", "reason": "hit the first block"}),
                json.dumps({"type": "done", "summary": "clicked the requested block"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Click the top-left block",
                dry_run=True,
                max_steps=4,
                grid_rows=4,
                grid_cols=4,
                trace_root="artifacts/audit/computer_use_grid_test",
            ),
            llm_client=llm,
        )
        result = runtime.run()
        assert result["ok"] is True
        rows = _trace_rows(result)
        step_rows = [row for row in rows if row.get("kind") == "step"]
        assert len(step_rows) == 1
        step = step_rows[0]
        assert step["action"]["type"] == "click_block"
        assert step["action"]["block"] == "A1"
        assert step["execution"]["block"] == "A1"
        assert int(step["execution"]["x"]) == 160
        assert int(step["execution"]["y"]) == 100
        assert '"block": "A1"' in str(llm.calls[0]["prompt"])
        summary = _summary_row(result)
        assert summary["action_counts"] == {"click_block": 1}
        assert summary["last_action"]["type"] == "click_block"
        assert summary["latest_nonempty_blocks"]


def test_computer_use_runtime_records_failure_summary() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "click_block", "block": "Z99", "reason": "bad block"}),
                json.dumps({"type": "fail", "reason": "cannot find a safe actionable target"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Click an invalid block then fail safely",
                dry_run=True,
                max_steps=4,
                trace_root="artifacts/audit/computer_use_failure_test",
            ),
            llm_client=llm,
        )
        result = runtime.run()
        assert result["ok"] is False
        assert result["status"] == "failed"
        summary = _summary_row(result)
        assert summary["failed_step_count"] == 1
        assert summary["action_counts"] == {"click_block": 1}
        assert summary["last_action"]["type"] == "click_block"
        assert "cannot find a safe actionable target" in str(summary["reason"])
        assert float(summary["verification_rate"]) == 0.0


def test_computer_use_runtime_writes_resumable_checkpoint_for_incomplete_run() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "wait", "seconds": 0.1, "reason": "pause before next step"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Pause and then continue later",
                dry_run=True,
                max_steps=1,
                trace_root="artifacts/audit/computer_use_resume_seed",
            ),
            llm_client=llm,
        )
        result = runtime.run()
        assert result["ok"] is False
        assert result["status"] == "incomplete"
        assert result["handoff_required"] is True
        assert result["handoff_reason_code"] == "max_steps_reached"
        assert result["resume_available"] is True
        assert result["resume_strategy"] == "resume_from_current_state"
        assert result["suggested_resume_command"] == "resume last computer-use task"
        recovery_path = Path(str(result["recovery_path"]))
        assert recovery_path.exists()
        payload = json.loads(recovery_path.read_text(encoding="utf-8"))
        assert payload["resume_available"] is True
        assert payload["resume_strategy"] == "resume_from_current_state"
        assert payload["repair_actions"] == []
        assert payload["query"] == "Pause and then continue later"


def test_resume_last_computer_use_task_continues_from_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        seed_llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "wait", "seconds": 0.1, "reason": "pause before next step"}),
            ]
        )
        seed_runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Pause and then continue later",
                dry_run=True,
                max_steps=1,
                trace_root="artifacts/audit/computer_use_resume_seed",
            ),
            llm_client=seed_llm,
        )
        seed_result = seed_runtime.run()
        assert seed_result["resume_available"] is True

        resume_llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "done", "summary": "resumed run completed"}),
            ]
        )
        resume_runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="resume last computer-use task",
                dry_run=True,
                max_steps=2,
                trace_root="artifacts/audit/computer_use_resume_continue",
            ),
            llm_client=resume_llm,
        )
        resumed = resume_runtime.run()
        assert resumed["ok"] is True
        assert resumed["status"] == "done"
        assert resumed["resume_mode"] is True
        assert resumed["resume_source_path"]
        assert resumed["summary"] == "resumed run completed"


def test_failed_desktop_action_builds_repair_replay_and_resume_succeeds() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        failing_llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "focus_window", "title": "QQ", "reason": "focus target app"}),
                json.dumps({"type": "click_block", "block": "A1", "reason": "open the first target"}),
                json.dumps({"type": "fail", "reason": "need to resume from the saved checkpoint"}),
            ]
        )
        failing_runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Focus QQ and click a target",
                dry_run=True,
                max_steps=3,
                trace_root="artifacts/audit/computer_use_repair_seed",
            ),
            llm_client=failing_llm,
        )
        original_execute = failing_runtime._execute_action

        def _failing_execute(action, observation):
            if action.type == "click_block":
                return {"verified": False, "ok": False, "error": "simulated click failure"}
            return original_execute(action, observation)

        failing_runtime._execute_action = _failing_execute  # type: ignore[method-assign]
        failed = failing_runtime.run()
        assert failed["ok"] is False
        assert failed["status"] == "failed"
        assert failed["resume_available"] is True
        assert failed["resume_strategy"] == "resume_from_last_verified_step"
        recovery_path = Path(str(failed["recovery_path"]))
        payload = json.loads(recovery_path.read_text(encoding="utf-8"))
        assert payload["resume_available"] is True
        assert payload["resume_strategy"] == "resume_from_last_verified_step"
        assert [row["type"] for row in payload["repair_actions"]] == ["focus_window", "click_block"]

        resume_llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "done", "summary": "recovery replay completed"}),
            ]
        )
        resume_runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="resume last computer-use task",
                dry_run=True,
                max_steps=3,
                trace_root="artifacts/audit/computer_use_repair_resume",
            ),
            llm_client=resume_llm,
        )
        resumed = resume_runtime.run()
        assert resumed["ok"] is True
        assert resumed["resume_mode"] is True
        assert resumed["resume_replayed_action_count"] == 2
        rows = _trace_rows(resumed)
        step_rows = [row for row in rows if row.get("kind") == "step"]
        assert [row["action"]["type"] for row in step_rows[:2]] == ["focus_window", "click_block"]
        assert all(bool(row.get("replayed", False)) for row in step_rows[:2])


def test_computer_use_runtime_handoffs_on_captcha_detection() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM([])
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Open the page and continue",
                dry_run=True,
                max_steps=2,
                trace_root="artifacts/audit/computer_use_captcha_test",
            ),
            llm_client=llm,
        )
        runtime._observe = lambda **kwargs: _fake_observation(  # type: ignore[method-assign]
            root,
            ocr_text="Security Check - Please complete CAPTCHA",
            ocr_items=[{"text": "Please complete CAPTCHA", "conf": 99.0, "left": 20, "top": 20, "width": 260, "height": 30, "block": "A1"}],
            nonempty_blocks=[{"block": "A1", "text": "Please complete CAPTCHA", "item_count": 1}],
        )
        result = runtime.run()
        assert result["ok"] is False
        assert result["status"] == "handoff_required"
        assert result["handoff_required"] is True
        assert result["handoff_reason_code"] == "captcha_detected"
        assert result["resume_available"] is True
        assert not llm.calls


def test_computer_use_runtime_handoffs_on_anti_automation_detection() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM([])
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Open the page and continue",
                dry_run=True,
                max_steps=2,
                trace_root="artifacts/audit/computer_use_anti_automation_test",
            ),
            llm_client=llm,
        )
        runtime._observe = lambda **kwargs: _fake_observation(  # type: ignore[method-assign]
            root,
            ocr_text="Access denied due to unusual traffic",
            ocr_items=[{"text": "Access denied", "conf": 99.0, "left": 20, "top": 20, "width": 160, "height": 30, "block": "A1"}],
            nonempty_blocks=[{"block": "A1", "text": "Access denied", "item_count": 1}],
        )
        result = runtime.run()
        assert result["ok"] is False
        assert result["status"] == "handoff_required"
        assert result["handoff_required"] is True
        assert result["handoff_reason_code"] == "anti_automation_detected"
        assert result["resume_available"] is True
        assert not llm.calls


def test_computer_use_runtime_blocks_high_risk_action_by_default() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "launch", "target": "powershell.exe", "reason": "open a shell"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Open a shell and continue",
                dry_run=True,
                max_steps=2,
                trace_root="artifacts/audit/computer_use_high_risk_test",
            ),
            llm_client=llm,
        )
        result = runtime.run()
        assert result["ok"] is False
        assert result["status"] == "handoff_required"
        assert result["handoff_required"] is True
        assert result["handoff_reason_code"] == "high_risk_launch"


def test_computer_use_runtime_rejects_blind_type_text_after_win_r_without_run_dialog() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "hotkey", "keys": "win+r", "reason": "open Run dialog"}),
                json.dumps({"type": "type_text", "text": "notepad", "reason": "type launch command"}),
                json.dumps({"type": "fail", "reason": "Run dialog never appeared"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Launch the requested desktop app safely",
                dry_run=True,
                max_steps=3,
                allow_high_risk_actions=True,
                trace_root="artifacts/audit/computer_use_missing_run_dialog_test",
            ),
            llm_client=llm,
        )
        observations = iter(
            [
                _fake_observation(root),
                _fake_observation(root, step_index=2),
                _fake_observation(root, step_index=3),
            ]
        )
        runtime._observe = lambda **kwargs: next(observations)  # type: ignore[method-assign]
        result = runtime.run()
        assert result["ok"] is False
        assert result["status"] == "failed"
        rows = _trace_rows(result)
        step_rows = [row for row in rows if row.get("kind") == "step"]
        assert [row["action"]["type"] for row in step_rows] == ["hotkey", "type_text"]
        assert step_rows[-1]["execution"]["plugin"] == "runtime_guard"
        assert step_rows[-1]["execution"]["reason_code"] == "missing_run_dialog_after_hotkey"


def test_computer_use_runtime_prefers_launch_or_focus_over_win_r_for_known_app_targets() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "hotkey", "keys": "win+r", "reason": "open Run dialog"}),
                json.dumps({"type": "launch", "target": "notepad", "reason": "launch app directly"}),
                json.dumps({"type": "done", "summary": "notepad launch completed"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Open Notepad safely",
                dry_run=True,
                max_steps=4,
                allow_high_risk_actions=True,
                trace_root="artifacts/audit/computer_use_prefer_launch_test",
            ),
            llm_client=llm,
        )
        observations = iter(
            [
                _fake_observation(root),
                _fake_observation(root, step_index=2),
                _fake_observation(
                    root,
                    step_index=3,
                    active_window={"title": "Untitled - Notepad", "process_name": "notepad.exe", "pid": 7, "rect": [20, 20, 1200, 700]},
                    open_windows=[{"title": "Untitled - Notepad", "process_name": "notepad.exe", "pid": 7}],
                    controls_window="Untitled - Notepad",
                    controls=[{"title": "Text Editor", "control_type": "Document", "automation_id": "15", "rect": [20, 60, 1180, 680]}],
                    ocr_text="Untitled - Notepad",
                    ocr_items=[{"text": "Untitled - Notepad", "conf": 99.0, "left": 20, "top": 20, "width": 220, "height": 30, "block": "A1"}],
                    nonempty_blocks=[{"block": "A1", "text": "Untitled - Notepad", "item_count": 1}],
                ),
            ]
        )
        runtime._observe = lambda **kwargs: next(observations)  # type: ignore[method-assign]
        result = runtime.run()
        assert result["ok"] is True
        assert result["status"] == "done"
        rows = _trace_rows(result)
        step_rows = [row for row in rows if row.get("kind") == "step"]
        assert [row["action"]["type"] for row in step_rows] == ["hotkey", "launch"]
        assert step_rows[0]["execution"]["plugin"] == "runtime_guard"
        assert step_rows[0]["execution"]["reason_code"] == "prefer_launch_or_focus_window_over_win_r"
        assert step_rows[1]["execution"]["plugin"] == "desktop_launch"


def test_computer_use_runtime_rejects_done_when_target_window_never_appears() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "hotkey", "keys": "win+r", "reason": "open Run dialog"}),
                json.dumps({"type": "type_text", "text": "notepad", "reason": "type launch command"}),
                json.dumps({"type": "hotkey", "keys": "enter", "reason": "submit launch command"}),
                json.dumps({"type": "wait", "seconds": 0.1, "reason": "allow the app to launch"}),
                json.dumps({"type": "done", "summary": "notepad launch completed"}),
                json.dumps({"type": "fail", "reason": "target window never appeared"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Launch the requested desktop app and wait until it is visible",
                dry_run=True,
                max_steps=6,
                allow_high_risk_actions=True,
                trace_root="artifacts/audit/computer_use_done_guard_test",
            ),
            llm_client=llm,
        )
        observations = iter(
            [
                _fake_observation(root),
                _fake_observation(
                    root,
                    step_index=2,
                    active_window={"title": "Run", "process_name": "explorer.exe", "pid": 11, "rect": [100, 100, 640, 420]},
                    open_windows=[{"title": "Run", "process_name": "explorer.exe", "pid": 11}],
                    controls_window="Run",
                    controls=[{"title": "Open", "control_type": "Edit", "automation_id": "1148", "rect": [140, 200, 520, 244]}],
                    ocr_text="Run Open: Browse",
                    ocr_items=[{"text": "Run Open: Browse", "conf": 99.0, "left": 20, "top": 20, "width": 220, "height": 30, "block": "A1"}],
                    nonempty_blocks=[{"block": "A1", "text": "Run Open: Browse", "item_count": 1}],
                ),
                _fake_observation(
                    root,
                    step_index=3,
                    active_window={"title": "Run", "process_name": "explorer.exe", "pid": 11, "rect": [100, 100, 640, 420]},
                    open_windows=[{"title": "Run", "process_name": "explorer.exe", "pid": 11}],
                    controls_window="Run",
                    controls=[{"title": "Open", "control_type": "Edit", "automation_id": "1148", "rect": [140, 200, 520, 244]}],
                    ocr_text="Run Open: Browse",
                    ocr_items=[{"text": "Run Open: Browse", "conf": 99.0, "left": 20, "top": 20, "width": 220, "height": 30, "block": "A1"}],
                    nonempty_blocks=[{"block": "A1", "text": "Run Open: Browse", "item_count": 1}],
                ),
                _fake_observation(root, step_index=4),
                _fake_observation(root, step_index=5),
                _fake_observation(root, step_index=6),
            ]
        )
        runtime._observe = lambda **kwargs: next(observations)  # type: ignore[method-assign]
        result = runtime.run()
        assert result["ok"] is False
        assert result["status"] == "failed"
        rows = _trace_rows(result)
        step_rows = [row for row in rows if row.get("kind") == "step"]
        assert [row["action"]["type"] for row in step_rows[:4]] == ["hotkey", "type_text", "hotkey", "wait"]
        assert step_rows[4]["action"]["type"] == "done"
        assert step_rows[4]["execution"]["plugin"] == "runtime_guard"
        assert step_rows[4]["execution"]["reason_code"] == "done_without_target_observed"


def test_computer_use_runtime_allows_high_risk_action_when_enabled() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "launch", "target": "powershell.exe", "reason": "open a shell for diagnostics"}),
                json.dumps({"type": "done", "summary": "launch path completed"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Open a shell and continue",
                dry_run=True,
                max_steps=3,
                allow_high_risk_actions=True,
                trace_root="artifacts/audit/computer_use_high_risk_allowed_test",
            ),
            llm_client=llm,
        )
        observations = iter(
            [
                _fake_observation(root),
                _fake_observation(
                    root,
                    step_index=2,
                    active_window={"title": "Windows PowerShell", "process_name": "powershell.exe", "pid": 2, "rect": [0, 0, 1280, 720]},
                    open_windows=[{"title": "Windows PowerShell", "process_name": "powershell.exe", "pid": 2}],
                    controls_window="Windows PowerShell",
                    controls=[{"title": "Console", "control_type": "Document", "automation_id": "console", "rect": [0, 0, 1280, 720]}],
                    ocr_text="Windows PowerShell",
                    ocr_items=[{"text": "Windows PowerShell", "conf": 99.0, "left": 20, "top": 20, "width": 220, "height": 30, "block": "A1"}],
                    nonempty_blocks=[{"block": "A1", "text": "Windows PowerShell", "item_count": 1}],
                ),
            ]
        )
        runtime._observe = lambda **kwargs: next(observations)  # type: ignore[method-assign]
        original_execute = runtime._execute_action

        def _launch_execute(action, observation):
            if action.type == "launch":
                return {"verified": True, "ok": True, "note": "simulated launch"}
            return original_execute(action, observation)

        runtime._execute_action = _launch_execute  # type: ignore[method-assign]
        result = runtime.run()
        assert result["ok"] is True
        assert result["status"] == "done"
        assert result["handoff_required"] is False
        rows = _trace_rows(result)
        step_rows = [row for row in rows if row.get("kind") == "step"]
        assert [row["action"]["type"] for row in step_rows] == ["launch"]
        summary = _summary_row(result)
        assert summary["action_counts"] == {"launch": 1}


def test_computer_use_runtime_accepts_model_requested_handoff() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "handoff", "reason": "please solve the challenge manually"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Open page and continue",
                dry_run=True,
                max_steps=2,
                trace_root="artifacts/audit/computer_use_model_handoff_test",
            ),
            llm_client=llm,
        )
        result = runtime.run()
        assert result["ok"] is False
        assert result["status"] == "handoff_required"
        assert result["handoff_required"] is True
        assert result["handoff_reason_code"] == "model_requested_handoff"
        assert "solve the challenge manually" in str(result["reason"])
        rows = _trace_rows(result)
        handoff_rows = [row for row in rows if row.get("kind") == "handoff_required"]
        assert handoff_rows
        assert handoff_rows[-1]["action"]["type"] == "handoff"


def test_computer_use_runtime_marks_special_drawn_visual_mode() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        llm = ScriptedComputerUseLLM(
            [
                json.dumps({"type": "done", "summary": "visual mode task inspected"}),
            ]
        )
        runtime = ComputerUseRuntime(
            ComputerUseConfig(
                project_root=root.as_posix(),
                goal="Inspect a custom rendered app",
                dry_run=True,
                max_steps=2,
                trace_root="artifacts/audit/computer_use_visual_mode_test",
            ),
            llm_client=llm,
        )
        runtime._observe = lambda **kwargs: _fake_observation(  # type: ignore[method-assign]
            root,
            controls=[],
            controls_window="Custom Canvas App",
            ocr_text="Canvas Button Play",
            ocr_items=[{"text": "Canvas Button Play", "conf": 95.0, "left": 100, "top": 100, "width": 180, "height": 30, "block": "B2"}],
            nonempty_blocks=[{"block": "B2", "text": "Canvas Button Play", "item_count": 1}],
            surface_mode="ocr_only",
            special_drawn_ui_likely=True,
            targeting_hint="prefer_text_or_block_actions",
        )
        result = runtime.run()
        assert result["ok"] is True
        summary = _summary_row(result)
        assert summary["latest_surface_mode"] == "ocr_only"
        assert summary["latest_special_drawn_ui_likely"] is True
        assert summary["latest_targeting_hint"] == "prefer_text_or_block_actions"


def test_computer_use_builder_uses_chat_client_for_deepseek() -> None:
    client = build_computer_use_llm_client(
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key="test-key",
    )
    assert isinstance(client, OpenAIChatComputerUseClient)


def main() -> None:
    test_computer_use_runtime_runs_scripted_dry_run_flow()
    test_computer_use_runtime_supports_grid_block_actions()
    test_computer_use_runtime_records_failure_summary()
    test_computer_use_runtime_writes_resumable_checkpoint_for_incomplete_run()
    test_resume_last_computer_use_task_continues_from_checkpoint()
    test_failed_desktop_action_builds_repair_replay_and_resume_succeeds()
    test_computer_use_runtime_handoffs_on_captcha_detection()
    test_computer_use_runtime_handoffs_on_anti_automation_detection()
    test_computer_use_runtime_blocks_high_risk_action_by_default()
    test_computer_use_runtime_rejects_blind_type_text_after_win_r_without_run_dialog()
    test_computer_use_runtime_prefers_launch_or_focus_over_win_r_for_known_app_targets()
    test_computer_use_runtime_rejects_done_when_target_window_never_appears()
    test_computer_use_runtime_allows_high_risk_action_when_enabled()
    test_computer_use_runtime_accepts_model_requested_handoff()
    test_computer_use_runtime_marks_special_drawn_visual_mode()
    test_computer_use_builder_uses_chat_client_for_deepseek()
    print("computer_use_runtime_ok")


if __name__ == "__main__":
    main()
