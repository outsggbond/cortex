from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.automation.executor import TaskExecutor
from system.core.planner import Plan, Task, TaskPlanner


def test_plugin_retries_after_unverified_result() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        trace_path = root / "trace.jsonl"
        old_trace = os.environ.get("COMPUTER_CONTROL_TRACE_PATH")
        try:
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            executor = TaskExecutor(project_root=root.as_posix())
            plan = Plan(
                goal="retry_unverified",
                steps=[
                    Task(
                        name="plugin:desktop_click",
                        detail="click 10 20",
                        payload={
                            "x": 10,
                            "y": 20,
                            "dry_run": True,
                            "max_retries": 1,
                            "retry_on_unverified": True,
                            "verify_text": "Dry Run Text",
                            "_simulate_unverified_attempts": 1,
                        },
                    )
                ],
                levels=[],
            )
            result = executor.run(plan)
            assert not result.failed
            payload = result.step_results[0]["result"]
            assert payload["verified"] is True
            assert int(payload["attempts"]) == 2
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert len(rows) == 2
            assert rows[0]["result"]["verified"] is False
            assert rows[1]["result"]["verified"] is True
        finally:
            if old_trace is None:
                os.environ.pop("COMPUTER_CONTROL_TRACE_PATH", None)
            else:
                os.environ["COMPUTER_CONTROL_TRACE_PATH"] = old_trace


def test_plugin_returns_unverified_after_retry_budget_exhausted() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        trace_path = root / "trace.jsonl"
        old_trace = os.environ.get("COMPUTER_CONTROL_TRACE_PATH")
        try:
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            executor = TaskExecutor(project_root=root.as_posix())
            plan = Plan(
                goal="retry_exhausted",
                steps=[
                    Task(
                        name="plugin:desktop_click",
                        detail="click 10 20",
                        payload={
                            "x": 10,
                            "y": 20,
                            "dry_run": True,
                            "max_retries": 1,
                            "retry_on_unverified": True,
                            "verify_text": "Missing Text",
                        },
                    )
                ],
                levels=[],
            )
            result = executor.run(plan)
            assert not result.failed
            payload = result.step_results[0]["result"]
            assert payload["verified"] is False
            assert int(payload["attempts"]) == 2
            assert payload["verification"] == "ocr_contains_text"
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert len(rows) == 2
        finally:
            if old_trace is None:
                os.environ.pop("COMPUTER_CONTROL_TRACE_PATH", None)
            else:
                os.environ["COMPUTER_CONTROL_TRACE_PATH"] = old_trace


def test_plugin_retries_after_transient_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        executor = TaskExecutor(project_root=root.as_posix())
        plan = Plan(
            goal="retry_error",
            steps=[
                Task(
                    name="plugin:desktop_click",
                    detail="click 10 20",
                    payload={
                        "x": 10,
                        "y": 20,
                        "dry_run": True,
                        "max_retries": 1,
                        "_simulate_error_attempts": 1,
                    },
                )
            ],
            levels=[],
        )
        result = executor.run(plan)
        assert not result.failed
        payload = result.step_results[0]["result"]
        assert payload["verified"] is True
        assert int(payload["attempts"]) == 2


def test_planner_parses_verification_and_retry_hints() -> None:
    planner = TaskPlanner()
    task = planner._parse_task('click 10 20 verify text "Dry Run Text" retries 2')
    assert task.name == "plugin:desktop_click"
    assert int(task.payload["x"]) == 10
    assert int(task.payload["y"]) == 20
    assert task.payload["verify_text"] == "Dry Run Text"
    assert int(task.payload["max_retries"]) == 2
    task2 = planner._parse_task('drag 10 20 -> 30 40 verify change')
    assert task2.name == "plugin:desktop_drag"
    assert task2.payload["verify_change"] is True


def test_planner_strips_wrapping_quotes_from_type_text() -> None:
    planner = TaskPlanner()
    task = planner._parse_task('type "晚安"')
    assert task.name == "plugin:desktop_type_text"
    assert task.payload["text"] == "晚安"


def main() -> None:
    test_plugin_retries_after_unverified_result()
    test_plugin_returns_unverified_after_retry_budget_exhausted()
    test_plugin_retries_after_transient_error()
    test_planner_parses_verification_and_retry_hints()
    test_planner_strips_wrapping_quotes_from_type_text()
    print("computer_plugin_retry_ok")


if __name__ == "__main__":
    main()
