from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from system.automation.runtime import AutomationRuntime


def _tempdir() -> tempfile.TemporaryDirectory[str]:
    base = ROOT / ".tmp"
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base.as_posix())


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_manual_workflow_run_persists_outputs() -> None:
    with _tempdir() as td:
        root = Path(td)
        config_path = root / "config" / "automation.json"
        state_path = root / "artifacts" / "audit" / "automation_state.json"
        runs_path = root / "artifacts" / "audit" / "automation_runs.jsonl"
        out_path = root / "artifacts" / "audit" / "automation" / "hello.txt"
        _write_json(
            config_path,
            {
                "workflows": [
                    {
                        "id": "write_hello",
                        "enabled": True,
                        "trigger": {"type": "manual"},
                        "steps": [
                            {
                                "name": "write_file",
                                "path": "artifacts/audit/automation/hello.txt",
                                "content": "hello automation\n",
                            }
                        ],
                    }
                ]
            },
        )
        runtime = AutomationRuntime(
            project_root=root.as_posix(),
            config_path=config_path.as_posix(),
            state_path=state_path.as_posix(),
            runs_path=runs_path.as_posix(),
        )
        summary = runtime.scan_once(workflow_ids=["write_hello"], force_run=True)
        assert summary["ok"] is True
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == "hello automation\n"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        wf_state = state["workflows"]["write_hello"]
        assert wf_state["last_status"] == "ok"
        assert int(wf_state["runs"]) == 1
        rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["workflow_id"] == "write_hello"
        assert rows[0]["ok"] is True


def test_interval_workflow_runs_once_then_waits() -> None:
    with _tempdir() as td:
        root = Path(td)
        config_path = root / "config" / "automation.json"
        state_path = root / "artifacts" / "audit" / "automation_state.json"
        runs_path = root / "artifacts" / "audit" / "automation_runs.jsonl"
        _write_json(
            config_path,
            {
                "workflows": [
                    {
                        "id": "interval_job",
                        "enabled": True,
                        "trigger": {"type": "interval", "every_s": 3600},
                        "steps": ["touch artifacts/audit/automation/interval.ok"],
                    }
                ]
            },
        )
        runtime = AutomationRuntime(
            project_root=root.as_posix(),
            config_path=config_path.as_posix(),
            state_path=state_path.as_posix(),
            runs_path=runs_path.as_posix(),
        )
        first = runtime.scan_once()
        assert len(first["triggered"]) == 1
        second = runtime.scan_once()
        assert len(second["triggered"]) == 0
        assert any(item["id"] == "interval_job" for item in second["skipped"])


def test_path_changed_workflow_detects_modified_file() -> None:
    with _tempdir() as td:
        root = Path(td)
        watched = root / "README.md"
        watched.write_text("hello\n", encoding="utf-8")
        config_path = root / "config" / "automation.json"
        state_path = root / "artifacts" / "audit" / "automation_state.json"
        runs_path = root / "artifacts" / "audit" / "automation_runs.jsonl"
        _write_json(
            config_path,
            {
                "workflows": [
                    {
                        "id": "watch_readme",
                        "enabled": True,
                        "trigger": {"type": "path_changed", "paths": ["README.md"]},
                        "steps": [
                            {
                                "name": "write_file",
                                "path": "artifacts/audit/automation/readme_watch.txt",
                                "content": "changed\n",
                            }
                        ],
                    }
                ]
            },
        )
        runtime = AutomationRuntime(
            project_root=root.as_posix(),
            config_path=config_path.as_posix(),
            state_path=state_path.as_posix(),
            runs_path=runs_path.as_posix(),
        )
        first = runtime.scan_once()
        assert len(first["triggered"]) == 0
        watched.write_text("hello again\n", encoding="utf-8")
        second = runtime.scan_once()
        assert len(second["triggered"]) == 1
        assert second["triggered"][0]["workflow_id"] == "watch_readme"


def test_run_script_workflow_generates_report_file() -> None:
    with _tempdir() as td:
        root = Path(td)
        script_path = root / "scripts" / "report.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            "from pathlib import Path\n"
            "out = Path('artifacts/audit/automation/report.txt')\n"
            "out.parent.mkdir(parents=True, exist_ok=True)\n"
            "out.write_text('report ok\\n', encoding='utf-8')\n"
            "print('report generated')\n",
            encoding="utf-8",
        )
        config_path = root / "config" / "automation.json"
        state_path = root / "artifacts" / "audit" / "automation_state.json"
        runs_path = root / "artifacts" / "audit" / "automation_runs.jsonl"
        _write_json(
            config_path,
            {
                "workflows": [
                    {
                        "id": "script_report",
                        "enabled": True,
                        "trigger": {"type": "manual"},
                        "steps": [
                            {
                                "name": "run_script",
                                "path": "scripts/report.py",
                            }
                        ],
                    }
                ]
            },
        )
        runtime = AutomationRuntime(
            project_root=root.as_posix(),
            config_path=config_path.as_posix(),
            state_path=state_path.as_posix(),
            runs_path=runs_path.as_posix(),
        )
        summary = runtime.scan_once(workflow_ids=["script_report"], force_run=True)
        assert summary["ok"] is True
        report_path = root / "artifacts" / "audit" / "automation" / "report.txt"
        assert report_path.exists()
        assert report_path.read_text(encoding="utf-8") == "report ok\n"
        rows = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[0]["step_results"][0]["result"]["output"] == "report generated"


def main() -> None:
    test_manual_workflow_run_persists_outputs()
    test_interval_workflow_runs_once_then_waits()
    test_path_changed_workflow_detects_modified_file()
    test_run_script_workflow_generates_report_file()
    print("automation_runtime_ok")


if __name__ == "__main__":
    main()
