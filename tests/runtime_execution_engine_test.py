from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())

from system.agent.task import RuntimeTask
from system.runtime.execution_engine import RuntimeExecutionEngine
from system.runtime.state_manager import RuntimeStateManager


def _tempdir() -> tempfile.TemporaryDirectory[str]:
    base = ROOT / ".tmp"
    base.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=base.as_posix())


def test_execution_engine_persists_success_state() -> None:
    with _tempdir() as td:
        root = Path(td)
        state_manager = RuntimeStateManager(
            project_root=root.as_posix(),
            state_root="artifacts/runtime/tasks",
        )
        engine = RuntimeExecutionEngine(state_manager=state_manager)
        engine.register("automation", lambda args: {"ok": True, "workflow": args.workflow})
        task = RuntimeTask(
            task_id="automation-1",
            task_type="automation",
            intent="workflow_run",
            goal="run workflow",
            entrypoint="system.domain.automation",
        )

        result = engine.execute(task, SimpleNamespace(workflow="workspace_inventory"))
        assert result["ok"] is True

        payload = json.loads(
            (root / "artifacts" / "runtime" / "tasks" / "automation-1.json").read_text(encoding="utf-8")
        )
        assert payload["status"] == "succeeded"
        assert payload["result"]["workflow"] == "workspace_inventory"
        assert payload["last_error"] == ""


def test_execution_engine_persists_failure_state() -> None:
    with _tempdir() as td:
        root = Path(td)
        state_manager = RuntimeStateManager(
            project_root=root.as_posix(),
            state_root="artifacts/runtime/tasks",
        )
        engine = RuntimeExecutionEngine(state_manager=state_manager)

        def _boom(_args):
            raise RuntimeError("forced failure")

        engine.register("knowledge", _boom)
        task = RuntimeTask(
            task_id="knowledge-1",
            task_type="knowledge",
            intent="rag_ingest",
            goal="rag ingest",
            entrypoint="system.app_runtime._run_rag_ingest",
        )

        try:
            engine.execute(task, SimpleNamespace())
        except RuntimeError as exc:
            assert "forced failure" in str(exc)
        else:
            raise AssertionError("expected runtime failure")

        payload = json.loads(
            (root / "artifacts" / "runtime" / "tasks" / "knowledge-1.json").read_text(encoding="utf-8")
        )
        assert payload["status"] == "failed"
        assert "forced failure" in str(payload.get("last_error", ""))
        assert payload["result"] == {}
