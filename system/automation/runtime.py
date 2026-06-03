from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from system.agent.executor import UnifiedExecutor
from system.agent.planner import UnifiedPlanner
from system.computer_use.action.validator import ActionValidator
from system.automation.executor import ExecutionResult
from system.core.planner import Plan, Task
from system.brain.world_model import WorldState


logger = logging.getLogger(__name__)


@dataclass
class TriggerDefinition:
    type: str = "manual"
    every_s: float = 0.0
    path: str = ""
    paths: List[str] = field(default_factory=list)
    cooldown_s: float = 0.0
    run_on_start: bool = False
    allow_repeat: bool = False

    @property
    def watched_paths(self) -> List[str]:
        out: List[str] = []
        if str(self.path or "").strip():
            out.append(str(self.path).strip())
        for item in list(self.paths or []):
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
        return out


@dataclass
class WorkflowDefinition:
    workflow_id: str
    description: str = ""
    enabled: bool = True
    triggers: List[TriggerDefinition] = field(default_factory=list)
    steps: List[Any] = field(default_factory=list)
    levels: List[List[Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def trigger_types(self) -> List[str]:
        if not self.triggers:
            return ["manual"]
        return [str(item.type or "manual").strip().lower() or "manual" for item in self.triggers]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("automation: failed to parse json %s", path)
        return default


def _write_json(path: Path, data: Any) -> None:
    _ensure_parent(path)
    tmp_path = path.with_name(f".{path.name}.tmp-{time.time_ns()}")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class AutomationRuntime:
    def __init__(
        self,
        *,
        project_root: str = ".",
        config_path: str = "config/automation_workflows.json",
        state_path: str = "artifacts/audit/automation_state.json",
        runs_path: str = "artifacts/audit/automation_runs.jsonl",
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.config_path = self._resolve_path(config_path)
        self.state_path = self._resolve_path(state_path)
        self.runs_path = self._resolve_path(runs_path)
        self.planner = UnifiedPlanner()
        self.validator = ActionValidator(project_root=self.project_root.as_posix())
        self.executor = UnifiedExecutor(project_root=self.project_root.as_posix())

    def _resolve_path(self, path_text: str) -> Path:
        path = Path(str(path_text or "").strip()).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def load_workflows(self) -> List[WorkflowDefinition]:
        raw = _read_json(self.config_path, default={"workflows": []})
        items = []
        if isinstance(raw, dict):
            items = list(raw.get("workflows") or [])
        elif isinstance(raw, list):
            items = list(raw)
        workflows: List[WorkflowDefinition] = []
        seen_ids: set[str] = set()
        for item in items:
            workflow = self._parse_workflow(item)
            low_id = workflow.workflow_id.lower()
            if low_id in seen_ids:
                raise ValueError(f"duplicate workflow id: {workflow.workflow_id}")
            seen_ids.add(low_id)
            workflows.append(workflow)
        return workflows

    def list_workflows(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for workflow in self.load_workflows():
            out.append(
                {
                    "id": workflow.workflow_id,
                    "enabled": bool(workflow.enabled),
                    "description": str(workflow.description or ""),
                    "triggers": workflow.trigger_types(),
                    "step_count": self._workflow_step_count(workflow),
                    "config_path": self.config_path.as_posix(),
                }
            )
        return out

    def scan_once(
        self,
        *,
        workflow_ids: Iterable[str] | None = None,
        force_run: bool = False,
    ) -> Dict[str, Any]:
        selected = {str(item or "").strip().lower() for item in list(workflow_ids or []) if str(item or "").strip()}
        workflows = self.load_workflows()
        workflow_map = {item.workflow_id.lower(): item for item in workflows}
        missing = sorted([item for item in selected if item not in workflow_map])
        if missing:
            raise ValueError("unknown automation workflow id(s): " + ", ".join(missing))

        state = self._load_state()
        now = time.time()
        triggered: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []

        for workflow in workflows:
            if selected and workflow.workflow_id.lower() not in selected:
                continue
            if not workflow.enabled:
                skipped.append({"id": workflow.workflow_id, "reason": "disabled"})
                continue
            due, reason = self._should_run_workflow(workflow, state, now=now, force_run=force_run)
            if not due:
                skipped.append({"id": workflow.workflow_id, "reason": reason})
                continue
            result = self._run_workflow(workflow, state, reason=reason, started_at=now)
            triggered.append(result)

        state["updated_at"] = time.time()
        _write_json(self.state_path, state)
        return {
            "ok": all(bool(item.get("ok", False)) for item in triggered) if triggered else True,
            "triggered": triggered,
            "skipped": skipped,
            "selected": sorted(selected),
            "config_path": self.config_path.as_posix(),
            "state_path": self.state_path.as_posix(),
            "runs_path": self.runs_path.as_posix(),
        }

    def run_loop(
        self,
        *,
        poll_s: float = 30.0,
        max_cycles: int = 0,
        workflow_ids: Iterable[str] | None = None,
    ) -> Dict[str, Any]:
        cycle = 0
        history: List[Dict[str, Any]] = []
        while True:
            cycle += 1
            summary = self.scan_once(workflow_ids=workflow_ids, force_run=False)
            summary["cycle"] = cycle
            history.append(summary)
            if max_cycles > 0 and cycle >= int(max_cycles):
                return {
                    "ok": all(bool(item.get("ok", False)) for item in history),
                    "cycles": cycle,
                    "history": history,
                }
            time.sleep(max(0.1, float(poll_s)))

    def _parse_workflow(self, item: Any) -> WorkflowDefinition:
        if not isinstance(item, dict):
            raise ValueError("workflow item must be an object")
        workflow_id = str(item.get("id") or item.get("name") or "").strip()
        if not workflow_id:
            raise ValueError("workflow id is required")
        description = str(item.get("description") or "").strip()
        enabled = bool(item.get("enabled", True))
        raw_triggers = item.get("triggers", item.get("trigger"))
        triggers = self._parse_triggers(raw_triggers)
        steps = list(item.get("steps") or [])
        levels = list(item.get("levels") or [])
        if not steps and not levels:
            raise ValueError(f"workflow {workflow_id} has no steps or levels")
        metadata = dict(item.get("metadata") or {})
        return WorkflowDefinition(
            workflow_id=workflow_id,
            description=description,
            enabled=enabled,
            triggers=triggers,
            steps=steps,
            levels=levels,
            metadata=metadata,
        )

    def _parse_triggers(self, raw: Any) -> List[TriggerDefinition]:
        if raw is None or raw == "":
            return [TriggerDefinition(type="manual")]
        items = raw if isinstance(raw, list) else [raw]
        triggers: List[TriggerDefinition] = []
        for item in items:
            if isinstance(item, str):
                trigger_type = str(item or "").strip().lower() or "manual"
                run_on_start = trigger_type == "interval"
                triggers.append(TriggerDefinition(type=trigger_type, run_on_start=run_on_start))
                continue
            if not isinstance(item, dict):
                raise ValueError("trigger must be a string or object")
            trigger_type = str(item.get("type") or "manual").strip().lower() or "manual"
            has_run_on_start = "run_on_start" in item
            run_on_start = bool(item.get("run_on_start", trigger_type == "interval")) if has_run_on_start else (trigger_type == "interval")
            trigger = TriggerDefinition(
                type=trigger_type,
                every_s=float(item.get("every_s") or item.get("interval_s") or 0.0),
                path=str(item.get("path") or "").strip(),
                paths=[str(x) for x in list(item.get("paths") or [])],
                cooldown_s=float(item.get("cooldown_s") or 0.0),
                run_on_start=run_on_start,
                allow_repeat=bool(item.get("allow_repeat", False)),
            )
            triggers.append(trigger)
        return triggers or [TriggerDefinition(type="manual")]

    def _workflow_step_count(self, workflow: WorkflowDefinition) -> int:
        if workflow.levels:
            return sum(len(level or []) for level in workflow.levels)
        return len(workflow.steps or [])

    def _load_state(self) -> Dict[str, Any]:
        state = _read_json(self.state_path, default={"version": 1, "workflows": {}, "updated_at": 0.0})
        if not isinstance(state, dict):
            state = {"version": 1, "workflows": {}, "updated_at": 0.0}
        state.setdefault("version", 1)
        state.setdefault("workflows", {})
        state.setdefault("updated_at", 0.0)
        return state

    def _workflow_state(self, state: Dict[str, Any], workflow_id: str) -> Dict[str, Any]:
        workflows = state.setdefault("workflows", {})
        if workflow_id not in workflows or not isinstance(workflows.get(workflow_id), dict):
            workflows[workflow_id] = {
                "last_run_at": 0.0,
                "last_status": "",
                "last_reason": "",
                "runs": 0,
                "failures": 0,
                "triggers": {},
            }
        return workflows[workflow_id]

    def _trigger_state(self, workflow_state: Dict[str, Any], index: int) -> Dict[str, Any]:
        triggers = workflow_state.setdefault("triggers", {})
        key = str(int(index))
        if key not in triggers or not isinstance(triggers.get(key), dict):
            triggers[key] = {}
        return triggers[key]

    def _should_run_workflow(
        self,
        workflow: WorkflowDefinition,
        state: Dict[str, Any],
        *,
        now: float,
        force_run: bool,
    ) -> Tuple[bool, str]:
        if force_run:
            return True, "manual_force"
        workflow_state = self._workflow_state(state, workflow.workflow_id)
        if not workflow.triggers:
            return False, "manual_only"
        for idx, trigger in enumerate(workflow.triggers):
            due, reason = self._evaluate_trigger(workflow_state, self._trigger_state(workflow_state, idx), trigger, now=now)
            if due:
                return True, reason
        return False, "no_trigger_due"

    def _evaluate_trigger(
        self,
        workflow_state: Dict[str, Any],
        trigger_state: Dict[str, Any],
        trigger: TriggerDefinition,
        *,
        now: float,
    ) -> Tuple[bool, str]:
        trigger_type = str(trigger.type or "manual").strip().lower() or "manual"
        cooldown_s = max(0.0, float(trigger.cooldown_s or 0.0))
        last_run_at = float(workflow_state.get("last_run_at") or 0.0)
        if cooldown_s > 0 and last_run_at > 0 and (now - last_run_at) < cooldown_s:
            return False, f"cooldown:{trigger_type}"

        if trigger_type == "manual":
            return False, "manual_only"

        if trigger_type == "interval":
            every_s = max(0.0, float(trigger.every_s or 0.0))
            if every_s <= 0.0:
                return False, "invalid_interval"
            if last_run_at <= 0:
                return bool(trigger.run_on_start), "interval_start"
            if (now - last_run_at) >= every_s:
                return True, "interval_due"
            return False, "interval_wait"

        if trigger_type == "path_exists":
            watched = trigger.watched_paths
            if not watched:
                return False, "missing_path"
            previous_exists = bool(trigger_state.get("last_exists", False))
            current_exists = any(self._resolve_path(path).exists() for path in watched)
            trigger_state["last_exists"] = bool(current_exists)
            if not current_exists:
                return False, "path_missing"
            if not previous_exists:
                return True, "path_exists"
            if trigger.allow_repeat:
                return True, "path_exists_repeat"
            return False, "path_exists_seen"

        if trigger_type == "path_changed":
            watched = trigger.watched_paths
            if not watched:
                return False, "missing_path"
            previous = dict(trigger_state.get("signatures") or {})
            current = {path: self._path_signature(path) for path in watched}
            trigger_state["signatures"] = current
            if not previous:
                return bool(trigger.run_on_start), "path_changed_start"
            for path in watched:
                if previous.get(path) != current.get(path):
                    return True, f"path_changed:{path}"
            return False, "path_unchanged"

        return False, f"unsupported_trigger:{trigger_type}"

    def _path_signature(self, path_text: str) -> Dict[str, Any]:
        path = self._resolve_path(path_text)
        if not path.exists():
            return {"exists": False}
        stat = path.stat()
        if path.is_dir():
            return {"exists": True, "kind": "dir", "mtime_ns": int(stat.st_mtime_ns)}
        return {
            "exists": True,
            "kind": "file",
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }

    def _run_workflow(
        self,
        workflow: WorkflowDefinition,
        state: Dict[str, Any],
        *,
        reason: str,
        started_at: float,
    ) -> Dict[str, Any]:
        plan = self._build_plan(workflow)
        world = WorldState(goal=workflow.workflow_id)
        workflow_state = self._workflow_state(state, workflow.workflow_id)
        workflow_state["last_run_at"] = started_at
        workflow_state["last_status"] = "running"
        workflow_state["last_reason"] = reason
        _write_json(self.state_path, state)
        invalid: List[Dict[str, Any]] = []
        for step in plan.steps:
            valid, note = self.validator.is_valid(step, world)
            if not valid:
                invalid.append({"step": step.detail, "name": step.name, "reason": note})
        if invalid:
            finished_at = time.time()
            workflow_state["last_run_at"] = finished_at
            workflow_state["last_status"] = "blocked"
            workflow_state["last_reason"] = reason
            workflow_state["runs"] = int(workflow_state.get("runs") or 0) + 1
            workflow_state["failures"] = int(workflow_state.get("failures") or 0) + 1
            row = {
                "ts": finished_at,
                "workflow_id": workflow.workflow_id,
                "ok": False,
                "status": "blocked",
                "reason": reason,
                "invalid": invalid,
            }
            _append_jsonl(self.runs_path, row)
            return row

        execution = self.executor.run(
            plan,
            task_id=workflow.workflow_id,
            metadata={"workflow_id": workflow.workflow_id, "reason": reason},
        )
        finished_at = time.time()
        ok = len(execution.failed) == 0
        workflow_state["last_run_at"] = finished_at
        workflow_state["last_status"] = "ok" if ok else "failed"
        workflow_state["last_reason"] = reason
        workflow_state["runs"] = int(workflow_state.get("runs") or 0) + 1
        if not ok:
            workflow_state["failures"] = int(workflow_state.get("failures") or 0) + 1
        row = {
            "ts": finished_at,
            "workflow_id": workflow.workflow_id,
            "ok": ok,
            "status": "ok" if ok else "failed",
            "reason": reason,
            "elapsed_s": round(max(0.0, finished_at - started_at), 3),
            "completed": list(execution.completed),
            "failed": list(execution.failed),
            "notes": list(execution.notes),
            "errors": list(execution.errors),
            "step_results": list(execution.step_results),
            "snapshot": dict(execution.snapshot or {}),
        }
        _append_jsonl(self.runs_path, row)
        return row

    def _build_plan(self, workflow: WorkflowDefinition) -> Plan:
        if workflow.levels:
            levels = [self._coerce_level(level, workflow.workflow_id) for level in workflow.levels]
        else:
            levels = [[self._coerce_task(item, workflow.workflow_id) for item in workflow.steps]]
        steps: List[Task] = []
        for level in levels:
            steps.extend(level)
        return Plan(goal=workflow.workflow_id, steps=steps, levels=levels)

    def _coerce_level(self, raw_level: Any, workflow_id: str) -> List[Task]:
        if not isinstance(raw_level, list):
            raise ValueError(f"workflow {workflow_id} level must be a list")
        return [self._coerce_task(item, workflow_id) for item in raw_level]

    def _coerce_task(self, item: Any, workflow_id: str) -> Task:
        if isinstance(item, str):
            task = self.planner.parse_task(item)
        elif isinstance(item, dict):
            if "command" in item or "text" in item:
                command = str(item.get("command") or item.get("text") or "").strip()
                if not command:
                    raise ValueError(f"workflow {workflow_id} command step is empty")
                task = self.planner.parse_task(command)
            else:
                name = str(item.get("name") or "").strip()
                if not name:
                    raise ValueError(f"workflow {workflow_id} step missing name")
                payload = item.get("payload")
                if payload is None:
                    payload = {
                        str(k): v
                        for k, v in dict(item).items()
                        if str(k) not in {"name", "detail", "priority", "status", "payload"}
                    }
                if payload is not None and not isinstance(payload, dict):
                    raise ValueError(f"workflow {workflow_id} payload must be an object")
                detail = str(item.get("detail") or name).strip() or name
                task = Task(
                    name=name,
                    detail=detail,
                    priority=max(1, int(item.get("priority") or 1)),
                    status=str(item.get("status") or "pending"),
                    payload=dict(payload or {}),
                )
        else:
            raise ValueError(f"workflow {workflow_id} step must be a string or object")
        if str(task.name or "").strip() == "step":
            raise ValueError(f"workflow {workflow_id} has unsupported step text: {task.detail}")
        return task


def run_automation_runtime(args: Any) -> Dict[str, Any]:
    runtime = build_automation_runtime(args)
    if bool(getattr(args, "automation_list", False)):
        items = runtime.list_workflows()
        for item in items:
            logger.info(
                "Automation workflow | id=%s enabled=%s triggers=%s steps=%s",
                item.get("id"),
                item.get("enabled"),
                ",".join(item.get("triggers") or []),
                item.get("step_count"),
            )
        return {"ok": True, "workflows": items}

    workflow_ids = list(getattr(args, "automation_run", []) or [])
    if workflow_ids:
        summary = runtime.scan_once(workflow_ids=workflow_ids, force_run=True)
        logger.info("Automation run summary: %s", summary)
        return summary

    if bool(getattr(args, "automation_loop", False)):
        summary = runtime.run_loop(
            poll_s=float(getattr(args, "automation_poll_s", 30.0) or 30.0),
            max_cycles=max(0, int(getattr(args, "automation_max_cycles", 0) or 0)),
        )
        logger.info("Automation loop summary: %s", summary)
        return summary

    summary = runtime.scan_once(force_run=False)
    logger.info("Automation scan summary: %s", summary)
    return summary


def build_automation_runtime(args: Any) -> AutomationRuntime:
    return AutomationRuntime(
        project_root=str(getattr(args, "automation_project_root", ".") or "."),
        config_path=str(getattr(args, "automation_config", "config/automation_workflows.json") or "config/automation_workflows.json"),
        state_path=str(getattr(args, "automation_state", "artifacts/audit/automation_state.json") or "artifacts/audit/automation_state.json"),
        runs_path=str(getattr(args, "automation_runs", "artifacts/audit/automation_runs.jsonl") or "artifacts/audit/automation_runs.jsonl"),
    )


__all__ = [
    "AutomationRuntime",
    "TriggerDefinition",
    "WorkflowDefinition",
    "build_automation_runtime",
    "run_automation_runtime",
]
