# -*- coding: utf-8 -*-
"""Conservative workspace agent loop for chat runtime v2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence, Tuple

from system.agent.executor import UnifiedExecutor
from system.agent.planner import UnifiedPlanner
from system.automation.executor import ExecutionResult
from system.brain.world_model import WorldModel
from system.computer_use.action.generator import ActionGenerator
from system.computer_use.action.stats import ActionStats
from system.computer_use.action.validator import ActionValidator
from system.core.planner import Plan, Task
from system.evaluation.self_heal import apply_self_heal
from system.l_utils.path_utils import (
    extract_paths,
    extract_search_pattern,
    extract_url,
    looks_like_filename,
    looks_like_path,
    normalize_path,
    pick_path,
    resolve_path,
)
from system.l_utils.text_utils import contains_cjk, snippet
from system.strategy.search_planner import SearchPlanner

from .computer_task_feedback import ComputerTaskFeedbackRecorder
from .computer_task_learning import LearnedComputerTaskStore
from .computer_task_recovery import (
    DEFAULT_COMPUTER_RECOVERY_PATH,
    build_computer_recovery_payload,
    build_resume_plan,
    load_computer_recovery_payload,
    save_computer_recovery_payload,
)
from .computer_task_templates import build_template_plan, parse_template_command
from .agent_classifier import RequestClassifier
from .intents import ChatIntent
from .llm import BaseLLMClient
from .types import ChatRequest
from .workspace_task_learning import LearnedWorkspaceTaskStore

from . import _agent_computer
from . import _agent_search
from . import _agent_summary


READ_ONLY_TASKS = {"read_file", "list_dir", "check_exists", "search_text", "search_files"}
WRITE_TASKS = {"write_file", "append_file", "touch_file", "mkdir", "copy_file"}
RISKY_WRITE_TASKS = {"move_file", "delete_file", "rollback", "chmod"}
EXEC_TASKS = {"run_script", "run_tests", "generate_code"}
BROWSER_PLUGIN_TASKS = {
    "plugin:browser_open_url",
    "plugin:browser_dom_open",
    "plugin:browser_dom_click",
    "plugin:browser_dom_type",
    "plugin:browser_dom_extract_text",
    "plugin:browser_dom_wait_text",
    "plugin:browser_dom_screenshot",
    "plugin:browser_dom_close",
}
DESKTOP_PLUGIN_TASKS = {
    "plugin:desktop_launch",
    "plugin:desktop_list_windows",
    "plugin:desktop_list_controls",
    "plugin:desktop_focus_window",
    "plugin:desktop_click_control",
    "plugin:desktop_type_control",
    "plugin:desktop_type_text",
    "plugin:desktop_hotkey",
    "plugin:desktop_screenshot",
    "plugin:desktop_click",
    "plugin:desktop_drag",
    "plugin:desktop_ocr",
    "plugin:desktop_click_text",
}
PATH_TOKEN_RE = re.compile(r"(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+(?:\.[A-Za-z0-9_]+)?")
URL_RE = re.compile(r"https?://[^\s`\"'>)]+", flags=re.IGNORECASE)

ZH_PROJECT = "\u9879\u76ee"
ZH_REPO = "\u4ed3\u5e93"
ZH_CODE = "\u4ee3\u7801"
ZH_DIR = "\u76ee\u5f55"
ZH_FILE = "\u6587\u4ef6"
ZH_TEST = "\u6d4b\u8bd5"
ZH_SEARCH = "\u641c\u7d22"
ZH_FIND = "\u67e5\u627e"
ZH_LOOK = "\u770b"
ZH_LOOK2 = "\u770b\u770b"
ZH_READ = "\u8bfb"
ZH_READ_FULL = "\u8bfb\u53d6"
ZH_OPEN = "\u6253\u5f00"
ZH_CONTENT = "\u5185\u5bb9"
ZH_INSIDE = "\u91cc\u9762"
ZH_LIST = "\u5217\u51fa"
ZH_STRUCTURE = "\u7ed3\u6784"
ZH_PROJECT_STRUCTURE = "\u9879\u76ee\u7ed3\u6784"
ZH_DIR_STRUCTURE = "\u76ee\u5f55\u7ed3\u6784"
ZH_WHAT_FILES = "\u6709\u54ea\u4e9b\u6587\u4ef6"
ZH_EXISTS = "\u5b58\u5728"
ZH_HAS = "\u6709\u6ca1\u6709"
ZH_HAS_Q = "\u662f\u5426\u6709"
ZH_SEARCH_SHORT = "\u641c"
ZH_FIND_SHORT = "\u627e"
ZH_WHERE = "\u5728\u54ea"
ZH_DEFINE = "\u5b9a\u4e49"
ZH_WRITE = "\u5199"
ZH_WRITE_FULL = "\u5199\u5165"
ZH_APPEND = "\u8ffd\u52a0"
ZH_CREATE = "\u521b\u5efa"
ZH_UPDATE = "\u66f4\u65b0"
ZH_MODIFY = "\u4fee\u6539"
ZH_EDIT = "\u7f16\u8f91"
ZH_RUN = "\u8fd0\u884c"
ZH_EXEC = "\u6267\u884c"
ZH_SCRIPT = "\u811a\u672c"
ZH_IN = "\u5728"
ZH_TO = "\u5230"
COMPUTER_GOAL_ACTION_PATTERNS = (
    "send",
    "message",
    "open",
    "launch",
    "search",
    "find",
    "type",
    "input",
    "click",
    "\u53d1\u9001",
    "\u53d1\u6d88\u606f",
    "\u53d1\u4e00\u53e5",
    "\u53d1\u4e2a",
    "\u53d1\u6761",
    "\u53d1\u7ed9",
    "\u6253\u5f00",
    "\u641c\u7d22",
    "\u67e5\u627e",
    "\u8f93\u5165",
    "\u70b9\u51fb",
    "\u70b9\u5f00",
)
COMPUTER_GOAL_OBJECT_PATTERNS = (
    "qq",
    "wechat",
    "wecom",
    "feishu",
    "contact",
    "chat",
    "message",
    "window",
    "desktop",
    "\u8054\u7cfb\u4eba",
    "\u804a\u5929",
    "\u6d88\u606f",
    "\u5bf9\u8bdd",
    "\u7a97\u53e3",
    "\u684c\u9762",
)


from ._agent_types import AgentConfig, AgentOutcome


class WorkspaceAgent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        llm_client: BaseLLMClient | None = None,
        registry: Any | None = None,  # ServiceRegistry for IoC
    ) -> None:
        self.config = config or AgentConfig()
        self.llm_client = llm_client
        self.project_root = str(self.config.project_root or ".")
        self.classifier = RequestClassifier()

        # Resolve dependencies from registry (preferred) or instantiate manually (fallback)
        _get = (lambda name: registry.get(name)) if registry is not None and registry.has("world_model") else None

        self.world_model = _get("world_model") if _get else WorldModel(project_root=self.project_root)
        self.action_stats = _get("action_stats") if _get else ActionStats()
        self.action_generator = _get("action_generator") if _get else ActionGenerator(project_root=self.project_root)
        self.validator = _get("action_validator") if _get else ActionValidator(project_root=self.project_root, stats=self.action_stats)
        self.task_planner = _get("task_planner") if _get else UnifiedPlanner()
        self.search_planner = (
            _get("search_planner") if _get
            else SearchPlanner(
                world_model=self.world_model,
                action_generator=self.action_generator,
                base_planner=self.task_planner,
                depth=max(1, int(self.config.search_depth)),
                beam_size=max(1, int(self.config.search_beam)),
                stats=self.action_stats,
                validator=self.validator,
                project_root=self.project_root,
            )
        )
        self.executor = _get("executor") if _get else UnifiedExecutor(project_root=self.project_root)
        self.computer_feedback = None
        self.computer_learning = None
        self.workspace_learning = None
        if bool(self.config.enable_computer_feedback):
            try:
                self.computer_feedback = ComputerTaskFeedbackRecorder(
                    feedback_path=str(self.config.computer_feedback_path or "artifacts/audit/computer_task_feedback.jsonl"),
                    stats_path=str(self.config.computer_feedback_stats_path or "artifacts/audit/computer_task_stats.json"),
                    reflections_path=str(self.config.computer_reflections_path or "artifacts/memory/dialogue_reflections.jsonl"),
                )
            except Exception:
                self.computer_feedback = None
        if bool(self.config.enable_computer_learning) and (self.config.allow_browser or self.config.allow_desktop):
            try:
                computer_learning_path = str(
                    os.environ.get("COMPUTER_TASK_LEARNING_PATH", "")
                    or self.config.computer_learning_path
                    or "artifacts/memory/computer_task_learnings.json"
                )
                if computer_learning_path and not Path(computer_learning_path).is_absolute():
                    computer_learning_path = (Path(self.project_root) / computer_learning_path).as_posix()
                self.computer_learning = LearnedComputerTaskStore(
                    path=computer_learning_path
                )
            except Exception:
                self.computer_learning = None
        if bool(self.config.enable_workspace_learning):
            try:
                workspace_learning_path = str(
                    os.environ.get("WORKSPACE_TASK_LEARNING_PATH", "")
                    or self.config.workspace_learning_path
                    or "artifacts/memory/workspace_task_learnings.json"
                )
                if workspace_learning_path and not Path(workspace_learning_path).is_absolute():
                    workspace_learning_path = (Path(self.project_root) / workspace_learning_path).as_posix()
                self.workspace_learning = LearnedWorkspaceTaskStore(
                    path=workspace_learning_path
                )
            except Exception:
                self.workspace_learning = None

    def should_handle(self, query: str, intent: ChatIntent) -> bool:
        if not self.config.enabled or intent != ChatIntent.TASK:
            return False
        text = str(query or "").strip()
        if not text:
            return False
        low = text.lower()
        if self._is_template_request(text) or self._is_recovery_request(text):
            return True
        if self._parse_qq_send_goal(text) is not None:
            return True
        has_path = bool(self._extract_paths(text))
        if self._is_list_request(text, low):
            return True
        if self._is_exists_request(text, low) and has_path:
            return True
        if self._is_search_request(text, low):
            return has_path or self._extract_search_pattern(text) is not None
        if self._is_read_request(text, low) and has_path:
            return True
        if self._is_write_request(text, low) and has_path:
            return True
        if self._is_browser_request(text, low) or self._is_desktop_request(text, low):
            return True
        if self._looks_like_natural_language_computer_goal(text, low):
            return True
        if self._is_exec_request(text, low):
            return has_path or "pytest" in low or "tests" in low or ZH_TEST in text
        workspace_terms = (
            "repo",
            "repository",
            "project",
            "workspace",
            "codebase",
            ZH_PROJECT,
            ZH_REPO,
            ZH_CODE,
            ZH_DIR,
            ZH_FILE,
        )
        return has_path or any(term in text or term in low for term in workspace_terms)

    def handle(self, req: ChatRequest, intent: ChatIntent) -> AgentOutcome | None:
        if not self.should_handle(req.user_text, intent):
            return None

        goal = str(req.user_text or "").strip()
        special_plan, special_error = self._special_computer_plan(goal)
        if special_error:
            return AgentOutcome(handled=True, text=special_error, source="agent")
        memories = self._history_text(req.history)
        state = self.world_model.build_state(goal=goal)
        draft_plan, plan_origin = self._draft_plan(goal, state, memories, req=req, special_plan=special_plan)
        if draft_plan is None or not draft_plan.steps:
            return None

        plan, blocked, invalid = self._filter_plan(draft_plan, state)
        if not plan.steps:
            if blocked:
                outcome = self._blocked_outcome(blocked)
                self._record_computer_feedback(
                    req=req,
                    plan=draft_plan,
                    execution=None,
                    blocked=blocked,
                    invalid=invalid,
                    outcome=outcome,
                    repaired=False,
                )
                return outcome
            return None

        final_plan = plan
        final_execution = self.executor.run(plan)
        repaired = False

        if final_execution.failed and self.config.self_heal:
            final_plan, final_execution, repaired, blocked, invalid = self._heal_once(
                goal,
                final_plan,
                final_execution,
                state,
                blocked,
                invalid,
            )

        replan_attempts = 0
        allow_replan = plan_origin == "planner"
        while allow_replan and final_execution.failed and replan_attempts < max(0, int(self.config.replan_max)):
            replan_attempts += 1
            next_state = self.world_model.build_state(goal=goal, execution=final_execution)
            feedback = self._feedback_from_errors(final_execution)
            replanned = self._plan_with_modules(goal, next_state, memories, feedback=feedback)
            if replanned is None or not replanned.steps:
                break
            replanned, extra_blocked, extra_invalid = self._filter_plan(replanned, next_state)
            blocked.extend(extra_blocked)
            invalid.extend(extra_invalid)
            if not replanned.steps:
                break
            final_plan = replanned
            final_execution = self.executor.run(replanned)

        text, source = self._compose_answer(
            req,
            final_plan,
            final_execution,
            blocked=blocked,
            invalid=invalid,
            repaired=repaired,
        )
        outcome = AgentOutcome(
            handled=True,
            text=text,
            source=source,
            metadata={
                "agent_plan_origin": plan_origin,
                "computer_plan_origin": plan_origin,
                "blocked": list(blocked),
                "invalid": list(invalid),
                "failed": list(final_execution.failed),
                "completed": list(final_execution.completed),
            },
        )
        learned = self._record_computer_learning(goal, final_plan, final_execution, plan_origin=plan_origin)
        if learned is not None:
            outcome.metadata["agent_learned_template"] = str(learned.get("template_name", "") or "")
            outcome.metadata["computer_learned_template"] = str(learned.get("template_name", "") or "")
            outcome.metadata["computer_learned_success_count"] = int(learned.get("success_count", 0) or 0)
        workspace_learned = self._record_workspace_learning(goal, final_plan, final_execution, plan_origin=plan_origin)
        if workspace_learned is not None:
            outcome.metadata["agent_learned_template"] = str(workspace_learned.get("template_name", "") or "")
            outcome.metadata["workspace_learned_template"] = str(workspace_learned.get("template_name", "") or "")
            outcome.metadata["workspace_learned_success_count"] = int(workspace_learned.get("success_count", 0) or 0)
        feedback_event = self._record_computer_feedback(
            req=req,
            plan=final_plan,
            execution=final_execution,
            blocked=blocked,
            invalid=invalid,
            outcome=outcome,
            repaired=repaired,
        )
        self._save_computer_recovery(
            goal,
            final_plan,
            final_execution,
            outcome,
            feedback_event=feedback_event,
        )
        return outcome

    def _heal_once(
        self,
        goal: str,
        plan: Plan,
        execution: ExecutionResult,
        state,
        blocked: List[str],
        invalid: List[str],
    ) -> Tuple[Plan, ExecutionResult, bool, List[str], List[str]]:
        del goal
        healed = apply_self_heal(
            plan,
            execution,
            project_root=self.project_root,
            allow_create=bool(self.config.allow_write and self.config.self_heal_create),
            allow_chmod=bool(self.config.allow_write and self.config.self_heal_chmod),
        )
        repaired = False
        final_plan = healed.plan
        final_execution = execution
        if healed.repair_steps:
            repair_plan = Plan(
                goal="agent_self_heal",
                steps=list(healed.repair_steps),
                levels=[list(healed.repair_steps)],
            )
            repair_plan, extra_blocked, extra_invalid = self._filter_plan(repair_plan, state)
            blocked.extend(extra_blocked)
            invalid.extend(extra_invalid)
            if repair_plan.steps:
                repair_execution = self.executor.run(repair_plan)
                if not repair_execution.failed:
                    final_execution = self.executor.run(final_plan)
                    repaired = True
        elif healed.applied:
            final_execution = self.executor.run(final_plan)
            repaired = True
        return final_plan, final_execution, repaired, blocked, invalid

    def _draft_plan(
        self,
        goal: str,
        state,
        memories: List[str],
        *,
        req: ChatRequest,
        special_plan: Plan | None = None,
    ) -> Tuple[Plan | None, str]:
        if special_plan is not None and special_plan.steps:
            return special_plan, "special"
        low = str(goal or "").lower()
        natural_language_computer_goal = self._looks_like_natural_language_computer_goal(goal, low)
        if natural_language_computer_goal:
            learned = self._learned_computer_plan(goal)
            if learned is not None and learned.steps:
                return learned, "learned_local"
        learned_workspace = self._learned_workspace_plan(goal)
        if learned_workspace is not None and learned_workspace.steps:
            return learned_workspace, "learned_local_workspace"
        direct = self._direct_plan(goal)
        if direct is not None and direct.steps:
            return direct, "direct"
        llm_computer = self._llm_computer_plan(goal, req=req)
        if llm_computer is not None and llm_computer.steps:
            return llm_computer, "llm_structured"
        if natural_language_computer_goal:
            return None, ""
        return self._plan_with_modules(goal, state, memories, feedback=""), "planner"

    def _plan_with_modules(self, goal: str, state, memories: List[str], feedback: str) -> Plan | None:
        if bool(self.config.search_plan):
            return self.search_planner.plan(
                goal=goal,
                state=state,
                memories=memories,
                model=None,
                examples=[],
                feedback=feedback,
            )
        return self.task_planner.plan(
            goal,
            model=None,
            memories=memories,
            feedback=feedback,
            max_candidates=max(1, int(self.config.planner_candidates)),
        )

    def _learned_computer_plan(self, goal: str) -> Plan | None:
        if self.computer_learning is None:
            return None
        try:
            return self.computer_learning.build_plan_for_query(
                goal,
                min_successes=max(1, int(self.config.computer_learning_min_successes)),
                goal=goal,
            )
        except Exception:
            return None

    def _learned_workspace_plan(self, goal: str) -> Plan | None:
        if self.workspace_learning is None:
            return None
        try:
            return self.workspace_learning.build_plan_for_query(
                goal,
                min_successes=max(1, int(self.config.workspace_learning_min_successes)),
                goal=goal,
            )
        except Exception:
            return None

    def _llm_computer_plan(self, goal: str, *, req: ChatRequest) -> Plan | None:
        if self.llm_client is None or not self.llm_client.available():
            return None
        text = str(goal or "").strip()
        low = text.lower()
        if not self._looks_like_natural_language_computer_goal(text, low):
            return None
        if self._looks_like_explicit_sequence(text) or self._parse_explicit_command(text) is not None:
            return None
        allowed_commands: List[str] = []
        if self.config.allow_desktop:
            allowed_commands.extend(
                [
                    'focus window "<title>"',
                    'list windows',
                    'list controls in window "<title>"',
                    'click control "<control>" in window "<title>" control type "<type>"',
                    'type "<text>" into control "<control>" in window "<title>" control type "<type>" clear first',
                    'click text "<screen text>"',
                    'type "<text>"',
                    'hotkey ctrl+l',
                    'launch "<path>"',
                    'screenshot',
                    'ocr screen',
                ]
            )
        if self.config.allow_browser:
            allowed_commands.extend(
                [
                    'browser open https://example.com',
                    'browser click "#selector"',
                    'browser type "<text>" into "#selector"',
                    'browser wait text "<text>"',
                    'browser screenshot',
                ]
            )
        if not allowed_commands:
            return None
        examples: List[str] = []
        if self.config.allow_desktop:
            examples.append(
                'Example request: \u6253\u5f00QQ\u5e76\u67e5\u770b\u7a97\u53e3\u63a7\u4ef6\n'
                'Example output: {"steps":["focus window QQ","list controls in window QQ"]}'
            )
            examples.append(
                'Example request: \u6253\u5f00QQ\u7ed9\u76db\u54e5\u53d1\u4e00\u53e5\u665a\u5b89\n'
                'Example output: {"steps":["focus window QQ","type \\"\u76db\u54e5\\" into control \u641c\u7d22 in window QQ control type Edit clear first","click control \\"\u76db\u54e5\\" in window QQ control type ListItem index 0","click control \\"\u804a\u5929\u8f93\u5165\u533a\\" in window QQ control type Group","type \\"\u665a\u5b89\\"","click control \u53d1\u9001 in window QQ control type Button"]}'
            )
        prompt = (
            "Convert the user's desktop/browser automation request into a strict JSON plan.\n"
            "Return JSON only. Do not add markdown or explanation.\n"
            'Use this schema: {"steps":["command 1","command 2"]}.\n'
            "Each step must be a deterministic local command from the allowed list.\n"
            "Prefer safe local execution. If the task is unclear, unsafe, or impossible with the allowed commands, return {\"steps\":[]}.\n"
            "If the request asks to send a QQ message, use the verified local QQ pattern from the examples and substitute the extracted contact and message.\n"
            "Allowed commands:\n- "
            + "\n- ".join(allowed_commands)
            + ("\n\nExamples:\n" + "\n\n".join(examples) if examples else "")
            + "\n\nUser request:\n"
            + text
        )
        try:
            raw = str(self.llm_client.generate(prompt, req.history) or "").strip()
        except Exception:
            return None
        commands = self._parse_llm_computer_steps(raw)
        if not commands:
            return None
        steps: List[Task] = []
        for command in commands:
            task = self._parse_explicit_command(command)
            if task is None:
                return None
            if str(task.name or "") not in BROWSER_PLUGIN_TASKS and str(task.name or "") not in DESKTOP_PLUGIN_TASKS:
                return None
            steps.append(task)
        if not steps:
            return None
        return Plan(goal=text, steps=steps, levels=[[step] for step in steps])

    def _parse_llm_computer_steps(self, raw: str) -> List[str]:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            payload = self._parse_json_object(text)
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            return []
        rows = payload.get("steps", [])
        if not isinstance(rows, list):
            return []
        out: List[str] = []
        for item in rows:
            if isinstance(item, str):
                command = item.strip()
            elif isinstance(item, dict):
                command = str(item.get("command", "") or item.get("text", "")).strip()
            else:
                command = ""
            if command:
                out.append(command)
        return out

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(text[start : end + 1])
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _record_computer_learning(
        self,
        query: str,
        plan: Plan,
        execution: ExecutionResult | None,
        *,
        plan_origin: str,
    ) -> Dict[str, Any] | None:
        if self.computer_learning is None:
            return None
        try:
            return self.computer_learning.remember_success(
                query=query,
                plan=plan,
                execution=execution,
                source=plan_origin,
            )
        except Exception:
            return None

    def _record_workspace_learning(
        self,
        query: str,
        plan: Plan,
        execution: ExecutionResult | None,
        *,
        plan_origin: str,
    ) -> Dict[str, Any] | None:
        if self.workspace_learning is None:
            return None
        try:
            return self.workspace_learning.remember_success(
                query=query,
                plan=plan,
                execution=execution,
                source=plan_origin,
            )
        except Exception:
            return None

    def _direct_plan(self, query: str) -> Plan | None:
        text = str(query or "").strip()
        if not text:
            return None
        low = text.lower()
        if self._is_project_inspection_request(text, low):
            project_plan = self._project_inspection_plan(text)
            if project_plan is not None and project_plan.steps:
                return project_plan
        if self._looks_like_explicit_sequence(text):
            seq = self.task_planner.split_plan(text)
            if seq.steps and all(str(step.name or "") != "step" for step in seq.steps):
                return seq
        explicit = self._parse_explicit_command(text)
        if explicit is not None:
            return Plan(goal=text, steps=[explicit], levels=[[explicit]])

        computer_task = self._computer_task(text, low)
        if computer_task is not None:
            return Plan(goal=text, steps=[computer_task], levels=[[computer_task]])

        primary_path = self._pick_path(text)
        search_path = "."
        if primary_path:
            candidate = self._resolve_path(primary_path)
            if candidate.is_dir():
                search_path = primary_path
            elif "/" in primary_path:
                search_path = primary_path.rsplit("/", 1)[0] or "."

        if self._is_write_request(text, low) and primary_path:
            task = Task(
                name="write_file",
                detail=f"write {primary_path} ->",
                payload={"path": primary_path, "content": ""},
            )
            return Plan(goal=text, steps=[task], levels=[[task]])

        if self._is_exec_request(text, low):
            if primary_path and primary_path.lower().endswith(".py"):
                task = Task(name="run_script", detail=f"run {primary_path}", payload={"path": primary_path})
                return Plan(goal=text, steps=[task], levels=[[task]])
            if "pytest" in low or "tests" in low or ZH_TEST in text:
                task = Task(name="run_tests", detail="run tests", payload={"command": ""})
                return Plan(goal=text, steps=[task], levels=[[task]])

        if self._is_list_request(text, low):
            task = Task(name="list_dir", detail=f"list {primary_path or '.'}", payload={"path": primary_path or "."})
            return Plan(goal=text, steps=[task], levels=[[task]])

        if self._is_exists_request(text, low) and primary_path:
            task = Task(name="check_exists", detail=f"exists {primary_path}", payload={"path": primary_path})
            return Plan(goal=text, steps=[task], levels=[[task]])

        pattern = self._extract_search_pattern(text)
        if self._is_search_request(text, low) and pattern:
            if self._looks_like_filename(pattern):
                task = Task(
                    name="search_files",
                    detail=f"find {pattern} in {search_path}",
                    payload={"pattern": pattern, "path": search_path},
                )
            else:
                task = Task(
                    name="search_text",
                    detail=f"search {pattern} in {search_path}",
                    payload={"pattern": pattern, "path": search_path},
                )
            return Plan(goal=text, steps=[task], levels=[[task]])

        if self._is_read_request(text, low) and primary_path:
            task = Task(name="read_file", detail=f"read {primary_path}", payload={"path": primary_path})
            return Plan(goal=text, steps=[task], levels=[[task]])

        if primary_path:
            task = Task(name="read_file", detail=f"read {primary_path}", payload={"path": primary_path})
            return Plan(goal=text, steps=[task], levels=[[task]])

        if any(term in text or term in low for term in ("project structure", "repo structure", ZH_DIR_STRUCTURE, ZH_PROJECT_STRUCTURE)):
            task = Task(name="list_dir", detail="list .", payload={"path": "."})
            return Plan(goal=text, steps=[task], levels=[[task]])
        return None

    def _project_inspection_plan(self, query: str) -> Plan | None:
        text = str(query or "").strip()
        if not text:
            return None
        steps = [
            Task(name="list_dir", detail="list .", payload={"path": "."}),
            Task(name="search_files", detail="find README in .", payload={"pattern": "README", "path": "."}),
            Task(name="search_files", detail="find pyproject.toml in .", payload={"pattern": "pyproject.toml", "path": "."}),
            Task(name="search_files", detail="find main.py in .", payload={"pattern": "main.py", "path": "."}),
            Task(name="search_files", detail="find app_runtime.py in system", payload={"pattern": "app_runtime.py", "path": "system"}),
            Task(name="search_files", detail="find runtime.py in system/chat_v2", payload={"pattern": "runtime.py", "path": "system/chat_v2"}),
            Task(name="list_dir", detail="list system", payload={"path": "system"}),
            Task(name="list_dir", detail="list docs", payload={"path": "docs"}),
            Task(name="list_dir", detail="list tests", payload={"path": "tests"}),
        ]
        return Plan(goal=text, steps=steps, levels=[[step] for step in steps])

    def _parse_explicit_command(self, text: str) -> Task | None:
        stripped = str(text or "").strip()
        low = stripped.lower()
        command_prefixes = (
            "browser open ",
            "browser click ",
            "browser type ",
            "browser text ",
            "browser extract ",
            "browser wait text ",
            "browser screenshot",
            "browser close",
            "open url ",
            "browse ",
            "visit ",
            "go to ",
            "navigate ",
            "double click ",
            "click ",
            "drag ",
            "ocr ",
            "click text ",
            "click control ",
            "list windows",
            "list controls",
            "focus window ",
            "launch ",
            "hotkey ",
            "type ",
            "screenshot",
            "read ",
            "write ",
            "append ",
            "touch ",
            "list ",
            "exists ",
            "mkdir ",
            "copy ",
            "move ",
            "rename ",
            "delete ",
            "remove ",
            "rm ",
            "search ",
            "find ",
            "run ",
            "chmod ",
            "rollback ",
        )
        if any(low.startswith(prefix) for prefix in command_prefixes):
            return self.task_planner.parse_task(stripped)
        return None

    def _filter_plan(self, plan: Plan, state) -> Tuple[Plan, List[str], List[str]]:
        allowed_steps: List[Task] = []
        blocked: List[str] = []
        invalid: List[str] = []
        levels: List[List[Task]] = []
        source_levels = list(plan.levels or [list(plan.steps or [])])
        for level in source_levels:
            allowed_level: List[Task] = []
            for step in list(level or []):
                permitted, reason = self._permission_reason(step)
                if not permitted:
                    blocked.append(f"{step.detail}: {reason}")
                    continue
                ok, why = self.validator.is_valid(step, state)
                if not ok and not self._repairable_invalid(step, why):
                    invalid.append(f"{step.detail}: {why}")
                    continue
                allowed_steps.append(step)
                allowed_level.append(step)
            if allowed_level:
                levels.append(allowed_level)
        return Plan(goal=plan.goal, steps=allowed_steps, levels=levels), blocked, invalid

    def _permission_reason(self, step: Task) -> Tuple[bool, str]:
        name = str(step.name or "")
        if name in READ_ONLY_TASKS:
            return True, "ok"
        if name in BROWSER_PLUGIN_TASKS:
            if self.config.allow_browser:
                return True, "ok"
            return False, "browser control is disabled; re-run with --v2-agent-allow-browser"
        if name in DESKTOP_PLUGIN_TASKS:
            if self.config.allow_desktop:
                return True, "ok"
            return False, "desktop control is disabled; re-run with --v2-agent-allow-desktop"
        if name in WRITE_TASKS:
            if self.config.allow_write:
                return True, "ok"
            return False, "chat agent is running in read-only mode; re-run with --v2-agent-allow-write"
        if name in RISKY_WRITE_TASKS:
            return False, "this action is too destructive for chat agent mode"
        if name in EXEC_TASKS or name.startswith("plugin:"):
            if self.config.allow_exec:
                return True, "ok"
            return False, "code execution is disabled; re-run with --v2-agent-allow-exec"
        return False, "unsupported action for chat agent mode"

    def _repairable_invalid(self, step: Task, reason: str) -> bool:
        text = str(reason or "").lower()
        return step.name in READ_ONLY_TASKS and any(token in text for token in ("missing", "not found", "read missing"))

    def _blocked_outcome(self, blocked: Sequence[str]) -> AgentOutcome:
        lines = ["I planned workspace actions, but they are blocked in the current safety mode."]
        for item in list(blocked)[:3]:
            lines.append(f"- {item}")
        lines.append("Current default is read-only inspection. Enable write, browser, desktop, or exec only when you need it.")
        return AgentOutcome(handled=True, text="\n".join(lines), source="agent_blocked")

    def _compose_answer(
        self,
        req: ChatRequest,
        plan: Plan,
        execution: ExecutionResult,
        *,
        blocked: Sequence[str],
        invalid: Sequence[str],
        repaired: bool,
    ) -> Tuple[str, str]:
        return _agent_summary._compose_answer(self, req, plan, execution, blocked=blocked, invalid=invalid, repaired=repaired)
    def _llm_summary(
        self,
        req: ChatRequest,
        plan: Plan,
        execution: ExecutionResult,
        *,
        blocked: Sequence[str],
        invalid: Sequence[str],
        repaired: bool,
    ) -> str:
        return _agent_summary._llm_summary(self, req, plan, execution, blocked=blocked, invalid=invalid, repaired=repaired)
    def _local_summary(
        self,
        req: ChatRequest,
        execution: ExecutionResult,
        *,
        blocked: Sequence[str],
        invalid: Sequence[str],
        repaired: bool,
    ) -> str:
        return _agent_summary._local_summary(self, req, execution, blocked=blocked, invalid=invalid, repaired=repaired)
    def _project_inspection_summary(
        self,
        execution: ExecutionResult,
        *,
        blocked: Sequence[str],
        invalid: Sequence[str],
        repaired: bool,
    ) -> str:
        return _agent_summary._project_inspection_summary(self, execution, blocked=blocked, invalid=invalid, repaired=repaired)
    def _template_summary(
        self,
        execution: ExecutionResult,
        *,
        blocked: Sequence[str],
        invalid: Sequence[str],
        repaired: bool,
    ) -> str:
        return _agent_summary._template_summary(self, execution, blocked=blocked, invalid=invalid, repaired=repaired)
    def _findings_block(self, execution: ExecutionResult) -> str:
        return _agent_summary._findings_block(self, execution)
    def _structured_findings(self, execution: ExecutionResult) -> List[str]:
        return _agent_summary._structured_findings(self, execution)
    def _project_search_hits(
        self,
        rows: Dict[Tuple[str, str], List[str]],
        pattern: str,
        *,
        path: str | None = None,
    ) -> List[str]:
        return _agent_search._project_search_hits(self, rows, pattern, path=path)
    def _preview_items(self, items: Sequence[str], *, limit: int = 8) -> str:
        return _agent_summary._preview_items(self, items, limit=limit)
    def _feedback_from_errors(self, execution: ExecutionResult) -> str:
        return _agent_summary._feedback_from_errors(self, execution)
    def _record_computer_feedback(
        self,
        *,
        req: ChatRequest,
        plan: Plan,
        execution: ExecutionResult | None,
        blocked: Sequence[str],
        invalid: Sequence[str],
        outcome: AgentOutcome,
        repaired: bool,
    ) -> Dict[str, Any] | None:
        return _agent_computer._record_computer_feedback(self, req=req, plan=plan, execution=execution, blocked=blocked, invalid=invalid, outcome=outcome, repaired=repaired)
    def _special_computer_plan(self, query: str) -> Tuple[Plan | None, str]:
        return _agent_computer._special_computer_plan(self, query)
    def _special_qq_send_plan(self, text: str) -> Plan | None:
        return _agent_computer._special_qq_send_plan(self, text)
    def _parse_qq_send_goal(self, text: str) -> Tuple[str, str, bool] | None:
        return _agent_computer._parse_qq_send_goal(self, text)
    def _strip_wrapping_quotes(self, text: str) -> str:
        return _agent_computer._strip_wrapping_quotes(self, text)

    def _guess_qq_launch_target(self) -> str:
        return _agent_computer._guess_qq_launch_target(self)

    def _save_computer_recovery(
        self,
        query: str,
        plan: Plan,
        execution: ExecutionResult | None,
        outcome: AgentOutcome,
        feedback_event: Dict[str, Any] | None = None,
    ) -> None:
        return _agent_computer._save_computer_recovery(self, query, plan, execution, outcome, feedback_event=feedback_event)

    def _is_template_request(self, text: str) -> bool:
        return _agent_computer._is_template_request(self, text)

    def _is_recovery_request(self, text: str) -> bool:
        return _agent_computer._is_recovery_request(self, text)

    def _step_list(self, steps: Sequence[Task]) -> str:
        if not steps:
            return "(none)"
        return "\n".join(f"- {step.detail}" for step in steps[:8])

    def _history_text(self, history: Sequence[Any]) -> List[str]:
        out: List[str] = []
        for turn in list(history)[-8:]:
            role = str(getattr(turn, "role", "") or "").strip()
            text = str(getattr(turn, "text", "") or "").strip()
            if role and text:
                out.append(f"{role}: {text}")
        return out

    def _snippet(self, text: str, limit: int = 320) -> str:
        return snippet(text, limit)

    def _normalize_path(self, value: str) -> str:
        return normalize_path(value)

    def _looks_like_path(self, value: str) -> bool:
        return looks_like_path(value)

    def _looks_like_filename(self, value: str) -> bool:
        return looks_like_filename(value)

    def _resolve_path(self, path: str) -> Path:
        return resolve_path(path, self.project_root)

    def _extract_paths(self, query: str) -> List[str]:
        return extract_paths(query, self.project_root)

    def _pick_path(self, query: str) -> str:
        return pick_path(query, self.project_root)

    def _extract_url(self, query: str) -> str:
        return extract_url(query)

    def _extract_search_pattern(self, query: str) -> str | None:
        return extract_search_pattern(query)

    def _looks_like_explicit_sequence(self, query: str) -> bool:
        text = str(query or "").strip().lower()
        if not text or not any(sep in text for sep in (" then ", " next ", " finally ", ",")):
            return False
        phases = text.replace("then", "|").replace("next", "|").replace("finally", "|").split("|")
        parsed = 0
        for phase in phases:
            phase = phase.strip(" ,")
            if phase and self._parse_explicit_command(phase) is not None:
                parsed += 1
        return parsed >= 2

    def _computer_task(self, text: str, low: str) -> Task | None:
        url = self._extract_url(text)
        if url and self._is_browser_request(text, low):
            return Task(name="plugin:browser_open_url", detail=f"open url {url}", payload={"url": url})
        if low.startswith(
            (
                "browser open ",
                "browser click ",
                "browser type ",
                "browser text ",
                "browser extract ",
                "browser wait text ",
                "browser screenshot",
                "browser close",
                "click text ",
                "click control ",
                "ocr ",
                "click ",
                "double click ",
                "drag ",
            )
        ):
            return self.task_planner.parse_task(text)
        if "list windows" in low or "windows list" in low or "\u7a97\u53e3" in text:
            if "list" in low or ZH_LIST in text or "\u54ea\u4e9b" in text:
                return Task(name="plugin:desktop_list_windows", detail="list windows", payload={})
        if low.startswith("focus window "):
            title = text[13:].strip()
            return Task(name="plugin:desktop_focus_window", detail=f"focus window {title}", payload={"title": title})
        if low.startswith("launch "):
            target = text[7:].strip()
            return Task(name="plugin:desktop_launch", detail=f"launch {target}", payload={"target": target})
        if low.startswith("hotkey "):
            hotkey = text[7:].strip()
            return Task(name="plugin:desktop_hotkey", detail=f"hotkey {hotkey}", payload={"hotkey": hotkey})
        if low.startswith("type "):
            typed = text[5:]
            return Task(name="plugin:desktop_type_text", detail="type text", payload={"text": typed})
        if low == "screenshot" or low.startswith("screenshot "):
            return Task(name="plugin:desktop_screenshot", detail="screenshot", payload={})
        return None

    def _looks_like_natural_language_computer_goal(self, text: str, low: str) -> bool:
        if not (self.config.allow_browser or self.config.allow_desktop):
            return False
        if not text:
            return False
        has_action = any(token in low or token in text for token in COMPUTER_GOAL_ACTION_PATTERNS)
        if not has_action:
            return False
        return any(token in low or token in text for token in COMPUTER_GOAL_OBJECT_PATTERNS)

    def _is_project_inspection_request(self, text: str, low: str) -> bool:
        if not text:
            return False
        project_terms = (
            "project",
            "repo",
            "repository",
            "workspace",
            "codebase",
            ZH_PROJECT,
            ZH_REPO,
            ZH_CODE,
        )
        inspection_terms = (
            "inspect",
            "check",
            "review",
            "analyze",
            "summarize",
            "overview",
            "architecture",
            "structure",
            "module",
            "entrypoint",
            "runtime flow",
            "检查",
            "查看",
            "看看",
            "分析",
            "梳理",
            "总结",
            "概览",
            "架构",
            "结构",
            "模块",
            "入口",
            "流程",
        )
        has_project_term = any(term in low or term in text for term in project_terms)
        has_inspection_term = any(term in low or term in text for term in inspection_terms)
        if has_project_term and has_inspection_term:
            return True
        return has_project_term and any(term in low or term in text for term in ("what files", "目录", "文件", "readme", "docs", "tests"))

    def _is_read_request(self, text: str, low: str) -> bool:
        return any(
            term in low or term in text
            for term in ("read", "open", "show", "inspect", "look at", "cat ", "review", ZH_LOOK, ZH_LOOK2, ZH_READ, ZH_READ_FULL, ZH_OPEN, ZH_CONTENT, ZH_INSIDE)
        )

    def _is_list_request(self, text: str, low: str) -> bool:
        return any(
            term in low or term in text
            for term in ("list", "ls", "tree", "structure", "files", "directories", ZH_DIR, ZH_LIST, ZH_STRUCTURE, ZH_WHAT_FILES)
        )

    def _is_exists_request(self, text: str, low: str) -> bool:
        return any(
            term in low or term in text
            for term in ("exists", "exist", "has ", "present", ZH_EXISTS, ZH_HAS, ZH_HAS_Q)
        )

    def _is_search_request(self, text: str, low: str) -> bool:
        return any(
            term in low or term in text
            for term in ("search", "find", "grep", "look for", "where", ZH_SEARCH, ZH_FIND, ZH_SEARCH_SHORT, ZH_FIND_SHORT, ZH_WHERE, ZH_DEFINE)
        )

    def _is_write_request(self, text: str, low: str) -> bool:
        return any(
            term in low or term in text
            for term in ("write", "append", "create", "update", "modify", "edit", "save", ZH_WRITE, ZH_WRITE_FULL, ZH_APPEND, ZH_CREATE, ZH_UPDATE, ZH_MODIFY, ZH_EDIT)
        )

    def _is_browser_request(self, text: str, low: str) -> bool:
        if self._extract_url(text):
            return True
        return any(
            term in low or term in text
            for term in (
                "browser ",
                "open url",
                "browse",
                "visit",
                "navigate",
                "selector",
                "dom",
                "\u7f51\u9875",
                "\u6d4f\u89c8\u5668",
                "\u8bbf\u95ee",
            )
        )

    def _is_desktop_request(self, text: str, low: str) -> bool:
        return any(
            term in low or term in text
            for term in (
                "click ",
                "double click ",
                "drag ",
                "ocr ",
                "click text ",
                "click control ",
                "list windows",
                "list controls",
                "focus window",
                "launch ",
                "hotkey ",
                "type ",
                "screenshot",
                "window",
                "control",
                "browser",
                "app ",
                "\u7a97\u53e3",
                "\u805a\u7126",
                "\u542f\u52a8",
                "\u622a\u56fe",
                "\u8f93\u5165",
            )
        )

    def _is_exec_request(self, text: str, low: str) -> bool:
        return any(
            term in low or term in text
            for term in ("run", "execute", "pytest", "test", "script", ZH_RUN, ZH_EXEC, ZH_TEST, ZH_SCRIPT)
        )

