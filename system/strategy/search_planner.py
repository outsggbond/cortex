from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time
import os
import re

from system.core.planner import TaskPlanner, Plan, Task
from system.brain.world_model import WorldModel, WorldState
from system.computer_use.action.generator import ActionGenerator
from system.computer_use.action.validator import ActionValidator
from system.computer_use.action.stats import ActionStats
from system.l_utils.metrics import estimate_state_cost
from system.brain.distiller import load_distilled_library, DistilledLibrary
from config.policy_manager import get_policy, get_planner_overrides


@dataclass
class SearchNode:
    steps: List[str]
    score: float
    conf: float
    state: WorldState
    g_cost: float


class SearchPlanner:
    def __init__(
        self,
        world_model: WorldModel,
        action_generator: ActionGenerator,
        base_planner: Optional[TaskPlanner] = None,
        depth: int = 3,
        beam_size: int = 4,
        stats: Optional[ActionStats] = None,
        validator: Optional[ActionValidator] = None,
        project_root: str = ".",
        distiller: Optional[DistilledLibrary] = None,
    ):
        self.world = world_model
        self.actions = action_generator
        self.base = base_planner or TaskPlanner()
        self.depth = max(1, int(depth))
        self.beam_size = max(1, int(beam_size))
        self.stats = stats
        self.policy = get_policy()
        planner_overrides = get_planner_overrides()
        distiller_enable = planner_overrides.get("distiller_enable")
        if distiller is not None:
            self.distiller = distiller
        else:
            use_distiller = None
            if distiller_enable is not None:
                use_distiller = bool(distiller_enable)
            if use_distiller is None:
                use_distiller = os.environ.get("DISTILLER_ENABLE", "1") == "1"
            if use_distiller:
                path = os.environ.get("DISTILLER_PATH", "artifacts/memory/distilled_experience.json")
                self.distiller = load_distilled_library(path)
            else:
                self.distiller = None
        if validator is None:
            self.validator = ActionValidator(project_root, stats=stats)
        else:
            self.validator = validator
            if self.stats is not None and getattr(self.validator, "stats", None) is None:
                self.validator.stats = self.stats
        self.last_stats: Dict[str, Any] = {}

    def _share_model(self, model) -> None:
        if model is None:
            return
        if hasattr(self.base, "_cloud_model"):
            self.base._cloud_model = model
        if hasattr(self.base, "_cloud_model_checked"):
            self.base._cloud_model_checked = True
        if hasattr(self.actions, "_cloud_model"):
            self.actions._cloud_model = model
        if hasattr(self.actions, "_cloud_model_checked"):
            self.actions._cloud_model_checked = True

    def _resolve_model(self, model=None):
        if model is not None:
            return model
        resolver = getattr(self.actions, "_resolve_model", None)
        if callable(resolver):
            resolved = resolver(None)
            if resolved is not None:
                self._share_model(resolved)
                return resolved
        resolver = getattr(self.base, "_resolve_model", None)
        if callable(resolver):
            resolved = resolver(None)
            if resolved is not None:
                self._share_model(resolved)
                return resolved
        return None

    def _parse_task(self, text: str) -> Task:
        parser = getattr(self.base, "parse_task", None)
        if callable(parser):
            return parser(text)
        legacy = getattr(self.base, "_parse_task")
        return legacy(text)

    def _base_plan(
        self,
        goal: str,
        *,
        model=None,
        memories: Optional[List[str]] = None,
        feedback: str = "",
        max_candidates: int = 3,
        previous_plan: Optional[Plan] = None,
    ) -> Plan:
        planner = getattr(self.base, "plan")
        return planner(
            goal,
            model=model,
            memories=list(memories or []),
            feedback=feedback,
            max_candidates=max_candidates,
            previous_plan=previous_plan,
        )

    def plan(
        self,
        goal: str,
        state: WorldState,
        memories: List[str],
        model=None,
        examples: Optional[List[Dict[str, Any]]] = None,
        feedback: str = "",
        experiences: Optional[List[Dict[str, Any]]] = None,
    ) -> Plan:
        model = self._resolve_model(model)
        start_ts = time.perf_counter()
        expanded_nodes = 0
        candidates_considered = 0
        # Beam search over step sequences.
        nodes: List[SearchNode] = [SearchNode(steps=[], score=0.0, conf=0.0, state=state, g_cost=0.0)]
        best_plan: Optional[Plan] = None
        best_score = -1e9
        macro_matches = self._match_macros(goal, state)
        abstractions = self._get_abstractions(state)
        seed_paths = self._seed_paths(goal, state)

        # Seed plans from distilled paths (cold-start).
        for seed_steps in seed_paths:
            seed_plan = self._build_plan(goal, seed_steps)
            seed_pred, seed_conf = self.world.predict_with_confidence(state, seed_plan, experiences=experiences)
            if self._predict_failure(state, seed_pred, seed_conf):
                continue
            seed_g = self._plan_action_cost(seed_plan.steps, state)
            seed_h = self._heuristic_cost(seed_pred, goal)
            seed_cost = seed_g + seed_h + self._goal_requirement_penalty(goal, seed_plan.steps)
            seed_score = -self._cost_weight() * seed_cost
            seed_score += 0.3 * self.world.score_plan(seed_plan, state, feedback=feedback)
            seed_score += self._predict_penalty(state, seed_pred, seed_conf)
            prior_weight = float(self.policy.get("prior_weight", os.environ.get("PRIOR_WEIGHT", "0.25")))
            seed_score += prior_weight * self._score_action_priors(seed_plan.steps, state)
            seed_score += 0.4 * self._goal_alignment_bonus(goal, seed_plan.steps)
            seed_score += self._macro_bonus(macro_matches, seed_plan.steps)
            seed_score += self._abstraction_bonus(goal, seed_plan.steps, abstractions)
            action_conf = self._plan_action_conf(seed_plan.steps, state)
            explore_weight = float(os.environ.get("EXPLORE_WEIGHT", "0.2"))
            action_explore = float(os.environ.get("ACTION_EXPLORE_WEIGHT", "0.15"))
            seed_score += explore_weight * (1.0 - seed_conf)
            seed_score += action_explore * (1.0 - action_conf)
            if seed_score > best_score:
                best_score = seed_score
                best_plan = seed_plan
        # Seed plan from action generator as a baseline.
        seed_actions = self.actions.generate(
            goal=goal,
            state=state,
            memories=memories,
            model=model,
            examples=examples,
            feedback=feedback,
            max_actions=max(self.depth, self.beam_size),
        )
        if seed_actions:
            seed_plan = self._build_plan(goal, seed_actions)
            seed_pred, seed_conf = self.world.predict_with_confidence(state, seed_plan, experiences=experiences)
            if not self._predict_failure(state, seed_pred, seed_conf):
                seed_g = self._plan_action_cost(seed_plan.steps, state)
                seed_h = self._heuristic_cost(seed_pred, goal)
                seed_cost = seed_g + seed_h + self._goal_requirement_penalty(goal, seed_plan.steps)
                seed_score = -self._cost_weight() * seed_cost
                seed_score += 0.3 * self.world.score_plan(seed_plan, state, feedback=feedback)
                seed_score += self._predict_penalty(state, seed_pred, seed_conf)
                prior_weight = float(self.policy.get("prior_weight", os.environ.get("PRIOR_WEIGHT", "0.25")))
                seed_score += prior_weight * self._score_action_priors(seed_plan.steps, state)
                seed_score += 0.4 * self._goal_alignment_bonus(goal, seed_plan.steps)
                seed_score += self._macro_bonus(macro_matches, seed_plan.steps)
                seed_score += self._abstraction_bonus(goal, seed_plan.steps, abstractions)
                action_conf = self._plan_action_conf(seed_plan.steps, state)
                explore_weight = float(os.environ.get("EXPLORE_WEIGHT", "0.2"))
                action_explore = float(os.environ.get("ACTION_EXPLORE_WEIGHT", "0.15"))
                seed_score += explore_weight * (1.0 - seed_conf)
                seed_score += action_explore * (1.0 - action_conf)
                best_plan = seed_plan
                best_score = seed_score

        for _depth in range(self.depth):
            next_nodes: List[SearchNode] = []
            for node in nodes:
                # If confidence is low, widen exploration by increasing candidate count.
                base_actions = self.beam_size
                low_conf = float(os.environ.get("PRED_CONF_LOW", "0.45"))
                extra = int(os.environ.get("EXPLORE_EXTRA", "2"))
                if self.world.predictor.global_model.trained and self.world.predictor.global_model.confidence() < low_conf:
                    base_actions = min(base_actions + extra, max(base_actions, self.beam_size * 2))
                candidates = self.actions.generate(
                    goal=goal,
                    state=node.state,
                    memories=memories,
                    model=model,
                    examples=examples,
                    feedback=feedback,
                    max_actions=base_actions,
                )
                macro_candidates = self._macro_next_steps(macro_matches, node.steps)
                if macro_candidates:
                    ordered = macro_candidates + [c for c in candidates if c not in macro_candidates]
                    candidates = ordered[: max(base_actions, len(macro_candidates))]
                for cand in candidates:
                    candidates_considered += 1
                    task = self._parse_task(cand)
                    ok, _reason = self.validator.is_valid(task, node.state)
                    if not ok:
                        continue
                    steps = node.steps + [cand]
                    plan = self._build_plan(goal, steps)
                    step_plan = self._build_plan(goal, [cand])
                    pred_state, conf = self.world.predict_with_confidence(
                        node.state, step_plan, experiences=experiences
                    )
                    if self._predict_failure(node.state, pred_state, conf):
                        continue
                    action_cost = self._action_cost(task, node.state)
                    g_cost = node.g_cost + action_cost
                    h_cost = self._heuristic_cost(pred_state, goal)
                    total_cost = g_cost + h_cost + self._goal_requirement_penalty(goal, plan.steps)
                    score = -self._cost_weight() * total_cost
                    score += 0.3 * self.world.score_plan(plan, node.state, feedback=feedback)
                    score += self._predict_penalty(node.state, pred_state, conf)
                    prior_weight = float(self.policy.get("prior_weight", os.environ.get("PRIOR_WEIGHT", "0.25")))
                    score += prior_weight * self._score_action_priors(plan.steps, node.state)
                    score += 0.4 * self._goal_alignment_bonus(goal, plan.steps)
                    score += self._macro_bonus(macro_matches, plan.steps)
                    score += self._abstraction_bonus(goal, plan.steps, abstractions)
                    explore_weight = float(os.environ.get("EXPLORE_WEIGHT", "0.2"))
                    action_conf = self._plan_action_conf(plan.steps, node.state)
                    score += explore_weight * (1.0 - conf)
                    action_explore = float(os.environ.get("ACTION_EXPLORE_WEIGHT", "0.15"))
                    score += action_explore * (1.0 - action_conf)
                    node_conf = self._combine_conf(conf, action_conf, node.conf)
                    next_nodes.append(
                        SearchNode(steps=steps, score=score, conf=node_conf, state=pred_state, g_cost=g_cost)
                    )
                    expanded_nodes += 1
                    if score > best_score:
                        best_score = score
                        best_plan = plan
            if not next_nodes:
                break
            next_nodes.sort(key=lambda n: n.score, reverse=True)
            avg_conf = sum(n.conf for n in next_nodes) / max(1.0, len(next_nodes))
            extra_beam = 0
            if avg_conf < float(os.environ.get("ACTION_CONF_LOW", "0.45")):
                extra_beam = int(os.environ.get("ACTION_EXPLORE_BEAM", "2"))
            nodes = next_nodes[: self.beam_size + extra_beam]

        planning_time_s = max(0.0, time.perf_counter() - start_ts)
        self.last_stats = {
            "expanded_nodes": int(expanded_nodes),
            "candidates_considered": int(candidates_considered),
            "depth": int(self.depth),
            "beam_size": int(self.beam_size),
            "planning_time_s": round(float(planning_time_s), 6),
        }
        if best_plan is not None:
            return best_plan
        fallback = self._codegen_fallback_plan(
            goal=goal,
            state=state,
            memories=memories,
            model=model,
            examples=examples,
            feedback=feedback,
        )
        if fallback is not None:
            return fallback
        return self._base_plan(
            goal,
            model=model,
            memories=memories,
            feedback=feedback,
            max_candidates=self.beam_size,
        )

    def _codegen_fallback_plan(
        self,
        goal: str,
        state: WorldState,
        memories: List[str],
        model=None,
        examples: Optional[List[Dict[str, Any]]] = None,
        feedback: str = "",
    ) -> Optional[Plan]:
        paths = self._extract_goal_paths(goal)
        output_path = self._guess_output_path(goal, paths)
        script_paths = self._guess_script_paths(paths)
        input_paths = self._guess_input_paths(goal, paths, output_path, script_paths)
        missing = ",".join(state.missing_paths[:3]) if state.missing_paths else ""
        perms = ",".join(state.perm_paths[:3]) if state.perm_paths else ""
        glue_request = (
            f"generate code glue: goal={goal}; "
            f"output={output_path}; inputs={','.join(input_paths)}; "
            f"missing={missing}; perm={perms}"
        )
        try:
            base_plan = self._base_plan(
                goal,
                model=model,
                memories=memories,
                feedback=feedback,
                max_candidates=self.beam_size,
            )
        except Exception:
            base_plan = None
        if base_plan is not None and base_plan.steps:
            if not any(t.name == "generate_code" for t in base_plan.steps):
                glue_task = self._parse_task(glue_request)
                base_plan.steps.insert(0, glue_task)
                base_plan.levels.insert(0, [glue_task])
            return base_plan
        return self._build_plan(goal, [glue_request])

    def _build_plan(self, goal: str, steps: List[str]) -> Plan:
        tasks: List[Task] = [self._parse_task(s) for s in steps if s.strip()]
        levels = [[t] for t in tasks]
        return Plan(goal=goal.strip(), steps=tasks, levels=levels)

    def _seed_paths(self, goal: str, state: WorldState) -> List[List[str]]:
        if not self.distiller:
            return []
        return self.distiller.seed_paths(goal, state, k=int(os.environ.get("DISTILLER_SEED_TOPK", "3")))

    def _match_macros(self, goal: str, state: WorldState) -> List[dict]:
        if not self.distiller:
            return []
        return self.distiller.match_macros(goal, state, k=int(os.environ.get("DISTILLER_MACRO_TOPK", "3")))

    def _macro_next_steps(self, macros: List[dict], steps: List[str]) -> List[str]:
        if not macros:
            return []
        out = []
        for m in macros:
            msteps = m.get("steps") or []
            if not msteps:
                continue
            prefix_ok = True
            for i, s in enumerate(steps):
                if i >= len(msteps) or s != msteps[i]:
                    prefix_ok = False
                    break
            if not prefix_ok:
                continue
            if len(steps) < len(msteps):
                out.append(msteps[len(steps)])
        return out

    def _macro_bonus(self, macros: List[dict], tasks: List[Task]) -> float:
        if not macros or not tasks:
            return 0.0
        steps = [t.detail for t in tasks]
        weight = float(self.policy.get("macro_weight", os.environ.get("MACRO_WEIGHT", "0.6")))
        bonus = 0.0
        for m in macros:
            msteps = m.get("steps") or []
            if not msteps:
                continue
            match = 0
            for i, s in enumerate(steps):
                if i >= len(msteps) or s != msteps[i]:
                    break
                match += 1
            if match == 0:
                continue
            ratio = match / max(1.0, float(len(msteps)))
            mw = float(m.get("weight") or 1.0)
            bonus += weight * mw * ratio
            if match == len(msteps):
                bonus += weight * mw * 0.5
        return bonus

    def _get_abstractions(self, state: WorldState) -> List[dict]:
        abs_list = getattr(state, "active_abstractions", None) or []
        out: List[dict] = []
        for item in abs_list:
            if not isinstance(item, dict):
                continue
            token = item.get("token")
            if not token:
                continue
            members = item.get("members") or []
            score = float(item.get("score", 0.5))
            out.append({"token": str(token), "members": [str(m) for m in members if m], "score": score})
        return out

    def _compress_steps(self, steps: List[Task], abstractions: List[dict]) -> List[str]:
        if not abstractions or not steps:
            return [t.detail for t in steps]
        text = " ".join(t.detail for t in steps)
        compressed = text
        for abs_item in abstractions:
            token = abs_item.get("token")
            members = abs_item.get("members") or []
            if not token or not members:
                continue
            hit = False
            for m in members[:6]:
                if m and m in compressed:
                    hit = True
                    break
            if hit:
                compressed += f" {token}"
        return [compressed]

    def _abstraction_bonus(self, goal: str, tasks: List[Task], abstractions: List[dict]) -> float:
        if not abstractions or not tasks:
            return 0.0
        weight = float(self.policy.get("abstraction_weight", os.environ.get("ABSTRACTION_WEIGHT", "0.35")))
        path_weight = float(self.policy.get("abstraction_path_weight", os.environ.get("ABSTRACTION_PATH_WEIGHT", "0.15")))
        text = " ".join(t.detail for t in tasks)
        goal_text = goal or ""
        bonus = 0.0
        for abs_item in abstractions:
            token = abs_item.get("token")
            members = abs_item.get("members") or []
            score = float(abs_item.get("score", 0.5))
            related = False
            if token and token in text:
                related = True
            if not related:
                for m in members[:6]:
                    if m and (m in text or m in goal_text):
                        related = True
                        break
            if related:
                bonus += weight * score
        # path compression reward
        compressed = self._compress_steps(tasks, abstractions)
        if compressed:
            compression_gain = max(0, len(tasks) - len(compressed))
            if compression_gain > 0:
                bonus += path_weight * float(compression_gain)
        return bonus

    def _plan_action_cost(self, tasks: List[Task], state: WorldState) -> float:
        if not tasks:
            return 0.0
        cost = 0.0
        for t in tasks:
            cost += self._action_cost(t, state)
        return cost

    def _action_cost(self, task: Task, state: WorldState) -> float:
        base_cost, success = self.validator.estimate_cost_success(task, state)
        cost = base_cost
        destructive = self.validator.is_destructive(task)
        if self.stats is not None:
            mean, conf, _n = self.stats.estimate(task.name)
            stats_cost = self.stats.get_action_cost(
                task.name, success_prob=success, destructive=destructive, policy=self.policy
            )
            cost = (1.0 - conf) * base_cost + conf * stats_cost
        else:
            from system.l_utils.metrics import action_cost

            cost = action_cost(
                task.name,
                duration_ms=0.0,
                success_prob=success,
                base_cost=base_cost,
                destructive=destructive,
                policy=self.policy,
            )
        return max(0.1, cost)

    def _heuristic_cost(self, state: WorldState, goal: str) -> float:
        return estimate_state_cost(state, goal=goal)

    def _cost_weight(self) -> float:
        return float(self.policy.get("cost_weight", os.environ.get("COST_WEIGHT", "1.0")))

    @staticmethod
    def _normalize_path(path: str) -> str:
        cleaned = path.replace("\\", "/").strip()
        cleaned = cleaned.strip(" \t\r\n\"'<>(),;")
        cleaned = cleaned.rstrip(".")
        return cleaned

    @staticmethod
    def _looks_like_path(text: str) -> bool:
        return "." in text or "/" in text or "\\" in text

    def _extract_goal_paths(self, goal: str) -> List[str]:
        if not goal:
            return []
        hits = re.findall(r"[A-Za-z0-9_\-./\\\\]+\.[A-Za-z0-9]+", goal)
        out = []
        seen = set()
        for h in hits:
            p = self._normalize_path(h)
            if not p or p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _guess_output_path(self, goal: str, paths: List[str]) -> str:
        if not goal or not paths:
            return ""
        patterns = [
            r"(?:create|write|generate|save|output|produce)\s+([^\s,;]+)",
            r"(?:\u521b\u5efa|\u751f\u6210|\u5199\u5165|\u8f93\u51fa|\u4fdd\u5b58)\s*([^\s,;]+)",
        ]
        for pat in patterns:
            m = re.search(pat, goal, flags=re.IGNORECASE)
            if m:
                candidate = self._normalize_path(m.group(1))
                if self._looks_like_path(candidate):
                    return candidate
        for p in paths:
            low = p.lower()
            if low.endswith((".json", ".txt", ".md", ".yaml", ".yml", ".csv")):
                return p
        return paths[0] if paths else ""

    def _guess_input_paths(self, goal: str, paths: List[str], output_path: str, scripts: List[str]) -> List[str]:
        inputs: List[str] = []
        if goal:
            patterns = [
                r"(?:from|using|use|based on)\s+([^\s,;]+)",
                r"(?:\u4ece|\u4f7f\u7528|\u5229\u7528|\u57fa\u4e8e)\s*([^\s,;]+)",
            ]
            for pat in patterns:
                for m in re.finditer(pat, goal, flags=re.IGNORECASE):
                    p = self._normalize_path(m.group(1))
                    if p and self._looks_like_path(p):
                        inputs.append(p)
        if not inputs:
            for p in paths:
                if p == output_path:
                    continue
                if p in scripts:
                    continue
                inputs.append(p)
        out = []
        seen = set()
        for p in inputs:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _guess_script_paths(self, paths: List[str]) -> List[str]:
        return [p for p in paths if p.lower().endswith(".py")]

    def _parent_path(self, path: str) -> str:
        p = self._normalize_path(path)
        if "/" not in p:
            return ""
        return p.rsplit("/", 1)[0]

    def _goal_requirement_penalty(self, goal: str, tasks: List[Task]) -> float:
        if not goal or not tasks:
            return 0.0
        penalty = 0.0
        low = goal.lower()
        paths = self._extract_goal_paths(goal)
        script_paths = self._guess_script_paths(paths)
        output_path = self._guess_output_path(goal, paths)
        input_paths = self._guess_input_paths(goal, paths, output_path, script_paths)

        if input_paths:
            if not any(
                t.name == "read_file" and self._normalize_path((t.payload or {}).get("path", "")) in input_paths
                for t in tasks
            ):
                penalty += 6.0

        needs_run = False
        if any(k in low for k in ("run", "execute", "decode", "convert", "\u8fd0\u884c", "\u6267\u884c", "\u89e3\u7801")):
            needs_run = True
        if script_paths and input_paths and output_path:
            needs_run = True
        if needs_run:
            if script_paths:
                ok_run = any(
                    t.name == "run_script" and self._normalize_path((t.payload or {}).get("path", "")) in script_paths
                    for t in tasks
                )
            else:
                ok_run = any(t.name == "run_script" for t in tasks)
            if not ok_run:
                penalty += 6.0

        if output_path:
            if not any(
                t.name == "write_file" and self._normalize_path((t.payload or {}).get("path", "")) == output_path
                for t in tasks
            ):
                penalty += 8.0
        if any(k in low for k in ("create", "write", "generate", "new file", "\u521b\u5efa", "\u751f\u6210")):
            if not any(t.name in {"write_file", "touch_file", "append_file"} for t in tasks):
                penalty += 4.0

        parent = self._parent_path(output_path) if output_path else ""
        if parent:
            conflict_detected = False
            try:
                target = self.validator.fm._resolve(parent)
                if target.exists() and target.is_file():
                    conflict_detected = True
            except Exception:
                conflict_detected = False
            has_mkdir = any(
                s.name == "mkdir" and self._normalize_path((s.payload or {}).get("path", "")) == parent
                for s in tasks
            )
            has_move = any(
                s.name == "move_file" and self._normalize_path((s.payload or {}).get("src", "")) == parent
                for s in tasks
            )
            has_delete = any(
                s.name == "delete_file" and self._normalize_path((s.payload or {}).get("path", "")) == parent
                for s in tasks
            )
            conflict_hints = (
                "conflict",
                "path conflict",
                "file exists",
                "already exists",
                "not a directory",
                "notadirectoryerror",
                "路径冲突",
                "冲突",
                "已存在",
                "不是目录",
            )
            if (conflict_detected or any(k in low for k in conflict_hints)) and not (has_move or has_delete):
                penalty += 6.0
            if not has_mkdir:
                penalty += 6.0

        # Ordering penalties for dependency chains.
        def _idx(pred):
            for i, t in enumerate(tasks):
                if pred(t):
                    return i
            return None

        idx_read = _idx(
            lambda t: t.name == "read_file" and self._normalize_path((t.payload or {}).get("path", "")) in input_paths
        )
        if script_paths:
            idx_run = _idx(
                lambda t: t.name == "run_script" and self._normalize_path((t.payload or {}).get("path", "")) in script_paths
            )
        else:
            idx_run = _idx(lambda t: t.name == "run_script")
        idx_write = _idx(
            lambda t: t.name == "write_file" and self._normalize_path((t.payload or {}).get("path", "")) == output_path
        )
        idx_move = _idx(
            lambda t: t.name == "move_file" and self._normalize_path((t.payload or {}).get("src", "")) == parent
        )
        idx_delete = _idx(
            lambda t: t.name == "delete_file" and self._normalize_path((t.payload or {}).get("path", "")) == parent
        )
        idx_mkdir = _idx(
            lambda t: t.name == "mkdir" and self._normalize_path((t.payload or {}).get("path", "")) == parent
        )
        if idx_run is not None and idx_read is not None and idx_run < idx_read:
            penalty += 4.0
        if idx_write is not None:
            if idx_read is None:
                penalty += 6.0
            elif idx_write < idx_read:
                penalty += 6.0
            if idx_run is None:
                penalty += 8.0
            elif idx_write < idx_run:
                penalty += 8.0
            if idx_move is not None and idx_write < idx_move:
                penalty += 6.0
            if idx_delete is not None and idx_write < idx_delete:
                penalty += 6.0
        if idx_mkdir is not None:
            if idx_move is not None and idx_mkdir < idx_move:
                penalty += 4.0
            if idx_delete is not None and idx_mkdir < idx_delete:
                penalty += 4.0
            if idx_write is not None and idx_write < idx_mkdir:
                penalty += 6.0

        weight = float(self.policy.get("goal_requirement_weight", os.environ.get("GOAL_REQUIREMENT_WEIGHT", "1.0")))
        return penalty * weight

    def _score_action_priors(self, tasks: List[Task], state: WorldState) -> float:
        if not tasks:
            return 0.0
        total_cost = 0.0
        total_success = 0.0
        for t in tasks:
            cost, success = self.validator.estimate_cost_success(t, state)
            total_cost += cost
            total_success += success
        avg_success = total_success / max(1.0, len(tasks))
        # lower cost, higher success
        return (avg_success * 1.2) - (total_cost * 0.15)

    def _plan_action_conf(self, tasks: List[Task], state: WorldState) -> float:
        if not tasks:
            return 0.0
        total = 0.0
        for t in tasks:
            total += self.validator.action_confidence(t, state)
        return total / max(1.0, len(tasks))

    def _goal_alignment_bonus(self, goal: str, tasks: List[Task]) -> float:
        if not goal or not tasks:
            return 0.0
        steps_text = " ".join(t.detail for t in tasks)
        bonus = 0.0
        paths = self._extract_goal_paths(goal)
        for p in paths:
            if p in steps_text:
                bonus += 0.3
            else:
                bonus -= 0.5
        low = goal.lower()
        script_paths = self._guess_script_paths(paths)
        if script_paths and any(k in low for k in ("run", "execute", "decode", "运行", "执行", "解码")):
            if any("run " in t.detail for t in tasks):
                bonus += 0.3
            else:
                bonus -= 0.6
        if any(k in low for k in ("create", "write", "generate", "创建", "生成")):
            if any("write " in t.detail for t in tasks):
                bonus += 0.2
            else:
                bonus -= 0.4
        return bonus
    def _predict_failure(self, before: WorldState, after: WorldState, conf: float) -> bool:
        # If prediction is confident and indicates worsening errors, avoid this action.
        if conf < float(os.environ.get("PRED_FAIL_CONF", "0.6")):
            return False
        if len(after.errors) > len(before.errors):
            return True
        if len(after.missing_paths) > len(before.missing_paths):
            return True
        if len(after.perm_paths) > len(before.perm_paths):
            return True
        return False

    def _predict_penalty(self, before: WorldState, after: WorldState, conf: float) -> float:
        # Soft penalty when prediction shows regression.
        if conf <= 0:
            return 0.0
        delta_err = len(after.errors) - len(before.errors)
        delta_missing = len(after.missing_paths) - len(before.missing_paths)
        delta_perm = len(after.perm_paths) - len(before.perm_paths)
        penalty = 0.0
        if delta_err > 0:
            penalty -= 0.4 * delta_err * conf
        if delta_missing > 0:
            penalty -= 0.2 * delta_missing * conf
        if delta_perm > 0:
            penalty -= 0.2 * delta_perm * conf
        return penalty

    @staticmethod
    def _combine_conf(pred_conf: float, action_conf: float, prior_conf: float) -> float:
        confs = [c for c in (pred_conf, action_conf, prior_conf) if c > 0]
        if not confs:
            return 0.0
        return sum(confs) / float(len(confs))
