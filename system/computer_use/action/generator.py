from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Any

from system.brain.world_model import WorldState
from system.computer_use.cloud_text_generator import build_cloud_text_generator_from_env
from config.policy_manager import get_policy


class ActionGenerator:
    def __init__(self, project_root: str = "."):
        self.max_actions = int(os.environ.get("ACTION_MAX", "6"))
        self.root = Path(project_root).resolve()
        self._cloud_model = None
        self._cloud_model_checked = False
        self.ignore_dirs = {
            ".venv",
            ".idea",
            "__pycache__",
            "checkpoints",
            "audit",
            "node_modules",
        }

    def _resolve_model(self, model=None):
        if model is not None:
            return model
        if self._cloud_model_checked:
            return self._cloud_model
        self._cloud_model_checked = True
        self._cloud_model = build_cloud_text_generator_from_env()
        return self._cloud_model

    def generate(
        self,
        goal: str,
        state: WorldState,
        memories: List[str],
        model=None,
        examples: List[Dict[str, Any]] | None = None,
        feedback: str = "",
        max_actions: int | None = None,
    ) -> List[str]:
        max_actions = max(1, min(10, int(max_actions or self.max_actions)))
        actions: List[str] = []
        if examples:
            for ex in examples:
                for step in ex.get("plan_steps", [])[:3]:
                    if step and step not in actions:
                        actions.append(step)
                        if len(actions) >= max_actions:
                            return actions

        use_llm = os.environ.get("ACTION_USE_LLM", "1") != "0"
        model = self._resolve_model(model)
        if model is not None and hasattr(model, "generate") and use_llm:
            prompt = self._build_prompt(goal, state, feedback, max_actions, examples)
            try:
                raw = model.generate(prompt, memories, raw=True) or ""
            except TypeError:
                raw = model.generate(prompt, memories) or ""
            actions.extend(self._parse_actions(raw))
            actions = self._dedupe(actions)
            return actions[:max_actions]

        actions.extend(self._heuristic_actions(goal, state))
        return self._dedupe(actions)[:max_actions]

    def _build_prompt(
        self,
        goal: str,
        state: WorldState,
        feedback: str,
        max_actions: int,
        examples: List[Dict[str, Any]] | None,
    ) -> str:
        example_block = ""
        if examples:
            lines = []
            for i, ex in enumerate(examples[:2], 1):
                steps = "; ".join(ex.get("plan_steps", [])[:4])
                if steps:
                    lines.append(f"Example {i}: {steps}")
            if lines:
                example_block = "Examples:\n" + "\n".join(lines)
        feedback_block = f"Failure feedback: {feedback}" if feedback else ""
        return (
            f"You are an action generator. Given a goal and current state, output {max_actions} actions.\n"
            "Constraints: one action per line, start with a verb, make it executable.\n"
            "Only output the action list.\n"
            f"Goal: {goal}\n"
            f"State: {state.to_text()}\n"
            f"{feedback_block}\n"
            f"{example_block}\n"
        ).strip()

    def _parse_actions(self, text: str) -> List[str]:
        if not text:
            return []
        actions: List[str] = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            ln = re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip()
            if not ln:
                continue
            actions.append(ln)
        if actions:
            return actions
        parts = re.split(r"[，,。.;；\n]", text)
        for p in parts:
            p = p.strip()
            if p:
                actions.append(p)
        return actions

    def _normalize_path(self, path: str) -> str:
        cleaned = path.replace("\\", "/").strip()
        cleaned = cleaned.strip(" \t\r\n\"'<>(),;")
        cleaned = cleaned.rstrip(".")
        return cleaned

    def _looks_like_path(self, text: str) -> bool:
        return "." in text or "/" in text or "\\" in text

    def _resolve_path(self, path: str) -> Path:
        try:
            return (self.root / path).resolve()
        except Exception:
            return self.root / path

    def _extract_paths(self, goal: str) -> List[str]:
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

    def _guess_output(self, goal: str, paths: List[str]) -> str:
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

    def _guess_inputs(self, goal: str, paths: List[str], output_path: str) -> List[str]:
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
                if p.lower().endswith(".py"):
                    continue
                inputs.append(p)
        out = []
        seen = set()
        for p in inputs:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _guess_scripts(self, paths: List[str]) -> List[str]:
        return [p for p in paths if p.lower().endswith(".py")]

    def _parent_path(self, path: str) -> str:
        p = self._normalize_path(path)
        if "/" not in p:
            return ""
        return p.rsplit("/", 1)[0]

    def _conflict_path(self, parent_path: str) -> str:
        if not parent_path:
            return ""
        p = self._resolve_path(parent_path)
        try:
            if p.exists() and p.is_file():
                return parent_path
        except Exception:
            return ""
        return ""

    def _is_ignored(self, path: Path) -> bool:
        return any(part in self.ignore_dirs for part in path.parts)

    def _find_script_by_keywords(self, keywords: List[str], limit: int = 200) -> str:
        if not keywords:
            return ""
        hits: List[Path] = []
        count = 0
        for p in self.root.rglob("*.py"):
            if self._is_ignored(p):
                continue
            name = p.name.lower()
            if any(k in name for k in keywords):
                hits.append(p)
            count += 1
            if count >= limit:
                break
        if not hits:
            return ""
        hits.sort(key=lambda x: len(x.as_posix()))
        try:
            rel = hits[0].relative_to(self.root).as_posix()
        except Exception:
            rel = hits[0].as_posix()
        return rel

    def _heuristic_actions(self, goal: str, state: WorldState) -> List[str]:
        actions: List[str] = []
        low = (goal or "").lower()

        paths = self._extract_paths(goal)
        output_path = self._guess_output(goal, paths)
        input_paths = self._guess_inputs(goal, paths, output_path)
        script_paths = self._guess_scripts(paths)
        if not script_paths and any(k in low for k in ("decode", "\u89e3\u7801")):
            found = self._find_script_by_keywords(["decode", "decoder"])
            if found:
                script_paths = [found]

        if state.perm_paths:
            actions.append(f"chmod {state.perm_paths[0]} 0o666")
        elif "permission" in low or "\u6743\u9650" in goal:
            if input_paths:
                actions.append(f"chmod {input_paths[0]} 0o644")

        for p in input_paths:
            actions.append(f"read {p}")

        if script_paths:
            run_args = "{{last_output}}" if input_paths else ""
            for sp in script_paths[:1]:
                actions.append(f"run {sp} {run_args}".strip())

        if output_path:
            parent = self._parent_path(output_path)
            conflict = self._conflict_path(parent)
            if conflict:
                policy = get_policy()
                destructive = float(policy.get("destructive_cost", 1.0))
                if destructive <= 1.0:
                    actions.append(f"delete {conflict}")
                else:
                    safe = conflict.replace("/", "__")
                    actions.append(f"move {conflict} -> .rollback/{safe}.block")
            if parent:
                actions.append(f"mkdir {parent}")
            if script_paths:
                content = "{{last_output}}"
            elif input_paths:
                content = f"{{{{read:{input_paths[0]}}}}}"
            else:
                content = ""
            actions.append(f"write {output_path} -> {content}".strip())

        if state.missing_paths:
            path = state.missing_paths[0]
            actions.append(f"exists {path}")
            if "/" in path or "\\" in path:
                parent = path.replace("\\", "/").rsplit("/", 1)[0]
                if parent:
                    actions.append(f"mkdir {parent}")
            actions.append(f"write {path} ->")
        if state.errors:
            actions.append("search error in .")
            actions.append("run tests")
        actions.append("list .")
        if "todo" in low or "\u5f85\u529e" in goal:
            actions.append("search TODO in .")
        return actions

    def _dedupe(self, items: List[str]) -> List[str]:
        seen = set()
        out = []
        for it in items:
            key = it.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out
