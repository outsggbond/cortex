from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from system.core.planner import Plan, Task
from config.policy_manager import get_policy


@dataclass
class HealResult:
    plan: Plan
    applied: List[str]
    repair_steps: List[Task]


_MISSING_PAT = re.compile(
    r"(no such file|not found|file not found|\u627e\u4e0d\u5230|\u4e0d\u5b58\u5728|\u6587\u4ef6\u4e0d\u5b58\u5728)",
    re.IGNORECASE,
)
_PERM_PAT = re.compile(
    r"(permission denied|access is denied|\u62d2\u7edd\u8bbf\u95ee|\u6743\u9650\u4e0d\u8db3)",
    re.IGNORECASE,
)
_NOTDIR_PAT = re.compile(r"(not a directory|notadirectoryerror|file exists|already exists)", re.IGNORECASE)


def apply_self_heal(
    plan: Plan,
    execution,
    project_root: str,
    allow_create: bool = False,
    allow_chmod: bool = False,
) -> HealResult:
    applied: List[str] = []
    repair_steps: List[Task] = []
    if not execution or not getattr(execution, "errors", None):
        return HealResult(plan, applied, repair_steps)

    root = Path(project_root).resolve()
    policy = get_policy()
    destructive_cost = float(policy.get("destructive_cost", 1.0))
    prefer_delete = destructive_cost <= 1.0
    ignored = {
        ".venv",
        ".idea",
        ".rollback",
        "__pycache__",
        "checkpoints",
        "audit",
        "node_modules",
    }

    def is_ignored(path: Path) -> bool:
        return any(part in ignored for part in path.parts)

    for err in execution.errors:
        name = (err.get("name") or "").strip()
        etype = (err.get("type") or "").strip()
        msg = (err.get("error") or "").strip()
        payload = err.get("payload") if isinstance(err.get("payload"), dict) else {}
        path = (payload.get("path") or "").strip()

        if not path:
            continue

        # Missing file: try to find a close match and rewrite plan paths.
        if etype == "FileNotFoundError" or _MISSING_PAT.search(msg):
            alt = _find_candidate(root, path, is_ignored)
            if alt:
                changed = _replace_path(plan, path, alt)
                if changed:
                    applied.append(f"path_fix {path} -> {alt}")
                    continue
            # Optionally create empty file for read_file only.
            if allow_create and name == "read_file":
                repair_steps.append(
                    Task(
                        name="write_file",
                        detail=f"write {path} ->",
                        payload={"path": path, "content": ""},
                    )
                )
                applied.append(f"create_empty {path}")
            continue

        # Permission issues: optionally chmod path.
        if allow_chmod and _PERM_PAT.search(msg):
            repair_steps.append(
                Task(
                    name="chmod",
                    detail=f"chmod {path} 0o666",
                    payload={"path": path, "mode": "0o666"},
                )
            )
            applied.append(f"chmod {path}")
            continue

        # Not a directory / path conflict: move blocking file and mkdir directory.
        if _NOTDIR_PAT.search(msg) or etype in {"NotADirectoryError", "FileExistsError"}:
            conflict = None
            target = root / path
            if target.exists() and target.is_file():
                conflict = target
            else:
                parent = target.parent
                if parent.exists() and parent.is_file():
                    conflict = parent
            if conflict is None:
                continue
            try:
                rel = conflict.relative_to(root).as_posix()
            except Exception:
                rel = conflict.as_posix()
            if prefer_delete:
                repair_steps.append(
                    Task(
                        name="delete_file",
                        detail=f"delete {rel}",
                        payload={"path": rel},
                    )
                )
            else:
                backup = Path(".rollback") / (rel.replace("/", "__") + ".block")
                repair_steps.append(
                    Task(
                        name="move_file",
                        detail=f"move {rel} -> {backup.as_posix()}",
                        payload={"src": rel, "dst": backup.as_posix()},
                    )
                )
            repair_steps.append(
                Task(
                    name="mkdir",
                    detail=f"mkdir {rel}",
                    payload={"path": rel},
                )
            )
            applied.append(f"path_conflict {rel}")

    return HealResult(plan, applied, repair_steps)


def _find_candidate(root: Path, missing: str, is_ignored) -> str | None:
    target = Path(missing).name
    if not target:
        return None
    matches: List[Path] = []
    for p in root.rglob(target):
        if p.is_dir():
            continue
        if is_ignored(p):
            continue
        matches.append(p)
        if len(matches) >= 20:
            break
    if not matches:
        return None
    # Prefer shortest relative path.
    matches.sort(key=lambda x: len(x.as_posix()))
    try:
        rel = matches[0].relative_to(root).as_posix()
    except Exception:
        rel = matches[0].as_posix()
    return rel


def _replace_path(plan: Plan, old: str, new: str) -> bool:
    changed = False
    for step in plan.steps:
        if not step.payload or not isinstance(step.payload, dict):
            continue
        if step.payload.get("path") == old:
            step.payload["path"] = new
            if old in step.detail:
                step.detail = step.detail.replace(old, new)
            changed = True
    return changed
