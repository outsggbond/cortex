from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from system.core.planner import TaskPlanner
from system.brain.world_model import WorldState


TOKEN_RE = re.compile(r"[a-zA-Z0-9_.]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    out = []
    for t in tokens:
        if len(t) >= 3 or "." in t:
            out.append(t)
    return out[:24]


def _features_from_state(state_dict: Dict[str, Any] | None) -> Dict[str, Any]:
    if not state_dict:
        return {
            "has_perm": False,
            "has_missing": False,
            "has_errors": False,
            "error_types": [],
            "missing_names": [],
            "perm_names": [],
        }
    errors = list(state_dict.get("errors") or [])
    error_types = list(state_dict.get("error_types") or [])
    missing = list(state_dict.get("missing_paths") or [])
    perm = list(state_dict.get("perm_paths") or [])
    missing_names = [Path(p).name for p in missing if p]
    perm_names = [Path(p).name for p in perm if p]
    return {
        "has_perm": bool(perm),
        "has_missing": bool(missing),
        "has_errors": bool(errors),
        "error_types": error_types[:6],
        "missing_names": missing_names[:6],
        "perm_names": perm_names[:6],
    }


def _match_score(entry: Dict[str, Any], goal_tokens: List[str], state: WorldState) -> float:
    score = 0.0
    feat = entry.get("features", {})
    if feat.get("has_perm") and state.perm_paths:
        score += 1.2
    if feat.get("has_missing") and state.missing_paths:
        score += 1.0
    if feat.get("has_errors") and state.errors:
        score += 0.6
    entry_goal = set(entry.get("goal_tokens") or [])
    if entry_goal and goal_tokens:
        overlap = len(entry_goal.intersection(goal_tokens)) / max(1.0, float(len(entry_goal)))
        score += 1.0 * overlap
    entry_err = set(feat.get("error_types") or [])
    if entry_err and state.error_types:
        overlap = len(entry_err.intersection(state.error_types)) / max(1.0, float(len(entry_err)))
        score += 0.6 * overlap
    missing_names = set(feat.get("missing_names") or [])
    if missing_names and state.missing_paths:
        names = {Path(p).name for p in state.missing_paths}
        if missing_names.intersection(names):
            score += 0.6
    perm_names = set(feat.get("perm_names") or [])
    if perm_names and state.perm_paths:
        names = {Path(p).name for p in state.perm_paths}
        if perm_names.intersection(names):
            score += 0.6
    weight = float(entry.get("weight") or 0.0)
    score += 0.2 * weight
    return score


class DistilledLibrary:
    def __init__(self, macros: List[Dict[str, Any]] | None = None, seeds: List[Dict[str, Any]] | None = None):
        self.macros = macros or []
        self.seeds = seeds or []

    @classmethod
    def load(cls, path: str | Path) -> Optional["DistilledLibrary"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        macros = list(data.get("macros") or [])
        seeds = list(data.get("seeds") or [])
        return cls(macros=macros, seeds=seeds)

    def match_macros(self, goal: str, state: WorldState, k: int = 3) -> List[Dict[str, Any]]:
        if not self.macros:
            return []
        goal_tokens = _tokenize(goal)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for m in self.macros:
            s = _match_score(m, goal_tokens, state)
            if s <= 0:
                continue
            scored.append((s, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]

    def seed_paths(self, goal: str, state: WorldState, k: int = 3) -> List[List[str]]:
        if not self.seeds:
            return []
        goal_tokens = _tokenize(goal)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for s in self.seeds:
            score = _match_score(s, goal_tokens, state)
            if score <= 0:
                continue
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _score, item in scored[:k]:
            steps = item.get("steps") or []
            if steps:
                out.append(list(steps))
        return out


def _pattern_segments(names: List[str], pattern: List[str]) -> List[Tuple[int, int]]:
    out = []
    n = len(pattern)
    if n == 0:
        return out
    for i in range(len(names) - n + 1):
        if names[i : i + n] == pattern:
            out.append((i, i + n))
    return out


def _extract_macros(tasks, steps, max_steps: int) -> List[Tuple[str, List[str]]]:
    macros: List[Tuple[str, List[str]]] = []
    names = [t.name for t in tasks]
    # Full prefix macro if small enough.
    if 2 <= len(steps) <= max_steps:
        sig = ">".join(names[: len(steps)])
        macros.append((sig, steps[:]))
    # Known useful patterns.
    patterns = [
        ["chmod", "read_file", "run_script"],
        ["read_file", "run_script", "write_file"],
        ["move_file", "mkdir"],
    ]
    for pat in patterns:
        for i, j in _pattern_segments(names, pat):
            seg = steps[i:j]
            sig = ">".join(names[i:j])
            macros.append((sig, seg))
    return macros


def distill(
    input_path: str = "artifacts/memory/experience_store.jsonl",
    output_path: str = "artifacts/memory/distilled_experience.json",
    max_macros: int = 200,
    max_seeds: int = 200,
) -> DistilledLibrary:
    src = Path(input_path)
    if not src.exists():
        return DistilledLibrary()
    lines = src.read_text(encoding="utf-8").splitlines()
    items: List[Dict[str, Any]] = []
    for line in lines:
        try:
            items.append(json.loads(line))
        except Exception:
            continue

    planner = TaskPlanner()
    macro_map: Dict[str, Dict[str, Any]] = {}
    seed_list: List[Dict[str, Any]] = []

    for item in items:
        if not item.get("success"):
            continue
        steps = item.get("plan_steps") or []
        if not steps:
            continue
        tasks = [planner._parse_task(s) for s in steps]
        goal = item.get("goal", "")
        score = float(item.get("score") or 0.0)
        meta = item.get("meta") or {}
        replans = int(meta.get("replans") or 0)
        score_adj = max(0.0, score - 0.2 * replans)
        weight = 1.0 + score_adj
        features = _features_from_state(item.get("before_state") or {})
        goal_tokens = _tokenize(goal)

        seed_list.append(
            {
                "steps": steps[:8],
                "goal_tokens": goal_tokens,
                "features": features,
                "weight": weight,
                "score": score_adj,
            }
        )

        for sig, seg in _extract_macros(tasks, steps, max_steps=4):
            existing = macro_map.get(sig)
            entry = {
                "name": f"macro_{sig.replace('>', '_')}",
                "signature": sig,
                "steps": seg,
                "goal_tokens": goal_tokens,
                "features": features,
                "weight": weight,
                "score": score_adj,
            }
            if existing is None or float(existing.get("weight") or 0.0) < weight:
                macro_map[sig] = entry

    macros = list(macro_map.values())
    macros.sort(key=lambda x: float(x.get("weight") or 0.0), reverse=True)
    macros = macros[:max_macros]

    seed_list.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    seeds = seed_list[:max_seeds]

    payload = {
        "generated_at": time.time(),
        "source": str(input_path),
        "macros": macros,
        "seeds": seeds,
    }
    Path(output_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return DistilledLibrary(macros=macros, seeds=seeds)


def load_distilled_library(path: str = "artifacts/memory/distilled_experience.json") -> Optional[DistilledLibrary]:
    return DistilledLibrary.load(path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Distill experience_store.jsonl into macro library.")
    parser.add_argument("--input", type=str, default="artifacts/memory/experience_store.jsonl")
    parser.add_argument("--output", type=str, default="artifacts/memory/distilled_experience.json")
    parser.add_argument("--max-macros", type=int, default=200)
    parser.add_argument("--max-seeds", type=int, default=200)
    args = parser.parse_args()
    distill(args.input, args.output, max_macros=args.max_macros, max_seeds=args.max_seeds)
    print(f"distilled_ok -> {args.output}")


if __name__ == "__main__":
    main()
