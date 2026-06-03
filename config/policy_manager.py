from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


DEFAULT_POLICY = {
    "fail_risk_penalty": 4.0,
    "destructive_cost": 1.5,
    "time_weight": 1.0,
    "macro_weight": 0.6,
    "cost_weight": 1.0,
    "prior_weight": 0.25,
    "goal_requirement_weight": 1.0,
}

DEFAULT_COSTS = {
    "action_base": {},
    "action_multiplier": {},
    "write_audit_cost": 0.0,
}

_CACHE: Dict[str, Any] = {}
_CACHE_KEY = None
_COST_CACHE: Dict[str, Any] = {}
_COST_CACHE_KEY = None
_CTX_CACHE: Dict[str, Any] = {}
_CTX_CACHE_KEY = None
_NB_CACHE: Dict[str, Any] = {}
_NB_CACHE_KEY = None


def reload() -> None:
    global _CACHE, _CACHE_KEY, _COST_CACHE, _COST_CACHE_KEY, _CTX_CACHE, _CTX_CACHE_KEY, _NB_CACHE, _NB_CACHE_KEY
    _CACHE = {}
    _CACHE_KEY = None
    _COST_CACHE = {}
    _COST_CACHE_KEY = None
    _CTX_CACHE = {}
    _CTX_CACHE_KEY = None
    _NB_CACHE = {}
    _NB_CACHE_KEY = None


def _parse_value(raw: str):
    raw = raw.strip()
    if raw == "":
        return ""
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except Exception:
        return raw


def _parse_simple_yaml(text: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    current: str | None = None
    current_map: Dict[str, Any] | None = None
    child_key: str | None = None
    child_indent = None
    for raw in text.splitlines():
        line = raw
        if "#" in line:
            line = line.split("#", 1)[0]
        line = line.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            child_key = None
            child_indent = None
            if stripped.endswith(":"):
                current = stripped[:-1].strip()
                if current:
                    current_map = {}
                    data[current] = current_map
            else:
                current = None
                current_map = None
            continue
        if current_map is None:
            continue
        if indent == 2 and stripped.endswith(":"):
            child_key = stripped[:-1].strip()
            if child_key:
                current_map[child_key] = {}
                child_indent = 4
            continue
        if ":" not in stripped:
            continue
        key, val = stripped.split(":", 1)
        key = key.strip()
        val = _parse_value(val)
        if not key:
            continue
        if child_key and child_indent is not None and indent >= child_indent:
            try:
                current_map[child_key][key] = val
            except Exception:
                current_map[key] = val
        else:
            current_map[key] = val
    return data


def _load_yaml(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        # Fallback to simple parser.
        return _parse_simple_yaml(path.read_text(encoding="utf-8"))


def load_policies(path: str | Path = "config/policies.yaml") -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {"default": dict(DEFAULT_POLICY)}
    try:
        data = _load_yaml(p)
        if not isinstance(data, dict):
            return {"default": dict(DEFAULT_POLICY)}
        if "default" not in data:
            data["default"] = dict(DEFAULT_POLICY)
        return data
    except Exception:
        return {"default": dict(DEFAULT_POLICY)}


def load_contexts(path: str | Path = "config/contexts.yaml") -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = _load_yaml(p)
        if not isinstance(data, dict):
            return {}
        normalized: Dict[str, Dict[str, Any]] = {}
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            key = k.strip().lower()
            if not key:
                continue
            normalized[key] = v if isinstance(v, dict) else {}
        return normalized
    except Exception:
        return {}


def load_costs(path: str | Path = "config/costs.yaml") -> Dict[str, Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {"default": dict(DEFAULT_COSTS)}
    try:
        data = _load_yaml(p)
        if isinstance(data, dict):
            if "default" not in data:
                data["default"] = dict(DEFAULT_COSTS)
            return data
        return {"default": dict(DEFAULT_COSTS)}
    except Exception:
        return {"default": dict(DEFAULT_COSTS)}


def select_context(name: str | None = None, path: str | Path = "config/contexts.yaml") -> Dict[str, Any]:
    ctx_name = (name or os.environ.get("AGENT_CONTEXT", "")).strip().lower()
    if not ctx_name:
        return {}
    contexts = load_contexts(path)
    return resolve_context(ctx_name, contexts)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_context(name: str, contexts: Dict[str, Dict[str, Any]], stack: Optional[List[str]] = None) -> Dict[str, Any]:
    if not name:
        return {}
    stack = stack or []
    if name in stack:
        return {}
    ctx = contexts.get(name, {})
    if not isinstance(ctx, dict):
        return {}
    base = {}
    inherit = ctx.get("inherit")
    if isinstance(inherit, str):
        bases = [inherit]
    elif isinstance(inherit, list):
        bases = [b for b in inherit if isinstance(b, str)]
    else:
        bases = []
    if bases:
        for b in bases:
            base_ctx = resolve_context(b.strip().lower(), contexts, stack + [name])
            if base_ctx:
                base = _deep_merge(base, base_ctx)
    ctx_override = {k: v for k, v in ctx.items() if k != "inherit"}
    return _deep_merge(base, ctx_override)


def select_policy(name: str | None = None, path: str | Path = "config/policies.yaml") -> Dict[str, Any]:
    policy_name = (name or os.environ.get("AGENT_POLICY", "default")).strip().lower()
    policies = load_policies(path)
    policy = policies.get(policy_name) or policies.get("default") or dict(DEFAULT_POLICY)
    merged = dict(DEFAULT_POLICY)
    merged.update(policy)
    return merged


def apply_env_overrides(policy: Dict[str, Any]) -> Dict[str, Any]:
    overrides = {
        "fail_risk_penalty": os.environ.get("FAIL_RISK_PENALTY"),
        "destructive_cost": os.environ.get("DESTRUCTIVE_COST"),
        "time_weight": os.environ.get("TIME_WEIGHT"),
        "macro_weight": os.environ.get("MACRO_WEIGHT"),
        "cost_weight": os.environ.get("COST_WEIGHT"),
        "prior_weight": os.environ.get("PRIOR_WEIGHT"),
        "goal_requirement_weight": os.environ.get("GOAL_REQUIREMENT_WEIGHT"),
    }
    for key, val in overrides.items():
        if val is None or str(val).strip() == "":
            continue
        try:
            policy[key] = float(val)
        except Exception:
            continue
    return policy


def get_policy(path: str | Path = "config/policies.yaml") -> Dict[str, Any]:
    global _CACHE, _CACHE_KEY
    p = Path(path)
    mtime = 0.0
    try:
        if p.exists():
            mtime = p.stat().st_mtime
    except Exception:
        mtime = 0.0
    ctx = select_context()
    ctx_mtime = 0.0
    try:
        ctx_path = Path("config/contexts.yaml")
        if ctx_path.exists():
            ctx_mtime = ctx_path.stat().st_mtime
    except Exception:
        ctx_mtime = 0.0
    ctx_policy = ctx.get("policy") if isinstance(ctx, dict) else None
    env_key = (
        os.environ.get("AGENT_CONTEXT", ""),
        os.environ.get("AGENT_POLICY", "default"),
        os.environ.get("FAIL_RISK_PENALTY", ""),
        os.environ.get("DESTRUCTIVE_COST", ""),
        os.environ.get("TIME_WEIGHT", ""),
        os.environ.get("MACRO_WEIGHT", ""),
        os.environ.get("COST_WEIGHT", ""),
        os.environ.get("PRIOR_WEIGHT", ""),
        os.environ.get("GOAL_REQUIREMENT_WEIGHT", ""),
    )
    cache_key = (mtime, ctx_mtime, env_key)
    if _CACHE and _CACHE_KEY == cache_key:
        return dict(_CACHE)
    policy = select_policy(name=ctx_policy, path=path)
    if isinstance(ctx, dict):
        overrides = ctx.get("policy_overrides") if isinstance(ctx.get("policy_overrides"), dict) else {}
        if overrides:
            policy.update(overrides)
    policy = apply_env_overrides(policy)
    _CACHE = dict(policy)
    _CACHE_KEY = cache_key
    return dict(policy)


def policy_json(path: str | Path = "config/policies.yaml") -> str:
    return json.dumps(get_policy(path), ensure_ascii=False, indent=2)


def apply_env_cost_overrides(costs: Dict[str, Any]) -> Dict[str, Any]:
    write_audit = os.environ.get("WRITE_AUDIT_COST")
    if write_audit:
        try:
            costs["write_audit_cost"] = float(write_audit)
        except Exception:
            pass
    for key, val in os.environ.items():
        if not key.startswith("ACTION_BASE_"):
            continue
        action = key[len("ACTION_BASE_") :].lower()
        try:
            costs.setdefault("action_base", {})[action] = float(val)
        except Exception:
            continue
    return costs


def get_costs(path: str | Path = "config/costs.yaml") -> Dict[str, Any]:
    global _COST_CACHE, _COST_CACHE_KEY
    p = Path(path)
    mtime = 0.0
    try:
        if p.exists():
            mtime = p.stat().st_mtime
    except Exception:
        mtime = 0.0
    ctx = select_context()
    ctx_mtime = 0.0
    try:
        ctx_path = Path("config/contexts.yaml")
        if ctx_path.exists():
            ctx_mtime = ctx_path.stat().st_mtime
    except Exception:
        ctx_mtime = 0.0
    env_key = (
        os.environ.get("AGENT_CONTEXT", ""),
        os.environ.get("WRITE_AUDIT_COST", ""),
    )
    cache_key = (mtime, ctx_mtime, env_key)
    if _COST_CACHE and _COST_CACHE_KEY == cache_key:
        return dict(_COST_CACHE)
    costs_all = load_costs(path)
    base = dict(DEFAULT_COSTS)
    base.update(costs_all.get("default", {}))
    if isinstance(ctx, dict):
        ctx_over = ctx.get("cost_overrides") if isinstance(ctx.get("cost_overrides"), dict) else {}
        if ctx_over:
            base.update(ctx_over)
    base = apply_env_cost_overrides(base)
    _COST_CACHE = dict(base)
    _COST_CACHE_KEY = cache_key
    return dict(base)


def get_planner_overrides(path: str | Path = "config/contexts.yaml") -> Dict[str, Any]:
    global _CTX_CACHE, _CTX_CACHE_KEY
    p = Path(path)
    mtime = 0.0
    try:
        if p.exists():
            mtime = p.stat().st_mtime
    except Exception:
        mtime = 0.0
    env_key = (os.environ.get("AGENT_CONTEXT", ""),)
    cache_key = (mtime, env_key)
    if _CTX_CACHE and _CTX_CACHE_KEY == cache_key:
        return dict(_CTX_CACHE)
    ctx = select_context(path=path)
    overrides = ctx.get("planner_overrides") if isinstance(ctx.get("planner_overrides"), dict) else {}
    _CTX_CACHE = dict(overrides) if overrides else {}
    _CTX_CACHE_KEY = cache_key
    return dict(_CTX_CACHE)


def get_nanobrain_overrides(path: str | Path = "config/contexts.yaml") -> Dict[str, Any]:
    global _NB_CACHE, _NB_CACHE_KEY
    p = Path(path)
    mtime = 0.0
    try:
        if p.exists():
            mtime = p.stat().st_mtime
    except Exception:
        mtime = 0.0
    env_key = (os.environ.get("AGENT_CONTEXT", ""),)
    cache_key = (mtime, env_key)
    if _NB_CACHE and _NB_CACHE_KEY == cache_key:
        return dict(_NB_CACHE)
    ctx = select_context(path=path)
    overrides = ctx.get("nanobrain_overrides") if isinstance(ctx.get("nanobrain_overrides"), dict) else {}
    _NB_CACHE = dict(overrides) if overrides else {}
    _NB_CACHE_KEY = cache_key
    return dict(_NB_CACHE)
