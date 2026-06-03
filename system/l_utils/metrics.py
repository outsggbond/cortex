from __future__ import annotations

from typing import Dict, List, Any

from config.policy_manager import get_policy, get_costs


BASE_ACTION_COST = {
    "read_file": 1.0,
    "write_file": 1.0,
    "append_file": 1.0,
    "touch_file": 1.0,
    "chmod": 1.0,
    "mkdir": 2.0,
    "move_file": 2.2,
    "copy_file": 2.0,
    "list_dir": 0.6,
    "check_exists": 0.6,
    "search_text": 1.5,
    "search_files": 1.5,
    "run_script": 10.0,
    "run_tests": 10.0,
    "generate_code": 8.0,
    "rollback": 5.0,
    "delete_file": 3.0,
}

REPLAN_COST = 50.0
DEFAULT_ACTION_COST = 3.0
FAILURE_RISK_WEIGHT = 4.0
TIME_PENALTY_CAP = 6.0
WRITE_ACTIONS = {"write_file", "append_file", "touch_file"}


def base_action_cost(action_name: str) -> float:
    return float(BASE_ACTION_COST.get(action_name, DEFAULT_ACTION_COST))


def action_cost(
    action_name: str,
    duration_ms: float = 0.0,
    success_prob: float | None = None,
    base_cost: float | None = None,
    destructive: bool = False,
    policy: Dict[str, Any] | None = None,
) -> float:
    pol = policy or get_policy()
    costs = get_costs()
    fail_weight = float(pol.get("fail_risk_penalty", FAILURE_RISK_WEIGHT))
    time_weight = float(pol.get("time_weight", 1.0))
    destructive_mul = float(pol.get("destructive_cost", 1.0))
    cost = float(base_cost if base_cost is not None else base_action_cost(action_name))
    action_base = costs.get("action_base") if isinstance(costs.get("action_base"), dict) else {}
    if action_name in action_base:
        try:
            cost = float(action_base.get(action_name))
        except Exception:
            pass
    action_mult = costs.get("action_multiplier") if isinstance(costs.get("action_multiplier"), dict) else {}
    if action_name in action_mult:
        try:
            cost *= float(action_mult.get(action_name))
        except Exception:
            pass
    if destructive:
        cost *= destructive_mul
    if action_name in WRITE_ACTIONS:
        try:
            cost += float(costs.get("write_audit_cost", 0.0))
        except Exception:
            pass
    if duration_ms > 0:
        cost += min(TIME_PENALTY_CAP, (float(duration_ms) / 1000.0) * time_weight)
    if success_prob is not None:
        risk = max(0.0, 1.0 - float(success_prob))
        cost += fail_weight * risk
    return max(0.1, cost)


def estimate_state_cost(state: Any, goal: str = "") -> float:
    if state is None:
        return 0.0
    errors = len(getattr(state, "errors", []) or [])
    missing = len(getattr(state, "missing_paths", []) or [])
    perm = len(getattr(state, "perm_paths", []) or [])
    added = int(getattr(state, "added_count", 0) or 0)
    cost = 1.0 * errors + 0.6 * missing + 0.6 * perm
    goal_low = (goal or "").lower()
    if any(k in goal_low for k in ("create", "write", "generate", "new file", ".json", ".py", ".txt")):
        if added > 0:
            cost -= min(1.5, 0.3 * added)
    return max(0.0, cost)


def compute_scores(trace: List[dict], execution: dict) -> Dict[str, float]:
    # simple joint score
    t_score = min((len(trace) / 5.0 if trace else 0.0), 1.0)
    total = len(execution.get("completed", [])) + len(execution.get("failed", []))
    e_score = (len(execution.get("completed", [])) / total) if total else 0.0
    return {"trace_score": t_score, "exec_score": e_score, "overall": (t_score + e_score) / 2}
