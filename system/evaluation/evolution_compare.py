from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - optional dependency
    scipy_stats = None


DEFAULT_LOG = "artifacts/audit/evolution_log.jsonl"
DEFAULT_OUT = "artifacts/audit/evolution_ab_compare.json"
DEFAULT_SUMMARY = "artifacts/audit/evolution_ab_compare.md"


@dataclass
class MetricSpec:
    name: str
    path: str
    kind: str  # "continuous" | "binary"
    higher_is_better: bool = True


METRICS: List[MetricSpec] = [
    MetricSpec("compression_ratio", "gnn.compression_ratio", "continuous", True),
    MetricSpec("compression_ratio_weight", "gnn.compression_ratio_weight", "continuous", True),
    MetricSpec("avg_fitness", "abstraction.avg_fitness", "continuous", True),
    MetricSpec("structural_score", "abstraction.structural_score", "continuous", True),
    MetricSpec("planning_time_s", "planning_time_s", "continuous", False),
    MetricSpec("expanded_nodes", "expanded_nodes", "continuous", False),
    MetricSpec("exec_time_s", "exec_time_s", "continuous", False),
    MetricSpec("prediction_error", "prediction_error", "continuous", False),
    MetricSpec("success", "success", "binary", True),
]


def _load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def _get_path(data: Dict[str, Any], path: str) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _group_by_run(records: List[Dict[str, Any]], phase: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        if phase and rec.get("phase") != phase:
            continue
        run_id = rec.get("run_id") or "unknown"
        grouped.setdefault(str(run_id), []).append(rec)
    return grouped


def _latest_runs(records: List[Dict[str, Any]], count: int = 2) -> List[str]:
    latest: Dict[str, float] = {}
    for rec in records:
        rid = rec.get("run_id") or "unknown"
        ts = float(rec.get("ts") or 0.0)
        latest[str(rid)] = max(latest.get(str(rid), 0.0), ts)
    ordered = sorted(latest.items(), key=lambda x: x[1], reverse=True)
    return [rid for rid, _ in ordered[:count]]


def _continuous_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    vals = [float(v) for v in values]
    return {
        "n": len(vals),
        "mean": round(float(statistics.fmean(vals)), 6),
        "std": round(float(statistics.pstdev(vals)), 6) if len(vals) > 1 else 0.0,
        "median": round(float(statistics.median(vals)), 6),
        "min": round(float(min(vals)), 6),
        "max": round(float(max(vals)), 6),
    }


def _binary_stats(values: List[bool]) -> Dict[str, Any]:
    if not values:
        return {"n": 0}
    total = len(values)
    success = sum(1 for v in values if v)
    return {
        "n": total,
        "success": success,
        "failure": total - success,
        "rate": round(success / max(1, total), 6),
    }


def _cohen_d(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    ma = statistics.fmean(a)
    mb = statistics.fmean(b)
    va = statistics.pvariance(a)
    vb = statistics.pvariance(b)
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / max(1.0, (len(a) + len(b) - 2))
    if pooled <= 0:
        return None
    return round((ma - mb) / math.sqrt(pooled), 6)


def _welch_ttest(a: List[float], b: List[float]) -> Dict[str, Any]:
    if scipy_stats is None:
        return {"test": "welch_t", "p_value": None, "t": None, "df": None, "note": "scipy_unavailable"}
    if len(a) < 2 or len(b) < 2:
        return {"test": "welch_t", "p_value": None, "t": None, "df": None, "note": "insufficient_samples"}
    res = scipy_stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return {
        "test": "welch_t",
        "t": round(float(res.statistic), 6) if res.statistic is not None else None,
        "p_value": round(float(res.pvalue), 8) if res.pvalue is not None else None,
    }


def _mann_whitney(a: List[float], b: List[float]) -> Dict[str, Any]:
    if scipy_stats is None:
        return {"test": "mann_whitney_u", "p_value": None, "u": None, "note": "scipy_unavailable"}
    if not a or not b:
        return {"test": "mann_whitney_u", "p_value": None, "u": None, "note": "insufficient_samples"}
    res = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "test": "mann_whitney_u",
        "u": round(float(res.statistic), 6) if res.statistic is not None else None,
        "p_value": round(float(res.pvalue), 8) if res.pvalue is not None else None,
    }


def _fisher_exact(success_a: int, fail_a: int, success_b: int, fail_b: int) -> Dict[str, Any]:
    if scipy_stats is None:
        return {"test": "fisher_exact", "p_value": None, "odds_ratio": None, "note": "scipy_unavailable"}
    table = [[success_a, fail_a], [success_b, fail_b]]
    odds_ratio, p_value = scipy_stats.fisher_exact(table, alternative="two-sided")
    return {
        "test": "fisher_exact",
        "odds_ratio": round(float(odds_ratio), 6) if odds_ratio is not None else None,
        "p_value": round(float(p_value), 8) if p_value is not None else None,
    }


def _diff_summary(a_stats: Dict[str, Any], b_stats: Dict[str, Any], higher_is_better: bool) -> Dict[str, Any]:
    if not a_stats or not b_stats:
        return {}
    mean_a = a_stats.get("mean")
    mean_b = b_stats.get("mean")
    median_a = a_stats.get("median")
    median_b = b_stats.get("median")
    diff = {}
    if isinstance(mean_a, (int, float)) and isinstance(mean_b, (int, float)):
        diff["mean_diff"] = round(float(mean_a - mean_b), 6)
        if mean_b != 0:
            diff["mean_rel"] = round(float((mean_a - mean_b) / abs(mean_b)), 6)
    if isinstance(median_a, (int, float)) and isinstance(median_b, (int, float)):
        diff["median_diff"] = round(float(median_a - median_b), 6)
    if "mean_diff" in diff:
        win = "A" if diff["mean_diff"] > 0 else ("B" if diff["mean_diff"] < 0 else "tie")
        if not higher_is_better:
            win = "B" if win == "A" else ("A" if win == "B" else "tie")
        diff["winner"] = win
    return diff


def compare_runs(
    records: List[Dict[str, Any]],
    run_a: str,
    run_b: str,
    phase: Optional[str],
) -> Dict[str, Any]:
    grouped = _group_by_run(records, phase)
    a_records = grouped.get(run_a, [])
    b_records = grouped.get(run_b, [])
    result = {
        "run_a": run_a,
        "run_b": run_b,
        "phase": phase or "all",
        "count_a": len(a_records),
        "count_b": len(b_records),
        "metrics": {},
    }
    for spec in METRICS:
        a_vals: List[Any] = []
        b_vals: List[Any] = []
        for rec in a_records:
            val = _get_path(rec, spec.path)
            if val is None:
                continue
            a_vals.append(val)
        for rec in b_records:
            val = _get_path(rec, spec.path)
            if val is None:
                continue
            b_vals.append(val)
        metric_out: Dict[str, Any] = {"kind": spec.kind, "higher_is_better": spec.higher_is_better}
        if spec.kind == "binary":
            a_bool = [bool(v) for v in a_vals]
            b_bool = [bool(v) for v in b_vals]
            a_stats = _binary_stats(a_bool)
            b_stats = _binary_stats(b_bool)
            metric_out["a"] = a_stats
            metric_out["b"] = b_stats
            if a_stats.get("n", 0) > 0 and b_stats.get("n", 0) > 0:
                test = _fisher_exact(
                    int(a_stats.get("success", 0)),
                    int(a_stats.get("failure", 0)),
                    int(b_stats.get("success", 0)),
                    int(b_stats.get("failure", 0)),
                )
                metric_out["test"] = test
            result["metrics"][spec.name] = metric_out
            continue
        a_f = [float(v) for v in a_vals if isinstance(v, (int, float))]
        b_f = [float(v) for v in b_vals if isinstance(v, (int, float))]
        a_stats = _continuous_stats(a_f)
        b_stats = _continuous_stats(b_f)
        metric_out["a"] = a_stats
        metric_out["b"] = b_stats
        metric_out["diff"] = _diff_summary(a_stats, b_stats, spec.higher_is_better)
        if a_stats.get("n", 0) > 1 and b_stats.get("n", 0) > 1:
            metric_out["test"] = {
                "welch_t": _welch_ttest(a_f, b_f),
                "mann_whitney_u": _mann_whitney(a_f, b_f),
                "effect_size_d": _cohen_d(a_f, b_f),
            }
        result["metrics"][spec.name] = metric_out
    return result


def _render_summary(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Evolution A/B Compare")
    lines.append("")
    lines.append(f"- Run A: `{result.get('run_a')}`")
    lines.append(f"- Run B: `{result.get('run_b')}`")
    lines.append(f"- Phase: `{result.get('phase')}`")
    lines.append(f"- Samples A: {result.get('count_a')}")
    lines.append(f"- Samples B: {result.get('count_b')}")
    lines.append("")
    lines.append("## Metrics")
    for name, metric in (result.get("metrics") or {}).items():
        lines.append(f"### {name}")
        a_stats = metric.get("a", {})
        b_stats = metric.get("b", {})
        if metric.get("kind") == "binary":
            lines.append(f"- A rate: {a_stats.get('rate')} (n={a_stats.get('n')})")
            lines.append(f"- B rate: {b_stats.get('rate')} (n={b_stats.get('n')})")
            test = (metric.get("test") or {})
            p_val = test.get("p_value")
            if p_val is not None:
                lines.append(f"- Fisher p: {p_val}")
        else:
            lines.append(f"- A mean: {a_stats.get('mean')} (n={a_stats.get('n')})")
            lines.append(f"- B mean: {b_stats.get('mean')} (n={b_stats.get('n')})")
            diff = metric.get("diff") or {}
            if diff:
                lines.append(f"- Mean diff (A-B): {diff.get('mean_diff')} winner={diff.get('winner')}")
            test = (metric.get("test") or {}).get("welch_t") or {}
            p_val = test.get("p_value")
            if p_val is not None:
                lines.append(f"- Welch p: {p_val}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolution A/B compare with significance tests")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG)
    parser.add_argument("--run-a", type=str, default="")
    parser.add_argument("--run-b", type=str, default="")
    parser.add_argument("--phase", type=str, default="", help="chat/cycle or empty for all")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=str, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    records = _load_records(args.log)
    if not records:
        raise SystemExit(f"No records found at {args.log}")

    run_a = args.run_a.strip()
    run_b = args.run_b.strip()
    if not run_a or not run_b:
        runs = _latest_runs(records, 2)
        if len(runs) < 2:
            raise SystemExit("Not enough distinct run_id values for A/B comparison.")
        run_a, run_b = runs[0], runs[1]

    phase = args.phase.strip() or None
    result = compare_runs(records, run_a, run_b, phase)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_render_summary(result), encoding="utf-8")


if __name__ == "__main__":
    main()
