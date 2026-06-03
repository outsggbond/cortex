from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict


def summarize_trace(log_path: str = "artifacts/audit/trace_log.jsonl", out_path: str = "artifacts/audit/trace_summary.json") -> Dict:
    lp = Path(log_path)
    if not lp.exists():
        return {}
    rule_counts = Counter()
    total = 0
    for line in lp.read_text(encoding="utf-8").splitlines():
        total += 1
        try:
            rec = json.loads(line)
            for t in rec.get("trace", []):
                rule_counts[t.get("rule", "unknown")] += 1
        except Exception:
            continue
    summary = {"total_records": total, "rule_counts": dict(rule_counts)}
    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize_rule_weights(log_path: str = "artifacts/audit/rule_weight.jsonl", out_path: str = "artifacts/audit/rule_weight_summary.json") -> Dict:
    lp = Path(log_path)
    if not lp.exists():
        return {}
    last = None
    for line in lp.read_text(encoding="utf-8").splitlines():
        try:
            last = json.loads(line)
        except Exception:
            continue
    if not last:
        return {}
    summary = {"rules": last.get("rules", [])}
    Path(out_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize_rule_weight_series(
    log_path: str = "artifacts/audit/rule_weight.jsonl", out_path: str = "artifacts/audit/rule_weight_series.json"
) -> Dict:
    lp = Path(log_path)
    if not lp.exists():
        return {}
    series = {}
    for line in lp.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ts = rec.get("ts")
        for r in rec.get("rules", []):
            name = r.get("name")
            w = r.get("weight")
            if name is None or w is None:
                continue
            series.setdefault(name, []).append({"ts": ts, "weight": w})
    Path(out_path).write_text(json.dumps(series, ensure_ascii=False, indent=2), encoding="utf-8")
    return series


def plot_rule_weight_series(
    series_path: str = "artifacts/audit/rule_weight_series.json", out_path: str = "artifacts/audit/rule_weight_series.png"
) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    sp = Path(series_path)
    if not sp.exists():
        return ""
    data = json.loads(sp.read_text(encoding="utf-8"))
    plt.figure(figsize=(8, 4))
    for name, points in data.items():
        xs = [p.get("ts") for p in points]
        ys = [p.get("weight") for p in points]
        if xs and ys:
            plt.plot(xs, ys, label=name)
    plt.legend(loc="best", fontsize=6)
    plt.title("Rule Weight Evolution")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path


def report_dag_rule_scores(
    plan_path: str = "artifacts/audit/plan_graph.png",
    rule_series_path: str = "artifacts/audit/rule_weight_series.json",
    out_path: str = "artifacts/audit/dag_rule_report.json",
) -> str:
    # lightweight report linking DAG and rule stats
    data = {"plan_graph": plan_path, "rule_series": rule_series_path}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    summarize_trace()
    summarize_rule_weights()
    summarize_rule_weight_series()
    plot_rule_weight_series()
    report_dag_rule_scores()
