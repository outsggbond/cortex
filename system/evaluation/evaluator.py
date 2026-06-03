from __future__ import annotations

import json
import os
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


@dataclass
class EvolutionEvalConfig:
    output_dir: str = "audit"
    log_path: str = "artifacts/audit/evolution_log.jsonl"
    report_path: str = "artifacts/audit/evolution_report.json"
    plot_path: str = "artifacts/audit/evolution_plot.png"
    edge_weight_threshold: float = 0.0
    success_score_threshold: float = 0.5
    abstract_prefix: str = "abs::"
    run_id: str = ""


class EvolutionEvaluator:
    def __init__(
        self,
        gnn=None,
        semantic_refiner=None,
        world_model=None,
        config: Optional[EvolutionEvalConfig] = None,
    ) -> None:
        self.gnn = gnn
        self.semantic_refiner = semantic_refiner
        self.world_model = world_model
        self.config = config or EvolutionEvalConfig()
        self.run_id = self.config.run_id or os.environ.get("EVOLUTION_RUN_ID", "") or time.strftime("%Y%m%d_%H%M%S")
        self._baseline_edges: Optional[int] = None
        self._baseline_weight: Optional[float] = None
        self._records: List[Dict[str, Any]] = []

    def start_timing(self) -> Dict[str, float]:
        return {"wall_start": time.perf_counter(), "cpu_start": time.process_time()}

    def stop_timing(self, token: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not token:
            return {}
        wall = max(0.0, time.perf_counter() - float(token.get("wall_start", 0.0)))
        cpu = max(0.0, time.process_time() - float(token.get("cpu_start", 0.0)))
        return {"wall_time_s": round(wall, 6), "cpu_time_s": round(cpu, 6)}

    def compute_prediction_error(self, predicted, actual) -> Optional[float]:
        if predicted is None or actual is None:
            return None
        try:
            diffs = [
                len(getattr(predicted, "errors", [])) - len(getattr(actual, "errors", [])),
                len(getattr(predicted, "missing_paths", [])) - len(getattr(actual, "missing_paths", [])),
                len(getattr(predicted, "perm_paths", [])) - len(getattr(actual, "perm_paths", [])),
                int(getattr(predicted, "added_count", 0)) - int(getattr(actual, "added_count", 0)),
                int(getattr(predicted, "modified_count", 0)) - int(getattr(actual, "modified_count", 0)),
                int(getattr(predicted, "removed_count", 0)) - int(getattr(actual, "removed_count", 0)),
                int(getattr(predicted, "total_files", 0)) - int(getattr(actual, "total_files", 0)),
            ]
            mse = sum(float(d) ** 2 for d in diffs) / max(1.0, float(len(diffs)))
            return round(float(mse), 6)
        except Exception:
            return None

    def record_cycle(
        self,
        cycle_id: int,
        phase: str = "cycle",
        *,
        success: Optional[bool] = None,
        score: Optional[float] = None,
        planning_time_s: Optional[float] = None,
        expanded_nodes: Optional[int] = None,
        plan_steps: Optional[int] = None,
        replan_count: Optional[int] = None,
        exec_time_s: Optional[float] = None,
        prediction_error: Optional[float] = None,
        timing: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if success is None and score is not None:
            success = bool(score >= float(self.config.success_score_threshold))
        record: Dict[str, Any] = {
            "ts": time.time(),
            "cycle_id": int(cycle_id),
            "phase": str(phase),
            "run_id": self.run_id,
            "success": success,
            "score": score,
            "planning_time_s": planning_time_s,
            "expanded_nodes": expanded_nodes,
            "plan_steps": plan_steps,
            "replan_count": replan_count,
            "exec_time_s": exec_time_s,
            "prediction_error": prediction_error,
        }
        record["flags"] = {
            "gnn_enabled": bool(self.gnn is not None),
            "semantic_refiner": bool(self.semantic_refiner is not None),
        }
        record["gnn"] = self._gnn_stats()
        record["abstraction"] = self._abstraction_stats()
        record["resources"] = self._resource_stats(timing=timing)
        if extra:
            record["extra"] = extra
        self._append_record(record)
        self._records.append(record)
        return record

    def _append_record(self, record: Dict[str, Any]) -> None:
        path = Path(self.config.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _gnn_stats(self) -> Dict[str, Any]:
        if self.gnn is None:
            return {}
        try:
            co_counts = getattr(self.gnn, "co_counts", {}) or {}
        except Exception:
            return {}
        abs_prefix = self.config.abstract_prefix
        edge_thresh = float(self.config.edge_weight_threshold or 0.0)
        total_edges = 0
        base_edges = 0
        abs_edges = 0
        base_weight = 0.0
        abs_weight = 0.0
        for key, weight in co_counts.items():
            try:
                w = float(weight)
            except Exception:
                continue
            if w < edge_thresh:
                continue
            try:
                a, b = key.split("\t", 1)
            except Exception:
                continue
            total_edges += 1
            if a.startswith(abs_prefix) or b.startswith(abs_prefix):
                abs_edges += 1
                abs_weight += w
            else:
                base_edges += 1
                base_weight += w
        if self._baseline_edges is None:
            self._baseline_edges = base_edges
        if self._baseline_weight is None:
            self._baseline_weight = base_weight
        compression_ratio = None
        compression_ratio_w = None
        if base_edges > 0 and self._baseline_edges is not None:
            compression_ratio = round(float(self._baseline_edges) / float(base_edges), 6)
        if base_weight > 0 and self._baseline_weight is not None:
            compression_ratio_w = round(float(self._baseline_weight) / float(base_weight), 6)
        return {
            "total_edges": total_edges,
            "base_edges": base_edges,
            "abstract_edges": abs_edges,
            "base_weight": round(base_weight, 6),
            "abstract_weight": round(abs_weight, 6),
            "baseline_base_edges": self._baseline_edges,
            "baseline_base_weight": round(self._baseline_weight or 0.0, 6),
            "compression_ratio": compression_ratio,
            "compression_ratio_weight": compression_ratio_w,
            "abstract_edge_ratio": round(abs_edges / max(1, total_edges), 6),
        }

    def _abstraction_stats(self) -> Dict[str, Any]:
        refiner = self.semantic_refiner
        if refiner is None:
            return {}
        try:
            stats = refiner.stats()
            return stats or {}
        except Exception:
            return {}

    def _resource_stats(self, timing: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if timing:
            out.update(timing)
        if psutil is not None:
            try:
                proc = psutil.Process()
                mem = proc.memory_info()
                out["rss_mb"] = round(mem.rss / (1024 ** 2), 2)
                out["vms_mb"] = round(mem.vms / (1024 ** 2), 2)
                out["cpu_percent"] = round(proc.cpu_percent(interval=None), 2)
            except Exception:
                pass
        if torch is not None and torch.cuda.is_available():
            try:
                alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
                reserv = torch.cuda.memory_reserved(0) / (1024 ** 2)
                out["gpu_mem_alloc_mb"] = round(float(alloc), 2)
                out["gpu_mem_reserved_mb"] = round(float(reserv), 2)
            except Exception:
                pass
        return out

    def compute_metrics(self, records: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if records is None:
            records = self._load_records()
        if not records:
            return {"summary": {}, "series": {}, "by_phase": {}}
        records = sorted(records, key=lambda r: (r.get("cycle_id", 0), r.get("ts", 0.0)))
        series = self._build_series(records)
        summary = self._summarize_records(records)
        by_phase: Dict[str, Any] = {}
        phases = {}
        for rec in records:
            phase = rec.get("phase", "cycle")
            phases.setdefault(phase, []).append(rec)
        for phase, recs in phases.items():
            by_phase[phase] = self._summarize_records(recs)
        return {"summary": summary, "series": series, "by_phase": by_phase}

    def export_report(self) -> Dict[str, Any]:
        report = self.compute_metrics()
        path = Path(self.config.report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self._export_plot(report)
        return report

    def _load_records(self) -> List[Dict[str, Any]]:
        path = Path(self.config.log_path)
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        records.append(rec)
                except Exception:
                    continue
        except Exception:
            return []
        return records

    def _summarize_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "count": len(records),
            "success_rate": None,
            "avg_planning_time_s": None,
            "avg_expanded_nodes": None,
            "avg_exec_time_s": None,
            "avg_prediction_error": None,
            "latest_structural": {},
        }
        if not records:
            return summary
        successes = [r.get("success") for r in records if r.get("success") is not None]
        if successes:
            summary["success_rate"] = round(sum(1 for s in successes if s) / len(successes), 6)
        plan_times = [r.get("planning_time_s") for r in records if isinstance(r.get("planning_time_s"), (int, float))]
        if plan_times:
            summary["avg_planning_time_s"] = round(sum(plan_times) / len(plan_times), 6)
        expanded = [r.get("expanded_nodes") for r in records if isinstance(r.get("expanded_nodes"), int)]
        if expanded:
            summary["avg_expanded_nodes"] = round(sum(expanded) / len(expanded), 3)
        exec_times = [r.get("exec_time_s") for r in records if isinstance(r.get("exec_time_s"), (int, float))]
        if exec_times:
            summary["avg_exec_time_s"] = round(sum(exec_times) / len(exec_times), 6)
        pred_errs = [r.get("prediction_error") for r in records if isinstance(r.get("prediction_error"), (int, float))]
        if pred_errs:
            summary["avg_prediction_error"] = round(sum(pred_errs) / len(pred_errs), 6)
        latest = records[-1]
        summary["latest_structural"] = {
            "compression_ratio": self._get_path(latest, "gnn.compression_ratio"),
            "compression_ratio_weight": self._get_path(latest, "gnn.compression_ratio_weight"),
            "survival_rate": self._get_path(latest, "abstraction.survival_rate"),
            "avg_fitness": self._get_path(latest, "abstraction.avg_fitness"),
            "structural_score": self._get_path(latest, "abstraction.structural_score"),
        }
        return summary

    def _build_series(self, records: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        series: Dict[str, List[Any]] = {
            "cycle_id": [],
            "compression_ratio": [],
            "avg_fitness": [],
            "structural_score": [],
            "planning_time_s": [],
            "expanded_nodes": [],
            "exec_time_s": [],
            "prediction_error": [],
            "success_rate": [],
        }
        success_count = 0
        success_total = 0
        for rec in records:
            series["cycle_id"].append(rec.get("cycle_id"))
            series["compression_ratio"].append(self._get_path(rec, "gnn.compression_ratio"))
            series["avg_fitness"].append(self._get_path(rec, "abstraction.avg_fitness"))
            series["structural_score"].append(self._get_path(rec, "abstraction.structural_score"))
            series["planning_time_s"].append(rec.get("planning_time_s"))
            series["expanded_nodes"].append(rec.get("expanded_nodes"))
            series["exec_time_s"].append(rec.get("exec_time_s"))
            series["prediction_error"].append(rec.get("prediction_error"))
            if rec.get("success") is not None:
                success_total += 1
                if rec.get("success"):
                    success_count += 1
                series["success_rate"].append(round(success_count / max(1, success_total), 6))
            else:
                series["success_rate"].append(None)
        return series

    def _export_plot(self, report: Dict[str, Any]) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        series = report.get("series") or {}
        cycles = series.get("cycle_id") or []
        if not cycles:
            return
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        ax1, ax2, ax3, ax4 = axes.flatten()
        self._plot_line(ax1, cycles, series.get("compression_ratio"), "Compression Ratio")
        self._plot_line(ax1, cycles, series.get("structural_score"), "Structural Score", color="tab:orange")
        ax1.legend(loc="best")
        self._plot_line(ax2, cycles, series.get("avg_fitness"), "Avg Fitness")
        self._plot_line(ax3, cycles, series.get("planning_time_s"), "Planning Time (s)")
        self._plot_line(ax3, cycles, series.get("expanded_nodes"), "Expanded Nodes", color="tab:orange")
        ax3.legend(loc="best")
        self._plot_line(ax4, cycles, series.get("success_rate"), "Success Rate")
        fig.tight_layout()
        path = Path(self.config.plot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.as_posix(), dpi=120)
        plt.close(fig)

    @staticmethod
    def _plot_line(ax, xs, ys, label: str, color: Optional[str] = None) -> None:
        if not xs or ys is None:
            return
        filtered_x: List[Any] = []
        filtered_y: List[Any] = []
        for x, y in zip(xs, ys):
            if y is None or (isinstance(y, float) and (math.isnan(y) or math.isinf(y))):
                continue
            filtered_x.append(x)
            filtered_y.append(y)
        if not filtered_x:
            return
        if color:
            ax.plot(filtered_x, filtered_y, label=label, color=color)
        else:
            ax.plot(filtered_x, filtered_y, label=label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _get_path(data: Dict[str, Any], path: str) -> Any:
        cur: Any = data
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur
