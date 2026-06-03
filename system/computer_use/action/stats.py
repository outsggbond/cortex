from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from system.core.planner import TaskPlanner
from system.core.planner import Task
from system.brain.world_model import WorldState
from system.l_utils.metrics import action_cost


logger = logging.getLogger(__name__)


@dataclass
class BetaStat:
    a: float = 1.0
    b: float = 1.0

    def mean(self) -> float:
        return self.a / max(1e-8, self.a + self.b)

    def confidence(self) -> float:
        # Confidence grows with samples; 0..1
        return min(1.0, (self.a + self.b) / (self.a + self.b + 10.0))


@dataclass
class DurationStat:
    mean_ms: float = 0.0
    count: int = 0

    def update(self, duration_ms: float, alpha: float = 0.2) -> None:
        if duration_ms <= 0:
            return
        if self.count == 0:
            self.mean_ms = float(duration_ms)
        else:
            self.mean_ms = (1.0 - alpha) * self.mean_ms + alpha * float(duration_ms)
        self.count += 1


@dataclass
class LogisticRegressor:
    weights: List[float]
    lr: float = 0.2
    l2: float = 0.01
    l1: float = 0.001
    min_count: int = 8
    count: int = 0

    def predict(self, x: List[float]) -> float:
        z = 0.0
        for w, v in zip(self.weights, x):
            z += w * v
        # sigmoid
        if z >= 0:
            ez = 1.0 / (1.0 + (2.71828 ** (-z)))
        else:
            ez = (2.71828 ** z) / (1.0 + (2.71828 ** z))
        return ez

    def update(self, x: List[float], y: float) -> None:
        pred = self.predict(x)
        err = y - pred
        for i in range(len(self.weights)):
            # SGD with L2
            grad = err * x[i] - self.l2 * self.weights[i]
            # L1 soft-threshold for sparsity
            if self.weights[i] > 0:
                grad -= self.l1
            elif self.weights[i] < 0:
                grad += self.l1
            self.weights[i] += self.lr * grad
        self.count += 1

    def confidence(self) -> float:
        return min(1.0, self.count / max(1.0, float(self.min_count * 4)))


class ActionStats:
    def __init__(self, path: str = "artifacts/memory/action_stats.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stats: Dict[str, BetaStat] = {}
        self.bucket_stats: Dict[str, Dict[str, BetaStat]] = {}
        self.duration_stats: Dict[str, DurationStat] = {}
        self.regressors: Dict[str, LogisticRegressor] = {}
        self.module_policy_path = Path("artifacts/audit/module_policy.json")
        self.cursor_path = self.path.with_suffix(".cursor")
        self.exec_log = Path("artifacts/audit/exec_log.jsonl")
        self._planner = TaskPlanner()
        self._last_pos = 0
        self._load()

    def record(self, action_name: str, success: bool, buckets: Optional[List[str]] = None) -> None:
        if not action_name:
            return
        stat = self.stats.get(action_name)
        if stat is None:
            stat = BetaStat()
            self.stats[action_name] = stat
        if success:
            stat.a += 1.0
        else:
            stat.b += 1.0
        if buckets:
            bmap = self.bucket_stats.get(action_name)
            if bmap is None:
                bmap = {}
                self.bucket_stats[action_name] = bmap
            for b in buckets:
                if not b:
                    continue
                bst = bmap.get(b)
                if bst is None:
                    bst = BetaStat()
                    bmap[b] = bst
                if success:
                    bst.a += 1.0
                else:
                    bst.b += 1.0

    def estimate(self, action_name: str, buckets: Optional[List[str]] = None) -> Tuple[float, float, int]:
        stat = self.stats.get(action_name)
        if not stat:
            return 0.6, 0.0, 0
        mean = stat.mean()
        conf = stat.confidence()
        samples = int(stat.a + stat.b - 2.0)
        # bucket blending
        if buckets:
            bmap = self.bucket_stats.get(action_name, {})
            for b in buckets:
                bst = bmap.get(b)
                if not bst:
                    continue
                bmean = bst.mean()
                bconf = bst.confidence()
                if bconf <= 0:
                    continue
                total = conf + bconf
                mean = (mean * conf + bmean * bconf) / max(1e-6, total)
                conf = min(1.0, total)
                samples += int(bst.a + bst.b - 2.0)
        return mean, conf, max(0, samples)

    def update_duration(self, action_name: str, duration_ms: float) -> None:
        if not action_name:
            return
        stat = self.duration_stats.get(action_name)
        if stat is None:
            stat = DurationStat()
            self.duration_stats[action_name] = stat
        stat.update(duration_ms)

    def duration_ms(self, action_name: str) -> float:
        stat = self.duration_stats.get(action_name)
        if not stat:
            return 0.0
        return float(stat.mean_ms)

    def get_action_cost(
        self, action_name: str, success_prob: float | None = None, destructive: bool = False, policy=None
    ) -> float:
        if not action_name:
            return 0.0
        if success_prob is None:
            mean, conf, _ = self.estimate(action_name)
            if conf > 0:
                success_prob = mean
        return action_cost(
            action_name,
            duration_ms=self.duration_ms(action_name),
            success_prob=success_prob,
            destructive=destructive,
            policy=policy,
        )

    def update_regressor(
        self,
        action_name: str,
        error_type: str,
        features: List[float],
        success: bool,
        group: str = "",
    ) -> None:
        if not action_name:
            return
        key = self._reg_key(action_name, error_type, group)
        reg = self.regressors.get(key)
        if reg is None or len(reg.weights) != len(features):
            reg = LogisticRegressor(weights=[0.0 for _ in features])
            self.regressors[key] = reg
        y = 1.0 if success else 0.0
        reg.update(features, y)

    def predict_regressor(
        self, action_name: str, error_type: str, features: List[float], group: str = ""
    ) -> Tuple[float, float]:
        key = self._reg_key(action_name, error_type, group)
        reg = self.regressors.get(key)
        if not reg:
            return 0.0, 0.0
        if reg.count < reg.min_count or len(reg.weights) != len(features):
            return 0.0, 0.0
        return reg.predict(features), reg.confidence()

    def update_from_exec_log(self) -> int:
        if not self.exec_log.exists():
            return 0
        try:
            if self._last_pos == 0 and self.cursor_path.exists():
                self._last_pos = int(self.cursor_path.read_text(encoding="utf-8") or "0")
        except Exception:
            self._last_pos = 0
        count = 0
        try:
            file_size = self.exec_log.stat().st_size
            if self._last_pos > file_size:
                # log rotated/truncated
                self._last_pos = 0
            with self.exec_log.open("r", encoding="utf-8") as f:
                f.seek(self._last_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    step = (data.get("step") or "").strip()
                    duration_ms = float(data.get("duration_ms") or 0.0)
                    if not step:
                        continue
                    task = self._planner._parse_task(step)
                    self.update_duration(task.name, duration_ms)
                    count += 1
                self._last_pos = f.tell()
            self.cursor_path.write_text(str(self._last_pos), encoding="utf-8")
        except Exception:
            return count
        return count

    def build_features(self, task: Task, state: WorldState, error_type: str) -> List[float]:
        payload = task.payload or {}
        path = payload.get("path") or payload.get("src") or payload.get("dst") or ""
        path = str(path).replace("\\", "/")
        depth = 0
        if path:
            depth = len([p for p in path.split("/") if p])
        is_py = 1.0 if path.endswith(".py") else 0.0
        is_dir = 1.0 if (path.endswith("/") or "." not in path.rsplit("/", 1)[-1]) else 0.0
        err_count = min(1.0, len(state.errors) / 5.0)
        miss_count = min(1.0, len(state.missing_paths) / 5.0)
        perm_count = min(1.0, len(state.perm_paths) / 5.0)
        err_flag = 1.0 if error_type else 0.0
        depth_norm = min(1.0, depth / 8.0)
        return [1.0, depth_norm, is_py, is_dir, err_count, miss_count, perm_count, err_flag]

    def action_group(self, task: Task) -> str:
        payload = task.payload or {}
        path = payload.get("path") or payload.get("src") or payload.get("dst") or ""
        if not path:
            return "none"
        path = str(path).replace("\\", "/")
        parts = [p for p in path.split("/") if p]
        if not parts:
            return "none"
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return parts[0]

    def save(self) -> None:
        data = {
            "stats": {k: {"a": v.a, "b": v.b} for k, v in self.stats.items()},
            "buckets": {
                k: {bk: {"a": bv.a, "b": bv.b} for bk, bv in bmap.items()}
                for k, bmap in self.bucket_stats.items()
            },
            "durations": {k: {"mean_ms": v.mean_ms, "count": v.count} for k, v in self.duration_stats.items()},
            "regressors": {
                k: {"weights": v.weights, "lr": v.lr, "l2": v.l2, "count": v.count}
                for k, v in self.regressors.items()
            },
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        # Also save module policy summary.
        try:
            policy = self._build_module_policy()
            self.module_policy_path.parent.mkdir(parents=True, exist_ok=True)
            self.module_policy_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("action_stats: module policy save failed", exc_info=True)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if "stats" in (data or {}):
                stats = (data or {}).get("stats", {})
            else:
                # backward-compatible: top-level action stats
                stats = {k: v for k, v in (data or {}).items() if isinstance(v, dict) and "a" in v and "b" in v}
            buckets = (data or {}).get("buckets", {})
            durs = (data or {}).get("durations", {})
            for k, v in stats.items():
                a = float(v.get("a", 1.0))
                b = float(v.get("b", 1.0))
                self.stats[k] = BetaStat(a=a, b=b)
            for k, bmap in buckets.items():
                for bk, bv in (bmap or {}).items():
                    a = float(bv.get("a", 1.0))
                    b = float(bv.get("b", 1.0))
                    self.bucket_stats.setdefault(k, {})[bk] = BetaStat(a=a, b=b)
            for k, v in durs.items():
                ms = float(v.get("mean_ms", 0.0))
                cnt = int(v.get("count", 0))
                self.duration_stats[k] = DurationStat(mean_ms=ms, count=cnt)
            for k, v in (data or {}).get("regressors", {}).items():
                weights = [float(x) for x in v.get("weights", [])]
                if not weights:
                    continue
                lr = float(v.get("lr", 0.2))
                l2 = float(v.get("l2", 0.01))
                cnt = int(v.get("count", 0))
                self.regressors[k] = LogisticRegressor(weights=weights, lr=lr, l2=l2, count=cnt)
        except Exception:
            return

    @staticmethod
    def _reg_key(action_name: str, error_type: str, group: str = "") -> str:
        et = (error_type or "none").strip().lower()
        grp = (group or "none").strip().lower()
        return f"{action_name}:{et}:{grp}"

    def _build_module_policy(self) -> dict:
        # Summarize success rates by top-level module/path group
        module_stats: Dict[str, Dict[str, int]] = {}
        for key, reg in self.regressors.items():
            try:
                _action, _err, group = key.split(":", 2)
            except Exception:
                continue
            if not group or group == "none":
                continue
            mod = group.split("/", 1)[0]
            d = module_stats.setdefault(mod, {"count": 0})
            d["count"] += reg.count
        # Convert to simple suggestions
        suggestions = []
        for mod, d in module_stats.items():
            if d["count"] < 10:
                continue
            suggestions.append(
                {
                    "module": mod,
                    "sample_count": d["count"],
                    "hint": f"{mod}: consider prioritizing actions with higher historical success.",
                }
            )
        return {"generated_at": time.time(), "suggestions": suggestions}
