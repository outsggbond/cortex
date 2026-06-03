from __future__ import annotations

import json
import time
import os
from pathlib import Path
from typing import Optional

from system.chat_v2.intent import parse_intent, Intent


class EvolutionEngine:
    def __init__(
        self,
        base_dir: str = "evolution",
        registry_path: str = "modules/registry.json",
        experience_path: str = "artifacts/memory/experience_store.jsonl",
        costs_path: str = "config/costs.yaml",
        nanobrain_path: str = "config/nanobrain.json",
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry = self.base_dir / "proposals.json"
        self.last_run_path = self.base_dir / "last_run.json"
        self.module_registry = Path(registry_path)
        self.experience_path = Path(experience_path)
        self.costs_path = Path(costs_path)
        self.nanobrain_path = Path(nanobrain_path)
        try:
            from system.core.planner import TaskPlanner

            self._planner = TaskPlanner()
        except Exception:
            self._planner = None
        if not self.module_registry.exists():
            self.module_registry.parent.mkdir(parents=True, exist_ok=True)
            self.module_registry.write_text("[]", encoding="utf-8")
        if not self.registry.exists():
            self.registry.write_text("[]", encoding="utf-8")

    def handle_message(self, text: str) -> Optional[str]:
        intent = parse_intent(text)
        if not intent:
            return None
        if intent.type == "add_module":
            return self._add_module(intent)
        if intent.type == "modify_code":
            return self._record_proposal(intent)
        if intent.type == "evolve":
            summary = self.auto_evolve()
            if summary:
                return summary
            return "\u6682\u65e0\u53ef\u7528\u7684\u5931\u8d25\u6837\u672c\uff0c\u672a\u89e6\u53d1\u53c2\u6570\u8c03\u6574\u3002"
        return None

    def auto_evolve(self) -> Optional[str]:
        if not self._should_run():
            return None
        window = int(os.environ.get("EVOLVE_WINDOW", "50"))
        min_fails = int(os.environ.get("EVOLVE_MIN_FAILS", "3"))
        stats = self.analyze_failures(window=window)
        if stats.get("failures", 0) < min_fails:
            return None
        changes = []
        cost_change = self._patch_costs(stats)
        if cost_change:
            changes.append({"costs": cost_change})
        nanobrain_change = self._patch_nanobrain(stats)
        if nanobrain_change:
            changes.append({"nanobrain": nanobrain_change})
        if not changes:
            return None
        self._append_registry(
            {
                "type": "auto_evolve",
                "detail": {
                    "stats": stats,
                    "changes": changes,
                },
            }
        )
        self._record_last_run()
        return "\u5df2\u57fa\u4e8e\u5931\u8d25\u7ecf\u9a8c\u5fae\u8c03\u7b56\u7565\u53c2\u6570\u3002"

    def analyze_failures(self, window: int = 50) -> dict:
        items = self._load_recent_items(window)
        failures = [x for x in items if not x.get("success", True)]
        total = len(items)
        fail_count = len(failures)
        action_counts: dict[str, int] = {}
        perm_hits = 0
        missing_hits = 0
        error_hits = 0
        run_fail = 0
        move_fail = 0
        delete_fail = 0
        for item in failures:
            steps = item.get("plan_steps") or []
            names = []
            for s in steps:
                name = self._parse_action_name(str(s))
                if name:
                    names.append(name)
                    action_counts[name] = action_counts.get(name, 0) + 1
            after_state = item.get("after_state") if isinstance(item.get("after_state"), dict) else {}
            if after_state.get("perm_paths"):
                perm_hits += 1
            if after_state.get("missing_paths"):
                missing_hits += 1
            if after_state.get("errors"):
                error_hits += 1
            if "run_script" in names:
                run_fail += 1
            if "move_file" in names:
                move_fail += 1
            if "delete_file" in names:
                delete_fail += 1
        fail_ratio = (fail_count / total) if total else 0.0
        return {
            "total": total,
            "failures": fail_count,
            "fail_ratio": fail_ratio,
            "perm_ratio": (perm_hits / fail_count) if fail_count else 0.0,
            "missing_ratio": (missing_hits / fail_count) if fail_count else 0.0,
            "error_ratio": (error_hits / fail_count) if fail_count else 0.0,
            "run_fail_ratio": (run_fail / fail_count) if fail_count else 0.0,
            "move_fail_ratio": (move_fail / fail_count) if fail_count else 0.0,
            "delete_fail_ratio": (delete_fail / fail_count) if fail_count else 0.0,
            "action_counts": action_counts,
        }

    def _parse_action_name(self, step: str) -> str:
        if not step:
            return ""
        if self._planner is not None:
            try:
                task = self._planner._parse_task(step)
                return task.name or ""
            except Exception:
                return ""
        return step.strip().split(" ", 1)[0].lower()

    def _load_recent_items(self, window: int) -> list[dict]:
        if not self.experience_path.exists():
            return []
        try:
            lines = self.experience_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        if window > 0:
            lines = lines[-window:]
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def _patch_costs(self, stats: dict) -> dict:
        try:
            import yaml  # type: ignore
        except Exception:
            return {}
        data = {}
        if self.costs_path.exists():
            try:
                data = yaml.safe_load(self.costs_path.read_text(encoding="utf-8")) or {}
            except Exception:
                data = {}
        if not isinstance(data, dict):
            data = {}
        default = data.get("default") if isinstance(data.get("default"), dict) else {}
        action_base = default.get("action_base") if isinstance(default.get("action_base"), dict) else {}
        write_audit = float(default.get("write_audit_cost", 0.0) or 0.0)

        def _clamp(val: float, low: float, high: float) -> float:
            return max(low, min(high, val))

        def _adjust(name: str, factor: float, low: float, high: float) -> Optional[dict]:
            from system.l_utils.metrics import base_action_cost

            base = float(action_base.get(name, base_action_cost(name)))
            new_val = _clamp(base * factor, low, high)
            if abs(new_val - base) < 1e-6:
                return None
            action_base[name] = round(new_val, 3)
            return {"action": name, "from": base, "to": new_val}

        changes = []
        if stats.get("perm_ratio", 0.0) >= 0.3:
            change = _adjust("chmod", 0.8, 0.2, 3.0)
            if change:
                changes.append(change)
        if stats.get("missing_ratio", 0.0) >= 0.3:
            change = _adjust("mkdir", 0.85, 0.4, 4.0)
            if change:
                changes.append(change)
        if stats.get("run_fail_ratio", 0.0) >= 0.2:
            change = _adjust("run_script", 1.15, 6.0, 20.0)
            if change:
                changes.append(change)
        if stats.get("move_fail_ratio", 0.0) >= 0.2:
            change = _adjust("move_file", 1.1, 1.0, 6.0)
            if change:
                changes.append(change)
        if stats.get("delete_fail_ratio", 0.0) >= 0.2:
            change = _adjust("delete_file", 1.1, 1.0, 8.0)
            if change:
                changes.append(change)
        if stats.get("fail_ratio", 0.0) >= 0.5:
            new_audit = _clamp(write_audit + 0.2, 0.0, 5.0)
            if abs(new_audit - write_audit) > 1e-6:
                default["write_audit_cost"] = round(new_audit, 3)
                changes.append({"write_audit_cost": {"from": write_audit, "to": new_audit}})

        if not changes:
            return {}
        default["action_base"] = action_base
        data["default"] = default
        self.costs_path.parent.mkdir(parents=True, exist_ok=True)
        self.costs_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return {"changes": changes}

    def _patch_nanobrain(self, stats: dict) -> dict:
        if not self.nanobrain_path.exists():
            return {}
        try:
            data = json.loads(self.nanobrain_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        threshold = float(data.get("threshold", 0.08))
        tau = float(data.get("tau", 0.04))
        updated = {}
        fail_ratio = float(stats.get("fail_ratio", 0.0))
        if fail_ratio >= 0.5:
            new_threshold = min(0.9, threshold + 0.02)
            new_tau = min(0.9, tau + 0.01)
            if new_threshold != threshold:
                data["threshold"] = round(new_threshold, 3)
                updated["threshold"] = {"from": threshold, "to": new_threshold}
            if new_tau != tau:
                data["tau"] = round(new_tau, 3)
                updated["tau"] = {"from": tau, "to": new_tau}
        elif fail_ratio > 0 and fail_ratio <= 0.15:
            new_threshold = max(0.02, threshold - 0.01)
            new_tau = max(0.01, tau - 0.005)
            if new_threshold != threshold:
                data["threshold"] = round(new_threshold, 3)
                updated["threshold"] = {"from": threshold, "to": new_threshold}
            if new_tau != tau:
                data["tau"] = round(new_tau, 3)
                updated["tau"] = {"from": tau, "to": new_tau}
        if not updated:
            return {}
        self.nanobrain_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        return updated

    def _should_run(self) -> bool:
        interval = float(os.environ.get("EVOLVE_INTERVAL", "300"))
        if interval <= 0:
            return True
        last = 0.0
        if self.last_run_path.exists():
            try:
                data = json.loads(self.last_run_path.read_text(encoding="utf-8"))
                last = float(data.get("ts", 0.0))
            except Exception:
                last = 0.0
        return (time.time() - last) >= interval

    def _record_last_run(self) -> None:
        try:
            self.last_run_path.write_text(json.dumps({"ts": time.time()}), encoding="utf-8")
        except Exception:
            pass

    def _add_module(self, intent: Intent) -> str:
        name = intent.name.strip()
        safe = "".join([c for c in name if c.isalnum() or c == "_"]).lower()
        if not safe:
            safe = "custom_feature"
        mod_name = f"{safe}_{int(time.time())}"
        mod_path = Path("modules") / f"{mod_name}.py"
        if not mod_path.parent.exists():
            mod_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "def run(context: dict) -> dict:\n"
            "    \"\"\"Custom module. Update logic as needed.\"\"\"\n"
            "    return {\"status\": \"ok\", \"context\": context}\n"
        )
        mod_path.write_text(content, encoding="utf-8")
        self._register_module(mod_path)
        self._append_registry(
            {
                "type": "add_module",
                "module": str(mod_path),
                "detail": intent.detail,
            }
        )
        return f"\u5df2\u521b\u5efa\u6a21\u5757\uff1a{mod_path}"

    def _record_proposal(self, intent: Intent) -> str:
        self._append_registry(
            {
                "type": intent.type,
                "detail": intent.detail,
            }
        )
        return "\u5df2\u8bb0\u5f55\u6f14\u8fdb\u610f\u56fe\uff0c\u751f\u6210\u8ba1\u5212\u8349\u6848\u3002"

    def _append_registry(self, item: dict) -> None:
        data = json.loads(self.registry.read_text(encoding="utf-8"))
        data.append(item)
        self.registry.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _register_module(self, mod_path: Path) -> None:
        data = json.loads(self.module_registry.read_text(encoding="utf-8"))
        data.append({"path": str(mod_path)})
        self.module_registry.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
