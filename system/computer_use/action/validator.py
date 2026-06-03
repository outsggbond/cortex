from __future__ import annotations

import os
from typing import List, Tuple

from system.l_utils.files import FileManager
from system.core.planner import Task
from system.brain.world_model import WorldState
from system.computer_use.action.stats import ActionStats
import json
from pathlib import Path


class ActionValidator:
    def __init__(self, project_root: str = ".", stats: ActionStats | None = None):
        self.fm = FileManager(project_root)
        self.allow_read_missing = os.environ.get("ALLOW_READ_MISSING", "0") == "1"
        self.stats = stats
        self.priority_cfg = {}
        self.priority_path = os.environ.get("ACTION_PRIORITY_PATH", "config/action_priority.json")
        self.priority_mtime = 0.0
        self._last_good_cfg = {}
        self._ensure_priority_config()

    def is_valid(self, task: Task, state: WorldState) -> Tuple[bool, str]:
        name = task.name
        payload = task.payload or {}
        try:
            if name in {"read_file", "append_file", "touch_file", "write_file"}:
                path = payload.get("path", "")
                if not path:
                    return False, "missing path"
                self.fm._resolve(path)
                if name == "read_file" and not self.fm.exists(path):
                    if self.allow_read_missing:
                        return True, "allow read missing"
                    return False, "read missing file"
                return True, "ok"
            if name == "run_script":
                path = payload.get("path", "")
                if not path:
                    return False, "missing path"
                self.fm._resolve(path)
                if not self.fm.exists(path):
                    return False, "script not found"
                if not str(path).lower().endswith(".py"):
                    return False, "only .py script"
                return True, "ok"
            if name in {"list_dir", "search_text", "search_files"}:
                path = payload.get("path", ".")
                self.fm._resolve(path)
                if not self.fm.exists(path):
                    return False, "path not found"
                return True, "ok"
            if name == "check_exists":
                path = payload.get("path", "")
                if not path:
                    return False, "missing path"
                self.fm._resolve(path)
                return True, "ok"
            if name == "mkdir":
                path = payload.get("path", "")
                if not path:
                    return False, "missing path"
                self.fm._resolve(path)
                return True, "ok"
            if name == "copy_file" or name == "move_file":
                src = payload.get("src", "")
                dst = payload.get("dst", "")
                if not src or not dst:
                    return False, "missing src/dst"
                self.fm._resolve(src)
                self.fm._resolve(dst)
                if not self.fm.exists(src):
                    return False, "src not found"
                return True, "ok"
            if name == "delete_file":
                path = payload.get("path", "")
                if not path:
                    return False, "missing path"
                self.fm._resolve(path)
                if not self.fm.exists(path):
                    return False, "path not found"
                return True, "ok"
            if name == "chmod":
                path = payload.get("path", "")
                if not path:
                    return False, "missing path"
                self.fm._resolve(path)
                if not self.fm.exists(path):
                    return False, "path not found"
                return True, "ok"
            if name in {"run_tests", "generate_code", "rollback"}:
                return True, "ok"
            if name.startswith("plugin:browser_open_url"):
                url = str(payload.get("url", "") or "").strip()
                if not url:
                    return False, "missing url"
                if not (url.startswith("http://") or url.startswith("https://")):
                    return False, "invalid url"
                return True, "ok"
            if name.startswith("plugin:browser_dom_open"):
                url = str(payload.get("url", "") or "").strip()
                if not url:
                    return False, "missing url"
                if not (url.startswith("http://") or url.startswith("https://")):
                    return False, "invalid url"
                return True, "ok"
            if name.startswith("plugin:browser_dom_click") or name.startswith("plugin:browser_dom_extract_text"):
                selector = str(payload.get("selector", "") or payload.get("query", "")).strip()
                if not selector:
                    return False, "missing selector"
                return True, "ok"
            if name.startswith("plugin:browser_dom_type"):
                selector = str(payload.get("selector", "") or "").strip()
                text = str(payload.get("text", "") or "")
                if not selector:
                    return False, "missing selector"
                if not text:
                    return False, "missing text"
                return True, "ok"
            if name.startswith("plugin:browser_dom_wait_text"):
                text = str(payload.get("text", "") or payload.get("query", "")).strip()
                if not text:
                    return False, "missing wait text"
                return True, "ok"
            if name.startswith("plugin:browser_dom_screenshot") or name.startswith("plugin:browser_dom_close"):
                return True, "ok"
            if name.startswith("plugin:desktop_launch"):
                target = str(payload.get("target", "") or payload.get("path", "") or payload.get("command", "")).strip()
                if not target:
                    return False, "missing launch target"
                return True, "ok"
            if name.startswith("plugin:desktop_focus_window"):
                title = str(payload.get("title", "") or payload.get("window", "")).strip()
                if not title:
                    return False, "missing window title"
                return True, "ok"
            if name.startswith("plugin:desktop_list_controls"):
                title = str(payload.get("title", "") or payload.get("window", "")).strip()
                if not title:
                    return False, "missing window title"
                return True, "ok"
            if name.startswith("plugin:desktop_click_control"):
                title = str(payload.get("title", "") or payload.get("window", "")).strip()
                control = str(payload.get("control", "") or payload.get("query", "")).strip()
                control_type = str(payload.get("control_type", "")).strip()
                if not title:
                    return False, "missing window title"
                if not control and not control_type:
                    return False, "missing control query"
                return True, "ok"
            if name.startswith("plugin:desktop_type_control"):
                title = str(payload.get("title", "") or payload.get("window", "")).strip()
                control = str(payload.get("control", "") or payload.get("query", "")).strip()
                control_type = str(payload.get("control_type", "")).strip()
                text = str(payload.get("text", "") or "")
                if not title:
                    return False, "missing window title"
                if not control and not control_type:
                    return False, "missing control query"
                if not text:
                    return False, "missing text"
                return True, "ok"
            if name.startswith("plugin:desktop_type_text"):
                text = str(payload.get("text", ""))
                if not text:
                    return False, "missing text"
                return True, "ok"
            if name.startswith("plugin:desktop_hotkey"):
                hotkey = payload.get("keys") or payload.get("hotkey") or ""
                if not str(hotkey).strip():
                    return False, "missing hotkey"
                return True, "ok"
            if name.startswith("plugin:desktop_click_text"):
                query = str(payload.get("text", "") or payload.get("query", "")).strip()
                if not query:
                    return False, "missing query text"
                return True, "ok"
            if name.startswith("plugin:desktop_click"):
                if "x" not in payload or "y" not in payload:
                    return False, "missing coordinates"
                return True, "ok"
            if name.startswith("plugin:desktop_drag"):
                for key in ("x1", "y1", "x2", "y2"):
                    if key not in payload:
                        return False, f"missing {key}"
                return True, "ok"
            if name.startswith("plugin:desktop_ocr"):
                return True, "ok"
            if name.startswith("plugin:desktop_screenshot") or name.startswith("plugin:desktop_list_windows"):
                return True, "ok"
        except Exception as e:
            return False, f"invalid: {e}"
        # default allow
        return True, "ok"

    def estimate_cost_success(self, task: Task, state: WorldState) -> Tuple[float, float]:
        # cost: higher means more expensive. success: likelihood [0,1]
        name = task.name
        payload = task.payload or {}
        cost = 1.0
        success = 0.7
        if name in {"read_file", "list_dir", "check_exists", "search_files", "search_text"}:
            cost = 0.6
            success = 0.85
        elif name in {"write_file", "append_file", "touch_file", "mkdir"}:
            cost = 0.9
            success = 0.8
        elif name in {"copy_file", "move_file"}:
            cost = 2.2 if name == "move_file" else 2.0
            success = 0.75
        elif name in {"run_script", "run_tests"}:
            cost = 2.2
            success = 0.55
        elif name.startswith("plugin:browser_open_url"):
            cost = 1.2
            success = 0.75
        elif name.startswith("plugin:browser_dom_open"):
            cost = 1.5
            success = 0.7
        elif name.startswith("plugin:browser_dom_click") or name.startswith("plugin:browser_dom_type"):
            cost = 1.8
            success = 0.6
        elif name.startswith("plugin:browser_dom_extract_text") or name.startswith("plugin:browser_dom_wait_text"):
            cost = 1.4
            success = 0.65
        elif name.startswith("plugin:browser_dom_screenshot") or name.startswith("plugin:browser_dom_close"):
            cost = 1.2
            success = 0.75
        elif name.startswith("plugin:desktop_list_windows") or name.startswith("plugin:desktop_screenshot"):
            cost = 1.0
            success = 0.8
        elif name.startswith("plugin:desktop_list_controls"):
            cost = 1.3
            success = 0.75
        elif name.startswith("plugin:desktop_focus_window"):
            cost = 1.3
            success = 0.65
        elif name.startswith("plugin:desktop_click_control"):
            cost = 1.9
            success = 0.62
        elif name.startswith("plugin:desktop_type_control"):
            cost = 2.0
            success = 0.58
        elif name.startswith("plugin:desktop_type_text") or name.startswith("plugin:desktop_hotkey"):
            cost = 1.6
            success = 0.55
        elif name.startswith("plugin:desktop_launch"):
            cost = 1.8
            success = 0.65
        elif name.startswith("plugin:desktop_ocr"):
            cost = 1.4
            success = 0.6
        elif name.startswith("plugin:desktop_click_text"):
            cost = 2.0
            success = 0.5
        elif name.startswith("plugin:desktop_click") or name.startswith("plugin:desktop_drag"):
            cost = 1.7
            success = 0.55
        elif name == "chmod":
            cost = 0.8
            success = 0.5
        elif name == "generate_code":
            cost = 2.0
            success = 0.45
        elif name == "rollback":
            cost = 1.5
            success = 0.7
        elif name == "delete_file":
            cost = 3.0
            success = 0.6

        path = payload.get("path") if isinstance(payload, dict) else None
        if name == "read_file" and path and not self.fm.exists(path):
            success *= 0.2
        if name in {"run_script", "run_tests"} and state.errors:
            # tests more likely to fail if errors already exist
            success *= 0.7
        self._ensure_priority_config()
        success = min(1.0, max(0.05, success))
        buckets = []
        if state.error_types:
            for et in state.error_types[:4]:
                buckets.append(f"err:{et}")
        buckets.extend(self._path_buckets(payload))
        if self.stats is not None:
            mean, conf, _n = self.stats.estimate(name, buckets=buckets)
            # Blend: higher confidence from history => more weight.
            success = (1.0 - conf) * success + conf * mean
            if mean < 0.5:
                cost *= 1.1
            duration_ms = self.stats.duration_ms(name)
            if duration_ms > 0:
                # cost increases with observed latency
                scale = 1.0 + min(2.0, duration_ms / 1000.0)
                cost *= scale
            # regression per action+error_type
            err_type = state.error_types[0] if state.error_types else ""
            feats = self.stats.build_features(task, state, err_type)
            group = self.stats.action_group(task)
            pred, pconf = self.stats.predict_regressor(name, err_type, feats, group=group)
            if pconf > 0:
                success = (1.0 - pconf) * success + pconf * pred
        # error-type bias for exploration (failure attribution)
        success *= self._error_type_bias(name, state.error_types)
        # priority mapping adjustments
        cost_mul, succ_mul = self._priority_multipliers(name, state.error_types)
        cost *= cost_mul
        success *= succ_mul
        success = min(1.0, max(0.03, success))
        return cost, success

    def is_destructive(self, task: Task) -> bool:
        name = task.name
        payload = task.payload or {}
        if name in {"delete_file", "rollback"}:
            return True
        if name == "move_file":
            return True
        if name.startswith("plugin:desktop_type_text") or name.startswith("plugin:desktop_hotkey"):
            return True
        if name.startswith("plugin:desktop_type_control"):
            return True
        if name.startswith("plugin:desktop_click_text"):
            return True
        if name.startswith("plugin:desktop_click_control"):
            return True
        if name.startswith("plugin:desktop_click") or name.startswith("plugin:desktop_drag"):
            return True
        if name.startswith("plugin:browser_dom_click") or name.startswith("plugin:browser_dom_type"):
            return True
        if name in {"write_file"}:
            path = payload.get("path", "")
            if path:
                try:
                    return self.fm.exists(path)
                except Exception:
                    return True
        return False

    def action_confidence(self, task: Task, state: WorldState) -> float:
        if self.stats is None:
            return 0.0
        buckets = []
        if state.error_types:
            for et in state.error_types[:4]:
                buckets.append(f"err:{et}")
        buckets.extend(self._path_buckets(task.payload or {}))
        _mean, conf, _n = self.stats.estimate(task.name, buckets=buckets)
        err_type = state.error_types[0] if state.error_types else ""
        feats = self.stats.build_features(task, state, err_type)
        group = self.stats.action_group(task)
        _pred, pconf = self.stats.predict_regressor(task.name, err_type, feats, group=group)
        return max(conf, pconf)

    def _path_buckets(self, payload: dict | None) -> list[str]:
        buckets: list[str] = []
        if not payload:
            return buckets
        path = payload.get("path") or payload.get("src") or payload.get("dst") or ""
        if not path:
            return buckets
        path = str(path).replace("\\", "/")
        parts = [p for p in path.split("/") if p]
        if path.endswith("/"):
            buckets.append("path:dir")
        else:
            buckets.append("path:file")
        if "." in parts[-1]:
            ext = "." + parts[-1].rsplit(".", 1)[-1].lower()
            buckets.append(f"path_ext:{ext}")
        depth = len(parts)
        buckets.append(f"path_depth:{depth}")
        if parts:
            buckets.append(f"path_dir:{parts[0]}")
        if len(parts) >= 2:
            buckets.append(f"path_dir2:{parts[0]}/{parts[1]}")
        return buckets

    def _error_type_bias(self, action_name: str, error_types: list[str]) -> float:
        if not error_types:
            return 1.0
        bias = 1.0
        # lightweight attribution: reduce likely-to-fail actions under certain errors
        for et in error_types:
            if et in {"import_missing", "syntax"} and action_name in {"run_tests", "run_script"}:
                bias *= 0.7
            if et in {"file_missing"} and action_name in {"run_script"}:
                bias *= 0.6
            if et in {"permission"} and action_name in {"write_file", "append_file", "touch_file"}:
                bias *= 0.7
            if et in {"not_a_dir", "path_conflict"} and action_name in {"write_file", "mkdir"}:
                bias *= 0.6
            if et in {"timeout"} and action_name in {"run_tests", "run_script"}:
                bias *= 0.8
        return max(0.5, min(1.2, bias))

    def _load_priority_config(self) -> dict:
        path = os.environ.get("ACTION_PRIORITY_PATH", "config/action_priority.json")
        p = Path(path)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _priority_multipliers(self, action_name: str, error_types: list[str]) -> tuple[float, float]:
        if not self.priority_cfg:
            return 1.0, 1.0
        cost_mul = 1.0
        succ_mul = 1.0
        overrides = self.priority_cfg.get("action_overrides", {})
        if action_name in overrides:
            ov = overrides.get(action_name, {})
            succ_mul *= float(ov.get("success", 1.0))
            cost_mul *= float(ov.get("cost", 1.0))
        et_map = self.priority_cfg.get("error_to_action", {})
        for et in error_types or []:
            amap = et_map.get(et, {})
            if action_name in amap:
                ov = amap.get(action_name, {})
                succ_mul *= float(ov.get("success", 1.0))
                cost_mul *= float(ov.get("cost", 1.0))
        return cost_mul, succ_mul

    def _ensure_priority_config(self) -> None:
        path = os.environ.get("ACTION_PRIORITY_PATH", self.priority_path)
        if path != self.priority_path:
            self.priority_path = path
            self.priority_mtime = 0.0
        p = Path(self.priority_path)
        if not p.exists():
            self.priority_cfg = {}
            self.priority_mtime = 0.0
            return
        try:
            mtime = p.stat().st_mtime
            if mtime != self.priority_mtime:
                data = json.loads(p.read_text(encoding="utf-8"))
                if self._validate_priority_config(data):
                    self.priority_cfg = data
                    self._last_good_cfg = data
                    self.priority_mtime = mtime
                else:
                    # rollback to last good config if validation fails
                    if self._last_good_cfg:
                        self.priority_cfg = self._last_good_cfg
                    # keep mtime so we don't re-parse until change
                    self.priority_mtime = mtime
        except Exception:
            return

    def _validate_priority_config(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        for k in ("action_overrides", "error_to_action"):
            if k in data and not isinstance(data[k], dict):
                return False
        # shallow validation of nested entries
        overrides = data.get("action_overrides", {})
        for action, cfg in overrides.items():
            if not isinstance(cfg, dict):
                return False
            if "success" in cfg and not isinstance(cfg["success"], (int, float)):
                return False
            if "cost" in cfg and not isinstance(cfg["cost"], (int, float)):
                return False
        err_map = data.get("error_to_action", {})
        for err, amap in err_map.items():
            if not isinstance(amap, dict):
                return False
            for action, cfg in amap.items():
                if not isinstance(cfg, dict):
                    return False
                if "success" in cfg and not isinstance(cfg["success"], (int, float)):
                    return False
                if "cost" in cfg and not isinstance(cfg["cost"], (int, float)):
                    return False
        return True

    def filter_tasks(self, tasks: List[Task], state: WorldState) -> List[Task]:
        out: List[Task] = []
        for t in tasks:
            ok, _ = self.is_valid(t, state)
            if ok:
                out.append(t)
        return out
