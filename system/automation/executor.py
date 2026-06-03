from __future__ import annotations

import asyncio
import subprocess
import sys
import time
import os
import hashlib
import traceback
import shutil
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

from system.l_utils.files import FileManager
from system.l_utils.codegen import CodeGenerator
from system.l_utils.loader import ModuleLoader
from system.strategy.task_flow import TaskFlow
from system.computer_use.plugins import registry as plugin_registry

from system.core.planner import Plan, Task


logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    completed: List[str]
    failed: List[str]
    notes: List[str]
    level_reports: List[Dict[str, Any]] | None = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    snapshot: Dict[str, Any] | None = None
    step_results: List[Dict[str, Any]] = field(default_factory=list)


class TaskExecutor:
    def __init__(self, project_root: str = ".", audit=None, exec_audit=None):
        self.fm = FileManager(project_root)
        self.codegen = CodeGenerator()
        self.loader = ModuleLoader()
        self.flow = TaskFlow()
        self.audit = audit
        self.exec_audit = exec_audit
        self.vars: Dict[str, str] = {}
        self.snapshot_ignore = {
            ".venv",
            ".idea",
            ".rollback",
            "__pycache__",
            "checkpoints",
            "audit",
            "node_modules",
        }
        self.snapshot_hash_limit = int(os.environ.get("SNAPSHOT_HASH_LIMIT", "262144"))
        self.snapshot_max_files = int(os.environ.get("SNAPSHOT_MAX_FILES", "8000"))

    def run(self, plan: Plan) -> ExecutionResult:
        completed: List[str] = []
        failed: List[str] = []
        notes: List[str] = []
        level_reports: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        step_results: List[Dict[str, Any]] = []
        snapshot_before = self._capture_snapshot() if self._should_snapshot() else None
        levels = plan.levels or [plan.steps]
        for idx, level in enumerate(levels):
            level_ok = True
            level_notes: List[str] = []

            def _make(step):
                def _f():
                    return self._execute_step(step)
                return _f

            funcs = [_make(step) for step in level]
            try:
                results = self.flow.run(funcs)
                results = self.flow.resolve(results)
            except Exception:
                results = [None for _ in level]
            for step, res in zip(level, results):
                try:
                    if isinstance(res, Exception):
                        raise res
                    if res is None:
                        raise RuntimeError("step returned no result")
                    note = res
                    duration_ms = 0.0
                    if isinstance(res, dict):
                        note = res.get("note", "")
                        duration_ms = float(res.get("duration_ms") or 0.0)
                    step.status = "done"
                    completed.append(step.detail)
                    level_notes.append(note or f"done: {step.detail}")
                    if self.audit:
                        self.audit.record("executor", f"done: {step.detail}")
                    if self.exec_audit:
                        self.exec_audit.record(step.detail, "done", duration_ms)
                    step_results.append(
                        {
                            "step": step.detail,
                            "name": step.name,
                            "ok": True,
                            "payload": dict(step.payload or {}),
                            "result": res if isinstance(res, dict) else {"note": note},
                        }
                    )
                except Exception as e:
                    step.status = "failed"
                    failed.append(step.detail)
                    level_ok = False
                    err_info = {
                        "step": step.detail,
                        "name": step.name,
                        "error": str(e),
                        "type": type(e).__name__,
                    }
                    if step.payload:
                        err_info["payload"] = step.payload
                    err_info["trace"] = traceback.format_exc(limit=3)
                    errors.append(err_info)
                    level_notes.append(f"failed: {step.detail}")
                    if self.audit:
                        self.audit.record("executor", f"failed: {step.detail} {type(e).__name__}: {e}")
                    if self.exec_audit:
                        self.exec_audit.record(step.detail, "failed", 0.0)
                    step_results.append(
                        {
                            "step": step.detail,
                            "name": step.name,
                            "ok": False,
                            "error": str(e),
                            "type": type(e).__name__,
                            "payload": dict(step.payload or {}),
                        }
                    )
            level_reports.append(
                {"level": idx + 1, "ok": level_ok, "notes": level_notes, "failed": not level_ok}
            )
            if not level_ok:
                notes.append(f"Level {idx + 1} failed, stopping.")
                break
        snapshot_after = self._capture_snapshot() if self._should_snapshot() else None
        snapshot_summary = None
        if snapshot_before is not None and snapshot_after is not None:
            diff = self._diff_snapshots(snapshot_before, snapshot_after)
            snapshot_summary = {"diff": diff, "before_count": snapshot_before.get("count", 0), "after_count": snapshot_after.get("count", 0)}
            if self.audit:
                try:
                    self.audit.record_json(
                        "executor_snapshot",
                        {
                            "plan_goal": plan.goal,
                            "before": snapshot_before,
                            "after": snapshot_after,
                            "diff": diff,
                            "errors": errors,
                        },
                    )
                except Exception:
                    logger.debug("executor: audit snapshot record failed", exc_info=True)
        return ExecutionResult(
            completed=completed,
            failed=failed,
            notes=notes,
            level_reports=level_reports,
            errors=errors,
            snapshot=snapshot_summary,
            step_results=step_results,
        )

    def _execute_step(self, step: Task) -> dict:
        start = time.time()
        payload = self._resolve_payload(step.payload or {})
        if step.name == "read_file":
            path = payload.get("path", "")
            content = self.fm.read_text(path)
            self._record_output(f"read:{path}", content)
            return {
                "note": f"read {path} ok len={len(content)}",
                "output": content,
                "duration_ms": (time.time() - start) * 1000.0,
            }
        if step.name == "write_file":
            path = payload.get("path", "")
            content = payload.get("content", "")
            self.fm.write_text(path, content)
            return {"note": f"write {path} ok", "duration_ms": (time.time() - start) * 1000.0}
        if step.name == "append_file":
            path = payload.get("path", "")
            content = payload.get("content", "")
            self.fm.append_text(path, content)
            return {"note": f"append {path} ok", "duration_ms": (time.time() - start) * 1000.0}
        if step.name == "touch_file":
            path = payload.get("path", "")
            self.fm.touch(path)
            return {"note": f"touch {path} ok", "duration_ms": (time.time() - start) * 1000.0}
        if step.name == "list_dir":
            path = payload.get("path", ".")
            entries = self.fm.list_dir(path)
            sample = entries[:50]
            return {
                "note": f"list {path} ok count={len(entries)}",
                "entries": sample,
                "duration_ms": (time.time() - start) * 1000.0,
            }
        if step.name == "check_exists":
            path = payload.get("path", "")
            ok = self.fm.exists(path)
            return {"note": f"exists {path}={ok}", "duration_ms": (time.time() - start) * 1000.0}
        if step.name == "mkdir":
            path = payload.get("path", "")
            self.fm.mkdir(path)
            return {"note": f"mkdir {path} ok", "duration_ms": (time.time() - start) * 1000.0}
        if step.name == "copy_file":
            src = payload.get("src", "")
            dst = payload.get("dst", "")
            self.fm.copy(src, dst)
            return {"note": f"copy {src} -> {dst} ok", "duration_ms": (time.time() - start) * 1000.0}
        if step.name == "move_file":
            src = payload.get("src", "")
            dst = payload.get("dst", "")
            self.fm.move(src, dst)
            return {"note": f"move {src} -> {dst} ok", "duration_ms": (time.time() - start) * 1000.0}
        if step.name == "delete_file":
            path = payload.get("path", "")
            self.fm.delete(path)
            return {"note": f"delete {path} ok", "duration_ms": (time.time() - start) * 1000.0}
        if step.name == "search_text":
            pattern = payload.get("pattern", "")
            path = payload.get("path", ".")
            hits = self._search_text(pattern, path)
            return {
                "note": f"search '{pattern}' in {path} hits={len(hits)}",
                "hits": hits[:50],
                "duration_ms": (time.time() - start) * 1000.0,
            }
        if step.name == "search_files":
            pattern = payload.get("pattern", "")
            path = payload.get("path", ".")
            hits = self._search_files(pattern, path)
            return {
                "note": f"find '{pattern}' in {path} hits={len(hits)}",
                "hits": hits[:50],
                "duration_ms": (time.time() - start) * 1000.0,
            }
        if step.name == "run_tests":
            cmd = payload.get("command") if payload else None
            res = self._run_tests(cmd)
            return {
                "note": res.get("note", "run tests"),
                "duration_ms": (time.time() - start) * 1000.0,
            }
        if step.name == "run_script":
            path = payload.get("path", "")
            args = payload.get("args") or []
            if isinstance(args, str):
                args = [args]
            p = self.fm._resolve(path)
            if p.suffix.lower() != ".py":
                raise ValueError("only .py script allowed")
            if not p.exists():
                raise FileNotFoundError(f"script not found: {p}")
            res = subprocess.run(
                [sys.executable, str(p), *[str(a) for a in args if a is not None]],
                cwd=self.fm.root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if res.returncode != 0:
                err = (res.stderr or res.stdout or "").strip()
                err = err[:300]
                raise RuntimeError(f"script failed code={res.returncode} {err}")
            out = (res.stdout or "").strip()
            if out:
                self._record_output(f"run:{path}", out)
            note = f"run {path} ok code=0"
            return {"note": note, "output": out, "duration_ms": (time.time() - start) * 1000.0}
        if step.name == "rollback":
            path = payload.get("path") if payload else ""
            if not path:
                raise ValueError("rollback requires path")
            ok = self.fm.rollback(path)
            return {"note": f"rollback {path} {'ok' if ok else 'fail'}", "duration_ms": (time.time() - start) * 1000.0}
        if step.name == "chmod":
            path = payload.get("path") if payload else ""
            mode = payload.get("mode") if payload else None
            if not path:
                raise ValueError("chmod requires path")
            if not mode:
                raise ValueError("chmod requires mode")
            p = self.fm._resolve(path)
            try:
                if isinstance(mode, str):
                    if mode.startswith("0o"):
                        mode_val = int(mode, 8)
                    else:
                        mode_val = int(mode)
                else:
                    mode_val = int(mode)
            except Exception:
                raise ValueError("chmod mode invalid")
            os.chmod(p, mode_val)
            return {"note": f"chmod {path} {mode}", "duration_ms": (time.time() - start) * 1000.0}
        if step.name.startswith("plugin:"):
            plugin_name = step.name.split(":", 1)[1]
            payload = payload or {}
            result = self._run_plugin_with_retry(step.name, plugin_name, payload)
            note = f"plugin {plugin_name} ok"
            if isinstance(result, dict):
                note = str(result.get("note", "") or note)
            elif result is not None:
                note = f"plugin {plugin_name}: {result}"
            payload_out = result if isinstance(result, dict) else {"value": result}
            payload_out["note"] = note
            return {
                **payload_out,
                "duration_ms": (time.time() - start) * 1000.0,
            }
        if step.name == "generate_code":
            req = payload.get("request", "") if payload else ""
            path, name = self.codegen.generate_module(req)
            mod = self.loader.load_module(path)
            if mod and hasattr(mod, "run"):
                out = mod.run({"request": req})
                return {"note": f"generated and ran module {name}, output: {out}", "duration_ms": (time.time() - start) * 1000.0}
            return {"note": f"generated module {name}, not executed", "duration_ms": (time.time() - start) * 1000.0}
        return {"note": f"done: {step.detail}", "duration_ms": (time.time() - start) * 1000.0}

    def _record_output(self, key: str, value: str) -> None:
        if value is None:
            return
        text = str(value)
        self.vars["last_output"] = text
        if key:
            self.vars[key] = text

    def _resolve_payload(self, payload: dict) -> dict:
        if not payload:
            return {}
        out: dict = {}
        for k, v in payload.items():
            out[k] = self._resolve_value(v)
        return out

    def _resolve_value(self, value):
        if isinstance(value, str):
            return self._resolve_text(value)
        if isinstance(value, list):
            return [self._resolve_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._resolve_value(v) for k, v in value.items()}
        return value

    def _resolve_text(self, text: str) -> str:
        if "{{" not in text:
            return text
        pattern = re.compile(r"\{\{\s*([^}]+)\s*\}\}")

        def repl(match):
            key = (match.group(1) or "").strip()
            if not key:
                return ""
            if key == "last_output":
                return self.vars.get("last_output", "")
            if key.startswith("read:"):
                return self.vars.get(f"read:{key[5:]}", "")
            if key.startswith("run:"):
                return self.vars.get(f"run:{key[4:]}", "")
            return self.vars.get(key, "")

        return pattern.sub(repl, text)

    def _run_plugin_once(self, plugin_name: str, payload: dict) -> dict:
        try:
            asyncio.get_running_loop()
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(plugin_registry.run(plugin_name, payload))
            finally:
                new_loop.close()
        except RuntimeError:
            return asyncio.run(plugin_registry.run(plugin_name, payload))

    def _plugin_retry_limit(self, step_name: str, payload: dict) -> int:
        if "max_retries" in payload:
            try:
                return max(0, int(payload.get("max_retries", 0)))
            except Exception:
                return 0
        if step_name.startswith("plugin:desktop_"):
            return max(0, int(os.environ.get("COMPUTER_ACTION_MAX_RETRIES", "1")))
        if step_name.startswith("plugin:browser_dom_"):
            return max(0, int(os.environ.get("COMPUTER_BROWSER_MAX_RETRIES", "1")))
        return 0

    def _plugin_retry_delay_s(self, payload: dict) -> float:
        if "retry_delay_s" in payload:
            try:
                return max(0.0, float(payload.get("retry_delay_s", 0.0)))
            except Exception:
                return 0.0
        return max(0.0, float(os.environ.get("COMPUTER_ACTION_RETRY_DELAY_S", "0.15")))

    def _plugin_retry_on_unverified(self, step_name: str, payload: dict) -> bool:
        if "retry_on_unverified" in payload:
            raw = str(payload.get("retry_on_unverified", "")).strip().lower()
            return raw in {"1", "true", "yes", "on"}
        return step_name.startswith("plugin:desktop_") or step_name.startswith("plugin:browser_dom_")

    def _run_plugin_with_retry(self, step_name: str, plugin_name: str, payload: dict) -> dict:
        retries = self._plugin_retry_limit(step_name, payload)
        delay_s = self._plugin_retry_delay_s(payload)
        last_result = None
        last_exc: Exception | None = None
        for attempt in range(1, retries + 2):
            attempt_payload = dict(payload or {})
            attempt_payload["retry_attempt"] = attempt
            try:
                result = self._run_plugin_once(plugin_name, attempt_payload)
                if isinstance(result, dict):
                    result = dict(result)
                    result["attempts"] = attempt
                    result["retried"] = max(0, attempt - 1)
                    if bool(result.get("verified", True)) is False and attempt <= retries and self._plugin_retry_on_unverified(step_name, payload):
                        last_result = result
                        if delay_s > 0:
                            time.sleep(delay_s)
                        continue
                return result
            except Exception as exc:
                last_exc = exc
                if attempt <= retries:
                    if delay_s > 0:
                        time.sleep(delay_s)
                    continue
                raise
        if last_result is not None:
            return last_result
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"plugin {plugin_name} returned no result")

    def _search_text(self, pattern: str, path: str) -> list[str]:
        pattern = pattern or ""
        if not pattern:
            return []
        if shutil.which("rg"):
            cmd = ["rg", "-n", pattern, path]
            res = subprocess.run(cmd, cwd=self.fm.root, capture_output=True, text=True)
            if res.returncode in (0, 1):  # 1 means no matches
                lines = (res.stdout or "").splitlines()
                return lines
        # fallback: naive search
        hits = []
        base = self.fm._resolve(path)
        if base.is_file():
            files = [base]
        else:
            files = [p for p in base.rglob("*") if p.is_file()]
        for p in files:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if pattern in text:
                hits.append(f"{p}:1:{pattern}")
            if len(hits) >= 200:
                break
        return hits

    def _search_files(self, pattern: str, path: str) -> list[str]:
        pattern = pattern or ""
        if not pattern:
            return []
        base = self.fm._resolve(path)
        hits = []
        for p in base.rglob("*"):
            if p.is_dir():
                continue
            if pattern.lower() in p.name.lower():
                hits.append(p.relative_to(self.fm.root).as_posix())
            if len(hits) >= 200:
                break
        return hits

    def _run_tests(self, command: str | None) -> dict:
        if command:
            cmd = command
        else:
            # default: pytest if available, otherwise unittest
            if shutil.which("pytest"):
                cmd = f"{sys.executable} -m pytest -q"
            else:
                cmd = f"{sys.executable} -m unittest"
        res = subprocess.run(
            cmd,
            cwd=self.fm.root,
            capture_output=True,
            text=True,
            timeout=60,
            shell=True,
        )
        if res.returncode != 0:
            err = (res.stderr or res.stdout or "").strip()[:400]
            raise RuntimeError(f"tests failed code={res.returncode} {err}")
        return {"note": f"tests ok: {cmd}"}

    def _should_snapshot(self) -> bool:
        return os.environ.get("EXEC_SNAPSHOT", "1") == "1"

    def _capture_snapshot(self) -> Dict[str, Any]:
        root = self.fm.root
        files: Dict[str, Dict[str, Any]] = {}
        count = 0
        for path in root.rglob("*"):
            try:
                if path.is_dir():
                    continue
                if self._is_ignored(path):
                    continue
                rel = path.relative_to(root).as_posix()
                stat = path.stat()
                entry = {"size": int(stat.st_size), "mtime": float(stat.st_mtime)}
                if stat.st_size <= self.snapshot_hash_limit:
                    entry["sha256"] = self._hash_file(path)
                files[rel] = entry
                count += 1
                if count >= self.snapshot_max_files:
                    break
            except Exception:
                continue
        return {
            "ts": time.time(),
            "root": str(root),
            "count": count,
            "files": files,
            "python": sys.version.split()[0],
        }

    def _diff_snapshots(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        b = before.get("files", {})
        a = after.get("files", {})
        b_keys = set(b.keys())
        a_keys = set(a.keys())
        added = sorted(a_keys - b_keys)
        removed = sorted(b_keys - a_keys)
        modified = []
        for k in sorted(a_keys & b_keys):
            bv = b.get(k, {})
            av = a.get(k, {})
            if bv.get("size") != av.get("size") or bv.get("mtime") != av.get("mtime") or bv.get("sha256") != av.get("sha256"):
                modified.append(k)
        return {
            "added": added[:200],
            "removed": removed[:200],
            "modified": modified[:200],
            "added_count": len(added),
            "removed_count": len(removed),
            "modified_count": len(modified),
        }

    def _hash_file(self, path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _is_ignored(self, path) -> bool:
        parts = set(path.parts)
        return any(p in parts for p in self.snapshot_ignore)
