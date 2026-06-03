# -*- coding: utf-8 -*-
"""Plugin dispatch helpers for the computer-use runtime."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

from system.computer_use.plugins import registry as plugin_registry
from system.computer_use._runtime_grid import _clean_str

logger = logging.getLogger(__name__)


def _run_plugin_once(runtime, plugin_name, payload):
    try:
        asyncio.get_running_loop()
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(plugin_registry.run(plugin_name, payload))
        finally:
            loop.close()
    except RuntimeError:
        return asyncio.run(plugin_registry.run(plugin_name, payload))


def _run_plugin(runtime, plugin_name, payload, *, required=False):
    attempts = max(1, int(runtime.config.plugin_retries))
    last_error = ""
    for attempt in range(1, attempts + 1):
        merged = dict(payload or {})
        if runtime.config.dry_run:
            merged["dry_run"] = True
        try:
            result = dict(_run_plugin_once(runtime, plugin_name, merged) or {})
            result.setdefault("verified", True)
            result.setdefault("plugin", plugin_name)
            result["attempt"] = attempt
            result["ok"] = bool(result.get("verified", True))
            return result
        except Exception as exc:
            last_error = str(exc)
            if attempt < attempts:
                time.sleep(0.15)
                continue
    if required:
        raise RuntimeError(f"{plugin_name} failed: {last_error or 'unknown error'}")
    return {
        "plugin": plugin_name,
        "verified": False,
        "ok": False,
        "error": last_error or "unknown error",
        "attempt": attempts,
    }


def _legacy_result(result, *, action_type):
    merged = dict(result or {})
    merged.setdefault("action_type", action_type)
    merged["success"] = bool(merged.get("ok", merged.get("verified", False)))
    return merged


def run_action(runtime, action_type, params=None):
    """Compatibility shim for older smoke tests that execute single actions."""
    action_type = _clean_str(action_type)
    params = dict(params or {})
    if not action_type:
        return {"action_type": action_type, "success": False, "error": "missing action_type"}

    try:
        if action_type == "screenshot":
            runtime._legacy_action_index += 1
            runtime.screenshot_dir.mkdir(parents=True, exist_ok=True)
            result = _run_plugin(
                runtime,
                "desktop_screenshot",
                {
                    "path": (
                        runtime.screenshot_dir
                        / f"legacy_action_{runtime._legacy_action_index:04d}.png"
                    ).as_posix(),
                },
                required=False,
            )
            return _legacy_result(result, action_type=action_type)

        if action_type == "desktop.launch_app":
            target = _clean_str(
                params.get("app")
                or params.get("target")
                or params.get("path")
                or params.get("command")
            )
            result = _run_plugin(
                runtime,
                "desktop_launch",
                {"target": target, "args": params.get("args") or []},
                required=False,
            )
            return _legacy_result(result, action_type=action_type)

        if action_type == "desktop.focus_window":
            title = _clean_str(
                params.get("window_title")
                or params.get("window")
                or params.get("title")
            )
            result = _run_plugin(
                runtime,
                "desktop_focus_window",
                {"title": title, "verify_window_title": title},
                required=False,
            )
            return _legacy_result(result, action_type=action_type)

        if action_type == "desktop.type_text":
            result = _run_plugin(
                runtime,
                "desktop_type_text",
                {"text": str(params.get("text", "") or "")},
                required=False,
            )
            return _legacy_result(result, action_type=action_type)

        if action_type == "desktop.hotkey":
            hotkey = _clean_str(params.get("key") or params.get("hotkey") or params.get("keys"))
            result = _run_plugin(
                runtime,
                "desktop_hotkey",
                {"hotkey": hotkey},
                required=False,
            )
            return _legacy_result(result, action_type=action_type)

        if action_type == "desktop.wait":
            wait_ms = max(0, int(params.get("ms", params.get("milliseconds", 0)) or 0))
            time.sleep(wait_ms / 1000.0)
            return {
                "action_type": action_type,
                "success": True,
                "verified": True,
                "ok": True,
                "wait_ms": wait_ms,
                "note": f"waited {wait_ms}ms",
            }
    except Exception as exc:
        logger.debug("computer-use legacy run_action failed: %s", exc, exc_info=True)
        return {"action_type": action_type, "success": False, "error": str(exc)}

    return {
        "action_type": action_type,
        "success": False,
        "error": f"unsupported action_type: {action_type}",
    }


def cleanup(runtime):
    """Compatibility no-op kept for older smoke tests."""
    return None
