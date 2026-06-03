# -*- coding: utf-8 -*-
"""Action executor for the computer-use runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from system.computer_use._runtime_grid import _clean_str, _find_grid_cell


def _execute_action(runtime, action, observation):
    window_hint = action.window or _clean_str(observation.get("controls_window")) or _clean_str(
        (observation.get("active_window") or {}).get("title")
    )
    if action.type == "focus_window":
        title = action.title or window_hint
        return runtime._run_plugin(
            "desktop_focus_window",
            {"title": title, "verify_window_title": title},
            required=False,
        )
    if action.type == "click_control":
        payload = {
            "window": window_hint,
            "control": action.control,
            "index": action.index,
            "fallback_to_ocr": True,
            "ocr_text": action.control,
        }
        if action.control_type:
            payload["control_type"] = action.control_type
        return runtime._run_plugin("desktop_click_control", payload, required=False)
    if action.type == "type_control":
        payload = {
            "window": window_hint,
            "control": action.control,
            "text": action.text,
            "index": action.index,
            "clear_first": bool(action.clear_first),
            "fallback_to_ocr": True,
            "ocr_text": action.control,
        }
        if action.control_type:
            payload["control_type"] = action.control_type
        return runtime._run_plugin("desktop_type_control", payload, required=False)
    if action.type == "click_text":
        payload = {"text": action.text}
        if window_hint:
            payload["title"] = window_hint
        return runtime._run_plugin("desktop_click_text", payload, required=False)
    if action.type == "click_block":
        cell = _find_grid_cell(observation.get("grid_cells", []) or [], action.block)
        if cell is None:
            return {"verified": False, "ok": False, "error": f"unknown block: {action.block}"}
        result = runtime._run_plugin(
            "desktop_click",
            {
                "x": int(cell.get("center_x", 0) or 0),
                "y": int(cell.get("center_y", 0) or 0),
                "button": action.button,
                "clicks": action.clicks,
            },
            required=False,
        )
        result["block"] = str(cell.get("id", "") or action.block)
        return result
    if action.type == "drag_blocks":
        start = _find_grid_cell(observation.get("grid_cells", []) or [], action.from_block)
        end = _find_grid_cell(observation.get("grid_cells", []) or [], action.to_block)
        if start is None or end is None:
            return {
                "verified": False,
                "ok": False,
                "error": f"unknown drag blocks: {action.from_block} -> {action.to_block}",
            }
        result = runtime._run_plugin(
            "desktop_drag",
            {
                "x1": int(start.get("center_x", 0) or 0),
                "y1": int(start.get("center_y", 0) or 0),
                "x2": int(end.get("center_x", 0) or 0),
                "y2": int(end.get("center_y", 0) or 0),
            },
            required=False,
        )
        result["from_block"] = str(start.get("id", "") or action.from_block)
        result["to_block"] = str(end.get("id", "") or action.to_block)
        return result
    if action.type == "type_text":
        return runtime._run_plugin("desktop_type_text", {"text": action.text}, required=False)
    if action.type == "hotkey":
        return runtime._run_plugin("desktop_hotkey", {"hotkey": action.keys}, required=False)
    if action.type == "launch":
        return runtime._run_plugin("desktop_launch", {"target": action.target}, required=False)
    if action.type == "scroll":
        # Scroll at screen center or specified coordinates
        x = int(getattr(action, 'x', 0) or 0)
        y = int(getattr(action, 'y', 0) or 0)
        if not x and not y:
            # Default: scroll at screen center
            from system.computer_use._runtime_grid import _image_size
            screen_path = observation.get("screen_path", "")
            try:
                w, h = _image_size(Path(screen_path)) if screen_path else (1920, 1080)
            except Exception:
                w, h = 1920, 1080
            x, y = w // 2, h // 2
        amount = int(float(getattr(action, 'seconds', 0) or 3) * -40)  # negative = scroll down
        return runtime._run_plugin("desktop_scroll", {"x": x, "y": y, "delta": amount}, required=False)
    if action.type == "wait":
        time.sleep(max(0.1, float(action.seconds)))
        return {
            "verified": True,
            "ok": True,
            "note": f"waited {action.seconds:.2f}s",
            "verification": "wait",
        }
    if action.type == "done":
        return {"verified": True, "ok": True, "note": action.summary or "goal completed"}
    if action.type == "fail":
        return {"verified": False, "ok": False, "error": action.reason or "goal failed"}
    return {"verified": False, "ok": False, "error": f"unsupported action type: {action.type}"}
