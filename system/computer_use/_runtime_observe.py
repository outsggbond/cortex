# -*- coding: utf-8 -*-
"""Observation builder for the computer-use runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from system.computer_use._runtime_grid import (
    _clean_str,
    _truncate,
    _build_grid,
    _file_sha1,
    _map_ocr_to_grid,
    _nonempty_blocks,
    _render_grid_overlay,
    _image_size,
)

logger = logging.getLogger(__name__)


def _observe(runtime, *, step_index, run_dir, preferred_window):
    screen_path = run_dir / f"step_{step_index:02d}_screen.png"
    grid_path = run_dir / f"step_{step_index:02d}_grid.png"
    active = runtime._run_plugin("desktop_active_window", {}, required=False)
    windows_result = runtime._run_plugin(
        "desktop_list_windows",
        {"limit": max(1, int(runtime.config.windows_limit))},
        required=False,
    )
    screen_result = runtime._run_plugin(
        "desktop_screenshot",
        {"path": screen_path.as_posix()},
        required=True,
    )
    actual_screen_path = Path(str(screen_result.get("path", "") or screen_path.as_posix()))
    ocr_result = runtime._run_plugin(
        "desktop_ocr",
        {"path": actual_screen_path.as_posix()},
        required=False,
    )
    try:
        width, height = _image_size(actual_screen_path)
    except Exception:
        width, height = 0, 0
    grid_cells = _build_grid(width, height, rows=runtime.config.grid_rows, cols=runtime.config.grid_cols)
    ocr_items = _map_ocr_to_grid(
        list(ocr_result.get("items", []) or []),
        grid_cells,
        limit=max(1, int(runtime.config.ocr_items_limit)),
    )
    overlay_saved = ""
    try:
        overlay_saved = _render_grid_overlay(
            actual_screen_path,
            grid_path,
            cells=grid_cells,
            ocr_items=ocr_items,
        )
    except Exception as exc:
        logger.debug("computer-use: failed to render grid overlay: %s", exc, exc_info=True)
    active_title = _clean_str(active.get("title") or active.get("active_title"))
    controls_window = _clean_str(preferred_window) or active_title
    controls_result = {}
    controls: List[Dict[str, Any]] = []
    if controls_window:
        controls_result = runtime._run_plugin(
            "desktop_list_controls",
            {
                "window": controls_window,
                "limit": max(1, int(runtime.config.controls_limit)),
            },
            required=False,
        )
        for item in list(controls_result.get("controls", []) or [])[: max(1, int(runtime.config.controls_limit))]:
            controls.append(
                {
                    "title": _clean_str((item or {}).get("title")),
                    "control_type": _clean_str((item or {}).get("control_type")),
                    "automation_id": _clean_str((item or {}).get("automation_id")),
                    "rect": list((item or {}).get("rect", []) or []),
                }
            )
    open_windows = []
    for item in list(windows_result.get("windows", []) or [])[: max(1, int(runtime.config.windows_limit))]:
        open_windows.append(
            {
                "title": _clean_str((item or {}).get("title")),
                "process_name": _clean_str((item or {}).get("process_name")),
                "pid": int((item or {}).get("pid", 0) or 0),
            }
        )
    if not controls_window and open_windows:
        controls_window = _clean_str((open_windows[0] or {}).get("title"))
        if controls_window:
            controls_result = runtime._run_plugin(
                "desktop_list_controls",
                {
                    "window": controls_window,
                    "limit": max(1, int(runtime.config.controls_limit)),
                },
                required=False,
            )
            controls = []
            for item in list(controls_result.get("controls", []) or [])[: max(1, int(runtime.config.controls_limit))]:
                controls.append(
                    {
                        "title": _clean_str((item or {}).get("title")),
                        "control_type": _clean_str((item or {}).get("control_type")),
                        "automation_id": _clean_str((item or {}).get("automation_id")),
                        "rect": list((item or {}).get("rect", []) or []),
                    }
                )
    surface_mode = "uia" if controls else ("ocr_only" if ocr_items else "visual_only")
    special_drawn_ui_likely = bool((not controls) and (bool(ocr_items) or bool(active_title) or bool(open_windows)))
    return {
        "step_index": step_index,
        "screen_path": actual_screen_path.as_posix(),
        "grid_path": overlay_saved or grid_path.as_posix(),
        "screen_sha1": _file_sha1(actual_screen_path),
        "image_width": width,
        "image_height": height,
        "active_window": {
            "title": active_title,
            "process_name": _clean_str(active.get("process_name")),
            "pid": int(active.get("pid", 0) or 0),
            "rect": list(active.get("rect", []) or []),
        },
        "open_windows": open_windows,
        "controls_window": controls_window,
        "controls": controls,
        "controls_error": _clean_str(controls_result.get("error")),
        "ocr_text": _truncate(str(ocr_result.get("text", "") or ""), 600),
        "ocr_engine": _clean_str(ocr_result.get("engine")),
        "ocr_items": ocr_items,
        "nonempty_blocks": _nonempty_blocks(ocr_items, limit=24),
        "grid_cells": grid_cells,
        "surface_mode": surface_mode,
        "special_drawn_ui_likely": special_drawn_ui_likely,
        "targeting_hint": (
            "prefer_control_actions"
            if controls
            else ("prefer_text_or_block_actions" if ocr_items else "prefer_block_actions")
        ),
    }
