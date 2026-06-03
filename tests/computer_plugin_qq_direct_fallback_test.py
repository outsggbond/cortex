# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.computer_use import plugins


def _restore_attr(name: str, value):
    setattr(plugins, name, value)


def test_qq_search_fallback_bypasses_uia() -> None:
    snapshot = os.environ.get("COMPUTER_CONTROL_DRY_RUN")
    os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
    old_verify_supported = plugins._verify_supported
    old_wait_uia = plugins._wait_for_top_window_uia
    old_focus_direct = plugins._focus_qq_search_input_direct
    old_input_text = plugins._input_text_windows
    old_active = plugins._active_window_title_windows
    old_verify = plugins._verify_visual_action
    old_clear = plugins._clear_focused_text_windows
    calls = []
    try:
        plugins._verify_supported = lambda payload: None
        plugins._wait_for_top_window_uia = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("UIA should not run for QQ search fallback")
        )
        plugins._focus_qq_search_input_direct = lambda window_title: {
            "matched_window": "QQ",
            "active_title": "QQ",
            "focus_strategy": "qq_search_hotkey_direct",
            "hotkey": "ctrl+f",
        }
        plugins._input_text_windows = lambda text, delay_ms=30: "clipboard_paste_fallback"
        plugins._active_window_title_windows = lambda: "QQ"
        plugins._verify_visual_action = (
            lambda payload, default_verified=True, default_verification="", before_path=None: {
                "verified": bool(default_verified),
                "verification": default_verification,
            }
        )
        plugins._clear_focused_text_windows = lambda: calls.append("clear")

        result = asyncio.run(
            plugins.plugin_desktop_type_control(
                {
                    "window": "QQ",
                    "control": "搜索",
                    "control_type": "Edit",
                    "text": "盛哥",
                    "clear_first": True,
                    "fallback_to_search_region": True,
                }
            )
        )

        assert result["verified"] is True
        assert result["fallback_strategy"] == "qq_search_hotkey_direct"
        assert result["input_strategy"] == "clipboard_paste_fallback"
        assert result["matched_window"] == "QQ"
        assert calls == ["clear"]
    finally:
        _restore_attr("_verify_supported", old_verify_supported)
        _restore_attr("_wait_for_top_window_uia", old_wait_uia)
        _restore_attr("_focus_qq_search_input_direct", old_focus_direct)
        _restore_attr("_input_text_windows", old_input_text)
        _restore_attr("_active_window_title_windows", old_active)
        _restore_attr("_verify_visual_action", old_verify)
        _restore_attr("_clear_focused_text_windows", old_clear)
        if snapshot is None:
            os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
        else:
            os.environ["COMPUTER_CONTROL_DRY_RUN"] = snapshot


def test_qq_message_input_fallback_bypasses_uia() -> None:
    snapshot = os.environ.get("COMPUTER_CONTROL_DRY_RUN")
    os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
    old_verify_supported = plugins._verify_supported
    old_wait_uia = plugins._wait_for_top_window_uia
    old_focus_direct = plugins._focus_qq_message_input_direct
    old_input_text = plugins._input_text_windows
    old_active = plugins._active_window_title_windows
    old_verify = plugins._verify_visual_action
    old_clear = plugins._clear_focused_text_windows
    calls = []
    try:
        plugins._verify_supported = lambda payload: None
        plugins._wait_for_top_window_uia = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("UIA should not run for QQ message fallback")
        )
        plugins._focus_qq_message_input_direct = lambda payload, window_title, anchor_text="发送": {
            "matched_window": "QQ",
            "active_title": "QQ",
            "focus_strategy": "qq_message_input_direct",
            "input_rect": [5, 6, 7, 8],
            "x": 220,
            "y": 340,
            "button": "left",
            "clicks": 1,
            "anchor_text": anchor_text,
            "anchor_engine": "rapidocr",
        }
        plugins._input_text_windows = lambda text, delay_ms=30: "clipboard_paste_fallback"
        plugins._active_window_title_windows = lambda: "QQ"
        plugins._verify_visual_action = (
            lambda payload, default_verified=True, default_verification="", before_path=None: {
                "verified": bool(default_verified),
                "verification": default_verification,
            }
        )
        plugins._clear_focused_text_windows = lambda: calls.append("clear")

        result = asyncio.run(
            plugins.plugin_desktop_type_control(
                {
                    "window": "QQ",
                    "control": "发送",
                    "control_type": "Button",
                    "text": "你好，我已经成功",
                    "clear_first": True,
                    "fallback_to_message_input": True,
                    "anchor_control": "发送",
                    "anchor_control_type": "Button",
                }
            )
        )

        assert result["verified"] is True
        assert result["fallback_strategy"] == "qq_message_input_direct"
        assert result["input_strategy"] == "clipboard_paste_fallback"
        assert result["matched_window"] == "QQ"
        assert result["control_title"] == "发送"
        assert calls == ["clear"]
    finally:
        _restore_attr("_verify_supported", old_verify_supported)
        _restore_attr("_wait_for_top_window_uia", old_wait_uia)
        _restore_attr("_focus_qq_message_input_direct", old_focus_direct)
        _restore_attr("_input_text_windows", old_input_text)
        _restore_attr("_active_window_title_windows", old_active)
        _restore_attr("_verify_visual_action", old_verify)
        _restore_attr("_clear_focused_text_windows", old_clear)
        if snapshot is None:
            os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
        else:
            os.environ["COMPUTER_CONTROL_DRY_RUN"] = snapshot


def test_qq_click_control_recognizes_chat_input_groupbox_query() -> None:
    snapshot = os.environ.get("COMPUTER_CONTROL_DRY_RUN")
    os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
    old_verify_supported = plugins._verify_supported
    old_wait_uia = plugins._wait_for_top_window_uia
    old_focus_wrapper = plugins._focus_wrapper_windows
    old_find_descendant = plugins._find_descendant_control
    old_focus_input = plugins._focus_qq_message_input
    old_focus_direct = plugins._focus_qq_message_input_direct
    old_verify = plugins._verify_visual_action
    try:
        plugins._verify_supported = lambda payload: None
        plugins._wait_for_top_window_uia = lambda *args, **kwargs: object()
        plugins._focus_wrapper_windows = lambda window, fallback_query="": {
            "matched_title": "QQ",
            "active_title": "QQ",
            "verified": True,
        }
        plugins._find_descendant_control = (
            lambda wrapper, title="", control_type="", index=0, exact_title=False, require_visible=True: object()
        )
        plugins._focus_qq_message_input = lambda wrapper, send_button: {
            "focus_strategy": "uia_control",
            "input_rect": [420, 520, 980, 760],
            "input_control_title": "聊天输入区",
            "input_control_type": "Group",
            "x": 700,
            "y": 640,
            "button": "left",
            "clicks": 1,
        }
        plugins._focus_qq_message_input_direct = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("direct QQ message input fallback should not run")
        )
        plugins._verify_visual_action = (
            lambda payload, default_verified=True, default_verification="", before_path=None: {
                "verified": bool(default_verified),
                "verification": default_verification,
            }
        )

        result = asyncio.run(
            plugins.plugin_desktop_click_control(
                {
                    "window": "QQ",
                    "control": "聊天输入区",
                    "control_type": "Group",
                    "anchor_control": "发送",
                    "anchor_control_type": "Button",
                }
            )
        )

        assert result["verified"] is True
        assert result["matched_window"] == "QQ"
        assert result["control_title"] == "聊天输入区"
        assert result["control_type"] == "Group"
        assert result["focus_strategy"] == "uia_control"
    finally:
        _restore_attr("_verify_supported", old_verify_supported)
        _restore_attr("_wait_for_top_window_uia", old_wait_uia)
        _restore_attr("_focus_wrapper_windows", old_focus_wrapper)
        _restore_attr("_find_descendant_control", old_find_descendant)
        _restore_attr("_focus_qq_message_input", old_focus_input)
        _restore_attr("_focus_qq_message_input_direct", old_focus_direct)
        _restore_attr("_verify_visual_action", old_verify)
        if snapshot is None:
            os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
        else:
            os.environ["COMPUTER_CONTROL_DRY_RUN"] = snapshot


def test_qq_click_control_prefers_search_result_direct_before_uia() -> None:
    snapshot = os.environ.get("COMPUTER_CONTROL_DRY_RUN")
    os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
    old_verify_supported = plugins._verify_supported
    old_wait_uia = plugins._wait_for_top_window_uia
    old_click_direct = plugins._click_qq_search_result_direct
    old_active = plugins._active_window_title_windows
    old_verify = plugins._verify_visual_action
    try:
        plugins._verify_supported = lambda payload: None
        plugins._wait_for_top_window_uia = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("UIA should not run for QQ search result direct fallback")
        )
        plugins._click_qq_search_result_direct = lambda payload, window_title, contact, button="left", clicks=1: {
            "matched_window": "QQ",
            "active_title": "QQ",
            "focus_strategy": "qq_search_result_direct",
            "control_rect": [42, 320, 340, 372],
            "matched_text": "\u76db\u54e5",
            "path": "shot.png",
            "engine": "rapidocr",
            "x": 191,
            "y": 346,
            "button": button,
            "clicks": clicks,
        }
        plugins._active_window_title_windows = lambda: "QQ"
        plugins._verify_visual_action = (
            lambda payload, default_verified=True, default_verification="", before_path=None: {
                "verified": bool(default_verified),
                "verification": default_verification,
            }
        )

        result = asyncio.run(
            plugins.plugin_desktop_click_control(
                {
                    "window": "QQ",
                    "control": "\u76db\u54e5",
                    "control_type": "ListItem",
                    "fallback_to_search_result": True,
                }
            )
        )

        assert result["verified"] is True
        assert result["fallback_strategy"] == "qq_search_result_direct"
        assert result["control_title"] == "\u76db\u54e5"
        assert result["matched_window"] == "QQ"
    finally:
        _restore_attr("_verify_supported", old_verify_supported)
        _restore_attr("_wait_for_top_window_uia", old_wait_uia)
        _restore_attr("_click_qq_search_result_direct", old_click_direct)
        _restore_attr("_active_window_title_windows", old_active)
        _restore_attr("_verify_visual_action", old_verify)
        if snapshot is None:
            os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
        else:
            os.environ["COMPUTER_CONTROL_DRY_RUN"] = snapshot


def test_click_qq_search_result_direct_matches_top_row_contact() -> None:
    old_focus_window = plugins._focus_window_windows
    old_matches = plugins._active_window_matches_query
    old_window_rect = plugins._window_rect_for_query_windows
    old_ocr_path = plugins._ocr_image_path
    old_ocr_extract = plugins._ocr_extract
    old_mouse_click = plugins._mouse_click_windows
    try:
        plugins._focus_window_windows = lambda title: {"matched_title": "QQ", "active_title": "QQ", "verified": True}
        plugins._active_window_matches_query = lambda title: True
        plugins._window_rect_for_query_windows = lambda title: ((0, 0, 640, 1168), "QQ")
        plugins._ocr_image_path = lambda payload, prefix="qq_search_result": (Path("shot.png"), (100, 200), "QQ")
        plugins._ocr_extract = lambda path, psm=6: {
            "engine": "rapidocr",
            "items": [
                {"text": "\u7f16\u8f91\u4e2a\u6027\u7b7e\u540d", "left": 154, "top": 115, "width": 148, "height": 32},
                {"text": "\u76db\u54e5", "left": 71, "top": 223, "width": 66, "height": 39},
                {"text": "\u5df2\u7ecf\u6210\u529f", "left": 223, "top": 488, "width": 100, "height": 30},
            ],
        }
        plugins._mouse_click_windows = lambda x, y, button="left", clicks=1: {
            "x": x,
            "y": y,
            "button": button,
            "clicks": clicks,
        }

        result = plugins._click_qq_search_result_direct({}, "QQ", contact="\u76db\u54e5")
        assert result["matched_window"] == "QQ"
        assert result["matched_text"] == "\u76db\u54e5"
        assert result["focus_strategy"] == "qq_search_result_direct"
        assert result["x"] == 292
        assert result["button"] == "left"
    finally:
        _restore_attr("_focus_window_windows", old_focus_window)
        _restore_attr("_active_window_matches_query", old_matches)
        _restore_attr("_window_rect_for_query_windows", old_window_rect)
        _restore_attr("_ocr_image_path", old_ocr_path)
        _restore_attr("_ocr_extract", old_ocr_extract)
        _restore_attr("_mouse_click_windows", old_mouse_click)


def test_click_qq_search_result_direct_falls_back_to_hotkeys_when_mouse_click_drifts() -> None:
    old_focus_window = plugins._focus_window_windows
    old_matches = plugins._active_window_matches_query
    old_window_rect = plugins._window_rect_for_query_windows
    old_ocr_path = plugins._ocr_image_path
    old_ocr_extract = plugins._ocr_extract
    old_mouse_click = plugins._mouse_click_windows
    old_hotkey_select = plugins._select_qq_search_result_hotkey_direct
    try:
        plugins._focus_window_windows = lambda title: {"matched_title": "QQ", "active_title": "QQ", "verified": True}
        states = iter([True, True, False])
        plugins._active_window_matches_query = lambda title: next(states)
        plugins._window_rect_for_query_windows = lambda title: ((0, 0, 640, 1168), "QQ")
        plugins._ocr_image_path = lambda payload, prefix="qq_search_result": (Path("shot.png"), (100, 200), "QQ")
        plugins._ocr_extract = lambda path, psm=6: {
            "engine": "rapidocr",
            "items": [
                {"text": "\u76db\u54e5", "left": 71, "top": 223, "width": 66, "height": 39},
            ],
        }
        plugins._mouse_click_windows = lambda x, y, button="left", clicks=1: {
            "x": x,
            "y": y,
            "button": button,
            "clicks": clicks,
        }
        plugins._select_qq_search_result_hotkey_direct = lambda window_title: {
            "matched_window": "QQ",
            "active_title": "QQ",
            "focus_strategy": "qq_search_result_hotkey_direct",
            "hotkeys": ["down", "enter"],
        }

        result = plugins._click_qq_search_result_direct({}, "QQ", contact="\u76db\u54e5")
        assert result["matched_window"] == "QQ"
        assert result["focus_strategy"] == "qq_search_result_hotkey_direct"
        assert result["matched_text"] == "\u76db\u54e5"
        assert result["hotkeys"] == ["down", "enter"]
        assert "active window drifted" in result["mouse_fallback_reason"]
    finally:
        _restore_attr("_focus_window_windows", old_focus_window)
        _restore_attr("_active_window_matches_query", old_matches)
        _restore_attr("_window_rect_for_query_windows", old_window_rect)
        _restore_attr("_ocr_image_path", old_ocr_path)
        _restore_attr("_ocr_extract", old_ocr_extract)
        _restore_attr("_mouse_click_windows", old_mouse_click)
        _restore_attr("_select_qq_search_result_hotkey_direct", old_hotkey_select)


def test_qq_click_control_prefers_ocr_fallback_before_uia() -> None:
    snapshot = os.environ.get("COMPUTER_CONTROL_DRY_RUN")
    os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
    old_verify_supported = plugins._verify_supported
    old_wait_uia = plugins._wait_for_top_window_uia
    old_focus_window = plugins._focus_window_windows
    old_click_ocr = plugins._click_text_with_ocr
    old_active = plugins._active_window_title_windows
    old_verify = plugins._verify_visual_action
    try:
        plugins._verify_supported = lambda payload: None
        plugins._wait_for_top_window_uia = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("UIA should not run for QQ OCR click fallback")
        )
        plugins._focus_window_windows = lambda title: {
            "matched_title": "QQ",
            "active_title": "QQ",
            "verified": True,
        }
        plugins._click_text_with_ocr = lambda payload, query, button="left": {
            "matched_text": "发送",
            "x": 480,
            "y": 720,
            "button": button,
            "clicks": 1,
            "path": "shot.png",
            "engine": "rapidocr",
        }
        plugins._active_window_title_windows = lambda: "QQ"
        plugins._verify_visual_action = (
            lambda payload, default_verified=True, default_verification="", before_path=None: {
                "verified": bool(default_verified),
                "verification": default_verification,
            }
        )

        result = asyncio.run(
            plugins.plugin_desktop_click_control(
                {
                    "window": "QQ",
                    "control": "发送",
                    "control_type": "Button",
                    "fallback_to_ocr": True,
                    "ocr_text": "发送",
                }
            )
        )

        assert result["verified"] is True
        assert result["fallback_strategy"] == "ocr_text"
        assert result["control_title"] == "发送"
        assert result["matched_window"] == "QQ"
    finally:
        _restore_attr("_verify_supported", old_verify_supported)
        _restore_attr("_wait_for_top_window_uia", old_wait_uia)
        _restore_attr("_focus_window_windows", old_focus_window)
        _restore_attr("_click_text_with_ocr", old_click_ocr)
        _restore_attr("_active_window_title_windows", old_active)
        _restore_attr("_verify_visual_action", old_verify)
        if snapshot is None:
            os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
        else:
            os.environ["COMPUTER_CONTROL_DRY_RUN"] = snapshot


def test_qq_click_control_falls_back_to_send_button_region_when_ocr_misses() -> None:
    snapshot = os.environ.get("COMPUTER_CONTROL_DRY_RUN")
    os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
    old_verify_supported = plugins._verify_supported
    old_wait_uia = plugins._wait_for_top_window_uia
    old_focus_window = plugins._focus_window_windows
    old_click_ocr = plugins._click_text_with_ocr
    old_click_direct = plugins._click_qq_send_button_direct
    old_active = plugins._active_window_title_windows
    old_verify = plugins._verify_visual_action
    try:
        plugins._verify_supported = lambda payload: None
        plugins._wait_for_top_window_uia = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("UIA should not run for QQ direct send button fallback")
        )
        plugins._focus_window_windows = lambda title: {
            "matched_title": "QQ",
            "active_title": "QQ",
            "verified": True,
        }
        plugins._click_text_with_ocr = lambda payload, query, button="left": (_ for _ in ()).throw(
            RuntimeError("ocr text not found: 发送")
        )
        plugins._click_qq_send_button_direct = lambda window_title, button="left", clicks=1: {
            "matched_window": "QQ",
            "active_title": "QQ",
            "focus_strategy": "qq_send_button_region_direct",
            "control_rect": [400, 500, 520, 560],
            "x": 480,
            "y": 530,
            "button": button,
            "clicks": clicks,
        }
        plugins._active_window_title_windows = lambda: "QQ"
        plugins._verify_visual_action = (
            lambda payload, default_verified=True, default_verification="", before_path=None: {
                "verified": bool(default_verified),
                "verification": default_verification,
            }
        )

        result = asyncio.run(
            plugins.plugin_desktop_click_control(
                {
                    "window": "QQ",
                    "control": "发送",
                    "control_type": "Button",
                    "fallback_to_ocr": True,
                    "ocr_text": "发送",
                }
            )
        )

        assert result["verified"] is True
        assert result["fallback_strategy"] == "qq_send_button_region_direct"
        assert result["control_title"] == "发送"
        assert result["matched_window"] == "QQ"
    finally:
        _restore_attr("_verify_supported", old_verify_supported)
        _restore_attr("_wait_for_top_window_uia", old_wait_uia)
        _restore_attr("_focus_window_windows", old_focus_window)
        _restore_attr("_click_text_with_ocr", old_click_ocr)
        _restore_attr("_click_qq_send_button_direct", old_click_direct)
        _restore_attr("_active_window_title_windows", old_active)
        _restore_attr("_verify_visual_action", old_verify)
        if snapshot is None:
            os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
        else:
            os.environ["COMPUTER_CONTROL_DRY_RUN"] = snapshot


def test_focus_qq_search_input_direct_prefers_ctrl_f_when_foreground_is_stable() -> None:
    old_focus_window = plugins._focus_window_windows
    old_sendkeys = plugins._powershell_sendkeys
    old_hotkey = plugins._hotkey_to_sendkeys
    old_matches = plugins._active_window_matches_query
    old_active_title = plugins._active_window_title_windows
    old_window_rect = plugins._window_rect_for_query_windows
    old_mouse_click = plugins._mouse_click_windows
    calls = []
    try:
        plugins._focus_window_windows = lambda title: {"matched_title": "QQ", "active_title": "QQ", "verified": True}
        plugins._powershell_sendkeys = lambda sequence, delay_ms=120: calls.append((sequence, delay_ms))
        plugins._hotkey_to_sendkeys = lambda keys: "^f"
        plugins._active_window_matches_query = lambda title: True
        plugins._active_window_title_windows = lambda: "QQ"
        plugins._window_rect_for_query_windows = lambda title: (_ for _ in ()).throw(
            AssertionError("window rect fallback should not run when ctrl+f succeeds")
        )
        plugins._mouse_click_windows = lambda x, y, button="left", clicks=1: (_ for _ in ()).throw(
            AssertionError("mouse click fallback should not run when ctrl+f succeeds")
        )

        result = plugins._focus_qq_search_input_direct("QQ")
        assert result["focus_strategy"] == "qq_search_hotkey_direct"
        assert result["hotkey"] == "ctrl+f"
        assert result["matched_window"] == "QQ"
        assert calls == [("^f", 120)]
    finally:
        _restore_attr("_focus_window_windows", old_focus_window)
        _restore_attr("_powershell_sendkeys", old_sendkeys)
        _restore_attr("_hotkey_to_sendkeys", old_hotkey)
        _restore_attr("_active_window_matches_query", old_matches)
        _restore_attr("_active_window_title_windows", old_active_title)
        _restore_attr("_window_rect_for_query_windows", old_window_rect)
        _restore_attr("_mouse_click_windows", old_mouse_click)


def test_focus_qq_search_input_direct_clicks_inside_search_rect_when_hotkey_fails() -> None:
    old_focus_window = plugins._focus_window_windows
    old_sendkeys = plugins._powershell_sendkeys
    old_hotkey = plugins._hotkey_to_sendkeys
    old_matches = plugins._active_window_matches_query
    old_window_rect = plugins._window_rect_for_query_windows
    old_mouse_click = plugins._mouse_click_windows
    try:
        plugins._focus_window_windows = lambda title: {"matched_title": "QQ", "active_title": "QQ", "verified": True}
        plugins._powershell_sendkeys = lambda sequence, delay_ms=120: None
        plugins._hotkey_to_sendkeys = lambda keys: "^f"
        plugins._active_window_matches_query = lambda title: False
        plugins._window_rect_for_query_windows = lambda title: ((0, 0, 774, 1704), "QQ")
        plugins._mouse_click_windows = lambda x, y, button="left", clicks=1: {
            "x": x,
            "y": y,
            "button": button,
            "clicks": clicks,
        }

        result = plugins._focus_qq_search_input_direct("QQ")
        assert result["input_rect"] == [58, 210, 340, 295]
        assert result["x"] == 199
        assert result["y"] == 252
    finally:
        _restore_attr("_focus_window_windows", old_focus_window)
        _restore_attr("_powershell_sendkeys", old_sendkeys)
        _restore_attr("_hotkey_to_sendkeys", old_hotkey)
        _restore_attr("_active_window_matches_query", old_matches)
        _restore_attr("_window_rect_for_query_windows", old_window_rect)
        _restore_attr("_mouse_click_windows", old_mouse_click)


def main() -> None:
    test_qq_search_fallback_bypasses_uia()
    test_qq_message_input_fallback_bypasses_uia()
    test_qq_click_control_recognizes_chat_input_groupbox_query()
    test_qq_click_control_prefers_search_result_direct_before_uia()
    test_click_qq_search_result_direct_matches_top_row_contact()
    test_click_qq_search_result_direct_falls_back_to_hotkeys_when_mouse_click_drifts()
    test_qq_click_control_prefers_ocr_fallback_before_uia()
    test_qq_click_control_falls_back_to_send_button_region_when_ocr_misses()
    test_focus_qq_search_input_direct_prefers_ctrl_f_when_foreground_is_stable()
    test_focus_qq_search_input_direct_clicks_inside_search_rect_when_hotkey_fails()
    print("computer_plugin_qq_direct_fallback_ok")


if __name__ == "__main__":
    main()
