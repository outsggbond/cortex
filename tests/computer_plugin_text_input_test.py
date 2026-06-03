# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.computer_use import plugins


def _restore_attr(name: str, value):
    setattr(plugins, name, value)


def test_desktop_type_text_falls_back_to_clipboard_when_sendinput_fails() -> None:
    snapshot = os.environ.get("COMPUTER_CONTROL_DRY_RUN")
    os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
    old_verify_supported = plugins._verify_supported
    old_type_text = plugins._type_text_windows
    old_sendkeys = plugins._powershell_sendkeys
    old_paste = plugins._paste_text_windows
    old_active = plugins._active_window_title_windows
    old_verify = plugins._verify_visual_action
    calls = []
    try:
        plugins._verify_supported = lambda payload: None
        plugins._type_text_windows = lambda text, delay_ms=30: (_ for _ in ()).throw(RuntimeError("SendInput failed"))
        plugins._powershell_sendkeys = lambda sequence, delay_ms=0: calls.append(("sendkeys", sequence, delay_ms))
        plugins._paste_text_windows = lambda text, delay_ms=150: calls.append(("paste", text, delay_ms))
        plugins._active_window_title_windows = lambda: "Unit Test Window"
        plugins._verify_visual_action = (
            lambda payload, default_verified=True, default_verification="", before_path=None: {
                "verified": bool(default_verified),
                "verification": default_verification,
            }
        )
        result = asyncio.run(plugins.plugin_desktop_type_text({"text": "notepad"}))
        assert result["verified"] is True
        assert result["input_strategy"] == "clipboard_paste_fallback"
        assert calls == [("paste", "notepad", 120)]
    finally:
        _restore_attr("_verify_supported", old_verify_supported)
        _restore_attr("_type_text_windows", old_type_text)
        _restore_attr("_powershell_sendkeys", old_sendkeys)
        _restore_attr("_paste_text_windows", old_paste)
        _restore_attr("_active_window_title_windows", old_active)
        _restore_attr("_verify_visual_action", old_verify)
        if snapshot is None:
            os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
        else:
            os.environ["COMPUTER_CONTROL_DRY_RUN"] = snapshot


def test_desktop_type_text_falls_back_to_sendkeys_when_sendinput_and_paste_fail() -> None:
    snapshot = os.environ.get("COMPUTER_CONTROL_DRY_RUN")
    os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
    old_verify_supported = plugins._verify_supported
    old_type_text = plugins._type_text_windows
    old_sendkeys = plugins._powershell_sendkeys
    old_paste = plugins._paste_text_windows
    old_active = plugins._active_window_title_windows
    old_verify = plugins._verify_visual_action
    calls = []
    try:
        plugins._verify_supported = lambda payload: None
        plugins._type_text_windows = lambda text, delay_ms=30: (_ for _ in ()).throw(RuntimeError("SendInput failed"))
        plugins._powershell_sendkeys = lambda sequence, delay_ms=0: calls.append(("sendkeys", sequence, delay_ms))
        plugins._paste_text_windows = (
            lambda text, delay_ms=150: (_ for _ in ()).throw(RuntimeError("paste failed"))
        )
        plugins._active_window_title_windows = lambda: "Unit Test Window"
        plugins._verify_visual_action = (
            lambda payload, default_verified=True, default_verification="", before_path=None: {
                "verified": bool(default_verified),
                "verification": default_verification,
            }
        )
        result = asyncio.run(plugins.plugin_desktop_type_text({"text": "notepad"}))
        assert result["verified"] is True
        assert result["input_strategy"] == "sendkeys_fallback"
        assert calls == [("sendkeys", "notepad", 80)]
    finally:
        _restore_attr("_verify_supported", old_verify_supported)
        _restore_attr("_type_text_windows", old_type_text)
        _restore_attr("_powershell_sendkeys", old_sendkeys)
        _restore_attr("_paste_text_windows", old_paste)
        _restore_attr("_active_window_title_windows", old_active)
        _restore_attr("_verify_visual_action", old_verify)
        if snapshot is None:
            os.environ.pop("COMPUTER_CONTROL_DRY_RUN", None)
        else:
            os.environ["COMPUTER_CONTROL_DRY_RUN"] = snapshot


def test_paste_text_windows_uses_native_shortcut() -> None:
    old_clipboard = plugins._set_clipboard_text_windows
    old_press = plugins._press_virtual_keys_windows
    old_sendkeys = plugins._powershell_sendkeys
    calls = []
    try:
        plugins._set_clipboard_text_windows = lambda text: calls.append(("clipboard", text))
        plugins._press_virtual_keys_windows = lambda keys, delay_ms=0: calls.append(("press", tuple(keys), delay_ms))
        plugins._powershell_sendkeys = (
            lambda sequence, delay_ms=0: (_ for _ in ()).throw(AssertionError("powershell sendkeys should not run"))
        )
        plugins._paste_text_windows("盛哥", delay_ms=150)
        assert calls == [("clipboard", "盛哥"), ("press", (0x11, 0x56), 150)]
    finally:
        _restore_attr("_set_clipboard_text_windows", old_clipboard)
        _restore_attr("_press_virtual_keys_windows", old_press)
        _restore_attr("_powershell_sendkeys", old_sendkeys)


def test_clear_focused_text_windows_uses_native_shortcuts() -> None:
    old_press = plugins._press_virtual_keys_windows
    old_sendkeys = plugins._powershell_sendkeys
    calls = []
    try:
        plugins._press_virtual_keys_windows = lambda keys, delay_ms=0: calls.append((tuple(keys), delay_ms))
        plugins._powershell_sendkeys = (
            lambda sequence, delay_ms=0: (_ for _ in ()).throw(AssertionError("powershell sendkeys should not run"))
        )
        plugins._clear_focused_text_windows()
        assert calls == [((0x11, 0x41), 120), ((0x08,), 0)]
    finally:
        _restore_attr("_press_virtual_keys_windows", old_press)
        _restore_attr("_powershell_sendkeys", old_sendkeys)


def main() -> None:
    test_desktop_type_text_falls_back_to_clipboard_when_sendinput_fails()
    test_desktop_type_text_falls_back_to_sendkeys_when_sendinput_and_paste_fail()
    test_paste_text_windows_uses_native_shortcut()
    test_clear_focused_text_windows_uses_native_shortcuts()
    print("computer_plugin_text_input_ok")


if __name__ == "__main__":
    main()
