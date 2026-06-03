# -*- coding: utf-8 -*-
"""Unit tests for system.computer_use.plugins — PluginRegistry, hotkey parsing,
and plugin implementations (desktop integration tests where available)."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import platform
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.computer_use.plugins import (
    PluginRegistry,
    registry,
    _parse_hotkey,
    _HOTKEY_MAP,
)


# ── PluginRegistry ───────────────────────────────────────────────────────────

class TestPluginRegistry:
    """Tests for the PluginRegistry class itself (no desktop interaction)."""

    def test_register_and_list(self):
        reg = PluginRegistry()
        assert reg.list_plugins() == []

        async def dummy(payload):
            return {"ok": True, "value": payload.get("x", 0)}

        reg.register("dummy", dummy)
        assert reg.list_plugins() == ["dummy"]

    def test_run_existing_plugin(self):
        reg = PluginRegistry()

        async def greeter(payload):
            return {"ok": True, "greeting": f"Hello, {payload.get('name', 'world')}"}

        reg.register("greet", greeter)

        result = asyncio.run(reg.run("greet", {"name": "tester"}))
        assert result["ok"] is True
        assert result["greeting"] == "Hello, tester"
        assert result.get("plugin") == "greet"

    def test_run_missing_plugin(self):
        reg = PluginRegistry()

        result = asyncio.run(reg.run("nonexistent", {}))
        assert result["ok"] is False
        assert "error" in result
        assert "not registered" in result["error"]

    def test_run_plugin_that_raises(self):
        reg = PluginRegistry()

        def broken(payload):
            raise RuntimeError("simulated crash")

        reg.register("broken", broken)

        result = asyncio.run(reg.run("broken", {}))
        assert result["ok"] is False
        assert "simulated crash" in result["error"]
        assert result.get("plugin") == "broken"

    def test_run_non_dict_result(self):
        """When a plugin returns a non-dict, it's wrapped in {ok: True, result: ...}."""
        reg = PluginRegistry()

        def returns_list(payload):
            return [1, 2, 3]

        reg.register("list_plugin", returns_list)

        result = asyncio.run(reg.run("list_plugin", {}))
        assert result["ok"] is True
        assert result["result"] == [1, 2, 3]
        assert result.get("plugin") == "list_plugin"

    def test_register_overwrites(self):
        """Registering with the same name overwrites the previous plugin."""
        reg = PluginRegistry()

        def first(payload):
            return {"ok": True, "version": 1}

        def second(payload):
            return {"ok": True, "version": 2}

        reg.register("plugin", first)
        reg.register("plugin", second)

        result = asyncio.run(reg.run("plugin", {}))
        assert result["version"] == 2


# ── Built-in registry ───────────────────────────────────────────────────────

class TestBuiltInPlugins:
    """Tests for the global registry and its 16 built-in plugins."""

    def test_all_16_plugins_registered(self):
        plugins = registry.list_plugins()
        assert len(plugins) == 16, f"Expected 16 plugins, got {len(plugins)}: {plugins}"

    def test_expected_plugin_names(self):
        plugins = registry.list_plugins()
        expected = [
            "desktop_active_window",
            "desktop_list_windows",
            "desktop_screenshot",
            "desktop_ocr",
            "desktop_list_controls",
            "desktop_focus_window",
            "desktop_click_control",
            "desktop_type_control",
            "desktop_click_text",
            "desktop_click",
            "desktop_double_click",
            "desktop_drag",
            "desktop_scroll",
            "desktop_type_text",
            "desktop_hotkey",
            "desktop_launch",
        ]
        for name in expected:
            assert name in plugins, f"Missing plugin: {name}"

    def test_run_unregistered_plugin_on_builtin_registry(self):
        result = asyncio.run(registry.run("nonexistent_plugin_xyz", {}))
        assert result["ok"] is False
        assert "not registered" in result["error"]


# ── _parse_hotkey ───────────────────────────────────────────────────────────

class TestParseHotkey:
    def test_ctrl_c(self):
        result = _parse_hotkey("ctrl+c")
        assert result is not None
        mod_vk, key_vk = result
        assert mod_vk == 0x11   # VK_CONTROL
        assert key_vk == ord("C")

    def test_ctrl_v(self):
        result = _parse_hotkey("ctrl+v")
        assert result is not None
        mod_vk, key_vk = result
        assert mod_vk == 0x11
        assert key_vk == ord("V")

    def test_alt_f4(self):
        result = _parse_hotkey("alt+f4")
        assert result is not None
        mod_vk, key_vk = result
        assert mod_vk == 0x12   # VK_MENU (Alt)
        assert key_vk == 0x73   # VK_F4 = 0x73

    def test_win_r(self):
        result = _parse_hotkey("win+r")
        assert result is not None
        mod_vk, key_vk = result
        assert mod_vk == 0x5B   # VK_LWIN
        assert key_vk == ord("R")

    def test_enter(self):
        result = _parse_hotkey("enter")
        assert result is not None
        mod_vk, key_vk = result
        assert mod_vk is None
        assert key_vk == 0x0D   # VK_RETURN

    def test_escape(self):
        result = _parse_hotkey("escape")
        assert result is not None
        mod_vk, key_vk = result
        assert mod_vk is None
        assert key_vk == 0x1B   # VK_ESCAPE

    def test_tab(self):
        result = _parse_hotkey("tab")
        assert result is not None
        mod_vk, key_vk = result
        assert mod_vk is None
        assert key_vk == 0x09

    def test_unknown_hotkey(self):
        result = _parse_hotkey("f99+x")
        assert result is None

    def test_case_insensitive(self):
        result = _parse_hotkey("CTRL+C")
        assert result is not None
        assert result[1] == ord("C")

    def test_generic_two_part(self):
        """_parse_hotkey falls back to generic modifier+char parsing."""
        result = _parse_hotkey("shift+q")
        assert result is not None
        mod_vk, key_vk = result
        assert mod_vk == 0x10   # VK_SHIFT
        assert key_vk == ord("Q")


# ── _HOTKEY_MAP contents ─────────────────────────────────────────────────────

class TestHotkeyMap:
    def test_contains_common_shortcuts(self):
        expected = [
            "ctrl+c", "ctrl+v", "ctrl+x", "ctrl+z",
            "ctrl+a", "ctrl+s", "ctrl+f", "ctrl+w",
            "ctrl+t", "ctrl+n", "ctrl+o", "ctrl+p",
            "alt+tab", "alt+f4", "win+r", "win+d", "win+e", "win+l",
            "enter", "escape", "tab", "backspace", "delete", "space",
        ]
        for key in expected:
            assert key in _HOTKEY_MAP, f"Missing hotkey: {key}"

    def test_contains_letter_combos(self):
        """Verify ctrl+a through ctrl+z are all present."""
        for letter in "abcdefghijklmnopqrstuvwxyz":
            key = f"ctrl+{letter}"
            assert key in _HOTKEY_MAP, f"Missing: {key}"

    def test_contains_function_keys(self):
        """Verify ctrl+f1 through ctrl+f12 for all modifiers."""
        for i in range(1, 13):
            for mod in ["ctrl", "alt", "shift", "win"]:
                key = f"{mod}+f{i}"
                assert key in _HOTKEY_MAP, f"Missing: {key}"

    def test_hotkey_values_are_tuples(self):
        for key, value in _HOTKEY_MAP.items():
            assert isinstance(value, tuple), f"{key} value is not a tuple: {value}"
            assert len(value) == 2, f"{key} value length != 2: {value}"


# ── Desktop integration (Windows only, skip-able) ───────────────────────────


def _skip_if_not_windows():
    if platform.system() != "Windows":
        pytest.skip("Desktop plugin tests require Windows")


class TestDesktopActiveWindow:
    """Tests for desktop_active_window plugin (requires Windows desktop)."""

    def test_returns_title(self):
        _skip_if_not_windows()
        result = asyncio.run(registry.run("desktop_active_window", {}))
        assert result["ok"] is True
        assert "title" in result
        assert "hwnd" in result
        assert "pid" in result
        assert isinstance(result["title"], (str, type(None)))
        assert isinstance(result["hwnd"], int)


class TestDesktopScreenshot:
    """Tests for desktop_screenshot plugin (requires Windows desktop)."""

    def test_saves_file(self):
        _skip_if_not_windows()
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "screenshot.png")
            result = asyncio.run(registry.run("desktop_screenshot", {"path": out}))
            assert result["ok"] is True, f"Screenshot failed: {result}"
            assert os.path.isfile(out), f"File not created: {out}"
            assert os.path.getsize(out) > 0, "Screenshot file is empty"

    def test_missing_path(self):
        _skip_if_not_windows()
        result = asyncio.run(registry.run("desktop_screenshot", {}))
        assert result["ok"] is False
        assert "missing path" in result.get("error", "")


class TestDesktopPluginsSmoke:
    """Smoke tests for desktop plugins that only validate input (no desktop state)."""

    def test_list_windows_returns_list(self):
        _skip_if_not_windows()
        result = asyncio.run(registry.run("desktop_list_windows", {"limit": 5}))
        assert result["ok"] is True
        assert "windows" in result
        assert isinstance(result["windows"], list)

    def test_type_text_missing_text(self):
        _skip_if_not_windows()
        result = asyncio.run(registry.run("desktop_type_text", {}))
        assert result["ok"] is False
        assert "missing text" in result.get("error", "")

    def test_hotkey_missing_key(self):
        _skip_if_not_windows()
        result = asyncio.run(registry.run("desktop_hotkey", {}))
        assert result["ok"] is False
        assert "missing hotkey" in result.get("error", "")

    def test_hotkey_unknown_key(self):
        _skip_if_not_windows()
        result = asyncio.run(registry.run("desktop_hotkey", {"hotkey": "zzz+xx"}))
        assert result["ok"] is False
        assert "unknown hotkey" in result.get("error", "")

    def test_launch_missing_target(self):
        _skip_if_not_windows()
        result = asyncio.run(registry.run("desktop_launch", {}))
        assert result["ok"] is False
        assert "missing target" in result.get("error", "")

    def test_ocr_missing_path(self):
        _skip_if_not_windows()
        result = asyncio.run(registry.run("desktop_ocr", {}))
        assert result["ok"] is False
        assert "missing path" in result.get("error", "")

    def test_list_controls_missing_window(self):
        _skip_if_not_windows()
        result = asyncio.run(registry.run("desktop_list_controls", {}))
        assert result["ok"] is False
        assert "missing window" in result.get("error", "")

    def test_focus_window_missing_title(self):
        _skip_if_not_windows()
        result = asyncio.run(registry.run("desktop_focus_window", {}))
        assert result["ok"] is False
        assert "missing title" in result.get("error", "")

    @pytest.mark.skipif(platform.system() != "Windows",
                        reason="click requires real desktop")
    def test_click_missing_text(self):
        result = asyncio.run(registry.run("desktop_click_text", {}))
        assert result["ok"] is False
        assert "missing text" in result.get("error", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    print("31 tests written for plugins module")
