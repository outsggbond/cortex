from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.computer_use import plugins


class _FakeWrapper:
    def __init__(self, handle: int, title: str) -> None:
        self.handle = handle
        self._title = title

    def window_text(self) -> str:
        return self._title


def test_focus_window_qq_can_match_process_name_when_chat_title_is_contact() -> None:
    original_list = plugins._list_windows_windows
    original_attempt = plugins._attempt_focus_window_windows
    original_active = plugins._active_window_handle_windows
    try:
        plugins._list_windows_windows = lambda limit=128: [
            {"handle": 101, "title": "盛哥", "pid": 9001, "process_name": "QQ.exe"},
            {"handle": 202, "title": "README.md - ai - Visual Studio Code", "pid": 42, "process_name": "Code.exe"},
        ]
        plugins._attempt_focus_window_windows = lambda hwnd, pulse_alt=False: "盛哥"
        plugins._active_window_handle_windows = lambda: 101

        result = plugins._focus_window_windows("QQ")
        assert result["verified"] is True
        assert result["matched_title"] == "盛哥"
        assert result["active_title"] == "盛哥"
        assert result["match_strategy"] == "process_name"
    finally:
        plugins._list_windows_windows = original_list
        plugins._attempt_focus_window_windows = original_attempt
        plugins._active_window_handle_windows = original_active


def test_focus_window_notepad_can_match_localized_title() -> None:
    original_list = plugins._list_windows_windows
    original_attempt = plugins._attempt_focus_window_windows
    original_active = plugins._active_window_handle_windows
    original_pid = plugins._window_pid_windows
    original_process = plugins._window_process_name_windows
    try:
        plugins._list_windows_windows = lambda limit=128: [
            {"handle": 303, "title": "无标题 - 记事本", "pid": 9101, "process_name": "Notepad.exe"},
            {"handle": 202, "title": "README.md - ai - Visual Studio Code", "pid": 42, "process_name": "Code.exe"},
        ]
        plugins._attempt_focus_window_windows = lambda hwnd, pulse_alt=False: "无标题 - 记事本"
        plugins._active_window_handle_windows = lambda: 303
        plugins._window_pid_windows = lambda hwnd: 9101 if int(hwnd) == 303 else 42
        plugins._window_process_name_windows = lambda pid: "Notepad.exe" if int(pid) == 9101 else "Code.exe"

        result = plugins._focus_window_windows("Notepad")
        assert result["verified"] is True
        assert result["matched_title"] == "无标题 - 记事本"
        assert result["active_title"] == "无标题 - 记事本"
        assert result["match_strategy"] == "title"
    finally:
        plugins._list_windows_windows = original_list
        plugins._attempt_focus_window_windows = original_attempt
        plugins._active_window_handle_windows = original_active
        plugins._window_pid_windows = original_pid
        plugins._window_process_name_windows = original_process


def test_focus_window_can_reinforce_foreground_with_app_activate() -> None:
    original_list = plugins._list_windows_windows
    original_attempt = plugins._attempt_focus_window_windows
    original_active = plugins._active_window_handle_windows
    original_title = plugins._active_window_title_windows
    original_pid = plugins._window_pid_windows
    original_process = plugins._window_process_name_windows
    original_activate = plugins._powershell_app_activate_pid_windows
    state = {"active_hwnd": 202}
    calls = []
    try:
        plugins._list_windows_windows = lambda limit=128: [
            {"handle": 101, "title": "QQ", "pid": 9001, "process_name": "QQ.exe"},
            {"handle": 202, "title": "README.md - ai - Visual Studio Code", "pid": 42, "process_name": "Code.exe"},
        ]
        plugins._attempt_focus_window_windows = lambda hwnd, pulse_alt=False: "欢迎 - ai - Visual Studio Code"
        plugins._active_window_handle_windows = lambda: state["active_hwnd"]
        plugins._active_window_title_windows = lambda: "QQ" if state["active_hwnd"] == 101 else "欢迎 - ai - Visual Studio Code"
        plugins._window_pid_windows = lambda hwnd: 9001 if int(hwnd) == 101 else 42
        plugins._window_process_name_windows = lambda pid: "QQ.exe" if int(pid) == 9001 else "Code.exe"

        def _activate(pid: int, delay_ms: int = 0) -> bool:
            calls.append((int(pid), int(delay_ms)))
            state["active_hwnd"] = 101
            return True

        plugins._powershell_app_activate_pid_windows = _activate

        result = plugins._focus_window_windows("QQ")
        assert result["verified"] is True
        assert result["matched_title"] == "QQ"
        assert result["active_title"] == "QQ"
        assert calls == [(9001, 50)]
    finally:
        plugins._list_windows_windows = original_list
        plugins._attempt_focus_window_windows = original_attempt
        plugins._active_window_handle_windows = original_active
        plugins._active_window_title_windows = original_title
        plugins._window_pid_windows = original_pid
        plugins._window_process_name_windows = original_process
        plugins._powershell_app_activate_pid_windows = original_activate


def test_wait_for_top_window_uia_can_match_process_name_when_chat_title_is_contact() -> None:
    original_loader = plugins._load_pywinauto_modules
    original_active = plugins._active_window_handle_windows
    original_pid = plugins._window_pid_windows
    original_process = plugins._window_process_name_windows
    try:
        wrappers = [
            _FakeWrapper(101, "盛哥"),
            _FakeWrapper(202, "README.md - ai - Visual Studio Code"),
        ]

        class _FakeDesktop:
            def __init__(self, backend: str = "uia") -> None:
                self.backend = backend

            def windows(self):
                return wrappers

        plugins._load_pywinauto_modules = lambda: _FakeDesktop
        plugins._active_window_handle_windows = lambda: 101
        plugins._window_pid_windows = lambda hwnd: 9001 if int(hwnd) == 101 else 42
        plugins._window_process_name_windows = lambda pid: "QQ.exe" if int(pid) == 9001 else "Code.exe"

        wrapper = plugins._wait_for_top_window_uia("QQ", timeout_s=0.1)
        assert int(wrapper.handle) == 101
    finally:
        plugins._load_pywinauto_modules = original_loader
        plugins._active_window_handle_windows = original_active
        plugins._window_pid_windows = original_pid
        plugins._window_process_name_windows = original_process


def test_wait_for_top_window_uia_can_match_notepad_localized_title() -> None:
    original_loader = plugins._load_pywinauto_modules
    original_active = plugins._active_window_handle_windows
    original_pid = plugins._window_pid_windows
    original_process = plugins._window_process_name_windows
    try:
        wrappers = [
            _FakeWrapper(303, "无标题 - 记事本"),
            _FakeWrapper(202, "README.md - ai - Visual Studio Code"),
        ]

        class _FakeDesktop:
            def __init__(self, backend: str = "uia") -> None:
                self.backend = backend

            def windows(self):
                return wrappers

        plugins._load_pywinauto_modules = lambda: _FakeDesktop
        plugins._active_window_handle_windows = lambda: 303
        plugins._window_pid_windows = lambda hwnd: 9101 if int(hwnd) == 303 else 42
        plugins._window_process_name_windows = lambda pid: "ApplicationFrameHost.exe" if int(pid) == 9101 else "Code.exe"

        wrapper = plugins._wait_for_top_window_uia("Notepad", timeout_s=0.1)
        assert int(wrapper.handle) == 303
    finally:
        plugins._load_pywinauto_modules = original_loader
        plugins._active_window_handle_windows = original_active
        plugins._window_pid_windows = original_pid
        plugins._window_process_name_windows = original_process


def test_focus_wrapper_can_verify_process_matched_window() -> None:
    original_attempt = plugins._attempt_focus_window_windows
    original_active = plugins._active_window_handle_windows
    original_pid = plugins._window_pid_windows
    original_process = plugins._window_process_name_windows
    try:
        wrapper = _FakeWrapper(101, "盛哥")
        plugins._attempt_focus_window_windows = lambda hwnd, pulse_alt=False: "盛哥"
        plugins._active_window_handle_windows = lambda: 101
        plugins._window_pid_windows = lambda hwnd: 9001
        plugins._window_process_name_windows = lambda pid: "QQ.exe"

        result = plugins._focus_wrapper_windows(wrapper, fallback_query="QQ")
        assert result["verified"] is True
        assert result["matched_title"] == "盛哥"
        assert result["active_title"] == "盛哥"
        assert result["match_strategy"] == "process_name"
    finally:
        plugins._attempt_focus_window_windows = original_attempt
        plugins._active_window_handle_windows = original_active
        plugins._window_pid_windows = original_pid
        plugins._window_process_name_windows = original_process


def main() -> None:
    test_focus_window_qq_can_match_process_name_when_chat_title_is_contact()
    test_focus_window_notepad_can_match_localized_title()
    test_focus_window_can_reinforce_foreground_with_app_activate()
    test_wait_for_top_window_uia_can_match_process_name_when_chat_title_is_contact()
    test_wait_for_top_window_uia_can_match_notepad_localized_title()
    test_focus_wrapper_can_verify_process_matched_window()
    print("computer_plugin_focus_test: ok")


if __name__ == "__main__":
    main()
