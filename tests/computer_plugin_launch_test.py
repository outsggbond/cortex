from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.computer_use import plugins


def test_launch_target_prefers_windows_shell_for_existing_local_exe() -> None:
    original_os_name = plugins.os.name
    original_shell = plugins._launch_target_windows_shell
    original_popen = plugins.subprocess.Popen
    try:
        captured: dict[str, object] = {}

        def _shell(target: Path, args):
            captured["target"] = target
            captured["args"] = list(args)
            return {"pid": 4321, "running": True, "returncode": None}

        def _fail_popen(*args, **kwargs):
            raise AssertionError("subprocess fallback should not run for existing local exe")

        plugins.os.name = "nt"
        plugins._launch_target_windows_shell = _shell
        plugins.subprocess.Popen = _fail_popen

        with tempfile.TemporaryDirectory() as td:
            exe_path = Path(td) / "QQ.exe"
            exe_path.write_bytes(b"MZ")
            result = plugins._launch_target(exe_path.as_posix(), [])

        assert result["pid"] == 4321
        assert result["running"] is True
        assert isinstance(captured["target"], Path)
        assert Path(captured["target"]).name == "QQ.exe"
        assert captured["args"] == []
    finally:
        plugins.os.name = original_os_name
        plugins._launch_target_windows_shell = original_shell
        plugins.subprocess.Popen = original_popen


def test_plugin_desktop_launch_marks_nonzero_fast_exit_as_failed() -> None:
    original_verify_supported = plugins._verify_supported
    original_launch_target = plugins._launch_target
    try:
        plugins._verify_supported = lambda payload: None
        plugins._launch_target = lambda target, args: {"pid": 5396, "running": False, "returncode": 4294930433}
        result = __import__("asyncio").run(
            plugins.plugin_desktop_launch({"target": r"D:\\qq\\QQ.exe"})
        )
        assert result["verified"] is False
        assert result["verification"] == "process_exit"
        assert "exited early" in str(result["note"])
    finally:
        plugins._verify_supported = original_verify_supported
        plugins._launch_target = original_launch_target


def test_plugin_desktop_launch_adds_qq_failure_diagnostics() -> None:
    original_verify_supported = plugins._verify_supported
    original_launch_target = plugins._launch_target
    original_diagnose = plugins._diagnose_launch_failure_windows
    original_dry_run = plugins._dry_run
    try:
        plugins._verify_supported = lambda payload: None
        plugins._launch_target = lambda target, args: {"pid": 2036, "running": False, "returncode": 4294930433}
        plugins._diagnose_launch_failure_windows = lambda target, launch: {
            "diagnostic_code": "qq_launcher_exited_early",
            "diagnostic": "QQ launcher exited before any visible QQ window was observed.",
        }
        plugins._dry_run = lambda payload: False
        result = __import__("asyncio").run(
            plugins.plugin_desktop_launch({"target": r"D:\\qq\\QQ.exe"})
        )
        assert result["verified"] is False
        assert result["diagnostic_code"] == "qq_launcher_exited_early"
        assert "visible QQ window" in str(result["diagnostic"])
    finally:
        plugins._verify_supported = original_verify_supported
        plugins._launch_target = original_launch_target
        plugins._diagnose_launch_failure_windows = original_diagnose
        plugins._dry_run = original_dry_run


def main() -> None:
    test_launch_target_prefers_windows_shell_for_existing_local_exe()
    test_plugin_desktop_launch_marks_nonzero_fast_exit_as_failed()
    test_plugin_desktop_launch_adds_qq_failure_diagnostics()
    print("computer_plugin_launch_test: ok")


if __name__ == "__main__":
    main()
