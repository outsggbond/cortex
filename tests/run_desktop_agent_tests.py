from __future__ import annotations

import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run(cmd: list[str]) -> int:
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(res.stdout)
    print(res.stderr)
    return int(res.returncode)


if __name__ == "__main__":
    code = 0
    code |= run([sys.executable, "tests/computer_use_runtime_test.py"])
    code |= run([sys.executable, "tests/chat_v2_computer_agent_test.py"])
    code |= run([sys.executable, "tests/chat_v2_computer_feedback_test.py"])
    code |= run([sys.executable, "tests/chat_v2_computer_learning_test.py"])
    code |= run([sys.executable, "tests/chat_v2_qq_local_plan_test.py"])
    code |= run([sys.executable, "tests/chat_v2_computer_template_recovery_test.py"])
    code |= run([sys.executable, "tests/computer_plugin_focus_test.py"])
    code |= run([sys.executable, "tests/computer_plugin_retry_test.py"])
    code |= run([sys.executable, "tests/automation_runtime_test.py"])
    print("desktop_agent_tests_ok" if code == 0 else "desktop_agent_tests_failed")
    raise SystemExit(code)
