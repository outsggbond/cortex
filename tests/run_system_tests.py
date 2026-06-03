import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run(cmd):
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(res.stdout)
    print(res.stderr)
    return res.returncode


if __name__ == "__main__":
    code = 0
    code |= run([sys.executable, "tests/smoke_test.py"])
    code |= run([sys.executable, "tests/regression_test.py"])
    code |= run([sys.executable, "tests/adversarial_test.py"])
    code |= run([sys.executable, "tests/reasoning_logic_test.py"])
    code |= run([sys.executable, "tests/reasoning_benchmark_test.py"])
    code |= run([sys.executable, "tests/runtime_adaptive_test.py"])
    code |= run([sys.executable, "tests/runtime_bootstrap_local_env_test.py"])
    code |= run([sys.executable, "tests/cli_runtime_adapt_test.py"])
    code |= run([sys.executable, "tests/chat_v2_runtime_test.py"])
    code |= run([sys.executable, "tests/chat_v2_router_test.py"])
    code |= run([sys.executable, "tests/chat_v2_clarify_test.py"])
    code |= run([sys.executable, "tests/chat_v2_followup_test.py"])
    code |= run([sys.executable, "tests/chat_v2_reasoning_test.py"])
    code |= run([sys.executable, "tests/chat_v2_validation_test.py"])
    code |= run([sys.executable, "tests/chat_v2_validation_feedback_test.py"])
    code |= run([sys.executable, "tests/chat_v2_cloud_test.py"])
    code |= run([sys.executable, "tests/chat_v2_rag_test.py"])
    code |= run([sys.executable, "tests/chat_v2_agent_test.py"])
    code |= run([sys.executable, "tests/chat_v2_workspace_learning_test.py"])
    code |= run([sys.executable, "tests/computer_use_runtime_test.py"])
    code |= run([sys.executable, "tests/chat_v2_computer_agent_test.py"])
    code |= run([sys.executable, "tests/chat_v2_computer_feedback_test.py"])
    code |= run([sys.executable, "tests/chat_v2_computer_learning_test.py"])
    code |= run([sys.executable, "tests/chat_v2_qq_local_plan_test.py"])
    code |= run([sys.executable, "tests/chat_v2_computer_template_recovery_test.py"])
    code |= run([sys.executable, "tests/computer_plugin_focus_test.py"])
    code |= run([sys.executable, "tests/computer_plugin_retry_test.py"])
    code |= run([sys.executable, "tests/automation_runtime_test.py"])
    code |= run([sys.executable, "tests/planner_cloud_test.py"])
    code |= run([sys.executable, "tests/auto_iterate_cloud_test.py"])
    code |= run([sys.executable, "tests/test_app_runtime_entry.py"])
    code |= run([sys.executable, "tests/continual_debate_test.py"])
    code |= run([sys.executable, "tests/dialogue_evolver_test.py"])
    code |= run([sys.executable, "tests/hybrid_semantic_index_test.py"])
    code |= run([sys.executable, "tests/runtime_helpers_test.py"])
    code |= run([sys.executable, "tests/adapter_pipeline_test.py"])
    code |= run([sys.executable, "tests/integration/test_labyrinth.py"])
    print("all_tests_done" if code == 0 else "tests_failed")
    raise SystemExit(code)
