from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system import app_runtime


@contextmanager
def _patched_attr(obj, name: str, value) -> Iterator[None]:
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextmanager
def _patched_argv(argv: list[str]) -> Iterator[None]:
    old = list(sys.argv)
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = old


@contextmanager
def _patched_cwd(path: str) -> Iterator[None]:
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _base_args(**overrides):
    data = dict(
        log_level=None,
        chat=False,
        pure_chat=False,
        require_embeddings=False,
        rag_ingest=False,
        rag_index="artifacts/memory/rag_index.json",
        rag_paths=[],
        rag_chunk_size=800,
        rag_chunk_overlap=120,
        rag_max_files=2000,
        federated_train=False,
        federated_script="scripts/train_adapter.py",
        federated_forward_args=[],
        transformer_model="local/base-model",
        local_text_model="",
        train_file="artifacts/memory/adapter_train.jsonl",
        runtime_arch="v2",
        artifact_registry="",
        artifact_maintain_now=False,
        artifact_auto_maintain=False,
        artifact_audit_now=False,
        artifact_restore_now=False,
        artifact_auto_prune=False,
        artifact_prune_now=False,
        automation_list=False,
        automation_scan=False,
        automation_loop=False,
        automation_run=[],
        automation_config="config/automation_workflows.json",
        automation_state="artifacts/audit/automation_state.json",
        automation_runs="artifacts/audit/automation_runs.jsonl",
        automation_project_root=".",
        automation_poll_s=30.0,
        automation_max_cycles=0,
        computer_use_goal="",
        computer_use_project_root=".",
        computer_use_max_steps=8,
        computer_use_grid_rows=8,
        computer_use_grid_cols=12,
        computer_use_step_delay_s=0.25,
        computer_use_target_window="",
        computer_use_trace_root="artifacts/audit/computer_use",
        computer_use_system_prompt="",
        computer_use_image_detail="auto",
        computer_use_dry_run=False,
        policy="",
        context="",
        empathy=None,
        empathy_mode="offset",
        dialogue_transformer_model="",
        dialogue_transformer_device="",
        dialogue_transformer_max_tokens=None,
        dialogue_transformer_allow_download=False,
        dialogue_transformer_local_only=False,
        cycles=3,
    )
    data.update(overrides)
    return SimpleNamespace(**data)


class _NoopStateManager:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def start(self, task):
        return task

    def succeed(self, task, result):
        return result

    def fail(self, task, error):
        return error


def test_build_federated_cmd_preserves_registry_and_passthrough() -> None:
    args = _base_args(
        federated_forward_args=["--fed-rounds", "3", "--fed-num-clients", "2"],
        artifact_registry="artifacts/checkpoints/custom_registry.json",
    )
    cmd = app_runtime._build_federated_cmd(args)
    text = " ".join(cmd)
    assert cmd[:4] == [sys.executable, "scripts/train_adapter.py", "--federated", "--model-path"]
    assert "local/base-model" in text
    assert "--artifact-registry-path" in cmd
    assert "artifacts/checkpoints/custom_registry.json" in cmd
    assert cmd[-4:] == ["--fed-rounds", "3", "--fed-num-clients", "2"]


def test_reject_unsupported_flags() -> None:
    args = _base_args(chat=True)
    try:
        app_runtime._reject_unsupported_flags(
            args,
            ["--chat", "--runtime-adapt", "--runtime-adapt-bandit"],
        )
    except SystemExit as exc:
        msg = str(exc)
        assert "--runtime-adapt" in msg
        assert "--runtime-adapt-bandit" in msg
    else:
        raise AssertionError("expected unsupported flag rejection")


def test_run_chat_falls_back_to_v2() -> None:
    calls: list[str] = []

    def _fake_run_chat_v2(args) -> None:
        calls.append(str(getattr(args, "runtime_arch", "")))

    import system.chat_v2 as chat_v2

    args = _base_args(chat=True, runtime_arch="legacy")
    with _patched_attr(chat_v2, "run_chat_v2", _fake_run_chat_v2):
        app_runtime._run_chat(args)
    assert calls == ["v2"]
    assert str(args.runtime_arch) == "v2"


def test_main_dispatches_rag_ingest() -> None:
    args = _base_args(rag_ingest=True)
    calls: list[str] = []

    def _record_offline() -> None:
        calls.append("offline")

    def _record_env(*_a, **_k) -> None:
        calls.append("env")

    def _record_setup(*_a, **_k) -> None:
        calls.append("setup")

    def _record_hw() -> None:
        calls.append("hw")

    def _record_artifacts(_args) -> bool:
        calls.append("artifact")
        return False

    def _record_rag(_args) -> None:
        calls.append("rag")

    def _fail(*_a, **_k):
        raise AssertionError("unexpected path")

    with _patched_argv(["main.py", "--rag-ingest"]):
        with _patched_attr(app_runtime, "apply_offline_runtime_defaults", _record_offline):
            with _patched_attr(app_runtime, "parse_args", lambda: args):
                with _patched_attr(app_runtime, "apply_runtime_env_overrides", _record_env):
                    with _patched_attr(app_runtime, "setup_logging", _record_setup):
                        with _patched_attr(app_runtime, "_maybe_save_hardware_snapshot", _record_hw):
                            with _patched_attr(app_runtime, "RuntimeStateManager", _NoopStateManager):
                                with _patched_attr(app_runtime, "_run_artifact_actions", _record_artifacts):
                                    with _patched_attr(app_runtime, "_run_rag_ingest", _record_rag):
                                        with _patched_attr(app_runtime, "_run_chat", _fail):
                                            with _patched_attr(app_runtime, "_run_federated_from_main", _fail):
                                                app_runtime.main()
    assert calls == ["offline", "env", "setup", "hw", "artifact", "rag"]


def test_main_dispatches_automation_runtime() -> None:
    args = _base_args(automation_scan=True)
    calls: list[str] = []

    def _record_offline() -> None:
        calls.append("offline")

    def _record_env(*_a, **_k) -> None:
        calls.append("env")

    def _record_setup(*_a, **_k) -> None:
        calls.append("setup")

    def _record_hw() -> None:
        calls.append("hw")

    def _record_artifacts(_args) -> bool:
        calls.append("artifact")
        return False

    def _record_automation(_args):
        calls.append("automation")
        return {"ok": True}

    def _fail(*_a, **_k):
        raise AssertionError("unexpected path")

    with _patched_argv(["main.py", "--automation-scan"]):
        with _patched_attr(app_runtime, "apply_offline_runtime_defaults", _record_offline):
            with _patched_attr(app_runtime, "parse_args", lambda: args):
                with _patched_attr(app_runtime, "apply_runtime_env_overrides", _record_env):
                    with _patched_attr(app_runtime, "setup_logging", _record_setup):
                        with _patched_attr(app_runtime, "_maybe_save_hardware_snapshot", _record_hw):
                            with _patched_attr(app_runtime, "RuntimeStateManager", _NoopStateManager):
                                with _patched_attr(app_runtime, "_run_artifact_actions", _record_artifacts):
                                    with _patched_attr(app_runtime, "_run_automation", _record_automation):
                                        with _patched_attr(app_runtime, "_run_rag_ingest", _fail):
                                            with _patched_attr(app_runtime, "_run_chat", _fail):
                                                with _patched_attr(app_runtime, "_run_federated_from_main", _fail):
                                                    app_runtime.main()
    assert calls == ["offline", "env", "setup", "hw", "artifact", "automation"]


def test_main_dispatches_computer_use_runtime() -> None:
    args = _base_args(computer_use_goal="open QQ and send hello")
    calls: list[str] = []

    def _record_offline() -> None:
        calls.append("offline")

    def _record_env(*_a, **_k) -> None:
        calls.append("env")

    def _record_setup(*_a, **_k) -> None:
        calls.append("setup")

    def _record_hw() -> None:
        calls.append("hw")

    def _record_artifacts(_args) -> bool:
        calls.append("artifact")
        return False

    def _record_computer_use(_args):
        calls.append("computer_use")
        return {"ok": True}

    def _fail(*_a, **_k):
        raise AssertionError("unexpected path")

    with _patched_argv(["main.py", "--computer-use-goal", "open QQ and send hello"]):
        with _patched_attr(app_runtime, "apply_offline_runtime_defaults", _record_offline):
            with _patched_attr(app_runtime, "parse_args", lambda: args):
                with _patched_attr(app_runtime, "apply_runtime_env_overrides", _record_env):
                    with _patched_attr(app_runtime, "setup_logging", _record_setup):
                        with _patched_attr(app_runtime, "_maybe_save_hardware_snapshot", _record_hw):
                            with _patched_attr(app_runtime, "RuntimeStateManager", _NoopStateManager):
                                with _patched_attr(app_runtime, "_run_artifact_actions", _record_artifacts):
                                    with _patched_attr(app_runtime, "_run_computer_use", _record_computer_use):
                                        with _patched_attr(app_runtime, "_run_automation", _fail):
                                            with _patched_attr(app_runtime, "_run_rag_ingest", _fail):
                                                with _patched_attr(app_runtime, "_run_chat", _fail):
                                                    with _patched_attr(app_runtime, "_run_federated_from_main", _fail):
                                                        app_runtime.main()
    assert calls == ["offline", "env", "setup", "hw", "artifact", "computer_use"]


def test_resolve_federated_train_data_uses_project_root_defaults() -> None:
    args = _base_args(train_file="")
    with tempfile.TemporaryDirectory() as td_root:
        with tempfile.TemporaryDirectory() as td_cwd:
            root = Path(td_root)
            data_file = root / "artifacts" / "memory" / "adapter_train.jsonl"
            data_file.parent.mkdir(parents=True, exist_ok=True)
            data_file.write_text('{"text":"hello"}\n', encoding="utf-8")
            with _patched_attr(app_runtime, "_PROJECT_ROOT", root):
                with _patched_cwd(td_cwd):
                    train_data = app_runtime._resolve_federated_train_data(args)
    assert train_data == "artifacts/memory/adapter_train.jsonl"


def test_run_artifact_actions_raises_for_explicit_prune_failure() -> None:
    args = _base_args(artifact_prune_now=True)

    def _boom(_args) -> None:
        raise RuntimeError("prune failed")

    with _patched_attr(app_runtime, "run_artifact_prune", _boom):
        try:
            app_runtime._run_artifact_actions(args)
        except RuntimeError as exc:
            assert "prune failed" in str(exc)
        else:
            raise AssertionError("expected explicit prune failure")


def test_main_short_circuits_artifact_actions() -> None:
    args = _base_args(artifact_maintain_now=True)
    calls: list[str] = []

    def _record(name: str):
        def _inner(*_a, **_k):
            calls.append(name)
            if name == "artifact":
                return True
            return None

        return _inner

    def _fail(*_a, **_k):
        raise AssertionError("unexpected path")

    with _patched_argv(["main.py", "--artifact-maintain-now"]):
        with _patched_attr(app_runtime, "parse_args", lambda: args):
            with _patched_attr(app_runtime, "apply_runtime_env_overrides", _record("env")):
                with _patched_attr(app_runtime, "setup_logging", _record("setup")):
                    with _patched_attr(app_runtime, "_maybe_save_hardware_snapshot", _record("hw")):
                        with _patched_attr(app_runtime, "_run_artifact_actions", _record("artifact")):
                            with _patched_attr(app_runtime, "_run_rag_ingest", _fail):
                                with _patched_attr(app_runtime, "_run_chat", _fail):
                                    with _patched_attr(app_runtime, "_run_federated_from_main", _fail):
                                        app_runtime.main()
    assert calls == ["env", "setup", "hw", "artifact"]


def main() -> None:
    test_build_federated_cmd_preserves_registry_and_passthrough()
    test_reject_unsupported_flags()
    test_run_chat_falls_back_to_v2()
    test_main_dispatches_rag_ingest()
    test_main_dispatches_automation_runtime()
    test_main_dispatches_computer_use_runtime()
    test_resolve_federated_train_data_uses_project_root_defaults()
    test_run_artifact_actions_raises_for_explicit_prune_failure()
    test_main_short_circuits_artifact_actions()
    print("app_runtime_entry_ok")


if __name__ == "__main__":
    main()
