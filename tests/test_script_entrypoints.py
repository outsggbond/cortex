from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path.as_posix())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_module_main(path: Path, module_name: str) -> None:
    module = _load_module(path, module_name)
    entry = getattr(module, "main", None)
    assert callable(entry), f"{path} has no callable main()"
    try:
        entry()
    except SystemExit as exc:
        code = 0 if exc.code is None else int(exc.code)
        assert code == 0, f"{path} exited with code {code}"


def _assert_marker(capsys, path: Path, module_name: str, marker: str) -> None:
    _run_module_main(path, module_name)
    captured = capsys.readouterr()
    assert marker in captured.out


def test_smoke_entrypoint(capsys) -> None:
    _assert_marker(capsys, ROOT / "tests" / "smoke_test.py", "legacy_smoke_test", "smoke_test_ok")


def test_labyrinth_entrypoint(capsys) -> None:
    _assert_marker(capsys, ROOT / "tests" / "integration" / "test_labyrinth.py", "legacy_labyrinth", "labyrinth_ok")


def test_chat_v2_entrypoint(capsys) -> None:
    _assert_marker(capsys, ROOT / "tests" / "chat_v2_runtime_test.py", "legacy_chat_v2_runtime", "chat_v2_runtime_ok")


def test_chat_v2_cloud_entrypoint(capsys) -> None:
    _assert_marker(capsys, ROOT / "tests" / "chat_v2_cloud_test.py", "legacy_chat_v2_cloud", "chat_v2_cloud_ok")


def test_chat_v2_rag_entrypoint(capsys) -> None:
    _assert_marker(capsys, ROOT / "tests" / "chat_v2_rag_test.py", "legacy_chat_v2_rag", "chat_v2_rag_ok")


def test_chat_v2_agent_entrypoint(capsys) -> None:
    _assert_marker(capsys, ROOT / "tests" / "chat_v2_agent_test.py", "legacy_chat_v2_agent", "chat_v2_agent_ok")


def test_automation_runtime_entrypoint(capsys) -> None:
    _assert_marker(
        capsys,
        ROOT / "tests" / "automation_runtime_test.py",
        "legacy_automation_runtime",
        "automation_runtime_ok",
    )


def test_computer_use_runtime_entrypoint(capsys) -> None:
    _assert_marker(
        capsys,
        ROOT / "tests" / "computer_use_runtime_test.py",
        "legacy_computer_use_runtime",
        "computer_use_runtime_ok",
    )


def test_planner_cloud_entrypoint(capsys) -> None:
    _assert_marker(capsys, ROOT / "tests" / "planner_cloud_test.py", "legacy_planner_cloud", "planner_cloud_ok")


def test_auto_iterate_cloud_entrypoint(capsys) -> None:
    _assert_marker(
        capsys,
        ROOT / "tests" / "auto_iterate_cloud_test.py",
        "legacy_auto_iterate_cloud",
        "auto_iterate_cloud_ok",
    )


def test_continual_debate_entrypoint(capsys) -> None:
    _assert_marker(capsys, ROOT / "tests" / "continual_debate_test.py", "legacy_continual_debate", "continual_debate_ok")


def test_dialogue_evolver_entrypoint(capsys) -> None:
    _assert_marker(capsys, ROOT / "tests" / "dialogue_evolver_test.py", "legacy_dialogue_evolver", "dialogue_evolver_ok")


def test_hybrid_semantic_index_entrypoint(capsys) -> None:
    _assert_marker(
        capsys,
        ROOT / "tests" / "hybrid_semantic_index_test.py",
        "legacy_hybrid_semantic_index",
        "hybrid_semantic_index_ok",
    )


def test_adapter_pipeline_entrypoint(capsys) -> None:
    _assert_marker(capsys, ROOT / "tests" / "adapter_pipeline_test.py", "legacy_adapter_pipeline", "adapter_pipeline_ok")
