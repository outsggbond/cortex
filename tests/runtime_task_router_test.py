from __future__ import annotations

from types import SimpleNamespace

from system.runtime.task_router import TaskRouter


def _args(**overrides):
    data = dict(
        chat=False,
        runtime_arch="v2",
        computer_use_goal="",
        computer_use_dry_run=False,
        computer_use_target_window="",
        automation_list=False,
        automation_scan=False,
        automation_loop=False,
        automation_run=[],
        automation_poll_s=30.0,
        federated_train=False,
        federated_script="scripts/train_adapter.py",
        rag_ingest=False,
        rag_paths=[],
    )
    data.update(overrides)
    return SimpleNamespace(**data)


def test_router_prefers_computer_goal_over_automation_flags() -> None:
    task = TaskRouter().route(
        _args(
            computer_use_goal="open QQ",
            automation_scan=True,
        )
    )
    assert task is not None
    assert task.task_type == "computer"
    assert task.intent == "desktop_execution"
    assert task.goal == "open QQ"
    assert task.entrypoint == "system.domain.computer"


def test_router_builds_automation_run_task() -> None:
    task = TaskRouter().route(_args(automation_run=["workspace_inventory", "lint"]))
    assert task is not None
    assert task.task_type == "automation"
    assert task.intent == "workflow_run"
    assert task.metadata["workflow_ids"] == ["workspace_inventory", "lint"]
    assert "workspace_inventory" in task.goal


def test_router_builds_chat_task() -> None:
    task = TaskRouter().route(_args(chat=True, runtime_arch="v2"))
    assert task is not None
    assert task.task_type == "chat"
    assert task.intent == "conversation"
    assert task.entrypoint == "system.domain.chat"
    assert task.metadata["runtime_arch"] == "v2"


def test_router_returns_none_without_supported_action() -> None:
    assert TaskRouter().route(_args()) is None
