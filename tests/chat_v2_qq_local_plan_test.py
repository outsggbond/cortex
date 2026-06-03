# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.agent import AgentConfig, WorkspaceAgent
from system.chat_v2.computer_task_templates import build_template_plan, parse_template_command
from system.chat_v2.intents import ChatIntent
from system.chat_v2.types import ChatRequest
from system.brain.world_model import WorldState


CONTACT = "\u76db\u54e5"
MESSAGE = "\u665a\u5b89"
SEARCH_LABEL = "\u641c\u7d22"
SEND_LABEL = "\u53d1\u9001"
CHAT_INPUT_LABEL = "\u804a\u5929\u8f93\u5165\u533a"


class NoopLLM:
    def available(self) -> bool:
        return False


def test_natural_language_qq_send_goal_uses_local_special_plan() -> None:
    os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
    agent = WorkspaceAgent(
        AgentConfig(
            enabled=True,
            project_root=".",
            allow_desktop=True,
            enable_computer_learning=False,
        ),
        llm_client=NoopLLM(),
    )
    outcome = agent.handle(
        ChatRequest(user_text="\u6253\u5f00QQ\u7ed9\u76db\u54e5\u53d1\u4e00\u53e5\u665a\u5b89", history=[]),
        ChatIntent.TASK,
    )
    assert outcome is not None
    assert outcome.metadata.get("computer_plan_origin") == "special"
    completed = list(outcome.metadata.get("completed", []) or [])
    assert "focus window QQ" in completed
    assert f'type "{CONTACT}" into control {SEARCH_LABEL} in window QQ control type Edit clear first' in completed
    assert f"click control {CONTACT} in window QQ control type ListItem index 0" in completed
    assert f"click control {CHAT_INPUT_LABEL} in window QQ control type Group" in completed
    assert f'type "{MESSAGE}"' in completed
    assert f"click control {SEND_LABEL} in window QQ control type Button" in completed


def test_template_command_decodes_unicode_escape_arguments() -> None:
    name, args = parse_template_command(
        r'template qq_send_message contact="\u76db\u54e5" message="\u665a\u5b89"'
    ) or ("", {})
    assert name == "qq_send_message"
    assert args["contact"] == CONTACT
    assert args["message"] == MESSAGE
    plan = build_template_plan(name, args, goal="test")
    details = [step.detail for step in plan.steps]
    assert len(details) == 6
    assert details[0] == "focus window QQ"
    assert f'type "{CONTACT}" into control {SEARCH_LABEL} in window QQ control type Edit clear first' in details
    assert f"click control {CONTACT} in window QQ control type ListItem index 0" in details
    assert f"click control {CHAT_INPUT_LABEL} in window QQ control type Group" in details
    assert f'type "{MESSAGE}"' in details
    assert f"click control {SEND_LABEL} in window QQ control type Button" in details
    search_step = plan.steps[1]
    select_step = plan.steps[2]
    input_focus_step = plan.steps[3]
    message_step = plan.steps[4]
    send_step = plan.steps[5]
    assert search_step.payload.get("fallback_to_search_region") is True
    assert search_step.payload.get("fallback_to_ocr") is True
    assert search_step.payload.get("ocr_text") == SEARCH_LABEL
    assert select_step.payload.get("window") == "QQ"
    assert select_step.payload.get("control") == CONTACT
    assert select_step.payload.get("control_type") == "ListItem"
    assert select_step.payload.get("index") == 0
    assert select_step.payload.get("fallback_to_search_result") is True
    assert input_focus_step.payload.get("window") == "QQ"
    assert input_focus_step.payload.get("control") == CHAT_INPUT_LABEL
    assert input_focus_step.payload.get("control_type") == "Group"
    assert input_focus_step.payload.get("fallback_to_message_input") is True
    assert input_focus_step.payload.get("prefer_uia_message_input") is True
    assert input_focus_step.payload.get("anchor_control") == SEND_LABEL
    assert message_step.payload.get("text") == MESSAGE
    assert message_step.payload.get("verify_change") is True
    assert send_step.payload.get("window") == "QQ"
    assert send_step.payload.get("control") == SEND_LABEL
    assert send_step.payload.get("control_type") == "Button"


def test_filter_plan_preserves_sequential_levels_for_qq_template() -> None:
    agent = WorkspaceAgent(
        AgentConfig(
            enabled=True,
            project_root=".",
            allow_desktop=True,
            enable_computer_learning=False,
        ),
        llm_client=NoopLLM(),
    )
    plan = build_template_plan("qq_send_message", {"contact": CONTACT, "message": MESSAGE}, goal="test")
    filtered, blocked, invalid = agent._filter_plan(plan, WorldState(goal="test"))
    assert blocked == []
    assert invalid == []
    assert len(filtered.steps) == len(plan.steps)
    assert len(filtered.levels or []) == len(plan.steps)
    assert all(len(level) == 1 for level in list(filtered.levels or []))


def main() -> None:
    test_natural_language_qq_send_goal_uses_local_special_plan()
    test_template_command_decodes_unicode_escape_arguments()
    test_filter_plan_preserves_sequential_levels_for_qq_template()
    print("chat_v2_qq_local_plan_test: ok")


if __name__ == "__main__":
    main()
