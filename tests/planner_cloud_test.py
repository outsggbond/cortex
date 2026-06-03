from __future__ import annotations

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.computer_use.action.generator import ActionGenerator
from system.computer_use.cloud_text_generator import build_cloud_text_generator_from_env
from system.core.planner import TaskPlanner
from system.strategy.search_planner import SearchPlanner
from system.brain.world_model import WorldState


class FakeCloudModel:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = []

    def generate(self, message: str, memories=None, context=None, draft=None, raw: bool = False) -> str:
        self.calls.append(
            {
                "message": str(message),
                "memories": list(memories or []),
                "context": dict(context or {}),
                "draft": draft,
                "raw": bool(raw),
            }
        )
        return self.reply


class FakeClient:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts = []

    def available(self) -> bool:
        return True

    def generate(self, user_text: str, history) -> str:
        self.prompts.append(
            {
                "user_text": str(user_text),
                "history": list(history or []),
            }
        )
        return self.reply


class FakeWorldModel:
    def __init__(self) -> None:
        self.predictor = SimpleNamespace(
            global_model=SimpleNamespace(
                trained=False,
                confidence=lambda: 0.0,
            )
        )

    def predict_with_confidence(self, state: WorldState, plan, experiences=None):
        del plan, experiences
        return state, 0.0

    def score_plan(self, plan, state: WorldState, feedback: str = "") -> float:
        del plan, state, feedback
        return 0.5


class FakeValidator:
    def __init__(self) -> None:
        self.stats = None
        self.fm = SimpleNamespace(_resolve=lambda path: path)

    def is_valid(self, task, state: WorldState):
        del task, state
        return True, ""

    def estimate_cost_success(self, task, state: WorldState):
        del task, state
        return 1.0, 0.8

    def action_confidence(self, task, state: WorldState) -> float:
        del task, state
        return 0.7

    def is_destructive(self, task) -> bool:
        del task
        return False


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_cloud_text_generator_uses_v2_env_fallbacks() -> None:
    from system.computer_use import cloud_text_generator as cloud_mod

    keys = [
        "CLOUD_LLM_ENABLE",
        "CLOUD_LLM_MODEL",
        "V2_LLM_MODEL",
        "V2_LLM_ENDPOINT",
        "OPENAI_API_KEY",
    ]
    snapshot = {key: os.environ.get(key) for key in keys}
    fake_client = FakeClient("cloud text ok")
    old = cloud_mod.build_llm_client
    try:
        os.environ["CLOUD_LLM_ENABLE"] = "1"
        os.environ.pop("CLOUD_LLM_MODEL", None)
        os.environ["V2_LLM_MODEL"] = "gpt-4.1-mini"
        os.environ["V2_LLM_ENDPOINT"] = "https://example.test/v1/chat/completions"
        os.environ["OPENAI_API_KEY"] = "sk-demo"
        cloud_mod.build_llm_client = lambda *args, **kwargs: fake_client
        model = build_cloud_text_generator_from_env()
        assert model is not None
        out = model.generate(
            "Build a small plan.",
            memories=["remember the README"],
            context={"mode": "planner"},
            draft="step 1",
        )
        assert out == "cloud text ok"
        assert fake_client.prompts
        prompt = str(fake_client.prompts[0]["user_text"])
        assert "Build a small plan." in prompt
        assert "Relevant memories" in prompt
        assert "remember the README" in prompt
        assert '"mode": "planner"' in prompt
        assert "Draft:" in prompt
    finally:
        cloud_mod.build_llm_client = old
        _restore_env(snapshot)


def test_task_planner_uses_cloud_model_when_enabled() -> None:
    from system.core import planner as planner_mod

    keys = [
        "CLOUD_LLM_ENABLE",
        "PLANNER_USE_LLM",
    ]
    snapshot = {key: os.environ.get(key) for key in keys}
    fake_model = FakeCloudModel("PLAN 1:\n- read README.md\n- run tests\n")
    old = planner_mod.build_cloud_text_generator_from_env
    try:
        os.environ["CLOUD_LLM_ENABLE"] = "1"
        os.environ["PLANNER_USE_LLM"] = "1"
        planner_mod.build_cloud_text_generator_from_env = lambda: fake_model
        planner = TaskPlanner()
        plan = planner.plan("check the repo and verify it", model=None, memories=["last run failed"])
        assert [step.detail for step in plan.steps] == ["read README.md", "run tests"]
        assert fake_model.calls
        assert "check the repo and verify it" in str(fake_model.calls[0]["message"])
        assert fake_model.calls[0]["memories"] == ["last run failed"]
        assert fake_model.calls[0]["raw"] is True
    finally:
        planner_mod.build_cloud_text_generator_from_env = old
        _restore_env(snapshot)


def test_action_generator_uses_cloud_model_when_enabled() -> None:
    from system.computer_use.action import generator as action_mod

    keys = [
        "CLOUD_LLM_ENABLE",
        "ACTION_USE_LLM",
    ]
    snapshot = {key: os.environ.get(key) for key in keys}
    fake_model = FakeCloudModel("- read src/main.py\n- run tests\n")
    old = action_mod.build_cloud_text_generator_from_env
    try:
        os.environ["CLOUD_LLM_ENABLE"] = "1"
        os.environ["ACTION_USE_LLM"] = "1"
        action_mod.build_cloud_text_generator_from_env = lambda: fake_model
        generator = ActionGenerator(project_root=ROOT)
        state = WorldState(goal="fix the parser", errors=["traceback"])
        actions = generator.generate("fix the parser", state, memories=["failing in CI"], model=None)
        assert actions[:2] == ["read src/main.py", "run tests"]
        assert fake_model.calls
        prompt = str(fake_model.calls[0]["message"])
        assert "Goal: fix the parser" in prompt
        assert "State:" in prompt
        assert fake_model.calls[0]["memories"] == ["failing in CI"]
        assert fake_model.calls[0]["raw"] is True
    finally:
        action_mod.build_cloud_text_generator_from_env = old
        _restore_env(snapshot)


def test_search_planner_uses_shared_cloud_model_when_enabled() -> None:
    from system.computer_use.action import generator as action_mod
    from system.core import planner as planner_mod

    keys = [
        "CLOUD_LLM_ENABLE",
        "ACTION_USE_LLM",
        "PLANNER_USE_LLM",
    ]
    snapshot = {key: os.environ.get(key) for key in keys}
    fake_model = FakeCloudModel("- inspect logs\n- run tests\n")
    calls = {"count": 0}

    def _build_once():
        calls["count"] += 1
        return fake_model

    old_action = action_mod.build_cloud_text_generator_from_env
    old_planner = planner_mod.build_cloud_text_generator_from_env
    try:
        os.environ["CLOUD_LLM_ENABLE"] = "1"
        os.environ["ACTION_USE_LLM"] = "1"
        os.environ["PLANNER_USE_LLM"] = "1"
        action_mod.build_cloud_text_generator_from_env = _build_once
        planner_mod.build_cloud_text_generator_from_env = _build_once
        search = SearchPlanner(
            world_model=FakeWorldModel(),
            action_generator=ActionGenerator(project_root=ROOT),
            base_planner=TaskPlanner(),
            depth=1,
            beam_size=2,
            validator=FakeValidator(),
        )
        state = WorldState(goal="diagnose the failing pipeline")
        plan = search.plan(
            goal="diagnose the failing pipeline",
            state=state,
            memories=["CI is red"],
            model=None,
            examples=[],
            feedback="",
        )
        assert [step.detail for step in plan.steps][:1] == ["inspect logs"]
        assert calls["count"] == 1
        assert fake_model.calls
        assert fake_model.calls[0]["memories"] == ["CI is red"]
    finally:
        action_mod.build_cloud_text_generator_from_env = old_action
        planner_mod.build_cloud_text_generator_from_env = old_planner
        _restore_env(snapshot)


def main() -> None:
    test_cloud_text_generator_uses_v2_env_fallbacks()
    test_task_planner_uses_cloud_model_when_enabled()
    test_action_generator_uses_cloud_model_when_enabled()
    test_search_planner_uses_shared_cloud_model_when_enabled()
    print("planner_cloud_ok")


if __name__ == "__main__":
    main()
