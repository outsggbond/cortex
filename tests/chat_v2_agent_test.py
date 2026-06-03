from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.automation.executor import ExecutionResult
from system.chat_v2.agent import AgentConfig, WorkspaceAgent
from system.chat_v2.llm import NoopLLMClient
from system.chat_v2.memory import CuratedMemoryStore
from system.chat_v2.pipeline import ChatPipeline, PipelineConfig, build_request
from system.core.planner import Plan, Task
from system.chat_v2.types import ChatRequest


def _memory_store(path: Path) -> CuratedMemoryStore:
    path.write_text("", encoding="utf-8")
    return CuratedMemoryStore(path=str(path))


def test_agent_reads_workspace_file_with_repair() -> None:
    os.environ["EXEC_SNAPSHOT"] = "0"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "README.md").write_text("Agent mode can inspect workspace files.", encoding="utf-8")
        memory = _memory_store(root / "memory.jsonl")
        agent = WorkspaceAgent(
            AgentConfig(
                enabled=True,
                project_root=str(root),
                replan_max=0,
                self_heal=True,
            ),
            llm_client=NoopLLMClient(),
        )
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False),
            agent_runner=agent,
        )
        out = pipeline.respond(build_request("read docs/README.md", []))
        assert out.source == "agent"
        assert "README.md" in out.text
        assert "Agent mode can inspect workspace files." in out.text
        assert "repaired" in out.text.lower()


def test_agent_blocks_write_by_default() -> None:
    os.environ["EXEC_SNAPSHOT"] = "0"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory = _memory_store(root / "memory.jsonl")
        agent = WorkspaceAgent(
            AgentConfig(enabled=True, project_root=str(root)),
            llm_client=NoopLLMClient(),
        )
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False),
            agent_runner=agent,
        )
        out = pipeline.respond(build_request("write notes.txt -> hello", []))
        assert out.source == "agent_blocked"
        assert "read-only" in out.text.lower()
        assert "--v2-agent-allow-write" in out.text


def test_policy_reply_is_not_hijacked_by_agent() -> None:
    os.environ["EXEC_SNAPSHOT"] = "0"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        memory = _memory_store(root / "memory.jsonl")
        agent = WorkspaceAgent(
            AgentConfig(enabled=True, project_root=str(root)),
            llm_client=NoopLLMClient(),
        )
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False),
            agent_runner=agent,
        )
        out = pipeline.respond(build_request("hello", []))
        assert out.source == "policy"


class _HallucinatingLLM:
    def available(self) -> bool:
        return True

    def generate(self, prompt: str, history=None) -> str:
        return "已成功发送。"


def test_failed_execution_uses_template_summary_not_llm() -> None:
    agent = WorkspaceAgent(
        AgentConfig(enabled=True, project_root="."),
        llm_client=_HallucinatingLLM(),
    )
    plan = Plan(goal="send qq message", steps=[Task(name="plugin:desktop_click_text", detail="click text 盛哥")], levels=[])
    execution = ExecutionResult(
        completed=["focus window QQ"],
        failed=["click text 盛哥"],
        notes=["Level 1 failed, stopping."],
        step_results=[
            {
                "step": "focus window QQ",
                "name": "plugin:desktop_focus_window",
                "ok": True,
                "payload": {"title": "QQ"},
                "result": {"matched_title": "QQ", "verified": True},
            },
            {
                "step": "click text 盛哥",
                "name": "plugin:desktop_click_text",
                "ok": False,
                "payload": {"text": "盛哥"},
                "error": "ocr text not found: 盛哥",
            },
        ],
    )
    text, source = agent._compose_answer(
        ChatRequest(user_text="打开QQ给盛哥发一句睡了", history=[]),
        plan,
        execution,
        blocked=[],
        invalid=[],
        repaired=False,
    )
    assert source == "agent"
    assert "did not fully complete" in text.lower()
    assert "Failed steps:" in text
    assert "已成功发送" not in text


def main() -> None:
    test_agent_reads_workspace_file_with_repair()
    test_agent_blocks_write_by_default()
    test_policy_reply_is_not_hijacked_by_agent()
    test_failed_execution_uses_template_summary_not_llm()
    print("chat_v2_agent_ok")


if __name__ == "__main__":
    main()
