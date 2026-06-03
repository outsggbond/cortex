from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.agent import AgentConfig, WorkspaceAgent
from system.chat_v2.llm import BaseLLMClient, NoopLLMClient
from system.chat_v2.memory import CuratedMemoryStore
from system.chat_v2.pipeline import ChatPipeline, PipelineConfig, build_request


class StructuredComputerLLM(BaseLLMClient):
    def __init__(self, plan_json: str) -> None:
        self.plan_json = plan_json
        self.prompts = []

    def available(self) -> bool:
        return True

    def generate(self, user_text: str, history) -> str:
        del history
        self.prompts.append(str(user_text))
        if "Return JSON only" in str(user_text):
            return self.plan_json
        return ""


def _memory_store(path: Path) -> CuratedMemoryStore:
    path.write_text("", encoding="utf-8")
    return CuratedMemoryStore(path=str(path))


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_llm_computer_plan_is_learned_and_replayed_locally() -> None:
    keys = [
        "EXEC_SNAPSHOT",
        "COMPUTER_CONTROL_DRY_RUN",
        "COMPUTER_CONTROL_TRACE_PATH",
        "COMPUTER_TASK_LEARNING_PATH",
    ]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["EXEC_SNAPSHOT"] = "0"
        os.environ["COMPUTER_CONTROL_DRY_RUN"] = "1"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            trace_path = root / "computer_trace.jsonl"
            learning_path = root / "computer_task_learnings.json"
            os.environ["COMPUTER_CONTROL_TRACE_PATH"] = trace_path.as_posix()
            os.environ["COMPUTER_TASK_LEARNING_PATH"] = learning_path.as_posix()
            query = "打开QQ并查看窗口控件"

            memory = _memory_store(root / "memory.jsonl")
            llm = StructuredComputerLLM('{"steps":["focus window QQ","list controls in window QQ"]}')
            agent = WorkspaceAgent(
                AgentConfig(
                    enabled=True,
                    project_root=str(root),
                    allow_desktop=True,
                    replan_max=0,
                ),
                llm_client=llm,
            )
            pipeline = ChatPipeline(
                memory_store=memory,
                llm_client=llm,
                config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False),
                agent_runner=agent,
            )
            out = pipeline.respond(build_request(query, []))
            assert out.source == "agent"
            assert "Focused window QQ" in out.text
            assert out.metadata["computer_plan_origin"] == "llm_structured"
            assert out.metadata["computer_learned_template"].startswith("learned_")
            payload = json.loads(learning_path.read_text(encoding="utf-8"))
            assert len(list(payload.get("items", []))) == 1
            assert int(payload["items"][0]["success_count"]) == 1
            template_name = str(payload["items"][0]["template_name"])

            trace_rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert [str(row.get("plugin", "")) for row in trace_rows] == ["desktop_focus_window", "desktop_list_controls"]

            trace_path.write_text("", encoding="utf-8")
            memory2 = _memory_store(root / "memory2.jsonl")
            agent2 = WorkspaceAgent(
                AgentConfig(
                    enabled=True,
                    project_root=str(root),
                    allow_desktop=True,
                    replan_max=0,
                ),
                llm_client=NoopLLMClient(),
            )
            pipeline2 = ChatPipeline(
                memory_store=memory2,
                llm_client=NoopLLMClient(),
                config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False),
                agent_runner=agent2,
            )
            out2 = pipeline2.respond(build_request(query, []))
            assert out2.source == "agent"
            assert "Listed window controls" in out2.text
            assert out2.metadata["computer_plan_origin"] == "learned_local"

            out3 = pipeline2.respond(build_request(f"template {template_name}", []))
            assert out3.source == "agent"
            assert "Focused window QQ" in out3.text
            assert "Listed window controls" in out3.text
    finally:
        _restore_env(snapshot)


def main() -> None:
    test_llm_computer_plan_is_learned_and_replayed_locally()
    print("chat_v2_computer_learning_ok")


if __name__ == "__main__":
    main()
