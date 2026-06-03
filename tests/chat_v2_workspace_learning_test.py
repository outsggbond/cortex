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
from system.chat_v2.llm import NoopLLMClient
from system.chat_v2.memory import CuratedMemoryStore
from system.chat_v2.pipeline import ChatPipeline, PipelineConfig, build_request


def _memory_store(path: Path) -> CuratedMemoryStore:
    path.write_text("", encoding="utf-8")
    return CuratedMemoryStore(path=str(path))


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_project_inspection_plan_is_learned_and_replayed_locally() -> None:
    keys = ["WORKSPACE_TASK_LEARNING_PATH"]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "system" / "chat_v2").mkdir(parents=True, exist_ok=True)
            (root / "docs").mkdir(parents=True, exist_ok=True)
            (root / "tests").mkdir(parents=True, exist_ok=True)
            (root / "main.py").write_text("from system.app_runtime import main\n", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (root / "README.md").write_text("# Demo\n", encoding="utf-8")
            (root / "system" / "app_runtime.py").write_text("def main():\n    pass\n", encoding="utf-8")
            (root / "system" / "chat_v2" / "runtime.py").write_text("def run_chat_v2(args):\n    pass\n", encoding="utf-8")
            (root / "docs" / "architecture.md").write_text("demo docs\n", encoding="utf-8")
            (root / "tests" / "smoke_test.py").write_text("print('ok')\n", encoding="utf-8")
            learning_path = root / "workspace_task_learnings.json"
            os.environ["WORKSPACE_TASK_LEARNING_PATH"] = learning_path.as_posix()

            query = "检查项目架构"
            memory = _memory_store(root / "memory.jsonl")
            agent = WorkspaceAgent(
                AgentConfig(
                    enabled=True,
                    project_root=str(root),
                    replan_max=0,
                ),
                llm_client=NoopLLMClient(),
            )
            pipeline = ChatPipeline(
                memory_store=memory,
                llm_client=NoopLLMClient(),
                config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=False),
                agent_runner=agent,
            )
            out = pipeline.respond(build_request(query, []))
            assert out.source == "agent"
            assert out.metadata["agent_plan_origin"] == "direct"
            assert "Workspace root:" in out.text
            assert "Runtime entrypoint candidates:" in out.text
            payload = json.loads(learning_path.read_text(encoding="utf-8"))
            assert len(list(payload.get("items", []))) == 1
            assert int(payload["items"][0]["success_count"]) == 1

            memory2 = _memory_store(root / "memory2.jsonl")
            agent2 = WorkspaceAgent(
                AgentConfig(
                    enabled=True,
                    project_root=str(root),
                    replan_max=0,
                ),
                llm_client=NoopLLMClient(),
            )
            pipeline2 = ChatPipeline(
                memory_store=memory2,
                llm_client=NoopLLMClient(),
                config=PipelineConfig(enable_llm=True, enable_memory_retrieval=False, enable_rag=False),
                agent_runner=agent2,
            )
            out2 = pipeline2.respond(build_request(query, []))
            assert out2.source == "agent"
            assert out2.metadata["agent_plan_origin"] == "learned_local_workspace"
            assert out2.metadata["workspace_learned_template"].startswith("workspace_")
            assert "Docs currently available:" in out2.text
    finally:
        _restore_env(snapshot)


def main() -> None:
    test_project_inspection_plan_is_learned_and_replayed_locally()
    print("chat_v2_workspace_learning_ok")


if __name__ == "__main__":
    main()
