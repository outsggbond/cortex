# -*- coding: utf-8 -*-
"""Entry runtime for modular chat architecture (v2)."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import sys
from typing import Any, Dict, List

from system.agent.executor import UnifiedExecutor
from system.agent.planner import UnifiedPlanner
from system.brain.world_model import WorldModel
from system.computer_use.action.generator import ActionGenerator
from system.computer_use.action.stats import ActionStats
from system.computer_use.action.validator import ActionValidator
from system.knowledge.rag_store import RagStore
from system.services import ServiceRegistry
from system.strategy.search_planner import SearchPlanner

from .agent import AgentConfig, WorkspaceAgent
from .llm import NoopLLMClient, build_llm_client
from .memory import CuratedMemoryStore
from .pipeline import ChatPipeline, PipelineConfig, build_request
from .reasoner import ChatReasoner


logger = logging.getLogger(__name__)
DEFAULT_CHAT_HISTORY_LIMIT = 24


@dataclass
class ChatRuntimeComponents:
    pipeline: ChatPipeline
    memory_store: CuratedMemoryStore
    llm_client: Any
    rag_store: Any | None
    agent_runner: Any | None
    reasoning_runner: ChatReasoner
    cloud_llm: bool
    llm_provider: str
    llm_base_url: str
    llm_endpoint: str
    memory_path: str
    rag_enabled: bool
    agent_enabled: bool


def _configure_stdio_encoding() -> None:
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCP(65001)
            kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _read_console_line_windows(prompt: str, *, _getwch=None, _stdout=None) -> str:
    stdout = _stdout or sys.stdout
    getwch = _getwch
    if getwch is None:
        import msvcrt

        getwch = msvcrt.getwch
    stdout.write(prompt)
    stdout.flush()
    chars: List[str] = []
    while True:
        ch = str(getwch() or "")
        if not ch:
            continue
        if ch in {"\x00", "\xe0"}:
            try:
                getwch()
            except Exception:
                pass
            continue
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x1a":
            stdout.write("\n")
            stdout.flush()
            raise EOFError
        if ch in {"\r", "\n"}:
            stdout.write("\n")
            stdout.flush()
            return "".join(chars)
        if ch in {"\b", "\x7f"}:
            if chars:
                chars.pop()
                stdout.write("\b \b")
                stdout.flush()
            continue
        chars.append(ch)
        stdout.write(ch)
        stdout.flush()


def _read_user_message(prompt: str = "You > ") -> str:
    if os.name == "nt":
        try:
            if bool(getattr(sys.stdin, "isatty", lambda: False)()):
                return _read_console_line_windows(prompt)
        except Exception:
            logger.debug("chat_v2: wide console input failed, falling back to input()", exc_info=True)
    return input(prompt)


def _bool_option(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


def _build_agent_config(args: Any) -> AgentConfig:
    defaults = AgentConfig()
    allow_browser = bool(getattr(args, "v2_agent_allow_browser", False))
    allow_desktop = bool(getattr(args, "v2_agent_allow_desktop", False))
    computer_feedback_enabled = bool(
        bool(getattr(args, "v2_agent", False))
        and (allow_browser or allow_desktop)
    )
    computer_recovery_enabled = bool(computer_feedback_enabled)
    self_heal_create = bool(getattr(args, "self_heal_create", False))
    self_heal_chmod = bool(getattr(args, "self_heal_chmod", False))
    self_heal_enabled = _bool_option(getattr(args, "self_heal", None), defaults.self_heal)
    if not self_heal_enabled and (self_heal_create or self_heal_chmod):
        self_heal_enabled = True
    return AgentConfig(
        enabled=True,
        project_root=str(getattr(args, "v2_agent_project_root", ".") or ".").strip() or ".",
        allow_write=bool(getattr(args, "v2_agent_allow_write", False)),
        allow_exec=bool(getattr(args, "v2_agent_allow_exec", False)),
        allow_browser=allow_browser,
        allow_desktop=allow_desktop,
        enable_computer_feedback=computer_feedback_enabled,
        enable_computer_recovery=computer_recovery_enabled,
        search_plan=_bool_option(getattr(args, "search_plan", None), defaults.search_plan),
        search_depth=max(1, int(getattr(args, "search_depth", defaults.search_depth))),
        search_beam=max(1, int(getattr(args, "search_beam", defaults.search_beam))),
        planner_candidates=max(1, int(getattr(args, "planner_candidates", defaults.planner_candidates))),
        replan_max=max(0, int(getattr(args, "replan_max", defaults.replan_max))),
        self_heal=self_heal_enabled,
        self_heal_create=self_heal_create,
        self_heal_chmod=self_heal_chmod,
    )


def build_chat_runtime_components(args: Any) -> ChatRuntimeComponents:
    cloud_llm = bool(getattr(args, "cloud_llm", False))
    memory_path = str(getattr(args, "v2_memory_path", "artifacts/memory/dialogue_curated.jsonl")).strip()
    llm_provider = str(getattr(args, "v2_llm_provider", "none")).strip()
    if cloud_llm and llm_provider.lower() in {"", "none"}:
        llm_provider = "openai"
    llm_model = str(getattr(args, "v2_llm_model", "") or "").strip()
    llm_timeout_s = float(getattr(args, "v2_llm_timeout_s", 20.0))
    llm_base_url = str(getattr(args, "v2_llm_base_url", "") or "").strip()
    llm_endpoint = str(getattr(args, "v2_llm_endpoint", "") or "").strip()
    llm_api_key = str(getattr(args, "v2_llm_api_key", "") or "").strip()
    llm_api_key_env = str(getattr(args, "v2_llm_api_key_env", "") or "").strip()
    llm_system_prompt = str(getattr(args, "v2_llm_system_prompt", "") or "").strip()
    llm_temperature = float(getattr(args, "v2_llm_temperature", 0.2))
    disable_memory = bool(getattr(args, "v2_no_memory_retrieval", False))
    rag_enabled = os.environ.get("RAG_ENABLE", "1") != "0"
    if bool(getattr(args, "rag_disable", False)):
        rag_enabled = False
    if bool(getattr(args, "rag_enable", False)):
        rag_enabled = True

    memory = CuratedMemoryStore(path=memory_path)
    rag_store = None
    if rag_enabled:
        try:
            rag_store = RagStore(
                index_path=str(getattr(args, "rag_index", "artifacts/memory/rag_index.json")).strip(),
                chunk_size=int(getattr(args, "rag_chunk_size", 800)),
                chunk_overlap=int(getattr(args, "rag_chunk_overlap", 120)),
                max_files=int(getattr(args, "rag_max_files", 2000)),
            )
        except Exception as exc:
            logger.warning("RAG store init failed for chat_v2: %s", exc)
            rag_store = None
    if cloud_llm:
        llm = build_llm_client(
            provider=llm_provider,
            model=llm_model,
            timeout_s=llm_timeout_s,
            endpoint=llm_endpoint,
            base_url=llm_base_url,
            api_key=llm_api_key,
            api_key_env=llm_api_key_env,
            system_prompt=llm_system_prompt,
            temperature=llm_temperature,
        )
    else:
        llm = NoopLLMClient()
    agent_enabled = bool(getattr(args, "v2_agent", False))
    agent_runner = None
    if agent_enabled:
        agent_config = _build_agent_config(args)
        # Wire dependencies through ServiceRegistry (Phase 5 refactoring)
        agent_registry = ServiceRegistry()
        project_root = str(agent_config.project_root or ".")
        agent_registry.register("world_model", lambda: WorldModel(project_root=project_root))
        agent_registry.register("action_stats", ActionStats)
        agent_registry.register("action_generator", lambda: ActionGenerator(project_root=project_root))
        agent_registry.register("action_validator", lambda: ActionValidator(project_root=project_root, stats=agent_registry.get("action_stats")))
        agent_registry.register("task_planner", UnifiedPlanner)
        agent_registry.register("search_planner", lambda: SearchPlanner(
            world_model=agent_registry.get("world_model"),
            action_generator=agent_registry.get("action_generator"),
            base_planner=agent_registry.get("task_planner"),
            depth=max(1, int(agent_config.search_depth)),
            beam_size=max(1, int(agent_config.search_beam)),
            stats=agent_registry.get("action_stats"),
            validator=agent_registry.get("action_validator"),
            project_root=project_root,
        ))
        agent_registry.register("executor", lambda: UnifiedExecutor(project_root=project_root))
        agent_runner = WorkspaceAgent(
            agent_config,
            llm_client=llm,
            registry=agent_registry,
        )
    reasoning_runner = ChatReasoner()
    pipeline = ChatPipeline(
        memory_store=memory,
        llm_client=llm,
        rag_store=rag_store,
        config=PipelineConfig(
            enable_llm=True,
            enable_memory_retrieval=(not disable_memory),
            enable_rag=bool(rag_enabled and rag_store is not None),
            enable_validation_feedback=True,
            rag_topk=max(1, int(getattr(args, "rag_topk", 4))),
            rag_min_sim=float(getattr(args, "rag_min_sim", 0.2)),
            rag_max_chars=max(120, int(getattr(args, "rag_max_chars", 1200))),
        ),
        agent_runner=agent_runner,
        reasoning_runner=reasoning_runner,
    )
    return ChatRuntimeComponents(
        pipeline=pipeline,
        memory_store=memory,
        llm_client=llm,
        rag_store=rag_store,
        agent_runner=agent_runner,
        reasoning_runner=reasoning_runner,
        cloud_llm=cloud_llm,
        llm_provider=llm_provider,
        llm_base_url=llm_base_url,
        llm_endpoint=llm_endpoint,
        memory_path=memory_path,
        rag_enabled=bool(rag_enabled and rag_store is not None),
        agent_enabled=agent_enabled,
    )


def append_chat_history(
    history: List[Dict[str, Any]] | None,
    user_text: str,
    response: Any,
    *,
    limit: int = DEFAULT_CHAT_HISTORY_LIMIT,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = list(history or [])
    out.append({"role": "user", "text": str(user_text or "")})
    out.append(
        {
            "role": "assistant",
            "text": str(getattr(response, "text", "") or ""),
            "source": str(getattr(response, "source", "") or ""),
            "metadata": dict(getattr(response, "metadata", {}) or {}),
        }
    )
    safe_limit = max(2, int(limit or DEFAULT_CHAT_HISTORY_LIMIT))
    if len(out) > safe_limit:
        out = out[-safe_limit:]
    return out


def run_chat_v2(args: Any) -> None:
    _configure_stdio_encoding()
    if not bool(getattr(args, "chat", False)):
        raise SystemExit("--runtime-arch v2 currently supports --chat mode only.")

    components = build_chat_runtime_components(args)

    logger.info(
        "Chat v2 ready. provider=%s llm_available=%s memory_entries=%d memory_path=%s cloud=%s base_url=%s endpoint=%s rag=%s agent=%s browser=%s desktop=%s",
        components.llm_provider,
        components.llm_client.available(),
        len(components.memory_store.entries),
        components.memory_path,
        components.cloud_llm,
        components.llm_base_url or "default",
        components.llm_endpoint or "default",
        components.rag_enabled,
        components.agent_enabled,
        bool(getattr(args, "v2_agent_allow_browser", False)),
        bool(getattr(args, "v2_agent_allow_desktop", False)),
    )
    logger.info("Chat mode (v2). Type exit to quit.")

    history: List[Dict[str, Any]] = []
    while True:
        try:
            msg = _read_user_message("You > ").strip()
        except KeyboardInterrupt:
            print("\nUser interrupted. Exiting.")
            break
        except EOFError:
            break
        if not msg:
            continue
        if msg.lower() in {"exit", "quit"}:
            break

        req = build_request(msg, history)
        out = components.pipeline.respond(req)
        print(f"Agent > {out.text}")
        history = append_chat_history(history, msg, out)


__all__ = [
    "ChatRuntimeComponents",
    "DEFAULT_CHAT_HISTORY_LIMIT",
    "append_chat_history",
    "build_chat_runtime_components",
    "run_chat_v2",
]
