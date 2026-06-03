# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.system_config import logger
from system.core.hardware import profile_hardware
from system.main_cli import parse_args
from system.knowledge.rag_store import RagStore
from system.runtime.bootstrap import apply_offline_runtime_defaults
from system.runtime.execution_engine import RuntimeExecutionEngine
from system.runtime.state_manager import RuntimeStateManager
from system.runtime.task_router import TaskRouter
from system.runtime.artifacts import (
    run_artifact_audit,
    run_artifact_maintain,
    run_artifact_prune,
    run_artifact_restore,
)
from system.runtime.support import (
    apply_runtime_env_overrides,
    configure_stdio_encoding,
    enforce_embeddings,
    env_flag,
    save_hardware,
    summarize_hardware,
)
from utils.logging import setup_logging


_UNSUPPORTED_THIN_RUNTIME_PREFIXES = (
    "--runtime-adapt",
    "--no-runtime-adapt",
    "--auto-tune",
    "--no-auto-tune",
    "--auto-tune-config",
    "--federated-auto",
    "--self-heal",
    "--search-",
    "--planner-",
    "--replan-",
    "--neo4j-",
    "--experience-",
)

# 未来如需限制 chat v2 的某些参数，可在此补充，当前为空占位元组
_UNSUPPORTED_CHAT_V2_PREFIXES: tuple[str, ...] = ()
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_runtime_path(path_text: str) -> Path:
    """将文本路径解析为绝对路径，空字符串或空白字符串会触发警告并回退到项目根目录"""
    stripped = str(path_text or "").strip()
    if not stripped:
        logger.warning("Empty path provided to _resolve_runtime_path, falling back to project root")
        return _PROJECT_ROOT
    path = Path(stripped).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _normalize_runtime_path(path_text: str) -> str:
    """将文本路径规范化为 POSIX 风格字符串，空字符串或空白字符串会触发警告并返回 '.' """
    stripped = str(path_text or "").strip()
    if not stripped:
        logger.warning("Empty path provided to _normalize_runtime_path, using '.' as default")
        return "."
    return Path(stripped).expanduser().as_posix()


def _collect_matching_flags(argv: List[str], prefixes: tuple[str, ...]) -> List[str]:
    hits: List[str] = []
    for token in argv:
        text = str(token or "").strip()
        if not text.startswith("--"):
            continue
        if any(text == prefix or text.startswith(prefix) for prefix in prefixes):
            hits.append(text)
    return hits


def _reject_unsupported_flags(args: Any, argv: List[str]) -> None:
    unsupported = _collect_matching_flags(argv, _UNSUPPORTED_THIN_RUNTIME_PREFIXES)
    if bool(getattr(args, "chat", False)):
        unsupported.extend(_collect_matching_flags(argv, _UNSUPPORTED_CHAT_V2_PREFIXES))
    if not unsupported:
        return
    flags = ", ".join(sorted(dict.fromkeys(unsupported)))
    raise SystemExit(
        "Unsupported flags for the thin main.py runtime entrypoint: "
        f"{flags}. Supported flows are --chat (v2), --computer-use-goal, automation workflow commands, "
        "--rag-ingest, --federated-train, and artifact maintenance commands."
    )


def _maybe_save_hardware_snapshot() -> None:
    try:
        hw = profile_hardware()
        save_hardware(hw)
        logger.info("Hardware profile: %s", summarize_hardware(hw))
    except Exception as exc:
        logger.debug("Hardware profiling skipped: %s", exc)


def _run_rag_ingest(args: Any) -> None:
    enforce_embeddings(True)
    rag_store = RagStore(
        index_path=args.rag_index,
        chunk_size=args.rag_chunk_size,
        chunk_overlap=args.rag_chunk_overlap,
        max_files=args.rag_max_files,
    )
    added = rag_store.ingest_paths(list(args.rag_paths or []))
    rag_store.save()
    rag_store.export_debug()
    logger.info("RAG ingest done. chunks=%d index=%s", added, args.rag_index)


def _train_adapter_script(args: Any) -> str:
    raw = str(getattr(args, "federated_script", "") or "").strip()
    script_path = raw or "scripts/train_adapter.py"
    resolved = _resolve_runtime_path(script_path)
    if not resolved.exists() or not resolved.is_file():
        raise SystemExit(f"federated_train_script_not_found: {script_path}")
    # 修复：直接返回规范化路径，移除永不会执行的 or 后备代码
    return _normalize_runtime_path(script_path)


def _resolve_federated_model_path(args: Any) -> str:
    candidate = str(getattr(args, "transformer_model", "") or "").strip()
    if candidate:
        return candidate
    candidate = str(getattr(args, "local_text_model", "") or "").strip()
    if candidate:
        return candidate
    return str(os.environ.get("TRANSFORMER_MODEL", "") or "").strip()


def _resolve_federated_train_data(args: Any) -> str:
    candidate = str(getattr(args, "train_file", "") or "").strip()
    if candidate:
        return _normalize_runtime_path(candidate)
    default_jsonl = _PROJECT_ROOT / "artifacts/memory/adapter_train.jsonl"
    if default_jsonl.exists():
        return default_jsonl.relative_to(_PROJECT_ROOT).as_posix()
    default_txt = _PROJECT_ROOT / "train.txt"
    if default_txt.exists():
        return default_txt.relative_to(_PROJECT_ROOT).as_posix()
    return ""


def _build_federated_cmd(args: Any, *, script_path_override: Optional[str] = None) -> List[str]:
    """
    构建 federated 训练命令，支持通过 script_path_override 避免重复调用 _train_adapter_script。
    """
    # 优先使用外部传入的脚本路径，否则自行解析
    script_path = script_path_override if script_path_override else _train_adapter_script(args)
    model_path = _resolve_federated_model_path(args)
    if not model_path:
        raise SystemExit(
            "federated_train_requires_model_path: set --transformer-model or TRANSFORMER_MODEL before using --federated-train"
        )

    cmd: List[str] = [sys.executable, script_path, "--federated", "--model-path", model_path]
    train_data = _resolve_federated_train_data(args)
    if train_data:
        cmd.extend(["--train-data", train_data])

    artifact_registry = str(getattr(args, "artifact_registry", "") or "").strip()
    if artifact_registry:
        cmd.extend(["--artifact-registry-path", artifact_registry])

    passthrough = [str(x) for x in list(getattr(args, "federated_forward_args", []) or []) if str(x).strip()]
    cmd.extend(passthrough)
    return cmd


def _run_federated_from_main(args: Any, *, raise_on_error: bool = False) -> Dict[str, Any]:
    # 修复：提前获取 script_path，避免通过 cmd[1] 依赖列表顺序
    script_path = _train_adapter_script(args)
    cmd = _build_federated_cmd(args, script_path_override=script_path)

    start = time.time()
    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, check=False)
    elapsed = time.time() - start
    stats = {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "command": cmd,
        "script": script_path,
        "elapsed_s": round(float(elapsed), 3),
    }
    if proc.returncode != 0 and raise_on_error:
        raise SystemExit(proc.returncode)
    return stats


def _run_artifact_actions(args: Any) -> bool:
    """
    执行 artifact 维护/审计/恢复/剪枝操作。
    显式指定 now 的操作将立即执行并返回 True；
    若无显式请求，自动维护/剪枝也会执行，并在执行后标记为“已执行”，
    最终返回 True 以避免程序继续路由至其他域。
    """
    # --- 显式维护 ----------
    if bool(getattr(args, "artifact_maintain_now", False)):
        stats = run_artifact_maintain(args, auto_mode=False)
        logger.info("Adapter artifact maintain: %s", stats)
        return True

    # --- 自动维护 ----------
    auto_maintain = bool(getattr(args, "artifact_auto_maintain", False)) or env_flag(
        "ADAPTER_ARTIFACT_AUTO_MAINTAIN",
        False,
    )
    # 用于记录是否执行了任何自动 artifact 操作
    any_auto_action = False

    if auto_maintain:
        try:
            stats = run_artifact_maintain(args, auto_mode=True)
            logger.info("Adapter artifact auto-maintain: %s", stats)
            any_auto_action = True  # 记录本次自动操作
        except Exception as exc:
            logger.warning("Adapter artifact auto-maintain failed: %s", exc)

    # --- 显式审计 ----------
    if bool(getattr(args, "artifact_audit_now", False)):
        stats = run_artifact_audit(args)
        logger.info("Adapter artifact audit: %s", stats)
        return True

    # --- 显式恢复 ----------
    if bool(getattr(args, "artifact_restore_now", False)):
        stats = run_artifact_restore(args)
        logger.info("Adapter artifact restore: %s", stats)
        return True

    # --- 显式剪枝 ----------
    explicit_prune = bool(getattr(args, "artifact_prune_now", False))
    if explicit_prune:
        stats = run_artifact_prune(args)
        logger.info("Adapter artifact prune: %s", stats)
        return True

    # --- 自动剪枝 ----------
    auto_prune = bool(getattr(args, "artifact_auto_prune", False)) or env_flag(
        "ADAPTER_ARTIFACT_AUTO_PRUNE",
        False,
    )
    if auto_prune:
        try:
            stats = run_artifact_prune(args)
            logger.info("Adapter artifact prune: %s", stats)
            any_auto_action = True  # 记录本次自动操作
        except Exception as exc:
            logger.warning("Adapter artifact prune failed: %s", exc)

    # 如果执行过任何自动操作，返回 True 以阻止后续域的路由
    return any_auto_action


def _run_chat(args: Any) -> None:
    from system.domain.chat import run_chat_domain

    if str(getattr(args, "runtime_arch", "legacy")).strip().lower() == "legacy":
        logger.info("Legacy chat runtime is retired; falling back to runtime-arch=v2 for --chat.")
        args.runtime_arch = "v2"
    if bool(getattr(args, "require_embeddings", False)):
        enforce_embeddings(True)
    run_chat_domain(args)


def _run_automation(args: Any) -> Dict[str, Any]:
    from system.domain.automation import run_automation_domain

    return run_automation_domain(args)


def _run_computer_use(args: Any) -> Dict[str, Any]:
    from system.domain.computer import run_computer_domain

    return run_computer_domain(args)


def _run_federated_task(args: Any) -> Dict[str, Any]:
    return _run_federated_from_main(args, raise_on_error=True)


def _build_runtime_engine() -> RuntimeExecutionEngine:
    engine = RuntimeExecutionEngine(
        state_manager=RuntimeStateManager(project_root=_PROJECT_ROOT.as_posix())
    )
    engine.register("chat", _run_chat)
    engine.register("computer", _run_computer_use)
    engine.register("automation", _run_automation)
    engine.register("knowledge", _run_rag_ingest)
    engine.register("training", _run_federated_task)
    return engine


def _run_api_server(args: Any) -> None:
    from system.interfaces.api.http import run_http_api_server

    host = str(getattr(args, "api_host", "127.0.0.1") or "127.0.0.1")
    port = int(getattr(args, "api_port", 8000) or 8000)
    logger.info("Starting HTTP API server on http://%s:%s", host, port)
    run_http_api_server(args, host=host, port=port)


def main() -> None:
    if sys.stderr is None:
        sys.stderr = sys.__stderr__
    apply_offline_runtime_defaults()
    configure_stdio_encoding()

    raw_argv = list(sys.argv[1:])
    args = parse_args()
    apply_runtime_env_overrides(args, argv=raw_argv)
    _reject_unsupported_flags(args, raw_argv)
    setup_logging(args.log_level)
    _maybe_save_hardware_snapshot()

    try:
        # API server mode — start HTTP server and block
        if bool(getattr(args, "api_server", False)):
            _run_api_server(args)
            return

        if _run_artifact_actions(args):
            return

        task = TaskRouter().route(args)
        if task is not None:
            result = _build_runtime_engine().execute(task, args)
            if isinstance(result, dict):
                logger.info("%s runtime finished: %s", task.task_type.capitalize(), result)
            else:
                logger.info("%s runtime finished.", task.task_type.capitalize())
            return

        raise SystemExit(
            "No supported runtime action selected. Use --chat, --computer-use-goal, "
            "--automation-scan/--automation-loop/--automation-run, --rag-ingest, "
            "--federated-train, --api-server, or an artifact maintenance flag."
        )
    except SystemExit:
        raise
    except Exception as exc:
        # 记录完整异常栈，避免丢失调试信息
        logger.exception("Runtime failed: %s", exc)
        raise SystemExit(2) from exc


__all__ = ["main"]