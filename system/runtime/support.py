# -*- coding: utf-8 -*-
"""Shared startup and environment helpers for runtime entrypoints."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from config.system_config import logger


def env_flag(name: str, default: bool = False) -> bool:
    """将环境变量解析为布尔值。"""
    raw = str(os.environ.get(name, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def env_float(name: str, default: float = 0.0) -> float:
    """将环境变量解析为浮点数。"""
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def env_int(name: str, default: int = 0) -> int:
    """将环境变量解析为整数。"""
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def configure_stdio_encoding() -> None:
    """配置标准输入输出的编码为 UTF-8。"""
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCP(65001)
            kernel32.SetConsoleOutputCP(65001)
        except Exception as e:
            logger.debug("Failed to set Windows console code page: %s", e)

    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.debug("Failed to reconfigure %s encoding: %s", stream_name, e)


def enforce_embeddings(required: bool) -> None:
    """强制检查嵌入模型是否可用，不可用则退出。"""
    if not required:
        return
    try:
        from system.core.embeddings import embedding_status

        status = embedding_status()
        if not status.get("available"):
            model = status.get("model", "")
            reason = status.get("reason", "unknown")
            if model:
                logger.error("Embedding model required but unavailable. model=%s reason=%s", model, reason)
            else:
                logger.error("Embedding model required but unavailable. reason=%s", reason)
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception as e:
        logger.error("Embedding model required but validation failed: %s", e)
        raise SystemExit(2) from e


def summarize_hardware(hw: Any) -> str:
    """提取硬件信息并返回摘要字符串。"""
    try:
        cores = hw.cpu_cores
        cpu = hw.cpu_name
        freq = hw.cpu_freq_mhz
        ram = f"{hw.available_ram_gb}/{hw.total_ram_gb} GB"
        gpu = "yes" if hw.gpu_available else "no"
        return f"cpu={cpu}, cores={cores}, freq={freq}MHz, ram={ram}, gpu={gpu}"
    except Exception:
        return str(hw)


def save_hardware(hw: Any, output_path: str = "artifacts/audit/hardware.json") -> None:
    """将硬件信息保存到指定的 JSON 文件。"""
    # 尝试多种方式获取字典数据，兼容 dataclass 和普通类
    if hasattr(hw, "__dict__"):
        payload = hw.__dict__
    elif hasattr(hw, "to_dict"):
        payload = hw.to_dict()
    else:
        logger.debug("Hardware object is not serializable to dict.")
        return

    try:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save hardware audit: %s", e)


def apply_runtime_env_overrides(args: Any, argv: Sequence[str] | None = None) -> None:
    """根据运行时参数覆盖环境变量。"""
    raw_argv = list(sys.argv[1:] if argv is None else argv)

    # 基础配置映射
    mapping = {
        "policy": "AGENT_POLICY",
        "context": "AGENT_CONTEXT",
        "empathy_mode": "NANOBRAIN_EMPATHY_MODE",
        "dialogue_transformer_model": "DIALOGUE_TRANSFORMER_MODEL",
        "dialogue_transformer_device": "DIALOGUE_TRANSFORMER_DEVICE",
        "artifact_registry": "TRANSFORMER_ADAPTER_ARTIFACT_REGISTRY",
    }

    for attr, env_var in mapping.items():
        val = str(getattr(args, attr, "") or "").strip()
        if val:
            os.environ[env_var] = val

    # 特殊类型处理
    if (empathy := getattr(args, "empathy", None)) is not None:
        os.environ["NANOBRAIN_EMPATHY"] = str(empathy)

    if (max_tokens := getattr(args, "dialogue_transformer_max_tokens", None)):
        os.environ["DIALOGUE_TRANSFORMER_MAX_TOKENS"] = str(int(max_tokens))

    # Embedding 逻辑
    require_embeddings = bool(getattr(args, "require_embeddings", False))
    if require_embeddings:
        os.environ["EMBEDDING_REQUIRE"] = "1"

    if bool(getattr(args, "pure_chat", False)) and not require_embeddings:
        os.environ["DISABLE_EMBEDDINGS_MODEL"] = "1"

    # 下载与本地模式逻辑
    if bool(getattr(args, "dialogue_transformer_allow_download", False)):
        os.environ["DIALOGUE_TRANSFORMER_LOCAL_ONLY"] = "0"
    elif bool(getattr(args, "dialogue_transformer_local_only", False)):
        os.environ["DIALOGUE_TRANSFORMER_LOCAL_ONLY"] = "1"

    # 周期逻辑：如果是聊天模式且未提供训练文件，且没有手动指定 --cycles，则强制 cycles 为 0
    if bool(getattr(args, "chat", False)) and not str(getattr(args, "train_file", "") or "").strip():
        if "--cycles" not in raw_argv:
            args.cycles = 0
