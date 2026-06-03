# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_OFFLINE_RUNTIME_ENV_DEFAULTS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
_LOCAL_ENV_CANDIDATES = (
    ".env.local",
    "config/runtime_secrets.env",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _iter_local_env_paths() -> Iterator[Path]:
    """生成器：按需生成本地环境文件的路径。"""
    root = _project_root()
    for rel in _LOCAL_ENV_CANDIDATES:
        yield root / rel


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """
    解析单行环境变量，支持行内注释。
    处理逻辑：
    1. 剥离前后空格。
    2. 忽略空行和以 # 开头的注释行。
    3. 寻找第一个 '='。
    4. 对 value 部分进行复杂的注释剥离（支持引号保护）。
    """
    clean_line = line.strip()
    if not clean_line or clean_line.startswith("#") or "=" not in clean_line:
        return None

    key_raw, val_raw = clean_line.split("=", 1)
    key = key_raw.strip()
    if not key:
        return None

    # 处理 Value 及其后的行内注释
    val_raw = val_raw.strip()
    
    if not val_raw:
        return key, ""

    # 如果值被引号包裹，我们需要找到闭合引号后的注释
    if val_raw[0] in {"'", '"'}:
        quote = val_raw[0]
        # 寻找下一个闭合引号
        end_quote_idx = val_raw.find(quote, 1)
        if end_quote_idx != -1:
            # 提取引号内的内容
            final_val = val_raw[1:end_quote_idx]
            # 引号后的内容（如果有）将被视为注释，直接忽略
            return key, final_val
        else:
            # 如果没有闭合引号，则视作普通字符串处理（或者报错，这里选择保守处理）
            val_raw = val_raw[1:]

    # 如果没有引号包裹，则直接按 # 切分，取第一部分
    final_val = val_raw.split("#")[0].strip()
    return key, final_val


def load_local_runtime_env() -> None:
    """加载本地环境文件，支持行内注释。"""
    for path in _iter_local_env_paths():
        if not path.is_file():
            continue
            
        try:
            # utf-8-sig 自动处理 BOM
            content = path.read_text(encoding="utf-8-sig")
            lines = content.splitlines()
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Skip env file {path}: {e}")
            continue
            
        for line in lines:
            result = _parse_env_line(line)
            if result:
                key, value = result
                os.environ.setdefault(key, value)


def apply_offline_runtime_defaults() -> None:
    """应用离线环境默认设置。"""
    load_local_runtime_env()
    for name, value in _OFFLINE_RUNTIME_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)


__all__ = ["apply_offline_runtime_defaults", "load_local_runtime_env"]
