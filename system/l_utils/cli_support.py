from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Mapping

from system.chat_v2.llm import resolve_openai_compat_config


_STRING_ENV_MAP = (
    ("cloud_provider", "CLOUD_LLM_PROVIDER"),
    ("cloud_model", "CLOUD_LLM_MODEL"),
    ("cloud_base_url", "CLOUD_LLM_BASE_URL"),
    ("cloud_endpoint", "CLOUD_LLM_ENDPOINT"),
    ("cloud_api_key", "CLOUD_LLM_API_KEY"),
    ("cloud_api_key_env", "CLOUD_LLM_API_KEY_ENV"),
    ("cloud_system_prompt", "CLOUD_LLM_SYSTEM_PROMPT"),
)

_FLOAT_ENV_MAP = (
    ("cloud_temperature", "CLOUD_LLM_TEMPERATURE"),
    ("cloud_timeout_s", "CLOUD_LLM_TIMEOUT_S"),
)


def add_cloud_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cloud-llm", action="store_true")
    parser.add_argument("--cloud-provider", type=str, default="")
    parser.add_argument("--cloud-model", type=str, default="")
    parser.add_argument("--cloud-base-url", type=str, default="")
    parser.add_argument("--cloud-endpoint", type=str, default="")
    parser.add_argument("--cloud-api-key", type=str, default="")
    parser.add_argument("--cloud-api-key-env", type=str, default="")
    parser.add_argument("--cloud-system-prompt", type=str, default="")
    parser.add_argument("--cloud-temperature", type=float, default=None)
    parser.add_argument("--cloud-timeout-s", type=float, default=None)


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _cloud_requested(args: Any) -> bool:
    if bool(getattr(args, "cloud_llm", False)):
        return True
    for attr, _env_name in _STRING_ENV_MAP:
        if _clean_str(getattr(args, attr, "")):
            return True
    for attr, _env_name in _FLOAT_ENV_MAP:
        if getattr(args, attr, None) is not None:
            return True
    return False


def cloud_env_overrides(args: Any) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if _cloud_requested(args):
        overrides["CLOUD_LLM_ENABLE"] = "1"
    for attr, env_name in _STRING_ENV_MAP:
        value = _clean_str(getattr(args, attr, ""))
        if value:
            overrides[env_name] = value
    for attr, env_name in _FLOAT_ENV_MAP:
        value = getattr(args, attr, None)
        if value is not None:
            overrides[env_name] = str(value)
    return overrides


def build_cloud_env(args: Any, base_env: Mapping[str, str] | None = None) -> Dict[str, str]:
    env = dict(base_env if base_env is not None else os.environ)
    env.update(cloud_env_overrides(args))
    return env


def _env_str(env: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = _clean_str(env.get(name, ""))
        if value:
            return value
    return _clean_str(default)


def _env_bool(env: Mapping[str, str], name: str) -> bool:
    value = _clean_str(env.get(name, "")).lower()
    return value in {"1", "true", "yes", "on"}


def cloud_runtime_summary(args: Any | None = None, env: Mapping[str, str] | None = None) -> Dict[str, Any]:
    merged_env = dict(env if env is not None else os.environ)
    if args is not None:
        merged_env.update(cloud_env_overrides(args))
    enabled = _env_bool(merged_env, "CLOUD_LLM_ENABLE")
    api_key_inline = bool(_env_str(merged_env, "CLOUD_LLM_API_KEY"))
    provider = _env_str(merged_env, "CLOUD_LLM_PROVIDER", "V2_LLM_PROVIDER", default="openai")
    resolved = resolve_openai_compat_config(
        provider,
        model=_env_str(merged_env, "CLOUD_LLM_MODEL", "V2_LLM_MODEL"),
        base_url=_env_str(merged_env, "CLOUD_LLM_BASE_URL", "V2_LLM_BASE_URL"),
        endpoint=_env_str(merged_env, "CLOUD_LLM_ENDPOINT", "V2_LLM_ENDPOINT"),
        api_key_env=_env_str(merged_env, "CLOUD_LLM_API_KEY_ENV", "V2_LLM_API_KEY_ENV"),
        env=merged_env,
    )
    return {
        "enabled": bool(enabled),
        "provider": str(resolved.get("provider", "") or provider or "openai"),
        "model": str(resolved.get("model", "")),
        "base_url": str(resolved.get("base_url", "")),
        "endpoint": str(resolved.get("endpoint", "")),
        "api_key_env": str(resolved.get("api_key_env", "")),
        "api_key_inline": bool(api_key_inline),
        "system_prompt_set": bool(_env_str(merged_env, "CLOUD_LLM_SYSTEM_PROMPT", "V2_LLM_SYSTEM_PROMPT")),
        "temperature": _env_str(merged_env, "CLOUD_LLM_TEMPERATURE", "V2_LLM_TEMPERATURE"),
        "timeout_s": _env_str(merged_env, "CLOUD_LLM_TIMEOUT_S", "V2_LLM_TIMEOUT_S"),
    }
