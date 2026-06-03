from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from system.chat_v2.llm import BaseLLMClient, build_llm_client


logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_float(*names: str, default: float) -> float:
    for name in names:
        raw = str(os.environ.get(name, "")).strip()
        if not raw:
            continue
        try:
            return float(raw)
        except Exception:
            logger.debug("cloud_text_generator: invalid float env %s=%r", name, raw)
    return float(default)


def _env_str(*names: str, default: str = "") -> str:
    for name in names:
        raw = str(os.environ.get(name, "")).strip()
        if raw:
            return raw
    return str(default or "")


@dataclass
class CloudTextGeneratorConfig:
    provider: str = "openai"
    model: str = ""
    timeout_s: float = 20.0
    base_url: str = ""
    endpoint: str = ""
    api_key: str = ""
    api_key_env: str = ""
    system_prompt: str = ""
    temperature: float = 0.2


class CloudTextGenerator:
    def __init__(self, config: CloudTextGeneratorConfig) -> None:
        self.config = config
        self.client: BaseLLMClient = build_llm_client(
            config.provider,
            config.model,
            config.timeout_s,
            base_url=config.base_url,
            endpoint=config.endpoint,
            api_key=config.api_key,
            api_key_env=config.api_key_env,
            system_prompt=config.system_prompt,
            temperature=config.temperature,
        )

    def available(self) -> bool:
        return bool(self.client and self.client.available())

    def generate(
        self,
        message: str,
        memories: List[str] | None = None,
        context: Dict[str, Any] | None = None,
        draft: str | None = None,
        raw: bool = False,
    ) -> str:
        del raw
        if not self.available():
            return ""
        prompt = self._build_prompt(message, memories=memories, context=context, draft=draft)
        return self.client.generate(prompt, [])

    @staticmethod
    def _build_prompt(
        message: str,
        *,
        memories: List[str] | None = None,
        context: Dict[str, Any] | None = None,
        draft: str | None = None,
    ) -> str:
        parts = [str(message or "").strip()]
        memory_items = [str(item).strip() for item in list(memories or []) if str(item).strip()]
        if memory_items:
            parts.append("Relevant memories:\n" + "\n".join(f"- {item}" for item in memory_items[:8]))
        if context:
            try:
                payload = json.dumps(context, ensure_ascii=False, sort_keys=True)
            except Exception:
                payload = str(context)
            parts.append("Context:\n" + payload)
        draft_text = str(draft or "").strip()
        if draft_text:
            parts.append("Draft:\n" + draft_text)
        return "\n\n".join(part for part in parts if part).strip()


def build_cloud_text_generator_from_env() -> CloudTextGenerator | None:
    if not _env_enabled("CLOUD_LLM_ENABLE"):
        return None
    config = CloudTextGeneratorConfig(
        provider=_env_str("CLOUD_LLM_PROVIDER", "V2_LLM_PROVIDER", default="openai") or "openai",
        model=_env_str("CLOUD_LLM_MODEL", "V2_LLM_MODEL", default=""),
        timeout_s=_env_float("CLOUD_LLM_TIMEOUT_S", "V2_LLM_TIMEOUT_S", default=20.0),
        base_url=_env_str(
            "CLOUD_LLM_BASE_URL",
            "V2_LLM_BASE_URL",
            default="",
        ),
        endpoint=_env_str("CLOUD_LLM_ENDPOINT", "V2_LLM_ENDPOINT", default=""),
        api_key=_env_str("CLOUD_LLM_API_KEY", "V2_LLM_API_KEY", default=""),
        api_key_env=_env_str("CLOUD_LLM_API_KEY_ENV", "V2_LLM_API_KEY_ENV", default=""),
        system_prompt=_env_str("CLOUD_LLM_SYSTEM_PROMPT", "V2_LLM_SYSTEM_PROMPT", default=""),
        temperature=_env_float("CLOUD_LLM_TEMPERATURE", "V2_LLM_TEMPERATURE", default=0.2),
    )
    model = CloudTextGenerator(config)
    if not model.available():
        logger.warning(
            "cloud_text_generator: cloud backend unavailable provider=%s model=%s target=%s",
            config.provider,
            config.model,
            config.endpoint or config.base_url or "<default>",
        )
        return None
    return model
