# -*- coding: utf-8 -*-
"""LLM client abstraction for chat runtime v2.

Provides a consistent interface over OpenAI-compatible APIs (DeepSeek, Kimi,
OpenAI, and any provider exposing a /v1/chat/completions endpoint).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean(value: Any) -> str:
    return str(value or "").strip()


def _truncate_err(exc: Exception, max_len: int = 200) -> str:
    """Truncate exception message for logging."""
    text = str(exc)
    if len(text) > max_len:
        text = text[:max_len - 3] + "..."
    return text


def _first(*values: Any) -> str:
    for v in values:
        text = _clean(v)
        if text:
            return text
    return ""


def _env(env: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        text = _clean(env.get(name, ""))
        if text:
            return text
    return _clean(default)


# ---------------------------------------------------------------------------
# well-known provider presets
# ---------------------------------------------------------------------------

_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4.1-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    "deepseek": {
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-v4-flash",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "kimi": {
        "endpoint": "https://api.moonshot.cn/v1/chat/completions",
        "model": "moonshot-v1-auto",
        "api_key_env": "MOONSHOT_API_KEY",
    },
}


def resolve_openai_compat_config(
    provider: str,
    *,
    model: str = "",
    endpoint: str = "",
    base_url: str = "",
    api_key_env: str = "",
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve provider details, filling gaps from presets and env vars."""
    env_map = env if env is not None else os.environ
    raw_provider = _clean(provider).lower()
    preset = _PROVIDER_PRESETS.get(raw_provider, {})

    resolved: dict[str, str] = {
        "provider": raw_provider if raw_provider else "openai",
        "model": "",
        "endpoint": "",
        "base_url": "",
        "api_key_env": "",
    }

    # --- api_key_env --------------------------------------------------------
    resolved["api_key_env"] = _first(
        api_key_env,
        _env(env_map, "V2_LLM_API_KEY_ENV", "CLOUD_LLM_API_KEY_ENV"),
        preset.get("api_key_env", "OPENAI_API_KEY"),
    )

    # --- model --------------------------------------------------------------
    resolved["model"] = _first(
        model,
        _env(env_map, "V2_LLM_MODEL", "CLOUD_LLM_MODEL"),
        preset.get("model", ""),
    )

    # --- endpoint -----------------------------------------------------------
    resolved["endpoint"] = _first(
        endpoint,
        _env(env_map, "V2_LLM_ENDPOINT", "CLOUD_LLM_ENDPOINT"),
        preset.get("endpoint", ""),
    )

    # --- base_url (used as fallback for endpoint) ---------------------------
    resolved["base_url"] = _first(
        base_url,
        _env(env_map, "V2_LLM_BASE_URL", "CLOUD_LLM_BASE_URL"),
    )

    # If we still have no endpoint but a base_url exists, construct from base_url
    if not resolved["endpoint"]:
        base = resolved["base_url"]
        if base:
            resolved["endpoint"] = base.rstrip("/") + "/chat/completions"

    return resolved


def build_llm_client(
    provider: str,
    model: str,
    timeout_s: float = 20.0,
    *,
    base_url: str = "",
    endpoint: str = "",
    api_key: str = "",
    api_key_env: str = "",
    system_prompt: str = "",
    temperature: float = 0.2,
    env: Mapping[str, str] | None = None,
) -> BaseLLMClient:
    """Convenience factory that resolves config then returns a concrete client."""
    resolved = resolve_openai_compat_config(
        provider,
        model=model,
        endpoint=endpoint,
        base_url=base_url,
        api_key_env=api_key_env,
        env=env,
    )
    return OpenAIChatClient(
        api_key=api_key,
        api_key_env=_clean(resolved.get("api_key_env", "") or "OPENAI_API_KEY"),
        model=_clean(resolved.get("model", "") or model),
        endpoint=_clean(resolved.get("endpoint", "") or endpoint),
        base_url=_clean(resolved.get("base_url", "") or base_url),
        timeout_s=timeout_s,
        system_prompt=system_prompt,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# abstract base
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Abstract LLM client — swap implementations behind this."""

    @abstractmethod
    def available(self) -> bool:
        """Return True if the client is configured and ready to call."""
        ...

    @abstractmethod
    def generate(self, prompt: str, history: Sequence[Any]) -> str:
        """Generate a completion for *prompt*, optionally conditioned on *history*."""
        ...


# ---------------------------------------------------------------------------
# no-op client (for testing / offline mode)
# ---------------------------------------------------------------------------

class NoopLLMClient(BaseLLMClient):
    """A client that never calls any API — always returns empty strings.

    Useful for unit tests and offline development where you don't want
    real LLM calls.
    """

    def available(self) -> bool:
        return False

    def generate(self, prompt: str, history: Sequence[Any]) -> str:
        return ""


# ---------------------------------------------------------------------------
# concrete OpenAI-compatible client
# ---------------------------------------------------------------------------

class OpenAIChatClient(BaseLLMClient):
    """Talks to any OpenAI-compatible /v1/chat/completions endpoint.

    Supports DeepSeek, Kimi, local proxies, and OpenAI itself.
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "gpt-4.1-mini",
        endpoint: str = "",
        base_url: str = "",
        timeout_s: float = 30.0,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> None:
        # resolve endpoint
        env_endpoint = _env(
            os.environ,
            "V2_LLM_ENDPOINT", "CLOUD_LLM_ENDPOINT",
            "OPENAI_API_ENDPOINT",
        )
        env_base = _env(
            os.environ,
            "V2_LLM_BASE_URL", "CLOUD_LLM_BASE_URL",
            "OPENAI_BASE_URL", "OPENAI_API_BASE_URL",
        )
        resolved_endpoint = _first(
            endpoint,
            env_endpoint,
        )
        if not resolved_endpoint and base_url:
            resolved_endpoint = base_url.rstrip("/") + "/chat/completions"
        if not resolved_endpoint and env_base:
            resolved_endpoint = env_base.rstrip("/") + "/chat/completions"
        self.endpoint = _first(
            resolved_endpoint,
            "https://api.openai.com/v1/chat/completions",
        )

        # model
        env_model = _env(os.environ, "V2_LLM_MODEL", "CLOUD_LLM_MODEL")
        self.model = _first(model, env_model, "gpt-4.1-mini")

        # Auto-detect endpoint from model name if using OpenAI default
        _model_lower = self.model.lower()
        if "deepseek" in _model_lower and "api.openai.com" in self.endpoint:
            self.endpoint = "https://api.deepseek.com/v1/chat/completions"
        elif "kimi" in _model_lower or "moonshot" in _model_lower and "api.openai.com" in self.endpoint:
            self.endpoint = "https://api.moonshot.cn/v1/chat/completions"

        # api key — auto-detect provider-specific env vars
        _provider_keys = []
        if "deepseek" in self.model.lower():
            _provider_keys.extend([os.environ.get("DEEPSEEK_API_KEY", "")])
        if "kimi" in self.model.lower() or "moonshot" in self.model.lower():
            _provider_keys.extend([os.environ.get("KIMI_API_KEY", ""), os.environ.get("MOONSHOT_API_KEY", "")])
        env_key = _first(
            os.environ.get(api_key_env, ""),
            os.environ.get("V2_LLM_API_KEY", ""),
            os.environ.get("CLOUD_LLM_API_KEY", ""),
            os.environ.get("OPENAI_API_KEY", ""),
            *_provider_keys,
        )
        self.api_key = _first(api_key, env_key)

        self.timeout_s = max(3.0, float(timeout_s))
        self.system_prompt = _clean(system_prompt)
        self.temperature = float(temperature)

    # -- BaseLLMClient interface ------------------------------------------------

    def available(self) -> bool:
        return bool(self.api_key and self.model and self.endpoint)

    def generate(self, prompt: str, history: Sequence[Any]) -> str:
        if not self.available():
            return ""
        prompt_text = _clean(prompt)
        if not prompt_text:
            return ""

        messages: List[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # fold in history
        for turn in list(history or []):
            role = str(getattr(turn, "role", "") or "").strip().lower()
            text = str(getattr(turn, "text", "") or "").strip()
            if role in ("user", "assistant", "system") and text:
                messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": prompt_text})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        body = self._request(payload)
        return self._parse_response_text(body)

    # -- request plumbing -------------------------------------------------------

    def _request(self, payload: Mapping[str, Any]) -> str:
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-Client-Request-Id": str(uuid.uuid4()),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            logger.warning("LLM HTTP %s: %s", e.code, _truncate_err(e))
            return ""
        except urllib.error.URLError as e:
            logger.warning("LLM connection failed: %s", e.reason)
            return ""
        except Exception as e:
            logger.warning("LLM request failed: %s", e)
            return ""

    @staticmethod
    def _parse_response_text(body: str) -> str:
        """Extract assistant message from a chat/completions (or responses) JSON body."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return ""

        # /v1/chat/completions shape
        choices = data.get("choices", [])
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            text = str(msg.get("content", "") or "").strip()
            if text:
                return text

        # /v1/responses shape (OpenAI Responses API)
        output = data.get("output", [])
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    for content_item in item.get("content", []) or []:
                        if isinstance(content_item, dict) and content_item.get("type") == "output_text":
                            text = str(content_item.get("text", "") or "").strip()
                            if text:
                                return text

        # fallback: search the whole object tree for an assistant content
        return _deep_find_assistant_content(data)


def _deep_find_assistant_content(data: Any, max_depth: int = 6) -> str:
    """Last-resort recursive search for assistant content in a nested dict."""
    if max_depth <= 0:
        return ""
    if isinstance(data, str):
        return ""
    if isinstance(data, dict):
        if data.get("role") == "assistant":
            return str(data.get("content", "") or "").strip()
        for _key, value in data.items():
            result = _deep_find_assistant_content(value, max_depth - 1)
            if result:
                return result
    if isinstance(data, (list, tuple)):
        for item in data:
            result = _deep_find_assistant_content(item, max_depth - 1)
            if result:
                return result
    return ""


__all__ = [
    "BaseLLMClient",
    "OpenAIChatClient",
    "resolve_openai_compat_config",
    "build_llm_client",
]
