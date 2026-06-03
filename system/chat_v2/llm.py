# -*- coding: utf-8 -*-
"""LLM client abstraction for chat runtime v2.

Provides a consistent interface over OpenAI-compatible APIs (DeepSeek, Kimi,
OpenAI, and any provider exposing a /v1/chat/completions endpoint).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ── Optional requests library (connection pooling, better retry support) ──────
try:
    import requests as _requests_lib
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    logger.debug("requests not installed; falling back to urllib. "
                 "Install requests for connection pooling: pip install requests")


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
        max_retries=3,
        cache_ttl_s=300.0,
    )


# ---------------------------------------------------------------------------
# simple in-memory response cache (TTL-based, thread-safe)
# ---------------------------------------------------------------------------

class _LLMCache:
    """Thread-safe LRU-ish cache for LLM responses with TTL."""

    def __init__(self, max_entries: int = 256, ttl_s: float = 300.0) -> None:
        self._store: Dict[str, Tuple[float, str]] = {}
        self._lock = Lock()
        self.max_entries = max(1, int(max_entries))
        self.ttl_s = max(0.0, float(ttl_s))

    def _key(self, model: str, payload: Mapping[str, Any]) -> str:
        raw = json.dumps({"model": model, "messages": payload.get("messages", [])},
                         sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, model: str, payload: Mapping[str, Any]) -> Optional[str]:
        if self.ttl_s <= 0:
            return None
        key = self._key(model, payload)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, text = entry
            if time.time() - ts > self.ttl_s:
                del self._store[key]
                return None
            return text

    def set(self, model: str, payload: Mapping[str, Any], text: str) -> None:
        if self.ttl_s <= 0 or not text:
            return
        key = self._key(model, payload)
        with self._lock:
            if len(self._store) >= self.max_entries:
                # Evict oldest entry
                oldest_key = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest_key]
            self._store[key] = (time.time(), str(text))

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ── Shared cache instance (TTL=5min, up to 256 entries) ──────────────────────
_llm_cache = _LLMCache(max_entries=256, ttl_s=300.0)


def clear_llm_cache() -> None:
    """Clear the shared LLM response cache."""
    _llm_cache.clear()


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
    Features: retry with exponential backoff, response caching, rate-limit awareness,
    and optional model fallback chain.
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
        # ── resilience knobs ──
        max_retries: int = 3,
        retry_base_s: float = 0.5,
        retry_max_s: float = 15.0,
        cache_ttl_s: float = 300.0,
        fallback_models: Optional[List[str]] = None,
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

        # ── resilience state ──
        self.max_retries = max(0, int(max_retries))
        self.retry_base_s = max(0.1, float(retry_base_s))
        self.retry_max_s = max(self.retry_base_s, float(retry_max_s))
        self.cache_ttl_s = float(cache_ttl_s)
        self.fallback_models: List[str] = [
            m for m in (fallback_models or []) if _clean(m)
        ]
        self._rate_limit_until: float = 0.0

    # -- BaseLLMClient interface ------------------------------------------------

    def available(self) -> bool:
        return bool(self.api_key and self.model and self.endpoint)

    def _build_messages(self, prompt: str, history: Sequence[Any]) -> List[dict[str, Any]]:
        messages: List[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for turn in list(history or []):
            role = str(getattr(turn, "role", "") or "").strip().lower()
            text = str(getattr(turn, "text", "") or "").strip()
            if role in ("user", "assistant", "system") and text:
                messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": _clean(prompt)})
        return messages

    def _build_payload(self, model: str, messages: List[dict[str, Any]]) -> Dict[str, Any]:
        return {"model": model, "messages": messages, "temperature": self.temperature}

    def generate(self, prompt: str, history: Sequence[Any]) -> str:
        if not self.available():
            return ""
        prompt_text = _clean(prompt)
        if not prompt_text:
            return ""

        messages = self._build_messages(prompt_text, history)

        # ── Try primary model with retry ──
        models_to_try = [self.model] + self.fallback_models
        for attempt_idx, model_name in enumerate(models_to_try):
            payload = self._build_payload(model_name, messages)

            # Check cache
            cached = _llm_cache.get(model_name, payload)
            if cached is not None:
                logger.debug("LLM cache hit for model=%s", model_name)
                return cached

            body = self._request_with_retry(payload, model_name)
            if body:
                text = self._parse_response_text(body)
                if text:
                    _llm_cache.set(model_name, payload, text)
                    return text

            if attempt_idx < len(models_to_try) - 1:
                logger.info("LLM falling back: %s -> %s", model_name,
                            models_to_try[attempt_idx + 1])

        logger.warning("LLM generate: all models exhausted, returning empty")
        return ""

    # -- request plumbing with retry --------------------------------------------

    def _request_with_retry(self, payload: Mapping[str, Any],
                            model_name: str = "") -> str:
        """Send request with exponential backoff retry and rate-limit awareness."""
        last_error = ""
        for attempt in range(self.max_retries + 1):
            # Respect rate-limit headers from previous responses
            now = time.time()
            if now < self._rate_limit_until:
                wait = self._rate_limit_until - now
                logger.debug("LLM rate-limited, waiting %.1fs", wait)
                time.sleep(wait)

            body, error, retry_after = self._request_once(payload)
            if body:
                return body

            last_error = error

            # If server told us to retry-after, respect it
            if retry_after is not None and retry_after > 0:
                self._rate_limit_until = time.time() + retry_after

            if attempt >= self.max_retries:
                break

            # Exponential backoff with jitter
            delay = min(self.retry_max_s,
                        self.retry_base_s * (2 ** attempt))
            jitter = delay * 0.25 * (hash(str(time.time())) % 1000 / 1000.0)
            delay += jitter
            logger.debug("LLM retry %d/%d in %.1fs: %s",
                         attempt + 1, self.max_retries, delay,
                         _truncate_err(Exception(last_error)) if last_error else "unknown")
            time.sleep(delay)

        logger.warning("LLM request failed after %d attempts: %s",
                       self.max_retries + 1, last_error or "unknown")
        return ""

    def _request_once(self, payload: Mapping[str, Any]) -> Tuple[str, str, Optional[float]]:
        """Make one HTTP request. Returns (body, error, retry_after_seconds)."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-Client-Request-Id": str(uuid.uuid4()),
        }

        # ── Try requests (connection pooling) first ──
        if _HAS_REQUESTS:
            try:
                resp = _requests_lib.post(
                    self.endpoint,
                    json=dict(payload),
                    headers=headers,
                    timeout=self.timeout_s,
                )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After",
                                     resp.headers.get("x-ratelimit-reset-after", "5")))
                    return "", f"rate_limited_429", retry_after
                if resp.status_code >= 500:
                    return "", f"server_error_{resp.status_code}", None
                if resp.status_code >= 400:
                    return "", f"client_error_{resp.status_code}: {_truncate_err(Exception(resp.text))}", None
                return resp.text, "", None
            except Exception as e:
                return "", f"requests_error: {_truncate_err(e)}", None

        # ── Fallback: urllib ──
        try:
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read().decode("utf-8", errors="ignore"), "", None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                return "", "rate_limited_429", 5.0
            return "", f"HTTP_{e.code}: {_truncate_err(e)}", None
        except urllib.error.URLError as e:
            return "", f"connection: {e.reason}", None
        except Exception as e:
            return "", f"urllib_error: {_truncate_err(e)}", None

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
    "NoopLLMClient",
    "resolve_openai_compat_config",
    "build_llm_client",
    "clear_llm_cache",
]
