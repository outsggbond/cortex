# -*- coding: utf-8 -*-
"""Multimodal structured LLM client for the computer-use runtime."""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

from system.chat_v2.llm import OpenAIChatClient, resolve_openai_compat_config


logger = logging.getLogger(__name__)


DEFAULT_COMPUTER_USE_SYSTEM_PROMPT = (
    "You are a careful Windows computer-use agent. "
    "You see desktop screenshots, a labeled grid overlay, OCR text, and UI control summaries. "
    "Choose exactly one next action that makes progress toward the user's goal. "
    "Prefer launch and focus_window for opening or switching desktop apps, and avoid win+r when launch or focus_window can do the job. "
    "Prefer structured control or text actions before blind coordinate clicks. "
    "Return strict JSON only."
)


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _clean_str(value)
        if text:
            return text
    return ""


def _env_first(env: Mapping[str, Any], *names: str, default: str = "") -> str:
    for name in names:
        text = _clean_str(env.get(name, ""))
        if text:
            return text
    return _clean_str(default)


def _normalize_responses_endpoint(target: str) -> str:
    raw = _clean_str(target).rstrip("/")
    if not raw:
        return ""
    if raw.endswith("/responses"):
        return raw
    if raw.endswith("/chat/completions"):
        return raw[: -len("/chat/completions")] + "/responses"
    if raw.endswith("/v1"):
        return raw + "/responses"
    return raw + "/responses"


def _normalize_chat_completions_endpoint(target: str) -> str:
    raw = _clean_str(target).rstrip("/")
    if not raw:
        return ""
    if raw.endswith("/chat/completions"):
        return raw
    if raw.endswith("/responses"):
        return raw[: -len("/responses")] + "/chat/completions"
    if raw.endswith("/v1"):
        return raw + "/chat/completions"
    return raw + "/chat/completions"


class BaseComputerUseLLMClient(ABC):
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def decide(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str],
        system_prompt: str = "",
    ) -> str:
        raise NotImplementedError


class NoopComputerUseLLMClient(BaseComputerUseLLMClient):
    def available(self) -> bool:
        return False

    def decide(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str],
        system_prompt: str = "",
    ) -> str:
        del prompt, image_paths, system_prompt
        return ""


class OpenAIResponsesComputerUseClient(BaseComputerUseLLMClient):
    def __init__(
        self,
        *,
        api_key: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "gpt-4.1-mini",
        endpoint: str = "",
        base_url: str = "",
        timeout_s: float = 45.0,
        system_prompt: str = "",
        temperature: float = 0.1,
        image_detail: str = "auto",
        vision_max_side: int = 1600,
    ) -> None:
        self.api_key_env = _clean_str(api_key_env) or "OPENAI_API_KEY"
        env_api_key = _first_non_empty(
            api_key,
            os.environ.get(self.api_key_env, ""),
            os.environ.get("COMPUTER_USE_API_KEY", ""),
            os.environ.get("V2_LLM_API_KEY", ""),
            os.environ.get("CLOUD_LLM_API_KEY", ""),
            os.environ.get("OPENAI_API_KEY", ""),
        )
        self.api_key = _clean_str(env_api_key)
        self.model = _clean_str(model or os.environ.get("COMPUTER_USE_MODEL", ""))
        env_endpoint = _env_first(
            os.environ,
            "COMPUTER_USE_ENDPOINT",
            "V2_LLM_ENDPOINT",
            "CLOUD_LLM_ENDPOINT",
            "OPENAI_API_ENDPOINT",
        )
        env_base_url = _env_first(
            os.environ,
            "COMPUTER_USE_BASE_URL",
            "V2_LLM_BASE_URL",
            "CLOUD_LLM_BASE_URL",
            "OPENAI_BASE_URL",
            "OPENAI_API_BASE_URL",
        )
        self.base_url = _clean_str(base_url or env_base_url)
        self.endpoint = _normalize_responses_endpoint(endpoint or env_endpoint or self.base_url)
        self.timeout_s = max(5.0, float(timeout_s))
        self.system_prompt = _clean_str(system_prompt) or DEFAULT_COMPUTER_USE_SYSTEM_PROMPT
        self.temperature = float(temperature)
        detail = _clean_str(image_detail).lower()
        self.image_detail = detail if detail in {"low", "high", "auto"} else "auto"
        self.vision_max_side = max(256, int(vision_max_side))

    def available(self) -> bool:
        return bool(self.api_key and self.model and self.endpoint)

    def _image_data_url(self, path_text: str) -> str:
        path = Path(str(path_text or "")).expanduser()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"computer-use image not found: {path}")
        try:
            from PIL import Image

            image = Image.open(path.as_posix())
            image.load()
            if max(image.size) > self.vision_max_side:
                image = image.convert("RGB")
                image.thumbnail((self.vision_max_side, self.vision_max_side))
            mime = "image/png"
            out = BytesIO()
            image.save(out, format="PNG", optimize=True)
            payload = out.getvalue()
        except Exception:
            payload = path.read_bytes()
            suffix = path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif suffix == ".webp":
                mime = "image/webp"
            else:
                mime = "image/png"
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _build_input_content(self, prompt: str, image_paths: Sequence[str]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        content.append({"type": "input_text", "text": str(prompt or "")})
        for path in list(image_paths or []):
            path_text = _clean_str(path)
            if not path_text:
                continue
            content.append(
                {
                    "type": "input_image",
                    "image_url": self._image_data_url(path_text),
                    "detail": self.image_detail,
                }
            )
        return content

    def _build_payload(self, prompt: str, image_paths: Sequence[str], system_prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "instructions": _clean_str(system_prompt) or self.system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": self._build_input_content(prompt, image_paths),
                }
            ],
            "temperature": self.temperature,
        }

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
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read().decode("utf-8", errors="ignore")

    def decide(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str],
        system_prompt: str = "",
    ) -> str:
        if not self.available():
            return ""
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return ""
        modes = [list(image_paths or []), []]
        last_error = ""
        for idx, paths in enumerate(modes):
            if idx > 0 and not last_error:
                continue
            payload = self._build_payload(prompt_text, paths, system_prompt)
            try:
                body = self._request(payload)
                return OpenAIChatClient._parse_response_text(body)
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    detail = str(exc)
                last_error = f"http {getattr(exc, 'code', 'unknown')}: {detail[:400]}"
                if paths:
                    logger.warning("computer-use multimodal request failed, retrying text-only: %s", last_error)
                    continue
                raise RuntimeError(f"computer-use llm request failed: {last_error}") from exc
            except Exception as exc:
                last_error = str(exc)
                if paths:
                    logger.warning("computer-use multimodal request failed, retrying text-only: %s", last_error)
                    continue
                raise RuntimeError(f"computer-use llm request failed: {last_error}") from exc
        if last_error:
            raise RuntimeError(f"computer-use llm request failed: {last_error}")
        return ""


class OpenAIChatComputerUseClient(BaseComputerUseLLMClient):
    def __init__(
        self,
        *,
        api_key: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "gpt-4.1-mini",
        endpoint: str = "",
        base_url: str = "",
        timeout_s: float = 45.0,
        system_prompt: str = "",
        temperature: float = 0.1,
    ) -> None:
        self.client = OpenAIChatClient(
            api_key=api_key,
            api_key_env=api_key_env,
            model=model,
            endpoint=_normalize_chat_completions_endpoint(endpoint or base_url),
            base_url=base_url,
            timeout_s=timeout_s,
            system_prompt=_clean_str(system_prompt) or DEFAULT_COMPUTER_USE_SYSTEM_PROMPT,
            temperature=temperature,
        )

    def available(self) -> bool:
        return self.client.available()

    def decide(
        self,
        *,
        prompt: str,
        image_paths: Sequence[str],
        system_prompt: str = "",
    ) -> str:
        del image_paths
        if system_prompt:
            original = self.client.system_prompt
            try:
                self.client.system_prompt = system_prompt
                return self.client.generate(prompt, [])
            finally:
                self.client.system_prompt = original
        return self.client.generate(prompt, [])


def build_computer_use_llm_client(
    *,
    provider: str = "",
    model: str = "",
    timeout_s: float = 45.0,
    endpoint: str = "",
    base_url: str = "",
    api_key: str = "",
    api_key_env: str = "",
    system_prompt: str = "",
    temperature: float = 0.1,
    image_detail: str = "auto",
    env: Mapping[str, Any] | None = None,
) -> BaseComputerUseLLMClient:
    env_map = env if env is not None else os.environ
    resolved_provider = _first_non_empty(
        provider,
        _env_first(env_map, "COMPUTER_USE_PROVIDER", "V2_LLM_PROVIDER", "CLOUD_LLM_PROVIDER"),
        "openai",
    )
    resolved = resolve_openai_compat_config(
        resolved_provider,
        model=_first_non_empty(model, _env_first(env_map, "COMPUTER_USE_MODEL", "V2_LLM_MODEL", "CLOUD_LLM_MODEL")),
        endpoint=_first_non_empty(endpoint, _env_first(env_map, "COMPUTER_USE_ENDPOINT")),
        base_url=_first_non_empty(base_url, _env_first(env_map, "COMPUTER_USE_BASE_URL")),
        api_key_env=_first_non_empty(
            api_key_env,
            _env_first(env_map, "COMPUTER_USE_API_KEY_ENV", "V2_LLM_API_KEY_ENV", "CLOUD_LLM_API_KEY_ENV"),
        ),
        env=env_map,
    )
    resolved_endpoint = _normalize_responses_endpoint(
        _first_non_empty(
            endpoint,
            _env_first(env_map, "COMPUTER_USE_ENDPOINT"),
            str(resolved.get("endpoint", "") or ""),
            _first_non_empty(base_url, _env_first(env_map, "COMPUTER_USE_BASE_URL"), str(resolved.get("base_url", "") or "")),
        )
    )
    if _clean_str(resolved.get("provider", "")).lower() == "none":
        return NoopComputerUseLLMClient()
    provider_name = _clean_str(resolved.get("provider", "")).lower()
    if provider_name in {"deepseek", "kimi"}:
        return OpenAIChatComputerUseClient(
            api_key=api_key,
            api_key_env=str(resolved.get("api_key_env", "") or "OPENAI_API_KEY"),
            model=str(resolved.get("model", "") or model),
            endpoint=_normalize_chat_completions_endpoint(
                _first_non_empty(
                    endpoint,
                    _env_first(env_map, "COMPUTER_USE_ENDPOINT"),
                    str(resolved.get("endpoint", "") or ""),
                    _first_non_empty(base_url, _env_first(env_map, "COMPUTER_USE_BASE_URL"), str(resolved.get("base_url", "") or "")),
                )
            ),
            base_url=str(resolved.get("base_url", "") or base_url),
            timeout_s=timeout_s,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    return OpenAIResponsesComputerUseClient(
        api_key=api_key,
        api_key_env=str(resolved.get("api_key_env", "") or "OPENAI_API_KEY"),
        model=str(resolved.get("model", "") or model),
        endpoint=resolved_endpoint,
        base_url=str(resolved.get("base_url", "") or base_url),
        timeout_s=timeout_s,
        system_prompt=system_prompt,
        temperature=temperature,
        image_detail=image_detail,
    )


__all__ = [
    "BaseComputerUseLLMClient",
    "NoopComputerUseLLMClient",
    "OpenAIChatComputerUseClient",
    "OpenAIResponsesComputerUseClient",
    "DEFAULT_COMPUTER_USE_SYSTEM_PROMPT",
    "build_computer_use_llm_client",
]
