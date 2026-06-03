from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.auto_iterate import parse_args as parse_auto_iterate_args
from scripts.auto_iterate_loop import parse_args as parse_auto_iterate_loop_args
from system.l_utils.cli_support import build_cloud_env, cloud_env_overrides, cloud_runtime_summary


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_auto_iterate_cloud_args_enable_env() -> None:
    args = parse_auto_iterate_args(
        [
            "--model-path",
            "demo-model",
            "--cloud-provider",
            "deepseek",
            "--cloud-model",
            "deepseek-chat",
            "--cloud-base-url",
            "https://api.deepseek.com",
            "--cloud-temperature",
            "0.35",
        ]
    )
    overrides = cloud_env_overrides(args)
    assert overrides["CLOUD_LLM_ENABLE"] == "1"
    assert overrides["CLOUD_LLM_PROVIDER"] == "deepseek"
    assert overrides["CLOUD_LLM_MODEL"] == "deepseek-chat"
    assert overrides["CLOUD_LLM_BASE_URL"] == "https://api.deepseek.com"
    assert overrides["CLOUD_LLM_TEMPERATURE"] == "0.35"


def test_auto_iterate_loop_cloud_args_are_consumed() -> None:
    args, passthrough = parse_auto_iterate_loop_args(
        [
            "--model-path",
            "demo-model",
            "--rounds",
            "2",
            "--cloud-llm",
            "--cloud-model",
            "gpt-4.1-mini",
            "--cloud-api-key-env",
            "CUSTOM_OPENAI_KEY",
            "--eval-max-samples",
            "12",
        ]
    )
    assert bool(args.cloud_llm) is True
    assert str(args.cloud_model) == "gpt-4.1-mini"
    assert str(args.cloud_api_key_env) == "CUSTOM_OPENAI_KEY"
    assert passthrough == ["--eval-max-samples", "12"]


def test_cloud_runtime_summary_uses_env_fallbacks_without_leaking_key() -> None:
    keys = [
        "CLOUD_LLM_ENABLE",
        "CLOUD_LLM_PROVIDER",
        "CLOUD_LLM_API_KEY",
        "CLOUD_LLM_API_KEY_ENV",
        "V2_LLM_MODEL",
        "V2_LLM_BASE_URL",
    ]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["CLOUD_LLM_ENABLE"] = "1"
        os.environ["CLOUD_LLM_PROVIDER"] = "openai"
        os.environ["CLOUD_LLM_API_KEY"] = "sk-secret-inline"
        os.environ["CLOUD_LLM_API_KEY_ENV"] = "CUSTOM_KEY_ENV"
        os.environ["V2_LLM_MODEL"] = "gpt-4.1-mini"
        os.environ["V2_LLM_BASE_URL"] = "https://example.test/v1"
        summary = cloud_runtime_summary()
        assert summary["enabled"] is True
        assert summary["provider"] == "openai"
        assert summary["model"] == "gpt-4.1-mini"
        assert summary["base_url"] == "https://example.test/v1"
        assert summary["endpoint"] == "https://example.test/v1/chat/completions"
        assert summary["api_key_env"] == "CUSTOM_KEY_ENV"
        assert summary["api_key_inline"] is True
        payload = json.dumps(summary, ensure_ascii=False)
        assert "sk-secret-inline" not in payload
    finally:
        _restore_env(snapshot)


def test_build_cloud_env_preserves_base_and_applies_overrides() -> None:
    args = parse_auto_iterate_args(
        [
            "--model-path",
            "demo-model",
            "--cloud-llm",
            "--cloud-model",
            "gpt-4.1-mini",
            "--cloud-timeout-s",
            "18",
        ]
    )
    env = build_cloud_env(args, base_env={"KEEP_ME": "1"})
    assert env["KEEP_ME"] == "1"
    assert env["CLOUD_LLM_ENABLE"] == "1"
    assert env["CLOUD_LLM_MODEL"] == "gpt-4.1-mini"
    assert env["CLOUD_LLM_TIMEOUT_S"] == "18.0"


def main() -> None:
    test_auto_iterate_cloud_args_enable_env()
    test_auto_iterate_loop_cloud_args_are_consumed()
    test_cloud_runtime_summary_uses_env_fallbacks_without_leaking_key()
    test_build_cloud_env_preserves_base_and_applies_overrides()
    print("auto_iterate_cloud_ok")


if __name__ == "__main__":
    main()
