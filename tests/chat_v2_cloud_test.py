from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.llm import OpenAIChatClient, build_llm_client
from system.chat_v2.types import ChatTurn
from system.main_cli import parse_args


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_cloud_alias_defaults_to_v2_openai() -> None:
    args = parse_args(["--chat", "--cloud-llm"])
    assert bool(args.cloud_llm) is True
    assert str(args.runtime_arch) == "v2"
    assert str(args.v2_llm_provider) == "openai"


def test_cloud_alias_preserves_explicit_provider_and_base_url() -> None:
    args = parse_args(
        [
            "--chat",
            "--cloud-llm",
            "--cloud-provider",
            "deepseek",
            "--cloud-model",
            "deepseek-reasoner",
            "--cloud-base-url",
            "https://api.deepseek.com",
            "--cloud-api-key-env",
            "DEEPSEEK_API_KEY",
            "--cloud-system-prompt",
            "You are a helpful cloud assistant.",
        ]
    )
    assert bool(args.cloud_llm) is True
    assert str(args.runtime_arch) == "v2"
    assert str(args.v2_llm_provider) == "deepseek"
    assert str(args.v2_llm_model) == "deepseek-reasoner"
    assert str(args.v2_llm_base_url) == "https://api.deepseek.com"
    assert str(args.v2_llm_api_key_env) == "DEEPSEEK_API_KEY"
    assert "helpful cloud assistant" in str(args.v2_llm_system_prompt)


def test_openai_chat_client_chat_completions_payload() -> None:
    import system.chat_v2.llm as llm_mod

    captured = {}

    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.get_header("Authorization")
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "cloud answer ok"}}]}))

    old = llm_mod.urllib.request.urlopen
    llm_mod.urllib.request.urlopen = _fake_urlopen
    try:
        client = OpenAIChatClient(
            api_key="sk-inline",
            model="demo-model",
            endpoint="https://example.test/v1/chat/completions",
            timeout_s=12.0,
            system_prompt="SYSTEM PROMPT",
            temperature=0.35,
        )
        out = client.generate("hello", [ChatTurn(role="user", text="earlier turn")])
        assert out == "cloud answer ok"
        assert captured["url"] == "https://example.test/v1/chat/completions"
        assert float(captured["timeout"]) == 12.0
        assert captured["auth"] == "Bearer sk-inline"
        assert str(captured["payload"]["model"]) == "demo-model"
        assert abs(float(captured["payload"]["temperature"]) - 0.35) < 1e-9
        assert str(captured["payload"]["messages"][0]["content"]) == "SYSTEM PROMPT"
        assert str(captured["payload"]["messages"][-1]["content"]) == "hello"
    finally:
        llm_mod.urllib.request.urlopen = old


def test_openai_chat_client_responses_api_and_env_key() -> None:
    import system.chat_v2.llm as llm_mod

    captured = {}
    old_env = os.environ.get("CLOUD_TEST_KEY")
    os.environ["CLOUD_TEST_KEY"] = "sk-env"

    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["auth"] = req.get_header("Authorization")
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(
            json.dumps(
                {
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "text": "responses api answer"},
                            ]
                        }
                    ]
                }
            )
        )

    old = llm_mod.urllib.request.urlopen
    llm_mod.urllib.request.urlopen = _fake_urlopen
    try:
        client = OpenAIChatClient(
            api_key="",
            api_key_env="CLOUD_TEST_KEY",
            model="demo-model",
            endpoint="https://example.test/v1/responses",
            timeout_s=9.0,
        )
        assert client.available() is True
        out = client.generate("need cloud", [])
        assert out == "responses api answer"
        assert captured["url"] == "https://example.test/v1/responses"
        assert captured["auth"] == "Bearer sk-env"
        assert str(captured["payload"]["model"]) == "demo-model"
        assert isinstance(captured["payload"]["input"], list) and captured["payload"]["input"]
        assert str(captured["payload"]["input"][-1]["content"][0]["text"]) == "need cloud"
    finally:
        llm_mod.urllib.request.urlopen = old
        if old_env is None:
            os.environ.pop("CLOUD_TEST_KEY", None)
        else:
            os.environ["CLOUD_TEST_KEY"] = old_env


def test_openai_chat_client_normalizes_base_url() -> None:
    import system.chat_v2.llm as llm_mod

    captured = {}

    def _fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": "deepseek ok"}}]}))

    old = llm_mod.urllib.request.urlopen
    llm_mod.urllib.request.urlopen = _fake_urlopen
    try:
        client = OpenAIChatClient(
            api_key="sk-inline",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            timeout_s=6.0,
        )
        out = client.generate("hello", [])
        assert out == "deepseek ok"
        assert captured["url"] == "https://api.deepseek.com/chat/completions"
        assert str(captured["payload"]["model"]) == "deepseek-chat"
    finally:
        llm_mod.urllib.request.urlopen = old


def test_build_llm_client_uses_provider_presets() -> None:
    deepseek = build_llm_client("deepseek", "", 5.0, api_key="sk-demo")
    assert isinstance(deepseek, OpenAIChatClient)
    assert deepseek.available() is True
    assert deepseek.model == "deepseek-v4-flash"
    assert deepseek.api_key_env == "DEEPSEEK_API_KEY"
    assert deepseek.endpoint == "https://api.deepseek.com/chat/completions"

    kimi = build_llm_client("kimi", "", 5.0, api_key="sk-demo")
    assert isinstance(kimi, OpenAIChatClient)
    assert kimi.available() is True
    assert kimi.model == "kimi-k2.5"
    assert kimi.api_key_env == "MOONSHOT_API_KEY"
    assert kimi.endpoint == "https://api.moonshot.cn/v1/chat/completions"


def test_build_llm_client_accepts_custom_openai_compatible_provider() -> None:
    custom = build_llm_client(
        "qwen",
        "qwen-plus",
        5.0,
        api_key="sk-demo",
        api_key_env="DASHSCOPE_API_KEY",
        base_url="https://example.test/compatible-mode/v1",
    )
    assert isinstance(custom, OpenAIChatClient)
    assert custom.available() is True
    assert custom.model == "qwen-plus"
    assert custom.api_key_env == "DASHSCOPE_API_KEY"
    assert custom.endpoint == "https://example.test/compatible-mode/v1/chat/completions"


def main() -> None:
    test_cloud_alias_defaults_to_v2_openai()
    test_cloud_alias_preserves_explicit_provider_and_base_url()
    test_openai_chat_client_chat_completions_payload()
    test_openai_chat_client_responses_api_and_env_key()
    test_openai_chat_client_normalizes_base_url()
    test_build_llm_client_uses_provider_presets()
    test_build_llm_client_accepts_custom_openai_compatible_provider()
    print("chat_v2_cloud_ok")


if __name__ == "__main__":
    main()
