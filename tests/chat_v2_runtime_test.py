import os
import sys
import tempfile
from pathlib import Path
from io import StringIO

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.intents import ChatIntent, classify_intent
from system.chat_v2.llm import NoopLLMClient
from system.chat_v2.memory import CuratedMemoryStore
from system.chat_v2.pipeline import ChatPipeline, PipelineConfig, build_request
from system.chat_v2.runtime import _build_agent_config, _read_console_line_windows
from system.main_cli import parse_args


def main():
    writer = StringIO()
    chars = iter(["打", "开", "Q", "Q", "\r"])
    assert _read_console_line_windows("You > ", _getwch=lambda: next(chars), _stdout=writer) == "打开QQ"
    assert writer.getvalue() == "You > 打开QQ\n"

    writer = StringIO()
    chars = iter(["盛", "哥", "\b", "哥", "\r"])
    assert _read_console_line_windows("You > ", _getwch=lambda: next(chars), _stdout=writer) == "盛哥"
    assert writer.getvalue() == "You > 盛哥\b \b哥\n"

    assert classify_intent("\u4f60\u53eb\u4ec0\u4e48\uff1f") == ChatIntent.IDENTITY
    assert classify_intent("\u4f60\u4f1a\u4ec0\u4e48\uff1f") == ChatIntent.CAPABILITY
    assert classify_intent("hello") == ChatIntent.GREETING
    assert classify_intent('type "\u665a\u5b89" into control \u53d1\u9001 in window \u76db\u54e5 control type Group clear first') == ChatIntent.TASK
    assert classify_intent("\u6253\u5f00QQ\u7ed9\u76db\u54e5\u53d1\u4e00\u53e5\u665a\u5b89") == ChatIntent.TASK

    with tempfile.TemporaryDirectory() as td:
        memory_path = Path(td) / "curated.jsonl"
        memory_path.write_text(
            "\n".join(
                [
                    '{"user":"python json\\u600e\\u4e48\\u8bfb\\u53d6","assistant":"Python\\u8bfb\\u53d6JSON\\u793a\\u4f8b\\uff1aimport json\\uff0c\\u7136\\u540e\\u7528 json.load(file) \\u3002"}',
                    '{"user":"\\u4f60\\u662f\\u8c01","assistant":"\\u6211\\u662f Tina\\uff0c\\u4e00\\u4e2a\\u5b66\\u4e60\\u578b AI \\u52a9\\u624b\\u3002"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        memory = CuratedMemoryStore(path=str(memory_path), min_similarity=0.45)
        pipeline = ChatPipeline(
            memory_store=memory,
            llm_client=NoopLLMClient(),
            config=PipelineConfig(enable_llm=True, enable_memory_retrieval=True),
        )

        out_identity = pipeline.respond(build_request("\u4f60\u53eb\u4ec0\u4e48\uff1f", []))
        assert "Tina" in out_identity.text
        assert out_identity.source == "policy"

        out_cap = pipeline.respond(build_request("\u4f60\u4f1a\u4ec0\u4e48\uff1f", []))
        assert "Tina" in out_cap.text
        assert out_cap.source == "policy"

        out_memory = pipeline.respond(build_request("python json\u600e\u4e48\u8bfb\u53d6", []))
        assert "json.load" in out_memory.text.lower()
        assert out_memory.source == "memory"

        memory_hits = memory.search_many("python json\u89e3\u6790\u6587\u4ef6", top_k=2)
        assert len(memory_hits) >= 1
        assert "json.load" in memory_hits[0].assistant.lower()
        assert float(memory_hits[0].score) >= 0.45

    args = parse_args(["--chat", "--runtime-arch", "v2"])
    assert args.runtime_arch == "v2"
    assert args.v2_memory_path
    assert args.v2_agent is False
    assert str(args.v2_llm_provider) == "none"

    agent_args = parse_args(["--chat", "--runtime-arch", "v2", "--v2-agent", "--v2-agent-allow-write"])
    assert agent_args.v2_agent is True
    assert agent_args.v2_agent_allow_write is True
    agent_config = _build_agent_config(agent_args)
    assert agent_config.search_plan is True
    assert agent_config.self_heal is False

    computer_agent_args = parse_args(
        [
            "--chat",
            "--runtime-arch",
            "v2",
            "--v2-agent",
            "--v2-agent-allow-browser",
            "--v2-agent-allow-desktop",
        ]
    )
    assert computer_agent_args.v2_agent is True
    assert computer_agent_args.v2_agent_allow_browser is True
    assert computer_agent_args.v2_agent_allow_desktop is True
    computer_agent_config = _build_agent_config(computer_agent_args)
    assert computer_agent_config.enable_computer_feedback is True
    assert computer_agent_config.enable_computer_recovery is True

    tuned_agent_args = parse_args(
        [
            "--chat",
            "--runtime-arch",
            "v2",
            "--v2-agent",
            "--no-search-plan",
            "--self-heal",
        ]
    )
    tuned_agent_config = _build_agent_config(tuned_agent_args)
    assert tuned_agent_config.search_plan is False
    assert tuned_agent_config.self_heal is True

    forced_self_heal_args = parse_args(
        [
            "--chat",
            "--runtime-arch",
            "v2",
            "--v2-agent",
            "--self-heal-create",
        ]
    )
    forced_self_heal_config = _build_agent_config(forced_self_heal_args)
    assert forced_self_heal_config.self_heal is True
    assert forced_self_heal_config.self_heal_create is True

    cloud_args = parse_args(
        [
            "--chat",
            "--cloud-llm",
            "--cloud-provider",
            "kimi",
            "--cloud-base-url",
            "https://api.moonshot.cn/v1",
            "--cloud-api-key-env",
            "MOONSHOT_API_KEY",
        ]
    )
    assert cloud_args.runtime_arch == "v2"
    assert cloud_args.v2_llm_provider == "kimi"
    assert cloud_args.v2_llm_base_url == "https://api.moonshot.cn/v1"
    assert cloud_args.v2_llm_api_key_env == "MOONSHOT_API_KEY"

    print("chat_v2_runtime_ok")


if __name__ == "__main__":
    main()
