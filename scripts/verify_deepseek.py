# -*- coding: utf-8 -*-
"""Verify DeepSeek connectivity for the chat v2 OpenAI-compatible client."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.chat_v2.llm import OpenAIChatClient, build_llm_client, normalize_llm_provider
from system.runtime.bootstrap import apply_offline_runtime_defaults


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify DeepSeek API configuration and connectivity")
    parser.add_argument("--provider", default="deepseek", help="LLM provider preset, default is deepseek")
    parser.add_argument("--model", default="", help="Model id override, default uses provider preset")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL override")
    parser.add_argument("--endpoint", default="", help="Full endpoint override")
    parser.add_argument("--api-key", default="", help="Inline API key override")
    parser.add_argument("--api-key-env", default="", help="API key env var override")
    parser.add_argument("--timeout-s", type=float, default=45.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--prompt",
        default="请只回复：deepseek_api_ok",
        help="Prompt used for the connectivity check",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Only verify resolved configuration, do not send a network request",
    )
    return parser


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def main() -> int:
    args = build_parser().parse_args()
    apply_offline_runtime_defaults()

    provider = normalize_llm_provider(args.provider) or "deepseek"
    client = build_llm_client(
        provider=provider,
        model=args.model,
        timeout_s=float(args.timeout_s),
        endpoint=args.endpoint,
        base_url=args.base_url,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
    )
    if not isinstance(client, OpenAIChatClient):
        print(f"provider={provider}")
        print("status=error")
        print("error=provider_resolved_to_noop_client")
        return 2

    api_key_present = bool(client.api_key)
    print(f"provider={provider}")
    print(f"model={client.model}")
    print(f"endpoint={client.endpoint}")
    print(f"api_key_env={client.api_key_env}")
    print(f"api_key_present={_yes_no(api_key_present)}")
    print(f"client_available={_yes_no(client.available())}")

    if args.config_only:
        print("status=ok" if client.available() else "status=error")
        if not client.available():
            print("error=missing_api_key_model_or_endpoint")
            return 2
        return 0

    if not client.available():
        print("status=error")
        print("error=missing_api_key_model_or_endpoint")
        return 2

    reply = client.generate(args.prompt, [])
    if reply:
        print("status=ok")
        print("response=" + reply.replace("\r", " ").replace("\n", " ").strip())
        return 0

    print("status=error")
    if client.last_http_status is not None:
        print(f"http_status={client.last_http_status}")
    print("error=" + str(client.last_error or "empty_response"))
    if client.last_http_status == 401:
        print("hint=DeepSeek 返回 401，请检查当前 DEEPSEEK_API_KEY 是否有效")
    elif client.last_http_status == 402:
        print("hint=DeepSeek 返回 402，请检查账号余额")
    elif client.last_http_status == 429:
        print("hint=DeepSeek 返回 429，请稍后重试")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
