# -*- coding: utf-8 -*-
import argparse
import os
import sys

from system.interfaces.cli import (
    _register_artifact_args,
    _register_automation_args,
    _register_chat_args,
    _register_common_args,
    _register_computer_use_args,
    _register_federated_args,
)


def _split_passthrough_args(argv=None):
    raw = sys.argv[1:] if argv is None else list(argv)
    if "--" not in raw:
        return raw, []
    idx = raw.index("--")
    return raw[:idx], raw[idx + 1 :]


def build_parser():
    parser = argparse.ArgumentParser(description="Offline Multimodal System")
    _register_common_args(parser)
    _register_chat_args(parser.add_argument_group("chat"))
    _register_computer_use_args(parser.add_argument_group("computer_use"))
    _register_automation_args(parser.add_argument_group("automation"))
    _register_artifact_args(parser.add_argument_group("artifact"))
    _register_federated_args(parser.add_argument_group("federated"))
    return parser


def parse_args(argv=None):
    parser = build_parser()
    main_argv, passthrough = _split_passthrough_args(argv)
    args, extra = parser.parse_known_args(main_argv)
    args.federated_train = bool(args.federated_train or args.federated)
    if bool(getattr(args, "cloud_llm", False)):
        args.runtime_arch = "v2"
        provider = str(getattr(args, "v2_llm_provider", "none") or "none").strip().lower()
        if provider in {"", "none"}:
            args.v2_llm_provider = "openai"
    args.federated_forward_args = list(passthrough)
    fed_auto_env = str(os.environ.get("FEDERATED_AUTO_TRAIN", "")).strip().lower() in {"1", "true", "yes", "on"}
    allow_federated_extra = bool(args.federated_train or args.federated_auto or fed_auto_env)
    if extra:
        parser.error("unrecognized arguments: " + " ".join(extra))
    if passthrough and not allow_federated_extra:
        parser.error("passthrough arguments after '--' require federated mode")
    return args
