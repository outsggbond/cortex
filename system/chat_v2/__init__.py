# -*- coding: utf-8 -*-
"""Modular chat runtime (v2) entrypoints."""


def run_chat_v2(*args, **kwargs):
    from .runtime import run_chat_v2 as _run_chat_v2

    return _run_chat_v2(*args, **kwargs)


__all__ = ["run_chat_v2"]
