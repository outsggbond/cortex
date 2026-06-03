# -*- coding: utf-8 -*-
"""CLI interface entrypoint adapter."""

from __future__ import annotations


def run_cli_entrypoint() -> None:
    from system.app_runtime import main as runtime_main

    runtime_main()


__all__ = ["run_cli_entrypoint"]
