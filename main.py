# -*- coding: utf-8 -*-
from system.runtime.bootstrap import apply_offline_runtime_defaults
from system.interfaces.cli import run_cli_entrypoint


apply_offline_runtime_defaults()

if __name__ == "__main__":
    run_cli_entrypoint()
