# -*- coding: utf-8 -*-
"""Public entry-point functions that build and run a ComputerUseRuntime from CLI args."""

from __future__ import annotations

import logging
from typing import Any, Dict

from system.computer_use._runtime_grid import _clean_str
from system.computer_use._runtime_types import (
    ComputerUseConfig,
    DEFAULT_COMPUTER_USE_RECOVERY_PATH,
)
from system.computer_use.llm import (
    DEFAULT_COMPUTER_USE_SYSTEM_PROMPT,
    build_computer_use_llm_client,
)

logger = logging.getLogger(__name__)


def run_computer_use_runtime(args: Any) -> Dict[str, Any]:
    goal = _clean_str(getattr(args, "computer_use_goal", ""))
    if not goal:
        raise SystemExit('computer-use goal is required; pass --computer-use-goal "..."')
    runtime = build_computer_use_runtime(args)
    result = runtime.run(goal)
    logger.info("Computer-use runtime finished: %s", result)
    return result


def build_computer_use_runtime(args: Any) -> "ComputerUseRuntime":
    from system.computer_use.runtime import ComputerUseRuntime

    goal = _clean_str(getattr(args, "computer_use_goal", ""))
    provider = _clean_str(getattr(args, "v2_llm_provider", ""))
    if provider.lower() in {"", "none"}:
        provider = "openai"
    llm_client = build_computer_use_llm_client(
        provider=provider,
        model=str(getattr(args, "v2_llm_model", "") or ""),
        timeout_s=float(getattr(args, "v2_llm_timeout_s", 45.0) or 45.0),
        endpoint=str(getattr(args, "v2_llm_endpoint", "") or ""),
        base_url=str(getattr(args, "v2_llm_base_url", "") or ""),
        api_key=str(getattr(args, "v2_llm_api_key", "") or ""),
        api_key_env=str(getattr(args, "v2_llm_api_key_env", "") or ""),
        system_prompt=str(getattr(args, "computer_use_system_prompt", "") or DEFAULT_COMPUTER_USE_SYSTEM_PROMPT),
        temperature=float(getattr(args, "v2_llm_temperature", 0.1) or 0.1),
        image_detail=str(getattr(args, "computer_use_image_detail", "auto") or "auto"),
    )
    return ComputerUseRuntime(
        ComputerUseConfig(
            project_root=str(getattr(args, "computer_use_project_root", ".") or "."),
            goal=goal,
            max_steps=max(1, int(getattr(args, "computer_use_max_steps", 8) or 8)),
            grid_rows=max(1, int(getattr(args, "computer_use_grid_rows", 8) or 8)),
            grid_cols=max(1, int(getattr(args, "computer_use_grid_cols", 12) or 12)),
            step_delay_s=max(0.0, float(getattr(args, "computer_use_step_delay_s", 0.25) or 0.25)),
            dry_run=bool(getattr(args, "computer_use_dry_run", False)),
            target_window=str(getattr(args, "computer_use_target_window", "") or ""),
            trace_root=str(getattr(args, "computer_use_trace_root", "artifacts/audit/computer_use") or "artifacts/audit/computer_use"),
            recovery_path=str(getattr(args, "computer_use_recovery_path", DEFAULT_COMPUTER_USE_RECOVERY_PATH) or DEFAULT_COMPUTER_USE_RECOVERY_PATH),
            allow_high_risk_actions=bool(getattr(args, "computer_use_allow_high_risk", False)),
            system_prompt=str(getattr(args, "computer_use_system_prompt", "") or DEFAULT_COMPUTER_USE_SYSTEM_PROMPT),
            image_detail=str(getattr(args, "computer_use_image_detail", "auto") or "auto"),
        ),
        llm_client=llm_client,
    )
