# -*- coding: utf-8 -*-
"""Private types and constants for the computer-use runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from system.computer_use.llm import DEFAULT_COMPUTER_USE_SYSTEM_PROMPT


BLOCK_ID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_COMPUTER_USE_RECOVERY_PATH = "artifacts/audit/computer_use_recovery.json"
_RESUME_COMPUTER_USE_GOALS = {
    "resume last computer-use task",
    "resume last computer task",
    "resume computer-use task",
    "resume computer task",
    "继续上次桌面任务",
    "恢复上次桌面任务",
}


@dataclass
class ComputerUseConfig:
    project_root: str = "."
    goal: str = ""
    max_steps: int = 8
    grid_rows: int = 8
    grid_cols: int = 12
    step_delay_s: float = 0.25
    dry_run: bool = False
    target_window: str = ""
    windows_limit: int = 16
    controls_limit: int = 20
    ocr_items_limit: int = 60
    trace_root: str = "artifacts/audit/computer_use"
    recovery_path: str = DEFAULT_COMPUTER_USE_RECOVERY_PATH
    allow_high_risk_actions: bool = False
    auto_handoff_on_captcha: bool = True
    auto_handoff_on_anti_automation: bool = True
    system_prompt: str = DEFAULT_COMPUTER_USE_SYSTEM_PROMPT
    image_detail: str = "auto"
    plugin_retries: int = 2
    debug: bool = False
    screenshot_dir: str = "artifacts/screenshots"


@dataclass
class ComputerAction:
    type: str
    reason: str = ""
    title: str = ""
    window: str = ""
    target: str = ""
    control: str = ""
    control_type: str = ""
    text: str = ""
    block: str = ""
    from_block: str = ""
    to_block: str = ""
    button: str = "left"
    clicks: int = 1
    keys: str = ""
    seconds: float = 1.0
    clear_first: bool = False
    index: int = 0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "reason": self.reason,
            "title": self.title,
            "window": self.window,
            "target": self.target,
            "control": self.control,
            "control_type": self.control_type,
            "text": self.text,
            "block": self.block,
            "from_block": self.from_block,
            "to_block": self.to_block,
            "button": self.button,
            "clicks": self.clicks,
            "keys": self.keys,
            "seconds": self.seconds,
            "clear_first": self.clear_first,
            "index": self.index,
            "summary": self.summary,
        }
