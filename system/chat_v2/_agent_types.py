# -*- coding: utf-8 -*-
"""Shared types for the workspace agent — extracted from agent.py to keep the god class leaner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .computer_task_recovery import DEFAULT_COMPUTER_RECOVERY_PATH


@dataclass
class AgentConfig:
    enabled: bool = False
    project_root: str = "."
    allow_write: bool = False
    allow_exec: bool = False
    allow_browser: bool = False
    allow_desktop: bool = False
    enable_computer_feedback: bool = False
    enable_computer_recovery: bool = False
    computer_feedback_path: str = "artifacts/audit/computer_task_feedback.jsonl"
    computer_feedback_stats_path: str = "artifacts/audit/computer_task_stats.json"
    computer_reflections_path: str = "artifacts/memory/dialogue_reflections.jsonl"
    computer_recovery_path: str = DEFAULT_COMPUTER_RECOVERY_PATH
    enable_computer_learning: bool = True
    computer_learning_path: str = "artifacts/memory/computer_task_learnings.json"
    computer_learning_min_successes: int = 1
    enable_workspace_learning: bool = True
    workspace_learning_path: str = "artifacts/memory/workspace_task_learnings.json"
    workspace_learning_min_successes: int = 1
    search_plan: bool = True
    search_depth: int = 3
    search_beam: int = 4
    planner_candidates: int = 3
    replan_max: int = 1
    self_heal: bool = False
    self_heal_create: bool = False
    self_heal_chmod: bool = False


@dataclass
class AgentOutcome:
    handled: bool
    text: str = ""
    source: str = "agent"
    metadata: Dict[str, Any] = field(default_factory=dict)
