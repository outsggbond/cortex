# -*- coding: utf-8 -*-
import os

def _register_computer_use_args(parser):
    parser.add_argument(
        "--computer-use-goal",
        type=str,
        default=str(os.environ.get("COMPUTER_USE_GOAL", "")),
        help="Run the desktop computer-use agent for one natural-language goal.",
    )
    parser.add_argument(
        "--computer-use-project-root",
        type=str,
        default=str(os.environ.get("COMPUTER_USE_PROJECT_ROOT", ".")),
        help="Project root used to store computer-use traces and artifacts.",
    )
    parser.add_argument(
        "--computer-use-max-steps",
        type=int,
        default=int(os.environ.get("COMPUTER_USE_MAX_STEPS", "8")),
        help="Maximum decision steps for a single computer-use run.",
    )
    parser.add_argument(
        "--computer-use-grid-rows",
        type=int,
        default=int(os.environ.get("COMPUTER_USE_GRID_ROWS", "8")),
        help="Grid rows used to label screenshot blocks for computer-use.",
    )
    parser.add_argument(
        "--computer-use-grid-cols",
        type=int,
        default=int(os.environ.get("COMPUTER_USE_GRID_COLS", "12")),
        help="Grid columns used to label screenshot blocks for computer-use.",
    )
    parser.add_argument(
        "--computer-use-step-delay-s",
        type=float,
        default=float(os.environ.get("COMPUTER_USE_STEP_DELAY_S", "0.25")),
        help="Delay in seconds between computer-use actions.",
    )
    parser.add_argument(
        "--computer-use-target-window",
        type=str,
        default=str(os.environ.get("COMPUTER_USE_TARGET_WINDOW", "")),
        help="Optional preferred window title for control discovery.",
    )
    parser.add_argument(
        "--computer-use-trace-root",
        type=str,
        default=str(os.environ.get("COMPUTER_USE_TRACE_ROOT", "artifacts/audit/computer_use")),
        help="Directory used for per-run computer-use traces, screenshots, and summaries.",
    )
    parser.add_argument(
        "--computer-use-recovery-path",
        type=str,
        default=str(os.environ.get("COMPUTER_USE_RECOVERY_PATH", "artifacts/audit/computer_use_recovery.json")),
        help="Path used to store the latest resumable computer-use checkpoint.",
    )
    parser.add_argument(
        "--computer-use-system-prompt",
        type=str,
        default=str(os.environ.get("COMPUTER_USE_SYSTEM_PROMPT", "")),
        help="Optional system prompt override for the computer-use agent.",
    )
    parser.add_argument(
        "--computer-use-image-detail",
        type=str,
        choices=["auto", "low", "high"],
        default=str(os.environ.get("COMPUTER_USE_IMAGE_DETAIL", "auto")),
        help="Vision detail level used when sending computer-use screenshots to the LLM.",
    )
    parser.add_argument(
        "--computer-use-dry-run",
        action="store_true",
        help="Run computer-use in dry-run mode without touching the live desktop.",
    )
    parser.add_argument(
        "--computer-use-allow-high-risk",
        action="store_true",
        help="Allow high-risk computer-use actions such as launching programs or sensitive typed input.",
    )

