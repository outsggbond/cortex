# -*- coding: utf-8 -*-
import os

def _register_automation_args(parser):
    parser.add_argument(
        "--automation-list",
        action="store_true",
        help="List configured automation workflows and exit.",
    )
    parser.add_argument(
        "--automation-scan",
        action="store_true",
        help="Scan configured automation workflows once and run those whose triggers are due.",
    )
    parser.add_argument(
        "--automation-loop",
        action="store_true",
        help="Run the automation worker loop and continuously evaluate workflow triggers.",
    )
    parser.add_argument(
        "--automation-run",
        nargs="+",
        default=[],
        help="Run one or more automation workflows immediately by workflow id.",
    )
    parser.add_argument(
        "--automation-config",
        type=str,
        default=str(os.environ.get("AUTOMATION_CONFIG", "config/automation_workflows.json")),
        help="Automation workflow configuration JSON path.",
    )
    parser.add_argument(
        "--automation-state",
        type=str,
        default=str(os.environ.get("AUTOMATION_STATE", "artifacts/audit/automation_state.json")),
        help="Automation worker state JSON path.",
    )
    parser.add_argument(
        "--automation-runs",
        type=str,
        default=str(os.environ.get("AUTOMATION_RUNS", "artifacts/audit/automation_runs.jsonl")),
        help="Automation worker run log JSONL path.",
    )
    parser.add_argument(
        "--automation-project-root",
        type=str,
        default=str(os.environ.get("AUTOMATION_PROJECT_ROOT", ".")),
        help="Project root used by the automation worker for file and task execution.",
    )
    parser.add_argument(
        "--automation-poll-s",
        type=float,
        default=float(os.environ.get("AUTOMATION_POLL_S", "30")),
        help="Automation worker poll interval in seconds when --automation-loop is enabled.",
    )
    parser.add_argument(
        "--automation-max-cycles",
        type=int,
        default=int(os.environ.get("AUTOMATION_MAX_CYCLES", "0")),
        help="Maximum automation worker cycles before exit (0 means run until interrupted).",
    )

