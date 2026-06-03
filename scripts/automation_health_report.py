from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{datetime.now(timezone.utc).timestamp():.6f}")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _read_jsonl_tail(path: Path, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0 or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return rows
    for line in lines[-limit:]:
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _iso_utc(ts: float | int | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _repo_file_stats(project_root: Path) -> Dict[str, Any]:
    exclude_names = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".tmp"}
    total_files = 0
    python_files = 0
    markdown_files = 0
    json_files = 0
    test_files = 0
    suffixes: Counter[str] = Counter()
    top_level_files: Counter[str] = Counter()

    for current_root, dirs, files in os.walk(project_root):
        dirs[:] = [name for name in dirs if name not in exclude_names]
        current = Path(current_root)
        for filename in files:
            path = current / filename
            total_files += 1
            suffix = path.suffix.lower() or "<noext>"
            suffixes[suffix] += 1
            rel = path.relative_to(project_root)
            top_level = rel.parts[0] if rel.parts else "."
            top_level_files[top_level] += 1
            if suffix == ".py":
                python_files += 1
            if suffix == ".md":
                markdown_files += 1
            if suffix == ".json":
                json_files += 1
            if "test" in rel.name.lower():
                test_files += 1

    return {
        "total_files": total_files,
        "python_files": python_files,
        "markdown_files": markdown_files,
        "json_files": json_files,
        "test_files": test_files,
        "top_suffixes": suffixes.most_common(8),
        "top_level_files": top_level_files.most_common(10),
    }


def _workflow_summary(config_path: Path, state_path: Path) -> Dict[str, Any]:
    config = _read_json(config_path, {"workflows": []})
    if isinstance(config, dict):
        workflows = list(config.get("workflows") or [])
    elif isinstance(config, list):
        workflows = list(config)
    else:
        workflows = []

    state = _read_json(state_path, {"workflows": {}})
    workflow_state = dict(state.get("workflows") or {}) if isinstance(state, dict) else {}

    items: List[Dict[str, Any]] = []
    enabled_count = 0
    for item in workflows:
        if not isinstance(item, dict):
            continue
        workflow_id = str(item.get("id") or item.get("name") or "").strip()
        if not workflow_id:
            continue
        enabled = bool(item.get("enabled", True))
        if enabled:
            enabled_count += 1
        raw_triggers = item.get("triggers", item.get("trigger"))
        trigger_items = raw_triggers if isinstance(raw_triggers, list) else [raw_triggers]
        triggers: List[str] = []
        for trigger in trigger_items:
            if isinstance(trigger, str):
                triggers.append(trigger.strip() or "manual")
            elif isinstance(trigger, dict):
                triggers.append(str(trigger.get("type") or "manual").strip() or "manual")
        state_row = workflow_state.get(workflow_id) if isinstance(workflow_state.get(workflow_id), dict) else {}
        items.append(
            {
                "id": workflow_id,
                "enabled": enabled,
                "triggers": triggers or ["manual"],
                "step_count": len(list(item.get("steps") or [])) + sum(len(level or []) for level in list(item.get("levels") or [])),
                "last_status": str(state_row.get("last_status") or ""),
                "last_run_at": _iso_utc(state_row.get("last_run_at")),
                "runs": int(state_row.get("runs") or 0),
            }
        )

    return {
        "count": len(items),
        "enabled_count": enabled_count,
        "items": items,
    }


def _focus_path_summary(project_root: Path, focus_paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in focus_paths:
        rel = str(raw or "").strip()
        if not rel:
            continue
        path = (project_root / rel).resolve()
        exists = path.exists()
        stat = path.stat() if exists else None
        rows.append(
            {
                "path": rel,
                "exists": exists,
                "kind": "dir" if exists and path.is_dir() else ("file" if exists else "missing"),
                "size": int(stat.st_size) if stat and path.is_file() else 0,
                "modified_at": _iso_utc(stat.st_mtime) if stat else "",
            }
        )
    return rows


def build_report(
    *,
    project_root: Path,
    config_path: Path,
    state_path: Path,
    runs_path: Path,
    focus_paths: List[str],
    recent_runs: int,
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    file_stats = _repo_file_stats(project_root)
    workflows = _workflow_summary(config_path, state_path)
    recent = _read_jsonl_tail(runs_path, recent_runs)
    focus = _focus_path_summary(project_root, focus_paths)
    return {
        "generated_at": now,
        "project_root": project_root.as_posix(),
        "file_stats": file_stats,
        "workflows": workflows,
        "recent_runs": recent,
        "focus_paths": focus,
        "paths": {
            "config_path": config_path.as_posix(),
            "state_path": state_path.as_posix(),
            "runs_path": runs_path.as_posix(),
        },
    }


def render_markdown(report: Dict[str, Any]) -> str:
    file_stats = dict(report.get("file_stats") or {})
    workflows = dict(report.get("workflows") or {})
    recent_runs = list(report.get("recent_runs") or [])
    focus_paths = list(report.get("focus_paths") or [])
    lines = [
        "# Automation Health Report",
        "",
        f"- Generated at: {report.get('generated_at', '')}",
        f"- Project root: {report.get('project_root', '')}",
        "",
        "## Workspace",
        "",
        f"- Total files scanned: {file_stats.get('total_files', 0)}",
        f"- Python files: {file_stats.get('python_files', 0)}",
        f"- Markdown files: {file_stats.get('markdown_files', 0)}",
        f"- JSON files: {file_stats.get('json_files', 0)}",
        f"- Test-like files: {file_stats.get('test_files', 0)}",
        "",
        "## Workflows",
        "",
        f"- Configured workflows: {workflows.get('count', 0)}",
        f"- Enabled workflows: {workflows.get('enabled_count', 0)}",
        "",
    ]

    workflow_items = list(workflows.get("items") or [])
    if workflow_items:
        for item in workflow_items:
            triggers = ", ".join(list(item.get("triggers") or []))
            last_status = item.get("last_status") or "never-run"
            last_run_at = item.get("last_run_at") or "n/a"
            lines.append(
                f"- `{item.get('id', '')}` | enabled={item.get('enabled', False)} | triggers={triggers} | steps={item.get('step_count', 0)} | last={last_status} @ {last_run_at}"
            )
        lines.append("")

    top_level_files = list(file_stats.get("top_level_files") or [])
    if top_level_files:
        lines.extend(["## Top-Level File Distribution", ""])
        for name, count in top_level_files:
            lines.append(f"- `{name}`: {count}")
        lines.append("")

    top_suffixes = list(file_stats.get("top_suffixes") or [])
    if top_suffixes:
        lines.extend(["## Most Common File Types", ""])
        for suffix, count in top_suffixes:
            lines.append(f"- `{suffix}`: {count}")
        lines.append("")

    if focus_paths:
        lines.extend(["## Focus Paths", ""])
        for item in focus_paths:
            lines.append(
                f"- `{item.get('path', '')}` | exists={item.get('exists', False)} | kind={item.get('kind', '')} | size={item.get('size', 0)} | modified_at={item.get('modified_at', '') or 'n/a'}"
            )
        lines.append("")

    lines.extend(["## Recent Automation Runs", ""])
    if recent_runs:
        for item in recent_runs:
            lines.append(
                f"- `{item.get('workflow_id', '')}` | ok={item.get('ok', False)} | status={item.get('status', '')} | reason={item.get('reason', '')} | ts={_iso_utc(item.get('ts')) or 'n/a'}"
            )
    else:
        lines.append("- No recent runs found.")
    lines.append("")

    lines.extend(
        [
            "## Try It",
            "",
            "- `python main.py --automation-list`",
            "- `python main.py --automation-run project_health_report`",
            "- `python main.py --automation-scan`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a workspace automation health report.")
    parser.add_argument("--project-root", default=".", help="Project root to inspect.")
    parser.add_argument("--config", default="config/automation_workflows.json", help="Automation workflow config path.")
    parser.add_argument("--state", default="artifacts/audit/automation_state.json", help="Automation state path.")
    parser.add_argument("--runs", default="artifacts/audit/automation_runs.jsonl", help="Automation runs log path.")
    parser.add_argument(
        "--markdown",
        default="artifacts/audit/automation/project_health_report.md",
        help="Markdown output path.",
    )
    parser.add_argument(
        "--json",
        default="artifacts/audit/automation/project_health_report.json",
        help="JSON output path.",
    )
    parser.add_argument("--focus-path", action="append", default=[], help="Focus on a specific file or directory.")
    parser.add_argument("--recent-runs", type=int, default=5, help="How many recent run rows to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    state_path = (project_root / args.state).resolve() if not Path(args.state).is_absolute() else Path(args.state)
    runs_path = (project_root / args.runs).resolve() if not Path(args.runs).is_absolute() else Path(args.runs)
    markdown_path = (project_root / args.markdown).resolve() if not Path(args.markdown).is_absolute() else Path(args.markdown)
    json_path = (project_root / args.json).resolve() if not Path(args.json).is_absolute() else Path(args.json)

    report = build_report(
        project_root=project_root,
        config_path=config_path,
        state_path=state_path,
        runs_path=runs_path,
        focus_paths=[str(item) for item in list(args.focus_path or [])],
        recent_runs=max(0, int(args.recent_runs)),
    )

    _atomic_write_text(markdown_path, render_markdown(report) + "\n")
    _atomic_write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"automation report written: {markdown_path.as_posix()}")
    print(f"automation report json: {json_path.as_posix()}")


if __name__ == "__main__":
    main()
