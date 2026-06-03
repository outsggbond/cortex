from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set


@dataclass
class FileInfo:
    path: Path
    size: int
    mtime: float


def _iter_candidates(roots: Iterable[Path], suffixes: Set[str]) -> List[FileInfo]:
    out: List[FileInfo] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if suffixes and p.suffix.lower() not in suffixes:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            out.append(FileInfo(path=p, size=int(st.st_size), mtime=float(st.st_mtime)))
    return out


def _keep_latest(files: List[FileInfo], per_dir: int) -> Set[Path]:
    if per_dir <= 0:
        return set()
    by_dir: Dict[Path, List[FileInfo]] = defaultdict(list)
    for item in files:
        by_dir[item.path.parent].append(item)
    keep: Set[Path] = set()
    for items in by_dir.values():
        items.sort(key=lambda x: x.mtime, reverse=True)
        for node in items[:per_dir]:
            keep.add(node.path)
    return keep


def _prune_by_age(files: List[FileInfo], keep: Set[Path], max_age_days: float) -> List[FileInfo]:
    if max_age_days <= 0:
        return []
    cutoff = time.time() - float(max_age_days) * 24.0 * 3600.0
    stale: List[FileInfo] = []
    for item in files:
        if item.path in keep:
            continue
        if item.mtime < cutoff:
            stale.append(item)
    stale.sort(key=lambda x: x.mtime)
    return stale


def _prune_by_size(files: List[FileInfo], keep: Set[Path], max_total_gb: float) -> List[FileInfo]:
    if max_total_gb <= 0:
        return []
    cap = int(float(max_total_gb) * 1024.0 * 1024.0 * 1024.0)
    total = sum(item.size for item in files)
    if total <= cap:
        return []
    removable = [item for item in files if item.path not in keep]
    removable.sort(key=lambda x: x.mtime)
    to_delete: List[FileInfo] = []
    for item in removable:
        if total <= cap:
            break
        to_delete.append(item)
        total -= int(item.size)
    return to_delete


def _delete_files(items: Iterable[FileInfo], dry_run: bool) -> Dict[str, int]:
    deleted = 0
    failed = 0
    reclaimed = 0
    for item in items:
        if dry_run:
            deleted += 1
            reclaimed += int(item.size)
            continue
        try:
            item.path.unlink(missing_ok=True)
            deleted += 1
            reclaimed += int(item.size)
        except OSError:
            failed += 1
    return {"deleted": int(deleted), "failed": int(failed), "reclaimed_bytes": int(reclaimed)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Runtime artifact housekeeping for artifacts/audit, artifacts/checkpoints, and artifacts/memory."
    )
    p.add_argument(
        "--policy",
        type=str,
        default="config/runtime_lifecycle.json",
        help="Optional JSON policy file for default housekeeping settings",
    )
    p.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help="Directories to scan for runtime artifacts (overrides policy)",
    )
    p.add_argument(
        "--suffixes",
        nargs="+",
        default=None,
        help="File suffixes eligible for pruning (overrides policy)",
    )
    p.add_argument(
        "--max-age-days",
        type=float,
        default=None,
        help="Delete files older than this many days (0 disables, overrides policy)",
    )
    p.add_argument(
        "--max-total-gb",
        type=float,
        default=None,
        help="Trim oldest files until total size is below this cap (0 disables, overrides policy)",
    )
    p.add_argument(
        "--keep-latest-per-dir",
        type=int,
        default=None,
        help="Always keep latest N files in each directory (overrides policy)",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not delete files, only simulate")
    p.add_argument(
        "--report-path",
        type=str,
        default="artifacts/audit/runtime_housekeeping_report.json",
        help="Write JSON report to this path",
    )
    return p.parse_args()


def _load_policy(path: Path) -> Dict[str, object]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def main() -> None:
    args = parse_args()
    policy_path = Path(str(args.policy or "").strip())
    policy = _load_policy(policy_path)
    roots_raw = args.roots if args.roots is not None else (policy.get("roots") or ["audit", "checkpoints", "memory"])
    suffix_raw = args.suffixes if args.suffixes is not None else (
        policy.get("suffixes")
        or [".json", ".jsonl", ".log", ".pt", ".npz", ".bin", ".safetensors", ".pkl", ".npy", ".txt"]
    )
    max_age_days = float(args.max_age_days) if args.max_age_days is not None else float(policy.get("max_age_days", 30.0))
    max_total_gb = float(args.max_total_gb) if args.max_total_gb is not None else float(policy.get("max_total_gb", 8.0))
    keep_latest_per_dir = (
        int(args.keep_latest_per_dir)
        if args.keep_latest_per_dir is not None
        else int(policy.get("keep_latest_per_dir", 40))
    )

    roots = [Path(str(x)) for x in (roots_raw or [])]
    suffixes = {str(s).lower() for s in (suffix_raw or [])}
    files = _iter_candidates(roots, suffixes)
    keep = _keep_latest(files, keep_latest_per_dir)

    prune_age = _prune_by_age(files, keep, max_age_days)
    paths_age = {x.path for x in prune_age}
    remain_after_age = [x for x in files if x.path not in paths_age]
    prune_size = _prune_by_size(remain_after_age, keep, max_total_gb)

    # De-duplicate, age first then size.
    selected: Dict[str, FileInfo] = {}
    for node in prune_age + prune_size:
        selected[node.path.as_posix()] = node
    to_delete = list(selected.values())

    stats = _delete_files(to_delete, bool(args.dry_run))
    total_bytes = sum(item.size for item in files)
    remain_bytes = max(0, total_bytes - int(stats["reclaimed_bytes"]))

    report = {
        "roots": [r.as_posix() for r in roots],
        "policy_path": policy_path.as_posix(),
        "suffixes": sorted(list(suffixes)),
        "dry_run": bool(args.dry_run),
        "max_age_days": float(max_age_days),
        "max_total_gb": float(max_total_gb),
        "keep_latest_per_dir": int(keep_latest_per_dir),
        "scanned_files": int(len(files)),
        "scanned_bytes": int(total_bytes),
        "candidate_delete_count": int(len(to_delete)),
        "deleted_count": int(stats["deleted"]),
        "delete_failed_count": int(stats["failed"]),
        "reclaimed_bytes": int(stats["reclaimed_bytes"]),
        "remaining_bytes_estimate": int(remain_bytes),
        "deleted_preview": [x.path.as_posix() for x in to_delete[:200]],
        "timestamp": int(time.time()),
    }

    report_path = Path(str(args.report_path or "").strip() or "artifacts/audit/runtime_housekeeping_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
