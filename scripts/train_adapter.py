from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # type: ignore
from torch.utils.data import Dataset
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.learning.dialogue.dialogue_dataset_builder import DialogueDatasetBuildConfig, build_dataset
from system.knowledge.data_etl import DataCleanConfig, clean_training_file, parse_allowed_languages
from system.evaluation.adapter_artifacts import prune_adapter_artifacts, register_adapter_artifact
from system.automation.federated_runtime import (
    CoordinatorConfig,
    DiscoveryConfig,
    FedBuffConfig,
    FederatedCoordinator,
    FederatedWorker,
    PrivacyConfig,
    SecureAggConfig,
    WorkerConfig,
)
try:
    from scripts.train_adapter_cli import parse_args
except ImportError:
    # CLI module was removed — provide minimal fallback
    def parse_args(argv=None):
        import argparse
        ap = argparse.ArgumentParser(description="Train LoRA adapter")
        ap.add_argument("--train-data", default="")
        ap.add_argument("--eval-data", default="")
        ap.add_argument("--model-path", default="")
        ap.add_argument("--output-dir", default="")
        ap.add_argument("--base-adapter", default="")
        ap.add_argument("--dataset-manifest", default="")
        ap.add_argument("--replay-data", default="")
        ap.add_argument("--fed-state-path", default="")
        ap.add_argument("--fed-workers-json", default="")
        ap.add_argument("--artifact-registry-path", default="artifacts/checkpoints/registry.json")
        ap.add_argument("--artifact-prune-archive-dir", default="")
        ap.add_argument("--target-modules", default="q_proj,v_proj")
        ap.add_argument("--dtype", default="auto")
        ap.add_argument("--bnb-compute-dtype", default="fp16")
        ap.add_argument("--lora-r", type=int, default=8)
        ap.add_argument("--lora-alpha", type=int, default=16)
        ap.add_argument("--lora-dropout", type=float, default=0.05)
        ap.add_argument("--learning-rate", type=float, default=2e-4)
        ap.add_argument("--batch-size", type=int, default=4)
        ap.add_argument("--grad-accum", type=int, default=4)
        ap.add_argument("--max-length", type=int, default=2048)
        ap.add_argument("--epochs", type=float, default=3.0)
        ap.add_argument("--seed", type=int, default=42)
        ap.add_argument("--warmup-ratio", type=float, default=0.1)
        ap.add_argument("--weight-decay", type=float, default=0.01)
        ap.add_argument("--logging-steps", type=int, default=10)
        ap.add_argument("--save-steps", type=int, default=500)
        ap.add_argument("--max-samples", type=int, default=0)
        ap.add_argument("--min-quality", type=float, default=0.3)
        ap.add_argument("--eval-ratio", type=float, default=0.1)
        ap.add_argument("--hard-min-ratio", type=float, default=0.05)
        ap.add_argument("--hard-max-ratio", type=float, default=0.3)
        ap.add_argument("--max-per-prompt", type=int, default=2000)
        ap.add_argument("--max-per-response", type=int, default=4000)
        ap.add_argument("--replay-ratio", type=float, default=0.1)
        ap.add_argument("--replay-max-samples", type=int, default=2000)
        ap.add_argument("--curriculum-easy-frac", type=float, default=0.3)
        ap.add_argument("--curriculum-stage1-ratio", type=float, default=0.5)
        ap.add_argument("--artifact-max-total-gb", type=float, default=10.0)
        ap.add_argument("--artifact-keep-latest", type=int, default=5)
        ap.add_argument("--artifact-keep-promoted", type=int, default=3)
        ap.add_argument("--fed-num-clients", type=int, default=2)
        ap.add_argument("--fed-split-method", default="iid")
        ap.add_argument("--fed-min-client-samples", type=int, default=100)
        ap.add_argument("--fed-rounds", type=int, default=3)
        ap.add_argument("--fed-min-active-clients", type=int, default=1)
        ap.add_argument("--fed-client-frac", type=float, default=1.0)
        ap.add_argument("--fed-max-client-failures", type=int, default=2)
        ap.add_argument("--fed-parallel-clients", type=int, default=1)
        ap.add_argument("--fed-async", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--fed-round-timeout-seconds", type=float, default=3600.0)
        ap.add_argument("--fed-client-timeout-seconds", type=float, default=1800.0)
        ap.add_argument("--fed-client-retries", type=int, default=1)
        ap.add_argument("--fed-retry-backoff-seconds", type=float, default=10.0)
        ap.add_argument("--fed-retry-switch-worker", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--fed-aggregation", default="fedavg")
        ap.add_argument("--fed-trim-ratio", type=float, default=0.1)
        ap.add_argument("--fed-regress-loss-tol", type=float, default=0.05)
        ap.add_argument("--fed-max-drift-l1", type=float, default=0.5)
        ap.add_argument("--fed-max-fail-rate", type=float, default=0.5)
        ap.add_argument("--fed-auto-degrade", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--fed-resume", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--fed-resume-strict", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--fed-resource-aware", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--fed-resource-pressure-alpha", type=float, default=0.5)
        ap.add_argument("--fed-reprobe-each-round", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--fed-worker-probe-timeout-seconds", type=float, default=5.0)
        ap.add_argument("--fed-worker-require-healthy", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--fed-worker-fail-threshold", type=int, default=3)
        ap.add_argument("--fed-worker-fail-cooldown-rounds", type=int, default=2)
        ap.add_argument("--fed-worker-cooldown-strict", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--fed-sched-use-perf", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--fed-sched-perf-alpha", type=float, default=0.6)
        ap.add_argument("--fed-sched-perf-fail-weight", type=float, default=1.5)
        ap.add_argument("--fed-sched-perf-latency-weight", type=float, default=0.5)
        ap.add_argument("--fed-sched-resource-weight", type=float, default=0.3)
        ap.add_argument("--fed-sched-perf-weight", type=float, default=0.7)
        ap.add_argument("--fed-client-weight-strategy", default="balanced")
        ap.add_argument("--fed-client-weight-loss-eps", type=float, default=1e-6)
        ap.add_argument("--fed-client-weight-loss-clip", type=float, default=10.0)
        ap.add_argument("--fed-client-weight-sample-power", type=float, default=0.5)
        ap.add_argument("--fed-client-weight-loss-power", type=float, default=1.0)
        ap.add_argument("--fed-client-weight-eval-loss-power", type=float, default=1.0)
        ap.add_argument("--fed-client-weight-timeout-penalty", type=float, default=0.3)
        ap.add_argument("--fed-client-weight-retry-penalty", type=float, default=0.1)
        ap.add_argument("--fed-client-weight-fail-attempt-penalty", type=float, default=0.2)
        ap.add_argument("--fed-client-weight-min", type=float, default=0.01)
        ap.add_argument("--fed-client-weight-max-ratio", type=float, default=5.0)
        ap.add_argument("--fed-client-weight-normalize-mean", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--fed-client-weight-max-share", type=float, default=0.6)
        ap.add_argument("--fed-client-bucket-balance", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--fed-client-bucket-balance-strength", type=float, default=0.3)
        ap.add_argument("--fed-client-bucket-balance-eps", type=float, default=0.01)
        ap.add_argument("--fed-client-bucket-target", default="uniform")
        ap.add_argument("--fed-prune-client-artifacts", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--fed-prune-keep-failed-clients", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--fed-prune-rounds-keep", type=int, default=2)
        ap.add_argument("--fed-dist-workers", type=int, default=1)
        ap.add_argument("--local-only", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--load-in-4bit", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--gradient-checkpointing", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--group-by-length", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--compact-output", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--curriculum", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--keep-templatey", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--no-bucket-classifier", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--bucket-classifier-min-conf", type=float, default=0.7)
        ap.add_argument("--bucket-classifier-min-docs", type=int, default=5)
        ap.add_argument("--bucket-classifier-alpha", type=float, default=0.3)
        ap.add_argument("--bucket-classifier-force-override", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--auto-build", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--federated", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--preset-7840hs", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--no-register-artifact", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--artifact-prune", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--artifact-prune-remove-invalid", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--artifact-prune-delete-files", type=lambda x: x.lower() in ("1", "true", "yes"), default=True)
        ap.add_argument("--artifact-prune-archive", type=lambda x: x.lower() in ("1", "true", "yes"), default=False)
        ap.add_argument("--_stage-epochs", type=float, default=1.0)
        return ap.parse_args(argv or sys.argv[1:])


def _open_text_auto(path: Path, mode: str):
    p = Path(path)
    m = str(mode)
    if "b" in m:
        m = m.replace("b", "")
    if "t" not in m:
        m = m + "t"
    if p.suffix.lower() == ".gz":
        return gzip.open(p.as_posix(), m, encoding="utf-8")
    return p.open(m, encoding="utf-8")


def _hf_cache_roots() -> List[Path]:
    roots: List[Path] = []
    home = Path.home()
    env_hf_home = str(os.environ.get("HF_HOME", "") or "").strip()
    env_hf_cache = str(os.environ.get("HUGGINGFACE_HUB_CACHE", "") or "").strip()
    if env_hf_cache:
        roots.append(Path(env_hf_cache).expanduser())
    if env_hf_home:
        roots.append(Path(env_hf_home).expanduser() / "hub")
    roots.append(home / ".cache" / "huggingface" / "hub")
    roots.append(home / ".huggingface" / "hub")
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: List[Path] = []
    for p in roots:
        key = p.as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _resolve_hf_cache_model_path(model_ref: str) -> str:
    ref = str(model_ref or "").strip()
    if not ref:
        return ""
    # Already a local path.
    p = Path(os.path.expanduser(ref))
    if p.exists():
        return p.as_posix()
    # Common hub repo id -> cached repo folder mapping.
    repo_dir_name = "models--" + ref.replace("/", "--")
    for root in _hf_cache_roots():
        try:
            repo_dir = root / repo_dir_name
            snaps_dir = repo_dir / "snapshots"
            if not snaps_dir.exists() or (not snaps_dir.is_dir()):
                continue
            snapshots: List[Path] = []
            for d in snaps_dir.iterdir():
                if not d.is_dir():
                    continue
                if (d / "config.json").exists():
                    snapshots.append(d)
            if not snapshots:
                continue
            snapshots.sort(key=lambda x: float(x.stat().st_mtime), reverse=True)
            return snapshots[0].as_posix()
        except Exception:
            continue
    return ""


def _resolve_model_ref(model_ref: str) -> Tuple[str, bool]:
    ref = str(model_ref or "").strip()
    if not ref:
        return "", False
    p = Path(os.path.expanduser(ref))
    if p.exists():
        return p.as_posix(), True
    cached = _resolve_hf_cache_model_path(ref)
    if cached:
        return cached, True
    return ref, False


def _base_model_ref(model_ref: str) -> str:
    resolved, _ = _resolve_model_ref(model_ref)
    return str(resolved or str(model_ref or "").strip())


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with _open_text_auto(path, "r") as f:
            for line in f:
                line = str(line).strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    out.append(item)
    except Exception:
        return []
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text_auto(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with _open_text_auto(path, "r") as f:
            raw = json.loads(f.read())
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text_auto(path, "w") as f:
        f.write(json.dumps(payload, ensure_ascii=False, indent=2))


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text_auto(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _path_storage_stats(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "file_count": 0,
        }
    files: List[Path] = []
    if p.is_file():
        files = [p]
    else:
        files = [x for x in p.rglob("*") if x.is_file()]
    size_bytes = 0
    for x in files:
        try:
            size_bytes += int(x.stat().st_size)
        except Exception:
            continue
    return {
        "exists": True,
        "size_bytes": int(size_bytes),
        "size_mb": round(float(size_bytes) / (1024.0 * 1024.0), 4),
        "file_count": int(len(files)),
    }


def _compact_training_output_dir(output_dir: Path) -> Dict[str, Any]:
    removed_dirs: List[str] = []
    removed_files: List[str] = []
    candidates: List[Path] = []
    for name in ("single", "curriculum_stage1", "curriculum_stage2"):
        candidates.append(output_dir / name)
    candidates.extend([x for x in output_dir.glob("checkpoint-*") if x.is_dir()])
    for p in candidates:
        if not p.exists():
            continue
        if p.name == "adapter":
            continue
        try:
            shutil.rmtree(p.as_posix(), ignore_errors=True)
            removed_dirs.append(p.as_posix())
        except Exception:
            continue
    for name in ("trainer_state.json", "all_results.json"):
        fp = output_dir / name
        if not fp.exists() or not fp.is_file():
            continue
        try:
            fp.unlink()
            removed_files.append(fp.as_posix())
        except Exception:
            continue
    return {
        "enabled": True,
        "removed_dir_count": int(len(removed_dirs)),
        "removed_file_count": int(len(removed_files)),
        "removed_dirs": removed_dirs,
        "removed_files": removed_files,
    }


def _register_artifact_from_summary(
    *,
    registry_path: str,
    adapter_path: str,
    base_model: str,
    report_path: str = "",
    source: str = "train_adapter",
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        node = register_adapter_artifact(
            registry_path=str(registry_path),
            adapter_path=str(adapter_path),
            base_model=str(base_model),
            report_path=str(report_path),
            source=str(source),
            promoted=False,
            used=True,
            metrics=(metrics if isinstance(metrics, dict) else None),
        )
        return {
            "ok": True,
            "id": str(node.get("id", "")),
            "path": str(node.get("adapter_path", "")),
            "registry_path": str(registry_path),
            "size_bytes": int(node.get("size_bytes", 0)),
            "file_count": int(node.get("file_count", 0)),
        }
    except Exception as e:
        return {
            "ok": False,
            "error": f"{e}",
            "registry_path": str(registry_path),
        }


def _maybe_prune_artifacts(args: argparse.Namespace, *, protect_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    if not bool(args.artifact_prune):
        return {"enabled": False}
    max_total_bytes = int(max(0.0, float(args.artifact_max_total_gb)) * (1024 ** 3))
    try:
        stats = prune_adapter_artifacts(
            registry_path=str(args.artifact_registry_path),
            keep_latest=max(0, int(args.artifact_keep_latest)),
            keep_promoted=max(0, int(args.artifact_keep_promoted)),
            max_total_bytes=max_total_bytes,
            protected_paths=list(protect_paths or []),
            remove_missing=True,
            remove_invalid=bool(args.artifact_prune_remove_invalid),
            delete_files=bool(args.artifact_prune_delete_files),
            archive_before_delete=bool(args.artifact_prune_archive),
            archive_dir=str(args.artifact_prune_archive_dir or "").strip(),
        )
        return {
            "enabled": True,
            "registry_path": str(args.artifact_registry_path),
            "keep_latest": int(args.artifact_keep_latest),
            "keep_promoted": int(args.artifact_keep_promoted),
            "max_total_gb": float(args.artifact_max_total_gb),
            "remove_invalid": bool(args.artifact_prune_remove_invalid),
            "archive_before_delete": bool(args.artifact_prune_archive),
            "archive_dir": str(args.artifact_prune_archive_dir or ""),
            **stats,
        }
    except Exception as e:
        return {
            "enabled": True,
            "error": f"{e}",
            "registry_path": str(args.artifact_registry_path),
        }


def _tail(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _run(cmd: List[str], cwd: Path, timeout_s: float = 0.0) -> Dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd.as_posix(),
            text=True,
            capture_output=True,
            timeout=(None if float(timeout_s) <= 0 else float(timeout_s)),
        )
    except subprocess.TimeoutExpired as e:
        elapsed = max(0.0, time.time() - start)
        return {
            "command": cmd,
            "returncode": -9,
            "elapsed_s": round(elapsed, 3),
            "stdout_tail": _tail(getattr(e, "stdout", "") or ""),
            "stderr_tail": _tail(getattr(e, "stderr", "") or ""),
            "timeout": True,
        }
    elapsed = max(0.0, time.time() - start)
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "elapsed_s": round(elapsed, 3),
        "stdout_tail": _tail(proc.stdout or ""),
        "stderr_tail": _tail(proc.stderr or ""),
        "timeout": False,
    }


def _default_local_worker() -> Dict[str, Any]:
    return {
        "name": "local",
        "enabled": True,
        "cmd_prefix": [],
        "python_bin": sys.executable,
        "root_dir": ROOT.as_posix(),
        "max_concurrency": 1,
        "probe_cmd": [sys.executable, "-V"],
        "resource_probe_cmd": [],
        "resource_weight": 1.0,
    }


def _norm_cmd_list(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for x in value:
            t = str(x).strip()
            if t:
                out.append(t)
        return out
    text = str(value or "").strip()
    if not text:
        return []
    return [x for x in text.split(" ") if str(x).strip()]


def _normalize_worker_node(node: Dict[str, Any], idx: int) -> Dict[str, Any]:
    name = str(node.get("name", f"worker_{idx}")).strip() or f"worker_{idx}"
    cmd_prefix = _norm_cmd_list(node.get("cmd_prefix", []))
    python_bin = str(node.get("python_bin", "")).strip() or sys.executable
    root_dir = str(node.get("root_dir", "")).strip()
    max_conc = max(1, int(node.get("max_concurrency", 1)))
    probe_cmd = _norm_cmd_list(node.get("probe_cmd", []))
    resource_probe_cmd = _norm_cmd_list(node.get("resource_probe_cmd", []))
    try:
        resource_weight = float(node.get("resource_weight", 1.0))
    except Exception:
        resource_weight = 1.0
    resource_weight = max(0.1, min(20.0, float(resource_weight)))
    if not probe_cmd:
        probe_cmd = [python_bin, "-V"]
    return {
        "name": name,
        "enabled": bool(node.get("enabled", True)),
        "cmd_prefix": cmd_prefix,
        "python_bin": python_bin,
        "root_dir": root_dir,
        "max_concurrency": int(max_conc),
        "probe_cmd": probe_cmd,
        "resource_probe_cmd": resource_probe_cmd,
        "resource_weight": float(resource_weight),
    }


def _load_federated_workers(args: argparse.Namespace) -> List[Dict[str, Any]]:
    src = str(args.fed_workers_json or "").strip()
    if not src:
        return [_default_local_worker()]
    raw = _read_json(Path(src))
    rows = raw.get("workers", raw if isinstance(raw, list) else [])
    out: List[Dict[str, Any]] = []
    if isinstance(rows, list):
        for i, node in enumerate(rows):
            if not isinstance(node, dict):
                continue
            w = _normalize_worker_node(node, i + 1)
            if bool(w.get("enabled", True)):
                out.append(w)
    if out:
        return out
    return [_default_local_worker()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def _parse_worker_resource_from_probe_output(text: str) -> Dict[str, Any]:
    src = str(text or "").strip()
    if not src:
        return {}
    candidates: List[str] = [src]
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    candidates.extend(reversed(lines))
    raw: Dict[str, Any] = {}
    for c in candidates:
        if "{" not in c or "}" not in c:
            continue
        left = c.find("{")
        right = c.rfind("}")
        if left < 0 or right < left:
            continue
        seg = c[left : right + 1]
        try:
            node = json.loads(seg)
        except Exception:
            continue
        if isinstance(node, dict):
            raw = node
            break
    if not raw:
        return {}
    out: Dict[str, Any] = {}
    if raw.get("pressure", None) is not None:
        out["pressure"] = _clamp01(_safe_float(raw.get("pressure"), 0.0))
    for key in ("cpu_percent", "mem_percent", "gpu_percent"):
        if raw.get(key, None) is None:
            continue
        out[key] = float(min(100.0, max(0.0, _safe_float(raw.get(key), 0.0))))
    if raw.get("free_mem_mb", None) is not None:
        out["free_mem_mb"] = float(max(0.0, _safe_float(raw.get("free_mem_mb"), 0.0)))
    return out


def _run_worker_resource_probe(worker: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    probe = _norm_cmd_list(worker.get("resource_probe_cmd", []))
    if not probe:
        return {"ok": False, "resource": {}, "step": {}}
    cmd = list(worker.get("cmd_prefix", [])) + probe
    step = _run(cmd, cwd=ROOT, timeout_s=float(timeout_s))
    ok = bool(int(step.get("returncode", -1)) == 0 and not bool(step.get("timeout", False)))
    resource = _parse_worker_resource_from_probe_output(str(step.get("stdout_tail", ""))) if ok else {}
    return {"ok": bool(ok), "resource": resource, "step": step}


def _worker_pressure(worker: Dict[str, Any], probe: Dict[str, Any], alpha: float) -> float:
    res: Dict[str, Any] = {}
    if isinstance(probe, dict):
        rnode = probe.get("resource", {})
        if isinstance(rnode, dict):
            res = rnode
    pressure = None
    if res.get("pressure", None) is not None:
        pressure = _clamp01(_safe_float(res.get("pressure"), 0.0))
    vals: List[float] = []
    for k in ("cpu_percent", "mem_percent", "gpu_percent"):
        if res.get(k, None) is None:
            continue
        vals.append(_clamp01(_safe_float(res.get(k), 0.0) / 100.0))
    if pressure is None:
        pressure = float(sum(vals) / len(vals)) if vals else 0.0
    scaled = _clamp01(float(pressure) * float(max(0.0, alpha)))
    weight = max(0.1, _safe_float(worker.get("resource_weight", 1.0), 1.0))
    # Higher resource_weight means the worker can absorb more load.
    return _clamp01(float(scaled) / float(weight))


def _build_worker_pressure_map(
    workers: List[Dict[str, Any]],
    worker_probes: List[Dict[str, Any]],
    *,
    enabled: bool,
    alpha: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not enabled:
        return out
    by_name = {str(x.get("name", "")): x for x in worker_probes if isinstance(x, dict)}
    for w in workers:
        name = str(w.get("name", ""))
        if not name:
            continue
        out[name] = _worker_pressure(w, by_name.get(name, {}), alpha=float(alpha))
    return out


def _sanitize_worker_health_state(
    state: Dict[str, Any],
    workers: List[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    src = state if isinstance(state, dict) else {}
    out: Dict[str, Dict[str, int]] = {}
    for w in workers:
        name = str(w.get("name", "")).strip()
        if not name:
            continue
        node = src.get(name, {})
        if not isinstance(node, dict):
            node = {}
        out[name] = {
            "total_success": max(0, int(node.get("total_success", 0))),
            "total_fail": max(0, int(node.get("total_fail", 0))),
            "total_timeout": max(0, int(node.get("total_timeout", 0))),
            "consecutive_fail_rounds": max(0, int(node.get("consecutive_fail_rounds", 0))),
            "cooldown_until_round": max(0, int(node.get("cooldown_until_round", 0))),
            "cooldown_events": max(0, int(node.get("cooldown_events", 0))),
        }
    return out


def _active_workers_for_round(
    workers: List[Dict[str, Any]],
    worker_health_state: Dict[str, Dict[str, int]],
    *,
    round_id: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    active: List[Dict[str, Any]] = []
    cooled: List[str] = []
    rid = int(round_id)
    for w in workers:
        name = str(w.get("name", "")).strip()
        if not name:
            continue
        node = worker_health_state.get(name, {})
        until = int(node.get("cooldown_until_round", 0)) if isinstance(node, dict) else 0
        if until > rid:
            cooled.append(name)
            continue
        active.append(w)
    return active, cooled


def _update_worker_health_after_round(
    worker_health_state: Dict[str, Dict[str, int]],
    *,
    client_steps: List[Dict[str, Any]],
    round_id: int,
    fail_threshold: int,
    cooldown_rounds: int,
) -> Dict[str, Any]:
    stats: Dict[str, Dict[str, int]] = {}
    for item in client_steps:
        attempts = item.get("attempts", [])
        rows: List[Dict[str, Any]] = []
        if isinstance(attempts, list) and attempts:
            rows = [x for x in attempts if isinstance(x, dict)]
        if not rows:
            rows = [
                {
                    "worker": str(item.get("worker", "")),
                    "ok": bool(item.get("ok", False)),
                    "timeout": bool((item.get("step") or {}).get("timeout", False))
                    if isinstance(item.get("step"), dict)
                    else False,
                }
            ]
        for row in rows:
            name = str(row.get("worker", "")).strip()
            if not name:
                continue
            node = stats.setdefault(name, {"success": 0, "fail": 0, "timeout": 0})
            ok = bool(row.get("ok", False))
            node["success"] += 1 if ok else 0
            node["fail"] += 0 if ok else 1
            node["timeout"] += 1 if bool(row.get("timeout", False)) else 0

    out = {str(k): dict(v) for k, v in worker_health_state.items()}
    cooled_this_round: List[str] = []
    for name, row in stats.items():
        cur = out.get(
            name,
            {
                "total_success": 0,
                "total_fail": 0,
                "total_timeout": 0,
                "consecutive_fail_rounds": 0,
                "cooldown_until_round": 0,
                "cooldown_events": 0,
            },
        )
        succ = int(row.get("success", 0))
        fail = int(row.get("fail", 0))
        tout = int(row.get("timeout", 0))
        cur["total_success"] = int(cur.get("total_success", 0)) + succ
        cur["total_fail"] = int(cur.get("total_fail", 0)) + fail
        cur["total_timeout"] = int(cur.get("total_timeout", 0)) + tout
        if fail > 0 and succ <= 0:
            cur["consecutive_fail_rounds"] = int(cur.get("consecutive_fail_rounds", 0)) + 1
        elif succ > 0:
            cur["consecutive_fail_rounds"] = 0

        if (
            int(fail_threshold) > 0
            and int(cooldown_rounds) > 0
            and int(cur.get("consecutive_fail_rounds", 0)) >= int(fail_threshold)
        ):
            cur["cooldown_until_round"] = max(
                int(cur.get("cooldown_until_round", 0)),
                int(round_id) + int(cooldown_rounds),
            )
            cur["consecutive_fail_rounds"] = 0
            cur["cooldown_events"] = int(cur.get("cooldown_events", 0)) + 1
            cooled_this_round.append(name)
        out[name] = cur

    return {
        "state": out,
        "round_stats": stats,
        "cooled_this_round": cooled_this_round,
    }


def _sanitize_worker_perf_state(
    state: Dict[str, Any],
    workers: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    src = state if isinstance(state, dict) else {}
    out: Dict[str, Dict[str, float]] = {}
    for w in workers:
        name = str(w.get("name", "")).strip()
        if not name:
            continue
        node = src.get(name, {})
        if not isinstance(node, dict):
            node = {}
        out[name] = {
            "ema_elapsed_s": max(0.0, _safe_float(node.get("ema_elapsed_s", 0.0), 0.0)),
            "ema_fail_rate": _clamp01(_safe_float(node.get("ema_fail_rate", 0.0), 0.0)),
            "total_attempts": float(max(0, int(_safe_float(node.get("total_attempts", 0), 0)))),
        }
    return out


def _update_worker_perf_after_round(
    worker_perf_state: Dict[str, Dict[str, float]],
    *,
    client_steps: List[Dict[str, Any]],
    workers: List[Dict[str, Any]],
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    a = _clamp01(alpha)
    if a <= 1e-6:
        a = 0.3
    out: Dict[str, Dict[str, float]] = _sanitize_worker_perf_state(worker_perf_state, workers)
    stats: Dict[str, Dict[str, float]] = {}
    for item in client_steps:
        attempts = item.get("attempts", [])
        rows: List[Dict[str, Any]] = []
        if isinstance(attempts, list) and attempts:
            rows = [x for x in attempts if isinstance(x, dict)]
        if not rows:
            step = item.get("step", {})
            rows = [
                {
                    "worker": str(item.get("worker", "")),
                    "ok": bool(item.get("ok", False)),
                    "elapsed_s": _safe_float((step or {}).get("elapsed_s", 0.0), 0.0) if isinstance(step, dict) else 0.0,
                }
            ]
        for row in rows:
            name = str(row.get("worker", "")).strip()
            if not name:
                continue
            node = stats.setdefault(name, {"attempts": 0.0, "fail": 0.0, "elapsed_total": 0.0})
            node["attempts"] += 1.0
            node["fail"] += 0.0 if bool(row.get("ok", False)) else 1.0
            node["elapsed_total"] += max(0.0, _safe_float(row.get("elapsed_s", 0.0), 0.0))

    for name, node in stats.items():
        cur = out.get(
            name,
            {"ema_elapsed_s": 0.0, "ema_fail_rate": 0.0, "total_attempts": 0.0},
        )
        attempts = max(1.0, float(node.get("attempts", 1.0)))
        fail_rate = _clamp01(float(node.get("fail", 0.0)) / attempts)
        avg_elapsed = max(0.0, float(node.get("elapsed_total", 0.0)) / attempts)

        prev_elapsed = max(0.0, _safe_float(cur.get("ema_elapsed_s", 0.0), 0.0))
        prev_fail = _clamp01(_safe_float(cur.get("ema_fail_rate", 0.0), 0.0))
        if float(cur.get("total_attempts", 0.0)) <= 0.0:
            cur["ema_elapsed_s"] = float(avg_elapsed)
            cur["ema_fail_rate"] = float(fail_rate)
        else:
            cur["ema_elapsed_s"] = float((1.0 - a) * prev_elapsed + a * avg_elapsed)
            cur["ema_fail_rate"] = float((1.0 - a) * prev_fail + a * fail_rate)
        cur["total_attempts"] = float(max(0.0, _safe_float(cur.get("total_attempts", 0.0), 0.0)) + attempts)
        out[name] = cur
    return out


def _build_worker_perf_penalty_map(
    workers: List[Dict[str, Any]],
    worker_perf_state: Dict[str, Dict[str, float]],
    *,
    enabled: bool,
    fail_weight: float,
    latency_weight: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not enabled:
        return out
    fw = max(0.0, float(fail_weight))
    lw = max(0.0, float(latency_weight))
    states = _sanitize_worker_perf_state(worker_perf_state, workers)
    elapsed_samples = [max(0.0, _safe_float(v.get("ema_elapsed_s", 0.0), 0.0)) for v in states.values() if _safe_float(v.get("total_attempts", 0.0), 0.0) > 0.0]
    baseline = 0.0
    if elapsed_samples:
        ordered = sorted(elapsed_samples)
        baseline = float(ordered[len(ordered) // 2])
    for w in workers:
        name = str(w.get("name", "")).strip()
        if not name:
            continue
        node = states.get(name, {})
        fail_rate = _clamp01(_safe_float(node.get("ema_fail_rate", 0.0), 0.0))
        elapsed = max(0.0, _safe_float(node.get("ema_elapsed_s", 0.0), 0.0))
        latency_pen = 0.0
        if baseline > 1e-6:
            latency_pen = _clamp01((elapsed / baseline - 1.0) / 2.0)
        penalty = _clamp01(fail_rate * fw + latency_pen * lw)
        out[name] = penalty
    return out


def _merge_worker_pressure_maps(
    primary: Dict[str, float],
    secondary: Dict[str, float],
    *,
    primary_weight: float,
    secondary_weight: float,
) -> Dict[str, float]:
    keys = set(primary.keys()) | set(secondary.keys())
    out: Dict[str, float] = {}
    pw = max(0.0, float(primary_weight))
    sw = max(0.0, float(secondary_weight))
    for k in keys:
        v = _safe_float(primary.get(k, 0.0), 0.0) * pw + _safe_float(secondary.get(k, 0.0), 0.0) * sw
        out[str(k)] = _clamp01(v)
    return out


def _probe_federated_worker(worker: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    cmd = list(worker.get("cmd_prefix", [])) + list(worker.get("probe_cmd", []))
    step = _run(cmd, cwd=ROOT, timeout_s=float(timeout_s))
    ok = bool(int(step.get("returncode", -1)) == 0 and not bool(step.get("timeout", False)))
    resource_probe = _run_worker_resource_probe(worker, timeout_s=float(timeout_s)) if ok else {"ok": False, "resource": {}}
    return {
        "name": str(worker.get("name", "")),
        "ok": bool(ok),
        "step": step,
        "max_concurrency": int(worker.get("max_concurrency", 1)),
        "resource_probe": resource_probe,
    }


def _assign_clients_to_workers(
    client_ids: List[int],
    workers: List[Dict[str, Any]],
    *,
    seed: int,
    worker_pressures: Optional[Dict[str, float]] = None,
) -> Dict[int, Dict[str, Any]]:
    if not workers:
        return {}
    rng = random.Random(int(seed))
    ids = [int(x) for x in client_ids]
    rng.shuffle(ids)
    load: Dict[str, int] = {str(w.get("name", "")): 0 for w in workers}
    pressures = dict(worker_pressures or {})
    by_id: Dict[int, Dict[str, Any]] = {}
    for cid in ids:
        best = workers[0]
        best_ratio = 1e9
        for w in workers:
            name = str(w.get("name", ""))
            cap = max(1, int(w.get("max_concurrency", 1)))
            pressure = _clamp01(_safe_float(pressures.get(name, 0.0), 0.0))
            ratio = (float(load.get(name, 0)) + pressure * float(cap + 1)) / float(cap)
            if ratio < best_ratio:
                best = w
                best_ratio = ratio
        bname = str(best.get("name", ""))
        load[bname] = int(load.get(bname, 0)) + 1
        by_id[int(cid)] = best
    return by_id


def _bucket_distribution(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}
    counts: Dict[str, int] = {}
    for row in rows:
        b = str(row.get("bucket", "")).strip().lower() or "smalltalk"
        counts[b] = int(counts.get(b, 0)) + 1
    total = float(sum(counts.values()))
    return {k: float(v / total) for k, v in counts.items()}


def _l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    return float(sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def _jain_fairness(values: List[float]) -> float:
    vals = [max(0.0, float(x)) for x in values if float(x) >= 0.0]
    if not vals:
        return 1.0
    s = float(sum(vals))
    ss = float(sum(x * x for x in vals))
    if ss <= 1e-12:
        return 1.0
    n = float(len(vals))
    return float((s * s) / (n * ss))


def _evaluate_federated_governance(
    *,
    client_steps: List[Dict[str, Any]],
    expected_clients: int,
    selected_rows: List[Dict[str, Any]],
    global_rows: List[Dict[str, Any]],
    best_loss: Optional[float],
    regress_tol: float,
    max_drift_l1: float,
    max_fail_rate: float,
) -> Dict[str, Any]:
    total = max(1, int(expected_clients))
    ok_steps = [x for x in client_steps if bool(x.get("ok", False))]
    fail_rate = float(1.0 - (len(ok_steps) / float(total)))

    losses: List[float] = []
    for item in ok_steps:
        summary = item.get("summary", {})
        if isinstance(summary, dict):
            try:
                losses.append(float(summary.get("train_loss", 0.0)))
            except Exception:
                continue
    round_loss = float(sum(losses) / len(losses)) if losses else None
    regress = False
    if round_loss is not None and best_loss is not None:
        regress = bool(round_loss > float(best_loss) + float(max(0.0, regress_tol)))

    drift = _l1_distance(_bucket_distribution(selected_rows), _bucket_distribution(global_rows))
    by_worker_rows: Dict[str, float] = {}
    by_worker_ok: Dict[str, float] = {}
    for item in client_steps:
        w = str(item.get("worker", ""))
        by_worker_rows[w] = float(by_worker_rows.get(w, 0.0) + float(item.get("rows", 0)))
        by_worker_ok[w] = float(by_worker_ok.get(w, 0.0) + (1.0 if bool(item.get("ok", False)) else 0.0))
    fairness_rows = _jain_fairness(list(by_worker_rows.values()))
    fairness_success = _jain_fairness(list(by_worker_ok.values()))

    reasons: List[str] = []
    if fail_rate > float(max_fail_rate):
        reasons.append(f"fail_rate({fail_rate:.4f}>{float(max_fail_rate):.4f})")
    if drift > float(max_drift_l1):
        reasons.append(f"drift_l1({drift:.4f}>{float(max_drift_l1):.4f})")
    if regress:
        reasons.append(
            f"train_loss_regress({float(round_loss):.6f}>{float(best_loss):.6f}+{float(regress_tol):.6f})"
        )
    passed = not bool(reasons)
    return {
        "passed": bool(passed),
        "reasons": reasons,
        "fail_rate": round(float(fail_rate), 6),
        "drift_l1": round(float(drift), 6),
        "round_train_loss": (None if round_loss is None else round(float(round_loss), 6)),
        "best_train_loss_before": (None if best_loss is None else round(float(best_loss), 6)),
        "fairness_rows_jain": round(float(fairness_rows), 6),
        "fairness_success_jain": round(float(fairness_success), 6),
    }


def _pair_key(row: Dict[str, Any]) -> Tuple[str, str]:
    prompt = str(row.get("prompt") or row.get("user") or "").strip()
    response = str(row.get("response") or row.get("assistant") or "").strip()
    return prompt, response


def _row_score(row: Dict[str, Any]) -> float:
    try:
        score = float(row.get("score", 0.6))
    except Exception:
        score = 0.6
    score = max(0.0, min(1.0, score))
    if bool(row.get("hard_sample", False)):
        score = min(1.0, score + 0.08)
    return score


def _row_difficulty(row: Dict[str, Any]) -> float:
    score = _row_score(row)
    bucket = str(row.get("bucket", "")).strip().lower()
    penalty = 0.0
    if bucket == "reasoning":
        penalty += 0.08
    elif bucket == "safety":
        penalty += 0.10
    elif bucket == "qa":
        penalty += 0.04
    if bool(row.get("hard_sample", False)):
        penalty += 0.08
    return max(0.0, min(1.5, (1.0 - score) + penalty))


def _dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = _pair_key(row)
        if not key[0] or not key[1]:
            continue
        cur = best.get(key)
        if cur is None or _row_score(row) > _row_score(cur):
            best[key] = dict(row)
    return list(best.values())


def _select_replay_rows(
    replay_rows: List[Dict[str, Any]],
    *,
    base_count: int,
    replay_ratio: float,
    replay_max_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if replay_ratio <= 0.0 or base_count <= 0:
        return []
    target = int(round(float(base_count) * float(replay_ratio)))
    if replay_max_samples > 0:
        target = min(target, int(replay_max_samples))
    if target <= 0:
        return []

    pool = _dedup_rows(replay_rows)
    if not pool:
        return []

    rng = random.Random(int(seed))
    pool = sorted(pool, key=lambda x: (_row_score(x), -_row_difficulty(x)), reverse=True)
    head_n = min(len(pool), max(target, target * 4))
    head = pool[:head_n]
    rng.shuffle(head)
    return head[: min(target, len(head))]


def _split_curriculum_rows(
    rows: List[Dict[str, Any]],
    *,
    easy_frac: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not rows:
        return [], []
    ordered = sorted(rows, key=_row_difficulty)
    cut = int(round(len(ordered) * float(easy_frac)))
    cut = min(max(1, cut), max(1, len(ordered) - 1))
    easy = list(ordered[:cut])
    hard = list(ordered[cut:])
    rng = random.Random(int(seed))
    rng.shuffle(easy)
    rng.shuffle(hard)
    return easy, hard


class DialogueSFTDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int) -> None:
        self.rows: List[Dict[str, Any]] = []
        eos_id = tokenizer.eos_token_id
        for row in rows:
            prompt = str(row.get("prompt") or row.get("user") or "").strip()
            response = str(row.get("response") or row.get("assistant") or "").strip()
            if not prompt or not response:
                continue
            prompt_text = f"User: {prompt}\nAssistant:"
            full_text = f"{prompt_text} {response}"
            p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            f_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if eos_id is not None and (not f_ids or f_ids[-1] != eos_id):
                f_ids = f_ids + [int(eos_id)]
            labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
            input_ids = f_ids
            if len(input_ids) > max_length:
                start = len(input_ids) - int(max_length)
                input_ids = input_ids[start:]
                labels = labels[start:]
            if not input_ids:
                continue
            if all(int(x) == -100 for x in labels):
                continue
            self.rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


@dataclass
class PackedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class DialogueCollator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = int(pad_id)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids: List[List[int]] = []
        masks: List[List[int]] = []
        labels: List[List[int]] = []
        for f in features:
            ln = len(f["input_ids"])
            pad = max_len - ln
            input_ids.append(list(f["input_ids"]) + [self.pad_id] * pad)
            masks.append(list(f["attention_mask"]) + [0] * pad)
            labels.append(list(f["labels"]) + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _resolve_dtype(dtype: str):
    key = str(dtype or "").strip().lower()
    if key == "bf16":
        return torch.bfloat16
    if key == "fp16":
        return torch.float16
    if key == "fp32":
        return torch.float32
    return None


def _apply_7840hs_preset(args: argparse.Namespace) -> Dict[str, Any]:
    changed: Dict[str, Any] = {}

    def _maybe_set(name: str, value: Any, default_value: Any) -> None:
        if getattr(args, name) == default_value:
            setattr(args, name, value)
            changed[name] = value

    _maybe_set("batch_size", 1, 1)
    _maybe_set("grad_accum", 32, 16)
    _maybe_set("max_length", 320, 384)
    _maybe_set("save_steps", 300, 200)
    _maybe_set("logging_steps", 20, 10)
    _maybe_set("warmup_ratio", 0.02, 0.03)
    _maybe_set("lora_r", 8, 8)
    _maybe_set("lora_alpha", 16, 16)
    _maybe_set("lora_dropout", 0.08, 0.05)
    _maybe_set("group_by_length", True, False)
    _maybe_set("gradient_checkpointing", True, False)
    _maybe_set("auto_build", True, False)
    _maybe_set("max_samples", 12000, 20000)
    _maybe_set("hard_min_ratio", 0.40, 0.30)
    _maybe_set("hard_max_ratio", 0.75, 0.65)
    _maybe_set("curriculum", True, False)
    _maybe_set("curriculum_easy_frac", 0.45, 0.40)
    _maybe_set("curriculum_stage1_ratio", 0.35, 0.30)
    _maybe_set("replay_ratio", 0.12, 0.0)
    if args.dtype == "auto":
        args.dtype = "fp16" if torch.cuda.is_available() else "fp32"
        changed["dtype"] = args.dtype
    if torch.cuda.is_available() and not bool(args.load_in_4bit):
        args.load_in_4bit = True
        changed["load_in_4bit"] = True
    return changed


def _resolve_bnb_compute_dtype(name: str, fallback: torch.dtype) -> torch.dtype:
    key = str(name or "").strip().lower()
    if key == "bf16":
        return torch.bfloat16
    if key == "fp32":
        return torch.float32
    return torch.float16 if fallback is None else fallback


def _build_quantization_config(
    *,
    load_in_4bit: bool,
    compute_dtype: torch.dtype,
) -> Tuple[Optional[Any], str]:
    if not load_in_4bit:
        return None, "disabled"
    if not torch.cuda.is_available():
        return None, "disabled_no_cuda"
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except Exception:
        return None, "disabled_bitsandbytes_missing"
    cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    return cfg, "enabled"


def _auto_build_if_needed(train_path: Path, eval_path: Path, args) -> Optional[Dict[str, Any]]:
    if train_path.exists() and eval_path.exists():
        return None
    cfg = DialogueDatasetBuildConfig(
        output_train_path=str(train_path),
        output_eval_path=str(eval_path),
        output_manifest_path=str(args.dataset_manifest),
        min_quality=float(args.min_quality),
        max_samples=int(args.max_samples),
        eval_ratio=float(args.eval_ratio),
        seed=int(args.seed),
        hard_min_ratio=float(args.hard_min_ratio),
        hard_max_ratio=float(args.hard_max_ratio),
        max_per_prompt=int(args.max_per_prompt),
        max_per_response=int(args.max_per_response),
        prevent_cross_split_prompt_leakage=not bool(getattr(args, "allow_cross_split_prompt", False)),
        diversity_penalty_enabled=not bool(getattr(args, "disable_diversity_penalty", False)),
        diversity_similarity_threshold=float(getattr(args, "diversity_similarity_threshold", 0.82)),
        diversity_penalty_strength=float(getattr(args, "diversity_penalty_strength", 0.20)),
        diversity_reference_cap=int(getattr(args, "diversity_reference_cap", 12)),
        strip_templatey_responses=not bool(args.keep_templatey),
        bucket_classifier_enabled=not bool(args.no_bucket_classifier),
        bucket_classifier_min_conf=float(args.bucket_classifier_min_conf),
        bucket_classifier_min_docs=int(args.bucket_classifier_min_docs),
        bucket_classifier_alpha=float(args.bucket_classifier_alpha),
        bucket_classifier_force_override=bool(args.bucket_classifier_force_override),
    )
    return build_dataset(cfg)


def _normalize_clean_report_path(path_text: str) -> Path:
    raw = str(path_text or "").strip()
    p = Path(raw) if raw else (Path("audit") / "data_clean_report.json")
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _resolve_clean_output(
    src: Path,
    *,
    args: argparse.Namespace,
    in_place: bool,
    out_dir: Path,
) -> Path:
    if in_place:
        return src
    suffix = str(src.suffix or "").strip()
    suffix = suffix if suffix else ".txt"
    return out_dir / f"{src.stem}.clean{suffix}"


def _maybe_clean_training_files(
    args: argparse.Namespace,
    *,
    train_path: Path,
    eval_path: Path,
    run_mode: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    if not bool(getattr(args, "clean_data", False)):
        return train_path, eval_path, {"enabled": False}
    if not train_path.exists():
        return train_path, eval_path, {
            "enabled": True,
            "skipped": True,
            "run_mode": str(run_mode),
            "reason": f"train_missing({train_path.as_posix()})",
        }

    in_place = bool(getattr(args, "clean_inplace", False))
    out_dir_raw = str(getattr(args, "clean_output_dir", "") or "").strip()
    out_dir = Path(out_dir_raw) if out_dir_raw else (train_path.parent / "cleaned")
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    if not in_place:
        out_dir.mkdir(parents=True, exist_ok=True)

    clean_cfg = DataCleanConfig(
        min_prompt_chars=max(1, int(getattr(args, "clean_min_prompt_chars", 2) or 2)),
        min_response_chars=max(1, int(getattr(args, "clean_min_response_chars", 4) or 4)),
        min_line_chars=max(1, int(getattr(args, "clean_min_line_chars", 2) or 2)),
        max_repeat_run=max(1, int(getattr(args, "clean_max_repeat_run", 14) or 14)),
        max_symbol_ratio=max(0.0, min(1.0, float(getattr(args, "clean_max_symbol_ratio", 0.60) or 0.60))),
        dedup=(not bool(getattr(args, "clean_no_dedup", False))),
        use_ftfy=bool(getattr(args, "clean_use_ftfy", False)),
        allowed_languages=parse_allowed_languages(str(getattr(args, "clean_allow_langs", "") or "")),
    )

    train_out = _resolve_clean_output(train_path, args=args, in_place=in_place, out_dir=out_dir)
    train_report = clean_training_file(train_path, train_out, cfg=clean_cfg)
    effective_train_path = Path(str(train_report.get("output_path", train_out.as_posix())))

    eval_report: Dict[str, Any] = {"ok": False, "skipped": True, "reason": "eval_missing"}
    effective_eval_path = eval_path
    if eval_path.exists():
        eval_out = _resolve_clean_output(eval_path, args=args, in_place=in_place, out_dir=out_dir)
        eval_report = clean_training_file(eval_path, eval_out, cfg=clean_cfg)
        effective_eval_path = Path(str(eval_report.get("output_path", eval_out.as_posix())))

    report = {
        "enabled": True,
        "run_mode": str(run_mode),
        "train": train_report,
        "eval": eval_report,
    }
    report_path = _normalize_clean_report_path(str(getattr(args, "clean_report_path", "") or ""))
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = report_path.as_posix()
    except Exception:
        pass
    return effective_train_path, effective_eval_path, report


def _build_training_args(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    epochs: float,
    use_cuda: bool,
    dtype: torch.dtype,
    has_eval: bool,
) -> TrainingArguments:
    bf16 = dtype == torch.bfloat16 and use_cuda
    fp16 = dtype == torch.float16 and use_cuda
    common_kwargs: Dict[str, Any] = {
        "output_dir": output_dir.as_posix(),
        "overwrite_output_dir": True,
        "num_train_epochs": float(epochs),
        "learning_rate": float(args.learning_rate),
        "per_device_train_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.grad_accum),
        "per_device_eval_batch_size": max(1, int(args.batch_size)),
        "logging_steps": max(1, int(args.logging_steps)),
        "save_steps": max(1, int(args.save_steps)),
        "save_total_limit": 2,
        "warmup_ratio": float(args.warmup_ratio),
        "weight_decay": float(args.weight_decay),
        "eval_steps": (max(1, int(args.save_steps)) if has_eval else None),
        "bf16": bool(bf16),
        "fp16": bool(fp16),
        "report_to": [],
        "remove_unused_columns": False,
        "seed": int(args.seed),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "group_by_length": bool(args.group_by_length),
    }
    strategy_val = "steps" if has_eval else "no"
    try:
        sig = inspect.signature(TrainingArguments.__init__)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()
    if "evaluation_strategy" in params:
        common_kwargs["evaluation_strategy"] = strategy_val
    elif "eval_strategy" in params:
        common_kwargs["eval_strategy"] = strategy_val
    else:
        common_kwargs["evaluation_strategy"] = strategy_val
    if params:
        filtered = {k: v for k, v in common_kwargs.items() if k in params}
    else:
        filtered = dict(common_kwargs)
    # Drop None fields for older transformers versions that reject explicit None.
    filtered = {k: v for k, v in filtered.items() if v is not None}
    return TrainingArguments(**filtered)


def _run_stage(
    *,
    stage_name: str,
    model,
    args: argparse.Namespace,
    base_output_dir: Path,
    train_ds: Dataset,
    eval_ds: Optional[Dataset],
    collator: DialogueCollator,
    use_cuda: bool,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    stage_out = base_output_dir / stage_name
    stage_out.mkdir(parents=True, exist_ok=True)
    train_args = _build_training_args(
        args=args,
        output_dir=stage_out,
        epochs=float(args.epochs if stage_name == "single" else args._stage_epochs),  # type: ignore[attr-defined]
        use_cuda=use_cuda,
        dtype=dtype,
        has_eval=bool(eval_ds is not None),
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    result = trainer.train()
    eval_metrics = {}
    if eval_ds is not None and len(eval_ds) > 0:
        eval_metrics = trainer.evaluate()
    metrics = getattr(result, "metrics", {}) or {}
    return {
        "stage": stage_name,
        "train_samples": int(len(train_ds)),
        "epochs": float(train_args.num_train_epochs),
        "train_runtime": float(metrics.get("train_runtime", 0.0)),
        "train_loss": float(metrics.get("train_loss", 0.0)),
        "eval_metrics": eval_metrics,
    }


def _partition_federated_rows(
    rows: List[Dict[str, Any]],
    *,
    num_clients: int,
    method: str,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    n = max(1, int(num_clients))
    key = str(method or "iid").strip().lower()
    clients: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
    if not rows:
        return clients
    rng = random.Random(int(seed))
    if key == "bucket":
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            bucket = str(row.get("bucket", "")).strip().lower() or "smalltalk"
            groups.setdefault(bucket, []).append(dict(row))
        for bucket_rows in groups.values():
            rng.shuffle(bucket_rows)
            for i, row in enumerate(bucket_rows):
                clients[i % n].append(dict(row))
    else:
        ordered = [dict(x) for x in rows]
        rng.shuffle(ordered)
        for i, row in enumerate(ordered):
            clients[i % n].append(dict(row))
    return clients


def _load_adapter_state_dict(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    sf = adapter_dir / "adapter_model.safetensors"
    if sf.exists():
        try:
            from safetensors.torch import load_file  # type: ignore

            raw = load_file(sf.as_posix(), device="cpu")
            return {str(k): v.detach().cpu() for k, v in raw.items()}
        except Exception:
            pass
    bn = adapter_dir / "adapter_model.bin"
    if bn.exists():
        raw = torch.load(bn.as_posix(), map_location="cpu")
        if isinstance(raw, dict):
            out: Dict[str, torch.Tensor] = {}
            for k, v in raw.items():
                if torch.is_tensor(v):
                    out[str(k)] = v.detach().cpu()
            if out:
                return out
    raise FileNotFoundError(f"adapter model weights not found under: {adapter_dir}")


def _state_signature(state: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(state.keys()):
        v = state.get(k)
        if v is None or (not torch.is_tensor(v)):
            continue
        t = v.detach().cpu().contiguous()
        h.update(str(k).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _adapter_signature(adapter_dir: Path) -> str:
    sd = _load_adapter_state_dict(adapter_dir)
    return _state_signature(sd)


def _save_adapter_state_dict(
    state: Dict[str, torch.Tensor],
    *,
    template_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in template_dir.iterdir():
        if not child.is_file():
            continue
        name = child.name
        if name in {"adapter_model.safetensors", "adapter_model.bin"}:
            continue
        shutil.copy2(child.as_posix(), (out_dir / name).as_posix())
    torch.save({k: v.detach().cpu() for k, v in state.items()}, (out_dir / "adapter_model.bin").as_posix())


def _build_state_layout(
    base_state: Dict[str, torch.Tensor],
    ref_state: Dict[str, torch.Tensor],
) -> List[Tuple[str, Tuple[int, ...], torch.dtype, int]]:
    key_ref: Dict[str, torch.Tensor] = {}
    for src in (base_state, ref_state):
        if not isinstance(src, dict):
            continue
        for k, v in src.items():
            if not torch.is_tensor(v):
                continue
            key = str(k)
            if key not in key_ref:
                key_ref[key] = v.detach().cpu()
    layout: List[Tuple[str, Tuple[int, ...], torch.dtype, int]] = []
    for key in sorted(key_ref.keys()):
        ref = key_ref[key]
        shape = tuple(int(x) for x in ref.shape)
        numel = int(ref.numel())
        if numel <= 0:
            continue
        layout.append((str(key), shape, ref.dtype, numel))
    return layout


def _layout_total_dim(layout: List[Tuple[str, Tuple[int, ...], torch.dtype, int]]) -> int:
    return int(sum(int(x[3]) for x in layout))


def _align_state_to_layout(
    state: Dict[str, torch.Tensor],
    *,
    layout: List[Tuple[str, Tuple[int, ...], torch.dtype, int]],
    fallback: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    src = state if isinstance(state, dict) else {}
    fb = fallback if isinstance(fallback, dict) else {}
    out: Dict[str, torch.Tensor] = {}
    for key, shape, dtype, _ in layout:
        cur = src.get(str(key))
        if cur is None or (not torch.is_tensor(cur)) or tuple(int(x) for x in cur.shape) != tuple(shape):
            cur = fb.get(str(key))
        if cur is None or (not torch.is_tensor(cur)) or tuple(int(x) for x in cur.shape) != tuple(shape):
            cur = torch.zeros(shape, dtype=torch.float32)
        out[str(key)] = cur.detach().cpu().to(dtype=dtype).contiguous()
    return out


def _state_diff_vector(
    *,
    base_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    layout: List[Tuple[str, Tuple[int, ...], torch.dtype, int]],
) -> torch.Tensor:
    aligned_base = _align_state_to_layout(base_state, layout=layout, fallback=None)
    aligned_target = _align_state_to_layout(target_state, layout=layout, fallback=aligned_base)
    parts: List[torch.Tensor] = []
    for key, _, _, _ in layout:
        delta = aligned_target[str(key)].to(dtype=torch.float32) - aligned_base[str(key)].to(dtype=torch.float32)
        parts.append(delta.reshape(-1).contiguous())
    if not parts:
        return torch.zeros((0,), dtype=torch.float32)
    return torch.cat(parts, dim=0).contiguous()


def _apply_delta_vector_to_state(
    *,
    base_state: Dict[str, torch.Tensor],
    delta_vector: torch.Tensor,
    layout: List[Tuple[str, Tuple[int, ...], torch.dtype, int]],
) -> Dict[str, torch.Tensor]:
    if not torch.is_tensor(delta_vector):
        raise ValueError("delta_vector must be torch.Tensor")
    vec = delta_vector.detach().cpu().to(dtype=torch.float32).reshape(-1).contiguous()
    total_dim = int(_layout_total_dim(layout))
    if int(vec.numel()) != int(total_dim):
        raise ValueError(f"delta_vector_dim_mismatch({int(vec.numel())}!={int(total_dim)})")
    aligned_base = _align_state_to_layout(base_state, layout=layout, fallback=None)
    out: Dict[str, torch.Tensor] = {}
    offset = 0
    for key, shape, dtype, numel in layout:
        seg = vec[offset : offset + int(numel)].reshape(shape)
        base = aligned_base[str(key)].to(dtype=torch.float32)
        out[str(key)] = (base + seg).to(dtype=dtype).detach().cpu().contiguous()
        offset += int(numel)
    return out


def _fedavg_state_dict(items: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
    return _aggregate_state_dict(items, method="avg", trim_ratio=0.1)


def _normalize_fed_aggregation(method: str) -> str:
    key = str(method or "").strip().lower()
    if key in {"avg", "mean", "fedavg"}:
        return "avg"
    if key in {"median", "coord_median", "coordinate_median"}:
        return "median"
    if key in {"trimmed_mean", "trimmed", "tmean"}:
        return "trimmed_mean"
    return "avg"


def _aggregate_state_dict(
    items: List[Tuple[Dict[str, torch.Tensor], float]],
    *,
    method: str,
    trim_ratio: float,
) -> Dict[str, torch.Tensor]:
    if not items:
        return {}
    keys = list(items[0][0].keys())
    mode = _normalize_fed_aggregation(method)
    t_ratio = min(0.45, max(0.0, float(trim_ratio)))
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        ref = items[0][0].get(k)
        if ref is None or (not torch.is_tensor(ref)):
            continue
        if not ref.is_floating_point():
            out[k] = ref.detach().cpu().clone()
            continue

        values: List[torch.Tensor] = []
        weights: List[float] = []
        for sd, w in items:
            cur = sd.get(k)
            if cur is None or (not torch.is_tensor(cur)):
                continue
            values.append(cur.detach().cpu().to(dtype=torch.float32))
            weights.append(float(max(1e-9, w)))
        if not values:
            continue

        if mode == "avg" or len(values) == 1:
            total = float(sum(weights))
            acc = torch.zeros_like(values[0], dtype=torch.float32, device="cpu")
            for val, wt in zip(values, weights):
                acc = acc + val * float(wt)
            merged = acc / float(max(1e-9, total))
        else:
            stack = torch.stack(values, dim=0)
            if mode == "median":
                merged = torch.median(stack, dim=0).values
            else:
                n = int(stack.shape[0])
                cut = int(n * t_ratio)
                cut = min(max(0, cut), max(0, (n - 1) // 2))
                sorted_vals = torch.sort(stack, dim=0).values
                kept = sorted_vals[cut : (n - cut)] if (n - 2 * cut) > 0 else sorted_vals
                merged = torch.mean(kept, dim=0)
        out[k] = merged.to(dtype=ref.dtype).detach().cpu()
    return out


def _aggregate_client_adapters(
    adapters: List[Tuple[Path, float]],
    *,
    out_dir: Path,
    method: str = "avg",
    trim_ratio: float = 0.1,
) -> Dict[str, Any]:
    valid: List[Tuple[Path, float, Dict[str, torch.Tensor], str]] = []
    for path, weight in adapters:
        p = Path(path)
        if not p.exists():
            continue
        try:
            sd = _load_adapter_state_dict(p)
            sig = _state_signature(sd)
        except Exception:
            continue
        valid.append((p, float(max(1e-6, weight)), sd, sig))
    if not valid:
        raise RuntimeError("no valid client adapters to aggregate")
    agg_items = [(sd, max(1e-6, w)) for _, w, sd, _ in valid]
    agg = _aggregate_state_dict(
        agg_items,
        method=str(method),
        trim_ratio=float(trim_ratio),
    )
    _save_adapter_state_dict(agg, template_dir=valid[0][0], out_dir=out_dir)
    total_weight = sum(max(1e-6, w) for _, w, _, _ in valid)
    client_signatures = {p.as_posix(): str(sig) for p, _, _, sig in valid}
    agg_signature = _state_signature(agg)
    return {
        "adapter_path": out_dir.as_posix(),
        "client_count": int(len(valid)),
        "total_weight": float(total_weight),
        "keys": int(len(agg)),
        "signature": str(agg_signature),
        "method": _normalize_fed_aggregation(method),
        "trim_ratio": float(min(0.45, max(0.0, float(trim_ratio)))),
        "client_signatures": client_signatures,
    }


def _normalize_client_weight_strategy(name: str) -> str:
    key = str(name or "").strip().lower()
    if key in {"samples", "sample", "rows"}:
        return "samples"
    if key in {"uniform", "equal"}:
        return "uniform"
    if key in {"inverse_loss", "inv_loss", "loss_inv"}:
        return "inverse_loss"
    if key in {"hybrid", "mixed"}:
        return "hybrid"
    if key in {"multi_signal", "multi", "quality_blend"}:
        return "multi_signal"
    return "samples"


def _extract_eval_loss_from_summary(summary: Dict[str, Any]) -> Optional[float]:
    if not isinstance(summary, dict):
        return None
    eval_node = summary.get("eval_metrics", {})
    if isinstance(eval_node, dict):
        try:
            val = float(eval_node.get("eval_loss", 0.0))
        except Exception:
            val = 0.0
        if val > 0.0:
            return float(val)
    stages = summary.get("stage_reports", [])
    if isinstance(stages, list) and stages:
        last = stages[-1]
        if isinstance(last, dict):
            ev = last.get("eval_metrics", {})
            if isinstance(ev, dict):
                try:
                    val2 = float(ev.get("eval_loss", 0.0))
                except Exception:
                    val2 = 0.0
                if val2 > 0.0:
                    return float(val2)
    return None


def _normalize_client_weight_map(
    weights: Dict[int, float],
    *,
    min_weight: float,
    max_ratio: float,
    normalize_mean: bool,
) -> Dict[int, float]:
    src = {int(k): float(v) for k, v in weights.items()}
    if not src:
        return {}
    floor = max(1e-9, float(min_weight))
    out: Dict[int, float] = {int(k): max(floor, float(v)) for k, v in src.items()}
    ratio = max(0.0, float(max_ratio))
    if ratio > 1.0:
        mn = min(float(v) for v in out.values())
        cap = float(mn * ratio)
        for k in list(out.keys()):
            out[k] = min(float(out[k]), float(cap))
            out[k] = max(float(out[k]), floor)
    if bool(normalize_mean):
        n = max(1, len(out))
        avg = float(sum(float(v) for v in out.values()) / float(n))
        if avg > 1e-12:
            scale = float(1.0 / avg)
            for k in list(out.keys()):
                out[k] = max(floor, float(out[k]) * scale)
    return out


def _cap_client_weight_share(
    weights: Dict[int, float],
    *,
    max_share: float,
    min_weight: float,
) -> Dict[int, float]:
    src = {int(k): max(float(min_weight), float(v)) for k, v in weights.items()}
    if not src:
        return {}
    cap = float(max_share)
    if cap <= 0.0 or cap >= 1.0:
        return src
    if len(src) <= 1:
        return src
    cap = min(cap, 1.0 - 1e-9)
    floor = max(1e-12, float(min_weight))
    out = dict(src)

    # Iteratively clip heavy clients and redistribute excess to lighter ones.
    for _ in range(len(out) * 4):
        total = float(sum(out.values()))
        if total <= 1e-12:
            break
        limit = float(total * cap)
        heavy = [k for k, v in out.items() if float(v) > float(limit) + 1e-12]
        if not heavy:
            break
        excess = 0.0
        for k in heavy:
            excess += float(out[k] - limit)
            out[k] = float(limit)
        light = [k for k in out.keys() if k not in set(heavy)]
        if not light:
            break
        light_total = float(sum(max(floor, out[k]) for k in light))
        if light_total <= 1e-12:
            add = float(excess / float(max(1, len(light))))
            for k in light:
                out[k] = float(out[k] + add)
            continue
        for k in light:
            frac = float(max(floor, out[k]) / light_total)
            out[k] = float(out[k] + excess * frac)
    return {int(k): max(floor, float(v)) for k, v in out.items()}


def _client_weight_stats(weights: Dict[int, float]) -> Dict[str, float]:
    src = {int(k): max(0.0, float(v)) for k, v in weights.items()}
    if not src:
        return {
            "client_count": 0.0,
            "total_weight": 0.0,
            "max_share": 0.0,
            "min_share": 0.0,
            "herfindahl": 0.0,
            "entropy_norm": 1.0,
        }
    total = float(sum(src.values()))
    if total <= 1e-12:
        n = float(len(src))
        return {
            "client_count": float(len(src)),
            "total_weight": 0.0,
            "max_share": float(1.0 / n) if n > 0 else 0.0,
            "min_share": float(1.0 / n) if n > 0 else 0.0,
            "herfindahl": float(1.0 / n) if n > 0 else 0.0,
            "entropy_norm": 1.0,
        }
    shares = [float(v / total) for v in src.values()]
    mx = float(max(shares))
    mn = float(min(shares))
    hhi = float(sum(s * s for s in shares))
    if len(shares) <= 1:
        ent_norm = 1.0
    else:
        ent = -float(sum(s * math.log(max(1e-12, s)) for s in shares))
        ent_norm = float(ent / math.log(float(len(shares))))
    return {
        "client_count": float(len(src)),
        "total_weight": float(total),
        "max_share": float(mx),
        "min_share": float(mn),
        "herfindahl": float(hhi),
        "entropy_norm": float(max(0.0, min(1.0, ent_norm))),
    }


def _apply_bucket_balance_to_client_weights(
    *,
    weights: Dict[int, float],
    client_bucket_dists: Dict[int, Dict[str, float]],
    target_dist: Dict[str, float],
    strength: float,
    eps: float,
) -> Dict[str, Any]:
    src = {int(k): max(1e-9, float(v)) for k, v in weights.items()}
    if not src:
        return {
            "weights": {},
            "target_bucket_dist": {},
            "raw_bucket_dist": {},
            "bucket_correction": {},
            "client_bucket_factor": {},
        }
    s = max(0.0, float(strength))
    e = max(1e-9, float(eps))
    tgt = {str(k): max(0.0, float(v)) for k, v in dict(target_dist or {}).items()}
    t_sum = float(sum(tgt.values()))
    if t_sum > 1e-12:
        tgt = {k: float(v / t_sum) for k, v in tgt.items()}
    else:
        tgt = {}
    if s <= 1e-9 or not tgt:
        return {
            "weights": dict(src),
            "target_bucket_dist": tgt,
            "raw_bucket_dist": {},
            "bucket_correction": {},
            "client_bucket_factor": {str(k): 1.0 for k in src.keys()},
        }

    keys = set(tgt.keys())
    for node in client_bucket_dists.values():
        if not isinstance(node, dict):
            continue
        keys.update(str(k) for k in node.keys())
    if not keys:
        return {
            "weights": dict(src),
            "target_bucket_dist": tgt,
            "raw_bucket_dist": {},
            "bucket_correction": {},
            "client_bucket_factor": {str(k): 1.0 for k in src.keys()},
        }

    raw_mass: Dict[str, float] = {str(k): 0.0 for k in keys}
    for cid, w in src.items():
        dist = client_bucket_dists.get(int(cid), {})
        if not isinstance(dist, dict):
            continue
        for b in keys:
            raw_mass[str(b)] = float(raw_mass.get(str(b), 0.0) + float(w) * float(max(0.0, float(dist.get(str(b), 0.0)))))
    raw_total = float(sum(raw_mass.values()))
    raw_dist = (
        {str(k): float(v / raw_total) for k, v in raw_mass.items()}
        if raw_total > 1e-12
        else {str(k): 0.0 for k in keys}
    )
    corr = {
        str(k): float((float(tgt.get(str(k), 0.0)) + e) / (float(raw_dist.get(str(k), 0.0)) + e)) for k in keys
    }
    factors: Dict[str, float] = {}
    adj: Dict[int, float] = {}
    for cid, w in src.items():
        dist = client_bucket_dists.get(int(cid), {})
        if not isinstance(dist, dict) or not dist:
            fac = 1.0
        else:
            fac = 0.0
            for b in keys:
                fac += float(max(0.0, float(dist.get(str(b), 0.0)))) * float(corr.get(str(b), 1.0))
            fac = max(e, float(fac))
        factors[str(int(cid))] = float(fac)
        adj[int(cid)] = float(max(e, float(w) * (float(fac) ** float(s))))
    return {
        "weights": adj,
        "target_bucket_dist": {str(k): float(v) for k, v in tgt.items()},
        "raw_bucket_dist": {str(k): float(v) for k, v in raw_dist.items()},
        "bucket_correction": {str(k): float(v) for k, v in corr.items()},
        "client_bucket_factor": factors,
    }


def _client_aggregation_weight(
    *,
    item: Dict[str, Any],
    rows_count: int,
    strategy: str,
    loss_eps: float,
    loss_clip: float,
    sample_power: float = 1.0,
    loss_power: float = 1.0,
    eval_loss_power: float = 0.0,
    timeout_penalty: float = 0.0,
    retry_penalty: float = 0.0,
    fail_attempt_penalty: float = 0.0,
) -> float:
    mode = _normalize_client_weight_strategy(strategy)
    rows_w = float(max(1, int(rows_count)))
    uniform_w = 1.0
    loss_eps_v = max(1e-9, float(loss_eps))
    loss_clip_v = max(1.0, float(loss_clip))
    sample_pow = max(0.0, float(sample_power))
    loss_pow = max(0.0, float(loss_power))
    eval_pow = max(0.0, float(eval_loss_power))
    inv_loss_w = 1.0
    summary = item.get("summary", {})
    if isinstance(summary, dict):
        try:
            train_loss = float(summary.get("train_loss", 0.0))
        except Exception:
            train_loss = 0.0
        if train_loss > 0.0:
            inv_loss_w = min(loss_clip_v, max(1e-6, 1.0 / (train_loss + loss_eps_v)))
    inv_eval_w = 1.0
    eval_loss = _extract_eval_loss_from_summary(summary if isinstance(summary, dict) else {})
    if eval_loss is not None and float(eval_loss) > 0.0:
        inv_eval_w = min(loss_clip_v, max(1e-6, 1.0 / (float(eval_loss) + loss_eps_v)))

    sample_term = float(max(1e-6, rows_w**sample_pow))
    train_term = float(max(1e-6, inv_loss_w**loss_pow))
    eval_term = float(max(1e-6, inv_eval_w**eval_pow))

    if mode == "inverse_loss":
        base = float(train_term)
    elif mode == "hybrid":
        # Geometric blend between sample-count and inverse-loss weight.
        base = float(max(1e-6, (sample_term * train_term) ** 0.5))
    elif mode == "multi_signal":
        terms = [sample_term, train_term]
        if eval_pow > 0.0:
            terms.append(eval_term)
        prod = 1.0
        for x in terms:
            prod *= float(max(1e-6, x))
        base = float(max(1e-6, prod ** (1.0 / float(max(1, len(terms))))))
    elif mode == "uniform":
        base = float(uniform_w)
    else:
        base = float(sample_term)

    t_pen = _clamp01(float(timeout_penalty))
    r_pen = _clamp01(float(retry_penalty))
    f_pen = _clamp01(float(fail_attempt_penalty))
    attempts = item.get("attempts", [])
    has_timeout = False
    fail_attempts = 0
    if isinstance(attempts, list):
        for a in attempts:
            if not isinstance(a, dict):
                continue
            if bool(a.get("timeout", False)):
                has_timeout = True
            if not bool(a.get("ok", False)):
                fail_attempts += 1
    else:
        step = item.get("step", {})
        if isinstance(step, dict) and bool(step.get("timeout", False)):
            has_timeout = True
        if not bool(item.get("ok", False)):
            fail_attempts = 1

    retries_used = 0
    try:
        retries_used = max(0, int(item.get("retries_used", 0)))
    except Exception:
        retries_used = 0

    mult = 1.0
    if has_timeout and t_pen > 0.0:
        mult *= float(1.0 - t_pen)
    if retries_used > 0 and r_pen > 0.0:
        mult *= float((1.0 - r_pen) ** retries_used)
    if fail_attempts > 0 and f_pen > 0.0:
        mult *= float((1.0 - f_pen) ** fail_attempts)
    return float(max(1e-6, base * mult))


def _build_federated_child_cmd(
    args: argparse.Namespace,
    *,
    train_data: Path,
    output_dir: Path,
    seed: int,
    base_adapter: str,
    worker: Optional[Dict[str, Any]] = None,
) -> List[str]:
    w = dict(worker or {})
    w_python = str(w.get("python_bin", "")).strip() or sys.executable
    w_root = str(w.get("root_dir", "")).strip()
    root_dir = Path(w_root) if w_root else ROOT
    cmd: List[str] = [
        w_python,
        (root_dir / "scripts" / "train_adapter.py").as_posix(),
        "--model-path",
        str(args.model_path),
        "--train-data",
        train_data.as_posix(),
        "--eval-data",
        str(args.eval_data),
        "--dataset-manifest",
        str(args.dataset_manifest),
        "--output-dir",
        output_dir.as_posix(),
        "--epochs",
        str(args.fed_client_epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--max-length",
        str(args.max_length),
        "--logging-steps",
        str(args.logging_steps),
        "--save-steps",
        str(args.save_steps),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(int(seed)),
        "--dtype",
        str(args.dtype),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--target-modules",
        str(args.target_modules),
        "--bnb-compute-dtype",
        str(args.bnb_compute_dtype),
        "--curriculum-easy-frac",
        str(args.curriculum_easy_frac),
        "--curriculum-stage1-ratio",
        str(args.curriculum_stage1_ratio),
        "--replay-ratio",
        str(args.replay_ratio),
        "--replay-max-samples",
        str(args.replay_max_samples),
        "--artifact-registry-path",
        str(args.artifact_registry_path),
        "--no-register-artifact",
    ]
    if str(args.replay_data or "").strip():
        cmd.extend(["--replay-data", str(args.replay_data)])
    if str(base_adapter or "").strip():
        cmd.extend(["--base-adapter", str(base_adapter)])
    if bool(getattr(args, "allow_cross_split_prompt", False)):
        cmd.append("--allow-cross-split-prompt")
    if bool(getattr(args, "disable_diversity_penalty", False)):
        cmd.append("--disable-diversity-penalty")
    cmd.extend(
        [
            "--diversity-similarity-threshold",
            str(float(getattr(args, "diversity_similarity_threshold", 0.82))),
            "--diversity-penalty-strength",
            str(float(getattr(args, "diversity_penalty_strength", 0.20))),
            "--diversity-reference-cap",
            str(int(getattr(args, "diversity_reference_cap", 12))),
        ]
    )
    if bool(args.local_only):
        cmd.append("--local-only")
    if bool(args.load_in_4bit):
        cmd.append("--load-in-4bit")
    if bool(args.gradient_checkpointing):
        cmd.append("--gradient-checkpointing")
    if bool(args.group_by_length):
        cmd.append("--group-by-length")
    if bool(args.compact_output):
        cmd.append("--compact-output")
    if bool(args.curriculum):
        cmd.append("--curriculum")
    prefix = _norm_cmd_list(w.get("cmd_prefix", []))
    if prefix:
        return prefix + cmd
    return cmd


def _validate_federated_client(
    *,
    step: Dict[str, Any],
    summary: Dict[str, Any],
    adapter_path: str,
    expected_rows: int,
) -> Dict[str, Any]:
    errors: List[str] = []
    if int(step.get("returncode", -1)) != 0:
        errors.append("non_zero_returncode")
    if bool(step.get("timeout", False)):
        errors.append("timeout")
    ap = Path(str(adapter_path or ""))
    if not ap.exists():
        errors.append("adapter_missing")
    if not isinstance(summary, dict) or not summary:
        errors.append("summary_missing")
    else:
        used = int(summary.get("train_samples_used", 0))
        if used <= 0:
            errors.append("summary_train_samples_used_invalid")
    signature = ""
    if not errors:
        try:
            signature = _adapter_signature(ap)
        except Exception:
            errors.append("adapter_signature_failed")
    return {
        "ok": not bool(errors),
        "errors": errors,
        "expected_rows": int(max(0, expected_rows)),
        "adapter_signature": str(signature),
    }


def _run_federated_client(
    *,
    args: argparse.Namespace,
    rid: int,
    cid: int,
    rows: List[Dict[str, Any]],
    round_dir: Path,
    base_adapter: str,
    worker: Optional[Dict[str, Any]] = None,
    worker_sem: Optional[threading.Semaphore] = None,
    timeout_s: float = 0.0,
) -> Dict[str, Any]:
    shard_path = round_dir / f"client_{cid}_train.jsonl"
    _write_jsonl(shard_path, rows)
    client_out = round_dir / f"client_{cid}_out"
    cmd = _build_federated_child_cmd(
        args,
        train_data=shard_path,
        output_dir=client_out,
        seed=int(args.seed) + rid * 1000 + int(cid),
        base_adapter=base_adapter,
        worker=worker,
    )
    if worker_sem is not None:
        worker_sem.acquire()
    try:
        step = _run(cmd, cwd=ROOT, timeout_s=float(timeout_s))
    finally:
        if worker_sem is not None:
            worker_sem.release()
    summary = _read_json(client_out / "train_summary.json")
    adapter_path = str(summary.get("adapter_path", "")).strip() or (client_out / "adapter").as_posix()
    health = _validate_federated_client(
        step=step,
        summary=summary,
        adapter_path=adapter_path,
        expected_rows=len(rows),
    )
    return {
        "client_id": int(cid),
        "worker": str((worker or {}).get("name", "local")),
        "rows": int(len(rows)),
        "adapter_path": str(adapter_path),
        "ok": bool(health.get("ok", False)),
        "health": health,
        "summary": summary,
        "step": step,
    }


def _pick_retry_worker(
    *,
    current_worker: Dict[str, Any],
    workers: List[Dict[str, Any]],
    rid: int,
    cid: int,
    attempt: int,
) -> Dict[str, Any]:
    if not workers:
        return dict(current_worker)
    cur_name = str(current_worker.get("name", ""))
    alternatives = [w for w in workers if str(w.get("name", "")) != cur_name]
    if not alternatives:
        return dict(current_worker)
    idx = int((rid * 10007 + cid * 97 + attempt) % len(alternatives))
    return dict(alternatives[idx])


def _run_federated_client_with_retries(
    *,
    args: argparse.Namespace,
    rid: int,
    cid: int,
    rows: List[Dict[str, Any]],
    round_dir: Path,
    base_adapter: str,
    first_worker: Dict[str, Any],
    all_workers: List[Dict[str, Any]],
    worker_sems: Dict[str, threading.Semaphore],
    timeout_s: float,
) -> Dict[str, Any]:
    retries = max(0, int(args.fed_client_retries))
    backoff = max(0.0, float(args.fed_retry_backoff_seconds))
    switch_worker = bool(args.fed_retry_switch_worker)
    attempts: List[Dict[str, Any]] = []
    worker = dict(first_worker)
    final_item: Optional[Dict[str, Any]] = None

    for attempt in range(retries + 1):
        sem = worker_sems.get(str(worker.get("name", "")))
        item = _run_federated_client(
            args=args,
            rid=int(rid),
            cid=int(cid),
            rows=rows,
            round_dir=round_dir,
            base_adapter=base_adapter,
            worker=worker,
            worker_sem=sem,
            timeout_s=float(timeout_s),
        )
        attempts.append(item)
        final_item = item
        if bool(item.get("ok", False)):
            break
        if attempt >= retries:
            break
        if backoff > 0.0:
            time.sleep(float(backoff) * float(attempt + 1))
        if switch_worker:
            worker = _pick_retry_worker(
                current_worker=worker,
                workers=all_workers,
                rid=int(rid),
                cid=int(cid),
                attempt=int(attempt + 1),
            )

    out = dict(final_item or {})
    out["retries_used"] = int(max(0, len(attempts) - 1))
    out["attempts"] = [
        {
            "worker": str(x.get("worker", "")),
            "ok": bool(x.get("ok", False)),
            "errors": list((x.get("health") or {}).get("errors", []))
            if isinstance(x.get("health"), dict)
            else [],
            "returncode": int((x.get("step") or {}).get("returncode", -1))
            if isinstance(x.get("step"), dict)
            else -1,
            "timeout": bool((x.get("step") or {}).get("timeout", False))
            if isinstance(x.get("step"), dict)
            else False,
            "elapsed_s": float((x.get("step") or {}).get("elapsed_s", 0.0))
            if isinstance(x.get("step"), dict)
            else 0.0,
        }
        for x in attempts
    ]
    return out


def _safe_adapter_signature(path: str) -> str:
    p = Path(str(path or "").strip())
    if not p.exists():
        return ""
    try:
        return _adapter_signature(p)
    except Exception:
        return ""


def _cleanup_round_client_artifacts(
    round_dir: Path,
    *,
    client_steps: List[Dict[str, Any]],
    keep_failed: bool,
) -> Dict[str, Any]:
    removed_files = 0
    removed_dirs = 0
    kept_failed = 0
    for item in client_steps:
        cid = int(item.get("client_id", -1))
        if cid < 0:
            continue
        ok = bool(item.get("ok", False))
        if (not ok) and bool(keep_failed):
            kept_failed += 1
            continue
        shard = round_dir / f"client_{cid}_train.jsonl"
        out_dir = round_dir / f"client_{cid}_out"
        if shard.exists():
            try:
                shard.unlink()
                removed_files += 1
            except Exception:
                pass
        if out_dir.exists():
            try:
                shutil.rmtree(out_dir.as_posix(), ignore_errors=True)
                removed_dirs += 1
            except Exception:
                pass
    return {
        "removed_files": int(removed_files),
        "removed_dirs": int(removed_dirs),
        "kept_failed_clients": int(kept_failed),
    }


def _prune_old_federated_round_dirs(
    fed_dir: Path,
    *,
    keep_latest: int,
    current_round: int,
) -> Dict[str, Any]:
    keep_n = max(0, int(keep_latest))
    if keep_n <= 0:
        return {"removed_rounds": []}
    rows: List[Tuple[int, Path]] = []
    for p in fed_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name.strip().lower()
        if not name.startswith("round_"):
            continue
        try:
            rid = int(name.split("_")[-1])
        except Exception:
            continue
        rows.append((rid, p))
    if not rows:
        return {"removed_rounds": []}
    rows = sorted(rows, key=lambda x: x[0])
    protect = {int(x[0]) for x in rows[-keep_n:]}
    protect.add(int(current_round))
    removed: List[int] = []
    for rid, path in rows:
        if int(rid) in protect:
            continue
        try:
            shutil.rmtree(path.as_posix(), ignore_errors=True)
            removed.append(int(rid))
        except Exception:
            continue
    return {"removed_rounds": removed}


def _compact_federated_client_step_for_state(node: Dict[str, Any]) -> Dict[str, Any]:
    def _as_int(value: Any, default: int) -> int:
        try:
            if value is None:
                return int(default)
            return int(value)
        except Exception:
            return int(default)

    def _as_float(value: Any, default: float) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    row = node if isinstance(node, dict) else {}
    step = row.get("step", {}) if isinstance(row.get("step"), dict) else {}
    health = row.get("health", {}) if isinstance(row.get("health"), dict) else {}
    errors = health.get("errors", []) if isinstance(health.get("errors"), list) else []
    return {
        "client_id": _as_int(row.get("client_id", -1), -1),
        "worker": str(row.get("worker", "") or ""),
        "ok": bool(row.get("ok", False)),
        "rows": _as_int(row.get("rows", 0), 0),
        "retries_used": _as_int(row.get("retries_used", 0), 0),
        "timeout": bool(step.get("timeout", False)),
        "returncode": _as_int(step.get("returncode", -1), -1),
        "elapsed_s": _as_float(step.get("elapsed_s", 0.0), 0.0),
        "error_count": int(len(errors)),
        "errors": [str(x) for x in errors[:6]],
    }


def _compact_federated_round_report_for_state(report: Dict[str, Any], *, compact: bool) -> Dict[str, Any]:
    src = report if isinstance(report, dict) else {}
    if not compact:
        return dict(src)
    out: Dict[str, Any] = {
        "round": int(src.get("round", 0) or 0),
        "selected_clients": [int(x) for x in (src.get("selected_clients", []) or [])],
        "active_workers": [str(x) for x in (src.get("active_workers", []) or [])],
        "cooled_workers": [str(x) for x in (src.get("cooled_workers", []) or [])],
        "parallel_clients": int(src.get("parallel_clients", 0) or 0),
        "async_mode": bool(src.get("async_mode", False)),
        "round_timeout_seconds": float(src.get("round_timeout_seconds", 0.0) or 0.0),
        "retry_stats": dict(src.get("retry_stats", {}) if isinstance(src.get("retry_stats"), dict) else {}),
        "governance": dict(src.get("governance", {}) if isinstance(src.get("governance"), dict) else {}),
        "degraded": bool(src.get("degraded", False)),
        "degrade_reason": str(src.get("degrade_reason", "") or ""),
        "global_adapter": str(src.get("global_adapter", "") or ""),
        "global_signature": str(src.get("global_signature", "") or ""),
        "stragglers": [int(x) for x in (src.get("stragglers", []) or [])],
    }
    agg = src.get("aggregate", {}) if isinstance(src.get("aggregate"), dict) else {}
    out["aggregate"] = {
        "method": str(agg.get("method", "") or ""),
        "adapter_path": str(agg.get("adapter_path", "") or ""),
        "signature": str(agg.get("signature", "") or ""),
        "total_weight": float(agg.get("total_weight", 0.0) or 0.0),
        "num_clients": int(agg.get("num_clients", 0) or 0),
    }
    steps = src.get("client_steps", [])
    if isinstance(steps, list):
        out["client_steps"] = [
            _compact_federated_client_step_for_state(x)
            for x in steps
            if isinstance(x, dict)
        ]
    else:
        out["client_steps"] = []
    return out


def _prepare_federated_state_round_reports(
    round_reports: List[Dict[str, Any]],
    *,
    compact: bool,
    keep_latest: int,
) -> List[Dict[str, Any]]:
    rows = [x for x in round_reports if isinstance(x, dict)]
    if int(keep_latest) > 0:
        rows = rows[-int(keep_latest) :]
    return [_compact_federated_round_report_for_state(x, compact=bool(compact)) for x in rows]


def _run_federated_training_distributed_runtime(
    args: argparse.Namespace,
    *,
    train_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    base_train_rows: List[Dict[str, Any]],
    replay_rows_raw: List[Dict[str, Any]],
    replay_rows: List[Dict[str, Any]],
    manifest: Optional[Dict[str, Any]],
    data_cleaning: Optional[Dict[str, Any]],
    output_dir: Path,
    fed_dir: Path,
) -> Dict[str, Any]:
    client_rows = _partition_federated_rows(
        train_rows,
        num_clients=int(args.fed_num_clients),
        method=str(args.fed_split_method),
        seed=int(args.seed),
    )
    min_samples = max(1, int(args.fed_min_client_samples))
    eligible = [i for i, rows in enumerate(client_rows) if len(rows) >= min_samples]
    if not eligible:
        raise SystemExit("No eligible federated clients after data partition")

    rounds = max(1, int(args.fed_rounds))
    rng = random.Random(int(args.seed))
    min_active = max(1, int(args.fed_min_active_clients))
    frac = max(0.01, min(1.0, float(args.fed_client_frac)))
    max_fail = max(0, int(args.fed_max_client_failures))
    report_keep_max = max(0, int(getattr(args, "fed_summary_keep_rounds", 24) or 24))

    dist_workers = max(1, int(getattr(args, "fed_dist_workers", 0) or max(1, int(args.fed_parallel_clients))))
    dist_coded = max(0, int(getattr(args, "fed_dist_coded_redundancy", 0) or 0))
    dist_timeout_s = max(0.0, float(getattr(args, "fed_dist_timeout_seconds", 0.0) or 0.0))
    dist_fedbuff_size = max(0, int(getattr(args, "fed_dist_fedbuff_size", 0) or 0))
    dist_transport = str(getattr(args, "fed_dist_transport", "tcp") or "tcp").strip().lower()
    if dist_transport not in {"tcp", "zeromq"}:
        dist_transport = "tcp"
    if dist_transport == "zeromq":
        try:
            import zmq  # type: ignore  # noqa: F401
        except Exception:
            raise SystemExit("fed_dist_transport_zeromq_requires_pyzmq")
    dist_advertise_host = str(getattr(args, "fed_dist_advertise_host", "") or "").strip()
    dist_discovery_enabled = bool(getattr(args, "fed_dist_discovery_enabled", False))
    dist_discovery_port = max(1, int(getattr(args, "fed_dist_discovery_port", 9136) or 9136))
    dist_discovery_service = str(getattr(args, "fed_dist_discovery_service", "fed_runtime") or "fed_runtime").strip()
    dist_discovery_broadcast_ip = str(
        getattr(args, "fed_dist_discovery_broadcast_ip", "255.255.255.255") or "255.255.255.255"
    ).strip()
    dist_discovery_timeout_s = max(0.1, float(getattr(args, "fed_dist_discovery_timeout_s", 1.0) or 1.0))
    dist_discovery_retries = max(1, int(getattr(args, "fed_dist_discovery_retries", 5) or 5))
    dist_secure_mode = str(getattr(args, "fed_dist_secure_mode", "plain") or "plain").strip().lower()
    if dist_secure_mode not in {"plain", "masked", "paillier"}:
        dist_secure_mode = "plain"
    dist_auth_token = str(getattr(args, "fed_dist_auth_token", "") or "").strip()
    dist_resume_state = bool(getattr(args, "fed_dist_resume_runtime_state", False))
    dist_secure_secret = str(getattr(args, "fed_dist_secure_secret", "") or "").strip()
    dist_secure_mask_scale = max(0.0, float(getattr(args, "fed_dist_secure_mask_scale", 0.03) or 0.03))
    dist_dp_clip_norm = max(0.0, float(getattr(args, "fed_dist_dp_clip_norm", 0.0) or 0.0))
    dist_dp_noise_sigma = max(0.0, float(getattr(args, "fed_dist_dp_noise_sigma", 0.0) or 0.0))
    dist_state_dir_raw = str(getattr(args, "fed_dist_state_dir", "") or "").strip()
    dist_state_dir = Path(dist_state_dir_raw) if dist_state_dir_raw else (fed_dir / "distributed_runtime")
    if not dist_state_dir.is_absolute():
        dist_state_dir = (Path.cwd() / dist_state_dir).resolve()
    dist_state_dir.mkdir(parents=True, exist_ok=True)

    current_global_adapter = str(args.base_adapter or "").strip()
    if current_global_adapter and not Path(current_global_adapter).exists():
        current_global_adapter = ""
    current_global_signature = _safe_adapter_signature(current_global_adapter)
    best_round_train_loss: Optional[float] = None
    round_reports: List[Dict[str, Any]] = []

    for rid in range(1, rounds + 1):
        round_dir = fed_dir / f"round_{rid}"
        round_dir.mkdir(parents=True, exist_ok=True)
        runtime_state_dir = dist_state_dir / f"round_{rid}"
        runtime_state_dir.mkdir(parents=True, exist_ok=True)
        runtime_vector_dir = round_dir / "runtime_vectors"
        runtime_vector_dir.mkdir(parents=True, exist_ok=True)

        select_n = max(min_active, int(round(len(eligible) * frac)))
        select_n = min(len(eligible), max(1, select_n))
        selected = list(eligible)
        rng.shuffle(selected)
        selected = selected[:select_n]
        selected_rows: List[Dict[str, Any]] = []
        for cid in selected:
            selected_rows.extend(list(client_rows[int(cid)]))
        selected_map = {int(i): int(cid) for i, cid in enumerate(selected)}

        round_workers_all = _load_federated_workers(args)
        round_worker_probes = [
            _probe_federated_worker(w, timeout_s=float(args.fed_worker_probe_timeout_seconds))
            for w in round_workers_all
        ]
        round_healthy_names = {
            str(x.get("name", ""))
            for x in round_worker_probes
            if isinstance(x, dict) and bool(x.get("ok", False))
        }
        round_workers = [w for w in round_workers_all if str(w.get("name", "")) in round_healthy_names]
        if bool(args.fed_worker_require_healthy) and (not round_workers):
            raise SystemExit("no_healthy_federated_workers_distributed_runtime")
        if not round_workers:
            round_workers = list(round_workers_all)
        if not round_workers:
            local_fallback = _default_local_worker()
            local_fallback["max_concurrency"] = int(max(1, dist_workers))
            round_workers = [local_fallback]
        round_worker_sems = {
            str(w.get("name", "")): threading.Semaphore(max(1, int(w.get("max_concurrency", 1))))
            for w in round_workers
        }
        round_worker_pressure = _build_worker_pressure_map(
            round_workers,
            round_worker_probes,
            enabled=bool(args.fed_resource_aware),
            alpha=float(args.fed_resource_pressure_alpha),
        )
        round_worker_by_client = _assign_clients_to_workers(
            [int(x) for x in selected],
            round_workers,
            seed=int(args.seed) + int(rid) * 17,
            worker_pressures=round_worker_pressure,
        )

        round_layout: List[Tuple[str, Tuple[int, ...], torch.dtype, int]] = []
        round_base_state: Dict[str, torch.Tensor] = {}
        round_template_dir: Optional[Path] = None
        if current_global_adapter and Path(current_global_adapter).exists():
            try:
                round_base_state = _load_adapter_state_dict(Path(current_global_adapter))
                round_layout = _build_state_layout(round_base_state, round_base_state)
                round_template_dir = Path(current_global_adapter)
            except Exception:
                round_base_state = {}
                round_layout = []
                round_template_dir = None

        layout_lock = threading.Lock()
        report_lock = threading.Lock()
        task_reports: Dict[str, Dict[str, Any]] = {}

        def _task_fn(task: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal round_layout, round_base_state, round_template_dir
            runtime_cid = int(task.get("client_id", -1) or -1)
            actual_cid = int(selected_map.get(runtime_cid, -1))
            if actual_cid < 0 or actual_cid >= len(client_rows):
                return {"error": "invalid_client_map"}
            rows = list(client_rows[int(actual_cid)])
            first_worker = round_worker_by_client.get(int(actual_cid), round_workers[0])
            item = _run_federated_client_with_retries(
                args=args,
                rid=int(rid),
                cid=int(actual_cid),
                rows=rows,
                round_dir=round_dir,
                base_adapter=str(current_global_adapter),
                first_worker=first_worker,
                all_workers=round_workers,
                worker_sems=round_worker_sems,
                timeout_s=max(0.0, float(args.fed_client_timeout_seconds)),
            )
            task_id = str(task.get("task_id", "") or f"round_{rid}_client_{actual_cid}")
            with report_lock:
                task_reports[task_id] = {
                    "runtime_client_id": int(runtime_cid),
                    "client_id": int(actual_cid),
                    "item": dict(item),
                }
            if not bool(item.get("ok", False)):
                return {"error": "client_train_failed"}
            ap = Path(str(item.get("adapter_path", "") or "").strip())
            if not ap.exists():
                return {"error": "adapter_missing"}
            try:
                client_state = _load_adapter_state_dict(ap)
            except Exception as e:
                return {"error": f"adapter_state_load_failed({e})"}

            with layout_lock:
                if not round_base_state:
                    round_base_state = {
                        str(k): torch.zeros_like(v.detach().cpu())
                        for k, v in client_state.items()
                        if torch.is_tensor(v)
                    }
                    if round_template_dir is None:
                        round_template_dir = Path(ap)
                if not round_layout:
                    round_layout = _build_state_layout(round_base_state, client_state)
                local_layout = list(round_layout)
                local_base = {str(k): v.detach().cpu() for k, v in round_base_state.items()}

            if not local_layout:
                return {"error": "round_layout_empty"}
            delta_vec = _state_diff_vector(
                base_state=local_base,
                target_state=client_state,
                layout=local_layout,
            )
            vec_path = runtime_vector_dir / f"{task_id}.pt"
            torch.save(delta_vec.detach().cpu(), vec_path.as_posix())
            sample_count = int(item.get("rows", len(rows)) or len(rows))
            return {
                "vector_path": vec_path.as_posix(),
                "vector_path_preprocessed": True,
                "vector_path_delete": True,
                "sample_count": int(max(1, sample_count)),
            }

        coord_cfg = CoordinatorConfig(
            host="127.0.0.1",
            port=0,
            transport=str(dist_transport),
            advertise_host=str(dist_advertise_host),
            auth_token=str(dist_auth_token),
            resume_state=bool(dist_resume_state),
            vector_dim=int(_layout_total_dim(round_layout)) if round_layout else 0,
            rounds=1,
            clients_per_round=int(len(selected)),
            required_clients=int(max(1, min_active)),
            coded_redundancy=int(dist_coded),
            lease_seconds=max(20.0, float(getattr(args, "fed_dist_lease_seconds", 40.0) or 40.0)),
            heartbeat_timeout_s=max(30.0, float(getattr(args, "fed_dist_heartbeat_timeout_s", 90.0) or 90.0)),
            state_path=(runtime_state_dir / "state.json").as_posix(),
            events_path=(runtime_state_dir / "events.jsonl").as_posix(),
            fedbuff=FedBuffConfig(
                enabled=True,
                buffer_size=int(dist_fedbuff_size if dist_fedbuff_size > 0 else max(1, len(selected))),
                flush_timeout_s=max(1.0, float(getattr(args, "fed_dist_fedbuff_timeout_s", 20.0) or 20.0)),
                mix_alpha=1.0,
            ),
            privacy=PrivacyConfig(
                clip_norm=float(dist_dp_clip_norm),
                noise_sigma=float(dist_dp_noise_sigma),
                seed=int(args.seed) + int(rid) * 101,
            ),
            secure=SecureAggConfig(
                mode=str(dist_secure_mode),
                shared_secret=str(dist_secure_secret),
                mask_scale=float(dist_secure_mask_scale),
            ),
            discovery=DiscoveryConfig(
                enabled=bool(dist_discovery_enabled),
                service=str(dist_discovery_service),
                udp_port=int(dist_discovery_port),
                broadcast_ip=str(dist_discovery_broadcast_ip),
                request_timeout_s=float(dist_discovery_timeout_s),
                retries=int(dist_discovery_retries),
            ),
        )
        coordinator = FederatedCoordinator(coord_cfg)
        coordinator.start()

        worker_results: List[Dict[str, Any]] = []
        worker_lock = threading.Lock()
        worker_threads: List[threading.Thread] = []

        def _worker_entry(worker_idx: int) -> None:
            w = FederatedWorker(
                WorkerConfig(
                    coordinator_host="127.0.0.1",
                    coordinator_port=int(coordinator.cfg.port),
                    transport=str(dist_transport),
                    auth_token=str(dist_auth_token),
                    node_id=f"fed_dist_w{int(worker_idx)+1}_r{int(rid)}",
                    poll_interval_s=0.05,
                    heartbeat_interval_s=0.6,
                    connect_timeout_s=10.0,
                    max_runtime_s=0.0,
                    max_tasks=0,
                    simulate_latency_s=0.0,
                    vector_scale=0.0,
                    privacy=PrivacyConfig(clip_norm=0.0, noise_sigma=0.0, seed=int(args.seed) + int(worker_idx)),
                    secure=SecureAggConfig(mode="plain", shared_secret="", mask_scale=0.0),
                    discover=DiscoveryConfig(
                        enabled=False,
                        service=str(dist_discovery_service),
                        udp_port=int(dist_discovery_port),
                        broadcast_ip=str(dist_discovery_broadcast_ip),
                        request_timeout_s=float(dist_discovery_timeout_s),
                        retries=int(dist_discovery_retries),
                    ),
                ),
                task_fn=_task_fn,
            )
            out = w.run()
            with worker_lock:
                worker_results.append(dict(out))

        for wi in range(max(1, int(dist_workers))):
            t = threading.Thread(target=_worker_entry, args=(wi,), daemon=True)
            worker_threads.append(t)
            t.start()

        round_timeout = float(dist_timeout_s if dist_timeout_s > 0.0 else max(300.0, float(len(selected)) * 90.0))
        coord_summary = coordinator.run_until_complete(timeout_s=float(round_timeout), poll_s=0.2)
        for t in worker_threads:
            t.join(timeout=3.0)
        coordinator.stop()

        if not bool(coord_summary.get("done", False)):
            raise SystemExit(f"federated_distributed_round_{rid}_timeout")

        with layout_lock:
            final_layout = list(round_layout)
            final_base_state = {str(k): v.detach().cpu() for k, v in round_base_state.items()}
            final_template = Path(round_template_dir) if round_template_dir is not None else None
        if not final_layout:
            raise SystemExit(f"federated_distributed_round_{rid}_layout_empty")
        delta_vec = torch.tensor(coordinator.get_global_vector(), dtype=torch.float32)
        agg_state = _apply_delta_vector_to_state(
            base_state=final_base_state,
            delta_vector=delta_vec,
            layout=final_layout,
        )
        if final_template is None or (not final_template.exists()):
            # fallback: use any successful client adapter as template directory
            for node in task_reports.values():
                item = node.get("item", {}) if isinstance(node, dict) else {}
                if isinstance(item, dict) and bool(item.get("ok", False)):
                    cand = Path(str(item.get("adapter_path", "") or ""))
                    if cand.exists():
                        final_template = cand
                        break
        if final_template is None or (not final_template.exists()):
            raise SystemExit(f"federated_distributed_round_{rid}_template_missing")

        round_agg_dir = round_dir / "aggregate_adapter"
        _save_adapter_state_dict(
            agg_state,
            template_dir=final_template,
            out_dir=round_agg_dir,
        )
        current_global_adapter = round_agg_dir.as_posix()
        current_global_signature = _safe_adapter_signature(current_global_adapter)

        client_steps: List[Dict[str, Any]] = []
        for _, node in sorted(task_reports.items(), key=lambda x: str(x[0])):
            item = node.get("item", {}) if isinstance(node, dict) else {}
            if isinstance(item, dict):
                client_steps.append(item)
        ok_steps = [x for x in client_steps if bool(x.get("ok", False))]
        fail_count = int(len(client_steps) - len(ok_steps))
        if len(ok_steps) < int(min_active):
            raise SystemExit(f"federated_distributed_round_{rid}_insufficient_clients({len(ok_steps)}<{int(min_active)})")
        if fail_count > int(max_fail):
            raise SystemExit(f"federated_distributed_round_{rid}_too_many_failures({int(fail_count)}>{int(max_fail)})")

        governance = _evaluate_federated_governance(
            client_steps=client_steps,
            expected_clients=int(len(selected)),
            selected_rows=selected_rows,
            global_rows=train_rows,
            best_loss=best_round_train_loss,
            regress_tol=float(args.fed_regress_loss_tol),
            max_drift_l1=float(args.fed_max_drift_l1),
            max_fail_rate=float(args.fed_max_fail_rate),
        )
        degraded = False
        degrade_reason = ""
        if not bool(governance.get("passed", False)):
            reasons = governance.get("reasons", [])
            detail = ",".join(str(x) for x in reasons) if isinstance(reasons, list) else "governance_failed"
            if bool(args.fed_auto_degrade):
                degraded = True
                degrade_reason = detail
            else:
                raise SystemExit(f"federated_governance_failed({detail})")
        if governance.get("round_train_loss", None) is not None:
            try:
                loss_val = float(governance.get("round_train_loss"))
                if best_round_train_loss is None or loss_val < float(best_round_train_loss):
                    best_round_train_loss = float(loss_val)
            except Exception:
                pass

        round_report = {
            "round": int(rid),
            "selected_clients": [int(x) for x in selected],
            "client_steps": client_steps,
            "ok_clients": int(len(ok_steps)),
            "fail_clients": int(fail_count),
            "aggregate": {
                "adapter_path": str(current_global_adapter),
                "adapter_signature": str(current_global_signature),
                "delta_dim": int(delta_vec.numel()),
            },
            "distributed_runtime": {
                "workers": int(dist_workers),
                "coded_redundancy": int(dist_coded),
                "timeout_s": float(round_timeout),
                "transport": str(dist_transport),
                "advertise_host": str(dist_advertise_host),
                "state_dir": runtime_state_dir.as_posix(),
                "coordinator": dict(coord_summary),
                "worker_results": [dict(x) for x in worker_results],
                "worker_pool": [
                    {
                        "name": str(w.get("name", "")),
                        "max_concurrency": int(w.get("max_concurrency", 1) or 1),
                        "resource_weight": float(w.get("resource_weight", 1.0) or 1.0),
                    }
                    for w in round_workers
                ],
                "worker_probes": [dict(x) for x in round_worker_probes if isinstance(x, dict)],
                "worker_assignments": {
                    str(cid): str(w.get("name", ""))
                    for cid, w in round_worker_by_client.items()
                    if isinstance(w, dict)
                },
                "resource_aware": bool(args.fed_resource_aware),
                "auth_token_enabled": bool(dist_auth_token),
                "resume_state": bool(dist_resume_state),
                "secure_mode": str(dist_secure_mode),
                "dp_clip_norm": float(dist_dp_clip_norm),
                "dp_noise_sigma": float(dist_dp_noise_sigma),
                "discovery_enabled": bool(dist_discovery_enabled),
                "discovery_port": int(dist_discovery_port),
                "discovery_service": str(dist_discovery_service),
            },
            "governance": governance,
            "degraded": bool(degraded),
            "degrade_reason": str(degrade_reason),
        }
        round_reports.append(round_report)
        if int(report_keep_max) > 0 and len(round_reports) > int(report_keep_max):
            round_reports = round_reports[-int(report_keep_max) :]

    final_adapter = str(current_global_adapter or "").strip()
    if not final_adapter or not Path(final_adapter).exists():
        raise SystemExit("federated_distributed_training_failed_no_final_adapter")

    artifact_registration = {"ok": False, "skipped": True}
    if not bool(args.no_register_artifact):
        artifact_registration = _register_artifact_from_summary(
            registry_path=str(args.artifact_registry_path),
            adapter_path=str(final_adapter),
            base_model=str(_base_model_ref(args.model_path)),
            report_path=(output_dir / "train_summary.json").as_posix(),
            source="train_adapter.federated_distributed_runtime",
            metrics=None,
        )
    artifact_prune = _maybe_prune_artifacts(args, protect_paths=[str(final_adapter)])

    summary = {
        "ts": time.time(),
        "mode": "federated_distributed_runtime",
        "base_model": str(_base_model_ref(args.model_path)),
        "adapter_path": str(final_adapter),
        "adapter_signature": str(current_global_signature or _safe_adapter_signature(final_adapter)),
        "adapter_storage": _path_storage_stats(Path(final_adapter)),
        "train_samples_raw": len(base_train_rows),
        "replay_samples_raw": len(replay_rows_raw),
        "replay_samples_used": len(replay_rows),
        "train_samples_merged": len(train_rows),
        "eval_samples_raw": len(eval_rows),
        "federated": {
            "enabled": True,
            "distributed_runtime": True,
            "num_clients": int(args.fed_num_clients),
            "eligible_clients": int(len(eligible)),
            "rounds": int(rounds),
            "client_frac": float(frac),
            "split_method": str(args.fed_split_method),
            "min_active_clients": int(min_active),
            "max_client_failures": int(max_fail),
            "dist_workers": int(dist_workers),
            "dist_coded_redundancy": int(dist_coded),
            "dist_timeout_seconds": float(dist_timeout_s),
            "dist_transport": str(dist_transport),
            "dist_advertise_host": str(dist_advertise_host),
            "dist_secure_mode": str(dist_secure_mode),
            "dist_auth_token_enabled": bool(dist_auth_token),
            "dist_resume_runtime_state": bool(dist_resume_state),
            "dist_dp_clip_norm": float(dist_dp_clip_norm),
            "dist_dp_noise_sigma": float(dist_dp_noise_sigma),
            "dist_discovery_enabled": bool(dist_discovery_enabled),
            "dist_discovery_port": int(dist_discovery_port),
            "dist_discovery_service": str(dist_discovery_service),
            "dist_state_dir": dist_state_dir.as_posix(),
            "round_reports": round_reports,
        },
        "args": vars(args),
        "data_cleaning": data_cleaning or {"enabled": False},
        "dataset_manifest": manifest or {},
        "artifact_registration": artifact_registration,
        "artifact_prune": artifact_prune,
    }
    _write_json(output_dir / "train_summary.json", summary)
    return summary


def _run_federated_training(args: argparse.Namespace) -> Dict[str, Any]:
    train_path = Path(args.train_data)
    eval_path = Path(args.eval_data)
    manifest = None
    if bool(args.auto_build):
        manifest = _auto_build_if_needed(train_path, eval_path, args)
    train_path, eval_path, data_cleaning = _maybe_clean_training_files(
        args,
        train_path=train_path,
        eval_path=eval_path,
        run_mode="federated",
    )

    base_train_rows = _read_jsonl(train_path)
    if not base_train_rows:
        raise SystemExit(f"No train samples: {train_path}")
    eval_rows = _read_jsonl(eval_path) if eval_path.exists() else []
    replay_rows_raw: List[Dict[str, Any]] = []
    if str(args.replay_data or "").strip():
        replay_rows_raw = _read_jsonl(Path(args.replay_data))
    replay_rows = _select_replay_rows(
        replay_rows_raw,
        base_count=len(base_train_rows),
        replay_ratio=float(args.replay_ratio),
        replay_max_samples=int(args.replay_max_samples),
        seed=int(args.seed),
    )
    train_rows = _dedup_rows(list(base_train_rows) + list(replay_rows))
    if not train_rows:
        raise SystemExit("No valid training rows after replay merge")

    ts = int(time.time())
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/checkpoints/adapters") / f"fed_candidate_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fed_dir = output_dir / "federated"
    fed_dir.mkdir(parents=True, exist_ok=True)

    if bool(getattr(args, "fed_distributed_runtime", False)):
        return _run_federated_training_distributed_runtime(
            args,
            train_rows=train_rows,
            eval_rows=eval_rows,
            base_train_rows=base_train_rows,
            replay_rows_raw=replay_rows_raw,
            replay_rows=replay_rows,
            manifest=manifest,
            data_cleaning=data_cleaning,
            output_dir=output_dir,
            fed_dir=fed_dir,
        )

    state_path = Path(args.fed_state_path) if str(args.fed_state_path).strip() else (fed_dir / "fed_state.json")
    client_rows = _partition_federated_rows(
        train_rows,
        num_clients=int(args.fed_num_clients),
        method=str(args.fed_split_method),
        seed=int(args.seed),
    )
    min_samples = max(1, int(args.fed_min_client_samples))
    eligible = [i for i, rows in enumerate(client_rows) if len(rows) >= min_samples]
    if not eligible:
        raise SystemExit("No eligible federated clients after data partition")

    all_workers = _load_federated_workers(args)
    worker_probes = [
        _probe_federated_worker(w, timeout_s=float(args.fed_worker_probe_timeout_seconds)) for w in all_workers
    ]
    healthy_names = {str(x.get("name", "")) for x in worker_probes if bool(x.get("ok", False))}
    healthy_workers = [w for w in all_workers if str(w.get("name", "")) in healthy_names]
    if bool(args.fed_worker_require_healthy) and not healthy_workers:
        raise SystemExit("no_healthy_federated_workers")
    if not healthy_workers:
        healthy_workers = [_default_local_worker()]
    worker_sems = {
        str(w.get("name", "")): threading.Semaphore(max(1, int(w.get("max_concurrency", 1))))
        for w in healthy_workers
    }
    worker_pressure_map = _build_worker_pressure_map(
        healthy_workers,
        worker_probes,
        enabled=bool(args.fed_resource_aware),
        alpha=float(args.fed_resource_pressure_alpha),
    )
    worker_health_state = _sanitize_worker_health_state({}, healthy_workers)
    worker_perf_state = _sanitize_worker_perf_state({}, healthy_workers)

    start_round = 1
    round_reports: List[Dict[str, Any]] = []
    current_global_adapter = str(args.base_adapter or "").strip()
    current_global_signature = _safe_adapter_signature(current_global_adapter)
    best_round_train_loss: Optional[float] = None
    if bool(args.fed_resume) and state_path.exists():
        st = _read_json(state_path)
        start_round = max(1, int(st.get("next_round", 1)))
        current_global_adapter = str(st.get("current_global_adapter", current_global_adapter)).strip()
        current_global_signature = str(st.get("current_global_signature", current_global_signature)).strip()
        try:
            if st.get("best_round_train_loss", None) is not None:
                best_round_train_loss = float(st.get("best_round_train_loss"))
        except Exception:
            best_round_train_loss = None
        rr = st.get("round_reports", [])
        if isinstance(rr, list):
            round_reports = [x for x in rr if isinstance(x, dict)]
            if best_round_train_loss is None:
                losses: List[float] = []
                for x in round_reports:
                    g = x.get("governance", {})
                    if isinstance(g, dict) and g.get("round_train_loss", None) is not None:
                        try:
                            losses.append(float(g.get("round_train_loss")))
                        except Exception:
                            pass
                if losses:
                    best_round_train_loss = min(losses)
        worker_health_state = _sanitize_worker_health_state(st.get("worker_health_state", {}), healthy_workers)
        worker_perf_state = _sanitize_worker_perf_state(st.get("worker_perf_state", {}), healthy_workers)
        actual_sig = _safe_adapter_signature(current_global_adapter)
        mismatch = False
        mismatch_reasons: List[str] = []
        if current_global_adapter and not Path(current_global_adapter).exists():
            mismatch = True
            mismatch_reasons.append("adapter_missing")
        if current_global_signature and actual_sig and current_global_signature != actual_sig:
            mismatch = True
            mismatch_reasons.append("signature_mismatch")
        if current_global_adapter and (not current_global_signature) and actual_sig:
            current_global_signature = actual_sig
        if mismatch:
            reason = ",".join(mismatch_reasons) if mismatch_reasons else "resume_state_mismatch"
            if bool(args.fed_resume_strict):
                raise SystemExit(f"federated_resume_integrity_failed({reason})")
            start_round = 1
            round_reports = []
            current_global_adapter = str(args.base_adapter or "").strip()
            current_global_signature = _safe_adapter_signature(current_global_adapter)
            best_round_train_loss = None
            worker_health_state = _sanitize_worker_health_state({}, healthy_workers)
            worker_perf_state = _sanitize_worker_perf_state({}, healthy_workers)

    rounds = max(1, int(args.fed_rounds))
    rng = random.Random(int(args.seed))
    min_active = max(1, int(args.fed_min_active_clients))
    frac = max(0.01, min(1.0, float(args.fed_client_frac)))
    max_fail = max(0, int(args.fed_max_client_failures))
    agg_method = _normalize_fed_aggregation(str(args.fed_aggregation))
    agg_trim_ratio = min(0.45, max(0.0, float(args.fed_trim_ratio)))
    parallel_limit = max(1, int(args.fed_parallel_clients))
    worker_fail_threshold = max(1, int(args.fed_worker_fail_threshold))
    worker_cooldown_rounds = max(0, int(args.fed_worker_fail_cooldown_rounds))
    sched_use_perf = bool(args.fed_sched_use_perf)
    sched_perf_alpha = _clamp01(float(args.fed_sched_perf_alpha))
    if sched_perf_alpha <= 1e-6:
        sched_perf_alpha = 0.3
    sched_perf_fail_weight = max(0.0, float(args.fed_sched_perf_fail_weight))
    sched_perf_latency_weight = max(0.0, float(args.fed_sched_perf_latency_weight))
    sched_resource_weight = max(0.0, float(args.fed_sched_resource_weight))
    sched_perf_weight = max(0.0, float(args.fed_sched_perf_weight))
    client_weight_strategy = _normalize_client_weight_strategy(str(args.fed_client_weight_strategy))
    client_weight_loss_eps = max(1e-9, float(args.fed_client_weight_loss_eps))
    client_weight_loss_clip = max(1.0, float(args.fed_client_weight_loss_clip))
    client_weight_sample_power = max(0.0, float(args.fed_client_weight_sample_power))
    client_weight_loss_power = max(0.0, float(args.fed_client_weight_loss_power))
    client_weight_eval_loss_power = max(0.0, float(args.fed_client_weight_eval_loss_power))
    client_weight_timeout_penalty = _clamp01(float(args.fed_client_weight_timeout_penalty))
    client_weight_retry_penalty = _clamp01(float(args.fed_client_weight_retry_penalty))
    client_weight_fail_attempt_penalty = _clamp01(float(args.fed_client_weight_fail_attempt_penalty))
    client_weight_min = max(1e-9, float(args.fed_client_weight_min))
    client_weight_max_ratio = max(0.0, float(args.fed_client_weight_max_ratio))
    client_weight_normalize_mean = bool(args.fed_client_weight_normalize_mean)
    client_weight_max_share = _clamp01(float(args.fed_client_weight_max_share))
    client_bucket_balance = bool(args.fed_client_bucket_balance)
    client_bucket_balance_strength = max(0.0, float(args.fed_client_bucket_balance_strength))
    client_bucket_balance_eps = max(1e-9, float(args.fed_client_bucket_balance_eps))
    client_bucket_target = str(args.fed_client_bucket_target).strip().lower()
    if client_bucket_target not in {"selected", "global"}:
        client_bucket_target = "selected"
    state_keep_rounds = max(0, int(getattr(args, "fed_state_keep_rounds", 12) or 0))
    state_compact = True
    if bool(getattr(args, "fed_state_full", False)):
        state_compact = False
    if bool(getattr(args, "fed_state_compact", False)):
        state_compact = True
    summary_keep_rounds = max(0, int(getattr(args, "fed_summary_keep_rounds", 24) or 0))
    summary_compact = True
    if bool(getattr(args, "fed_summary_full", False)):
        summary_compact = False
    if bool(getattr(args, "fed_summary_compact", False)):
        summary_compact = True
    round_report_compact = not bool(getattr(args, "fed_round_report_full", False))
    round_report_jsonl = (
        Path(str(getattr(args, "fed_round_report_jsonl", "")).strip())
        if str(getattr(args, "fed_round_report_jsonl", "")).strip()
        else (fed_dir / "round_reports.jsonl")
    )
    if not round_report_jsonl.is_absolute():
        round_report_jsonl = (Path.cwd() / round_report_jsonl).resolve()
    report_keep_max = max(
        int(state_keep_rounds) if int(state_keep_rounds) > 0 else 0,
        int(summary_keep_rounds) if int(summary_keep_rounds) > 0 else 0,
    )
    try:
        round_report_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if int(start_round) <= 1:
            round_report_jsonl.write_text("", encoding="utf-8")
    except Exception:
        pass

    for rid in range(start_round, rounds + 1):
        round_dir = fed_dir / f"round_{rid}"
        round_dir.mkdir(parents=True, exist_ok=True)
        select_n = max(min_active, int(round(len(eligible) * frac)))
        select_n = min(len(eligible), max(1, select_n))
        selected = list(eligible)
        rng.shuffle(selected)
        selected = selected[:select_n]
        active_workers, cooled_workers = _active_workers_for_round(
            healthy_workers,
            worker_health_state,
            round_id=int(rid),
        )
        if not active_workers:
            if bool(args.fed_worker_cooldown_strict):
                raise SystemExit("no_active_workers_after_cooldown")
            active_workers = list(healthy_workers)
            cooled_workers = []
        round_probe_updates = 0
        round_resource_pressure_map: Dict[str, float] = {}
        if bool(args.fed_resource_aware) and bool(args.fed_reprobe_each_round):
            round_probes = [
                _probe_federated_worker(w, timeout_s=float(args.fed_worker_probe_timeout_seconds)) for w in active_workers
            ]
            probe_by_name = {str(x.get("name", "")): x for x in worker_probes if isinstance(x, dict)}
            for p in round_probes:
                if not isinstance(p, dict):
                    continue
                name = str(p.get("name", ""))
                if not name or not bool(p.get("ok", False)):
                    continue
                probe_by_name[name] = p
                round_probe_updates += 1
            worker_probes = list(probe_by_name.values())
        round_resource_pressure_map = _build_worker_pressure_map(
            active_workers,
            worker_probes,
            enabled=bool(args.fed_resource_aware),
            alpha=float(args.fed_resource_pressure_alpha),
        )
        worker_pressure_map = dict(round_resource_pressure_map)
        round_perf_penalty_map = _build_worker_perf_penalty_map(
            active_workers,
            worker_perf_state,
            enabled=bool(sched_use_perf),
            fail_weight=float(sched_perf_fail_weight),
            latency_weight=float(sched_perf_latency_weight),
        )
        round_worker_pressure_map = _merge_worker_pressure_maps(
            round_resource_pressure_map,
            round_perf_penalty_map,
            primary_weight=float(sched_resource_weight),
            secondary_weight=float(sched_perf_weight),
        )
        round_parallel_cap = sum(max(1, int(w.get("max_concurrency", 1))) for w in active_workers)
        parallel = min(int(parallel_limit), max(1, int(round_parallel_cap)))
        worker_by_client = _assign_clients_to_workers(
            [int(x) for x in selected],
            active_workers,
            seed=int(args.seed) + int(rid) * 17,
            worker_pressures=round_worker_pressure_map,
        )
        selected_rows: List[Dict[str, Any]] = []
        for cid in selected:
            selected_rows.extend(list(client_rows[int(cid)]))

        client_steps: List[Dict[str, Any]] = []
        good_adapter_rows: List[Tuple[int, Path, float]] = []
        good_adapters: List[Tuple[Path, float]] = []
        good_client_weights: Dict[int, float] = {}
        good_client_weights_raw: Dict[int, float] = {}
        good_client_bucket_dists: Dict[int, Dict[str, float]] = {}
        fail_count = 0
        async_mode = bool(args.fed_async)
        round_timeout = max(0.0, float(args.fed_round_timeout_seconds))
        client_timeout = max(0.0, float(args.fed_client_timeout_seconds))
        if async_mode and round_timeout > 0.0:
            client_timeout = round_timeout if client_timeout <= 0.0 else min(client_timeout, round_timeout)

        if parallel <= 1:
            for cid in selected:
                rows = list(client_rows[cid])
                worker = worker_by_client.get(int(cid), active_workers[0])
                item = _run_federated_client_with_retries(
                    args=args,
                    rid=int(rid),
                    cid=int(cid),
                    rows=rows,
                    round_dir=round_dir,
                    base_adapter=current_global_adapter,
                    first_worker=worker,
                    all_workers=active_workers,
                    worker_sems=worker_sems,
                    timeout_s=client_timeout,
                )
                if bool(item.get("ok", False)):
                    w = _client_aggregation_weight(
                        item=item,
                        rows_count=len(rows),
                        strategy=client_weight_strategy,
                        loss_eps=float(client_weight_loss_eps),
                        loss_clip=float(client_weight_loss_clip),
                        sample_power=float(client_weight_sample_power),
                        loss_power=float(client_weight_loss_power),
                        eval_loss_power=float(client_weight_eval_loss_power),
                        timeout_penalty=float(client_weight_timeout_penalty),
                        retry_penalty=float(client_weight_retry_penalty),
                        fail_attempt_penalty=float(client_weight_fail_attempt_penalty),
                    )
                    ap = Path(str(item.get("adapter_path", "")))
                    good_adapter_rows.append((int(cid), ap, float(w)))
                    good_client_weights_raw[int(cid)] = float(w)
                    good_client_bucket_dists[int(cid)] = _bucket_distribution(rows)
                else:
                    fail_count += 1
                client_steps.append(item)
        else:
            futures = []
            with ThreadPoolExecutor(max_workers=parallel) as ex:
                for cid in selected:
                    rows = list(client_rows[cid])
                    worker = worker_by_client.get(int(cid), active_workers[0])
                    futures.append(
                        ex.submit(
                            _run_federated_client_with_retries,
                            args=args,
                            rid=int(rid),
                            cid=int(cid),
                            rows=rows,
                            round_dir=round_dir,
                            base_adapter=current_global_adapter,
                            first_worker=worker,
                            all_workers=active_workers,
                            worker_sems=worker_sems,
                            timeout_s=client_timeout,
                        )
                    )
                fut_iter = None
                if async_mode and round_timeout > 0.0:
                    fut_iter = as_completed(futures, timeout=round_timeout)
                else:
                    fut_iter = as_completed(futures)
                try:
                    for fut in fut_iter:
                        item = fut.result()
                        cid = int(item.get("client_id", -1))
                        rows = list(client_rows[cid]) if 0 <= cid < len(client_rows) else []
                        if bool(item.get("ok", False)):
                            w = _client_aggregation_weight(
                                item=item,
                                rows_count=len(rows),
                                strategy=client_weight_strategy,
                                loss_eps=float(client_weight_loss_eps),
                                loss_clip=float(client_weight_loss_clip),
                                sample_power=float(client_weight_sample_power),
                                loss_power=float(client_weight_loss_power),
                                eval_loss_power=float(client_weight_eval_loss_power),
                                timeout_penalty=float(client_weight_timeout_penalty),
                                retry_penalty=float(client_weight_retry_penalty),
                                fail_attempt_penalty=float(client_weight_fail_attempt_penalty),
                            )
                            ap = Path(str(item.get("adapter_path", "")))
                            good_adapter_rows.append((int(cid), ap, float(w)))
                            good_client_weights_raw[int(cid)] = float(w)
                            good_client_bucket_dists[int(cid)] = _bucket_distribution(rows)
                        else:
                            fail_count += 1
                        client_steps.append(item)
                except FuturesTimeout:
                    pass
                if async_mode and round_timeout > 0.0:
                    for fut in futures:
                        if fut.done():
                            continue
                        fut.cancel()
            client_steps = sorted(client_steps, key=lambda x: int(x.get("client_id", 10**9)))
        processed_ids = {int(x.get("client_id", -1)) for x in client_steps}
        straggler_ids = [int(x) for x in selected if int(x) not in processed_ids]

        if len(good_adapter_rows) < min_active:
            raise SystemExit(f"federated_round_{rid}_insufficient_clients({len(good_adapter_rows)}<{min_active})")
        if fail_count > max_fail:
            raise SystemExit(f"federated_round_{rid}_too_many_failures({fail_count}>{max_fail})")
        bucket_balance_report: Dict[str, Any] = {
            "enabled": bool(client_bucket_balance),
            "target": str(client_bucket_target),
            "strength": float(client_bucket_balance_strength),
            "eps": float(client_bucket_balance_eps),
        }
        if bool(client_bucket_balance) and len(good_client_weights_raw) >= 2:
            target_rows = selected_rows if str(client_bucket_target) == "selected" else train_rows
            target_dist = _bucket_distribution(target_rows)
            balanced = _apply_bucket_balance_to_client_weights(
                weights=good_client_weights_raw,
                client_bucket_dists=good_client_bucket_dists,
                target_dist=target_dist,
                strength=float(client_bucket_balance_strength),
                eps=float(client_bucket_balance_eps),
            )
            good_client_weights_raw = {
                int(k): float(v) for k, v in dict(balanced.get("weights", {})).items()
            }
            bucket_balance_report.update(
                {
                    "target_bucket_dist": dict(balanced.get("target_bucket_dist", {})),
                    "raw_bucket_dist": dict(balanced.get("raw_bucket_dist", {})),
                    "bucket_correction": dict(balanced.get("bucket_correction", {})),
                    "client_bucket_factor": dict(balanced.get("client_bucket_factor", {})),
                }
            )
        good_client_weights = _normalize_client_weight_map(
            good_client_weights_raw,
            min_weight=float(client_weight_min),
            max_ratio=float(client_weight_max_ratio),
            normalize_mean=bool(client_weight_normalize_mean),
        )
        good_client_weights = _cap_client_weight_share(
            good_client_weights,
            max_share=float(client_weight_max_share),
            min_weight=float(client_weight_min),
        )
        client_weight_stats = _client_weight_stats(good_client_weights)
        good_adapters = []
        for cid, ap, raw_w in good_adapter_rows:
            w = float(good_client_weights.get(int(cid), max(float(client_weight_min), float(raw_w))))
            good_adapters.append((ap, w))

        governance = _evaluate_federated_governance(
            client_steps=client_steps,
            expected_clients=int(len(selected)),
            selected_rows=selected_rows,
            global_rows=train_rows,
            best_loss=best_round_train_loss,
            regress_tol=float(args.fed_regress_loss_tol),
            max_drift_l1=float(args.fed_max_drift_l1),
            max_fail_rate=float(args.fed_max_fail_rate),
        )

        degraded = False
        degrade_reason = ""
        agg: Dict[str, Any] = {}
        if bool(governance.get("passed", False)):
            global_dir = round_dir / "global_adapter"
            agg = _aggregate_client_adapters(
                good_adapters,
                out_dir=global_dir,
                method=agg_method,
                trim_ratio=agg_trim_ratio,
            )
            current_global_adapter = str(agg.get("adapter_path", "")).strip()
            current_global_signature = str(agg.get("signature", current_global_signature)).strip()
            if governance.get("round_train_loss", None) is not None:
                rloss = float(governance.get("round_train_loss"))
                if best_round_train_loss is None:
                    best_round_train_loss = rloss
                else:
                    best_round_train_loss = min(float(best_round_train_loss), rloss)
        elif bool(args.fed_auto_degrade):
            degraded = True
            reasons = governance.get("reasons", [])
            if isinstance(reasons, list) and reasons:
                degrade_reason = ";".join(str(x) for x in reasons)
            else:
                degrade_reason = "governance_failed"
        else:
            reasons = governance.get("reasons", [])
            detail = ";".join(str(x) for x in reasons) if isinstance(reasons, list) and reasons else "unknown"
            raise SystemExit(f"federated_governance_failed({detail})")

        retry_counts = [int(x.get("retries_used", 0)) for x in client_steps]
        total_retries = int(sum(retry_counts)) if retry_counts else 0
        clients_retried = int(sum(1 for x in retry_counts if int(x) > 0)) if retry_counts else 0
        max_retries_used = int(max(retry_counts)) if retry_counts else 0
        worker_perf_state = _update_worker_perf_after_round(
            worker_perf_state,
            client_steps=client_steps,
            workers=healthy_workers,
            alpha=float(sched_perf_alpha),
        )
        round_perf_penalty_map_next = _build_worker_perf_penalty_map(
            active_workers,
            worker_perf_state,
            enabled=bool(sched_use_perf),
            fail_weight=float(sched_perf_fail_weight),
            latency_weight=float(sched_perf_latency_weight),
        )
        worker_health_upd = _update_worker_health_after_round(
            worker_health_state,
            client_steps=client_steps,
            round_id=int(rid),
            fail_threshold=int(worker_fail_threshold),
            cooldown_rounds=int(worker_cooldown_rounds),
        )
        worker_health_state = _sanitize_worker_health_state(
            worker_health_upd.get("state", {}),
            healthy_workers,
        )
        cleanup_stats: Dict[str, Any] = {}
        if bool(args.fed_prune_client_artifacts):
            cleanup_stats = _cleanup_round_client_artifacts(
                round_dir,
                client_steps=client_steps,
                keep_failed=bool(args.fed_prune_keep_failed_clients),
            )
        prune_stats = _prune_old_federated_round_dirs(
            fed_dir,
            keep_latest=int(args.fed_prune_rounds_keep),
            current_round=int(rid),
        )
        round_report = {
            "round": int(rid),
            "selected_clients": [int(x) for x in selected],
            "active_workers": [str(x.get("name", "")) for x in active_workers],
            "cooled_workers": [str(x) for x in cooled_workers],
            "parallel_clients": int(parallel),
            "async_mode": bool(async_mode),
            "round_timeout_seconds": float(round_timeout),
            "resource_aware": {
                "enabled": bool(args.fed_resource_aware),
                "reprobe_each_round": bool(args.fed_reprobe_each_round),
                "pressure_alpha": float(args.fed_resource_pressure_alpha),
                "probe_updates": int(round_probe_updates),
                "worker_pressure_map": dict(round_resource_pressure_map),
            },
            "scheduler": {
                "use_perf": bool(sched_use_perf),
                "resource_weight": float(sched_resource_weight),
                "perf_weight": float(sched_perf_weight),
                "perf_alpha": float(sched_perf_alpha),
                "perf_fail_weight": float(sched_perf_fail_weight),
                "perf_latency_weight": float(sched_perf_latency_weight),
                "perf_penalty_map_before": dict(round_perf_penalty_map),
                "perf_penalty_map_after": dict(round_perf_penalty_map_next),
                "combined_pressure_map": dict(round_worker_pressure_map),
                "worker_perf_state": dict(worker_perf_state),
            },
            "retry_policy": {
                "client_retries": int(args.fed_client_retries),
                "retry_backoff_seconds": float(args.fed_retry_backoff_seconds),
                "retry_switch_worker": bool(args.fed_retry_switch_worker),
            },
            "retry_stats": {
                "clients_retried": int(clients_retried),
                "total_retries_used": int(total_retries),
                "max_retries_used": int(max_retries_used),
            },
            "worker_health": {
                "fail_threshold": int(worker_fail_threshold),
                "cooldown_rounds": int(worker_cooldown_rounds),
                "cooled_this_round": [str(x) for x in worker_health_upd.get("cooled_this_round", [])],
                "round_stats": dict(worker_health_upd.get("round_stats", {})),
                "state": dict(worker_health_state),
            },
            "worker_assignments": {str(k): str(v.get("name", "")) for k, v in worker_by_client.items()},
            "client_steps": client_steps,
            "client_weights_raw": {
                str(k): float(v) for k, v in sorted(good_client_weights_raw.items(), key=lambda x: int(x[0]))
            },
            "client_weights": {str(k): float(v) for k, v in sorted(good_client_weights.items(), key=lambda x: int(x[0]))},
            "stragglers": straggler_ids,
            "aggregate": agg,
            "aggregation_policy": {
                "strategy": str(client_weight_strategy),
                "loss_eps": float(client_weight_loss_eps),
                "loss_clip": float(client_weight_loss_clip),
                "sample_power": float(client_weight_sample_power),
                "loss_power": float(client_weight_loss_power),
                "eval_loss_power": float(client_weight_eval_loss_power),
                "timeout_penalty": float(client_weight_timeout_penalty),
                "retry_penalty": float(client_weight_retry_penalty),
                "fail_attempt_penalty": float(client_weight_fail_attempt_penalty),
                "min_weight": float(client_weight_min),
                "max_ratio": float(client_weight_max_ratio),
                "normalize_mean": bool(client_weight_normalize_mean),
                "max_share": float(client_weight_max_share),
                "stats": dict(client_weight_stats),
                "bucket_balance": bucket_balance_report,
            },
            "global_adapter": current_global_adapter,
            "global_signature": str(current_global_signature),
            "governance": governance,
            "degraded": bool(degraded),
            "degrade_reason": str(degrade_reason),
            "cleanup": {
                "prune_client_artifacts": bool(args.fed_prune_client_artifacts),
                "keep_failed_clients": bool(args.fed_prune_keep_failed_clients),
                **cleanup_stats,
            },
            "prune_rounds": {
                "keep_latest": int(args.fed_prune_rounds_keep),
                **prune_stats,
            },
        }
        round_reports.append(round_report)
        if int(report_keep_max) > 0 and len(round_reports) > int(report_keep_max):
            round_reports = round_reports[-int(report_keep_max) :]
        try:
            round_report_row = _compact_federated_round_report_for_state(
                round_report,
                compact=bool(round_report_compact),
            )
            _append_jsonl(round_report_jsonl, round_report_row)
        except Exception:
            pass
        state_round_reports = _prepare_federated_state_round_reports(
            round_reports,
            compact=bool(state_compact),
            keep_latest=int(state_keep_rounds),
        )
        _write_json(
            state_path,
            {
                "next_round": int(rid + 1),
                "current_global_adapter": current_global_adapter,
                "current_global_signature": current_global_signature,
                "best_round_train_loss": best_round_train_loss,
                "worker_health_state": worker_health_state,
                "worker_perf_state": worker_perf_state,
                "round_reports": state_round_reports,
                "state_compact": bool(state_compact),
                "state_keep_rounds": int(state_keep_rounds),
                "round_report_jsonl_path": round_report_jsonl.as_posix(),
                "round_report_compact": bool(round_report_compact),
                "ts": time.time(),
            },
        )

    final_adapter = current_global_adapter
    if not final_adapter or not Path(final_adapter).exists():
        raise SystemExit("federated_training_failed_no_final_adapter")

    artifact_registration = {"ok": False, "skipped": True}
    if not bool(args.no_register_artifact):
        artifact_registration = _register_artifact_from_summary(
            registry_path=str(args.artifact_registry_path),
            adapter_path=str(final_adapter),
            base_model=str(_base_model_ref(args.model_path)),
            report_path=(output_dir / "train_summary.json").as_posix(),
            source="train_adapter.federated",
            metrics=None,
        )
    artifact_prune = _maybe_prune_artifacts(args, protect_paths=[str(final_adapter)])
    summary_round_reports = _prepare_federated_state_round_reports(
        round_reports,
        compact=bool(summary_compact),
        keep_latest=int(summary_keep_rounds),
    )

    summary = {
        "ts": time.time(),
        "mode": "federated",
        "base_model": str(_base_model_ref(args.model_path)),
        "adapter_path": str(final_adapter),
        "adapter_signature": str(current_global_signature or _safe_adapter_signature(final_adapter)),
        "adapter_storage": _path_storage_stats(Path(final_adapter)),
        "train_samples_raw": len(base_train_rows),
        "replay_samples_raw": len(replay_rows_raw),
        "replay_samples_used": len(replay_rows),
        "train_samples_merged": len(train_rows),
        "eval_samples_raw": len(eval_rows),
        "federated": {
            "enabled": True,
            "num_clients": int(args.fed_num_clients),
            "eligible_clients": int(len(eligible)),
            "rounds": int(rounds),
            "client_frac": float(frac),
            "split_method": str(args.fed_split_method),
            "parallel_clients": int(parallel),
            "client_timeout_seconds": float(args.fed_client_timeout_seconds),
            "async_mode": bool(args.fed_async),
            "round_timeout_seconds": float(args.fed_round_timeout_seconds),
            "resource_aware": {
                "enabled": bool(args.fed_resource_aware),
                "reprobe_each_round": bool(args.fed_reprobe_each_round),
                "pressure_alpha": float(args.fed_resource_pressure_alpha),
                "worker_pressure_map": dict(worker_pressure_map),
            },
            "scheduler": {
                "use_perf": bool(sched_use_perf),
                "resource_weight": float(sched_resource_weight),
                "perf_weight": float(sched_perf_weight),
                "perf_alpha": float(sched_perf_alpha),
                "perf_fail_weight": float(sched_perf_fail_weight),
                "perf_latency_weight": float(sched_perf_latency_weight),
                "worker_perf_state": dict(worker_perf_state),
            },
            "aggregation": {
                "method": str(agg_method),
                "trim_ratio": float(agg_trim_ratio),
                "client_weight_strategy": str(client_weight_strategy),
                "client_weight_loss_eps": float(client_weight_loss_eps),
                "client_weight_loss_clip": float(client_weight_loss_clip),
                "client_weight_sample_power": float(client_weight_sample_power),
                "client_weight_loss_power": float(client_weight_loss_power),
                "client_weight_eval_loss_power": float(client_weight_eval_loss_power),
                "client_weight_timeout_penalty": float(client_weight_timeout_penalty),
                "client_weight_retry_penalty": float(client_weight_retry_penalty),
                "client_weight_fail_attempt_penalty": float(client_weight_fail_attempt_penalty),
                "client_weight_min": float(client_weight_min),
                "client_weight_max_ratio": float(client_weight_max_ratio),
                "client_weight_normalize_mean": bool(client_weight_normalize_mean),
                "client_weight_max_share": float(client_weight_max_share),
                "client_bucket_balance": bool(client_bucket_balance),
                "client_bucket_balance_strength": float(client_bucket_balance_strength),
                "client_bucket_balance_eps": float(client_bucket_balance_eps),
                "client_bucket_target": str(client_bucket_target),
            },
            "retry_policy": {
                "client_retries": int(args.fed_client_retries),
                "retry_backoff_seconds": float(args.fed_retry_backoff_seconds),
                "retry_switch_worker": bool(args.fed_retry_switch_worker),
            },
            "cleanup": {
                "prune_client_artifacts": bool(args.fed_prune_client_artifacts),
                "keep_failed_clients": bool(args.fed_prune_keep_failed_clients),
                "prune_rounds_keep": int(args.fed_prune_rounds_keep),
                "compact_output": bool(args.compact_output),
            },
            "worker_health": {
                "fail_threshold": int(worker_fail_threshold),
                "cooldown_rounds": int(worker_cooldown_rounds),
                "cooldown_strict": bool(args.fed_worker_cooldown_strict),
                "state": dict(worker_health_state),
            },
            "worker_count": int(len(healthy_workers)),
            "workers": [dict(x) for x in healthy_workers],
            "worker_probes": worker_probes,
            "governance": {
                "regress_loss_tol": float(args.fed_regress_loss_tol),
                "max_drift_l1": float(args.fed_max_drift_l1),
                "max_fail_rate": float(args.fed_max_fail_rate),
                "auto_degrade": bool(args.fed_auto_degrade),
                "best_round_train_loss": best_round_train_loss,
            },
            "resume": {
                "enabled": bool(args.fed_resume),
                "strict": bool(args.fed_resume_strict),
            },
            "state": {
                "compact": bool(state_compact),
                "keep_rounds": int(state_keep_rounds),
            },
            "summary": {
                "compact": bool(summary_compact),
                "keep_rounds": int(summary_keep_rounds),
            },
            "round_report_stream": {
                "path": round_report_jsonl.as_posix(),
                "compact": bool(round_report_compact),
            },
            "state_path": state_path.as_posix(),
            "round_reports": summary_round_reports,
        },
        "args": vars(args),
        "data_cleaning": data_cleaning or {"enabled": False},
        "dataset_manifest": manifest or {},
        "artifact_registration": artifact_registration,
        "artifact_prune": artifact_prune,
    }
    _write_json(output_dir / "train_summary.json", summary)
    return summary

def main() -> None:
    args = parse_args()
    model_ref, model_is_local = _resolve_model_ref(args.model_path)
    if not model_ref:
        raise SystemExit("Model path/model id is empty")
    if bool(args.local_only) and (not bool(model_is_local)):
        raise SystemExit(f"Model not found in local path/HF cache (local-only): {args.model_path}")

    if bool(args.federated):
        summary = _run_federated_training(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    preset_changes: Dict[str, Any] = {}
    if bool(args.preset_7840hs):
        preset_changes = _apply_7840hs_preset(args)

    train_path = Path(args.train_data)
    eval_path = Path(args.eval_data)
    manifest = None
    if args.auto_build:
        manifest = _auto_build_if_needed(train_path, eval_path, args)
        if manifest:
            print(f"dataset_built={json.dumps(manifest, ensure_ascii=False)}")
    train_path, eval_path, data_cleaning = _maybe_clean_training_files(
        args,
        train_path=train_path,
        eval_path=eval_path,
        run_mode="single",
    )

    base_train_rows = _read_jsonl(train_path)
    eval_rows = _read_jsonl(eval_path) if eval_path.exists() else []
    if not base_train_rows:
        raise SystemExit(f"No train samples: {train_path}")

    replay_rows_raw: List[Dict[str, Any]] = []
    if str(args.replay_data or "").strip():
        replay_rows_raw = _read_jsonl(Path(args.replay_data))
    replay_rows = _select_replay_rows(
        replay_rows_raw,
        base_count=len(base_train_rows),
        replay_ratio=float(args.replay_ratio),
        replay_max_samples=int(args.replay_max_samples),
        seed=int(args.seed),
    )
    train_rows = _dedup_rows(list(base_train_rows) + list(replay_rows))
    if not train_rows:
        raise SystemExit("No valid training rows after replay merge")

    ts = int(time.time())
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/checkpoints/adapters") / f"candidate_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_ref),
        local_files_only=bool(args.local_only),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    dtype = _resolve_dtype(args.dtype)
    if dtype is None and torch.cuda.is_available():
        dtype = torch.float16
    if dtype is None:
        dtype = torch.float32

    bnb_compute_dtype = _resolve_bnb_compute_dtype(args.bnb_compute_dtype, dtype)
    quant_cfg, quant_mode = _build_quantization_config(
        load_in_4bit=bool(args.load_in_4bit),
        compute_dtype=bnb_compute_dtype,
    )

    model_kwargs: Dict[str, Any] = {
        "local_files_only": bool(args.local_only),
        "trust_remote_code": True,
    }
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_ref), **model_kwargs)
    except Exception as e:
        if quant_cfg is None:
            raise
        print(f"[warn] 4bit load failed, fallback to normal dtype load: {e}")
        quant_mode = "fallback_full_precision"
        model = AutoModelForCausalLM.from_pretrained(
            str(model_ref),
            local_files_only=bool(args.local_only),
            trust_remote_code=True,
            torch_dtype=dtype,
        )

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    target_modules = [x.strip() for x in str(args.target_modules).split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        target_modules=target_modules,
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    base_adapter_loaded = ""
    base_adapter_path = Path(str(args.base_adapter)).expanduser() if str(args.base_adapter or "").strip() else None
    if base_adapter_path is not None and base_adapter_path.exists():
        try:
            model = PeftModel.from_pretrained(
                model,
                base_adapter_path.as_posix(),
                is_trainable=True,
            )
            base_adapter_loaded = base_adapter_path.as_posix()
        except Exception as e:
            print(f"[warn] failed to load base adapter, fallback to fresh LoRA: {e}")
            model = get_peft_model(model, lora_cfg)
    else:
        model = get_peft_model(model, lora_cfg)
    if bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    model.config.use_cache = False

    eval_ds = DialogueSFTDataset(eval_rows, tokenizer=tokenizer, max_length=int(args.max_length)) if eval_rows else None
    collator = DialogueCollator(pad_id=int(tokenizer.pad_token_id))
    use_cuda = torch.cuda.is_available()

    stage_reports: List[Dict[str, Any]] = []
    final_eval: Dict[str, Any] = {}
    train_samples_used = 0

    if bool(args.curriculum) and len(train_rows) >= 8:
        easy_rows, hard_rows = _split_curriculum_rows(
            train_rows,
            easy_frac=float(args.curriculum_easy_frac),
            seed=int(args.seed),
        )
        stage1_rows = easy_rows
        stage2_rows = list(train_rows)
        stage1_ds = DialogueSFTDataset(stage1_rows, tokenizer=tokenizer, max_length=int(args.max_length))
        stage2_ds = DialogueSFTDataset(stage2_rows, tokenizer=tokenizer, max_length=int(args.max_length))
        if len(stage1_ds) > 0 and len(stage2_ds) > 0:
            total_epochs = max(0.1, float(args.epochs))
            stage1_ratio = min(max(0.1, float(args.curriculum_stage1_ratio)), 0.8)
            stage1_epochs = max(0.1, total_epochs * stage1_ratio)
            stage2_epochs = max(0.1, total_epochs - stage1_epochs)

            setattr(args, "_stage_epochs", stage1_epochs)
            s1 = _run_stage(
                stage_name="curriculum_stage1",
                model=model,
                args=args,
                base_output_dir=output_dir,
                train_ds=stage1_ds,
                eval_ds=eval_ds,
                collator=collator,
                use_cuda=use_cuda,
                dtype=dtype,
            )
            stage_reports.append(s1)

            setattr(args, "_stage_epochs", stage2_epochs)
            s2 = _run_stage(
                stage_name="curriculum_stage2",
                model=model,
                args=args,
                base_output_dir=output_dir,
                train_ds=stage2_ds,
                eval_ds=eval_ds,
                collator=collator,
                use_cuda=use_cuda,
                dtype=dtype,
            )
            stage_reports.append(s2)
            final_eval = dict(s2.get("eval_metrics", {}))
            train_samples_used = int(len(stage2_ds))
            delattr(args, "_stage_epochs")
            curriculum_info = {
                "enabled": True,
                "easy_rows": len(stage1_rows),
                "hard_rows": len(hard_rows),
                "stage1_epochs": stage1_epochs,
                "stage2_epochs": stage2_epochs,
            }
        else:
            curriculum_info = {
                "enabled": False,
                "fallback_reason": "empty_stage_dataset_after_tokenization",
            }
    else:
        curriculum_info = {
            "enabled": False,
            "fallback_reason": "disabled_or_insufficient_rows",
        }

    if not stage_reports:
        train_ds = DialogueSFTDataset(train_rows, tokenizer=tokenizer, max_length=int(args.max_length))
        if len(train_ds) <= 0:
            raise SystemExit("No valid train rows after tokenization")
        setattr(args, "_stage_epochs", float(args.epochs))
        s0 = _run_stage(
            stage_name="single",
            model=model,
            args=args,
            base_output_dir=output_dir,
            train_ds=train_ds,
            eval_ds=eval_ds,
            collator=collator,
            use_cuda=use_cuda,
            dtype=dtype,
        )
        stage_reports.append(s0)
        final_eval = dict(s0.get("eval_metrics", {}))
        train_samples_used = int(len(train_ds))
        delattr(args, "_stage_epochs")

    adapter_out = output_dir / "adapter"
    adapter_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_out.as_posix())
    tokenizer.save_pretrained(adapter_out.as_posix())

    output_storage_before = _path_storage_stats(output_dir)
    output_compaction: Dict[str, Any] = {"enabled": False}
    if bool(args.compact_output):
        output_compaction = _compact_training_output_dir(output_dir)
    output_storage_after = _path_storage_stats(output_dir)
    artifact_registration = {"ok": False, "skipped": True}
    if not bool(args.no_register_artifact):
        artifact_registration = _register_artifact_from_summary(
            registry_path=str(args.artifact_registry_path),
            adapter_path=adapter_out.as_posix(),
            base_model=str(model_ref),
            report_path=(output_dir / "train_summary.json").as_posix(),
            source="train_adapter.single",
            metrics=None,
        )
    artifact_prune = _maybe_prune_artifacts(args, protect_paths=[adapter_out.as_posix()])

    final_stage = stage_reports[-1] if stage_reports else {}
    summary = {
        "ts": time.time(),
        "base_model": str(model_ref),
        "base_adapter_input": str(args.base_adapter or ""),
        "base_adapter_loaded": str(base_adapter_loaded),
        "adapter_path": adapter_out.as_posix(),
        "adapter_storage": _path_storage_stats(adapter_out),
        "output_storage_before": output_storage_before,
        "output_storage_after": output_storage_after,
        "output_compaction": output_compaction,
        "train_samples_raw": len(base_train_rows),
        "replay_samples_raw": len(replay_rows_raw),
        "replay_samples_used": len(replay_rows),
        "train_samples_merged": len(train_rows),
        "eval_samples_raw": len(eval_rows),
        "train_samples_used": train_samples_used,
        "eval_samples_used": len(eval_ds) if eval_ds is not None else 0,
        "train_runtime": float(final_stage.get("train_runtime", 0.0)),
        "train_loss": float(final_stage.get("train_loss", 0.0)),
        "eval_metrics": final_eval,
        "stage_reports": stage_reports,
        "curriculum": curriculum_info,
        "artifact_registration": artifact_registration,
        "artifact_prune": artifact_prune,
        "args": vars(args),
        "data_cleaning": data_cleaning or {"enabled": False},
        "dataset_manifest": manifest or {},
        "runtime_profile": {
            "cuda": bool(use_cuda),
            "dtype": str(dtype),
            "quant_mode": quant_mode,
            "preset_7840hs": bool(args.preset_7840hs),
            "preset_changes": preset_changes,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "group_by_length": bool(args.group_by_length),
            "compact_output": bool(args.compact_output),
        },
    }
    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
