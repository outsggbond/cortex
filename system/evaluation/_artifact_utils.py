from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_path(path: str) -> str:
    src = str(path or "").strip()
    if not src:
        return ""
    p = Path(os.path.expanduser(src))
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p.as_posix()


def _normalize_model_id(model: str) -> str:
    src = str(model or "").strip()
    if not src:
        return ""
    if "://" in src:
        return src.lower()
    # For local paths normalize to absolute path; for model ids keep lower-cased text.
    if any(ch in src for ch in ("/", "\\", ":")) or src.startswith(".") or src.startswith("~"):
        return _normalize_path(src).lower()
    return src.lower()


def _is_base_model_match(candidate: str, preferred: str) -> bool:
    c = _normalize_model_id(candidate)
    p = _normalize_model_id(preferred)
    if not p:
        return True
    if not c:
        return False
    return c == p


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for child in path.rglob("*"):
        if child.is_file():
            yield child


def _adapter_validity(path: Path) -> tuple[bool, str]:
    p = Path(path)
    if not p.exists():
        return False, "missing"
    if not p.is_dir():
        return False, "not_directory"
    cfg = p / "adapter_config.json"
    w_sf = p / "adapter_model.safetensors"
    w_bin = p / "adapter_model.bin"
    if not cfg.exists():
        return False, "missing_adapter_config"
    if not (w_sf.exists() or w_bin.exists()):
        return False, "missing_adapter_weights"
    return True, "ok"


def _find_adapter_dir(root: Path) -> Path | None:
    r = Path(root)
    if not r.exists():
        return None
    ok, _reason = _adapter_validity(r)
    if ok:
        return r
    for cand in r.rglob("*"):
        if not cand.is_dir():
            continue
        ok, _reason = _adapter_validity(cand)
        if ok:
            return cand
    return None


def _dir_stats(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "file_count": 0,
            "modified_at": 0.0,
        }
    size_bytes = 0
    file_count = 0
    modified_at = 0.0
    for child in _iter_files(path):
        try:
            st = child.stat()
        except Exception:
            continue
        file_count += 1
        size_bytes += int(st.st_size)
        modified_at = max(modified_at, float(st.st_mtime))
    if modified_at <= 0.0:
        try:
            modified_at = float(path.stat().st_mtime)
        except Exception:
            modified_at = 0.0
    return {
        "exists": True,
        "size_bytes": int(size_bytes),
        "file_count": int(file_count),
        "modified_at": float(modified_at),
    }


def _artifact_id(real_path: str, size_bytes: int, modified_at: float) -> str:
    h = hashlib.sha1()
    h.update(str(real_path).encode("utf-8", errors="ignore"))
    h.update(str(int(size_bytes)).encode("utf-8"))
    h.update(str(int(modified_at * 1000)).encode("utf-8"))
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_archive_signing_key(explicit_key: str = "") -> str:
    key = str(explicit_key or "").strip()
    if key:
        return key
    return str(os.environ.get("ADAPTER_ARCHIVE_SIGNING_KEY", "")).strip()


def _resolve_archive_signing_key_id(explicit_key_id: str = "") -> str:
    kid = str(explicit_key_id or "").strip()
    if kid:
        return kid
    return str(os.environ.get("ADAPTER_ARCHIVE_SIGNING_KEY_ID", "")).strip()


def _load_archive_signing_keys_map(raw_json: str = "") -> Dict[str, str]:
    src = str(raw_json or "").strip() or str(os.environ.get("ADAPTER_ARCHIVE_SIGNING_KEYS_JSON", "")).strip()
    if not src:
        return {}
    try:
        node = json.loads(src)
    except Exception:
        return {}
    if not isinstance(node, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in node.items():
        kk = str(k or "").strip()
        vv = str(v or "").strip()
        if kk and vv:
            out[kk] = vv
    return out


def _resolve_archive_signing_key_for_row(
    *,
    explicit_key: str,
    row: Dict[str, Any] | None = None,
    signing_keys_json: str = "",
) -> str:
    direct = _resolve_archive_signing_key(explicit_key)
    if direct:
        return direct
    key_map = _load_archive_signing_keys_map(signing_keys_json)
    if not key_map:
        return ""
    node = row if isinstance(row, dict) else {}
    key_id = str(node.get("archive_signing_key_id", "") or node.get("archive_key_id", "")).strip()
    if key_id and key_id in key_map:
        return str(key_map.get(key_id, "")).strip()
    return ""


def _archive_signature_payload(
    *,
    archive_sha256: str,
    artifact_id: str,
    base_model_norm: str,
    archive_size_bytes: int,
) -> str:
    return (
        f"v1|sha256={str(archive_sha256 or '').strip().lower()}"
        f"|artifact_id={str(artifact_id or '').strip()}"
        f"|base_model_norm={str(base_model_norm or '').strip().lower()}"
        f"|archive_size_bytes={int(max(0, int(archive_size_bytes or 0)))}"
    )


def _archive_hmac_sha256(
    *,
    archive_sha256: str,
    artifact_id: str,
    base_model_norm: str,
    archive_size_bytes: int,
    signing_key: str,
) -> str:
    key = str(signing_key or "").encode("utf-8", errors="ignore")
    payload = _archive_signature_payload(
        archive_sha256=archive_sha256,
        artifact_id=artifact_id,
        base_model_norm=base_model_norm,
        archive_size_bytes=archive_size_bytes,
    ).encode("utf-8", errors="ignore")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    keep = {}
    for k in ("sample_count", "avg_quality", "avg_overlap", "avg_semantic", "pass_rate"):
        if k in metrics:
            keep[k] = metrics.get(k)
    buckets = metrics.get("by_bucket", {})
    if isinstance(buckets, dict):
        small: Dict[str, Dict[str, Any]] = {}
        for bk, node in buckets.items():
            if not isinstance(node, dict):
                continue
            small[str(bk)] = {
                "sample_count": node.get("sample_count", 0),
                "avg_quality": node.get("avg_quality", 0.0),
                "avg_semantic": node.get("avg_semantic", 0.0),
                "pass_rate": node.get("pass_rate", 0.0),
            }
        keep["by_bucket"] = small
    return keep


def _clamp01(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _metric_quality_score(row: Dict[str, Any]) -> float:
    node = row.get("metrics", {})
    if not isinstance(node, dict):
        return 0.0
    q = _clamp01(node.get("avg_quality", 0.0))
    p = _clamp01(node.get("pass_rate", 0.0))
    s = _clamp01(node.get("avg_semantic", 0.0))
    n = max(0, _safe_int(node.get("sample_count", 0), 0))
    # Higher score means better quality confidence.
    return float(0.45 * p + 0.35 * q + 0.20 * s + 0.02 * math.log1p(n))


def _freshness_score(row: Dict[str, Any]) -> float:
    return max(
        _safe_float(row.get("last_promoted_at", 0.0), 0.0),
        _safe_float(row.get("last_used_at", 0.0), 0.0),
        _safe_float(row.get("updated_at", 0.0), 0.0),
    )


def _artifact_rank_tuple(row: Dict[str, Any]) -> tuple:
    promoted_bias = 1 if _safe_float(row.get("last_promoted_at", 0.0), 0.0) > 0.0 else 0
    quality = _metric_quality_score(row)
    fresh = _freshness_score(row)
    return (
        int(promoted_bias),
        float(quality),
        float(fresh),
    )
