from __future__ import annotations

import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from system.evaluation._artifact_registry import _find_artifact_idx, _load_registry
from system.evaluation._artifact_utils import (
    _adapter_validity,
    _archive_hmac_sha256,
    _artifact_id,
    _artifact_rank_tuple,
    _clamp01,
    _compact_metrics,
    _dir_stats,
    _file_sha256,
    _find_adapter_dir,
    _freshness_score,
    _is_base_model_match,
    _iter_files,
    _load_archive_signing_keys_map,
    _metric_quality_score,
    _normalize_model_id,
    _normalize_path,
    _read_json,
    _resolve_archive_signing_key,
    _resolve_archive_signing_key_for_row,
    _resolve_archive_signing_key_id,
    _safe_float,
    _safe_int,
    _write_json,
)


def register_adapter_artifact(
    *,
    registry_path: str,
    adapter_path: str,
    base_model: str = "",
    metrics: Dict[str, Any] | None = None,
    report_path: str = "",
    reason: str = "",
    source: str = "",
    promoted: bool = False,
    used: bool = True,
    now: float | None = None,
) -> Dict[str, Any]:
    ts = float(now if now is not None else time.time())
    reg_path = Path(_normalize_path(registry_path) or registry_path)
    adapter_norm = _normalize_path(adapter_path)
    p = Path(adapter_norm) if adapter_norm else Path("")
    stats = _dir_stats(p) if adapter_norm else {"exists": False, "size_bytes": 0, "file_count": 0, "modified_at": 0.0}
    real_path = ""
    if adapter_norm:
        try:
            real_path = p.resolve().as_posix()
        except Exception:
            real_path = adapter_norm
    art_id = _artifact_id(real_path or adapter_norm, int(stats["size_bytes"]), float(stats["modified_at"]))

    reg = _load_registry(reg_path)
    rows: List[Dict[str, Any]] = list(reg.get("artifacts", []))
    idx = _find_artifact_idx(rows, real_path=real_path, adapter_path=adapter_norm)
    if idx >= 0:
        node = dict(rows[idx])
    else:
        node = {
            "id": art_id,
            "adapter_path": adapter_norm,
            "real_path": real_path,
            "first_seen_at": ts,
            "use_count": 0,
            "promoted_count": 0,
        }

    node["id"] = art_id
    node["adapter_path"] = adapter_norm
    node["real_path"] = real_path
    node["exists"] = bool(stats.get("exists", False))
    node["size_bytes"] = int(stats.get("size_bytes", 0))
    node["file_count"] = int(stats.get("file_count", 0))
    node["modified_at"] = float(stats.get("modified_at", 0.0))
    node["updated_at"] = ts
    valid, valid_reason = _adapter_validity(p) if adapter_norm else (False, "empty_path")
    node["valid"] = bool(valid)
    node["valid_reason"] = str(valid_reason)
    if base_model:
        node["base_model"] = str(base_model)
        node["base_model_norm"] = _normalize_model_id(base_model)
    if report_path:
        node["report_path"] = _normalize_path(report_path)
    if reason:
        node["reason"] = str(reason)
    if source:
        node["source"] = str(source)
    if isinstance(metrics, dict) and metrics:
        node["metrics"] = _compact_metrics(metrics)
    if used:
        node["last_used_at"] = ts
        node["use_count"] = _safe_int(node.get("use_count", 0), 0) + 1
    if promoted:
        node["last_promoted_at"] = ts
        node["promoted_count"] = _safe_int(node.get("promoted_count", 0), 0) + 1
        reg["active_id"] = art_id

    if idx >= 0:
        rows[idx] = node
    else:
        rows.append(node)
    rows.sort(
        key=lambda x: (
            _artifact_rank_tuple(x),
            _safe_float(x.get("last_used_at", 0.0), 0.0),
            _safe_float(x.get("updated_at", 0.0), 0.0),
        ),
        reverse=True,
    )
    reg["artifacts"] = rows
    reg["updated_at"] = ts
    _write_json(reg_path, reg)
    return node


def resolve_best_adapter_path(
    *,
    active_registry_path: str = "artifacts/checkpoints/active_adapter.json",
    artifact_registry_path: str = "artifacts/checkpoints/adapter_artifacts.json",
    preferred_base_model: str = "",
    strict_base_model: bool = False,
    require_valid: bool = True,
) -> str:
    active = _read_json(Path(_normalize_path(active_registry_path) or active_registry_path))
    active_base = str(active.get("base_model", "")).strip()
    active_base_ok = _is_base_model_match(active_base, preferred_base_model) if active_base else (not bool(strict_base_model))
    for key in ("active_adapter", "previous_adapter"):
        if bool(strict_base_model) and (not active_base_ok):
            continue
        cand = _normalize_path(str(active.get(key, "")).strip())
        if not cand or (not Path(cand).exists()):
            continue
        if bool(require_valid):
            ok, _reason = _adapter_validity(Path(cand))
            if not ok:
                continue
        if cand:
            return cand

    reg = _load_registry(Path(_normalize_path(artifact_registry_path) or artifact_registry_path))
    active_id = str(reg.get("active_id", "")).strip()
    rows = [x for x in reg.get("artifacts", []) if isinstance(x, dict)]
    if active_id:
        for row in rows:
            if str(row.get("id", "")).strip() != active_id:
                continue
            row_base = str(row.get("base_model_norm", "") or row.get("base_model", "")).strip()
            if bool(strict_base_model) and (not _is_base_model_match(row_base, preferred_base_model)):
                continue
            cand = _normalize_path(str(row.get("adapter_path", "")).strip())
            if not cand or (not Path(cand).exists()):
                continue
            if bool(require_valid):
                is_valid = bool(row.get("valid", True))
                if not is_valid:
                    continue
                ok, _reason = _adapter_validity(Path(cand))
                if not ok:
                    continue
            if cand:
                return cand
    available: List[tuple[Dict[str, Any], str]] = []
    for row in rows:
        row_base = str(row.get("base_model_norm", "") or row.get("base_model", "")).strip()
        base_match = _is_base_model_match(row_base, preferred_base_model)
        if bool(strict_base_model) and (not base_match):
            continue
        cand = _normalize_path(str(row.get("adapter_path", "")).strip())
        if not cand or (not Path(cand).exists()):
            continue
        if bool(require_valid):
            is_valid = bool(row.get("valid", True))
            if not is_valid:
                continue
            ok, _reason = _adapter_validity(Path(cand))
            if not ok:
                continue
        node = dict(row)
        node["base_match"] = bool(base_match)
        available.append((node, cand))
    if not available:
        return ""
    available.sort(
        key=lambda item: (
            1 if bool(item[0].get("base_match", False)) else 0,
            _artifact_rank_tuple(item[0]),
            _safe_float(item[0].get("updated_at", 0.0), 0.0),
        ),
        reverse=True,
    )
    for _row, cand in available:
        if cand:
            return cand
    return ""


def mark_adapter_used(
    *,
    registry_path: str,
    adapter_path: str,
    source: str = "runtime",
    base_model: str = "",
    now: float | None = None,
) -> Dict[str, Any]:
    return register_adapter_artifact(
        registry_path=registry_path,
        adapter_path=adapter_path,
        source=source,
        base_model=base_model,
        promoted=False,
        used=True,
        now=now,
    )


def prune_adapter_artifacts(
    *,
    registry_path: str,
    keep_latest: int = 0,
    keep_promoted: int = 0,
    max_total_bytes: int = 0,
    protected_paths: List[str] | None = None,
    remove_missing: bool = True,
    remove_invalid: bool = False,
    delete_files: bool = False,
    archive_before_delete: bool = False,
    archive_dir: str = "",
    archive_signing_key: str = "",
    archive_signing_key_id: str = "",
    now: float | None = None,
) -> Dict[str, Any]:
    def _row_id(row: Dict[str, Any], index: int) -> str:
        rid = str(row.get("id", "")).strip()
        if rid:
            return rid
        fallback_path = _normalize_path(str(row.get("adapter_path", "")).strip()) or f"row_{index}"
        rid = _artifact_id(
            fallback_path,
            _safe_int(row.get("size_bytes", 0), 0),
            _safe_float(row.get("modified_at", 0.0), 0.0),
        )
        row["id"] = rid
        return rid

    def _safe_stem(raw: str) -> str:
        src = str(raw or "").strip() or "artifact"
        buf = [ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in src]
        out = "".join(buf).strip("._")
        return out or "artifact"

    def _archive_artifact(src_path: Path, artifact_id: str, archive_root: str, at: float) -> str:
        target_root = Path(archive_root)
        target_root.mkdir(parents=True, exist_ok=True)
        stem = _safe_stem(src_path.name)
        aid = _safe_stem(str(artifact_id or "na"))[:12] or "na"
        stamp = int(at * 1000)
        idx = 0
        while True:
            suffix = "" if idx == 0 else f"_{idx}"
            out_zip = target_root / f"{stem}_{aid}_{stamp}{suffix}.zip"
            if not out_zip.exists():
                break
            idx += 1
        with zipfile.ZipFile(out_zip.as_posix(), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if src_path.is_dir():
                for fp in sorted(src_path.rglob("*")):
                    if not fp.is_file():
                        continue
                    rel = fp.relative_to(src_path).as_posix()
                    zf.write(fp.as_posix(), arcname=f"{src_path.name}/{rel}")
            else:
                zf.write(src_path.as_posix(), arcname=src_path.name)
        return out_zip.as_posix()

    ts = float(now if now is not None else time.time())
    reg_path = Path(_normalize_path(registry_path) or registry_path)
    reg = _load_registry(reg_path)
    rows = [dict(x) for x in reg.get("artifacts", []) if isinstance(x, dict)]
    archive_rows = [dict(x) for x in reg.get("archives", []) if isinstance(x, dict)]
    if not rows:
        return {
            "removed": 0,
            "removed_by_policy": 0,
            "removed_missing": 0,
            "removed_invalid": 0,
            "deleted_files": 0,
            "deleted_dirs": 0,
            "delete_failures": 0,
            "archived": 0,
            "archive_failures": 0,
            "archived_paths": [],
            "remaining": 0,
        }

    norm_protect = {_normalize_path(x) for x in (protected_paths or []) if str(x or "").strip()}
    norm_protect = {x for x in norm_protect if x}
    active_id = str(reg.get("active_id", "")).strip()

    row_by_id: Dict[str, Dict[str, Any]] = {}
    missing_ids: set[str] = set()
    invalid_ids: set[str] = set()
    for idx, row in enumerate(rows):
        rid = _row_id(row, idx)
        row_by_id[rid] = row
        path = _normalize_path(str(row.get("adapter_path", "")).strip())
        row["adapter_path"] = path
        ok, _reason = _adapter_validity(Path(path)) if path else (False, "empty_path")
        if not path or not Path(path).exists():
            missing_ids.add(rid)
        if not ok:
            invalid_ids.add(rid)

    scan_removed_ids: set[str] = set()
    if remove_missing:
        scan_removed_ids.update(missing_ids)
    if remove_invalid:
        invalid_base = invalid_ids - (missing_ids if remove_missing else set())
        scan_removed_ids.update(invalid_base)

    base_rows = [row for row in rows if str(row.get("id", "")).strip() not in scan_removed_ids]
    keep_ids: set[str] = set()
    if active_id:
        keep_ids.add(active_id)
    for row in base_rows:
        rid = str(row.get("id", "")).strip()
        path = _normalize_path(str(row.get("adapter_path", "")).strip())
        if rid and path and path in norm_protect:
            keep_ids.add(rid)

    kp = max(0, int(keep_promoted))
    if kp > 0:
        promoted_sorted = sorted(
            base_rows,
            key=lambda x: _safe_float(x.get("last_promoted_at", 0.0), 0.0),
            reverse=True,
        )
        for row in promoted_sorted[:kp]:
            rid = str(row.get("id", "")).strip()
            if rid:
                keep_ids.add(rid)

    kl = max(0, int(keep_latest))
    if kl > 0:
        latest_sorted = sorted(
            base_rows,
            key=lambda x: _safe_float(x.get("updated_at", 0.0), 0.0),
            reverse=True,
        )
        for row in latest_sorted[:kl]:
            rid = str(row.get("id", "")).strip()
            if rid:
                keep_ids.add(rid)

    removed_policy_ids: set[str] = set()
    if kl > 0 or kp > 0:
        for row in base_rows:
            rid = str(row.get("id", "")).strip()
            if rid and rid not in keep_ids:
                removed_policy_ids.add(rid)

    kept_rows = [row for row in base_rows if str(row.get("id", "")).strip() not in removed_policy_ids]
    mt = max(0, int(max_total_bytes))
    if mt > 0:
        total = sum(max(0, _safe_int(r.get("size_bytes", 0), 0)) for r in kept_rows)
        if total > mt:
            drop_candidates = sorted(
                [r for r in kept_rows if str(r.get("id", "")).strip() not in keep_ids],
                key=lambda x: _safe_float(x.get("updated_at", 0.0), 0.0),
            )
            for row in drop_candidates:
                if total <= mt:
                    break
                rid = str(row.get("id", "")).strip()
                if not rid or rid in keep_ids:
                    continue
                removed_policy_ids.add(rid)
                total -= max(0, _safe_int(row.get("size_bytes", 0), 0))

    requested_remove_ids = set(scan_removed_ids) | set(removed_policy_ids)
    deleted_files = 0
    deleted_dirs = 0
    delete_failures = 0
    archived = 0
    archive_failures = 0
    archived_paths: List[str] = []
    archive_records: List[Dict[str, Any]] = []
    failed_ids: set[str] = set()
    arch_dir = _normalize_path(archive_dir)
    sign_key = _resolve_archive_signing_key(archive_signing_key)
    sign_key_id = _resolve_archive_signing_key_id(archive_signing_key_id)

    if delete_files and requested_remove_ids:
        for rid in sorted(requested_remove_ids):
            row = row_by_id.get(rid)
            if not isinstance(row, dict):
                continue
            path = _normalize_path(str(row.get("adapter_path", "")).strip())
            if not path or path in norm_protect:
                continue
            p = Path(path)
            if not p.exists():
                continue

            rec: Dict[str, Any] | None = None
            if archive_before_delete:
                try:
                    if not arch_dir:
                        raise RuntimeError("archive_dir_empty")
                    archived_path = _archive_artifact(p, rid, arch_dir, ts)
                    archived_file = Path(archived_path)
                    archive_sha256 = _file_sha256(archived_file).lower()
                    archive_size_bytes = int(archived_file.stat().st_size) if archived_file.exists() else 0
                    base_model_norm = str(row.get("base_model_norm", "")).strip()
                    if not base_model_norm:
                        base_model_norm = _normalize_model_id(str(row.get("base_model", "")))
                    archive_hmac_sha256 = ""
                    archive_sig_algo = ""
                    if sign_key:
                        archive_hmac_sha256 = _archive_hmac_sha256(
                            archive_sha256=archive_sha256,
                            artifact_id=rid,
                            base_model_norm=base_model_norm,
                            archive_size_bytes=archive_size_bytes,
                            signing_key=sign_key,
                        ).lower()
                        archive_sig_algo = "hmac-sha256-v1"
                    archived += 1
                    archived_paths.append(archived_path)
                    rec = {
                        "artifact_id": rid,
                        "archive_path": archived_path,
                        "archive_sha256": archive_sha256,
                        "archive_hmac_sha256": archive_hmac_sha256,
                        "archive_sig_algo": archive_sig_algo,
                        "archive_signed": bool(archive_hmac_sha256),
                        "archive_signed_at": ts if archive_hmac_sha256 else 0.0,
                        "archive_signing_key_id": sign_key_id if archive_hmac_sha256 else "",
                        "archive_size_bytes": archive_size_bytes,
                        "source_adapter_path": path,
                        "archived_at": ts,
                        "base_model": str(row.get("base_model", "")),
                        "base_model_norm": base_model_norm,
                        "size_bytes": int(row.get("size_bytes", 0)),
                        "file_count": int(row.get("file_count", 0)),
                        "metrics": row.get("metrics", {}),
                        "restore_count": 0,
                    }
                except Exception:
                    archive_failures += 1
                    failed_ids.add(rid)
                    continue

            try:
                if p.is_file():
                    p.unlink()
                    deleted_files += 1
                elif p.is_dir():
                    shutil.rmtree(p.as_posix(), ignore_errors=False)
                    deleted_dirs += 1
            except Exception:
                delete_failures += 1
                failed_ids.add(rid)
                if isinstance(rec, dict):
                    rec["delete_succeeded"] = False
                    archive_records.append(rec)
                continue

            if isinstance(rec, dict):
                rec["delete_succeeded"] = True
                archive_records.append(rec)

    final_removed_ids = {rid for rid in requested_remove_ids if rid not in failed_ids}
    final_rows = [row for row in rows if str(row.get("id", "")).strip() not in final_removed_ids]

    merged_archives: Dict[str, Dict[str, Any]] = {}
    for row in archive_rows:
        ap = _normalize_path(str(row.get("archive_path", "")).strip())
        if not ap:
            continue
        node = dict(row)
        node["archive_path"] = ap
        node["restore_count"] = _safe_int(node.get("restore_count", 0), 0)
        merged_archives[ap] = node
    for rec in archive_records:
        ap = _normalize_path(str(rec.get("archive_path", "")).strip())
        if not ap:
            continue
        node = dict(rec)
        node["archive_path"] = ap
        node["restore_count"] = _safe_int(node.get("restore_count", 0), 0)
        merged_archives[ap] = node
    final_archives = sorted(
        merged_archives.values(),
        key=lambda x: max(
            _safe_float(x.get("archived_at", 0.0), 0.0),
            _safe_float(x.get("last_restored_at", 0.0), 0.0),
        ),
        reverse=True,
    )[:4000]

    reg["artifacts"] = final_rows
    reg["archives"] = final_archives
    if active_id and all(str(x.get("id", "")).strip() != active_id for x in final_rows):
        reg["active_id"] = ""
    reg["updated_at"] = ts
    _write_json(reg_path, reg)

    removed_missing_count = int(len(final_removed_ids & missing_ids)) if remove_missing else 0
    invalid_base_for_count = invalid_ids - (missing_ids if remove_missing else set())
    removed_invalid_count = int(len(final_removed_ids & invalid_base_for_count)) if remove_invalid else 0
    removed_by_policy_count = int(len(final_removed_ids & removed_policy_ids))
    return {
        "removed": int(len(final_removed_ids)),
        "removed_by_policy": removed_by_policy_count,
        "removed_missing": removed_missing_count,
        "removed_invalid": removed_invalid_count,
        "deleted_files": int(deleted_files),
        "deleted_dirs": int(deleted_dirs),
        "delete_failures": int(delete_failures),
        "archived": int(archived),
        "archive_failures": int(archive_failures),
        "archived_paths": archived_paths[:20],
        "remaining": int(len(final_rows)),
    }


def restore_archived_artifact(
    *,
    registry_path: str,
    archive_path: str = "",
    artifact_id: str = "",
    preferred_base_model: str = "",
    strict_base_model: bool = False,
    output_dir: str = "artifacts/checkpoints/restored_adapters",
    activate: bool = False,
    verify_archive_hash: bool = True,
    verify_archive_signature: bool = True,
    require_archive_signature: bool = False,
    archive_signing_key: str = "",
    archive_signing_keys_json: str = "",
    overwrite: bool = False,
    now: float | None = None,
) -> Dict[str, Any]:
    def _safe_stem(raw: str) -> str:
        src = str(raw or "").strip() or "adapter"
        buf = [ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in src]
        out = "".join(buf).strip("._")
        return out or "adapter"

    def _safe_extract_zip(zf: zipfile.ZipFile, dst: Path) -> None:
        root = dst.resolve()
        for info in zf.infolist():
            name = str(info.filename or "")
            if not name:
                continue
            target = (root / name).resolve()
            try:
                target.relative_to(root)
            except Exception:
                raise RuntimeError(f"unsafe_archive_path:{name}")
        zf.extractall(root.as_posix())

    def _verify_candidate(
        row: Dict[str, Any],
        src_zip: Path,
    ) -> tuple[bool, str, str, str, str, str, bool]:
        expected_hash_local = str(row.get("archive_sha256", "")).strip().lower()
        actual_hash_local = ""
        if bool(verify_archive_hash) and expected_hash_local:
            try:
                actual_hash_local = _file_sha256(src_zip).lower()
            except Exception as e:
                return False, f"archive_hash_compute_failed:{e}", expected_hash_local, actual_hash_local, "", "", False
            if actual_hash_local != expected_hash_local:
                return False, "archive_hash_mismatch", expected_hash_local, actual_hash_local, "", "", False

        expected_sig_local = str(row.get("archive_hmac_sha256", "")).strip().lower()
        if bool(require_archive_signature) and (not expected_sig_local):
            return False, "archive_signature_missing", expected_hash_local, actual_hash_local, expected_sig_local, "", False

        actual_sig_local = ""
        sig_verified_local = False
        if bool(verify_archive_signature) and expected_sig_local:
            sign_key = _resolve_archive_signing_key_for_row(
                explicit_key=archive_signing_key,
                row=row,
                signing_keys_json=archive_signing_keys_json,
            )
            if not sign_key:
                return (
                    False,
                    "archive_signature_key_missing",
                    expected_hash_local,
                    actual_hash_local,
                    expected_sig_local,
                    actual_sig_local,
                    sig_verified_local,
                )
            if not actual_hash_local:
                try:
                    actual_hash_local = _file_sha256(src_zip).lower()
                except Exception as e:
                    return (
                        False,
                        f"archive_hash_compute_failed:{e}",
                        expected_hash_local,
                        actual_hash_local,
                        expected_sig_local,
                        actual_sig_local,
                        sig_verified_local,
                    )
            sig_artifact_id = str(row.get("artifact_id", "")).strip()
            sig_base_model_norm = str(row.get("base_model_norm", "")).strip() or _normalize_model_id(
                str(row.get("base_model", ""))
            )
            sig_size = _safe_int(row.get("archive_size_bytes", 0), 0)
            if sig_size <= 0:
                try:
                    sig_size = int(src_zip.stat().st_size)
                except Exception:
                    sig_size = 0
            actual_sig_local = _archive_hmac_sha256(
                archive_sha256=actual_hash_local,
                artifact_id=sig_artifact_id,
                base_model_norm=sig_base_model_norm,
                archive_size_bytes=sig_size,
                signing_key=sign_key,
            ).lower()
            if actual_sig_local != expected_sig_local:
                return (
                    False,
                    "archive_signature_mismatch",
                    expected_hash_local,
                    actual_hash_local,
                    expected_sig_local,
                    actual_sig_local,
                    sig_verified_local,
                )
            sig_verified_local = True
        return (
            True,
            "ok",
            expected_hash_local,
            actual_hash_local,
            expected_sig_local,
            actual_sig_local,
            sig_verified_local,
        )

    ts = float(now if now is not None else time.time())
    reg_path = Path(_normalize_path(registry_path) or registry_path)
    reg = _load_registry(reg_path)
    raw_archive_rows = [dict(x) for x in reg.get("archives", []) if isinstance(x, dict)]
    archive_rows: List[Dict[str, Any]] = []
    for row in raw_archive_rows:
        node = dict(row)
        ap = _normalize_path(str(node.get("archive_path", "")).strip())
        if not ap:
            continue
        node["archive_path"] = ap
        archive_rows.append(node)
    archive_rows.sort(
        key=lambda x: max(
            _safe_float(x.get("archived_at", 0.0), 0.0),
            _safe_float(x.get("last_restored_at", 0.0), 0.0),
        ),
        reverse=True,
    )

    pref_base = str(preferred_base_model or "").strip()
    strict_base = bool(strict_base_model)

    def _archive_base_match(row: Dict[str, Any]) -> bool:
        row_base = str(row.get("base_model_norm", "") or row.get("base_model", "")).strip()
        return _is_base_model_match(row_base, pref_base)

    norm_archive_path = _normalize_path(archive_path)
    aid_filter = str(artifact_id or "").strip()
    candidate_rows: List[Dict[str, Any]] = []
    if norm_archive_path:
        matched = []
        for row in archive_rows:
            ap = str(row.get("archive_path", "")).strip()
            if ap and ap == norm_archive_path:
                node = dict(row)
                node["archive_path"] = ap
                matched.append(node)
        if matched:
            candidate_rows = matched
        else:
            candidate_rows = [{"archive_path": norm_archive_path, "artifact_id": aid_filter}]
    elif aid_filter:
        for row in archive_rows:
            if str(row.get("artifact_id", "")).strip() != aid_filter:
                continue
            ap = str(row.get("archive_path", "")).strip()
            if not ap:
                continue
            node = dict(row)
            node["archive_path"] = ap
            candidate_rows.append(node)
    else:
        ranked: List[tuple[int, int, float, Dict[str, Any]]] = []
        for row in archive_rows:
            ap = str(row.get("archive_path", "")).strip()
            if not ap or (not Path(ap).is_file()):
                continue
            base_match = bool(_archive_base_match(row))
            if strict_base and pref_base and (not base_match):
                continue
            freshness = max(
                _safe_float(row.get("archived_at", 0.0), 0.0),
                _safe_float(row.get("last_restored_at", 0.0), 0.0),
            )
            sig_bias = 1 if str(row.get("archive_hmac_sha256", "")).strip() else 0
            ranked.append((1 if base_match else 0, int(sig_bias), float(freshness), dict(row)))
        ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        candidate_rows = [x[3] for x in ranked]
    if (not candidate_rows) and strict_base and pref_base and (not norm_archive_path) and (not aid_filter):
        return {
            "restored": False,
            "reason": "archive_not_found_for_base_model",
            "archive_path": "",
            "preferred_base_model": pref_base,
        }
    if not candidate_rows:
        return {
            "restored": False,
            "reason": "archive_not_found",
            "archive_path": "",
        }
    out_root_norm = _normalize_path(output_dir) or _normalize_path("artifacts/checkpoints/restored_adapters")
    out_root = Path(out_root_norm)
    out_root.mkdir(parents=True, exist_ok=True)

    selected_row: Dict[str, Any] | None = None
    selected_src_archive = ""
    selected_restore_root: Path | None = None
    selected_adapter_path = ""
    selected_expected_hash = ""
    selected_actual_hash = ""
    selected_expected_sig = ""
    selected_actual_sig = ""
    selected_sig_verified = False
    candidate_failures: List[Dict[str, Any]] = []

    for cand_idx, row in enumerate(candidate_rows):
        src_archive = _normalize_path(str(row.get("archive_path", "")).strip())
        if not src_archive:
            candidate_failures.append({"archive_path": "", "reason": "archive_path_empty"})
            continue
        src_zip = Path(src_archive)
        if not src_zip.is_file():
            candidate_failures.append({"archive_path": src_archive, "reason": "archive_missing"})
            continue

        ok, reason, expected_hash, actual_hash, expected_sig, actual_sig, sig_verified = _verify_candidate(row, src_zip)
        if not ok:
            fail_payload: Dict[str, Any] = {"archive_path": src_archive, "reason": reason}
            if expected_hash:
                fail_payload["expected_sha256"] = expected_hash
            if actual_hash:
                fail_payload["actual_sha256"] = actual_hash
            if expected_sig:
                fail_payload["expected_signature"] = expected_sig
            if actual_sig:
                fail_payload["actual_signature"] = actual_sig
            candidate_failures.append(fail_payload)
            continue

        source_name = Path(str(row.get("source_adapter_path", "")).strip() or "").name or src_zip.stem
        stem = _safe_stem(source_name)
        stamp = int(ts * 1000)
        restore_root = out_root / f"{stem}_{stamp}_{cand_idx}"
        if restore_root.exists() and overwrite:
            shutil.rmtree(restore_root.as_posix(), ignore_errors=True)
        if restore_root.exists() and (not overwrite):
            idx = 1
            while True:
                alt = out_root / f"{stem}_{stamp}_{cand_idx}_{idx}"
                if not alt.exists():
                    restore_root = alt
                    break
                idx += 1
        restore_root.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(src_zip.as_posix(), "r") as zf:
                _safe_extract_zip(zf, restore_root)
        except Exception as e:
            shutil.rmtree(restore_root.as_posix(), ignore_errors=True)
            candidate_failures.append({"archive_path": src_archive, "reason": f"extract_failed:{e}"})
            continue

        adapter_dir = _find_adapter_dir(restore_root)
        if adapter_dir is None:
            shutil.rmtree(restore_root.as_posix(), ignore_errors=True)
            candidate_failures.append({"archive_path": src_archive, "reason": "adapter_dir_not_found"})
            continue

        selected_row = dict(row)
        selected_src_archive = src_archive
        selected_restore_root = restore_root
        selected_adapter_path = adapter_dir.resolve().as_posix()
        selected_expected_hash = expected_hash
        selected_actual_hash = actual_hash
        selected_expected_sig = expected_sig
        selected_actual_sig = actual_sig
        selected_sig_verified = bool(sig_verified)
        break

    if selected_row is None:
        reason = str(candidate_failures[-1].get("reason", "archive_not_found")) if candidate_failures else "archive_not_found"
        out = {
            "restored": False,
            "reason": reason,
            "archive_path": str(candidate_failures[-1].get("archive_path", "")) if candidate_failures else "",
            "preferred_base_model": pref_base,
            "strict_base_model": bool(strict_base),
        }
        if candidate_failures:
            out["failed_candidates"] = candidate_failures[:8]
        return out

    adapter_path = selected_adapter_path
    base_model = str(selected_row.get("base_model", "")).strip()
    node = register_adapter_artifact(
        registry_path=reg_path.as_posix(),
        adapter_path=adapter_path,
        base_model=base_model,
        source="archive_restore",
        promoted=bool(activate),
        used=True,
        now=ts,
    )

    reg2 = _load_registry(reg_path)
    arch_rows2 = [dict(x) for x in reg2.get("archives", []) if isinstance(x, dict)]
    src_norm = _normalize_path(selected_src_archive)
    touched = False
    for row in arch_rows2:
        ap = _normalize_path(str(row.get("archive_path", "")).strip())
        if not ap or ap != src_norm:
            continue
        row["archive_path"] = ap
        row["last_restored_at"] = ts
        row["restore_count"] = _safe_int(row.get("restore_count", 0), 0) + 1
        row["restored_adapter_path"] = adapter_path
        row["restored_artifact_id"] = str(node.get("id", "")).strip()
        row["restored_activated"] = bool(activate)
        if selected_expected_hash:
            row["archive_sha256"] = selected_expected_hash
        if selected_expected_sig:
            row["archive_hmac_sha256"] = selected_expected_sig
            row["archive_sig_algo"] = str(row.get("archive_sig_algo", "")).strip() or "hmac-sha256-v1"
            row["archive_signed"] = True
        touched = True
        break
    if not touched:
        arch_rows2.append(
            {
                "artifact_id": str(selected_row.get("artifact_id", "") or artifact_id or "").strip(),
                "archive_path": src_norm,
                "source_adapter_path": "",
                "archived_at": 0.0,
                "archive_sha256": selected_expected_hash or selected_actual_hash,
                "archive_hmac_sha256": selected_expected_sig or selected_actual_sig,
                "archive_sig_algo": "hmac-sha256-v1" if (selected_expected_sig or selected_actual_sig) else "",
                "archive_signed": bool(selected_expected_sig or selected_actual_sig),
                "restore_count": 1,
                "last_restored_at": ts,
                "restored_adapter_path": adapter_path,
                "restored_artifact_id": str(node.get("id", "")).strip(),
                "restored_activated": bool(activate),
            }
        )
    normalized_archives: List[Dict[str, Any]] = []
    for row in arch_rows2:
        ap = _normalize_path(str(row.get("archive_path", "")).strip())
        if not ap:
            continue
        node2 = dict(row)
        node2["archive_path"] = ap
        node2["restore_count"] = _safe_int(node2.get("restore_count", 0), 0)
        normalized_archives.append(node2)
    reg2["archives"] = sorted(
        normalized_archives,
        key=lambda x: max(
            _safe_float(x.get("archived_at", 0.0), 0.0),
            _safe_float(x.get("last_restored_at", 0.0), 0.0),
        ),
        reverse=True,
    )[:4000]
    reg2["updated_at"] = ts
    _write_json(reg_path, reg2)
    return {
        "restored": True,
        "archive_path": src_norm,
        "restore_root": selected_restore_root.as_posix() if selected_restore_root is not None else "",
        "adapter_path": adapter_path,
        "artifact_id": str(node.get("id", "")).strip(),
        "activated": bool(activate),
        "base_model": base_model,
        "base_model_match": bool(_is_base_model_match(base_model, pref_base)),
        "archive_sha256": selected_expected_hash or selected_actual_hash,
        "archive_hash_verified": bool(verify_archive_hash and bool(selected_expected_hash)),
        "archive_signature": selected_expected_sig or selected_actual_sig,
        "archive_signature_verified": bool(selected_sig_verified),
        "archive_signature_required": bool(require_archive_signature),
        "attempted_candidates": int(len(candidate_failures) + 1),
        "failed_candidates": candidate_failures[:8],
    }


def audit_adapter_artifacts(
    *,
    registry_path: str,
    refresh_stats: bool = True,
    fix: bool = False,
    drop_missing: bool = False,
    drop_invalid: bool = False,
    drop_missing_archives: bool = False,
    verify_archive_hash: bool = False,
    verify_archive_signature: bool = False,
    require_archive_signature: bool = False,
    sign_missing_archives: bool = False,
    archive_signing_key: str = "",
    archive_signing_key_id: str = "",
    archive_signing_keys_json: str = "",
    now: float | None = None,
) -> Dict[str, Any]:
    ts = float(now if now is not None else time.time())
    reg_path = Path(_normalize_path(registry_path) or registry_path)
    reg = _load_registry(reg_path)
    raw_rows = [dict(x) for x in reg.get("artifacts", []) if isinstance(x, dict)]
    raw_archives = [dict(x) for x in reg.get("archives", []) if isinstance(x, dict)]

    missing_artifacts = 0
    invalid_artifacts = 0
    duplicate_artifacts = 0
    dropped_missing = 0
    dropped_invalid = 0
    artifact_map: Dict[str, Dict[str, Any]] = {}

    for idx, row in enumerate(raw_rows):
        node = dict(row)
        rid = str(node.get("id", "")).strip()
        ap = _normalize_path(str(node.get("adapter_path", "")).strip())
        if not rid:
            rid = _artifact_id(
                ap or f"row_{idx}",
                _safe_int(node.get("size_bytes", 0), 0),
                _safe_float(node.get("modified_at", 0.0), 0.0),
            )
        node["id"] = rid
        node["adapter_path"] = ap
        rp = ""
        if ap:
            try:
                rp = Path(ap).resolve().as_posix()
            except Exception:
                rp = ap
        node["real_path"] = rp
        if str(node.get("base_model", "")).strip():
            node["base_model_norm"] = _normalize_model_id(str(node.get("base_model", "")))

        exists = bool(ap and Path(ap).exists())
        valid, valid_reason = _adapter_validity(Path(ap)) if ap else (False, "empty_path")
        if not exists:
            missing_artifacts += 1
        if not valid:
            invalid_artifacts += 1

        if refresh_stats:
            stats = _dir_stats(Path(ap)) if ap else {"exists": False, "size_bytes": 0, "file_count": 0, "modified_at": 0.0}
            node["exists"] = bool(stats.get("exists", False))
            node["size_bytes"] = int(stats.get("size_bytes", 0))
            node["file_count"] = int(stats.get("file_count", 0))
            node["modified_at"] = float(stats.get("modified_at", 0.0))
            node["valid"] = bool(valid)
            node["valid_reason"] = str(valid_reason)

        if fix and bool(drop_missing) and (not exists):
            dropped_missing += 1
            continue
        if fix and bool(drop_invalid) and (not valid):
            dropped_invalid += 1
            continue

        key = ap or f"id:{rid}"
        prev = artifact_map.get(key)
        if prev is None:
            artifact_map[key] = node
            continue
        duplicate_artifacts += 1
        keep_new = _safe_float(node.get("updated_at", 0.0), 0.0) >= _safe_float(prev.get("updated_at", 0.0), 0.0)
        if keep_new:
            artifact_map[key] = node

    audited_rows = list(artifact_map.values())
    audited_rows.sort(
        key=lambda x: (
            _artifact_rank_tuple(x),
            _safe_float(x.get("last_used_at", 0.0), 0.0),
            _safe_float(x.get("updated_at", 0.0), 0.0),
        ),
        reverse=True,
    )

    archive_missing = 0
    archive_duplicates = 0
    dropped_missing_archive_rows = 0
    archive_hash_verified = 0
    archive_hash_mismatch = 0
    archive_hash_compute_failed = 0
    archive_signature_verified = 0
    archive_signature_missing = 0
    archive_signature_key_missing = 0
    archive_signature_mismatch = 0
    archive_signed_missing = 0
    archive_map: Dict[str, Dict[str, Any]] = {}
    sign_key = _resolve_archive_signing_key(archive_signing_key)
    sign_key_id = _resolve_archive_signing_key_id(archive_signing_key_id)
    for row in raw_archives:
        node = dict(row)
        ap = _normalize_path(str(node.get("archive_path", "")).strip())
        if not ap:
            continue
        exists = bool(Path(ap).is_file())
        if not exists:
            archive_missing += 1
            if fix and bool(drop_missing_archives):
                dropped_missing_archive_rows += 1
                continue
        node["archive_path"] = ap
        node["restore_count"] = _safe_int(node.get("restore_count", 0), 0)
        node["exists"] = exists

        need_hash = bool(verify_archive_hash or verify_archive_signature or sign_missing_archives)
        actual_hash = ""
        if exists and need_hash:
            try:
                actual_hash = _file_sha256(Path(ap)).lower()
            except Exception:
                archive_hash_compute_failed += 1
                actual_hash = ""

        expected_hash = str(node.get("archive_sha256", "")).strip().lower()
        if verify_archive_hash and expected_hash and actual_hash:
            if expected_hash == actual_hash:
                archive_hash_verified += 1
            else:
                archive_hash_mismatch += 1

        expected_sig = str(node.get("archive_hmac_sha256", "")).strip().lower()
        if verify_archive_signature:
            if not expected_sig:
                if require_archive_signature:
                    archive_signature_missing += 1
            elif not actual_hash:
                archive_hash_compute_failed += 1
            else:
                sig_key = _resolve_archive_signing_key_for_row(
                    explicit_key=archive_signing_key,
                    row=node,
                    signing_keys_json=archive_signing_keys_json,
                )
                if not sig_key:
                    archive_signature_key_missing += 1
                else:
                    sig_artifact_id = str(node.get("artifact_id", "")).strip()
                    sig_base_model_norm = str(node.get("base_model_norm", "")).strip() or _normalize_model_id(
                        str(node.get("base_model", ""))
                    )
                    sig_size = _safe_int(node.get("archive_size_bytes", 0), 0)
                    if sig_size <= 0:
                        try:
                            sig_size = int(Path(ap).stat().st_size)
                        except Exception:
                            sig_size = 0
                    actual_sig = _archive_hmac_sha256(
                        archive_sha256=actual_hash,
                        artifact_id=sig_artifact_id,
                        base_model_norm=sig_base_model_norm,
                        archive_size_bytes=sig_size,
                        signing_key=sig_key,
                    ).lower()
                    if actual_sig == expected_sig:
                        archive_signature_verified += 1
                    else:
                        archive_signature_mismatch += 1

        if bool(fix and sign_missing_archives and exists and actual_hash and sign_key and (not expected_sig)):
            sig_artifact_id = str(node.get("artifact_id", "")).strip()
            sig_base_model_norm = str(node.get("base_model_norm", "")).strip() or _normalize_model_id(
                str(node.get("base_model", ""))
            )
            sig_size = _safe_int(node.get("archive_size_bytes", 0), 0)
            if sig_size <= 0:
                try:
                    sig_size = int(Path(ap).stat().st_size)
                except Exception:
                    sig_size = 0
            new_sig = _archive_hmac_sha256(
                archive_sha256=actual_hash,
                artifact_id=sig_artifact_id,
                base_model_norm=sig_base_model_norm,
                archive_size_bytes=sig_size,
                signing_key=sign_key,
            ).lower()
            node["archive_sha256"] = actual_hash
            node["archive_size_bytes"] = int(sig_size)
            node["archive_hmac_sha256"] = new_sig
            node["archive_sig_algo"] = "hmac-sha256-v1"
            node["archive_signed"] = True
            node["archive_signed_at"] = ts
            node["archive_signing_key_id"] = sign_key_id
            archive_signed_missing += 1

        prev = archive_map.get(ap)
        if prev is None:
            archive_map[ap] = node
            continue
        archive_duplicates += 1
        keep_new = max(
            _safe_float(node.get("archived_at", 0.0), 0.0),
            _safe_float(node.get("last_restored_at", 0.0), 0.0),
        ) >= max(
            _safe_float(prev.get("archived_at", 0.0), 0.0),
            _safe_float(prev.get("last_restored_at", 0.0), 0.0),
        )
        if keep_new:
            archive_map[ap] = node

    audited_archives = sorted(
        archive_map.values(),
        key=lambda x: max(
            _safe_float(x.get("archived_at", 0.0), 0.0),
            _safe_float(x.get("last_restored_at", 0.0), 0.0),
        ),
        reverse=True,
    )[:4000]

    active_id = str(reg.get("active_id", "")).strip()
    if active_id and all(str(x.get("id", "")).strip() != active_id for x in audited_rows):
        active_id = ""

    wrote = False
    if bool(fix):
        reg["artifacts"] = audited_rows
        reg["archives"] = audited_archives
        reg["active_id"] = active_id
        reg["updated_at"] = ts
        _write_json(reg_path, reg)
        wrote = True

    return {
        "registry_path": reg_path.as_posix(),
        "total_artifacts": int(len(raw_rows)),
        "remaining_artifacts": int(len(audited_rows)),
        "missing_artifacts": int(missing_artifacts),
        "invalid_artifacts": int(invalid_artifacts),
        "duplicate_artifacts": int(duplicate_artifacts),
        "dropped_missing": int(dropped_missing),
        "dropped_invalid": int(dropped_invalid),
        "total_archives": int(len(raw_archives)),
        "remaining_archives": int(len(audited_archives)),
        "missing_archives": int(archive_missing),
        "duplicate_archives": int(archive_duplicates),
        "dropped_missing_archives": int(dropped_missing_archive_rows),
        "archive_hash_verified": int(archive_hash_verified),
        "archive_hash_mismatch": int(archive_hash_mismatch),
        "archive_hash_compute_failed": int(archive_hash_compute_failed),
        "archive_signature_verified": int(archive_signature_verified),
        "archive_signature_missing": int(archive_signature_missing),
        "archive_signature_key_missing": int(archive_signature_key_missing),
        "archive_signature_mismatch": int(archive_signature_mismatch),
        "archive_signed_missing": int(archive_signed_missing),
        "active_id": active_id,
        "fixed": bool(fix),
        "wrote": bool(wrote),
    }


def maintain_adapter_artifacts(
    *,
    registry_path: str,
    active_registry_path: str = "artifacts/checkpoints/active_adapter.json",
    preferred_base_model: str = "",
    strict_base_model: bool = False,
    audit_fix: bool = True,
    audit_drop_missing: bool = True,
    audit_drop_invalid: bool = False,
    audit_drop_missing_archives: bool = True,
    prune: bool = False,
    keep_latest: int = 12,
    keep_promoted: int = 4,
    max_total_bytes: int = 0,
    prune_remove_invalid: bool = False,
    prune_delete_files: bool = False,
    prune_archive_before_delete: bool = False,
    prune_archive_dir: str = "artifacts/checkpoints/adapter_archive",
    prune_archive_signing_key: str = "",
    prune_archive_signing_key_id: str = "",
    audit_verify_archive_hash: bool = False,
    audit_verify_archive_signature: bool = False,
    audit_require_archive_signature: bool = False,
    audit_sign_missing_archives: bool = False,
    audit_archive_signing_key: str = "",
    audit_archive_signing_key_id: str = "",
    audit_archive_signing_keys_json: str = "",
    auto_restore_missing: bool = False,
    restore_output_dir: str = "artifacts/checkpoints/restored_adapters",
    restore_activate: bool = True,
    restore_verify_archive_hash: bool = True,
    restore_verify_archive_signature: bool = True,
    restore_require_archive_signature: bool = False,
    restore_archive_signing_key: str = "",
    restore_archive_signing_keys_json: str = "",
    sync_active_registry: bool = True,
    now: float | None = None,
) -> Dict[str, Any]:
    ts = float(now if now is not None else time.time())
    reg_path = _normalize_path(registry_path) or registry_path
    active_path = _normalize_path(active_registry_path) or active_registry_path
    base_model = str(preferred_base_model or "").strip()

    audit_stats = audit_adapter_artifacts(
        registry_path=reg_path,
        refresh_stats=True,
        fix=bool(audit_fix),
        drop_missing=bool(audit_drop_missing),
        drop_invalid=bool(audit_drop_invalid),
        drop_missing_archives=bool(audit_drop_missing_archives),
        verify_archive_hash=bool(audit_verify_archive_hash),
        verify_archive_signature=bool(audit_verify_archive_signature),
        require_archive_signature=bool(audit_require_archive_signature),
        sign_missing_archives=bool(audit_sign_missing_archives),
        archive_signing_key=str(audit_archive_signing_key or "").strip(),
        archive_signing_key_id=str(audit_archive_signing_key_id or "").strip(),
        archive_signing_keys_json=str(audit_archive_signing_keys_json or "").strip(),
        now=ts,
    )

    prune_stats: Dict[str, Any] = {}
    if bool(prune):
        prune_stats = prune_adapter_artifacts(
            registry_path=reg_path,
            keep_latest=max(0, int(keep_latest)),
            keep_promoted=max(0, int(keep_promoted)),
            max_total_bytes=max(0, int(max_total_bytes)),
            protected_paths=[],
            remove_missing=True,
            remove_invalid=bool(prune_remove_invalid),
            delete_files=bool(prune_delete_files),
            archive_before_delete=bool(prune_archive_before_delete),
            archive_dir=str(prune_archive_dir or "").strip(),
            archive_signing_key=str(prune_archive_signing_key or "").strip(),
            archive_signing_key_id=str(prune_archive_signing_key_id or "").strip(),
            now=ts,
        )

    resolved = resolve_best_adapter_path(
        active_registry_path=active_path,
        artifact_registry_path=reg_path,
        preferred_base_model=base_model,
        strict_base_model=bool(strict_base_model),
        require_valid=True,
    )
    restore_stats: Dict[str, Any] = {}
    if (not resolved) and bool(auto_restore_missing):
        restore_stats = restore_archived_artifact(
            registry_path=reg_path,
            preferred_base_model=base_model,
            strict_base_model=bool(strict_base_model),
            output_dir=str(restore_output_dir or "").strip() or "artifacts/checkpoints/restored_adapters",
            activate=bool(restore_activate),
            verify_archive_hash=bool(restore_verify_archive_hash),
            verify_archive_signature=bool(restore_verify_archive_signature),
            require_archive_signature=bool(restore_require_archive_signature),
            archive_signing_key=str(restore_archive_signing_key or "").strip(),
            archive_signing_keys_json=str(restore_archive_signing_keys_json or "").strip(),
            now=ts,
        )
        if bool(restore_stats.get("restored", False)):
            resolved = _normalize_path(str(restore_stats.get("adapter_path", "")).strip())

    mark_used_ok = False
    if resolved:
        try:
            mark_adapter_used(
                registry_path=reg_path,
                adapter_path=resolved,
                source="artifact_maintain",
                base_model=base_model,
                now=ts,
            )
            mark_used_ok = True
        except Exception:
            mark_used_ok = False

    active_sync_ok = False
    if resolved and bool(sync_active_registry):
        try:
            active_payload = _read_json(Path(active_path))
            prev = _normalize_path(str(active_payload.get("active_adapter", "")).strip())
            if prev and prev != resolved:
                active_payload["previous_adapter"] = prev
            active_payload["active_adapter"] = resolved
            if base_model:
                active_payload["base_model"] = base_model
            active_payload["updated_at"] = ts
            _write_json(Path(active_path), active_payload)
            active_sync_ok = True
        except Exception:
            active_sync_ok = False

    healthy = False
    if resolved:
        ok, _reason = _adapter_validity(Path(resolved))
        healthy = bool(ok)

    return {
        "registry_path": reg_path,
        "active_registry_path": active_path,
        "preferred_base_model": base_model,
        "strict_base_model": bool(strict_base_model),
        "audit": audit_stats,
        "prune": prune_stats,
        "restore": restore_stats,
        "resolved_adapter_path": str(resolved or ""),
        "healthy": bool(healthy),
        "mark_used_ok": bool(mark_used_ok),
        "active_registry_synced": bool(active_sync_ok),
        "timestamp": ts,
    }
