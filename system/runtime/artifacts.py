# -*- coding: utf-8 -*-
"""Artifact lifecycle helpers extracted from the legacy runtime entrypoint."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict

from system.evaluation.adapter_artifacts import (
    audit_adapter_artifacts,
    maintain_adapter_artifacts,
    prune_adapter_artifacts,
    restore_archived_artifact,
)
from system.runtime.support import env_flag

# 提取默认路径为常量
DEFAULT_REGISTRY = "artifacts/checkpoints/adapter_artifacts.json"
DEFAULT_RESTORE_DIR = "artifacts/checkpoints/restored_adapters"

def _str_arg(args: Any, name: str, default: str = "") -> str:
    val = getattr(args, name, default)
    return str(val).strip() if val is not None else default


def _bool_arg(args: Any, name: str, default: bool = False) -> bool:
    return bool(getattr(args, name, default))


def _int_arg(args: Any, name: str, default: int = 0) -> int:
    try:
        val = getattr(args, name, default)
        return int(val) if val not in (None, "") else int(default)
    except (ValueError, TypeError):
        return int(default)


def _float_arg(args: Any, name: str, default: float = 0.0) -> float:
    try:
        val = getattr(args, name, default)
        return float(val) if val not in (None, "") else float(default)
    except (ValueError, TypeError):
        return float(default)


def disk_free_gb(path: str) -> float:
    """获取指定路径剩余空间 (GB)，路径不存在则尝试父目录。"""
    p = Path(str(path or "").strip() or ".")
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        usage = shutil.disk_usage(p.as_posix())
        return usage.free / (1024 ** 3)
    except Exception:
        return -1.0


def run_artifact_prune(args: Any) -> Dict[str, Any]:
    reg_path = _str_arg(args, "artifact_registry") or DEFAULT_REGISTRY
    max_total_bytes = int(max(0.0, _float_arg(args, "artifact_max_total_gb")) * (1024 ** 3))
    keep_latest = max(0, _int_arg(args, "artifact_keep_latest", 12))
    keep_promoted = max(0, _int_arg(args, "artifact_keep_promoted", 4))
    
    stats = prune_adapter_artifacts(
        registry_path=reg_path,
        keep_latest=keep_latest,
        keep_promoted=keep_promoted,
        max_total_bytes=max_total_bytes,
        protected_paths=[],
        remove_missing=True,
        remove_invalid=_bool_arg(args, "artifact_prune_remove_invalid"),
        delete_files=_bool_arg(args, "artifact_prune_delete_files"),
        archive_before_delete=_bool_arg(args, "artifact_prune_archive"),
        archive_dir=_str_arg(args, "artifact_prune_archive_dir"),
        archive_signing_key_id=_str_arg(args, "artifact_archive_signing_key_id"),
    )
    
    return {
        "registry": reg_path,
        "keep_latest": keep_latest,
        "keep_promoted": keep_promoted,
        "max_total_gb": _float_arg(args, "artifact_max_total_gb"),
        "remove_invalid": _bool_arg(args, "artifact_prune_remove_invalid"),
        "archive_before_delete": _bool_arg(args, "artifact_prune_archive"),
        **stats,
    }


def run_artifact_audit(args: Any) -> Dict[str, Any]:
    reg_path = _str_arg(args, "artifact_registry") or DEFAULT_REGISTRY
    stats = audit_adapter_artifacts(
        registry_path=reg_path,
        refresh_stats=True,
        fix=_bool_arg(args, "artifact_audit_fix"),
        drop_missing=_bool_arg(args, "artifact_audit_drop_missing"),
        drop_invalid=_bool_arg(args, "artifact_audit_drop_invalid"),
        drop_missing_archives=_bool_arg(args, "artifact_audit_drop_missing_archives"),
        verify_archive_hash=_bool_arg(args, "artifact_audit_verify_hash"),
        verify_archive_signature=_bool_arg(args, "artifact_audit_verify_signature"),
        require_archive_signature=_bool_arg(args, "artifact_audit_require_signature"),
        sign_missing_archives=_bool_arg(args, "artifact_audit_sign_missing"),
        archive_signing_key_id=_str_arg(args, "artifact_archive_signing_key_id"),
        archive_signing_keys_json=_str_arg(args, "artifact_signing_keys_json"),
    )
    return {
        "registry": reg_path,
        "fix": _bool_arg(args, "artifact_audit_fix"),
        **stats,
    }


def run_artifact_maintain(args: Any, *, auto_mode: bool = False, force_prune: bool = False) -> Dict[str, Any]:
    reg_path = _str_arg(args, "artifact_registry") or DEFAULT_REGISTRY
    active_reg = os.environ.get("TRANSFORMER_ADAPTER_REGISTRY", "artifacts/checkpoints/active_adapter.json").strip()
    preferred_model = _str_arg(args, "transformer_model") or os.environ.get("TRANSFORMER_MODEL", "").strip()
    
    # 这里已修复：将变量名改回 strict_base_model
    strict_base_model = _bool_arg(args, "adapter_strict_base_model") or env_flag("TRANSFORMER_ADAPTER_STRICT_BASE_MODEL")
    max_total_bytes = int(max(0.0, _float_arg(args, "artifact_max_total_gb")) * (1024 ** 3))

    enable_fix = _bool_arg(args, "artifact_maintain_fix") or auto_mode
    enable_prune = _bool_arg(args, "artifact_maintain_prune") or force_prune
    enable_auto_restore = _bool_arg(args, "artifact_maintain_auto_restore") or auto_mode

    stats = maintain_adapter_artifacts(
        registry_path=reg_path,
        active_registry_path=active_reg,
        preferred_base_model=preferred_model,
        strict_base_model=bool(strict_base_model), # 报错位置已修正
        audit_fix=enable_fix,
        audit_drop_missing=_bool_arg(args, "artifact_audit_drop_missing") or enable_fix,
        audit_drop_invalid=_bool_arg(args, "artifact_audit_drop_invalid"),
        audit_drop_missing_archives=_bool_arg(args, "artifact_audit_drop_missing_archives") or enable_fix,
        audit_verify_archive_hash=_bool_arg(args, "artifact_audit_verify_hash"),
        audit_verify_archive_signature=_bool_arg(args, "artifact_audit_verify_signature"),
        audit_require_archive_signature=_bool_arg(args, "artifact_audit_require_signature"),
        audit_sign_missing_archives=_bool_arg(args, "artifact_audit_sign_missing"),
        audit_archive_signing_key_id=_str_arg(args, "artifact_archive_signing_key_id"),
        audit_archive_signing_keys_json=_str_arg(args, "artifact_signing_keys_json"),
        prune=enable_prune,
        keep_latest=max(0, _int_arg(args, "artifact_keep_latest", 12)),
        keep_promoted=max(0, _int_arg(args, "artifact_keep_promoted", 4)),
        max_total_bytes=max_total_bytes,
        prune_remove_invalid=_bool_arg(args, "artifact_prune_remove_invalid"),
        prune_delete_files=_bool_arg(args, "artifact_prune_delete_files"),
        prune_archive_before_delete=_bool_arg(args, "artifact_prune_archive"),
        prune_archive_dir=_str_arg(args, "artifact_prune_archive_dir"),
        prune_archive_signing_key_id=_str_arg(args, "artifact_archive_signing_key_id"),
        auto_restore_missing=enable_auto_restore,
        restore_output_dir=_str_arg(args, "artifact_auto_restore_output_dir") or DEFAULT_RESTORE_DIR,
        restore_activate=_bool_arg(args, "artifact_auto_restore_activate") or enable_auto_restore,
        restore_verify_archive_hash=not _bool_arg(args, "artifact_restore_no_hash_verify"),
        restore_verify_archive_signature=not _bool_arg(args, "artifact_restore_no_signature_verify"),
        restore_require_archive_signature=_bool_arg(args, "artifact_restore_require_signature"),
        restore_archive_signing_keys_json=_str_arg(args, "artifact_signing_keys_json"),
        sync_active_registry=True,
    )
    
    return {
        "registry": reg_path,
        "auto_mode": auto_mode,
        "force_prune": force_prune,
        "fix": enable_fix,
        "prune": enable_prune,
        "auto_restore": enable_auto_restore,
        **stats,
    }


def run_artifact_restore(args: Any) -> Dict[str, Any]:
    reg_path = _str_arg(args, "artifact_registry") or DEFAULT_REGISTRY
    preferred_restore_model = (
        _str_arg(args, "artifact_restore_preferred_base_model")
        or _str_arg(args, "transformer_model")
        or os.environ.get("TRANSFORMER_MODEL", "").strip()
    )
    
    stats = restore_archived_artifact(
        registry_path=reg_path,
        archive_path=_str_arg(args, "artifact_restore_archive_path"),
        artifact_id=_str_arg(args, "artifact_restore_artifact_id"),
        preferred_base_model=preferred_restore_model,
        strict_base_model=_bool_arg(args, "artifact_restore_strict_base_model"),
        output_dir=_str_arg(args, "artifact_restore_output_dir") or DEFAULT_RESTORE_DIR,
        activate=_bool_arg(args, "artifact_restore_activate"),
        verify_archive_hash=not _bool_arg(args, "artifact_restore_no_hash_verify"),
        verify_archive_signature=not _bool_arg(args, "artifact_restore_no_signature_verify"),
        require_archive_signature=_bool_arg(args, "artifact_restore_require_signature"),
        archive_signing_keys_json=_str_arg(args, "artifact_signing_keys_json"),
        overwrite=_bool_arg(args, "artifact_restore_overwrite"),
    )
    
    return {
        "registry": reg_path,
        "artifact_id": _str_arg(args, "artifact_restore_artifact_id"),
        "output_dir": _str_arg(args, "artifact_restore_output_dir") or DEFAULT_RESTORE_DIR,
        **stats,
    }
