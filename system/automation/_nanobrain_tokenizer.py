from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from system.automation._nanobrain_config import NanoBrainConfig


logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+|[一-鿿]+")
CJK_RE = re.compile(r"[一-鿿]+")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")


def _load_config(path: Path) -> NanoBrainConfig:
    if not path.exists():
        return NanoBrainConfig()
    try:
        data = json.loads(_read_text(path))
    except Exception:
        try:
            import yaml

            data = yaml.safe_load(_read_text(path)) or {}
        except Exception:
            data = {}
    cfg = NanoBrainConfig()
    payload = dict(data or {})
    policies = payload.get("policies") if isinstance(payload.get("policies"), dict) else {}
    default_policy = payload.get("default_policy", "")
    for k, v in payload.items():
        if k in {"policies", "default_policy"}:
            continue
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                continue
    policy_name = os.environ.get("AGENT_POLICY", "").strip() or str(default_policy or "").strip()
    if policy_name and policy_name in policies and isinstance(policies[policy_name], dict):
        for k, v in policies[policy_name].items():
            if hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    continue
    try:
        from config.policy_manager import get_nanobrain_overrides

        overrides = get_nanobrain_overrides()
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    try:
                        setattr(cfg, k, v)
                    except Exception:
                        continue
    except Exception:
        pass
    env_map = {
        "NANOBRAIN_LOGIC_WEIGHT": "logic_weight",
        "NANOBRAIN_EMOTION_WEIGHT": "emotion_weight",
        "NANOBRAIN_CROSS_WEIGHT": "cross_weight",
        "NANOBRAIN_PMI_BETA": "pmi_beta",
        "NANOBRAIN_THRESHOLD": "threshold",
        "NANOBRAIN_TAU": "tau",
        "NANOBRAIN_DECAY": "decay",
        "NANOBRAIN_FLUIDITY": "fluidity",
        "NANOBRAIN_GRAVITY": "gravity",
        "NANOBRAIN_BASE_GAIN": "base_gain",
    }
    for env_key, attr in env_map.items():
        val = os.environ.get(env_key)
        if val is None or val == "":
            continue
        try:
            num = float(val)
        except Exception:
            continue
        try:
            setattr(cfg, attr, num)
        except Exception:
            continue
    offset = os.environ.get("NANOBRAIN_EMPATHY", "")
    if offset != "":
        try:
            offset_val = float(offset)
        except Exception:
            offset_val = 0.0
        mode = os.environ.get("NANOBRAIN_EMPATHY_MODE", "offset").strip().lower()
        if mode == "absolute":
            cfg.empathy = max(0.0, min(1.0, float(offset_val)))
        else:
            cfg.empathy = max(0.0, min(1.0, float(cfg.empathy) + offset_val))
    if not cfg.critical_keywords:
        cfg.critical_keywords = ["critical", "risk", "danger", "high", "urgent"]
    return cfg


def _load_templates(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {
            "action": ["Suggested actions: {items}"],
            "risk": ["Risk signals: {items}"],
            "target": ["Targets: {items}"],
            "concept": ["Related concepts: {items}"],
            "path": ["Association path: {path}"],
            "clarify": [
                "I need a clearer goal or constraints (output, path, success criteria).",
            ],
        }
    try:
        data = json.loads(_read_text(path))
    except Exception:
        try:
            import yaml

            data = yaml.safe_load(_read_text(path)) or {}
        except Exception:
            data = {}
    out: Dict[str, Any] = {}
    for key, val in (data or {}).items():
        if isinstance(val, list):
            out[key] = [str(x) for x in val if str(x).strip()]
        elif isinstance(val, dict):
            cleaned: Dict[str, List[str]] = {}
            for lvl, items in val.items():
                if isinstance(items, list):
                    cleaned[str(lvl)] = [str(x) for x in items if str(x).strip()]
            if cleaned:
                out[key] = cleaned
    return out


def _try_jieba(text: str) -> Optional[List[str]]:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            import jieba

        return [t.strip() for t in jieba.cut(text) if t.strip()]
    except Exception:
        return None


def _split_cjk(chunk: str) -> List[str]:
    if len(chunk) <= 3:
        return [chunk]
    out = [chunk]
    for i in range(len(chunk) - 1):
        out.append(chunk[i : i + 2])
    return out


def tokenize(text: str, max_tokens: int = 128) -> List[str]:
    if not text:
        return []
    jieba_tokens = _try_jieba(text)
    tokens: List[str] = []
    if jieba_tokens:
        tokens.extend(jieba_tokens)
    else:
        for chunk in TOKEN_RE.findall(text):
            if not chunk:
                continue
            if CJK_RE.fullmatch(chunk):
                tokens.extend(_split_cjk(chunk))
            else:
                tokens.append(chunk)
    cleaned: List[str] = []
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if not CJK_RE.fullmatch(t):
            t = t.lower()
        if len(t) <= 1:
            continue
        cleaned.append(t)
        if len(cleaned) >= max_tokens:
            break
    return cleaned


def _hash_choice(items: List[str], seed: str) -> str:
    if not items:
        return ""
    try:
        import hashlib

        h = hashlib.md5(seed.encode("utf-8", errors="ignore")).hexdigest()
        idx = int(h[:8], 16) % len(items)
    except Exception:
        idx = 0
    return items[idx]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return items


def _doc_tokens(item: Dict[str, Any], max_tokens: int) -> List[str]:
    parts: List[str] = []
    for key in ("goal", "state_text"):
        val = item.get(key)
        if isinstance(val, str):
            parts.append(val)
    steps = item.get("plan_steps")
    if isinstance(steps, list):
        parts.append(" ".join([str(s) for s in steps if s]))
    errs = item.get("error_types") or []
    if isinstance(errs, list):
        parts.append(" ".join([str(e) for e in errs if e]))
    return tokenize(" ".join(parts), max_tokens=max_tokens)


def _extract_action_tokens(item: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    steps = item.get("plan_steps") or []
    for s in steps:
        if not isinstance(s, str):
            continue
        head = s.strip().split(" ", 1)[0].strip().lower()
        if head:
            out.append(head)
    return out


def _extract_path_tokens(item: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for key in ("missing_paths", "perm_paths"):
        vals = item.get("before_state", {}).get(key) or []
        if isinstance(vals, list):
            paths.extend([str(v) for v in vals if v])
    steps = item.get("plan_steps") or []
    for s in steps:
        if not isinstance(s, str):
            continue
        for token in s.split():
            if "/" in token or "\\" in token:
                paths.append(token)
    return paths


def _is_path_token(token: str) -> bool:
    return "/" in token or "\\" in token or "." in token
