from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"\s+")

_PROMPT_KEYS: Tuple[str, ...] = (
    "prompt",
    "query",
    "instruction",
    "input",
    "user",
    "question",
)

_RESPONSE_KEYS: Tuple[str, ...] = (
    "response",
    "answer",
    "output",
    "assistant",
    "target",
)


def _try_ftfy():
    try:
        import ftfy  # type: ignore

        return ftfy
    except Exception:
        return None


def _try_langdetect():
    try:
        from langdetect import detect as _detect  # type: ignore

        return _detect
    except Exception:
        return None


def parse_allowed_languages(raw: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(raw, str):
        src = [x.strip().lower() for x in str(raw).replace(";", ",").split(",")]
    else:
        src = [str(x).strip().lower() for x in raw]
    out: List[str] = []
    seen: set[str] = set()
    for item in src:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _encoding_candidates() -> Tuple[str, ...]:
    return ("utf-8", "utf-8-sig", "gb18030", "gbk", "big5", "latin-1")


def _decode_text(raw: bytes) -> Tuple[str, str]:
    for enc in _encoding_candidates():
        try:
            return raw.decode(enc), enc
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def _norm_space(text: str) -> str:
    t = _CTRL_RE.sub(" ", str(text or ""))
    t = t.replace("\ufeff", " ")
    t = t.replace("\ufffd", " ")
    t = _WS_RE.sub(" ", t)
    return t.strip()


def _normalize_text(text: str, *, use_ftfy: bool, ftfy_mod: Any) -> Tuple[str, bool]:
    src = str(text or "")
    cur = src
    changed = False
    if bool(use_ftfy) and ftfy_mod is not None:
        try:
            fixed = str(ftfy_mod.fix_text(cur))
            if fixed != cur:
                cur = fixed
                changed = True
        except Exception:
            pass
    compact = _norm_space(cur)
    if compact != cur:
        changed = True
    return compact, changed


def _max_repeat_run(text: str) -> int:
    s = str(text or "")
    if not s:
        return 0
    best = 1
    run = 1
    prev = s[0]
    for ch in s[1:]:
        if ch == prev:
            run += 1
            if run > best:
                best = run
        else:
            prev = ch
            run = 1
    return int(best)


def _symbol_ratio(text: str) -> float:
    total = 0
    symbol = 0
    for ch in str(text or ""):
        if ch.isspace():
            continue
        total += 1
        if ch.isalnum():
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            continue
        symbol += 1
    if total <= 0:
        return 1.0
    return float(symbol) / float(total)


def _pick_text(node: Dict[str, Any], keys: Iterable[str]) -> Tuple[str, str]:
    for k in keys:
        v = node.get(k)
        if isinstance(v, str):
            return str(k), v
    return "", ""


def _from_messages(node: Dict[str, Any]) -> Tuple[str, str]:
    msgs = node.get("messages")
    if not isinstance(msgs, list):
        return "", ""
    user_text = ""
    assistant_text = ""
    for item in msgs:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        txt = str(item.get("content", item.get("text", ""))).strip()
        if not txt:
            continue
        if role == "user":
            user_text = txt
        elif role == "assistant":
            assistant_text = txt
    return user_text, assistant_text


def _lang_allowed(lang: str, allowed: Tuple[str, ...]) -> bool:
    if not allowed:
        return True
    cur = str(lang or "").strip().lower()
    if not cur:
        return True
    for a in allowed:
        if cur == a or cur.startswith(a + "-"):
            return True
    return False


def _estimate_clean_quality(
    *,
    rows_total: int,
    rows_output: int,
    rows_dropped: Dict[str, int],
) -> float:
    total = max(0, int(rows_total))
    kept = max(0, int(rows_output))
    if total <= 0:
        return 0.0
    keep_ratio = float(kept) / float(total)

    dropped = dict(rows_dropped or {})
    invalid_ratio = float(max(0, int(dropped.get("invalid_json", 0)))) / float(total)
    missing_ratio = float(max(0, int(dropped.get("missing_fields", 0)))) / float(total)
    too_short_ratio = float(max(0, int(dropped.get("too_short", 0)))) / float(total)
    low_quality_ratio = float(max(0, int(dropped.get("low_quality", 0)))) / float(total)
    duplicate_ratio = float(max(0, int(dropped.get("duplicate", 0)))) / float(total)
    language_ratio = float(max(0, int(dropped.get("language", 0)))) / float(total)

    penalty = 0.0
    penalty += (1.0 - float(keep_ratio)) * 0.45
    penalty += float(low_quality_ratio) * 0.20
    penalty += float(too_short_ratio) * 0.10
    penalty += float(missing_ratio) * 0.10
    penalty += float(invalid_ratio) * 0.08
    penalty += float(duplicate_ratio) * 0.04
    penalty += float(language_ratio) * 0.03
    return max(0.0, min(1.0, 1.0 - float(penalty)))


@dataclass
class DataCleanConfig:
    min_prompt_chars: int = 2
    min_response_chars: int = 4
    min_line_chars: int = 2
    max_repeat_run: int = 14
    max_symbol_ratio: float = 0.60
    dedup: bool = True
    use_ftfy: bool = False
    allowed_languages: Tuple[str, ...] = ()
    keep_invalid_json: bool = False


def clean_jsonl_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cfg: Optional[DataCleanConfig] = None,
) -> Dict[str, Any]:
    conf = cfg or DataCleanConfig()
    src = Path(str(input_path))
    out = Path(str(output_path))
    raw = src.read_bytes()
    text, used_encoding = _decode_text(raw)
    ftfy_mod = _try_ftfy() if bool(conf.use_ftfy) else None
    detector = _try_langdetect() if bool(conf.allowed_languages) else None

    seen: set[str] = set()
    cleaned_rows: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = {
        "invalid_json": 0,
        "missing_fields": 0,
        "too_short": 0,
        "low_quality": 0,
        "language": 0,
        "duplicate": 0,
    }
    changed_text = 0
    rows_total = 0
    lang_skipped = 0

    for line in text.splitlines():
        line = str(line).strip()
        if not line:
            continue
        rows_total += 1
        item: Dict[str, Any]
        try:
            raw_node = json.loads(line)
            if not isinstance(raw_node, dict):
                raise ValueError("non_dict")
            item = dict(raw_node)
        except Exception:
            dropped["invalid_json"] += 1
            if not bool(conf.keep_invalid_json):
                continue
            item = {"prompt": line, "response": ""}

        pk, prompt_raw = _pick_text(item, _PROMPT_KEYS)
        rk, response_raw = _pick_text(item, _RESPONSE_KEYS)
        if (not prompt_raw) or (not response_raw):
            msg_user, msg_assistant = _from_messages(item)
            if not prompt_raw and msg_user:
                prompt_raw = msg_user
            if not response_raw and msg_assistant:
                response_raw = msg_assistant
        if not prompt_raw or not response_raw:
            dropped["missing_fields"] += 1
            continue

        prompt, prompt_changed = _normalize_text(prompt_raw, use_ftfy=bool(conf.use_ftfy), ftfy_mod=ftfy_mod)
        response, response_changed = _normalize_text(response_raw, use_ftfy=bool(conf.use_ftfy), ftfy_mod=ftfy_mod)
        if prompt_changed or response_changed:
            changed_text += 1
        if len(prompt) < max(1, int(conf.min_prompt_chars)) or len(response) < max(1, int(conf.min_response_chars)):
            dropped["too_short"] += 1
            continue

        merged = f"{prompt}\n{response}"
        if _max_repeat_run(merged) > int(max(1, conf.max_repeat_run)):
            dropped["low_quality"] += 1
            continue
        if _symbol_ratio(merged) > float(max(0.0, min(1.0, conf.max_symbol_ratio))):
            dropped["low_quality"] += 1
            continue
        if prompt == response and len(response) <= 64:
            dropped["low_quality"] += 1
            continue

        detected_lang = ""
        if conf.allowed_languages:
            if detector is None:
                lang_skipped += 1
            else:
                try:
                    detected_lang = str(detector(merged)).strip().lower()
                except Exception:
                    detected_lang = ""
                if (detected_lang and (not _lang_allowed(detected_lang, conf.allowed_languages))):
                    dropped["language"] += 1
                    continue

        digest = hashlib.sha256(f"{prompt}\n{response}".encode("utf-8", errors="ignore")).hexdigest()
        if bool(conf.dedup):
            if digest in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(digest)

        if not pk:
            pk = "prompt"
        if not rk:
            rk = "response"
        item[pk] = prompt
        item[rk] = response
        # Keep canonical fields for downstream scripts.
        item["prompt"] = prompt
        item["response"] = response
        if detected_lang:
            item["lang"] = detected_lang
        cleaned_rows.append(item)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in cleaned_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    keep_ratio = float(len(cleaned_rows)) / float(rows_total) if rows_total > 0 else 0.0
    quality_score = _estimate_clean_quality(
        rows_total=int(rows_total),
        rows_output=int(len(cleaned_rows)),
        rows_dropped=dropped,
    )
    return {
        "ok": True,
        "kind": "jsonl",
        "input_path": src.as_posix(),
        "output_path": out.as_posix(),
        "encoding_detected": str(used_encoding),
        "rows_total": int(rows_total),
        "rows_output": int(len(cleaned_rows)),
        "keep_ratio": float(keep_ratio),
        "quality_score": float(quality_score),
        "rows_dropped": dict(dropped),
        "normalized_rows": int(changed_text),
        "lang_filter": {
            "enabled": bool(conf.allowed_languages),
            "allowed": list(conf.allowed_languages),
            "detector_available": bool(detector is not None),
            "detect_skipped_rows": int(lang_skipped),
        },
        "config": {
            "min_prompt_chars": int(conf.min_prompt_chars),
            "min_response_chars": int(conf.min_response_chars),
            "max_repeat_run": int(conf.max_repeat_run),
            "max_symbol_ratio": float(conf.max_symbol_ratio),
            "dedup": bool(conf.dedup),
            "use_ftfy": bool(conf.use_ftfy),
        },
    }


def clean_text_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cfg: Optional[DataCleanConfig] = None,
) -> Dict[str, Any]:
    conf = cfg or DataCleanConfig()
    src = Path(str(input_path))
    out = Path(str(output_path))
    raw = src.read_bytes()
    text, used_encoding = _decode_text(raw)
    ftfy_mod = _try_ftfy() if bool(conf.use_ftfy) else None
    detector = _try_langdetect() if bool(conf.allowed_languages) else None

    seen: set[str] = set()
    kept: List[str] = []
    dropped = {
        "too_short": 0,
        "low_quality": 0,
        "language": 0,
        "duplicate": 0,
    }
    changed = 0
    total = 0
    lang_skipped = 0

    for line in text.splitlines():
        raw_line = str(line or "")
        if not raw_line.strip():
            continue
        total += 1
        cur, is_changed = _normalize_text(raw_line, use_ftfy=bool(conf.use_ftfy), ftfy_mod=ftfy_mod)
        if is_changed:
            changed += 1
        if len(cur) < max(1, int(conf.min_line_chars)):
            dropped["too_short"] += 1
            continue
        if _max_repeat_run(cur) > int(max(1, conf.max_repeat_run)):
            dropped["low_quality"] += 1
            continue
        if _symbol_ratio(cur) > float(max(0.0, min(1.0, conf.max_symbol_ratio))):
            dropped["low_quality"] += 1
            continue

        detected_lang = ""
        if conf.allowed_languages:
            if detector is None:
                lang_skipped += 1
            else:
                try:
                    detected_lang = str(detector(cur)).strip().lower()
                except Exception:
                    detected_lang = ""
                if detected_lang and (not _lang_allowed(detected_lang, conf.allowed_languages)):
                    dropped["language"] += 1
                    continue

        if bool(conf.dedup):
            digest = hashlib.sha256(cur.encode("utf-8", errors="ignore")).hexdigest()
            if digest in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(digest)
        kept.append(cur)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(("\n".join(kept) + ("\n" if kept else "")), encoding="utf-8")

    keep_ratio = float(len(kept)) / float(total) if total > 0 else 0.0
    quality_score = _estimate_clean_quality(
        rows_total=int(total),
        rows_output=int(len(kept)),
        rows_dropped=dropped,
    )
    return {
        "ok": True,
        "kind": "text",
        "input_path": src.as_posix(),
        "output_path": out.as_posix(),
        "encoding_detected": str(used_encoding),
        "rows_total": int(total),
        "rows_output": int(len(kept)),
        "keep_ratio": float(keep_ratio),
        "quality_score": float(quality_score),
        "rows_dropped": dict(dropped),
        "normalized_rows": int(changed),
        "lang_filter": {
            "enabled": bool(conf.allowed_languages),
            "allowed": list(conf.allowed_languages),
            "detector_available": bool(detector is not None),
            "detect_skipped_rows": int(lang_skipped),
        },
        "config": {
            "min_line_chars": int(conf.min_line_chars),
            "max_repeat_run": int(conf.max_repeat_run),
            "max_symbol_ratio": float(conf.max_symbol_ratio),
            "dedup": bool(conf.dedup),
            "use_ftfy": bool(conf.use_ftfy),
        },
    }


def clean_training_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    cfg: Optional[DataCleanConfig] = None,
) -> Dict[str, Any]:
    src = Path(str(input_path))
    suffix = str(src.suffix or "").strip().lower()
    if suffix in {".jsonl", ".json"}:
        return clean_jsonl_file(src, output_path, cfg=cfg)
    return clean_text_file(src, output_path, cfg=cfg)
