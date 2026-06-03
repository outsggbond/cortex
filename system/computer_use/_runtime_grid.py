# -*- coding: utf-8 -*-
"""Private grid/OCR/image utility functions for the computer-use runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from system.computer_use._runtime_types import (
    BLOCK_ID_CHARS,
    _RESUME_COMPUTER_USE_GOALS,
)


_EXPLICIT_LAUNCHABLE_APPS = {
    "qq": ("qq", "QQ"),
    "notepad": ("notepad", "Notepad", "记事本"),
}


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _truncate(text: str, limit: int = 240) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3].rstrip() + "..."


def _round_ratio(numerator: int, denominator: int) -> float:
    base = max(0, int(denominator))
    if base <= 0:
        return 0.0
    return round(float(max(0, int(numerator))) / float(base), 4)


def _load_json_dict(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _is_resume_request(text: str) -> bool:
    return _clean_str(text).lower() in _RESUME_COMPUTER_USE_GOALS


def _match_pattern(text: str, patterns: Sequence[str]) -> str:
    low = _clean_str(text).lower()
    if not low:
        return ""
    for pattern in list(patterns or []):
        probe = _clean_str(pattern).lower()
        if probe and probe in low:
            return pattern
    return ""


def _normalize_hotkey(keys: str) -> str:
    return _clean_str(keys).lower().replace(" ", "")


def _hint_variants(text: str) -> List[str]:
    raw = _clean_str(text).strip("\"'").replace("\\", "/").rstrip("/")
    if not raw:
        return []
    variants: List[str] = []
    for candidate in (raw, raw.split("/")[-1]):
        probe = _clean_str(candidate).lower()
        if not probe or probe in variants:
            continue
        variants.append(probe)
        if probe.endswith(".exe"):
            trimmed = probe[: -len(".exe")].strip()
            if trimmed and trimmed not in variants:
                variants.append(trimmed)
    return variants


def _looks_like_launch_text(text: str) -> bool:
    raw = _clean_str(text)
    if not raw or any(ch in raw for ch in "\r\n\t"):
        return False
    if len(raw) > 120:
        return False
    parts = [part for part in raw.replace("\\", "/").split("/") if _clean_str(part)]
    tail = parts[-1] if parts else raw
    token = tail.lower()
    if token.endswith(".exe"):
        token = token[: -len(".exe")]
    if not token:
        return False
    return all(ch.isalnum() or ch in {"_", "-", ".", " "} for ch in raw)


def _goal_launch_hints(goal: str) -> List[str]:
    low = _clean_str(goal).lower()
    hints: List[str] = []
    for variants in _EXPLICIT_LAUNCHABLE_APPS.values():
        probes = [_clean_str(item) for item in variants if _clean_str(item)]
        if any(probe.lower() in low for probe in probes):
            for item in probes:
                if item not in hints:
                    hints.append(item)
    return hints


def _resolve_path(project_root: Path, path_text: str) -> Path:
    path = Path(str(path_text or "").strip()).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_sha1(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _column_label(index: int) -> str:
    value = max(0, int(index))
    out = ""
    while True:
        value, rem = divmod(value, 26)
        out = BLOCK_ID_CHARS[rem] + out
        if value == 0:
            return out
        value -= 1


def _block_id(row_index: int, col_index: int) -> str:
    return f"{_column_label(col_index)}{int(row_index) + 1}"


def _build_grid(width: int, height: int, *, rows: int, cols: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    safe_rows = max(1, int(rows))
    safe_cols = max(1, int(cols))
    for row in range(safe_rows):
        top = int(round((height * row) / safe_rows))
        bottom = int(round((height * (row + 1)) / safe_rows))
        for col in range(safe_cols):
            left = int(round((width * col) / safe_cols))
            right = int(round((width * (col + 1)) / safe_cols))
            out.append(
                {
                    "id": _block_id(row, col),
                    "row": row + 1,
                    "col": col + 1,
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "center_x": int(round((left + right) / 2.0)),
                    "center_y": int(round((top + bottom) / 2.0)),
                }
            )
    return out


def _find_grid_cell(cells: Sequence[Dict[str, Any]], block_id: str) -> Dict[str, Any] | None:
    target = _clean_str(block_id).upper()
    if not target:
        return None
    for cell in list(cells or []):
        if _clean_str((cell or {}).get("id")).upper() == target:
            return dict(cell)
    return None


def _map_ocr_to_grid(items: Sequence[Dict[str, Any]], cells: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in list(items or [])[: max(1, int(limit))]:
        left = int(item.get("left", 0) or 0)
        top = int(item.get("top", 0) or 0)
        width = int(item.get("width", 0) or 0)
        height = int(item.get("height", 0) or 0)
        center_x = left + max(1, width) // 2
        center_y = top + max(1, height) // 2
        block = ""
        for cell in list(cells or []):
            if (
                int(cell.get("left", 0) or 0) <= center_x < int(cell.get("right", 0) or 0)
                and int(cell.get("top", 0) or 0) <= center_y < int(cell.get("bottom", 0) or 0)
            ):
                block = str(cell.get("id", "") or "")
                break
        out.append(
            {
                "text": _clean_str((item or {}).get("text")),
                "conf": float(item.get("conf", -1.0) or -1.0),
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "block": block,
            }
        )
    return out


def _nonempty_blocks(ocr_items: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[str]] = {}
    for item in list(ocr_items or []):
        block = _clean_str((item or {}).get("block"))
        text = _clean_str((item or {}).get("text"))
        if not block or not text:
            continue
        grouped.setdefault(block, [])
        if text not in grouped[block]:
            grouped[block].append(text)
    rows: List[Dict[str, Any]] = []
    for block in sorted(grouped):
        texts = grouped.get(block) or []
        rows.append({"block": block, "text": " ".join(texts[:4]), "item_count": len(texts)})
    return rows[: max(1, int(limit))]


def _extract_json_object(text: str) -> str:
    raw = str(text or "").strip()
    start = raw.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(raw)):
        char = raw[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : idx + 1]
    return ""


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    image = Image.open(path.as_posix())
    image.load()
    return int(image.width or 0), int(image.height or 0)


def _render_grid_overlay(
    source_path: Path,
    dest_path: Path,
    *,
    cells: Sequence[Dict[str, Any]],
    ocr_items: Sequence[Dict[str, Any]],
) -> str:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(source_path.as_posix()).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    for cell in list(cells or []):
        left = int(cell.get("left", 0) or 0)
        top = int(cell.get("top", 0) or 0)
        right = int(cell.get("right", 0) or 0)
        bottom = int(cell.get("bottom", 0) or 0)
        label = _clean_str(cell.get("id"))
        draw.rectangle([left, top, right, bottom], outline=(40, 210, 255, 180), width=1)
        if label:
            draw.rectangle([left + 2, top + 2, left + 34, top + 16], fill=(0, 0, 0, 170))
            draw.text((left + 5, top + 4), label, fill=(255, 255, 255, 255), font=font)
    for item in list(ocr_items or [])[:80]:
        left = int(item.get("left", 0) or 0)
        top = int(item.get("top", 0) or 0)
        width = int(item.get("width", 0) or 0)
        height = int(item.get("height", 0) or 0)
        text = _clean_str(item.get("text"))
        if width <= 0 or height <= 0:
            continue
        draw.rectangle([left, top, left + width, top + height], outline=(255, 184, 77, 220), width=2)
        if text:
            draw.rectangle([left, max(0, top - 14), min(image.width, left + 120), top], fill=(255, 184, 77, 180))
            draw.text((left + 2, max(0, top - 12)), _truncate(text, 14), fill=(20, 20, 20, 255), font=font)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest_path.as_posix())
    return dest_path.as_posix()
