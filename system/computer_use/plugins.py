# -*- coding: utf-8 -*-
"""Windows Desktop Automation Plugins

Real desktop automation via Win32 API (SendInput, keyboard, mouse),
with optional pywinauto UIA, Tesseract OCR, and Playwright browser support.

Architecture
────────────
Each plugin is an async callable ``(payload: dict) -> dict`` registered in the
global ``registry``.  Plugins gracefully degrade when optional dependencies are
missing — callers already handle ``{"ok": False, ...}`` responses via the
``required=False`` convention in ``computer_use/runtime.py``.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from ctypes import wintypes
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 1. Win32 API helpers
# ═══════════════════════════════════════════════════════════════════

# --- SendInput types ----------------------------------------------------------

INPUT_MOUSE    = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

MOUSEEVENTF_MOVE       = 0x0001
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOUSEEVENTF_RIGHTDOWN  = 0x0008
MOUSEEVENTF_RIGHTUP    = 0x0010
MOUSEEVENTF_ABSOLUTE   = 0x8000
MOUSEEVENTF_WHEEL      = 0x0800

KEYEVENTF_KEYUP        = 0x0002
KEYEVENTF_SCANCODE     = 0x0008
KEYEVENTF_UNICODE      = 0x0004

# Virtual-key codes for common modifiers
VK_SHIFT   = 0x10
VK_CONTROL = 0x11
VK_MENU    = 0x12  # Alt
VK_LWIN    = 0x5B
VK_RWIN    = 0x5C
VK_RETURN  = 0x0D
VK_TAB     = 0x09
VK_ESCAPE  = 0x1B
VK_BACK    = 0x08
VK_DELETE  = 0x2E
VK_SPACE   = 0x20

# Virtual-key → descriptive name
_VK_NAMES: Dict[int, str] = {
    VK_SHIFT: "shift", VK_CONTROL: "ctrl", VK_MENU: "alt",
    VK_LWIN: "win", VK_RWIN: "win",
    VK_RETURN: "enter", VK_TAB: "tab", VK_ESCAPE: "esc",
    VK_BACK: "backspace", VK_DELETE: "delete", VK_SPACE: "space",
}

# Hotkey string → (modifier_vk, key_vk_or_char)
_HOTKEY_MAP: Dict[str, tuple] = {}
for _mod, _vk in [("ctrl", VK_CONTROL), ("alt", VK_MENU), ("shift", VK_SHIFT), ("win", VK_LWIN)]:
    for _c in "abcdefghijklmnopqrstuvwxyz0123456789":
        _HOTKEY_MAP[f"{_mod}+{_c}"] = (_vk, ord(_c.upper()))
    for _f in range(1, 13):
        _HOTKEY_MAP[f"{_mod}+f{_f}"] = (_vk, 0x6F + _f)  # VK_F1 = 0x70
_HOTKEY_MAP.update({
    "ctrl+v": (VK_CONTROL, ord('V')), "ctrl+c": (VK_CONTROL, ord('C')),
    "ctrl+x": (VK_CONTROL, ord('X')), "ctrl+z": (VK_CONTROL, ord('Z')),
    "ctrl+a": (VK_CONTROL, ord('A')), "ctrl+s": (VK_CONTROL, ord('S')),
    "ctrl+f": (VK_CONTROL, ord('F')), "ctrl+w": (VK_CONTROL, ord('W')),
    "ctrl+t": (VK_CONTROL, ord('T')), "ctrl+n": (VK_CONTROL, ord('N')),
    "ctrl+o": (VK_CONTROL, ord('O')), "ctrl+p": (VK_CONTROL, ord('P')),
    "alt+tab": (VK_MENU, VK_TAB), "alt+f4": (VK_MENU, 0x73),
    "win+r": (VK_LWIN, ord('R')), "win+d": (VK_LWIN, ord('D')),
    "win+e": (VK_LWIN, ord('E')), "win+l": (VK_LWIN, ord('L')),
    "enter": (None, VK_RETURN), "escape": (None, VK_ESCAPE),
    "tab": (None, VK_TAB), "backspace": (None, VK_BACK),
    "delete": (None, VK_DELETE), "space": (None, VK_SPACE),
})


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx",          wintypes.LONG),
        ("dy",          wintypes.LONG),
        ("mouseData",   wintypes.DWORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",         wintypes.WORD),
        ("wScan",       wintypes.WORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUT_UNION),
    ]


# Win32 function signatures
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

user32.SendInput.restype = wintypes.UINT
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int]

user32.GetCursorPos.restype = wintypes.BOOL
user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]

user32.SetCursorPos.restype = wintypes.BOOL
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]

user32.GetForegroundWindow.restype = wintypes.HWND
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]

user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]

user32.SetForegroundWindow.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]

user32.EnumWindows.restype = wintypes.BOOL
user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM), wintypes.LPARAM]

user32.IsWindowVisible.restype = wintypes.BOOL
user32.IsWindowVisible.argtypes = [wintypes.HWND]

user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]

user32.GetSystemMetrics.restype = ctypes.c_int
user32.GetSystemMetrics.argtypes = [ctypes.c_int]

user32.MapVirtualKeyW.restype = wintypes.UINT
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]

user32.VkKeyScanW.restype = ctypes.c_short
user32.VkKeyScanW.argtypes = [wintypes.WCHAR]

SM_CXSCREEN = 0
SM_CYSCREEN = 1


def _send_input(*inputs: _INPUT) -> int:
    """Send one or more INPUT structs; return count successfully sent."""
    if not inputs:
        return 0
    arr = (_INPUT * len(inputs))(*inputs)
    return user32.SendInput(len(arr), arr, ctypes.sizeof(_INPUT))


def _mouse_input(flags: int, x: int = 0, y: int = 0, data: int = 0) -> _INPUT:
    inp = _INPUT()
    inp.type = INPUT_MOUSE
    inp.union.mi.dx = x
    inp.union.mi.dy = y
    inp.union.mi.mouseData = data
    inp.union.mi.dwFlags = flags
    inp.union.mi.time = 0
    inp.union.mi.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    return inp


def _keyboard_input(vk: int, flags: int = 0, scan: int = 0) -> _INPUT:
    inp = _INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki.wVk = vk
    inp.union.ki.wScan = scan
    inp.union.ki.dwFlags = flags
    inp.union.ki.time = 0
    inp.union.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    return inp


def _unicode_input(char: str) -> _INPUT:
    """Send a Unicode character via KEYEVENTF_UNICODE."""
    inp = _INPUT()
    inp.type = INPUT_KEYBOARD
    inp.union.ki.wVk = 0
    inp.union.ki.wScan = ord(char)
    inp.union.ki.dwFlags = KEYEVENTF_UNICODE
    inp.union.ki.time = 0
    inp.union.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
    return inp


# ═══════════════════════════════════════════════════════════════════
# 2. Keyboard
# ═══════════════════════════════════════════════════════════════════

def _press_key(vk: int) -> None:
    _send_input(_keyboard_input(vk, 0))


def _release_key(vk: int) -> None:
    _send_input(_keyboard_input(vk, KEYEVENTF_KEYUP))


def _tap_key(vk: int) -> None:
    _press_key(vk)
    time.sleep(0.02)
    _release_key(vk)


def _send_text_via_sendinput(text: str) -> int:
    """Type text using Unicode SendInput. Returns number of characters sent."""
    inputs: List[_INPUT] = []
    for ch in text:
        inputs.append(_unicode_input(ch))
        inputs.append(_keyboard_input(0, KEYEVENTF_KEYUP | KEYEVENTF_UNICODE, ord(ch)))
    if inputs:
        return _send_input(*inputs)
    return 0


def _send_hotkey(mod_vk: Optional[int], key_vk: int) -> None:
    """Press mod (if any) + key, then release in reverse order."""
    if mod_vk is not None:
        _press_key(mod_vk)
        time.sleep(0.01)
    _press_key(key_vk)
    time.sleep(0.02)
    _release_key(key_vk)
    time.sleep(0.01)
    if mod_vk is not None:
        _release_key(mod_vk)


def _parse_hotkey(hotkey_str: str) -> Optional[tuple]:
    """Parse a hotkey string like 'ctrl+c' → (mod_vk, key_vk)."""
    key = hotkey_str.strip().lower()
    if key in _HOTKEY_MAP:
        return _HOTKEY_MAP[key]
    # Try generic: modifier+char
    parts = key.split("+")
    if len(parts) == 2:
        mod_str, key_str = parts[0].strip(), parts[1].strip()
        mod_vk = {"ctrl": VK_CONTROL, "alt": VK_MENU, "shift": VK_SHIFT, "win": VK_LWIN}.get(mod_str)
        if mod_vk and len(key_str) == 1:
            return (mod_vk, ord(key_str.upper()))
    return None


# ═══════════════════════════════════════════════════════════════════
# 3. Mouse
# ═══════════════════════════════════════════════════════════════════

def _get_cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)


def _set_cursor_pos(x: int, y: int) -> bool:
    return bool(user32.SetCursorPos(x, y))


def _click_at(x: int, y: int, button: str = "left", clicks: int = 1) -> None:
    """Click at screen coordinates."""
    _set_cursor_pos(x, y)
    time.sleep(0.01)
    down_flag = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
    up_flag   = MOUSEEVENTF_LEFTUP   if button == "left" else MOUSEEVENTF_RIGHTUP
    for _ in range(max(1, clicks)):
        _send_input(_mouse_input(down_flag, 0, 0))
        time.sleep(0.02)
        _send_input(_mouse_input(up_flag, 0, 0))
        time.sleep(0.02)


def _double_click_at(x: int, y: int, button: str = "left") -> None:
    _click_at(x, y, button=button, clicks=2)


def _drag(x1: int, y1: int, x2: int, y2: int, button: str = "left") -> None:
    """Drag from (x1,y1) to (x2,y2)."""
    down_flag = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
    up_flag   = MOUSEEVENTF_LEFTUP   if button == "left" else MOUSEEVENTF_RIGHTUP
    _set_cursor_pos(x1, y1)
    time.sleep(0.01)
    _send_input(_mouse_input(down_flag, 0, 0))
    time.sleep(0.02)
    # Move in small steps for reliability
    steps = max(1, max(abs(x2 - x1), abs(y2 - y1)) // 5)
    for i in range(1, steps + 1):
        t = i / steps
        _set_cursor_pos(int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t))
        time.sleep(0.002)
    _set_cursor_pos(x2, y2)
    time.sleep(0.01)
    _send_input(_mouse_input(up_flag, 0, 0))


def _scroll_at(x: int, y: int, delta: int = -120) -> None:
    """Scroll at position. Negative delta = scroll down."""
    _set_cursor_pos(x, y)
    time.sleep(0.01)
    _send_input(_mouse_input(MOUSEEVENTF_WHEEL, 0, 0, delta))


# ═══════════════════════════════════════════════════════════════════
# 4. Window management
# ═══════════════════════════════════════════════════════════════════

def _get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _get_window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _get_process_name(pid: int) -> str:
    try:
        handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
        if not handle:
            return ""
        buf = ctypes.create_unicode_buffer(260)
        size = wintypes.DWORD(260)
        # QueryFullProcessImageNameW
        ret = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        kernel32.CloseHandle(handle)
        if ret:
            return Path(buf.value).name
    except Exception:
        pass
    return ""


def _list_windows(limit: int = 20) -> List[Dict[str, Any]]:
    """Enumerate visible top-level windows."""
    windows: List[Dict[str, Any]] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum_proc(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _get_window_title(hwnd)
        if not title.strip():
            return True
        pid = _get_window_pid(hwnd)
        windows.append({
            "title": title,
            "hwnd": hwnd,
            "pid": pid,
            "process_name": _get_process_name(pid),
        })
        return len(windows) < limit

    user32.EnumWindows(_enum_proc, 0)
    return windows


def _find_window_by_title(title: str) -> Optional[int]:
    """Find a visible window whose title contains `title` (case-insensitive)."""
    found: List[int] = []
    query = title.strip().lower()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum_proc(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        if query in _get_window_title(hwnd).strip().lower():
            found.append(hwnd)
            return False  # stop enumeration
        return True

    user32.EnumWindows(_enum_proc, 0)
    return found[0] if found else None


def _focus_window(title: str) -> Dict[str, Any]:
    """Find and focus a window by title substring."""
    hwnd = _find_window_by_title(title)
    if hwnd is None:
        return {"ok": False, "error": f"window not found: {title!r}"}
    try:
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.1)
    except Exception as exc:
        return {"ok": False, "error": f"SetForegroundWindow failed: {exc}"}
    actual_title = _get_window_title(hwnd)
    return {"ok": True, "title": actual_title, "hwnd": hwnd}


# ═══════════════════════════════════════════════════════════════════
# 5. Screenshot
# ═══════════════════════════════════════════════════════════════════

def _capture_screen(path: str) -> Dict[str, Any]:
    """Capture full virtual screen to a PNG file. Returns {path, width, height}."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab(all_screens=True)
        img.save(str(dest), "PNG")
        return {"ok": True, "path": str(dest), "width": img.width, "height": img.height}
    except ImportError:
        # Fallback: try mss or pyautogui
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[0]  # all-in-one
                sct.shot(mon=-1, output=str(dest))
                return {"ok": True, "path": str(dest), "width": monitor["width"], "height": monitor["height"]}
        except ImportError:
            pass
    return {"ok": False, "error": "no screenshot backend (Pillow or mss required)"}


# ═══════════════════════════════════════════════════════════════════
# 6. OCR
# ═══════════════════════════════════════════════════════════════════

def _ocr_image(path: str) -> Dict[str, Any]:
    """Run OCR on an image file. Returns {items: [{text, bbox: [x,y,w,h], confidence}]}."""
    img_path = Path(path)
    if not img_path.exists():
        return {"ok": False, "error": f"image not found: {path}"}
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(img_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        items: List[Dict[str, Any]] = []
        for i in range(len(data.get("text", []))):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            conf = float(data.get("conf", [0])[i] or 0)
            items.append({
                "text": text,
                "bbox": [
                    int(data["left"][i] or 0),
                    int(data["top"][i] or 0),
                    int(data["width"][i] or 0),
                    int(data["height"][i] or 0),
                ],
                "confidence": conf,
                "block": int(data["block_num"][i] or 0),
            })
        return {"ok": True, "items": items, "count": len(items)}
    except ImportError:
        # Fallback: try Windows OCR via subprocess
        pass
    return {"ok": False, "error": "OCR not available (install pytesseract + tesseract-ocr)"}


# ═══════════════════════════════════════════════════════════════════
# 7. UI Automation (pywinauto)
# ═══════════════════════════════════════════════════════════════════

_pywinauto_available = False
try:
    import pywinauto
    _pywinauto_available = True
except ImportError:
    pass


def _list_controls(window_title: str, limit: int = 50) -> Dict[str, Any]:
    """List UI automation controls for a window."""
    if not _pywinauto_available:
        return {"ok": False, "error": "pywinauto not installed", "controls": []}
    try:
        from pywinauto import Desktop
        dlg = Desktop(backend="uia").window(title=window_title)
        if not dlg.exists():
            dlg = Desktop(backend="win32").window(title=window_title)
        if not dlg.exists():
            return {"ok": False, "error": f"window not found via UIA: {window_title!r}", "controls": []}
        controls: List[Dict[str, Any]] = []
        for ctrl in dlg.descendants()[:limit]:
            try:
                rect = ctrl.rectangle()
                controls.append({
                    "title": ctrl.window_text() or "",
                    "control_type": ctrl.element_info.control_type or "",
                    "automation_id": ctrl.element_info.automation_id or "",
                    "class_name": ctrl.element_info.class_name or "",
                    "enabled": ctrl.is_enabled() if hasattr(ctrl, "is_enabled") else True,
                    "visible": ctrl.is_visible() if hasattr(ctrl, "is_visible") else True,
                    "rect": [rect.left, rect.top, rect.right, rect.bottom] if rect else [0, 0, 0, 0],
                })
            except Exception:
                continue
        return {"ok": True, "controls": controls, "count": len(controls)}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "controls": []}


def _click_control(window: str, control_text: str, index: int = 0,
                   control_type: Optional[str] = None,
                   fallback_to_ocr: bool = True,
                   ocr_text: str = "") -> Dict[str, Any]:
    """Click a UI control by name/text/index."""
    if not _pywinauto_available:
        return {"ok": False, "error": "pywinauto not installed"}
    try:
        from pywinauto import Desktop
        dlg = Desktop(backend="uia").window(title=window)
        if not dlg.exists():
            dlg = Desktop(backend="win32").window(title=window)
        if not dlg.exists():
            return {"ok": False, "error": f"window not found: {window!r}"}
        dlg.set_focus()
        time.sleep(0.1)
        # Try to find the control
        candidates = []
        for ctrl in dlg.descendants():
            ctrl_name = ctrl.window_text() or ""
            ctrl_aid  = ctrl.element_info.automation_id or ""
            if control_text and (control_text.lower() in ctrl_name.lower() or control_text.lower() in ctrl_aid.lower()):
                candidates.append(ctrl)
        if candidates and 0 <= index < len(candidates):
            candidates[index].click_input()
            return {"ok": True, "clicked": control_text, "index": index}
        # Fallback: click by index on all controls
        all_ctrls = list(dlg.descendants())
        if 0 <= index < len(all_ctrls):
            all_ctrls[index].click_input()
            return {"ok": True, "clicked": f"control[{index}]", "index": index}
        return {"ok": False, "error": f"control {control_text!r} not found in {window!r}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _type_control(window: str, control_text: str, text: str, index: int = 0,
                  clear_first: bool = True, control_type: Optional[str] = None,
                  fallback_to_ocr: bool = True, ocr_text: str = "") -> Dict[str, Any]:
    """Type into a UI control."""
    if not _pywinauto_available:
        return {"ok": False, "error": "pywinauto not installed"}
    try:
        from pywinauto import Desktop
        dlg = Desktop(backend="uia").window(title=window)
        if not dlg.exists():
            dlg = Desktop(backend="win32").window(title=window)
        if not dlg.exists():
            return {"ok": False, "error": f"window not found: {window!r}"}
        dlg.set_focus()
        time.sleep(0.1)
        candidates = []
        for ctrl in dlg.descendants():
            ctrl_name = ctrl.window_text() or ""
            ctrl_aid  = ctrl.element_info.automation_id or ""
            if control_text and (control_text.lower() in ctrl_name.lower() or control_text.lower() in ctrl_aid.lower()):
                candidates.append(ctrl)
        target = None
        if candidates and 0 <= index < len(candidates):
            target = candidates[index]
        elif 0 <= index < len(list(dlg.descendants())):
            target = list(dlg.descendants())[index]
        if target is not None:
            if clear_first:
                try:
                    target.set_edit_text("")
                except Exception:
                    pass
            target.type_keys(text)
            return {"ok": True, "typed_into": control_text or f"control[{index}]"}
        return {"ok": False, "error": f"control {control_text!r} not found"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
# 8. OCR-based clicking
# ═══════════════════════════════════════════════════════════════════

def _click_ocr_text(text: str, title: str = "") -> Dict[str, Any]:
    """Find text on screen via OCR and click its center."""
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".png"))
    try:
        # If a specific window title is given, try to focus it first
        if title:
            _focus_window(title)
            time.sleep(0.3)
        # Capture screen
        from PIL import ImageGrab
        img = ImageGrab.grab(all_screens=True)
        img.save(str(tmp), "PNG")
        # OCR
        ocr_result = _ocr_image(str(tmp))
        if not ocr_result.get("ok"):
            return {"ok": False, "error": f"OCR failed: {ocr_result.get('error')}"}
        # Find the best-matching text item
        query = text.strip().lower()
        best = None
        for item in ocr_result.get("items", []):
            item_text = item.get("text", "").strip().lower()
            if query in item_text or item_text in query:
                bbox = item.get("bbox", [0, 0, 0, 0])
                cx = bbox[0] + bbox[2] // 2
                cy = bbox[1] + bbox[3] // 2
                best = (cx, cy, item_text)
                break
        if best:
            _click_at(best[0], best[1])
            return {"ok": True, "clicked_text": text, "matched": best[2], "x": best[0], "y": best[1]}
        return {"ok": False, "error": f"text {text!r} not found on screen"}
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# 9. Application launch
# ═══════════════════════════════════════════════════════════════════

def _launch_app(target: str, args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Launch an application by name or path."""
    cmd = [target] + (list(args or []))
    try:
        proc = subprocess.Popen(cmd, shell=True)
        time.sleep(0.5)
        return {"ok": True, "pid": proc.pid, "target": target}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════
# 10. Browser automation plugins (Playwright-based, optional)
# ═══════════════════════════════════════════════════════════════════

def _plugin_browser_dom_open(payload: dict) -> dict:
    return {"ok": False, "error": "Playwright not installed — browser plugins require 'pip install playwright'"}

def _plugin_browser_dom_click(payload: dict) -> dict:
    return {"ok": False, "error": "Playwright not installed"}

def _plugin_browser_dom_type(payload: dict) -> dict:
    return {"ok": False, "error": "Playwright not installed"}

def _plugin_browser_dom_screenshot(payload: dict) -> dict:
    return {"ok": False, "error": "Playwright not installed"}


# ═══════════════════════════════════════════════════════════════════
# 11. Plugin registry
# ═══════════════════════════════════════════════════════════════════

class PluginRegistry:
    """Registry of named async plugin functions."""

    def __init__(self) -> None:
        self._plugins: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._plugins[name] = fn

    async def run(self, plugin_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._plugins.get(plugin_name)
        if fn is None:
            logger.warning("Plugin %r not registered", plugin_name)
            return {"ok": False, "error": f"plugin {plugin_name!r} not registered"}
        try:
            result = fn(payload)
            if asyncio.iscoroutine(result):
                result = await result
            if not isinstance(result, dict):
                result = {"ok": True, "result": result}
            result.setdefault("plugin", plugin_name)
            return result
        except Exception as exc:
            logger.exception("Plugin %r failed", plugin_name)
            return {"ok": False, "error": str(exc), "plugin": plugin_name}

    def list_plugins(self) -> List[str]:
        return sorted(self._plugins.keys())


# ═══════════════════════════════════════════════════════════════════
# 11. Plugin implementations
# ═══════════════════════════════════════════════════════════════════

def _plugin_desktop_active_window(_payload: dict) -> dict:
    hwnd = user32.GetForegroundWindow()
    title = _get_window_title(hwnd or 0)
    pid = _get_window_pid(hwnd or 0)
    return {
        "ok": True,
        "title": title,
        "hwnd": hwnd or 0,
        "pid": pid,
        "process_name": _get_process_name(pid),
    }


def _plugin_desktop_list_windows(payload: dict) -> dict:
    limit = max(1, int(payload.get("limit", 20) or 20))
    windows = _list_windows(limit=limit)
    return {"ok": True, "windows": windows, "count": len(windows)}


def _plugin_desktop_screenshot(payload: dict) -> dict:
    path = payload.get("path", "")
    if not path:
        return {"ok": False, "error": "missing path"}
    return _capture_screen(str(path))


def _plugin_desktop_ocr(payload: dict) -> dict:
    path = payload.get("path", "")
    if not path:
        return {"ok": False, "error": "missing path"}
    return _ocr_image(str(path))


def _plugin_desktop_list_controls(payload: dict) -> dict:
    window = payload.get("window", "")
    limit = max(1, int(payload.get("limit", 50) or 50))
    if not window:
        return {"ok": False, "error": "missing window", "controls": []}
    return _list_controls(str(window), limit=limit)


def _plugin_desktop_focus_window(payload: dict) -> dict:
    title = payload.get("title", "")
    if not title:
        return {"ok": False, "error": "missing title"}
    return _focus_window(str(title))


def _plugin_desktop_click_control(payload: dict) -> dict:
    return _click_control(
        window=str(payload.get("window", "")),
        control_text=str(payload.get("control", "")),
        index=int(payload.get("index", 0) or 0),
        control_type=payload.get("control_type"),
        fallback_to_ocr=bool(payload.get("fallback_to_ocr", True)),
        ocr_text=str(payload.get("ocr_text", "")),
    )


def _plugin_desktop_type_control(payload: dict) -> dict:
    return _type_control(
        window=str(payload.get("window", "")),
        control_text=str(payload.get("control", "")),
        text=str(payload.get("text", "")),
        index=int(payload.get("index", 0) or 0),
        clear_first=bool(payload.get("clear_first", True)),
        control_type=payload.get("control_type"),
        fallback_to_ocr=bool(payload.get("fallback_to_ocr", True)),
        ocr_text=str(payload.get("ocr_text", "")),
    )


def _plugin_desktop_click_text(payload: dict) -> dict:
    text = str(payload.get("text", ""))
    title = str(payload.get("title", ""))
    if not text:
        return {"ok": False, "error": "missing text"}
    return _click_ocr_text(text, title=title)


def _plugin_desktop_click(payload: dict) -> dict:
    x = int(payload.get("x", 0) or 0)
    y = int(payload.get("y", 0) or 0)
    button = str(payload.get("button", "left") or "left")
    clicks = max(1, int(payload.get("clicks", 1) or 1))
    try:
        _click_at(x, y, button=button, clicks=clicks)
        return {"ok": True, "x": x, "y": y, "button": button, "clicks": clicks}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _plugin_desktop_double_click(payload: dict) -> dict:
    x = int(payload.get("x", 0) or 0)
    y = int(payload.get("y", 0) or 0)
    button = str(payload.get("button", "left") or "left")
    try:
        _double_click_at(x, y, button=button)
        return {"ok": True, "x": x, "y": y, "button": button}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _plugin_desktop_drag(payload: dict) -> dict:
    x1 = int(payload.get("x1", 0) or 0)
    y1 = int(payload.get("y1", 0) or 0)
    x2 = int(payload.get("x2", 0) or 0)
    y2 = int(payload.get("y2", 0) or 0)
    try:
        _drag(x1, y1, x2, y2)
        return {"ok": True, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _plugin_desktop_scroll(payload: dict) -> dict:
    x = int(payload.get("x", 0) or 0)
    y = int(payload.get("y", 0) or 0)
    amount = int(payload.get("amount", payload.get("delta", -120)) or -120)
    try:
        _scroll_at(x, y, delta=amount)
        return {"ok": True, "x": x, "y": y, "delta": amount}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _plugin_desktop_type_text(payload: dict) -> dict:
    text = str(payload.get("text", ""))
    if not text:
        return {"ok": False, "error": "missing text"}
    try:
        _send_text_via_sendinput(text)
        return {"ok": True, "characters": len(text)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _plugin_desktop_hotkey(payload: dict) -> dict:
    hotkey = str(payload.get("hotkey", ""))
    if not hotkey:
        return {"ok": False, "error": "missing hotkey"}
    parsed = _parse_hotkey(hotkey)
    if parsed is None:
        return {"ok": False, "error": f"unknown hotkey: {hotkey!r}"}
    mod_vk, key_vk = parsed
    try:
        _send_hotkey(mod_vk, key_vk)
        return {"ok": True, "hotkey": hotkey}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _plugin_desktop_launch(payload: dict) -> dict:
    target = str(payload.get("target", ""))
    if not target:
        return {"ok": False, "error": "missing target"}
    args = list(payload.get("args", []) or [])
    return _launch_app(target, args)


# ═══════════════════════════════════════════════════════════════════
# 12. Global registry — register all plugins
# ═══════════════════════════════════════════════════════════════════

registry = PluginRegistry()

registry.register("desktop_active_window",  _plugin_desktop_active_window)
registry.register("desktop_list_windows",   _plugin_desktop_list_windows)
registry.register("desktop_screenshot",     _plugin_desktop_screenshot)
registry.register("desktop_ocr",            _plugin_desktop_ocr)
registry.register("desktop_list_controls",  _plugin_desktop_list_controls)
registry.register("desktop_focus_window",   _plugin_desktop_focus_window)
registry.register("desktop_click_control",  _plugin_desktop_click_control)
registry.register("desktop_type_control",   _plugin_desktop_type_control)
registry.register("desktop_click_text",     _plugin_desktop_click_text)
registry.register("desktop_click",          _plugin_desktop_click)
registry.register("desktop_double_click",   _plugin_desktop_double_click)
registry.register("desktop_drag",           _plugin_desktop_drag)
registry.register("desktop_scroll",         _plugin_desktop_scroll)
registry.register("desktop_type_text",      _plugin_desktop_type_text)
registry.register("desktop_hotkey",         _plugin_desktop_hotkey)
registry.register("desktop_launch",         _plugin_desktop_launch)
registry.register("browser_dom_open",       _plugin_browser_dom_open)
registry.register("browser_dom_click",      _plugin_browser_dom_click)
registry.register("browser_dom_type",       _plugin_browser_dom_type)
registry.register("browser_dom_screenshot", _plugin_browser_dom_screenshot)
