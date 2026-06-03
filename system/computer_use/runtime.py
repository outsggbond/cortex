# -*- coding: utf-8 -*-
"""Stepwise desktop computer-use runtime with grid-aware observation."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from system.computer_use.llm import (
    BaseComputerUseLLMClient,
    DEFAULT_COMPUTER_USE_SYSTEM_PROMPT,
    build_computer_use_llm_client,
)
from system.computer_use._runtime_types import (
    BLOCK_ID_CHARS,
    ComputerAction,
    ComputerUseConfig,
    DEFAULT_COMPUTER_USE_RECOVERY_PATH,
    _RESUME_COMPUTER_USE_GOALS,
)

from system.computer_use.trace.audit import ErrorAudit, ExecAudit, TraceAudit

from system.computer_use._runtime_grid import (
    _clean_str,
    _truncate,
    _round_ratio,
    _load_json_dict,
    _is_resume_request,
    _match_pattern,
    _normalize_hotkey,
    _hint_variants,
    _looks_like_launch_text,
    _goal_launch_hints,
    _resolve_path,
    _append_jsonl,
    _write_json,
    _file_sha1,
    _column_label,
    _block_id,
    _build_grid,
    _find_grid_cell,
    _map_ocr_to_grid,
    _nonempty_blocks,
    _extract_json_object,
    _image_size,
    _render_grid_overlay,
)

import system.computer_use._runtime_plugins as _runtime_plugins
import system.computer_use._runtime_observe as _runtime_observe
import system.computer_use._runtime_actions as _runtime_actions


logger = logging.getLogger(__name__)


_STATEFUL_DESKTOP_ACTION_TYPES = {
    "click_control",
    "type_control",
    "click_text",
    "click_block",
    "drag_blocks",
    "type_text",
    "hotkey",
}
_DESKTOP_CONTEXT_ACTION_TYPES = {
    "focus_window",
    "launch",
}
_CAPTCHA_PATTERNS = (
    "captcha",
    "not a robot",
    "verify you are human",
    "verify you're human",
    "robot check",
    "security check",
    "challenge",
    "验证码",
    "安全验证",
    "人机验证",
    "请完成验证",
    "滑动验证",
    "拖动滑块",
    "拼图验证",
)
_ANTI_AUTOMATION_PATTERNS = (
    "automation",
    "bot detected",
    "access denied",
    "too many requests",
    "unusual traffic",
    "request blocked",
    "blocked",
    "频繁操作",
    "访问受限",
    "异常流量",
    "检测到异常",
    "操作过于频繁",
)
_HIGH_RISK_TEXT_PATTERNS = (
    "password",
    "passcode",
    "2fa",
    "otp",
    "token",
    "secret",
    "bank",
    "wallet",
    "payment",
    "pay now",
    "wire transfer",
    "terminal",
    "powershell",
    "cmd.exe",
    "regedit",
    "delete",
    "format disk",
    "erase",
    "密码",
    "验证码",
    "口令",
    "支付",
    "付款",
    "转账",
    "银行",
    "钱包",
    "终端",
    "命令行",
    "删除",
    "格式化",
)
_HIGH_RISK_UI_PATTERNS = (
    "delete",
    "erase",
    "format",
    "payment",
    "pay",
    "transfer",
    "withdraw",
    "submit order",
    "purchase",
    "删除",
    "清空",
    "格式化",
    "支付",
    "付款",
    "转账",
    "提现",
    "购买",
    "提交订单",
)
_HIGH_RISK_HOTKEYS = {
    "alt+f4",
    "ctrl+alt+del",
    "ctrl+shift+esc",
    "win+r",
    "windows+r",
    "win+x",
    "windows+x",
}


class ComputerUseRuntime:
    def __init__(
        self,
        config: ComputerUseConfig | None = None,
        *,
        llm_client: BaseComputerUseLLMClient | None = None,
    ) -> None:
        self.config = config or ComputerUseConfig()
        self.project_root = Path(str(self.config.project_root or ".")).resolve()
        self.trace_root = _resolve_path(self.project_root, str(self.config.trace_root or "artifacts/audit/computer_use"))
        self.recovery_path = _resolve_path(
            self.project_root,
            str(self.config.recovery_path or DEFAULT_COMPUTER_USE_RECOVERY_PATH),
        )
        self.screenshot_dir = _resolve_path(
            self.project_root,
            str(self.config.screenshot_dir or "artifacts/screenshots"),
        )
        self._legacy_action_index = 0
        self.trace = TraceAudit(str(self.trace_root / "trace_log.jsonl"))
        self.error_log = ErrorAudit(str(self.trace_root / "error_log.jsonl"))
        self.exec_log = ExecAudit(str(self.trace_root / "exec_log.jsonl"))
        self.llm_client = llm_client or build_computer_use_llm_client(
            system_prompt=str(self.config.system_prompt or DEFAULT_COMPUTER_USE_SYSTEM_PROMPT),
            image_detail=str(self.config.image_detail or "auto"),
        )

    def _action_from_dict(self, payload: Dict[str, Any]) -> ComputerAction:
        return self._parse_action(json.dumps(dict(payload or {}), ensure_ascii=False))

    def _observation_text(self, goal: str, observation: Dict[str, Any]) -> str:
        parts: List[str] = [goal]
        active = dict(observation.get("active_window", {}) or {})
        parts.append(_clean_str(active.get("title")))
        parts.append(_clean_str(active.get("process_name")))
        parts.append(_clean_str(observation.get("controls_window")))
        parts.append(_clean_str(observation.get("ocr_text")))
        for item in list(observation.get("controls", []) or [])[:20]:
            node = dict(item or {})
            parts.append(_clean_str(node.get("title")))
            parts.append(_clean_str(node.get("control_type")))
        for item in list(observation.get("open_windows", []) or [])[:12]:
            node = dict(item or {})
            parts.append(_clean_str(node.get("title")))
            parts.append(_clean_str(node.get("process_name")))
        return " | ".join([item for item in parts if _clean_str(item)])

    def _observation_search_space(self, observation: Dict[str, Any], *, include_ocr: bool = True) -> List[str]:
        parts: List[str] = []
        active = dict(observation.get("active_window", {}) or {})
        parts.extend(
            [
                _clean_str(active.get("title")),
                _clean_str(active.get("process_name")),
                _clean_str(observation.get("controls_window")),
            ]
        )
        for item in list(observation.get("controls", []) or [])[:24]:
            node = dict(item or {})
            parts.extend(
                [
                    _clean_str(node.get("title")),
                    _clean_str(node.get("control_type")),
                    _clean_str(node.get("automation_id")),
                ]
            )
        for item in list(observation.get("open_windows", []) or [])[:24]:
            node = dict(item or {})
            parts.extend(
                [
                    _clean_str(node.get("title")),
                    _clean_str(node.get("process_name")),
                ]
            )
        if include_ocr:
            parts.append(_clean_str(observation.get("ocr_text")))
        return [part for part in parts if part]

    def _observation_matches_hints(self, observation: Dict[str, Any], hints: Sequence[str]) -> bool:
        candidates = [part.lower() for part in self._observation_search_space(observation)]
        if not candidates:
            return False
        for hint in list(hints or []):
            for variant in _hint_variants(hint):
                if any(variant in candidate for candidate in candidates):
                    return True
        return False

    def _observation_shows_run_dialog(self, observation: Dict[str, Any]) -> bool:
        active = dict(observation.get("active_window", {}) or {})
        title_fields = [
            _clean_str(active.get("title")),
            _clean_str(observation.get("controls_window")),
        ]
        for item in list(observation.get("open_windows", []) or [])[:16]:
            title_fields.append(_clean_str((item or {}).get("title")))
        return any(field.lower() == "run" or "运行" in field for field in title_fields if field)

    def _last_successful_row(self, history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        for row in reversed(list(history or [])):
            candidate = dict(row or {})
            if bool(candidate.get("ok", False)) and bool(candidate.get("verified", False)):
                return candidate
        return {}

    def _run_dialog_launch_hint(self, history: Sequence[Dict[str, Any]]) -> str:
        rows = list(history or [])
        for idx in range(len(rows) - 1, 1, -1):
            submit_row = dict(rows[idx] or {})
            submit_action = dict(submit_row.get("action", {}) or {})
            if not (bool(submit_row.get("ok", False)) and bool(submit_row.get("verified", False))):
                continue
            if _clean_str(submit_action.get("type")).lower() != "hotkey":
                continue
            if _normalize_hotkey(_clean_str(submit_action.get("keys"))) != "enter":
                continue

            text_row = dict(rows[idx - 1] or {})
            text_action = dict(text_row.get("action", {}) or {})
            launch_text = _clean_str(text_action.get("text"))
            if not (bool(text_row.get("ok", False)) and bool(text_row.get("verified", False))):
                continue
            if _clean_str(text_action.get("type")).lower() != "type_text" or not _looks_like_launch_text(launch_text):
                continue

            hotkey_row = dict(rows[idx - 2] or {})
            hotkey_action = dict(hotkey_row.get("action", {}) or {})
            if not (bool(hotkey_row.get("ok", False)) and bool(hotkey_row.get("verified", False))):
                continue
            if _clean_str(hotkey_action.get("type")).lower() != "hotkey":
                continue
            if _normalize_hotkey(_clean_str(hotkey_action.get("keys"))) not in {"win+r", "windows+r"}:
                continue
            return launch_text
        return ""

    def _completion_target_hints(self, history: Sequence[Dict[str, Any]]) -> List[str]:
        hints: List[str] = []

        def _add_hint(value: Any) -> None:
            hint = _clean_str(value)
            if hint and hint not in hints:
                hints.append(hint)

        _add_hint(self.config.target_window)
        launch_hint = self._run_dialog_launch_hint(history)
        if launch_hint:
            _add_hint(launch_hint)
        for row in reversed(list(history or [])):
            candidate = dict(row or {})
            if not (bool(candidate.get("ok", False)) and bool(candidate.get("verified", False))):
                continue
            action = dict(candidate.get("action", {}) or {})
            action_type = _clean_str(action.get("type")).lower()
            if action_type == "launch":
                _add_hint(action.get("target"))
            elif action_type == "focus_window":
                _add_hint(action.get("title") or action.get("window"))
            elif action_type in {"click_control", "type_control", "click_text"}:
                _add_hint(action.get("window"))
            if len(hints) >= 8:
                break
        return hints

    def _explicit_launch_or_focus_hints(self, goal: str, history: Sequence[Dict[str, Any]]) -> List[str]:
        hints: List[str] = []
        for item in _goal_launch_hints(goal):
            if item not in hints:
                hints.append(item)
        for item in self._completion_target_hints(history):
            if item not in hints:
                hints.append(item)
        return hints

    def _pending_target_hints(
        self,
        *,
        history: Sequence[Dict[str, Any]],
        observation: Dict[str, Any],
    ) -> List[str]:
        pending: List[str] = []
        for hint in self._completion_target_hints(history):
            if hint and not self._observation_matches_hints(observation, [hint]) and hint not in pending:
                pending.append(hint)
        return pending

    def _validate_runtime_action(
        self,
        *,
        goal: str,
        action: ComputerAction,
        observation: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if action.type == "hotkey":
            normalized = _normalize_hotkey(_clean_str(action.keys))
            if normalized in {"win+r", "windows+r"}:
                explicit_targets = self._explicit_launch_or_focus_hints(goal, history)
                if explicit_targets:
                    return {
                        "reason": (
                            f"Use launch or focus_window for the target app ({explicit_targets[0]}) "
                            "instead of win+r when an explicit app target is known."
                        ),
                        "reason_code": "prefer_launch_or_focus_window_over_win_r",
                    }
        if action.type == "type_text":
            last_row = self._last_successful_row(history)
            last_action = dict(last_row.get("action", {}) or {})
            if (
                _clean_str(last_action.get("type")).lower() == "hotkey"
                and _normalize_hotkey(_clean_str(last_action.get("keys"))) in {"win+r", "windows+r"}
                and not self._observation_shows_run_dialog(observation)
            ):
                return {
                    "reason": "Run dialog was not observed after win+r; refusing blind free-text input.",
                    "reason_code": "missing_run_dialog_after_hotkey",
                }
            pending_hints = self._pending_target_hints(history=history, observation=observation)
            if pending_hints:
                return {
                    "reason": f"Target app or window is not visible yet ({pending_hints[0]}); refusing blind free-text input.",
                    "reason_code": "target_not_observed_before_type_text",
                }
        if action.type == "done":
            completion_hints = self._completion_target_hints(history)
            if completion_hints and not self._observation_matches_hints(observation, completion_hints):
                return {
                    "reason": f"Completion rejected because the target app or window is not visible in the latest observation ({completion_hints[0]}).",
                    "reason_code": "done_without_target_observed",
                }
        return None

    def _detect_handoff_from_observation(
        self,
        *,
        goal: str,
        observation: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        combined = self._observation_text(goal, observation)
        if bool(self.config.auto_handoff_on_captcha):
            matched = _match_pattern(combined, _CAPTCHA_PATTERNS)
            if matched:
                return {
                    "reason": f"captcha or human-verification challenge detected ({matched})",
                    "reason_code": "captcha_detected",
                }
        if bool(self.config.auto_handoff_on_anti_automation):
            matched = _match_pattern(combined, _ANTI_AUTOMATION_PATTERNS)
            if matched:
                return {
                    "reason": f"anti-automation or access challenge detected ({matched})",
                    "reason_code": "anti_automation_detected",
                }
        return None

    def _classify_high_risk_action(
        self,
        *,
        goal: str,
        action: ComputerAction,
        observation: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        if bool(self.config.allow_high_risk_actions):
            return None
        if action.type == "launch":
            return {
                "reason": "launch actions are treated as high risk by default; rerun with explicit high-risk allowance",
                "reason_code": "high_risk_launch",
            }
        if action.type == "hotkey":
            normalized = _clean_str(action.keys).lower().replace(" ", "")
            if normalized in {item.replace(" ", "") for item in _HIGH_RISK_HOTKEYS}:
                return {
                    "reason": f"high-risk hotkey blocked by default ({action.keys})",
                    "reason_code": "high_risk_hotkey",
                }
        if action.type == "type_text":
            matched = _match_pattern(f"{goal} | {action.text}", _HIGH_RISK_TEXT_PATTERNS)
            if matched:
                return {
                    "reason": f"sensitive free-text action blocked by default ({matched})",
                    "reason_code": "high_risk_text",
                }
        if action.type in {"click_control", "click_text", "click_block"}:
            target_text = " | ".join(
                [
                    _clean_str(action.control),
                    _clean_str(action.text),
                    _clean_str(action.block),
                    self._observation_text(goal, observation),
                ]
            )
            matched = _match_pattern(target_text, _HIGH_RISK_UI_PATTERNS)
            if matched:
                return {
                    "reason": f"high-risk UI action blocked by default ({matched})",
                    "reason_code": "high_risk_ui_action",
                }
        return None

    def _run_plugin_once(self, plugin_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return _runtime_plugins._run_plugin_once(self, plugin_name, payload)

    def _run_plugin(self, plugin_name: str, payload: Dict[str, Any], *, required: bool = False) -> Dict[str, Any]:
        return _runtime_plugins._run_plugin(self, plugin_name, payload, required=required)

    def _legacy_result(self, result: Dict[str, Any], *, action_type: str) -> Dict[str, Any]:
        return _runtime_plugins._legacy_result(result, action_type=action_type)

    def run_action(self, action_type: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return _runtime_plugins.run_action(self, action_type, params)

    def cleanup(self) -> None:
        return _runtime_plugins.cleanup(self)

    def _observe(
        self,
        *,
        step_index: int,
        run_dir: Path,
        preferred_window: str,
    ) -> Dict[str, Any]:
        return _runtime_observe._observe(self, step_index=step_index, run_dir=run_dir, preferred_window=preferred_window)

    def _decision_prompt(
        self,
        *,
        goal: str,
        observation: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
    ) -> str:
        recent_steps = []
        for item in list(history or [])[-5:]:
            recent_steps.append(
                {
                    "step": int(item.get("step", 0) or 0),
                    "action": dict(item.get("action", {}) or {}),
                    "result": {
                        "ok": bool(item.get("ok", False)),
                        "verified": bool(item.get("verified", False)),
                        "note": _truncate(str(item.get("note", "") or ""), 160),
                        "error": _truncate(str(item.get("error", "") or ""), 160),
                    },
                }
            )
        observation_payload = {
            "step_index": int(observation.get("step_index", 0) or 0),
            "max_steps": max(1, int(self.config.max_steps)),
            "goal": goal,
            "active_window": dict(observation.get("active_window", {}) or {}),
            "open_windows": list(observation.get("open_windows", []) or [])[:12],
            "controls_window": _clean_str(observation.get("controls_window")),
            "controls": list(observation.get("controls", []) or [])[:12],
            "surface_mode": _clean_str(observation.get("surface_mode")),
            "special_drawn_ui_likely": bool(observation.get("special_drawn_ui_likely", False)),
            "targeting_hint": _clean_str(observation.get("targeting_hint")),
            "ocr_text_preview": _clean_str(observation.get("ocr_text")),
            "ocr_items": list(observation.get("ocr_items", []) or [])[:24],
            "nonempty_blocks": list(observation.get("nonempty_blocks", []) or [])[:24],
            "image_size": {
                "width": int(observation.get("image_width", 0) or 0),
                "height": int(observation.get("image_height", 0) or 0),
            },
            "images": {
                "raw_screenshot": _clean_str(observation.get("screen_path")),
                "grid_overlay": _clean_str(observation.get("grid_path")),
            },
            "recent_steps": recent_steps,
        }
        return (
            "Choose one next desktop action.\n"
            "Images attached in order: first is the raw screenshot, second is the same screenshot with labeled grid blocks.\n"
            "Return one JSON object only.\n"
            "Allowed action shapes:\n"
            '{"type":"focus_window","title":"QQ","reason":"short reason"}\n'
            '{"type":"click_control","window":"QQ","control":"Send","control_type":"Button","reason":"short reason"}\n'
            '{"type":"type_control","window":"QQ","control":"Search","control_type":"Edit","text":"hello","clear_first":true,"reason":"short reason"}\n'
            '{"type":"click_text","text":"Send","window":"QQ","reason":"short reason"}\n'
            '{"type":"click_block","block":"B3","button":"left","clicks":1,"reason":"short reason"}\n'
            '{"type":"drag_blocks","from_block":"B3","to_block":"G3","reason":"short reason"}\n'
            '{"type":"type_text","text":"hello","reason":"short reason"}\n'
            '{"type":"hotkey","keys":"ctrl+l","reason":"short reason"}\n'
            '{"type":"launch","target":"C:/Path/App.exe","reason":"short reason"}\n'
            '{"type":"wait","seconds":1.0,"reason":"short reason"}\n'
            '{"type":"handoff","reason":"human action required before automation can continue"}\n'
            '{"type":"done","summary":"goal completed"}\n'
            '{"type":"fail","reason":"what blocked progress"}\n'
            "Rules:\n"
            "- For desktop apps such as QQ or Notepad, prefer launch or focus_window before trying win+r.\n"
            "- Avoid win+r when the target app name or window is already known from the goal, history, or observation.\n"
            "- If the target app is already visible in open_windows, use focus_window instead of relaunching it.\n"
            "- Prefer focus_window, click_control, type_control, and click_text before click_block.\n"
            "- Use click_block or drag_blocks only when OCR and controls are not enough.\n"
            "- If surface_mode is ocr_only or visual_only, prefer click_text or click_block instead of control-only actions.\n"
            "- If a captcha, verification challenge, anti-automation page, or high-risk operation is visible, return handoff.\n"
            "- Reuse observed window/control names when possible.\n"
            "- If the task is already finished, return done.\n"
            "- If the task cannot continue safely, return fail.\n\n"
            "Observation JSON:\n"
            f"{json.dumps(observation_payload, ensure_ascii=False, indent=2)}"
        )

    def _parse_action(self, raw_text: str) -> ComputerAction:
        payload_text = _extract_json_object(raw_text)
        if not payload_text:
            raise ValueError("model did not return a JSON object")
        data = json.loads(payload_text)
        if isinstance(data, dict) and isinstance(data.get("action"), dict):
            data = dict(data.get("action") or {})
        if not isinstance(data, dict):
            raise ValueError("model action payload must be an object")
        action_type = _clean_str(data.get("type")).lower()
        if not action_type:
            raise ValueError("model action missing type")
        action = ComputerAction(
            type=action_type,
            reason=_clean_str(data.get("reason")),
            title=_clean_str(data.get("title")),
            window=_clean_str(data.get("window")),
            target=_clean_str(data.get("target")),
            control=_clean_str(data.get("control")),
            control_type=_clean_str(data.get("control_type")),
            text=str(data.get("text", "") or ""),
            block=_clean_str(data.get("block")).upper(),
            from_block=_clean_str(data.get("from_block")).upper(),
            to_block=_clean_str(data.get("to_block")).upper(),
            button=_clean_str(data.get("button") or "left").lower() or "left",
            clicks=max(1, int(data.get("clicks", 1) or 1)),
            keys=_clean_str(data.get("keys")),
            seconds=max(0.1, float(data.get("seconds", 1.0) or 1.0)),
            clear_first=bool(data.get("clear_first", False)),
            index=max(0, int(data.get("index", 0) or 0)),
            summary=_clean_str(data.get("summary")),
        )
        if action.type == "focus_window":
            action.title = action.title or action.window
            if not action.title:
                raise ValueError("focus_window requires title")
        elif action.type == "click_control":
            if not action.control:
                raise ValueError("click_control requires control")
        elif action.type == "type_control":
            if not action.control:
                raise ValueError("type_control requires control")
            if not action.text:
                raise ValueError("type_control requires text")
        elif action.type == "click_text":
            if not action.text:
                raise ValueError("click_text requires text")
        elif action.type == "click_block":
            if not action.block:
                raise ValueError("click_block requires block")
        elif action.type == "drag_blocks":
            if not action.from_block or not action.to_block:
                raise ValueError("drag_blocks requires from_block and to_block")
        elif action.type == "type_text":
            if not action.text:
                raise ValueError("type_text requires text")
        elif action.type == "hotkey":
            if not action.keys:
                raise ValueError("hotkey requires keys")
        elif action.type == "launch":
            if not action.target:
                raise ValueError("launch requires target")
        elif action.type == "wait":
            pass
        elif action.type in {"handoff", "request_human"}:
            action.type = "handoff"
            action.reason = action.reason or action.summary or "human action required"
        elif action.type == "done":
            action.summary = action.summary or action.reason or "goal completed"
        elif action.type == "fail":
            action.reason = action.reason or action.summary or "no reason provided"
        else:
            raise ValueError(f"unsupported action type: {action.type}")
        return action

    def _decide_action(
        self,
        *,
        goal: str,
        observation: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
    ) -> tuple[ComputerAction, str]:
        prompt = self._decision_prompt(goal=goal, observation=observation, history=history)
        image_paths = [
            str(observation.get("screen_path", "") or ""),
            str(observation.get("grid_path", "") or ""),
        ]
        raw = self.llm_client.decide(
            prompt=prompt,
            image_paths=image_paths,
            system_prompt=str(self.config.system_prompt or DEFAULT_COMPUTER_USE_SYSTEM_PROMPT),
        )
        try:
            return self._parse_action(raw), raw
        except Exception as exc:
            repair_prompt = (
                f"{prompt}\n\n"
                f"Your previous reply was invalid because: {exc}\n"
                "Return one corrected JSON object only."
            )
            repaired = self.llm_client.decide(
                prompt=repair_prompt,
                image_paths=image_paths,
                system_prompt=str(self.config.system_prompt or DEFAULT_COMPUTER_USE_SYSTEM_PROMPT),
            )
            return self._parse_action(repaired), repaired

    def _execute_action(self, action: ComputerAction, observation: Dict[str, Any]) -> Dict[str, Any]:
        return _runtime_actions._execute_action(self, action, observation)

    def _find_last_verified_index(self, history: Sequence[Dict[str, Any]], before_index: int) -> int:
        limit = max(0, min(int(before_index), len(history)))
        for idx in range(limit - 1, -1, -1):
            row = dict(history[idx] or {})
            if bool(row.get("ok", False)) and bool(row.get("verified", False)):
                return idx
        return -1

    def _find_last_context_index(self, history: Sequence[Dict[str, Any]], before_index: int) -> int:
        limit = max(0, min(int(before_index), len(history)))
        for idx in range(limit - 1, -1, -1):
            row = dict(history[idx] or {})
            action = dict(row.get("action", {}) or {})
            action_type = _clean_str(action.get("type")).lower()
            if action_type in _DESKTOP_CONTEXT_ACTION_TYPES and bool(row.get("ok", False)):
                return idx
        return -1

    def _build_recovery_payload(
        self,
        *,
        goal: str,
        status: str,
        reason: str,
        history: Sequence[Dict[str, Any]],
        latest_observation: Dict[str, Any] | None,
        run_dir: Path,
        trace_path: Path,
        summary_path: Path,
    ) -> Dict[str, Any]:
        rows = list(history or [])
        status_key = _clean_str(status).lower()
        failed_index = next((idx for idx, row in enumerate(rows) if not bool((row or {}).get("ok", False))), len(rows))
        completed_count = max(0, min(failed_index, len(rows)))
        resume_available = status_key in {"failed", "incomplete", "handoff_required"} and (bool(rows) or latest_observation is not None)
        resume_strategy = "none"
        repair_from_index = max(0, min(failed_index, len(rows)))
        repair_actions: List[Dict[str, Any]] = []
        if resume_available:
            if status_key in {"incomplete", "handoff_required"}:
                resume_strategy = "resume_from_current_state"
                repair_from_index = len(rows)
            elif failed_index < len(rows):
                failed_row = dict(rows[failed_index] or {})
                failed_action = dict(failed_row.get("action", {}) or {})
                failed_type = _clean_str(failed_action.get("type")).lower()
                repair_from_index = failed_index
                resume_strategy = "resume_from_failed_step"
                if failed_type in _STATEFUL_DESKTOP_ACTION_TYPES:
                    verified_index = self._find_last_verified_index(rows, failed_index)
                    if 0 <= verified_index < failed_index:
                        repair_from_index = verified_index
                        resume_strategy = "resume_from_last_verified_step"
                    else:
                        context_index = self._find_last_context_index(rows, failed_index)
                        if 0 <= context_index < failed_index:
                            repair_from_index = context_index
                            resume_strategy = "repair_desktop_context"
                elif failed_type in _DESKTOP_CONTEXT_ACTION_TYPES:
                    resume_strategy = "retry_from_failed_context_step"
                repair_actions = [
                    dict((row.get("action") or {}) or {})
                    for row in rows[repair_from_index : failed_index + 1]
                    if isinstance((row.get("action") or {}), dict)
                ]
            else:
                resume_strategy = "resume_from_current_state"
                repair_from_index = len(rows)
        payload = {
            "version": 1,
            "ts": time.time(),
            "query": goal,
            "status": status_key or "unknown",
            "reason": reason,
            "resume_available": bool(resume_available),
            "resume_strategy": resume_strategy,
            "resume_from_index": int(max(0, min(failed_index, len(rows)))),
            "completed_step_count": int(completed_count),
            "step_count": int(len(rows)),
            "history": [
                {
                    "step": int(row.get("step", 0) or 0),
                    "action": dict(row.get("action", {}) or {}),
                    "ok": bool(row.get("ok", False)),
                    "verified": bool(row.get("verified", False)),
                    "note": _clean_str(row.get("note")),
                    "error": _clean_str(row.get("error")),
                    "replayed": bool(row.get("replayed", False)),
                }
                for row in rows
            ],
            "repair_from_index": int(repair_from_index),
            "repair_action_count": int(len(repair_actions)),
            "repair_actions": repair_actions,
            "last_action": dict((rows[-1].get("action") or {}) or {}) if rows else {},
            "latest_active_window": dict((latest_observation or {}).get("active_window", {}) or {}),
            "latest_controls_window": _clean_str((latest_observation or {}).get("controls_window")),
            "latest_nonempty_blocks": list((latest_observation or {}).get("nonempty_blocks", []) or [])[:8],
            "run_dir": run_dir.as_posix(),
            "trace_path": trace_path.as_posix(),
            "summary_path": summary_path.as_posix(),
            "suggested_command": "resume last computer-use task" if resume_available else "",
        }
        return payload

    def _save_recovery_payload(self, payload: Dict[str, Any], run_dir: Path) -> tuple[str, str]:
        run_recovery_path = run_dir / "recovery.json"
        _write_json(run_recovery_path, payload)
        _write_json(self.recovery_path, payload)
        return self.recovery_path.as_posix(), run_recovery_path.as_posix()

    def _build_run_summary(
        self,
        *,
        goal: str,
        status: str,
        ok: bool,
        started_at: float,
        run_dir: Path,
        trace_path: Path,
        summary_path: Path,
        history: Sequence[Dict[str, Any]],
        latest_observation: Dict[str, Any] | None = None,
        completion_summary: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        rows = list(history or [])
        action_counts: Dict[str, int] = {}
        unique_windows: List[str] = []
        for row in rows:
            action = dict(row.get("action", {}) or {})
            action_type = _clean_str(action.get("type"))
            if action_type:
                action_counts[action_type] = int(action_counts.get(action_type, 0) or 0) + 1
            for candidate in (
                _clean_str(action.get("window")),
                _clean_str(action.get("title")),
                _clean_str(((row.get("observation") or {}).get("active_window") or {}).get("title")),
                _clean_str((row.get("observation") or {}).get("controls_window")),
            ):
                if candidate and candidate not in unique_windows:
                    unique_windows.append(candidate)
        verified_steps = sum(1 for row in rows if bool(row.get("verified", False)))
        failed_steps = sum(1 for row in rows if not bool(row.get("ok", False)))
        recent_steps: List[Dict[str, Any]] = []
        for row in rows[-5:]:
            action = dict(row.get("action", {}) or {})
            recent_steps.append(
                {
                    "step": int(row.get("step", 0) or 0),
                    "action_type": _clean_str(action.get("type")),
                    "target": (
                        _clean_str(action.get("window"))
                        or _clean_str(action.get("title"))
                        or _clean_str(action.get("control"))
                        or _clean_str(action.get("text"))
                        or _clean_str(action.get("block"))
                    ),
                    "ok": bool(row.get("ok", False)),
                    "verified": bool(row.get("verified", False)),
                    "note": _truncate(_clean_str(row.get("note")), 120),
                    "error": _truncate(_clean_str(row.get("error")), 120),
                }
            )
        latest = dict(latest_observation or {})
        latest_active_window = dict(latest.get("active_window", {}) or {})
        latest_controls_window = _clean_str(latest.get("controls_window"))
        latest_nonempty_blocks = list(latest.get("nonempty_blocks", []) or [])[:8]
        last_action = {}
        if rows:
            last_action = dict((rows[-1].get("action") or {}) or {})
        result = {
            "ok": bool(ok),
            "status": status,
            "goal": goal,
            "summary": completion_summary or "",
            "reason": reason or "",
            "step_count": int(len(rows)),
            "verified_step_count": int(verified_steps),
            "failed_step_count": int(failed_steps),
            "verification_rate": _round_ratio(verified_steps, len(rows)),
            "elapsed_s": round(max(0.0, time.time() - started_at), 3),
            "action_counts": action_counts,
            "unique_windows": unique_windows[:12],
            "last_action": last_action,
            "recent_steps": recent_steps,
            "latest_active_window": latest_active_window,
            "latest_controls_window": latest_controls_window,
            "latest_nonempty_blocks": latest_nonempty_blocks,
            "latest_surface_mode": _clean_str(latest.get("surface_mode")),
            "latest_special_drawn_ui_likely": bool(latest.get("special_drawn_ui_likely", False)),
            "latest_targeting_hint": _clean_str(latest.get("targeting_hint")),
            "latest_screen_path": _clean_str(latest.get("screen_path")),
            "latest_grid_path": _clean_str(latest.get("grid_path")),
            "run_dir": run_dir.as_posix(),
            "trace_path": trace_path.as_posix(),
            "summary_path": summary_path.as_posix(),
        }
        return result

    def _decorate_result(
        self,
        *,
        result: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
        resume_mode: bool,
        resume_source_path: str,
        resume_strategy_requested: str,
        recovery_payload: Dict[str, Any],
        fixed_recovery_path: str,
        run_recovery_path: str,
        handoff_required: bool = False,
        handoff_reason: str = "",
        handoff_reason_code: str = "",
    ) -> Dict[str, Any]:
        result["resume_mode"] = bool(resume_mode)
        result["resume_source_path"] = resume_source_path
        result["resume_requested_strategy"] = resume_strategy_requested
        result["resume_replayed_action_count"] = int(sum(1 for row in history if bool((row or {}).get("replayed", False))))
        result["resume_available"] = bool(recovery_payload.get("resume_available", False))
        result["resume_strategy"] = _clean_str(recovery_payload.get("resume_strategy"))
        result["suggested_resume_command"] = _clean_str(recovery_payload.get("suggested_command"))
        result["recovery_path"] = fixed_recovery_path
        result["run_recovery_path"] = run_recovery_path
        result["handoff_required"] = bool(handoff_required)
        result["handoff_reason"] = _clean_str(handoff_reason)
        result["handoff_reason_code"] = _clean_str(handoff_reason_code)
        return result

    def run(self, goal: str | None = None) -> Dict[str, Any]:
        requested_goal = _clean_str(goal or self.config.goal)
        resume_payload = None
        resume_mode = False
        resume_source_path = ""
        resume_strategy_requested = ""
        replay_actions: List[ComputerAction] = []
        if _is_resume_request(requested_goal):
            resume_payload = _load_json_dict(self.recovery_path)
            if not resume_payload or not bool(resume_payload.get("resume_available", False)):
                raise RuntimeError("no resumable computer-use task found")
            resolved_goal = _clean_str(resume_payload.get("query")) or requested_goal
            resume_mode = True
            resume_source_path = self.recovery_path.as_posix()
            resume_strategy_requested = _clean_str(resume_payload.get("resume_strategy"))
            for item in list(resume_payload.get("repair_actions", []) or []):
                if not isinstance(item, dict):
                    continue
                replay_actions.append(self._action_from_dict(item))
        else:
            resolved_goal = requested_goal
        if not resolved_goal:
            raise ValueError("computer-use goal is required")
        if self.llm_client is None or not self.llm_client.available():
            raise RuntimeError(
                "computer-use runtime requires an available LLM client. "
                "Configure --cloud-llm / --v2-llm-provider and an API key."
            )
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.trace_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = run_dir / "trace.jsonl"
        summary_path = run_dir / "summary.json"
        preferred_window = _clean_str(self.config.target_window)
        history: List[Dict[str, Any]] = []
        previous_sha1 = ""
        started_at = time.time()
        fixed_recovery_path = self.recovery_path.as_posix()
        run_recovery_path = (run_dir / "recovery.json").as_posix()
        total_budget = max(1, int(self.config.max_steps)) + len(replay_actions)
        if resume_mode:
            _append_jsonl(
                trace_path,
                {
                    "kind": "resume_loaded",
                    "goal": resolved_goal,
                    "resume_strategy": resume_strategy_requested,
                    "resume_source_path": resume_source_path,
                    "replay_action_count": len(replay_actions),
                },
            )
        for step_index in range(1, total_budget + 1):
            observation = self._observe(step_index=step_index, run_dir=run_dir, preferred_window=preferred_window)
            same_screen = bool(previous_sha1 and previous_sha1 == _clean_str(observation.get("screen_sha1")))
            previous_sha1 = _clean_str(observation.get("screen_sha1"))
            if not replay_actions:
                handoff_signal = self._detect_handoff_from_observation(goal=resolved_goal, observation=observation)
                if handoff_signal is not None:
                    recovery_payload = self._build_recovery_payload(
                        goal=resolved_goal,
                        status="handoff_required",
                        reason=_clean_str(handoff_signal.get("reason")),
                        history=history,
                        latest_observation=observation,
                        run_dir=run_dir,
                        trace_path=trace_path,
                        summary_path=summary_path,
                    )
                    fixed_recovery_path, run_recovery_path = self._save_recovery_payload(recovery_payload, run_dir)
                    result = self._build_run_summary(
                        goal=resolved_goal,
                        status="handoff_required",
                        ok=False,
                        started_at=started_at,
                        run_dir=run_dir,
                        trace_path=trace_path,
                        summary_path=summary_path,
                        history=history,
                        latest_observation=observation,
                        reason=_clean_str(handoff_signal.get("reason")),
                    )
                    self._decorate_result(
                        result=result,
                        history=history,
                        resume_mode=resume_mode,
                        resume_source_path=resume_source_path,
                        resume_strategy_requested=resume_strategy_requested,
                        recovery_payload=recovery_payload,
                        fixed_recovery_path=fixed_recovery_path,
                        run_recovery_path=run_recovery_path,
                        handoff_required=True,
                        handoff_reason=_clean_str(handoff_signal.get("reason")),
                        handoff_reason_code=_clean_str(handoff_signal.get("reason_code")),
                    )
                    _append_jsonl(
                        trace_path,
                        {
                            "kind": "handoff_required",
                            "step": step_index,
                            "reason": _clean_str(handoff_signal.get("reason")),
                            "reason_code": _clean_str(handoff_signal.get("reason_code")),
                            "observation": observation,
                        },
                    )
                    _write_json(summary_path, result)
                    return result
            replayed = False
            if replay_actions:
                action = replay_actions.pop(0)
                raw_response = json.dumps(
                    {
                        "type": action.type,
                        "reason": action.reason or "resume replay action",
                        "recovery_replay": True,
                        "payload": action.to_dict(),
                    },
                    ensure_ascii=False,
                )
                replayed = True
            else:
                try:
                    action, raw_response = self._decide_action(
                        goal=resolved_goal,
                        observation=observation,
                        history=history,
                    )
                except Exception as exc:
                    recovery_payload = self._build_recovery_payload(
                        goal=resolved_goal,
                        status="failed",
                        reason=str(exc),
                        history=history,
                        latest_observation=observation,
                        run_dir=run_dir,
                        trace_path=trace_path,
                        summary_path=summary_path,
                    )
                    fixed_recovery_path, run_recovery_path = self._save_recovery_payload(recovery_payload, run_dir)
                    result = self._build_run_summary(
                        goal=resolved_goal,
                        status="failed",
                        ok=False,
                        started_at=started_at,
                        run_dir=run_dir,
                        trace_path=trace_path,
                        summary_path=summary_path,
                        history=history,
                        latest_observation=observation,
                        reason=str(exc),
                    )
                    self._decorate_result(
                        result=result,
                        history=history,
                        resume_mode=resume_mode,
                        resume_source_path=resume_source_path,
                        resume_strategy_requested=resume_strategy_requested,
                        recovery_payload=recovery_payload,
                        fixed_recovery_path=fixed_recovery_path,
                        run_recovery_path=run_recovery_path,
                    )
                    _append_jsonl(trace_path, {"kind": "decision_error", "step": step_index, "error": str(exc)})
                    _write_json(summary_path, result)
                    return result
            risk_signal = self._classify_high_risk_action(goal=resolved_goal, action=action, observation=observation)
            if risk_signal is not None:
                recovery_payload = self._build_recovery_payload(
                    goal=resolved_goal,
                    status="handoff_required",
                    reason=_clean_str(risk_signal.get("reason")),
                    history=history,
                    latest_observation=observation,
                    run_dir=run_dir,
                    trace_path=trace_path,
                    summary_path=summary_path,
                )
                fixed_recovery_path, run_recovery_path = self._save_recovery_payload(recovery_payload, run_dir)
                result = self._build_run_summary(
                    goal=resolved_goal,
                    status="handoff_required",
                    ok=False,
                    started_at=started_at,
                    run_dir=run_dir,
                    trace_path=trace_path,
                    summary_path=summary_path,
                    history=history,
                    latest_observation=observation,
                    reason=_clean_str(risk_signal.get("reason")),
                )
                self._decorate_result(
                    result=result,
                    history=history,
                    resume_mode=resume_mode,
                    resume_source_path=resume_source_path,
                    resume_strategy_requested=resume_strategy_requested,
                    recovery_payload=recovery_payload,
                    fixed_recovery_path=fixed_recovery_path,
                    run_recovery_path=run_recovery_path,
                    handoff_required=True,
                    handoff_reason=_clean_str(risk_signal.get("reason")),
                    handoff_reason_code=_clean_str(risk_signal.get("reason_code")),
                )
                _append_jsonl(
                    trace_path,
                    {
                        "kind": "handoff_required",
                        "step": step_index,
                        "reason": _clean_str(risk_signal.get("reason")),
                        "reason_code": _clean_str(risk_signal.get("reason_code")),
                        "action": action.to_dict(),
                        "observation": observation,
                    },
                )
                _write_json(summary_path, result)
                return result
            validation_signal = self._validate_runtime_action(
                goal=resolved_goal,
                action=action,
                observation=observation,
                history=history,
            )
            if validation_signal is not None:
                step_row = {
                    "step": step_index,
                    "action": action.to_dict(),
                    "raw_response": raw_response,
                    "ok": False,
                    "verified": False,
                    "note": "",
                    "error": _clean_str(validation_signal.get("reason")),
                    "replayed": replayed,
                    "same_screen_as_previous": same_screen,
                    "observation": {
                        "active_window": dict(observation.get("active_window", {}) or {}),
                        "controls_window": _clean_str(observation.get("controls_window")),
                        "surface_mode": _clean_str(observation.get("surface_mode")),
                        "special_drawn_ui_likely": bool(observation.get("special_drawn_ui_likely", False)),
                        "targeting_hint": _clean_str(observation.get("targeting_hint")),
                        "screen_path": _clean_str(observation.get("screen_path")),
                        "grid_path": _clean_str(observation.get("grid_path")),
                        "nonempty_blocks": list(observation.get("nonempty_blocks", []) or [])[:12],
                    },
                    "execution": {
                        "plugin": "runtime_guard",
                        "guard": True,
                        "ok": False,
                        "verified": False,
                        "error": _clean_str(validation_signal.get("reason")),
                        "reason_code": _clean_str(validation_signal.get("reason_code")),
                    },
                }
                history.append(step_row)
                _append_jsonl(trace_path, {"kind": "step", **step_row})
                time.sleep(max(0.0, float(self.config.step_delay_s)))
                continue
            if action.type == "handoff":
                recovery_payload = self._build_recovery_payload(
                    goal=resolved_goal,
                    status="handoff_required",
                    reason=action.reason or "human action required",
                    history=history,
                    latest_observation=observation,
                    run_dir=run_dir,
                    trace_path=trace_path,
                    summary_path=summary_path,
                )
                fixed_recovery_path, run_recovery_path = self._save_recovery_payload(recovery_payload, run_dir)
                result = self._build_run_summary(
                    goal=resolved_goal,
                    status="handoff_required",
                    ok=False,
                    started_at=started_at,
                    run_dir=run_dir,
                    trace_path=trace_path,
                    summary_path=summary_path,
                    history=history,
                    latest_observation=observation,
                    reason=action.reason or "human action required",
                )
                self._decorate_result(
                    result=result,
                    history=history,
                    resume_mode=resume_mode,
                    resume_source_path=resume_source_path,
                    resume_strategy_requested=resume_strategy_requested,
                    recovery_payload=recovery_payload,
                    fixed_recovery_path=fixed_recovery_path,
                    run_recovery_path=run_recovery_path,
                    handoff_required=True,
                    handoff_reason=action.reason or "human action required",
                    handoff_reason_code="model_requested_handoff",
                )
                _append_jsonl(
                    trace_path,
                    {
                        "kind": "handoff_required",
                        "step": step_index,
                        "observation": observation,
                        "raw_response": raw_response,
                        "action": action.to_dict(),
                    },
                )
                _write_json(summary_path, result)
                return result
            if action.type == "done":
                recovery_payload = self._build_recovery_payload(
                    goal=resolved_goal,
                    status="done",
                    reason="",
                    history=history,
                    latest_observation=observation,
                    run_dir=run_dir,
                    trace_path=trace_path,
                    summary_path=summary_path,
                )
                fixed_recovery_path, run_recovery_path = self._save_recovery_payload(recovery_payload, run_dir)
                result = self._build_run_summary(
                    goal=resolved_goal,
                    status="done",
                    ok=True,
                    started_at=started_at,
                    run_dir=run_dir,
                    trace_path=trace_path,
                    summary_path=summary_path,
                    history=history,
                    latest_observation=observation,
                    completion_summary=action.summary or action.reason or "goal completed",
                )
                self._decorate_result(
                    result=result,
                    history=history,
                    resume_mode=resume_mode,
                    resume_source_path=resume_source_path,
                    resume_strategy_requested=resume_strategy_requested,
                    recovery_payload=recovery_payload,
                    fixed_recovery_path=fixed_recovery_path,
                    run_recovery_path=run_recovery_path,
                )
                _append_jsonl(
                    trace_path,
                    {
                        "kind": "done",
                        "step": step_index,
                        "observation": observation,
                        "raw_response": raw_response,
                        "action": action.to_dict(),
                    },
                )
                _write_json(summary_path, result)
                return result
            if action.type == "fail":
                recovery_payload = self._build_recovery_payload(
                    goal=resolved_goal,
                    status="failed",
                    reason=action.reason or "model declared failure",
                    history=history,
                    latest_observation=observation,
                    run_dir=run_dir,
                    trace_path=trace_path,
                    summary_path=summary_path,
                )
                fixed_recovery_path, run_recovery_path = self._save_recovery_payload(recovery_payload, run_dir)
                result = self._build_run_summary(
                    goal=resolved_goal,
                    status="failed",
                    ok=False,
                    started_at=started_at,
                    run_dir=run_dir,
                    trace_path=trace_path,
                    summary_path=summary_path,
                    history=history,
                    latest_observation=observation,
                    reason=action.reason or "model declared failure",
                )
                self._decorate_result(
                    result=result,
                    history=history,
                    resume_mode=resume_mode,
                    resume_source_path=resume_source_path,
                    resume_strategy_requested=resume_strategy_requested,
                    recovery_payload=recovery_payload,
                    fixed_recovery_path=fixed_recovery_path,
                    run_recovery_path=run_recovery_path,
                )
                _append_jsonl(
                    trace_path,
                    {
                        "kind": "fail",
                        "step": step_index,
                        "observation": observation,
                        "raw_response": raw_response,
                        "action": action.to_dict(),
                    },
                )
                _write_json(summary_path, result)
                return result
            execution = self._execute_action(action, observation)
            if action.type == "focus_window" and bool(execution.get("ok", False)):
                preferred_window = action.title or action.window or preferred_window
            elif action.window:
                preferred_window = action.window
            step_row = {
                "step": step_index,
                "action": action.to_dict(),
                "raw_response": raw_response,
                "ok": bool(execution.get("ok", execution.get("verified", False))),
                "verified": bool(execution.get("verified", False)),
                "note": _clean_str(execution.get("note")),
                "error": _clean_str(execution.get("error")),
                "replayed": replayed,
                "same_screen_as_previous": same_screen,
                "observation": {
                    "active_window": dict(observation.get("active_window", {}) or {}),
                    "controls_window": _clean_str(observation.get("controls_window")),
                    "surface_mode": _clean_str(observation.get("surface_mode")),
                    "special_drawn_ui_likely": bool(observation.get("special_drawn_ui_likely", False)),
                    "targeting_hint": _clean_str(observation.get("targeting_hint")),
                    "screen_path": _clean_str(observation.get("screen_path")),
                    "grid_path": _clean_str(observation.get("grid_path")),
                    "nonempty_blocks": list(observation.get("nonempty_blocks", []) or [])[:12],
                },
                "execution": dict(execution or {}),
            }
            history.append(step_row)
            _append_jsonl(trace_path, {"kind": "step", **step_row})
            time.sleep(max(0.0, float(self.config.step_delay_s)))
        latest_observation = observation if "observation" in locals() else None
        recovery_payload = self._build_recovery_payload(
            goal=resolved_goal,
            status="incomplete",
            reason="max steps reached before the model declared done",
            history=history,
            latest_observation=latest_observation,
            run_dir=run_dir,
            trace_path=trace_path,
            summary_path=summary_path,
        )
        fixed_recovery_path, run_recovery_path = self._save_recovery_payload(recovery_payload, run_dir)
        result = self._build_run_summary(
            goal=resolved_goal,
            status="incomplete",
            ok=False,
            started_at=started_at,
            run_dir=run_dir,
            trace_path=trace_path,
            summary_path=summary_path,
            history=history,
            latest_observation=latest_observation,
            reason="max steps reached before the model declared done",
        )
        self._decorate_result(
            result=result,
            history=history,
            resume_mode=resume_mode,
            resume_source_path=resume_source_path,
            resume_strategy_requested=resume_strategy_requested,
            recovery_payload=recovery_payload,
            fixed_recovery_path=fixed_recovery_path,
            run_recovery_path=run_recovery_path,
            handoff_required=True,
            handoff_reason="max steps reached before the model declared done",
            handoff_reason_code="max_steps_reached",
        )
        _append_jsonl(trace_path, {"kind": "incomplete", "history": history[-5:]})
        _write_json(summary_path, result)
        return result


from system.computer_use._runtime_entry import (
    build_computer_use_runtime,
    run_computer_use_runtime,
)


__all__ = [
    "ComputerAction",
    "ComputerUseConfig",
    "ComputerUseRuntime",
    "DEFAULT_COMPUTER_USE_RECOVERY_PATH",
    "build_computer_use_runtime",
    "run_computer_use_runtime",
]
