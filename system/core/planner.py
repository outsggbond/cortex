from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import logging
import os
import re
import shlex

logger = logging.getLogger(__name__)


def _strip_wrapped_quotes(text: str) -> str:
    value = str(text or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'", "`"}:
        return value[1:-1]
    return value


@dataclass
class Task:
    name: str
    detail: str
    priority: int = 1
    status: str = "pending"
    payload: dict | None = None


@dataclass
class Plan:
    goal: str
    steps: List[Task]
    status: str = "active"
    levels: List[List[Task]] | None = None


@dataclass
class PlanCandidate:
    plan: Plan
    score: float
    rationale: str
    raw: str = ""


class TaskPlanner:
    def __init__(self):
        self._cloud_model = None
        self._cloud_model_checked = False
        self.action_keywords = (
            "read",
            "write",
            "run",
            "check",
            "verify",
            "analyze",
            "summarize",
            "implement",
            "fix",
            "optimize",
            "test",
            "diagnose",
            "review",
            "list",
            "search",
            "find",
            "mkdir",
            "append",
            "copy",
            "move",
            "rename",
            "delete",
            "remove",
            "rm",
            "touch",
            "chmod",
            "设计",
            "修复",
            "优化",
            "测试",
            "验证",
            "分析",
            "总结",
            "对比",
            "生成",
            "检查",
            "运行",
            "调试",
            "列出",
            "搜索",
            "查找",
            "创建",
            "新建",
            "追加",
            "复制",
            "移动",
            "重命名",
            "权限",
        )

    def _resolve_model(self, model=None):
        if model is not None:
            return model
        if self._cloud_model_checked:
            return self._cloud_model
        self._cloud_model_checked = True
        try:
            from system.computer_use.cloud_text_generator import build_cloud_text_generator_from_env  # noqa: PLC0415
            self._cloud_model = build_cloud_text_generator_from_env()
        except Exception:
            self._cloud_model = None
        return self._cloud_model

    def plan(
        self,
        message: str,
        model=None,
        memories: Optional[List[str]] = None,
        feedback: str = "",
        max_candidates: int = 3,
        previous_plan: Optional[Plan] = None,
    ) -> Plan:
        use_llm = os.environ.get("PLANNER_USE_LLM", "1") != "0"
        model = self._resolve_model(model)
        if model is not None and hasattr(model, "generate") and use_llm:
            candidates = self._plan_with_llm(
                message,
                model,
                memories or [],
                feedback=feedback,
                max_candidates=max_candidates,
                previous_plan=previous_plan,
            )
            best = self._select_best(candidates, message, feedback=feedback)
            if best is not None:
                return best.plan
        return self._plan_by_split(message)

    def _plan_by_split(self, message: str) -> Plan:
        t = message.strip()
        if not t:
            return Plan(goal="", steps=[], levels=[])
        tasks: List[Task] = []
        levels: List[List[Task]] = []
        phases = (
            t.replace("then", "|")
            .replace("next", "|")
            .replace("finally", "|")
            .split("|")
        )
        for phase in phases:
            phase = phase.strip()
            if not phase:
                continue
            phase_tasks: List[Task] = []
            for part in phase.split(","):
                p = part.strip()
                if p:
                    task = self._parse_task(p)
                    phase_tasks.append(task)
                    tasks.append(task)
            if phase_tasks:
                levels.append(phase_tasks)
        if not tasks:
            task = self._parse_task(t)
            tasks.append(task)
            levels.append([task])
        return Plan(goal=t, steps=tasks, levels=levels)

    def _parse_task(self, text: str) -> Task:
        t = text.strip()
        t, control_payload = self._extract_computer_control_hints(t)
        low = t.lower()
        drag_match = re.match(r"^drag\s+(-?\d+)\s+(-?\d+)\s*->\s*(-?\d+)\s+(-?\d+)(?:\s+duration\s+([0-9.]+))?$", low)
        click_match = re.match(r"^(double click|click)\s+(-?\d+)\s+(-?\d+)(?:\s+(left|right|middle))?$", low)
        if low == "browser close":
            return Task(name="plugin:browser_dom_close", detail=t, payload={**{"session": "default"}, **control_payload})
        if low == "browser screenshot" or low.startswith("browser screenshot "):
            payload = {"session": "default"}
            if low.startswith("browser screenshot "):
                payload["path"] = t[len("browser screenshot ") :].strip()
            return Task(name="plugin:browser_dom_screenshot", detail=t, payload={**payload, **control_payload})
        if low.startswith("browser open "):
            return Task(name="plugin:browser_dom_open", detail=t, payload={**{"url": t[13:].strip(), "session": "default"}, **control_payload})
        if low.startswith("browser click "):
            body = t[14:].strip()
            url = ""
            if " in http" in body.lower():
                parts = re.split(r"\s+in\s+(https?://\S+)\s*$", body, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 3:
                    body, url = parts[0].strip(), parts[1].strip()
            selector = body.strip().strip("`").strip("\"").strip("'")
            payload = {"selector": selector, "session": "default"}
            if url:
                payload["url"] = url
            return Task(name="plugin:browser_dom_click", detail=t, payload={**payload, **control_payload})
        if low.startswith("browser type ") and " into " in low:
            match = re.match(
                r'^browser type\s+(".*?"|\'.*?\'|`.*?`|.+?)\s+into\s+(".*?"|\'.*?\'|`.*?`|.+?)(?:\s+in\s+(https?://\S+))?$',
                t,
                flags=re.IGNORECASE,
            )
            if match:
                text_value = str(match.group(1) or "").strip().strip("`").strip("\"").strip("'")
                selector = str(match.group(2) or "").strip().strip("`").strip("\"").strip("'")
                payload = {"text": text_value, "selector": selector, "session": "default"}
                if str(match.group(3) or "").strip():
                    payload["url"] = str(match.group(3) or "").strip()
                return Task(name="plugin:browser_dom_type", detail=t, payload={**payload, **control_payload})
        if low.startswith("browser text ") or low.startswith("browser extract "):
            body = t[13:].strip() if low.startswith("browser text ") else t[16:].strip()
            url = ""
            if " in http" in body.lower():
                parts = re.split(r"\s+in\s+(https?://\S+)\s*$", body, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 3:
                    body, url = parts[0].strip(), parts[1].strip()
            selector = body.strip().strip("`").strip("\"").strip("'") or "body"
            payload = {"selector": selector, "session": "default"}
            if url:
                payload["url"] = url
            return Task(name="plugin:browser_dom_extract_text", detail=t, payload={**payload, **control_payload})
        if low.startswith("browser wait text "):
            body = t[18:].strip()
            url = ""
            if " in http" in body.lower():
                parts = re.split(r"\s+in\s+(https?://\S+)\s*$", body, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) == 3:
                    body, url = parts[0].strip(), parts[1].strip()
            text_value = body.strip().strip("`").strip("\"").strip("'")
            payload = {"text": text_value, "session": "default"}
            if url:
                payload["url"] = url
            return Task(name="plugin:browser_dom_wait_text", detail=t, payload={**payload, **control_payload})
        list_controls_match = re.match(
            r'^list controls\s+in\s+window\s+(".*?"|\'.*?\'|`.*?`|.+?)(?:\s+control\s+(".*?"|\'.*?\'|`.*?`|.+?))?(?:\s+control\s+type\s+(".*?"|\'.*?\'|`.*?`|.+?))?(?:\s+index\s+(\d+))?$',
            t,
            flags=re.IGNORECASE,
        )
        if list_controls_match:
            window_title = str(list_controls_match.group(1) or "").strip().strip("`").strip("\"").strip("'")
            control_name = str(list_controls_match.group(2) or "").strip().strip("`").strip("\"").strip("'")
            control_type = str(list_controls_match.group(3) or "").strip().strip("`").strip("\"").strip("'")
            payload = {"window": window_title}
            if control_name:
                payload["control"] = control_name
            if control_type:
                payload["control_type"] = control_type
            if str(list_controls_match.group(4) or "").strip():
                payload["index"] = int(str(list_controls_match.group(4) or "0"))
            return Task(name="plugin:desktop_list_controls", detail=t, payload={**payload, **control_payload})
        click_control_match = re.match(
            r'^click control\s+(".*?"|\'.*?\'|`.*?`|.+?)\s+in\s+window\s+(".*?"|\'.*?\'|`.*?`|.+?)(?:\s+control\s+type\s+(".*?"|\'.*?\'|`.*?`|.+?))?(?:\s+index\s+(\d+))?$',
            t,
            flags=re.IGNORECASE,
        )
        if click_control_match:
            control_name = str(click_control_match.group(1) or "").strip().strip("`").strip("\"").strip("'")
            window_title = str(click_control_match.group(2) or "").strip().strip("`").strip("\"").strip("'")
            control_type = str(click_control_match.group(3) or "").strip().strip("`").strip("\"").strip("'")
            payload = {"window": window_title, "control": control_name}
            if control_type:
                payload["control_type"] = control_type
            if str(click_control_match.group(4) or "").strip():
                payload["index"] = int(str(click_control_match.group(4) or "0"))
            return Task(name="plugin:desktop_click_control", detail=t, payload={**payload, **control_payload})
        type_control_match = re.match(
            r'^type\s+(".*?"|\'.*?\'|`.*?`|.+?)\s+into\s+control\s+(".*?"|\'.*?\'|`.*?`|.+?)\s+in\s+window\s+(".*?"|\'.*?\'|`.*?`|.+?)(?:\s+control\s+type\s+(".*?"|\'.*?\'|`.*?`|.+?))?(?:\s+index\s+(\d+))?(\s+clear\s+first)?$',
            t,
            flags=re.IGNORECASE,
        )
        if type_control_match:
            text_value = str(type_control_match.group(1) or "").strip().strip("`").strip("\"").strip("'")
            control_name = str(type_control_match.group(2) or "").strip().strip("`").strip("\"").strip("'")
            window_title = str(type_control_match.group(3) or "").strip().strip("`").strip("\"").strip("'")
            control_type = str(type_control_match.group(4) or "").strip().strip("`").strip("\"").strip("'")
            payload = {"window": window_title, "control": control_name, "text": text_value}
            if control_type:
                payload["control_type"] = control_type
            if str(type_control_match.group(5) or "").strip():
                payload["index"] = int(str(type_control_match.group(5) or "0"))
            if str(type_control_match.group(6) or "").strip():
                payload["clear_first"] = True
            return Task(name="plugin:desktop_type_control", detail=t, payload={**payload, **control_payload})
        if low == "ocr screen" or low.startswith("ocr "):
            payload = {}
            if low.startswith("ocr ") and low != "ocr screen":
                path = t[4:].strip()
                if path and path.lower() != "screen":
                    payload["path"] = path
            return Task(name="plugin:desktop_ocr", detail=t, payload={**payload, **control_payload})
        if low.startswith("click text "):
            return Task(name="plugin:desktop_click_text", detail=t, payload={**{"text": t[11:].strip().strip("`").strip("\"").strip("'")}, **control_payload})
        if click_match:
            clicks = 2 if str(click_match.group(1) or "").lower().startswith("double") else 1
            return Task(
                name="plugin:desktop_click",
                detail=t,
                payload={
                    "x": int(click_match.group(2)),
                    "y": int(click_match.group(3)),
                    "button": str(click_match.group(4) or "left"),
                    "clicks": clicks,
                    **control_payload,
                },
            )
        if drag_match:
            payload = {
                "x1": int(drag_match.group(1)),
                "y1": int(drag_match.group(2)),
                "x2": int(drag_match.group(3)),
                "y2": int(drag_match.group(4)),
            }
            if str(drag_match.group(5) or "").strip():
                payload["duration_s"] = float(drag_match.group(5))
            return Task(name="plugin:desktop_drag", detail=t, payload={**payload, **control_payload})
        if low.startswith(("open url ", "browse ", "visit ", "go to ", "navigate ")):
            if low.startswith("open url "):
                url = t[9:].strip()
            elif low.startswith("browse "):
                url = t[7:].strip()
            elif low.startswith("visit "):
                url = t[6:].strip()
            elif low.startswith("go to "):
                url = t[6:].strip()
            else:
                url = t[9:].strip()
            return Task(name="plugin:browser_open_url", detail=t, payload={**{"url": url}, **control_payload})
        if low == "list windows" or low.startswith("list windows "):
            return Task(name="plugin:desktop_list_windows", detail=t, payload=dict(control_payload))
        if low.startswith("focus window "):
            return Task(name="plugin:desktop_focus_window", detail=t, payload={**{"title": t[13:].strip()}, **control_payload})
        if low.startswith("launch "):
            return Task(name="plugin:desktop_launch", detail=t, payload={**{"target": t[7:].strip()}, **control_payload})
        if low.startswith("hotkey "):
            return Task(name="plugin:desktop_hotkey", detail=t, payload={**{"hotkey": t[7:].strip()}, **control_payload})
        if low.startswith("type "):
            return Task(
                name="plugin:desktop_type_text",
                detail=t,
                payload={**{"text": _strip_wrapped_quotes(t[5:])}, **control_payload},
            )
        if low == "screenshot" or low.startswith("screenshot "):
            return Task(name="plugin:desktop_screenshot", detail=t, payload=dict(control_payload))
        if t.startswith("run "):
            rest = t.split(" ", 1)[1].strip()
            parts = shlex.split(rest)
            if not parts:
                payload = {"path": rest}
                return Task(name="run_script", detail=t, payload=payload)
            path = parts[0]
            args = parts[1:]
            payload = {"path": path}
            if args:
                payload["args"] = args
            return Task(name="run_script", detail=t, payload=payload)
        if t.startswith("read "):
            payload = {"path": t.split(" ", 1)[1].strip()}
            return Task(name="read_file", detail=t, payload=payload)
        if t.startswith("write "):
            rest = t.split(" ", 1)[1].strip()
            if "->" in rest:
                path, content = rest.split("->", 1)
                payload = {"path": path.strip(), "content": content.strip()}
            else:
                payload = {"path": rest.strip(), "content": ""}
            return Task(name="write_file", detail=t, payload=payload)
        if t.startswith("append "):
            rest = t.split(" ", 1)[1].strip()
            if "->" in rest:
                path, content = rest.split("->", 1)
                payload = {"path": path.strip(), "content": content.strip()}
            else:
                payload = {"path": rest.strip(), "content": ""}
            return Task(name="append_file", detail=t, payload=payload)
        if t.startswith("chmod "):
            rest = t.split(" ", 1)[1].strip()
            parts = shlex.split(rest)
            path = ""
            mode = ""
            if len(parts) >= 2:
                path = parts[0]
                mode = parts[1]
            elif " " in rest:
                path, mode = rest.split(" ", 1)
            else:
                path = rest
            return Task(name="chmod", detail=t, payload={"path": path.strip(), "mode": mode.strip()})
        if t.startswith("touch "):
            payload = {"path": t.split(" ", 1)[1].strip()}
            return Task(name="touch_file", detail=t, payload=payload)
        if t.startswith("list "):
            payload = {"path": t.split(" ", 1)[1].strip()}
            return Task(name="list_dir", detail=t, payload=payload)
        if t.startswith("exists "):
            payload = {"path": t.split(" ", 1)[1].strip()}
            return Task(name="check_exists", detail=t, payload=payload)
        if t.startswith("mkdir "):
            payload = {"path": t.split(" ", 1)[1].strip()}
            return Task(name="mkdir", detail=t, payload=payload)
        if t.startswith("copy "):
            rest = t.split(" ", 1)[1].strip()
            if "->" in rest:
                src, dst = rest.split("->", 1)
                payload = {"src": src.strip(), "dst": dst.strip()}
                return Task(name="copy_file", detail=t, payload=payload)
        if t.startswith("delete ") or t.startswith("rm ") or t.startswith("remove "):
            parts = t.split(" ", 1)
            path = parts[1].strip() if len(parts) > 1 else ""
            return Task(name="delete_file", detail=t, payload={"path": path})
        if t.startswith("move ") or t.startswith("rename "):
            rest = t.split(" ", 1)[1].strip()
            if "->" in rest:
                src, dst = rest.split("->", 1)
                payload = {"src": src.strip(), "dst": dst.strip()}
                return Task(name="move_file", detail=t, payload=payload)
        if low.startswith("search "):
            rest = t.split(" ", 1)[1].strip()
            if " in " in rest:
                pattern, path = rest.split(" in ", 1)
                payload = {"pattern": pattern.strip(), "path": path.strip()}
            else:
                payload = {"pattern": rest.strip(), "path": "."}
            return Task(name="search_text", detail=t, payload=payload)
        if low.startswith("find "):
            rest = t.split(" ", 1)[1].strip()
            if " in " in rest:
                pattern, path = rest.split(" in ", 1)
                payload = {"pattern": pattern.strip(), "path": path.strip()}
            else:
                payload = {"pattern": rest.strip(), "path": "."}
            return Task(name="search_files", detail=t, payload=payload)
        if "run tests" in low or "pytest" in low:
            payload = {"command": ""}
            return Task(name="run_tests", detail=t, payload=payload)
        if "generate code" in t:
            return Task(name="generate_code", detail=t, payload={"request": t})
        if t.startswith("rollback"):
            path = ""
            if " " in t:
                path = t.split(" ", 1)[1].strip()
            return Task(name="rollback", detail=t, payload={"path": path})

        # Chinese patterns
        if t.startswith(("列出", "查看目录", "列出文件")):
            path = t
            for p in ("列出", "查看目录", "列出文件"):
                if path.startswith(p):
                    path = path[len(p) :]
                    break
            path = path.strip() or "."
            return Task(name="list_dir", detail=t, payload={"path": path})
        if t.startswith(("搜索", "查找")):
            rest = t
            for p in ("搜索", "查找"):
                if rest.startswith(p):
                    rest = rest[len(p) :].strip()
                    break
            if "在" in rest:
                pattern, path = rest.split("在", 1)
                return Task(
                    name="search_text",
                    detail=t,
                    payload={"pattern": pattern.strip(), "path": path.strip()},
                )
            return Task(name="search_text", detail=t, payload={"pattern": rest, "path": "."})
        if t.startswith(("创建目录", "新建目录", "建立目录")):
            path = t
            for p in ("创建目录", "新建目录", "建立目录"):
                if path.startswith(p):
                    path = path[len(p) :]
                    break
            path = path.strip()
            return Task(name="mkdir", detail=t, payload={"path": path})
        if t.startswith(("创建文件", "新建文件", "生成文件")):
            path = t
            for p in ("创建文件", "新建文件", "生成文件"):
                if path.startswith(p):
                    path = path[len(p) :]
                    break
            path = path.strip()
            return Task(name="touch_file", detail=t, payload={"path": path})
        if t.startswith(("追加", "附加")):
            rest = t
            for p in ("追加", "附加"):
                if rest.startswith(p):
                    rest = rest[len(p) :].strip()
                    break
            if "->" in rest:
                path, content = rest.split("->", 1)
                return Task(
                    name="append_file",
                    detail=t,
                    payload={"path": path.strip(), "content": content.strip()},
                )
            return Task(name="append_file", detail=t, payload={"path": rest.strip(), "content": ""})
        if t.startswith(("复制", "拷贝")) and "->" in t:
            rest = t
            for p in ("复制", "拷贝"):
                if rest.startswith(p):
                    rest = rest[len(p) :].strip()
                    break
            src, dst = rest.split("->", 1)
            return Task(name="copy_file", detail=t, payload={"src": src.strip(), "dst": dst.strip()})
        if t.startswith(("移动", "重命名")) and "->" in t:
            rest = t
            for p in ("移动", "重命名"):
                if rest.startswith(p):
                    rest = rest[len(p) :].strip()
                    break
            src, dst = rest.split("->", 1)
            return Task(name="move_file", detail=t, payload={"src": src.strip(), "dst": dst.strip()})
        if t.startswith(("运行测试", "跑测试", "执行测试")):
            return Task(name="run_tests", detail=t, payload={"command": ""})
        return Task(name="step", detail=t, priority=1, status="pending")

    def _plan_with_llm(
        self,
        message: str,
        model,
        memories: List[str],
        feedback: str = "",
        max_candidates: int = 3,
        previous_plan: Optional[Plan] = None,
    ) -> List[PlanCandidate]:
        count = max(1, min(5, int(max_candidates or 1)))
        prior = ""
        if previous_plan and previous_plan.steps:
            prior_steps = "；".join([s.detail for s in previous_plan.steps][:6])
            prior = f"上一版计划：{prior_steps}"
        feedback_block = f"失败反馈：{feedback}" if feedback else ""
        prompt = (
            f"你是任务规划器。为用户问题生成 {count} 个可执行计划。\n"
            "要求：每个计划 2-6 步，每步以动词开头。\n"
            "输出格式（严格遵守）：\n"
            "PLAN 1:\n"
            "- step...\n"
            "PLAN 2:\n"
            "- step...\n"
            "只输出计划，不要解释。\n"
            f"用户问题：{message}\n"
            f"{prior}\n"
            f"{feedback_block}\n"
        ).strip()
        try:
            raw = model.generate(prompt, memories, raw=True) or ""
        except TypeError:
            raw = model.generate(prompt, memories) or ""
        blocks = self._extract_plan_blocks(raw)
        candidates: List[PlanCandidate] = []
        for block in blocks[:count]:
            steps = self._parse_steps(block)
            if not steps:
                continue
            tasks = [self._parse_task(s) for s in steps]
            levels = [[t] for t in tasks]
            plan = Plan(goal=message.strip(), steps=tasks, levels=levels)
            score, reason = self._score_plan(plan, message, feedback=feedback)
            candidates.append(PlanCandidate(plan=plan, score=score, rationale=reason, raw=block))
        return candidates

    def _extract_plan_blocks(self, text: str) -> List[str]:
        if not text:
            return []
        pattern = re.compile(r"(?i)plan\s*\d+\s*[:：]")
        matches = list(pattern.finditer(text))
        if not matches:
            return [text.strip()]
        blocks = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            if block:
                blocks.append(block)
        return blocks

    def _parse_steps(self, block: str) -> List[str]:
        if not block:
            return []
        steps: List[str] = []
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        for ln in lines:
            ln = re.sub(r"^[-*\d\.\)\s]+", "", ln).strip()
            if not ln:
                continue
            steps.append(ln)
        if steps:
            return steps
        # fallback: split by punctuation
        parts = re.split(r"[;,，。；\n]+", block)
        for p in parts:
            p = p.strip()
            if p:
                steps.append(p)
        return steps

    def _score_plan(self, plan: Plan, message: str, feedback: str = "") -> tuple[float, str]:
        if not plan.steps:
            return -1.0, "empty"
        score = 0.0
        reasons = []
        n = len(plan.steps)
        if 2 <= n <= 6:
            score += 0.4
            reasons.append("step_count_ok")
        elif n == 1:
            score += 0.1
            reasons.append("single_step")
        else:
            score -= 0.2
            reasons.append("too_many_steps")
        hit = 0
        for step in plan.steps:
            text = step.detail.lower()
            if any(k in text for k in self.action_keywords):
                hit += 1
        if hit:
            score += min(0.3, 0.06 * hit)
            reasons.append(f"action_hits={hit}")
        # semantic alignment
        try:
            from system.core.embeddings import encode_text, cosine, embedding_available

            if embedding_available():
                qv = encode_text(message, require_model=True)
                pv = encode_text(" ".join(s.detail for s in plan.steps), require_model=True)
                score += 0.6 * cosine(qv, pv)
                reasons.append("semantic_match")
        except Exception:
            logger.debug("planner: embedding scoring failed", exc_info=True)
        # penalize repeating failed hints
        if feedback:
            bad_terms = [t for t in re.split(r"[\s,，。；、]+", feedback) if len(t) >= 2]
            penalty = 0.0
            for term in bad_terms[:8]:
                if any(term in s.detail for s in plan.steps):
                    penalty += 0.05
            if penalty:
                score -= min(0.3, penalty)
                reasons.append("feedback_penalty")
        return score, ",".join(reasons)

    def _select_best(self, candidates: List[PlanCandidate], message: str, feedback: str = "") -> Optional[PlanCandidate]:
        if not candidates:
            return None
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[0]

    def _extract_computer_control_hints(self, text: str) -> tuple[str, dict]:
        node = str(text or "").strip()
        payload: dict = {}
        patterns = (
            (r"\s+retries\s+(\d+)\s*$", "max_retries", int),
            (r'\s+verify\s+not\s+text\s+(".*?"|\'.*?\'|`.*?`|.+?)\s*$', "verify_not_text", str),
            (r'\s+verify\s+text\s+(".*?"|\'.*?\'|`.*?`|.+?)\s*$', "verify_text", str),
            (r'\s+verify\s+window\s+(".*?"|\'.*?\'|`.*?`|.+?)\s*$', "verify_window_title", str),
            (r"\s+verify\s+change\s*$", "verify_change", bool),
        )
        changed = True
        while changed and node:
            changed = False
            for pattern, key, caster in patterns:
                match = re.search(pattern, node, flags=re.IGNORECASE)
                if not match:
                    continue
                raw = str(match.group(1) or "").strip() if match.groups() else ""
                if caster is bool:
                    payload[key] = True
                elif caster is int:
                    payload[key] = int(raw)
                else:
                    payload[key] = raw.strip("`").strip('"').strip("'")
                node = node[: match.start()].strip()
                changed = True
                break
        return node, payload
