# -*- coding: utf-8 -*-
"""Evidence validator — post-generation validation, grounding, clarification, feedback.

Extracted from ChatPipeline (pipeline.py) to improve cohesion.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from system.brain.persona import persona_prompt_reply
from system.l_utils.text_utils import contains_cjk, is_valid_reply, is_valid_rag_reply
from system.l_utils.tokens import code_tokens, path_tokens, salient_keys, time_tokens

from .router import RouteKind, RouteProfile
from .validation_feedback import ValidationFeedbackRecorder


class EvidenceValidator:
    """Validates LLM-generated answers against thread context and retrieved evidence."""

    def __init__(
        self,
        *,
        enable_post_validation: bool = True,
        enable_clarification: bool = True,
        enable_validation_feedback: bool = False,
        feedback_path: str = "artifacts/audit/chat_v2_validation_feedback.jsonl",
        stats_path: str = "artifacts/audit/chat_v2_validation_stats.json",
    ) -> None:
        self.enable_post_validation = bool(enable_post_validation)
        self.enable_clarification = bool(enable_clarification)
        self.feedback: ValidationFeedbackRecorder | None = None
        if enable_validation_feedback:
            try:
                self.feedback = ValidationFeedbackRecorder(
                    feedback_path=str(feedback_path or "artifacts/audit/chat_v2_validation_feedback.jsonl"),
                    stats_path=str(stats_path or "artifacts/audit/chat_v2_validation_stats.json"),
                )
            except Exception:
                self.feedback = None

    # -- evidence construction ---------------------------------------------------

    def evidence_blocks(
        self, memory_hits: Sequence[Any], rag_context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for hit in list(memory_hits or []):
            text = str(getattr(hit, "assistant", "") or "").strip()
            if not text:
                continue
            blocks.append({
                "kind": "memory", "text": text,
                "paths": path_tokens(text), "codes": code_tokens(text),
                "times": time_tokens(text), "keys": salient_keys(text),
            })
        if isinstance(rag_context, dict):
            for item in list(rag_context.get("snippets", []) or []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "") or "").strip()
                path_str = str(item.get("path", "") or "").strip()
                if not text:
                    continue
                blocks.append({
                    "kind": "rag", "text": text,
                    "paths": path_tokens(text) | path_tokens(path_str),
                    "codes": code_tokens(text), "times": time_tokens(text),
                    "keys": salient_keys(text),
                })
        return blocks

    def supported_by_evidence(self, answer: str, evidence_blocks: Sequence[Dict[str, Any]]) -> bool:
        if not evidence_blocks:
            return True
        answer_paths = path_tokens(answer)
        answer_codes = code_tokens(answer)
        answer_times = time_tokens(answer)
        answer_keys = salient_keys(answer)
        if not answer_paths and not answer_codes and not answer_times and not answer_keys:
            return False
        for block in list(evidence_blocks):
            block_paths = set(block.get("paths", set()))
            block_codes = set(block.get("codes", set()))
            block_times = set(block.get("times", set()))
            block_keys = set(block.get("keys", set()))
            if answer_paths and block_paths and (answer_paths & block_paths):
                return True
            if answer_codes and block_codes and (answer_codes & block_codes):
                return True
            if answer_times and block_times and (answer_times & block_times):
                return True
            overlap = answer_keys & block_keys
            if len(overlap) >= 2:
                return True
            if overlap and (len(overlap) / float(max(1, len(answer_keys)))) >= 0.45:
                return True
        return False

    def evidence_issue(
        self, answer: str, route: RouteProfile,
        memory_hits: Sequence[Any], rag_context: Optional[Dict[str, Any]],
    ) -> Dict[str, str] | None:
        evidence_blocks = self.evidence_blocks(memory_hits, rag_context)
        if not evidence_blocks:
            return None
        answer_paths = path_tokens(answer)
        answer_codes = code_tokens(answer)
        answer_times = time_tokens(answer)
        evidence_paths: set[str] = set()
        evidence_codes: set[str] = set()
        evidence_times: set[str] = set()
        for block in evidence_blocks:
            evidence_paths |= set(block.get("paths", set()))
            evidence_codes |= set(block.get("codes", set()))
            evidence_times |= set(block.get("times", set()))
        if answer_paths and evidence_paths and not (answer_paths & evidence_paths):
            return {"kind": "evidence_path_conflict"}
        if answer_codes and evidence_codes and not (answer_codes & evidence_codes):
            return {"kind": "evidence_code_conflict"}
        if answer_times and evidence_times and not (answer_times & evidence_times):
            return {"kind": "evidence_time_conflict"}
        if route.kind not in {RouteKind.PROJECT_QA, RouteKind.FACTUAL_QA, RouteKind.REASONING}:
            return None
        if not self.supported_by_evidence(answer, evidence_blocks):
            return {"kind": "unsupported_by_evidence"}
        return None

    # -- clarification -----------------------------------------------------------

    def clarify_text(
        self, query: str, route: RouteProfile, *, reasoning_out: Any | None = None,
    ) -> str:
        cjk = contains_cjk(query)
        text = str(query or "").strip()
        low = text.lower()
        reasoning_meta = (
            getattr(reasoning_out, "metadata", {})
            if reasoning_out is not None and isinstance(getattr(reasoning_out, "metadata", {}), dict)
            else {}
        )
        reasoning_action = str(reasoning_meta.get("action", "") or "").strip().lower()
        reasoning_followup = str(reasoning_meta.get("followup", "") or "").strip()

        if route.kind == RouteKind.WORKSPACE_ACTION:
            return "你要我看哪个文件或路径？" if cjk else "Which file or path should I inspect?"
        if route.kind == RouteKind.PROJECT_QA:
            if any(t in low for t in ("runtime", "flow", "request path", "agent loop", "memory", "rag")):
                return "你想看请求路径、检索链路，还是 agent 执行流？" if cjk else "Request path, retrieval chain, or agent loop?"
            if any(t in low for t in ("setup", "install", "config", "env", "environment")):
                return "你想先看安装步骤、环境变量，还是运行参数？" if cjk else "Install steps, environment variables, or runtime flags?"
            return "你想先看架构、运行流程，还是环境配置？" if cjk else "Architecture, runtime flow, or setup?"
        if route.kind == RouteKind.REASONING:
            if reasoning_action == "clarify_constraints":
                return "时限、安全，还是回滚，哪个优先？" if cjk else "Deadline, safety, or rollback: which one wins?"
            if reasoning_followup:
                if cjk:
                    return f"先说明：{reasoning_followup}？"
                if reasoning_followup.endswith("?"):
                    return reasoning_followup
                node = reasoning_followup[0].upper() + reasoning_followup[1:] if reasoning_followup else reasoning_followup
                return f"{node}?"
            if any(t in low for t in ("why", "cause", "reason", "error", "bug", "issue")):
                return "它是从哪个现象或变更开始的？" if cjk else "What changed right before it started?"
            if any(t in low for t in ("plan", "strategy", "approach", "roadmap")):
                return "你的成功标准和可接受的回滚窗口是什么？" if cjk else "What outcome and rollback window are you targeting?"
            return "你更看重速度、风险，还是回滚余地？" if cjk else "What matters most here: speed, risk, or rollback?"
        if route.kind == RouteKind.FACTUAL_QA:
            if "json" in low:
                if "file" in low or "文件" in text:
                    return "你用的是 Python、JavaScript，还是其他语言？" if cjk else "Are you using Python, JavaScript, or another language?"
                return "JSON 是来自文件、字符串，还是 HTTP 响应？" if cjk else "Is your JSON coming from a file, a string, or an HTTP response?"
            if any(t in low for t in ("error", "exception", "traceback", "bug")):
                return "把完整报错贴给我，可以吗？" if cjk else "What is the exact error text?"
            return "你用的是什么语言或运行环境？" if cjk else "What language or runtime are you using?"
        return "你想要的结果是什么？" if cjk else "What outcome are you aiming for?"

    # -- validation orchestration ------------------------------------------------

    def thread_conflict_reply(self, query: str, issue: Dict[str, str]) -> str:
        label = str(issue.get("label", "") or "").strip()
        value = str(issue.get("value", "") or "").strip()
        cjk = contains_cjk(query)
        if not value:
            return ""
        if cjk:
            if label:
                return f"你刚刚给的 {label} 是 {value}。我还要继续按这个方向取舍吗？"
            return f"我还需要对齐：{value}。你想优先按这个约束继续吗？"
        if label:
            return f"You said {label.lower()} is {value}. Should I keep optimizing around that?"
        return f"I still need to honor this context: {value}. Should I keep optimizing around it?"

    def grounded_fallback(
        self, query: str, route: RouteProfile,
        memory_hits: Sequence[Any], rag_context: Optional[Dict[str, Any]],
        rag_reply_fn,  # callable
    ) -> tuple[str, str] | None:
        rag_text = rag_reply_fn(query, rag_context)
        memory_text = ""
        if memory_hits:
            memory_text = str(getattr(memory_hits[0], "assistant", "") or "").strip()
        candidates: List[tuple[str, str]] = []
        if route.kind in {RouteKind.PROJECT_QA, RouteKind.WORKSPACE_ACTION}:
            if is_valid_rag_reply(rag_text):
                candidates.append((rag_text, "rag"))
            if is_valid_reply(memory_text):
                candidates.append((memory_text, "memory"))
        else:
            if is_valid_reply(memory_text):
                candidates.append((memory_text, "memory"))
            if is_valid_rag_reply(rag_text):
                candidates.append((rag_text, "rag"))
        for text, source in candidates:
            return text, source
        return None

    def post_validate_candidate(
        self, *, query: str, text: str, source: str,
        metadata: Dict[str, Any], route: RouteProfile,
        memory_hits: Sequence[Any], rag_context: Optional[Dict[str, Any]],
        thread_context: Dict[str, Any] | None,
        thread_context_tracker,  # ThreadContextTracker
        rag_reply_fn,
        reasoning_out: Any | None = None,
    ) -> tuple[str, str, Dict[str, Any]]:
        issue = self._validation_issue(
            text, source, route, memory_hits, rag_context,
            thread_context_tracker, thread_context,
        )
        if issue is None:
            out = dict(metadata or {})
            out["validation"] = "passed"
            return text, source, out

        issue_kind = str(issue.get("kind", "") or "").strip()
        if issue_kind.startswith("thread_"):
            thread_text = thread_context_tracker.thread_conflict_reply(query, issue).strip()  # use tracker's version
            # fallback to our version
            if not thread_text:
                thread_text = self.thread_conflict_reply(query, issue).strip()
            if thread_text and is_valid_reply(thread_text):
                out = thread_context_tracker.attach_thread_context(
                    {"validation": issue_kind, "validation_override": "clarify"},
                    thread_context, resolved_query=query,
                )
                return thread_text, "clarify", out

        grounded = self.grounded_fallback(query, route, memory_hits, rag_context, rag_reply_fn)
        if grounded is not None:
            grounded_text, grounded_source = grounded
            out = thread_context_tracker.attach_thread_context(
                {"validation": issue_kind, "validation_override": grounded_source},
                thread_context, resolved_query=query,
            )
            return grounded_text, grounded_source, out

        return persona_prompt_reply(), "fallback", {}

    def _validation_issue(
        self, answer: str, source: str, route: RouteProfile,
        memory_hits: Sequence[Any], rag_context: Optional[Dict[str, Any]],
        thread_context_tracker, thread_context: Dict[str, Any] | None,
    ) -> Dict[str, str] | None:
        if not bool(self.enable_post_validation):
            return None
        if not str(source or "").startswith(("llm", "reasoning")):
            return None
        route_kind_str = str(getattr(route.kind, "value", route.kind) or "unknown")
        issue = thread_context_tracker.thread_context_issue(answer, route_kind_str, thread_context)
        if issue is not None:
            return issue
        return self.evidence_issue(answer, route, memory_hits, rag_context)
