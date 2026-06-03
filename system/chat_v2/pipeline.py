# -*- coding: utf-8 -*-
"""Single-path response pipeline for chat runtime v2."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from system.brain.persona import normalize_persona_reply, persona_prompt_reply
from system.evaluation.response_guard import guard_response, is_low_signal_response
from system.l_utils.text_utils import contains_cjk, is_valid_reply, is_valid_rag_reply
from system.l_utils.tokens import (
    code_tokens,
    match_key,
    normalize_structured_token,
    path_tokens,
    salient_keys,
    salient_tokens,
    time_tokens,
)

from .intents import ChatIntent, classify_intent, policy_reply
from .llm import BaseLLMClient
from .memory import CuratedMemoryStore
from .router import RouteKind, RouteProfile, classify_route
from .types import ChatRequest, ChatResponse
from .validation_feedback import ValidationFeedbackRecorder


@dataclass
class PipelineConfig:
    enable_llm: bool = True
    enable_memory_retrieval: bool = True
    enable_rag: bool = False
    enable_clarification: bool = True
    enable_post_validation: bool = True
    enable_validation_feedback: bool = False
    validation_feedback_path: str = "artifacts/audit/chat_v2_validation_feedback.jsonl"
    validation_stats_path: str = "artifacts/audit/chat_v2_validation_stats.json"
    memory_topk: int = 3
    memory_max_chars: int = 900
    rag_topk: int = 4
    rag_min_sim: float = 0.2
    rag_max_chars: int = 1200


from .thread_context import ThreadContextTracker
from .retrieval import RetrievalManager
from .evidence import EvidenceValidator

# ── Self-evolution learning loop ─────────────────────────────────────────
try:
    from system.learning.learning_loop import LearnEvent, LearningLoop
    _LEARNING_LOOP_AVAILABLE = True
except Exception:
    _LEARNING_LOOP_AVAILABLE = False

# ── FAQ retrieval ────────────────────────────────────────────────────────
try:
    from system.knowledge.semantic.faq import FAQRetriever
    _FAQ_AVAILABLE = True
except Exception:
    _FAQ_AVAILABLE = False


class ChatPipeline:
    def __init__(
        self,
        memory_store: CuratedMemoryStore,
        llm_client: BaseLLMClient,
        rag_store: Any | None = None,
        config: PipelineConfig | None = None,
        agent_runner: Any | None = None,
        reasoning_runner: Any | None = None,
        learning_loop: Any | None = None,
    ) -> None:
        self.memory_store = memory_store
        self.llm_client = llm_client
        self.rag_store = rag_store
        self.config = config or PipelineConfig()
        self.agent_runner = agent_runner
        self.reasoning_runner = reasoning_runner

        # Self-evolution learning loop
        if learning_loop is not None:
            self.learning_loop = learning_loop
        elif _LEARNING_LOOP_AVAILABLE:
            self.learning_loop = LearningLoop(enabled=True)
        else:
            self.learning_loop = None

        # FAQ retriever — embedding-based FAQ search
        self.faq_retriever = None
        if _FAQ_AVAILABLE:
            self.faq_retriever = FAQRetriever()

        # Extracted components (Phase 2 refactoring)
        self.thread = ThreadContextTracker()
        self.retrieval = RetrievalManager(
            memory_store=memory_store,
            rag_store=rag_store,
            memory_topk=self.config.memory_topk,
            memory_max_chars=self.config.memory_max_chars,
            rag_topk=self.config.rag_topk,
            rag_min_sim=self.config.rag_min_sim,
            rag_max_chars=self.config.rag_max_chars,
            enable_memory_retrieval=self.config.enable_memory_retrieval,
            enable_rag=self.config.enable_rag,
        )
        self.evidence = EvidenceValidator(
            enable_post_validation=self.config.enable_post_validation,
            enable_clarification=self.config.enable_clarification,
            enable_validation_feedback=self.config.enable_validation_feedback,
            feedback_path=self.config.validation_feedback_path,
            stats_path=self.config.validation_stats_path,
        )
        self.validation_feedback = self.evidence.feedback

    def _contains_cjk(self, text): return contains_cjk(text)
    def _should_try_agent_first(self, route, query):
        if self.agent_runner is None: return False
        if route.prefers_agent: return True
        if route.kind == RouteKind.PROJECT_QA and self._is_project_inspection_query(query): return True
        return False
    def _is_project_inspection_query(self, text):
        raw = str(text or "").strip()
        if not raw: return False
        low = raw.lower()
        pt = ("project","repo","repository","workspace","codebase","项目","仓库","代码","工程")
        it = ("inspect","check","review","analyze","summarize","overview","architecture","structure","module","entrypoint","runtime flow","检查","查看","看看","分析","梳理","总结","概览","架构","结构","模块","入口","流程")
        return any(t in low or t in raw for t in pt) and any(t in low or t in raw for t in it)

    # ── Thread context delegation ──────────────────────────────────────────
    def _last_assistant_turn(self, history): return self.thread.last_assistant_turn(history)
    def _default_context_label(self, kind, *, cjk): return self.thread.default_context_label(kind, cjk=cjk)
    def _context_label(self, kind, query, answer): return self.thread.context_label(kind, query, answer)
    def _normalize_thread_context(self, raw): return self.thread.normalize_thread_context(raw)
    def _assistant_thread_context(self, history): return self.thread.assistant_thread_context(history)
    def _pending_clarification(self, history): return self.thread.pending_clarification(history)
    def _looks_like_new_task(self, text): return self.thread.looks_like_new_task(text)
    def _looks_like_context_update(self, text): return self.thread.looks_like_context_update(text)
    def _append_thread_context(self, thread, answer): return self.thread.append_thread_context(thread, answer)
    def _compose_thread_query(self, thread): return self.thread.compose_thread_query(thread)
    def _expand_followup_query(self, req): return self.thread.expand_followup_query(req.history, req.user_text)
    def _attach_thread_context(self, meta, ctx, *, resolved_query=""): return self.thread.attach_thread_context(meta, ctx, resolved_query=resolved_query)
    def _thread_context_issue(self, answer, route, ctx):
        rk = str(getattr(route.kind, "value", route.kind) or "unknown") if hasattr(route, 'kind') else str(route or "")
        return self.thread.thread_context_issue(answer, rk, ctx)

    # ── Response validation (pipeline-specific guards) ─────────────────────
    def _valid_reply(self, query: str, reply: str) -> bool:
        if not is_valid_reply(reply):
            return False
        if is_low_signal_response(reply):
            return False
        if guard_response(query, reply) is None:
            return False
        return True

    def _valid_rag_reply(self, reply: str) -> bool:
        return is_valid_rag_reply(reply)
    def _attach_thread_context(self, meta, ctx, *, resolved_query=""): return self.thread.attach_thread_context(meta, ctx, resolved_query=resolved_query)
    def _thread_context_issue(self, answer, route, ctx):
        rk = str(getattr(route.kind, "value", route.kind) or "unknown") if hasattr(route, 'kind') else str(route or "")
        return self.thread.thread_context_issue(answer, rk, ctx)

    # ── Retrieval delegation ───────────────────────────────────────────────
    def _memory_score_threshold(self, route): return self.retrieval.memory_score_threshold(route)
    def _filter_memory_hits(self, hits, route): return self.retrieval.filter_memory_hits(hits, route)
    def _memory_hits(self, query, route=None): return self.retrieval.memory_hits(query, route)
    def _rag_context(self, query): return self.retrieval.rag_context(query)
    def _memory_block(self, hits): return self.retrieval.memory_block(hits)
    def _rag_block(self, ctx): return self.retrieval.rag_block(ctx)
    def _augment_query_with_context(self, q, hits, ctx, route): return self.retrieval.augment_query_with_context(q, hits, ctx, route)

    def _image_context(self, req: ChatRequest) -> str:
        """Analyze any image attached to the request and return a text description."""
        image = req.image_data
        if image is None and req.image_path:
            try:
                from system.perception.image_analyzer import load_image
                image = load_image(req.image_path)
            except Exception:
                pass
        if image is None:
            return ""
        try:
            from system.perception.image_analyzer import analyze_image
            info = analyze_image(image)
            caption = info.get("caption", "")
            if not caption:
                from system.perception.image_analyzer import _basic_image_description
                caption = _basic_image_description(image)
            return f"[Image: {caption}]"
        except Exception:
            return ""
    def _route_guidance(self, route): return self.retrieval.route_guidance(route)
    def _rag_reply(self, q, ctx): return self.retrieval.rag_reply(q, ctx)
    def _has_rag_evidence(self, ctx): return self.retrieval.has_rag_evidence(ctx)
    def _rag_snippet_count(self, ctx): return self.retrieval.rag_snippet_count(ctx)

    # ── Evidence delegation ────────────────────────────────────────────────
    def _evidence_blocks(self, hits, ctx): return self.evidence.evidence_blocks(hits, ctx)
    def _is_critical_context_label(self, label): return self.thread.is_critical_context_label(label)
    def _supported_by_evidence(self, answer, blocks): return self.evidence.supported_by_evidence(answer, blocks)
    def _evidence_issue(self, answer, route, hits, ctx): return self.evidence.evidence_issue(answer, route, hits, ctx)
    def _validation_issue(self, answer, source, route, hits, ctx, thread_ctx):
        return self.evidence._validation_issue(answer, source, route, hits, ctx, self.thread, thread_ctx)
    def _thread_conflict_reply(self, query, issue): return self.evidence.thread_conflict_reply(query, issue)
    def _grounded_fallback(self, query, route, hits, ctx):
        return self.evidence.grounded_fallback(query, route, hits, ctx, self._rag_reply)
    def _clarify_text(self, query, route, *, reasoning_out=None):
        return self.evidence.clarify_text(query, route, reasoning_out=reasoning_out)
    def _preview_text(self, text, *, limit=180):
        import re
        raw = re.sub(r"\s+", " ", str(text or "").strip())
        return raw[:limit - 3].rstrip() + "..." if len(raw) > limit else raw

    def _record_validation_feedback(
        self,
        *,
        query: str,
        route: RouteProfile,
        candidate_source: str,
        candidate_text: str,
        final_source: str,
        final_text: str,
        validation: str,
        validation_override: str,
        memory_hits: Sequence[Any],
        rag_context: Optional[Dict[str, Any]],
        thread_context: Dict[str, Any] | None,
        reasoning_out: Any | None = None,
    ) -> None:
        if self.validation_feedback is None:
            return
        reasoning_meta = {}
        if reasoning_out is not None and isinstance(getattr(reasoning_out, "metadata", {}), dict):
            reasoning_meta = dict(getattr(reasoning_out, "metadata", {}) or {})
        thread = self._normalize_thread_context(thread_context) or {}
        payload = {
            "query_preview": self._preview_text(query, limit=240),
            "route_kind": str(getattr(route.kind, "value", route.kind) or "unknown"),
            "candidate_source": str(candidate_source or "").strip() or "unknown",
            "final_source": str(final_source or "").strip() or "unknown",
            "validation": str(validation or "").strip() or "unknown",
            "validation_override": str(validation_override or "").strip() or "unknown",
            "candidate_preview": self._preview_text(candidate_text),
            "final_preview": self._preview_text(final_text),
            "memory_hits": int(len(list(memory_hits or []))),
            "rag_snippets": self._rag_snippet_count(rag_context),
            "thread_items": int(len(list(thread.get("items", []) or []))),
            "thread_labels": [str(item.get("label", "") or "").strip() for item in list(thread.get("items", []) or [])[:4] if isinstance(item, dict)],
            "query_mode": str(reasoning_meta.get("query_mode", "") or "").strip(),
            "reasoning_action": str(reasoning_meta.get("action", "") or "").strip(),
        }
        try:
            self.validation_feedback.record(payload)
        except Exception:
            pass

    def _post_validate_candidate(
        self,
        *,
        query: str,
        text: str,
        source: str,
        metadata: Dict[str, Any],
        route: RouteProfile,
        memory_hits: Sequence[Any],
        rag_context: Optional[Dict[str, Any]],
        thread_context: Dict[str, Any] | None,
        reasoning_out: Any | None = None,
    ) -> tuple[str, str, Dict[str, Any]]:
        issue = self._validation_issue(text, source, route, memory_hits, rag_context, thread_context)
        if issue is None:
            out = dict(metadata or {})
            out["validation"] = "passed"
            self._record_validation_feedback(
                query=query,
                route=route,
                candidate_source=source,
                candidate_text=text,
                final_source=source,
                final_text=text,
                validation="passed",
                validation_override="pass",
                memory_hits=memory_hits,
                rag_context=rag_context,
                thread_context=thread_context,
                reasoning_out=reasoning_out,
            )
            return text, source, out

        issue_kind = str(issue.get("kind", "") or "").strip()
        if issue_kind.startswith("thread_"):
            thread_text = self._thread_conflict_reply(query, issue).strip()
            if thread_text and self._valid_reply(query, thread_text):
                out = self._attach_thread_context(
                    {"validation": issue_kind, "validation_override": "clarify"},
                    thread_context,
                    resolved_query=query,
                )
                self._record_validation_feedback(
                    query=query,
                    route=route,
                    candidate_source=source,
                    candidate_text=text,
                    final_source="clarify",
                    final_text=thread_text,
                    validation=issue_kind,
                    validation_override="clarify",
                    memory_hits=memory_hits,
                    rag_context=rag_context,
                    thread_context=thread_context,
                    reasoning_out=reasoning_out,
                )
                return thread_text, "clarify", out

        grounded = self._grounded_fallback(query, route, memory_hits, rag_context)
        if grounded is not None:
            grounded_text, grounded_source = grounded
            out = self._attach_thread_context(
                {"validation": issue_kind, "validation_override": grounded_source},
                thread_context,
                resolved_query=query,
            )
            self._record_validation_feedback(
                query=query,
                route=route,
                candidate_source=source,
                candidate_text=text,
                final_source=grounded_source,
                final_text=grounded_text,
                validation=issue_kind,
                validation_override=grounded_source,
                memory_hits=memory_hits,
                rag_context=rag_context,
                thread_context=thread_context,
                reasoning_out=reasoning_out,
            )
            return grounded_text, grounded_source, out

        clarification = self._clarification_reply(
            query,
            route,
            memory_hits=list(memory_hits or []),
            rag_context=rag_context,
            reasoning_out=reasoning_out,
            thread_context=thread_context,
            force=True,
        )
        if clarification is not None:
            text_out, source_out, meta_out = clarification
            meta = dict(meta_out or {})
            meta["validation"] = issue_kind
            meta["validation_override"] = source_out
            self._record_validation_feedback(
                query=query,
                route=route,
                candidate_source=source,
                candidate_text=text,
                final_source=source_out,
                final_text=text_out,
                validation=issue_kind,
                validation_override=source_out,
                memory_hits=memory_hits,
                rag_context=rag_context,
                thread_context=thread_context,
                reasoning_out=reasoning_out,
            )
            return text_out, source_out, meta

        out = self._attach_thread_context(
            {"validation": issue_kind, "validation_override": "fallback"},
            thread_context,
            resolved_query=query,
        )
        self._record_validation_feedback(
            query=query,
            route=route,
            candidate_source=source,
            candidate_text=text,
            final_source="fallback",
            final_text=persona_prompt_reply(),
            validation=issue_kind,
            validation_override="fallback",
            memory_hits=memory_hits,
            rag_context=rag_context,
            thread_context=thread_context,
            reasoning_out=reasoning_out,
        )
        return persona_prompt_reply(), "fallback", out

    def _clarification_reply(
        self,
        query: str,
        route: RouteProfile,
        *,
        memory_hits: List[Any],
        rag_context: Optional[Dict[str, Any]],
        reasoning_out: Any | None = None,
        thread_context: Dict[str, Any] | None = None,
        force: bool = False,
    ) -> Optional[tuple[str, str, Dict[str, Any]]]:
        if not bool(self.config.enable_clarification):
            return None
        if not force and (memory_hits or self._has_rag_evidence(rag_context)):
            return None
        text = self._clarify_text(query, route, reasoning_out=reasoning_out).strip()
        if not text:
            return None
        if not self._valid_reply(query, text):
            return None
        base_thread = self._normalize_thread_context(thread_context)
        if base_thread is None:
            base_thread = {
                "original_query": str(query or "").strip(),
                "route_kind": str(route.kind.value),
                "items": [],
            }
        metadata: Dict[str, Any] = {
            "route_kind": str(base_thread.get("route_kind", "") or route.kind.value),
            "original_query": str(base_thread.get("original_query", "") or query).strip(),
            "clarify_prompt": text,
            "thread_context": base_thread,
        }
        if reasoning_out is not None and isinstance(getattr(reasoning_out, "metadata", {}), dict):
            reasoning_meta = dict(getattr(reasoning_out, "metadata", {}) or {})
            if reasoning_meta:
                metadata["reasoning"] = reasoning_meta
        return text, "clarify", metadata

    def _decide(
        self,
        req: ChatRequest,
        intent: ChatIntent,
        *,
        thread_context: Dict[str, Any] | None = None,
    ) -> tuple[str, str, Dict[str, Any]]:
        policy = policy_reply(intent, req.user_text)
        if policy:
            return policy, "policy", {}

        route = classify_route(req.user_text, intent)

        if self._should_try_agent_first(route, req.user_text):
            try:
                agent_out = self.agent_runner.handle(req, intent)
            except Exception:
                agent_out = None
            if agent_out is not None and bool(getattr(agent_out, "handled", False)):
                text = str(getattr(agent_out, "text", "") or "").strip()
                source = str(getattr(agent_out, "source", "agent") or "agent").strip()
                if text:
                    metadata = self._attach_thread_context(
                        dict(getattr(agent_out, "metadata", {}) or {}),
                        thread_context,
                        resolved_query=req.user_text,
                    )
                    return text, source, metadata

        memory_hits = self._memory_hits(req.user_text, route=route) if route.prefers_memory else []
        rag_context = self._rag_context(req.user_text) if route.prefers_rag else None
        reasoning_out = None
        if route.prefers_reasoning and self.reasoning_runner is not None:
            try:
                reasoning_out = self.reasoning_runner.handle(
                    req,
                    intent,
                    memory_hits=memory_hits,
                    rag_context=rag_context,
                )
            except Exception:
                reasoning_out = None
            if reasoning_out is not None and bool(getattr(reasoning_out, "handled", False)):
                text = str(getattr(reasoning_out, "text", "") or "").strip()
                source = str(getattr(reasoning_out, "source", "reasoning") or "reasoning").strip()
                if text and self._valid_reply(req.user_text, text):
                    metadata = self._attach_thread_context(
                        dict(getattr(reasoning_out, "metadata", {}) or {}),
                        thread_context,
                        resolved_query=req.user_text,
                    )
                    return self._post_validate_candidate(
                        query=req.user_text,
                        text=text,
                        source=source,
                        metadata=metadata,
                        route=route,
                        memory_hits=memory_hits,
                        rag_context=rag_context,
                        thread_context=thread_context,
                        reasoning_out=reasoning_out,
                    )

        if self.config.enable_llm and self.llm_client.available():
            try:
                llm_query = self._augment_query_with_context(req.user_text, memory_hits, rag_context, route)
                # Inject image analysis into LLM query if image is attached
                img_ctx = self._image_context(req)
                if img_ctx:
                    llm_query = f"{img_ctx}\n\nUser query: {llm_query}"
                llm_text = self.llm_client.generate(llm_query, req.history)
            except Exception:
                logger.debug("LLM call failed, falling through to next source", exc_info=True)
                llm_text = ""
            if self._valid_reply(req.user_text, llm_text):
                metadata = self._attach_thread_context({}, thread_context, resolved_query=req.user_text)
                if memory_hits and rag_context:
                    source = "llm_memory_rag"
                elif memory_hits:
                    source = "llm_memory"
                else:
                    source = "llm_rag" if rag_context else "llm"
                return self._post_validate_candidate(
                    query=req.user_text,
                    text=llm_text,
                    source=source,
                    metadata=metadata,
                    route=route,
                    memory_hits=memory_hits,
                    rag_context=rag_context,
                    thread_context=thread_context,
                )

        # FAQ retrieval — embedding-based question matching
        if self.faq_retriever is not None:
            try:
                faq_result = self.faq_retriever.search(req.user_text)
                if faq_result is not None:
                    faq_text = str(getattr(faq_result, "answer", "") or "").strip()
                    if self._valid_reply(req.user_text, faq_text):
                        metadata = self._attach_thread_context(
                            {"faq_question": getattr(faq_result, "question", ""), "faq_source": getattr(faq_result, "source", "")},
                            thread_context, resolved_query=req.user_text,
                        )
                        return faq_text, "faq", metadata
            except Exception:
                logger.debug("FAQ retrieval failed", exc_info=True)

        if route.kind == RouteKind.PROJECT_QA:
            rag_text = self._rag_reply(req.user_text, rag_context)
            if self._valid_rag_reply(rag_text):
                metadata = self._attach_thread_context({}, thread_context, resolved_query=req.user_text)
                return rag_text, "rag", metadata

        if memory_hits:
            hit = memory_hits[0]
            reply = str(getattr(hit, "assistant", "") or "").strip()
            if self._valid_reply(req.user_text, reply):
                metadata = self._attach_thread_context({}, thread_context, resolved_query=req.user_text)
                return reply, "memory", metadata

        rag_text = self._rag_reply(req.user_text, rag_context)
        if self._valid_rag_reply(rag_text):
            metadata = self._attach_thread_context({}, thread_context, resolved_query=req.user_text)
            return rag_text, "rag", metadata

        clarification = self._clarification_reply(
            req.user_text,
            route,
            memory_hits=memory_hits,
            rag_context=rag_context,
            reasoning_out=reasoning_out,
            thread_context=thread_context,
        )
        if clarification is not None:
            return clarification

        return persona_prompt_reply(), "fallback", {}

    def respond(self, req: ChatRequest) -> ChatResponse:
        import time as _time
        _start_ts = _time.time()

        effective_text, thread_context = self._expand_followup_query(req)
        work_req = ChatRequest(user_text=effective_text, history=req.history)
        intent = classify_intent(work_req.user_text)
        text, source, metadata = self._decide(work_req, intent, thread_context=thread_context)
        text = normalize_persona_reply(req.user_text, text)

        # ── Derive validation outcome for learning ─────────────────────────
        validation = str((metadata or {}).get("validation", "") or "passed")
        final_source = source
        was_fallback = False

        if str(source).startswith("agent"):
            if len(str(text or "").strip()) < 2:
                text = persona_prompt_reply()
                source = "fallback"
                final_source = "fallback"
                was_fallback = True
                metadata = {}
            text = normalize_persona_reply(req.user_text, text)
        elif source == "rag":
            if not self._valid_rag_reply(text):
                text = persona_prompt_reply()
                source = "fallback"
                final_source = "fallback"
                was_fallback = True
                metadata = {}
            text = normalize_persona_reply(req.user_text, text)
        elif not self._valid_reply(work_req.user_text, text):
            text = persona_prompt_reply()
            source = "fallback"
            final_source = "fallback"
            was_fallback = True
            metadata = {}
        text = normalize_persona_reply(req.user_text, text)

        # ── Self-evolution: observe & learn ──────────────────────────────────
        if self.learning_loop is not None:
            try:
                route = classify_route(work_req.user_text, intent)
                success = not was_fallback and not str(validation).startswith("thread_") and not str(validation).startswith("evidence_")
                rule_name = str((metadata or {}).get("reasoning_rule", "") or "")
                self.learning_loop.observe(
                    LearnEvent(
                        ts=_start_ts,
                        query=str(work_req.user_text or ""),
                        route_kind=str(route.kind.name),
                        source=str(source or ""),
                        validation=str(validation or "passed"),
                        final_source=str(final_source or source or ""),
                        rule_name=rule_name,
                        success=success,
                        metadata=dict(metadata or {}),
                    )
                )
                # Periodic learning trigger
                self.learning_loop.maybe_learn()
            except Exception:
                logger.debug("LearningLoop: observe/maybe_learn failed", exc_info=True)

        return ChatResponse(text=text, source=source, intent=str(intent), metadata=metadata)


def build_request(user_text: str, history: List[dict]) -> ChatRequest:
    from .types import ChatTurn

    turns = [
        ChatTurn(
            role=str(x.get("role", "")),
            text=str(x.get("text", "")),
            source=str(x.get("source", "") or ""),
            metadata=dict(x.get("metadata", {}) or {}),
        )
        for x in history
    ]
    return ChatRequest(user_text=str(user_text or ""), history=turns)
