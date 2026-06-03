# -*- coding: utf-8 -*-
"""Deterministic reasoning route for chat runtime v2."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence

from system.brain.reasoning import HierarchicalReasoner

from .intents import ChatIntent
from .types import ChatRequest


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_PATH_RE = re.compile(r"(?:[A-Za-z]:)?[A-Za-z0-9_./\\-]+(?:\.[A-Za-z0-9_]+)?")

_REASON_PATTERNS = (
    r"\bwhy\b",
    r"\bbecause\b",
    r"\breason\b",
    r"\bcause\b",
    r"\broot cause\b",
    r"\u4e3a\u4ec0\u4e48",
    r"\u539f\u56e0",
    r"\u4e3a\u4f55",
    r"\u56e0\u4e3a",
)

_PLAN_PATTERNS = (
    r"\bplan\b",
    r"\bstrategy\b",
    r"\bapproach\b",
    r"\broadmap\b",
    r"\bsteps?\b",
    r"\btrade[\s-]?off\b",
    r"\brollback\b",
    r"\bsafely\b",
    r"\brisk\b",
    r"\bconstraint\b",
    r"\bpriorit",
    r"\u8ba1\u5212",
    r"\u65b9\u6848",
    r"\u6b65\u9aa4",
    r"\u7b56\u7565",
    r"\u53d6\u820d",
    r"\u6743\u8861",
    r"\u8def\u7ebf",
    r"\u56de\u6eda",
    r"\u98ce\u9669",
    r"\u7ea6\u675f",
)

_DECISION_PATTERNS = (
    r"\bshould\b",
    r"\bwhether\b",
    r"\bwhich\b.{0,24}\bbetter\b",
    r"\bcompare\b",
    r"\bchoose\b",
    r"\bbetter\b",
    r"\bsafer\b",
    r"\bcan we\b",
    r"\bcan i\b",
    r"\bdo we need to\b",
    r"\bdo i need to\b",
    r"\u5e94\u8be5",
    r"\u662f\u5426",
    r"\u80fd\u5426",
    r"\u53ef\u4e0d\u53ef\u4ee5",
    r"\u8981\u4e0d\u8981",
    r"\u8be5\u4e0d\u8be5",
    r"\u54ea\u4e2a\u66f4\u597d",
    r"\u54ea\u4e2a\u66f4\u7a33",
    r"\u6bd4\u8f83",
    r"\u5bf9\u6bd4",
    r"\u9009\u54ea\u4e2a",
)

_CONSTRAINT_PATTERNS = (
    r"\bmust\b.{0,40}\bcannot\b",
    r"\bmust\b.{0,40}\bmust not\b",
    r"\bmust\b.{0,40}\bcan not\b",
    r"\bshould\b.{0,40}\bshould not\b",
    r"\bcan't\b",
    r"\bconflict\b",
    r"\bcontradict",
    r"\bconstraint\b",
    r"\u5fc5\u987b.{0,20}\u4e0d\u80fd",
    r"\u5fc5\u987b.{0,20}\u4e0d\u5f97",
    r"\u5e94\u8be5.{0,20}\u4e0d\u5e94\u8be5",
    r"\u51b2\u7a81",
    r"\u77db\u76fe",
    r"\u7ea6\u675f",
)

_EXPLICIT_WORKSPACE_PATTERNS = (
    r"\bread\b",
    r"\bopen\b",
    r"\blist\b",
    r"\bsearch\b",
    r"\bfind\b",
    r"\bwrite\b",
    r"\bedit\b",
    r"\bpatch\b",
    r"\brun\b",
    r"\bpytest\b",
    r"\btests?\b",
    r"\bfile\b",
    r"\bfolder\b",
    r"\bdirectory\b",
    r"\brepo\b",
    r"\brepository\b",
    r"\u8bfb\u53d6",
    r"\u6253\u5f00",
    r"\u67e5\u770b",
    r"\u5217\u51fa",
    r"\u641c\u7d22",
    r"\u67e5\u627e",
    r"\u5199\u5165",
    r"\u4fee\u6539",
    r"\u8fd0\u884c",
    r"\u6587\u4ef6",
    r"\u76ee\u5f55",
    r"\u4ee3\u7801\u5e93",
)

_STRIP_PUNCT = " \t\r\n?!.,;:\u3002\uff01\uff1f\uff0c\uff1b\uff1a"


@dataclass
class ReasoningConfig:
    enabled: bool = True
    feedback_path: str = "artifacts/audit/reasoning_rule_feedback.json"
    profile_path: str = "config/reasoning_profile.json"
    load_feedback: bool = True
    max_memory_hits: int = 3
    max_rag_snippets: int = 3
    max_history_turns: int = 4


@dataclass
class ReasoningOutcome:
    handled: bool
    text: str = ""
    source: str = "reasoning"
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChatReasoner:
    def __init__(
        self,
        config: ReasoningConfig | None = None,
        *,
        reasoner: HierarchicalReasoner | None = None,
    ) -> None:
        self.config = config or ReasoningConfig()
        self.reasoner = reasoner or HierarchicalReasoner(
            feedback_path=str(self.config.feedback_path or "artifacts/audit/reasoning_rule_feedback.json"),
            profile_path=str(self.config.profile_path or "config/reasoning_profile.json"),
            load_feedback=bool(self.config.load_feedback),
        )

    def should_handle(self, query: str, intent: ChatIntent) -> bool:
        if not bool(self.config.enabled) or intent != ChatIntent.TASK:
            return False
        text = str(query or "").strip()
        if not text:
            return False
        low = text.lower()
        if self._looks_like_workspace_task(text, low):
            return False
        return any(
            (
                self._has_pattern(text, low, _CONSTRAINT_PATTERNS),
                self._has_pattern(text, low, _REASON_PATTERNS),
                self._has_pattern(text, low, _PLAN_PATTERNS),
                self._has_pattern(text, low, _DECISION_PATTERNS),
            )
        )

    def handle(
        self,
        req: ChatRequest,
        intent: ChatIntent,
        *,
        memory_hits: Sequence[Any] | None = None,
        rag_context: Dict[str, Any] | None = None,
    ) -> Optional[ReasoningOutcome]:
        if not self.should_handle(req.user_text, intent):
            return None

        query = str(req.user_text or "").strip()
        memory_rows = list(memory_hits or [])[: max(0, int(self.config.max_memory_hits))]
        rag_snippets = self._rag_snippets(rag_context)[: max(0, int(self.config.max_rag_snippets))]
        goal = self._infer_goal(query, req.history)
        facts = self._build_facts(query, goal, req.history, memory_rows, rag_snippets)
        low_rules = self._build_low_rules(query, goal, memory_rows, rag_snippets)

        steps = self.reasoner.reason(facts, low_rules)
        meta = self.reasoner.summarize(steps)
        text = self._compose_reply(
            query,
            goal,
            meta=meta,
            memory_hits=memory_rows,
            rag_snippets=rag_snippets,
        )
        return ReasoningOutcome(
            handled=bool(text.strip()),
            text=text,
            source=self._source_name(memory_rows, rag_snippets),
            metadata={
                "query_mode": str(self._query_mode(query)),
                "followup": str(self._followup_hint(query)),
                "action": str(meta.get("action", "") or "").strip(),
                "confidence": float(meta.get("confidence", 0.0) or 0.0),
                "best_conclusion": str(meta.get("best_conclusion", "") or "").strip(),
                "requires_clarification": bool(meta.get("requires_clarification", False)),
                "memory_hits": int(len(memory_rows)),
                "rag_snippets": int(len(rag_snippets)),
            },
        )

    def _has_pattern(self, text: str, low: str, patterns: Sequence[str]) -> bool:
        for pattern in patterns:
            haystack = low if "\\b" in str(pattern) else text
            if re.search(str(pattern), haystack, flags=re.IGNORECASE):
                return True
        return False

    def _looks_like_workspace_task(self, text: str, low: str) -> bool:
        if not self._has_pattern(text, low, _EXPLICIT_WORKSPACE_PATTERNS):
            return False
        if _PATH_RE.search(text):
            return True
        if "`" in text:
            return True
        if re.search(r"--[a-z0-9][a-z0-9-]*", low):
            return True
        return False

    def _contains_cjk(self, text: str) -> bool:
        return _CJK_RE.search(str(text or "")) is not None

    def _query_mode(self, query: str) -> str:
        text = str(query or "").strip()
        low = text.lower()
        if self._has_pattern(text, low, _CONSTRAINT_PATTERNS):
            return "constraints"
        if self._has_pattern(text, low, _REASON_PATTERNS):
            return "explain"
        if self._has_pattern(text, low, _PLAN_PATTERNS):
            return "plan"
        if self._has_pattern(text, low, _DECISION_PATTERNS):
            return "decision"
        return "general"

    def _infer_goal(self, query: str, history: Sequence[Any]) -> str:
        text = str(query or "").strip()
        if not text:
            return ""
        cleaned = re.sub(r"^[Pp]lease\s+", "", text)
        cleaned = re.sub(
            r"^(?:can|could|would|should)\s+(?:you|we|i)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"^(?:why|how|what|which)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip(_STRIP_PUNCT)
        if len(cleaned) >= 12:
            return cleaned
        history_tail = self._history_block(history)
        if history_tail:
            return f"{cleaned} {history_tail}".strip()
        return cleaned or text

    def _history_block(self, history: Sequence[Any]) -> str:
        rows: List[str] = []
        for turn in list(history or [])[-max(0, int(self.config.max_history_turns)) :]:
            role = str(getattr(turn, "role", "") or "").strip()
            text = str(getattr(turn, "text", "") or "").strip()
            if not role or not text:
                continue
            rows.append(f"{role}: {text}")
        return "\n".join(rows)

    def _rag_snippets(self, rag_context: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        if not isinstance(rag_context, dict):
            return []
        snippets = list(rag_context.get("snippets", []) or [])
        out: List[Dict[str, Any]] = []
        for item in snippets:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "text": text,
                    "path": str(item.get("path", "") or "").strip(),
                    "score": float(item.get("score", 0.0) or 0.0),
                }
            )
        return out

    def _build_facts(
        self,
        query: str,
        goal: str,
        history: Sequence[Any],
        memory_hits: Sequence[Any],
        rag_snippets: Sequence[Dict[str, Any]],
    ) -> Dict[str, str]:
        facts: Dict[str, str] = {
            "message": str(query or "").strip(),
            "goal": str(goal or "").strip(),
        }
        history_block = self._history_block(history)
        if history_block:
            facts["history"] = history_block
        for idx, hit in enumerate(list(memory_hits), 1):
            user = str(getattr(hit, "user", "") or "").strip()
            assistant = str(getattr(hit, "assistant", "") or "").strip()
            if user:
                facts[f"memory_user_{idx}"] = user
            if assistant:
                facts[f"memory_assistant_{idx}"] = assistant
        for idx, item in enumerate(list(rag_snippets), 1):
            text = str(item.get("text", "") or "").strip()
            path = str(item.get("path", "") or "").strip()
            if text:
                facts[f"rag_text_{idx}"] = text
            if path:
                facts[f"rag_path_{idx}"] = path
        return facts

    def _candidate_text(self, text: str, *, limit: int = 220) -> str:
        raw = re.sub(r"\s+", " ", str(text or "").strip())
        if not raw:
            return ""
        parts = re.split(r"(?<=[.!?;\u3002\uff01\uff1f\uff1b])\s+", raw)
        best = str(parts[0] or "").strip()
        if len(best) < 24 and len(parts) >= 2:
            merged = f"{best} {str(parts[1] or '').strip()}".strip()
            best = merged or best
        if len(best) <= limit:
            return best
        return best[: limit - 3].rstrip() + "..."

    def _domain_hint(self, goal: str, query: str, *, cjk: bool) -> str:
        blob = f"{goal} {query}".lower()
        if any(
            token in blob
            for token in (
                "deploy",
                "release",
                "migration",
                "patch",
                "database",
                "rollout",
                "\u53d1\u5e03",
                "\u4e0a\u7ebf",
                "\u8fc1\u79fb",
                "\u6570\u636e\u5e93",
                "\u56de\u6eda",
            )
        ):
            if cjk:
                return "\u5148\u786e\u8ba4\u524d\u7f6e\u6761\u4ef6\uff0c\u518d\u505a\u5907\u4efd\u3001\u9a8c\u8bc1\u548c\u56de\u6eda\u51c6\u5907"
            return "start with prerequisites, backup, verification, and rollback readiness"
        if any(
            token in blob
            for token in (
                "latency",
                "performance",
                "slow",
                "timeout",
                "\u6027\u80fd",
                "\u5ef6\u8fdf",
                "\u8d85\u65f6",
                "\u6162",
            )
        ):
            if cjk:
                return "\u5148\u5b9a\u4f4d\u74f6\u9888\uff0c\u518d\u9010\u6b65\u9a8c\u8bc1\u6bcf\u4e2a\u6539\u52a8\u7684\u6548\u679c"
            return "measure the bottleneck first, then validate one change at a time"
        if any(
            token in blob
            for token in (
                "error",
                "incident",
                "down",
                "failure",
                "bug",
                "issue",
                "\u9519\u8bef",
                "\u4e8b\u6545",
                "\u6545\u969c",
                "\u5f02\u5e38",
                "\u95ee\u9898",
            )
        ):
            if cjk:
                return "\u5148\u786e\u8ba4\u73b0\u8c61\u548c\u89e6\u53d1\u6761\u4ef6\uff0c\u518d\u5b9a\u4f4d\u6839\u56e0\u5e76\u9a8c\u8bc1\u4fee\u590d"
            return "confirm symptoms and triggers first, then verify the root cause before fixing"
        if cjk:
            return "\u5148\u660e\u786e\u7ea6\u675f\uff0c\u518d\u6309\u987a\u5e8f\u6267\u884c\u3001\u9a8c\u8bc1\u5e76\u4fdd\u7559\u56de\u6eda\u7a7a\u95f4"
        return "sequence the work with constraints, verification, and a reversible path"

    def _heuristic_candidates(self, query: str, goal: str) -> List[Dict[str, Any]]:
        cjk = self._contains_cjk(query)
        mode = self._query_mode(query)
        target = str(goal or query or "").strip()
        domain_hint = self._domain_hint(goal, query, cjk=cjk)
        out: List[Dict[str, Any]] = []
        if mode == "constraints":
            text = (
                "\u5148\u89e3\u51b3\u51b2\u7a81\u7ea6\u675f\uff0c\u518d\u51b3\u5b9a\u662f\u5426\u6267\u884c\u540e\u7eed\u52a8\u4f5c"
                if cjk
                else "resolve the conflicting constraints before taking any irreversible step"
            )
            out.append(
                {
                    "rule": "chat_constraint_reasoning",
                    "score": 0.78,
                    "evidence": "constraint_pattern",
                    "conclusion": text,
                }
            )
        if mode == "explain":
            text = (
                f"\u56f4\u7ed5 {target}\uff0c\u5148\u89e3\u91ca\u6700\u53ef\u80fd\u7684\u539f\u56e0\uff0c\u518d\u7ed9\u51fa\u9a8c\u8bc1\u987a\u5e8f\u548c\u540e\u7eed\u52a8\u4f5c"
                if cjk
                else f"for {target}, explain the most likely cause first, then outline the validation sequence and next action"
            )
            out.append(
                {
                    "rule": "chat_explain_reasoning",
                    "score": 0.64,
                    "evidence": "explain_pattern",
                    "conclusion": text,
                }
            )
        if mode == "plan":
            text = (
                f"\u56f4\u7ed5 {target}\uff0c{domain_hint}"
                if cjk
                else f"for {target}, {domain_hint}"
            )
            out.append(
                {
                    "rule": "chat_plan_reasoning",
                    "score": 0.68,
                    "evidence": "plan_pattern",
                    "conclusion": text,
                }
            )
        if mode == "decision":
            text = (
                f"\u4f18\u5148\u9009\u62e9\u6700\u7b26\u5408 {target} \u4e14\u66f4\u53ef\u56de\u6eda\u3001\u66f4\u4f4e\u98ce\u9669\u7684\u8def\u5f84"
                if cjk
                else f"prefer the option that best supports {target} while keeping the path reversible and lower risk"
            )
            out.append(
                {
                    "rule": "chat_decision_reasoning",
                    "score": 0.66,
                    "evidence": "decision_pattern",
                    "conclusion": text,
                }
            )
        if not out:
            text = (
                f"\u56f4\u7ed5 {target}\uff0c{domain_hint}"
                if cjk
                else f"for {target}, {domain_hint}"
            )
            out.append(
                {
                    "rule": "chat_general_reasoning",
                    "score": 0.55,
                    "evidence": "general_pattern",
                    "conclusion": text,
                }
            )
        return out

    def _memory_rules(self, memory_hits: Sequence[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, hit in enumerate(list(memory_hits), 1):
            assistant = self._candidate_text(str(getattr(hit, "assistant", "") or "").strip())
            if not assistant:
                continue
            score = 0.40 + 0.45 * max(0.0, min(1.0, float(getattr(hit, "score", 0.0) or 0.0)))
            out.append(
                {
                    "rule": f"memory_hit_{idx}",
                    "score": min(0.92, score),
                    "evidence": f"memory_similarity={float(getattr(hit, 'score', 0.0) or 0.0):.3f}",
                    "conclusion": assistant,
                }
            )
        return out

    def _rag_rules(self, rag_snippets: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(list(rag_snippets), 1):
            text = self._candidate_text(str(item.get("text", "") or "").strip())
            if not text:
                continue
            score = 0.34 + 0.42 * max(0.0, min(1.0, float(item.get("score", 0.0) or 0.0)))
            path = str(item.get("path", "") or "").strip()
            evidence = f"rag_similarity={float(item.get('score', 0.0) or 0.0):.3f}"
            if path:
                evidence += f",path={path}"
            out.append(
                {
                    "rule": f"rag_hit_{idx}",
                    "score": min(0.88, score),
                    "evidence": evidence,
                    "conclusion": text,
                }
            )
        return out

    def _build_low_rules(
        self,
        query: str,
        goal: str,
        memory_hits: Sequence[Any],
        rag_snippets: Sequence[Dict[str, Any]],
    ) -> List[Any]:
        rows = self._memory_rules(memory_hits) + self._rag_rules(rag_snippets) + self._heuristic_candidates(query, goal)
        out: List[Any] = []
        for row in rows:
            payload = dict(row)

            def _rule(_facts: Dict[str, str], payload: Dict[str, Any] = payload) -> Dict[str, Any]:
                return dict(payload)

            out.append(_rule)
        return out

    def _source_name(self, memory_hits: Sequence[Any], rag_snippets: Sequence[Dict[str, Any]]) -> str:
        if memory_hits and rag_snippets:
            return "reasoning_memory_rag"
        if memory_hits:
            return "reasoning_memory"
        if rag_snippets:
            return "reasoning_rag"
        return "reasoning"

    def _normalize_match(self, text: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(text or "").lower())

    def _is_query_echo(self, query: str, text: str) -> bool:
        left = self._normalize_match(query)
        right = self._normalize_match(text)
        if not left or not right:
            return False
        if left == right or left in right or right in left:
            return True
        left_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", left))
        right_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", right))
        if not left_tokens or not right_tokens:
            return False
        overlap = len(left_tokens & right_tokens) / float(max(1, len(left_tokens | right_tokens)))
        return overlap >= 0.82

    def _fallback_conclusion(self, query: str, goal: str) -> str:
        cjk = self._contains_cjk(query)
        mode = self._query_mode(query)
        domain_hint = self._domain_hint(goal, query, cjk=cjk)
        if cjk:
            if mode == "constraints":
                return "\u5148\u89e3\u51b3\u7ea6\u675f\u51b2\u7a81"
            if mode == "explain":
                return "\u5148\u9a8c\u8bc1\u6700\u53ef\u80fd\u7684\u539f\u56e0"
            if "backup" in domain_hint.lower() or "\u5907\u4efd" in domain_hint:
                return "\u5148\u5907\u4efd\uff0c\u518d\u8fc1\u79fb"
            if mode in {"plan", "decision"}:
                return "\u5148\u8d70\u53ef\u56de\u6eda\u3001\u4f4e\u98ce\u9669\u7684\u8def\u5f84"
            return "\u5148\u660e\u786e\u76ee\u6807\u548c\u7ea6\u675f"
        if mode == "constraints":
            return "Resolve priority first"
        if mode == "explain":
            return "Verify the likely cause first"
        if "backup" in domain_hint.lower():
            return "Backup first, then migrate"
        if mode in {"plan", "decision"}:
            return "Take the reversible, lower-risk path"
        return "Clarify the goal and constraints first"

    def _compact_conclusion(self, query: str, text: str) -> str:
        raw = self._candidate_text(text, limit=96)
        if not raw:
            return ""
        cjk = self._contains_cjk(query)
        low = raw.lower()
        if cjk:
            if ("\u5907\u4efd" in raw or "\u6821\u9a8c" in raw) and any(
                token in raw for token in ("\u8fc1\u79fb", "\u53d1\u5e03", "\u4e0a\u7ebf")
            ):
                return "\u5148\u5907\u4efd\uff0c\u518d\u8fc1\u79fb"
            if "\u56de\u6eda" in raw:
                return "\u5148\u51c6\u5907\u56de\u6eda"
            if any(token in raw for token in ("\u6839\u56e0", "\u539f\u56e0", "\u89e6\u53d1")):
                return "\u5148\u9a8c\u8bc1\u6839\u56e0"
            if any(token in raw for token in ("\u7ea6\u675f", "\u51b2\u7a81", "\u77db\u76fe")):
                return "\u5148\u89e3\u51b3\u7ea6\u675f\u51b2\u7a81"
            return raw[:24].rstrip(_STRIP_PUNCT)
        if "backup" in low and any(token in low for token in ("migrat", "deploy", "release", "database")):
            return "Backup first, then migrate"
        if "rollback" in low:
            return "Keep rollback ready"
        if "root cause" in low or ("cause" in low and "verify" in low):
            return "Verify the root cause first"
        if "constraint" in low or "irreversible" in low:
            return "Resolve priority first"
        words = raw.split()
        if len(words) > 6:
            return " ".join(words[:6]).rstrip(",.;:")
        return raw

    def _best_conclusion(
        self,
        query: str,
        goal: str,
        meta: Dict[str, Any],
        memory_hits: Sequence[Any],
        rag_snippets: Sequence[Dict[str, Any]],
    ) -> str:
        best = str(meta.get("best_conclusion", "") or "").strip()
        lowered = best.lower()
        if best and "need_more_evidence_or_constraints" not in lowered and not self._is_query_echo(query, best):
            compact = self._compact_conclusion(query, best)
            if compact:
                return compact
        if memory_hits:
            text = self._candidate_text(str(getattr(memory_hits[0], "assistant", "") or "").strip())
            if text:
                compact = self._compact_conclusion(query, text)
                if compact:
                    return compact
        if rag_snippets:
            text = self._candidate_text(str(rag_snippets[0].get("text", "") or "").strip())
            if text:
                compact = self._compact_conclusion(query, text)
                if compact:
                    return compact
        return self._fallback_conclusion(query, goal)

    def _evidence_note(self, query: str, memory_hits: Sequence[Any], rag_snippets: Sequence[Dict[str, Any]]) -> str:
        cjk = self._contains_cjk(query)
        if memory_hits and rag_snippets:
            if cjk:
                return "\u53c2\u8003\u4e86\u5bf9\u8bdd\u8bb0\u5fc6\u548c\u9879\u76ee\u4e0a\u4e0b\u6587\u3002"
            return "Uses dialogue memory and project context."
        if memory_hits:
            if cjk:
                return "\u53c2\u8003\u4e86\u5bf9\u8bdd\u8bb0\u5fc6\u3002"
            return "Uses dialogue memory."
        if rag_snippets:
            if cjk:
                return "\u53c2\u8003\u4e86\u9879\u76ee\u4e0a\u4e0b\u6587\u3002"
            return "Uses retrieved project context."
        return ""

    def _followup_hint(self, query: str) -> str:
        cjk = self._contains_cjk(query)
        mode = self._query_mode(query)
        if cjk:
            mapping = {
                "constraints": "\u54ea\u6761\u7ea6\u675f\u4f18\u5148",
                "explain": "\u5df2\u786e\u8ba4\u7684\u73b0\u8c61\u548c\u6700\u65b0\u53d8\u66f4",
                "plan": "\u6210\u529f\u6807\u51c6\u3001\u7ea6\u675f\u548c\u56de\u6eda\u7a97\u53e3",
                "decision": "\u901f\u5ea6\u3001\u98ce\u9669\u3001\u6210\u672c\u6216\u56de\u6eda",
                "general": "\u76ee\u6807\u3001\u7ea6\u675f\u548c\u98ce\u9669\u8fb9\u754c",
            }
        else:
            mapping = {
                "constraints": "which constraint wins",
                "explain": "the confirmed symptoms and latest change",
                "plan": "success criteria, constraints, and rollback window",
                "decision": "whether speed, risk, cost, or rollback matters most",
                "general": "the goal, constraints, and risk boundary",
            }
        return str(mapping.get(mode, mapping["general"]))

    def _compose_reply(
        self,
        query: str,
        goal: str,
        *,
        meta: Dict[str, Any],
        memory_hits: Sequence[Any],
        rag_snippets: Sequence[Dict[str, Any]],
    ) -> str:
        cjk = self._contains_cjk(query)
        best = self._best_conclusion(query, goal, meta, memory_hits, rag_snippets)
        action = str(meta.get("action", "") or "").strip().lower()
        evidence_note = self._evidence_note(query, memory_hits, rag_snippets)
        followup = self._followup_hint(query)

        if cjk:
            if action == "clarify_constraints":
                parts = ["\u7ea6\u675f\u51b2\u7a81\u3002\u66f4\u7a33\u7684\u65b9\u5411\u662f\u5148\u78ba\u5b9a\u4f18\u5148\u7ea7\u3002"]
            elif action == "collect_goal_details":
                parts = [f"\u5f53\u524d\u503e\u5411\uff1a{best}\u3002", f"\u8fd8\u8981\u786e\u8ba4\uff1a{followup}\u3002"]
            elif action == "proceed_with_caution":
                parts = [f"\u5f53\u524d\u503e\u5411\uff1a{best}\u3002"]
                if not evidence_note:
                    parts.append(f"\u5148\u786e\u8ba4\uff1a{followup}\u3002")
            else:
                parts = [f"{best}\u3002"]
            if evidence_note:
                parts.append(evidence_note)
            return " ".join(part for part in parts if part).strip()

        if action == "clarify_constraints":
            parts = ["Constraints conflict. Pin down priority first."]
        elif action == "collect_goal_details":
            parts = [f"Lean: {best}.", f"Need: {followup}."]
        elif action == "proceed_with_caution":
            parts = [f"Lean: {best}."]
            if not evidence_note:
                parts.append(f"Check {followup}.")
        else:
            parts = [f"{best}."]
        if evidence_note:
            parts.append(evidence_note)
        return " ".join(part for part in parts if part).strip()


__all__ = ["ChatReasoner", "ReasoningConfig", "ReasoningOutcome"]
