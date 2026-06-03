# -*- coding: utf-8 -*-
"""Online learning loop — connects pipeline feedback to continuous improvement.

ARCHITECTURE
────────────
The LearningLoop sits beside the ChatPipeline and observes every validation
event. When patterns emerge (same failure repeated, same rule producing good
results), it triggers learning actions that make the system "smarter with use":

    Pipeline._post_validate_candidate()
      └─ LearningLoop.observe(event)
           ├─ accumulate per-pattern counters
           ├─ detect emerging patterns
           └─ maybe_learn() when threshold reached
                ├─ RuleFeedbackMemory.update()   → rules get better priors
                ├─ ExperienceStore.add()         → episodes available for RAG
                └─ DialogueFailureReflector      → suggest prompt improvements
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Thresholds ───────────────────────────────────────────────────────────────
WINDOW_SIZE = 50          # sliding window of recent events
LEARN_MIN_EVENTS = 3      # need at least this many similar events to learn
LEARN_SCORE_THRESHOLD = 0.6  # only act on patterns with score >= threshold
LEARN_COOLDOWN_S = 300    # don't re-trigger for same pattern within 5 min


@dataclass
class LearnEvent:
    """A single observation from the pipeline."""
    ts: float
    query: str
    route_kind: str          # e.g. "GENERAL", "PROJECT_QA", "TASK"
    source: str              # e.g. "llm", "agent", "reasoning", "memory", "rag"
    validation: str          # "passed", "thread_context_issue", "evidence_issue", "fallback"
    final_source: str        # what actually served the response
    rule_name: str = ""      # reasoning rule name (if applicable)
    success: bool = True     # whether the outcome was good
    query_sig: str = ""      # normalized query signature for grouping
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnPattern:
    """An emerging pattern detected from a cluster of events."""
    key: str                 # pattern key (route:validation:query_group)
    events: List[LearnEvent] = field(default_factory=list)
    count: int = 0
    success_rate: float = 0.0
    last_triggered: float = 0.0
    suggestion: str = ""


class LearningLoop:
    """Observes pipeline events and triggers learning when patterns emerge.

    Usage in ChatPipeline::

        loop = LearningLoop()
        # ... after _post_validate_candidate:
        loop.observe(LearnEvent(
            ts=time.time(), query=..., route_kind=..., ...
        ))
        # ... at end of respond():
        result = loop.maybe_learn()
        if result:
            logger.info("LearningLoop: %s", result)
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        window_size: int = WINDOW_SIZE,
        learn_min_events: int = LEARN_MIN_EVENTS,
        learn_threshold: float = LEARN_SCORE_THRESHOLD,
        cooldown_s: float = LEARN_COOLDOWN_S,
        rule_feedback_path: str = "artifacts/audit/reasoning_rule_feedback.json",
        experience_store_path: str = "artifacts/memory/learning_experience.jsonl",
        pattern_store_path: str = "artifacts/audit/learning_patterns.json",
    ) -> None:
        self.enabled = enabled
        self.window_size = max(10, int(window_size))
        self.learn_min_events = max(2, int(learn_min_events))
        self.learn_threshold = float(learn_threshold)
        self.cooldown_s = float(cooldown_s)

        # Sliding window
        self._events: List[LearnEvent] = []

        # Pattern tracking
        self._patterns: Dict[str, LearnPattern] = {}
        self._trigger_history: Dict[str, float] = {}  # key → last_triggered_ts
        self._total_observations = 0
        self._total_learn_actions = 0

        # Persistence paths
        self._rule_feedback_path = Path(rule_feedback_path)
        self._experience_path = Path(experience_store_path)
        self._pattern_path = Path(pattern_store_path)

        # Lazy-loaded learning modules
        self._rule_feedback = None
        self._experience_store = None
        self._failure_reflector = None
        self._evolution_engine = None
        self._continual_memory = None
        self._evaluator = None

    # ── Public API ───────────────────────────────────────────────────────────

    def observe(self, event: LearnEvent) -> None:
        """Record a pipeline validation event."""
        if not self.enabled:
            return

        self._events.append(event)
        self._total_observations += 1
        if len(self._events) > self.window_size:
            self._events = self._events[-self.window_size :]

        # Update pattern counters
        pattern_key = self._pattern_key(event)
        if pattern_key not in self._patterns:
            self._patterns[pattern_key] = LearnPattern(key=pattern_key)
        pat = self._patterns[pattern_key]
        pat.events.append(event)
        pat.count += 1
        if pat.count > self.window_size:
            # Trim old events
            pat.events = pat.events[-self.window_size :]
            pat.count = len(pat.events)
        successes = sum(1 for e in pat.events if e.success)
        pat.success_rate = successes / max(1, pat.count)

    def maybe_learn(self) -> Optional[Dict[str, Any]]:
        """Check for patterns and trigger learning if thresholds are met.

        Returns a summary dict if learning occurred, None otherwise.
        """
        if not self.enabled:
            return None

        results: List[Dict[str, Any]] = []
        now = time.time()

        for key, pat in self._patterns.items():
            if pat.count < self.learn_min_events:
                continue
            # Check cooldown
            last = self._trigger_history.get(key, 0.0)
            if now - last < self.cooldown_s:
                continue

            # Trigger on low-success patterns (failures to learn from)
            if pat.success_rate < 0.5:
                result = self._learn_from_failures(key, pat)
                if result:
                    results.append(result)
                    self._trigger_history[key] = now

            # Also trigger on high-success patterns (reinforce good rules)
            elif pat.success_rate >= 0.85 and pat.count >= self.learn_min_events * 2:
                result = self._learn_from_success(key, pat)
                if result:
                    results.append(result)
                    self._trigger_history[key] = now

        if results:
            self._total_learn_actions += len(results)
            self._save_patterns()
            return {
                "actions": len(results),
                "results": results,
                "total_observations": self._total_observations,
                "total_learn_actions": self._total_learn_actions,
            }
        return None

    def force_learn(self, query: str, feedback: str, success: bool) -> Dict[str, Any]:
        """Explicit learning from user feedback (e.g., thumbs up/down)."""
        event = LearnEvent(
            ts=time.time(),
            query=query,
            route_kind="EXPLICIT_FEEDBACK",
            source="user",
            validation="user_feedback",
            final_source="user",
            success=success,
            metadata={"feedback": feedback},
        )
        self.observe(event)
        # Immediately learn from explicit feedback
        return self._learn_from_explicit(event) or {"action": "recorded"}

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "observations": self._total_observations,
            "learn_actions": self._total_learn_actions,
            "active_patterns": len(self._patterns),
            "window_size": len(self._events),
        }

    # ── Pattern detection ────────────────────────────────────────────────────

    def _pattern_key(self, event: LearnEvent) -> str:
        """Create a grouping key for similar events."""
        route = (event.route_kind or "UNKNOWN").strip()
        validation = (event.validation or "passed").strip()
        # Group similar queries by their salient tokens
        qsig = self._query_signature(event.query)
        return f"{route}:{validation}:{qsig}"

    def _query_signature(self, query: str) -> str:
        """Create a coarse query signature for grouping similar user intents."""
        text = (query or "").strip().lower()
        if not text:
            return "empty"
        # Use length bucket + key terms
        length_bucket = "short" if len(text) < 20 else ("long" if len(text) > 80 else "medium")
        # Extract key domain words (simple heuristic)
        domains = {
            "code": ["code", "function", "class", "import", "def ", "error", "bug", "fix", "test", "python", "js", "ts"],
            "file": ["file", "path", "directory", "folder", "read", "write", "open", "save", "delete"],
            "desktop": ["click", "type", "window", "screen", "app", "launch", "browser", "notepad"],
            "qa": ["what", "how", "why", "where", "when", "who", "explain", "tell", "show", "find"],
            "task": ["do", "run", "execute", "build", "create", "make", "deploy", "check"],
        }
        matched = []
        for domain, keywords in domains.items():
            if any(kw in text for kw in keywords):
                matched.append(domain)
        domain_str = "+".join(matched) if matched else "general"
        return f"{domain_str}:{length_bucket}"

    # ── Learning actions ─────────────────────────────────────────────────────

    def _learn_from_failures(self, key: str, pat: LearnPattern) -> Optional[Dict[str, Any]]:
        """Learn from a cluster of failures."""
        actions = []
        events = pat.events[-self.learn_min_events * 2 :]  # recent relevant events

        # 1. If there's a rule involved, penalize it
        rule_name = ""
        for e in events:
            if e.rule_name:
                rule_name = e.rule_name
                break
        if rule_name:
            rf = self._get_rule_feedback()
            result = rf.update(rule_name, success=False, weight=0.5)
            actions.append({"type": "rule_update", "rule": rule_name, "result": result})

        # 2. Record the failure pattern for RAG retrieval
        self._record_experience(events, success=False)
        actions.append({"type": "experience_recorded", "count": len(events)})

        # 3. Reflect on failures to generate suggestions
        suggestion = self._reflect_on_failures(events)
        pat.suggestion = suggestion
        if suggestion:
            actions.append({"type": "reflection", "suggestion": suggestion})

        # 4. Track failure for continual learning (EWC importance estimation)
        self._track_for_continual(events, success=False)

        # 5. Submit to evolution engine for strategy proposal
        evo_result = self._propose_evolution(events, success=False)
        if evo_result:
            actions.append({"type": "evolution_proposal", "result": evo_result})

        logger.info(
            "LearningLoop: learned from %d failures [%s] — %d actions",
            pat.count, key, len(actions),
        )
        return {"pattern": key, "count": pat.count, "success_rate": pat.success_rate, "actions": actions}

    def _learn_from_success(self, key: str, pat: LearnPattern) -> Optional[Dict[str, Any]]:
        """Reinforce patterns that consistently work well."""
        actions = []
        events = pat.events[-self.learn_min_events :]

        # 1. Reinforce good rules
        rule_name = ""
        for e in events:
            if e.rule_name:
                rule_name = e.rule_name
                break
        if rule_name:
            rf = self._get_rule_feedback()
            result = rf.update(rule_name, success=True, weight=1.0)
            actions.append({"type": "rule_reinforce", "rule": rule_name, "result": result})

        # 2. Record successful patterns for future reference
        self._record_experience(events, success=True)
        actions.append({"type": "experience_recorded", "count": len(events)})

        logger.info(
            "LearningLoop: reinforced %d successes [%s] — score=%.2f",
            pat.count, key, pat.success_rate,
        )
        return {"pattern": key, "count": pat.count, "success_rate": pat.success_rate, "actions": actions}

    def _learn_from_explicit(self, event: LearnEvent) -> Optional[Dict[str, Any]]:
        """Learn from explicit user feedback."""
        actions = []
        rf = self._get_rule_feedback()
        rule = event.rule_name or "user_feedback"
        result = rf.update(rule, success=event.success, weight=2.0)  # explicit feedback weighs more
        actions.append({"type": "explicit_feedback", "rule": rule, "result": result})

        self._record_experience([event], success=event.success)
        actions.append({"type": "experience_recorded"})

        logger.info("LearningLoop: explicit feedback → rule=%s success=%s", rule, event.success)
        return {"pattern": "explicit", "actions": actions}

    # ── Support modules (lazy-loaded) ────────────────────────────────────────

    def _get_rule_feedback(self):
        if self._rule_feedback is None:
            try:
                from system.brain._reasoning_feedback import RuleFeedbackMemory
                self._rule_feedback = RuleFeedbackMemory(path=str(self._rule_feedback_path))
                self._rule_feedback.load()
            except Exception:
                logger.debug("LearningLoop: RuleFeedbackMemory not available", exc_info=True)
                self._rule_feedback = _DummyRuleFeedback()
        return self._rule_feedback

    def _get_experience_store(self):
        if self._experience_store is None:
            try:
                from system.evaluation.experience_store import ExperienceStore
                self._experience_store = ExperienceStore(path=str(self._experience_path))
            except Exception:
                logger.debug("LearningLoop: ExperienceStore not available", exc_info=True)
                self._experience_store = _DummyStore()
        return self._experience_store

    def _record_experience(self, events: List[LearnEvent], success: bool) -> None:
        """Record a learning episode to the experience store."""
        if not events:
            return
        try:
            store = self._get_experience_store()
            queries = [e.query for e in events if e.query]
            combined_text = " | ".join(queries[:3])
            plan_steps = [
                f"{e.route_kind}:{e.source}→{e.final_source}({e.validation})"
                for e in events
            ]
            avg_score = sum(1.0 for e in events if e.success) / max(1, len(events))
            store.add(
                state_text=combined_text,
                goal=events[0].route_kind or "learning",
                plan_steps=plan_steps,
                success=success,
                score=float(avg_score),
                meta={
                    "source": "LearningLoop",
                    "pattern_count": len(events),
                    "route_kind": events[0].route_kind,
                },
            )
        except Exception:
            logger.debug("LearningLoop: failed to record experience", exc_info=True)

    def _reflect_on_failures(self, events: List[LearnEvent]) -> str:
        """Analyze failures and produce a human-readable suggestion."""
        if not events:
            return ""
        try:
            from system.learning.dialogue.dialogue_evolver import DialogueFailureReflector
            if self._failure_reflector is None:
                self._failure_reflector = DialogueFailureReflector()
            # Collect the queries and outcomes
            samples = [
                {"query": e.query, "validation": e.validation, "success": e.success}
                for e in events[-5:]
            ]
            return self._failure_reflector.analyze(samples)
        except Exception:
            pass

        # Fallback: simple heuristic
        validation_kinds = set(e.validation for e in events)
        if "evidence_issue" in validation_kinds:
            return "Consider adding more grounding evidence to the knowledge base for these query types."
        if "thread_context_issue" in validation_kinds:
            return "The system is losing track of context — consider shorter conversation turns."
        if "fallback" in validation_kinds:
            return "LLM frequently fails on this type of query — consider adding curated examples."
        return "Review these failure patterns to identify missing knowledge or prompt improvements."

    # ── Evolution & continual learning hooks ──────────────────────────────────

    def _get_evolution_engine(self):
        if self._evolution_engine is None:
            try:
                from system.evaluation.evolution import EvolutionEngine
                self._evolution_engine = EvolutionEngine()
            except Exception:
                self._evolution_engine = False  # mark as unavailable
        return self._evolution_engine if self._evolution_engine is not False else None

    def _get_continual_memory(self):
        if self._continual_memory is None:
            try:
                from system.learning.continual import EWCMemory
                self._continual_memory = EWCMemory()
            except Exception:
                self._continual_memory = False
        return self._continual_memory if self._continual_memory is not False else None

    def _track_for_continual(self, events: List[LearnEvent], success: bool) -> None:
        """Feed events into EWC-based continual learning for importance estimation."""
        ewc = self._get_continual_memory()
        if ewc is None:
            return
        try:
            for e in events[-3:]:
                task_id = self._pattern_key(e)
                loss = 0.0 if success else 1.0
                ewc.record(task_id, loss)
        except Exception:
            pass

    def _propose_evolution(self, events: List[LearnEvent], success: bool) -> Optional[Dict[str, Any]]:
        """Ask the EvolutionEngine to analyze failures and propose improvements."""
        engine = self._get_evolution_engine()
        if engine is None:
            return None
        try:
            stats = engine.analyze_failures(window=min(50, len(events)))
            if stats:
                proposal = engine.auto_evolve()
                return {"stats": stats, "proposal": proposal}
            return None
        except Exception:
            return None

    def _save_patterns(self) -> None:
        """Persist learning patterns for inspection."""
        try:
            self._pattern_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": time.time(),
                "total_patterns": len(self._patterns),
                "total_observations": self._total_observations,
                "total_learn_actions": self._total_learn_actions,
                "patterns": {
                    key: {
                        "count": pat.count,
                        "success_rate": pat.success_rate,
                        "suggestion": pat.suggestion,
                        "last_event_ts": pat.events[-1].ts if pat.events else 0,
                    }
                    for key, pat in self._patterns.items()
                    if pat.count >= self.learn_min_events
                },
            }
            self._pattern_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass


# ── Dummy fallbacks ──────────────────────────────────────────────────────────

class _DummyRuleFeedback:
    def update(self, rule, success, weight=1.0):
        return {"rule": rule, "wins": 1.0 if success else 0.0, "total": 1.0}

    def prior(self, rule):
        return 0.5

    def load(self):
        pass


class _DummyStore:
    def add(self, **kwargs):
        pass
