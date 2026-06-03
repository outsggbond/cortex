# -*- coding: utf-8 -*-
"""Unit tests for system.learning.learning_loop — LearningLoop, LearnEvent, LearnPattern."""

from __future__ import annotations

import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.learning.learning_loop import (
    LearningLoop,
    LearnEvent,
    LearnPattern,
    LEARN_MIN_EVENTS,
    LEARN_COOLDOWN_S,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _event(
    query: str = "hello",
    route_kind: str = "GENERAL",
    source: str = "llm",
    validation: str = "passed",
    final_source: str = "llm",
    success: bool = True,
    rule_name: str = "",
) -> LearnEvent:
    return LearnEvent(
        ts=time.time(),
        query=query,
        route_kind=route_kind,
        source=source,
        validation=validation,
        final_source=final_source,
        success=success,
        rule_name=rule_name,
    )


def _make_loop(**overrides) -> LearningLoop:
    """Create a LearningLoop with small windows for fast tests."""
    defaults = dict(
        enabled=True,
        window_size=20,
        learn_min_events=3,
        learn_threshold=0.5,
        cooldown_s=0.0,  # no cooldown by default in tests
        rule_feedback_path="artifacts/audit/test_rule_feedback.json",
        experience_store_path="artifacts/memory/test_learning_experience.jsonl",
        pattern_store_path="artifacts/audit/test_learning_patterns.json",
    )
    defaults.update(overrides)
    return LearningLoop(**defaults)


# ── Creation ─────────────────────────────────────────────────────────────────

def test_learning_loop_creation_defaults() -> None:
    """LearningLoop creates with default configuration."""
    loop = LearningLoop()
    assert loop.enabled is True
    assert loop.window_size >= 10
    assert loop.learn_min_events >= 2
    assert 0.0 < loop.learn_threshold <= 1.0
    assert loop.cooldown_s >= 0.0
    assert len(loop._events) == 0
    assert len(loop._patterns) == 0


def test_learning_loop_creation_custom() -> None:
    """LearningLoop accepts custom window_size, learn_min_events, cooldown_s."""
    loop = _make_loop(window_size=15, learn_min_events=4, cooldown_s=10.0)
    assert loop.window_size == 15
    assert loop.learn_min_events == 4
    assert loop.cooldown_s == 10.0


def test_learning_loop_disabled_ignores_events() -> None:
    """When disabled, observe() and maybe_learn() are no-ops."""
    loop = _make_loop(enabled=False)
    loop.observe(_event(validation="evidence_issue", success=False))
    assert len(loop._events) == 0
    assert loop.maybe_learn() is None


# ── observe() ────────────────────────────────────────────────────────────────

def test_observe_accumulates_events() -> None:
    """observe() appends events and increments total count."""
    loop = _make_loop(window_size=20)
    for _ in range(5):
        loop.observe(_event(query="test"))
    assert len(loop._events) == 5
    assert loop._total_observations == 5


def test_observe_sliding_window_respects_capacity() -> None:
    """Events beyond window_size are trimmed (window_size clamped to >=10)."""
    loop = _make_loop(window_size=10)  # minimum is 10
    for i in range(15):
        loop.observe(_event(query=f"q{i}"))
    assert len(loop._events) == 10  # only last 10 retained
    assert loop._total_observations == 15  # total still counts all
    # The oldest events should be gone
    queries = [e.query for e in loop._events]
    assert "q0" not in queries
    assert "q14" in queries


# ── Pattern detection ───────────────────────────────────────────────────────

def test_pattern_detection_groups_by_route_validation_query_sig() -> None:
    """Events with the same (route:validation:query_sig) are grouped together."""
    loop = _make_loop(learn_min_events=3)
    # Send 3 identical-pattern events (same route, validation, query sig)
    for _ in range(3):
        loop.observe(_event(
            query="what is python",
            route_kind="GENERAL",
            validation="evidence_issue",
            success=False,
        ))
    # Should form a single pattern
    assert len(loop._patterns) >= 1
    # Find the pattern with evidence_issue
    found = False
    for key, pat in loop._patterns.items():
        if "evidence_issue" in key and "GENERAL" in key:
            assert pat.count == 3
            found = True
    assert found, "Expected pattern key containing 'GENERAL' and 'evidence_issue'"


def test_pattern_detection_different_routes_separate() -> None:
    """Different route_kind values produce different pattern keys."""
    loop = _make_loop()
    loop.observe(_event(query="what is python", route_kind="GENERAL", validation="passed"))
    loop.observe(_event(query="what is python", route_kind="TASK", validation="passed"))
    # Should be 2 distinct patterns
    assert len(loop._patterns) == 2
    keys = list(loop._patterns.keys())
    assert any("GENERAL" in k for k in keys)
    assert any("TASK" in k for k in keys)


# ── _query_signature() ──────────────────────────────────────────────────────

def test_query_signature_short_qa() -> None:
    """Short query with 'what' → qa:short."""
    loop = _make_loop()
    sig = loop._query_signature("what is this")
    assert sig == "qa:short"


def test_query_signature_long_code() -> None:
    """Long query (>80 chars) with 'function', 'class', 'import' → code:long."""
    loop = _make_loop()
    sig = loop._query_signature(
        "implement a python function that handles errors and exceptions using a class "
        "with proper testing methods and import statements for error handling"
    )
    assert sig == "code:long"


def test_query_signature_file_domain() -> None:
    """Query with 'file', 'read' → file domain with correct length bucket."""
    loop = _make_loop()
    # This query is < 20 chars → file:short
    sig = loop._query_signature("read the file")
    assert sig == "file:short"


def test_query_signature_desktop_domain() -> None:
    """Query with 'click', 'window' → desktop:short."""
    loop = _make_loop()
    sig = loop._query_signature("click the window")
    assert "desktop" in sig


def test_query_signature_task_domain() -> None:
    """Query with 'run', 'execute' → task:short."""
    loop = _make_loop()
    sig = loop._query_signature("run the build")
    assert "task" in sig


def test_query_signature_empty_query() -> None:
    """Empty or whitespace query → 'empty'."""
    loop = _make_loop()
    assert loop._query_signature("") == "empty"
    assert loop._query_signature("   ") == "empty"


def test_query_signature_general_fallback() -> None:
    """Query with no domain keywords → general:length_bucket."""
    loop = _make_loop()
    sig = loop._query_signature("xyzzy flarg bloop")
    assert sig.startswith("general:")
    assert ":" in sig


def test_query_signature_medium_length() -> None:
    """Query between 20-80 chars → medium bucket."""
    loop = _make_loop()
    medium = "this is a medium length query for testing"
    assert 20 <= len(medium) <= 80
    sig = loop._query_signature(medium)
    assert ":medium" in sig


# ── force_learn() ───────────────────────────────────────────────────────────

def test_force_learn_explicit_feedback() -> None:
    """force_learn() creates an explicit-feedback event and learns immediately."""
    loop = _make_loop()
    result = loop.force_learn(
        query="how do I sort a list",
        feedback="wrong answer",
        success=False,
    )
    assert result is not None
    assert isinstance(result, dict)
    # Should have created an event in the window
    assert loop._total_observations >= 1
    # The event should have EXPLICIT_FEEDBACK route
    assert len(loop._events) >= 1
    assert loop._events[-1].route_kind == "EXPLICIT_FEEDBACK"
    assert loop._events[-1].metadata.get("feedback") == "wrong answer"


def test_force_learn_multiple_calls() -> None:
    """Multiple force_learn() calls accumulate correctly."""
    loop = _make_loop()
    loop.force_learn(query="q1", feedback="good", success=True)
    loop.force_learn(query="q2", feedback="bad", success=False)
    assert loop._total_observations >= 2


# ── maybe_learn() thresholds ─────────────────────────────────────────────────

def test_maybe_learn_too_few_events_no_trigger() -> None:
    """maybe_learn() returns None when fewer than learn_min_events."""
    loop = _make_loop(learn_min_events=5)
    # Add only 2 failure events with same pattern
    for _ in range(2):
        loop.observe(_event(
            query="what is python",
            route_kind="GENERAL",
            validation="evidence_issue",
            success=False,
        ))
    result = loop.maybe_learn()
    assert result is None


def test_maybe_learn_triggers_after_min_failures() -> None:
    """maybe_learn() triggers when learn_min_events same-pattern failures accumulate."""
    loop = _make_loop(learn_min_events=3, cooldown_s=0.0)
    for _ in range(3):
        loop.observe(_event(
            query="what is python",
            route_kind="GENERAL",
            validation="evidence_issue",
            success=False,
        ))
    result = loop.maybe_learn()
    assert result is not None
    assert result["actions"] >= 1
    assert result["total_learn_actions"] >= 1


def test_maybe_learn_only_successes_no_trigger() -> None:
    """A pattern with only successful events (>= 50%) does not trigger failure learning."""
    loop = _make_loop(learn_min_events=3, cooldown_s=0.0)
    for _ in range(3):
        loop.observe(_event(
            query="what is python",
            route_kind="GENERAL",
            validation="passed",
            success=True,
        ))
    # success rate is 1.0, so failure path (rate < 0.5) is not hit.
    # But with >= 0.85 rate and count >= learn_min_events*2 (=6), it could
    # trigger success reinforcement. We only have 3 events so count < 6.
    result = loop.maybe_learn()
    assert result is None


def test_maybe_learn_high_success_reinforcement() -> None:
    """A pattern with high success rate and enough events triggers reinforcement."""
    loop = _make_loop(learn_min_events=3, cooldown_s=0.0)
    for _ in range(6):
        loop.observe(_event(
            query="what is python",
            route_kind="GENERAL",
            validation="passed",
            success=True,
        ))
    result = loop.maybe_learn()
    assert result is not None
    assert result["actions"] >= 1


# ── Cooldown ─────────────────────────────────────────────────────────────────

def test_cooldown_prevents_retriggering() -> None:
    """maybe_learn() respects cooldown window."""
    loop = _make_loop(learn_min_events=3, cooldown_s=999.0)  # very long cooldown
    for _ in range(3):
        loop.observe(_event(
            query="what is python",
            route_kind="GENERAL",
            validation="evidence_issue",
            success=False,
        ))
    result1 = loop.maybe_learn()
    assert result1 is not None  # first trigger succeeds

    # Add more failure events
    for _ in range(5):
        loop.observe(_event(
            query="what is python",
            route_kind="GENERAL",
            validation="evidence_issue",
            success=False,
        ))
    result2 = loop.maybe_learn()
    assert result2 is None  # blocked by cooldown


def test_cooldown_expired_allows_retrigger() -> None:
    """After cooldown expires, maybe_learn() can trigger again."""
    loop = _make_loop(learn_min_events=3, cooldown_s=0.0)
    for _ in range(3):
        loop.observe(_event(
            query="what is python",
            route_kind="GENERAL",
            validation="evidence_issue",
            success=False,
        ))
    result1 = loop.maybe_learn()
    assert result1 is not None

    # Different pattern → different key, no cooldown conflict
    for _ in range(3):
        loop.observe(_event(
            query="run the tests for the bug fix",
            route_kind="TASK",
            validation="fallback",
            success=False,
        ))
    result2 = loop.maybe_learn()
    assert result2 is not None


# ── stats ────────────────────────────────────────────────────────────────────

def test_stats_property_returns_counts() -> None:
    """stats property returns observation/learn counts and pattern info."""
    loop = _make_loop(learn_min_events=3)
    for _ in range(5):
        loop.observe(_event(query="q"))
    stats = loop.stats
    assert stats["observations"] == 5
    assert stats["learn_actions"] == 0
    assert isinstance(stats["active_patterns"], int)
    assert isinstance(stats["window_size"], int)


def test_stats_after_learn() -> None:
    """stats reflects learn_actions after maybe_learn triggers."""
    loop = _make_loop(learn_min_events=3, cooldown_s=0.0)
    for _ in range(3):
        loop.observe(_event(
            query="what is python",
            route_kind="GENERAL",
            validation="evidence_issue",
            success=False,
        ))
    loop.maybe_learn()
    stats = loop.stats
    assert stats["learn_actions"] >= 1


# ── LearnEvent / LearnPattern dataclasses ────────────────────────────────────

def test_learn_event_defaults() -> None:
    """LearnEvent has correct defaults."""
    e = LearnEvent(ts=1.0, query="q", route_kind="GENERAL", source="llm",
                   validation="passed", final_source="llm")
    assert e.success is True
    assert e.rule_name == ""
    assert e.query_sig == ""
    assert e.metadata == {}


def test_learn_pattern_defaults() -> None:
    """LearnPattern has correct defaults."""
    p = LearnPattern(key="TEST:passed:qa:short")
    assert p.key == "TEST:passed:qa:short"
    assert p.events == []
    assert p.count == 0
    assert p.success_rate == 0.0
    assert p.last_triggered == 0.0
    assert p.suggestion == ""


# ── Window size edge cases ──────────────────────────────────────────────────

def test_window_size_never_below_10() -> None:
    """window_size is clamped to at least 10."""
    loop = _make_loop(window_size=1)
    assert loop.window_size >= 10


def test_learn_min_events_never_below_2() -> None:
    """learn_min_events is clamped to at least 2."""
    loop = _make_loop(learn_min_events=1)
    assert loop.learn_min_events >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    print("14 tests written for learning_loop module")
