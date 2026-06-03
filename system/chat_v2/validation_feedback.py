# -*- coding: utf-8 -*-
"""Persistence for chat_v2 post-validation outcomes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def _bump(bucket: Dict[str, int], key: str) -> None:
    node = str(key or "").strip() or "unknown"
    bucket[node] = int(bucket.get(node, 0)) + 1


class ValidationFeedbackRecorder:
    def __init__(
        self,
        feedback_path: str = "artifacts/audit/chat_v2_validation_feedback.jsonl",
        stats_path: str = "artifacts/audit/chat_v2_validation_stats.json",
    ) -> None:
        self.feedback_path = Path(str(feedback_path or "artifacts/audit/chat_v2_validation_feedback.jsonl"))
        self.stats_path = Path(str(stats_path or "artifacts/audit/chat_v2_validation_stats.json"))
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(event or {})
        payload["ts"] = float(payload.get("ts", 0.0) or time.time())
        self._append_jsonl(payload)
        stats = self._update_stats(payload)
        return stats

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        with self.feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _load_stats(self) -> Dict[str, Any]:
        if not self.stats_path.exists():
            return self._default_stats()
        try:
            payload = json.loads(self.stats_path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_stats()
        if not isinstance(payload, dict):
            return self._default_stats()
        base = self._default_stats()
        base.update(payload)
        for key in (
            "totals",
            "by_validation",
            "by_override",
            "by_candidate_source",
            "by_final_source",
            "by_route_kind",
        ):
            if not isinstance(base.get(key), dict):
                base[key] = {}
        return base

    def _default_stats(self) -> Dict[str, Any]:
        return {
            "totals": {
                "events": 0,
                "passed": 0,
                "overrides": 0,
            },
            "by_validation": {},
            "by_override": {},
            "by_candidate_source": {},
            "by_final_source": {},
            "by_route_kind": {},
            "last_event": {},
        }

    def _update_stats(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        stats = self._load_stats()
        totals = dict(stats.get("totals", {}) or {})
        totals["events"] = int(totals.get("events", 0)) + 1
        validation = str(payload.get("validation", "") or "unknown")
        override = str(payload.get("validation_override", "") or "unknown")
        candidate_source = str(payload.get("candidate_source", "") or "unknown")
        final_source = str(payload.get("final_source", "") or "unknown")
        route_kind = str(payload.get("route_kind", "") or "unknown")

        if validation == "passed":
            totals["passed"] = int(totals.get("passed", 0)) + 1
        if candidate_source != final_source or validation != "passed":
            totals["overrides"] = int(totals.get("overrides", 0)) + 1

        by_validation = dict(stats.get("by_validation", {}) or {})
        by_override = dict(stats.get("by_override", {}) or {})
        by_candidate_source = dict(stats.get("by_candidate_source", {}) or {})
        by_final_source = dict(stats.get("by_final_source", {}) or {})
        by_route_kind = dict(stats.get("by_route_kind", {}) or {})

        _bump(by_validation, validation)
        _bump(by_override, override)
        _bump(by_candidate_source, candidate_source)
        _bump(by_final_source, final_source)
        _bump(by_route_kind, route_kind)

        stats["totals"] = totals
        stats["by_validation"] = by_validation
        stats["by_override"] = by_override
        stats["by_candidate_source"] = by_candidate_source
        stats["by_final_source"] = by_final_source
        stats["by_route_kind"] = by_route_kind
        stats["last_event"] = {
            "ts": float(payload.get("ts", 0.0) or 0.0),
            "validation": validation,
            "validation_override": override,
            "candidate_source": candidate_source,
            "final_source": final_source,
            "route_kind": route_kind,
        }
        self.stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        return stats


__all__ = ["ValidationFeedbackRecorder"]
