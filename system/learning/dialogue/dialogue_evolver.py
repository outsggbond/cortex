from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from system.evaluation.response_guard import guard_response, is_low_signal_response


def _char_jaccard(a: str, b: str) -> float:
    sa = set((a or "").strip())
    sb = set((b or "").strip())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    if union <= 0:
        return 0.0
    return float(inter / union)


def _repeat_ratio(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    uniq = len(set(t))
    return 1.0 - (uniq / max(1, len(t)))


def score_response_quality(query: str, answer: str, expected: str = "") -> float:
    q = (query or "").strip()
    a = (answer or "").strip()
    e = (expected or "").strip()
    if not a:
        return 0.0
    guarded = 1.0 if guard_response(q, a) is not None else 0.0
    low_signal = 1.0 if is_low_signal_response(a) else 0.0
    ref_len = max(8.0, float(os.environ.get("DIALOGUE_QUALITY_REF_LEN", "40")))
    length_score = min(1.0, len(a) / ref_len)
    rep_limit = max(0.1, float(os.environ.get("DIALOGUE_QUALITY_MAX_REPEAT", "0.75")))
    rep = _repeat_ratio(a)
    rep_score = 1.0 - min(1.0, rep / rep_limit)
    base = (0.30 * guarded) + (0.25 * (1.0 - low_signal)) + (0.25 * length_score) + (0.20 * rep_score)
    if e:
        expected_sim = _char_jaccard(a, e)
        base = (0.55 * base) + (0.45 * expected_sim)
    return float(max(0.0, min(1.0, base)))


@dataclass
class RouteStats:
    attempts: int = 0
    successes: int = 0
    quality_sum: float = 0.0
    fail_streak: int = 0
    context_attempts: Dict[str, int] = field(default_factory=dict)
    context_successes: Dict[str, int] = field(default_factory=dict)
    context_quality_sum: Dict[str, float] = field(default_factory=dict)

    def avg_quality(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return float(self.quality_sum / max(1, self.attempts))

    def success_rate(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return float(self.successes / max(1, self.attempts))

    def context_score(self, context: str) -> float:
        key = str(context or "").strip()
        if not key:
            return 0.0
        attempts = int(self.context_attempts.get(key, 0))
        if attempts <= 0:
            return 0.0
        succ = float(self.context_successes.get(key, 0))
        qsum = float(self.context_quality_sum.get(key, 0.0))
        succ_rate = succ / max(1.0, float(attempts))
        qavg = qsum / max(1.0, float(attempts))
        return float((0.55 * succ_rate) + (0.45 * qavg))


class DialogueRouteSelector:
    def __init__(
        self,
        stats_path: str = "artifacts/audit/dialogue_route_stats.json",
        trace_path: str = "artifacts/audit/dialogue_route_trace.jsonl",
        enabled: Optional[bool] = None,
        auto_persist: Optional[bool] = None,
    ) -> None:
        self.stats_path = Path(stats_path or "artifacts/audit/dialogue_route_stats.json")
        self.trace_path = Path(trace_path or "artifacts/audit/dialogue_route_trace.jsonl")
        if enabled is None:
            self.enabled = os.environ.get("DIALOGUE_ROUTE_SELECT", "1") != "0"
        else:
            self.enabled = bool(enabled)
        if auto_persist is None:
            self.auto_persist = os.environ.get("DIALOGUE_ROUTE_PERSIST", "1") != "0"
        else:
            self.auto_persist = bool(auto_persist)
        self.explore = max(0.0, float(os.environ.get("DIALOGUE_ROUTE_EXPLORE", "0.2")))
        self.prior = max(0.0, min(1.0, float(os.environ.get("DIALOGUE_ROUTE_PRIOR", "0.45"))))
        self.save_every = max(1, int(os.environ.get("DIALOGUE_ROUTE_SAVE_EVERY", "8")))
        self._pending_since_save = 0
        self._stats: Dict[str, RouteStats] = {}
        self.load()

    def _score_route(self, route: str, context: str, total_attempts: int, idx: int) -> Dict[str, Any]:
        s = self._stats.get(route)
        attempts = int(s.attempts) if s is not None else 0
        successes = int(s.successes) if s is not None else 0
        avg_q = float(s.avg_quality()) if s is not None and attempts > 0 else self.prior
        succ = float(s.success_rate()) if s is not None and attempts > 0 else self.prior
        ctx = float(s.context_score(context)) if s is not None and attempts > 0 else self.prior
        fail_streak = int(s.fail_streak) if s is not None else 0
        base = (0.40 * avg_q) + (0.35 * succ) + (0.25 * ctx)
        base -= min(0.25, 0.04 * max(0, fail_streak))
        explore = 0.0
        if self.enabled:
            explore = self.explore * math.sqrt(math.log(total_attempts + 1.0) / float(attempts + 1))
        score = float(base + explore)
        return {
            "route": route,
            "score": score,
            "base_score": float(base),
            "explore_bonus": float(explore),
            "attempts": attempts,
            "successes": successes,
            "success_rate": float(succ),
            "avg_quality": float(avg_q),
            "context_score": float(ctx),
            "fail_streak": fail_streak,
            "_idx": int(idx),
        }

    def leaderboard(
        self,
        context: str = "chat",
        limit: int = 8,
        routes: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        route_list: List[str] = []
        seen = set()
        source = routes if routes is not None else list(self._stats.keys())
        for r in source:
            key = str(r or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            route_list.append(key)
        if not route_list:
            return []
        total_attempts = sum(int(s.attempts) for s in self._stats.values()) + 1
        scored = [
            self._score_route(route=route, context=context, total_attempts=total_attempts, idx=idx)
            for idx, route in enumerate(route_list)
        ]
        scored.sort(key=lambda x: (float(x["score"]), -int(x["_idx"])), reverse=True)
        out: List[Dict[str, Any]] = []
        for item in scored[: max(1, int(limit))] if int(limit) > 0 else scored:
            clean = dict(item)
            clean.pop("_idx", None)
            out.append(clean)
        return out

    def choose_order(self, routes: Sequence[str], context: str = "chat") -> List[str]:
        dedup: List[str] = []
        seen = set()
        for r in routes:
            key = str(r or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            dedup.append(key)
        if len(dedup) <= 1:
            return dedup
        board = self.leaderboard(context=context, limit=0, routes=dedup)
        if not board:
            return dedup
        return [str(item.get("route", "")) for item in board if str(item.get("route", ""))]

    def record(
        self,
        route: str,
        success: bool,
        quality: float,
        context: str = "chat",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = str(route or "").strip()
        if not key:
            return
        q = float(max(0.0, min(1.0, quality)))
        s = self._stats.get(key)
        if s is None:
            s = RouteStats()
            self._stats[key] = s
        s.attempts += 1
        if success:
            s.successes += 1
            s.fail_streak = 0
        else:
            s.fail_streak += 1
        s.quality_sum += q
        ctx = str(context or "").strip() or "chat"
        s.context_attempts[ctx] = int(s.context_attempts.get(ctx, 0)) + 1
        if success:
            s.context_successes[ctx] = int(s.context_successes.get(ctx, 0)) + 1
        s.context_quality_sum[ctx] = float(s.context_quality_sum.get(ctx, 0.0)) + q
        self._write_trace(
            {
                "ts": time.time(),
                "route": key,
                "context": ctx,
                "success": bool(success),
                "quality": q,
                "attempts": int(s.attempts),
                "successes": int(s.successes),
                "avg_quality": float(s.avg_quality()),
                "success_rate": float(s.success_rate()),
                "fail_streak": int(s.fail_streak),
                "meta": meta or {},
            }
        )
        self._pending_since_save += 1
        if self.auto_persist and self._pending_since_save >= self.save_every:
            self.save()

    def save(self) -> bool:
        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "ts": time.time(),
                "routes": {
                    route: {
                        "attempts": int(stats.attempts),
                        "successes": int(stats.successes),
                        "quality_sum": float(stats.quality_sum),
                        "fail_streak": int(stats.fail_streak),
                        "context_attempts": dict(stats.context_attempts),
                        "context_successes": dict(stats.context_successes),
                        "context_quality_sum": dict(stats.context_quality_sum),
                    }
                    for route, stats in self._stats.items()
                },
            }
            self.stats_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self._pending_since_save = 0
            return True
        except Exception:
            return False

    def load(self) -> bool:
        if not self.stats_path.exists():
            return False
        try:
            raw = json.loads(self.stats_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return False
            routes = raw.get("routes")
            if not isinstance(routes, dict):
                return False
            out: Dict[str, RouteStats] = {}
            for route, item in routes.items():
                if not isinstance(item, dict):
                    continue
                out[str(route)] = RouteStats(
                    attempts=int(item.get("attempts", 0)),
                    successes=int(item.get("successes", 0)),
                    quality_sum=float(item.get("quality_sum", 0.0)),
                    fail_streak=int(item.get("fail_streak", 0)),
                    context_attempts={
                        str(k): int(v)
                        for k, v in (item.get("context_attempts", {}) or {}).items()
                        if isinstance(v, (int, float))
                    },
                    context_successes={
                        str(k): int(v)
                        for k, v in (item.get("context_successes", {}) or {}).items()
                        if isinstance(v, (int, float))
                    },
                    context_quality_sum={
                        str(k): float(v)
                        for k, v in (item.get("context_quality_sum", {}) or {}).items()
                        if isinstance(v, (int, float))
                    },
                )
            self._stats = out
            self._pending_since_save = 0
            return True
        except Exception:
            return False

    def _write_trace(self, payload: Dict[str, Any]) -> None:
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass


class DialogueFailureReflector:
    def __init__(self, path: str = "artifacts/memory/dialogue_reflections.jsonl") -> None:
        self.path = Path(path or "artifacts/memory/dialogue_reflections.jsonl")

    def record(
        self,
        query: str,
        reply: str,
        route: str,
        reason: str,
        *,
        context: str = "chat",
        quality: Optional[float] = None,
        expected: str = "",
        repaired: str = "",
    ) -> Dict[str, Any]:
        q = (query or "").strip()
        a = (reply or "").strip()
        e = (expected or "").strip()
        r = (repaired or "").strip()
        tags: List[str] = []
        if not a:
            tags.append("empty_reply")
        if is_low_signal_response(a):
            tags.append("low_signal")
        if _repeat_ratio(a) > float(os.environ.get("LEARN_MAX_REPEAT", "0.7")):
            tags.append("high_repeat")
        exp_sim = _char_jaccard(a, e) if e else None
        if exp_sim is not None and exp_sim < float(os.environ.get("DIALOGUE_EXPECTED_SIM_MIN", "0.35")):
            tags.append("expected_mismatch")
        action = "adjust_route_weight"
        if "expected_mismatch" in tags:
            action = "writeback_expected_answer"
        elif "low_signal" in tags:
            action = "block_low_signal_learning"
        payload: Dict[str, Any] = {
            "ts": time.time(),
            "context": str(context or "chat"),
            "route": str(route or ""),
            "reason": str(reason or ""),
            "query": q,
            "reply": a,
            "quality": float(quality) if quality is not None else score_response_quality(q, a),
            "expected": e,
            "expected_sim": exp_sim,
            "repaired": r,
            "tags": tags,
            "suggested_action": action,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return payload


class DialogueSelfPlay:
    def __init__(
        self,
        log_path: str = "artifacts/audit/dialogue_selfplay.jsonl",
        enabled: Optional[bool] = None,
        every_turns: Optional[int] = None,
        batch_size: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.log_path = Path(log_path or "artifacts/audit/dialogue_selfplay.jsonl")
        if enabled is None:
            self.enabled = os.environ.get("DIALOGUE_SELFPLAY_ENABLE", "1") != "0"
        else:
            self.enabled = bool(enabled)
        self.every_turns = max(1, int(every_turns or os.environ.get("DIALOGUE_SELFPLAY_EVERY", "8")))
        self.batch_size = max(1, int(batch_size or os.environ.get("DIALOGUE_SELFPLAY_BATCH", "3")))
        self.min_expected_sim = float(os.environ.get("DIALOGUE_SELFPLAY_MIN_EXPECTED_SIM", "0.35"))
        self.min_quality = float(os.environ.get("DIALOGUE_SELFPLAY_MIN_QUALITY", "0.5"))
        self._rng = random.Random(int(seed or os.environ.get("DIALOGUE_SELFPLAY_SEED", "7")))

    def _has_expected_pair(self, dialogue_gen, intent: Optional[str], user: str, expected: str) -> bool:
        memory = getattr(dialogue_gen, "memory", None)
        patterns = getattr(memory, "patterns", None)
        if not isinstance(patterns, dict):
            return False
        key = str(intent or "").strip()
        pattern = patterns.get(key)
        if pattern is None:
            return False
        pair = (str(user or "").strip(), str(expected or "").strip())
        if not pair[0] or not pair[1]:
            return False
        pair_set = getattr(pattern, "_pair_set", None)
        if isinstance(pair_set, set):
            return pair in pair_set
        pairs = getattr(pattern, "pairs", None)
        if not isinstance(pairs, list):
            return False
        return pair in pairs

    def should_run(self, turn_id: int) -> bool:
        if not self.enabled:
            return False
        if int(turn_id) <= 0:
            return False
        return int(turn_id) % int(self.every_turns) == 0

    def run(
        self,
        *,
        turn_id: int,
        dialogue_gen,
        available_routes: Sequence[str],
        selector: Optional[DialogueRouteSelector],
        run_route: Callable[[str, str, Optional[str]], str],
        reflector: Optional[DialogueFailureReflector] = None,
    ) -> Dict[str, Any]:
        pairs = self._sample_pairs(dialogue_gen, limit=self.batch_size)
        routes = [str(r).strip() for r in available_routes if str(r).strip()]
        if not pairs or not routes:
            return {"turn_id": int(turn_id), "runs": 0, "passed": 0, "failed": 0, "writeback": 0}
        runs = 0
        passed = 0
        failed = 0
        writeback = 0
        for intent, user, expected in pairs:
            route_order = selector.choose_order(routes, context="selfplay") if selector is not None else list(routes)
            route = route_order[0] if route_order else routes[0]
            try:
                reply = str(run_route(route, user, intent) or "").strip()
            except Exception:
                reply = ""
            quality = score_response_quality(user, reply, expected=expected)
            sim = _char_jaccard(reply, expected)
            guarded = guard_response(user, reply) is not None
            ok = bool(guarded and quality >= float(self.min_quality) and sim >= float(self.min_expected_sim))
            if selector is not None:
                selector.record(
                    route=route,
                    success=ok,
                    quality=quality,
                    context="selfplay",
                    meta={"turn_id": int(turn_id), "expected_sim": sim},
                )
            if ok:
                passed += 1
            else:
                failed += 1
                wrote = False
                try:
                    wrote = bool(
                        dialogue_gen.learn_sample(
                            user_input=user,
                            response=expected,
                            intent=intent,
                            persist=False,
                        )
                    )
                except Exception:
                    wrote = False
                if wrote or self._has_expected_pair(dialogue_gen, intent, user, expected):
                    writeback += 1
                if reflector is not None:
                    reflector.record(
                        query=user,
                        reply=reply,
                        route=route,
                        reason="selfplay_mismatch",
                        context="selfplay",
                        quality=quality,
                        expected=expected,
                        repaired=expected if wrote else "",
                    )
            runs += 1
        out = {
            "ts": time.time(),
            "turn_id": int(turn_id),
            "runs": int(runs),
            "passed": int(passed),
            "failed": int(failed),
            "writeback": int(writeback),
            "routes": routes,
        }
        self._write_log(out)
        return out

    def _sample_pairs(self, dialogue_gen, limit: int) -> List[Tuple[str, str, str]]:
        if dialogue_gen is None:
            return []
        memory = getattr(dialogue_gen, "memory", None)
        patterns = getattr(memory, "patterns", None)
        if not isinstance(patterns, dict):
            return []
        pool: List[Tuple[str, str, str]] = []
        for intent, pattern in patterns.items():
            pairs = getattr(pattern, "pairs", None)
            if not isinstance(pairs, list):
                continue
            for user, assistant in pairs[-200:]:
                u = str(user or "").strip()
                a = str(assistant or "").strip()
                if not u or not a:
                    continue
                pool.append((str(intent), u, a))
        if not pool:
            return []
        if len(pool) <= int(limit):
            self._rng.shuffle(pool)
            return pool
        return self._rng.sample(pool, int(limit))

    def _write_log(self, payload: Dict[str, Any]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
