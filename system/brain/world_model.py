from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from system.core.embeddings import encode_text, cosine, embedding_available
from system.core.planner import Plan
from system.core.error_normalizer import normalize_error

if TYPE_CHECKING:
    from system.core.state_encoder import StateEncoder
    from system.brain.world_predictor import WorldPredictor


logger = logging.getLogger(__name__)


_MISSING_PAT = re.compile(r"(no such file|not found|file not found|找不到|不存在|文件不存在)", re.IGNORECASE)
_PERM_PAT = re.compile(r"(permission denied|access is denied|拒绝访问|权限不足)", re.IGNORECASE)


@dataclass
class WorldState:
    goal: str
    errors: List[str] = field(default_factory=list)
    error_types: List[str] = field(default_factory=list)
    error_signatures: List[str] = field(default_factory=list)
    missing_paths: List[str] = field(default_factory=list)
    perm_paths: List[str] = field(default_factory=list)
    added_count: int = 0
    modified_count: int = 0
    removed_count: int = 0
    total_files: int = 0
    notes: List[str] = field(default_factory=list)
    active_abstractions: List[Dict[str, Any]] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_text(self) -> str:
        parts = [
            f"goal={self.goal}",
            f"errors={len(self.errors)}",
            f"missing={len(self.missing_paths)}",
            f"perm={len(self.perm_paths)}",
            f"added={self.added_count}",
            f"modified={self.modified_count}",
            f"removed={self.removed_count}",
            f"files={self.total_files}",
        ]
        if self.active_abstractions:
            tokens = [str(a.get("token")) for a in self.active_abstractions if a.get("token")]
            if tokens:
                parts.append("abs=" + ",".join(tokens[:5]))
        if self.error_types:
            parts.append("error_types=" + ",".join(self.error_types[:5]))
        if self.error_signatures:
            parts.append("error_sigs=" + ",".join(self.error_signatures[:3]))
        if self.missing_paths:
            parts.append("missing_paths=" + ",".join(self.missing_paths[:5]))
        if self.perm_paths:
            parts.append("perm_paths=" + ",".join(self.perm_paths[:5]))
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "errors": list(self.errors),
            "error_types": list(self.error_types),
            "error_signatures": list(self.error_signatures),
            "missing_paths": list(self.missing_paths),
            "perm_paths": list(self.perm_paths),
            "added_count": int(self.added_count),
            "modified_count": int(self.modified_count),
            "removed_count": int(self.removed_count),
            "total_files": int(self.total_files),
            "notes": list(self.notes),
            "active_abstractions": list(self.active_abstractions),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WorldState":
        if not data:
            return WorldState(goal="")
        return WorldState(
            goal=data.get("goal", "") or "",
            errors=list(data.get("errors") or []),
            error_types=list(data.get("error_types") or []),
            error_signatures=list(data.get("error_signatures") or []),
            missing_paths=list(data.get("missing_paths") or []),
            perm_paths=list(data.get("perm_paths") or []),
            added_count=int(data.get("added_count") or 0),
            modified_count=int(data.get("modified_count") or 0),
            removed_count=int(data.get("removed_count") or 0),
            total_files=int(data.get("total_files") or 0),
            notes=list(data.get("notes") or []),
            active_abstractions=list(data.get("active_abstractions") or []),
        )


class WorldModel:
    def __init__(self, project_root: str = "."):
        # Delayed imports to avoid circular dependencies
        from system.core.state_encoder import StateEncoder
        from system.brain.world_predictor import WorldPredictor
        
        self.root = Path(project_root).resolve()
        self.ignore = {
            ".venv",
            ".idea",
            ".rollback",
            "__pycache__",
            "checkpoints",
            "audit",
            "node_modules",
        }
        self.max_files = int(os.environ.get("WORLD_MAX_FILES", "8000"))
        self.encoder = StateEncoder()
        self.predictor = WorldPredictor(dim=self.encoder.dim)
        self.predictor.load()

    def build_state(
        self,
        goal: str,
        execution: Optional[Any] = None,
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> WorldState:
        errors: List[str] = []
        error_types: List[str] = []
        error_signatures: List[str] = []
        missing_paths: List[str] = []
        perm_paths: List[str] = []
        added = modified = removed = 0
        total_files = 0

        if execution is not None:
            for err in getattr(execution, "errors", []) or []:
                etype = (err.get("type") or "").strip()
                msg = (err.get("error") or "").strip()
                errors.append(msg or etype)
                cat, sig = normalize_error(etype, msg)
                if cat:
                    error_types.append(cat)
                if sig:
                    error_signatures.append(sig)
                payload = err.get("payload") if isinstance(err.get("payload"), dict) else {}
                path = (payload.get("path") or "").strip()
                if (etype == "FileNotFoundError" or _MISSING_PAT.search(msg)) and path:
                    missing_paths.append(path)
                if _PERM_PAT.search(msg) and path:
                    perm_paths.append(path)
            snap = getattr(execution, "snapshot", None)
            if isinstance(snap, dict):
                diff = snap.get("diff") if isinstance(snap.get("diff"), dict) else {}
                added = int(diff.get("added_count", 0))
                modified = int(diff.get("modified_count", 0))
                removed = int(diff.get("removed_count", 0))
                total_files = int(snap.get("after_count", 0) or snap.get("before_count", 0))

        if snapshot is None:
            snapshot = self.capture_snapshot()
        if snapshot:
            total_files = total_files or int(snapshot.get("count", 0))

        return WorldState(
            goal=goal or "",
            errors=errors,
            error_types=error_types,
            error_signatures=error_signatures,
            missing_paths=missing_paths,
            perm_paths=perm_paths,
            added_count=added,
            modified_count=modified,
            removed_count=removed,
            total_files=total_files,
        )

    def capture_snapshot(self) -> Dict[str, Any]:
        files = 0
        for path in self.root.rglob("*"):
            if path.is_dir():
                continue
            if self._is_ignored(path):
                continue
            files += 1
            if files >= self.max_files:
                break
        return {
            "ts": time.time(),
            "root": str(self.root),
            "count": files,
        }

    def evaluate_state(self, state: WorldState) -> float:
        # Higher is better.
        score = 1.0
        score -= 0.4 * len(state.errors)
        score -= 0.2 * len(state.missing_paths)
        score -= 0.2 * len(state.perm_paths)
        score -= 0.05 * (state.added_count + state.modified_count + state.removed_count)
        return max(0.0, score)

    def score_plan(self, plan: Plan, state: WorldState, feedback: str = "") -> float:
        if not plan.steps:
            return -1.0
        score = 0.0
        n = len(plan.steps)
        if 2 <= n <= 6:
            score += 0.4
        elif n == 1:
            score += 0.1
        else:
            score -= 0.2
        # Align with goal
        try:
            if embedding_available():
                qv = encode_text(state.goal, require_model=True)
                pv = encode_text(" ".join(s.detail for s in plan.steps), require_model=True)
                score += 0.7 * cosine(qv, pv)
        except Exception:
            logger.debug("world_model: embedding scoring failed", exc_info=True)
        # Error-aware hints
        if state.missing_paths:
            if any("read" in s.detail or "write" in s.detail or "check" in s.detail for s in plan.steps):
                score += 0.15
        if state.perm_paths:
            if any("chmod" in s.detail or "权限" in s.detail for s in plan.steps):
                score += 0.1
        # Feedback penalty
        if feedback:
            bad_terms = [t for t in re.split(r"[\\s,，。；、]+", feedback) if len(t) >= 2]
            penalty = 0.0
            for term in bad_terms[:8]:
                if any(term in s.detail for s in plan.steps):
                    penalty += 0.05
            score -= min(0.3, penalty)
        return score

    def predict(
        self,
        state: WorldState,
        plan: Plan,
        experiences: Optional[List[Dict[str, Any]]] = None,
    ) -> WorldState:
        pred, _conf = self.predict_with_confidence(state, plan, experiences=experiences)
        return pred

    def predict_with_confidence(
        self,
        state: WorldState,
        plan: Plan,
        experiences: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[WorldState, float]:
        # Start from current state
        pred = WorldState.from_dict(state.to_dict())
        steps = [s.detail for s in plan.steps]
        action_sig = self.action_signature(steps)

        # Experience-based delta
        deltas = []
        weights = []
        if experiences:
            for ex in experiences:
                before = ex.get("before_state")
                after = ex.get("after_state")
                if not before or not after:
                    continue
                ex_sig = (ex.get("action_sig") or "").strip().lower()
                if action_sig and ex_sig and action_sig != ex_sig:
                    continue
                b = WorldState.from_dict(before)
                a = WorldState.from_dict(after)
                w = self._state_similarity(state, b)
                if w < 0.15:
                    continue
                deltas.append(
                    {
                        "errors": len(a.errors) - len(b.errors),
                        "missing": len(a.missing_paths) - len(b.missing_paths),
                        "perm": len(a.perm_paths) - len(b.perm_paths),
                        "added": a.added_count - b.added_count,
                        "modified": a.modified_count - b.modified_count,
                        "removed": a.removed_count - b.removed_count,
                    }
                )
                weights.append(w)
        if deltas:
            avg = self._weighted_avg(deltas, weights)
            pred = self._apply_delta(pred, avg)
            conf = min(1.0, max(weights) if weights else 0.4)
        else:
            model_delta, conf = self.predictor.predict(state, action_sig=action_sig)
            if model_delta:
                pred = self._apply_delta(pred, model_delta)
            else:
                # heuristic fallback
                pred = self._apply_heuristic(pred, steps)
                conf = 0.2
        return pred, conf

    def action_signature(self, steps: List[str]) -> str:
        if not steps:
            return ""
        first = steps[0].strip()
        if not first:
            return ""
        # Chinese verb: first 2 chars; otherwise first token
        if re.search(r"[\u4e00-\u9fff]", first):
            return first[:2]
        return first.split()[0].lower()

    def _apply_delta(self, state: WorldState, delta: Dict[str, float]) -> WorldState:
        def clamp(val, lo=0):
            return max(lo, int(round(val)))

        errors = clamp(len(state.errors) + delta.get("errors", 0))
        missing = clamp(len(state.missing_paths) + delta.get("missing", 0))
        perm = clamp(len(state.perm_paths) + delta.get("perm", 0))
        state.errors = state.errors[:errors]
        state.missing_paths = state.missing_paths[:missing]
        state.perm_paths = state.perm_paths[:perm]
        state.added_count = clamp(state.added_count + delta.get("added", 0))
        state.modified_count = clamp(state.modified_count + delta.get("modified", 0))
        state.removed_count = clamp(state.removed_count + delta.get("removed", 0))
        return state

    def _apply_heuristic(self, state: WorldState, steps: List[str]) -> WorldState:
        text = " ".join(steps).lower()
        if any(k in text for k in ("write", "create", "touch", "生成", "创建", "新建")):
            if state.missing_paths:
                state.missing_paths = state.missing_paths[1:]
        if "chmod" in text or "权限" in text:
            if state.perm_paths:
                state.perm_paths = state.perm_paths[1:]
        if any(k in text for k in ("run tests", "pytest", "测试", "验证")):
            if state.errors:
                state.errors = state.errors[1:]
        return state

    def _state_similarity(self, a: WorldState, b: WorldState) -> float:
        try:
            va = self.encoder.encode(a, a.goal)
            vb = self.encoder.encode(b, b.goal)
            denom = (float(va @ va) ** 0.5) * (float(vb @ vb) ** 0.5) + 1e-8
            return float(va @ vb) / denom
        except Exception:
            return 0.0

    def _weighted_avg(self, deltas: List[Dict[str, float]], weights: List[float]) -> Dict[str, float]:
        if not deltas:
            return {}
        if not weights or len(weights) != len(deltas):
            weights = [1.0 for _ in deltas]
        total = sum(weights) or 1.0
        out = {k: 0.0 for k in deltas[0].keys()}
        for d, w in zip(deltas, weights):
            for k in out.keys():
                out[k] += d.get(k, 0.0) * w
        for k in out.keys():
            out[k] /= total
        return out

    def update_predictor(self, before: WorldState, after: WorldState, action_sig: str = "") -> None:
        try:
            self.predictor.update(before, after, action_sig=action_sig)
            self.predictor.save()
        except Exception:
            logger.debug("world_model: predictor update failed", exc_info=True)

    def _is_ignored(self, path: Path) -> bool:
        return any(part in self.ignore for part in path.parts)
