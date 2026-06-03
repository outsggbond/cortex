from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from system.brain.gnn_language import GNNLanguageEngine


@dataclass
class SemanticRefinerConfig:
    enabled: bool = True
    mode: str = "hybrid"  # "stat" | "llm" | "hybrid"
    min_co_count: float = 2.0
    min_cluster_size: int = 3
    max_cluster_size: int = 6
    min_density: float = 0.3
    min_avg_weight: float = 1.0
    decay: float = 0.9
    max_new_tokens: int = 3
    cooldown_s: float = 120.0
    min_resource_score: float = 0.6
    abstract_prefix: str = "abs::"
    state_path: str = "artifacts/memory/semantic_abstractions.json"
    generation_interval: int = 20
    retain_ratio: float = 0.4
    kill_ratio: float = 0.2
    mutate_ratio: float = 0.4
    max_population: int = 50
    merge_overlap: float = 0.6
    decay_on_kill: float = 0.3
    generalization_cap: int = 30
    age_half_life_s: float = 3600.0
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "alpha": 1.0,
            "beta": 0.8,
            "gamma": 0.6,
            "delta": 0.4,
            "epsilon": 0.5,
        }
    )

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticRefinerConfig":
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


@dataclass
class AbstractionRecord:
    token: str
    members: List[str]
    count: int = 1
    score: float = 0.0
    strength: float = 1.0
    fitness: float = 0.0
    usage_count: int = 0
    success_contribution: float = 0.0
    failure_penalty: float = 0.0
    compression_benefit: float = 0.0
    generalization_hits: int = 0
    goal_signatures: List[str] = field(default_factory=list)
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    active: bool = True
    created_ts: float = field(default_factory=time.time)
    last_ts: float = field(default_factory=time.time)


class SemanticRefiner:
    def __init__(
        self,
        gnn: GNNLanguageEngine,
        config: Optional[SemanticRefinerConfig] = None,
        name_fn: Optional[Callable[[List[str]], str]] = None,
    ):
        self.gnn = gnn
        self.config = config or SemanticRefinerConfig()
        self.name_fn = name_fn
        self._last_refine_ts = 0.0
        self._records: Dict[str, AbstractionRecord] = {}
        self._generation = 0
        self._task_count = 0
        self._load_state()

    def _load_state(self) -> None:
        path = Path(self.config.state_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        for key, rec in data.items():
            if not isinstance(rec, dict):
                continue
            members = [str(x) for x in rec.get("members", []) if x]
            token = rec.get("token") or key
            self._records[key] = AbstractionRecord(
                token=str(token),
                members=members,
                count=int(rec.get("count", 1)),
                score=float(rec.get("score", 0.0)),
                strength=float(rec.get("strength", 1.0)),
                fitness=float(rec.get("fitness", 0.0)),
                usage_count=int(rec.get("usage_count", 0)),
                success_contribution=float(rec.get("success_contribution", 0.0)),
                failure_penalty=float(rec.get("failure_penalty", 0.0)),
                compression_benefit=float(rec.get("compression_benefit", 0.0)),
                generalization_hits=int(rec.get("generalization_hits", 0)),
                goal_signatures=list(rec.get("goal_signatures") or []),
                generation=int(rec.get("generation", 0)),
                parent_ids=list(rec.get("parent_ids") or []),
                active=bool(rec.get("active", True)),
                created_ts=float(rec.get("created_ts", time.time())),
                last_ts=float(rec.get("last_ts", time.time())),
            )
        self._generation = int(data.get("_generation", 0))
        self._task_count = int(data.get("_task_count", 0))

    def _save_state(self) -> None:
        path = Path(self.config.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"_generation": self._generation, "_task_count": self._task_count}
        for key, rec in self._records.items():
            payload[key] = {
                "token": rec.token,
                "members": rec.members,
                "count": rec.count,
                "score": rec.score,
                "strength": rec.strength,
                "fitness": rec.fitness,
                "usage_count": rec.usage_count,
                "success_contribution": rec.success_contribution,
                "failure_penalty": rec.failure_penalty,
                "compression_benefit": rec.compression_benefit,
                "generalization_hits": rec.generalization_hits,
                "goal_signatures": rec.goal_signatures,
                "generation": rec.generation,
                "parent_ids": rec.parent_ids,
                "active": rec.active,
                "created_ts": rec.created_ts,
                "last_ts": rec.last_ts,
            }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def refine(self, resource_score: float = 0.0) -> Dict[str, float]:
        now = time.time()
        if not self.config.enabled:
            return {"created": 0, "updated": 0, "score": 0.0}
        if now - self._last_refine_ts < float(self.config.cooldown_s):
            return {"created": 0, "updated": 0, "score": 0.0}
        if resource_score < float(self.config.min_resource_score):
            return {"created": 0, "updated": 0, "score": 0.0}
        if not self.gnn or not self.gnn.available():
            return {"created": 0, "updated": 0, "score": 0.0}

        comps = self._find_components()
        if not comps:
            return {"created": 0, "updated": 0, "score": 0.0}
        comps.sort(key=lambda x: x[1], reverse=True)
        created = 0
        updated = 0
        for nodes, score, avg_w in comps:
            if created >= int(self.config.max_new_tokens):
                break
            key = self._cluster_key(nodes)
            rec = self._records.get(key)
            if rec is not None:
                rec.count += 1
                rec.last_ts = now
                rec.score = max(rec.score, score)
                self._strengthen_abstraction(rec.token, nodes, avg_w, rec.count)
                updated += 1
                continue
            token = self._make_abstract_token(nodes)
            if not token:
                continue
            self._create_abstraction(token, nodes, avg_w)
            self._records[key] = AbstractionRecord(
                token=token,
                members=nodes,
                count=1,
                score=score,
                created_ts=now,
                last_ts=now,
            )
            created += 1

        if created or updated:
            self.gnn._adj_dirty = True
            try:
                self.gnn._save_edges()
            except Exception:
                pass
            try:
                self.gnn._save_vocab()
            except Exception:
                pass
            self._save_state()
            self._last_refine_ts = now
        return {"created": created, "updated": updated, "score": self._structural_score()}

    def get_top_abstractions(self, k: int = 5) -> List[Dict[str, float | int | List[str]]]:
        if not self._records:
            return []
        def _score(rec: AbstractionRecord) -> float:
            base = float(rec.fitness) if rec.fitness else float(rec.score)
            return base * (1.0 + 0.1 * min(5, rec.count))
        items = sorted([r for r in self._records.values() if r.active], key=_score, reverse=True)
        out = []
        for rec in items[: max(1, int(k))]:
            out.append(
                {
                    "token": rec.token,
                    "members": list(rec.members),
                    "score": float(rec.fitness or rec.score),
                    "count": int(rec.count),
                }
            )
        return out

    def stats(self) -> Dict[str, Any]:
        records = list(self._records.values())
        total = len(records)
        active = [r for r in records if r.active]
        active_count = len(active)
        def _score(rec: AbstractionRecord) -> float:
            return float(rec.fitness) if rec.fitness else float(rec.score)
        avg_fitness = 0.0
        if active:
            avg_fitness = sum(_score(r) for r in active) / max(1.0, float(len(active)))
        gen_map: Dict[int, List[float]] = {}
        for rec in active:
            gen_map.setdefault(int(rec.generation), []).append(_score(rec))
        avg_by_gen = {str(k): round(sum(v) / max(1.0, float(len(v))), 6) for k, v in gen_map.items()}
        return {
            "total": total,
            "active": active_count,
            "survival_rate": round(active_count / max(1, total), 6) if total else None,
            "avg_fitness": round(avg_fitness, 6),
            "avg_fitness_by_generation": avg_by_gen,
            "structural_score": self._structural_score(),
            "generation": int(self._generation),
            "task_count": int(self._task_count),
        }

    def record_outcome(self, goal: str, steps: List[str], success: bool) -> None:
        if not self._records:
            return
        self._task_count += 1
        now = time.time()
        text = " ".join([s for s in steps if s])
        goal_sig = self._goal_signature(goal)
        compression_gain = self._estimate_compression_gain(text)
        related = self._related_abstractions(text, goal)
        for rec in related:
            rec.usage_count += 1
            if success:
                rec.success_contribution += 1.0
            else:
                rec.failure_penalty += 1.0
            rec.compression_benefit += compression_gain
            if goal_sig and goal_sig not in rec.goal_signatures:
                rec.goal_signatures.append(goal_sig)
                if len(rec.goal_signatures) > int(self.config.generalization_cap):
                    rec.goal_signatures = rec.goal_signatures[-int(self.config.generalization_cap) :]
                rec.generalization_hits = len(rec.goal_signatures)
            rec.last_ts = now
            rec.fitness = self._compute_fitness(rec, now)
        if self._task_count % max(1, int(self.config.generation_interval)) == 0:
            self._generation_update(now)
        self._save_state()

    def _goal_signature(self, goal: str) -> str:
        if not goal:
            return ""
        cleaned = re.sub(r"\s+", " ", goal.strip().lower())
        return cleaned[:80]

    def _related_abstractions(self, steps_text: str, goal: str) -> List[AbstractionRecord]:
        out = []
        for rec in self._records.values():
            if not rec.active:
                continue
            if rec.token and rec.token in steps_text:
                out.append(rec)
                continue
            hit = False
            for m in rec.members[:8]:
                if not m:
                    continue
                if m in steps_text or m in goal:
                    hit = True
                    break
            if hit:
                out.append(rec)
        return out

    def _estimate_compression_gain(self, steps_text: str) -> float:
        if not steps_text:
            return 0.0
        gain = 0.0
        for rec in self._records.values():
            if not rec.active:
                continue
            for m in rec.members[:6]:
                if m and m in steps_text:
                    gain += 1.0
                    break
        return gain

    def _compute_fitness(self, rec: AbstractionRecord, now: float) -> float:
        w = self.config.fitness_weights or {}
        alpha = float(w.get("alpha", 1.0))
        beta = float(w.get("beta", 0.8))
        gamma = float(w.get("gamma", 0.6))
        delta = float(w.get("delta", 0.4))
        epsilon = float(w.get("epsilon", 0.5))
        usage = max(1, rec.usage_count)
        success_rate = rec.success_contribution / usage
        failure_rate = rec.failure_penalty / usage
        compression = rec.compression_benefit / usage
        generalization = rec.generalization_hits / max(1, int(self.config.generalization_cap))
        age = max(0.0, now - rec.created_ts)
        age_pen = age / max(1.0, float(self.config.age_half_life_s))
        fitness = (
            alpha * success_rate
            - beta * failure_rate
            + gamma * compression
            - delta * age_pen
            + epsilon * generalization
        )
        return float(round(fitness, 4))

    def _generation_update(self, now: float) -> None:
        actives = [r for r in self._records.values() if r.active]
        if not actives:
            return
        for rec in actives:
            if rec.fitness == 0.0:
                rec.fitness = self._compute_fitness(rec, now)
        actives.sort(key=lambda r: r.fitness, reverse=True)
        total = len(actives)
        retain = int(max(1, round(total * float(self.config.retain_ratio))))
        kill = int(max(0, round(total * float(self.config.kill_ratio))))
        max_pop = int(self.config.max_population)
        retained = actives[:retain]
        to_kill = actives[-kill:] if kill > 0 else []
        # enforce population cap
        if max_pop > 0 and len(retained) > max_pop:
            retained = retained[:max_pop]
            to_kill = [r for r in actives if r not in retained]
        for rec in to_kill:
            rec.active = False
            rec.generation = self._generation + 1
            self._decay_abstraction(rec.token, rec.members, float(self.config.decay_on_kill))
        # mutate middle group via merge
        mutate_n = int(max(0, round(total * float(self.config.mutate_ratio))))
        middle = [r for r in actives if r not in retained and r not in to_kill]
        middle = middle[:mutate_n]
        self._merge_mutations(middle, now)
        self._generation += 1

    def _decay_abstraction(self, token: str, members: List[str], factor: float) -> None:
        if factor <= 0 or factor >= 1.0:
            factor = 0.3
        for m in members:
            key = self.gnn._co_key(m, token)
            if key in self.gnn.co_counts:
                self.gnn.co_counts[key] = float(self.gnn.co_counts.get(key, 0.0)) * factor

    def _merge_mutations(self, candidates: List[AbstractionRecord], now: float) -> None:
        if len(candidates) < 2:
            return
        used = set()
        for i, a in enumerate(candidates):
            if a.token in used:
                continue
            for b in candidates[i + 1 :]:
                if b.token in used:
                    continue
                overlap = self._member_overlap(a.members, b.members)
                if overlap < float(self.config.merge_overlap):
                    continue
                members = list(dict.fromkeys(a.members + b.members))
                token = self._make_abstract_token(members)
                if not token:
                    continue
                self._create_abstraction(token, members, avg_w=max(a.score, b.score, 0.5))
                rec = AbstractionRecord(
                    token=token,
                    members=members,
                    count=1,
                    score=max(a.score, b.score),
                    strength=1.0,
                    fitness=0.0,
                    usage_count=0,
                    success_contribution=0.0,
                    failure_penalty=0.0,
                    compression_benefit=0.0,
                    generalization_hits=0,
                    goal_signatures=[],
                    generation=self._generation + 1,
                    parent_ids=[a.token, b.token],
                    active=True,
                    created_ts=now,
                    last_ts=now,
                )
                self._records[self._cluster_key(members)] = rec
                used.add(a.token)
                used.add(b.token)
                break

    @staticmethod
    def _member_overlap(a: List[str], b: List[str]) -> float:
        if not a or not b:
            return 0.0
        sa = set(a)
        sb = set(b)
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / max(1.0, float(union))

    def _find_components(self) -> List[Tuple[List[str], float, float]]:
        min_w = float(self.config.min_co_count)
        edges: Dict[str, Dict[str, float]] = {}
        for key, w in (self.gnn.co_counts or {}).items():
            if float(w) < min_w:
                continue
            a, b = key.split("\t", 1)
            if a.startswith(self.config.abstract_prefix) and b.startswith(self.config.abstract_prefix):
                continue
            edges.setdefault(a, {})[b] = float(w)
            edges.setdefault(b, {})[a] = float(w)
        visited = set()
        comps = []
        for node in edges.keys():
            if node in visited:
                continue
            queue = [node]
            comp = []
            visited.add(node)
            while queue:
                cur = queue.pop()
                comp.append(cur)
                for nxt in edges.get(cur, {}):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            if len(comp) < int(self.config.min_cluster_size):
                continue
            comp = self._trim_component(comp, edges)
            if len(comp) < int(self.config.min_cluster_size):
                continue
            density, avg_w = self._component_stats(comp, edges)
            if density < float(self.config.min_density):
                continue
            if avg_w < float(self.config.min_avg_weight):
                continue
            score = density * avg_w * (len(comp) / max(1, int(self.config.max_cluster_size)))
            comps.append((comp, score, avg_w))
        return comps

    def _trim_component(self, comp: List[str], edges: Dict[str, Dict[str, float]]) -> List[str]:
        max_size = int(self.config.max_cluster_size)
        if len(comp) <= max_size:
            return comp
        degrees = []
        for n in comp:
            degrees.append((n, len(edges.get(n, {}))))
        degrees.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in degrees[:max_size]]

    def _component_stats(self, comp: List[str], edges: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
        total_w = 0.0
        edge_count = 0
        comp_set = set(comp)
        for i, a in enumerate(comp):
            for b in comp[i + 1 :]:
                w = edges.get(a, {}).get(b)
                if w is None:
                    continue
                total_w += float(w)
                edge_count += 1
        n = len(comp)
        max_edges = n * (n - 1) / 2.0 if n > 1 else 1.0
        density = edge_count / max_edges
        avg_w = total_w / max(1, edge_count)
        return density, avg_w

    def _cluster_key(self, nodes: List[str]) -> str:
        return "|".join(sorted(set(nodes)))

    def _make_abstract_token(self, nodes: List[str]) -> str:
        base = ""
        if self.config.mode in {"llm", "hybrid"} and self.name_fn is not None:
            try:
                base = self.name_fn(nodes) or ""
            except Exception:
                base = ""
        if not base:
            base = "_".join(nodes[:3])
        base = base.strip().lower()
        base = re.sub(r"[^\w\u4e00-\u9fff]+", "_", base)
        base = base.strip("_")
        if not base:
            return ""
        token = f"{self.config.abstract_prefix}{base}"
        # avoid collision
        if token in self.gnn.token_to_id:
            suffix = 1
            while f"{token}_{suffix}" in self.gnn.token_to_id:
                suffix += 1
            token = f"{token}_{suffix}"
        return token

    def _create_abstraction(self, token: str, nodes: List[str], avg_w: float) -> None:
        self.gnn._ensure_tokens([token])
        for a in nodes:
            key = self.gnn._co_key(a, token)
            self.gnn.co_counts[key] = float(self.gnn.co_counts.get(key, 0.0)) + max(avg_w, 0.5)
        self._decay_edges(nodes)

    def _strengthen_abstraction(self, token: str, nodes: List[str], avg_w: float, count: int) -> None:
        boost = min(3.0, 0.5 + 0.1 * float(count))
        for a in nodes:
            key = self.gnn._co_key(a, token)
            self.gnn.co_counts[key] = float(self.gnn.co_counts.get(key, 0.0)) + max(avg_w, 0.5) * boost
        self._decay_edges(nodes)

    def _decay_edges(self, nodes: List[str]) -> None:
        decay = float(self.config.decay)
        if decay <= 0 or decay >= 1.0:
            decay = 0.9
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                key = self.gnn._co_key(a, b)
                if key in self.gnn.co_counts:
                    self.gnn.co_counts[key] = float(self.gnn.co_counts.get(key, 0.0)) * decay

    def _structural_score(self) -> float:
        vocab_size = len(self.gnn.vocab)
        if vocab_size <= 0:
            return 0.0
        abs_tokens = [t for t in self.gnn.vocab if t.startswith(self.config.abstract_prefix)]
        abs_ratio = len(abs_tokens) / vocab_size
        total_edges = len(self.gnn.co_counts or {})
        abs_edges = 0
        for key in (self.gnn.co_counts or {}):
            a, b = key.split("\t", 1)
            if a.startswith(self.config.abstract_prefix) or b.startswith(self.config.abstract_prefix):
                abs_edges += 1
        edge_ratio = abs_edges / max(1, total_edges)
        return round(0.6 * edge_ratio + 0.4 * abs_ratio, 4)
