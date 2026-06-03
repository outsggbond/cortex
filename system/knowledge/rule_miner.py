# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import Counter
from typing import List

from system.brain.neurosymbolic import Rule, NeuroSymbolicEngine
from system.core.embeddings import encode_text, cosine


class RuleMiner:
    def __init__(self, engine: NeuroSymbolicEngine):
        self.engine = engine

    def mine(self, texts: List[str]) -> List[Rule]:
        rules: List[Rule] = []
        patterns = []
        for t in texts:
            if "if" in t and "then" in t:
                try:
                    cond, cons = t.split("if", 1)[1].split("then", 1)
                    patterns.append((cond.strip(), cons.strip()))
                except Exception:
                    continue
            if "如果" in t and "那么" in t:
                try:
                    cond, cons = t.split("如果", 1)[1].split("那么", 1)
                    patterns.append((cond.strip(), cons.strip()))
                except Exception:
                    continue
        counts = Counter(patterns)
        for (cond, cons), n in counts.items():
            if n < 2:
                continue
            rules.append(
                self.engine.make_contains_rule(
                    name=f"mined_if_{cond[:8]}",
                    keyword=cond,
                    conclusion=cons,
                    weight=min(1.2, 0.5 + 0.1 * n),
                    evidence=f"mined from {n} samples",
                )
            )
        return self._semantic_merge(rules)

    def _semantic_merge(self, rules: List[Rule], sim_threshold: float = 0.8) -> List[Rule]:
        merged: List[Rule] = []
        emb_cache = {}
        for r in rules:
            key = r.meta.get("conclusion", r.name)
            emb = emb_cache.get(key)
            if emb is None:
                emb = encode_text(key)
                emb_cache[key] = emb
            found = False
            for m in merged:
                m_key = m.meta.get("conclusion", m.name)
                m_emb = emb_cache.get(m_key)
                if m_emb is None:
                    m_emb = encode_text(m_key)
                    emb_cache[m_key] = m_emb
                if cosine(emb, m_emb) >= sim_threshold:
                    m.weight = (m.weight + r.weight) / 2
                    found = True
                    break
            if not found:
                merged.append(r)
        return merged


class EvolutionaryRuleOptimizer:
    def __init__(self):
        try:
            from deap import base, creator, tools
        except Exception:
            self.available = False
            return
        self.available = True
        self.base = base
        self.creator = creator
        self.tools = tools

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

    def optimize_weights(self, rules: List[Rule]) -> List[Rule]:
        if not self.available or not rules:
            return rules
        base, creator, tools = self.base, self.creator, self.tools

        toolbox = base.Toolbox()
        toolbox.register("attr_float", lambda: 0.5)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(rules))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def eval_individual(ind):
            score = 0.0
            for w in ind:
                score += 1.0 - abs(w - 0.7)
            return (score,)

        toolbox.register("evaluate", eval_individual)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=10)
        for _ in range(5):
            offspring = tools.selBest(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if len(child1) == len(child2):
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = toolbox.evaluate(ind)
            pop[:] = offspring

        best = tools.selBest(pop, 1)[0]
        for r, w in zip(rules, best):
            r.weight = max(0.0, min(1.5, float(w)))
        return rules
