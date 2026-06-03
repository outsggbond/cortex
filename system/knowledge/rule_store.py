from __future__ import annotations

import json
from pathlib import Path
from typing import List

from system.brain.neurosymbolic import Rule, NeuroSymbolicEngine


class RuleStore:
    def __init__(self, path: str = "rules/dynamic_rules.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, rules: List[Rule]) -> None:
        data = []
        for r in rules:
            data.append(
                {
                    "name": r.name,
                    "weight": r.weight,
                    "type": r.meta.get("type", "contains"),
                    "keyword": r.meta.get("keyword", ""),
                    "conclusion": r.meta.get("conclusion", ""),
                }
            )
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_into(self, engine: NeuroSymbolicEngine) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        for item in data:
            name = item.get("name", "dyn_loaded")
            weight = float(item.get("weight", 0.6))
            rtype = item.get("type", "contains")
            keyword = item.get("keyword", "")
            conclusion = item.get("conclusion", "")
            if rtype == "contains" and keyword and conclusion:
                engine.add_dynamic_rule(
                    engine.make_contains_rule(
                        name=name,
                        keyword=keyword,
                        conclusion=conclusion,
                        weight=weight,
                        evidence="loaded",
                    )
                )
            elif rtype == "static" and conclusion:
                engine.add_dynamic_rule(engine.make_static_rule(name=name, conclusion=conclusion, weight=weight))
            elif rtype == "and" and keyword and conclusion:
                parts = keyword.split("|")
                engine.add_dynamic_rule(engine.make_and_rule(name=name, keywords=parts, conclusion=conclusion, weight=weight))
