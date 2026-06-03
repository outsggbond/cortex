from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class Rule:
    name: str
    when: Callable[[dict], bool]
    then: Callable[[dict], dict]


class RuleEngine:
    def __init__(self):
        self._rules: List[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)

    def run(self, context: dict) -> Dict[str, dict]:
        results: Dict[str, dict] = {}
        for rule in self._rules:
            try:
                if rule.when(context):
                    results[rule.name] = rule.then(context)
            except Exception:
                continue
        return results
