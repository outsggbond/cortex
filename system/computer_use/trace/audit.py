from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


class TraceAudit:
    def __init__(self, path: str = "artifacts/audit/trace_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: Dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class ErrorAudit:
    def __init__(self, path: str = "artifacts/audit/error_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, where: str, error: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"where": where, "error": error}, ensure_ascii=False) + "\n")

    def record_json(self, where: str, data: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"where": where, "data": data}, ensure_ascii=False) + "\n")


class ExecAudit:
    def __init__(self, path: str = "artifacts/audit/exec_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, step: str, status: str, duration_ms: float) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"step": step, "status": status, "duration_ms": duration_ms},
                    ensure_ascii=False,
                )
                + "\n"
            )


class RuleWeightAudit:
    def __init__(self, path: str = "artifacts/audit/rule_weight.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, rules: List[dict]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"rules": rules, "ts": self._now()}, ensure_ascii=False) + "\n")

    def _now(self) -> float:
        import time

        return time.time()
