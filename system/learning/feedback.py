from __future__ import annotations

from typing import Dict, List


def build_feedback(execution: dict) -> Dict[str, object]:
    completed = execution.get("completed", [])
    failed = execution.get("failed", [])
    total = len(completed) + len(failed)
    rate = (len(completed) / total) if total else 0.0
    suggestions: List[str] = []
    if failed:
        suggestions.append("存在失败步骤，建议缩小范围或补充细节。")
    if rate < 0.5:
        suggestions.append("成功率较低，建议降低复杂度。")
    if not suggestions:
        suggestions.append("执行顺利，可推进下一阶段。")
    return {"success_rate": rate, "suggestions": suggestions}
