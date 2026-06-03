from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    """Tokenize text with jieba for CJK support, falling back to simple split."""
    try:
        import jieba

        return [t.strip() for t in jieba.cut(str(text or "")) if t.strip()]
    except Exception:
        pass
    # Fallback: split on whitespace and CJK punctuation
    raw = str(text or "")
    raw = re.sub(r"[。，；：！？、\n\r\t]", " ", raw)
    return [t.strip() for t in raw.split() if t.strip()]


def _has_cjk(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", str(text or "")))


@dataclass
class CognitiveState:
    focus_terms: List[str]
    working_memory: List[str]
    plan_hint: str


class AttentionManager:
    def select(self, text: str, top_k: int = 5) -> List[str]:
        tokens = _tokenize(text)
        # Filter stopwords for both EN and CJK
        stops = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "of", "in", "on", "at", "to", "for", "with", "by", "from",
            "and", "or", "but", "if", "then", "that", "this", "it", "its",
            "的", "了", "在", "是", "我", "你", "他", "她", "它", "们",
            "就", "都", "而", "及", "与", "着", "或", "一个", "没有",
        }
        tokens = [t for t in tokens if t.lower() not in stops and len(t) > 1]
        # Prefer longer tokens (more specific) but balance with frequency
        tokens.sort(key=lambda t: len(t), reverse=True)
        return tokens[:top_k]


class WorkingMemory:
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._items: List[str] = []

    def update(self, item: str) -> None:
        if not item or not str(item).strip():
            return
        text = str(item).strip()
        # Deduplicate — move existing to front instead of adding duplicate
        if text in self._items:
            self._items.remove(text)
        self._items.append(text)
        if len(self._items) > self.capacity:
            self._items.pop(0)

    def items(self) -> List[str]:
        return list(self._items)


class StrategyPlanner:
    def plan(self, text: str, focus_terms: List[str]) -> str:
        has_cjk = _has_cjk(text)
        if has_cjk:
            if "目标" in text:
                return "分解目标为步骤并验证约束"
            if "为什么" in text or "如何" in text or "怎么" in text:
                return "先解释原理，再给出步骤"
            if "比较" in text or "对比" in text or "区别" in text:
                return "逐维度对比，给出选择建议"
            if "错误" in text or "报错" in text or "失败" in text:
                return "先诊断根因，再给出修复步骤"
            if focus_terms:
                return f"围绕关键词 {', '.join(focus_terms[:3])} 提供推理"
            return "给出最小可行解释"
        else:
            if "why" in text.lower() or "how" in text.lower():
                return "Explain the principle first, then provide steps"
            if "compare" in text.lower() or "vs" in text.lower() or "versus" in text.lower():
                return "Compare dimension by dimension, then recommend"
            if "error" in text.lower() or "bug" in text.lower() or "fail" in text.lower():
                return "Diagnose root cause first, then provide fix steps"
            if "goal" in text.lower() or "target" in text.lower():
                return "Decompose goal into steps and verify constraints"
            if focus_terms:
                return f"Reason around keywords: {', '.join(focus_terms[:3])}"
            return "Provide a minimal viable explanation"


class CognitiveArchitecture:
    def __init__(self, wm_capacity: int = 8):
        self.attention = AttentionManager()
        self.working_memory = WorkingMemory(capacity=wm_capacity)
        self.strategy = StrategyPlanner()

    def run(self, text: str) -> CognitiveState:
        focus = self.attention.select(text)
        self.working_memory.update(text)
        plan_hint = self.strategy.plan(text, focus)
        return CognitiveState(
            focus_terms=focus,
            working_memory=self.working_memory.items(),
            plan_hint=plan_hint,
        )
