from __future__ import annotations

import logging
import os
import re
from typing import Optional


logger = logging.getLogger(__name__)


def _looks_like_code_answer(text: str) -> bool:
    cur = str(text or "").strip()
    if not cur:
        return False
    if "```" in cur:
        return True
    lines = [line.rstrip() for line in cur.splitlines() if str(line or "").strip()]
    if len(lines) < 2:
        return False
    markers = (
        "#include",
        "import ",
        "from ",
        "def ",
        "class ",
        "function ",
        "return ",
        "std::",
        "console.",
        "printf(",
        "cout",
        "System.out",
        "public static void main",
    )
    if any(marker in cur for marker in markers):
        return True
    punct = sum(cur.count(token) for token in ("{", "}", "(", ")", ";", "::", "=>"))
    if punct >= 4 and len(lines) >= 3:
        return True
    return False


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk / max(1, len(text))


def _repetition_ratio(text: str) -> float:
    if not text:
        return 0.0
    tokens = list(text)
    if not tokens:
        return 0.0
    unique = len(set(tokens))
    return 1.0 - unique / max(1, len(tokens))


def guard_response(question: str, answer: str) -> Optional[str]:
    if not answer:
        return None
    code_like = _looks_like_code_answer(answer)
    min_len = int(os.environ.get("RESPONSE_MIN_LEN", "2"))
    if len(answer.strip()) < min_len:
        return None
    if "<|assistant|>" in answer or "<|user|>" in answer:
        return None
    if not code_like and _repetition_ratio(answer) > float(os.environ.get("RESPONSE_MAX_REPEAT", "0.65")):
        return None

    # If question is Chinese, require answer to be Chinese-ish.
    q_cjk = _cjk_ratio(question)
    if not code_like and q_cjk >= 0.2 and _cjk_ratio(answer) < float(os.environ.get("RESPONSE_MIN_CJK", "0.1")):
        return None

    # Optional semantic similarity (only when embedding model is set).
    if os.environ.get("EMBEDDING_MODEL", "").strip():
        try:
            from system.core.embeddings import cosine, encode_text

            qv = encode_text(question)
            av = encode_text(answer)
            sim = cosine(qv, av)
            if sim < float(os.environ.get("RESPONSE_MIN_SIM", "0.2")):
                return None
        except Exception:
            logger.debug("response_guard: embedding similarity failed", exc_info=True)
    return answer


def is_low_signal_response(answer: str) -> bool:
    """Detect generic / low-information replies that should not be learned."""
    text = (answer or "").strip()
    if not text:
        return True
    if _looks_like_code_answer(text):
        return False
    if len(text) < int(os.environ.get("LEARN_MIN_LEN", "4")):
        return True
    text_lower = text.lower()
    compact_lower = re.sub(r"\s+", "", text_lower)
    low_signal_markers = (
        "need more information",
        "provide more details",
        "still learning",
        "tell me your goal and constraints",
        "i will provide an executable plan",
        "i can help right now: share your goal",
        "cannot answer this",
        "not sure",
        "please clarify",
        "missing constraints",
        "missing requirements",
        "我需要更多信息",
        "请补充更多信息",
        "请提供更多细节",
        "先告诉我你的目标和约束",
        "需要更明确的目标或约束",
    )
    if any(m in text_lower for m in low_signal_markers):
        return True
    compact_markers = (
        "tellmeyourgoalandconstraints",
        "iwillprovideanexecutableplan",
        "icanhelprightnow:shareyourgoal",
        "需要更明确的目标或约束",
        "告诉我你的目标和约束",
    )
    if any(m in compact_lower for m in compact_markers):
        return True
    if _repetition_ratio(text) > float(os.environ.get("LEARN_MAX_REPEAT", "0.7")):
        return True
    return False


def should_learn_from_chat(user_text: str, assistant_text: str) -> bool:
    """Quality gate before writing online chat samples into artifacts/memory/experience."""
    q = (user_text or "").strip()
    a = (assistant_text or "").strip()
    if not q or not a:
        return False
    if is_low_signal_response(a):
        return False
    if len(a) < int(os.environ.get("LEARN_MIN_LEN", "4")):
        return False
    if _repetition_ratio(a) > float(os.environ.get("LEARN_MAX_REPEAT", "0.7")):
        return False
    # Keep language consistency for Chinese prompts.
    if _cjk_ratio(q) >= 0.2 and _cjk_ratio(a) < float(os.environ.get("LEARN_MIN_CJK", "0.1")):
        return False
    return True
