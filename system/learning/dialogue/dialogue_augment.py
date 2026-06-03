from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class AugmentConfig:
    factor: int = 2
    strength: str = "mixed"  # light | strong | mixed
    seed: int = 7


_REPLACEMENTS: List[Tuple[str, str]] = [
    ("怎么", "如何"),
    ("如何", "怎么"),
    ("能不能", "可以吗"),
    ("能否", "可以"),
    ("可以", "能否"),
    ("请", "麻烦"),
    ("帮我", "帮我一下"),
    ("读取", "读"),
    ("文件", "档案"),
    ("谢谢", "感谢"),
    ("你好", "您好"),
    ("早上好", "早安"),
    ("晚上好", "晚安"),
    ("名字", "称呼"),
    ("什么", "啥"),
]

_GREET_KW = ["你好", "您好", "早上好", "早安", "晚上好", "晚安", "嗨", "哈喽", "hello", "hi"]
_THANK_KW = ["谢谢", "感谢", "多谢", "辛苦"]

_GREET_VARIANTS = [
    "嗨，见到你很高兴",
    "您好",
    "早安",
    "晚上好",
    "哈喽",
]
_THANK_VARIANTS = [
    "非常感谢",
    "多谢帮忙",
    "谢啦",
    "辛苦了",
]

_SYN_MAP = {
    "读取文件": "查看文件",
    "读文件": "打开文件",
    "运行python脚本": "执行python程序",
    "运行Python脚本": "执行Python程序",
    "运行脚本": "执行脚本",
    "修复bug": "排查问题",
    "bug": "问题",
    "优化": "提升性能",
    "代码": "程序",
    "举个例子": "给个示例",
    "再详细": "更详细",
    "成本": "预算",
    "风险": "风险点",
    "时间复杂度": "复杂度",
    "叫什么名字": "怎么称呼你",
    "你能干什么": "你有什么功能",
}

_QUESTION_TEMPLATES = [
    "请问{obj}？",
    "能否协助{obj}？",
    "可以帮忙{obj}吗？",
    "想咨询：{obj}怎么处理？",
    "请教一下，{obj}应该怎么做？",
]

_STOPWORDS = [
    "可以",
    "请问",
    "请教",
    "能否",
    "能不能",
    "可不可以",
    "帮我",
    "帮忙",
    "一下",
    "协助",
]


def _normalize(text: str) -> str:
    s = text
    for k, v in _SYN_MAP.items():
        s = s.replace(k, v)
    return s


def _extract_obj(text: str) -> str:
    s = _normalize(text)
    for w in _STOPWORDS:
        s = s.replace(w, "")
    s = s.strip()
    s = re.sub(r"[？?。！!]+", "", s)
    return s


def _paraphrase_light(text: str, rng: random.Random) -> str:
    original = text
    s = text
    replacements = _REPLACEMENTS[:]
    rng.shuffle(replacements)
    applied = 0
    for pat, repl in replacements:
        if pat in s:
            s2 = s.replace(pat, repl)
            if s2 != s:
                s = s2
                applied += 1
            if applied >= 2:
                break
    if s == original:
        prefixes = ["请问", "麻烦问下", "能否请教", "想咨询"]
        s = rng.choice(prefixes) + s
    if not s.endswith("？") and not s.endswith("?"):
        if s.endswith("。"):
            s = s[:-1] + "？"
        else:
            s = s + "？"
    return s


def _paraphrase_strong(text: str, rng: random.Random) -> str:
    t = text.strip()
    if any(k in t for k in _GREET_KW):
        return rng.choice(_GREET_VARIANTS) + "！"
    if any(k in t for k in _THANK_KW):
        return rng.choice(_THANK_VARIANTS) + "！"
    obj = _extract_obj(t)
    if not obj or len(obj) < 2:
        return _paraphrase_light(text, rng)
    template = rng.choice(_QUESTION_TEMPLATES)
    return template.format(obj=obj)


def _paraphrase(text: str, rng: random.Random, strength: str) -> str:
    if strength == "light":
        return _paraphrase_light(text, rng)
    if strength == "strong":
        return _paraphrase_strong(text, rng)
    # mixed
    if rng.random() < 0.5:
        return _paraphrase_light(text, rng)
    return _paraphrase_strong(text, rng)


def augment_dialogue_pairs(
    data: List[Dict[str, str]],
    factor: int = 2,
    strength: str = "mixed",
    seed: int = 7,
) -> List[Dict[str, str]]:
    if factor <= 0:
        return []
    rng = random.Random(seed)
    augmented: List[Dict[str, str]] = []
    seen = set()
    for item in data:
        user = str(item.get("user", "") or "").strip()
        assistant = str(item.get("assistant", "") or "").strip()
        if not user or not assistant:
            continue
        for _ in range(factor):
            new_user = _paraphrase(user, rng, strength)
            if not new_user or new_user == user:
                continue
            key = (new_user, assistant)
            if key in seen:
                continue
            seen.add(key)
            augmented.append({"user": new_user, "assistant": assistant})
    return augmented
