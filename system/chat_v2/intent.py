from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Intent:
    type: str
    name: str = ""
    detail: str = ""


def parse_intent(text: str) -> Optional[Intent]:
    t = text.strip().lower()
    # add module
    m = re.search(r"(添加|新增|增加).*(模块|功能)\s*([a-zA-Z0-9_\-]+)?", text)
    if m:
        name = m.group(3) or "custom_feature"
        return Intent(type="add_module", name=name, detail=text.strip())
    # evolve request
    if "进化" in text or "自我进化" in text:
        return Intent(type="evolve", detail=text.strip())
    # code change request
    if "修改代码" in text or "改代码" in text:
        return Intent(type="modify_code", detail=text.strip())
    return None
