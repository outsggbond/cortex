from __future__ import annotations

import re
from typing import Tuple, Dict, List

# =========================================================
# 错误分类配置 (数据与逻辑分离)
# =========================================================
ERROR_RULES: Dict[str, List[str]] = {
    "import_missing": ["modulenotfounderror", "no module named", "importerror"],
    "file_missing": ["filenotfounderror", "no such file", "not found"],
    "permission": ["permission", "access is denied", "权限", "拒绝"],
    "not_a_dir": ["notadirectoryerror", "not a directory", "不是目录", "非目录"],
    "path_conflict": ["file exists", "already exists", "已存在", "存在同名"],
    "syntax": ["syntaxerror", "indentationerror"],
    "type": ["typeerror"],
    "value": ["valueerror"],
    "key": ["keyerror"],
    "index": ["indexerror"],
    "assert": ["assert", "assertionerror"],
    "timeout": ["timeout", "timed out"],
    "network": ["connection", "network"],
}


# =========================================================
# 签名清理工具类 (职责单一化)
# =========================================================
class SignatureSanitizer:
    """处理错误信息脱敏与格式化的专用工具类"""
    
    # 修复了原代码中 \\d+ 的转义问题，并统一管理正则对象
    _PATH_PAT = re.compile(r"([A-Za-z]:)?[\\/][\w\-\.\/]+")
    _NUM_PAT = re.compile(r"\d+")
    _SPACE_PAT = re.compile(r"\s+")

    @classmethod
    def sanitize(cls, text: str) -> str:
        """
        去除字符串中的路径、数字及多余空格，生成通用签名。
        """
        sig = cls._PATH_PAT.sub("<path>", text)
        sig = cls._NUM_PAT.sub("<num>", sig)
        sig = cls._SPACE_PAT.sub(" ", sig).strip()
        return sig


# =========================================================
# 主处理函数 (只负责编排流程)
# =========================================================
def normalize_error(error_type: str, message: str) -> Tuple[str, str]:
    """
    标准化错误信息，返回错误分类和脱敏后的错误签名。
    """
    et = (error_type or "").strip()
    msg = (message or "").strip()
    low = f"{et} {msg}".lower()

    # 1. 确定错误分类
    category = "unknown"
    for cat, keywords in ERROR_RULES.items():
        if any(kw in low for kw in keywords):
            category = cat
            break

    # 2. 生成脱敏签名 (调用专门的工具类)
    sig = SignatureSanitizer.sanitize(msg)
    
    # 3. 兜底处理
    if not sig:
        sig = et or "error"
        
    return category, sig
