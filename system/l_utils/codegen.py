from __future__ import annotations

import ast
import json
import logging
import os
import re
import secrets
import threading
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# 允许执行的命令白名单（可配置）
SAFE_COMMANDS = {
    "git", "docker", "curl", "wget", "pip", "pytest",
    "python", "python3", "echo", "cat", "ls", "mkdir",
    "cp", "mv", "rm", "chmod", "chown",
}

class CodeGenerator:
    """安全代码生成器，动态生成模块并原子化注册。"""

    def __init__(self, modules_dir: str = "modules", registry_path: str = "modules/registry.json"):
        self.modules_dir = Path(modules_dir).resolve()
        self.modules_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = Path(registry_path).resolve()
        self._lock = threading.Lock()
        if not self.registry_path.exists():
            self._atomic_write_registry([])

    # ------------------------------------------------------------------
    # 注册表原子读写（与改进后的 loader 一致）
    # ------------------------------------------------------------------
    def _read_registry(self) -> List[Dict[str, str]]:
        try:
            with self._lock:
                text = self.registry_path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("Registry top-level must be a list")
            for item in data:
                if not isinstance(item, dict) or "path" not in item:
                    raise ValueError("Invalid registry entry format")
            return data
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.error("Failed to read registry %s: %s", self.registry_path, e)
            return []

    def _atomic_write_registry(self, data: List[Dict[str, str]]) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.registry_path.parent, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.registry_path)
        except OSError as e:
            logger.error("Atomic write failed for registry: %s", e)
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 公共 API（保持接口不变）
    # ------------------------------------------------------------------
    def generate_module(self, request: str) -> Tuple[str, str]:
        """根据请求生成一个模块文件，返回 (文件绝对路径, 模块名)。"""
        # 生成唯一名称
        unique = secrets.token_hex(4)
        name = f"auto_module_{int(time.time())}_{unique}"
        path = self.modules_dir / f"{name}.py"
        try:
            intent = self._analyze_intent(request)
            if intent == "DATA_WRITE":
                content = self._build_data_write_module(request)
            elif intent == "BATCH_PROCESS":
                content = self._build_batch_process_module(request)
            elif intent == "SUBPROCESS_WRAPPER":
                content = self._build_subprocess_wrapper(request)
            elif intent == "GLUE":
                content = self._build_glue_module(request)
            else:
                content = self._build_basic_module(request)

            # 安全审查
            ok, reason = self._is_safe_code(content)
            if not ok:
                logger.warning("Generated code blocked for safety: %s", reason)
                content = self._build_blocked_module(request, reason)

            # 写入模块文件
            path.write_text(content, encoding="utf-8")
            # 注册到注册表（存储相对路径）
            self._register(str(path))
            logger.info("Module generated: %s", path)
            return str(path), name
        except Exception as e:
            logger.exception("Failed to generate module for request: %s", request)
            # 尝试写入一个安全的占位模块，避免调用方崩溃
            try:
                fallback = self._build_blocked_module(request, f"generation error: {e}")
                path.write_text(fallback, encoding="utf-8")
                self._register(str(path))
            except Exception:
                logger.critical("Completely failed to create fallback module")
            return str(path), name

    # ------------------------------------------------------------------
    # 注册逻辑（内部调用）
    # ------------------------------------------------------------------
    def _register(self, absolute_path: str) -> None:
        """将模块加入注册表，自动转换为相对于 modules_dir 的路径。"""
        abs_path = Path(absolute_path).resolve()
        try:
            rel_path = str(abs_path.relative_to(self.modules_dir))
        except ValueError:
            logger.error("Module path %s is outside modules_dir %s, skipping register", abs_path, self.modules_dir)
            return

        with self._lock:
            data = self._read_registry()
            if any(item["path"] == rel_path for item in data):
                logger.debug("Module %s already registered", rel_path)
                return
            data.append({"path": rel_path})
            self._atomic_write_registry(data)

    # ------------------------------------------------------------------
    # 意图分析（保持不变，仅调整了些许健壮性）
    # ------------------------------------------------------------------
    def _analyze_intent(self, request: str) -> str:
        low = (request or "").lower()
        if "intent=data_write" in low:
            return "DATA_WRITE"
        if "intent=batch_process" in low:
            return "BATCH_PROCESS"
        if "intent=subprocess_wrapper" in low:
            return "SUBPROCESS_WRAPPER"
        if "intent=glue" in low:
            return "GLUE"
        if any(ext in low for ext in (".json", ".yaml", ".yml")) and any(
            k in low
            for k in (
                "write", "generate", "create", "save", "report", "audit",
                "\u914d\u7f6e", "\u751f\u6210", "\u4fdd\u5b58", "\u62a5\u544a",
            )
        ):
            return "DATA_WRITE"
        if any(
            k in low
            for k in (
                "batch", "rename", "move", "compress", "archive", "zip", "gzip", "filter",
                "\u6279\u91cf", "\u91cd\u547d\u540d", "\u79fb\u52a8", "\u538b\u7f29", "\u8fc7\u6ee4",
            )
        ):
            return "BATCH_PROCESS"
        if any(
            k in low
            for k in (
                "git", "docker", "curl", "wget", "pip", "pytest", "subprocess", "command", "shell",
                "\u547d\u4ee4", "\u6267\u884c",
            )
        ):
            return "SUBPROCESS_WRAPPER"
        if self._looks_like_glue_request(request):
            return "GLUE"
        return "BASIC"

    def _looks_like_glue_request(self, request: str) -> bool:
        low = (request or "").lower()
        if "glue" in low:
            return True
        keywords = ("fix path", "path conflict", "mkdir", "chmod", "move", "rename", "conflict")
        return any(k in low for k in keywords)

    # ------------------------------------------------------------------
    # 路径/标签提取工具
    # ------------------------------------------------------------------
    def _extract_paths(self, text: str) -> List[str]:
        if not text:
            return []
        hits = re.findall(r"[A-Za-z0-9_\-./\\\\]+\.[A-Za-z0-9]+", text)
        out, seen = [], set()
        for h in hits:
            p = h.strip().strip(" \t\r\n\"'<>(),;")
            p = p.rstrip(".")
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _guess_output_path(self, text: str, paths: List[str]) -> str:
        if not text or not paths:
            return ""
        patterns = [
            r"(?:create|write|generate|save|output|produce)\s+([^\s,;]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1).strip().strip("\"'<>(),;")
                if candidate:
                    return candidate
        for p in paths:
            low = p.lower()
            if low.endswith((".json", ".txt", ".md", ".yaml", ".yml", ".csv")):
                return p
        return paths[0] if paths else ""

    def _extract_tag(self, text: str, key: str) -> str:
        if not text:
            return ""
        key = key.rstrip("=") + "="
        idx = text.find(key)
        if idx == -1:
            return ""
        tail = text[idx + len(key):]
        if ";" in tail:
            tail = tail.split(";", 1)[0]
        return tail.strip().strip("\"'<>(),")

    def _parse_size_threshold(self, text: str) -> int:
        if not text:
            return 0
        m = re.search(r"(\d+(?:\.\d+)?)\s*(kb|mb|gb)", text, flags=re.IGNORECASE)
        if not m:
            return 0
        val = float(m.group(1))
        unit = m.group(2).lower()
        mult = {"kb": 1024, "mb": 1024**2, "gb": 1024**3}.get(unit, 1)
        return int(val * mult)

    # ------------------------------------------------------------------
    # 模块代码生成
    # ------------------------------------------------------------------
    def _build_basic_module(self, request: str) -> str:
        return (
            "def run(context: dict) -> dict:\n"
            "    \"\"\"Auto-generated module.\"\"\"\n"
            f"    return {{\"status\": \"ok\", \"request\": {request!r}, \"context\": context}}\n"
        )

    def _build_data_write_module(self, request: str) -> str:
        paths = self._extract_paths(request)
        output = self._extract_tag(request, "output") or self._guess_output_path(request, paths)
        goal = self._extract_tag(request, "goal") or request
        fmt = "json"
        if output.lower().endswith((".yaml", ".yml")):
            fmt = "yaml"
        return (
            "import os\n"
            "import json\n"
            "from datetime import datetime\n"
            "try:\n"
            "    import yaml\n"
            "except Exception:\n"
            "    yaml = None\n\n"
            "def run(context: dict) -> dict:\n"
            "    \"\"\"Write structured data to target file.\"\"\"\n"
            f"    goal = {goal!r}\n"
            f"    output = {output!r}\n"
            "    if not output:\n"
            "        return {\"status\": \"blocked\", \"reason\": \"missing output path\"}\n"
            "    parent = os.path.dirname(output)\n"
            "    if parent:\n"
            "        os.makedirs(parent, exist_ok=True)\n"
            "    payload = {\n"
            "        \"goal\": goal,\n"
            "        \"timestamp\": datetime.utcnow().isoformat() + \"Z\",\n"
            "        \"context\": context or {},\n"
            "    }\n"
            f"    fmt = {fmt!r}\n"
            "    if fmt == \"yaml\" and yaml is not None:\n"
            "        with open(output, \"w\", encoding=\"utf-8\") as f:\n"
            "            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)\n"
            "    else:\n"
            "        with open(output, \"w\", encoding=\"utf-8\") as f:\n"
            "            json.dump(payload, f, ensure_ascii=False, indent=2)\n"
            "    return {\"status\": \"ok\", \"output\": output, \"format\": fmt}\n"
        )

    def _build_batch_process_module(self, request: str) -> str:
        paths = self._extract_paths(request)
        root = self._extract_tag(request, "root") or (paths[0] if paths else ".")
        dest = self._extract_tag(request, "dest") or ".batch_output"
        threshold = self._parse_size_threshold(request) or 10 * 1024 * 1024
        return (
            "import os\n"
            "import shutil\n"
            "import gzip\n"
            "from pathlib import Path\n\n"
            "def _compress_file(src: Path, dst: Path) -> None:\n"
            "    dst.parent.mkdir(parents=True, exist_ok=True)\n"
            "    with open(src, \"rb\") as f_in:\n"
            "        with gzip.open(dst, \"wb\") as f_out:\n"
            "            shutil.copyfileobj(f_in, f_out)\n\n"
            "def run(context: dict) -> dict:\n"
            "    \"\"\"Batch process files (compress/move/rename).\"\"\"\n"
            f"    root = Path({root!r})\n"
            f"    dest = Path({dest!r})\n"
            f"    threshold = int({threshold})\n"
            "    moved = []\n"
            "    if not root.exists():\n"
            "        return {\"status\": \"blocked\", \"reason\": f\"root not found: {root}\"}\n"
            "    for p in root.rglob(\"*\"):\n"
            "        if not p.is_file():\n"
            "            continue\n"
            "        try:\n"
            "            size = p.stat().st_size\n"
            "        except Exception:\n"
            "            continue\n"
            "        if size < threshold:\n"
            "            continue\n"
            "        rel = p.relative_to(root)\n"
            "        gz_name = rel.as_posix() + \".gz\"\n"
            "        dst = dest / gz_name\n"
            "        try:\n"
            "            _compress_file(p, dst)\n"
            "            moved.append(str(dst))\n"
            "        except Exception:\n"
            "            continue\n"
            "    return {\"status\": \"ok\", \"moved\": moved, \"count\": len(moved)}\n"
        )

    def _build_subprocess_wrapper(self, request: str) -> str:
        """生成安全的子进程包装器，使用命令白名单 + 参数列表，拒绝 shell 元字符。"""
        cmd = self._extract_tag(request, "cmd") or request
        # 提取第一个词作为命令名
        first_word = cmd.strip().split()[0] if cmd.strip() else ""
        if first_word not in SAFE_COMMANDS:
            return (
                "def run(context: dict) -> dict:\n"
                "    \"\"\"Blocked: unsafe command.\"\"\"\n"
                f"    return {{\"status\": \"blocked\", \"reason\": \"command not in whitelist: {first_word}\", \"request\": {request!r}}}\n"
            )
        # 用 shlex 安全分割
        return (
            "import subprocess\n"
            "import shlex\n\n"
            "def run(context: dict) -> dict:\n"
            "    \"\"\"Run external command safely.\"\"\"\n"
            f"    cmd_str = {cmd!r}\n"
            "    # 再次检查白名单\n"
            f"    allowed = {sorted(SAFE_COMMANDS)!r}\n"
            "    args = shlex.split(cmd_str)\n"
            "    if not args or args[0] not in allowed:\n"
            "        return {\"status\": \"blocked\", \"reason\": \"command rejected\"}\n"
            "    try:\n"
            "        res = subprocess.run(args, capture_output=True, text=True, timeout=30)\n"
            "        return {\"status\": \"ok\" if res.returncode == 0 else \"error\", \"output\": res.stdout, \"stderr\": res.stderr, \"code\": res.returncode}\n"
            "    except Exception as e:\n"
            "        return {\"status\": \"error\", \"message\": str(e)}\n"
        )

    def _build_glue_module(self, request: str) -> str:
        return (
            "import os\n"
            "import re\n"
            "import shutil\n"
            "from pathlib import Path\n\n"
            "def run(context: dict) -> dict:\n"
            "    \"\"\"Glue module that fixes path conflicts and permissions.\"\"\"\n"
            f"    request = {request!r}\n"
            "    text = str(request)\n"
            "    paths = re.findall(r\"[A-Za-z0-9_\\-./\\\\\\\\]+\\\\.[A-Za-z0-9]+\", text)\n"
            "    cleaned = []\n"
            "    seen = set()\n"
            "    for p in paths:\n"
            "        p = p.strip().strip(' \\t\\r\\n\\\"\\'<>(),;').rstrip('.')\n"
            "        if p and p not in seen:\n"
            "            seen.add(p)\n"
            "            cleaned.append(p)\n"
            "    paths = cleaned\n"
            "    output = \"\"\n"
            "    m = re.search(r\"(?:create|write|generate|save|output|produce)\\s+([^\\s,;]+)\", text, flags=re.I)\n"
            "    if m:\n"
            "        output = m.group(1).strip().strip('\\\"\\'<>(),;')\n"
            "    if not output:\n"
            "        for p in paths:\n"
            "            low = p.lower()\n"
            "            if low.endswith((\".json\", \".txt\", \".md\", \".yaml\", \".yml\", \".csv\")):\n"
            "                output = p\n"
            "                break\n"
            "    if output:\n"
            "        parent = os.path.dirname(output)\n"
            "        if parent and os.path.isfile(parent):\n"
            "            os.makedirs('.rollback', exist_ok=True)\n"
            "            safe = parent.replace('/', '__').replace('\\\\\\\\', '__')\n"
            "            dst = os.path.join('.rollback', f\"{safe}.block\")\n"
            "            try:\n"
            "                shutil.move(parent, dst)\n"
            "            except Exception:\n"
            "                pass\n"
            "        if parent:\n"
            "            os.makedirs(parent, exist_ok=True)\n"
            "    for p in paths:\n"
            "        try:\n"
            "            if os.path.isfile(p):\n"
            "                os.chmod(p, 0o644)\n"
            "        except Exception:\n"
            "            pass\n"
            "    return {\"status\": \"ok\", \"output\": output, \"paths\": paths}\n"
        )

    def _build_blocked_module(self, request: str, reason: str) -> str:
        return (
            "def run(context: dict) -> dict:\n"
            "    \"\"\"Blocked by safety guard.\"\"\"\n"
            f"    return {{\"status\": \"blocked\", \"reason\": {reason!r}, \"request\": {request!r}}}\n"
        )

    # ------------------------------------------------------------------
    # 安全审查（增强版 AST 检查）
    # ------------------------------------------------------------------
    def _is_safe_code(self, content: str) -> Tuple[bool, str]:
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        forbidden_calls = {
            "eval", "exec", "compile", "open",
            "__import__", "breakpoint",
            "os.system", "os.popen", "os.spawn", "os.fork",
            "subprocess.call", "subprocess.check_call", "subprocess.check_output",
            "subprocess.run",  # 允许但需要严格检查 shell 参数，见下
        }
        dangerous_patterns = (
            "rm -rf", "del /f /s", "format c:", "shutdown",
        )

        def _full_name(node):
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                base = _full_name(node.value)
                if base:
                    return f"{base}.{node.attr}"
                return node.attr
            return ""

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fname = _full_name(node.func)
                if fname in forbidden_calls:
                    if fname == "subprocess.run":
                        # 允许 subprocess.run，但必须没有 shell=True
                        for kw in node.keywords:
                            if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value:
                                return False, "subprocess.run with shell=True is prohibited"
                        continue
                    return False, f"Forbidden call: {fname}"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                low = node.value.lower()
                if any(p in low for p in dangerous_patterns):
                    return False, "Dangerous shell pattern detected"

        return True, "ok"
