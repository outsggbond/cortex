from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from system.agent.state import AgentTaskState, TaskStatus
from system.agent.task import RuntimeTask


# 简单的文件锁实现（不引入外部依赖，基于文件系统）
class _FileLock:
    """基于 os.O_EXCL 的跨平台文件锁，锁定期间会创建一个临时文件。"""
    def __init__(self, lock_path: Path, timeout: float = 5.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._acquired = False

    def acquire(self) -> bool:
        start = time.monotonic()
        while True:
            try:
                # 使用 O_EXCL 确保原子性创建文件，如果文件已存在则抛 FileExistsError
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                self._acquired = True
                return True
            except FileExistsError:
                # 文件已存在，检查是否超时
                if time.monotonic() - start >= self.timeout:
                    return False
                time.sleep(0.05)
            except OSError:  # 其他系统错误立即放弃
                return False

    def release(self) -> None:
        if self._acquired:
            try:
                os.unlink(str(self.lock_path))
            except OSError:
                pass
            self._acquired = False

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock for {self.lock_path}")
        return self

    def __exit__(self, *args):
        self.release()
    """跨进程文件锁，使用 os.link 原子创建锁文件。"""
    def __init__(self, lock_path: Path, timeout: float = 5.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._acquired = False

    def acquire(self) -> bool:
        start = time.monotonic()
        while True:
            try:
                os.link(str(self.lock_path), str(self.lock_path) + ".lock")
                self._acquired = True
                return True
            except OSError:
                if time.monotonic() - start >= self.timeout:
                    return False
                time.sleep(0.05)

    def release(self) -> None:
        if self._acquired:
            try:
                (str(self.lock_path) + ".lock")
                os.unlink(str(self.lock_path) + ".lock")
            except OSError:
                pass
            self._acquired = False

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock for {self.lock_path}")
        return self

    def __exit__(self, *args):
        self.release()


class RuntimeStateManager:
    def __init__(
        self,
        *,
        project_root: str = ".",
        state_root: str = "artifacts/runtime/tasks",
    ) -> None:
        self.project_root = Path(str(project_root or ".")).resolve()
        self.state_root = self._resolve_path(state_root)

    # ---------- 公共接口保持不变 ----------
    def start(self, task: RuntimeTask) -> AgentTaskState:
        state = self._load_or_default(task)
        state.status = TaskStatus.RUNNING
        state.started_at = time.time()
        state.finished_at = 0.0
        state.last_error = ""
        state.result = {}
        self._write_state(state)
        return state

    def succeed(self, task: RuntimeTask, result: Any) -> AgentTaskState:
        state = self._load_or_default(task)
        state.status = TaskStatus.SUCCEEDED
        state.finished_at = time.time()
        state.last_error = ""
        state.result = self._sanitize(result) if result is not None else {}  # 保持原有语义，返回 {} 而不是 None
        self._write_state(state)
        return state

    def fail(self, task: RuntimeTask, error: Exception | str) -> AgentTaskState:
        state = self._load_or_default(task)
        state.status = TaskStatus.FAILED
        state.finished_at = time.time()
        # 改进：保存完整的错误信息（类型+消息+截断的traceback）
        state.last_error = self._format_error(error)
        self._write_state(state)
        return state

    def load(self, task_id: str) -> dict[str, Any] | None:
        path = self.path_for(task_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def path_for(self, task_id: str) -> Path:
        # 问题1修复：仅允许安全字符，杜绝路径穿越
        safe_id = self._sanitize_task_id(task_id)
        return self.state_root / f"{safe_id}.json"

    # ---------- 私有辅助方法 ----------
    def _load_or_default(self, task: RuntimeTask) -> AgentTaskState:
        path = self.path_for(task.task_id)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return AgentTaskState(
                task_id=self._get_field(raw, "task_id", task.task_id),
                task_type=self._get_field(raw, "task_type", task.task_type),
                intent=self._get_field(raw, "intent", task.intent),
                goal=self._get_field(raw, "goal", task.goal),
                entrypoint=self._get_field(raw, "entrypoint", task.entrypoint),
                status=self._get_field(raw, "status", TaskStatus.PENDING),
                source=self._get_field(raw, "source", task.source or "cli"),
                started_at=self._get_field(raw, "started_at", 0.0),
                finished_at=self._get_field(raw, "finished_at", 0.0),
                last_error=self._get_field(raw, "last_error", ""),
                metadata=dict(self._get_field(raw, "metadata", task.metadata or {})),
                result=dict(self._get_field(raw, "result", {})),
            )
        return AgentTaskState.from_task(task)

    def _write_state(self, state: AgentTaskState) -> None:
        path = self.path_for(state.task_id)
        self._atomic_write(path, state)

    def _resolve_path(self, path_text: str) -> Path:
        # 问题1 & 9 改进：空字符串直接视为无效，保持原有绝对路径优先逻辑但增加明确处理
        if not path_text or not path_text.strip():
            raise ValueError("state_root must not be empty")
        path = Path(path_text.strip()).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _sanitize(self, value: Any, _depth: int = 0) -> Any:
        """递归清理对象，限制深度，处理常见不可序列化类型。"""
        # 问题4：增加深度限制，防止循环引用导致栈溢出
        if _depth > 10:
            return {"_error": "max recursion depth exceeded"}

        if value is None:
            return {}
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            # 保持平台原生路径，不强制 as_posix()
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._sanitize(v, _depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize(item, _depth + 1) for item in value]
        # 常见类型处理
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="replace")
        # 扩展：处理 datetime, Decimal 等
        if hasattr(value, "isoformat"):  # datetime/date
            return value.isoformat()
        if hasattr(value, "__float__"):  # Decimal
            return float(value)
        if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            return self._sanitize(value.to_dict(), _depth + 1)
        if hasattr(value, "__dict__"):
            return self._sanitize(vars(value), _depth + 1)
        # 最后的保底：返回字符串表示，但加上类型信息
        return {"_type": type(value).__name__, "_value": str(value)}

    def _sanitize_task_id(self, task_id: str) -> str:
        """问题1核心修复：仅允许字母数字、下划线、短横线的文件名，否则哈希化。"""
        safe = str(task_id or "").strip()
        if not safe:
            raise ValueError("task_id must not be empty")
        # 白名单校验
        if re.fullmatch(r"[a-zA-Z0-9_.-]+", safe):
            return safe
        # 不安全时使用哈希值，完全避免路径穿越
        import hashlib
        return hashlib.sha256(safe.encode()).hexdigest()

    def _format_error(self, error: Exception | str) -> str:
        """问题7：保留异常类型和截断的 traceback，避免 last_error 无限长。"""
        if isinstance(error, str):
            return error.strip()
        tb = traceback.format_exception(type(error), error, error.__traceback__)
        full = "".join(tb).strip()
        # 限制到 1000 字符，防止文件过大
        return full[:1000] if len(full) > 1000 else full

    def _get_field(self, raw: dict, key: str, default: Any) -> Any:
        """问题3修复：仅当键不存在时使用默认值，空字符串/0不会被覆盖。"""
        if key in raw:
            val = raw[key]
            # 对于数值字段，确保类型正确
            if key in ("started_at", "finished_at") and not isinstance(val, (int, float)):
                return default
            return val
        return default

    def _atomic_write(self, path: Path, state: AgentTaskState) -> None:
        """原子写入：使用临时文件 + os.replace，无锁（单进程安全）。"""
        # 确保目录存在
        path.parent.mkdir(parents=True, exist_ok=True)
        # 写入临时文件
        tmp_path = path.with_suffix(".tmp")
        content = json.dumps(
            self._sanitize(state.to_dict()),
            ensure_ascii=False,
            indent=2,
            default=str,
        ) + "\n"
        tmp_path.write_text(content, encoding="utf-8")
        # 原子替换（Windows 上 replace 会覆盖目标）
        tmp_path.replace(path)
