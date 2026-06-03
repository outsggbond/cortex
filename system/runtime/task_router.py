from __future__ import annotations

import time
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

from system.agent.task import RuntimeTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路由规则抽象基类（可插拔的处理器）
# ---------------------------------------------------------------------------
class RouteRule(ABC):
    """每个具体的路由规则继承此类，实现 evaluate 方法"""

    @abstractmethod
    def evaluate(self, args: Any) -> Optional[RuntimeTask]:
        """如果该规则匹配给定的 args，返回 RuntimeTask；否则返回 None"""
        ...

    @staticmethod
    def _make_task(
        task_type: str,
        intent: str,
        goal: str,
        entrypoint: str,
        metadata: dict[str, Any],
    ) -> RuntimeTask:
        """统一的工厂方法，负责构建唯一任务ID和任务实例"""
        # 使用 uuid 片段 + 时间戳保证高并发下的唯一性
        unique_id = f"{uuid.uuid4().hex[:8]}-{int(time.time() * 1000)}"
        task_id = f"{task_type}-{unique_id}"
        return RuntimeTask(
            task_id=task_id,
            task_type=task_type,
            intent=intent,
            goal=goal,
            entrypoint=entrypoint,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _clean_items(values: Iterable[Any]) -> List[str]:
        """去重、清理字符串列表的辅助方法"""
        out: List[str] = []
        for val in list(values or []):
            text = str(val or "").strip()
            if text and text not in out:
                out.append(text)
        return out


# ---------------------------------------------------------------------------
# 具体的路由规则实现（每个规则对应原来一个 if 分支，使用工厂方法）
# ---------------------------------------------------------------------------
class ComputerRule(RouteRule):
    def evaluate(self, args: Any) -> Optional[RuntimeTask]:
        goal = str(getattr(args, "computer_use_goal", "") or "").strip()
        if not goal:
            return None
        return self._make_task(
            task_type="computer",
            intent="desktop_execution",
            goal=goal,
            entrypoint="system.domain.computer",
            metadata={
                "dry_run": bool(getattr(args, "computer_use_dry_run", False)),
                "target_window": str(getattr(args, "computer_use_target_window", "") or "").strip(),
            },
        )


class AutomationListRule(RouteRule):
    def evaluate(self, args: Any) -> Optional[RuntimeTask]:
        if not bool(getattr(args, "automation_list", False)):
            return None
        return self._make_task(
            task_type="automation",
            intent="workflow_list",
            goal="list automation workflows",
            entrypoint="system.domain.automation",
            metadata={},
        )


class AutomationRunRule(RouteRule):
    def evaluate(self, args: Any) -> Optional[RuntimeTask]:
        ids = list(getattr(args, "automation_run", []) or [])
        if not ids:
            return None
        goal = "run automation workflows: " + ", ".join(str(item) for item in ids)
        return self._make_task(
            task_type="automation",
            intent="workflow_run",
            goal=goal,
            entrypoint="system.domain.automation",
            metadata={"workflow_ids": [str(item) for item in ids]},
        )


class AutomationLoopRule(RouteRule):
    def evaluate(self, args: Any) -> Optional[RuntimeTask]:
        if not bool(getattr(args, "automation_loop", False)):
            return None
        return self._make_task(
            task_type="automation",
            intent="workflow_loop",
            goal="automation workflow loop",
            entrypoint="system.domain.automation",
            metadata={"poll_s": float(getattr(args, "automation_poll_s", 30.0) or 30.0)},
        )


class AutomationScanRule(RouteRule):
    def evaluate(self, args: Any) -> Optional[RuntimeTask]:
        if not bool(getattr(args, "automation_scan", False)):
            return None
        return self._make_task(
            task_type="automation",
            intent="workflow_scan",
            goal="scan automation workflows",
            entrypoint="system.domain.automation",
            metadata={},
        )


class FederatedTrainRule(RouteRule):
    def evaluate(self, args: Any) -> Optional[RuntimeTask]:
        if not bool(getattr(args, "federated_train", False)):
            return None
        script = str(getattr(args, "federated_script", "") or "").strip() or "scripts/train_adapter.py"
        return self._make_task(
            task_type="training",
            intent="federated_train",
            goal=f"federated adapter training via {script}",
            entrypoint="system.app_runtime._run_federated_from_main",
            metadata={"script": script},
        )


class RagIngestRule(RouteRule):
    def evaluate(self, args: Any) -> Optional[RuntimeTask]:
        if not bool(getattr(args, "rag_ingest", False)):
            return None
        paths = self._clean_items(getattr(args, "rag_paths", []) or [])
        goal = "rag ingest"
        if paths:
            goal = goal + ": " + ", ".join(paths)
        return self._make_task(
            task_type="knowledge",
            intent="rag_ingest",
            goal=goal,
            entrypoint="system.app_runtime._run_rag_ingest",
            metadata={"paths": paths},
        )


class ChatRule(RouteRule):
    def evaluate(self, args: Any) -> Optional[RuntimeTask]:
        if not bool(getattr(args, "chat", False)):
            return None
        return self._make_task(
            task_type="chat",
            intent="conversation",
            goal="interactive chat session",
            entrypoint="system.domain.chat",
            metadata={"runtime_arch": str(getattr(args, "runtime_arch", "") or "").strip()},
        )


# ---------------------------------------------------------------------------
# 路由器（支持注入规则，保持原公开接口）
# ---------------------------------------------------------------------------
class TaskRouter:
    def __init__(self, rules: List[RouteRule] | None = None):
        # 允许外部注入规则列表，否则使用默认的优先级顺序
        self._rules: List[RouteRule] = rules if rules is not None else [
            ComputerRule(),
            AutomationListRule(),
            AutomationRunRule(),
            AutomationLoopRule(),
            AutomationScanRule(),
            FederatedTrainRule(),
            RagIngestRule(),
            ChatRule(),
        ]

    def route(self, args: Any) -> RuntimeTask | None:
        """保持与原接口完全一致，使用责任链模式遍历规则，并记录匹配信息"""
        for rule in self._rules:
            task = rule.evaluate(args)
            if task is not None:
                logger.info(
                    "Task routed by %s -> %s (type=%s)",
                    type(rule).__name__,
                    task.task_id,
                    task.task_type,
                )
                return task
        logger.debug("No rule matched for args: %s", args)
        return None
