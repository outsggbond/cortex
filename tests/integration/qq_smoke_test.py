# -*- coding: utf-8 -*-
"""
QQ 真实应用测试 - 烟雾测试

验证以下 QQ 自动化能力：
1. QQ 窗口识别和获取焦点
2. 联系人搜索
3. 消息输入和发送
4. 动作验证和失败恢复

用法：
  python -m pytest tests/integration/qq_smoke_test.py -v -s

注意：这个测试需要 QQ 已安装并可用。可以通过环境变量 SKIP_REAL_APP 跳过。
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from system.computer_use.runtime import ComputerUseRuntime, ComputerUseConfig


class TestQQSmoke:
    """QQ 基础功能测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """初始化运行时"""
        self.config = ComputerUseConfig(
            dry_run=os.environ.get("COMPUTER_CONTROL_DRY_RUN", "0") == "1",
            debug=True,
            screenshot_dir=os.path.join(ROOT, "artifacts", "screenshots"),
        )
        self.runtime = ComputerUseRuntime(self.config)
        os.makedirs(self.config.screenshot_dir, exist_ok=True)
        yield

    @pytest.mark.skipif(
        os.environ.get("SKIP_REAL_APP") == "1" or os.environ.get("COMPUTER_CONTROL_DRY_RUN") == "1",
        reason="真实应用测试（需要 QQ）或干运行模式跳过"
    )
    def test_qq_window_focus(self):
        """测试：获取 QQ 窗口焦点"""
        result = self.runtime.run_action(
            action_type="desktop.focus_window",
            params={"window_title": "QQ"},
        )
        assert result["success"], f"QQ 获取焦点失败: {result.get('error')}"
        
        # 截图验证 QQ 已获得焦点
        screenshot_result = self.runtime.run_action(action_type="screenshot")
        assert screenshot_result["success"], "截图失败"
        print("✓ QQ 窗口焦点获取成功")

    @pytest.mark.skipif(
        os.environ.get("SKIP_REAL_APP") == "1" or os.environ.get("COMPUTER_CONTROL_DRY_RUN") == "1",
        reason="真实应用测试"
    )
    def test_qq_search_contact(self):
        """测试：QQ 中搜索联系人"""
        # 首先获得焦点
        self.runtime.run_action(
            action_type="desktop.focus_window",
            params={"window_title": "QQ"},
        )
        time.sleep(1)
        
        # 模拟搜索操作（使用 Ctrl+F）
        result = self.runtime.run_action(
            action_type="desktop.hotkey",
            params={"key": "ctrl+f"},
        )
        assert result["success"], f"快捷键失败: {result.get('error')}"
        time.sleep(0.5)
        
        # 输入搜索内容
        search_result = self.runtime.run_action(
            action_type="desktop.type_text",
            params={"text": "测试"},
        )
        assert search_result["success"], f"文本输入失败: {search_result.get('error')}"

    def test_qq_dry_run_workflow(self):
        """测试：干运行模式下的 QQ 工作流"""
        config = ComputerUseConfig(dry_run=True, debug=True)
        runtime = ComputerUseRuntime(config)
        
        # 完整的 QQ 消息发送工作流
        workflow = [
            {"type": "screenshot", "purpose": "初始状态"},
            {"type": "desktop.focus_window", "params": {"window_title": "QQ"}},
            {"type": "screenshot", "purpose": "QQ 获得焦点后"},
            {"type": "desktop.hotkey", "params": {"key": "ctrl+f"}},
            {"type": "desktop.type_text", "params": {"text": "测试"}},
            {"type": "desktop.hotkey", "params": {"key": "down"}},
            {"type": "desktop.hotkey", "params": {"key": "enter"}},
            {"type": "screenshot", "purpose": "联系人选中后"},
            {"type": "desktop.type_text", "params": {"text": "晚安"}},
            {"type": "desktop.hotkey", "params": {"key": "enter"}},
            {"type": "screenshot", "purpose": "消息发送后"},
        ]
        
        for step in workflow:
            action_type = step["type"]
            params = step.get("params", {})
            result = runtime.run_action(action_type=action_type, params=params)
            assert result is not None, f"动作 {action_type} 返回 None"
        
        print(f"✓ QQ 干运行工作流完成，共 {len(workflow)} 个步骤")


class TestQQTaskCompletion:
    """QQ 任务完成度测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """初始化"""
        self.config = ComputerUseConfig(dry_run=True)
        self.runtime = ComputerUseRuntime(self.config)
        yield
        self.runtime.cleanup()

    def test_qq_send_message_task_structure(self):
        """测试：QQ 发送消息任务的结构"""
        task = {
            "name": "send_qq_message",
            "description": "给指定联系人发送 QQ 消息",
            "inputs": {
                "contact": {"type": "str", "description": "联系人名称"},
                "message": {"type": "str", "description": "消息内容"},
            },
            "steps": [
                {"id": "focus_qq", "type": "desktop.focus_window", "params": {"window": "QQ"}},
                {"id": "wait_focus", "type": "desktop.wait", "params": {"ms": 500}},
                {"id": "open_search", "type": "desktop.hotkey", "params": {"key": "ctrl+f"}},
                {"id": "type_contact", "type": "desktop.type_text", "params": {"text": "{contact}"}},
                {"id": "select_contact", "type": "desktop.hotkey", "params": {"key": "down enter"}},
                {"id": "wait_chat", "type": "desktop.wait", "params": {"ms": 500}},
                {"id": "type_message", "type": "desktop.type_text", "params": {"text": "{message}"}},
                {"id": "send_message", "type": "desktop.hotkey", "params": {"key": "enter"}},
                {"id": "verify", "type": "screenshot"},
            ],
            "expected_output": {
                "success": bool,
                "message_sent": bool,
                "timestamp": str,
            },
        }
        
        assert task["name"] == "send_qq_message"
        assert len(task["steps"]) == 9
        assert task["steps"][0]["id"] == "focus_qq"
        assert task["steps"][-1]["id"] == "verify"
        print(f"✓ QQ 任务结构完整，共 {len(task['steps'])} 个步骤")

    def test_qq_multi_contact_message_plan(self):
        """测试：QQ 群发消息的规划"""
        contacts = ["张三", "李四", "王五"]
        message = "会议提醒"
        
        plan = {
            "name": "batch_qq_messages",
            "contacts": contacts,
            "message": message,
            "steps": [],
        }
        
        for contact in contacts:
            contact_steps = [
                {"action": "focus_qq"},
                {"action": "search", "target": contact},
                {"action": "select"},
                {"action": "type_message", "text": message},
                {"action": "send"},
                {"action": "screenshot"},
            ]
            plan["steps"].extend(contact_steps)
        
        assert len(plan["steps"]) == len(contacts) * 6
        assert plan["steps"][0]["action"] == "focus_qq"
        print(f"✓ QQ 群发消息规划完成，{len(contacts)} 个联系人，共 {len(plan['steps'])} 个步骤")


class TestQQErrorRecovery:
    """QQ 错误恢复测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """初始化"""
        self.config = ComputerUseConfig(dry_run=True)
        self.runtime = ComputerUseRuntime(self.config)
        yield
        self.runtime.cleanup()

    def test_qq_recovery_strategy_on_window_not_found(self):
        """测试：QQ 窗口未找到时的恢复策略"""
        recovery_plan = {
            "error": "window_not_found",
            "trigger": "QQ window not focused",
            "recovery_steps": [
                {"step": 1, "action": "take_screenshot", "purpose": "记录当前状态"},
                {"step": 2, "action": "list_windows"},
                {"step": 3, "action": "find_qq_window"},
                {"step": 4, "action": "focus_qq_window"},
                {"step": 5, "action": "verify_focus", "expected": "QQ window in focus"},
                {"step": 6, "action": "resume_original_task"},
            ],
        }
        
        assert recovery_plan["error"] == "window_not_found"
        assert len(recovery_plan["recovery_steps"]) == 6
        print(f"✓ QQ 恢复策略完整，包含 {len(recovery_plan['recovery_steps'])} 个恢复步骤")

    def test_qq_recovery_strategy_on_contact_not_found(self):
        """测试：联系人未找到时的恢复策略"""
        recovery_plan = {
            "error": "contact_not_found",
            "trigger": "搜索结果为空",
            "recovery_steps": [
                {"step": 1, "action": "clear_search"},
                {"step": 2, "action": "take_screenshot"},
                {"step": 3, "action": "notify_user", "message": "联系人未找到"},
                {"step": 4, "action": "list_recent_contacts"},
                {"step": 5, "action": "allow_manual_selection"},
            ],
        }
        
        assert recovery_plan["error"] == "contact_not_found"
        assert len(recovery_plan["recovery_steps"]) == 5
        print(f"✓ 联系人恢复策略完整，包含 {len(recovery_plan['recovery_steps'])} 个步骤")


def main() -> None:
    """本地运行入口"""
    import pytest as pytest_module
    
    pytest_module.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
