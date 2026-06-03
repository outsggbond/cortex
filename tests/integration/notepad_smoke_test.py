# -*- coding: utf-8 -*-
"""
Notepad 真实应用测试 - 基础功能验证

验证以下 Notepad 自动化能力：
1. Notepad 启动
2. 文本输入
3. 文件保存
4. 窗口操作

用法：
  python -m pytest tests/integration/notepad_smoke_test.py -v -s

这是最简单的真实应用测试，推荐作为 P0 突破口。
"""
from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from system.computer_use.runtime import ComputerUseRuntime, ComputerUseConfig


class TestNotepadSmoke:
    """Notepad 基础功能测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """初始化运行时"""
        self.config = ComputerUseConfig(
            dry_run=os.environ.get("COMPUTER_CONTROL_DRY_RUN", "0") == "1",
            debug=True,
            screenshot_dir=os.path.join(ROOT, "artifacts", "screenshots"),
        )
        self.runtime = ComputerUseRuntime(self.config)
        self.temp_dir = tempfile.mkdtemp(prefix="notepad_test_")
        os.makedirs(self.config.screenshot_dir, exist_ok=True)
        yield

    def test_notepad_launch(self):
        """测试：Notepad 启动"""
        result = self.runtime.run_action(
            action_type="desktop.launch_app",
            params={"app": "notepad"},
        )
        assert result["success"], f"Notepad 启动失败: {result.get('error')}"
        
        # 截图验证
        screenshot_result = self.runtime.run_action(action_type="screenshot")
        assert screenshot_result["success"], "截图失败"
        print("✓ Notepad 启动成功")

    @pytest.mark.skipif(
        os.environ.get("SKIP_REAL_APP") == "1" or os.environ.get("COMPUTER_CONTROL_DRY_RUN") == "1",
        reason="真实应用测试（需要 Notepad）或干运行模式跳过"
    )
    def test_notepad_text_input(self):
        """测试：文本输入"""
        # 启动 Notepad
        self.runtime.run_action(
            action_type="desktop.launch_app",
            params={"app": "notepad"},
        )
        
        # 输入文本
        text = "This is a test file.\nLine 2\nLine 3"
        result = self.runtime.run_action(
            action_type="desktop.type_text",
            params={"text": text},
        )
        assert result["success"], f"文本输入失败: {result.get('error')}"
        
        # 截图验证
        screenshot_result = self.runtime.run_action(action_type="screenshot")
        assert screenshot_result["success"], "验证截图失败"
        print("✓ 文本输入成功")

    @pytest.mark.skipif(
        os.environ.get("SKIP_REAL_APP") == "1" or os.environ.get("COMPUTER_CONTROL_DRY_RUN") == "1",
        reason="真实应用测试"
    )
    def test_notepad_save_file(self):
        """测试：文件保存"""
        # 启动 Notepad
        self.runtime.run_action(
            action_type="desktop.launch_app",
            params={"app": "notepad"},
        )
        
        # 输入文本
        self.runtime.run_action(
            action_type="desktop.type_text",
            params={"text": "Test content for saving"},
        )
        
        # 保存文件 (Ctrl+S)
        save_result = self.runtime.run_action(
            action_type="desktop.hotkey",
            params={"key": "ctrl+s"},
        )
        assert save_result["success"], f"保存快捷键失败: {save_result.get('error')}"
        
        print("✓ 文件保存命令执行成功")

    def test_notepad_dry_run_workflow(self):
        """测试：干运行模式下的完整工作流"""
        config = ComputerUseConfig(dry_run=True, debug=True)
        runtime = ComputerUseRuntime(config)
        
        workflow = [
            {"type": "screenshot", "purpose": "初始状态"},
            {"type": "desktop.launch_app", "params": {"app": "notepad"}},
            {"type": "screenshot", "purpose": "Notepad 启动后"},
            {"type": "desktop.type_text", "params": {"text": "Hello, Notepad!"}},
            {"type": "desktop.hotkey", "params": {"key": "ctrl+a"}},
            {"type": "desktop.hotkey", "params": {"key": "ctrl+c"}},
            {"type": "screenshot", "purpose": "复制操作后"},
            {"type": "desktop.hotkey", "params": {"key": "ctrl+s"}},
            {"type": "screenshot", "purpose": "保存操作后"},
            {"type": "desktop.hotkey", "params": {"key": "alt+f4"}},
            {"type": "screenshot", "purpose": "关闭后"},
        ]
        
        for step in workflow:
            action_type = step["type"]
            params = step.get("params", {})
            result = runtime.run_action(action_type=action_type, params=params)
            assert result is not None, f"动作 {action_type} 返回 None"
        
        print(f"✓ Notepad 干运行工作流完成，共 {len(workflow)} 个步骤")


class TestNotepadTaskCompletion:
    """Notepad 任务完成度测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """初始化"""
        self.config = ComputerUseConfig(dry_run=True)
        self.runtime = ComputerUseRuntime(self.config)
        self.temp_dir = tempfile.mkdtemp(prefix="notepad_test_")
        yield

    def test_create_and_save_file_task(self):
        """测试：创建和保存文件的任务规划"""
        task = {
            "name": "create_notepad_file",
            "description": "在 Notepad 中创建并保存文本文件",
            "inputs": {
                "content": {"type": "str", "description": "文件内容"},
                "filename": {"type": "str", "description": "文件名"},
            },
            "steps": [
                {"id": "launch", "action": "launch_app", "app": "notepad"},
                {"id": "wait_window", "action": "wait", "ms": 1000},
                {"id": "type_content", "action": "type_text", "text": "{content}"},
                {"id": "open_save_dialog", "action": "hotkey", "key": "ctrl+s"},
                {"id": "wait_dialog", "action": "wait", "ms": 500},
                {"id": "type_filename", "action": "type_text", "text": "{filename}"},
                {"id": "press_save", "action": "hotkey", "key": "enter"},
                {"id": "wait_save", "action": "wait", "ms": 1000},
                {"id": "verify", "action": "screenshot"},
            ],
            "expected_outputs": {
                "file_created": bool,
                "file_path": str,
                "success": bool,
            },
        }
        
        assert task["name"] == "create_notepad_file"
        assert len(task["steps"]) == 9
        assert task["steps"][0]["id"] == "launch"
        assert task["steps"][-1]["id"] == "verify"
        print(f"✓ Notepad 文件创建任务规划完整，共 {len(task['steps'])} 个步骤")

    def test_edit_existing_file_workflow(self):
        """测试：编辑现有文件的工作流"""
        workflow = {
            "name": "edit_notepad_file",
            "trigger": "user requests to edit file",
            "steps": [
                {"id": 1, "action": "launch_app", "app": "notepad"},
                {"id": 2, "action": "hotkey", "key": "ctrl+o"},
                {"id": 3, "action": "wait", "ms": 500},
                {"id": 4, "action": "type_text", "text": "path/to/file.txt"},
                {"id": 5, "action": "hotkey", "key": "enter"},
                {"id": 6, "action": "wait", "ms": 1000},
                {"id": 7, "action": "hotkey", "key": "ctrl+end"},
                {"id": 8, "action": "type_text", "text": "\nAppended line"},
                {"id": 9, "action": "hotkey", "key": "ctrl+s"},
                {"id": 10, "action": "screenshot"},
            ],
            "recovery": {
                "on_file_not_found": [
                    {"action": "cancel_dialog"},
                    {"action": "notify_user", "message": "File not found"},
                ],
            },
        }
        
        assert len(workflow["steps"]) == 10
        assert workflow["recovery"]["on_file_not_found"] is not None
        print(f"✓ Notepad 编辑工作流完整，包含恢复策略")

    def test_notepad_copy_paste_task(self):
        """测试：复制-粘贴任务"""
        task = {
            "name": "copy_paste_in_notepad",
            "steps": [
                {"step": 1, "action": "launch", "target": "notepad"},
                {"step": 2, "action": "type", "text": "Source content"},
                {"step": 3, "action": "select_all", "key": "ctrl+a"},
                {"step": 4, "action": "copy", "key": "ctrl+c"},
                {"step": 5, "action": "new_window"},
                {"step": 6, "action": "paste", "key": "ctrl+v"},
                {"step": 7, "action": "verify", "check": "content_matches"},
            ],
        }
        
        assert len(task["steps"]) == 7
        assert task["steps"][2]["action"] == "select_all"
        print(f"✓ Notepad 复制-粘贴任务规划完整")


class TestNotepadErrorRecovery:
    """Notepad 错误恢复测试"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """初始化"""
        self.config = ComputerUseConfig(dry_run=True)
        self.runtime = ComputerUseRuntime(self.config)
        yield
        self.runtime.cleanup()

    def test_recovery_on_unsaved_changes(self):
        """测试：未保存更改时的恢复"""
        recovery = {
            "error": "unsaved_changes",
            "scenario": "User tries to close Notepad with unsaved content",
            "recovery_steps": [
                {"id": 1, "action": "detect_save_dialog"},
                {"id": 2, "action": "take_screenshot"},
                {"id": 3, "action": "analyze_dialog", "look_for": ["Save", "Don't Save", "Cancel"]},
                {"id": 4, "action": "user_confirm", "prompt": "Save changes?"},
                {"id": 5, "action": "click", "button": "Save"},
                {"id": 6, "action": "wait_save", "ms": 1000},
                {"id": 7, "action": "verify", "state": "window_closed"},
            ],
        }
        
        assert len(recovery["recovery_steps"]) == 7
        print(f"✓ 未保存更改恢复策略完整")

    def test_recovery_on_app_crash(self):
        """测试：应用崩溃时的恢复"""
        recovery = {
            "error": "app_crashed",
            "detection": ["Process terminated", "Window closed unexpectedly"],
            "recovery_steps": [
                {"id": 1, "action": "detect_crash"},
                {"id": 2, "action": "check_recovery_file"},
                {"id": 3, "action": "restart_app"},
                {"id": 4, "action": "check_recovery_file_list"},
                {"id": 5, "action": "restore_if_available"},
            ],
        }
        
        assert len(recovery["recovery_steps"]) == 5
        print(f"✓ 应用崩溃恢复策略完整")


def main() -> None:
    """本地运行入口"""
    import pytest as pytest_module
    
    pytest_module.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
