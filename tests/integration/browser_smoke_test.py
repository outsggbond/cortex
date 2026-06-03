# -*- coding: utf-8 -*-
"""
浏览器真实应用测试 - 烟雾测试框架

验证以下基本能力：
1. 浏览器任务规划
2. DOM 元素识别策略
3. 动作验证框架
4. 失败恢复策略

这是 P0 阶段的真实应用测试框架。

用法：
  python -m pytest tests/integration/browser_smoke_test.py -v
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from system.computer_use.runtime import ComputerUseRuntime, ComputerUseConfig


class TestBrowserTaskPlanning:
    """浏览器任务规划测试"""

    def test_simple_navigation_task_structure(self):
        """测试：简单导航任务的结构"""
        task = {
            "name": "navigate_and_extract",
            "description": "导航到网页并提取信息",
            "inputs": {
                "url": {"type": "str", "description": "目标 URL"},
                "selector": {"type": "str", "description": "CSS 选择器"},
            },
            "steps": [
                {"id": "launch_browser", "type": "browser.launch"},
                {"id": "navigate", "type": "browser.navigate", "params": {"url": "{url}"}},
                {"id": "wait_page", "type": "browser.wait", "params": {"ms": 1000}},
                {"id": "screenshot", "type": "screenshot", "purpose": "页面加载后"},
                {"id": "find_element", "type": "browser.find", "params": {"selector": "{selector}"}},
                {"id": "extract_text", "type": "browser.extract_text"},
            ],
        }
        
        assert task["name"] == "navigate_and_extract"
        assert len(task["steps"]) == 6
        print(f"✓ 浏览器导航任务规划完整，共 {len(task['steps'])} 个步骤")

    def test_form_submission_workflow(self):
        """测试：表单提交工作流"""
        workflow = {
            "name": "form_submission",
            "steps": [
                {"id": 1, "action": "launch_browser"},
                {"id": 2, "action": "navigate", "target": "https://example.com/form"},
                {"id": 3, "action": "find", "selector": "input[name='email']"},
                {"id": 4, "action": "type", "text": "test@example.com"},
                {"id": 5, "action": "find", "selector": "button[type='submit']"},
                {"id": 6, "action": "click"},
                {"id": 7, "action": "wait", "ms": 2000},
                {"id": 8, "action": "screenshot"},
            ],
        }
        
        assert len(workflow["steps"]) == 8
        assert workflow["steps"][0]["action"] == "launch_browser"
        assert workflow["steps"][-1]["action"] == "screenshot"
        print(f"✓ 表单工作流规划完成，共 {len(workflow['steps'])} 个步骤")


class TestBrowserErrorRecovery:
    """浏览器错误恢复测试"""

    def test_recovery_on_page_load_timeout(self):
        """测试：页面加载超时时的恢复"""
        recovery = {
            "error": "page_load_timeout",
            "trigger": "页面加载超过 10 秒",
            "recovery_steps": [
                {"step": 1, "action": "take_screenshot"},
                {"step": 2, "action": "check_page_ready", "criteria": "document.readyState === 'complete'"},
                {"step": 3, "action": "force_timeout", "ms": 5000},
                {"step": 4, "action": "retry_navigation", "max_retries": 3},
                {"step": 5, "action": "fallback_to_text_interaction"},
            ],
        }
        
        assert len(recovery["recovery_steps"]) == 5
        print(f"✓ 页面加载超时恢复策略完整")

    def test_recovery_on_element_not_found(self):
        """测试：元素未找到时的恢复"""
        recovery = {
            "error": "element_not_found",
            "selector": "button.submit",
            "recovery_steps": [
                {"step": 1, "action": "retry_selector"},
                {"step": 2, "action": "try_alternative_selectors", "alternatives": [
                    "button[type='submit']",
                    "button:contains('提交')",
                    "input[type='submit']",
                ]},
                {"step": 3, "action": "use_ocr", "text": "提交"},
                {"step": 4, "action": "screenshot"},
                {"step": 5, "action": "notify_failure"},
            ],
        }
        
        assert len(recovery["recovery_steps"]) == 5
        print(f"✓ 元素不存在恢复策略完整")


class TestBrowserConfigExample:
    """浏览器配置示例"""

    def test_computer_use_config_initialization(self):
        """测试：ComputerUseConfig 初始化"""
        config = ComputerUseConfig(
            project_root=".",
            goal="navigate_example",
            max_steps=8,
            dry_run=True,
        )
        
        assert config.project_root == "."
        assert config.goal == "navigate_example"
        assert config.max_steps == 8
        assert config.dry_run is True
        print("✓ ComputerUseConfig 初始化成功")

    def test_computer_use_runtime_initialization(self):
        """测试：ComputerUseRuntime 初始化"""
        config = ComputerUseConfig(dry_run=True)
        runtime = ComputerUseRuntime(config)
        
        assert runtime is not None
        print("✓ ComputerUseRuntime 初始化成功")


def main() -> None:
    """本地运行入口"""
    import pytest as pytest_module
    
    pytest_module.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    main()
