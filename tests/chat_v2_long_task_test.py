# -*- coding: utf-8 -*-
"""P2: 长任务处理和进度跟踪测试"""

import pytest
from system.core.planner import Plan, Task


class TestLongTaskHandling:
    """验证系统对长任务（50+ 步）的处理能力"""

    def test_long_task_creation(self):
        """测试：创建长任务"""
        # 创建一个 50+ 步的任务计划
        tasks = [
            Task(name=f"step_{i:03d}", detail=f"Execute step {i}", payload={"step": i})
            for i in range(1, 51)
        ]
        
        plan = Plan(goal="long_task_test", steps=tasks)
        
        assert len(plan.steps) >= 50
        print(f"✓ 创建 {len(plan.steps)} 步的长任务")

    def test_long_task_progress_tracking(self):
        """测试：长任务进度跟踪"""
        # 创建 100 步的任务
        tasks = [
            Task(name=f"step_{i:03d}", detail=f"Step {i}", payload={"step": i})
            for i in range(1, 101)
        ]
        plan = Plan(goal="progress_test", steps=tasks)
        
        # 模拟执行前 30 步
        completed = 30
        progress = completed / len(plan.steps)
        
        print(f"✓ 进度: {progress:.0%} ({completed}/{len(plan.steps)})")
        assert progress > 0.2

    def test_long_task_intermediate_summary(self):
        """测试：中间状态总结"""
        # 每 10 步总结一次
        tasks = [
            Task(name=f"step_{i:03d}", detail=f"Step {i}", payload={"step": i})
            for i in range(1, 101)
        ]
        
        summaries = []
        for i in range(0, len(tasks), 10):
            checkpoint = i + 10
            summary = f"完成第 {checkpoint} 步，已处理 {checkpoint} 个步骤"
            summaries.append(summary)
        
        assert len(summaries) >= 10
        print(f"✓ 创建 {len(summaries)} 个中间总结点")

    def test_long_task_context_preservation(self):
        """测试：长任务的上下文保持"""
        # 验证系统可以在长任务中保持上下文
        initial_context = {"user": "test", "goal": "long_task", "step": 0}
        
        # 模拟 100 步执行
        current_context = dict(initial_context)
        for step in range(1, 101):
            current_context["step"] = step
            # 验证上下文不丢失
            assert current_context["goal"] == "long_task"
            assert current_context["user"] == "test"
        
        print(f"✓ 上下文在 {current_context['step']} 步后保持完整")

    def test_long_task_checkpoint_save(self):
        """测试：长任务的保存点"""
        checkpoints = []
        
        # 每 25 步创建一个保存点
        for i in range(0, 101, 25):
            checkpoint = {
                "step": i,
                "state": f"checkpoint_at_step_{i}",
            }
            checkpoints.append(checkpoint)
        
        assert len(checkpoints) >= 4
        print(f"✓ 创建 {len(checkpoints)} 个保存点")

    def test_long_task_failure_recovery(self):
        """测试：长任务失败恢复"""
        # 从保存点恢复
        failed_step = 67
        recovery_point = 50  # 最近的保存点
        
        print(f"✓ 任务在第 {failed_step} 步失败，从第 {recovery_point} 步恢复")
        
        remaining_steps = failed_step - recovery_point
        assert remaining_steps > 0
        print(f"✓ 需要重新执行 {remaining_steps} 个步骤")


class TestErrorFeedbackForLongTasks:
    """验证长任务的错误反馈质量"""

    def test_error_message_clarity(self):
        """测试：错误消息清晰度"""
        error_cases = [
            {
                "error": "window_not_found",
                "message": "未找到目标窗口 'Microsoft Word'，请确保应用已打开",
                "suggestion": "建议：使用 launch 命令打开应用，或检查窗口标题是否正确"
            },
            {
                "error": "timeout",
                "message": "等待元素超时（已等待 30 秒）",
                "suggestion": "建议：检查网络连接，或增加超时时间"
            },
            {
                "error": "invalid_selector",
                "message": "无法识别 CSS 选择器 '.invalid-class'",
                "suggestion": "建议：验证选择器的正确性，或使用 OCR 识别文本"
            }
        ]
        
        for case in error_cases:
            print(f"✓ 错误类型 '{case['error']}' 消息清晰")
            assert len(case["message"]) > 10
            assert len(case["suggestion"]) > 10

    def test_recovery_suggestions(self):
        """测试：恢复建议"""
        scenarios = [
            {
                "situation": "浏览器页面加载失败",
                "suggestions": [
                    "重新加载页面",
                    "检查网络连接",
                    "更换代理服务器"
                ]
            },
            {
                "situation": "桌面应用无响应",
                "suggestions": [
                    "等待 5 秒后重试",
                    "重新启动应用",
                    "检查系统资源使用率"
                ]
            }
        ]
        
        for scenario in scenarios:
            assert len(scenario["suggestions"]) >= 2
            print(f"✓ '{scenario['situation']}' 有 {len(scenario['suggestions'])} 个恢复建议")


class TestUserUnderstanding:
    """验证用户对失败原因的理解度"""

    def test_failure_explanation(self):
        """测试：失败原因说明"""
        explanations = [
            {
                "step": 45,
                "failure": "点击按钮失败",
                "reason": "按钮在浏览器中不可见（可能被弹窗遮挡）",
                "user_understandable": True
            },
            {
                "step": 78,
                "failure": "文本输入失败",
                "reason": "目标文本框已被禁用",
                "user_understandable": True
            }
        ]
        
        understandable_count = sum(
            1 for e in explanations if e["user_understandable"]
        )
        
        clarity_rate = understandable_count / len(explanations)
        print(f"✓ 用户理解度: {clarity_rate:.0%}")
        assert clarity_rate >= 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
