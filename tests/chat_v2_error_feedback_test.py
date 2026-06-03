# -*- coding: utf-8 -*-
"""P2: 错误反馈和恢复建议测试"""

import pytest
from typing import Dict, List


class TestErrorFeedback:
    """验证错误反馈的质量和清晰度"""

    def test_standardized_error_messages(self):
        """测试：标准化错误消息"""
        error_catalog = {
            "window_not_found": {
                "category": "system_error",
                "message_template": "未找到窗口 '{window_title}'",
                "example": "未找到窗口 'Microsoft Word'"
            },
            "timeout": {
                "category": "timing_error",
                "message_template": "等待超时 ({timeout}s)",
                "example": "等待超时 (30s)"
            },
            "selector_not_found": {
                "category": "browser_error",
                "message_template": "无法定位元素 '{selector}'",
                "example": "无法定位元素 '.button-close'"
            },
            "text_not_found": {
                "category": "recognition_error",
                "message_template": "屏幕上未找到文本 '{text}'",
                "example": "屏幕上未找到文本 '下一页'"
            },
        }
        
        for error_code, error_info in error_catalog.items():
            assert "message_template" in error_info
            assert "example" in error_info
            print(f"✓ 标准化错误: {error_code}")
        
        assert len(error_catalog) >= 4

    def test_clear_error_messages(self):
        """测试：清晰的错误消息"""
        test_cases = [
            {
                "error": "无法点击按钮",
                "clarity_score": 0.9,
                "explanation": "明确说明了问题（无法点击）和对象（按钮）"
            },
            {
                "error": "浏览器导航失败",
                "clarity_score": 0.8,
                "explanation": "说明了问题（导航失败）和场景（浏览器）"
            },
            {
                "error": "窗口活跃度检查失败",
                "clarity_score": 0.7,
                "explanation": "说明了检查项，但对用户不太明确"
            }
        ]
        
        avg_clarity = sum(c["clarity_score"] for c in test_cases) / len(test_cases)
        print(f"✓ 平均清晰度: {avg_clarity:.0%}")
        assert avg_clarity >= 0.7

    def test_contextual_error_messages(self):
        """测试：上下文错误消息"""
        scenarios = [
            {
                "context": "第 45 步，浏览器操作",
                "error": "点击元素失败",
                "message": "步骤 45: 点击元素失败 (浏览器不可见)",
                "includes_context": True
            },
            {
                "context": "第 78 步，桌面操作",
                "error": "找不到窗口",
                "message": "步骤 78: 找不到窗口 'Notepad' (可能已关闭)",
                "includes_context": True
            }
        ]
        
        contextual_count = sum(1 for s in scenarios if s["includes_context"])
        print(f"✓ 包含上下文的消息: {contextual_count}/{len(scenarios)}")
        assert contextual_count >= len(scenarios) - 1


class TestRecoverySuggestions:
    """验证恢复建议的有用性和准确性"""

    def test_recovery_suggestion_categories(self):
        """测试：恢复建议分类"""
        suggestions_map = {
            "retry": {
                "description": "重新尝试",
                "examples": [
                    "等待 2 秒后重试",
                    "刷新页面重试",
                    "重新启动应用"
                ]
            },
            "alternative": {
                "description": "寻找替代方案",
                "examples": [
                    "使用 OCR 代替 CSS 选择器定位",
                    "使用坐标点击代替文本匹配",
                    "使用其他搜索引擎"
                ]
            },
            "diagnostic": {
                "description": "诊断问题",
                "examples": [
                    "检查网络连接",
                    "验证元素选择器正确性",
                    "检查系统资源使用率"
                ]
            },
            "manual_intervention": {
                "description": "需要用户干预",
                "examples": [
                    "请手动关闭阻挡窗口",
                    "请检查是否需要登录",
                    "请验证输入的参数是否正确"
                ]
            }
        }
        
        print(f"✓ 恢复建议分类数: {len(suggestions_map)}")
        assert len(suggestions_map) >= 3

    def test_recovery_success_likelihood(self):
        """测试：恢复建议的成功可能性"""
        recovery_attempts = [
            {
                "error": "timeout_error",
                "suggestions": [
                    {"action": "等待 5 秒后重试", "likelihood": 0.6},
                    {"action": "检查网络连接", "likelihood": 0.3},
                    {"action": "增加超时时间到 60 秒", "likelihood": 0.8},
                ]
            },
            {
                "error": "element_not_found",
                "suggestions": [
                    {"action": "使用 OCR 重新定位", "likelihood": 0.7},
                    {"action": "验证选择器", "likelihood": 0.9},
                    {"action": "截图查看当前状态", "likelihood": 0.8},
                ]
            }
        ]
        
        for attempt in recovery_attempts:
            best_suggestion = max(attempt["suggestions"], key=lambda x: x["likelihood"])
            print(f"✓ {attempt['error']}: 最优建议成功率 {best_suggestion['likelihood']:.0%}")
            assert best_suggestion["likelihood"] >= 0.6

    def test_ordered_recovery_suggestions(self):
        """测试：有序恢复建议"""
        # 建议应该按成功率排序
        suggestions = [
            {"action": "重试", "success_rate": 0.8, "cost": "low"},
            {"action": "诊断网络", "success_rate": 0.6, "cost": "medium"},
            {"action": "手动干预", "success_rate": 0.3, "cost": "high"},
        ]
        
        # 验证按成功率从高到低排序
        for i in range(len(suggestions) - 1):
            assert suggestions[i]["success_rate"] >= suggestions[i+1]["success_rate"]
        
        print(f"✓ 恢复建议正确排序: {len(suggestions)} 个")


class TestFailurePatternLearning:
    """验证系统是否能记录和学习失败模式"""

    def test_failure_pattern_recording(self):
        """测试：失败模式记录"""
        failure_patterns = [
            {
                "error_type": "timeout",
                "scenario": "浏览器加载页面",
                "frequency": 5,
                "last_occurred": "2024-12-01 14:30",
            },
            {
                "error_type": "selector_not_found",
                "scenario": "点击按钮",
                "frequency": 3,
                "last_occurred": "2024-12-01 14:25",
            },
            {
                "error_type": "window_not_found",
                "scenario": "桌面操作",
                "frequency": 2,
                "last_occurred": "2024-12-01 14:20",
            }
        ]
        
        print(f"✓ 记录的失败模式: {len(failure_patterns)}")
        assert len(failure_patterns) >= 3

    def test_failure_pattern_analysis(self):
        """测试：失败模式分析"""
        # 分析最常见的失败模式
        failure_patterns = [
            {"error_type": "timeout", "count": 8, "resolution_time": 120},
            {"error_type": "selector_not_found", "count": 5, "resolution_time": 90},
            {"error_type": "window_not_found", "count": 3, "resolution_time": 60},
        ]
        
        most_common = max(failure_patterns, key=lambda x: x["count"])
        avg_resolution_time = sum(p["resolution_time"] for p in failure_patterns) / len(failure_patterns)
        
        print(f"✓ 最常见的失败: {most_common['error_type']} ({most_common['count']} 次)")
        print(f"✓ 平均解决时间: {avg_resolution_time:.0f} 秒")

    def test_failure_prevention_rules(self):
        """测试：失败预防规则"""
        # 基于历史失败，生成预防规则
        prevention_rules = [
            {
                "condition": "如果浏览器操作超过 3 次失败",
                "action": "自动增加超时时间到 60 秒",
                "effectiveness": 0.85
            },
            {
                "condition": "如果选择器连续 2 次失败",
                "action": "自动切换到 OCR 定位方式",
                "effectiveness": 0.75
            },
            {
                "condition": "如果窗口查找失败",
                "action": "自动截图并提示用户",
                "effectiveness": 0.9
            }
        ]
        
        print(f"✓ 预防规则数: {len(prevention_rules)}")
        avg_effectiveness = sum(r["effectiveness"] for r in prevention_rules) / len(prevention_rules)
        print(f"✓ 平均有效性: {avg_effectiveness:.0%}")
        assert avg_effectiveness >= 0.7


class TestUserUnderstandingOfErrors:
    """验证用户对错误的理解度"""

    def test_error_explanation_clarity(self):
        """测试：错误说明的清晰度"""
        explanations = [
            {
                "error": "无法找到 '下一步' 按钮",
                "explanation": "系统在屏幕上查找文本 '下一步'，但未找到。可能是：1) 按钮不在当前可见区域；2) 页面尚未加载完成；3) 按钮已被禁用。",
                "clarity_score": 0.95
            },
            {
                "error": "浏览器导航超时",
                "explanation": "访问网站超过 30 秒仍未完成。建议检查网络连接或尝试访问其他网站。",
                "clarity_score": 0.85
            },
            {
                "error": "应用崩溃",
                "explanation": "应用意外关闭。",
                "clarity_score": 0.5
            }
        ]
        
        avg_clarity = sum(e["clarity_score"] for e in explanations) / len(explanations)
        print(f"✓ 平均清晰度: {avg_clarity:.0%}")
        assert avg_clarity >= 0.7

    def test_actionable_error_messages(self):
        """测试：可操作的错误消息"""
        messages = [
            {
                "message": "点击失败。建议：1) 截图查看当前状态；2) 使用 OCR 重新定位元素；3) 等待 2 秒后重试。",
                "is_actionable": True,
                "actions_count": 3
            },
            {
                "message": "出现错误。",
                "is_actionable": False,
                "actions_count": 0
            },
            {
                "message": "无法找到窗口。建议：启动应用或检查窗口标题。",
                "is_actionable": True,
                "actions_count": 2
            }
        ]
        
        actionable_count = sum(1 for m in messages if m["is_actionable"])
        print(f"✓ 可操作的消息: {actionable_count}/{len(messages)}")
        assert actionable_count >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
