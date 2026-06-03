# -*- coding: utf-8 -*-
"""P2: 意图识别准确率测试"""

import pytest
from system.chat_v2.intents import ChatIntent, classify_intent


class TestIntentRecognition:
    """验证意图识别的准确性"""

    def test_identity_questions(self):
        """测试：身份相关问题"""
        test_cases = [
            ("你叫什么名字", ChatIntent.IDENTITY),
            ("你是谁", ChatIntent.IDENTITY),
            ("你的名字是什么", ChatIntent.IDENTITY),
            ("What's your name", ChatIntent.IDENTITY),
            ("你是谁啊", ChatIntent.IDENTITY),
        ]
        
        for text, expected in test_cases:
            result = classify_intent(text)
            assert result == expected, f"Expected {expected} for '{text}', got {result}"
        
        print(f"✓ 身份问题识别: {len(test_cases)}/5 通过")

    def test_capability_questions(self):
        """测试：能力相关问题"""
        test_cases = [
            ("你能做什么", ChatIntent.CAPABILITY),
            ("你会什么", ChatIntent.CAPABILITY),
            ("你的功能", ChatIntent.CAPABILITY),
            ("What can you do", ChatIntent.CAPABILITY),
            ("你能干什么", ChatIntent.CAPABILITY),
        ]
        
        for text, expected in test_cases:
            result = classify_intent(text)
            assert result == expected, f"Expected {expected} for '{text}', got {result}"
        
        print(f"✓ 能力问题识别: {len(test_cases)}/5 通过")

    def test_greeting_queries(self):
        """测试：问候"""
        test_cases = [
            ("你好", ChatIntent.GREETING),
            ("早上好", ChatIntent.GREETING),
            ("晚上好", ChatIntent.GREETING),
            ("hello", ChatIntent.GREETING),
            ("早安", ChatIntent.GREETING),
        ]
        
        for text, expected in test_cases:
            result = classify_intent(text)
            assert result == expected, f"Expected {expected} for '{text}', got {result}"
        
        print(f"✓ 问候识别: {len(test_cases)}/5 通过")

    def test_computer_tasks_explicit(self):
        """测试：明确的计算机任务（指令式）"""
        test_cases = [
            ("screenshot", ChatIntent.TASK),
            ("template qq_send_message", ChatIntent.TASK),
            ("runtemplate browser_search", ChatIntent.TASK),
            ("click on the button", ChatIntent.TASK),
            ("type hello", ChatIntent.TASK),
        ]
        
        for text, expected in test_cases:
            result = classify_intent(text)
            assert result == expected, f"Expected {expected} for '{text}', got {result}"
        
        print(f"✓ 明确任务识别: {len(test_cases)}/5 通过")

    def test_computer_tasks_natural_language(self):
        """测试：自然语言计算机任务"""
        test_cases = [
            ("给张三发送一条消息", ChatIntent.TASK),
            ("打开浏览器搜索 Python", ChatIntent.TASK),
            ("创建一个文件", ChatIntent.TASK),
            ("点击桌面上的图标", ChatIntent.TASK),
            ("发送一条 QQ 消息", ChatIntent.TASK),
        ]
        
        for text, expected in test_cases:
            result = classify_intent(text)
            assert result == expected, f"Expected {expected} for '{text}', got {result}"
        
        print(f"✓ 自然语言任务识别: {len(test_cases)}/5 通过")

    def test_intent_recognition_accuracy(self):
        """测试：整体意图识别准确率"""
        all_tests = [
            # 身份问题
            ("你叫什么名字", ChatIntent.IDENTITY),
            ("你是谁", ChatIntent.IDENTITY),
            # 能力问题
            ("你会什么", ChatIntent.CAPABILITY),
            ("你的功能", ChatIntent.CAPABILITY),
            # 问候
            ("你好", ChatIntent.GREETING),
            ("早上好", ChatIntent.GREETING),
            # 任务
            ("screenshot", ChatIntent.TASK),
            ("给张三发消息", ChatIntent.TASK),
            ("打开浏览器", ChatIntent.TASK),
        ]
        
        correct = 0
        for text, expected in all_tests:
            result = classify_intent(text)
            if result == expected:
                correct += 1
        
        accuracy = correct / len(all_tests)
        print(f"✓ 整体准确率: {accuracy:.0%} ({correct}/{len(all_tests)})")
        
        # 期望至少 85% 的准确率
        assert accuracy >= 0.85, f"准确率 {accuracy:.0%} 低于目标 85%"


class TestParameterExtraction:
    """验证参数提取的准确性"""

    def test_qq_message_parameters(self):
        """测试：QQ 消息参数提取"""
        test_cases = [
            {
                "text": "给张三发送晚安",
                "expected_params": {
                    "recipient": "张三",
                    "message": "晚安"
                }
            },
            {
                "text": "给李四发送一条消息，说我今天很忙",
                "expected_params": {
                    "recipient": "李四",
                    "message": "我今天很忙"
                }
            },
        ]
        
        # 注：这是框架，具体提取逻辑待实现
        for test_case in test_cases:
            text = test_case["text"]
            # 未来的 extract_parameters 函数应该能从自然语言中提取这些参数
            print(f"✓ 参数提取框架准备好: {text}")

    def test_browser_search_parameters(self):
        """测试：浏览器搜索参数提取"""
        test_cases = [
            {
                "text": "搜索 Python 教程",
                "expected_params": {
                    "query": "Python 教程",
                    "engine": "default"
                }
            },
        ]
        
        for test_case in test_cases:
            text = test_case["text"]
            print(f"✓ 浏览器参数提取框架准备好: {text}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
