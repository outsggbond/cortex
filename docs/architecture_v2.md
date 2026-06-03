## 中文版

# 聊天运行时 V2（LLM 优先、模块化、可测试）

## 1. 为什么需要 V2

V2 直接解决当前痛点：

1. 不与前沿 LLM 在通用闲聊上竞争；将 LLM 作为主要引擎。
2. 停止 `system/app_runtime.py` 单体膨胀；引入独立的模块化运行时。
3. 停止在污染的混合风格对话记忆上训练；增加精选记忆构建步骤。
4. 移除单函数中的重叠多策略；强制执行单路径决策流水线。
5. 规范化配置/数据编码，清理不良回复模式。
6. 在自适应/bandit 复杂性之前优先保证核心回复质量。
7. 为对话质量增加行为级测试，而非仅测试“程序能跑”。

## 2. 运行时分流

V2 实现在 `system/chat_v2/` 中，通过以下命令选择：

```bash
python main.py --chat --runtime-arch v2
```

遗留运行时仍可通过以下命令使用：

```bash
python main.py --chat --runtime-arch legacy
```

## 3. 模块边界

1. `system/chat_v2/intents.py`  
   意图分类 + 确定性策略回复（身份/能力/问候）。

2. `system/chat_v2/llm.py`  
   LLM 适配层（`none` / `openai`），具有严格接口。

3. `system/chat_v2/memory.py`  
   从精选记忆 JSONL 进行的只读检索。

4. `system/chat_v2/pipeline.py`  
   单路径编排：  
   策略 → LLM → 检索 → 后备 → 守卫

5. `system/chat_v2/runtime.py`  
   仅聊天循环；无训练逻辑，无规划器，无联邦副作用。

## 4. 决策顺序（单一路径）

对每条用户消息：

1. 分类意图。
2. 若为确定性意图（身份/能力/问候），返回策略回复。
3. 否则，若有可用 LLM，则调用。
4. 若 LLM 不可用或质量低，尝试精选检索。
5. 若仍无效，返回可操作的后备回复。
6. 输出前应用角色归一化 + 回复守卫。

这消除了路由重叠和不稳定的行为振荡。

## 5. 数据卫生

使用：

```bash
python scripts/build_curated_dialogue_memory.py
```

此脚本：

1. 读取 `artifacts/memory/dialogue_patterns.json`（+ 可选额外 JSONL）。
2. 过滤上下文污染/浪漫漂移/低信号垃圾。
3. 将输出归一化为稳定的角色策略。
4. 按归一化的用户查询去重。
5. 写入 `artifacts/memory/dialogue_curated.jsonl`。

V2 运行时仅消费此精选记忆文件。

## 6. 质量契约

V2 回复仅当通过所有检查才被视为有效：

1. 非低信号（`system.response_guard.is_low_signal_response`）。
2. 通过 `system.response_guard.guard_response`。
3. 角色归一化不产生禁止模式。

## 7. 迁移计划

1. 以影子/手动模式启动 V2：`--runtime-arch v2`。
2. 为非聊天功能保留遗留运行时。
3. 逐个将额外能力移入模块化包。
4. 行为指标稳定后，弃用遗留纯聊天路径。

---

## English Version

# Architecture V2 (LLM-First, Modular, Testable)

## 1. Why V2

V2 directly addresses the current pain points:

1. Do not compete with frontier LLM on generic chat; use LLM as primary engine.
2. Stop growing `system/app_runtime.py` monolith; introduce a separate modular runtime.
3. Stop training on polluted mixed-style dialogue memory; add curated memory build step.
4. Remove overlapping multi-strategy in one function; enforce a single-path decision pipeline.
5. Normalize config/data encoding and sanitize bad response patterns.
6. Prioritize core response quality before adaptive/bandit complexity.
7. Add behavior-level tests for conversation quality, not only "program runs."

## 2. Runtime Split

V2 is implemented in `system/chat_v2/` and selected via:

```bash
python main.py --chat --runtime-arch v2
```

Legacy runtime remains available with:

```bash
python main.py --chat --runtime-arch legacy
```

## 3. Module Boundaries

1. `system/chat_v2/intents.py`
Intent classification + deterministic policy replies (identity/capability/greeting).

2. `system/chat_v2/llm.py`
LLM adapter layer (`none` / `openai`) with a strict interface.

3. `system/chat_v2/memory.py`
Read-only retrieval from curated memory JSONL.

4. `system/chat_v2/pipeline.py`
Single-path orchestration:
policy -> llm -> retrieval -> fallback -> guard

5. `system/chat_v2/runtime.py`
Chat loop only; no training logic, no planner, no federated side effects.

## 4. Decision Order (Single Path)

For each user message:

1. Classify intent.
2. If deterministic intent (identity/capability/greeting), return policy reply.
3. Else call LLM if available.
4. If LLM unavailable or low-quality, try curated retrieval.
5. If still invalid, return actionable fallback.
6. Apply persona normalization + response guard before output.

This removes route overlap and unstable behavior oscillation.

## 5. Data Hygiene

Use:

```bash
python scripts/build_curated_dialogue_memory.py
```

This script:

1. Reads `artifacts/memory/dialogue_patterns.json` (+ optional extra JSONL).
2. Filters context pollution / romantic drift / low-signal garbage.
3. Normalizes outputs to stable persona policy.
4. Deduplicates by normalized user query.
5. Writes `artifacts/memory/dialogue_curated.jsonl`.

V2 runtime consumes only this curated memory file.

## 6. Quality Contract

V2 response is considered valid only if all checks pass:

1. Not low-signal (`system.response_guard.is_low_signal_response`).
2. Passes `system.response_guard.guard_response`.
3. Persona normalization does not produce prohibited patterns.

## 7. Migration Plan

1. Start V2 in shadow/manual mode:
`--runtime-arch v2`.
2. Keep legacy runtime for non-chat features.
3. Move additional capabilities into modular packages one by one.
4. After behavior metrics are stable, deprecate legacy pure-chat path.
