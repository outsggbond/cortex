# 桌面代理路线图

## 1. 目标

把当前仓库收口为一个可持续迭代的 Windows 桌面智能体系统，优先解决三件事：

1. 看得懂桌面
2. 动作可验证
3. 失败可恢复、可复盘、可回归

主线产品形态按优先级分三层推进：

1. 独立 `computer-use` runtime
2. `chat_v2` 交互式桌面代理
3. automation worker 持久化工作流

训练、联邦、推理优化暂时视为支撑能力，不作为第一阶段对外主叙事。

## 2. 当前基线

仓库现在已经具备以下基础：

1. `system/computer_use_runtime.py`
支持截图、网格覆盖、OCR、窗口/控件摘要、多模态决策、一步一执行、trace 落盘。

2. `system/chat_v2/agent.py`
支持浏览器 / 桌面动作的交互式 agent loop，可把自然语言请求转为本地动作序列。

3. `system/computer_plugins.py`
已经覆盖浏览器 DOM、桌面点击/输入/热键、Windows UIA 控件读取与操作等动作面。

4. `system/automation_runtime.py`
支持持久化 workflow、trigger、状态与运行日志。

5. 测试基础
已存在 `tests/computer_use_runtime_test.py`、`tests/chat_v2_computer_agent_test.py`、`tests/computer_plugin_focus_test.py`、`tests/automation_runtime_test.py`。

## 3. 设计原则

1. 先闭环，再扩面。
优先保证一批常见桌面任务稳定跑通，而不是过早追求“所有应用都能用”。

2. 先验证，再自动化。
优先增加观察、确认、回放和恢复能力，再扩动作数量。

3. 先 Windows-first，再跨平台。
当前桌面能力明显依赖 UIA、OCR、窗口焦点和本地插件，先把 Windows 路线做稳。

4. 先真实任务，再训练优化。
先沉淀任务 trace、失败样本、恢复样本，再决定哪些数据值得进入训练或 replay。

## 4. 阶段划分

### P0 稳定基线

目标：

1. `computer-use` dry-run 和 live-run 都能稳定完成最小任务闭环
2. 桌面聚焦测试可重复通过
3. trace、summary、恢复点格式稳定

验收建议：

1. 跑通 `tests/computer_use_runtime_test.py`
2. 跑通 `tests/chat_v2_computer_agent_test.py`
3. 跑通 `tests/computer_plugin_focus_test.py`
4. 至少有一个真实应用任务可复现，例如 QQ / 浏览器 / Notepad

### P1 单应用可靠性

目标：

1. 先选 2-3 个重点应用做深，而不是横向铺很多应用
2. 完善窗口匹配、控件匹配、OCR 兜底、动作验证
3. 提升恢复成功率

推荐优先应用：

1. QQ 或微信
2. 浏览器
3. 文件管理器或 Notepad

重点增强：

1. 更稳定的窗口识别
2. 基于控件优先、OCR 兜底的动作策略
3. 每一步的成功判定
4. 失败后的 repair plan

### P2 交互式产品入口

目标：

1. 把 `chat_v2` 变成主要的人机入口
2. 让自然语言请求更可靠地落到结构化动作
3. 把 learned templates 用起来，减少重复规划成本

重点增强：

1. 自然语言任务拆解质量
2. learned template 命中率
3. 对长任务的中间状态总结
4. 用户可理解的失败原因与下一步建议

### P3 自动化沉淀

目标：

1. 把稳定动作和稳定流程沉淀为 automation workflows
2. 增强定时触发、文件触发、运行状态持久化
3. 形成“桌面任务库”

重点增强：

1. workflow 模板化
2. 任务回放与重跑
3. 失败任务自动落审计
4. 常见桌面任务的可复用工作流

## 5. 关键指标

建议优先跟踪这些指标，而不是先追求模型参数层面的复杂优化：

1. 任务完成率
2. 单步验证通过率
3. 恢复成功率
4. 平均步数
5. 真实任务与 dry-run 结果的一致性
6. 重点应用的回归通过率

## 6. 当前建议删减或降级的方向

为了让桌面主线跑得更快，建议暂时降级这些叙事：

1. 泛聊天人格塑造
2. 强陪伴型训练语料的主线地位
3. "万能 AI 平台"叙事
4. 过早把联邦训练和桌面智能体并列成双主线

它们不是必须删除，但应该明确放到次线或实验区。

## 7. 近一步就能落地的任务

如果按一周一个主题推进，推荐顺序：

1. 固化桌面基线测试入口和 README 的主线说明
2. 为 QQ / 浏览器各补一组更真实的回归用例
3. 把失败恢复策略写成更清楚的状态机或文档
4. 给 `computer-use` 的 trace 增加更容易人工判断的摘要
5. 把稳定任务沉淀成 automation workflow 示例

## 8. 暂不作为第一阶段目标

1. 完整企业级 RPA 控制面
2. 多租户 secret 管理
3. 企业级联邦训练控制台
4. 通用跨平台桌面代理
5. 在线训练闭环直接改底座参数

## 9. 回归测试入口

桌面代理聚焦回归测试应使用：

```bash
python tests/run_desktop_agent_tests.py
```

---

# Desktop Agent Roadmap

## 1. Objective

Converge the current repository into a sustainably iterable Windows desktop agent system, prioritizing three things:

1. Understand the desktop visually
2. Actions must be verifiable
3. Failures are recoverable, reviewable, and regressible

The main product forms advance in three priority layers:

1. Standalone `computer-use` runtime
2. `chat_v2` interactive desktop agent
3. Automation worker with persistent workflows

Training, federation, and inference optimization are treated as supporting capabilities for now, not the primary narrative of the first phase.

## 2. Current Baseline

The repository already has the following foundations:

1. `system/computer_use_runtime.py`
Supports screenshots, grid overlay, OCR, window/control summary, multimodal decision-making, one-action-per-step, and trace persistence.

2. `system/chat_v2/agent.py`
Supports an interactive agent loop for browser/desktop actions, turning natural language requests into local action sequences.

3. `system/computer_plugins.py`
Already covers browser DOM, desktop clicks/inputs/hotkeys, Windows UIA control reading and manipulation, and other action surfaces.

4. `system/automation_runtime.py`
Supports persistent workflows, triggers, state, and run logs.

5. Test foundation
Existing tests: `tests/computer_use_runtime_test.py`, `tests/chat_v2_computer_agent_test.py`, `tests/computer_plugin_focus_test.py`, `tests/automation_runtime_test.py`.

## 3. Design Principles

1. Close the loop first, then broaden.
Prioritize keeping a set of common desktop tasks running stably, rather than prematurely aiming for "works with every application."

2. Verify first, then automate.
Prioritize adding observation, confirmation, replay, and recovery capabilities before expanding the number of actions.

3. Windows-first, then cross-platform.
Current desktop capabilities clearly depend on UIA, OCR, window focus, and local plugins; stabilize the Windows path first.

4. Real tasks first, then training optimization.
Accumulate task traces, failure samples, and recovery samples first, then decide which data is worth feeding into training or replay.

## 4. Phases

### P0 Stable Baseline

Goals:

1. `computer-use` dry-run and live-run both stably complete a minimal task loop.
2. Desktop focus tests are repeatably passing.
3. Trace, summary, and recovery checkpoint formats are stable.

Acceptance suggestions:

1. Pass `tests/computer_use_runtime_test.py`
2. Pass `tests/chat_v2_computer_agent_test.py`
3. Pass `tests/computer_plugin_focus_test.py`
4. At least one real application task is reproducible, e.g., QQ / Browser / Notepad

### P1 Single-Application Reliability

Goals:

1. Go deep on 2–3 priority applications first, rather than covering many horizontally.
2. Improve window matching, control matching, OCR fallback, and action verification.
3. Increase recovery success rate.

Recommended priority applications:

1. QQ or WeChat
2. Browser
3. File Explorer or Notepad

Key enhancements:

1. More stable window identification
2. Control-first action strategy with OCR fallback
3. Success determination for each step
4. Repair plan after failure

### P2 Interactive Product Entry

Goals:

1. Make `chat_v2` the primary human-machine interface.
2. Make natural language requests map more reliably to structured actions.
3. Use learned templates to reduce repeated planning costs.

Key enhancements:

1. Quality of natural language task decomposition
2. Learned template hit rate
3. Intermediate state summarization for long tasks
4. User-understandable failure reasons and next-step suggestions

### P3 Automation Accumulation

Goals:

1. Solidify stable actions and stable processes into automation workflows.
2. Enhance scheduled triggers, file triggers, and run state persistence.
3. Form a "desktop task library."

Key enhancements:

1. Workflow templating
2. Task replay and rerun
3. Automatic audit logging for failed tasks
4. Reusable workflows for common desktop tasks

## 5. Key Metrics

Prioritize tracking these metrics rather than pursuing complex model-parameter-level optimization first:

1. Task completion rate
2. Single-step verification pass rate
3. Recovery success rate
4. Average number of steps
5. Consistency between real tasks and dry-run results
6. Regression pass rate for priority applications

## 6. Directions Recommended for Reduction or De-prioritization

To make the desktop mainline move faster, temporarily de-prioritize these narratives:

1. Generic chat personality shaping
2. Heavy companion-style training data as a mainline focus
3. "Universal AI platform" narrative
4. Prematurely making federated training and desktop agent parallel mainlines

They are not necessarily to be deleted, but should be explicitly placed on a secondary track or experimental area.

## 7. Tasks That Can Land Immediately

If advancing one theme per week, the recommended order is:

1. Solidify the desktop baseline test entry and the mainline description in the README.
2. Add a more realistic regression test suite for QQ / Browser each.
3. Write the failure recovery strategy as a clearer state machine or document.
4. Add more human-readable summaries to `computer-use` traces.
5. Solidify stable tasks as example automation workflows.

## 8. Not First-Phase Goals

1. Full enterprise RPA control plane
2. Multi-tenant secret management
3. Enterprise federated training console
4. Universal cross-platform desktop agent
5. Closed-loop online training directly modifying base model parameters

## 9. Regression Entry

Desktop-agent focus regression should use:

```bash
python tests/run_desktop_agent_tests.py
```
