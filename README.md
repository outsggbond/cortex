# Offline Multimodal System

一个以本地运行和多能力编排为目标的 Python 项目，当前主入口已经收敛到 `main.py`，核心运行时位于 `system/`。

目前实际可用的主路径：

- `--chat`：对话运行时，支持 OpenAI 兼容接口，包括 DeepSeek。
- `--computer-use-goal`：桌面任务执行。
- `--automation-*`：自动化工作流执行。
- `--rag-ingest`：知识库索引构建。

## 目录概览

```text
.
|-- main.py
|-- system/                  # 运行时、领域服务、接口适配层
|-- config/                  # 配置文件
|-- scripts/                 # 辅助脚本
|-- tests/                   # 测试与脚本化验证
|-- artifacts/               # 运行产物
`-- README.md
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

如果要跑验证和测试，再补装开发依赖：

```powershell
pip install -e ".[dev]"
```

## DeepSeek 配置

推荐在项目根目录放一个 `.env.local`：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
V2_LLM_PROVIDER=deepseek
V2_LLM_MODEL=deepseek-v4-flash
```

项目启动时会自动读取：

- `.env.local`
- `config/runtime_secrets.env`

## DeepSeek 运行

交互式聊天：

```powershell
.\.venv\Scripts\python.exe main.py `
  --chat `
  --cloud-llm `
  --cloud-provider deepseek `
  --cloud-model deepseek-v4-flash `
  --rag-disable
```

如果已经在 `.env.local` 里设置了 `V2_LLM_PROVIDER=deepseek` 和 `V2_LLM_MODEL=deepseek-v4-flash`，命令可以再简化。

## DeepSeek 验证

先做配置检查，不发请求：

```powershell
.\.venv\Scripts\python.exe scripts\verify_deepseek.py --config-only
```

再做真实 API 调用验证：

```powershell
.\.venv\Scripts\python.exe scripts\verify_deepseek.py
```

脚本会输出：

- 解析后的 `provider` / `model` / `endpoint`
- 是否检测到 API Key
- 真实网络调用结果
- 常见错误提示，例如 `401` 无效密钥、`402` 余额不足

## 本地验证

仓库里有几组可以直接运行的轻量验证脚本：

```powershell
.\.venv\Scripts\python.exe tests\chat_v2_cloud_test.py
.\.venv\Scripts\python.exe tests\runtime_bootstrap_local_env_test.py
.\.venv\Scripts\python.exe tests\chat_v2_runtime_test.py
.\.venv\Scripts\python.exe tests\interfaces_entrypoints_test.py
```

如果已安装 `pytest`，也可以直接跑：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\chat_v2_cloud_test.py tests\runtime_bootstrap_local_env_test.py tests\chat_v2_runtime_test.py tests\interfaces_entrypoints_test.py -q
```



# 可以增强的部分

## L1 基础设施层

### `system/core/`

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `hardware.py` | CPU/RAM/GPU 检测 | USB 设备枚举、磁盘健康、电池状态、温度传感器、外接显示器检测 |
| `embeddings.py` | 文本嵌入向量生成 | 多语言嵌入、图像嵌入、CLIP 多模态嵌入、本地小模型替代 |
| `model_loader.py` | Transformer 模型加载 | 支持 ONNX、TensorRT 加速、量化模型加载、LoRA 热插拔 |
| `state_encoder.py` | 状态编码 | 增加时序状态编码（历史状态序列）、图状态编码 |
| `planner.py` | 规划原语 | 分层规划、并行子任务、条件分支规划、规划回退策略 |
| `error_normalizer.py` | 错误规范化 | 错误分类（可恢复/不可恢复/需人工）、错误趋势分析 |

### `system/runtime/`

| 文件                     | 当前能力                     | 可增强方向 |
|------                    |---------                    |-----------|
| `bootstrap.py` | 启动配置 | 插件化启动、按需加载、启动性能优化 |
| `support.py` | 环境变量覆盖 | 配置文件热加载、远程配置拉取 |
| `artifacts.py` | 制品维护/清理 | 制品版本管理、制品完整性校验、云存储同步 |
| `adaptive.py` | 硬件自适应控制器 | 多维度自适应（网络带宽、磁盘IO、电量）、策略预学习 |
| `task_router.py` | 任务类型路由 | 动态路由（根据负载选域）、优先级队列、任务依赖关系 |
| `execution_engine.py` | 统一任务执行 | 异步任务执行、任务暂停/恢复、任务超时控制 |
| `state_manager.py` | 任务状态持久化 | 父子任务关系、任务血缘追踪、状态快照对比 |

### `system/l_utils/`没有增强的----------------------------------------------------------------------------------------------

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `loader.py` | 模块加载 | 插件发现机制、热加载/热卸载 |
| `codegen.py` | 代码生成 | 模板引擎升级、多语言代码生成 |
| `files.py` | 文件操作 | 大文件流处理、文件变更监听、哈希校验 |---------------------------这个开始
| `metrics.py` | 指标统计 | Prometheus 指标暴露、自定义仪表盘数据 |
| `cli_support.py` | CLI 工具 | 自动补全、交互式向导、彩色输出美化 |
| `graphviz_export.py` | 图形导出 | 实时依赖图可视化、状态机图导出 |

---

## L2 感知与知识层

### `system/perception/`

| 当前能力 | 可增强方向 |
|---------|-----------|
| 视觉预处理、OCR 对齐、UI 树融合 | 接入 YOLO/SAM 做像素级 UI 元素分割 |
| | 接入 Grounding DINO 做自然语言定位（“红色按钮”） |
| | 视频流理解（录屏回放分析） |
| | 多模态 VLM 本地部署做端侧视觉理解 |
| | 跨应用 UI 模式识别（识别“这是一个登录表单”） |

### `system/knowledge/`

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `rag_store.py` | RAG 索引检索 | 混合检索（稀疏+稠密）、多模态 RAG（搜图）、对话式检索、增量索引更新 |
| `graph/kb.py` | 知识库 | 自动知识抽取（从操作记录中提取流程）、时序知识图谱 |
| `graph/query.py` | 查询引擎 | 自然语言转图查询、图推理（多跳推理） |
| `graph/repl.py` | 交互式 REPL | 可视化查询结果、查询历史记录 |
| `semantic/index.py` | 混合语义索引 | 多语言语义对齐、跨模态语义索引 |
| `semantic/faq.py` | FAQ 检索 | 主动 FAQ 生成（从日志中提取常见问题）、FAQ 自动更新 |
| `memory.py` | 内存与向量数据库 | 长期记忆/短期记忆分离、记忆遗忘曲线、记忆优先级排序 |

---

## L3 智能推理层

### `system/brain/`

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `reasoning.py` | 层次化推理 | Chain-of-Thought 可视化、多步推理回溯、推理过程解释生成 |
| `neural_reasoner.py` | 神经推理 | 接入更强本地模型、推理与检索混合 |
| `debate_reasoner.py` | 辩论式推理 | 多模型辩论（不同模型互相质疑）、辩论结果置信度 |
| `neurosymbolic.py` | 神经符号混合 | 可解释规则提取、规则库可视化编辑 |
| `profile.py` | 推理配置 | 按场景自动切换配置、A/B 测试框架 |
| `world_model.py` | 世界状态建模 | 跨设备世界模型、时序预测（下一步状态是什么） |
| `world_predictor.py` | 状态预测 | 多步预测、预测不确定性量化 |
| `stimulus_engine.py` | 刺激与反应 | 主动触发（不是被动等指令）、上下文感知推送 |

### `system/learning/`

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `continual.py` | 持续学习框架 | 在线学习（边用边学）、灾难性遗忘防护 |
| `feedback.py` | 反馈处理 | 隐式反馈提取（从操作结果反推好坏）、反馈加权 |
| `loader.py` | 训练数据加载 | 流式加载、多格式支持（Parquet、Arrow） |
| `dialogue/` 系列 | 对话生成/增强/演进 | 多模态对话数据（含图）、角色一致性约束 |

### `system/strategy/`

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `strategy.py` | 策略生成 | 多策略比较、策略模拟评估 |
| `task_flow.py` | 任务流程 | 可视化流程编辑器、流程模板市场 |
| `search_planner.py` | 搜索规划 | 蒙特卡洛树搜索、A* 规划、规划缓存复用 |

---

## L4 执行与应用层

### `system/chat_v2/`

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `intents.py` | 意图分类 | 多语言意图、复合意图分解、意图置信度阈值可配 |
| `llm.py` | LLM 适配层 | 支持更多模型（Claude、Gemini、本地模型）、模型热切换、流式输出 |
| `memory.py` | 只读检索 | 多轮对话上下文管理、会话摘要 |
| `pipeline.py` | 单路径流水线 | 可插拔流水线节点、自定义流水线编排 |
| `runtime.py` | 聊天循环 | 多会话管理、会话恢复、多模态输入（语音、图片） |

### `system/computer_use/`

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `runtime.py` | 桌面智能体主循环 | 多窗口并行操作、操作速度可配、批量操作 |
| `llm.py` | 决策 LLM | 本地模型替代云端、多模型投票决策 |
| `plugins.py` | 桌面插件 | 更多应用插件（Office、企业微信、钉钉）、插件热加载 |
| `action/generator.py` | 动作生成 | 动作模板学习、复合动作（拖拽+键盘） |
| `action/validator.py` | 动作验证 | 视觉对比验证、预期状态断言 |
| `action/stats.py` | 动作统计 | 成功率分应用统计、耗时统计 |
| `trace/audit.py` | 审计日志 | 审计日志结构化查询、异常检测 |
| `trace/visual.py` | 轨迹可视化 | 操作回放、步骤对比 |

### `system/automation/`

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `runtime.py` | 工作流运行时 | 定时触发升级为 cron 表达式、事件触发（文件变更/网络事件） |
| `executor.py` | 步骤执行 | 并行步骤、条件分支、循环步骤、子工作流调用 |

### `system/evaluation/`

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `evaluator.py` | 性能评估 | 多维度评分（速度/准确率/资源消耗）、趋势报告 |
| `reflect.py` | 自我反思 | 失败模式自动聚类、反思知识积累 |
| `self_heal.py` | 自我修复 | 自动回滚配置、自动切换备用策略 |
| `response_guard.py` | 响应守卫 | 内容安全扩展（多语言）、敏感信息过滤 |

---

## 统一抽象层

### `system/agent/`

| 文件 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `task.py` | 任务模型 | 任务优先级、截止时间、任务标签、自定义任务属性 |
| `planner.py` | 规划门面 | 规划串行化（可保存/恢复规划）、规划版本管理 |
| `executor.py` | 执行器门面 | 执行器插件机制、执行器性能监控 |
| `state.py` | 状态模型 | 状态历史链、状态变更事件、状态校验 |

---

## 接口层

### `system/interfaces/`

| 目录 | 当前能力 | 可增强方向 |
|------|---------|-----------|
| `cli/` | 命令行参数注册 | 交互式 CLI、脚本模式（批量执行）、输出格式化（JSON/表格/纯文本） |
| `api/` | HTTP API 适配器 | RESTful API 完善、WebSocket 实时推送、OpenAPI 文档自动生成 |
| `gui/` | GUI 会话适配 | Web UI、桌面托盘应用、移动端控制面板 |

---

## 工程体系

| 能力 | 可增强方向 |
|------|-----------|
| 测试 | 端到端集成测试、性能基准测试、混沌测试（随机注入故障） |
| CI/CD | 自动发布、版本号管理、changelog 自动生成 |
| 监控 | 运行时指标面板、告警规则、操作回放审计 |
| 安全 | 操作权限分级、敏感操作二次确认、操作沙箱 |

---

## 你之前提的方向

| 你想要的 | 对应增强位置 |
|---------|------------|
| 深入操作系统 | L1 `core/hardware.py` + 新建 L4 `system/system_control/` |
| USB 检测手机 | L1 `core/hardware.py` 扩展 USB 枚举 |
| 操作手机 | 新建 L4 `system/phone_use/`，对标 `computer_use/` |
| 跨设备协同 | L3 `agent/` 父子任务 + `state_manager.py` 血缘追踪 |
| MCP 协议接入 | L2 `knowledge/` 或新建 L4 适配模块 |
| 本地模型替代 | L3 `brain/reasoning.py` + L1 `model_loader.py` |

---


# CLAUDE.md — OmniEvolve

A **local-first, self-evolving, multi-modal AI agent system** — runs entirely on user hardware, handles text/images/audio, and continuously improves through online learning.

## Project Identity

| Line | Module | Purpose |
|------|--------|---------|
| Chat | `system/chat_v2/` | Conversational AI: intent routing, memory, RAG, reasoning |
| Desktop | `system/computer_use/` | Windows automation: Win32 SendInput, OCR, pywinauto, plugins |
| Workflow | `system/automation/` | Trigger-driven execution + NanoBrain + federated learning |
| Knowledge | `system/knowledge/` | RAG, hybrid semantic index, Neo4j graph KB, FAQ retrieval |
| Learning | `system/learning/` | Online learning loop, dialogue evolution, continual learning |

**~66,000 lines of Python**, 273 modules, layered architecture, zero dead code.

## Quick Start

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-minimal.txt

# Chat
.\.venv\Scripts\python.exe main.py --chat --cloud-llm --cloud-provider deepseek --cloud-model deepseek-v4-flash

# Tests
.\.venv\Scripts\python.exe -m pytest tests/ -o "addopts=" -q
```

## Architecture

```
main.py (bootstrap)
  └─ system/app_runtime.py          # Unified entry: parse → route → execute
       ├─ system/runtime/             # TaskRouter, ExecutionEngine, StateManager, AdaptiveController
       ├─ system/domain/              # Business facades (chat/computer/automation)
       ├─ system/agent/               # Unified Task/Planner/Executor/State models
       ├─ system/interfaces/          # CLI/API/GUI adapters
       ├─ system/chat_v2/             # Chat with LearningLoop integration
       ├─ system/computer_use/        # Desktop automation + plugin registry
       ├─ system/automation/          # Workflow + federated learning
       ├─ system/knowledge/           # RAG, rules, semantic index, FAQ
       ├─ system/learning/            # Online learning loop, dialogue evolution
       ├─ system/evaluation/          # Adapter artifacts, evolution, self-heal
       ├─ system/brain/               # Reasoning, world model, persona, distiller
       ├─ system/core/                # Hardware, planner, embeddings, auto-tune
       ├─ system/strategy/            # Search planner, strategy config
       ├─ system/perception/          # Multi-modal encoders, reasoning
       ├─ system/agent/               # Unified abstraction layer
       └─ system/l_utils/             # CLI support, codegen, files, metrics
```

**Layered design**: L1 infrastructure → L2 intelligence/knowledge → L3 business → L4 interfaces.

## Key Modules

| Module | Lines | Role |
|--------|-------|------|
| `computer_use/runtime.py` | 1384 | Desktop agent: observation, action dispatch, plugin orchestration |
| `evaluation/adapter_artifacts.py` | 1347 | LoRA adapter lifecycle management |
| `chat_v2/agent.py` | 1220 | Workspace agent: plan → execute → self-heal → learn |
| `automation/nanobrain.py` | 1055 | Cognitive graph: PMI-based logic/emotion graphs, A* path |
| `computer_use/plugins.py` | 887 | Windows desktop automation (SendInput, OCR, pywinauto, 16 plugins) |
| `chat_v2/reasoner.py` | 738 | Hierarchical reasoner interface |
| `brain/reasoning.py` | 695 | Hierarchical reasoner + profile + chains + feedback |
| `chat_v2/pipeline.py` | 595 | Full chat pipeline with LearningLoop + FAQ integration |
| `automation/executor.py` | 579 | Step executor: file ops, scripts, plugins, retry, diff |
| `knowledge/semantic/refiner.py` | 564 | Semantic index refiner |
| `learning/learning_loop.py` | 470 | Online self-evolution: observe → pattern detect → trigger learning |

## Self-Evolution Pipeline

```
user_input → Pipeline.respond()
  ├─ classify_intent → classify_route → _decide
  │    ├─ agent/reasoning/LLM/memory/RAG/FAQ/clarify/fallback
  │    └─ _post_validate_candidate (evidence + thread context check)
  └─ LearningLoop.observe(event)
       ├─ accumulate events per (route:validation:query_pattern)
       ├─ detect emerging patterns
       └─ maybe_learn() when threshold met
            ├─ RuleFeedbackMemory.update()       → better rule priors
            ├─ ExperienceStore.add()              → episodes for RAG
            ├─ EvolutionEngine.analyze_failures() → strategy proposals
            ├─ EWCMemory.record()                 → continual learning
            └─ DialogueFailureReflector           → prompt improvements
```

## Import Rules

All imports use the **full layered path**. Flat shims deleted.

| Old (removed) | Correct |
|---------------|---------|
| `system.hardware` | `system.core.hardware` |
| `system.planner` | `system.core.planner` |
| `system.reasoning` | `system.brain.reasoning` |
| `system.executor` | `system.automation.executor` |
| `system.embeddings` | `system.core.embeddings` |
| `system.runtime_bootstrap` | `system.runtime.bootstrap` |

## Test Status

- **57+ passing** (runtime, chat_v2, adaptive, reasoning, interfaces, validation)
- **1 xfail** (bandit uncertainty guard)
- ~40 tests skipped (need torch, playwright, pywinauto, network, desktop)

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -o "addopts=" -q
```

## Configuration

- `.env.local` — API keys (auto-loaded on startup)
- `config/` — YAML/JSON: `auto_tune.json`, `policies.yaml`, `costs.yaml`, `contexts.yaml`
- `config/policy_manager.py` — Runtime policy resolution
- `config/system_config.py` — SystemConfig dataclass

## Conventions

1. **`main.py`** — bootstrap only, no business logic.
2. **Domain facades** — stable entry points; implementations in subdirectories.
3. **`system/` root** — only `app_runtime.py`, `main_cli.py`, `services.py`.
4. **LLM clients** — extend `BaseLLMClient` from `system/chat_v2/llm.py`.
5. **Every module connected** — no dead code; each file imported by ≥1 other module or is an entry point.
6. **Optional deps** — torch, neo4j, playwright, pywinauto use try/except guards.
7. **Test flags** — `--maxfail=1` in pyproject.toml; override with `-o "addopts="`.
