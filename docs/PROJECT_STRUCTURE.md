# 项目结构与落位约定

这份文档只回答一件事：后续开发时，看到目录就知道“功能放哪里、逻辑在哪层、应该改哪一段”。

## 一句话原则

- `main.py` 只做 bootstrap，并通过 `system/interfaces/cli/` 转发 CLI 入口。
- `system/app_runtime.py` 只做运行时 orchestration。
- `system/main_cli.py` 只做 CLI 组装。
- `system/interfaces/cli/` 按域拆 CLI 参数注册。
- `system/interfaces/api/` 放程序化/API 适配器。
- `system/interfaces/gui/` 放 GUI 会话适配器。
- `system/runtime/` 管统一路由、执行、状态等跨域基础设施。
- `system/domain/` 管业务域门面，不复制实现。
- `system/chat_v2/`、`system/computer_use/`、`system/automation/` 管具体业务实现。
- `system/agent/` 管统一 `Task / Planner / Executor / State` 抽象。
- `config/` 放配置，`artifacts/` 放产物，`tests/` 放回归。

## 推荐分层

### 1. 启动与运行时层

- `main.py`
  统一启动入口。
- `system/app_runtime.py`
  统一 runtime 入口。
- `system/main_cli.py`
  参数解析层。
- `system/interfaces/cli/`
  按 `chat / computer_use / automation / artifact / federated` 分域注册参数，并承接 CLI 根入口。
- `system/interfaces/api/`
  程序化入口适配层，供 HTTP/API handler 复用；当前已挂 `chat / computer / automation`。
- `system/interfaces/gui/`
  GUI 会话适配层，供桌面/Web 壳复用；当前已挂 `chat / computer / automation`。
- `system/runtime/`
  `TaskRouter`、`RuntimeExecutionEngine`、`RuntimeStateManager`、bootstrap、artifact 维护等。

### 2. 业务域门面层

- `system/domain/chat/`
- `system/domain/computer/`
- `system/domain/automation/`

这层的职责是稳定暴露业务域入口，避免根入口直接依赖具体实现目录。

### 3. 统一 Agent 抽象层

- `system/agent/task.py`
- `system/agent/planner.py`
- `system/agent/executor.py`
- `system/agent/state.py`

这层解决统一建模问题，让 chat、desktop、automation 逐步共享任务与执行语义。

### 4. 业务实现层

- `system/chat_v2/`
  对话运行时、意图识别、记忆、workspace agent。
- `system/computer_use/`
  桌面观察、动作规划、动作执行、trace、恢复。
- `system/automation/`
  workflow 配置解析、触发器、执行器编排。

### 5. 能力与支撑层

- `system/core/`
  真正跨域复用的基础能力。
- `system/brain/`
  world model、reasoning、predictor 等。
- `system/knowledge/`
  RAG、规则、知识图谱。
- `system/evaluation/`
  评估、反馈、自修复、审计。
- `system/learning/`
  数据构建、持续学习。

## 顶层目录建议

```text
.
|-- main.py
|-- system/
|   |-- app_runtime.py
|   |-- main_cli.py
|   |-- interfaces/
|   |   |-- cli/
|   |   |-- api/
|   |   `-- gui/
|   |-- runtime/
|   |-- domain/
|   |-- agent/
|   |-- chat_v2/
|   |-- computer_use/
|   |-- automation/
|   |-- core/
|   |-- brain/
|   |-- knowledge/
|   |-- evaluation/
|   `-- learning/
|-- config/
|-- scripts/
|-- docs/
|-- tests/
|-- artifacts/
|-- checkpoints/
`-- utils/
```

## 放置规则

- 新增 CLI 路由或运行时编排：放 `system/runtime/`
- 新增 CLI 参数定义：放 `system/interfaces/cli/`
- 新增 API 适配器：放 `system/interfaces/api/`
- 新增 GUI 会话适配器：放 `system/interfaces/gui/`
- 新增统一任务或执行抽象：放 `system/agent/`
- 新增域公开入口：放 `system/domain/`
- 新增聊天逻辑：放 `system/chat_v2/`
- 新增桌面代理逻辑：放 `system/computer_use/`
- 新增 workflow 或触发器：放 `system/automation/`
- 新增跨域基础能力：放 `system/core/` 或 `system/brain/`

如果一个模块只服务于某一条主线，就不要塞进 `system/core/`。

## 不推荐继续扩散的做法

- 在 `main.py` 或 `system/app_runtime.py` 写具体业务逻辑。
- 在 `config/` 写计算逻辑或副作用逻辑。
- 在 `utils/` 堆高耦合业务函数，最后变成杂物间。
- 让业务实现反向依赖根入口。

## 兼容层约定

为了避免迁移期直接打断旧测试或旧脚本，当前允许保留薄兼容层：

- `system/runtime_bootstrap.py`
- `system/automation_runtime.py`
- `system/planner.py`
- `system/executor.py`

兼容层只做转发，不复制实现。

## 建议阅读顺序

### 想看统一运行时

- `system/app_runtime.py`
- `system/main_cli.py`
- `system/interfaces/cli/`
- `system/interfaces/api/`
- `system/interfaces/gui/`
- `system/runtime/task_router.py`
- `system/runtime/execution_engine.py`
- `system/runtime/state_manager.py`

### 想看聊天主线

- `system/domain/chat/`
- `system/chat_v2/runtime.py`
- `system/chat_v2/pipeline.py`
- `system/interfaces/api/chat.py`
- `system/interfaces/gui/chat.py`

### 想看桌面主线

- `system/domain/computer/`
- `system/computer_use/runtime.py`
- `system/interfaces/api/computer.py`
- `system/interfaces/gui/computer.py`
- `system/computer_use/action/`
- `system/computer_use/trace/`

### 想看自动化主线

- `system/domain/automation/`
- `system/automation/runtime.py`
- `system/interfaces/api/automation.py`
- `system/interfaces/gui/automation.py`
- `system/automation/executor.py`
- `config/automation_workflows.json`

## 判断结构是否清晰

如果后续开发者能快速回答下面几个问题，这套结构就算站住了：

- 入口在哪里？
- CLI、API、GUI 各自的适配层在哪里？
- 统一 runtime 在哪里？
- 业务域门面在哪里？
- 具体业务实现在哪里？
- 新功能该落哪一层？
- 测试和运行产物分别在哪里？

---

# Project Structure and Placement Conventions

This document answers a single question: when developing further, by looking at the directory you know "where functionality goes, which layer logic belongs to, and which section to modify."

## One-Sentence Principles

- `main.py` only does bootstrapping and forwards CLI entry via `system/interfaces/cli/`.
- `system/app_runtime.py` only does runtime orchestration.
- `system/main_cli.py` only does CLI assembly.
- `system/interfaces/cli/` registers CLI arguments by domain.
- `system/interfaces/api/` holds programmatic/API adapters.
- `system/interfaces/gui/` holds GUI session adapters.
- `system/runtime/` manages cross-cutting infrastructure like unified routing, execution, and state.
- `system/domain/` manages business domain facades without duplicating implementations.
- `system/chat_v2/`, `system/computer_use/`, `system/automation/` manage concrete business implementations.
- `system/agent/` manages unified `Task / Planner / Executor / State` abstractions.
- `config/` holds configuration, `artifacts/` holds outputs, `tests/` holds regression tests.

## Recommended Layers

### 1. Startup and Runtime Layer

- `main.py`
  Unified startup entry point.
- `system/app_runtime.py`
  Unified runtime entry point.
- `system/main_cli.py`
  Argument parsing layer.
- `system/interfaces/cli/`
  Registers parameters by domain (`chat / computer_use / automation / artifact / federated`) and serves as the CLI root entry.
- `system/interfaces/api/`
  Programmatic entry adapter layer for reuse by HTTP/API handlers; currently supports `chat / computer / automation`.
- `system/interfaces/gui/`
  GUI session adapter layer for reuse by desktop/web shells; currently supports `chat / computer / automation`.
- `system/runtime/`
  `TaskRouter`, `RuntimeExecutionEngine`, `RuntimeStateManager`, bootstrap, artifact maintenance, etc.

### 2. Business Domain Facade Layer

- `system/domain/chat/`
- `system/domain/computer/`
- `system/domain/automation/`

This layer's responsibility is to stably expose business domain entry points, preventing root entries from directly depending on concrete implementation directories.

### 3. Unified Agent Abstraction Layer

- `system/agent/task.py`
- `system/agent/planner.py`
- `system/agent/executor.py`
- `system/agent/state.py`

This layer solves the unified modeling problem, allowing chat, desktop, and automation to gradually share task and execution semantics.

### 4. Business Implementation Layer

- `system/chat_v2/`
  Dialogue runtime, intent recognition, memory, workspace agent.
- `system/computer_use/`
  Desktop observation, action planning, action execution, trace, recovery.
- `system/automation/`
  Workflow configuration parsing, triggers, executor orchestration.

### 5. Capabilities and Support Layer

- `system/core/`
  Truly cross-domain reusable foundational capabilities.
- `system/brain/`
  World model, reasoning, predictor, etc.
- `system/knowledge/`
  RAG, rules, knowledge graph.
- `system/evaluation/`
  Evaluation, feedback, self-healing, audit.
- `system/learning/`
  Data construction, continual learning.

## Recommended Top-Level Directory Layout

```text
.
|-- main.py
|-- system/
|   |-- app_runtime.py
|   |-- main_cli.py
|   |-- interfaces/
|   |   |-- cli/
|   |   |-- api/
|   |   `-- gui/
|   |-- runtime/
|   |-- domain/
|   |-- agent/
|   |-- chat_v2/
|   |-- computer_use/
|   |-- automation/
|   |-- core/
|   |-- brain/
|   |-- knowledge/
|   |-- evaluation/
|   `-- learning/
|-- config/
|-- scripts/
|-- docs/
|-- tests/
|-- artifacts/
|-- checkpoints/
`-- utils/
```

## Placement Rules

- New CLI routing or runtime orchestration: place in `system/runtime/`
- New CLI argument definitions: place in `system/interfaces/cli/`
- New API adapters: place in `system/interfaces/api/`
- New GUI session adapters: place in `system/interfaces/gui/`
- New unified task or execution abstractions: place in `system/agent/`
- New domain public entry points: place in `system/domain/`
- New chat logic: place in `system/chat_v2/`
- New desktop agent logic: place in `system/computer_use/`
- New workflows or triggers: place in `system/automation/`
- New cross-domain foundational capabilities: place in `system/core/` or `system/brain/`

If a module only serves one mainline, do not put it into `system/core/`.

## Practices to Avoid Spreading Further

- Writing concrete business logic in `main.py` or `system/app_runtime.py`.
- Writing computation logic or side-effectful logic in `config/`.
- Piling highly coupled business functions into `utils/`, turning it into a junk drawer.
- Making business implementations reverse-depend on root entry points.

## Compatibility Layer Convention

To avoid disrupting old tests or scripts during migration, thin compatibility layers are currently allowed to remain:

- `system/runtime_bootstrap.py`
- `system/automation_runtime.py`
- `system/planner.py`
- `system/executor.py`

Compatibility layers only do forwarding and do not duplicate implementations.

## Suggested Reading Order

### For the Unified Runtime

- `system/app_runtime.py`
- `system/main_cli.py`
- `system/interfaces/cli/`
- `system/interfaces/api/`
- `system/interfaces/gui/`
- `system/runtime/task_router.py`
- `system/runtime/execution_engine.py`
- `system/runtime/state_manager.py`

### For the Chat Mainline

- `system/domain/chat/`
- `system/chat_v2/runtime.py`
- `system/chat_v2/pipeline.py`
- `system/interfaces/api/chat.py`
- `system/interfaces/gui/chat.py`

### For the Desktop Mainline

- `system/domain/computer/`
- `system/computer_use/runtime.py`
- `system/interfaces/api/computer.py`
- `system/interfaces/gui/computer.py`
- `system/computer_use/action/`
- `system/computer_use/trace/`

### For the Automation Mainline

- `system/domain/automation/`
- `system/automation/runtime.py`
- `system/interfaces/api/automation.py`
- `system/interfaces/gui/automation.py`
- `system/automation/executor.py`
- `config/automation_workflows.json`

## Determining if the Structure is Clear

If future developers can quickly answer the following questions, the structure can be considered solid:

- Where is the entry point?
- Where are the adapter layers for CLI, API, and GUI?
- Where is the unified runtime?
- Where are the business domain facades?
- Where are the concrete business implementations?
- Which layer should a new feature be placed in?
- Where are tests and runtime artifacts located, respectively?
