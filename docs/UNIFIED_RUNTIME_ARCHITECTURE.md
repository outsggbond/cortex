# 统一运行时架构

## 目标

把仓库从"3 个并列系统"收口成"1 个统一 AI runtime + 3 个业务域实现"。

这次调整的重点不是物理大搬家，而是先补齐统一抽象，并用薄兼容层挂回现有实现：

- 不推翻 `chat_v2 / computer_use / automation`
- 不把旧逻辑直接塞进新目录
- 先把入口、路由、执行、状态建成稳定 API

## 核心抽象

### 1. Task

统一入口任务模型：

- 定义：`system/agent/task.py`
- 路由：`system/runtime/task_router.py`

当前覆盖的任务类型：

- `chat`
- `computer`
- `automation`
- `knowledge`
- `training`

### 2. Planner

统一规划门面：

- 门面：`system/agent/planner.py`
- 当前实现：`system/core/planner.py`

这意味着 chat agent、automation workflow、未来的 desktop subtask decomposition 都可以逐步收口到一套 planner API。

### 3. Executor

统一执行分成两层：

- 运行时级别分发：`system/runtime/execution_engine.py`
- 计划级别执行：`system/agent/executor.py` -> `system/automation/executor.py`

前者解决"任务该交给哪个业务域"；后者解决"计划里的 step 如何执行"。

### 4. State

统一状态模型和落盘：

- 状态模型：`system/agent/state.py`
- 持久化：`system/runtime/state_manager.py`

默认输出目录：

```text
artifacts/runtime/tasks/
```

每次从 CLI 进入的任务都会生成状态文件，记录：

- task id
- task type
- intent
- goal
- entrypoint
- status
- result / last_error

## 分层

### Runtime Layer

- `main.py`
  只做 bootstrap。
- `system/interfaces/cli/`
  CLI 根入口与分域参数注册层。
- `system/app_runtime.py`
  统一 runtime 入口。
- `system/main_cli.py`
  薄 CLI 参数收口层。
- `system/interfaces/api/`
  程序化/API 适配层，承接非 CLI 的 `chat / computer / automation` 入口。
- `system/interfaces/gui/`
  GUI 会话适配层，承接桌面/Web 壳的人机入口。
- `system/runtime/*`
  任务路由、执行调度、状态落盘、artifact 维护等跨域基础设施。

### Domain Layer

- `system/domain/chat`
- `system/domain/computer`
- `system/domain/automation`

这层的作用是把业务域作为稳定接口暴露出去，而不是让入口直接依赖具体实现目录。

### Domain Implementations

- `system/chat_v2`
- `system/computer_use`
- `system/automation`

现有实现仍然留在这里，避免一次性迁移带来的回归风险。

### Agent Layer

- `system/agent/task.py`
- `system/agent/planner.py`
- `system/agent/executor.py`
- `system/agent/state.py`

这层负责统一任务、规划、执行、状态抽象，是后续继续收口 chat / computer / automation 的桥。

### Capability Layer

- `system/core`
- `system/brain`
- `system/knowledge`

这层放共享能力，而不是业务入口。

## 运行时流程

当前 CLI 主路径：

```text
main.py
  -> system.interfaces.cli.run_cli_entrypoint()
  -> system.app_runtime.main()
  -> TaskRouter.route(args)
  -> RuntimeExecutionEngine.execute(task, args)
  -> system.domain.<domain>
  -> existing implementation
  -> RuntimeStateManager persists task state
```

当前非 CLI 入口建议路径：

```text
API / GUI shell
  -> system.interfaces.api or system.interfaces.gui
  -> runtime/domain builder helpers
  -> chat/computer/automation implementation
```

## 为什么这很重要

这套结构解决了之前最容易继续发散的几个问题：

### 入口不再直接绑死业务模块

以前 `app_runtime.py` 直接按 if/else 跳到各主线；
现在先变成统一 `Task`，再交给执行引擎。

### 业务域从"入口"降级为"实现"

`chat_v2 / computer_use / automation` 现在更明确地处在实现层，而不是系统根入口。

### 执行状态可追踪

统一状态文件让 CLI 级任务有了最基本的可观测性，这对后续做 audit、resume、replay、dashboard 都是基础。

### API / GUI 也开始走统一接口层

`system/interfaces/api` 和 `system/interfaces/gui` 现在已经覆盖 chat、computer、automation，并直接复用 runtime builder，而不是各自复制一套装配逻辑。

## 迁移策略

当前采用的是"兼容优先"的迁移方式：

1. 新增统一抽象和门面层。
2. 保留老实现目录。
3. 让入口先依赖新抽象。
4. 后续再逐步把内部逻辑收口。

换句话说，这次不是重写，而是为后续重构创造稳定支点。

## 推荐后续步骤

下一阶段最值得继续推进的方向：

1. 让 `WorkspaceAgent` 和 `AutomationRuntime` 统一消费 `system/agent/planner.py`。
2. 把 computer-use 的动作决策逐步抽成可复用 `Action` schema。
3. 给 `RuntimeStateManager` 增加 resume / lineage / parent-child task 关系。
4. 继续把 `system/interfaces/api` / `system/interfaces/gui` 挂到更多非主线入口或服务壳。
5. 给 `system/domain/*`、`system/agent/*`、`system/interfaces/*` 补模块级 README，降低后续认知成本。

---

# Unified Runtime Architecture

## Goal

Converge the repository from "3 parallel systems" into "1 unified AI runtime + 3 business domain implementations".

The focus of this adjustment is not a physical move, but first filling in unified abstractions and using thin compatibility layers to hang back to existing implementations:

- Do not overturn `chat_v2 / computer_use / automation`
- Do not directly stuff old logic into new directories
- First build stable APIs for entry, routing, execution, and state

## Core Abstractions

### 1. Task

Unified entry task model:

- Definition: `system/agent/task.py`
- Routing: `system/runtime/task_router.py`

Currently covered task types:

- `chat`
- `computer`
- `automation`
- `knowledge`
- `training`

### 2. Planner

Unified planning facade:

- Facade: `system/agent/planner.py`
- Current implementation: `system/core/planner.py`

This means the chat agent, automation workflow, and future desktop subtask decomposition can all gradually converge on a single set of planner APIs.

### 3. Executor

Unified execution is split into two layers:

- Runtime-level dispatch: `system/runtime/execution_engine.py`
- Plan-level execution: `system/agent/executor.py` -> `system/automation/executor.py`

The former solves "which business domain should this task be handed to"; the latter solves "how the steps in the plan are executed".

### 4. State

Unified state model and persistence:

- State model: `system/agent/state.py`
- Persistence: `system/runtime/state_manager.py`

Default output directory:

```text
artifacts/runtime/tasks/
```

Every task entered from the CLI generates a state file recording:

- task id
- task type
- intent
- goal
- entrypoint
- status
- result / last_error

## Layering

### Runtime Layer

- `main.py`
  Only does bootstrapping.
- `system/interfaces/cli/`
  CLI root entry and per-domain parameter registration layer.
- `system/app_runtime.py`
  Unified runtime entry.
- `system/main_cli.py`
  Thin CLI argument collecting layer.
- `system/interfaces/api/`
  Programmatic/API adapter layer, handling non-CLI `chat / computer / automation` entries.
- `system/interfaces/gui/`
  GUI session adapter layer, handling human-machine entry for desktop/web shells.
- `system/runtime/*`
  Cross-cutting infrastructure: task routing, execution scheduling, state persistence, artifact maintenance, etc.

### Domain Layer

- `system/domain/chat`
- `system/domain/computer`
- `system/domain/automation`

This layer's role is to expose business domains as stable interfaces, rather than letting entries directly depend on concrete implementation directories.

### Domain Implementations

- `system/chat_v2`
- `system/computer_use`
- `system/automation`

Existing implementations remain here to avoid regression risks from a one-shot migration.

### Agent Layer

- `system/agent/task.py`
- `system/agent/planner.py`
- `system/agent/executor.py`
- `system/agent/state.py`

This layer is responsible for unified task, planning, execution, and state abstractions, serving as the bridge for further converging chat / computer / automation.

### Capability Layer

- `system/core`
- `system/brain`
- `system/knowledge`

This layer holds shared capabilities rather than business entries.

## Runtime Flow

Current CLI main path:

```text
main.py
  -> system.interfaces.cli.run_cli_entrypoint()
  -> system.app_runtime.main()
  -> TaskRouter.route(args)
  -> RuntimeExecutionEngine.execute(task, args)
  -> system.domain.<domain>
  -> existing implementation
  -> RuntimeStateManager persists task state
```

Current non-CLI recommended entry path:

```text
API / GUI shell
  -> system.interfaces.api or system.interfaces.gui
  -> runtime/domain builder helpers
  -> chat/computer/automation implementation
```

## Why This Matters

This structure solves several problems that were previously most prone to continuing divergence:

### Entries are no longer directly bound to concrete business modules

Previously `app_runtime.py` directly jumped to each mainline via if/else;
now it first turns into a unified `Task` and then hands it to the execution engine.

### Business domains are downgraded from "entry points" to "implementations"

`chat_v2 / computer_use / automation` are now more clearly in the implementation layer rather than the system root entry.

### Execution state is trackable

Unified state files give CLI-level tasks basic observability, which is foundational for later audit, resume, replay, and dashboard.

### API / GUI also start using the unified interface layer

`system/interfaces/api` and `system/interfaces/gui` now cover chat, computer, automation and directly reuse the runtime builder instead of each copying a set of assembly logic.

## Migration Policy

The current approach is a "compatibility-first" migration:

1. Add unified abstractions and facade layers.
2. Keep old implementation directories.
3. Let entries depend on the new abstractions first.
4. Gradually converge internal logic later.

In other words, this is not a rewrite, but creating stable fulcrums for subsequent refactoring.

## Recommended Next Steps

The most worthwhile directions to advance in the next phase:

1. Have `WorkspaceAgent` and `AutomationRuntime` uniformly consume `system/agent/planner.py`.
2. Gradually extract the computer-use action decisions into reusable `Action` schemas.
3. Add resume / lineage / parent-child task relationships to `RuntimeStateManager`.
4. Continue attaching `system/interfaces/api` / `system/interfaces/gui` to more non-mainline entries or service shells.
5. Add module-level READMEs for `system/domain/*`, `system/agent/*`, `system/interfaces/*` to reduce future cognitive overhead.
