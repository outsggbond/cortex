# system/

`system/` 是项目的主运行时代码目录。

当前建议按 5 层理解，而不是把它看成一堆并列模块：

1. `runtime`
   统一任务路由、执行编排、状态管理、启动支持。
2. `domain`
   业务域门面层，对外暴露 `chat` / `computer` / `automation`。
3. `agent`
   统一 `Task / Planner / Executor / State` 抽象。
4. `chat_v2` / `computer_use` / `automation`
   三条主线的具体实现。
5. `core` / `brain` / `knowledge` / `evaluation` / `learning`
   跨域复用能力与支撑模块。

## 目录职责

- `app_runtime.py`
  根运行时入口，只做 orchestration。
- `main_cli.py`
  薄 CLI 入口。
- `interfaces/cli/`
  CLI 入口与分域参数注册层。
- `interfaces/api/`
  程序化 API 适配层。
- `interfaces/gui/`
  GUI 会话适配层。
- `runtime/`
  统一 runtime 基础设施。
- `domain/`
  业务域门面层。
- `agent/`
  统一任务、规划、执行、状态抽象。
- `chat_v2/`
  对话与 workspace agent 的具体实现。
- `computer_use/`
  桌面智能体的具体实现。
- `automation/`
  workflow 触发与执行的具体实现。
- `knowledge/`
  RAG、规则、知识图谱等知识能力。
- `evaluation/`
  评估、反馈、自修复、审计。
- `learning/`
  学习、数据构建、持续改进。
- `core/`
  真正跨模块复用的基础能力。

## 当前推荐的入口路径

如果你要继续改主线逻辑，建议先看：

1. `system/app_runtime.py`
2. `system/main_cli.py`
3. `system/interfaces/cli/`
4. `system/interfaces/api/`
5. `system/interfaces/gui/`
6. `system/runtime/task_router.py`
7. `system/runtime/execution_engine.py`
8. `system/runtime/state_manager.py`
9. 对应业务实现目录

## 放置规则

- 新增入口分发逻辑：放 `runtime/`
- 新增 CLI 参数注册：放 `interfaces/cli/`
- 新增程序化/API 适配器：放 `interfaces/api/`
- 新增 GUI 会话或展示适配器：放 `interfaces/gui/`
- 新增统一任务/规划/执行抽象：放 `agent/`
- 新增业务域公开入口：放 `domain/`
- 新增聊天能力：放 `chat_v2/`
- 新增桌面代理能力：放 `computer_use/`
- 新增自动化流程能力：放 `automation/`
- 新增通用基础能力：放 `core/` 或 `brain/`

## 导入规范

所有导入必须使用完整的层级路径，例如：
- `from system.runtime.bootstrap import ...`（不是 `from system.runtime_bootstrap`）
- `from system.core.planner import ...`（不是 `from system.planner`）
- `from system.brain.reasoning import ...`（不是 `from system.reasoning`）

禁止在 `system/` 根目录放置新的扁平模块。
