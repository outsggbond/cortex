好的，以下是中英文合并版的《分层架构治理文档》：



### 架构概览

本项目采用**四层依赖分治架构**，确保代码的可维护性和可扩展性。

```
┌─────────────────────────────────────────────────────┐
│  L4 执行与应用层 (The Body & Interface)             │
│  computer_use/ automation/ chat_v2/ evaluation/   │
├─────────────────────────────────────────────────────┤
│  L3 智能推理层 (The Brain)                          │
│  brain/ learning/ strategy/                       │
├─────────────────────────────────────────────────────┤
│  L2 感知与知识层 (The Senses & Memory)              │
│  perception/ knowledge/                           │
├─────────────────────────────────────────────────────┤
│  L1 基础设施层 (The Bedrock)                        │
│  core/ runtime/ l_utils/                          │
└─────────────────────────────────────────────────────┘
```

### 分层规则

#### 📍 单向依赖（CRITICAL）

```
╔════════════════════════════════════════╗
║ 允许的导入方向                          ║
╠════════════════════════════════════════╣
║ L4 → L3 ✅  (执行层可引用推理层)       ║
║ L4 → L2 ✅  (执行层可引用知识层)       ║
║ L4 → L1 ✅  (执行层可引用基础设施)     ║
║                                        ║
║ L3 → L2 ✅  (推理层可引用知识层)       ║
║ L3 → L1 ✅  (推理层可引用基础设施)     ║
║                                        ║
║ L2 → L1 ✅  (知识层可引用基础设施)     ║
║                                        ║
║ L1 → L2 ❌  (基础设施严禁引用知识层)   ║
║ L1 → L3 ❌  (基础设施严禁引用推理层)   ║
║ L1 → L4 ❌  (基础设施严禁引用应用层)   ║
║                                        ║
║ L2 → L3 ❌  (知识层严禁引用推理层)     ║
║ L2 → L4 ❌  (知识层严禁引用应用层)     ║
║                                        ║
║ L3 → L4 ❌  (推理层严禁引用应用层)     ║
╚════════════════════════════════════════╝
```

### 各层详细说明

#### L1 基础设施层 (The Bedrock)

**不依赖其他层，提供原始功能。**

- **core/**: 核心基础设施
  - `hardware.py` - 硬件检测与监控
  - `embeddings.py` - 嵌入向量生成
  - `model_loader.py` - 模型加载器
  - `state_encoder.py` - 状态编码
  - `planner.py` - 规划原语
  - `error_normalizer.py` - 错误规范化

- **runtime/**: 运行时基础设施
  - `bootstrap.py` - 启动配置
  - `support.py` - 运行时支持
  - `artifacts.py` - 制品管理
  - `adaptive.py` - 自适应运行时

- **l_utils/**: 通用工具（防止 utils/ 循环污染）
  - `loader.py` - 模块加载
  - `codegen.py` - 代码生成
  - `files.py` - 文件操作
  - `metrics.py` - 指标统计
  - `cli_support.py` - CLI 支持
  - `graphviz_export.py` - 图形导出

#### L2 感知与知识层 (The Senses & Memory)

**依赖 L1，提供感知和知识管理能力。**

- **perception/**: 多模态感知融合
  - 视觉预处理、OCR 对齐、UI 树融合
  - 未来扩展：YOLO, SAM, 多模态 VLM
  - *保持独立目录，避免与知识层混淆*

- **knowledge/**: 知识管理与检索
  - `rag_store.py` - RAG 存储
  - **graph/**: 知识图谱
    - `kb.py` - 知识库
    - `query.py` - 查询引擎
    - `repl.py` - 交互式 REPL
    - `cli.py` - 命令行接口
  - **semantic/**: 语义索引与检索
    - `index.py` - 混合语义索引
    - `faq.py` - FAQ 检索
    - `refiner.py` - 语义优化
  - `memory.py` - 内存与向量数据库管理

#### L3 智能推理层 (The Brain)

**依赖 L1/L2，提供推理和学习能力。**

- **brain/**: 推理与世界模型（reasoning + world 合并）
  - `reasoning.py` - 核心推理引擎
  - `neural_reasoner.py` - 神经推理
  - `debate_reasoner.py` - 辩论式推理
  - `neurosymbolic.py` - 神经符号混合
  - `profile.py` - 推理配置优化
  - `world_model.py` - 世界状态建模
  - `world_predictor.py` - 状态预测
  - `stimulus_engine.py` - 刺激与反应

- **learning/**: 持续学习与训练
  - `continual.py` - 持续学习框架
  - `feedback.py` - 反馈处理
  - `loader.py` - 训练数据加载
  - **dialogue/**: 对话学习
    - `dialogue_generator.py` - 对话生成
    - `dialogue_augment.py` - 对话增强
    - `dialogue_evolver.py` - 对话演进
    - `dialogue_dataset_builder.py` - 数据集构建

- **strategy/**: 策略与规划
  - `strategy.py` - 策略生成
  - `task_flow.py` - 任务流程
  - `search_planner.py` - 搜索规划

#### L4 执行与应用层 (The Body & Interface)

**依赖所有下层，提供最终用户接口。**

- **computer_use/**: 桌面智能体运行时
  - `runtime.py` - 主运行时
  - `llm.py` - 决策 LLM
  - `plugins.py` - 桌面/浏览器插件
  - **action/**: 动作系统
    - `generator.py` - 动作生成
    - `validator.py` - 动作验证
    - `stats.py` - 动作统计
  - **trace/**: 审计与可视化
    - `audit.py` - 审计日志
    - `visual.py` - 轨迹可视化

- **automation/**: 自动化工作流执行
  - `runtime.py` - 工作流运行时
  - `executor.py` - 执行器

- **chat_v2/**: 模块化聊天运行时（保持不变）
  - 支持规则回复、检索回退、LLM 接入、Agent 模式

- **evaluation/**: 评估与守卫
  - `evaluator.py` - 性能评估
  - `reflect.py` - 自我反思
  - `self_heal.py` - 自我修复
  - `response_guard.py` - 响应守卫

### 防循环引用策略

#### ✅ 正确做法

```python
# ✓ 在 L1 模块中可以定义接口
# core/planner.py
from typing import Protocol

class PlanExecutor(Protocol):
    """规划执行器接口"""
    def execute(self, plan: Plan) -> Result: ...

# ✓ 在 L3/L4 模块中实现接口
# brain/reasoning.py
from system.core.planner import PlanExecutor

class ConcreteExecutor(PlanExecutor):
    def execute(self, plan: Plan) -> Result:
        ...

# ✓ 在 L3/L4 中使用依赖注入
# computer_use/runtime.py
def run(executor: PlanExecutor) -> None:
    # 不直接导入具体实现
    ...
```

#### ❌ 禁止做法

```python
# ✗ L1 导入 L2/L3/L4
# core/hardware.py
from system.brain.reasoning import Reasoner  # ❌ 违反依赖方向

# ✗ 在 __init__.py 中写业务逻辑
# knowledge/__init__.py
class RagStore:  # ❌ 应该在独立模块中
    def search(self): ...

# ✗ 双向导入
# learning/continual.py
from system.computer_use.runtime import Runtime  # ❌ L3 引用 L4
```

### 检查清单

在添加新代码时，问自己：

- [ ] 我的模块应该放在哪一层？
- [ ] 它依赖的所有导入都来自同层或下层吗？
- [ ] 我是否在 `__init__.py` 中写了业务逻辑？（应该只做模块暴露）
- [ ] 是否存在潜在的循环依赖？
- [ ] 这个依赖是否真的必要，或者应该使用接口/依赖注入？

### 工具与命令

#### 验证依赖完整性

```bash
# 检查所有导入是否正确
python -m py_compile system/**/*.py

# 生成依赖图
python scripts/analyze_imports_detailed.py
```

#### 添加新模块

```bash
# 1. 创建模块目录
mkdir -p system/your_layer/your_module

# 2. 创建 __init__.py
echo "# -*- coding: utf-8 -*-" > system/your_layer/your_module/__init__.py

# 3. 添加代码，确保满足分层规则

# 4. 验证
python -m py_compile system/your_layer/your_module/*.py
```

### 迁移状态

- ✅ 循环依赖已修复（world_model.py）
- ✅ 57 个文件已迁移到新结构
- ✅ 48 个文件的导入路径已更新
- ✅ 所有 __init__.py 已创建
- ⏳ 需要运行完整测试套件验证
- ⏳ 更新相关文档引用
- ⏳ 更新 CI/CD 配置

### 后续优化方向

1. **降低导入深度**：从目前的 7 层降低到 4-5 层
2. **增加接口定义**：减少具体依赖，增加协议/抽象
3. **异步加载**：对于重型模块采用延迟导入
4. **模块隔离**：考虑基于特性的模块分组

---

**文档版本**: 1.0  
**最后更新**: 2026-04-22  
**维护者**: AI 系统架构组

---

## English Version

### Architecture Overview

This project adopts a **four-layer dependency-governed architecture** to ensure maintainability and scalability.

```
┌─────────────────────────────────────────────────────┐
│  L4 Execution & Application (The Body & Interface)  │
│  computer_use/ automation/ chat_v2/ evaluation/   │
├─────────────────────────────────────────────────────┤
│  L3 Intelligent Reasoning (The Brain)               │
│  brain/ learning/ strategy/                       │
├─────────────────────────────────────────────────────┤
│  L2 Perception & Knowledge (The Senses & Memory)    │
│  perception/ knowledge/                           │
├─────────────────────────────────────────────────────┤
│  L1 Infrastructure (The Bedrock)                    │
│  core/ runtime/ l_utils/                          │
└─────────────────────────────────────────────────────┘
```

### Layering Rules

#### 📍 Unidirectional Dependencies (CRITICAL)

```
╔════════════════════════════════════════╗
║ Allowed Import Directions              ║
╠════════════════════════════════════════╣
║ L4 → L3 ✅                             ║
║ L4 → L2 ✅                             ║
║ L4 → L1 ✅                             ║
║                                        ║
║ L3 → L2 ✅                             ║
║ L3 → L1 ✅                             ║
║                                        ║
║ L2 → L1 ✅                             ║
║                                        ║
║ L1 → L2 ❌ (Infra must not import Knowledge) ║
║ L1 → L3 ❌ (Infra must not import Reasoning) ║
║ L1 → L4 ❌ (Infra must not import App)       ║
║                                        ║
║ L2 → L3 ❌ (Knowledge must not import Reasoning) ║
║ L2 → L4 ❌ (Knowledge must not import App)       ║
║                                        ║
║ L3 → L4 ❌ (Reasoning must not import App)       ║
╚════════════════════════════════════════╝
```

### Layer Details

#### L1 Infrastructure Layer (The Bedrock)

**Depends on nothing; provides foundational capabilities.**

- **core/**: Core infrastructure
  - `hardware.py` - Hardware detection & monitoring
  - `embeddings.py` - Embedding vector generation
  - `model_loader.py` - Model loader
  - `state_encoder.py` - State encoding
  - `planner.py` - Planning primitives
  - `error_normalizer.py` - Error normalization

- **runtime/**: Runtime infrastructure
  - `bootstrap.py` - Startup configuration
  - `support.py` - Runtime support
  - `artifacts.py` - Artifact management
  - `adaptive.py` - Adaptive runtime

- **l_utils/**: General utilities (avoids `utils/` pollution)
  - `loader.py` - Module loading
  - `codegen.py` - Code generation
  - `files.py` - File operations
  - `metrics.py` - Metrics
  - `cli_support.py` - CLI support
  - `graphviz_export.py` - Graph export

#### L2 Perception & Knowledge Layer (The Senses & Memory)

**Depends on L1; provides perception and knowledge management.**

- **perception/**: Multimodal perception fusion
  - Visual preprocessing, OCR alignment, UI tree fusion
  - Future: YOLO, SAM, multimodal VLMs
  - *Kept separate to avoid confusion with knowledge layer*

- **knowledge/**: Knowledge management & retrieval
  - `rag_store.py` - RAG store
  - **graph/**: Knowledge graph
    - `kb.py` - Knowledge base
    - `query.py` - Query engine
    - `repl.py` - Interactive REPL
    - `cli.py` - CLI interface
  - **semantic/**: Semantic indexing & retrieval
    - `index.py` - Hybrid semantic index
    - `faq.py` - FAQ retrieval
    - `refiner.py` - Semantic refinement
  - `memory.py` - In-memory & vector DB management

#### L3 Intelligent Reasoning Layer (The Brain)

**Depends on L1/L2; provides reasoning and learning.**

- **brain/**: Reasoning & world model (merged reasoning + world)
  - `reasoning.py` - Core reasoning engine
  - `neural_reasoner.py` - Neural reasoning
  - `debate_reasoner.py` - Debate-style reasoning
  - `neurosymbolic.py` - Neuro-symbolic hybrid
  - `profile.py` - Reasoning profile optimization
  - `world_model.py` - World state modeling
  - `world_predictor.py` - State prediction
  - `stimulus_engine.py` - Stimulus & response

- **learning/**: Continual learning & training
  - `continual.py` - Continual learning framework
  - `feedback.py` - Feedback processing
  - `loader.py` - Training data loader
  - **dialogue/**: Dialogue learning
    - `dialogue_generator.py` - Dialogue generation
    - `dialogue_augment.py` - Dialogue augmentation
    - `dialogue_evolver.py` - Dialogue evolution
    - `dialogue_dataset_builder.py` - Dataset construction

- **strategy/**: Strategy & planning
  - `strategy.py` - Strategy generation
  - `task_flow.py` - Task flow
  - `search_planner.py` - Search planning

#### L4 Execution & Application Layer (The Body & Interface)

**Depends on all lower layers; provides end-user interfaces.**

- **computer_use/**: Desktop agent runtime
  - `runtime.py` - Main runtime
  - `llm.py` - Decision LLM
  - `plugins.py` - Desktop/browser plugins
  - **action/**: Action system
    - `generator.py` - Action generation
    - `validator.py` - Action validation
    - `stats.py` - Action statistics
  - **trace/**: Audit & visualization
    - `audit.py` - Audit log
    - `visual.py` - Trace visualization

- **automation/**: Automation workflow execution
  - `runtime.py` - Workflow runtime
  - `executor.py` - Executor

- **chat_v2/**: Modular chat runtime (unchanged)
  - Supports rule-based replies, retrieval fallback, LLM access, Agent mode

- **evaluation/**: Evaluation & guard
  - `evaluator.py` - Performance evaluation
  - `reflect.py` - Self-reflection
  - `self_heal.py` - Self-healing
  - `response_guard.py` - Response guard

### Anti-Circular-Dependency Strategy

#### ✅ Correct Practices

```python
# ✓ Define interfaces in L1
# core/planner.py
from typing import Protocol

class PlanExecutor(Protocol):
    """Planner executor interface"""
    def execute(self, plan: Plan) -> Result: ...

# ✓ Implement interfaces in L3/L4
# brain/reasoning.py
from system.core.planner import PlanExecutor

class ConcreteExecutor(PlanExecutor):
    def execute(self, plan: Plan) -> Result:
        ...

# ✓ Use dependency injection in L3/L4
# computer_use/runtime.py
def run(executor: PlanExecutor) -> None:
    # no direct import of concrete implementations
    ...
```

#### ❌ Forbidden Practices

```python
# ✗ L1 importing from L2/L3/L4
# core/hardware.py
from system.brain.reasoning import Reasoner  # ❌ Violates dependency direction

# ✗ Business logic in __init__.py
# knowledge/__init__.py
class RagStore:  # ❌ Should be in a separate module
    def search(self): ...

# ✗ Bidirectional imports
# learning/continual.py
from system.computer_use.runtime import Runtime  # ❌ L3 references L4
```

### Checklist

When adding new code, ask yourself:

- [ ] Which layer should my module belong to?
- [ ] Are all its imports from the same layer or lower?
- [ ] Did I put business logic in `__init__.py`? (Should only expose modules)
- [ ] Is there potential circular dependency?
- [ ] Is this dependency really necessary, or should I use an interface / dependency injection?

### Tools & Commands

#### Verify Dependency Integrity

```bash
# Check all imports for correctness
python -m py_compile system/**/*.py

# Generate dependency graph
python scripts/analyze_imports_detailed.py
```

#### Adding a New Module

```bash
# 1. Create module directory
mkdir -p system/your_layer/your_module

# 2. Create __init__.py
echo "# -*- coding: utf-8 -*-" > system/your_layer/your_module/__init__.py

# 3. Add code, ensuring layering rules are met

# 4. Verify
python -m py_compile system/your_layer/your_module/*.py
```

### Migration Status

- ✅ Circular dependencies fixed (world_model.py)
- ✅ 57 files migrated to new structure
- ✅ 48 files have updated import paths
- ✅ All __init__.py files created
- ⏳ Full test suite execution pending
- ⏳ Update related documentation references
- ⏳ Update CI/CD configuration

### Future Optimization Directions

1. **Reduce import depth**: from 7 layers down to 4–5
2. **Increase interface definitions**: favor protocols/abstractions over concrete dependencies
3. **Lazy loading**: use deferred imports for heavy modules
4. **Module isolation**: consider feature-based module grouping

---

**Document Version**: 1.0  
**Last Updated**: 2026-04-22  
**Maintainer**: AI System Architecture Group
