
## 中文版

本目录包含了项目的架构设计、迁移记录、开发指南及测试说明等核心文档。

- **ARCHITECTURE.md**  
  分层架构治理文档。定义四层依赖分治架构、单向依赖规则、各层职责、防循环引用策略以及验证工具。

- **DEPENDENCY_QUICK_REFERENCE.md**  
  导入依赖关系快速参考。列出核心发现、依赖层级结构、L1 高频模块详解、循环依赖风险及立即行动清单。

- **IMPLEMENTATION_REPORT.md**  
  分层架构实施完成报告。记录 81 个文件的迁移结果、12 个待优化依赖违规及后续建议。

- **IMPORT_DEPENDENCY_ANALYSIS.md**  
  导入依赖关系详细分析报告。包含模块统计、循环依赖检测、依赖链深度分析及分阶段改进路线图。

- **LAYERED_ARCHITECTURE_SUMMARY.md**  
  分层架构重组完成总结。概述新项目结构、三大关键改进、迁移状态和工具命令。

- **MIGRATION_PLAN.md**  
  迁移映射计划。列出从原始平铺结构到 L1–L4 分层目录的文件移动对应关系。

- **architecture_v2.md**  
  Chat V2 运行时设计说明。阐述 LLM 优先、模块化、可测试的聊天架构，包括单路径决策、数据卫生和质量契约。

- **computer_use_runtime.md**  
  桌面智能体运行时说明。介绍截图、网格标注、OCR、多模态决策、一步一执行及审计追踪的工作方式。

- **desktop_agent_roadmap.md**  
  桌面代理路线图。规划阶段性目标（P0–P3）、设计原则、关键指标及暂缓方向。

- **dialogue_manual_test.md**  
  对话手工测试脚本。提供回归测试用例、启动命令、通过/失败标准及测试记录模板。

- **engineering_upgrade.md**  
  工程升级指南。记录从工作空间卫生、入口拆分、可重现性到推理升级、硬件自适应、Bandit 优化等一系列工程化改进。

- **PROJECT_STRUCTURE.md**  
  项目结构与落位约定。明确各目录职责、分层原则、放置规则及开发者阅读顺序。

- **UNIFIED_RUNTIME_ARCHITECTURE.md**  
  统一运行时架构说明。介绍 Task/Planner/Executor/State 核心抽象、运行时流程、领域门面及迁移策略。

- **README.md**（项目根目录）  
  项目的整体介绍、三条业务主线、统一运行时抽象、快速启动命令和维护建议。

## English Version

This directory contains core documentation covering architecture design, migration records, development guides, and testing instructions.

- **ARCHITECTURE.md**  
  Layered architecture governance document. Defines the four-layer dependency structure, unidirectional dependency rules, module responsibilities, anti-circular-dependency strategies, and verification tools.

- **DEPENDENCY_QUICK_REFERENCE.md**  
  Quick reference for import dependency analysis. Includes key findings, dependency hierarchy, details of high-usage L1 modules, circular dependency risks, and immediate action items.

- **IMPLEMENTATION_REPORT.md**  
  Implementation completion report for the layered architecture. Records the migration of 81 files, 12 remaining dependency violations, and further optimization suggestions.

- **IMPORT_DEPENDENCY_ANALYSIS.md**  
  Detailed import dependency analysis report. Contains module statistics, circular dependency detection, dependency chain depth analysis, and a phased improvement roadmap.

- **LAYERED_ARCHITECTURE_SUMMARY.md**  
  Reorganization summary of the layered architecture. Highlights the new project structure, three key improvements, migration status, and useful commands.

- **MIGRATION_PLAN.md**  
  Migration mapping plan. Lists file movement correspondences from the original flat structure to the L1–L4 layered directories.

- **architecture_v2.md**  
  Design notes for the Chat V2 runtime. Describes the LLM-first, modular, and testable chat architecture, including single-path decision making, data hygiene, and quality contracts.

- **computer_use_runtime.md**  
  Desktop agent runtime guide. Explains the workflow of screenshot capture, grid overlay, OCR, multimodal decision, step-by-step execution, and audit tracing.

- **desktop_agent_roadmap.md**  
  Desktop agent roadmap. Outlines phased goals (P0–P3), design principles, key metrics, and directions that are intentionally deprioritized.

- **dialogue_manual_test.md**  
  Manual dialogue test script. Provides regression test cases, startup commands, pass/fail criteria, and a test record template.

- **engineering_upgrade.md**  
  Engineering upgrade guide. Documents a series of engineering improvements, from workspace hygiene and entry-point splitting to reasoning upgrades, hardware auto-adaptation, and Bandit optimization.

- **PROJECT_STRUCTURE.md**  
  Project structure and placement conventions. Clarifies directory responsibilities, layering principles, code placement rules, and recommended reading order for developers.

- **UNIFIED_RUNTIME_ARCHITECTURE.md**  
  Unified runtime architecture documentation. Introduces the core abstractions (Task/Planner/Executor/State), runtime flow, domain facades, and migration policy.

- **README.md** (project root)  
  Overall project introduction, covering the three main business lines, unified runtime abstractions, quick-start commands, and maintenance suggestions.
