# 工程升级指南

## P0: 工作空间卫生

1. 缓存和临时文件现已在 `.gitignore` 中忽略：
   - `.venv/`
   - `__pycache__/`
   - `*.pyc`
   - 常见的运行时产物输出
2. 需要时运行缓存清理：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/clean_project_cache.ps1
```

## P0: 入口点拆分

1. 薄根入口点：
   - `main.py`
2. 运行时编排移至：
   - `system/app_runtime.py`
3. CLI 参数层移至：
   - `system/main_cli.py`
4. 适配器训练 CLI 拆分：
   - `scripts/train_adapter.py` (运行时)
   - `scripts/train_adapter_cli.py` (参数解析)

## P1: 可重现性

1. 包元数据和扩展：
   - `pyproject.toml`
2. 锁定快照：
   - `requirements.lock`
3. 安装模式：

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## P1: 测试 + 覆盖率 + CI

1. 带覆盖率运行测试：

```bash
pytest
```

2. CI 工作流：
   - `.github/workflows/ci.yml`

## P1: 运行时数据生命周期

使用清理工具修剪旧的/大的运行时产物：

```bash
python scripts/runtime_housekeeping.py --dry-run
python scripts/runtime_housekeeping.py --max-age-days 21 --max-total-gb 6
```

## P2: 编码一致性

1. `.editorconfig` 强制使用 UTF-8 默认编码。
2. 推荐 Windows shell 设置：

```powershell
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
```

3. 运行时已在主编排中重新配置 stdio 编码。

## 推理升级（最新）

1. 层次化推理现在包括：
   - 候选投票
   - 矛盾检查
   - 显式目标一致性验证
2. 神经符号推理现在返回：
   - `best_conclusion`
   - `consensus_score`
   - `conflicts`
3. 运行时生成上下文现在包含推理摘要，以实现更强的“先推理，后回答”行为。

## 推理升级（当前版本）

1. 加强了中英文约束理解：
   - 模态冲突检测现在支持 `must/cannot` 和 `必须/不能/不得/不应该` 模式。
2. 添加了高层综合步骤：
   - `final_decision`（包含动作 + 置信度证据）
   - 当约束冲突或目标证据薄弱时 `request_clarification`。
3. 运行时现在消费结构化推理元数据：
   - `reasoning_decision`
   - `reasoning_confidence`
   - `reasoning_action`
   - `reasoning_need_clarification`
4. 神经相似度不再直接覆盖逻辑分数：
   - 步骤分数现在是逻辑置信度和语义相似度的加权混合。

## 硬件自适应升级（当前版本）

1. 运行时现包含闭环自适应控制器：
   - 通过 EMA 平滑监测 CPU/RAM/GPU 压力。
   - 在持续压力下快速降级。
   - 当资源释放后逐渐恢复到硬件目标策略。
2. 在线调整的自适应旋钮：
   - `reasoning_k`
   - `chat_max_memories`
   - `similarity_threshold`
3. Transformer 路由调度现在具有硬件感知能力：
   - 在节流压力期间，重路由被降级/禁用。
   - 当压力较低且资源分数健康时再次优先使用。
4. 自适应事件持久化以供审计：
   - `audit/runtime_adapt_events.jsonl`
5. 运行时自适应现支持 CLI 覆盖和在线学习：
   - CLI：`--runtime-adapt*`、`--runtime-adapt-bandit*`、`--runtime-transformer-hot-tune`
   - Bandit 状态持久化：`audit/runtime_adapt_bandit_state.json`
6. Transformer 生成参数可在运行时热调优：
   - 令牌/输入限制和采样控制根据实时资源信号进行调整。
   - 显式 CLI transformer 标志仍然优先。
7. 规划器复杂度现为运行时自适应：
   - 搜索 `depth/beam` 由硬件压力 + bandit 乘数热调优。
   - CLI 覆盖：`--runtime-planner-hot-tune` 和 `--runtime-planner-*-min/max`。
8. 添加后台自适应心跳：
   - 运行时自适应可在空闲阶段继续（不仅仅在请求/轮次边界）。
9. 运行时自适应节奏现具有压力感知能力：
   - 高压/节流 -> 更快的采样。
   - 稳定的低压 -> 更慢的采样以减少开销。
10. 运行时自适应持久化具有崩溃韧性：
    - 状态写入现使用原子替换。
    - 当主状态文件损坏时，启动可回退到 `.bak` 快照。
11. 运行时压力阈值现可自校准：
    - `pressure_high/pressure_low` 在线自适应，具有冷却和有限范围。
    - 持续压力使降级触发更早；持续健康阶段逐步放松阈值。
12. 添加运行时健康遥测：
    - 发送周期性 `runtime_health` 事件，包含奖励 EMA、正率 EMA、压力、分数、规划器复杂度和有效间隔。
13. Bandit 探索现为自调优：
    - 奖励差/波动 -> 自动增加探索。
    - 稳定高奖励 -> 自动衰减探索以加快收敛。
14. 添加 Bandit 漂移自修复：
    - 持续低奖励阶段触发陈旧 bandit 记忆的受控衰减/重置。
    - 记录重置事件并提升探索以在工作负载漂移下重新发现更好的臂。
15. 添加上下文 Bandit 硬件偏置：
    - 臂选择现包含实时压力/资源上下文（超出纯粹的历史奖励）。
    - 资源压力低倾向于保守臂；健康余量倾向于更强臂。
    - 支持 CLI 覆盖压力/分数权重和偏置上限。
16. 添加 Bandit 硬件签名热启动保护：
    - bandit 状态现携带硬件签名以区分热启动来源。
    - 当签名不匹配时，运行时可自动衰减陈旧的计数/值并提升探索。
    - 可选严格模式可完全拒绝不匹配的 bandit 状态。
17. 添加上下文冷启动臂选择：
    - 早期 bandit 选择现使用实时余量（`resource_score - pressure`）来选择更好的第一个臂。
    - 高压/低分启动倾向于保守臂；健康启动倾向于激进臂。
    - 可通过 CLI 禁用或调优（`--runtime-adapt-bandit-context-cold-start*`）。
18. 添加 Bandit 陈旧记忆时间衰减：
    - bandit 计数/值现随时间衰减，以减少对旧工作负载状态的锁定。
    - 衰减使用半衰期、最小因子和冷却控制，行为可预测。
    - 衰减遥测包含在 bandit 反馈事件中以供审计。
19. 添加 Bandit 臂可靠性守卫：
    - 每个臂现跟踪奖励 EMA/方差，并在臂选择期间获得有界风险惩罚。
    - 不稳定/高负臂在引起重复回归前被降级。
    - 臂冷却策略和臂风险策略现通过 CLI/运行时配置暴露以进行细粒度控制。
20. 添加 Bandit 安全包络：
    - 在危险硬件信号下（高压/低资源分数/节流），bandit 臂选择自动限制在安全乘数范围内。
    - 这防止历史上高奖励的激进臂在不稳定资源窗口期间过度触发。
    - 安全状态和原因现在运行时自适应遥测中发出，以供审计和调优。
21. 添加 Bandit 状态分割记忆：
    - 运行时 bandit 现为 `normal` 和 `stress` 硬件状态保存单独的臂记忆。
    - 高压/低分学习不再污染正常状态臂偏好。
    - 状态元数据现包含在决策和反馈遥测中，用于调试和策略调优。
22. 添加 Bandit 冲击保护：
    - 短时间内重复的灾难性奖励现触发 bandit 乘数的临时保守上限。
    - 冲击状态（`active/until/reason`）被持久化并在遥测中显示。
    - 这为突发故障（如重复 OOM/超时期间）增加了一层快速回滚。
23. 添加 Bandit 压力冲击保护：
    - 突然的硬件压力跳变/资源分数下降现在在反馈循环跟上之前触发临时保守上限。
    - 冲击保护在正常安全卫士禁用时也有效，充当紧急抢先保护。
    - 冲击状态（`active/until`）被持久化并在运行时健康/自适应遥测中发出。
24. 添加 Bandit 全冷却故障安全回退：
    - 当所有臂都处于冷却状态时，臂选择现回退到最保守的乘数臂，而不是重新打开所有臂。
    - 这防止灾难性突发后立即重新选择激进臂。
    - 策略可通过 `bandit_all_cooled_safe_fallback_enabled` 和相应的运行时 CLI 标志配置。
25. 升级推理候选排序：
    - 最佳候选选择现在混合原始规则分数、目标一致性和事实基础，而不是仅使用原始分数减去全局惩罚。
    - 无支持/高新颖性的结论受到惩罚，以减少在稀疏证据下的离题幻觉。
    - 发出新的 `candidate_ranking` 中级跟踪以提升可观测性和回归分析。
26. 添加数据清洗质量评分：
    - 训练数据清洗现在输出每个清洗文件的 `keep_ratio` 和 `quality_score`。
    - `scripts/clean_training_data.py` 聚合 `avg_quality_score` 和 `avg_keep_ratio`，使数据质量随时间可追踪。
27. 添加推理基准基线：
    - `tests/reasoning_benchmark_test.py` 引入一个小型确定性基准套件。
    - 强制执行通过率阈值，以防止目标一致性/约束处理行为中的静默回归。
28. 添加推理规则反馈学习：
    - `system/reasoning.py` 现在保留持久化的规则可靠性记忆（`audit/reasoning_rule_feedback.json`）。
    - 候选排序包含规则先验奖励，因此历史可靠的规则获得适度偏好。
    - `HierarchicalReasoner.learn_from_feedback(...)` 使离线评估结果能够更新规则先验。
29. 升级推理基准分析：
    - `system/reasoning_benchmark.py` 添加可重用的基准案例、类别指标和失败类型分解。
    - `scripts/eval_reasoning_benchmark.py` 添加报告生成 + 门控阈值 + 可选反馈学习。
    - 基准门控现支持全局通过率和每类别最低率约束。
30. 为 CI 添加数据清洗门控：
    - `scripts/clean_training_data.py` 现支持 `--min-quality-score`、`--min-keep-ratio` 和 `--fail-on-gate`。
    - 低质量训练数据现在可以在自动化中快速失败，而不是静默进入训练流水线。
31. 加强推理歧义处理：
    - `system/reasoning.py` 现在在顶部候选接近且可能不稳定时发出 `candidate_ambiguity`。
    - 最终动作选择现考虑歧义差距/相似性/冲突，减少在接近平局的分歧选项上过度自信的 `proceed`。
    - 接近平局的 `goal_restate` 现在可以让位于具体候选，以避免浅层重述占主导。
32. 添加推理优化循环自动化：
    - `scripts/optimize_reasoning.py` 运行多轮基准门控优化，可选反馈学习。
    - 每轮可写入报告和最终摘要，实现可重复的“优化直到目标阈值”的工作流。
33. 添加推理配置文件参数化：
    - `system/reasoning.py` 现支持可配置的 `ReasoningProfile`（权重、惩罚、歧义阈值、动作置信度门）。
    - `HierarchicalReasoner` 可从文件加载配置文件并应用运行时覆盖，从而无需硬编码即可进行受控行为调优。
34. 添加基准驱动的配置文件自动调优：
    - `system/reasoning_profile_tuner.py` 针对推理基准门控执行候选配置文件搜索。
    - `scripts/tune_reasoning_profile.py` 写入最佳配置文件和调优报告，以实现可重现的优化。
    - `scripts/optimize_reasoning.py` 现支持 `--auto-tune-profile`，用于闭环“调优 + 评估”轮次。
35. 添加置信度校准：
    - `system/reasoning.py` 现在在决策后应用置信度校准，以减少在弱证据下的过度自信动作。
    - 当动作/置信度被调整时，校准发出 `confidence_calibration` 跟踪步骤。
36. 添加鲁棒性基准套件：
    - `system/reasoning_benchmark.py` 现支持 `core` + `robustness` 套件和 `by_suite` 指标。
    - 基准门控现支持套件级阈值（`min_suite_rate`、`required_suites`）以及类别阈值。
    - `scripts/eval_reasoning_benchmark.py`、`scripts/tune_reasoning_profile.py` 和 `scripts/optimize_reasoning.py` 现支持 `--with-robustness` 管道选项。
37. 添加校准指标和门控：
    - 推理基准现报告置信度校准（`ece`、`brier`、桶统计），用于客观的可靠性跟踪。
    - 门控逻辑现支持 `max_ece` 和 `max_brier`，以防止过度自信但脆弱的升级。
38. 添加校准感知优化循环：
    - 配置文件调优分数现包含校准惩罚，因此高通过率但校准不良的配置文件被降级。
    - `eval/tune/optimize` 脚本现暴露校准门控标志，用于端到端自动治理。
39. 添加难度感知推理治理：
    - 基准案例现携带 `difficulty` 并报告 `by_difficulty`、`weighted_pass_rate` 和 `hard_case_pass_rate`。
    - 门控逻辑现支持 `min_weighted_pass_rate`、`min_difficulty_rate`、`required_difficulties` 和 `min_hard_case_rate`。
    - `scripts/eval_reasoning_benchmark.py`、`scripts/tune_reasoning_profile.py` 和 `scripts/optimize_reasoning.py` 现支持难度感知目标标志，使优化优先考虑困难示例，而非仅平均通过率。
40. 添加困难负样本重放优化：
    - 基准现暴露 `replay_reasoning_cases(...)`，从失败样本生成更困难的重放案例。
    - 优化循环现支持 `--with-failure-replay`、`--replay-max-cases` 和 `--replay-multiplier`。
    - 先前轮次的失败样本可在后续轮次（以及调优输入）中重放，使升级专注于持久失败模式，而不是反复过拟合简单胜利。
41. 添加对抗套件治理：
    - 基准现暴露 `adversarial_reasoning_cases(...)` 包含干扰/语码切换/反事实扰动，用于更强的压力评估。
    - `eval/tune/optimize` 现支持 `--with-adversarial`，实现套件门控回归检查，超越基本鲁棒性噪声。
    - 对抗套件与难度感知门控集成，因此可以在对抗输入下强制执行困难案例质量。
42. 添加失败签名因果调优：
    - 配置文件调优现支持来自基准信号的失败引导候选扩展（`failure_breakdown` + 表现不佳的类别）。
    - 候选生成应用目标参数增量（对齐/安全/歧义/校准轴），而不仅仅是盲目的局部网格扰动。
    - 添加 CLI 控制以进行治理：`--disable-failure-guided` 和 `--failure-guided-max-variants`（在调优 + 优化流程中可用）。
    - 失败引导调优现支持持久信号学习记忆（`audit/reasoning_failure_guided_state.json`），使历史有效的信号随时间权重更高。
    - 状态持久化可通过 `--failure-guided-state-path` 和 `--disable-failure-guided-state-persist` 在调优和优化脚本中控制。
43. 添加跨分割数据泄漏治理：
    - `scripts/clean_training_data.py` 现支持清洗输出上的可选跨文件泄漏检查（`--check-cross-leakage`）。
    - 报告现包含跨文件的配对/提示重叠诊断以及训练/评估特定的重叠摘要。
    - 质量门控现支持重叠阈值：`--max-cross-pair-overlap-ratio`、`--max-cross-prompt-overlap-ratio`、`--max-train-eval-pair-overlap-ratio`、`--max-train-eval-prompt-overlap-ratio`。
    - 添加语义近似重复泄漏检查：`--check-semantic-leakage`，带有可配置的相似度阈值和采样预算。
    - 语义泄漏门控支持全局/训练-评估阈值：`--max-semantic-prompt-overlap-ratio`、`--max-train-eval-semantic-prompt-overlap-ratio`。
44. 添加数据集分割泄漏保护：
    - `system/dialogue_dataset_builder.py` 现支持提示组分分割保护（`prevent_cross_split_prompt_leakage`，默认开启），以避免训练/评估之间的提示泄漏。
    - 添加 CLI 控制：`--allow-cross-split-prompt`（显式地选择退出保护）。
    - 数据集清单现报告 `cross_split_prompt_overlap_*` 和 `cross_split_pair_overlap_*` 指标，以提供可审计的分割质量。
45. 添加多样性感知样本选择：
    - `system/dialogue_dataset_builder.py` 现在在最终采样之前，在同一提示的候选内应用可选的近似重复多样性惩罚。
    - 当 `max_per_prompt` 很紧张时，这会降低模板式重写的权重，并有利于更多信息多样化的响应。
    - 新控制：`--disable-diversity-penalty`、`--diversity-similarity-threshold`、`--diversity-penalty-strength`、`--diversity-reference-cap`。
    - 清单现暴露 `near_duplicate_samples` 和 `avg_diversity_penalty` 以进行跟踪。
46. 添加 CI 泄漏烟雾门控：
    - `.github/workflows/ci.yml` 现在在 pytest 之前，针对烟雾训练/评估集运行 `scripts/clean_training_data.py`，并带有严格的重叠门控。
    - 当烟雾数据集管道中出现训练/评估泄漏时，CI 现快速失败。
47. 添加语义配对级泄漏治理：
    - `scripts/clean_training_data.py` 现在计算跨文件的提示级文本和配对级文本（`prompt + response`）的语义重叠。
    - 报告现包含配对级语义诊断（`semantic_pair_*`）以及全局/训练-评估范围的摘要最大比率。
    - 语义门控现支持配对级阈值：`--max-semantic-pair-overlap-ratio`、`--max-train-eval-semantic-pair-overlap-ratio`。
48. 添加数据集清单感知候选评分：
    - `scripts/auto_iterate.py` 现使用 `train_summary.dataset_manifest` 将数据集质量风险纳入候选排序。
    - 候选分数由 `avg_diversity_penalty`、`near_duplicate_ratio`、`cross_split_prompt_overlap_ratio` 和 `cross_split_pair_overlap_ratio` 的惩罚调整。
    - 权重可通过 `--candidate-dataset-*-penalty-weight` 配置，并可通过 `--no-candidate-dataset-penalty` 禁用。
    - 自动迭代现将数据集构建控制（`--allow-cross-split-prompt`、`--disable-diversity-penalty` 和多样性阈值）转发到 `train_adapter.py` 中。
49. 添加统计置信度基准门控：
    - `system/reasoning_benchmark.py` 现报告全局/类别/套件/难度通过率和困难案例通过率的 Wilson CI95 界限。
    - `gate_reasoning_report(...)` 现支持下限治理（`*_ci95_lower`），因此晋升可依赖于置信区间而非仅点估计。
    - `scripts/eval_reasoning_benchmark.py`、`scripts/tune_reasoning_profile.py` 和 `scripts/optimize_reasoning.py` 现暴露 CI95 下限目标标志，用于端到端优化和门控。
50. 添加基线相对显著性门控：
    - `system/reasoning_benchmark.py` 现支持配对报告比较（`compare_reasoning_reports`），包含增量通过率、不一致胜率以及 McNemar/符号检验统计。
    - `gate_reasoning_report(...)` 现支持可选的基线相对门控（`min_delta_pass_rate`、`max_mcnemar_pvalue`、`min_discordant_win_rate`、`min_discordant_win_rate_ci95_lower`、`min_comparable_cases`）。
    - `scripts/eval_reasoning_benchmark.py` 和 `scripts/optimize_reasoning.py` 现支持基线输入（`--baseline-report` / `--baseline-profile-path`）和基线相对目标标志，以实现按改进晋升治理。
51. 添加基线感知配置文件调优：
    - `system/reasoning_profile_tuner.py` 现支持基线相对调优门控，并在搜索期间将每个候选与基线报告/配置文件进行比较。
    - 调优器候选分数现包含基线改进奖励（增量通过率 + 不一致胜率置信度 - 弱显著性惩罚），优先考虑稳健改进而非脆弱的绝对增益。
    - `scripts/tune_reasoning_profile.py` 现暴露基线调优标志（`--baseline-report`、`--baseline-profile-path`、`--min-delta-pass-rate`、`--max-mcnemar-pvalue`、`--min-discordant-win-rate`、`--min-discordant-win-rate-ci95-lower`、`--min-comparable-cases`）。

---

# Engineering Upgrade Guide

## P0: Workspace Hygiene

1. Cache and transient files are now ignored in `.gitignore`:
   - `.venv/`
   - `__pycache__/`
   - `*.pyc`
   - common runtime artifact outputs
2. Run cache cleanup when needed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/clean_project_cache.ps1
```

## P0: Entry-Point Split

1. Thin root entrypoint:
   - `main.py`
2. Runtime orchestration moved to:
   - `system/app_runtime.py`
3. CLI argument layer moved to:
   - `system/main_cli.py`
4. Adapter training CLI split:
   - `scripts/train_adapter.py` (runtime)
   - `scripts/train_adapter_cli.py` (argument parsing)

## P1: Reproducibility

1. Package metadata and extras:
   - `pyproject.toml`
2. Locked snapshot:
   - `requirements.lock`
3. Install patterns:

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

## P1: Test + Coverage + CI

1. Run tests with coverage:

```bash
pytest
```

2. CI workflow:
   - `.github/workflows/ci.yml`

## P1: Runtime Data Lifecycle

Use housekeeping to trim old/large runtime artifacts:

```bash
python scripts/runtime_housekeeping.py --dry-run
python scripts/runtime_housekeeping.py --max-age-days 21 --max-total-gb 6
```

## P2: Encoding Consistency

1. `.editorconfig` enforces UTF-8 defaults.
2. Recommended Windows shell setup:

```powershell
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
```

3. Runtime already reconfigures stdio encoding in main orchestration.

## Reasoning Upgrade (Latest)

1. Hierarchical reasoning now includes:
   - candidate voting
   - contradiction checks
   - explicit goal-alignment verification
2. Neuro-symbolic inference now returns:
   - `best_conclusion`
   - `consensus_score`
   - `conflicts`
3. Runtime generation context now includes reasoning summaries for stronger "reason first, answer second" behavior.

## Reasoning Upgrade (Current Pass)

1. Chinese/English constraint understanding was strengthened:
   - modal conflict detection now supports `must/cannot` and `必须/不能/不得/不应该` patterns.
2. A high-level synthesis step was added:
   - `final_decision` (with action + confidence evidence)
   - `request_clarification` when constraints conflict or goal evidence is weak.
3. Runtime now consumes structured reasoning meta:
   - `reasoning_decision`
   - `reasoning_confidence`
   - `reasoning_action`
   - `reasoning_need_clarification`
4. Neural similarity no longer overwrites logical scores directly:
   - step score is now a weighted blend of logical confidence and semantic similarity.

## Hardware Auto-Adapt Upgrade (Current Pass)

1. Runtime now includes a closed-loop adaptive controller:
   - monitors CPU/RAM/GPU pressure with EMA smoothing.
   - degrades quickly under sustained pressure.
   - recovers gradually to hardware-target strategy when resources free up.
2. Adaptive knobs adjusted online:
   - `reasoning_k`
   - `chat_max_memories`
   - `similarity_threshold`
3. Transformer route scheduling is now hardware-aware:
   - heavy route is deprioritized/disabled during throttle pressure.
   - preferred again when pressure is low and resource score is healthy.
4. Adaptive events are persisted for audit:
   - `audit/runtime_adapt_events.jsonl`
5. Runtime adaptive now supports CLI overrides and online learning:
   - CLI: `--runtime-adapt*`, `--runtime-adapt-bandit*`, `--runtime-transformer-hot-tune`
   - Bandit state persistence: `audit/runtime_adapt_bandit_state.json`
6. Transformer generation parameters can be hot-tuned during runtime:
   - token/input limits and sampling controls are adjusted from live resource signals.
   - explicit CLI transformer flags still take precedence.
7. Planner complexity is now runtime-adaptive:
   - search `depth/beam` are hot-tuned by hardware pressure + bandit multiplier.
   - CLI overrides: `--runtime-planner-hot-tune` and `--runtime-planner-*-min/max`.
8. Background adaptive heartbeat added:
   - runtime adaptation can continue during idle phases (not only on request/turn boundaries).
9. Runtime adaptive cadence is now pressure-aware:
   - high pressure / throttle -> faster sampling.
   - stable low pressure -> slower sampling to reduce overhead.
10. Runtime adaptive persistence is crash-hardened:
   - state writes now use atomic replace.
   - startup can fallback to `.bak` snapshot when the primary state file is corrupted.
11. Runtime pressure thresholds now self-calibrate:
   - `pressure_high/pressure_low` adapt online with cooldown and bounded range.
   - sustained pressure makes degradation trigger earlier; sustained healthy phases relax thresholds gradually.
12. Runtime health telemetry added:
   - emits periodic `runtime_health` events with reward EMA, positive-rate EMA, pressure, score, planner complexity and effective interval.
13. Bandit exploration is now self-tuning:
   - poor reward / volatile reward -> exploration increases automatically.
   - stable high reward -> exploration decays automatically for faster convergence.
14. Bandit drift self-healing added:
   - sustained low-reward phases trigger a controlled decay/reset of stale bandit memory.
   - reset event is logged and exploration is boosted to re-discover better arms under workload drift.
15. Contextual bandit hardware bias added:
   - arm selection now includes live pressure/resource context (beyond pure historical reward).
   - low-resource pressure tends to favor conservative arms; healthy headroom tends to favor stronger arms.
   - supports CLI overrides for pressure/score weight and bias cap.
16. Bandit hardware-signature warm-start guard added:
   - bandit state now carries a hardware signature to distinguish warm-start provenance.
   - when signatures mismatch, runtime can decay stale counts/values and boost exploration automatically.
   - optional strict mode can reject mismatched bandit state entirely.
17. Contextual cold-start arm pick added:
   - early bandit selection now uses live headroom (`resource_score - pressure`) to pick a better first arm.
   - high-pressure/low-score starts favor conservative arms; healthy starts favor aggressive arms.
   - can be disabled or tuned via CLI (`--runtime-adapt-bandit-context-cold-start*`).
18. Bandit stale-memory time decay added:
   - bandit counts/values now decay with elapsed time to reduce lock-in to old workload regimes.
   - decay uses half-life, minimum factor and cooldown controls for predictable behavior.
   - decay telemetry is included in bandit feedback events for auditability.
19. Bandit arm reliability guard added:
   - each arm now tracks reward EMA/variance and receives a bounded risk penalty during arm selection.
   - unstable/high-negative arms are de-prioritized before they cause repeated regressions.
   - arm cooldown policy and arm-risk policy are now exposed through CLI/runtime config for fine-grained control.
20. Bandit safety envelope added:
   - under dangerous hardware signals (high pressure / low resource-score / throttle), bandit arm choice is automatically capped to a safe multiplier range.
   - this prevents historical high-reward aggressive arms from overfiring during unstable resource windows.
   - safety status and reason are now emitted in runtime adaptive telemetry for auditing and tuning.
21. Bandit regime split memory added:
   - runtime bandit now keeps separate arm memory for `normal` and `stress` hardware regimes.
   - high-pressure/low-score learning no longer pollutes normal-state arm preferences.
   - regime metadata is now included in decision and feedback telemetry for debugging and policy tuning.
22. Bandit shock guard added:
   - repeated catastrophic rewards in a short window now trigger a temporary conservative cap on bandit multiplier.
   - shock state (`active/until/reason`) is persisted and surfaced in telemetry.
   - this adds a fast rollback layer for burst failures such as repeated OOM/timeout periods.
23. Bandit pressure-spike guard added:
   - sudden hardware pressure jumps / resource-score drops now trigger a temporary conservative cap before feedback loops catch up.
   - spike guard works even when normal safety guard is disabled, acting as emergency preemptive protection.
   - spike state (`active/until`) is persisted and emitted in runtime health/adapt telemetry.
24. Bandit all-cooled fail-safe fallback added:
   - when all arms are in cooldown, arm selection now falls back to the most conservative multiplier arm(s) instead of reopening all arms.
   - this prevents immediate re-selection of aggressive arms right after catastrophic bursts.
   - policy is configurable via `bandit_all_cooled_safe_fallback_enabled` and corresponding runtime CLI flag.
25. Reasoning candidate ranking upgraded:
   - best-candidate selection now blends raw rule score, goal alignment and fact grounding instead of using raw score minus a global penalty only.
   - unsupported/high-novelty conclusions are penalized to reduce off-goal hallucination under sparse evidence.
   - a new `candidate_ranking` mid-level trace is emitted for observability and regression analysis.
26. Data cleaning quality scoring added:
   - training data cleaning now outputs `keep_ratio` and `quality_score` for each cleaned file.
   - `scripts/clean_training_data.py` aggregates `avg_quality_score` and `avg_keep_ratio` so data quality becomes trackable over time.
27. Reasoning benchmark baseline added:
   - `tests/reasoning_benchmark_test.py` introduces a small deterministic benchmark suite.
   - pass-rate threshold is enforced to prevent silent regressions in goal alignment/constraint handling behavior.
28. Reasoning rule-feedback learning added:
   - `system/reasoning.py` now keeps a persisted rule reliability memory (`audit/reasoning_rule_feedback.json`).
   - candidate ranking includes rule prior bonus so historically reliable rules get modest preference.
   - `HierarchicalReasoner.learn_from_feedback(...)` enables offline eval outcomes to update rule priors.
29. Reasoning benchmark analytics upgraded:
   - `system/reasoning_benchmark.py` adds reusable benchmark cases, category metrics, and failure-type breakdown.
   - `scripts/eval_reasoning_benchmark.py` adds report generation + gate thresholds + optional feedback learning.
   - benchmark gate now supports both global pass-rate and per-category minimum-rate constraints.
30. Data cleaning gate for CI added:
   - `scripts/clean_training_data.py` now supports `--min-quality-score`, `--min-keep-ratio`, and `--fail-on-gate`.
   - low-quality training data can now fail fast in automation instead of silently entering training pipelines.
31. Reasoning ambiguity handling strengthened:
   - `system/reasoning.py` now emits `candidate_ambiguity` when top candidates are close and potentially unstable.
   - final action selection now factors in ambiguity gap/similarity/conflict, reducing over-confident `proceed` on near-tie divergent options.
   - near-tie `goal_restate` can now yield to a concrete candidate to avoid shallow restatement dominance.
32. Reasoning optimization loop automation added:
   - `scripts/optimize_reasoning.py` runs multi-round benchmark-gated optimization with optional feedback learning.
   - each round can write a report and final summary, enabling repeatable "optimize until target threshold" workflows.
33. Reasoning profile parameterization added:
   - `system/reasoning.py` now supports a configurable `ReasoningProfile` (weights, penalties, ambiguity thresholds, action confidence gates).
   - `HierarchicalReasoner` can load profile from file and apply runtime overrides, enabling controlled behavior tuning without hardcoding.
34. Benchmark-driven profile auto-tuning added:
   - `system/reasoning_profile_tuner.py` performs candidate-profile search against reasoning benchmark gates.
   - `scripts/tune_reasoning_profile.py` writes the best profile and tuning report for reproducible optimization.
   - `scripts/optimize_reasoning.py` now supports `--auto-tune-profile` for closed-loop "tune + evaluate" rounds.
35. Confidence calibration added:
   - `system/reasoning.py` now applies post-decision confidence calibration to reduce overconfident actions under weak evidence.
   - calibration emits `confidence_calibration` trace step when action/confidence is adjusted.
36. Robustness benchmark suite added:
   - `system/reasoning_benchmark.py` now supports `core` + `robustness` suites and `by_suite` metrics.
   - benchmark gate now supports suite-level thresholds (`min_suite_rate`, `required_suites`) in addition to category thresholds.
   - `scripts/eval_reasoning_benchmark.py`, `scripts/tune_reasoning_profile.py`, and `scripts/optimize_reasoning.py` now support `--with-robustness` pipeline options.
37. Calibration metrics and gates added:
   - reasoning benchmark now reports confidence calibration (`ece`, `brier`, bucket stats) for objective reliability tracking.
   - gate logic now supports `max_ece` and `max_brier` to prevent overconfident-yet-fragile upgrades.
38. Calibration-aware optimization loop added:
    - profile tuning score now includes calibration penalties so high pass-rate but miscalibrated profiles are de-prioritized.
    - `eval/tune/optimize` scripts now expose calibration gate flags for end-to-end automated governance.
39. Difficulty-aware reasoning governance added:
    - benchmark cases now carry `difficulty` and report `by_difficulty`, `weighted_pass_rate`, and `hard_case_pass_rate`.
    - gate logic now supports `min_weighted_pass_rate`, `min_difficulty_rate`, `required_difficulties`, and `min_hard_case_rate`.
    - `scripts/eval_reasoning_benchmark.py`, `scripts/tune_reasoning_profile.py`, and `scripts/optimize_reasoning.py` now support difficulty-aware target flags so optimization prioritizes hard examples instead of only average pass-rate.
40. Hard-negative replay optimization added:
    - benchmark now exposes `replay_reasoning_cases(...)` to generate harder replay cases from failed samples.
    - optimize loop now supports `--with-failure-replay`, `--replay-max-cases`, and `--replay-multiplier`.
    - failed samples from previous rounds can be replayed in later rounds (and tuning input) so upgrades focus on persistent failure modes rather than repeatedly overfitting easy wins.
41. Adversarial suite governance added:
    - benchmark now exposes `adversarial_reasoning_cases(...)` with distractor/code-switch/counterfactual perturbations for stronger stress evaluation.
    - `eval/tune/optimize` now support `--with-adversarial`, enabling suite-gated regression checks beyond basic robustness noise.
    - adversarial suite integrates with difficulty-aware gates so hard-case quality can be enforced under adversarial inputs.
42. Failure-signature causal tuning added:
    - profile tuning now supports failure-guided candidate expansion from benchmark signals (`failure_breakdown` + underperforming categories).
    - candidate generation applies targeted parameter deltas (alignment/safety/ambiguity/calibration axes) instead of only blind local grid perturbation.
    - CLI controls added for governance: `--disable-failure-guided` and `--failure-guided-max-variants` (available in tune + optimize flows).
    - failure-guided tuning now supports persistent signal-learning memory (`audit/reasoning_failure_guided_state.json`) so historically effective signals are weighted higher over time.
    - state persistence can be controlled with `--failure-guided-state-path` and `--disable-failure-guided-state-persist` in both tune and optimize scripts.
43. Cross-split data leakage governance added:
    - `scripts/clean_training_data.py` now supports optional cross-file leakage checks (`--check-cross-leakage`) over cleaned outputs.
    - report now includes pair/prompt overlap diagnostics across files and train/eval-specific overlap summaries.
    - quality gate now supports overlap thresholds: `--max-cross-pair-overlap-ratio`, `--max-cross-prompt-overlap-ratio`, `--max-train-eval-pair-overlap-ratio`, `--max-train-eval-prompt-overlap-ratio`.
    - semantic near-duplicate leakage check added: `--check-semantic-leakage` with configurable similarity threshold and sampling budget.
    - semantic leakage gate supports global/train-eval thresholds: `--max-semantic-prompt-overlap-ratio`, `--max-train-eval-semantic-prompt-overlap-ratio`.
44. Dataset split leakage guard added:
    - `system/dialogue_dataset_builder.py` now supports prompt-group split protection (`prevent_cross_split_prompt_leakage`, default on) to avoid prompt leakage between train/eval.
    - CLI control added: `--allow-cross-split-prompt` (explicitly opt out of the protection).
    - dataset manifest now reports `cross_split_prompt_overlap_*` and `cross_split_pair_overlap_*` metrics for auditable split quality.
45. Diversity-aware sample selection added:
    - `system/dialogue_dataset_builder.py` now applies an optional near-duplicate diversity penalty within same-prompt candidates before final sampling.
    - this down-weights template-like rewrites and favors more information-diverse responses when `max_per_prompt` is tight.
    - new controls: `--disable-diversity-penalty`, `--diversity-similarity-threshold`, `--diversity-penalty-strength`, `--diversity-reference-cap`.
    - manifest now exposes `near_duplicate_samples` and `avg_diversity_penalty` for tracking.
46. CI leakage smoke gate added:
    - `.github/workflows/ci.yml` now runs `scripts/clean_training_data.py` against smoke train/eval sets with strict overlap gates before pytest.
    - CI now fails fast when train/eval leakage appears in the smoke dataset pipeline.
47. Semantic pair-level leakage governance added:
    - `scripts/clean_training_data.py` now computes semantic overlap for both prompt-level text and pair-level text (`prompt + response`) across files.
    - report now includes pair-level semantic diagnostics (`semantic_pair_*`) and summary max ratios for global/train-eval scopes.
    - semantic gate now supports pair-level thresholds: `--max-semantic-pair-overlap-ratio`, `--max-train-eval-semantic-pair-overlap-ratio`.
48. Dataset-manifest-aware candidate scoring added:
    - `scripts/auto_iterate.py` now folds dataset quality risks into candidate ranking using `train_summary.dataset_manifest`.
    - candidate score is adjusted by penalties from `avg_diversity_penalty`, `near_duplicate_ratio`, `cross_split_prompt_overlap_ratio`, and `cross_split_pair_overlap_ratio`.
    - weights are configurable via `--candidate-dataset-*-penalty-weight` and can be disabled with `--no-candidate-dataset-penalty`.
    - auto-iterate now forwards dataset build controls (`--allow-cross-split-prompt`, `--disable-diversity-penalty`, and diversity thresholds) into `train_adapter.py`.
49. Statistical-confidence benchmark gates added:
    - `system/reasoning_benchmark.py` now reports Wilson CI95 bounds for global/category/suite/difficulty pass rates and hard-case pass-rate.
    - `gate_reasoning_report(...)` now supports lower-bound governance (`*_ci95_lower`) so promotion can depend on confidence intervals instead of point estimates only.
    - `scripts/eval_reasoning_benchmark.py`, `scripts/tune_reasoning_profile.py`, and `scripts/optimize_reasoning.py` now expose CI95 lower-bound target flags for end-to-end optimization and gating.
50. Baseline-relative significance gating added:
    - `system/reasoning_benchmark.py` now supports paired report comparison (`compare_reasoning_reports`) with delta pass-rate, discordant win-rate, and McNemar/sign-test statistics.
    - `gate_reasoning_report(...)` now supports optional baseline-relative gates (`min_delta_pass_rate`, `max_mcnemar_pvalue`, `min_discordant_win_rate`, `min_discordant_win_rate_ci95_lower`, `min_comparable_cases`).
    - `scripts/eval_reasoning_benchmark.py` and `scripts/optimize_reasoning.py` now support baseline inputs (`--baseline-report` / `--baseline-profile-path`) and baseline-relative target flags for promotion-by-improvement governance.
51. Baseline-aware profile tuning added:
    - `system/reasoning_profile_tuner.py` now supports baseline-relative tuning gates and compares each candidate against a baseline report/profile during search.
    - tuner candidate score now includes a baseline-improvement bonus (delta pass-rate + discordant win-rate confidence - weak-significance penalty), prioritizing robust improvements over fragile absolute gains.
    - `scripts/tune_reasoning_profile.py` now exposes baseline tuning flags (`--baseline-report`, `--baseline-profile-path`, `--min-delta-pass-rate`, `--max-mcnemar-pvalue`, `--min-discordant-win-rate`, `--min-discordant-win-rate-ci95-lower`, `--min-comparable-cases`).
