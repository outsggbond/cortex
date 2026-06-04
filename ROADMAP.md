# Enhancement Roadmap

## L1 — Infrastructure

### `system/core/`

| File | Current | Potential |
|------|---------|-----------|
| `hardware.py` | CPU/RAM/GPU detection | USB device enumeration, disk health, battery status, temperature sensors, external display detection |
| `embeddings.py` | Text embedding vectors | Multilingual embeddings, image embeddings, CLIP multimodal embeddings, local small model replacement |
| `model_loader.py` | Transformer model loading | ONNX/TensorRT acceleration, quantized model loading, LoRA hot-swap |
| `state_encoder.py` | State encoding | Temporal state encoding (history sequences), graph state encoding |
| `planner.py` | Planning primitives | Hierarchical planning, parallel subtasks, conditional branching, plan fallback strategies |
| `error_normalizer.py` | Error normalization | Error classification (recoverable/unrecoverable/manual), error trend analysis |

### `system/runtime/`

| File | Current | Potential |
|------|---------|-----------|
| `bootstrap.py` | Startup config | Plugin-based startup, on-demand loading, startup performance optimization |
| `support.py` | Env var overrides | Config file hot-reload, remote config pull |
| `artifacts.py` | Artifact maintenance/cleanup | Artifact versioning, integrity checks, cloud storage sync |
| `adaptive.py` | Hardware adaptive controller | Multi-dimension adaptation (network bandwidth, disk IO, battery), strategy pre-learning |
| `task_router.py` | Task type routing | Dynamic routing (load-based domain selection), priority queues, task dependencies |
| `execution_engine.py` | Unified task execution | Async task execution, task pause/resume, task timeout control |
| `state_manager.py` | Task state persistence | Parent-child task relations, task lineage tracking, state snapshot comparison |

### `system/l_utils/`

| File | Current | Potential |
|------|---------|-----------|
| `loader.py` | Module loading | Plugin discovery, hot-load/unload |
| `codegen.py` | Code generation | Template engine upgrade, multi-language code generation |
| `files.py` | File operations | Large file streaming, file change monitoring, hash verification |
| `metrics.py` | Metric statistics | Prometheus metrics export, custom dashboard data |
| `cli_support.py` | CLI utilities | Auto-completion, interactive wizards, colored output beautification |
| `graphviz_export.py` | Graph export | Real-time dependency graph visualization, state machine diagram export |

---

## L2 — Perception & Knowledge

### `system/perception/`

| Current | Potential |
|---------|-----------|
| Visual preprocessing, OCR alignment, UI tree fusion | YOLO/SAM for pixel-level UI element segmentation |
| | Grounding DINO for natural language grounding ("the red button") |
| | Video stream understanding (screen recording replay analysis) |
| | Local multimodal VLM deployment for on-device visual understanding |
| | Cross-application UI pattern recognition ("this is a login form") |

### `system/knowledge/`

| File | Current | Potential |
|------|---------|-----------|
| `rag_store.py` | RAG index retrieval | Hybrid retrieval (sparse + dense), multimodal RAG (image search), conversational retrieval, incremental index updates |
| `graph/kb.py` | Knowledge base | Automatic knowledge extraction (extract workflows from operation logs), temporal knowledge graph |
| `graph/query.py` | Query engine | Natural language to graph queries, graph reasoning (multi-hop) |
| `graph/repl.py` | Interactive REPL | Visual query results, query history |
| `semantic/index.py` | Hybrid semantic index | Multilingual semantic alignment, cross-modal semantic index |
| `semantic/faq.py` | FAQ retrieval | Proactive FAQ generation (extract common issues from logs), FAQ auto-update |
| `memory.py` | In-memory + vector DB | Long-term/short-term memory separation, memory forgetting curves, memory priority ranking |

---

## L3 — Reasoning & Learning

### `system/brain/`

| File | Current | Potential |
|------|---------|-----------|
| `reasoning.py` | Hierarchical reasoning | Chain-of-Thought visualization, multi-step reasoning backtracking, reasoning explanation generation |
| `neural_reasoner.py` | Neural reasoning | Stronger local model integration, reasoning + retrieval hybrid |
| `debate_reasoner.py` | Debate-style reasoning | Multi-model debate (different models challenge each other), debate result confidence |
| `neurosymbolic.py` | Neuro-symbolic hybrid | Explainable rule extraction, rule base visual editor |
| `profile.py` | Reasoning config | Per-scenario auto-switching config, A/B testing framework |
| `world_model.py` | World state modeling | Cross-device world model, temporal prediction (what's the next state) |
| `world_predictor.py` | State prediction | Multi-step prediction, prediction uncertainty quantification |
| `stimulus_engine.py` | Stimulus & response | Proactive triggering (not passive command-waiting), context-aware push |

### `system/learning/`

| File | Current | Potential |
|------|---------|-----------|
| `continual.py` | Continual learning framework | Online learning (learn while using), catastrophic forgetting prevention |
| `feedback.py` | Feedback processing | Implicit feedback extraction (infer quality from operation results), feedback weighting |
| `loader.py` | Training data loading | Streaming loading, multi-format support (Parquet, Arrow) |
| `dialogue/` series | Dialogue generation/augmentation/evolution | Multimodal dialogue data (with images), role consistency constraints |

### `system/strategy/`

| File | Current | Potential |
|------|---------|-----------|
| `strategy.py` | Strategy generation | Multi-strategy comparison, strategy simulation evaluation |
| `task_flow.py` | Task flow | Visual flow editor, flow template marketplace |
| `search_planner.py` | Search planning | Monte Carlo tree search, A* planning, plan cache reuse |

---

## L4 — Execution & Applications

### `system/chat_v2/`

| File | Current | Potential |
|------|---------|-----------|
| `intents.py` | Intent classification | Multilingual intents, compound intent decomposition, configurable confidence thresholds |
| `llm.py` | LLM adapter layer | More model support (Claude, Gemini, local models), hot-swap models, streaming output |
| `memory.py` | Read-only retrieval | Multi-turn conversation context management, session summarization |
| `pipeline.py` | Single-path pipeline | Pluggable pipeline nodes, custom pipeline orchestration |
| `runtime.py` | Chat loop | Multi-session management, session resume, multimodal input (voice, images) |

### `system/computer_use/`

| File | Current | Potential |
|------|---------|-----------|
| `runtime.py` | Desktop agent main loop | Multi-window parallel ops, configurable operation speed, batch operations |
| `llm.py` | Decision LLM | Local model replacement for cloud, multi-model voting decisions |
| `plugins.py` | Desktop plugins | More app plugins (Office, WeChat Work, DingTalk), plugin hot-loading |
| `action/generator.py` | Action generation | Action template learning, compound actions (drag + keyboard) |
| `action/validator.py` | Action validation | Visual comparison verification, expected state assertions |
| `action/stats.py` | Action statistics | Per-app success rate stats, time consumption stats |
| `trace/audit.py` | Audit logs | Structured audit log queries, anomaly detection |
| `trace/visual.py` | Trace visualization | Operation replay, step comparison |

### `system/automation/`

| File | Current | Potential |
|------|---------|-----------|
| `runtime.py` | Workflow runtime | Cron expression triggers, event triggers (file change / network event) |
| `executor.py` | Step execution | Parallel steps, conditional branches, loop steps, sub-workflow calls |

### `system/evaluation/`

| File | Current | Potential |
|------|---------|-----------|
| `evaluator.py` | Performance evaluation | Multi-dimension scoring (speed/accuracy/resource), trend reports |
| `reflect.py` | Self-reflection | Automatic failure pattern clustering, reflection knowledge accumulation |
| `self_heal.py` | Self-healing | Auto-rollback config, auto-switch to fallback strategies |
| `response_guard.py` | Response guard | Content safety expansion (multilingual), sensitive info filtering |

---

## Unified Abstraction Layer

### `system/agent/`

| File | Current | Potential |
|------|---------|-----------|
| `task.py` | Task model | Task priority, deadlines, tags, custom attributes |
| `planner.py` | Planner facade | Plan serialization (save/restore), plan versioning |
| `executor.py` | Executor facade | Executor plugin mechanism, executor performance monitoring |
| `state.py` | State model | State history chain, state change events, state validation |

---

## Interfaces

### `system/interfaces/`

| Directory | Current | Potential |
|-----------|---------|-----------|
| `cli/` | CLI arg registration | Interactive CLI, script mode (batch execution), output formatting (JSON/table/plain text) |
| `api/` | HTTP API adapter | RESTful API completion, WebSocket real-time push, auto-generated OpenAPI docs |
| `gui/` | GUI session adapter | Web UI, desktop tray app, mobile control panel |

---

## Engineering Systems

| Area | Potential |
|------|-----------|
| Testing | End-to-end integration tests, performance benchmarks, chaos testing (random fault injection) |
| CI/CD | Auto-release, version number management, auto-generated changelogs |
| Monitoring | Runtime metrics dashboard, alerting rules, operation replay audit |
| Security | Operation permission tiers, sensitive operation re-confirmation, operation sandboxing |

---

## User-Requested Directions

| Request | Target |
|---------|--------|
| Deep OS integration | L1 `core/hardware.py` + new L4 `system/system_control/` |
| USB phone detection | L1 `core/hardware.py` USB enumeration extension |
| Phone operation | New L4 `system/phone_use/`, mirroring `computer_use/` |
| Cross-device coordination | L3 `agent/` parent-child tasks + `state_manager.py` lineage tracking |
| MCP protocol integration | L2 `knowledge/` or new L4 adapter module |
| Local model replacement | L3 `brain/reasoning.py` + L1 `model_loader.py` |
