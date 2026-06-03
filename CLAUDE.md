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
