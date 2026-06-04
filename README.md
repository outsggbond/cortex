# OmniEvolve

**Local-first, self-evolving, multi-modal AI agent system** — runs on user hardware, handles text/image/audio, continuously improves through online learning.

~66,000 lines of Python, 273 modules, layered architecture.

## Quick Start

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-minimal.txt

# Chat (DeepSeek)
.\.venv\Scripts\python.exe main.py --chat --cloud-llm --cloud-provider deepseek --cloud-model deepseek-v4-flash

# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -o "addopts=" -q
```

## Capabilities

| Module | Purpose |
|--------|---------|
| `system/chat_v2/` | Conversational AI: intent routing, memory, RAG, reasoning |
| `system/computer_use/` | Windows desktop automation: Win32 SendInput, OCR, pywinauto, plugins |
| `system/automation/` | Trigger-driven workflow execution + NanoBrain + federated learning |
| `system/knowledge/` | RAG, hybrid semantic index, Neo4j graph KB, FAQ retrieval |
| `system/learning/` | Online learning loop, dialogue evolution, continual learning |

CLI entry points:

| Flag | Function |
|------|----------|
| `--chat` | Interactive conversation |
| `--computer-use-goal` | Desktop task execution |
| `--automation-*` | Workflow automation |
| `--rag-ingest` | Knowledge base indexing |

## Architecture

```
main.py (bootstrap)
  └─ system/app_runtime.py          # Parse → route → execute
       ├─ system/runtime/             # TaskRouter, ExecutionEngine, StateManager, AdaptiveController
       ├─ system/domain/              # Business facades (chat / computer / automation)
       ├─ system/agent/               # Unified Task / Planner / Executor / State models
       ├─ system/interfaces/          # CLI / API / GUI adapters
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
       └─ system/l_utils/             # CLI support, codegen, files, metrics
```

**Layered design**: L1 infrastructure → L2 perception/knowledge → L3 reasoning/learning → L4 execution/interfaces.

## Self-Evolution Pipeline

```
user_input → Pipeline.respond()
  ├─ classify_intent → classify_route → _decide
  │    ├─ agent / reasoning / LLM / memory / RAG / FAQ / clarify / fallback
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

## Configuration

- `.env.local` — API keys and runtime overrides (auto-loaded on startup)
- `config/` — YAML/JSON: `auto_tune.json`, `policies.yaml`, `costs.yaml`, `contexts.yaml`
- `config/runtime_secrets.env` — alternative secrets file

Example `.env.local`:

```dotenv
DEEPSEEK_API_KEY=your_key
V2_LLM_PROVIDER=deepseek
V2_LLM_MODEL=deepseek-v4-flash
```

## Testing

```powershell
# Full suite (~57 passing, ~40 skipped for optional deps)
.\.venv\Scripts\python.exe -m pytest tests/ -o "addopts=" -q

# Individual verification scripts
.\.venv\Scripts\python.exe tests/chat_v2_cloud_test.py
.\.venv\Scripts\python.exe tests/runtime_bootstrap_local_env_test.py
.\.venv\Scripts\python.exe tests/chat_v2_runtime_test.py
.\.venv\Scripts\python.exe tests/interfaces_entrypoints_test.py
.\.venv\Scripts\python.exe scripts/verify_deepseek.py
```

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

## Conventions

1. **`main.py`** — bootstrap only, no business logic
2. **Domain facades** — stable entry points; implementations in subdirectories
3. **`system/` root** — only `app_runtime.py`, `main_cli.py`, `services.py`
4. **LLM clients** — extend `BaseLLMClient` from `system/chat_v2/llm.py`
5. **Every module connected** — no dead code; each file imported by ≥1 other module or is an entry point
6. **Optional deps** — torch, neo4j, playwright, pywinauto use try/except guards
7. **Imports** — always use full layered path (e.g. `system.core.hardware`, not `system.hardware`)
8. **Test flags** — `--maxfail=1` in pyproject.toml; override with `-o "addopts="`

## Enhancement Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed enhancement directions across all layers (L1 infrastructure through L4 execution, interfaces, and engineering systems).
