# -*- coding: utf-8 -*-
import os

def _register_chat_args(parser):
    parser.add_argument(
        "--runtime-arch",
        type=str,
        default=str(os.environ.get("RUNTIME_ARCH", "legacy")),
        choices=["legacy", "v2"],
        help="Select runtime architecture. 'legacy' uses app_runtime monolith; 'v2' uses modular chat pipeline.",
    )
    parser.add_argument(
        "--v2-memory-path",
        type=str,
        default=str(os.environ.get("V2_MEMORY_PATH", "artifacts/memory/dialogue_curated.jsonl")),
        help="Curated retrieval memory JSONL path for runtime-arch=v2.",
    )
    parser.add_argument(
        "--v2-llm-provider",
        "--cloud-provider",
        dest="v2_llm_provider",
        type=str,
        default=str(os.environ.get("V2_LLM_PROVIDER", "none")),
        help="LLM provider for runtime-arch=v2. Presets: openai, deepseek, kimi, compatible. Other values use generic OpenAI-compatible mode.",
    )
    parser.add_argument(
        "--v2-llm-model",
        "--cloud-model",
        dest="v2_llm_model",
        type=str,
        default=str(os.environ.get("V2_LLM_MODEL", "")),
        help="Model id used by v2 LLM provider. Leave empty to use the provider preset default when available.",
    )
    parser.add_argument(
        "--v2-llm-timeout-s",
        type=float,
        default=float(os.environ.get("V2_LLM_TIMEOUT_S", "45")),
        help="Request timeout seconds for v2 LLM provider.",
    )
    parser.add_argument(
        "--v2-llm-base-url",
        "--cloud-base-url",
        dest="v2_llm_base_url",
        type=str,
        default=str(os.environ.get("V2_LLM_BASE_URL", "")),
        help="Optional OpenAI-compatible base URL. Examples: https://api.deepseek.com or https://api.moonshot.cn/v1.",
    )
    parser.add_argument(
        "--v2-llm-endpoint",
        "--cloud-endpoint",
        dest="v2_llm_endpoint",
        type=str,
        default=str(os.environ.get("V2_LLM_ENDPOINT", "")),
        help="Optional custom endpoint for v2 LLM provider (for example /v1/chat/completions or /v1/responses). Overrides base URL when provided.",
    )
    parser.add_argument(
        "--v2-llm-api-key",
        "--cloud-api-key",
        dest="v2_llm_api_key",
        type=str,
        default=str(os.environ.get("V2_LLM_API_KEY", "")),
        help="Optional inline API key for v2 LLM provider. Prefer env vars in shared shells.",
    )
    parser.add_argument(
        "--v2-llm-api-key-env",
        "--cloud-api-key-env",
        dest="v2_llm_api_key_env",
        type=str,
        default=str(os.environ.get("V2_LLM_API_KEY_ENV", "") or os.environ.get("CLOUD_LLM_API_KEY_ENV", "")),
        help="Environment variable name used to resolve the v2 LLM API key. Leave empty to use the provider preset default.",
    )
    parser.add_argument(
        "--v2-llm-system-prompt",
        "--cloud-system-prompt",
        dest="v2_llm_system_prompt",
        type=str,
        default=str(os.environ.get("V2_LLM_SYSTEM_PROMPT", "")),
        help="Optional system prompt override for v2 API-backed chat.",
    )
    parser.add_argument(
        "--v2-llm-temperature",
        "--cloud-temperature",
        dest="v2_llm_temperature",
        type=float,
        default=float(os.environ.get("V2_LLM_TEMPERATURE", "0.2")),
        help="Sampling temperature used by v2 API-backed chat.",
    )
    parser.add_argument(
        "--cloud-llm",
        action="store_true",
        help="Shortcut for API-backed chat: forces runtime-arch=v2 and keeps the selected provider preset.",
    )
    parser.add_argument(
        "--v2-no-memory-retrieval",
        action="store_true",
        help="Disable retrieval fallback in runtime-arch=v2.",
    )
    parser.add_argument(
        "--v2-agent",
        action="store_true",
        help="Enable conservative workspace agent mode in runtime-arch=v2 chat.",
    )
    parser.add_argument(
        "--v2-agent-project-root",
        type=str,
        default=str(os.environ.get("V2_AGENT_PROJECT_ROOT", ".")),
        help="Workspace root used by the v2 chat agent for file inspection actions.",
    )
    parser.add_argument(
        "--v2-agent-allow-write",
        action="store_true",
        help="Allow non-destructive write actions in v2 chat agent mode.",
    )
    parser.add_argument(
        "--v2-agent-allow-exec",
        action="store_true",
        help="Allow script/test execution actions in v2 chat agent mode.",
    )
    parser.add_argument(
        "--v2-agent-allow-browser",
        action="store_true",
        help="Allow OpenAI-compatible browser control actions in v2 chat agent mode.",
    )
    parser.add_argument(
        "--v2-agent-allow-desktop",
        action="store_true",
        help="Allow desktop window/input/screenshot actions in v2 chat agent mode.",
    )
    parser.add_argument("--train-file", type=str, default="")
    parser.add_argument("--no-faq", action="store_true")
    parser.add_argument("--neo4j-uri", type=str, default="")
    parser.add_argument("--neo4j-user", type=str, default="")
    parser.add_argument("--neo4j-password", type=str, default="")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--replan-max", type=int, default=1)
    parser.add_argument("--planner-candidates", type=int, default=3)
    parser.add_argument("--require-embeddings", action="store_true")
    parser.add_argument(
        "--self-heal",
        dest="self_heal",
        action="store_true",
        default=None,
        help="Enable agent self-healing repairs in runtime-arch=v2.",
    )
    parser.add_argument(
        "--no-self-heal",
        dest="self_heal",
        action="store_false",
        help="Disable agent self-healing repairs in runtime-arch=v2.",
    )
    parser.add_argument("--self-heal-create", action="store_true")
    parser.add_argument("--self-heal-chmod", action="store_true")
    parser.add_argument(
        "--search-plan",
        dest="search_plan",
        action="store_true",
        default=None,
        help="Enable search-based planning in runtime-arch=v2 agent mode.",
    )
    parser.add_argument(
        "--no-search-plan",
        dest="search_plan",
        action="store_false",
        help="Disable search-based planning in runtime-arch=v2 agent mode.",
    )
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--search-beam", type=int, default=4)
    parser.add_argument("--experience-store", type=str, default="artifacts/memory/experience_store.jsonl")
    parser.add_argument("--experience-topk", type=int, default=2)
    parser.add_argument("--experience-disable", action="store_true")
    parser.add_argument("--policy", type=str, default="")
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--empathy", type=float, default=None)
    parser.add_argument("--empathy-mode", type=str, choices=["offset", "absolute"], default="offset")
    parser.add_argument("--auto-tune", action="store_true", help="Enable hardware-aware auto tuning")
    parser.add_argument("--no-auto-tune", action="store_true", help="Disable hardware-aware auto tuning")
    parser.add_argument("--auto-tune-config", type=str, default="config/auto_tune.json", help="Auto tuning config path")
    parser.add_argument(
        "--runtime-adapt",
        dest="runtime_adapt_enabled",
        action="store_true",
        default=None,
        help="Enable runtime closed-loop hardware adaptation",
    )
    parser.add_argument(
        "--no-runtime-adapt",
        dest="runtime_adapt_enabled",
        action="store_false",
        help="Disable runtime closed-loop hardware adaptation",
    )
    parser.add_argument(
        "--runtime-adapt-sample-interval-s",
        type=float,
        default=None,
        help="Runtime adaptive sampling interval in seconds",
    )
    parser.add_argument(
        "--runtime-adapt-cooldown-s",
        type=float,
        default=None,
        help="Minimum seconds between runtime adaptive config updates",
    )
    parser.add_argument(
        "--runtime-adapt-pressure-high",
        type=float,
        default=None,
        help="Pressure threshold to trigger degradation",
    )
    parser.add_argument(
        "--runtime-adapt-pressure-low",
        type=float,
        default=None,
        help="Pressure threshold to trigger recovery",
    )
    parser.add_argument(
        "--runtime-adapt-heavy-route-min-score",
        type=float,
        default=None,
        help="Minimum resource score to keep heavy (transformer) route preferred",
    )
    parser.add_argument(
        "--runtime-adapt-heavy-route-max-pressure",
        type=float,
        default=None,
        help="Maximum pressure to keep heavy (transformer) route preferred",
    )
    parser.add_argument(
        "--runtime-adapt-bandit",
        dest="runtime_adapt_bandit_enabled",
        action="store_true",
        default=None,
        help="Enable lightweight bandit learning for runtime adaptation",
    )
    parser.add_argument(
        "--runtime-adapt-no-bandit",
        dest="runtime_adapt_bandit_enabled",
        action="store_false",
        help="Disable lightweight bandit learning for runtime adaptation",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-explore",
        type=float,
        default=None,
        help="Bandit exploration strength (UCB factor)",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-auto-explore",
        dest="runtime_adapt_bandit_auto_explore",
        action="store_true",
        default=None,
        help="Enable adaptive tuning of bandit exploration strength",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-auto-explore",
        dest="runtime_adapt_bandit_auto_explore",
        action="store_false",
        help="Disable adaptive tuning of bandit exploration strength",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-explore-min",
        type=float,
        default=None,
        help="Minimum bandit exploration strength when auto-explore is enabled",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-explore-max",
        type=float,
        default=None,
        help="Maximum bandit exploration strength when auto-explore is enabled",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-explore-step-up",
        type=float,
        default=None,
        help="Bandit exploration increment when recent reward is poor",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-explore-step-down",
        type=float,
        default=None,
        help="Bandit exploration decrement when recent reward is stable and good",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-contextual-bias",
        dest="runtime_adapt_bandit_contextual_bias",
        action="store_true",
        default=None,
        help="Enable contextual bandit bias from live pressure/resource signals",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-contextual-bias",
        dest="runtime_adapt_bandit_contextual_bias",
        action="store_false",
        help="Disable contextual bandit bias from live pressure/resource signals",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-context-pressure-weight",
        type=float,
        default=None,
        help="Contextual bandit pressure weight (higher favors conservative arms under load)",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-context-score-weight",
        type=float,
        default=None,
        help="Contextual bandit resource-score weight (higher favors aggressive arms on healthy hardware)",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-context-bias-cap",
        type=float,
        default=None,
        help="Absolute cap for contextual bandit bias term",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-context-cold-start",
        dest="runtime_adapt_bandit_context_cold_start",
        action="store_true",
        default=None,
        help="Enable contextual cold-start arm pick before bandit has enough samples",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-context-cold-start",
        dest="runtime_adapt_bandit_context_cold_start",
        action="store_false",
        help="Disable contextual cold-start arm pick before bandit has enough samples",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-context-cold-start-gain",
        type=float,
        default=None,
        help="Gain for mapping headroom(score-pressure) to cold-start target multiplier",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-hardware-guard",
        dest="runtime_adapt_bandit_hardware_guard",
        action="store_true",
        default=None,
        help="Enable hardware-signature guard for bandit warm-start state",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-hardware-guard",
        dest="runtime_adapt_bandit_hardware_guard",
        action="store_false",
        help="Disable hardware-signature guard for bandit warm-start state",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-hardware-strict",
        dest="runtime_adapt_bandit_hardware_strict",
        action="store_true",
        default=None,
        help="Reject bandit warm-start state when hardware signature mismatches",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-hardware-strict",
        dest="runtime_adapt_bandit_hardware_strict",
        action="store_false",
        help="Allow loading mismatched bandit state with decay instead of hard reject",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-hardware-mismatch-count-decay",
        type=float,
        default=None,
        help="Count decay ratio applied when loading mismatched hardware bandit state",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-hardware-mismatch-value-decay",
        type=float,
        default=None,
        help="Value decay ratio applied when loading mismatched hardware bandit state",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-hardware-mismatch-explore-boost",
        type=float,
        default=None,
        help="Explore boost applied when loading mismatched hardware bandit state",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-time-decay",
        dest="runtime_adapt_bandit_time_decay",
        action="store_true",
        default=None,
        help="Enable time-based decay of stale bandit memory",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-time-decay",
        dest="runtime_adapt_bandit_time_decay",
        action="store_false",
        help="Disable time-based decay of stale bandit memory",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-time-decay-half-life-s",
        type=float,
        default=None,
        help="Half-life in seconds for bandit time decay",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-time-decay-min-factor",
        type=float,
        default=None,
        help="Minimum decay factor applied during bandit time decay",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-time-decay-cooldown-s",
        type=float,
        default=None,
        help="Minimum seconds between consecutive bandit time-decay applications",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-arm-cooldown",
        dest="runtime_adapt_bandit_arm_cooldown",
        action="store_true",
        default=None,
        help="Enable per-arm cooldown after repeated catastrophic rewards",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-arm-cooldown",
        dest="runtime_adapt_bandit_arm_cooldown",
        action="store_false",
        help="Disable per-arm cooldown after repeated catastrophic rewards",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-arm-cooldown-reward-threshold",
        type=float,
        default=None,
        help="Reward threshold used to count catastrophic events for per-arm cooldown",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-arm-cooldown-trigger-streak",
        type=int,
        default=None,
        help="Catastrophic streak count before cooling down the active arm",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-arm-cooldown-s",
        type=float,
        default=None,
        help="Cooldown duration in seconds for a cooled arm",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-all-cooled-safe-fallback",
        dest="runtime_adapt_bandit_all_cooled_safe_fallback",
        action="store_true",
        default=None,
        help="When all arms are cooling down, fallback to the most conservative arm set",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-all-cooled-safe-fallback",
        dest="runtime_adapt_bandit_all_cooled_safe_fallback",
        action="store_false",
        help="Disable conservative fallback when all bandit arms are cooling down",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-arm-risk-aware",
        dest="runtime_adapt_bandit_arm_risk_aware",
        action="store_true",
        default=None,
        help="Enable per-arm risk-aware penalty during bandit arm selection",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-arm-risk-aware",
        dest="runtime_adapt_bandit_arm_risk_aware",
        action="store_false",
        help="Disable per-arm risk-aware penalty during bandit arm selection",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-arm-risk-alpha",
        type=float,
        default=None,
        help="EMA alpha for per-arm risk statistics",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-arm-risk-weight",
        type=float,
        default=None,
        help="Weight of per-arm reward variance in risk penalty",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-arm-risk-neg-weight",
        type=float,
        default=None,
        help="Weight of negative per-arm reward EMA in risk penalty",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-arm-risk-cap",
        type=float,
        default=None,
        help="Maximum per-arm risk penalty applied in bandit scoring",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-safety-guard",
        dest="runtime_adapt_bandit_safety_guard",
        action="store_true",
        default=None,
        help="Enable safety guard that caps aggressive bandit arms under dangerous hardware signals",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-safety-guard",
        dest="runtime_adapt_bandit_safety_guard",
        action="store_false",
        help="Disable safety guard for aggressive bandit arm capping",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-safety-pressure-gate",
        type=float,
        default=None,
        help="Pressure gate for activating bandit safety guard",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-safety-score-gate",
        type=float,
        default=None,
        help="Resource-score gate for activating bandit safety guard",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-safety-max-multiplier",
        type=float,
        default=None,
        help="Maximum multiplier allowed by bandit safety guard when active",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-uncertainty-guard",
        dest="runtime_adapt_bandit_uncertainty_guard",
        action="store_true",
        default=None,
        help="Enable high-uncertainty guard to cap aggressive arms under volatile rewards",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-uncertainty-guard",
        dest="runtime_adapt_bandit_uncertainty_guard",
        action="store_false",
        help="Disable high-uncertainty guard",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-uncertainty-var-gate",
        type=float,
        default=None,
        help="Reward-variance EMA gate for uncertainty guard activation",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-uncertainty-min-samples",
        type=int,
        default=None,
        help="Minimum total bandit samples before uncertainty guard can activate",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-uncertainty-pressure-gate",
        type=float,
        default=None,
        help="Pressure gate for uncertainty guard activation",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-uncertainty-score-gate",
        type=float,
        default=None,
        help="Resource-score gate for uncertainty guard activation",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-uncertainty-max-multiplier",
        type=float,
        default=None,
        help="Maximum allowed multiplier while uncertainty guard is active",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-regime-split",
        dest="runtime_adapt_bandit_regime_split",
        action="store_true",
        default=None,
        help="Enable normal/stress split memory for runtime bandit learning",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-regime-split",
        dest="runtime_adapt_bandit_regime_split",
        action="store_false",
        help="Disable normal/stress split memory for runtime bandit learning",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-regime-pressure-gate",
        type=float,
        default=None,
        help="Pressure gate for switching bandit memory to stress regime",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-regime-score-gate",
        type=float,
        default=None,
        help="Resource-score gate for switching bandit memory to stress regime",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-shock-guard",
        dest="runtime_adapt_bandit_shock_guard",
        action="store_true",
        default=None,
        help="Enable catastrophic-feedback shock guard for temporary conservative bandit capping",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-shock-guard",
        dest="runtime_adapt_bandit_shock_guard",
        action="store_false",
        help="Disable catastrophic-feedback shock guard",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-shock-reward-threshold",
        type=float,
        default=None,
        help="Reward threshold considered catastrophic for shock guard triggering",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-shock-trigger-count",
        type=int,
        default=None,
        help="Number of catastrophic rewards inside shock window required to trigger guard",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-shock-window-s",
        type=float,
        default=None,
        help="Window size in seconds for counting catastrophic rewards",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-shock-cooldown-s",
        type=float,
        default=None,
        help="Shock guard active duration in seconds after trigger",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-shock-max-multiplier",
        type=float,
        default=None,
        help="Maximum allowed multiplier while shock guard is active",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-spike-guard",
        dest="runtime_adapt_bandit_spike_guard",
        action="store_true",
        default=None,
        help="Enable pressure-spike guard for temporary conservative bandit capping",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-no-spike-guard",
        dest="runtime_adapt_bandit_spike_guard",
        action="store_false",
        help="Disable pressure-spike guard for runtime bandit",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-spike-pressure-delta-threshold",
        type=float,
        default=None,
        help="Minimum pressure jump (against EMA baseline) required to trigger spike guard",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-spike-score-drop-threshold",
        type=float,
        default=None,
        help="Minimum resource-score drop (against EMA baseline) required to trigger spike guard",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-spike-min-pressure",
        type=float,
        default=None,
        help="Minimum absolute pressure required before spike guard can trigger",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-spike-cooldown-s",
        type=float,
        default=None,
        help="Duration in seconds for keeping spike guard active after trigger",
    )
    parser.add_argument(
        "--runtime-adapt-bandit-spike-max-multiplier",
        type=float,
        default=None,
        help="Maximum allowed multiplier while spike guard is active",
    )
    parser.add_argument(
        "--runtime-adapt-state-path",
        type=str,
        default=None,
        help="Runtime adaptive state snapshot path (set empty to disable snapshots)",
    )
    parser.add_argument(
        "--runtime-adapt-load-state",
        dest="runtime_adapt_load_state",
        action="store_true",
        default=None,
        help="Load runtime adaptive snapshot at startup",
    )
    parser.add_argument(
        "--runtime-adapt-no-load-state",
        dest="runtime_adapt_load_state",
        action="store_false",
        help="Disable runtime adaptive snapshot restore at startup",
    )
    parser.add_argument(
        "--runtime-adapt-save-interval-s",
        type=float,
        default=None,
        help="Minimum seconds between runtime adaptive snapshot writes",
    )
    parser.add_argument(
        "--runtime-adapt-dynamic-interval",
        dest="runtime_adapt_dynamic_interval",
        action="store_true",
        default=None,
        help="Enable pressure-aware dynamic runtime adaptive sampling interval",
    )
    parser.add_argument(
        "--runtime-adapt-no-dynamic-interval",
        dest="runtime_adapt_dynamic_interval",
        action="store_false",
        help="Disable pressure-aware dynamic runtime adaptive sampling interval",
    )
    parser.add_argument(
        "--runtime-adapt-interval-min-s",
        type=float,
        default=None,
        help="Minimum effective runtime adaptive sampling interval in seconds",
    )
    parser.add_argument(
        "--runtime-adapt-interval-max-s",
        type=float,
        default=None,
        help="Maximum effective runtime adaptive sampling interval in seconds",
    )
    parser.add_argument(
        "--runtime-adapt-interval-high-mul",
        type=float,
        default=None,
        help="Sampling interval multiplier under high pressure",
    )
    parser.add_argument(
        "--runtime-adapt-interval-low-mul",
        type=float,
        default=None,
        help="Sampling interval multiplier under low pressure",
    )
    parser.add_argument(
        "--runtime-adapt-interval-throttle-mul",
        type=float,
        default=None,
        help="Sampling interval multiplier while throttle is active",
    )
    parser.add_argument(
        "--runtime-adapt-threshold-auto",
        dest="runtime_adapt_threshold_auto",
        action="store_true",
        default=None,
        help="Enable automatic pressure threshold self-calibration",
    )
    parser.add_argument(
        "--runtime-adapt-no-threshold-auto",
        dest="runtime_adapt_threshold_auto",
        action="store_false",
        help="Disable automatic pressure threshold self-calibration",
    )
    parser.add_argument(
        "--runtime-adapt-threshold-min-high",
        type=float,
        default=None,
        help="Minimum pressure_high during threshold self-calibration",
    )
    parser.add_argument(
        "--runtime-adapt-threshold-max-high",
        type=float,
        default=None,
        help="Maximum pressure_high during threshold self-calibration",
    )
    parser.add_argument(
        "--runtime-adapt-threshold-gap",
        type=float,
        default=None,
        help="Target gap between pressure_high and pressure_low in threshold self-calibration",
    )
    parser.add_argument(
        "--runtime-adapt-threshold-step-up",
        type=float,
        default=None,
        help="Upward step for pressure_high when resources are consistently healthy",
    )
    parser.add_argument(
        "--runtime-adapt-threshold-step-down",
        type=float,
        default=None,
        help="Downward step for pressure_high when pressure is sustained high",
    )
    parser.add_argument(
        "--runtime-transformer-hot-tune",
        dest="runtime_transformer_hot_tune",
        action="store_true",
        default=None,
        help="Enable runtime hot tuning of transformer generation parameters",
    )
    parser.add_argument(
        "--runtime-transformer-no-hot-tune",
        dest="runtime_transformer_hot_tune",
        action="store_false",
        help="Disable runtime hot tuning of transformer generation parameters",
    )
    parser.add_argument(
        "--runtime-planner-hot-tune",
        dest="runtime_planner_hot_tune",
        action="store_true",
        default=None,
        help="Enable runtime hot tuning for search planner depth/beam",
    )
    parser.add_argument(
        "--runtime-planner-no-hot-tune",
        dest="runtime_planner_hot_tune",
        action="store_false",
        help="Disable runtime hot tuning for search planner depth/beam",
    )
    parser.add_argument(
        "--runtime-planner-depth-min",
        type=int,
        default=None,
        help="Minimum planner depth for runtime planner hot tuning",
    )
    parser.add_argument(
        "--runtime-planner-depth-max",
        type=int,
        default=None,
        help="Maximum planner depth for runtime planner hot tuning",
    )
    parser.add_argument(
        "--runtime-planner-beam-min",
        type=int,
        default=None,
        help="Minimum planner beam size for runtime planner hot tuning",
    )
    parser.add_argument(
        "--runtime-planner-beam-max",
        type=int,
        default=None,
        help="Maximum planner beam size for runtime planner hot tuning",
    )
    parser.add_argument("--generative-only", action="store_true", help="Use only generative response mode")
    parser.add_argument("--allow-matching", action="store_true", help="Allow matching/template/FAQ response")
    parser.add_argument("--deterministic", action="store_true", help="Disable stochastic sampling")
    parser.add_argument("--sampling", action="store_true", help="Enable stochastic sampling")

    parser.add_argument("--train-dialogue", action="store_true", help="Train dialogue subsystem")
    parser.add_argument("--dialogue-data", type=str, default="train.txt", help="Dialogue data file")
    parser.add_argument("--dialogue-epochs", type=int, default=10, help="Dialogue training epochs")
    parser.add_argument("--dialogue-lr", type=float, default=0.01, help="Dialogue learning rate")
    parser.add_argument("--eval-dialogue", action="store_true", help="Evaluate dialogue subsystem")
    parser.add_argument("--train-gnn", action="store_true", help="Train GNN language model")
    parser.add_argument("--gnn-data", type=str, default="train.txt", help="GNN training data file")
    parser.add_argument("--gnn-epochs", type=int, default=3, help="GNN epochs")
    parser.add_argument("--gnn-steps", type=int, default=40, help="GNN online steps per epoch")
    parser.add_argument("--augment-dialogue", action="store_true", help="Enable dialogue data augmentation")
    parser.add_argument("--augment-factor", type=int, default=2, help="Augmented samples per item")
    parser.add_argument(
        "--augment-strength",
        type=str,
        choices=["light", "strong", "mixed"],
        default="mixed",
        help="Rewrite strength",
    )
    parser.add_argument("--augment-seed", type=int, default=7, help="Augmentation seed")
    parser.add_argument("--augment-save", type=str, default="", help="Optional path to save augmented data")
    parser.add_argument(
        "--dialogue-transformer-model",
        type=str,
        default="",
        help="Local transformer model path for dialogue semantics",
    )
    parser.add_argument(
        "--dialogue-transformer-device",
        type=str,
        default="",
        help="Device for dialogue transformer (cpu/cuda)",
    )
    parser.add_argument(
        "--dialogue-transformer-max-tokens",
        type=int,
        default=256,
        help="Max input tokens for dialogue transformer",
    )
    parser.add_argument(
        "--dialogue-transformer-local-only",
        action="store_true",
        help="Load dialogue transformer from local files only",
    )
    parser.add_argument(
        "--dialogue-transformer-allow-download",
        action="store_true",
        help="Allow downloading dialogue transformer assets",
    )

    parser.add_argument("--transformer-model", type=str, default="", help="Local transformer generation model")
    parser.add_argument(
        "--transformer-mode",
        type=str,
        choices=["primary", "polish", "off"],
        default="primary",
        help="primary=main generator, polish=refiner only, off=disabled",
    )
    parser.add_argument("--transformer-max-new", type=int, default=96, help="Max new tokens")
    parser.add_argument("--transformer-max-input", type=int, default=0, help="Max input tokens (0=auto)")
    parser.add_argument("--transformer-temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--transformer-top-p", type=float, default=0.9, help="Sampling top-p")
    parser.add_argument("--transformer-top-k", type=int, default=40, help="Sampling top-k")
    parser.add_argument("--transformer-repeat-penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--transformer-device", type=str, default="", help="Transformer device")
    parser.add_argument("--evolution-eval", action="store_true", help="Enable evolution evaluator output")
    parser.add_argument("--no-evolution-eval", action="store_true", help="Disable evolution evaluator output")
    parser.add_argument(
        "--transformer-trust-remote-code",
        action="store_true",
        help="Allow trust_remote_code when loading local transformer model",
    )
    parser.add_argument("--rag-enable", action="store_true", help="Enable RAG")
    parser.add_argument("--rag-disable", action="store_true", help="Disable RAG")
    parser.add_argument("--rag-index", type=str, default="artifacts/memory/rag_index.json", help="RAG index path")
    parser.add_argument("--rag-ingest", action="store_true", help="Build RAG index then exit")
    parser.add_argument("--rag-paths", nargs="*", default=[], help="Paths for RAG ingestion")
    parser.add_argument("--rag-topk", type=int, default=4, help="RAG top-k")
    parser.add_argument("--rag-min-sim", type=float, default=0.2, help="RAG min similarity")
    parser.add_argument("--rag-max-chars", type=int, default=1200, help="RAG max context chars")
    parser.add_argument("--rag-chunk-size", type=int, default=800, help="RAG chunk size")
    parser.add_argument("--rag-chunk-overlap", type=int, default=120, help="RAG chunk overlap")
    parser.add_argument("--rag-max-files", type=int, default=2000, help="RAG max files")

