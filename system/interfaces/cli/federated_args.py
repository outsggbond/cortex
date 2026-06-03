# -*- coding: utf-8 -*-
import os
import sys

def _register_federated_args(parser):
    parser.add_argument(
        "--federated-train",
        action="store_true",
        help="Run scripts/train_adapter.py in federated mode, then exit",
    )
    parser.add_argument(
        "--federated",
        action="store_true",
        help="Alias for --federated-train",
    )
    parser.add_argument(
        "--federated-script",
        type=str,
        default="scripts/train_adapter.py",
        help="Federated training entry script path",
    )
    parser.add_argument(
        "--federated-python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter for federated training entry",
    )
    parser.add_argument(
        "--federated-auto",
        action="store_true",
        help="Enable periodic federated training auto-trigger in runtime loops",
    )
    parser.add_argument(
        "--federated-auto-interval-s",
        type=float,
        default=None,
        help="Auto-trigger federated training after this interval in seconds (0 disables interval trigger)",
    )
    parser.add_argument(
        "--federated-auto-min-train-growth",
        type=int,
        default=None,
        help="Auto-trigger when train-data non-empty line growth reaches this value (0 disables growth trigger)",
    )
    parser.add_argument(
        "--federated-auto-cooldown-s",
        type=float,
        default=None,
        help="Minimum seconds between federated auto-trigger attempts",
    )
    parser.add_argument(
        "--federated-auto-min-free-gb",
        type=float,
        default=None,
        help="Skip federated auto-trigger when free disk is below this GB threshold (0 disables)",
    )
    parser.add_argument(
        "--federated-auto-disk-path",
        type=str,
        default="",
        help="Disk path used to evaluate free-space condition for federated auto-trigger",
    )
    parser.add_argument(
        "--federated-auto-max-runs",
        type=int,
        default=None,
        help="Optional max successful federated auto-train runs per process (0 means unlimited)",
    )
    parser.add_argument(
        "--federated-auto-state-path",
        type=str,
        default="",
        help="State file for federated auto-trigger cooldown/growth tracking",
    )
    parser.add_argument(
        "--federated-auto-events-path",
        type=str,
        default="",
        help="Optional JSONL event log path for federated auto-trigger lifecycle",
    )
    parser.add_argument(
        "--federated-auto-log-dir",
        type=str,
        default="",
        help="Optional directory for async federated auto-train subprocess logs",
    )
    parser.add_argument(
        "--federated-auto-lock-path",
        type=str,
        default="",
        help="Optional cross-process lock file path for federated auto-train exclusivity",
    )
    parser.add_argument(
        "--federated-auto-lock-stale-s",
        type=float,
        default=None,
        help="Treat federated auto lock as stale after this many seconds (0 disables stale timeout)",
    )
    parser.add_argument(
        "--federated-auto-lock-skip-log-interval-s",
        type=float,
        default=None,
        help="Minimum interval between lock-busy skip logs/events (0 disables throttling)",
    )
    parser.add_argument(
        "--federated-auto-async",
        action="store_true",
        help="Run federated auto-train in background subprocess mode",
    )
    parser.add_argument(
        "--federated-auto-sync",
        action="store_true",
        help="Force blocking federated auto-train mode",
    )
    parser.add_argument(
        "--federated-auto-fail-backoff-mult",
        type=float,
        default=None,
        help="Exponential backoff multiplier on consecutive auto-train failures",
    )
    parser.add_argument(
        "--federated-auto-fail-backoff-max-s",
        type=float,
        default=None,
        help="Cap for exponential failure backoff cooldown in seconds",
    )
    parser.add_argument(
        "--federated-auto-max-fail-streak",
        type=int,
        default=None,
        help="Stop auto-trigger attempts after this many consecutive failures (0 disables)",
    )
    parser.add_argument(
        "--federated-auto-max-runtime-s",
        type=float,
        default=None,
        help="Terminate async federated auto-train run when runtime exceeds this threshold in seconds (0 disables)",
    )
    parser.add_argument(
        "--federated-auto-no-resume",
        action="store_true",
        help="Disable implicit --fed-resume default in federated auto mode",
    )
    parser.add_argument(
        "--federated-auto-default-output-dir",
        type=str,
        default="",
        help="Default --output-dir for federated auto mode when not provided in forwarded args",
    )
    parser.add_argument(
        "--federated-auto-default-fed-state-path",
        type=str,
        default="",
        help="Default --fed-state-path for federated auto mode when not provided in forwarded args",
    )
    parser.add_argument(
        "--federated-auto-default-round-report-jsonl",
        type=str,
        default="",
        help="Default --fed-round-report-jsonl for federated auto mode when not provided in forwarded args",
    )
    parser.add_argument(
        "--federated-auto-default-prune-rounds-keep",
        type=int,
        default=None,
        help="Default --fed-prune-rounds-keep value in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-state-keep-rounds",
        type=int,
        default=None,
        help="Default --fed-state-keep-rounds value in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-summary-keep-rounds",
        type=int,
        default=None,
        help="Default --fed-summary-keep-rounds value in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-compact-output",
        dest="federated_auto_default_compact_output",
        action="store_true",
        default=None,
        help="Default to injecting --compact-output in federated auto mode",
    )
    parser.add_argument(
        "--federated-auto-default-no-compact-output",
        dest="federated_auto_default_compact_output",
        action="store_false",
        help="Disable default --compact-output injection in federated auto mode",
    )
    parser.add_argument(
        "--federated-auto-default-artifact-prune",
        dest="federated_auto_default_artifact_prune",
        action="store_true",
        default=None,
        help="Default to injecting --artifact-prune in federated auto mode",
    )
    parser.add_argument(
        "--federated-auto-default-no-artifact-prune",
        dest="federated_auto_default_artifact_prune",
        action="store_false",
        help="Disable default --artifact-prune injection in federated auto mode",
    )
    parser.add_argument(
        "--federated-auto-default-artifact-keep-latest",
        type=int,
        default=None,
        help="Default --artifact-keep-latest value in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-artifact-keep-promoted",
        type=int,
        default=None,
        help="Default --artifact-keep-promoted value in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-rounds",
        type=int,
        default=None,
        help="Default --fed-rounds in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-num-clients",
        type=int,
        default=None,
        help="Default --fed-num-clients in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-client-frac",
        type=float,
        default=None,
        help="Default --fed-client-frac in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-parallel-clients",
        type=int,
        default=None,
        help="Default --fed-parallel-clients in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-distributed-runtime",
        dest="federated_auto_default_fed_distributed_runtime",
        action="store_true",
        default=None,
        help="Default to injecting --fed-distributed-runtime in federated auto mode",
    )
    parser.add_argument(
        "--federated-auto-default-no-fed-distributed-runtime",
        dest="federated_auto_default_fed_distributed_runtime",
        action="store_false",
        help="Disable default --fed-distributed-runtime injection in federated auto mode",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-workers",
        type=int,
        default=None,
        help="Default --fed-dist-workers in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-timeout-seconds",
        type=float,
        default=None,
        help="Default --fed-dist-timeout-seconds in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-coded-redundancy",
        type=int,
        default=None,
        help="Default --fed-dist-coded-redundancy in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-fedbuff-size",
        type=int,
        default=None,
        help="Default --fed-dist-fedbuff-size in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-state-dir",
        type=str,
        default="",
        help="Default --fed-dist-state-dir in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-secure-mode",
        type=str,
        choices=["", "plain", "masked", "paillier"],
        default="",
        help="Default --fed-dist-secure-mode in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-secure-secret",
        type=str,
        default="",
        help="Default --fed-dist-secure-secret in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-secure-mask-scale",
        type=float,
        default=None,
        help="Default --fed-dist-secure-mask-scale in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-dp-clip-norm",
        type=float,
        default=None,
        help="Default --fed-dist-dp-clip-norm in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-dp-noise-sigma",
        type=float,
        default=None,
        help="Default --fed-dist-dp-noise-sigma in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-lease-seconds",
        type=float,
        default=None,
        help="Default --fed-dist-lease-seconds in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-heartbeat-timeout-s",
        type=float,
        default=None,
        help="Default --fed-dist-heartbeat-timeout-s in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-default-fed-dist-fedbuff-timeout-s",
        type=float,
        default=None,
        help="Default --fed-dist-fedbuff-timeout-s in federated auto mode when not provided",
    )
    parser.add_argument(
        "--federated-auto-adaptive",
        dest="federated_auto_adaptive",
        action="store_true",
        default=None,
        help="Enable adaptive federated auto profile tuning based on recent failures/successes",
    )
    parser.add_argument(
        "--federated-auto-no-adaptive",
        dest="federated_auto_adaptive",
        action="store_false",
        help="Disable adaptive federated auto profile tuning",
    )
    parser.add_argument(
        "--federated-auto-adapt-fail-threshold",
        type=int,
        default=None,
        help="Increase adaptive level when fail streak reaches this threshold",
    )
    parser.add_argument(
        "--federated-auto-adapt-max-level",
        type=int,
        default=None,
        help="Maximum adaptive degradation level",
    )
    parser.add_argument(
        "--federated-auto-adapt-recover-successes",
        type=int,
        default=None,
        help="Decrease adaptive level after this many consecutive successes",
    )
    parser.add_argument(
        "--federated-auto-adapt-round-decay",
        type=float,
        default=None,
        help="Per-level multiplicative decay for --fed-rounds (0<decay<=1)",
    )
    parser.add_argument(
        "--federated-auto-adapt-client-decay",
        type=float,
        default=None,
        help="Per-level multiplicative decay for --fed-num-clients (0<decay<=1)",
    )
    parser.add_argument(
        "--federated-auto-adapt-frac-decay",
        type=float,
        default=None,
        help="Per-level multiplicative decay for --fed-client-frac (0<decay<=1)",
    )
    parser.add_argument(
        "--federated-auto-adapt-parallel-decay",
        type=float,
        default=None,
        help="Per-level multiplicative decay for --fed-parallel-clients (0<decay<=1)",
    )
    parser.add_argument(
        "--federated-auto-adapt-min-rounds",
        type=int,
        default=None,
        help="Minimum rounds allowed by adaptive profile",
    )
    parser.add_argument(
        "--federated-auto-adapt-min-num-clients",
        type=int,
        default=None,
        help="Minimum client count allowed by adaptive profile",
    )
    parser.add_argument(
        "--federated-auto-adapt-min-client-frac",
        type=float,
        default=None,
        help="Minimum client fraction allowed by adaptive profile",
    )
    parser.add_argument(
        "--federated-auto-adapt-min-parallel-clients",
        type=int,
        default=None,
        help="Minimum parallel clients allowed by adaptive profile",
    )
    parser.add_argument(
        "--federated-auto-adapt-up-step",
        type=int,
        default=None,
        help="Adaptive level increment step when degradation is triggered",
    )
    parser.add_argument(
        "--federated-auto-adapt-down-step",
        type=int,
        default=None,
        help="Adaptive level decrement step when recovery is triggered",
    )
    parser.add_argument(
        "--federated-auto-adapt-up-cooldown-s",
        type=float,
        default=None,
        help="Minimum seconds between adaptive level increases",
    )
    parser.add_argument(
        "--federated-auto-adapt-down-cooldown-s",
        type=float,
        default=None,
        help="Minimum seconds between adaptive level decreases",
    )
    parser.add_argument(
        "--federated-auto-adapt-down-success-scale-per-level",
        type=float,
        default=None,
        help="Extra recovery-success requirement multiplier per adaptive level",
    )
    parser.add_argument(
        "--federated-auto-adapt-idle-decay-interval-s",
        type=float,
        default=None,
        help="Passive adaptive level decay interval when system stays stable",
    )
    parser.add_argument(
        "--federated-auto-adapt-idle-decay-step",
        type=int,
        default=None,
        help="Passive adaptive level decay step per idle interval",
    )
    parser.add_argument(
        "--federated-auto-adapt-idle-decay-no-fail-s",
        type=float,
        default=None,
        help="Require no federated auto failure for this long before passive decay",
    )
    parser.add_argument(
        "--federated-auto-adapt-idle-decay-no-resource-block-s",
        type=float,
        default=None,
        help="Require no resource-guard block for this long before passive decay",
    )
    parser.add_argument(
        "--federated-auto-resource-aware",
        dest="federated_auto_resource_aware",
        action="store_true",
        default=None,
        help="Enable local resource guard for federated auto trigger",
    )
    parser.add_argument(
        "--federated-auto-no-resource-aware",
        dest="federated_auto_resource_aware",
        action="store_false",
        help="Disable local resource guard for federated auto trigger",
    )
    parser.add_argument(
        "--federated-auto-max-cpu-percent",
        type=float,
        default=None,
        help="Skip federated auto trigger when local CPU usage is above this percent (0 disables)",
    )
    parser.add_argument(
        "--federated-auto-min-available-ram-gb",
        type=float,
        default=None,
        help="Skip federated auto trigger when available RAM is below this GB threshold (0 disables)",
    )
    parser.add_argument(
        "--federated-auto-min-resource-score",
        type=float,
        default=None,
        help="Skip federated auto trigger when hardware resource score is below this threshold (0 disables)",
    )
    parser.add_argument(
        "--federated-auto-resource-check-interval-s",
        type=float,
        default=None,
        help="Minimum interval between resource profile checks for federated auto guard",
    )
    parser.add_argument(
        "--federated-auto-resource-cooldown-s",
        type=float,
        default=None,
        help="Cooldown after resource-pressure skip before next federated auto attempt",
    )
    parser.add_argument(
        "--federated-auto-resource-adapt-boost",
        dest="federated_auto_resource_adapt_boost",
        action="store_true",
        default=None,
        help="When resource guard blocks, increase adaptive degradation level if adaptive mode is enabled",
    )
    parser.add_argument(
        "--federated-auto-no-resource-adapt-boost",
        dest="federated_auto_resource_adapt_boost",
        action="store_false",
        help="Disable adaptive-level boost on resource-pressure skip",
    )
    parser.add_argument(
        "--federated-auto-resource-skip-log-interval-s",
        type=float,
        default=None,
        help="Minimum interval between resource-guard skip logs/events (0 disables throttling)",
    )
    parser.add_argument(
        "--federated-auto-resource-pass-required",
        type=int,
        default=None,
        help="Required consecutive resource-pass checks before federated auto trigger (1 disables stability wait)",
    )
    parser.add_argument(
        "--federated-auto-hw-adapt",
        dest="federated_auto_hw_adapt",
        action="store_true",
        default=None,
        help="Auto-derive federated auto profile and resource guard from current hardware profile",
    )
    parser.add_argument(
        "--federated-auto-no-hw-adapt",
        dest="federated_auto_hw_adapt",
        action="store_false",
        help="Disable hardware-derived federated auto profile defaults",
    )
    parser.add_argument(
        "--federated-auto-multi-device",
        dest="federated_auto_multi_device",
        action="store_true",
        default=None,
        help="Enable cross-device federated distributed runtime defaults (discovery/auth/resume)",
    )
    parser.add_argument(
        "--federated-auto-no-multi-device",
        dest="federated_auto_multi_device",
        action="store_false",
        help="Disable cross-device federated distributed runtime defaults",
    )
    parser.add_argument(
        "--federated-auto-multi-device-service",
        type=str,
        default="",
        help="Discovery service name used when --federated-auto-multi-device is enabled",
    )
    parser.add_argument(
        "--federated-auto-multi-device-port",
        type=int,
        default=None,
        help="Discovery UDP port used when --federated-auto-multi-device is enabled",
    )
    parser.add_argument(
        "--federated-auto-multi-device-auth-token",
        type=str,
        default="",
        help="Auth token injected into distributed runtime when --federated-auto-multi-device is enabled",
    )
    parser.add_argument(
        "--federated-auto-workers-json",
        type=str,
        default="",
        help="Worker manifest path injected as --fed-workers-json for cross-device federated orchestration",
    )

