from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.automation.federated_runtime import (  # noqa: E402
    CoordinatorConfig,
    DiscoveryConfig,
    FedBuffConfig,
    FederatedCoordinator,
    FederatedWorker,
    PrivacyConfig,
    SecureAggConfig,
    WorkerConfig,
    run_demo_cluster,
)


def _has_zmq() -> bool:
    return bool(importlib.util.find_spec("zmq"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distributed federated runtime (coordinator/worker/demo)")
    p.add_argument("--mode", choices=["coordinator", "worker", "demo"], default="demo")

    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=9135)
    p.add_argument("--transport", choices=["tcp", "zeromq"], default="tcp")
    p.add_argument("--advertise-host", type=str, default="")
    p.add_argument("--auth-token", type=str, default="")
    p.add_argument("--resume-state", action="store_true")
    p.add_argument("--state-dir", type=str, default="artifacts/audit/federated_runtime")
    p.add_argument("--events-path", type=str, default="")
    p.add_argument("--state-path", type=str, default="")
    p.add_argument("--discovery-enabled", action="store_true")
    p.add_argument("--discovery-port", type=int, default=9136)
    p.add_argument("--discovery-service", type=str, default="fed_runtime")
    p.add_argument("--discovery-broadcast-ip", type=str, default="255.255.255.255")
    p.add_argument("--discovery-timeout-s", type=float, default=1.0)
    p.add_argument("--discovery-retries", type=int, default=5)

    p.add_argument("--vector-dim", type=int, default=32)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--clients-per-round", type=int, default=8)
    p.add_argument("--required-clients", type=int, default=6)
    p.add_argument("--coded-redundancy", type=int, default=1)
    p.add_argument("--lease-seconds", type=float, default=40.0)
    p.add_argument("--heartbeat-timeout-s", type=float, default=90.0)

    p.add_argument("--fedbuff-enabled", action="store_true")
    p.add_argument("--fedbuff-size", type=int, default=4)
    p.add_argument("--fedbuff-timeout-s", type=float, default=12.0)
    p.add_argument("--fedbuff-mix-alpha", type=float, default=1.0)

    p.add_argument("--secure-mode", choices=["plain", "masked", "paillier"], default="masked")
    p.add_argument("--secure-secret", type=str, default="runtime_shared_secret")
    p.add_argument("--secure-mask-scale", type=float, default=0.03)
    p.add_argument("--dp-clip-norm", type=float, default=1.0)
    p.add_argument("--dp-noise-sigma", type=float, default=0.0)
    p.add_argument("--dp-seed", type=int, default=7)

    p.add_argument("--node-id", type=str, default="worker_local")
    p.add_argument("--poll-interval-s", type=float, default=0.2)
    p.add_argument("--worker-heartbeat-interval-s", type=float, default=1.5)
    p.add_argument("--worker-max-runtime-s", type=float, default=0.0)
    p.add_argument("--worker-max-tasks", type=int, default=0)
    p.add_argument("--worker-sim-latency-s", type=float, default=0.05)
    p.add_argument("--worker-vector-scale", type=float, default=0.10)
    p.add_argument("--connect-timeout-s", type=float, default=8.0)

    p.add_argument("--demo-workers", type=int, default=3)
    p.add_argument("--demo-timeout-s", type=float, default=40.0)
    return p.parse_args()


def _paths(args: argparse.Namespace) -> Dict[str, str]:
    base = Path(str(args.state_dir))
    events = Path(str(args.events_path)).expanduser() if str(args.events_path or "").strip() else (base / "events.jsonl")
    state = Path(str(args.state_path)).expanduser() if str(args.state_path or "").strip() else (base / "state.json")
    if not events.is_absolute():
        events = (Path.cwd() / events).resolve()
    if not state.is_absolute():
        state = (Path.cwd() / state).resolve()
    return {"events": events.as_posix(), "state": state.as_posix()}


def _build_privacy(args: argparse.Namespace) -> PrivacyConfig:
    return PrivacyConfig(
        clip_norm=float(args.dp_clip_norm),
        noise_sigma=float(args.dp_noise_sigma),
        seed=int(args.dp_seed),
    )


def _build_secure(args: argparse.Namespace) -> SecureAggConfig:
    return SecureAggConfig(
        mode=str(args.secure_mode),
        shared_secret=str(args.secure_secret),
        mask_scale=float(args.secure_mask_scale),
    )


def _build_discovery(args: argparse.Namespace) -> DiscoveryConfig:
    return DiscoveryConfig(
        enabled=bool(args.discovery_enabled),
        service=str(args.discovery_service),
        udp_port=max(1, int(args.discovery_port)),
        broadcast_ip=str(args.discovery_broadcast_ip),
        request_timeout_s=max(0.1, float(args.discovery_timeout_s)),
        retries=max(1, int(args.discovery_retries)),
    )


def run_coordinator(args: argparse.Namespace) -> None:
    paths = _paths(args)
    cfg = CoordinatorConfig(
        host=str(args.host),
        port=int(args.port),
        transport=str(args.transport),
        advertise_host=str(args.advertise_host),
        auth_token=str(args.auth_token),
        resume_state=bool(args.resume_state),
        vector_dim=max(1, int(args.vector_dim)),
        rounds=max(1, int(args.rounds)),
        clients_per_round=max(1, int(args.clients_per_round)),
        required_clients=max(1, int(args.required_clients)),
        coded_redundancy=max(0, int(args.coded_redundancy)),
        lease_seconds=max(5.0, float(args.lease_seconds)),
        heartbeat_timeout_s=max(10.0, float(args.heartbeat_timeout_s)),
        state_path=str(paths["state"]),
        events_path=str(paths["events"]),
        fedbuff=FedBuffConfig(
            enabled=bool(args.fedbuff_enabled),
            buffer_size=max(1, int(args.fedbuff_size)),
            flush_timeout_s=max(0.0, float(args.fedbuff_timeout_s)),
            mix_alpha=max(0.0, min(1.0, float(args.fedbuff_mix_alpha))),
        ),
        privacy=_build_privacy(args),
        secure=_build_secure(args),
        discovery=_build_discovery(args),
    )
    coord = FederatedCoordinator(cfg)
    coord.start()
    print(
        json.dumps(
            {
                "ok": True,
                "mode": "coordinator",
                "host": cfg.host,
                "port": int(coord.cfg.port),
                "transport": str(cfg.transport),
                "auth_token_enabled": bool(str(cfg.auth_token).strip()),
                "resume_state": bool(cfg.resume_state),
                "discovery_enabled": bool(cfg.discovery.enabled),
                "discovery_service": str(cfg.discovery.service),
                "discovery_port": int(cfg.discovery.udp_port),
                "state_path": paths["state"],
                "events_path": paths["events"],
            },
            ensure_ascii=False,
        )
    )
    try:
        while True:
            row = coord.run_until_complete(timeout_s=2.0, poll_s=0.2)
            if bool(row.get("done", False)):
                print(json.dumps({"ok": True, "mode": "coordinator", "done": True, **row}, ensure_ascii=False))
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print(json.dumps({"ok": True, "mode": "coordinator", "interrupted": True}, ensure_ascii=False))
    finally:
        coord.stop()


def run_worker(args: argparse.Namespace) -> None:
    cfg = WorkerConfig(
        coordinator_host=str(args.host),
        coordinator_port=int(args.port),
        transport=str(args.transport),
        auth_token=str(args.auth_token),
        node_id=str(args.node_id),
        poll_interval_s=max(0.05, float(args.poll_interval_s)),
        heartbeat_interval_s=max(0.5, float(args.worker_heartbeat_interval_s)),
        connect_timeout_s=max(1.0, float(args.connect_timeout_s)),
        max_runtime_s=max(0.0, float(args.worker_max_runtime_s)),
        max_tasks=max(0, int(args.worker_max_tasks)),
        simulate_latency_s=max(0.0, float(args.worker_sim_latency_s)),
        vector_scale=max(0.001, float(args.worker_vector_scale)),
        privacy=_build_privacy(args),
        secure=_build_secure(args),
        discover=_build_discovery(args),
    )
    out = FederatedWorker(cfg).run()
    print(json.dumps(out, ensure_ascii=False))


def run_demo(args: argparse.Namespace) -> None:
    out = run_demo_cluster(
        workers=max(1, int(args.demo_workers)),
        rounds=max(1, int(args.rounds)),
        clients_per_round=max(1, int(args.clients_per_round)),
        required_clients=max(1, int(args.required_clients)),
        coded_redundancy=max(0, int(args.coded_redundancy)),
        vector_dim=max(1, int(args.vector_dim)),
        timeout_s=max(5.0, float(args.demo_timeout_s)),
        state_dir=str(args.state_dir),
        transport=str(args.transport),
        discovery=bool(args.discovery_enabled),
        discovery_port=max(1, int(args.discovery_port)),
        discovery_service=str(args.discovery_service),
        auth_token=str(args.auth_token),
        resume_state=bool(args.resume_state),
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if str(args.transport).strip().lower() == "zeromq" and (not _has_zmq()):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "zeromq_transport_requires_pyzmq",
                    "hint": "pip install pyzmq",
                    "mode": str(args.mode),
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)
    if str(args.mode) == "coordinator":
        run_coordinator(args)
        return
    if str(args.mode) == "worker":
        run_worker(args)
        return
    run_demo(args)


if __name__ == "__main__":
    main()
