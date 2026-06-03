from __future__ import annotations

import hashlib
import json
import math
import os
import random
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _try_import_zmq():
    try:
        import zmq  # type: ignore

        return zmq
    except Exception:
        return None


def _normalize_transport(name: str) -> str:
    key = str(name or "").strip().lower()
    if key in {"zeromq", "zmq"}:
        return "zeromq"
    return "tcp"


def _local_ip_hint() -> str:
    sock: Optional[socket.socket] = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = str(sock.getsockname()[0] or "").strip()
        if ip:
            return ip
    except Exception:
        pass
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
    try:
        host = socket.gethostname()
        ip = str(socket.gethostbyname(host) or "").strip()
        if ip:
            return ip
    except Exception:
        pass
    return "127.0.0.1"


def _resolve_advertise_host(host: str, advertise_host: str) -> str:
    adv = str(advertise_host or "").strip()
    if adv:
        return adv
    h = str(host or "").strip()
    if h and h not in {"0.0.0.0", "::", "::0"}:
        return h
    return _local_ip_hint()


def _now() -> float:
    return float(time.time())


def _stable_seed(parts: List[Any]) -> int:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{int(time.time() * 1000)}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clip_l2(vec: List[float], clip_norm: float) -> List[float]:
    if clip_norm <= 0.0:
        return [float(x) for x in vec]
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= clip_norm or norm <= 1e-12:
        return [float(x) for x in vec]
    s = float(clip_norm) / float(norm)
    return [float(x) * s for x in vec]


def _gaussian_noise(vec: List[float], sigma: float, seed: int) -> List[float]:
    if sigma <= 0.0:
        return [float(x) for x in vec]
    rng = random.Random(int(seed))
    return [float(x) + float(rng.gauss(0.0, sigma)) for x in vec]


def _make_mask(
    dim: int,
    *,
    secret: str,
    node_id: str,
    round_id: int,
    client_id: int,
    replica_id: int,
    scale: float,
) -> List[float]:
    if dim <= 0 or not secret or scale <= 0.0:
        return [0.0 for _ in range(max(0, int(dim)))]
    seed = _stable_seed([secret, node_id, round_id, client_id, replica_id, "mask"])
    rng = random.Random(seed)
    return [float(rng.uniform(-scale, scale)) for _ in range(int(dim))]


def _vector_digest(vec: List[float]) -> str:
    h = hashlib.sha256()
    for v in vec:
        h.update(f"{float(v):.10f}|".encode("utf-8"))
    return h.hexdigest()


def _vector_storage_payload(vec: List[float]) -> Dict[str, Any]:
    dim = int(len(vec))
    norm = float(math.sqrt(sum(float(x) * float(x) for x in vec)))
    payload: Dict[str, Any] = {
        "dim": int(dim),
        "norm": float(norm),
        "sha256": str(_vector_digest(vec)),
    }
    if dim <= 256:
        payload["full"] = [float(x) for x in vec]
    else:
        payload["head"] = [float(x) for x in vec[:32]]
        payload["tail"] = [float(x) for x in vec[-32:]]
    return payload


def _load_vector_from_path(path: str) -> List[float]:
    p = Path(str(path or "").strip())
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"vector_path_not_found({p.as_posix()})")
    suffix = str(p.suffix).strip().lower()
    if suffix in {".pt", ".pth", ".bin"}:
        try:
            import torch  # type: ignore

            raw = torch.load(p.as_posix(), map_location="cpu")
            if hasattr(raw, "detach"):
                raw = raw.detach().cpu().reshape(-1).tolist()
            if isinstance(raw, list):
                return [float(x) for x in raw]
        except Exception as e:
            raise RuntimeError(f"vector_path_torch_load_failed({e})")
    if suffix == ".npy":
        try:
            import numpy as np  # type: ignore

            arr = np.load(p.as_posix())
            return [float(x) for x in arr.reshape(-1).tolist()]
        except Exception as e:
            raise RuntimeError(f"vector_path_numpy_load_failed({e})")
    text = p.read_text(encoding="utf-8")
    raw = json.loads(text)
    if isinstance(raw, list):
        return [float(x) for x in raw]
    if isinstance(raw, dict) and isinstance(raw.get("vector"), list):
        return [float(x) for x in raw.get("vector", [])]
    raise RuntimeError("vector_path_invalid_content")


@dataclass
class FedBuffConfig:
    enabled: bool = True
    buffer_size: int = 4
    flush_timeout_s: float = 12.0
    mix_alpha: float = 1.0


@dataclass
class PrivacyConfig:
    clip_norm: float = 1.0
    noise_sigma: float = 0.0
    seed: int = 7


@dataclass
class SecureAggConfig:
    mode: str = "plain"  # plain | masked | paillier
    shared_secret: str = ""
    mask_scale: float = 0.03


@dataclass
class DiscoveryConfig:
    enabled: bool = False
    service: str = "fed_runtime"
    udp_port: int = 9136
    broadcast_ip: str = "255.255.255.255"
    request_timeout_s: float = 1.0
    retries: int = 5


@dataclass
class CoordinatorConfig:
    host: str = "127.0.0.1"
    port: int = 9135
    transport: str = "tcp"  # tcp | zeromq
    advertise_host: str = ""
    auth_token: str = ""
    resume_state: bool = False
    vector_dim: int = 64
    rounds: int = 2
    clients_per_round: int = 6
    required_clients: int = 4
    coded_redundancy: int = 1
    lease_seconds: float = 40.0
    heartbeat_timeout_s: float = 90.0
    state_path: str = "artifacts/audit/federated_runtime/state.json"
    events_path: str = "artifacts/audit/federated_runtime/events.jsonl"
    fedbuff: FedBuffConfig = field(default_factory=FedBuffConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    secure: SecureAggConfig = field(default_factory=SecureAggConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)


@dataclass
class WorkerConfig:
    coordinator_host: str = "127.0.0.1"
    coordinator_port: int = 9135
    transport: str = "tcp"  # tcp | zeromq
    auth_token: str = ""
    node_id: str = "worker_local"
    poll_interval_s: float = 0.2
    heartbeat_interval_s: float = 1.5
    connect_timeout_s: float = 8.0
    max_runtime_s: float = 0.0
    max_tasks: int = 0
    simulate_latency_s: float = 0.05
    vector_scale: float = 0.1
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    secure: SecureAggConfig = field(default_factory=SecureAggConfig)
    discover: DiscoveryConfig = field(default_factory=DiscoveryConfig)
