from __future__ import annotations

import json
import os
import random
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._fed_coordinator import FederatedCoordinator
from ._fed_config import (
    CoordinatorConfig,
    DiscoveryConfig,
    FedBuffConfig,
    PrivacyConfig,
    SecureAggConfig,
    WorkerConfig,
    _clip_l2,
    _gaussian_noise,
    _make_mask,
    _normalize_transport,
    _now,
    _stable_seed,
    _try_import_zmq,
)


class FederatedWorker:
    def __init__(self, cfg: WorkerConfig, task_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None):
        self.cfg = cfg
        self.task_fn = task_fn
        self._last_event_id = 0
        self._transport = _normalize_transport(str(cfg.transport))
        self._discovery_lock = threading.Lock()

    def _with_auth(self, req: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(req or {})
        token = str(self.cfg.auth_token or "").strip()
        if token:
            payload["auth_token"] = token
        return payload

    def _discover_endpoint(self) -> Dict[str, Any]:
        dc = self.cfg.discover
        if not bool(dc.enabled):
            return {
                "ok": False,
                "reason": "discovery_disabled",
                "host": str(self.cfg.coordinator_host),
                "port": int(self.cfg.coordinator_port),
            }
        timeout_s = max(0.1, float(dc.request_timeout_s))
        retries = max(1, int(dc.retries))
        service = str(dc.service or "").strip() or "fed_runtime"
        target_ip = str(dc.broadcast_ip or "").strip() or "255.255.255.255"
        udp_port = max(1, int(dc.udp_port))
        request = {
            "op": "discover",
            "service": str(service),
            "transport": str(self._transport),
            "node_id": str(self.cfg.node_id),
            "pid": int(os.getpid()),
            "ts": float(_now()),
        }
        payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
        last_error = ""
        for _ in range(retries):
            sock: Optional[socket.socket] = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                except Exception:
                    pass
                sock.settimeout(timeout_s)
                targets: List[Tuple[str, int]] = [(target_ip, int(udp_port))]
                direct_host = str(self.cfg.coordinator_host or "").strip()
                if direct_host and direct_host not in {"0.0.0.0", "::"} and direct_host != target_ip:
                    targets.append((direct_host, int(udp_port)))
                for host, prt in targets:
                    try:
                        sock.sendto(payload, (str(host), int(prt)))
                    except Exception:
                        continue
                wait_until = _now() + timeout_s
                while _now() < wait_until:
                    try:
                        raw, _addr = sock.recvfrom(8192)
                    except socket.timeout:
                        break
                    except Exception as e:
                        last_error = f"{e}"
                        break
                    try:
                        rep = json.loads(raw.decode("utf-8", errors="ignore").strip())
                    except Exception:
                        continue
                    if not isinstance(rep, dict):
                        continue
                    if str(rep.get("op", "")).strip().lower() != "discover_reply":
                        continue
                    if str(rep.get("service", "")).strip() != service:
                        continue
                    host = str(rep.get("host", "") or "").strip()
                    port = int(rep.get("port", 0) or 0)
                    t = _normalize_transport(str(rep.get("transport", "") or self._transport))
                    if (not host) or port <= 0:
                        continue
                    if t != str(self._transport):
                        continue
                    self.cfg.coordinator_host = str(host)
                    self.cfg.coordinator_port = int(port)
                    return {
                        "ok": True,
                        "host": str(host),
                        "port": int(port),
                        "transport": str(t),
                    }
            except Exception as e:
                last_error = f"{e}"
            finally:
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:
                        pass
        return {
            "ok": False,
            "reason": "discover_timeout",
            "error": str(last_error),
            "host": str(self.cfg.coordinator_host),
            "port": int(self.cfg.coordinator_port),
        }

    def _rpc_tcp(self, req: Dict[str, Any]) -> Dict[str, Any]:
        addr = (str(self.cfg.coordinator_host), int(self.cfg.coordinator_port))
        with socket.create_connection(addr, timeout=float(self.cfg.connect_timeout_s)) as sock:
            sock.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
            line = b""
            while b"\n" not in line:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                line += chunk
        raw = line.decode("utf-8", errors="ignore").strip()
        if not raw:
            return {"ok": False, "error": "empty_response"}
        try:
            out = json.loads(raw)
        except Exception:
            return {"ok": False, "error": "invalid_response"}
        return out if isinstance(out, dict) else {"ok": False, "error": "invalid_response_type"}

    def _rpc_zmq(self, req: Dict[str, Any]) -> Dict[str, Any]:
        zmq = _try_import_zmq()
        if zmq is None:
            return {"ok": False, "error": "zeromq_transport_requires_pyzmq"}
        endpoint = f"tcp://{str(self.cfg.coordinator_host)}:{int(self.cfg.coordinator_port)}"
        timeout_ms = max(100, int(float(self.cfg.connect_timeout_s) * 1000.0))
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        try:
            sock.linger = 0
            sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
            sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
            sock.connect(endpoint)
            sock.send_string(json.dumps(req, ensure_ascii=False))
            raw = sock.recv_string()
        except Exception as e:
            return {"ok": False, "error": f"rpc_zmq_failed({e})"}
        finally:
            try:
                sock.close(0)
            except Exception:
                pass
        text = str(raw or "").strip()
        if not text:
            return {"ok": False, "error": "empty_response"}
        try:
            out = json.loads(text)
        except Exception:
            return {"ok": False, "error": "invalid_response"}
        return out if isinstance(out, dict) else {"ok": False, "error": "invalid_response_type"}

    def _rpc(self, req: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._with_auth(req)
        if str(self._transport) == "zeromq":
            return self._rpc_zmq(payload)
        return self._rpc_tcp(payload)

    def _synthetic(self, task: Dict[str, Any]) -> Dict[str, Any]:
        rid = int(task.get("round_id", 0) or 0)
        cid = int(task.get("client_id", 0) or 0)
        rep = int(task.get("replica_id", 0) or 0)
        dim = int(task.get("vector_dim", 64) or 64)
        seed = _stable_seed([self.cfg.node_id, self.cfg.privacy.seed, rid, cid, rep, "upd"])
        rng = random.Random(seed)
        vec = [float(rng.gauss(0.0, float(self.cfg.vector_scale))) + float((cid % 7) * 0.01) for _ in range(dim)]
        return {"vector": vec, "sample_count": int(task.get("sample_count_hint", 64) or 64)}

    def run(self) -> Dict[str, Any]:
        start = _now()
        pulls = 0
        done = 0
        hbs = 0
        empty_pulls = 0
        submit_rejected = 0
        rpc_failures = 0
        idle_backoff = max(0.05, float(self.cfg.poll_interval_s))
        last_hb = 0.0
        sec_mode = str(self.cfg.secure.mode)
        discovery_result: Dict[str, Any] = {"ok": False, "reason": "not_used"}
        if bool(self.cfg.discover.enabled):
            discovery_result = self._discover_endpoint()
        reg = self._rpc({"op": "register", "node_id": str(self.cfg.node_id)})
        if (not bool(reg.get("ok", False))) and bool(self.cfg.discover.enabled):
            discovery_result = self._discover_endpoint()
            reg = self._rpc({"op": "register", "node_id": str(self.cfg.node_id)})
        if not bool(reg.get("ok", False)):
            return {
                "ok": False,
                "error": str(reg.get("error", "register_failed")),
                "transport": str(self._transport),
                "coordinator_host": str(self.cfg.coordinator_host),
                "coordinator_port": int(self.cfg.coordinator_port),
                "discovery": dict(discovery_result),
            }
        sec_mode = str(reg.get("secure_mode_effective", sec_mode) or sec_mode)
        while True:
            now_ts = _now()
            if float(self.cfg.max_runtime_s) > 0.0 and now_ts - start >= float(self.cfg.max_runtime_s):
                break
            if int(self.cfg.max_tasks) > 0 and int(done) >= int(self.cfg.max_tasks):
                break
            if now_ts - last_hb >= max(0.5, float(self.cfg.heartbeat_interval_s)):
                hb = self._rpc({"op": "heartbeat", "node_id": str(self.cfg.node_id), "since_event_id": int(self._last_event_id), "stats": {"tasks_done": int(done), "pulls": int(pulls)}})
                hbs += 1
                last_hb = now_ts
                if not bool(hb.get("ok", False)):
                    rpc_failures += 1
                    idle_backoff = min(2.0, max(0.05, idle_backoff * 1.5))
                    time.sleep(float(idle_backoff))
                    continue
                gossip = hb.get("gossip", [])
                if isinstance(gossip, list) and gossip:
                    self._last_event_id = max(int(self._last_event_id), max(int(x.get("id", 0) or 0) for x in gossip if isinstance(x, dict)))
                if bool(hb.get("done", False)):
                    break
            pull = self._rpc({"op": "pull_task", "node_id": str(self.cfg.node_id)})
            pulls += 1
            if not bool(pull.get("ok", False)):
                rpc_failures += 1
                idle_backoff = min(2.0, max(0.05, idle_backoff * 1.5))
                time.sleep(float(idle_backoff))
                continue
            if bool(pull.get("done", False)):
                break
            task = pull.get("task", None)
            if not isinstance(task, dict):
                empty_pulls += 1
                hint = max(0.05, float(pull.get("retry_after_s", self.cfg.poll_interval_s) or self.cfg.poll_interval_s))
                idle_backoff = min(2.0, max(float(hint), idle_backoff * 1.4))
                time.sleep(float(idle_backoff))
                continue
            idle_backoff = max(0.05, float(self.cfg.poll_interval_s))
            t0 = _now()
            if float(self.cfg.simulate_latency_s) > 0.0:
                time.sleep(float(self.cfg.simulate_latency_s))
            try:
                result = self.task_fn(task) if callable(self.task_fn) else self._synthetic(task)
            except Exception:
                rpc_failures += 1
                continue
            if not isinstance(result, dict):
                rpc_failures += 1
                continue
            if bool(result.get("error")):
                rpc_failures += 1
                continue

            sample_count = int(max(1, int(result.get("sample_count", task.get("sample_count_hint", 64)) or 64)))
            preprocessed = bool(result.get("vector_path_preprocessed", False))
            vec_path = str(result.get("vector_path", "") or "").strip()
            masked = bool(result.get("masked", False)) if preprocessed else False
            payload: Dict[str, Any] = {
                "op": "submit_update",
                "node_id": str(self.cfg.node_id),
                "task_id": str(task.get("task_id", "")),
                "round_id": int(task.get("round_id", 0) or 0),
                "client_id": int(task.get("client_id", 0) or 0),
                "replica_id": int(task.get("replica_id", 0) or 0),
                "sample_count": int(sample_count),
                "latency_s": float(max(0.0, _now() - t0)),
                "masked": bool(masked),
            }
            if vec_path:
                payload["vector_path"] = str(vec_path)
                payload["vector_path_delete"] = bool(result.get("vector_path_delete", False))
            else:
                vec = [float(x) for x in list(result.get("vector", []) or [])]
                if not preprocessed:
                    vec = _clip_l2(vec, float(self.cfg.privacy.clip_norm))
                    if float(self.cfg.privacy.noise_sigma) > 0.0:
                        s = _stable_seed([self.cfg.privacy.seed, self.cfg.node_id, task.get("task_id", ""), "worker_noise"])
                        vec = _gaussian_noise(vec, float(self.cfg.privacy.noise_sigma), int(s))
                    if str(sec_mode) == "masked":
                        m = _make_mask(
                            len(vec),
                            secret=str(self.cfg.secure.shared_secret),
                            node_id=str(self.cfg.node_id),
                            round_id=int(task.get("round_id", 0) or 0),
                            client_id=int(task.get("client_id", 0) or 0),
                            replica_id=int(task.get("replica_id", 0) or 0),
                            scale=float(self.cfg.secure.mask_scale),
                        )
                        vec = [float(v) + float(x) for v, x in zip(vec, m)]
                        payload["masked"] = True
                payload["vector"] = vec

            sub = self._rpc(payload)
            if bool(sub.get("ok", False)) and bool(sub.get("accepted", False)):
                done += 1
            elif bool(sub.get("ok", False)) and (not bool(sub.get("accepted", True))):
                submit_rejected += 1
            else:
                rpc_failures += 1
        return {
            "ok": True,
            "node_id": str(self.cfg.node_id),
            "transport": str(self._transport),
            "coordinator_host": str(self.cfg.coordinator_host),
            "coordinator_port": int(self.cfg.coordinator_port),
            "tasks_done": int(done),
            "pulls": int(pulls),
            "heartbeats": int(hbs),
            "empty_pulls": int(empty_pulls),
            "submit_rejected": int(submit_rejected),
            "rpc_failures": int(rpc_failures),
            "elapsed_s": float(max(0.0, _now() - start)),
            "discovery": dict(discovery_result),
        }


def run_demo_cluster(
    *,
    workers: int = 3,
    rounds: int = 2,
    clients_per_round: int = 6,
    required_clients: int = 4,
    coded_redundancy: int = 1,
    vector_dim: int = 16,
    timeout_s: float = 40.0,
    state_dir: str = "artifacts/audit/federated_runtime_demo",
    transport: str = "tcp",
    discovery: bool = False,
    discovery_port: int = 9136,
    discovery_service: str = "fed_runtime_demo",
    auth_token: str = "",
    resume_state: bool = False,
) -> Dict[str, Any]:
    base = Path(str(state_dir))
    transport_norm = _normalize_transport(str(transport))
    coord = FederatedCoordinator(
        CoordinatorConfig(
            host="127.0.0.1",
            port=0,
            transport=str(transport_norm),
            advertise_host="127.0.0.1",
            auth_token=str(auth_token or ""),
            resume_state=bool(resume_state),
            vector_dim=int(vector_dim),
            rounds=int(rounds),
            clients_per_round=int(clients_per_round),
            required_clients=int(required_clients),
            coded_redundancy=int(coded_redundancy),
            state_path=(base / "state.json").as_posix(),
            events_path=(base / "events.jsonl").as_posix(),
            fedbuff=FedBuffConfig(enabled=True, buffer_size=max(1, int(required_clients // 2)), flush_timeout_s=5.0, mix_alpha=1.0),
            privacy=PrivacyConfig(clip_norm=2.0, noise_sigma=0.0, seed=13),
            secure=SecureAggConfig(mode="masked", shared_secret="demo_secret", mask_scale=0.03),
            discovery=DiscoveryConfig(
                enabled=bool(discovery),
                service=str(discovery_service),
                udp_port=max(1, int(discovery_port)),
                broadcast_ip="255.255.255.255",
                request_timeout_s=0.5,
                retries=3,
            ),
        )
    )
    coord.start()
    out_rows: List[Dict[str, Any]] = []
    lock = threading.Lock()
    threads: List[threading.Thread] = []

    def _worker_entry(i: int) -> None:
        w = FederatedWorker(
            WorkerConfig(
                coordinator_host=("" if bool(discovery) else "127.0.0.1"),
                coordinator_port=(0 if bool(discovery) else int(coord.cfg.port)),
                transport=str(transport_norm),
                auth_token=str(auth_token or ""),
                node_id=f"worker_{i+1}",
                poll_interval_s=0.05,
                heartbeat_interval_s=0.6,
                max_runtime_s=float(timeout_s),
                simulate_latency_s=0.01 + 0.01 * float(i),
                vector_scale=0.08 + 0.02 * float(i),
                privacy=PrivacyConfig(clip_norm=1.5, noise_sigma=0.0, seed=31 + i),
                secure=SecureAggConfig(mode="masked", shared_secret="demo_secret", mask_scale=0.03),
                discover=DiscoveryConfig(
                    enabled=bool(discovery),
                    service=str(discovery_service),
                    udp_port=max(1, int(discovery_port)),
                    broadcast_ip="255.255.255.255",
                    request_timeout_s=0.5,
                    retries=4,
                ),
            )
        )
        row = w.run()
        with lock:
            out_rows.append(row)

    for i in range(max(1, int(workers))):
        t = threading.Thread(target=_worker_entry, args=(i,), daemon=True)
        threads.append(t)
        t.start()
    coord_summary = coord.run_until_complete(timeout_s=float(timeout_s), poll_s=0.2)
    for t in threads:
        t.join(timeout=2.0)
    coord.stop()
    return {
        "ok": bool(coord_summary.get("done", False)),
        "transport": str(transport_norm),
        "discovery": bool(discovery),
        "auth_token_enabled": bool(str(auth_token or "").strip()),
        "resume_state": bool(resume_state),
        "coordinator": coord_summary,
        "workers": out_rows,
    }
