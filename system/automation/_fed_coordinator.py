from __future__ import annotations

import hmac
import json
import os
import socketserver
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from ._fed_config import (
    CoordinatorConfig,
    DiscoveryConfig,
    _append_jsonl,
    _atomic_write_json,
    _clip_l2,
    _gaussian_noise,
    _load_vector_from_path,
    _make_mask,
    _normalize_transport,
    _now,
    _resolve_advertise_host,
    _stable_seed,
    _try_import_zmq,
    _vector_storage_payload,
)


class _TCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr: Tuple[str, int], handler: Any, runtime: "FederatedCoordinator"):
        self.runtime = runtime
        super().__init__(addr, handler)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        rt: FederatedCoordinator = self.server.runtime  # type: ignore[attr-defined]
        while True:
            line = self.rfile.readline()
            if not line:
                return
            raw = str(line.decode("utf-8", errors="ignore")).strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
            except Exception:
                resp = {"ok": False, "error": "invalid_json"}
            else:
                if not isinstance(req, dict):
                    req = {}
                resp = rt.handle_rpc(req, f"{self.client_address[0]}:{self.client_address[1]}")
            try:
                self.wfile.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                return


class FederatedCoordinator:
    def __init__(self, cfg: CoordinatorConfig):
        self.cfg = cfg
        self._lock = threading.RLock()
        self._server: Optional[_TCPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._server_kind = "none"
        self._transport = _normalize_transport(str(cfg.transport))
        self._advertise_host = _resolve_advertise_host(str(cfg.host), str(cfg.advertise_host))
        self._zmq_ctx = None
        self._zmq_sock = None
        self._serve_stop = threading.Event()
        self._discovery_thread: Optional[threading.Thread] = None
        self._discovery_stop = threading.Event()
        self._started_at = _now()
        self._done = False

        self._state_path = Path(str(cfg.state_path))
        self._events_path = Path(str(cfg.events_path))
        self._workers: Dict[str, Dict[str, Any]] = {}
        self._events: Deque[Dict[str, Any]] = deque(maxlen=2048)
        self._event_id = 0

        self._round = 1
        self._round_tasks: Dict[int, List[Dict[str, Any]]] = self._build_tasks()
        self._pending: Deque[Dict[str, Any]] = deque(self._round_tasks.get(1, []))
        self._leases: Dict[str, Dict[str, Any]] = {}
        self._completed_tasks: Dict[str, Dict[str, Any]] = {}
        self._completed_clients: Dict[int, Dict[int, Dict[str, Any]]] = {}
        self._buffers: Dict[int, List[Dict[str, Any]]] = {}
        self._buffer_ts: Dict[int, float] = {}
        self._round_done: Dict[int, bool] = {}
        self._round_flush_count: Dict[int, int] = {}
        self._global_vec: List[float] = [0.0 for _ in range(max(1, int(cfg.vector_dim)))]
        self._global_updates = 0
        self._secure_mode = self._resolve_secure_mode(str(cfg.secure.mode))
        restored = False
        if bool(self.cfg.resume_state):
            restored = bool(self._restore_from_state())
        self._emit(
            "init",
            {
                "secure_mode_requested": str(cfg.secure.mode),
                "secure_mode_effective": str(self._secure_mode),
                "resume_state": bool(self.cfg.resume_state),
                "state_restored": bool(restored),
            },
        )
        self._save()

    def _resolve_secure_mode(self, mode: str) -> str:
        key = str(mode or "").strip().lower()
        if key in {"plain", "masked"}:
            return key
        if key == "paillier":
            try:
                import phe  # type: ignore  # noqa: F401
                return "paillier"
            except Exception:
                return "masked"
        return "plain"

    def _auth_required(self) -> bool:
        return bool(str(self.cfg.auth_token or "").strip())

    def _auth_ok(self, req: Dict[str, Any]) -> bool:
        expected = str(self.cfg.auth_token or "").strip()
        if not expected:
            return True
        got = str(req.get("auth_token", "") or "").strip()
        if not got:
            return False
        return bool(hmac.compare_digest(got, expected))

    def _task_catalog(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for rows in self._round_tasks.values():
            for task in rows:
                if not isinstance(task, dict):
                    continue
                tid = str(task.get("task_id", "") or "").strip()
                if not tid:
                    continue
                out[tid] = dict(task)
        return out

    def _state_snapshot_runtime(self) -> Dict[str, Any]:
        leases: List[Dict[str, Any]] = []
        for tid, node in self._leases.items():
            task = node.get("task", {}) if isinstance(node, dict) else {}
            if not isinstance(task, dict):
                continue
            leases.append(
                {
                    "task_id": str(tid),
                    "lease_until": float(node.get("lease_until", 0.0) or 0.0),
                    "task": dict(task),
                }
            )
        completed_by_round: Dict[str, List[int]] = {}
        for rid, rows in self._completed_clients.items():
            if not isinstance(rows, dict):
                continue
            cids: List[int] = []
            for cid in rows.keys():
                try:
                    cids.append(int(cid))
                except Exception:
                    continue
            completed_by_round[str(int(rid))] = sorted(set(cids))
        return {
            "pending": [dict(x) for x in list(self._pending)],
            "leases": leases,
            "completed_task_ids": sorted(str(x) for x in self._completed_tasks.keys()),
            "completed_clients": completed_by_round,
        }

    def _restore_from_state(self) -> bool:
        p = self._state_path
        if not p.exists():
            return False
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(raw, dict):
            return False
        snap = raw.get("runtime_snapshot", {})
        if not isinstance(snap, dict):
            snap = {}
        catalog = self._task_catalog()
        completed_ids: set[str] = set()
        for tid in list(snap.get("completed_task_ids", []) or []):
            key = str(tid or "").strip()
            if key and key in catalog:
                completed_ids.add(key)
                self._completed_tasks[key] = {"node_id": "restored", "ts": _now()}

        pending_rows: List[Dict[str, Any]] = []
        for node in list(snap.get("pending", []) or []):
            if not isinstance(node, dict):
                continue
            tid = str(node.get("task_id", "") or "").strip()
            if (not tid) or (tid not in catalog) or (tid in completed_ids):
                continue
            pending_rows.append(dict(catalog[tid]))

        now_ts = _now()
        lease_rows = list(snap.get("leases", []) or [])
        for node in lease_rows:
            if not isinstance(node, dict):
                continue
            tid = str(node.get("task_id", "") or "").strip()
            if (not tid) or (tid not in catalog) or (tid in completed_ids):
                continue
            lease_until = float(node.get("lease_until", 0.0) or 0.0)
            if lease_until <= now_ts:
                pending_rows.append(dict(catalog[tid]))
                continue
            self._leases[tid] = {
                "task": dict(catalog[tid]),
                "lease_until": float(lease_until),
            }

        seen_pending: set[str] = set()
        dedup_pending: List[Dict[str, Any]] = []
        for task in pending_rows:
            tid = str(task.get("task_id", "") or "").strip()
            if not tid:
                continue
            if tid in seen_pending or tid in self._leases or tid in completed_ids:
                continue
            seen_pending.add(tid)
            dedup_pending.append(dict(task))
        if (not dedup_pending) and (not self._leases):
            for rows in self._round_tasks.values():
                for task in rows:
                    if not isinstance(task, dict):
                        continue
                    tid = str(task.get("task_id", "") or "").strip()
                    if (not tid) or (tid in completed_ids):
                        continue
                    if tid in seen_pending:
                        continue
                    seen_pending.add(tid)
                    dedup_pending.append(dict(task))
        self._pending = deque(dedup_pending)

        cc = snap.get("completed_clients", {})
        if isinstance(cc, dict):
            restored_clients: Dict[int, Dict[int, Dict[str, Any]]] = {}
            for rid_text, rows in cc.items():
                try:
                    rid = int(rid_text)
                except Exception:
                    continue
                if not isinstance(rows, list):
                    continue
                bucket: Dict[int, Dict[str, Any]] = {}
                for cid in rows:
                    try:
                        c = int(cid)
                    except Exception:
                        continue
                    bucket[c] = {"node_id": "restored", "ts": now_ts}
                if bucket:
                    restored_clients[int(rid)] = bucket
            if restored_clients:
                self._completed_clients = restored_clients

        try:
            round_id = int(raw.get("round", 1) or 1)
        except Exception:
            round_id = 1
        if self._pending:
            try:
                round_id = int(self._pending[0].get("round_id", round_id) or round_id)
            except Exception:
                pass
        self._round = max(1, min(int(self.cfg.rounds), int(round_id)))
        self._done = bool(raw.get("done", False))
        try:
            self._global_updates = max(0, int(raw.get("global_updates", 0) or 0))
        except Exception:
            self._global_updates = 0
        for rid in range(1, int(self.cfg.rounds) + 1):
            cc_rows = self._completed_clients.get(int(rid), {})
            self._round_done[int(rid)] = bool(len(cc_rows) >= max(1, int(self.cfg.required_clients)))
        return True

    def _build_tasks(self) -> Dict[int, List[Dict[str, Any]]]:
        out: Dict[int, List[Dict[str, Any]]] = {}
        rounds = max(1, int(self.cfg.rounds))
        clients = max(1, int(self.cfg.clients_per_round))
        red = max(0, int(self.cfg.coded_redundancy))
        for rid in range(1, rounds + 1):
            rows: List[Dict[str, Any]] = []
            for cid in range(clients):
                hint = 32 + 8 * (cid % 5)
                for rep in range(red + 1):
                    rows.append(
                        {
                            "task_id": f"r{rid}_c{cid}_rep{rep}",
                            "round_id": int(rid),
                            "client_id": int(cid),
                            "replica_id": int(rep),
                            "sample_count_hint": int(hint),
                            "vector_dim": int(self.cfg.vector_dim),
                        }
                    )
            out[int(rid)] = rows
        return out

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        self._event_id += 1
        row = {"id": int(self._event_id), "ts": _now(), "event": str(event), "round": int(self._round), **(payload or {})}
        self._events.append(row)
        _append_jsonl(self._events_path, row)

    def _active_workers(self) -> List[Dict[str, Any]]:
        now_ts = _now()
        max_age = max(1.0, float(self.cfg.heartbeat_timeout_s))
        out: List[Dict[str, Any]] = []
        for node_id, n in self._workers.items():
            seen = float(n.get("last_seen", 0.0) or 0.0)
            if now_ts - seen > max_age:
                continue
            out.append({"node_id": str(node_id), "last_seen": float(seen), "latency_ema_s": float(n.get("latency_ema_s", 0.0) or 0.0), "fail_ema": float(n.get("fail_ema", 0.0) or 0.0), "tasks_done": int(n.get("tasks_done", 0) or 0)})
        out.sort(key=lambda x: str(x.get("node_id", "")))
        return out

    def _dht(self) -> Dict[str, List[str]]:
        table: Dict[str, List[str]] = {}
        for n in self._active_workers():
            node_id = str(n.get("node_id", ""))
            b = int(_stable_seed([node_id, "dht"]) % 16)
            k = f"b{b:02d}"
            table.setdefault(k, []).append(node_id)
        for k in list(table.keys()):
            table[k] = sorted(table[k])
        return table

    def _save(self) -> None:
        vec_payload = _vector_storage_payload(self._global_vec)
        runtime_snapshot = self._state_snapshot_runtime()
        payload = {
            "ok": True,
            "state_version": 2,
            "done": bool(self._done),
            "round": int(self._round),
            "rounds": int(self.cfg.rounds),
            "transport": str(self._transport),
            "advertise_host": str(self._advertise_host),
            "resume_state": bool(self.cfg.resume_state),
            "discovery": {
                "enabled": bool(self.cfg.discovery.enabled),
                "service": str(self.cfg.discovery.service),
                "udp_port": int(self.cfg.discovery.udp_port),
            },
            "auth_token_enabled": bool(self._auth_required()),
            "pending_tasks": int(len(self._pending)),
            "leased_tasks": int(len(self._leases)),
            "completed_tasks": int(len(self._completed_tasks)),
            "completed_clients_by_round": {str(k): int(len(v)) for k, v in self._completed_clients.items()},
            "round_done": {str(k): bool(v) for k, v in self._round_done.items()},
            "round_flush_count": {str(k): int(v) for k, v in self._round_flush_count.items()},
            "workers": self._active_workers(),
            "dht_table": self._dht(),
            "global_vector_dim": int(len(self._global_vec)),
            "global_vector": list(vec_payload.get("full", [])) if "full" in vec_payload else [],
            "global_vector_storage": vec_payload,
            "global_updates": int(self._global_updates),
            "secure_mode_effective": str(self._secure_mode),
            "runtime_snapshot": runtime_snapshot,
        }
        _atomic_write_json(self._state_path, payload)

    def _requeue_expired(self) -> None:
        now_ts = _now()
        expired: List[str] = []
        for tid, lease in self._leases.items():
            if now_ts >= float(lease.get("lease_until", 0.0) or 0.0):
                expired.append(str(tid))
        for tid in expired:
            lease = self._leases.pop(tid, {})
            task = lease.get("task", {})
            if isinstance(task, dict):
                self._pending.appendleft(task)
            self._emit("lease_expired", {"task_id": str(tid)})

    def _maybe_flush_timeout(self) -> None:
        if not bool(self.cfg.fedbuff.enabled):
            return
        timeout_s = max(0.0, float(self.cfg.fedbuff.flush_timeout_s))
        if timeout_s <= 0.0:
            return
        now_ts = _now()
        for rid, rows in list(self._buffers.items()):
            if not rows:
                continue
            last_ts = float(self._buffer_ts.get(int(rid), 0.0) or 0.0)
            if last_ts > 0.0 and now_ts - last_ts >= timeout_s:
                self._flush(int(rid), "timeout")

    def _flush(self, rid: int, reason: str) -> Dict[str, Any]:
        rows = list(self._buffers.get(int(rid), []))
        if not rows:
            return {"ok": False, "reason": "empty"}
        dim = int(self.cfg.vector_dim)
        acc = [0.0 for _ in range(dim)]
        total_w = 0.0
        for r in rows:
            vec = [float(x) for x in list(r.get("vector", []) or [])]
            if len(vec) != dim:
                continue
            w = max(1.0, float(r.get("sample_count", 1.0) or 1.0))
            for i in range(dim):
                acc[i] += float(vec[i]) * float(w)
            total_w += float(w)
        if total_w <= 0.0:
            self._buffers[int(rid)] = []
            return {"ok": False, "reason": "bad_weight"}
        merged = [float(x) / float(total_w) for x in acc]
        mix = min(1.0, max(0.0, float(self.cfg.fedbuff.mix_alpha)))
        if self._global_updates <= 0:
            self._global_vec = merged
        else:
            self._global_vec = [float((1.0 - mix) * g + mix * m) for g, m in zip(self._global_vec, merged)]
        self._global_updates += 1
        self._buffers[int(rid)] = []
        self._round_flush_count[int(rid)] = int(self._round_flush_count.get(int(rid), 0) or 0) + 1
        self._emit("buffer_flushed", {"round_id": int(rid), "reason": str(reason), "count": int(len(rows)), "global_updates": int(self._global_updates)})
        return {"ok": True}

    def _unmask(self, vec: List[float], node_id: str, rid: int, cid: int, rep: int, masked: bool) -> List[float]:
        if str(self._secure_mode) != "masked" or (not masked):
            return [float(x) for x in vec]
        mask = _make_mask(len(vec), secret=str(self.cfg.secure.shared_secret), node_id=str(node_id), round_id=int(rid), client_id=int(cid), replica_id=int(rep), scale=float(self.cfg.secure.mask_scale))
        return [float(v) - float(m) for v, m in zip(vec, mask)]

    def _finish_round_if_needed(self, rid: int) -> None:
        done_clients = self._completed_clients.get(int(rid), {})
        if len(done_clients) < max(1, int(self.cfg.required_clients)):
            return
        if self._buffers.get(int(rid)):
            self._flush(int(rid), "round_complete")
        self._round_done[int(rid)] = True
        self._emit("round_done", {"round_id": int(rid), "completed_clients": int(len(done_clients))})
        if int(rid) >= int(self.cfg.rounds):
            self._done = True
            self._emit("training_done", {"round_id": int(rid)})
            return
        self._round = int(rid) + 1
        self._pending = deque(self._round_tasks.get(int(self._round), []))
        self._emit("round_activated", {"round_id": int(self._round), "task_count": int(len(self._pending))})

    def handle_rpc(self, req: Dict[str, Any], client_addr: str = "") -> Dict[str, Any]:
        op = str(req.get("op", "") or "").strip().lower()
        with self._lock:
            if not self._auth_ok(req):
                self._emit("auth_rejected", {"from": str(client_addr), "op": str(op)})
                return {"ok": False, "error": "unauthorized"}
            self._requeue_expired()
            self._maybe_flush_timeout()
            if op == "register":
                node_id = str(req.get("node_id", "") or "").strip()
                if not node_id:
                    return {"ok": False, "error": "missing_node_id"}
                n = self._workers.get(node_id, {})
                n["last_seen"] = _now()
                n.setdefault("latency_ema_s", 0.0)
                n.setdefault("fail_ema", 0.0)
                n.setdefault("tasks_done", 0)
                self._workers[node_id] = n
                self._emit("worker_registered", {"node_id": str(node_id), "from": str(client_addr)})
                self._save()
                return {
                    "ok": True,
                    "done": bool(self._done),
                    "round_id": int(self._round),
                    "secure_mode_effective": str(self._secure_mode),
                    "transport": str(self._transport),
                    "coordinator": {"host": str(self._advertise_host), "port": int(self.cfg.port)},
                    "peers": self._active_workers(),
                    "dht_table": self._dht(),
                    "auth_required": bool(self._auth_required()),
                }
            if op == "heartbeat":
                node_id = str(req.get("node_id", "") or "").strip()
                if node_id:
                    n = self._workers.setdefault(node_id, {})
                    n["last_seen"] = _now()
                    self._workers[node_id] = n
                since = int(req.get("since_event_id", 0) or 0)
                gossip = [x for x in self._events if int(x.get("id", 0) or 0) > since][-50:]
                return {
                    "ok": True,
                    "done": bool(self._done),
                    "round_id": int(self._round),
                    "transport": str(self._transport),
                    "coordinator": {"host": str(self._advertise_host), "port": int(self.cfg.port)},
                    "peers": self._active_workers(),
                    "dht_table": self._dht(),
                    "gossip": gossip,
                    "auth_required": bool(self._auth_required()),
                }
            if op == "pull_task":
                if self._done:
                    return {"ok": True, "done": True, "task": None}
                if not self._pending:
                    self._finish_round_if_needed(int(self._round))
                    return {
                        "ok": True,
                        "done": bool(self._done),
                        "task": None,
                        "retry_after_s": max(0.05, min(2.0, float(self.cfg.fedbuff.flush_timeout_s) / 8.0 if float(self.cfg.fedbuff.flush_timeout_s) > 0.0 else 0.2)),
                    }
                task = self._pending.popleft()
                tid = str(task.get("task_id", ""))
                self._leases[tid] = {"task": dict(task), "lease_until": _now() + max(5.0, float(self.cfg.lease_seconds))}
                return {"ok": True, "done": False, "task": dict(task)}
            if op == "submit_update":
                node_id = str(req.get("node_id", "") or "").strip()
                tid = str(req.get("task_id", "") or "").strip()
                rid = int(req.get("round_id", 0) or 0)
                cid = int(req.get("client_id", 0) or 0)
                rep = int(req.get("replica_id", 0) or 0)
                if not node_id:
                    return {"ok": False, "error": "missing_node_id"}
                if not tid:
                    return {"ok": False, "error": "missing_task_id"}
                if rid <= 0 or rid > int(self.cfg.rounds):
                    n_bad = self._workers.setdefault(str(node_id), {})
                    prev_bad = max(0.0, min(1.0, float(n_bad.get("fail_ema", 0.0) or 0.0)))
                    n_bad["fail_ema"] = float(min(1.0, 0.7 * prev_bad + 0.3))
                    n_bad["last_seen"] = _now()
                    self._workers[str(node_id)] = n_bad
                    return {"ok": False, "error": "invalid_round_id"}
                if rid < int(self._round) or bool(self._round_done.get(int(rid), False)):
                    return {"ok": True, "accepted": False, "reason": "stale_round"}
                if rid > int(self._round):
                    return {"ok": True, "accepted": False, "reason": "future_round"}
                vec_path = str(req.get("vector_path", "") or "").strip()
                vec: List[float] = []
                if vec_path:
                    try:
                        vec = _load_vector_from_path(vec_path)
                    except Exception as e:
                        return {"ok": False, "error": f"vector_path_load_failed({e})"}
                    if bool(req.get("vector_path_delete", False)):
                        try:
                            vp = Path(vec_path)
                            if not vp.is_absolute():
                                vp = (Path.cwd() / vp).resolve()
                            if vp.exists():
                                vp.unlink()
                        except Exception:
                            pass
                else:
                    vec = [float(x) for x in list(req.get("vector", []) or [])]
                if int(self.cfg.vector_dim) <= 0:
                    self.cfg.vector_dim = int(len(vec))
                    self._global_vec = [0.0 for _ in range(max(0, int(self.cfg.vector_dim)))]
                if len(vec) != int(self.cfg.vector_dim):
                    return {"ok": False, "error": f"invalid_vector_dim({len(vec)}!={int(self.cfg.vector_dim)})"}
                if tid in self._completed_tasks:
                    return {"ok": True, "accepted": False, "reason": "duplicate_task"}
                self._leases.pop(tid, None)
                vec = self._unmask(vec, node_id, rid, cid, rep, bool(req.get("masked", False)))
                vec = _clip_l2(vec, float(self.cfg.privacy.clip_norm))
                if float(self.cfg.privacy.noise_sigma) > 0.0:
                    s = _stable_seed([self.cfg.privacy.seed, rid, cid, rep, node_id, "srv_noise"])
                    vec = _gaussian_noise(vec, float(self.cfg.privacy.noise_sigma), int(s))
                self._buffers.setdefault(int(rid), []).append({"task_id": str(tid), "client_id": int(cid), "replica_id": int(rep), "sample_count": int(max(1, int(req.get("sample_count", 1) or 1))), "vector": vec})
                self._buffer_ts[int(rid)] = _now()
                self._completed_tasks[tid] = {"node_id": str(node_id), "ts": _now()}
                self._completed_clients.setdefault(int(rid), {}).setdefault(int(cid), {"node_id": str(node_id), "ts": _now()})
                n = self._workers.setdefault(str(node_id), {})
                lat = max(0.0, float(req.get("latency_s", 0.0) or 0.0))
                prev = max(0.0, float(n.get("latency_ema_s", 0.0) or 0.0))
                n["latency_ema_s"] = float(lat if prev <= 0.0 else (0.7 * prev + 0.3 * lat))
                prev_fail = max(0.0, min(1.0, float(n.get("fail_ema", 0.0) or 0.0)))
                n["fail_ema"] = float(max(0.0, 0.85 * prev_fail))
                n["tasks_done"] = int(n.get("tasks_done", 0) or 0) + 1
                n["last_seen"] = _now()
                self._workers[str(node_id)] = n
                flush = {"ok": False}
                if bool(self.cfg.fedbuff.enabled) and len(self._buffers.get(int(rid), [])) >= max(1, int(self.cfg.fedbuff.buffer_size)):
                    flush = self._flush(int(rid), "fedbuff_size")
                self._finish_round_if_needed(int(rid))
                self._save()
                return {"ok": True, "accepted": True, "done": bool(self._done), "flush": flush, "global_updates": int(self._global_updates)}
            if op == "status":
                self._save()
                return {
                    "ok": True,
                    "done": bool(self._done),
                    "round_id": int(self._round),
                    "transport": str(self._transport),
                    "coordinator": {"host": str(self._advertise_host), "port": int(self.cfg.port)},
                    "pending_tasks": int(len(self._pending)),
                    "global_updates": int(self._global_updates),
                    "workers": self._active_workers(),
                    "dht_table": self._dht(),
                    "secure_mode_effective": str(self._secure_mode),
                    "auth_required": bool(self._auth_required()),
                }
            if op == "shutdown":
                self._done = True
                self._emit("shutdown", {"from": str(client_addr)})
                self._save()
                return {"ok": True, "done": True}
            return {"ok": False, "error": "unknown_op"}

    def _serve_zmq_loop(self) -> None:
        zmq = _try_import_zmq()
        if zmq is None:
            return
        sock = self._zmq_sock
        if sock is None:
            return
        while not self._serve_stop.is_set():
            try:
                raw = sock.recv_string()
            except Exception:
                continue
            req: Dict[str, Any] = {}
            try:
                node = json.loads(str(raw).strip())
                if isinstance(node, dict):
                    req = node
            except Exception:
                req = {}
            resp = self.handle_rpc(req, "zmq")
            try:
                sock.send_string(json.dumps(resp, ensure_ascii=False))
            except Exception:
                continue

    def _discovery_reply_payload(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "op": "discover_reply",
            "service": str(self.cfg.discovery.service),
            "transport": str(self._transport),
            "host": str(self._advertise_host),
            "port": int(self.cfg.port),
            "secure_mode_effective": str(self._secure_mode),
            "auth_required": bool(self._auth_required()),
            "ts": float(_now()),
        }

    def _run_discovery_responder(self) -> None:
        sock: Optional[socket.socket] = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            except Exception:
                pass
            sock.bind(("", max(1, int(self.cfg.discovery.udp_port))))
            sock.settimeout(0.5)
            while not self._discovery_stop.is_set():
                try:
                    raw, addr = sock.recvfrom(8192)
                except socket.timeout:
                    continue
                except Exception:
                    continue
                try:
                    req = json.loads(raw.decode("utf-8", errors="ignore").strip())
                except Exception:
                    continue
                if not isinstance(req, dict):
                    continue
                op = str(req.get("op", "") or "").strip().lower()
                if op not in {"discover", "discover_request"}:
                    continue
                service = str(req.get("service", "") or "").strip()
                if service and service != str(self.cfg.discovery.service):
                    continue
                t_req = _normalize_transport(str(req.get("transport", "") or ""))
                if t_req and t_req != str(self._transport):
                    continue
                payload = self._discovery_reply_payload()
                try:
                    sock.sendto(json.dumps(payload, ensure_ascii=False).encode("utf-8"), addr)
                except Exception:
                    continue
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    def start(self) -> None:
        with self._lock:
            if self._server is not None or self._thread is not None or self._zmq_sock is not None:
                return
            self._serve_stop.clear()
            if str(self._transport) == "zeromq":
                zmq = _try_import_zmq()
                if zmq is None:
                    raise RuntimeError("zeromq_transport_requires_pyzmq")
                ctx = zmq.Context.instance()
                sock = ctx.socket(zmq.REP)
                sock.linger = 0
                timeout_ms = 500
                sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
                sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
                bind_host = str(self.cfg.host)
                bind_ep = f"tcp://{bind_host}"
                if int(self.cfg.port) <= 0:
                    self.cfg.port = int(sock.bind_to_random_port(bind_ep))
                else:
                    sock.bind(f"{bind_ep}:{int(self.cfg.port)}")
                self._zmq_ctx = ctx
                self._zmq_sock = sock
                self._thread = threading.Thread(target=self._serve_zmq_loop, daemon=True)
                self._thread.start()
                self._server_kind = "zeromq"
            else:
                self._server = _TCPServer((str(self.cfg.host), int(self.cfg.port)), _Handler, self)
                self.cfg.port = int(self._server.server_address[1])
                self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
                self._thread.start()
                self._server_kind = "tcp"
            self._advertise_host = _resolve_advertise_host(str(self.cfg.host), str(self.cfg.advertise_host))
            if bool(self.cfg.discovery.enabled):
                self._discovery_stop.clear()
                self._discovery_thread = threading.Thread(target=self._run_discovery_responder, daemon=True)
                self._discovery_thread.start()
                self._emit(
                    "discovery_started",
                    {
                        "service": str(self.cfg.discovery.service),
                        "udp_port": int(self.cfg.discovery.udp_port),
                        "advertise_host": str(self._advertise_host),
                    },
                )
            self._emit(
                "server_started",
                {
                    "host": str(self.cfg.host),
                    "port": int(self.cfg.port),
                    "transport": str(self._transport),
                    "advertise_host": str(self._advertise_host),
                },
            )
            self._save()

    def stop(self) -> None:
        disc_thread: Optional[threading.Thread] = None
        srv_thread: Optional[threading.Thread] = None
        with self._lock:
            if self._server is None and self._thread is None and self._zmq_sock is None:
                return
            self._serve_stop.set()
            self._discovery_stop.set()
            disc_thread = self._discovery_thread
            self._discovery_thread = None
            try:
                if self._server is not None:
                    self._server.shutdown()
            except Exception:
                pass
            try:
                if self._server is not None:
                    self._server.server_close()
            except Exception:
                pass
            try:
                if self._zmq_sock is not None:
                    self._zmq_sock.close(0)
            except Exception:
                pass
            self._server = None
            self._zmq_sock = None
            self._zmq_ctx = None
            srv_thread = self._thread
            self._thread = None
            self._server_kind = "none"
        if disc_thread is not None:
            disc_thread.join(timeout=2.0)
        if srv_thread is not None:
            srv_thread.join(timeout=2.0)

    def get_global_vector(self) -> List[float]:
        with self._lock:
            return [float(x) for x in self._global_vec]

    def run_until_complete(self, timeout_s: float = 0.0, poll_s: float = 0.2) -> Dict[str, Any]:
        start = _now()
        while True:
            with self._lock:
                if bool(self._done):
                    self._save()
                    return {
                        "ok": True,
                        "done": True,
                        "round_id": int(self._round),
                        "global_updates": int(self._global_updates),
                        "port": int(self.cfg.port),
                        "transport": str(self._transport),
                        "advertise_host": str(self._advertise_host),
                    }
            if float(timeout_s) > 0.0 and (_now() - start) >= float(timeout_s):
                with self._lock:
                    self._emit("timeout", {"timeout_s": float(timeout_s)})
                    self._save()
                    return {
                        "ok": True,
                        "done": bool(self._done),
                        "round_id": int(self._round),
                        "global_updates": int(self._global_updates),
                        "port": int(self.cfg.port),
                        "transport": str(self._transport),
                        "advertise_host": str(self._advertise_host),
                    }
            time.sleep(max(0.05, float(poll_s)))
