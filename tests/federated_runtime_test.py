from __future__ import annotations

import json
import os
import sys
import importlib.util
import shutil
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.automation.federated_runtime import CoordinatorConfig, FederatedCoordinator, FederatedWorker, WorkerConfig, run_demo_cluster


def _has_zmq() -> bool:
    return bool(importlib.util.find_spec("zmq"))


def test_federated_runtime_demo_cluster() -> None:
    out = run_demo_cluster(
        workers=3,
        rounds=2,
        clients_per_round=5,
        required_clients=3,
        coded_redundancy=1,
        vector_dim=12,
        timeout_s=25.0,
        state_dir="artifacts/audit/federated_runtime_test",
    )
    assert bool(out.get("ok", False))
    coord = out.get("coordinator", {})
    assert isinstance(coord, dict)
    assert bool(coord.get("done", False))
    assert int(coord.get("global_updates", 0) or 0) > 0
    workers = out.get("workers", [])
    assert isinstance(workers, list) and workers
    assert any(bool(w.get("ok", False)) for w in workers if isinstance(w, dict))

    state_path = Path("artifacts/audit/federated_runtime_test/state.json")
    assert state_path.exists()
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert int(raw.get("global_updates", 0) or 0) > 0


def test_federated_runtime_zeromq_discovery() -> None:
    if not _has_zmq():
        return
    out = run_demo_cluster(
        workers=2,
        rounds=1,
        clients_per_round=3,
        required_clients=2,
        coded_redundancy=0,
        vector_dim=8,
        timeout_s=20.0,
        state_dir="artifacts/audit/federated_runtime_test_zmq",
        transport="zeromq",
        discovery=True,
        discovery_port=19436,
        discovery_service="fed_runtime_test_zmq",
    )
    assert bool(out.get("ok", False))
    assert str(out.get("transport", "")) == "zeromq"
    workers = out.get("workers", [])
    assert isinstance(workers, list) and workers
    assert any(int(w.get("tasks_done", 0) or 0) > 0 for w in workers if isinstance(w, dict))


def test_federated_runtime_auth_token() -> None:
    token = "federated_test_token"
    coord = FederatedCoordinator(
        CoordinatorConfig(
            host="127.0.0.1",
            port=0,
            auth_token=token,
            rounds=1,
            clients_per_round=1,
            required_clients=1,
            vector_dim=8,
            state_path="artifacts/audit/federated_runtime_auth/state.json",
            events_path="artifacts/audit/federated_runtime_auth/events.jsonl",
        )
    )
    coord.start()
    try:
        bad = FederatedWorker(
            WorkerConfig(
                coordinator_host="127.0.0.1",
                coordinator_port=int(coord.cfg.port),
                auth_token="wrong_token",
                node_id="auth_bad_worker",
                max_runtime_s=1.0,
            )
        ).run()
        assert not bool(bad.get("ok", True))
        assert "unauthorized" in str(bad.get("error", ""))

        good = FederatedWorker(
            WorkerConfig(
                coordinator_host="127.0.0.1",
                coordinator_port=int(coord.cfg.port),
                auth_token=token,
                node_id="auth_good_worker",
                max_runtime_s=6.0,
            )
        ).run()
        assert bool(good.get("ok", False))

        done = coord.run_until_complete(timeout_s=8.0, poll_s=0.2)
        assert bool(done.get("done", False))
    finally:
        coord.stop()


def test_federated_runtime_resume_state() -> None:
    base = Path("artifacts/audit/federated_runtime_resume")
    if base.exists():
        shutil.rmtree(base.as_posix(), ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    state_path = base / "state.json"
    events_path = base / "events.jsonl"

    coord1 = FederatedCoordinator(
        CoordinatorConfig(
            host="127.0.0.1",
            port=0,
            rounds=1,
            clients_per_round=2,
            required_clients=2,
            coded_redundancy=0,
            vector_dim=4,
            resume_state=False,
            state_path=state_path.as_posix(),
            events_path=events_path.as_posix(),
        )
    )
    first_task_id = ""
    try:
        reg = coord1.handle_rpc({"op": "register", "node_id": "resume_w1"}, "local")
        assert bool(reg.get("ok", False))
        pull = coord1.handle_rpc({"op": "pull_task", "node_id": "resume_w1"}, "local")
        task = pull.get("task", {})
        assert isinstance(task, dict) and task
        first_task_id = str(task.get("task_id", ""))
        sub = coord1.handle_rpc(
            {
                "op": "submit_update",
                "node_id": "resume_w1",
                "task_id": str(task.get("task_id", "")),
                "round_id": int(task.get("round_id", 0) or 0),
                "client_id": int(task.get("client_id", 0) or 0),
                "replica_id": int(task.get("replica_id", 0) or 0),
                "sample_count": 8,
                "vector": [0.1, -0.1, 0.05, 0.2],
            },
            "local",
        )
        assert bool(sub.get("ok", False))
        before = json.loads(state_path.read_text(encoding="utf-8"))
        assert int(before.get("completed_tasks", 0) or 0) >= 1
    finally:
        coord1.stop()

    coord2 = FederatedCoordinator(
        CoordinatorConfig(
            host="127.0.0.1",
            port=0,
            rounds=1,
            clients_per_round=2,
            required_clients=2,
            coded_redundancy=0,
            vector_dim=4,
            resume_state=True,
            state_path=state_path.as_posix(),
            events_path=events_path.as_posix(),
        )
    )
    try:
        reg2 = coord2.handle_rpc({"op": "register", "node_id": "resume_w2"}, "local")
        assert bool(reg2.get("ok", False))
        pull2 = coord2.handle_rpc({"op": "pull_task", "node_id": "resume_w2"}, "local")
        task2 = pull2.get("task", {})
        assert isinstance(task2, dict) and task2
        assert str(task2.get("task_id", "")) != first_task_id
        sub2 = coord2.handle_rpc(
            {
                "op": "submit_update",
                "node_id": "resume_w2",
                "task_id": str(task2.get("task_id", "")),
                "round_id": int(task2.get("round_id", 0) or 0),
                "client_id": int(task2.get("client_id", 0) or 0),
                "replica_id": int(task2.get("replica_id", 0) or 0),
                "sample_count": 8,
                "vector": [0.0, 0.0, 0.1, 0.1],
            },
            "local",
        )
        assert bool(sub2.get("ok", False))
        assert bool(sub2.get("done", False))
    finally:
        coord2.stop()


def main() -> None:
    test_federated_runtime_demo_cluster()
    test_federated_runtime_zeromq_discovery()
    test_federated_runtime_auth_token()
    test_federated_runtime_resume_state()
    print("federated_runtime_ok")


if __name__ == "__main__":
    main()
