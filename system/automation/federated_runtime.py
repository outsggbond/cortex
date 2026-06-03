from __future__ import annotations

# -*- Compatibility re-export module — implementation split across three sub-modules -*-
# _fed_config   — dataclasses + utilities
# _fed_coordinator — FederatedCoordinator + _TCPServer + _Handler
# _fed_worker   — FederatedWorker + run_demo_cluster

from ._fed_config import (
    CoordinatorConfig,
    DiscoveryConfig,
    FedBuffConfig,
    PrivacyConfig,
    SecureAggConfig,
    WorkerConfig,
    _append_jsonl,
    _atomic_write_json,
    _clip_l2,
    _gaussian_noise,
    _load_vector_from_path,
    _local_ip_hint,
    _make_mask,
    _normalize_transport,
    _now,
    _resolve_advertise_host,
    _stable_seed,
    _try_import_zmq,
    _vector_digest,
    _vector_storage_payload,
)

from ._fed_coordinator import (
    FederatedCoordinator,
    _TCPServer,
    _Handler,
)

from ._fed_worker import (
    FederatedWorker,
    run_demo_cluster,
)
