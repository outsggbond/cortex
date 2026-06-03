from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.evaluation.adapter_artifacts import register_adapter_artifact
from system.runtime.bootstrap import apply_offline_runtime_defaults
from system.runtime.artifacts import disk_free_gb, run_artifact_prune
from system.runtime.support import (
    apply_runtime_env_overrides,
    env_flag,
    env_float,
    env_int,
    save_hardware,
    summarize_hardware,
)


def _restore_env(snapshot: dict[str, str | None]) -> None:
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_runtime_env_helpers() -> None:
    keys = [
        "TEST_RUNTIME_FLAG",
        "TEST_RUNTIME_FLOAT",
        "TEST_RUNTIME_INT",
    ]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        os.environ["TEST_RUNTIME_FLAG"] = "yes"
        os.environ["TEST_RUNTIME_FLOAT"] = "oops"
        os.environ["TEST_RUNTIME_INT"] = "7"
        assert env_flag("TEST_RUNTIME_FLAG") is True
        assert env_flag("TEST_RUNTIME_FLAG_MISSING", default=True) is True
        assert float(env_float("TEST_RUNTIME_FLOAT", 1.25)) == 1.25
        assert int(env_int("TEST_RUNTIME_INT", 3)) == 7
    finally:
        _restore_env(snapshot)


def test_apply_offline_runtime_defaults() -> None:
    keys = [
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
    ]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        apply_offline_runtime_defaults()
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"

        os.environ["HF_HUB_OFFLINE"] = "0"
        apply_offline_runtime_defaults()
        assert os.environ["HF_HUB_OFFLINE"] == "0"
    finally:
        _restore_env(snapshot)


def test_apply_runtime_env_overrides_and_hardware_persist() -> None:
    keys = [
        "AGENT_POLICY",
        "AGENT_CONTEXT",
        "NANOBRAIN_EMPATHY",
        "NANOBRAIN_EMPATHY_MODE",
        "EMBEDDING_REQUIRE",
        "DISABLE_EMBEDDINGS_MODEL",
        "DIALOGUE_TRANSFORMER_MODEL",
        "DIALOGUE_TRANSFORMER_DEVICE",
        "DIALOGUE_TRANSFORMER_MAX_TOKENS",
        "DIALOGUE_TRANSFORMER_LOCAL_ONLY",
        "TRANSFORMER_ADAPTER_ARTIFACT_REGISTRY",
    ]
    snapshot = {key: os.environ.get(key) for key in keys}
    try:
        args = SimpleNamespace(
            policy="steady",
            context="ops",
            empathy=0.65,
            empathy_mode="absolute",
            require_embeddings=False,
            pure_chat=True,
            dialogue_transformer_model="local/demo",
            dialogue_transformer_device="cpu",
            dialogue_transformer_max_tokens=96,
            dialogue_transformer_allow_download=False,
            dialogue_transformer_local_only=True,
            chat=True,
            train_file="",
            cycles=3,
            artifact_registry="artifacts/checkpoints/custom_registry.json",
        )
        apply_runtime_env_overrides(args, argv=["--chat"])
        assert os.environ["AGENT_POLICY"] == "steady"
        assert os.environ["AGENT_CONTEXT"] == "ops"
        assert os.environ["NANOBRAIN_EMPATHY"] == "0.65"
        assert os.environ["NANOBRAIN_EMPATHY_MODE"] == "absolute"
        assert os.environ["DISABLE_EMBEDDINGS_MODEL"] == "1"
        assert os.environ["DIALOGUE_TRANSFORMER_MODEL"] == "local/demo"
        assert os.environ["DIALOGUE_TRANSFORMER_DEVICE"] == "cpu"
        assert os.environ["DIALOGUE_TRANSFORMER_MAX_TOKENS"] == "96"
        assert os.environ["DIALOGUE_TRANSFORMER_LOCAL_ONLY"] == "1"
        assert os.environ["TRANSFORMER_ADAPTER_ARTIFACT_REGISTRY"] == "artifacts/checkpoints/custom_registry.json"
        assert int(args.cycles) == 0

        hw = SimpleNamespace(
            cpu_cores=8,
            cpu_name="unit-test-cpu",
            cpu_freq_mhz=3600,
            available_ram_gb=12.5,
            total_ram_gb=32.0,
            gpu_available=False,
        )
        assert "cpu=unit-test-cpu" in summarize_hardware(hw)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "hardware.json"
            save_hardware(hw, output_path=out.as_posix())
            assert out.exists()
            assert '"cpu_name": "unit-test-cpu"' in out.read_text(encoding="utf-8")
    finally:
        _restore_env(snapshot)


def test_runtime_artifact_prune_wrapper() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        registry = root / "adapter_artifacts.json"
        old_dir = root / "adapter_old"
        new_dir = root / "adapter_new"
        for path, payload in ((old_dir, b"old"), (new_dir, b"new")):
            path.mkdir(parents=True, exist_ok=True)
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (path / "adapter_model.bin").write_bytes(payload)

        register_adapter_artifact(
            registry_path=registry.as_posix(),
            adapter_path=old_dir.as_posix(),
            source="runtime_helper_old",
            used=True,
        )
        time.sleep(0.02)
        register_adapter_artifact(
            registry_path=registry.as_posix(),
            adapter_path=new_dir.as_posix(),
            source="runtime_helper_new",
            used=True,
        )

        args = SimpleNamespace(
            artifact_registry=registry.as_posix(),
            artifact_max_total_gb=0.0,
            artifact_keep_latest=1,
            artifact_keep_promoted=0,
            artifact_prune_remove_invalid=False,
            artifact_prune_delete_files=False,
            artifact_prune_archive=False,
            artifact_prune_archive_dir="",
            artifact_archive_signing_key_id="",
        )
        stats = run_artifact_prune(args)
        assert stats["registry"] == registry.as_posix()
        assert int(stats["remaining"]) == 1
        assert int(stats["removed_by_policy"]) == 1
        assert disk_free_gb(root.as_posix()) > 0.0


def main() -> None:
    test_runtime_env_helpers()
    test_apply_offline_runtime_defaults()
    test_apply_runtime_env_overrides_and_hardware_persist()
    test_runtime_artifact_prune_wrapper()
    print("runtime_helpers_ok")


if __name__ == "__main__":
    main()
