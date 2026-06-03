import json
import os
import random
import shutil
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.learning.dialogue.dialogue_dataset_builder import DialogueDatasetBuildConfig, build_dataset
from scripts.eval_and_promote import BucketMetrics, BucketThreshold, EvalMetrics, decide_promotion
from scripts.build_replay_pool import ReplayBuildConfig, build_replay_pool
from scripts.auto_iterate import _build_train_cmd as _build_auto_train_cmd
from scripts.auto_iterate import _dataset_manifest_penalty, _load_score_weights, _metric_score, _resolve_candidate_grid
from scripts.train_adapter import (
    _append_jsonl,
    _active_workers_for_round,
    _adapter_signature,
    _assign_clients_to_workers,
    _aggregate_client_adapters,
    _build_federated_child_cmd,
    _compact_federated_round_report_for_state,
    _cap_client_weight_share,
    _client_weight_stats,
    _build_worker_perf_penalty_map,
    _client_aggregation_weight,
    _cleanup_round_client_artifacts,
    _evaluate_federated_governance,
    _fedavg_state_dict,
    _load_federated_workers,
    _apply_bucket_balance_to_client_weights,
    _normalize_client_weight_map,
    _parse_worker_resource_from_probe_output,
    _partition_federated_rows,
    _prepare_federated_state_round_reports,
    _prune_old_federated_round_dirs,
    _read_jsonl,
    _sanitize_worker_health_state,
    _sanitize_worker_perf_state,
    _update_worker_health_after_round,
    _update_worker_perf_after_round,
    _validate_federated_client,
    _write_jsonl,
)
from scripts.auto_iterate_loop import (
    _bandit_epsilon_for_round,
    _build_bandit_reward,
    _build_adaptive_score_weights,
    _build_fallback_manifest_from_report,
    _build_bucket_threshold_payload,
    _build_budget_overrides,
    _build_candidate_space_overrides,
    _build_dataset_penalty_overrides,
    _blend_bandit_stats,
    _choose_bandit_profile,
    _build_round_adaptive_overrides,
    _build_target_module_overrides,
    _build_trend_memory_overrides,
    _build_train_policy_overrides,
    _extract_flag_value,
    _load_bandit_state,
    _load_score_weights as _load_loop_score_weights,
    _merge_replay_files,
    _normalize_bandit_context,
    _resolve_bandit_contexts,
    _resolve_bandit_profiles,
    _save_bandit_state,
    _select_bandit_context,
    should_stop_early,
)
from scripts.mine_hard_cases import (
    build_manifest as build_hard_manifest,
    collect_hard_cases_from_report,
    merge_hard_cases,
)
from system.evaluation.adapter_artifacts import (
    audit_adapter_artifacts,
    maintain_adapter_artifacts,
    prune_adapter_artifacts,
    register_adapter_artifact,
    resolve_best_adapter_path,
    restore_archived_artifact,
)


def _test_dataset_builder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        patterns = root / "dialogue_patterns.json"
        chat_mem = root / "chat_memory.jsonl"
        refl = root / "dialogue_reflections.jsonl"
        train_out = root / "adapter_train.jsonl"
        eval_out = root / "adapter_eval.jsonl"
        manifest_out = root / "adapter_manifest.json"

        patterns.write_text(
            json.dumps(
                {
                    "smalltalk": {
                        "pairs": [
                            ["hello", "hello there"],
                            ["how are you", "I am fine, thanks"],
                            ["good night", "Sleep well and rest"],
                            ["tell me a joke", "Here is a short joke about coding."],
                        ]
                    },
                    "qa": {
                        "pairs": [
                            ["what is machine learning", "Machine learning models patterns from data."],
                            ["where should I start Python", "Start with syntax, lists, and small scripts."],
                        ]
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        chat_mem.write_text(
            "\n".join(
                [
                    json.dumps({"role": "user", "text": "what is ml"}, ensure_ascii=False),
                    json.dumps({"role": "assistant", "text": "ml means machine learning"}, ensure_ascii=False),
                    json.dumps({"role": "user", "text": "why is debugging useful"}, ensure_ascii=False),
                    json.dumps(
                        {
                            "role": "assistant",
                            "text": "Debugging finds defects and improves software reliability.",
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        refl.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "query": "how to plan a study routine",
                            "reply": "idk",
                            "expected": "create a weekly plan and track progress",
                            "repaired": "create weekly goals, track tasks, review weekend",
                            "quality": 0.90,
                            "context": "selfplay",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "query": "why does overfitting happen",
                            "reply": "not sure",
                            "expected": "it memorizes noise and fails to generalize",
                            "repaired": "model fits noise and misses broad patterns",
                            "quality": 0.86,
                            "context": "selfplay",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "query": "how to start ml",
                            "reply": "idk",
                            "expected": "start with linear algebra",
                            "repaired": "start with linear algebra and probability",
                            "quality": 0.88,
                            "context": "selfplay",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "query": "how to debug a crash",
                            "reply": "guess and try",
                            "expected": "collect logs and isolate the failing step",
                            "repaired": "read logs, isolate crash, then test a patch",
                            "quality": 0.85,
                            "context": "selfplay",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "query": "what causes slow code",
                            "reply": "maybe hardware",
                            "expected": "inefficient loops and repeated allocations can slow code",
                            "repaired": "slow code comes from nested loops, copy churn, blocked io",
                            "quality": 0.82,
                            "context": "selfplay",
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        cfg = DialogueDatasetBuildConfig(
            patterns_path=patterns.as_posix(),
            chat_memory_path=chat_mem.as_posix(),
            reflections_path=refl.as_posix(),
            output_train_path=train_out.as_posix(),
            output_eval_path=eval_out.as_posix(),
            output_manifest_path=manifest_out.as_posix(),
            eval_ratio=0.25,
            min_quality=0.3,
            max_samples=6,
            hard_min_ratio=0.5,
            hard_max_ratio=0.8,
            max_per_prompt=10,
            max_per_response=10,
            bucket_classifier_enabled=True,
            bucket_classifier_min_docs=2,
            bucket_classifier_min_conf=0.55,
            seed=3,
        )
        manifest = build_dataset(cfg)
        assert manifest["total_samples"] == 6
        assert manifest["hard_samples"] >= 3
        assert 0.45 <= float(manifest["hard_ratio"]) <= 0.85
        assert isinstance(manifest.get("by_bucket"), dict) and manifest["by_bucket"]
        assert "rebucketed_samples" in manifest
        assert train_out.exists()
        assert eval_out.exists()
        assert manifest_out.exists()
        train_rows = [json.loads(ln) for ln in train_out.read_text(encoding="utf-8").splitlines() if ln.strip()]
        eval_rows = [json.loads(ln) for ln in eval_out.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert train_rows
        assert eval_rows
        all_rows = train_rows + eval_rows
        assert any(bool(row.get("hard_sample", False)) for row in all_rows)
        assert all(str(row.get("bucket", "")).strip() for row in all_rows)


def _test_jsonl_gzip_helpers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gz_path = root / "round_reports.jsonl.gz"
        rows = [
            {"round": 1, "score": 0.5},
            {"round": 2, "score": 0.7},
        ]
        _write_jsonl(gz_path, rows)
        loaded = _read_jsonl(gz_path)
        assert len(loaded) == 2
        assert int(loaded[0].get("round", 0)) == 1
        _append_jsonl(gz_path, {"round": 3, "score": 0.8})
        loaded2 = _read_jsonl(gz_path)
        assert len(loaded2) == 3
        assert int(loaded2[-1].get("round", 0)) == 3


def _test_promote_decision() -> None:
    thresholds = {
        "smalltalk": BucketThreshold(min_quality=0.55, min_pass_rate=0.45),
        "qa": BucketThreshold(min_quality=0.58, min_pass_rate=0.50),
        "reasoning": BucketThreshold(min_quality=0.60, min_pass_rate=0.52),
        "safety": BucketThreshold(min_quality=0.62, min_pass_rate=0.56),
    }
    candidate = EvalMetrics(
        sample_count=20,
        avg_quality=0.66,
        avg_overlap=0.51,
        pass_rate=0.62,
        avg_semantic=0.63,
        by_bucket={
            "qa": BucketMetrics(sample_count=6, avg_quality=0.64, avg_overlap=0.50, pass_rate=0.67, avg_semantic=0.60),
            "reasoning": BucketMetrics(sample_count=7, avg_quality=0.65, avg_overlap=0.51, pass_rate=0.57, avg_semantic=0.62),
            "safety": BucketMetrics(sample_count=7, avg_quality=0.69, avg_overlap=0.55, pass_rate=0.71, avg_semantic=0.68),
        },
    )
    baseline = EvalMetrics(
        sample_count=20,
        avg_quality=0.61,
        avg_overlap=0.48,
        pass_rate=0.57,
        avg_semantic=0.57,
        by_bucket={
            "qa": BucketMetrics(sample_count=6, avg_quality=0.61, avg_overlap=0.47, pass_rate=0.58, avg_semantic=0.55),
            "reasoning": BucketMetrics(sample_count=7, avg_quality=0.61, avg_overlap=0.48, pass_rate=0.54, avg_semantic=0.57),
            "safety": BucketMetrics(sample_count=7, avg_quality=0.65, avg_overlap=0.52, pass_rate=0.64, avg_semantic=0.60),
        },
    )
    promote, reason = decide_promotion(
        candidate=candidate,
        baseline=baseline,
        min_quality=0.58,
        min_pass_rate=0.45,
        min_semantic=0.45,
        min_improve=0.015,
        bucket_thresholds=thresholds,
        min_bucket_samples=1,
    )
    assert promote, reason

    bucket_fail = EvalMetrics(
        sample_count=20,
        avg_quality=0.66,
        avg_overlap=0.51,
        pass_rate=0.62,
        avg_semantic=0.62,
        by_bucket={
            "qa": BucketMetrics(sample_count=6, avg_quality=0.64, avg_overlap=0.50, pass_rate=0.67, avg_semantic=0.60),
            "safety": BucketMetrics(sample_count=7, avg_quality=0.69, avg_overlap=0.55, pass_rate=0.40, avg_semantic=0.64),
        },
    )
    promote2, _ = decide_promotion(
        candidate=bucket_fail,
        baseline=baseline,
        min_quality=0.58,
        min_pass_rate=0.45,
        min_semantic=0.45,
        min_improve=0.015,
        bucket_thresholds=thresholds,
        min_bucket_samples=1,
    )
    assert not promote2

    regress = EvalMetrics(
        sample_count=20,
        avg_quality=0.63,
        avg_overlap=0.50,
        pass_rate=0.58,
        avg_semantic=0.54,
        by_bucket={
            "qa": BucketMetrics(sample_count=6, avg_quality=0.63, avg_overlap=0.50, pass_rate=0.65, avg_semantic=0.57),
            "reasoning": BucketMetrics(sample_count=7, avg_quality=0.56, avg_overlap=0.47, pass_rate=0.48, avg_semantic=0.48),
            "safety": BucketMetrics(sample_count=7, avg_quality=0.66, avg_overlap=0.53, pass_rate=0.64, avg_semantic=0.58),
        },
    )
    promote3, _ = decide_promotion(
        candidate=regress,
        baseline=baseline,
        min_quality=0.58,
        min_pass_rate=0.45,
        min_semantic=0.45,
        min_improve=0.015,
        bucket_thresholds=thresholds,
        min_bucket_samples=1,
        bucket_max_quality_regress=0.02,
        bucket_max_pass_regress=0.03,
        bucket_max_semantic_regress=0.03,
    )
    assert not promote3

    semantic_fail = EvalMetrics(
        sample_count=20,
        avg_quality=0.67,
        avg_overlap=0.54,
        pass_rate=0.64,
        avg_semantic=0.20,
    )
    promote4, _ = decide_promotion(
        candidate=semantic_fail,
        baseline=baseline,
        min_quality=0.58,
        min_pass_rate=0.45,
        min_semantic=0.30,
        min_improve=0.015,
        bucket_thresholds=thresholds,
        min_bucket_samples=1,
    )
    assert not promote4


def _test_replay_builder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        chat_mem = root / "chat_memory.jsonl"
        refl = root / "dialogue_reflections.jsonl"
        replay_out = root / "adapter_replay.jsonl"
        replay_manifest = root / "adapter_replay_manifest.json"

        chat_mem.write_text(
            "\n".join(
                [
                    json.dumps({"role": "user", "text": "what is python"}, ensure_ascii=False),
                    json.dumps({"role": "assistant", "text": "Python is a popular programming language."}, ensure_ascii=False),
                    json.dumps({"role": "user", "text": "hi"}, ensure_ascii=False),
                    json.dumps({"role": "assistant", "text": "hello there"}, ensure_ascii=False),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        refl.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "query": "how to debug crash",
                            "expected": "collect logs and isolate failing path",
                            "repaired": "collect logs, isolate failing path, and verify minimal repro",
                            "quality": 0.9,
                            "context": "selfplay",
                        },
                        ensure_ascii=False,
                    )
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        cfg = ReplayBuildConfig(
            chat_memory_path=chat_mem.as_posix(),
            reflections_path=refl.as_posix(),
            output_path=replay_out.as_posix(),
            manifest_path=replay_manifest.as_posix(),
            max_samples=20,
            min_score=0.5,
            seed=3,
        )
        manifest = build_replay_pool(cfg)
        assert replay_out.exists()
        assert replay_manifest.exists()
        assert int(manifest.get("total_samples", 0)) >= 1
        assert "by_bucket" in manifest
        rows = [json.loads(ln) for ln in replay_out.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert rows
        assert all(str(r.get("prompt", "")).strip() and str(r.get("response", "")).strip() for r in rows)


def _test_hard_case_mining() -> None:
    report = {
        "candidate_runs": [
            {
                "eval_report": {
                    "candidate_examples": [
                        {
                            "prompt": "p1",
                            "expected": "e1",
                            "generated": "g1",
                            "pass": False,
                            "quality": 0.10,
                            "overlap": 0.20,
                            "semantic": 0.30,
                            "bucket": "qa",
                        },
                        {
                            "prompt": "ok",
                            "expected": "ok",
                            "generated": "ok",
                            "pass": True,
                            "quality": 0.90,
                            "overlap": 0.90,
                            "semantic": 0.90,
                            "bucket": "smalltalk",
                        },
                    ]
                }
            }
        ],
        "eval": {
            "report": {
                "candidate_examples": [
                    {
                        "prompt": "p2",
                        "expected": "e2",
                        "generated": "g2",
                        "pass": False,
                        "quality": 0.25,
                        "overlap": 0.30,
                        "semantic": 0.35,
                        "bucket": "reasoning",
                    }
                ]
            }
        },
        "canary": {
            "report": {
                "candidate_examples": [
                    {
                        "prompt": "p3",
                        "expected": "e3",
                        "generated": "g3",
                        "pass": False,
                        "quality": 0.45,
                        "overlap": 0.40,
                        "semantic": 0.40,
                        "bucket": "safety",
                    }
                ]
            }
        },
    }
    mined = collect_hard_cases_from_report(report, report_name="round_1.json", min_severity=0.8)
    assert len(mined) == 3
    assert all(bool(x.get("hard_sample", False)) for x in mined)
    assert all(str(x.get("source", "")) == "hard_mine" for x in mined)
    assert all(str(x.get("error_tag", "")).strip() for x in mined)
    assert all(str(x.get("meta", {}).get("report", "")) == "round_1.json" for x in mined)

    existing = [
        {"prompt": "p1", "response": "e1", "source": "seed", "bucket": "qa", "score": 0.75, "_severity": 0.20},
        {"prompt": "base", "response": "base_r", "source": "seed", "bucket": "qa", "score": 0.78, "_severity": 0.15},
    ]
    merged, added = merge_hard_cases(existing_rows=existing, new_rows=mined, max_add=3, max_total=10, max_add_per_tag=1)
    assert added >= 1
    assert len(merged) >= 3
    by_key = {(str(x.get("prompt", "")), str(x.get("response", ""))): x for x in merged}
    assert ("p1", "e1") in by_key
    assert float(by_key[("p1", "e1")].get("_severity", 0.0)) >= 2.0

    manifest = build_hard_manifest(merged, added=added, reports=["round_1.json"])
    assert int(manifest.get("total_samples", 0)) == len(merged)
    assert int(manifest.get("added_samples", 0)) == int(added)
    assert "hard_mine" in dict(manifest.get("by_source", {}))
    assert isinstance(manifest.get("by_error_tag"), dict) and manifest.get("by_error_tag")


def _test_fallback_manifest_from_report() -> None:
    report = {
        "candidate_runs": [
            {
                "eval_report": {
                    "candidate_examples": [
                        {
                            "bucket": "reasoning",
                            "quality": 0.30,
                            "overlap": 0.20,
                            "semantic": 0.20,
                            "pass": False,
                        },
                        {
                            "bucket": "qa",
                            "quality": 0.55,
                            "overlap": 0.10,
                            "semantic": 0.30,
                            "pass": False,
                        },
                        {
                            "bucket": "smalltalk",
                            "quality": 0.90,
                            "overlap": 0.90,
                            "semantic": 0.90,
                            "pass": True,
                        },
                    ]
                }
            }
        ],
        "eval": {
            "report": {
                "candidate_examples": [
                    {
                        "bucket": "safety",
                        "quality": 0.40,
                        "overlap": 0.40,
                        "semantic": 0.40,
                        "pass": False,
                    }
                ]
            }
        },
        "canary": {
            "report": {
                "candidate_examples": [
                    {
                        "bucket": "qa",
                        "quality": 0.62,
                        "overlap": 0.40,
                        "semantic": 0.41,
                        "pass": False,
                    }
                ]
            }
        },
    }
    fallback = _build_fallback_manifest_from_report(report)
    assert fallback
    assert str(fallback.get("source", "")) == "report_fallback"
    assert int(fallback.get("total_samples", 0)) == 4
    by_tag = dict(fallback.get("by_error_tag", {}))
    assert int(by_tag.get("semantic_drift", 0)) >= 1
    assert int(by_tag.get("hallucination", 0)) >= 1
    assert int(by_tag.get("safety", 0)) >= 1
    by_bucket = dict(fallback.get("by_bucket", {}))
    assert int(by_bucket.get("qa", 0)) >= 2
    assert float(fallback.get("avg_severity", 0.0)) > 0.0

    assert _build_fallback_manifest_from_report({}) == {}


def _test_loop_replay_helpers() -> None:
    args = ["--x", "1", "--replay-data", "a.jsonl", "--y", "2", "--replay-data", "b.jsonl"]
    assert _extract_flag_value(args, "--replay-data") == "b.jsonl"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src_a = root / "a.jsonl"
        src_b = root / "b.jsonl"
        merged = root / "merged.jsonl"

        src_a.write_text(
            "\n".join(
                [
                    json.dumps({"prompt": "p1", "response": "r1", "score": 0.4}, ensure_ascii=False),
                    json.dumps({"prompt": "p2", "response": "r2", "score": 0.8}, ensure_ascii=False),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        src_b.write_text(
            "\n".join(
                [
                    json.dumps({"prompt": "p1", "response": "r1", "score": 0.9}, ensure_ascii=False),
                    json.dumps({"prompt": "p3", "response": "r3", "score": "0.6"}, ensure_ascii=False),
                    json.dumps({"prompt": "bad", "response": "", "score": 1.0}, ensure_ascii=False),
                    json.dumps({"prompt": "p4", "response": "r4", "score": None}, ensure_ascii=False),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        count = _merge_replay_files(src_a.as_posix(), src_b.as_posix(), merged, max_total=3)
        assert count == 3
        rows = [json.loads(ln) for ln in merged.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(rows) == 3
        keys = {(str(x.get("prompt", "")), str(x.get("response", ""))): x for x in rows}
        assert ("p1", "r1") in keys
        assert abs(float(keys[("p1", "r1")].get("score", 0.0)) - 0.9) < 1e-9


def _test_federated_train_helpers() -> None:
    rows = []
    for i in range(18):
        rows.append(
            {
                "prompt": f"p{i}",
                "response": f"r{i}",
                "bucket": "reasoning" if i % 3 == 0 else ("safety" if i % 3 == 1 else "qa"),
                "score": 0.6,
            }
        )
    parts_bucket = _partition_federated_rows(rows, num_clients=4, method="bucket", seed=7)
    parts_iid = _partition_federated_rows(rows, num_clients=4, method="iid", seed=7)
    assert len(parts_bucket) == 4 and len(parts_iid) == 4
    assert sum(len(x) for x in parts_bucket) == len(rows)
    assert sum(len(x) for x in parts_iid) == len(rows)

    workers = [
        {"name": "w1", "max_concurrency": 1},
        {"name": "w2", "max_concurrency": 2},
    ]
    assign = _assign_clients_to_workers([0, 1, 2, 3, 4], workers, seed=7)
    assert set(assign.keys()) == {0, 1, 2, 3, 4}
    used_workers = {str(v.get("name", "")) for v in assign.values()}
    assert "w1" in used_workers and "w2" in used_workers
    assign_pressure = _assign_clients_to_workers(
        [0, 1, 2, 3, 4, 5],
        [{"name": "hot", "max_concurrency": 1}, {"name": "cool", "max_concurrency": 1}],
        seed=7,
        worker_pressures={"hot": 0.9, "cool": 0.0},
    )
    counts = {"hot": 0, "cool": 0}
    for item in assign_pressure.values():
        counts[str(item.get("name", ""))] += 1
    assert int(counts.get("cool", 0)) > int(counts.get("hot", 0))

    h0 = _sanitize_worker_health_state({}, workers)
    active0, cooled0 = _active_workers_for_round(workers, h0, round_id=1)
    assert len(active0) == 2
    assert cooled0 == []
    h1 = _update_worker_health_after_round(
        h0,
        client_steps=[
            {"worker": "w1", "ok": False, "step": {"timeout": True}},
            {"worker": "w2", "ok": True, "step": {"timeout": False}},
        ],
        round_id=1,
        fail_threshold=1,
        cooldown_rounds=2,
    )
    cooled_now = list(h1.get("cooled_this_round", []))
    assert "w1" in cooled_now
    h1_state = _sanitize_worker_health_state(h1.get("state", {}), workers)
    active1, cooled1 = _active_workers_for_round(workers, h1_state, round_id=2)
    assert "w1" in cooled1
    assert len(active1) == 1
    active2, cooled2 = _active_workers_for_round(workers, h1_state, round_id=3)
    assert "w1" not in cooled2
    assert len(active2) == 2
    h2 = _update_worker_health_after_round(
        h1_state,
        client_steps=[
            {
                "worker": "w2",
                "ok": True,
                "attempts": [
                    {"worker": "w2", "ok": False, "timeout": True},
                    {"worker": "w2", "ok": True, "timeout": False},
                ],
            }
        ],
        round_id=3,
        fail_threshold=3,
        cooldown_rounds=1,
    )
    h2_state = _sanitize_worker_health_state(h2.get("state", {}), workers)
    assert int(h2_state["w2"]["total_fail"]) >= 1
    assert int(h2_state["w2"]["total_success"]) >= 2

    p0 = _sanitize_worker_perf_state({}, workers)
    p1 = _update_worker_perf_after_round(
        p0,
        client_steps=[
            {
                "worker": "w1",
                "ok": True,
                "step": {"elapsed_s": 2.0},
                "attempts": [
                    {"worker": "w1", "ok": False, "elapsed_s": 1.0},
                    {"worker": "w1", "ok": True, "elapsed_s": 2.0},
                ],
            },
            {"worker": "w2", "ok": True, "step": {"elapsed_s": 0.5}},
        ],
        workers=workers,
        alpha=0.5,
    )
    p1s = _sanitize_worker_perf_state(p1, workers)
    assert float(p1s["w1"]["ema_fail_rate"]) > 0.0
    assert float(p1s["w1"]["ema_elapsed_s"]) >= 1.0
    perf_pen = _build_worker_perf_penalty_map(
        workers,
        p1s,
        enabled=True,
        fail_weight=0.7,
        latency_weight=0.3,
    )
    assert float(perf_pen.get("w1", 0.0)) >= float(perf_pen.get("w2", 0.0))
    w_samples = _client_aggregation_weight(
        item={"summary": {"train_loss": 0.5}},
        rows_count=20,
        strategy="samples",
        loss_eps=1e-4,
        loss_clip=10.0,
    )
    w_uniform = _client_aggregation_weight(
        item={"summary": {"train_loss": 0.5}},
        rows_count=20,
        strategy="uniform",
        loss_eps=1e-4,
        loss_clip=10.0,
    )
    w_inv = _client_aggregation_weight(
        item={"summary": {"train_loss": 0.5}},
        rows_count=20,
        strategy="inverse_loss",
        loss_eps=1e-4,
        loss_clip=10.0,
    )
    w_hybrid = _client_aggregation_weight(
        item={"summary": {"train_loss": 0.5}},
        rows_count=20,
        strategy="hybrid",
        loss_eps=1e-4,
        loss_clip=10.0,
    )
    w_multi = _client_aggregation_weight(
        item={"summary": {"train_loss": 0.5, "eval_metrics": {"eval_loss": 0.25}}},
        rows_count=20,
        strategy="multi_signal",
        loss_eps=1e-4,
        loss_clip=10.0,
        sample_power=1.0,
        loss_power=1.0,
        eval_loss_power=1.0,
    )
    w_pen = _client_aggregation_weight(
        item={
            "summary": {"train_loss": 0.5},
            "retries_used": 1,
            "attempts": [
                {"ok": False, "timeout": True},
                {"ok": True, "timeout": False},
            ],
        },
        rows_count=20,
        strategy="uniform",
        loss_eps=1e-4,
        loss_clip=10.0,
        timeout_penalty=0.5,
        retry_penalty=0.2,
        fail_attempt_penalty=0.1,
    )
    assert abs(float(w_uniform) - 1.0) < 1e-9
    assert float(w_samples) > float(w_uniform)
    assert float(w_inv) > float(w_uniform)
    assert float(w_hybrid) > float(w_uniform)
    assert float(w_multi) > float(w_uniform)
    assert float(w_pen) < float(w_uniform)
    norm_w = _normalize_client_weight_map(
        {0: 0.1, 1: 10.0},
        min_weight=1e-6,
        max_ratio=4.0,
        normalize_mean=True,
    )
    assert 0 in norm_w and 1 in norm_w
    assert float(norm_w[1]) / float(norm_w[0]) <= 4.000001
    mean_w = (float(norm_w[0]) + float(norm_w[1])) / 2.0
    assert abs(mean_w - 1.0) < 1e-6
    capped = _cap_client_weight_share(
        {0: 9.0, 1: 1.0, 2: 1.0},
        max_share=0.6,
        min_weight=1e-6,
    )
    cstats = _client_weight_stats(capped)
    assert float(cstats.get("max_share", 1.0)) <= 0.600001
    assert float(cstats.get("entropy_norm", 0.0)) > 0.0
    bb = _apply_bucket_balance_to_client_weights(
        weights={0: 1.0, 1: 1.0},
        client_bucket_dists={0: {"safety": 1.0}, 1: {"qa": 1.0}},
        target_dist={"safety": 0.8, "qa": 0.2},
        strength=1.0,
        eps=1e-6,
    )
    bbw = dict(bb.get("weights", {}))
    assert float(bbw.get(0, 0.0)) > float(bbw.get(1, 0.0))
    bb0 = _apply_bucket_balance_to_client_weights(
        weights={0: 1.0, 1: 1.0},
        client_bucket_dists={0: {"safety": 1.0}, 1: {"qa": 1.0}},
        target_dist={"safety": 0.8, "qa": 0.2},
        strength=0.0,
        eps=1e-6,
    )
    bb0w = dict(bb0.get("weights", {}))
    assert abs(float(bb0w.get(0, 0.0)) - 1.0) < 1e-9
    assert abs(float(bb0w.get(1, 0.0)) - 1.0) < 1e-9

    parsed_res = _parse_worker_resource_from_probe_output(
        'probe_ok\n{"cpu_percent": 17, "mem_percent": 40, "gpu_percent": 5, "pressure": 0.26}'
    )
    assert abs(float(parsed_res.get("pressure", 0.0)) - 0.26) < 1e-9
    assert float(parsed_res.get("cpu_percent", -1.0)) == 17.0
    assert float(parsed_res.get("mem_percent", -1.0)) == 40.0

    sd1 = {"lora_A": torch.tensor([1.0, 3.0], dtype=torch.float32)}
    sd2 = {"lora_A": torch.tensor([4.0, 6.0], dtype=torch.float32)}
    avg = _fedavg_state_dict([(sd1, 2.0), (sd2, 1.0)])
    vec = avg.get("lora_A")
    assert vec is not None
    assert abs(float(vec[0]) - 2.0) < 1e-6
    assert abs(float(vec[1]) - 4.0) < 1e-6

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        c1 = root / "c1"
        c2 = root / "c2"
        c3 = root / "c3"
        c1.mkdir(parents=True, exist_ok=True)
        c2.mkdir(parents=True, exist_ok=True)
        c3.mkdir(parents=True, exist_ok=True)
        (c1 / "adapter_config.json").write_text("{}", encoding="utf-8")
        (c2 / "adapter_config.json").write_text("{}", encoding="utf-8")
        (c3 / "adapter_config.json").write_text("{}", encoding="utf-8")
        torch.save({"lora_A": torch.tensor([1.0, 1.0])}, (c1 / "adapter_model.bin").as_posix())
        torch.save({"lora_A": torch.tensor([4.0, 7.0])}, (c2 / "adapter_model.bin").as_posix())
        torch.save({"lora_A": torch.tensor([100.0, 100.0])}, (c3 / "adapter_model.bin").as_posix())
        out = root / "global"
        rep = _aggregate_client_adapters([(c1, 1.0), (c2, 3.0)], out_dir=out)
        assert Path(str(rep.get("adapter_path", ""))).exists()
        assert str(rep.get("signature", "")).strip()
        assert str(rep.get("method", "")) == "avg"
        csign = dict(rep.get("client_signatures", {}))
        assert len(csign) == 2
        agg_sd = torch.load((out / "adapter_model.bin").as_posix(), map_location="cpu")
        t = agg_sd.get("lora_A")
        assert t is not None
        assert abs(float(t[0]) - 3.25) < 1e-6
        assert abs(float(t[1]) - 5.5) < 1e-6
        sig = _adapter_signature(out)
        assert str(sig) == str(rep.get("signature", ""))

        loww_out = root / "global_loww"
        loww_rep = _aggregate_client_adapters([(c1, 0.1), (c2, 0.9)], out_dir=loww_out)
        loww_sd = torch.load((loww_out / "adapter_model.bin").as_posix(), map_location="cpu")
        loww_t = loww_sd.get("lora_A")
        assert loww_t is not None
        assert abs(float(loww_t[0]) - 3.7) < 1e-6
        assert abs(float(loww_t[1]) - 6.4) < 1e-6
        assert abs(float(loww_rep.get("total_weight", 0.0)) - 1.0) < 1e-6

        med_out = root / "global_median"
        med_rep = _aggregate_client_adapters([(c1, 1.0), (c2, 1.0), (c3, 1.0)], out_dir=med_out, method="median")
        med_sd = torch.load((med_out / "adapter_model.bin").as_posix(), map_location="cpu")
        med_t = med_sd.get("lora_A")
        assert med_t is not None
        assert str(med_rep.get("method", "")) == "median"
        assert abs(float(med_t[0]) - 4.0) < 1e-6
        assert abs(float(med_t[1]) - 7.0) < 1e-6

        trim_out = root / "global_trim"
        trim_rep = _aggregate_client_adapters(
            [(c1, 1.0), (c2, 1.0), (c3, 1.0)],
            out_dir=trim_out,
            method="trimmed_mean",
            trim_ratio=0.34,
        )
        trim_sd = torch.load((trim_out / "adapter_model.bin").as_posix(), map_location="cpu")
        trim_t = trim_sd.get("lora_A")
        assert trim_t is not None
        assert str(trim_rep.get("method", "")) == "trimmed_mean"
        assert abs(float(trim_t[0]) - 4.0) < 1e-6
        assert abs(float(trim_t[1]) - 7.0) < 1e-6

        ok_health = _validate_federated_client(
            step={"returncode": 0, "timeout": False},
            summary={"train_samples_used": 9},
            adapter_path=out.as_posix(),
            expected_rows=9,
        )
        assert bool(ok_health.get("ok", False))
        assert str(ok_health.get("adapter_signature", "")).strip()

        bad_health = _validate_federated_client(
            step={"returncode": 1, "timeout": True},
            summary={},
            adapter_path=(root / "missing").as_posix(),
            expected_rows=9,
        )
        assert not bool(bad_health.get("ok", True))
        errs = list(bad_health.get("errors", []))
        assert "non_zero_returncode" in errs
        assert "timeout" in errs
        assert "adapter_missing" in errs

        round_dir = root / "round_x"
        (round_dir / "client_0_out").mkdir(parents=True, exist_ok=True)
        (round_dir / "client_1_out").mkdir(parents=True, exist_ok=True)
        (round_dir / "client_0_train.jsonl").write_text("", encoding="utf-8")
        (round_dir / "client_1_train.jsonl").write_text("", encoding="utf-8")
        cleanup = _cleanup_round_client_artifacts(
            round_dir,
            client_steps=[
                {"client_id": 0, "ok": True},
                {"client_id": 1, "ok": False},
            ],
            keep_failed=True,
        )
        assert int(cleanup.get("removed_files", 0)) >= 1
        assert int(cleanup.get("removed_dirs", 0)) >= 1
        assert int(cleanup.get("kept_failed_clients", 0)) == 1
        assert not (round_dir / "client_0_out").exists()
        assert (round_dir / "client_1_out").exists()

        fed_dir = root / "fed"
        (fed_dir / "round_1").mkdir(parents=True, exist_ok=True)
        (fed_dir / "round_2").mkdir(parents=True, exist_ok=True)
        (fed_dir / "round_3").mkdir(parents=True, exist_ok=True)
        prune = _prune_old_federated_round_dirs(fed_dir, keep_latest=1, current_round=3)
        removed = sorted(int(x) for x in prune.get("removed_rounds", []))
        assert removed == [1, 2]
        assert (fed_dir / "round_3").exists()

    sample_round = {
        "round": 2,
        "selected_clients": [0, 1],
        "active_workers": ["w1"],
        "cooled_workers": [],
        "parallel_clients": 1,
        "async_mode": False,
        "round_timeout_seconds": 0.0,
        "retry_stats": {"clients_retried": 1},
        "governance": {"passed": True, "round_train_loss": 0.2},
        "degraded": False,
        "degrade_reason": "",
        "global_adapter": "ga",
        "global_signature": "sig",
        "stragglers": [],
        "aggregate": {"method": "avg", "adapter_path": "ga", "signature": "sig", "total_weight": 2.0, "num_clients": 2},
        "client_steps": [
            {
                "client_id": 0,
                "worker": "w1",
                "ok": True,
                "rows": 10,
                "retries_used": 0,
                "step": {"timeout": False, "returncode": 0, "elapsed_s": 1.0},
                "health": {"errors": []},
            }
        ],
    }
    compact_round = _compact_federated_round_report_for_state(sample_round, compact=True)
    assert int(compact_round.get("round", 0)) == 2
    assert isinstance(compact_round.get("client_steps", []), list)
    assert int(compact_round.get("client_steps", [{}])[0].get("client_id", -1)) == 0
    full_round = _compact_federated_round_report_for_state(sample_round, compact=False)
    assert isinstance(full_round.get("client_steps", []), list)
    state_rows = _prepare_federated_state_round_reports([{"round": 1}, sample_round], compact=True, keep_latest=1)
    assert len(state_rows) == 1
    assert int(state_rows[0].get("round", 0)) == 2

    client_steps = [
        {"ok": True, "rows": 10, "worker": "w1", "summary": {"train_loss": 0.30}},
        {"ok": True, "rows": 12, "worker": "w2", "summary": {"train_loss": 0.35}},
        {"ok": False, "rows": 8, "worker": "w2", "summary": {}},
    ]
    gov = _evaluate_federated_governance(
        client_steps=client_steps,
        expected_clients=3,
        selected_rows=rows[:20],
        global_rows=rows,
        best_loss=0.29,
        regress_tol=0.02,
        max_drift_l1=0.9,
        max_fail_rate=0.7,
    )
    assert isinstance(gov.get("passed", None), bool)
    assert float(gov.get("fail_rate", 0.0)) >= 0.0
    assert float(gov.get("fairness_rows_jain", 0.0)) > 0.0

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "workers.json"
        cfg.write_text(
            json.dumps(
                {
                    "workers": [
                        {
                            "name": "n1",
                            "cmd_prefix": [],
                            "python_bin": sys.executable,
                            "max_concurrency": 2,
                            "resource_weight": 2.5,
                            "resource_probe_cmd": [sys.executable, "-V"],
                        },
                        {"name": "n2", "enabled": False},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        args = SimpleNamespace(fed_workers_json=cfg.as_posix())
        ws = _load_federated_workers(args)
        assert len(ws) == 1
        assert str(ws[0].get("name", "")) == "n1"
        assert int(ws[0].get("max_concurrency", 0)) == 2
        assert abs(float(ws[0].get("resource_weight", 0.0)) - 2.5) < 1e-9
        assert isinstance(ws[0].get("resource_probe_cmd", []), list)

    child_args = SimpleNamespace(
        model_path="base_model",
        eval_data="eval.jsonl",
        dataset_manifest="manifest.json",
        fed_client_epochs=0.5,
        learning_rate=2e-4,
        batch_size=1,
        grad_accum=2,
        max_length=128,
        logging_steps=10,
        save_steps=20,
        warmup_ratio=0.03,
        weight_decay=0.01,
        dtype="fp32",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules="q_proj,v_proj",
        bnb_compute_dtype="fp16",
        curriculum_easy_frac=0.4,
        curriculum_stage1_ratio=0.3,
        replay_ratio=0.1,
        replay_max_samples=20,
        artifact_registry_path="artifacts/checkpoints/adapter_artifacts.json",
        replay_data="",
        local_only=True,
        load_in_4bit=False,
        gradient_checkpointing=False,
        group_by_length=False,
        compact_output=True,
        curriculum=False,
    )
    child_cmd = _build_federated_child_cmd(
        child_args,
        train_data=Path("client_train.jsonl"),
        output_dir=Path("client_out"),
        seed=11,
        base_adapter="base_adapter",
        worker=None,
    )
    assert "--no-register-artifact" in child_cmd
    assert "--artifact-registry-path" in child_cmd and "artifacts/checkpoints/adapter_artifacts.json" in " ".join(child_cmd)


def _test_auto_iterate_federated_train_cmd() -> None:
    args = SimpleNamespace(
        python_bin=sys.executable,
        model_path="m",
        train_data="train.jsonl",
        eval_data="eval.jsonl",
        dataset_manifest="manifest.json",
        epochs=1.0,
        batch_size=1,
        grad_accum=2,
        max_length=128,
        logging_steps=10,
        save_steps=20,
        warmup_ratio=0.03,
        seed=7,
        dtype="fp32",
        target_modules="q_proj,v_proj",
        build_min_quality=0.4,
        build_max_samples=100,
        build_eval_ratio=0.1,
        hard_min_ratio=0.3,
        hard_max_ratio=0.6,
        max_per_prompt=3,
        max_per_response=3,
        allow_cross_split_prompt=True,
        disable_diversity_penalty=True,
        diversity_similarity_threshold=0.90,
        diversity_penalty_strength=0.30,
        diversity_reference_cap=7,
        bucket_classifier_min_conf=0.6,
        bucket_classifier_min_docs=4,
        bucket_classifier_alpha=1.0,
        bnb_compute_dtype="fp16",
        curriculum_easy_frac=0.4,
        curriculum_stage1_ratio=0.3,
        replay_ratio=0.1,
        replay_max_samples=50,
        no_auto_build=True,
        keep_templatey=False,
        no_bucket_classifier=False,
        bucket_classifier_force_override=False,
        preset_7840hs=False,
        load_in_4bit=False,
        gradient_checkpointing=False,
        group_by_length=False,
        compact_train_output=True,
        train_no_register_artifact=False,
        curriculum=False,
        base_adapter="base_adapter_path",
        federated_train=True,
        fed_num_clients=3,
        fed_rounds=2,
        fed_client_frac=0.66,
        fed_min_active_clients=1,
        fed_min_client_samples=4,
        fed_split_method="bucket",
        fed_client_epochs=0.5,
        fed_max_client_failures=1,
        fed_parallel_clients=2,
        fed_client_timeout_seconds=123.0,
        fed_resource_aware=True,
        fed_reprobe_each_round=True,
        fed_resource_pressure_alpha=1.2,
        fed_sched_use_perf=True,
        fed_sched_perf_alpha=0.35,
        fed_sched_perf_fail_weight=0.8,
        fed_sched_perf_latency_weight=0.2,
        fed_sched_resource_weight=1.0,
        fed_sched_perf_weight=0.7,
        fed_client_retries=2,
        fed_retry_backoff_seconds=1.5,
        fed_retry_switch_worker=True,
        fed_async=True,
        fed_round_timeout_seconds=222.0,
        fed_workers_json="workers.json",
        fed_worker_probe_timeout_seconds=9.0,
        fed_worker_require_healthy=True,
        fed_worker_fail_threshold=3,
        fed_worker_fail_cooldown_rounds=2,
        fed_worker_cooldown_strict=True,
        fed_aggregation="median",
        fed_trim_ratio=0.2,
        fed_client_weight_strategy="multi_signal",
        fed_client_weight_loss_eps=1e-4,
        fed_client_weight_loss_clip=9.0,
        fed_client_weight_sample_power=1.0,
        fed_client_weight_loss_power=0.8,
        fed_client_weight_eval_loss_power=0.6,
        fed_client_weight_timeout_penalty=0.25,
        fed_client_weight_retry_penalty=0.10,
        fed_client_weight_fail_attempt_penalty=0.05,
        fed_client_weight_min=1e-5,
        fed_client_weight_max_ratio=8.0,
        fed_client_weight_normalize_mean=True,
        fed_client_weight_max_share=0.55,
        fed_client_bucket_balance=True,
        fed_client_bucket_balance_strength=0.9,
        fed_client_bucket_balance_eps=1e-6,
        fed_client_bucket_target="selected",
        fed_regress_loss_tol=0.03,
        fed_max_drift_l1=0.25,
        fed_max_fail_rate=0.4,
        fed_auto_degrade=True,
        fed_prune_client_artifacts=True,
        fed_prune_keep_failed_clients=True,
        fed_prune_rounds_keep=2,
        fed_state_path="fed_state.json",
        fed_resume=True,
        fed_resume_strict=True,
        local_only=True,
        artifact_registry_path="artifacts/checkpoints/adapter_artifacts.json",
    )
    cmd = _build_auto_train_cmd(
        args,
        train_output_dir=Path("out"),
        seed=11,
        learning_rate=2e-4,
        lora_dropout=0.05,
        lora_r=8,
        lora_alpha=16,
        weight_decay=0.01,
        replay_data="replay.jsonl",
    )
    text = " ".join(cmd)
    assert "--federated" in cmd
    assert "--fed-num-clients" in cmd and "3" in text
    assert "--fed-rounds" in cmd and "2" in text
    assert "--fed-state-path" in cmd and "fed_state.json" in text
    assert "--fed-resume" in cmd
    assert "--fed-parallel-clients" in cmd and "2" in text
    assert "--fed-client-timeout-seconds" in cmd and "123.0" in text
    assert "--allow-cross-split-prompt" in cmd
    assert "--disable-diversity-penalty" in cmd
    assert "--diversity-similarity-threshold" in cmd and "0.9" in text
    assert "--diversity-penalty-strength" in cmd and "0.3" in text
    assert "--diversity-reference-cap" in cmd and "7" in text
    assert "--fed-resource-aware" in cmd
    assert "--fed-reprobe-each-round" in cmd
    assert "--fed-resource-pressure-alpha" in cmd and "1.2" in text
    assert "--fed-sched-use-perf" in cmd
    assert "--fed-sched-perf-alpha" in cmd and "0.35" in text
    assert "--fed-sched-perf-fail-weight" in cmd and "0.8" in text
    assert "--fed-sched-perf-latency-weight" in cmd and "0.2" in text
    assert "--fed-sched-resource-weight" in cmd and "1.0" in text
    assert "--fed-sched-perf-weight" in cmd and "0.7" in text
    assert "--fed-client-retries" in cmd and "2" in text
    assert "--fed-retry-backoff-seconds" in cmd and "1.5" in text
    assert "--fed-retry-switch-worker" in cmd
    assert "--fed-async" in cmd
    assert "--fed-round-timeout-seconds" in cmd and "222.0" in text
    assert "--fed-workers-json" in cmd and "workers.json" in text
    assert "--fed-worker-probe-timeout-seconds" in cmd and "9.0" in text
    assert "--fed-worker-require-healthy" in cmd
    assert "--fed-worker-fail-threshold" in cmd and "3" in text
    assert "--fed-worker-fail-cooldown-rounds" in cmd and "2" in text
    assert "--fed-worker-cooldown-strict" in cmd
    assert "--fed-aggregation" in cmd and "median" in text
    assert "--fed-trim-ratio" in cmd and "0.2" in text
    assert "--fed-client-weight-strategy" in cmd and "multi_signal" in text
    assert "--fed-client-weight-loss-eps" in cmd and "0.0001" in text
    assert "--fed-client-weight-loss-clip" in cmd and "9.0" in text
    assert "--fed-client-weight-sample-power" in cmd and "1.0" in text
    assert "--fed-client-weight-loss-power" in cmd and "0.8" in text
    assert "--fed-client-weight-eval-loss-power" in cmd and "0.6" in text
    assert "--fed-client-weight-timeout-penalty" in cmd and "0.25" in text
    assert "--fed-client-weight-retry-penalty" in cmd and "0.1" in text
    assert "--fed-client-weight-fail-attempt-penalty" in cmd and "0.05" in text
    assert "--fed-client-weight-min" in cmd and "1e-05" in text
    assert "--fed-client-weight-max-ratio" in cmd and "8.0" in text
    assert "--fed-client-weight-normalize-mean" in cmd
    assert "--fed-client-weight-max-share" in cmd and "0.55" in text
    assert "--fed-client-bucket-balance" in cmd
    assert "--fed-client-bucket-balance-strength" in cmd and "0.9" in text
    assert "--fed-client-bucket-balance-eps" in cmd and "1e-06" in text
    assert "--fed-client-bucket-target" in cmd and "selected" in text
    assert "--fed-regress-loss-tol" in cmd and "0.03" in text
    assert "--fed-max-drift-l1" in cmd and "0.25" in text
    assert "--fed-max-fail-rate" in cmd and "0.4" in text
    assert "--fed-auto-degrade" in cmd
    assert "--fed-prune-client-artifacts" in cmd
    assert "--fed-prune-keep-failed-clients" in cmd
    assert "--fed-prune-rounds-keep" in cmd and "2" in text
    assert "--fed-resume-strict" in cmd
    assert "--base-adapter" in cmd and "base_adapter_path" in text
    assert "--compact-output" in cmd
    assert "--artifact-registry-path" in cmd and "artifacts/checkpoints/adapter_artifacts.json" in text


def _test_adaptive_score_weights() -> None:
    base = _load_loop_score_weights("", "balanced")
    manifest = {
        "by_bucket": {"safety": 12, "reasoning": 2, "qa": 1},
        "by_error_tag": {"safety": 6, "reasoning": 1},
    }
    out = _build_adaptive_score_weights(base, manifest, shift=0.20, min_global_sum=0.45)
    assert out
    assert float(out.get("safety_pass", 0.0)) > float(base.get("safety_pass", 0.0))
    assert float(out.get("global_pass", 0.0)) < float(base.get("global_pass", 0.0))
    gsum = float(out.get("global_pass", 0.0)) + float(out.get("global_quality", 0.0)) + float(out.get("global_semantic", 0.0))
    assert gsum >= 0.45 - 1e-6
    total = sum(float(v) for v in out.values())
    assert abs(total - 1.0) < 1e-5


def _test_adaptive_round_overrides() -> None:
    cfg = SimpleNamespace(
        adaptive_replay_ratio_min=0.0,
        adaptive_replay_ratio_max=0.35,
        adaptive_replay_ratio_step=0.05,
        adaptive_num_candidates_step=1,
        adaptive_num_candidates_max=6,
        adaptive_base_min_pass_rate=0.45,
        adaptive_min_pass_rate_step=0.02,
        adaptive_min_pass_rate_max=0.70,
        adaptive_base_min_quality=0.58,
        adaptive_min_quality_step=0.02,
        adaptive_min_quality_max=0.72,
    )
    cur = [
        "--num-candidates",
        "2",
        "--replay-ratio",
        "0.10",
        "--min-pass-rate",
        "0.45",
        "--min-quality",
        "0.58",
    ]
    safety_manifest = {"by_error_tag": {"safety": 8, "reasoning": 1}}
    out = _build_round_adaptive_overrides(cur, safety_manifest, promoted=False, fail_streak=2, cfg=cfg)
    ov = dict(out.get("overrides", {}))
    assert str(out.get("top_error_tag", "")) == "safety"
    assert ov.get("--candidate-score-profile") == "safety"
    assert float(ov.get("--replay-ratio", "0.0")) > 0.10
    assert int(ov.get("--num-candidates", "1")) == 3
    assert float(ov.get("--min-pass-rate", "0.0")) > 0.45

    factual_manifest = {"by_error_tag": {"factual": 6}}
    out2 = _build_round_adaptive_overrides(cur, factual_manifest, promoted=False, fail_streak=1, cfg=cfg)
    ov2 = dict(out2.get("overrides", {}))
    assert ov2.get("--candidate-score-profile") == "factual"
    assert float(ov2.get("--min-quality", "0.0")) > 0.58

    out3 = _build_round_adaptive_overrides(cur, {}, promoted=True, fail_streak=0, cfg=cfg)
    ov3 = dict(out3.get("overrides", {}))
    assert ov3.get("--candidate-score-profile") == "balanced"
    assert float(ov3.get("--replay-ratio", "1.0")) <= 0.10


def _test_adaptive_train_policy() -> None:
    cfg = SimpleNamespace(
        adaptive_policy_fail_threshold=2,
        adaptive_policy_replay_max_samples_base=0,
        adaptive_policy_replay_max_samples_step=800,
        adaptive_policy_replay_max_samples_max=12000,
        adaptive_policy_eval_max_samples_base=64,
        adaptive_policy_eval_max_samples_step=16,
        adaptive_policy_eval_max_samples_max=192,
        adaptive_policy_safety_min_pass_base=0.0,
        adaptive_policy_safety_min_pass_step=0.02,
        adaptive_policy_safety_min_pass_max=0.35,
        adaptive_policy_curriculum_easy_min=0.25,
        adaptive_policy_curriculum_stage1_min=0.20,
        adaptive_policy_curriculum_frac_step=0.04,
        adaptive_policy_curriculum_stage1_step=0.04,
    )
    cur = [
        "--replay-max-samples",
        "1200",
        "--eval-max-samples",
        "80",
        "--safety-min-pass-rate",
        "0.02",
        "--curriculum-easy-frac",
        "0.40",
        "--curriculum-stage1-ratio",
        "0.30",
    ]
    reasoning_manifest = {"by_error_tag": {"reasoning": 7, "safety": 1}}
    out = _build_train_policy_overrides(cur, reasoning_manifest, promoted=False, fail_streak=2, cfg=cfg)
    vo = dict(out.get("value_overrides", {}))
    bo = list(out.get("bool_overrides", []))
    assert str(out.get("top_error_tag", "")) == "reasoning"
    assert int(vo.get("--replay-max-samples", "0")) > 1200
    assert int(vo.get("--eval-max-samples", "0")) > 80
    assert float(vo.get("--curriculum-easy-frac", "1.0")) < 0.40
    assert "--curriculum" in bo
    assert "--search-from-history" in bo

    safety_manifest = {"by_error_tag": {"safety": 6}}
    out2 = _build_train_policy_overrides(cur, safety_manifest, promoted=False, fail_streak=1, cfg=cfg)
    vo2 = dict(out2.get("value_overrides", {}))
    assert float(vo2.get("--safety-min-pass-rate", "0.0")) > 0.02

    out3 = _build_train_policy_overrides(cur, {}, promoted=True, fail_streak=0, cfg=cfg)
    vo3 = dict(out3.get("value_overrides", {}))
    assert int(vo3.get("--replay-max-samples", "99999")) <= 1200
    assert int(vo3.get("--eval-max-samples", "99999")) <= 80


def _test_adaptive_dataset_penalty() -> None:
    cfg = SimpleNamespace(
        adaptive_dataset_diversity_penalty_base=0.15,
        adaptive_dataset_near_duplicate_penalty_base=0.20,
        adaptive_dataset_cross_prompt_penalty_base=0.30,
        adaptive_dataset_cross_pair_penalty_base=0.35,
        adaptive_dataset_diversity_penalty_step=0.03,
        adaptive_dataset_near_duplicate_penalty_step=0.03,
        adaptive_dataset_cross_prompt_penalty_step=0.04,
        adaptive_dataset_cross_pair_penalty_step=0.04,
        adaptive_dataset_penalty_min=0.0,
        adaptive_dataset_penalty_max=0.95,
        adaptive_dataset_fail_threshold=2,
        adaptive_dataset_penalty_fail_boost=0.50,
        adaptive_dataset_penalty_relax_factor=0.50,
    )
    cur = [
        "--candidate-dataset-diversity-penalty-weight",
        "0.15",
        "--candidate-dataset-near-duplicate-penalty-weight",
        "0.20",
        "--candidate-dataset-cross-prompt-penalty-weight",
        "0.30",
        "--candidate-dataset-cross-pair-penalty-weight",
        "0.35",
    ]
    m_reason = {"by_error_tag": {"semantic_drift": 8}}
    out = _build_dataset_penalty_overrides(cur, m_reason, promoted=False, fail_streak=2, cfg=cfg)
    vo = dict(out.get("value_overrides", {}))
    assert str(out.get("top_error_tag", "")) == "semantic_drift"
    assert float(vo.get("--candidate-dataset-diversity-penalty-weight", "0")) > 0.15
    assert float(vo.get("--candidate-dataset-near-duplicate-penalty-weight", "0")) > 0.20
    assert float(vo.get("--candidate-dataset-cross-prompt-penalty-weight", "0")) > 0.30
    assert float(vo.get("--candidate-dataset-cross-pair-penalty-weight", "0")) >= 0.35

    m_fact = {"by_error_tag": {"hallucination": 6}}
    out2 = _build_dataset_penalty_overrides(cur, m_fact, promoted=False, fail_streak=1, cfg=cfg)
    vo2 = dict(out2.get("value_overrides", {}))
    assert str(out2.get("top_error_tag", "")) == "hallucination"
    assert float(vo2.get("--candidate-dataset-cross-prompt-penalty-weight", "0")) > 0.30
    assert float(vo2.get("--candidate-dataset-cross-pair-penalty-weight", "0")) > 0.35

    cur2 = [
        "--candidate-dataset-diversity-penalty-weight",
        "0.22",
        "--candidate-dataset-near-duplicate-penalty-weight",
        "0.27",
        "--candidate-dataset-cross-prompt-penalty-weight",
        "0.42",
        "--candidate-dataset-cross-pair-penalty-weight",
        "0.46",
    ]
    out3 = _build_dataset_penalty_overrides(cur2, m_fact, promoted=True, fail_streak=0, cfg=cfg)
    vo3 = dict(out3.get("value_overrides", {}))
    assert float(vo3.get("--candidate-dataset-diversity-penalty-weight", "1")) < 0.22
    assert float(vo3.get("--candidate-dataset-cross-prompt-penalty-weight", "1")) < 0.42
    assert float(vo3.get("--candidate-dataset-cross-pair-penalty-weight", "1")) < 0.46


def _test_adaptive_candidate_space() -> None:
    cfg = SimpleNamespace(
        adaptive_candidate_fail_threshold=2,
        adaptive_candidate_min_r=4,
        adaptive_candidate_max_r=32,
        adaptive_candidate_lr_min=1e-5,
        adaptive_candidate_lr_max=0.002,
        adaptive_candidate_search_topk_step=2,
        adaptive_candidate_search_topk_max=10,
    )
    cur = [
        "--learning-rate",
        "0.0002",
        "--lora-dropout",
        "0.05",
        "--lora-r",
        "8",
        "--lora-alpha",
        "16",
        "--weight-decay",
        "0.01",
        "--search-history-topk",
        "4",
    ]
    m_reason = {"by_error_tag": {"reasoning": 7}}
    out = _build_candidate_space_overrides(cur, m_reason, promoted=False, fail_streak=2, cfg=cfg)
    vo = dict(out.get("value_overrides", {}))
    bo = list(out.get("bool_overrides", []))
    assert str(out.get("top_error_tag", "")) == "reasoning"
    assert "--search-from-history" in bo
    assert "12" in str(vo.get("--candidate-lora-rs", ""))
    assert int(vo.get("--search-history-topk", "0")) >= 5

    m_safety = {"by_error_tag": {"safety": 6}}
    out2 = _build_candidate_space_overrides(cur, m_safety, promoted=False, fail_streak=1, cfg=cfg)
    vo2 = dict(out2.get("value_overrides", {}))
    assert str(out2.get("top_error_tag", "")) == "safety"
    drops = [float(x) for x in str(vo2.get("--candidate-dropouts", "")).split(",") if str(x).strip()]
    assert drops and max(drops) >= 0.08

    out3 = _build_candidate_space_overrides(cur, {}, promoted=True, fail_streak=0, cfg=cfg)
    vo3 = dict(out3.get("value_overrides", {}))
    lrs3 = [float(x) for x in str(vo3.get("--candidate-lrs", "")).split(",") if str(x).strip()]
    assert lrs3 and min(lrs3) >= 1e-5


def _test_adaptive_target_modules() -> None:
    cfg = SimpleNamespace(
        adaptive_target_fail_threshold=2,
        adaptive_target_default_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        adaptive_target_safety_modules="q_proj,k_proj,v_proj,o_proj",
        adaptive_target_reasoning_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        adaptive_target_factual_modules="q_proj,k_proj,v_proj,o_proj,up_proj",
    )
    cur = [
        "--target-modules",
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    ]

    out_s = _build_target_module_overrides(cur, {"by_error_tag": {"safety": 5}}, promoted=False, fail_streak=1, cfg=cfg)
    ms = set(str(out_s.get("value_overrides", {}).get("--target-modules", "")).split(","))
    assert "q_proj" in ms and "o_proj" in ms
    assert "gate_proj" not in ms

    out_r = _build_target_module_overrides(cur, {"by_error_tag": {"reasoning": 6}}, promoted=False, fail_streak=1, cfg=cfg)
    mr = set(str(out_r.get("value_overrides", {}).get("--target-modules", "")).split(","))
    assert "gate_proj" in mr and "down_proj" in mr

    out_f = _build_target_module_overrides(cur, {"by_error_tag": {"factual": 4}}, promoted=False, fail_streak=3, cfg=cfg)
    mf = set(str(out_f.get("value_overrides", {}).get("--target-modules", "")).split(","))
    assert "up_proj" in mf and "gate_proj" in mf

    out_p = _build_target_module_overrides(cur, {"by_error_tag": {"safety": 9}}, promoted=True, fail_streak=0, cfg=cfg)
    mp = set(str(out_p.get("value_overrides", {}).get("--target-modules", "")).split(","))
    assert "gate_proj" in mp and "down_proj" in mp


def _test_adaptive_budget_overrides() -> None:
    cfg = SimpleNamespace(
        adaptive_budget_target_round_seconds=600.0,
        adaptive_budget_shrink_threshold=1.15,
        adaptive_budget_grow_threshold=0.70,
        adaptive_budget_min_candidates=1,
        adaptive_budget_max_candidates=8,
        adaptive_budget_candidate_step=1,
        adaptive_budget_eval_max_base=64,
        adaptive_budget_eval_min=32,
        adaptive_budget_eval_max=256,
        adaptive_budget_eval_step=16,
        adaptive_budget_max_new_tokens_base=96,
        adaptive_budget_token_min=64,
        adaptive_budget_token_max=192,
        adaptive_budget_token_step=16,
        adaptive_budget_fail_shrink_threshold=2,
    )
    cur = [
        "--num-candidates",
        "4",
        "--eval-max-samples",
        "128",
        "--max-new-tokens",
        "128",
    ]
    shrink = _build_budget_overrides(cur, round_elapsed_s=1100.0, promoted=False, fail_streak=1, cfg=cfg)
    svo = dict(shrink.get("value_overrides", {}))
    assert str(shrink.get("mode", "")) == "shrink"
    assert int(svo.get("--num-candidates", "99")) < 4
    assert int(svo.get("--eval-max-samples", "999")) <= 128
    assert int(svo.get("--max-new-tokens", "999")) <= 128

    grow = _build_budget_overrides(cur, round_elapsed_s=260.0, promoted=False, fail_streak=1, cfg=cfg)
    gvo = dict(grow.get("value_overrides", {}))
    assert str(grow.get("mode", "")) == "grow"
    assert int(gvo.get("--num-candidates", "0")) >= 5
    assert int(gvo.get("--eval-max-samples", "0")) >= 128

    hold = _build_budget_overrides(cur, round_elapsed_s=560.0, promoted=True, fail_streak=0, cfg=cfg)
    hvo = dict(hold.get("value_overrides", {}))
    assert str(hold.get("mode", "")) == "hold"
    assert int(hvo.get("--num-candidates", "0")) == 4


def _test_adaptive_bucket_thresholds() -> None:
    cfg = SimpleNamespace(
        adaptive_bucket_base_min_quality=0.58,
        adaptive_bucket_base_min_pass_rate=0.45,
        adaptive_bucket_base_min_semantic=0.0,
        adaptive_bucket_pass_step=0.015,
        adaptive_bucket_quality_step=0.015,
        adaptive_bucket_semantic_step=0.01,
        adaptive_bucket_relax_step=0.008,
        adaptive_bucket_fail_threshold=2,
        adaptive_bucket_max_quality=0.85,
        adaptive_bucket_max_pass=0.85,
        adaptive_bucket_max_semantic=0.65,
    )
    cur = ["--min-quality", "0.58", "--min-pass-rate", "0.45", "--min-semantic", "0.00"]

    s = _build_bucket_threshold_payload(cur, {"by_error_tag": {"safety": 8}}, promoted=False, fail_streak=2, cfg=cfg)
    sp = dict(s.get("payload", {}))
    assert str(s.get("top_error_tag", "")) == "safety"
    assert float(sp.get("safety", {}).get("min_pass_rate", 0.0)) > 0.53

    r = _build_bucket_threshold_payload(cur, {"by_error_tag": {"reasoning": 7}}, promoted=False, fail_streak=1, cfg=cfg)
    rp = dict(r.get("payload", {}))
    assert float(rp.get("reasoning", {}).get("min_quality", 0.0)) > 0.60

    f = _build_bucket_threshold_payload(cur, {"by_error_tag": {"factual": 6}}, promoted=False, fail_streak=1, cfg=cfg)
    fp = dict(f.get("payload", {}))
    assert float(fp.get("qa", {}).get("min_quality", 0.0)) >= 0.58
    assert float(fp.get("qa", {}).get("min_pass_rate", 0.0)) >= 0.45

    p = _build_bucket_threshold_payload(cur, {"by_error_tag": {"safety": 9}}, promoted=True, fail_streak=0, cfg=cfg)
    pp = dict(p.get("payload", {}))
    assert float(pp.get("safety", {}).get("min_pass_rate", 0.0)) <= float(sp.get("safety", {}).get("min_pass_rate", 1.0))


def _test_adaptive_trend_memory() -> None:
    cfg = SimpleNamespace(
        trend_memory_trigger_streak=2,
        trend_memory_max_boost_rounds=4,
        trend_memory_num_candidates_step=1,
        trend_memory_replay_ratio_step=0.03,
        trend_memory_replay_max_samples_step=600,
        trend_memory_history_topk_step=1,
        trend_memory_max_new_tokens_step=8,
        trend_memory_min_pass_rate_step=0.01,
        trend_memory_min_quality_step=0.01,
        adaptive_num_candidates_max=6,
        adaptive_budget_max_candidates=8,
        adaptive_replay_ratio_min=0.0,
        adaptive_replay_ratio_max=0.35,
        adaptive_policy_replay_max_samples_base=0,
        adaptive_policy_replay_max_samples_max=12000,
        adaptive_candidate_search_topk_max=10,
        adaptive_budget_max_new_tokens_base=96,
        adaptive_budget_token_max=192,
        adaptive_base_min_pass_rate=0.45,
        adaptive_min_pass_rate_max=0.70,
        adaptive_base_min_quality=0.58,
        adaptive_min_quality_max=0.72,
    )
    cur = [
        "--num-candidates",
        "3",
        "--replay-ratio",
        "0.10",
        "--replay-max-samples",
        "1200",
        "--search-history-topk",
        "4",
        "--max-new-tokens",
        "96",
        "--min-pass-rate",
        "0.45",
        "--min-quality",
        "0.58",
    ]
    base_overrides = {"--num-candidates": "4"}

    off = _build_trend_memory_overrides(
        cur,
        base_overrides,
        top_error_tag="reasoning",
        top_error_streak=1,
        cfg=cfg,
    )
    assert not bool(off.get("active", True))

    on = _build_trend_memory_overrides(
        cur,
        base_overrides,
        top_error_tag="reasoning",
        top_error_streak=4,
        cfg=cfg,
    )
    vo = dict(on.get("value_overrides", {}))
    bo = list(on.get("bool_overrides", []))
    assert bool(on.get("active", False))
    assert int(on.get("boost", 0)) >= 2
    assert int(vo.get("--num-candidates", "0")) >= 5
    assert int(vo.get("--search-history-topk", "0")) >= 5
    assert "--search-from-history" in bo
    assert "--curriculum" in bo

    safe = _build_trend_memory_overrides(
        cur,
        {},
        top_error_tag="safety",
        top_error_streak=3,
        cfg=cfg,
    )
    svo = dict(safe.get("value_overrides", {}))
    assert float(svo.get("--min-pass-rate", "0.0")) > 0.45


def _test_adaptive_bandit() -> None:
    profiles = _resolve_bandit_profiles("reasoning,safety,foo,reasoning")
    assert profiles == ["reasoning", "safety"]
    fallback = _resolve_bandit_profiles("foo,bar")
    assert fallback == ["balanced", "safety", "reasoning", "factual"]

    eps0 = _bandit_epsilon_for_round(base=0.20, decay=0.9, min_eps=0.05, pulls=0)
    eps8 = _bandit_epsilon_for_round(base=0.20, decay=0.9, min_eps=0.05, pulls=8)
    assert abs(eps0 - 0.20) < 1e-9
    assert 0.05 <= eps8 < 0.20

    rng = random.Random(3)
    stats_cold = {
        "reasoning": {"pulls": 0, "reward_sum": 0.0, "reward_mean": 0.0},
        "safety": {"pulls": 4, "reward_sum": 1.2, "reward_mean": 0.3},
    }
    cold = _choose_bandit_profile(stats_cold, profiles, epsilon=0.5, ucb_c=0.35, rng=rng)
    assert str(cold.get("mode", "")) == "cold_start"
    assert str(cold.get("profile", "")) == "reasoning"

    stats_live = {
        "reasoning": {"pulls": 4, "reward_sum": 1.4, "reward_mean": 0.35},
        "safety": {"pulls": 4, "reward_sum": 0.5, "reward_mean": 0.125},
    }
    exploit = _choose_bandit_profile(stats_live, profiles, epsilon=0.0, ucb_c=0.0, rng=rng)
    assert str(exploit.get("mode", "")) == "exploit"
    assert str(exploit.get("profile", "")) == "reasoning"

    cfg = SimpleNamespace(
        bandit_reward_score_delta_weight=1.0,
        bandit_reward_promote_bonus=0.06,
        bandit_reward_improve_bonus=0.03,
        bandit_reward_fail_penalty=0.04,
    )
    good = _build_bandit_reward(score=0.72, prev_score=0.60, promote=True, ok=True, improved=True, cfg=cfg)
    bad = _build_bandit_reward(score=0.54, prev_score=0.60, promote=False, ok=False, improved=False, cfg=cfg)
    assert float(good.get("reward", 0.0)) > 0.0
    assert float(bad.get("reward", 0.0)) < 0.0
    assert float(good.get("score_delta", 0.0)) > 0.0
    assert float(bad.get("fail_penalty", 0.0)) > 0.0


def _test_adaptive_bandit_contextual() -> None:
    ctx = _resolve_bandit_contexts("reasoning,safety,foo,reasoning")
    assert ctx == ["reasoning", "safety"]
    fallback = _resolve_bandit_contexts("foo,bar")
    assert fallback == ["neutral", "safety", "reasoning", "factual"]

    assert _normalize_bandit_context("semantic_drift") == "reasoning"
    assert _normalize_bandit_context("hallucination") == "factual"
    assert _normalize_bandit_context("unknown") == "neutral"
    assert _select_bandit_context("semantic_drift", ["neutral", "reasoning"]) == "reasoning"
    assert _select_bandit_context("safety", ["neutral", "reasoning"]) == "neutral"

    profiles = ["safety", "reasoning"]
    global_stats = {
        "safety": {"pulls": 8, "reward_sum": 0.8, "reward_mean": 0.10},
        "reasoning": {"pulls": 8, "reward_sum": 4.0, "reward_mean": 0.50},
    }
    local_stats = {
        "safety": {"pulls": 3, "reward_sum": 2.4, "reward_mean": 0.80},
        "reasoning": {"pulls": 1, "reward_sum": 0.1, "reward_mean": 0.10},
    }
    mixed = _blend_bandit_stats(global_stats, local_stats, profiles, prior_pulls=2)
    rng = random.Random(7)
    choose = _choose_bandit_profile(mixed, profiles, epsilon=0.0, ucb_c=0.0, rng=rng)
    assert str(choose.get("profile", "")) == "safety"


def _test_adaptive_bandit_persistence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state_path = root / "bandit_state.json"
        profiles = ["balanced", "reasoning"]
        contexts = ["neutral", "reasoning"]

        cold = _load_bandit_state(state_path, profiles, contexts=contexts, reset=False)
        assert int(cold.get("total_pulls", -1)) == 0
        cold_stats = dict(cold.get("stats", {}))
        assert set(cold_stats.keys()) == set(profiles)
        cold_ctx = dict(cold.get("context_states", {}))
        assert set(cold_ctx.keys()) == set(contexts)
        assert not bool(cold.get("has_prev_score", True))

        mutated = dict(cold)
        mutated["stats"] = dict(cold_stats)
        mutated["stats"]["balanced"] = {
            "pulls": 3,
            "reward_sum": 0.9,
            "last_round": 3,
            "last_reward": 0.2,
        }
        mutated["total_pulls"] = 3
        mutated["prev_score"] = 0.66
        mutated["has_prev_score"] = True
        mutated["context_states"] = dict(cold_ctx)
        mutated["context_states"]["reasoning"] = {
            "total_pulls": 2,
            "stats": {
                "balanced": {"pulls": 1, "reward_sum": 0.1, "reward_mean": 0.1},
                "reasoning": {"pulls": 1, "reward_sum": 0.4, "reward_mean": 0.4},
            },
        }
        saved = _save_bandit_state(state_path, mutated, profiles, contexts=contexts)
        assert state_path.exists()
        assert int(saved.get("total_pulls", 0)) == 3

        hot = _load_bandit_state(state_path, profiles, contexts=contexts, reset=False)
        hot_stats = dict(hot.get("stats", {}))
        b = dict(hot_stats.get("balanced", {}))
        hot_ctx = dict(hot.get("context_states", {}))
        hot_reason = dict(hot_ctx.get("reasoning", {}))
        hot_reason_stats = dict(hot_reason.get("stats", {}))
        assert int(hot.get("total_pulls", 0)) == 3
        assert int(b.get("pulls", 0)) == 3
        assert abs(float(b.get("reward_mean", 0.0)) - 0.3) < 1e-9
        assert int(hot_reason.get("total_pulls", 0)) == 2
        assert float(hot_reason_stats.get("reasoning", {}).get("reward_mean", 0.0)) > 0.0
        assert abs(float(hot.get("prev_score", 0.0)) - 0.66) < 1e-9
        assert bool(hot.get("has_prev_score", False))

        reset_state = _load_bandit_state(state_path, profiles, contexts=contexts, reset=True)
        assert int(reset_state.get("total_pulls", -1)) == 0
        reset_ctx = dict(reset_state.get("context_states", {}))
        assert set(reset_ctx.keys()) == set(contexts)
        assert not bool(reset_state.get("has_prev_score", True))


def _test_candidate_grid_and_score() -> None:
    args = SimpleNamespace(
        num_candidates=5,
        candidate_seeds="11,23",
        candidate_lrs="0.0002,0.00016",
        candidate_dropouts="0.05,0.08",
        candidate_lora_rs="8,12",
        candidate_lora_alphas="16,24",
        candidate_weight_decays="0.01,0.005",
        seed=7,
        learning_rate=2e-4,
        lora_dropout=0.05,
        lora_r=8,
        lora_alpha=16,
        weight_decay=0.01,
        search_from_history=False,
        search_history_glob="",
        search_history_topk=4,
        search_jitter_lr=0.2,
        search_jitter_dropout=0.03,
        search_jitter_weight_decay=0.5,
        search_jitter_lora_r=4,
        search_jitter_lora_alpha=8,
        candidate_select_metric="composite",
    )
    grid = _resolve_candidate_grid(args)
    assert len(grid) == 5
    for row in grid:
        assert int(row["seed"]) in {11, 23}
        assert int(row["lora_r"]) in {8, 12}
        assert int(row["lora_alpha"]) in {16, 24}
        assert float(row["learning_rate"]) in {0.0002, 0.00016}

    report = {
        "candidate_metrics": {
            "avg_quality": 0.64,
            "pass_rate": 0.58,
            "avg_semantic": 0.62,
            "by_bucket": {
                "safety": {"pass_rate": 0.90, "avg_quality": 0.71},
                "reasoning": {"pass_rate": 0.48, "avg_quality": 0.55},
                "qa": {"pass_rate": 0.59, "avg_quality": 0.61},
            },
        }
    }
    balanced_w = _load_score_weights("", "balanced")
    safety_w = _load_score_weights("", "safety")
    reasoning_w = _load_score_weights("", "reasoning")
    sb = _metric_score(report, "composite", score_weights=balanced_w)
    ss = _metric_score(report, "composite", score_weights=safety_w)
    sr = _metric_score(report, "composite", score_weights=reasoning_w)
    assert ss >= sb
    assert sr <= ss
    assert _metric_score(report, "quality", score_weights=balanced_w) == 0.64
    assert _metric_score(report, "pass_rate", score_weights=balanced_w) == 0.58
    assert _metric_score(report, "semantic", score_weights=balanced_w) == 0.62


def _test_history_guided_grid() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        r1 = root / "auto_iterate_a.json"
        r2 = root / "auto_iterate_b.json"
        report_a = {
            "candidate_runs": [
                {
                    "seed": 11,
                    "learning_rate": 0.0002,
                    "lora_dropout": 0.06,
                    "lora_r": 8,
                    "lora_alpha": 16,
                    "weight_decay": 0.01,
                    "select_score": 0.61,
                    "passed_gate": True,
                    "eval_report": {
                        "candidate_metrics": {"avg_quality": 0.63, "pass_rate": 0.57, "avg_semantic": 0.60}
                    },
                }
            ]
        }
        report_b = {
            "candidate_runs": [
                {
                    "seed": 23,
                    "learning_rate": 0.00016,
                    "lora_dropout": 0.05,
                    "lora_r": 12,
                    "lora_alpha": 24,
                    "weight_decay": 0.005,
                    "select_score": 0.71,
                    "passed_gate": True,
                    "eval_report": {
                        "candidate_metrics": {"avg_quality": 0.66, "pass_rate": 0.62, "avg_semantic": 0.64}
                    },
                }
            ]
        }
        r1.write_text(json.dumps(report_a, ensure_ascii=False), encoding="utf-8")
        r2.write_text(json.dumps(report_b, ensure_ascii=False), encoding="utf-8")

        args = SimpleNamespace(
            num_candidates=4,
            candidate_seeds="31,41",
            candidate_lrs="",
            candidate_dropouts="",
            candidate_lora_rs="",
            candidate_lora_alphas="",
            candidate_weight_decays="",
            seed=7,
            learning_rate=2e-4,
            lora_dropout=0.05,
            lora_r=8,
            lora_alpha=16,
            weight_decay=0.01,
            search_from_history=True,
            search_history_glob=(root / "auto_iterate_*.json").as_posix(),
            search_history_topk=2,
            search_jitter_lr=0.15,
            search_jitter_dropout=0.02,
            search_jitter_weight_decay=0.5,
            search_jitter_lora_r=4,
            search_jitter_lora_alpha=8,
            candidate_select_metric="composite",
        )
        w = _load_score_weights("", "balanced")
        grid = _resolve_candidate_grid(args, score_weights=w)
        assert len(grid) == 4
        lrs = {round(float(x["learning_rate"]), 8) for x in grid}
        assert 0.00016 in lrs or 0.0002 in lrs


def _test_auto_iterate_dataset_manifest_penalty() -> None:
    summary = {
        "dataset_manifest": {
            "train_rows": 20,
            "eval_rows": 10,
            "near_duplicate_samples": 9,
            "avg_diversity_penalty": 0.50,
            "cross_split_prompt_overlap_ratio": 0.40,
            "cross_split_pair_overlap_ratio": 0.30,
        }
    }
    args = SimpleNamespace(
        no_candidate_dataset_penalty=False,
        candidate_dataset_diversity_penalty_weight=0.15,
        candidate_dataset_near_duplicate_penalty_weight=0.20,
        candidate_dataset_cross_prompt_penalty_weight=0.30,
        candidate_dataset_cross_pair_penalty_weight=0.35,
    )
    penalty, stats = _dataset_manifest_penalty(summary, args)
    expected = 0.15 * 0.50 + 0.20 * (9.0 / 30.0) + 0.30 * 0.40 + 0.35 * 0.30
    assert bool(stats.get("available", False)) is True
    assert abs(float(penalty) - float(expected)) < 1e-9

    args.no_candidate_dataset_penalty = True
    penalty_off, _ = _dataset_manifest_penalty(summary, args)
    assert float(penalty_off) == 0.0


def _test_adapter_artifact_store() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        registry_path = root / "adapter_artifacts.json"
        active_registry_path = root / "active_adapter.json"

        adapter_a = root / "adapter_a"
        adapter_b = root / "adapter_b"
        adapter_a.mkdir(parents=True, exist_ok=True)
        adapter_b.mkdir(parents=True, exist_ok=True)
        (adapter_a / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_b / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_a / "adapter_model.bin").write_bytes(b"aaaaaaaa")
        (adapter_b / "adapter_model.bin").write_bytes(b"bbbbbbbbbbbb")

        a = register_adapter_artifact(
            registry_path=registry_path.as_posix(),
            adapter_path=adapter_a.as_posix(),
            base_model="mock-base",
            promoted=True,
            used=True,
            source="test",
        )
        b = register_adapter_artifact(
            registry_path=registry_path.as_posix(),
            adapter_path=adapter_b.as_posix(),
            base_model="mock-base",
            promoted=False,
            used=True,
            source="test",
        )
        assert str(a.get("id", "")).strip()
        assert str(b.get("id", "")).strip()

        # Active registry points to missing path; resolver should fallback to artifact registry.
        active_registry_path.write_text(
            json.dumps({"active_adapter": (root / "missing_adapter").as_posix()}, ensure_ascii=False),
            encoding="utf-8",
        )
        resolved = resolve_best_adapter_path(
            active_registry_path=active_registry_path.as_posix(),
            artifact_registry_path=registry_path.as_posix(),
        )
        assert resolved == adapter_a.resolve().as_posix()

        # Invalid adapter should not be selected even with strong metrics.
        invalid_adapter = root / "adapter_invalid"
        invalid_adapter.mkdir(parents=True, exist_ok=True)
        (invalid_adapter / "adapter_model.bin").write_bytes(b"cccc")
        register_adapter_artifact(
            registry_path=registry_path.as_posix(),
            adapter_path=invalid_adapter.as_posix(),
            metrics={"avg_quality": 0.99, "pass_rate": 0.99, "avg_semantic": 0.99, "sample_count": 32},
            promoted=False,
            used=True,
            source="test_invalid",
        )
        resolved_valid_only = resolve_best_adapter_path(
            active_registry_path=active_registry_path.as_posix(),
            artifact_registry_path=registry_path.as_posix(),
            require_valid=True,
        )
        assert resolved_valid_only != invalid_adapter.resolve().as_posix()

        shutil.rmtree(adapter_b.as_posix(), ignore_errors=True)
        prune = prune_adapter_artifacts(
            registry_path=registry_path.as_posix(),
            keep_latest=0,
            keep_promoted=1,
            max_total_bytes=0,
            remove_missing=True,
            delete_files=False,
        )
        assert int(prune.get("removed_missing", 0)) >= 1

        # Quality-aware fallback: when no active/promoted pointer is usable,
        # resolver should prefer higher-quality artifact over merely newer updates.
        quality_registry = root / "adapter_artifacts_quality.json"
        adapter_b.mkdir(parents=True, exist_ok=True)
        (adapter_b / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_b / "adapter_model.bin").write_bytes(b"bbbbbbbbbbbb")
        register_adapter_artifact(
            registry_path=quality_registry.as_posix(),
            adapter_path=adapter_a.as_posix(),
            metrics={"avg_quality": 0.55, "pass_rate": 0.50, "avg_semantic": 0.45, "sample_count": 48},
            promoted=False,
            used=True,
            source="test_quality",
        )
        register_adapter_artifact(
            registry_path=quality_registry.as_posix(),
            adapter_path=adapter_b.as_posix(),
            metrics={"avg_quality": 0.78, "pass_rate": 0.81, "avg_semantic": 0.74, "sample_count": 48},
            promoted=False,
            used=True,
            source="test_quality",
        )
        # Make adapter_a appear fresher; adapter_b should still win by quality.
        register_adapter_artifact(
            registry_path=quality_registry.as_posix(),
            adapter_path=adapter_a.as_posix(),
            metrics={"avg_quality": 0.55, "pass_rate": 0.50, "avg_semantic": 0.45, "sample_count": 48},
            promoted=False,
            used=True,
            source="test_quality_refresh",
        )
        resolved_q = resolve_best_adapter_path(
            active_registry_path=(root / "missing_active.json").as_posix(),
            artifact_registry_path=quality_registry.as_posix(),
        )
        assert resolved_q == adapter_b.resolve().as_posix()

        # Preferred base model should bias resolver; strict mode should enforce it.
        model_registry = root / "adapter_artifacts_model.json"
        register_adapter_artifact(
            registry_path=model_registry.as_posix(),
            adapter_path=adapter_a.as_posix(),
            base_model="model/a",
            metrics={"avg_quality": 0.40, "pass_rate": 0.42, "avg_semantic": 0.38, "sample_count": 32},
            promoted=False,
            used=True,
            source="test_model",
        )
        register_adapter_artifact(
            registry_path=model_registry.as_posix(),
            adapter_path=adapter_b.as_posix(),
            base_model="model/b",
            metrics={"avg_quality": 0.90, "pass_rate": 0.91, "avg_semantic": 0.86, "sample_count": 32},
            promoted=False,
            used=True,
            source="test_model",
        )
        resolved_pref = resolve_best_adapter_path(
            active_registry_path=(root / "missing_active2.json").as_posix(),
            artifact_registry_path=model_registry.as_posix(),
            preferred_base_model="model/a",
            strict_base_model=False,
        )
        assert resolved_pref == adapter_a.resolve().as_posix()
        resolved_strict = resolve_best_adapter_path(
            active_registry_path=(root / "missing_active2.json").as_posix(),
            artifact_registry_path=model_registry.as_posix(),
            preferred_base_model="model/a",
            strict_base_model=True,
        )
        assert resolved_strict == adapter_a.resolve().as_posix()

        # Prune with archive should produce zip and delete dropped artifact path.
        archive_registry = root / "adapter_artifacts_archive.json"
        archive_dir = root / "archives"
        old_adapter = root / "adapter_old"
        new_adapter = root / "adapter_new"
        old_adapter.mkdir(parents=True, exist_ok=True)
        new_adapter.mkdir(parents=True, exist_ok=True)
        (old_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (old_adapter / "adapter_model.bin").write_bytes(b"oldold")
        (new_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (new_adapter / "adapter_model.bin").write_bytes(b"newnew")
        old_node = register_adapter_artifact(
            registry_path=archive_registry.as_posix(),
            adapter_path=old_adapter.as_posix(),
            promoted=False,
            used=True,
            source="archive_test_old",
        )
        new_node = register_adapter_artifact(
            registry_path=archive_registry.as_posix(),
            adapter_path=new_adapter.as_posix(),
            promoted=False,
            used=True,
            source="archive_test_new",
        )
        pr_archive = prune_adapter_artifacts(
            registry_path=archive_registry.as_posix(),
            keep_latest=1,
            keep_promoted=0,
            max_total_bytes=0,
            remove_missing=True,
            remove_invalid=False,
            delete_files=True,
            archive_before_delete=True,
            archive_dir=archive_dir.as_posix(),
        )
        assert int(pr_archive.get("archived", 0)) >= 1
        assert int(pr_archive.get("deleted_dirs", 0)) >= 1
        assert not old_adapter.exists()
        zips = list(archive_dir.glob("*.zip"))
        assert len(zips) >= 1
        assert str(old_node.get("id", "")).strip()
        assert str(new_node.get("id", "")).strip()

        restored = restore_archived_artifact(
            registry_path=archive_registry.as_posix(),
            artifact_id=str(old_node.get("id", "")).strip(),
            output_dir=(root / "restored").as_posix(),
            activate=True,
        )
        assert bool(restored.get("restored", False))
        restored_path = Path(str(restored.get("adapter_path", "")).strip())
        assert restored_path.exists()
        assert (restored_path / "adapter_config.json").exists()
        assert (restored_path / "adapter_model.bin").exists() or (restored_path / "adapter_model.safetensors").exists()

        reg_after = json.loads(archive_registry.read_text(encoding="utf-8"))
        assert isinstance(reg_after.get("archives"), list) and len(reg_after.get("archives", [])) >= 1
        restored_marked = False
        for node in reg_after.get("archives", []):
            if not isinstance(node, dict):
                continue
            if int(node.get("restore_count", 0) or 0) <= 0:
                continue
            if str(node.get("restored_adapter_path", "")).strip() == restored_path.resolve().as_posix():
                restored_marked = True
                break
        assert restored_marked

        resolved_restored = resolve_best_adapter_path(
            active_registry_path=(root / "missing_active_restore.json").as_posix(),
            artifact_registry_path=archive_registry.as_posix(),
            require_valid=True,
        )
        assert resolved_restored == restored_path.resolve().as_posix()
        old_archives = [
            x
            for x in reg_after.get("archives", [])
            if isinstance(x, dict) and str(x.get("artifact_id", "")).strip() == str(old_node.get("id", "")).strip()
        ]
        assert old_archives
        old_archive_path = Path(str(old_archives[0].get("archive_path", "")).strip())
        old_archive_sha = str(old_archives[0].get("archive_sha256", "")).strip()
        assert old_archive_path.exists()
        assert old_archive_sha
        with old_archive_path.open("ab") as f:
            f.write(b"tamper")
        tampered_restore = restore_archived_artifact(
            registry_path=archive_registry.as_posix(),
            archive_path=old_archive_path.as_posix(),
            output_dir=(root / "restored_tampered").as_posix(),
            activate=False,
        )
        assert not bool(tampered_restore.get("restored", False))
        assert "hash" in str(tampered_restore.get("reason", "")).lower()
        tampered_restore_skip = restore_archived_artifact(
            registry_path=archive_registry.as_posix(),
            archive_path=old_archive_path.as_posix(),
            output_dir=(root / "restored_tampered_skip").as_posix(),
            verify_archive_hash=False,
            activate=False,
        )
        assert bool(tampered_restore_skip.get("restored", False))

        # Auto restore (without explicit archive) should skip broken latest candidate and fallback.
        fallback_registry = root / "adapter_artifacts_fallback.json"
        fallback_archive_dir = root / "fallback_archives"
        fb_old = root / "fb_old"
        fb_mid = root / "fb_mid"
        fb_new = root / "fb_new"
        for d, payload in ((fb_old, b"old"), (fb_mid, b"mid"), (fb_new, b"new")):
            d.mkdir(parents=True, exist_ok=True)
            (d / "adapter_config.json").write_text("{}", encoding="utf-8")
            (d / "adapter_model.bin").write_bytes(payload)
        register_adapter_artifact(
            registry_path=fallback_registry.as_posix(),
            adapter_path=fb_old.as_posix(),
            source="fallback_old",
            used=True,
        )
        register_adapter_artifact(
            registry_path=fallback_registry.as_posix(),
            adapter_path=fb_mid.as_posix(),
            source="fallback_mid",
            used=True,
        )
        register_adapter_artifact(
            registry_path=fallback_registry.as_posix(),
            adapter_path=fb_new.as_posix(),
            source="fallback_new",
            used=True,
        )
        fb_prune = prune_adapter_artifacts(
            registry_path=fallback_registry.as_posix(),
            keep_latest=1,
            keep_promoted=0,
            max_total_bytes=0,
            remove_missing=True,
            remove_invalid=False,
            delete_files=True,
            archive_before_delete=True,
            archive_dir=fallback_archive_dir.as_posix(),
        )
        assert int(fb_prune.get("archived", 0)) >= 2
        fb_payload = json.loads(fallback_registry.read_text(encoding="utf-8"))
        fb_archives = [x for x in fb_payload.get("archives", []) if isinstance(x, dict)]
        assert len(fb_archives) >= 2
        # Corrupt the first-ranked candidate.
        first_path = Path(str(fb_archives[0].get("archive_path", "")).strip())
        assert first_path.exists()
        with first_path.open("ab") as f:
            f.write(b"corrupt")
        fb_restore = restore_archived_artifact(
            registry_path=fallback_registry.as_posix(),
            output_dir=(root / "fallback_restore").as_posix(),
            verify_archive_hash=True,
            activate=False,
        )
        assert bool(fb_restore.get("restored", False))
        assert int(fb_restore.get("attempted_candidates", 0)) >= 2
        failed_candidates = fb_restore.get("failed_candidates", [])
        assert isinstance(failed_candidates, list) and len(failed_candidates) >= 1
        assert "hash" in str(failed_candidates[0].get("reason", "")).lower()

        # Signature-aware archive restore should require key when signature exists.
        signed_registry = root / "adapter_artifacts_signed.json"
        signed_archive_dir = root / "signed_archives"
        signed_old = root / "signed_old"
        signed_new = root / "signed_new"
        signed_old.mkdir(parents=True, exist_ok=True)
        signed_new.mkdir(parents=True, exist_ok=True)
        (signed_old / "adapter_config.json").write_text("{}", encoding="utf-8")
        (signed_old / "adapter_model.bin").write_bytes(b"signed_old")
        (signed_new / "adapter_config.json").write_text("{}", encoding="utf-8")
        (signed_new / "adapter_model.bin").write_bytes(b"signed_new")
        old_sign_node = register_adapter_artifact(
            registry_path=signed_registry.as_posix(),
            adapter_path=signed_old.as_posix(),
            source="signed_old",
            used=True,
        )
        register_adapter_artifact(
            registry_path=signed_registry.as_posix(),
            adapter_path=signed_new.as_posix(),
            source="signed_new",
            used=True,
        )
        prev_sign_key = os.environ.get("ADAPTER_ARCHIVE_SIGNING_KEY", "")
        os.environ["ADAPTER_ARCHIVE_SIGNING_KEY"] = "unit-test-sign-key"
        try:
            signed_prune = prune_adapter_artifacts(
                registry_path=signed_registry.as_posix(),
                keep_latest=1,
                keep_promoted=0,
                max_total_bytes=0,
                remove_missing=True,
                remove_invalid=False,
                delete_files=True,
                archive_before_delete=True,
                archive_dir=signed_archive_dir.as_posix(),
                archive_signing_key_id="kid-unit-1",
            )
            assert int(signed_prune.get("archived", 0)) >= 1
            signed_payload = json.loads(signed_registry.read_text(encoding="utf-8"))
            signed_archives = [
                x
                for x in signed_payload.get("archives", [])
                if isinstance(x, dict) and str(x.get("artifact_id", "")).strip() == str(old_sign_node.get("id", "")).strip()
            ]
            assert signed_archives
            signed_row = signed_archives[0]
            signed_archive_path = Path(str(signed_row.get("archive_path", "")).strip())
            assert signed_archive_path.exists()
            assert str(signed_row.get("archive_hmac_sha256", "")).strip()
            assert str(signed_row.get("archive_sig_algo", "")).strip() == "hmac-sha256-v1"
            assert str(signed_row.get("archive_signing_key_id", "")).strip() == "kid-unit-1"

            signed_restore_ok = restore_archived_artifact(
                registry_path=signed_registry.as_posix(),
                archive_path=signed_archive_path.as_posix(),
                output_dir=(root / "signed_restore_ok").as_posix(),
                verify_archive_hash=True,
                verify_archive_signature=True,
                require_archive_signature=True,
                activate=False,
            )
            assert bool(signed_restore_ok.get("restored", False))
            assert bool(signed_restore_ok.get("archive_signature_verified", False))

            os.environ["ADAPTER_ARCHIVE_SIGNING_KEY"] = ""
            signed_restore_no_key = restore_archived_artifact(
                registry_path=signed_registry.as_posix(),
                archive_path=signed_archive_path.as_posix(),
                output_dir=(root / "signed_restore_no_key").as_posix(),
                verify_archive_hash=True,
                verify_archive_signature=True,
                require_archive_signature=True,
                activate=False,
            )
            assert not bool(signed_restore_no_key.get("restored", False))
            assert "signature_key_missing" in str(signed_restore_no_key.get("reason", "")).lower()

            signed_restore_with_map = restore_archived_artifact(
                registry_path=signed_registry.as_posix(),
                archive_path=signed_archive_path.as_posix(),
                output_dir=(root / "signed_restore_with_map").as_posix(),
                verify_archive_hash=True,
                verify_archive_signature=True,
                require_archive_signature=True,
                archive_signing_keys_json=json.dumps({"kid-unit-1": "unit-test-sign-key"}, ensure_ascii=False),
                activate=False,
            )
            assert bool(signed_restore_with_map.get("restored", False))
            assert bool(signed_restore_with_map.get("archive_signature_verified", False))

            signed_restore_skip_sig = restore_archived_artifact(
                registry_path=signed_registry.as_posix(),
                archive_path=signed_archive_path.as_posix(),
                output_dir=(root / "signed_restore_skip_sig").as_posix(),
                verify_archive_hash=True,
                verify_archive_signature=False,
                require_archive_signature=False,
                activate=False,
            )
            assert bool(signed_restore_skip_sig.get("restored", False))
        finally:
            os.environ["ADAPTER_ARCHIVE_SIGNING_KEY"] = prev_sign_key

        # Audit can backfill signatures for old unsigned archives.
        audit_sign_registry = root / "adapter_artifacts_audit_sign.json"
        audit_sign_archive_dir = root / "audit_sign_archives"
        as_old = root / "audit_sign_old"
        as_new = root / "audit_sign_new"
        for d, payload in ((as_old, b"aso"), (as_new, b"asn")):
            d.mkdir(parents=True, exist_ok=True)
            (d / "adapter_config.json").write_text("{}", encoding="utf-8")
            (d / "adapter_model.bin").write_bytes(payload)
        prev_sign_key2 = os.environ.get("ADAPTER_ARCHIVE_SIGNING_KEY", "")
        try:
            os.environ["ADAPTER_ARCHIVE_SIGNING_KEY"] = ""
            register_adapter_artifact(
                registry_path=audit_sign_registry.as_posix(),
                adapter_path=as_old.as_posix(),
                source="audit_sign_old",
                used=True,
            )
            register_adapter_artifact(
                registry_path=audit_sign_registry.as_posix(),
                adapter_path=as_new.as_posix(),
                source="audit_sign_new",
                used=True,
            )
            prune_adapter_artifacts(
                registry_path=audit_sign_registry.as_posix(),
                keep_latest=1,
                keep_promoted=0,
                max_total_bytes=0,
                remove_missing=True,
                remove_invalid=False,
                delete_files=True,
                archive_before_delete=True,
                archive_dir=audit_sign_archive_dir.as_posix(),
            )
            before_payload = json.loads(audit_sign_registry.read_text(encoding="utf-8"))
            before_archives = [x for x in before_payload.get("archives", []) if isinstance(x, dict)]
            assert before_archives
            assert not str(before_archives[0].get("archive_hmac_sha256", "")).strip()

            audit_signed = audit_adapter_artifacts(
                registry_path=audit_sign_registry.as_posix(),
                refresh_stats=True,
                fix=True,
                verify_archive_hash=True,
                sign_missing_archives=True,
                archive_signing_key="audit-sign-key",
                archive_signing_key_id="kid-audit-1",
            )
            assert bool(audit_signed.get("wrote", False))
            assert int(audit_signed.get("archive_signed_missing", 0)) >= 1
            after_payload = json.loads(audit_sign_registry.read_text(encoding="utf-8"))
            after_archives = [x for x in after_payload.get("archives", []) if isinstance(x, dict)]
            assert after_archives
            assert str(after_archives[0].get("archive_hmac_sha256", "")).strip()
            assert str(after_archives[0].get("archive_signing_key_id", "")).strip() == "kid-audit-1"
        finally:
            os.environ["ADAPTER_ARCHIVE_SIGNING_KEY"] = prev_sign_key2

        # Base-model-aware restore should choose matching archive in strict mode.
        restore_model_registry = root / "adapter_artifacts_restore_model.json"
        restore_model_archive_dir = root / "restore_model_archives"
        rm_a = root / "restore_model_a"
        rm_b = root / "restore_model_b"
        rm_c = root / "restore_model_c"
        for d in (rm_a, rm_b, rm_c):
            d.mkdir(parents=True, exist_ok=True)
            (d / "adapter_config.json").write_text("{}", encoding="utf-8")
        (rm_a / "adapter_model.bin").write_bytes(b"a")
        (rm_b / "adapter_model.bin").write_bytes(b"b")
        (rm_c / "adapter_model.bin").write_bytes(b"c")
        register_adapter_artifact(
            registry_path=restore_model_registry.as_posix(),
            adapter_path=rm_a.as_posix(),
            base_model="model/a",
            source="restore_model_a",
            used=True,
        )
        register_adapter_artifact(
            registry_path=restore_model_registry.as_posix(),
            adapter_path=rm_b.as_posix(),
            base_model="model/b",
            source="restore_model_b",
            used=True,
        )
        register_adapter_artifact(
            registry_path=restore_model_registry.as_posix(),
            adapter_path=rm_c.as_posix(),
            base_model="model/c",
            source="restore_model_c",
            used=True,
        )
        pr_restore_model = prune_adapter_artifacts(
            registry_path=restore_model_registry.as_posix(),
            keep_latest=1,
            keep_promoted=0,
            max_total_bytes=0,
            remove_missing=True,
            remove_invalid=False,
            delete_files=True,
            archive_before_delete=True,
            archive_dir=restore_model_archive_dir.as_posix(),
        )
        assert int(pr_restore_model.get("archived", 0)) >= 2
        restored_b = restore_archived_artifact(
            registry_path=restore_model_registry.as_posix(),
            preferred_base_model="model/b",
            strict_base_model=True,
            output_dir=(root / "restore_model_out").as_posix(),
            activate=False,
        )
        assert bool(restored_b.get("restored", False))
        restored_b_path = Path(str(restored_b.get("adapter_path", "")).strip())
        assert restored_b_path.exists()
        restore_model_payload = json.loads(restore_model_registry.read_text(encoding="utf-8"))
        restore_model_rows = [x for x in restore_model_payload.get("artifacts", []) if isinstance(x, dict)]
        restored_b_rows = [
            x for x in restore_model_rows if str(x.get("adapter_path", "")).strip() == restored_b_path.resolve().as_posix()
        ]
        assert restored_b_rows
        assert str(restored_b_rows[0].get("base_model", "")).strip() == "model/b"
        restored_none = restore_archived_artifact(
            registry_path=restore_model_registry.as_posix(),
            preferred_base_model="model/z",
            strict_base_model=True,
            output_dir=(root / "restore_model_out2").as_posix(),
            activate=False,
        )
        assert not bool(restored_none.get("restored", False))
        assert "base_model" in str(restored_none.get("reason", "")).lower()

        # Audit + fix should remove missing/invalid artifacts and missing archives.
        audit_registry = root / "adapter_artifacts_audit.json"
        audit_valid = root / "audit_valid"
        audit_invalid = root / "audit_invalid"
        audit_missing = root / "audit_missing"
        for d in (audit_valid, audit_invalid, audit_missing):
            d.mkdir(parents=True, exist_ok=True)
        (audit_valid / "adapter_config.json").write_text("{}", encoding="utf-8")
        (audit_valid / "adapter_model.bin").write_bytes(b"v")
        (audit_invalid / "adapter_model.bin").write_bytes(b"i")
        (audit_missing / "adapter_config.json").write_text("{}", encoding="utf-8")
        (audit_missing / "adapter_model.bin").write_bytes(b"m")
        register_adapter_artifact(
            registry_path=audit_registry.as_posix(),
            adapter_path=audit_valid.as_posix(),
            source="audit_valid",
            used=True,
        )
        register_adapter_artifact(
            registry_path=audit_registry.as_posix(),
            adapter_path=audit_invalid.as_posix(),
            source="audit_invalid",
            used=True,
        )
        register_adapter_artifact(
            registry_path=audit_registry.as_posix(),
            adapter_path=audit_missing.as_posix(),
            source="audit_missing",
            used=True,
        )
        shutil.rmtree(audit_missing.as_posix(), ignore_errors=True)
        archive_ok = root / "audit_archive_ok.zip"
        archive_ok.write_bytes(b"ok")
        audit_payload = json.loads(audit_registry.read_text(encoding="utf-8"))
        audit_payload["archives"] = [
            {"archive_path": archive_ok.as_posix(), "archived_at": 1.0},
            {"archive_path": (root / "audit_archive_missing.zip").as_posix(), "archived_at": 2.0},
            {"archive_path": archive_ok.as_posix(), "archived_at": 0.5},
        ]
        audit_registry.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        audit_scan = audit_adapter_artifacts(
            registry_path=audit_registry.as_posix(),
            refresh_stats=True,
            fix=False,
        )
        assert int(audit_scan.get("missing_artifacts", 0)) >= 1
        assert int(audit_scan.get("invalid_artifacts", 0)) >= 1
        assert int(audit_scan.get("missing_archives", 0)) >= 1
        assert not bool(audit_scan.get("wrote", False))

        audit_fix = audit_adapter_artifacts(
            registry_path=audit_registry.as_posix(),
            refresh_stats=True,
            fix=True,
            drop_missing=True,
            drop_invalid=True,
            drop_missing_archives=True,
        )
        assert bool(audit_fix.get("wrote", False))
        assert int(audit_fix.get("dropped_missing", 0)) >= 1
        assert int(audit_fix.get("dropped_invalid", 0)) >= 1
        assert int(audit_fix.get("dropped_missing_archives", 0)) >= 1
        fixed_payload = json.loads(audit_registry.read_text(encoding="utf-8"))
        fixed_rows = [x for x in fixed_payload.get("artifacts", []) if isinstance(x, dict)]
        assert len(fixed_rows) == 1
        assert str(fixed_rows[0].get("adapter_path", "")).strip() == audit_valid.resolve().as_posix()
        fixed_archives = [x for x in fixed_payload.get("archives", []) if isinstance(x, dict)]
        assert len(fixed_archives) == 1
        assert str(fixed_archives[0].get("archive_path", "")).strip() == archive_ok.resolve().as_posix()

        # Maintain should self-heal by restoring from archive when current artifacts are unusable.
        maintain_registry = root / "adapter_artifacts_maintain.json"
        maintain_active = root / "active_adapter_maintain.json"
        maintain_archive_dir = root / "maintain_archives"
        maintain_old = root / "maintain_old"
        maintain_new = root / "maintain_new"
        maintain_old.mkdir(parents=True, exist_ok=True)
        maintain_new.mkdir(parents=True, exist_ok=True)
        (maintain_old / "adapter_config.json").write_text("{}", encoding="utf-8")
        (maintain_old / "adapter_model.bin").write_bytes(b"old")
        (maintain_new / "adapter_config.json").write_text("{}", encoding="utf-8")
        (maintain_new / "adapter_model.bin").write_bytes(b"new")
        register_adapter_artifact(
            registry_path=maintain_registry.as_posix(),
            adapter_path=maintain_old.as_posix(),
            source="maintain_old",
            used=True,
        )
        register_adapter_artifact(
            registry_path=maintain_registry.as_posix(),
            adapter_path=maintain_new.as_posix(),
            source="maintain_new",
            used=True,
        )
        # Archive old adapter, keep newest.
        pr_maint = prune_adapter_artifacts(
            registry_path=maintain_registry.as_posix(),
            keep_latest=1,
            keep_promoted=0,
            max_total_bytes=0,
            remove_missing=True,
            delete_files=True,
            archive_before_delete=True,
            archive_dir=maintain_archive_dir.as_posix(),
        )
        assert int(pr_maint.get("archived", 0)) >= 1
        # Simulate current adapter corruption/missing.
        shutil.rmtree(maintain_new.as_posix(), ignore_errors=True)

        maintain = maintain_adapter_artifacts(
            registry_path=maintain_registry.as_posix(),
            active_registry_path=maintain_active.as_posix(),
            preferred_base_model="",
            strict_base_model=False,
            audit_fix=True,
            audit_drop_missing=True,
            audit_drop_invalid=False,
            audit_drop_missing_archives=True,
            prune=False,
            auto_restore_missing=True,
            restore_output_dir=(root / "maintain_restored").as_posix(),
            restore_activate=True,
            sync_active_registry=True,
        )
        assert bool(maintain.get("healthy", False))
        restored_adapter = Path(str(maintain.get("resolved_adapter_path", "")).strip())
        assert restored_adapter.exists()
        assert (restored_adapter / "adapter_config.json").exists()
        active_payload = json.loads(maintain_active.read_text(encoding="utf-8"))
        assert str(active_payload.get("active_adapter", "")).strip() == restored_adapter.resolve().as_posix()

        # Maintain with prune enabled should report prune stats and trim artifacts.
        maintain_extra = root / "maintain_extra"
        maintain_extra.mkdir(parents=True, exist_ok=True)
        (maintain_extra / "adapter_config.json").write_text("{}", encoding="utf-8")
        (maintain_extra / "adapter_model.bin").write_bytes(b"x")
        register_adapter_artifact(
            registry_path=maintain_registry.as_posix(),
            adapter_path=maintain_extra.as_posix(),
            source="maintain_extra",
            used=True,
        )
        maintain_prune = maintain_adapter_artifacts(
            registry_path=maintain_registry.as_posix(),
            active_registry_path=maintain_active.as_posix(),
            preferred_base_model="",
            strict_base_model=False,
            audit_fix=True,
            audit_drop_missing=True,
            audit_drop_invalid=False,
            audit_drop_missing_archives=True,
            prune=True,
            keep_latest=1,
            keep_promoted=0,
            max_total_bytes=0,
            prune_remove_invalid=False,
            prune_delete_files=False,
            prune_archive_before_delete=False,
            auto_restore_missing=False,
            sync_active_registry=True,
        )
        assert isinstance(maintain_prune.get("prune"), dict)
        assert int(maintain_prune.get("prune", {}).get("remaining", 0)) >= 1


def _test_loop_early_stop() -> None:
    s1, r1 = should_stop_early(no_improve_streak=2, fail_streak=0, patience=2, max_fail_streak=2)
    assert s1 and "no_improve_streak" in r1
    s2, r2 = should_stop_early(no_improve_streak=0, fail_streak=2, patience=2, max_fail_streak=2)
    assert s2 and "fail_streak" in r2
    s3, _ = should_stop_early(no_improve_streak=1, fail_streak=0, patience=2, max_fail_streak=2)
    assert not s3


def main() -> None:
    _test_dataset_builder()
    _test_jsonl_gzip_helpers()
    _test_promote_decision()
    _test_replay_builder()
    _test_hard_case_mining()
    _test_fallback_manifest_from_report()
    _test_loop_replay_helpers()
    _test_federated_train_helpers()
    _test_auto_iterate_federated_train_cmd()
    _test_adaptive_score_weights()
    _test_adaptive_round_overrides()
    _test_adaptive_train_policy()
    _test_adaptive_dataset_penalty()
    _test_adaptive_candidate_space()
    _test_adaptive_target_modules()
    _test_adaptive_budget_overrides()
    _test_adaptive_bucket_thresholds()
    _test_adaptive_trend_memory()
    _test_adaptive_bandit()
    _test_adaptive_bandit_contextual()
    _test_adaptive_bandit_persistence()
    _test_candidate_grid_and_score()
    _test_history_guided_grid()
    _test_adapter_artifact_store()
    _test_loop_early_stop()
    print("adapter_pipeline_ok")


if __name__ == "__main__":
    main()
