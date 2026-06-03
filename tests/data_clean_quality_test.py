from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.knowledge.data_etl import DataCleanConfig, clean_training_file


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, str):
                f.write(row + "\n")
            else:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_data_clean_quality_score_reflects_noise() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        good_in = root / "good.jsonl"
        bad_in = root / "bad.jsonl"
        good_out = root / "good_clean.jsonl"
        bad_out = root / "bad_clean.jsonl"

        _write_jsonl(
            good_in,
            [
                {"prompt": "How to backup db?", "response": "Use snapshot and verify checksum."},
                {"prompt": "How to rollback?", "response": "Prepare checklist and staged restore."},
                {"prompt": "How to validate?", "response": "Run smoke tests and compare metrics."},
            ],
        )
        _write_jsonl(
            bad_in,
            [
                {"prompt": "How to backup db?", "response": "Use snapshot and verify checksum."},
                {"prompt": "x", "response": "y"},
                {"prompt": "aaaaabbbbbcccccdddddeeeeefffff", "response": "zzzzzzzzzzzzzzzzzzzz"},
                {"prompt": "How to backup db?", "response": "Use snapshot and verify checksum."},
                "{\"broken_json\":",
                {"messages": [{"role": "user", "content": "u-only"}]},
            ],
        )

        cfg = DataCleanConfig(
            min_prompt_chars=2,
            min_response_chars=4,
            max_repeat_run=14,
            max_symbol_ratio=0.60,
            dedup=True,
            use_ftfy=False,
        )
        good_report = clean_training_file(good_in, good_out, cfg=cfg)
        bad_report = clean_training_file(bad_in, bad_out, cfg=cfg)

        assert float(good_report.get("quality_score", 0.0)) > float(bad_report.get("quality_score", 0.0))
        assert float(good_report.get("keep_ratio", 0.0)) > float(bad_report.get("keep_ratio", 0.0))
        assert float(good_report.get("quality_score", 0.0)) >= 0.70
        assert float(bad_report.get("quality_score", 0.0)) <= 0.65


def test_clean_training_script_quality_gate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        bad_in = root / "bad.jsonl"
        report_path = root / "report.json"
        out_dir = root / "out"

        _write_jsonl(
            bad_in,
            [
                {"prompt": "x", "response": "y"},
                {"prompt": "x", "response": "y"},
                "{\"broken_json\":",
            ],
        )

        cmd = [
            sys.executable,
            "scripts/clean_training_data.py",
            "--paths",
            bad_in.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--report",
            report_path.as_posix(),
            "--min-quality-score",
            "0.95",
            "--min-keep-ratio",
            "0.80",
            "--fail-on-gate",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 2
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        gate = payload.get("gate", {})
        assert isinstance(gate, dict)
        assert bool(gate.get("passed", True)) is False


def test_clean_training_script_cross_leakage_gate_train_eval_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train_in = root / "adapter_train.jsonl"
        eval_in = root / "adapter_eval.jsonl"
        report_path = root / "report_cross_fail.json"
        out_dir = root / "out"

        shared = {"prompt": "How to backup database safely?", "response": "Create snapshot, verify checksum, then proceed."}
        _write_jsonl(
            train_in,
            [
                shared,
                {"prompt": "How to rollback service?", "response": "Use staged rollback checklist and verification."},
            ],
        )
        _write_jsonl(
            eval_in,
            [
                shared,
                {"prompt": "How to monitor latency?", "response": "Track p95, p99 and error rates continuously."},
            ],
        )

        cmd = [
            sys.executable,
            "scripts/clean_training_data.py",
            "--paths",
            train_in.as_posix(),
            eval_in.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--report",
            report_path.as_posix(),
            "--check-cross-leakage",
            "--max-train-eval-pair-overlap-ratio",
            "0.0",
            "--fail-on-gate",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 2
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        gate = payload.get("gate", {})
        assert isinstance(gate, dict)
        assert bool(gate.get("passed", True)) is False
        reasons = [str(x) for x in list(gate.get("reasons", []) or [])]
        assert any("train_eval_pair_overlap_above_threshold" in r for r in reasons)
        leakage = payload.get("cross_leakage", {})
        assert isinstance(leakage, dict)
        summary = leakage.get("summary", {})
        assert isinstance(summary, dict)
        assert float(summary.get("train_eval_pair_overlap_ratio_max", 0.0)) > 0.0


def test_clean_training_script_cross_leakage_gate_train_eval_pass() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train_in = root / "adapter_train.jsonl"
        eval_in = root / "adapter_eval.jsonl"
        report_path = root / "report_cross_pass.json"
        out_dir = root / "out"

        _write_jsonl(
            train_in,
            [
                {"prompt": "How to backup database safely?", "response": "Create snapshot, verify checksum, then proceed."},
                {"prompt": "How to rollback service?", "response": "Use staged rollback checklist and verification."},
            ],
        )
        _write_jsonl(
            eval_in,
            [
                {"prompt": "How to monitor latency?", "response": "Track p95, p99 and error rates continuously."},
                {"prompt": "How to debug memory leak?", "response": "Collect heap profiles and compare deltas."},
            ],
        )

        cmd = [
            sys.executable,
            "scripts/clean_training_data.py",
            "--paths",
            train_in.as_posix(),
            eval_in.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--report",
            report_path.as_posix(),
            "--check-cross-leakage",
            "--max-cross-pair-overlap-ratio",
            "0.0",
            "--max-cross-prompt-overlap-ratio",
            "0.0",
            "--max-train-eval-pair-overlap-ratio",
            "0.0",
            "--max-train-eval-prompt-overlap-ratio",
            "0.0",
            "--fail-on-gate",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        gate = payload.get("gate", {})
        assert isinstance(gate, dict)
        assert bool(gate.get("passed", False)) is True
        leakage = payload.get("cross_leakage", {})
        assert isinstance(leakage, dict)
        summary = leakage.get("summary", {})
        assert isinstance(summary, dict)
        assert float(summary.get("train_eval_pair_overlap_ratio_max", 1.0)) <= 0.0


def test_clean_training_script_semantic_leakage_gate_train_eval_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train_in = root / "adapter_train.jsonl"
        eval_in = root / "adapter_eval.jsonl"
        report_path = root / "report_semantic_fail.json"
        out_dir = root / "out"

        _write_jsonl(
            train_in,
            [
                {"prompt": "backup database before migration with checksum", "response": "prepare backup and verify checksum"},
                {"prompt": "rollback service safely with checklist", "response": "execute staged rollback and validate"},
            ],
        )
        _write_jsonl(
            eval_in,
            [
                {
                    "prompt": "before migration backup database with checksum verification",
                    "response": "create backup then run checksum verification",
                },
                {"prompt": "monitor latency with p95 and error rate", "response": "track p95 and error rate over time"},
            ],
        )

        cmd = [
            sys.executable,
            "scripts/clean_training_data.py",
            "--paths",
            train_in.as_posix(),
            eval_in.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--report",
            report_path.as_posix(),
            "--check-semantic-leakage",
            "--semantic-overlap-threshold",
            "0.70",
            "--semantic-max-items-per-file",
            "256",
            "--semantic-max-pair-comparisons",
            "50000",
            "--max-train-eval-semantic-prompt-overlap-ratio",
            "0.0",
            "--fail-on-gate",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 2
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        gate = payload.get("gate", {})
        assert isinstance(gate, dict)
        assert bool(gate.get("passed", True)) is False
        reasons = [str(x) for x in list(gate.get("reasons", []) or [])]
        assert any("train_eval_semantic_prompt_overlap_above_threshold" in r for r in reasons)
        leakage = payload.get("cross_leakage", {})
        assert isinstance(leakage, dict)
        summary = leakage.get("summary", {})
        assert isinstance(summary, dict)
        assert float(summary.get("train_eval_semantic_prompt_overlap_ratio_max", 0.0)) > 0.0


def test_clean_training_script_semantic_leakage_gate_train_eval_pass() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train_in = root / "adapter_train.jsonl"
        eval_in = root / "adapter_eval.jsonl"
        report_path = root / "report_semantic_pass.json"
        out_dir = root / "out"

        _write_jsonl(
            train_in,
            [
                {"prompt": "rollback service safely with checklist", "response": "execute staged rollback and validate"},
                {"prompt": "backup database before migration with checksum", "response": "prepare backup and verify checksum"},
            ],
        )
        _write_jsonl(
            eval_in,
            [
                {"prompt": "weather is sunny today", "response": "carry an umbrella if rain is expected"},
                {"prompt": "scale kubernetes cluster automatically", "response": "set autoscaler thresholds and observe"},
            ],
        )

        cmd = [
            sys.executable,
            "scripts/clean_training_data.py",
            "--paths",
            train_in.as_posix(),
            eval_in.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--report",
            report_path.as_posix(),
            "--check-semantic-leakage",
            "--semantic-overlap-threshold",
            "0.70",
            "--semantic-max-items-per-file",
            "256",
            "--semantic-max-pair-comparisons",
            "50000",
            "--max-train-eval-semantic-prompt-overlap-ratio",
            "0.0",
            "--fail-on-gate",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        gate = payload.get("gate", {})
        assert isinstance(gate, dict)
        assert bool(gate.get("passed", False)) is True
        leakage = payload.get("cross_leakage", {})
        assert isinstance(leakage, dict)
        summary = leakage.get("summary", {})
        assert isinstance(summary, dict)
        assert float(summary.get("train_eval_semantic_prompt_overlap_ratio_max", 1.0)) <= 0.0


def test_clean_training_script_semantic_pair_leakage_gate_train_eval_fail() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train_in = root / "adapter_train.jsonl"
        eval_in = root / "adapter_eval.jsonl"
        report_path = root / "report_semantic_pair_fail.json"
        out_dir = root / "out"

        _write_jsonl(
            train_in,
            [
                {
                    "prompt": "backup database before migration with checksum validation",
                    "response": "create snapshot and validate checksum before migration",
                },
                {"prompt": "rollback service safely with checklist", "response": "execute staged rollback and validate"},
            ],
        )
        _write_jsonl(
            eval_in,
            [
                {
                    "prompt": "before migration do database backup and checksum validation",
                    "response": "create snapshot and validate checksum before migration",
                },
                {"prompt": "monitor latency with p95 and error rate", "response": "track p95 and error rate over time"},
            ],
        )

        cmd = [
            sys.executable,
            "scripts/clean_training_data.py",
            "--paths",
            train_in.as_posix(),
            eval_in.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--report",
            report_path.as_posix(),
            "--check-semantic-leakage",
            "--semantic-overlap-threshold",
            "0.70",
            "--semantic-max-items-per-file",
            "256",
            "--semantic-max-pair-comparisons",
            "50000",
            "--max-train-eval-semantic-pair-overlap-ratio",
            "0.0",
            "--fail-on-gate",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 2
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        gate = payload.get("gate", {})
        assert isinstance(gate, dict)
        assert bool(gate.get("passed", True)) is False
        reasons = [str(x) for x in list(gate.get("reasons", []) or [])]
        assert any("train_eval_semantic_pair_overlap_above_threshold" in r for r in reasons)
        leakage = payload.get("cross_leakage", {})
        assert isinstance(leakage, dict)
        summary = leakage.get("summary", {})
        assert isinstance(summary, dict)
        assert float(summary.get("train_eval_semantic_pair_overlap_ratio_max", 0.0)) > 0.0


def test_clean_training_script_semantic_pair_leakage_gate_train_eval_pass() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        train_in = root / "adapter_train.jsonl"
        eval_in = root / "adapter_eval.jsonl"
        report_path = root / "report_semantic_pair_pass.json"
        out_dir = root / "out"

        _write_jsonl(
            train_in,
            [
                {
                    "prompt": "rollback service safely with checklist",
                    "response": "execute staged rollback and validate",
                },
                {
                    "prompt": "backup database before migration with checksum",
                    "response": "prepare backup and verify checksum",
                },
            ],
        )
        _write_jsonl(
            eval_in,
            [
                {
                    "prompt": "weather is sunny today",
                    "response": "carry an umbrella if rain is expected",
                },
                {
                    "prompt": "scale kubernetes cluster automatically",
                    "response": "set autoscaler thresholds and observe",
                },
            ],
        )

        cmd = [
            sys.executable,
            "scripts/clean_training_data.py",
            "--paths",
            train_in.as_posix(),
            eval_in.as_posix(),
            "--out-dir",
            out_dir.as_posix(),
            "--report",
            report_path.as_posix(),
            "--check-semantic-leakage",
            "--semantic-overlap-threshold",
            "0.70",
            "--semantic-max-items-per-file",
            "256",
            "--semantic-max-pair-comparisons",
            "50000",
            "--max-train-eval-semantic-pair-overlap-ratio",
            "0.0",
            "--fail-on-gate",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        gate = payload.get("gate", {})
        assert isinstance(gate, dict)
        assert bool(gate.get("passed", False)) is True
        leakage = payload.get("cross_leakage", {})
        assert isinstance(leakage, dict)
        summary = leakage.get("summary", {})
        assert isinstance(summary, dict)
        assert float(summary.get("train_eval_semantic_pair_overlap_ratio_max", 1.0)) <= 0.0
