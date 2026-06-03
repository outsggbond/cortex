from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_reasoning_benchmark_cli_report_and_feedback() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_path = root / "reasoning_report.json"
        feedback_path = root / "reasoning_feedback.json"

        cmd = [
            sys.executable,
            "scripts/eval_reasoning_benchmark.py",
            "--report",
            report_path.as_posix(),
            "--feedback-path",
            feedback_path.as_posix(),
            "--min-pass-rate",
            "0.80",
            "--min-category-rate",
            "0.60",
            "--learn-feedback",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert isinstance(payload.get("gate"), dict)
        assert bool(payload.get("gate", {}).get("passed", False)) is True
        assert isinstance(payload.get("by_category"), dict)
        assert feedback_path.exists()


def test_reasoning_benchmark_cli_with_baseline_profile_gate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_path = root / "reasoning_report_baseline.json"
        baseline_profile = root / "baseline_profile.json"
        baseline_profile.write_text("{}", encoding="utf-8")

        cmd = [
            sys.executable,
            "scripts/eval_reasoning_benchmark.py",
            "--report",
            report_path.as_posix(),
            "--min-pass-rate",
            "0.80",
            "--min-category-rate",
            "0.60",
            "--baseline-profile-path",
            baseline_profile.as_posix(),
            "--min-delta-pass-rate",
            "-0.01",
            "--min-comparable-cases",
            "7",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        gate = payload.get("gate", {})
        assert isinstance(gate, dict)
        assert bool(gate.get("baseline_enabled", False)) is True
        compare = payload.get("baseline_compare", {})
        assert isinstance(compare, dict)
        assert int(compare.get("comparable_cases", 0)) >= 7
        assert str(compare.get("source", "")).startswith("profile:")


def test_reasoning_benchmark_cli_with_robustness_suite() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_path = root / "reasoning_report_robust.json"
        cmd = [
            sys.executable,
            "scripts/eval_reasoning_benchmark.py",
            "--report",
            report_path.as_posix(),
            "--min-pass-rate",
            "0.80",
            "--min-weighted-pass-rate",
            "0.80",
            "--min-pass-rate-ci95-lower",
            "0.30",
            "--min-category-rate",
            "0.60",
            "--min-category-rate-ci95-lower",
            "0.20",
            "--min-difficulty-rate",
            "0.70",
            "--min-difficulty-rate-ci95-lower",
            "0.20",
            "--min-hard-case-rate",
            "0.75",
            "--min-hard-case-rate-ci95-lower",
            "0.20",
            "--with-robustness",
            "--min-suite-rate",
            "0.70",
            "--min-suite-rate-ci95-lower",
            "0.20",
            "--max-ece",
            "0.40",
            "--max-brier",
            "0.40",
            "--required-suites",
            "core,robustness",
            "--required-difficulties",
            "easy,medium,hard,extreme",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        by_suite = payload.get("by_suite", {})
        assert isinstance(by_suite, dict)
        assert "robustness" in by_suite
        calibration = payload.get("calibration", {})
        assert isinstance(calibration, dict)
        assert float(calibration.get("ece", 1.0)) <= 1.0
        gate = payload.get("gate", {})
        assert isinstance(gate, dict)
        assert float(gate.get("min_weighted_pass_rate", 0.0)) > 0.0
        assert float(gate.get("min_hard_case_rate", 0.0)) > 0.0
        assert float(gate.get("min_pass_rate_ci95_lower", 0.0)) > 0.0
        assert float(gate.get("min_category_rate_ci95_lower", 0.0)) > 0.0
        assert float(gate.get("min_difficulty_rate_ci95_lower", 0.0)) > 0.0
        assert float(gate.get("min_hard_case_rate_ci95_lower", 0.0)) > 0.0
        assert float(gate.get("min_suite_rate_ci95_lower", 0.0)) > 0.0
        assert bool(payload.get("gate", {}).get("passed", False)) is True


def test_reasoning_benchmark_cli_with_adversarial_suite() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_path = root / "reasoning_report_adv.json"
        cmd = [
            sys.executable,
            "scripts/eval_reasoning_benchmark.py",
            "--report",
            report_path.as_posix(),
            "--min-pass-rate",
            "0.78",
            "--min-weighted-pass-rate",
            "0.78",
            "--min-category-rate",
            "0.58",
            "--min-difficulty-rate",
            "0.70",
            "--min-hard-case-rate",
            "0.75",
            "--with-adversarial",
            "--min-suite-rate",
            "0.70",
            "--max-ece",
            "0.40",
            "--max-brier",
            "0.40",
            "--required-suites",
            "core,adversarial",
            "--required-difficulties",
            "easy,medium,hard,extreme",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        by_suite = payload.get("by_suite", {})
        assert isinstance(by_suite, dict)
        assert "adversarial" in by_suite
        gate = payload.get("gate", {})
        assert isinstance(gate, dict)
        assert bool(gate.get("with_adversarial", False)) is True
        assert bool(payload.get("gate", {}).get("passed", False)) is True
