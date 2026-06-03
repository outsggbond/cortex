from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_reasoning_profile_tune_cli_writes_profile() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        profile_path = root / "reasoning_profile.json"
        report_path = root / "reasoning_profile_tune_report.json"
        feedback_path = root / "feedback.json"

        cmd = [
            sys.executable,
            "scripts/tune_reasoning_profile.py",
            "--profile-path",
            profile_path.as_posix(),
            "--report",
            report_path.as_posix(),
            "--feedback-path",
            feedback_path.as_posix(),
            "--max-candidates",
            "24",
            "--min-pass-rate",
            "0.80",
            "--min-category-rate",
            "0.60",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        assert profile_path.exists()
        assert report_path.exists()

        profile_payload = json.loads(profile_path.read_text(encoding="utf-8"))
        assert isinstance(profile_payload, dict)
        assert "goal_align_weight" in profile_payload
        assert "proceed_confidence_threshold" in profile_payload

        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(report_payload, dict)
        assert bool(report_payload.get("ok", False)) is True
        result = report_payload.get("result", {})
        assert isinstance(result, dict)
        assert int(result.get("candidates_evaluated", 0)) >= 1
        fg = result.get("failure_guided", {})
        assert isinstance(fg, dict)
        assert str(fg.get("state_path", "")).endswith("reasoning_failure_guided_state.json")
        assert bool(fg.get("state_persisted", False)) is True


def test_reasoning_profile_tune_cli_with_robustness() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        profile_path = root / "reasoning_profile_robust.json"
        report_path = root / "reasoning_profile_tune_robust_report.json"

        cmd = [
            sys.executable,
            "scripts/tune_reasoning_profile.py",
            "--profile-path",
            profile_path.as_posix(),
            "--report",
            report_path.as_posix(),
            "--max-candidates",
            "20",
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
            "--with-robustness",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("with_robustness", False)) is True
        assert bool(target.get("failure_guided", False)) is True
        assert str(target.get("failure_guided_state_path", "")).endswith("reasoning_failure_guided_state.json")
        assert bool(target.get("persist_failure_guided_state", False)) is True
        assert float(target.get("min_weighted_pass_rate", 0.0)) > 0.0
        assert float(target.get("min_hard_case_rate", 0.0)) > 0.0
        assert float(target.get("min_pass_rate_ci95_lower", 0.0)) > 0.0
        assert float(target.get("min_category_rate_ci95_lower", 0.0)) > 0.0
        assert float(target.get("min_difficulty_rate_ci95_lower", 0.0)) > 0.0
        assert float(target.get("min_hard_case_rate_ci95_lower", 0.0)) > 0.0
        assert float(target.get("min_suite_rate_ci95_lower", 0.0)) > 0.0
        assert float(target.get("max_ece", 0.0)) > 0.0
        assert bool(payload.get("ok", False)) is True


def test_reasoning_profile_tune_cli_with_adversarial() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        profile_path = root / "reasoning_profile_adv.json"
        report_path = root / "reasoning_profile_tune_adv_report.json"

        cmd = [
            sys.executable,
            "scripts/tune_reasoning_profile.py",
            "--profile-path",
            profile_path.as_posix(),
            "--report",
            report_path.as_posix(),
            "--max-candidates",
            "20",
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
            "--with-adversarial",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("with_adversarial", False)) is True
        assert bool(target.get("failure_guided", False)) is True
        assert str(target.get("failure_guided_state_path", "")).endswith("reasoning_failure_guided_state.json")
        assert bool(target.get("persist_failure_guided_state", False)) is True
        assert bool(payload.get("ok", False)) is True


def test_reasoning_profile_tune_cli_with_baseline_profile_gate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        profile_path = root / "reasoning_profile_baseline.json"
        report_path = root / "reasoning_profile_tune_baseline_report.json"
        baseline_profile = root / "baseline_profile.json"
        baseline_profile.write_text("{}", encoding="utf-8")

        cmd = [
            sys.executable,
            "scripts/tune_reasoning_profile.py",
            "--profile-path",
            profile_path.as_posix(),
            "--report",
            report_path.as_posix(),
            "--max-candidates",
            "16",
            "--min-pass-rate",
            "0.78",
            "--min-category-rate",
            "0.58",
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
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("baseline_enabled", False)) is True
        result = payload.get("result", {})
        assert isinstance(result, dict)
        baseline = result.get("baseline", {})
        assert isinstance(baseline, dict)
        assert bool(baseline.get("enabled", False)) is True
        best = result.get("best_metrics", {})
        assert isinstance(best, dict)
        assert int(best.get("baseline_comparable_cases", 0)) >= 7


def test_reasoning_profile_tune_cli_baseline_gate_requires_input() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        profile_path = root / "reasoning_profile_baseline_missing.json"
        report_path = root / "reasoning_profile_tune_baseline_missing_report.json"

        cmd = [
            sys.executable,
            "scripts/tune_reasoning_profile.py",
            "--profile-path",
            profile_path.as_posix(),
            "--report",
            report_path.as_posix(),
            "--max-candidates",
            "12",
            "--min-pass-rate",
            "0.78",
            "--min-category-rate",
            "0.58",
            "--min-delta-pass-rate",
            "0.02",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode != 0


def test_reasoning_profile_tune_cli_disable_failure_guided() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        profile_path = root / "reasoning_profile_no_fg.json"
        report_path = root / "reasoning_profile_tune_no_fg_report.json"

        cmd = [
            sys.executable,
            "scripts/tune_reasoning_profile.py",
            "--profile-path",
            profile_path.as_posix(),
            "--report",
            report_path.as_posix(),
            "--max-candidates",
            "16",
            "--min-pass-rate",
            "0.78",
            "--min-category-rate",
            "0.58",
            "--disable-failure-guided",
            "--failure-guided-max-variants",
            "0",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("failure_guided", True)) is False
        assert bool(target.get("persist_failure_guided_state", False)) is True
        result = payload.get("result", {})
        assert isinstance(result, dict)
        fg = result.get("failure_guided", {})
        assert isinstance(fg, dict)
        assert bool(fg.get("enabled", True)) is False


def test_reasoning_profile_tune_cli_disable_failure_guided_state_persist() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        profile_path = root / "reasoning_profile_no_state_persist.json"
        report_path = root / "reasoning_profile_tune_no_state_persist_report.json"
        state_path = root / "fg_state.json"

        cmd = [
            sys.executable,
            "scripts/tune_reasoning_profile.py",
            "--profile-path",
            profile_path.as_posix(),
            "--report",
            report_path.as_posix(),
            "--max-candidates",
            "16",
            "--min-pass-rate",
            "0.78",
            "--min-category-rate",
            "0.58",
            "--failure-guided-state-path",
            state_path.as_posix(),
            "--disable-failure-guided-state-persist",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("persist_failure_guided_state", True)) is False
        result = payload.get("result", {})
        assert isinstance(result, dict)
        fg = result.get("failure_guided", {})
        assert isinstance(fg, dict)
        assert bool(fg.get("enabled", False)) is True
        assert bool(fg.get("state_persisted", True)) is False
        assert state_path.exists() is False
