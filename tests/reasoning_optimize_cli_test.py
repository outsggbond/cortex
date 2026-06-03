from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(__file__))


def test_reasoning_optimize_cli_summary() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_dir = root / "round_reports"
        summary_path = root / "summary.json"
        feedback_path = root / "feedback.json"

        cmd = [
            sys.executable,
            "scripts/optimize_reasoning.py",
            "--rounds",
            "3",
            "--target-pass-rate",
            "0.80",
            "--target-category-rate",
            "0.60",
            "--report-dir",
            report_dir.as_posix(),
            "--summary",
            summary_path.as_posix(),
            "--feedback-path",
            feedback_path.as_posix(),
            "--learn-feedback",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        assert summary_path.exists()
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert bool(payload.get("ok", False)) is True
        assert int(payload.get("rounds_executed", 0)) >= 1
        assert report_dir.exists()


def test_reasoning_optimize_cli_with_profile_tuning() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_dir = root / "round_reports"
        summary_path = root / "summary.json"
        feedback_path = root / "feedback.json"
        profile_path = root / "reasoning_profile.json"

        cmd = [
            sys.executable,
            "scripts/optimize_reasoning.py",
            "--rounds",
            "2",
            "--target-pass-rate",
            "0.80",
            "--target-category-rate",
            "0.60",
            "--report-dir",
            report_dir.as_posix(),
            "--summary",
            summary_path.as_posix(),
            "--feedback-path",
            feedback_path.as_posix(),
            "--profile-path",
            profile_path.as_posix(),
            "--auto-tune-profile",
            "--tune-max-candidates",
            "20",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        assert summary_path.exists()
        assert profile_path.exists()
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert bool(payload.get("ok", False)) is True
        assert bool(payload.get("auto_tune_profile", False)) is True
        history = payload.get("history", [])
        assert isinstance(history, list)
        assert len(history) >= 1
        first = history[0] if history else {}
        assert isinstance(first, dict)
        assert isinstance(first.get("profile"), dict)
        tune = first.get("profile_tune", {})
        assert isinstance(tune, dict)
        fg = tune.get("failure_guided", {})
        assert isinstance(fg, dict)
        assert bool(fg.get("enabled", False)) is True
        assert bool(fg.get("state_persisted", False)) is True


def test_reasoning_optimize_cli_with_baseline_profile_targets() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_dir = root / "round_reports"
        summary_path = root / "summary.json"
        baseline_profile = root / "baseline_profile.json"
        baseline_profile.write_text("{}", encoding="utf-8")

        cmd = [
            sys.executable,
            "scripts/optimize_reasoning.py",
            "--rounds",
            "2",
            "--target-pass-rate",
            "0.80",
            "--target-category-rate",
            "0.60",
            "--baseline-profile-path",
            baseline_profile.as_posix(),
            "--target-min-delta-pass-rate",
            "-0.01",
            "--target-min-comparable-cases",
            "7",
            "--report-dir",
            report_dir.as_posix(),
            "--summary",
            summary_path.as_posix(),
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("baseline_enabled", False)) is True
        assert str(target.get("baseline_source", "")).startswith("profile:")
        assert int(target.get("min_comparable_cases", 0)) >= 7
        history = payload.get("history", [])
        assert isinstance(history, list)
        assert len(history) >= 1
        first = history[0] if history else {}
        assert isinstance(first, dict)
        compare = first.get("baseline_compare", {})
        assert isinstance(compare, dict)
        assert int(compare.get("comparable_cases", 0)) >= 7


def test_reasoning_optimize_cli_autotune_with_baseline_profile_targets() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_dir = root / "round_reports"
        summary_path = root / "summary.json"
        profile_path = root / "reasoning_profile.json"
        baseline_profile = root / "baseline_profile.json"
        baseline_profile.write_text("{}", encoding="utf-8")

        cmd = [
            sys.executable,
            "scripts/optimize_reasoning.py",
            "--rounds",
            "2",
            "--target-pass-rate",
            "0.80",
            "--target-category-rate",
            "0.60",
            "--baseline-profile-path",
            baseline_profile.as_posix(),
            "--target-min-delta-pass-rate",
            "-0.01",
            "--target-min-comparable-cases",
            "7",
            "--report-dir",
            report_dir.as_posix(),
            "--summary",
            summary_path.as_posix(),
            "--profile-path",
            profile_path.as_posix(),
            "--auto-tune-profile",
            "--tune-max-candidates",
            "12",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        history = payload.get("history", [])
        assert isinstance(history, list)
        assert len(history) >= 1
        first = history[0] if history else {}
        assert isinstance(first, dict)
        tune = first.get("profile_tune", {})
        assert isinstance(tune, dict)
        baseline = tune.get("baseline", {})
        assert isinstance(baseline, dict)
        assert bool(baseline.get("enabled", False)) is True


def test_reasoning_optimize_cli_with_robustness_gate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_dir = root / "round_reports"
        summary_path = root / "summary.json"
        feedback_path = root / "feedback.json"
        profile_path = root / "reasoning_profile.json"

        cmd = [
            sys.executable,
            "scripts/optimize_reasoning.py",
            "--rounds",
            "2",
            "--target-pass-rate",
            "0.80",
            "--target-weighted-pass-rate",
            "0.80",
            "--target-pass-rate-ci95-lower",
            "0.30",
            "--target-category-rate",
            "0.60",
            "--target-category-rate-ci95-lower",
            "0.20",
            "--target-difficulty-rate",
            "0.70",
            "--target-difficulty-rate-ci95-lower",
            "0.20",
            "--target-hard-case-rate",
            "0.75",
            "--target-hard-case-rate-ci95-lower",
            "0.20",
            "--target-suite-rate",
            "0.70",
            "--target-suite-rate-ci95-lower",
            "0.20",
            "--target-max-ece",
            "0.40",
            "--target-max-brier",
            "0.40",
            "--required-suites",
            "core,robustness",
            "--required-difficulties",
            "easy,medium,hard,extreme",
            "--with-robustness",
            "--report-dir",
            report_dir.as_posix(),
            "--summary",
            summary_path.as_posix(),
            "--feedback-path",
            feedback_path.as_posix(),
            "--profile-path",
            profile_path.as_posix(),
            "--auto-tune-profile",
            "--tune-max-candidates",
            "18",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("with_robustness", False)) is True
        assert bool(target.get("failure_guided", False)) is True
        assert str(target.get("failure_guided_state_path", "")).endswith("reasoning_failure_guided_state.json")
        assert bool(target.get("persist_failure_guided_state", False)) is True
        assert float(target.get("weighted_pass_rate", 0.0)) > 0.0
        assert float(target.get("hard_case_rate", 0.0)) > 0.0
        assert float(target.get("pass_rate_ci95_lower", 0.0)) > 0.0
        assert float(target.get("category_rate_ci95_lower", 0.0)) > 0.0
        assert float(target.get("difficulty_rate_ci95_lower", 0.0)) > 0.0
        assert float(target.get("hard_case_rate_ci95_lower", 0.0)) > 0.0
        assert float(target.get("suite_rate_ci95_lower", 0.0)) > 0.0
        assert float(target.get("max_ece", 0.0)) > 0.0
        assert bool(payload.get("ok", False)) is True


def test_reasoning_optimize_cli_failure_replay_path() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_dir = root / "round_reports"
        summary_path = root / "summary.json"

        cmd = [
            sys.executable,
            "scripts/optimize_reasoning.py",
            "--rounds",
            "2",
            "--target-pass-rate",
            "1.10",
            "--target-category-rate",
            "1.10",
            "--with-failure-replay",
            "--replay-max-cases",
            "2",
            "--replay-multiplier",
            "2",
            "--report-dir",
            report_dir.as_posix(),
            "--summary",
            summary_path.as_posix(),
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode != 0
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("with_failure_replay", False)) is True
        history = payload.get("history", [])
        assert isinstance(history, list)
        assert len(history) == 2
        second = history[1] if len(history) > 1 else {}
        assert isinstance(second, dict)
        assert int(second.get("replay_cases", 0)) >= 1


def test_reasoning_optimize_cli_with_adversarial_gate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_dir = root / "round_reports"
        summary_path = root / "summary.json"
        feedback_path = root / "feedback.json"
        profile_path = root / "reasoning_profile.json"

        cmd = [
            sys.executable,
            "scripts/optimize_reasoning.py",
            "--rounds",
            "2",
            "--target-pass-rate",
            "0.78",
            "--target-weighted-pass-rate",
            "0.78",
            "--target-category-rate",
            "0.58",
            "--target-difficulty-rate",
            "0.70",
            "--target-hard-case-rate",
            "0.75",
            "--target-suite-rate",
            "0.70",
            "--target-max-ece",
            "0.40",
            "--target-max-brier",
            "0.40",
            "--required-suites",
            "core,adversarial",
            "--required-difficulties",
            "easy,medium,hard,extreme",
            "--with-adversarial",
            "--report-dir",
            report_dir.as_posix(),
            "--summary",
            summary_path.as_posix(),
            "--feedback-path",
            feedback_path.as_posix(),
            "--profile-path",
            profile_path.as_posix(),
            "--auto-tune-profile",
            "--tune-max-candidates",
            "18",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("with_adversarial", False)) is True
        assert bool(target.get("failure_guided", False)) is True
        assert str(target.get("failure_guided_state_path", "")).endswith("reasoning_failure_guided_state.json")
        assert bool(target.get("persist_failure_guided_state", False)) is True
        assert bool(payload.get("ok", False)) is True


def test_reasoning_optimize_cli_disable_failure_guided() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_dir = root / "round_reports"
        summary_path = root / "summary.json"
        profile_path = root / "reasoning_profile.json"

        cmd = [
            sys.executable,
            "scripts/optimize_reasoning.py",
            "--rounds",
            "2",
            "--target-pass-rate",
            "0.80",
            "--target-category-rate",
            "0.60",
            "--report-dir",
            report_dir.as_posix(),
            "--summary",
            summary_path.as_posix(),
            "--profile-path",
            profile_path.as_posix(),
            "--auto-tune-profile",
            "--disable-failure-guided",
            "--failure-guided-max-variants",
            "0",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("failure_guided", True)) is False
        assert bool(target.get("persist_failure_guided_state", False)) is True
        history = payload.get("history", [])
        assert isinstance(history, list)
        assert len(history) >= 1
        first = history[0] if history else {}
        assert isinstance(first, dict)
        tune = first.get("profile_tune", {})
        assert isinstance(tune, dict)
        fg = tune.get("failure_guided", {})
        assert isinstance(fg, dict)
        assert bool(fg.get("enabled", True)) is False


def test_reasoning_optimize_cli_disable_failure_guided_state_persist() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        report_dir = root / "round_reports"
        summary_path = root / "summary.json"
        profile_path = root / "reasoning_profile.json"
        state_path = root / "fg_state.json"

        cmd = [
            sys.executable,
            "scripts/optimize_reasoning.py",
            "--rounds",
            "2",
            "--target-pass-rate",
            "0.80",
            "--target-category-rate",
            "0.60",
            "--report-dir",
            report_dir.as_posix(),
            "--summary",
            summary_path.as_posix(),
            "--profile-path",
            profile_path.as_posix(),
            "--failure-guided-state-path",
            state_path.as_posix(),
            "--disable-failure-guided-state-persist",
            "--auto-tune-profile",
        ]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        assert res.returncode == 0
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        target = payload.get("target", {})
        assert isinstance(target, dict)
        assert bool(target.get("persist_failure_guided_state", True)) is False
        history = payload.get("history", [])
        assert isinstance(history, list)
        assert len(history) >= 1
        first = history[0] if history else {}
        assert isinstance(first, dict)
        tune = first.get("profile_tune", {})
        assert isinstance(tune, dict)
        fg = tune.get("failure_guided", {})
        assert isinstance(fg, dict)
        assert bool(fg.get("enabled", False)) is True
        assert bool(fg.get("state_persisted", True)) is False
        assert state_path.exists() is False
