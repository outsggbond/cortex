from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.learning.dialogue.dialogue_dataset_builder import DialogueDatasetBuildConfig, build_dataset


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = str(line).strip()
        if not row:
            continue
        try:
            node = json.loads(row)
        except Exception:
            continue
        if isinstance(node, dict):
            out.append(node)
    return out


def _build_patterns(path: Path) -> None:
    pairs = [
        ["How to harden production release?", "backup database with checksum step one"],
        ["How to harden production release?", "backup database with checksum step two"],
        ["How to harden production release?", "backup database with checksum step three"],
        ["How to harden production release?", "rotate api keys and audit access logs"],
    ]
    path.write_text(json.dumps({"qa": {"pairs": pairs}}, ensure_ascii=False), encoding="utf-8")


def test_dataset_diversity_penalty_prefers_varied_responses() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        patterns = root / "dialogue_patterns.json"
        train_out = root / "adapter_train.jsonl"
        eval_out = root / "adapter_eval.jsonl"
        manifest_out = root / "manifest.json"
        _build_patterns(patterns)

        cfg = DialogueDatasetBuildConfig(
            patterns_path=patterns.as_posix(),
            chat_memory_path=(root / "missing_chat.jsonl").as_posix(),
            reflections_path=(root / "missing_reflections.jsonl").as_posix(),
            output_train_path=train_out.as_posix(),
            output_eval_path=eval_out.as_posix(),
            output_manifest_path=manifest_out.as_posix(),
            include_patterns=True,
            include_chat_memory=False,
            include_reflections=False,
            max_samples=4,
            eval_ratio=0.0,
            max_per_prompt=3,
            max_per_response=4,
            diversity_penalty_enabled=True,
            diversity_similarity_threshold=0.60,
            diversity_penalty_strength=0.35,
            diversity_reference_cap=3,
            bucket_classifier_enabled=False,
            seed=13,
        )
        manifest = build_dataset(cfg)
        rows = _read_jsonl(train_out)
        responses = [str(x.get("response", "")).strip().lower() for x in rows]
        assert len(rows) == 3
        assert any("step one" in r for r in responses)
        assert any("rotate api keys" in r for r in responses)
        assert int(manifest.get("near_duplicate_samples", 0)) >= 1
        assert float(manifest.get("avg_diversity_penalty", 0.0)) > 0.0


def test_dataset_diversity_penalty_can_be_disabled() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        patterns = root / "dialogue_patterns.json"
        train_out = root / "adapter_train.jsonl"
        eval_out = root / "adapter_eval.jsonl"
        manifest_out = root / "manifest.json"
        _build_patterns(patterns)

        cfg = DialogueDatasetBuildConfig(
            patterns_path=patterns.as_posix(),
            chat_memory_path=(root / "missing_chat.jsonl").as_posix(),
            reflections_path=(root / "missing_reflections.jsonl").as_posix(),
            output_train_path=train_out.as_posix(),
            output_eval_path=eval_out.as_posix(),
            output_manifest_path=manifest_out.as_posix(),
            include_patterns=True,
            include_chat_memory=False,
            include_reflections=False,
            max_samples=4,
            eval_ratio=0.0,
            max_per_prompt=3,
            max_per_response=4,
            diversity_penalty_enabled=False,
            bucket_classifier_enabled=False,
            seed=13,
        )
        manifest = build_dataset(cfg)
        rows = _read_jsonl(train_out)
        responses = [str(x.get("response", "")).strip().lower() for x in rows]
        assert len(rows) == 3
        assert any("step one" in r for r in responses)
        assert any("step two" in r for r in responses) or any("step three" in r for r in responses)
        assert all("rotate api keys" not in r for r in responses)
        assert int(manifest.get("near_duplicate_samples", 0)) == 0
        assert float(manifest.get("avg_diversity_penalty", 0.0)) <= 0.0


def main() -> None:
    test_dataset_diversity_penalty_prefers_varied_responses()
    test_dataset_diversity_penalty_can_be_disabled()
    print("dialogue_dataset_diversity_ok")


if __name__ == "__main__":
    main()
