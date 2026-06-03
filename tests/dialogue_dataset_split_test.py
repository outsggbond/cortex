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


def test_dataset_split_prevents_prompt_leakage() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        patterns = root / "dialogue_patterns.json"
        train_out = root / "adapter_train.jsonl"
        eval_out = root / "adapter_eval.jsonl"
        manifest_out = root / "manifest.json"

        pairs = []
        for i in range(5):
            pairs.append(["How to backup database safely?", f"Use snapshot plan step {i + 1}."])
        for i in range(5):
            pairs.append(["How to rollback safely?", f"Use rollback checklist step {i + 1}."])
        patterns.write_text(json.dumps({"qa": {"pairs": pairs}}, ensure_ascii=False), encoding="utf-8")

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
            eval_ratio=0.40,
            max_samples=10,
            max_per_prompt=10,
            max_per_response=10,
            bucket_classifier_enabled=False,
            prevent_cross_split_prompt_leakage=True,
            seed=11,
        )
        manifest = build_dataset(cfg)
        train_rows = _read_jsonl(train_out)
        eval_rows = _read_jsonl(eval_out)
        train_prompts = {str(x.get("prompt", "")).strip() for x in train_rows if str(x.get("prompt", "")).strip()}
        eval_prompts = {str(x.get("prompt", "")).strip() for x in eval_rows if str(x.get("prompt", "")).strip()}
        assert train_rows
        assert eval_rows
        assert len(train_prompts & eval_prompts) == 0
        assert int(manifest.get("cross_split_prompt_overlap_count", 1)) == 0
        assert float(manifest.get("cross_split_prompt_overlap_ratio", 1.0)) <= 0.0


def test_dataset_split_can_allow_prompt_overlap() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        patterns = root / "dialogue_patterns.json"
        train_out = root / "adapter_train.jsonl"
        eval_out = root / "adapter_eval.jsonl"
        manifest_out = root / "manifest.json"

        pairs = []
        for i in range(8):
            pairs.append(["How to backup database safely?", f"Use snapshot plan variant {i + 1}."])
        patterns.write_text(json.dumps({"qa": {"pairs": pairs}}, ensure_ascii=False), encoding="utf-8")

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
            eval_ratio=0.50,
            max_samples=8,
            max_per_prompt=10,
            max_per_response=10,
            bucket_classifier_enabled=False,
            prevent_cross_split_prompt_leakage=False,
            seed=7,
        )
        manifest = build_dataset(cfg)
        assert int(manifest.get("train_samples", 0)) >= 1
        assert int(manifest.get("eval_samples", 0)) >= 1
        assert int(manifest.get("cross_split_prompt_overlap_count", 0)) >= 1
        assert float(manifest.get("cross_split_prompt_overlap_ratio", 0.0)) > 0.0


def main() -> None:
    test_dataset_split_prevents_prompt_leakage()
    test_dataset_split_can_allow_prompt_overlap()
    print("dialogue_dataset_split_ok")


if __name__ == "__main__":
    main()
