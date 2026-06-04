#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean training data files and run leakage / quality gates.

Supports:
  - JSONL/text file cleaning via `clean_training_file`
  - Cross-file exact-text leakage detection (prompt & prompt+response overlap)
  - Semantic near-duplicate leakage detection (embedding cosine similarity)
  - Quality gates (min quality score, min keep ratio)
  - Overlap gates (exact and semantic, global and train/eval)
  - --fail-on-gate (exit code 2)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from system.knowledge.data_etl import DataCleanConfig, clean_training_file


# ---------------------------------------------------------------------------
# helpers for reading cleaned outputs back
# ---------------------------------------------------------------------------

def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    text = path.read_bytes()
    raw_text: str
    try:
        raw_text = text.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = text.decode("utf-8", errors="replace")
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            node = json.loads(line)
            if isinstance(node, dict):
                rows.append(node)
        except Exception:
            continue
    return rows


def _extract_prompt(row: Dict[str, Any]) -> str:
    for k in ("prompt", "query", "instruction", "input", "user", "question"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _extract_pair(row: Dict[str, Any]) -> str:
    prompt = _extract_prompt(row)
    response = ""
    for k in ("response", "answer", "output", "assistant", "target"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            response = v.strip()
            break
    return f"{prompt}\n{response}"


# ---------------------------------------------------------------------------
# exact-text cross-file leakage
# ---------------------------------------------------------------------------

def _compute_cross_leakage(
    file_paths: Sequence[Path],
) -> Dict[str, Any]:
    """Compute exact-text overlap ratios across cleaned files."""
    n = len(file_paths)
    file_prompts: List[List[str]] = []
    file_pairs: List[List[str]] = []

    for fp in file_paths:
        rows = _load_jsonl_rows(fp)
        prompts = [_extract_prompt(r) for r in rows]
        prompts = [p for p in prompts if p]
        pair_texts = [_extract_pair(r) for r in rows]
        pair_texts = [p for p in pair_texts if p.strip()]
        file_prompts.append(prompts)
        file_pairs.append(pair_texts)

    per_file: List[Dict[str, Any]] = []
    pair_overlap_matrix: List[List[float]] = []
    prompt_overlap_matrix: List[List[float]] = []
    train_eval_pair_max = 0.0
    train_eval_prompt_max = 0.0

    for i in range(n):
        pi_set: Set[str] = set(file_prompts[i])
        pa_set: Set[str] = set(file_pairs[i])
        per_file.append({
            "path": file_paths[i].as_posix(),
            "prompts": len(pi_set),
            "pairs": len(pa_set),
        })
        pair_row: List[float] = []
        prompt_row: List[float] = []
        for j in range(n):
            if i == j:
                pair_row.append(0.0)
                prompt_row.append(0.0)
            else:
                pj_set: Set[str] = set(file_prompts[j])
                pj_pa_set: Set[str] = set(file_pairs[j])
                # pair overlap
                inter_pa = float(len(pa_set & pj_pa_set))
                denom_pa = max(1.0, float(min(len(pa_set), len(pj_pa_set))))
                pair_overlap = inter_pa / denom_pa
                pair_row.append(pair_overlap)
                # prompt overlap
                inter_pr = float(len(pi_set & pj_set))
                denom_pr = max(1.0, float(min(len(pi_set), len(pj_set))))
                prompt_overlap = inter_pr / denom_pr
                prompt_row.append(prompt_overlap)
        pair_overlap_matrix.append(pair_row)
        prompt_overlap_matrix.append(prompt_row)

    # train/eval: first two files
    train_eval_pair_overlap_ratio = 0.0
    train_eval_prompt_overlap_ratio = 0.0
    if n >= 2:
        train_eval_pair_overlap_ratio = pair_overlap_matrix[0][1]
        train_eval_prompt_overlap_ratio = prompt_overlap_matrix[0][1]

    # global maxes
    max_cross_pair = 0.0
    max_cross_prompt = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                max_cross_pair = max(max_cross_pair, pair_overlap_matrix[i][j])
                max_cross_prompt = max(max_cross_prompt, prompt_overlap_matrix[i][j])

    return {
        "files": per_file,
        "pair_overlap_ratios": pair_overlap_matrix,
        "prompt_overlap_ratios": prompt_overlap_matrix,
        "summary": {
            "train_eval_pair_overlap_ratio_max": train_eval_pair_overlap_ratio,
            "train_eval_prompt_overlap_ratio_max": train_eval_prompt_overlap_ratio,
            "cross_pair_overlap_ratio_max": max_cross_pair,
            "cross_prompt_overlap_ratio_max": max_cross_prompt,
        },
    }


# ---------------------------------------------------------------------------
# semantic near-duplicate leakage (with optional sentence-transformers)
# ---------------------------------------------------------------------------

def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = max(1e-12, sum(x * x for x in a) ** 0.5)
    nb = max(1e-12, sum(x * x for x in b) ** 0.5)
    return float(dot / (na * nb))


def _embed_texts(
    texts: List[str],
    *,
    model: Any,
    batch_size: int = 64,
) -> Optional[List[List[float]]]:
    if model is None:
        return None
    try:
        if hasattr(model, "encode"):
            out = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
            if hasattr(out, "tolist"):
                return out.tolist()
            return [list(map(float, v)) for v in out]
        return None
    except Exception:
        return None


def _try_semantic_model():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def _compute_semantic_leakage(
    file_paths: Sequence[Path],
    *,
    threshold: float = 0.70,
    max_items_per_file: int = 400,
    max_pair_comparisons: int = 200000,
) -> Dict[str, Any]:
    """Compute semantic overlap ratios across cleaned files.

    Uses sentence-transformers if available; otherwise uses a simple
    token-Jaccard fallback with adjusted threshold.
    """
    sem_model = _try_semantic_model()
    n = len(file_paths)
    file_prompts: List[List[str]] = []
    file_pairs: List[List[str]] = []
    sampling_occurred = False

    for fp in file_paths:
        rows = _load_jsonl_rows(fp)
        prompts = [_extract_prompt(r) for r in rows]
        prompts = [p for p in prompts if p]
        pair_texts = [_extract_pair(r) for r in rows]
        pair_texts = [p for p in pair_texts if p.strip()]

        if len(prompts) > max_items_per_file:
            sampling_occurred = True
            step = max(1, len(prompts) // max_items_per_file)
            prompts = prompts[::step][:max_items_per_file]
        if len(pair_texts) > max_items_per_file:
            pair_texts = pair_texts[:: max(1, len(pair_texts) // max_items_per_file)][
                :max_items_per_file
            ]

        file_prompts.append(prompts)
        file_pairs.append(pair_texts)

    prompt_embeddings: List[Optional[List[List[float]]]] = [None] * n
    pair_embeddings: List[Optional[List[List[float]]]] = [None] * n

    if sem_model is not None:
        for i, prompts in enumerate(file_prompts):
            prompt_embeddings[i] = _embed_texts(prompts, model=sem_model)
        for i, pairs in enumerate(file_pairs):
            pair_embeddings[i] = _embed_texts(pairs, model=sem_model)

    prompt_ratios: List[List[float]] = [[0.0] * n for _ in range(n)]
    pair_ratios: List[List[float]] = [[0.0] * n for _ in range(n)]
    total_comparisons = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if total_comparisons >= max_pair_comparisons:
                break

            # semantic prompt overlap
            if prompt_embeddings[i] is not None and prompt_embeddings[j] is not None:
                matches = 0
                denom = max(1, min(len(prompt_embeddings[i]), len(prompt_embeddings[j])))
                comp_count = 0
                for ei in prompt_embeddings[i]:
                    for ej in prompt_embeddings[j]:
                        comp_count += 1
                        total_comparisons += 1
                        if _cosine_sim(ei, ej) >= threshold:
                            matches += 1
                            break
                    if total_comparisons >= max_pair_comparisons:
                        break
                prompt_ratios[i][j] = float(matches) / float(denom)
            else:
                # Per-item Jaccard fallback: count items in file i that have a
                # token-Jaccard match in file j above the fallback threshold.
                fallback_threshold = max(0.30, float(threshold) * 0.55)
                denom = max(1, min(len(file_prompts[i]), len(file_prompts[j])))
                matches = 0
                for ti in file_prompts[i]:
                    ti_tokens = set(ti.lower().split())
                    if not ti_tokens:
                        continue
                    for tj in file_prompts[j]:
                        tj_tokens = set(tj.lower().split())
                        if not tj_tokens:
                            continue
                        inter = len(ti_tokens & tj_tokens)
                        union = max(1, len(ti_tokens | tj_tokens))
                        if float(inter) / float(union) >= fallback_threshold:
                            matches += 1
                            break
                prompt_ratios[i][j] = float(matches) / float(denom)

            # semantic pair overlap
            if pair_embeddings[i] is not None and pair_embeddings[j] is not None:
                matches = 0
                denom = max(1, min(len(pair_embeddings[i]), len(pair_embeddings[j])))
                for ei in pair_embeddings[i]:
                    for ej in pair_embeddings[j]:
                        total_comparisons += 1
                        if _cosine_sim(ei, ej) >= threshold:
                            matches += 1
                            break
                    if total_comparisons >= max_pair_comparisons:
                        break
                pair_ratios[i][j] = float(matches) / float(denom)
            else:
                fallback_threshold = max(0.30, float(threshold) * 0.55)
                denom = max(1, min(len(file_pairs[i]), len(file_pairs[j])))
                matches = 0
                for ti in file_pairs[i]:
                    ti_tokens = set(ti.lower().split())
                    if not ti_tokens:
                        continue
                    for tj in file_pairs[j]:
                        tj_tokens = set(tj.lower().split())
                        if not tj_tokens:
                            continue
                        inter = len(ti_tokens & tj_tokens)
                        union = max(1, len(ti_tokens | tj_tokens))
                        if float(inter) / float(union) >= fallback_threshold:
                            matches += 1
                            break
                pair_ratios[i][j] = float(matches) / float(denom)

        if total_comparisons >= max_pair_comparisons:
            break

    train_eval_prompt_max = 0.0
    train_eval_pair_max = 0.0
    global_prompt_max = 0.0
    global_pair_max = 0.0
    if n >= 2:
        train_eval_prompt_max = prompt_ratios[0][1]
        train_eval_pair_max = pair_ratios[0][1]
    for i in range(n):
        for j in range(n):
            if i != j:
                global_prompt_max = max(global_prompt_max, prompt_ratios[i][j])
                global_pair_max = max(global_pair_max, pair_ratios[i][j])

    return {
        "model_available": bool(sem_model is not None),
        "sampling_occurred": sampling_occurred,
        "threshold": threshold,
        "pair_comparisons_used": total_comparisons,
        "prompt_overlap_ratios": prompt_ratios,
        "pair_overlap_ratios": pair_ratios,
        "summary": {
            "train_eval_semantic_prompt_overlap_ratio_max": train_eval_prompt_max,
            "train_eval_semantic_pair_overlap_ratio_max": train_eval_pair_max,
            "semantic_prompt_overlap_ratio_max": global_prompt_max,
            "semantic_pair_overlap_ratio_max": global_pair_max,
        },
    }


# ---------------------------------------------------------------------------
# gate evaluation
# ---------------------------------------------------------------------------

def _evaluate_gates(
    *,
    file_reports: List[Dict[str, Any]],
    cross_leakage: Optional[Dict[str, Any]],
    semantic_leakage: Optional[Dict[str, Any]],
    min_quality_score: Optional[float],
    min_keep_ratio: Optional[float],
    max_cross_pair_overlap_ratio: Optional[float],
    max_cross_prompt_overlap_ratio: Optional[float],
    max_train_eval_pair_overlap_ratio: Optional[float],
    max_train_eval_prompt_overlap_ratio: Optional[float],
    max_semantic_prompt_overlap_ratio: Optional[float],
    max_semantic_pair_overlap_ratio: Optional[float],
    max_train_eval_semantic_prompt_overlap_ratio: Optional[float],
    max_train_eval_semantic_pair_overlap_ratio: Optional[float],
) -> Dict[str, Any]:
    """Evaluate all gates and return {passed, reasons, ...}."""
    passed = True
    reasons: List[str] = []
    details: Dict[str, Any] = {}

    # quality / keep-ratio gates (aggregated across all files)
    if min_quality_score is not None or min_keep_ratio is not None:
        scores = [r.get("quality_score", 0.0) for r in file_reports]
        keeps = [r.get("keep_ratio", 0.0) for r in file_reports]
        avg_quality = float(sum(scores) / max(1, len(scores)))
        avg_keep = float(sum(keeps) / max(1, len(keeps)))
        details["avg_quality_score"] = avg_quality
        details["avg_keep_ratio"] = avg_keep
        if min_quality_score is not None and avg_quality < float(min_quality_score):
            passed = False
            reasons.append("avg_quality_below_min")
        if min_keep_ratio is not None and avg_keep < float(min_keep_ratio):
            passed = False
            reasons.append("avg_keep_ratio_below_min")

    # cross leakage gates
    if cross_leakage is not None:
        s = cross_leakage.get("summary", {})
        details.update(
            {f"cross_{k}": v for k, v in s.items() if isinstance(v, (int, float))}
        )
        gates_cross = [
            ("cross_pair_overlap_above_threshold", s.get("cross_pair_overlap_ratio_max", 0.0), max_cross_pair_overlap_ratio),
            ("cross_prompt_overlap_above_threshold", s.get("cross_prompt_overlap_ratio_max", 0.0), max_cross_prompt_overlap_ratio),
            ("train_eval_pair_overlap_above_threshold", s.get("train_eval_pair_overlap_ratio_max", 0.0), max_train_eval_pair_overlap_ratio),
            ("train_eval_prompt_overlap_above_threshold", s.get("train_eval_prompt_overlap_ratio_max", 0.0), max_train_eval_prompt_overlap_ratio),
        ]
        for reason_key, actual, limit in gates_cross:
            if limit is not None and float(actual) > float(limit):
                passed = False
                reasons.append(reason_key)

    # semantic leakage gates
    if semantic_leakage is not None:
        s = semantic_leakage.get("summary", {})
        details.update(
            {f"semantic_{k}": v for k, v in s.items() if isinstance(v, (int, float))}
        )
        gates_semantic = [
            ("semantic_prompt_overlap_above_threshold", s.get("semantic_prompt_overlap_ratio_max", 0.0), max_semantic_prompt_overlap_ratio),
            ("semantic_pair_overlap_above_threshold", s.get("semantic_pair_overlap_ratio_max", 0.0), max_semantic_pair_overlap_ratio),
            ("train_eval_semantic_prompt_overlap_above_threshold", s.get("train_eval_semantic_prompt_overlap_ratio_max", 0.0), max_train_eval_semantic_prompt_overlap_ratio),
            ("train_eval_semantic_pair_overlap_above_threshold", s.get("train_eval_semantic_pair_overlap_ratio_max", 0.0), max_train_eval_semantic_pair_overlap_ratio),
        ]
        for reason_key, actual, limit in gates_semantic:
            if limit is not None and float(actual) > float(limit):
                passed = False
                reasons.append(reason_key)

    return {"passed": passed, "reasons": reasons, "details": details}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _argparse() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean training data and run leakage/quality gates")
    p.add_argument("--paths", nargs="+", required=True, help="Input file paths to clean")
    p.add_argument("--out-dir", required=True, help="Output directory for cleaned files")
    p.add_argument("--report", required=True, help="Path to write JSON report")

    # quality gates
    p.add_argument("--min-quality-score", type=float, default=None)
    p.add_argument("--min-keep-ratio", type=float, default=None)

    # cross leakage
    p.add_argument("--check-cross-leakage", action="store_true")
    p.add_argument("--max-cross-pair-overlap-ratio", type=float, default=None)
    p.add_argument("--max-cross-prompt-overlap-ratio", type=float, default=None)
    p.add_argument("--max-train-eval-pair-overlap-ratio", type=float, default=None)
    p.add_argument("--max-train-eval-prompt-overlap-ratio", type=float, default=None)

    # semantic leakage
    p.add_argument("--check-semantic-leakage", action="store_true")
    p.add_argument("--semantic-overlap-threshold", type=float, default=0.70)
    p.add_argument("--semantic-max-items-per-file", type=int, default=400)
    p.add_argument("--semantic-max-pair-comparisons", type=int, default=200000)
    p.add_argument("--max-semantic-prompt-overlap-ratio", type=float, default=None)
    p.add_argument("--max-semantic-pair-overlap-ratio", type=float, default=None)
    p.add_argument("--max-train-eval-semantic-prompt-overlap-ratio", type=float, default=None)
    p.add_argument("--max-train-eval-semantic-pair-overlap-ratio", type=float, default=None)

    p.add_argument("--fail-on-gate", action="store_true")
    return p


def main() -> None:
    args = _argparse().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DataCleanConfig()
    file_reports: List[Dict[str, Any]] = []
    cleaned_paths: List[Path] = []

    for input_path_str in args.paths:
        src = Path(input_path_str)
        stem = src.stem or "cleaned"
        out_path = out_dir / f"{stem}_clean{src.suffix or '.jsonl'}"
        report = clean_training_file(src, out_path, cfg=cfg)
        file_reports.append(report)
        cleaned_paths.append(out_path)

    # collect aggregate stats
    scores = [r.get("quality_score", 0.0) for r in file_reports]
    keeps = [r.get("keep_ratio", 0.0) for r in file_reports]
    avg_quality = float(sum(scores) / max(1, len(scores)))
    avg_keep = float(sum(keeps) / max(1, len(keeps)))

    cross_leakage: Optional[Dict[str, Any]] = None
    semantic_leakage: Optional[Dict[str, Any]] = None

    if args.check_cross_leakage:
        cross_leakage = _compute_cross_leakage(cleaned_paths)

    if args.check_semantic_leakage:
        semantic_leakage = _compute_semantic_leakage(
            cleaned_paths,
            threshold=float(args.semantic_overlap_threshold),
            max_items_per_file=int(args.semantic_max_items_per_file),
            max_pair_comparisons=int(args.semantic_max_pair_comparisons),
        )

    gate = _evaluate_gates(
        file_reports=file_reports,
        cross_leakage=cross_leakage,
        semantic_leakage=semantic_leakage,
        min_quality_score=args.min_quality_score,
        min_keep_ratio=args.min_keep_ratio,
        max_cross_pair_overlap_ratio=args.max_cross_pair_overlap_ratio,
        max_cross_prompt_overlap_ratio=args.max_cross_prompt_overlap_ratio,
        max_train_eval_pair_overlap_ratio=args.max_train_eval_pair_overlap_ratio,
        max_train_eval_prompt_overlap_ratio=args.max_train_eval_prompt_overlap_ratio,
        max_semantic_prompt_overlap_ratio=args.max_semantic_prompt_overlap_ratio,
        max_semantic_pair_overlap_ratio=args.max_semantic_pair_overlap_ratio,
        max_train_eval_semantic_prompt_overlap_ratio=args.max_train_eval_semantic_prompt_overlap_ratio,
        max_train_eval_semantic_pair_overlap_ratio=args.max_train_eval_semantic_pair_overlap_ratio,
    )

    report: Dict[str, Any] = {
        "files": file_reports,
        "avg_quality_score": avg_quality,
        "avg_keep_ratio": avg_keep,
        "gate": gate,
    }
    # Merge all leakage data under "cross_leakage" (tests always read from this key)
    combined_leakage: Dict[str, Any] = {}
    if cross_leakage is not None:
        combined_leakage.update(cross_leakage)
    if semantic_leakage is not None:
        # Merge semantic summary into cross_leakage.summary
        if "summary" in combined_leakage:
            combined_leakage["summary"].update(semantic_leakage.get("summary", {}))
        else:
            combined_leakage["summary"] = dict(semantic_leakage.get("summary", {}))
        # Also keep semantic-specific fields
        combined_leakage["semantic"] = semantic_leakage
    if combined_leakage:
        report["cross_leakage"] = combined_leakage

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.fail_on_gate and not bool(gate.get("passed", True)):
        print(f"Gate FAILED: {gate.get('reasons', [])}", file=sys.stderr)
        sys.exit(2)
    else:
        print(f"Gate {'PASSED' if gate.get('passed', True) else 'FAILED (no --fail-on-gate)'}")


if __name__ == "__main__":
    main()
