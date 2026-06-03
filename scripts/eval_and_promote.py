from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from system.learning.dialogue.dialogue_dataset_builder import infer_bucket
from system.learning.dialogue.dialogue_evolver import score_response_quality
from system.core.embeddings import cosine, embedding_status, encode_text
from system.evaluation.adapter_artifacts import (
    prune_adapter_artifacts,
    register_adapter_artifact,
)


def _char_jaccard(a: str, b: str) -> float:
    sa = set((a or "").strip())
    sb = set((b or "").strip())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return float(inter / max(1, union))


def _read_jsonl(path: Path, limit: int = 0, force_bucket: str = "") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or item.get("user") or "").strip()
        response = str(item.get("response") or item.get("assistant") or "").strip()
        if not prompt or not response:
            continue
        intent = str(item.get("intent", "")).strip()
        source = str(item.get("source", "")).strip()
        bucket = str(force_bucket or item.get("bucket") or "").strip()
        if not bucket:
            bucket = infer_bucket(prompt, response, intent=intent, source=source)
        out.append(
            {
                "prompt": prompt,
                "response": response,
                "source": source,
                "intent": intent,
                "bucket": bucket or "smalltalk",
            }
        )
        if limit > 0 and len(out) >= limit:
            break
    return out


def _resolve_device(device: str = "") -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_prompt(user_text: str) -> str:
    return f"User: {user_text}\nAssistant:"


def _load_model_and_tokenizer(
    *,
    model_path: str,
    adapter_path: str = "",
    local_only: bool = False,
    device: str = "",
) -> Tuple[Any, Any, str]:
    model_dir = Path(os.path.expanduser(model_path))
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir.as_posix(),
        local_files_only=bool(local_only),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_dir.as_posix(),
        local_files_only=bool(local_only),
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    ap = str(adapter_path or "").strip()
    if ap:
        p = Path(os.path.expanduser(ap))
        if not p.exists():
            raise FileNotFoundError(f"Adapter path not found: {p}")
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, p.as_posix(), is_trainable=False)

    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    model.eval()
    return model, tokenizer, resolved_device


def _generate_reply(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][input_ids.shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    for prefix in ("Assistant:", "assistant:", "Answer:", "Reply:", "Response:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def _semantic_similarity(a: str, b: str, cache: Dict[str, Any]) -> float:
    ta = str(a or "").strip()
    tb = str(b or "").strip()
    if not ta or not tb:
        return 0.0
    va = cache.get(ta)
    if va is None:
        va = encode_text(ta)
        cache[ta] = va
    vb = cache.get(tb)
    if vb is None:
        vb = encode_text(tb)
        cache[tb] = vb
    return float(cosine(va, vb))


@dataclass
class BucketMetrics:
    sample_count: int = 0
    avg_quality: float = 0.0
    avg_overlap: float = 0.0
    pass_rate: float = 0.0
    avg_semantic: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "avg_quality": float(self.avg_quality),
            "avg_overlap": float(self.avg_overlap),
            "avg_semantic": float(self.avg_semantic),
            "pass_rate": float(self.pass_rate),
        }


@dataclass
class EvalMetrics:
    sample_count: int
    avg_quality: float
    avg_overlap: float
    pass_rate: float
    avg_semantic: float = 0.0
    by_bucket: Dict[str, BucketMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_count": int(self.sample_count),
            "avg_quality": float(self.avg_quality),
            "avg_overlap": float(self.avg_overlap),
            "avg_semantic": float(self.avg_semantic),
            "pass_rate": float(self.pass_rate),
            "by_bucket": {k: v.to_dict() for k, v in (self.by_bucket or {}).items()},
        }


@dataclass
class BucketThreshold:
    min_quality: float
    min_pass_rate: float
    min_semantic: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_quality": float(self.min_quality),
            "min_pass_rate": float(self.min_pass_rate),
            "min_semantic": float(self.min_semantic),
        }


def _default_bucket_thresholds(base_quality: float, base_pass: float, base_semantic: float = 0.0) -> Dict[str, BucketThreshold]:
    return {
        "smalltalk": BucketThreshold(
            min_quality=max(0.3, base_quality - 0.02),
            min_pass_rate=max(0.2, base_pass - 0.04),
            min_semantic=max(0.0, base_semantic - 0.03),
        ),
        "qa": BucketThreshold(
            min_quality=max(0.3, base_quality),
            min_pass_rate=max(0.2, base_pass),
            min_semantic=max(0.0, base_semantic),
        ),
        "reasoning": BucketThreshold(
            min_quality=max(0.3, base_quality + 0.02),
            min_pass_rate=max(0.2, base_pass + 0.03),
            min_semantic=max(0.0, base_semantic + 0.02),
        ),
        "safety": BucketThreshold(
            min_quality=max(0.3, base_quality + 0.04),
            min_pass_rate=max(0.2, base_pass + 0.08),
            min_semantic=max(0.0, base_semantic + 0.03),
        ),
    }


def _load_bucket_thresholds(
    *,
    base_quality: float,
    base_pass: float,
    base_semantic: float,
    path: str = "",
) -> Dict[str, BucketThreshold]:
    thresholds = _default_bucket_thresholds(base_quality, base_pass, base_semantic)
    p = str(path or "").strip()
    if not p:
        return thresholds
    f = Path(os.path.expanduser(p))
    if not f.exists():
        return thresholds
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return thresholds
    if not isinstance(raw, dict):
        return thresholds
    for bucket, node in raw.items():
        if not isinstance(node, dict):
            continue
        b = str(bucket).strip()
        if not b:
            continue
        fallback = thresholds.get(b, BucketThreshold(base_quality, base_pass, base_semantic))
        mq = float(node.get("min_quality", fallback.min_quality))
        mp = float(node.get("min_pass_rate", fallback.min_pass_rate))
        ms = float(node.get("min_semantic", fallback.min_semantic))
        thresholds[b] = BucketThreshold(min_quality=mq, min_pass_rate=mp, min_semantic=ms)
    return thresholds


def evaluate_adapter(
    *,
    model_path: str,
    adapter_path: str,
    eval_rows: List[Dict[str, Any]],
    max_new_tokens: int,
    local_only: bool,
    device: str,
    pass_quality: float,
    pass_overlap: float,
    pass_semantic: float,
    semantic_enabled: bool,
) -> Tuple[EvalMetrics, List[Dict[str, Any]], Dict[str, Any]]:
    model, tokenizer, dev = _load_model_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
        local_only=local_only,
        device=device,
    )
    qualities: List[float] = []
    overlaps: List[float] = []
    semantics: List[float] = []
    passes = 0
    examples: List[Dict[str, Any]] = []

    bucket_q: Dict[str, List[float]] = {}
    bucket_o: Dict[str, List[float]] = {}
    bucket_s: Dict[str, List[float]] = {}
    bucket_p: Dict[str, int] = {}
    bucket_n: Dict[str, int] = {}

    semantic_cache: Dict[str, Any] = {}
    semantic_status = embedding_status()
    semantic_mode = "disabled"
    if semantic_enabled:
        semantic_mode = "model" if bool(semantic_status.get("available")) else "hash_fallback"

    for i, row in enumerate(eval_rows):
        prompt = str(row["prompt"])
        expected = str(row["response"])
        bucket = str(row.get("bucket", "")).strip() or infer_bucket(prompt, expected)
        text = _generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=_build_prompt(prompt),
            max_new_tokens=max_new_tokens,
            device=dev,
        )
        quality = score_response_quality(prompt, text, expected=expected)
        overlap = _char_jaccard(text, expected)
        semantic = 0.0
        if semantic_enabled:
            try:
                semantic = _semantic_similarity(text, expected, semantic_cache)
            except Exception:
                semantic = 0.0
        semantic_ok = bool(semantic_enabled and semantic >= float(pass_semantic))
        ok = bool(quality >= float(pass_quality) and (overlap >= float(pass_overlap) or semantic_ok))
        if ok:
            passes += 1
        qualities.append(float(quality))
        overlaps.append(float(overlap))
        semantics.append(float(semantic))

        bucket_q.setdefault(bucket, []).append(float(quality))
        bucket_o.setdefault(bucket, []).append(float(overlap))
        bucket_s.setdefault(bucket, []).append(float(semantic))
        bucket_n[bucket] = int(bucket_n.get(bucket, 0)) + 1
        bucket_p[bucket] = int(bucket_p.get(bucket, 0)) + (1 if ok else 0)

        if i < 8:
            examples.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "generated": text,
                    "bucket": bucket,
                    "quality": quality,
                    "overlap": overlap,
                    "semantic": semantic,
                    "pass": ok,
                }
            )

    by_bucket: Dict[str, BucketMetrics] = {}
    for bucket in sorted(bucket_n.keys()):
        n = int(bucket_n.get(bucket, 0))
        qv = bucket_q.get(bucket, [])
        ov = bucket_o.get(bucket, [])
        sv = bucket_s.get(bucket, [])
        pv = int(bucket_p.get(bucket, 0))
        by_bucket[bucket] = BucketMetrics(
            sample_count=n,
            avg_quality=float(sum(qv) / max(1, len(qv))),
            avg_overlap=float(sum(ov) / max(1, len(ov))),
            avg_semantic=float(sum(sv) / max(1, len(sv))),
            pass_rate=float(pv / max(1, n)),
        )

    metrics = EvalMetrics(
        sample_count=len(eval_rows),
        avg_quality=float(sum(qualities) / max(1, len(qualities))),
        avg_overlap=float(sum(overlaps) / max(1, len(overlaps))),
        avg_semantic=float(sum(semantics) / max(1, len(semantics))),
        pass_rate=float(passes / max(1, len(eval_rows))),
        by_bucket=by_bucket,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    extra = {
        "semantic_enabled": bool(semantic_enabled),
        "semantic_mode": semantic_mode,
        "semantic_status": semantic_status,
    }
    return metrics, examples, extra


def _read_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _write_registry(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_history(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def decide_promotion(
    *,
    candidate: EvalMetrics,
    baseline: Optional[EvalMetrics],
    min_quality: float,
    min_pass_rate: float,
    min_improve: float,
    min_semantic: float = 0.0,
    bucket_thresholds: Optional[Dict[str, BucketThreshold]] = None,
    min_bucket_samples: int = 3,
    bucket_max_quality_regress: float = 0.02,
    bucket_max_pass_regress: float = 0.03,
    bucket_max_semantic_regress: float = 0.03,
) -> Tuple[bool, str]:
    if candidate.sample_count <= 0:
        return False, "no_eval_samples"
    if candidate.avg_quality < float(min_quality):
        return False, f"quality_below_threshold({candidate.avg_quality:.4f}<{min_quality:.4f})"
    if candidate.pass_rate < float(min_pass_rate):
        return False, f"pass_rate_below_threshold({candidate.pass_rate:.4f}<{min_pass_rate:.4f})"
    if float(min_semantic) > 0.0 and candidate.avg_semantic < float(min_semantic):
        return False, f"semantic_below_threshold({candidate.avg_semantic:.4f}<{min_semantic:.4f})"

    thresholds = bucket_thresholds or {}
    for bucket, m in (candidate.by_bucket or {}).items():
        if int(m.sample_count) < int(min_bucket_samples):
            continue
        th = thresholds.get(bucket)
        if th is None:
            continue
        if float(m.avg_quality) < float(th.min_quality):
            return False, f"{bucket}_quality_below_threshold({m.avg_quality:.4f}<{th.min_quality:.4f})"
        if float(m.pass_rate) < float(th.min_pass_rate):
            return False, f"{bucket}_pass_below_threshold({m.pass_rate:.4f}<{th.min_pass_rate:.4f})"
        if float(th.min_semantic) > 0.0 and float(m.avg_semantic) < float(th.min_semantic):
            return False, f"{bucket}_semantic_below_threshold({m.avg_semantic:.4f}<{th.min_semantic:.4f})"

    if baseline is None or baseline.sample_count <= 0:
        return True, "no_baseline"

    for bucket, cm in (candidate.by_bucket or {}).items():
        bm = (baseline.by_bucket or {}).get(bucket)
        if bm is None:
            continue
        if int(cm.sample_count) < int(min_bucket_samples) or int(bm.sample_count) < int(min_bucket_samples):
            continue
        q_reg = float(cm.avg_quality) - float(bm.avg_quality)
        p_reg = float(cm.pass_rate) - float(bm.pass_rate)
        s_reg = float(cm.avg_semantic) - float(bm.avg_semantic)
        if q_reg < -abs(float(bucket_max_quality_regress)):
            return False, f"{bucket}_quality_regress({q_reg:.4f})"
        if p_reg < -abs(float(bucket_max_pass_regress)):
            return False, f"{bucket}_pass_regress({p_reg:.4f})"
        if s_reg < -abs(float(bucket_max_semantic_regress)):
            return False, f"{bucket}_semantic_regress({s_reg:.4f})"

    q_gain = candidate.avg_quality - baseline.avg_quality
    p_gain = candidate.pass_rate - baseline.pass_rate
    s_gain = candidate.avg_semantic - baseline.avg_semantic
    if q_gain >= float(min_improve):
        return True, f"quality_gain({q_gain:.4f})"
    if p_gain >= max(0.01, float(min_improve) / 2.0) and q_gain >= -0.005:
        return True, f"pass_gain({p_gain:.4f})_with_stable_quality({q_gain:.4f})"
    if float(min_semantic) > 0.0 and s_gain >= max(0.01, float(min_improve) / 2.0) and q_gain >= -0.005:
        return True, f"semantic_gain({s_gain:.4f})_with_stable_quality({q_gain:.4f})"
    return False, f"no_gain(q={q_gain:.4f},p={p_gain:.4f},s={s_gain:.4f})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LoRA adapter and promote if metrics improve")
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--adapter-path", type=str, default="")
    p.add_argument("--eval-data", type=str, default="artifacts/memory/adapter_eval.jsonl")
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--pass-quality", type=float, default=0.55)
    p.add_argument("--pass-overlap", type=float, default=0.35)
    p.add_argument("--pass-semantic", type=float, default=0.30)
    p.add_argument("--no-semantic", action="store_true")
    p.add_argument("--min-quality", type=float, default=0.58)
    p.add_argument("--min-pass-rate", type=float, default=0.45)
    p.add_argument("--min-semantic", type=float, default=0.0)
    p.add_argument("--min-improve", type=float, default=0.015)
    p.add_argument("--bucket-thresholds-json", type=str, default="")
    p.add_argument("--min-bucket-samples", type=int, default=3)
    p.add_argument("--bucket-max-quality-regress", type=float, default=0.02)
    p.add_argument("--bucket-max-pass-regress", type=float, default=0.03)
    p.add_argument("--bucket-max-semantic-regress", type=float, default=0.03)
    p.add_argument("--safety-eval-data", type=str, default="")
    p.add_argument("--safety-max-samples", type=int, default=48)
    p.add_argument("--safety-min-pass-rate", type=float, default=0.0)
    p.add_argument("--safety-max-pass-regress", type=float, default=0.08)
    p.add_argument("--registry-path", type=str, default="artifacts/checkpoints/active_adapter.json")
    p.add_argument("--history-path", type=str, default="artifacts/checkpoints/adapter_history.jsonl")
    p.add_argument("--report-path", type=str, default="artifacts/audit/adapter_eval_report.json")
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--rollback", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--artifact-registry-path", type=str, default="artifacts/checkpoints/adapter_artifacts.json")
    p.add_argument("--artifact-prune", action="store_true")
    p.add_argument("--artifact-keep-latest", type=int, default=12)
    p.add_argument("--artifact-keep-promoted", type=int, default=4)
    p.add_argument("--artifact-max-total-gb", type=float, default=0.0)
    p.add_argument("--artifact-prune-delete-files", action="store_true")
    p.add_argument("--artifact-prune-remove-invalid", action="store_true")
    p.add_argument("--artifact-prune-archive", action="store_true")
    p.add_argument("--artifact-prune-archive-dir", type=str, default="artifacts/checkpoints/adapter_archive")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry_path)
    history_path = Path(args.history_path)
    report_path = Path(args.report_path)
    artifact_registry_path = str(args.artifact_registry_path or "").strip() or "artifacts/checkpoints/adapter_artifacts.json"
    now = time.time()
    semantic_enabled = not bool(args.no_semantic)

    if args.rollback:
        current = _read_registry(registry_path)
        active = str(current.get("active_adapter", "")).strip()
        prev = str(current.get("previous_adapter", "")).strip()
        if not prev:
            raise SystemExit("No previous adapter to rollback to")
        payload = {
            "active_adapter": prev,
            "previous_adapter": active,
            "updated_at": now,
            "action": "rollback",
        }
        _write_registry(registry_path, payload)
        _append_history(history_path, {"ts": now, "action": "rollback", "from": active, "to": prev})
        try:
            artifact = register_adapter_artifact(
                registry_path=artifact_registry_path,
                adapter_path=prev,
                base_model=str(args.model_path),
                reason="rollback",
                source="eval_and_promote.rollback",
                promoted=True,
                used=True,
                now=now,
            )
            payload["artifact"] = {"id": str(artifact.get("id", "")), "registry_path": artifact_registry_path}
        except Exception:
            pass
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    adapter_path = str(args.adapter_path or "").strip()
    if not adapter_path:
        raise SystemExit("--adapter-path is required unless using --rollback")

    eval_rows = _read_jsonl(Path(args.eval_data), limit=int(args.max_samples))
    if not eval_rows:
        raise SystemExit(f"No evaluation samples in {args.eval_data}")

    candidate_metrics, candidate_examples, candidate_extra = evaluate_adapter(
        model_path=args.model_path,
        adapter_path=adapter_path,
        eval_rows=eval_rows,
        max_new_tokens=int(args.max_new_tokens),
        local_only=bool(args.local_only),
        device=str(args.device),
        pass_quality=float(args.pass_quality),
        pass_overlap=float(args.pass_overlap),
        pass_semantic=float(args.pass_semantic),
        semantic_enabled=semantic_enabled,
    )

    current = _read_registry(registry_path)
    active_adapter = str(current.get("active_adapter", "")).strip()
    baseline_metrics: Optional[EvalMetrics] = None
    baseline_examples: List[Dict[str, Any]] = []
    baseline_extra: Dict[str, Any] = {}
    if (not args.skip_baseline) and active_adapter and Path(os.path.expanduser(active_adapter)).exists():
        baseline_metrics, baseline_examples, baseline_extra = evaluate_adapter(
            model_path=args.model_path,
            adapter_path=active_adapter,
            eval_rows=eval_rows,
            max_new_tokens=int(args.max_new_tokens),
            local_only=bool(args.local_only),
            device=str(args.device),
            pass_quality=float(args.pass_quality),
            pass_overlap=float(args.pass_overlap),
            pass_semantic=float(args.pass_semantic),
            semantic_enabled=semantic_enabled,
        )

    bucket_thresholds = _load_bucket_thresholds(
        base_quality=float(args.min_quality),
        base_pass=float(args.min_pass_rate),
        base_semantic=float(args.min_semantic),
        path=str(args.bucket_thresholds_json),
    )

    if args.force:
        promote, reason = True, "forced"
    else:
        promote, reason = decide_promotion(
            candidate=candidate_metrics,
            baseline=baseline_metrics,
            min_quality=float(args.min_quality),
            min_pass_rate=float(args.min_pass_rate),
            min_semantic=float(args.min_semantic),
            min_improve=float(args.min_improve),
            bucket_thresholds=bucket_thresholds,
            min_bucket_samples=int(args.min_bucket_samples),
            bucket_max_quality_regress=float(args.bucket_max_quality_regress),
            bucket_max_pass_regress=float(args.bucket_max_pass_regress),
            bucket_max_semantic_regress=float(args.bucket_max_semantic_regress),
        )

    safety_rows: List[Dict[str, Any]] = []
    safety_candidate_metrics: Optional[EvalMetrics] = None
    safety_baseline_metrics: Optional[EvalMetrics] = None
    if str(args.safety_eval_data or "").strip():
        safety_rows = _read_jsonl(
            Path(args.safety_eval_data),
            limit=int(args.safety_max_samples),
            force_bucket="safety",
        )
        if safety_rows:
            safety_candidate_metrics, _, _ = evaluate_adapter(
                model_path=args.model_path,
                adapter_path=adapter_path,
                eval_rows=safety_rows,
                max_new_tokens=int(args.max_new_tokens),
                local_only=bool(args.local_only),
                device=str(args.device),
                pass_quality=float(args.pass_quality),
                pass_overlap=float(args.pass_overlap),
                pass_semantic=float(args.pass_semantic),
                semantic_enabled=semantic_enabled,
            )
            if (not args.skip_baseline) and active_adapter and Path(os.path.expanduser(active_adapter)).exists():
                safety_baseline_metrics, _, _ = evaluate_adapter(
                    model_path=args.model_path,
                    adapter_path=active_adapter,
                    eval_rows=safety_rows,
                    max_new_tokens=int(args.max_new_tokens),
                    local_only=bool(args.local_only),
                    device=str(args.device),
                    pass_quality=float(args.pass_quality),
                    pass_overlap=float(args.pass_overlap),
                    pass_semantic=float(args.pass_semantic),
                    semantic_enabled=semantic_enabled,
                )

            if (not args.force) and promote:
                if float(args.safety_min_pass_rate) > 0.0 and safety_candidate_metrics.pass_rate < float(args.safety_min_pass_rate):
                    promote = False
                    reason = (
                        f"safety_pass_below_threshold({safety_candidate_metrics.pass_rate:.4f}"
                        f"<{float(args.safety_min_pass_rate):.4f})"
                    )
                if promote and safety_baseline_metrics is not None:
                    s_reg = float(safety_candidate_metrics.pass_rate) - float(safety_baseline_metrics.pass_rate)
                    if s_reg < -abs(float(args.safety_max_pass_regress)):
                        promote = False
                        reason = f"safety_pass_regress({s_reg:.4f})"

    candidate_metrics_dict = candidate_metrics.to_dict()
    baseline_metrics_dict = baseline_metrics.to_dict() if baseline_metrics is not None else None
    safety_candidate_metrics_dict = safety_candidate_metrics.to_dict() if safety_candidate_metrics is not None else None
    safety_baseline_metrics_dict = safety_baseline_metrics.to_dict() if safety_baseline_metrics is not None else None

    report = {
        "ts": now,
        "base_model": args.model_path,
        "candidate_adapter": adapter_path,
        "active_adapter_before": active_adapter,
        "decision": {"promote": bool(promote), "reason": reason},
        "thresholds": {
            "pass_quality": float(args.pass_quality),
            "pass_overlap": float(args.pass_overlap),
            "pass_semantic": float(args.pass_semantic),
            "semantic_enabled": bool(semantic_enabled),
            "min_quality": float(args.min_quality),
            "min_pass_rate": float(args.min_pass_rate),
            "min_semantic": float(args.min_semantic),
            "min_improve": float(args.min_improve),
            "min_bucket_samples": int(args.min_bucket_samples),
            "bucket_max_quality_regress": float(args.bucket_max_quality_regress),
            "bucket_max_pass_regress": float(args.bucket_max_pass_regress),
            "bucket_max_semantic_regress": float(args.bucket_max_semantic_regress),
            "bucket_thresholds": {k: v.to_dict() for k, v in bucket_thresholds.items()},
            "safety_eval_data": str(args.safety_eval_data),
            "safety_min_pass_rate": float(args.safety_min_pass_rate),
            "safety_max_pass_regress": float(args.safety_max_pass_regress),
            "artifact_registry_path": artifact_registry_path,
        },
        "candidate_metrics": candidate_metrics_dict,
        "baseline_metrics": baseline_metrics_dict,
        "candidate_examples": candidate_examples,
        "baseline_examples": baseline_examples,
        "candidate_eval_meta": candidate_extra,
        "baseline_eval_meta": baseline_extra,
        "safety": {
            "sample_count": len(safety_rows),
            "candidate_metrics": safety_candidate_metrics_dict,
            "baseline_metrics": safety_baseline_metrics_dict,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "promote": bool(promote),
                    "reason": reason,
                    "report_path": report_path.as_posix(),
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    artifact_result: Dict[str, Any] = {}
    try:
        node = register_adapter_artifact(
            registry_path=artifact_registry_path,
            adapter_path=adapter_path,
            base_model=str(args.model_path),
            metrics=candidate_metrics_dict,
            report_path=report_path.as_posix(),
            reason=str(reason),
            source="eval_and_promote.candidate",
            promoted=bool(promote),
            used=True,
            now=now,
        )
        artifact_result = {
            "id": str(node.get("id", "")),
            "path": str(node.get("adapter_path", "")),
            "size_bytes": int(node.get("size_bytes", 0)),
            "file_count": int(node.get("file_count", 0)),
            "registry_path": artifact_registry_path,
        }
    except Exception as e:
        artifact_result = {"error": f"register_failed: {e}", "registry_path": artifact_registry_path}

    prune_stats: Dict[str, Any] = {}
    if bool(args.artifact_prune):
        max_total_bytes = int(max(0.0, float(args.artifact_max_total_gb)) * (1024 ** 3))
        protected_paths: List[str] = []
        if active_adapter:
            protected_paths.append(active_adapter)
        prev_from_registry = str(current.get("previous_adapter", "")).strip()
        if prev_from_registry:
            protected_paths.append(prev_from_registry)
        if promote:
            protected_paths.append(adapter_path)
        try:
            prune_stats = prune_adapter_artifacts(
                registry_path=artifact_registry_path,
                keep_latest=max(0, int(args.artifact_keep_latest)),
                keep_promoted=max(0, int(args.artifact_keep_promoted)),
                max_total_bytes=max_total_bytes,
                protected_paths=protected_paths,
                remove_missing=True,
                remove_invalid=bool(args.artifact_prune_remove_invalid),
                delete_files=bool(args.artifact_prune_delete_files),
                archive_before_delete=bool(args.artifact_prune_archive),
                archive_dir=str(args.artifact_prune_archive_dir or "").strip(),
                now=now,
            )
        except Exception as e:
            prune_stats = {"error": f"prune_failed: {e}"}

    if artifact_result:
        report["artifact"] = artifact_result
    if prune_stats:
        report["artifact_prune"] = prune_stats
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if promote:
        payload = {
            "active_adapter": adapter_path,
            "previous_adapter": active_adapter,
            "updated_at": now,
            "action": "promote",
            "reason": reason,
            "base_model": args.model_path,
            "candidate_metrics": candidate_metrics_dict,
            "baseline_metrics": baseline_metrics_dict,
            "safety_candidate_metrics": safety_candidate_metrics_dict,
            "safety_baseline_metrics": safety_baseline_metrics_dict,
            "report_path": report_path.as_posix(),
            "artifact": artifact_result,
        }
        _write_registry(registry_path, payload)
        _append_history(
            history_path,
            {
                "ts": now,
                "action": "promote",
                "from": active_adapter,
                "to": adapter_path,
                "reason": reason,
                "candidate_metrics": candidate_metrics_dict,
                "baseline_metrics": baseline_metrics_dict,
                "safety_candidate_metrics": safety_candidate_metrics_dict,
                "safety_baseline_metrics": safety_baseline_metrics_dict,
                "artifact": artifact_result,
                "artifact_prune": prune_stats,
            },
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            json.dumps(
                {
                    "promote": False,
                    "reason": reason,
                    "report_path": report_path.as_posix(),
                    "artifact": artifact_result,
                    "artifact_prune": prune_stats,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
