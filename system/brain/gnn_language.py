from __future__ import annotations

import contextlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - runtime optional
    torch = None  # type: ignore[assignment]
    nn = None
    F = None


# ---------------------------------------------------------------------------
# Stub base class so the module and its importers are importable without torch
# ---------------------------------------------------------------------------
class _StubModule:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass


class _StubTensor:
    pass


if nn is None:
    nn = type(  # type: ignore[assignment]
        "nn",
        (),
        {
            "Module": _StubModule,
            "Embedding": _StubModule,
            "Linear": _StubModule,
            "Parameter": _StubModule,
        },
    )
if torch is None:
    torch = type(  # type: ignore[assignment]
        "torch",
        (),
        {
            "Tensor": _StubTensor,
            "no_grad": staticmethod(contextlib.nullcontext),
            "cuda": type("cuda", (), {"is_available": staticmethod(lambda: False)}),
            "relu": staticmethod(lambda x: x),
            "matmul": staticmethod(lambda a, b: None),
            "sparse": type(
                "sparse",
                (),
                {
                    "mm": staticmethod(lambda a, b: None),
                    "sum": staticmethod(lambda a, dim: None),
                },
            ),
            "sparse_coo_tensor": staticmethod(lambda indices, values, size: None),
            "tensor": staticmethod(lambda data, dtype=None: None),
            "long": None,
            "float32": None,
            "device": staticmethod(lambda kind: None),
        },
    )
if F is None:
    F = type("F", (), {})  # type: ignore[assignment]


SPECIAL_BOS = "<BOS>"
SPECIAL_EOS = "<EOS>"
SPECIAL_UNK = "<UNK>"


@dataclass
class GNNLanguageConfig:
    vocab_path: str = "artifacts/memory/gnn_vocab.json"
    edges_path: str = "artifacts/memory/gnn_edges.json"
    corpus_path: str = "artifacts/memory/gnn_corpus.jsonl"
    model_path: str = "artifacts/memory/gnn_model.pt"
    max_vocab: int = 20000
    window_size: int = 4
    context_size: int = 8
    min_corpus_tokens: int = 80
    min_vocab: int = 32
    embed_dim: int = 64
    train_steps: int = 6
    batch_size: int = 16
    lr: float = 0.02
    temperature: float = 0.6
    top_k: int = 6
    nb_bias: float = 0.6
    use_cuda: bool = False
    corpus_max_lines: int = 4000
    ord_factor: float = 1.2
    min_output_len: int = 2
    min_unique_ratio: float = 0.35
    max_repeat_run: int = 6
    min_confidence: float = 0.2
    deterministic: bool = True


class _CharTokenizer:
    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        out: List[str] = []
        for ch in text:
            if ch.isspace():
                continue
            out.append(ch)
        return out


class _GNNLangModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lin1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lin2 = nn.Linear(embed_dim, embed_dim, bias=False)

    def propagate(self, adj: torch.Tensor) -> torch.Tensor:
        x = self.emb.weight
        h1 = torch.relu(torch.sparse.mm(adj, self.lin1(x)))
        h2 = torch.relu(torch.sparse.mm(adj, self.lin2(h1)))
        return h2


class GNNLanguageEngine:
    def __init__(self, config: Optional[GNNLanguageConfig] = None):
        self.config = config or GNNLanguageConfig()
        self._tokenizer = _CharTokenizer()
        self.vocab: List[str] = []
        self.token_to_id: Dict[str, int] = {}
        self.token_counts: Dict[str, int] = {}
        self.co_counts: Dict[str, float] = {}
        self.ord_counts: Dict[str, float] = {}
        self.corpus_tokens: List[List[str]] = []
        self._adj: Optional[torch.Tensor] = None
        self._adj_dirty = True
        self._model: Optional[_GNNLangModel] = None
        self._optim = None
        self._device = None
        self._available = torch is not None
        if self._available:
            use_cuda = bool(self.config.use_cuda) or os.environ.get("GNN_USE_CUDA", "0") == "1"
            if use_cuda and torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        self._load_state()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def available(self) -> bool:
        return self._available and self._model is not None and self._adj is not None

    def ready(self) -> bool:
        if not self.available():
            return False
        if self.vocab_size < self.config.min_vocab:
            return False
        total = sum(len(seq) for seq in self.corpus_tokens)
        return total >= self.config.min_corpus_tokens

    def observe_text(self, text: str, role: str = "") -> None:
        tokens = self._tokenizer.tokenize(text)
        if not tokens:
            return
        seq = [SPECIAL_BOS] + tokens + [SPECIAL_EOS]
        self._ensure_tokens(seq)
        self._update_edges(seq)
        self._append_corpus(text, role)
        self._adj_dirty = True

    def add_cross_modal_edges(
        self,
        token_groups: Dict[str, List[str]],
        alias_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if not token_groups:
            return
        reverse_alias: Dict[str, str] = {}
        if isinstance(alias_map, dict):
            for canon, variants in alias_map.items():
                if not canon:
                    continue
                reverse_alias[str(canon)] = str(canon)
                if isinstance(variants, (list, tuple, set)):
                    for v in variants:
                        if v:
                            reverse_alias[str(v)] = str(canon)
        expanded: Dict[str, Dict[str, float]] = {}
        for modality, tokens in token_groups.items():
            if not tokens:
                continue
            bucket: Dict[str, float] = {}
            for t in tokens:
                tok = None
                conf = 1.0
                if isinstance(t, dict):
                    tok = t.get("token")
                    conf = float(t.get("score", 1.0))
                elif isinstance(t, (tuple, list)):
                    tok = t[0] if t else None
                    conf = float(t[1]) if len(t) > 1 else 1.0
                else:
                    tok = str(t)
                    conf = 1.0
                if not tok:
                    continue
                bucket[tok] = max(bucket.get(tok, 0.0), conf)
                canon = reverse_alias.get(str(tok))
                if canon:
                    bucket[canon] = max(bucket.get(canon, 0.0), conf)
            if bucket:
                expanded[modality] = bucket
        if len(expanded) < 2:
            return
        # ensure tokens exist
        for toks in expanded.values():
            self._ensure_tokens(list(toks.keys()))
        mods = list(expanded.keys())
        max_pairs = int(os.environ.get("CROSS_MODAL_MAX_PAIRS", "256"))
        pairs_used = 0
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                a_map = expanded.get(mods[i], {})
                b_map = expanded.get(mods[j], {})
                for a, a_conf in a_map.items():
                    for b, b_conf in b_map.items():
                        if pairs_used >= max_pairs:
                            break
                        if not a or not b or a == b:
                            continue
                        key = self._co_key(a, b)
                        weight = min(float(a_conf), float(b_conf))
                        self.co_counts[key] = float(self.co_counts.get(key, 0.0)) + max(weight, 0.01)
                        pairs_used += 1
                    if pairs_used >= max_pairs:
                        break
                if pairs_used >= max_pairs:
                    break
            if pairs_used >= max_pairs:
                break
        if pairs_used > 0:
            self._adj_dirty = True
            self._save_edges()

    def train_online(self, steps: Optional[int] = None) -> None:
        if not self.available():
            return
        if not self.corpus_tokens:
            return
        steps = int(steps or self.config.train_steps)
        if steps <= 0:
            return
        self._refresh_graph()
        if self._model is None or self._optim is None:
            return
        self._model.train()
        for _ in range(steps):
            batch = self._sample_batch(self.config.batch_size, self.config.context_size)
            if not batch:
                break
            contexts, targets = batch
            loss = self._train_batch(contexts, targets)
            if loss is None:
                break
        self._save_model()

    def generate(
        self,
        prompt: str,
        nanobrain=None,
        max_len: int = 48,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        if not self.ready():
            return ""
        self._refresh_graph()
        if self._model is None or self._adj is None:
            return ""
        temperature = float(temperature or self.config.temperature)
        top_k = int(top_k or self.config.top_k)
        ctx_len = int(self.config.context_size)
        tokens = self._tokenizer.tokenize(prompt)
        if not tokens:
            tokens = [SPECIAL_BOS]
        ctx = tokens[-ctx_len:]
        bias = self._build_nb_bias(nanobrain, prompt) if nanobrain is not None else None

        self._model.eval()
        with torch.no_grad():
            h = self._model.propagate(self._adj)
            generated: List[str] = []
            repeat_run = 0
            confidences: List[float] = []
            for _ in range(max_len):
                ctx_ids = self._tokens_to_ids(ctx)
                if not ctx_ids:
                    break
                ctx_vec = h[ctx_ids].mean(dim=0)
                logits = torch.matmul(ctx_vec, h.t())
                if bias is not None:
                    logits = logits + bias
                # avoid special tokens
                for sp in (SPECIAL_BOS, SPECIAL_UNK):
                    sid = self.token_to_id.get(sp)
                    if sid is not None:
                        logits[sid] = -1e9
                # avoid early stop
                if len(generated) < int(self.config.min_output_len):
                    eos_id = self.token_to_id.get(SPECIAL_EOS)
                    if eos_id is not None:
                        logits[eos_id] = -1e9
                # avoid repeated run of the same token
                if ctx:
                    last_id = self.token_to_id.get(ctx[-1])
                else:
                    last_id = None
                if last_id is not None and repeat_run >= int(self.config.max_repeat_run):
                    logits[last_id] = -1e9
                next_id, max_prob = self._sample_logits_with_confidence(
                    logits, temperature=temperature, top_k=top_k
                )
                confidences.append(max_prob)
                tok = self.vocab[next_id]
                if tok == SPECIAL_EOS:
                    break
                if tok not in (SPECIAL_BOS, SPECIAL_UNK):
                    generated.append(tok)
                if ctx and tok == ctx[-1]:
                    repeat_run += 1
                else:
                    repeat_run = 1
                ctx = (ctx + [tok])[-ctx_len:]
        text = "".join(generated).strip()
        if not self._passes_quality(text, confidences):
            return ""
        return text

    def _load_state(self) -> None:
        self._load_vocab()
        self._load_edges()
        self._load_corpus()
        self._refresh_graph()
        self._load_model()

    def _load_vocab(self) -> None:
        path = Path(self.config.vocab_path)
        if not path.exists():
            self.vocab = [SPECIAL_BOS, SPECIAL_EOS, SPECIAL_UNK]
            self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
            self.token_counts = {t: 1 for t in self.vocab}
            self._save_vocab()
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tokens = data.get("tokens") if isinstance(data, dict) else data
            counts = data.get("counts") if isinstance(data, dict) else None
            if isinstance(tokens, list) and tokens:
                self.vocab = [str(t) for t in tokens]
            else:
                self.vocab = [SPECIAL_BOS, SPECIAL_EOS, SPECIAL_UNK]
            self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
            if isinstance(counts, dict):
                self.token_counts = {str(k): int(v) for k, v in counts.items()}
            else:
                self.token_counts = {t: 1 for t in self.vocab}
        except Exception:
            self.vocab = [SPECIAL_BOS, SPECIAL_EOS, SPECIAL_UNK]
            self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
            self.token_counts = {t: 1 for t in self.vocab}

    def _save_vocab(self) -> None:
        path = Path(self.config.vocab_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"tokens": self.vocab, "counts": self.token_counts}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_edges(self) -> None:
        path = Path(self.config.edges_path)
        if not path.exists():
            self.co_counts = {}
            self.ord_counts = {}
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.co_counts = {str(k): float(v) for k, v in (data.get("co") or {}).items()}
            self.ord_counts = {str(k): float(v) for k, v in (data.get("ord") or {}).items()}
        except Exception:
            self.co_counts = {}
            self.ord_counts = {}

    def _save_edges(self) -> None:
        path = Path(self.config.edges_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"co": self.co_counts, "ord": self.ord_counts}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _append_corpus(self, text: str, role: str = "") -> None:
        path = Path(self.config.corpus_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"role": role, "text": text}, ensure_ascii=False) + "\n")
        except Exception:
            return
        tokens = self._tokenizer.tokenize(text)
        if tokens:
            self.corpus_tokens.append(tokens)
            max_lines = int(self.config.corpus_max_lines)
            if max_lines > 0 and len(self.corpus_tokens) > max_lines:
                self.corpus_tokens = self.corpus_tokens[-max_lines:]

    def _load_corpus(self) -> None:
        path = Path(self.config.corpus_path)
        if not path.exists():
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        max_lines = int(self.config.corpus_max_lines)
        if max_lines > 0:
            lines = lines[-max_lines:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get("text", "")
            except Exception:
                text = ""
            tokens = self._tokenizer.tokenize(text)
            if tokens:
                self.corpus_tokens.append(tokens)

    def _ensure_tokens(self, tokens: List[str]) -> None:
        for t in tokens:
            if t not in self.token_to_id:
                if len(self.vocab) >= int(self.config.max_vocab):
                    continue
                self.token_to_id[t] = len(self.vocab)
                self.vocab.append(t)
            self.token_counts[t] = int(self.token_counts.get(t, 0)) + 1
        self._save_vocab()

    def _update_edges(self, tokens: List[str]) -> None:
        win = max(1, int(self.config.window_size))
        n = len(tokens)
        for i in range(n):
            t_i = tokens[i]
            for j in range(i + 1, min(n, i + win + 1)):
                t_j = tokens[j]
                key = self._co_key(t_i, t_j)
                self.co_counts[key] = float(self.co_counts.get(key, 0.0)) + 1.0
        for i in range(n - 1):
            a = tokens[i]
            b = tokens[i + 1]
            key = self._ord_key(a, b)
            self.ord_counts[key] = float(self.ord_counts.get(key, 0.0)) + 1.0
        self._save_edges()

    def _refresh_graph(self) -> None:
        if not self._available:
            return
        if not self._adj_dirty and self._adj is not None:
            return
        self._adj = self._build_adj().to(self._device)
        self._adj_dirty = False
        self._ensure_model()

    def _build_adj(self) -> torch.Tensor:
        vocab_size = len(self.vocab)
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []

        def add_edge(src: str, dst: str, weight: float) -> None:
            if src not in self.token_to_id or dst not in self.token_to_id:
                return
            rows.append(self.token_to_id[src])
            cols.append(self.token_to_id[dst])
            vals.append(weight)

        for key, cnt in self.co_counts.items():
            a, b = key.split("\t", 1)
            w = math.log1p(cnt)
            add_edge(a, b, w)
            add_edge(b, a, w)
        ord_factor = float(self.config.ord_factor)
        for key, cnt in self.ord_counts.items():
            a, b = key.split("\t", 1)
            w = math.log1p(cnt) * ord_factor
            add_edge(a, b, w)

        for i in range(vocab_size):
            rows.append(i)
            cols.append(i)
            vals.append(1.0)

        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        adj = torch.sparse_coo_tensor(indices, values, (vocab_size, vocab_size))
        adj = adj.coalesce()
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        row_sum = torch.clamp(row_sum, min=1e-6)
        new_vals = values / row_sum[indices[0]]
        return torch.sparse_coo_tensor(indices, new_vals, (vocab_size, vocab_size)).coalesce()

    def _ensure_model(self) -> None:
        if not self._available:
            return
        if self._model is None:
            self._model = _GNNLangModel(len(self.vocab), self.config.embed_dim).to(self._device)
            self._optim = torch.optim.Adam(self._model.parameters(), lr=self.config.lr)
            return
        if self._model.emb.num_embeddings != len(self.vocab):
            self._resize_embeddings(len(self.vocab))

    def _resize_embeddings(self, new_size: int) -> None:
        if self._model is None:
            return
        old_emb = self._model.emb.weight.data
        old_size, dim = old_emb.shape
        new_emb = nn.Embedding(new_size, dim).to(self._device)
        nn.init.normal_(new_emb.weight, mean=0.0, std=0.02)
        new_emb.weight.data[:old_size] = old_emb
        self._model.emb = new_emb
        self._optim = torch.optim.Adam(self._model.parameters(), lr=self.config.lr)

    def _load_model(self) -> None:
        if not self._available:
            return
        path = Path(self.config.model_path)
        if not path.exists():
            self._ensure_model()
            return
        self._ensure_model()
        try:
            data = torch.load(str(path), map_location=self._device)
            state = data.get("model") if isinstance(data, dict) else data
            if state:
                self._model.load_state_dict(state, strict=False)
        except Exception:
            pass

    def _save_model(self) -> None:
        if not self._available or self._model is None:
            return
        path = Path(self.config.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self._model.state_dict(), "vocab_size": len(self.vocab)}
        try:
            torch.save(payload, str(path))
        except Exception:
            pass

    def _sample_batch(self, batch_size: int, ctx_len: int) -> Optional[Tuple[List[List[int]], List[int]]]:
        if not self.corpus_tokens:
            return None
        contexts: List[List[int]] = []
        targets: List[int] = []
        for _ in range(batch_size):
            seq = random.choice(self.corpus_tokens)
            if not seq:
                continue
            ids = self._tokens_to_ids([SPECIAL_BOS] + seq + [SPECIAL_EOS])
            if len(ids) < 2:
                continue
            idx = random.randint(1, len(ids) - 1)
            start = max(0, idx - ctx_len)
            ctx = ids[start:idx]
            if not ctx:
                continue
            contexts.append(ctx)
            targets.append(ids[idx])
        if not contexts:
            return None
        return contexts, targets

    def _train_batch(self, contexts: List[List[int]], targets: List[int]) -> Optional[float]:
        if self._model is None or self._adj is None:
            return None
        h = self._model.propagate(self._adj)
        ctx_vecs = []
        for ctx in contexts:
            ctx_vecs.append(h[ctx].mean(dim=0))
        ctx_mat = torch.stack(ctx_vecs, dim=0)
        logits = torch.matmul(ctx_mat, h.t())
        target_tensor = torch.tensor(targets, dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, target_tensor)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        return float(loss.item())

    def _build_nb_bias(self, nanobrain, prompt: str) -> Optional[torch.Tensor]:
        if nanobrain is None or not hasattr(nanobrain, "stimulate"):
            return None
        try:
            activation, _idxs = nanobrain.stimulate(prompt)
        except Exception:
            return None
        if activation is None:
            return None
        bias = torch.zeros(len(self.vocab), device=self._device, dtype=torch.float32)
        nb_index = getattr(nanobrain, "index", {})
        if not isinstance(nb_index, dict):
            return None
        scale = float(self.config.nb_bias)
        for tok, tid in self.token_to_id.items():
            nb_id = nb_index.get(tok)
            if nb_id is None:
                continue
            try:
                bias[tid] = float(activation[nb_id]) * scale
            except Exception:
                continue
        return bias

    def _tokens_to_ids(self, tokens: List[str]) -> List[int]:
        out = []
        unk = self.token_to_id.get(SPECIAL_UNK, 0)
        for t in tokens:
            out.append(self.token_to_id.get(t, unk))
        return out

    def _sample_logits(self, logits: torch.Tensor, temperature: float, top_k: int) -> int:
        next_id, _conf = self._sample_logits_with_confidence(logits, temperature, top_k)
        return next_id

    def _sample_logits_with_confidence(
        self, logits: torch.Tensor, temperature: float, top_k: int
    ) -> Tuple[int, float]:
        if bool(self.config.deterministic):
            probs = torch.softmax(logits, dim=0)
            max_prob, idx = torch.max(probs, dim=0)
            return int(idx.item()), float(max_prob.item())
        if temperature <= 0:
            temperature = 1.0
        logits = logits / temperature
        if top_k > 0:
            top_k = min(top_k, logits.numel())
            vals, idxs = torch.topk(logits, k=top_k)
            probs = torch.softmax(vals, dim=0)
            pick = torch.multinomial(probs, 1).item()
            max_prob = float(torch.max(probs).item())
            return int(idxs[pick].item()), max_prob
        probs = torch.softmax(logits, dim=0)
        max_prob = float(torch.max(probs).item())
        return int(torch.multinomial(probs, 1).item()), max_prob

    def _passes_quality(self, text: str, confidences: List[float]) -> bool:
        if not text:
            return False
        min_len = int(self.config.min_output_len)
        if len(text) < min_len:
            return False
        placeholder_cnt = sum(1 for ch in text if ch in {"?", "？", "�", "□"})
        if placeholder_cnt / max(1, len(text)) > float(os.environ.get("GNN_MAX_PLACEHOLDER_RATIO", "0.25")):
            return False
        useful_cnt = sum(1 for ch in text if ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"))
        if useful_cnt <= 0:
            return False
        tokens = self._tokenizer.tokenize(text)
        if not tokens:
            return False
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        if unique_ratio < float(self.config.min_unique_ratio):
            return False
        if confidences:
            avg_conf = sum(confidences) / max(1, len(confidences))
            if avg_conf < float(self.config.min_confidence):
                return False
        return True

    @staticmethod
    def _co_key(a: str, b: str) -> str:
        return f"{a}\t{b}" if a <= b else f"{b}\t{a}"

    @staticmethod
    def _ord_key(a: str, b: str) -> str:
        return f"{a}\t{b}"
