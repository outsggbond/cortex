from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover - runtime fallback
    torch = None  # type: ignore


TensorMap = Dict[str, "torch.Tensor"]  # type: ignore[name-defined]


@dataclass
class EWCMemory:
    """Lightweight EWC memory for continual adaptation."""

    fisher: TensorMap = field(default_factory=dict)
    anchor: TensorMap = field(default_factory=dict)
    decay: float = 0.95

    def snapshot(self, named_params: Iterable[Tuple[str, "torch.Tensor"]]) -> None:  # type: ignore[name-defined]
        if torch is None:
            return
        out: TensorMap = {}
        for name, param in named_params:
            if param is None or (not torch.is_tensor(param)):
                continue
            out[str(name)] = param.detach().clone().float().cpu()
        self.anchor = out

    def update_fisher_from_grads(self, named_grads: Iterable[Tuple[str, "torch.Tensor"]]) -> None:  # type: ignore[name-defined]
        if torch is None:
            return
        d = max(0.0, min(0.999, float(self.decay)))
        for name, grad in named_grads:
            if grad is None or (not torch.is_tensor(grad)):
                continue
            g2 = grad.detach().float().cpu().pow(2.0)
            old = self.fisher.get(str(name))
            if old is None:
                self.fisher[str(name)] = g2
            else:
                self.fisher[str(name)] = old * d + g2 * (1.0 - d)

    def penalty(
        self,
        named_params: Iterable[Tuple[str, "torch.Tensor"]],  # type: ignore[name-defined]
        *,
        lambda_ewc: float = 1.0,
        default_importance: float = 0.0,
    ) -> "torch.Tensor":  # type: ignore[name-defined]
        if torch is None:
            raise RuntimeError("torch unavailable")
        lam = max(0.0, float(lambda_ewc))
        if lam <= 0.0:
            return torch.zeros((), dtype=torch.float32)
        reg = torch.zeros((), dtype=torch.float32)
        for name, p in named_params:
            if p is None or (not torch.is_tensor(p)):
                continue
            key = str(name)
            anchor = self.anchor.get(key)
            if anchor is None:
                if default_importance <= 0.0:
                    continue
                fisher = torch.full_like(p.detach().float().cpu(), float(default_importance))
            else:
                fisher = self.fisher.get(key)
                if fisher is None:
                    if default_importance <= 0.0:
                        continue
                    fisher = torch.full_like(anchor, float(default_importance))
                if fisher.shape != anchor.shape:
                    continue
            diff = p.detach().float().cpu() - anchor
            reg = reg + torch.sum(fisher * diff * diff)
        return reg * float(lam)


@dataclass
class EpisodicGradientMemory:
    """Simplified GEM memory with projection onto stored gradients."""

    max_items: int = 16
    vectors: List["torch.Tensor"] = field(default_factory=list)  # type: ignore[name-defined]

    def add(self, grad_vector: "torch.Tensor") -> None:  # type: ignore[name-defined]
        if torch is None:
            return
        if grad_vector is None or (not torch.is_tensor(grad_vector)):
            return
        vec = grad_vector.detach().float().cpu().view(-1)
        self.vectors.append(vec)
        if len(self.vectors) > max(1, int(self.max_items)):
            self.vectors = self.vectors[-int(self.max_items) :]

    def project(self, grad_vector: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        if torch is None:
            raise RuntimeError("torch unavailable")
        g = grad_vector.detach().float().cpu().view(-1)
        if not self.vectors:
            return g
        out = g.clone()
        eps = 1e-12
        for mem in self.vectors:
            denom = float(torch.dot(mem, mem).item())
            if denom <= eps:
                continue
            dot = float(torch.dot(out, mem).item())
            if dot < 0.0:
                out = out - (dot / denom) * mem
        return out

