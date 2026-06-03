from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from system.learning.continual import EWCMemory, EpisodicGradientMemory
from system.brain.debate_reasoner import DebateReasoner


def _test_ewc_memory() -> None:
    w = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32))
    m = EWCMemory(decay=0.9)
    m.snapshot([("w", w)])
    grad = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float32)
    m.update_fisher_from_grads([("w", grad)])
    with torch.no_grad():
        w.copy_(torch.tensor([1.2, -2.3, 3.4], dtype=torch.float32))
    p = m.penalty([("w", w)], lambda_ewc=2.0)
    assert float(p.item()) > 0.0


def _test_gem_projection() -> None:
    gem = EpisodicGradientMemory(max_items=8)
    g_mem = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    g_cur = torch.tensor([-1.0, 0.5, 0.1], dtype=torch.float32)
    gem.add(g_mem)
    out = gem.project(g_cur)
    assert float(torch.dot(out, g_mem).item()) >= -1e-6


def _test_debate_reasoner() -> None:
    d = DebateReasoner()
    out = d.debate("why does a dropped cup fall down", context={"domain": "physics"})
    assert bool(out.get("ok", False))
    assert str(out.get("winner", "")) in {"proposer", "challenger"}
    assert str(out.get("answer", "")).strip()


def main() -> None:
    _test_ewc_memory()
    _test_gem_projection()
    _test_debate_reasoner()
    print("continual_debate_ok")


if __name__ == "__main__":
    main()

