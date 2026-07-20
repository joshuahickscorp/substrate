from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn


def fisher_trace(model: nn.Module, batches: Iterable, loss_fn: Callable, samples: int = 16) -> float:
    total = 0.0
    seen = 0
    for batch in batches:
        model.zero_grad(set_to_none=True)
        loss_fn(batch).backward()
        total += sum(float((p.grad**2).sum()) for p in model.parameters() if p.grad is not None)
        seen += 1
        if seen >= samples:
            break
    model.zero_grad(set_to_none=True)
    return total / max(1, seen)


def fisher_trace_over_training(
    make_model: Callable[[], nn.Module],
    data_batches: list,
    loss_fn_factory: Callable[[nn.Module], Callable],
    checkpoints: int = 8,
    steps_per_ckpt: int = 10,
    lr: float = 1e-2,
) -> list[float]:
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = loss_fn_factory(model)
    trace = []
    it = 0
    for _ in range(checkpoints):
        trace.append(fisher_trace(model, data_batches, loss_fn, samples=min(8, len(data_batches))))
        for _ in range(steps_per_ckpt):
            b = data_batches[it % len(data_batches)]
            it += 1
            opt.zero_grad()
            loss_fn(b).backward()
            opt.step()
    return trace


def critical_period_signature(trace: list[float]) -> dict:
    if len(trace) < 3:
        return {"peak_index": 0, "rise_then_fall": False}
    peak = max(range(len(trace)), key=lambda i: trace[i])
    rise_then_fall = 0 < peak < len(trace) - 1 and trace[peak] > trace[0] and trace[peak] > trace[-1]
    return {"peak_index": peak, "rise_then_fall": bool(rise_then_fall), "trace": trace}
