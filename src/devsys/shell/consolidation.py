"""Synaptic consolidation: the weight-space dual of replay. EWC (diagonal Fisher proxy) and
SI (path integral of the gradient). Both add a per-parameter quadratic penalty pulling
important weights back toward their post-task values. Composable (method=both).

EWC:  penalty = lambda * sum_i F_i (theta_i - theta*_i)^2,   F_i ~ mean grad_i^2
SI:   penalty = c      * sum_i Omega_i (theta_i - theta*_i)^2,
      Omega_i += w_i / ((Dtheta_i)^2 + xi),  w_i = sum_t (-grad_i,t * dtheta_i,t)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn


def _named_trainable(model: nn.Module):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


class EWC:
    def __init__(self, lam: float = 1e3):
        self.lam = lam
        self.fisher: dict[str, torch.Tensor] = {}
        self.star: dict[str, torch.Tensor] = {}

    def estimate(self, model: nn.Module, batches: Iterable, loss_fn: Callable, samples: int = 64) -> None:
        """Accumulate diagonal Fisher = mean of squared grads over `samples`, snapshot theta*."""
        fisher = {n: torch.zeros_like(p) for n, p in _named_trainable(model)}
        seen = 0
        for batch in batches:
            model.zero_grad(set_to_none=True)
            loss = loss_fn(batch)
            loss.backward()
            for n, p in _named_trainable(model):
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            seen += 1
            if seen >= samples:
                break
        seen = max(1, seen)
        # accumulate across tasks (Fisher adds), refresh anchor to current weights
        for n, p in _named_trainable(model):
            self.fisher[n] = self.fisher.get(n, 0) + fisher[n] / seen
            self.star[n] = p.detach().clone()
        model.zero_grad(set_to_none=True)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        total = torch.zeros((), device=next(model.parameters()).device)
        for n, p in _named_trainable(model):
            if n in self.fisher:
                total = total + (self.fisher[n] * (p - self.star[n]) ** 2).sum()
        return self.lam * total


class SI:
    def __init__(self, c: float = 0.1, xi: float = 1e-3):
        self.c, self.xi = c, xi
        self.omega: dict[str, torch.Tensor] = {}
        self.star: dict[str, torch.Tensor] = {}
        self._w: dict[str, torch.Tensor] = {}
        self._prev: dict[str, torch.Tensor] = {}
        self._task_start: dict[str, torch.Tensor] = {}

    def begin_task(self, model: nn.Module) -> None:
        for n, p in _named_trainable(model):
            self._w[n] = torch.zeros_like(p)
            self._prev[n] = p.detach().clone()
            self._task_start[n] = p.detach().clone()

    def before_step(self, model: nn.Module) -> None:
        for n, p in _named_trainable(model):
            self._prev[n] = p.detach().clone()

    def after_step(self, model: nn.Module) -> None:
        """Call right after optimizer.step() while grads still populated. Accumulates the
        path integral w_i += -grad_i * (theta_i_new - theta_i_old)."""
        for n, p in _named_trainable(model):
            if p.grad is not None:
                self._w[n] += -p.grad.detach() * (p.detach() - self._prev[n])

    def consolidate(self, model: nn.Module) -> None:
        for n, p in _named_trainable(model):
            d2 = (p.detach() - self._task_start[n]) ** 2
            self.omega[n] = self.omega.get(n, 0) + self._w[n].clamp_min(0) / (d2 + self.xi)
            self.star[n] = p.detach().clone()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        total = torch.zeros((), device=next(model.parameters()).device)
        for n, p in _named_trainable(model):
            if n in self.omega:
                total = total + (self.omega[n] * (p - self.star[n]) ** 2).sum()
        return self.c * total


class Consolidation:
    """Selects none|ewc|si|both and exposes one penalty()."""

    def __init__(self, cfg):
        self.method = str(cfg.method)
        self.ewc = EWC(float(cfg.ewc_lambda)) if self.method in ("ewc", "both") else None
        self.si = SI(float(cfg.si_c), float(cfg.si_xi)) if self.method in ("si", "both") else None
        self.fisher_samples = int(cfg.fisher_samples)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        z = torch.zeros((), device=next(model.parameters()).device)
        if self.ewc:
            z = z + self.ewc.penalty(model)
        if self.si:
            z = z + self.si.penalty(model)
        return z

    def begin_task(self, model: nn.Module) -> None:
        if self.si:
            self.si.begin_task(model)

    def before_step(self, model: nn.Module) -> None:
        if self.si:
            self.si.before_step(model)

    def after_step(self, model: nn.Module) -> None:
        if self.si:
            self.si.after_step(model)

    def consolidate(self, model: nn.Module, batches=None, loss_fn=None) -> None:
        if self.ewc and batches is not None and loss_fn is not None:
            self.ewc.estimate(model, batches, loss_fn, self.fisher_samples)
        if self.si:
            self.si.consolidate(model)
