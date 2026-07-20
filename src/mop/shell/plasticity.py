
from __future__ import annotations

import math

import torch
from torch import nn


def _named_trainable(model: nn.Module):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


class PlasticityController:
    def __init__(self, cfg, seed: int = 0):
        self.schedule = str(cfg.schedule)
        self.base_lr = float(cfg.lr)
        self.rigidity_w = float(cfg.rigidity)
        self.pnn_fraction = float(cfg.pnn_fraction)
        self.floor = 0.1
        self.close_at = 0.5  # hard: close after half the schedule
        self.reopen_threshold = 1.0  # signal z-score that reopens plasticity
        self.k = 4.0  # soft decay rate
        self._slope = 6.0  # learned-gate slope
        self.g = torch.Generator().manual_seed(seed)
        self._anchor: dict[str, torch.Tensor] = {}
        self._rigidity: dict[str, torch.Tensor] = {}
        self._pnn_mask: dict[str, torch.Tensor] = {}

    def lr_scale(self, progress: float, signal: float = 0.0) -> float:
        if self.schedule == "hard":
            base = 1.0 if progress < self.close_at else self.floor
        elif self.schedule == "soft":
            base = self.floor + (1 - self.floor) * math.exp(-self.k * progress)
        else:  # learned: sigmoid gate closing around mid-schedule
            base = self.floor + (1 - self.floor) / (1 + math.exp(self._slope * (progress - 0.5)))
        if signal > self.reopen_threshold:  # triggered reopening
            reopen = min(1.0, (signal - self.reopen_threshold) / self.reopen_threshold)
            base = base + (1 - base) * reopen
        return float(max(self.floor, min(1.0, base)))

    def lr(self, progress: float, signal: float = 0.0) -> float:
        return self.base_lr * self.lr_scale(progress, signal)

    def init_pnn(self, model: nn.Module) -> None:
        for n, p in _named_trainable(model):
            self._anchor[n] = p.detach().clone()
            self._rigidity[n] = torch.zeros_like(p)
            mask = torch.rand(p.shape, generator=self.g) < self.pnn_fraction
            self._pnn_mask[n] = mask

    def update_rigidity(self, model: nn.Module, ema: float = 0.9) -> None:
        for n, p in _named_trainable(model):
            if n not in self._anchor:
                self._anchor[n] = p.detach().clone()
                self._rigidity[n] = torch.zeros_like(p)
            move = (p.detach() - self._anchor[n]).abs()
            stability = torch.exp(-move)  # ~1 when unmoved, ->0 when moved
            self._rigidity[n] = ema * self._rigidity[n] + (1 - ema) * stability
            self._anchor[n] = p.detach().clone()

    def rigidity_penalty(self, model: nn.Module) -> torch.Tensor:
        total = torch.zeros((), device=next(model.parameters()).device)
        if not self.rigidity_w or not self._rigidity:
            return total
        for n, p in _named_trainable(model):
            if n in self._rigidity:
                total = total + (self._rigidity[n] * (p - self._anchor[n]) ** 2).sum()
        return self.rigidity_w * total

    def apply_pnn_freeze(self, model: nn.Module) -> None:
        for n, p in _named_trainable(model):
            if n in self._pnn_mask and p.grad is not None:
                p.grad[self._pnn_mask[n]] = 0.0
