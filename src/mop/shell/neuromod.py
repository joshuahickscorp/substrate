"""Neuromodulation (Category D). Scalar global signals derived from the system's OWN
statistics that gate other processes:
  DA  = reward / surprise = prediction-error magnitude  (gates learning rate)
  ACh = expected uncertainty = novelty                  (gates memory write / priors)
  NE  = unexpected uncertainty = ensemble disagreement / context shift (gates reset/explore)

Each raw signal is z-normalized by a running EMA estimator, squashed through a sigmoid, and
mapped into a multiplicative gain in [floor, ceil]. The bridge to plasticity: a high
surprise/disagreement gain is what triggers critical-period reopening in the controller.
"""

from __future__ import annotations

import torch


class RunningStat:
    """EMA mean/var for online z-scoring. Bias-light; var floored for stability."""

    def __init__(self, momentum: float = 0.99):
        self.m = momentum
        self.mean = 0.0
        self.var = 1.0
        self._init = False

    def update(self, x: float) -> float:
        if not self._init:
            self.mean, self.var, self._init = x, 1.0, True
        else:
            d = x - self.mean
            self.mean += (1 - self.m) * d
            self.var = self.m * self.var + (1 - self.m) * d * d
        return self.z(x)

    def z(self, x: float) -> float:
        return (x - self.mean) / (self.var**0.5 + 1e-6)


class Neuromodulation:
    def __init__(self, cfg):
        self.enabled = bool(cfg.enabled)
        self.gains = {
            "surprise": float(cfg.surprise_gain),
            "novelty": float(cfg.novelty_gain),
            "uncertainty": float(cfg.uncertainty_gain),
        }
        self.floor = float(cfg.gate_floor)
        self.ceil = float(cfg.gate_ceil)
        self.stats = {k: RunningStat() for k in self.gains}

    @staticmethod
    def surprise(pred: torch.Tensor, target: torch.Tensor) -> float:
        return float((pred - target).abs().mean())

    def signals(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        disagreement: torch.Tensor | float | None = None,
        novelty: torch.Tensor | float | None = None,
    ) -> dict[str, float]:
        s = {"surprise": self.surprise(pred, target)}
        s["uncertainty"] = (
            float(torch.as_tensor(disagreement).mean()) if disagreement is not None else s["surprise"]
        )
        s["novelty"] = float(torch.as_tensor(novelty).mean()) if novelty is not None else s["surprise"]
        return s

    def gate(self, name: str, value: float) -> float:
        """Map a raw signal to a multiplicative gain in [floor, ceil]. z-score -> sigmoid ->
        scale by the configured per-channel gain, clamped."""
        if not self.enabled:
            return 1.0
        z = self.stats[name].update(value)
        sig = 1.0 / (1.0 + torch.tensor(-z).exp().item())  # in (0,1)
        gain = self.floor + (self.ceil - self.floor) * sig
        gain = 1.0 + self.gains[name] * (gain - 1.0)
        return float(max(self.floor, min(self.ceil, gain)))

    def gates(self, signals: dict[str, float]) -> dict[str, float]:
        return {k: self.gate(k, v) for k, v in signals.items()}
