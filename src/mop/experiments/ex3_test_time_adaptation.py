
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..seeding import seed_everything
from .base import Experiment, _mean


class _Affine(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x + self.beta


def _acc(head, x, y, overlay=None) -> float:
    with torch.no_grad():
        z = overlay(x) if overlay is not None else x
        return float((head(z).argmax(-1) == y).float().mean())


class EX3(Experiment):
    id = "ex3_test_time_adaptation"
    metric = ("shift_acc_tta", "shift_acc_frozen", "base_retention_after_revert")
    baseline = "the frozen head on the shifted domain; full source accuracy is the upper anchor"
    ablation = "label-free entropy-min fast-weight overlay vs the frozen head at matched params"
    null_hypothesis = (
        "test-time adaptation does not beat the frozen head on the shifted domain at matched parameters: "
        "the unlabeled entropy proxy carries no usable adaptation signal, or adapting corrupts the base"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim = int(e.dim)
        tta_acc, frozen_acc, base_ret = [], [], []
        for s in seeds:
            seed_everything(s)
            stream = make_task_stream(
                n_tasks=2,
                dim=dim,
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                incremental="domain",
                seed=s,
            )
            src, shift = stream[0], stream[1]
            nc = int(max(src.y.max(), shift.y.max())) + 1
            head = nn.Sequential(nn.Linear(dim, int(e.hidden)), nn.GELU(), nn.Linear(int(e.hidden), nc))
            opt = torch.optim.Adam(head.parameters(), lr=float(e.lr))
            for _ in range(int(e.epochs)):
                opt.zero_grad()
                F.cross_entropy(head(src.x), src.y).backward()
                opt.step()
            base0 = _acc(head, src.x, src.y)

            frozen_acc.append(_acc(head, shift.x, shift.y))  # frozen head on the shift

            overlay = _Affine(dim)
            for p in head.parameters():
                p.requires_grad_(False)
            topt = torch.optim.Adam(overlay.parameters(), lr=float(e.tta_lr))
            for _ in range(int(e.tta_steps)):
                topt.zero_grad()
                logits = head(overlay(shift.x))
                p = logits.softmax(-1)
                entropy = -(p * (p + 1e-8).log()).sum(-1).mean()
                entropy.backward()
                topt.step()
            tta_acc.append(_acc(head, shift.x, shift.y, overlay))
            base_ret.append(_acc(head, src.x, src.y))  # revert == drop overlay; base must be intact
            assert abs(base_ret[-1] - base0) < 1e-6  # the overlay never touched the slow head

        ta, fa = _mean(tta_acc), _mean(frozen_acc)
        out = {
            "shift_acc_tta": round(ta, 4),
            "shift_acc_frozen": round(fa, 4),
            "tta_gain": round(ta - fa, 4),
            "base_retention_after_revert": round(_mean(base_ret), 4),
            "margin": float(e.margin),
            "seeds": list(seeds),
            "null_supported": bool((ta - fa) <= float(e.margin)),
            "tta_helps": bool((ta - fa) > float(e.margin)),
        }
        return out
