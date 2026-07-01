"""EX3: test-time adaptation on frozen latents. A trained classifier sees a SHIFTED domain (same
labels, independent latent geometry, the domain-incremental setting). A small fast-weight overlay (a
per-feature affine on the latent, initialized to identity) is fit at inference by LABEL-FREE entropy
minimization on the shifted unlabeled latents (TENT-style), then evaluated; reverting the overlay must
restore the base. The question is whether the unlabeled proxy carries usable adaptation signal.

NULL: TTA does not beat the frozen head on the shifted domain at matched parameters (the unlabeled
proxy carries no usable signal, or adapting corrupts the base). Taxonomy slot 3 (the shift is not
decodable, nothing to adapt to) or 4 (overlay too small / proxy mis-specified). cpu-now, seconds.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..seeding import seed_everything
from .base import Experiment


class _Affine(nn.Module):
    """A per-feature fast-weight overlay y = gamma * x + beta, identity at init. Tiny (2*dim params),
    the slow head is never touched; reverting is dropping this module (the base is restored exactly)."""

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
            # two domains, SHARED labels, independent geometry: domain 0 = source, domain 1 = shift
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
            # train the slow head on the source domain
            head = nn.Sequential(nn.Linear(dim, int(e.hidden)), nn.GELU(), nn.Linear(int(e.hidden), nc))
            opt = torch.optim.Adam(head.parameters(), lr=float(e.lr))
            for _ in range(int(e.epochs)):
                opt.zero_grad()
                F.cross_entropy(head(src.x), src.y).backward()
                opt.step()
            base0 = _acc(head, src.x, src.y)

            frozen_acc.append(_acc(head, shift.x, shift.y))  # frozen head on the shift

            # label-free TTA: fit the affine overlay by entropy minimization on shift's UNLABELED latents
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

        def mean(v):
            return sum(v) / len(v)

        ta, fa = mean(tta_acc), mean(frozen_acc)
        out = {
            "shift_acc_tta": round(ta, 4),
            "shift_acc_frozen": round(fa, 4),
            "tta_gain": round(ta - fa, 4),
            "base_retention_after_revert": round(mean(base_ret), 4),
            "margin": float(e.margin),
            "seeds": list(seeds),
            # the explicit null: TTA does not beat the frozen head by more than the margin
            "null_supported": bool((ta - fa) <= float(e.margin)),
            "tta_helps": bool((ta - fa) > float(e.margin)),
        }
        return out
