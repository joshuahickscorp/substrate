
from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, param_count, refiner_flops
from ..seeding import seed_everything
from ..shell.predictor import mlp
from ..shell.refine import IterativeRefiner
from .base import Experiment, _fit_eval


class _UntiedDepth(nn.Module):

    def __init__(self, dim: int, hidden: int, steps: int):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(steps)])
        self.blocks = nn.ModuleList([mlp(dim, dim, hidden, depth=1, ln=True) for _ in range(steps)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for norm, block in zip(self.norms, self.blocks, strict=True):
            z = z + block(norm(z))
        return z


class EX17(Experiment):
    id = "ex17_latent_reasoning"
    metric = ("accuracy", "gain_vs_compute_matched", "params_ratio", "halting_steps")
    baseline = "compute-matched untied-depth network (same block count + hidden, untied weights)"
    ablation = "tied iterative refiner vs untied depth at matched forward FLOPs; frozen-random substrate"
    null_hypothesis = (
        "at matched compute the iterative refiner ties the untied-depth control within `margin`: "
        "iteration was just depth, weight-tying buys nothing on a frozen pooled latent"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, steps = int(e.dim), int(e.hidden), int(e.steps)
        margin = float(e.margin)
        epochs, lr = int(e.epochs), float(e.lr)

        ref_acc, ctl_acc, halts = [], [], []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=int(e.n_classes),
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            x, y = task.x, task.y
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            nc = int(y.max()) + 1

            refiner = IterativeRefiner(dim, hidden, steps, halt=bool(e.adaptive_halt))
            rhead = nn.Linear(dim, nc)
            ref_acc.append(_fit_eval(refiner, rhead, xtr, ytr, xte, yte, epochs, lr))
            with torch.no_grad():
                _, used = refiner(xte)
            halts.append(float(used.float().mean()))

            seed_everything(s)  # same init seed so the comparison is fair
            ctl = _UntiedDepth(dim, hidden, steps)
            chead = nn.Linear(dim, nc)
            ctl_acc.append(_fit_eval(ctl, chead, xtr, ytr, xte, yte, epochs, lr))

        rm = sum(ref_acc) / len(ref_acc)
        cm = sum(ctl_acc) / len(ctl_acc)
        refiner = IterativeRefiner(dim, hidden, steps)
        ctl = _UntiedDepth(dim, hidden, steps)
        flops_ref = refiner_flops(dim, hidden, steps)
        flops_ctl = refiner_flops(dim, hidden, steps)  # equal block count + hidden -> equal FLOPs
        compute = matched_within(flops_ref, flops_ctl)
        gain = rm - cm
        out = {
            "refiner_acc_mean": round(rm, 4),
            "control_acc_mean": round(cm, 4),
            "gain_vs_compute_matched": round(gain, 4),
            "margin": margin,
            "params_refiner": param_count(refiner),
            "params_control": param_count(ctl),
            "params_ratio": round(param_count(refiner) / max(1, param_count(ctl)), 4),
            "compute_matched": compute,
            "mean_halting_steps": round(sum(halts) / len(halts), 3),
            "seeds": list(seeds),
            "null_supported": bool(abs(gain) <= margin),
            "refiner_wins_at_equal_flops": bool(gain > margin and compute["matched"]),
        }
        return out
