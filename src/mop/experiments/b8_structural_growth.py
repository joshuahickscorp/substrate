
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import param_count
from ..seeding import seed_everything
from ..substrate.datasets import make_task_stream
from .base import Experiment, _mean, _spread
from .base import _split_xy as _split


class _GrowableHead(nn.Module):

    def __init__(self, dim: int, width: int, nc: int, gen: torch.Generator):
        super().__init__()
        self.dim, self.nc = dim, nc
        self.gen = gen
        self.in_proj = nn.Linear(dim, width)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(width, nc)

    @property
    def width(self) -> int:
        return self.in_proj.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.act(self.in_proj(x)))

    @torch.no_grad()
    def grow(self, add: int) -> None:
        if add <= 0:
            return
        w_old = self.width
        new_in = nn.Linear(self.dim, w_old + add)
        new_in.weight.zero_()
        new_in.bias.zero_()
        new_in.weight[:w_old].copy_(self.in_proj.weight)
        new_in.bias[:w_old].copy_(self.in_proj.bias)
        scale = (2.0 / self.dim) ** 0.5
        new_in.weight[w_old:].copy_(torch.randn(add, self.dim, generator=self.gen) * scale)
        new_out = nn.Linear(w_old + add, self.nc)
        new_out.weight.zero_()
        new_out.bias.copy_(self.out_proj.bias)
        new_out.weight[:, :w_old].copy_(self.out_proj.weight)
        self.in_proj = new_in
        self.out_proj = new_out


def _fresh_opt(head: nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(head.parameters(), lr=lr)


class B8(Experiment):
    id = "b8_structural_growth"
    metric = ("grown_vs_fixed_final_delta", "capacity_matched", "growth_events")
    baseline = "fixed-capacity shell whose width EQUALS the grown final width (matched-final-capacity)"
    ablation = (
        "experience-driven growth (add units on plateau) vs fixed-final-capacity vs fixed-initial-capacity"
    )
    null_hypothesis = (
        "a shell that GROWS capacity in response to experience (adding hidden units when learning "
        "plateaus) does not beat a fixed shell of the SAME final capacity trained the same way; any gain "
        "is just the extra capacity, not the growth process"
    )
    tier = "cpu-now"

    @staticmethod
    def _train_epochs(head: nn.Module, opt, xtr, ytr, xva, yva, epochs) -> list[float]:
        val_trace = []
        for _ in range(epochs):
            opt.zero_grad()
            F.cross_entropy(head(xtr), ytr).backward()
            opt.step()
            with torch.no_grad():
                val_trace.append(float(F.cross_entropy(head(xva), yva)))
        return val_trace

    @staticmethod
    def _plateaued(trace: list[float], patience: int, min_delta: float) -> bool:
        if len(trace) <= patience:
            return False
        best_before = min(trace[:-patience])
        best_recent = min(trace[-patience:])
        return (best_before - best_recent) < min_delta

    def _run_grown(self, xtr, ytr, xva, yva, dim, nc, e, gen, lr) -> tuple[float, int, int, nn.Module]:
        w_init, w_final = int(e.w_init), int(e.w_final)
        grow_add = int(e.grow_add)
        patience, min_delta = int(e.patience), float(e.min_delta)
        total_epochs = int(e.epochs)

        head = _GrowableHead(dim, w_init, nc, gen)
        opt = _fresh_opt(head, lr)
        trace: list[float] = []
        events = 0
        for _ep in range(total_epochs):
            opt.zero_grad()
            F.cross_entropy(head(xtr), ytr).backward()
            opt.step()
            with torch.no_grad():
                trace.append(float(F.cross_entropy(head(xva), yva)))
            if head.width < w_final and self._plateaued(trace, patience, min_delta):
                add = min(grow_add, w_final - head.width)
                head.grow(add)
                opt = _fresh_opt(head, lr)
                events += 1
                trace = trace[-1:]  # keep the last point as the new baseline, reset the window
        with torch.no_grad():
            acc = float((head(xva).argmax(-1) == yva).float().mean())
        return acc, events, head.width, head

    def _run_fixed(self, xtr, ytr, xva, yva, dim, nc, e, gen, lr, width) -> tuple[float, nn.Module]:
        head = _GrowableHead(dim, width, nc, gen)
        opt = _fresh_opt(head, lr)
        self._train_epochs(head, opt, xtr, ytr, xva, yva, int(e.epochs))
        with torch.no_grad():
            acc = float((head(xva).argmax(-1) == yva).float().mean())
        return acc, head

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc = int(e.dim), int(e.n_classes)
        lr = float(e.lr)
        w_init, w_final = int(e.w_init), int(e.w_final)

        grown_acc, fixed_final_acc, fixed_init_acc = [], [], []
        events_list, grown_widths = [], []
        param_grown, param_fixed_final, param_fixed_init = [], [], []
        for s in seeds:
            seed_everything(s)
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            xtr, ytr, xva, yva = _split(task.x, task.y)

            g_grown = torch.Generator().manual_seed(s + 101)
            ga, ev, gw, ghead = self._run_grown(xtr, ytr, xva, yva, dim, nc, e, g_grown, lr)
            grown_acc.append(ga)
            events_list.append(ev)
            grown_widths.append(gw)
            param_grown.append(param_count(ghead))

            g_ff = torch.Generator().manual_seed(s + 202)
            fa, fhead = self._run_fixed(xtr, ytr, xva, yva, dim, nc, e, g_ff, lr, w_final)
            fixed_final_acc.append(fa)
            param_fixed_final.append(param_count(fhead))

            g_fi = torch.Generator().manual_seed(s + 303)
            ia, ihead = self._run_fixed(xtr, ytr, xva, yva, dim, nc, e, g_fi, lr, w_init)
            fixed_init_acc.append(ia)
            param_fixed_init.append(param_count(ihead))

        gm, ffm, fim = _mean(grown_acc), _mean(fixed_final_acc), _mean(fixed_init_acc)
        spread = max(_spread(grown_acc), _spread(fixed_final_acc))
        grown_vs_fixed_final = gm - ffm  # THE null-testing number (capacity held constant)
        grown_vs_fixed_init = gm - fim  # sanity: did growing do anything vs never growing

        widths_all_final = all(w == w_final for w in grown_widths)
        capacity_matched = bool(param_grown == param_fixed_final) and widths_all_final

        growth_helps = capacity_matched and (grown_vs_fixed_final > spread)
        null = bool((not capacity_matched) or (grown_vs_fixed_final <= spread))
        return {
            "grown_final_acc": round(gm, 4),
            "fixed_final_acc": round(ffm, 4),
            "fixed_initial_acc": round(fim, 4),
            "grown_vs_fixed_final_delta": round(grown_vs_fixed_final, 4),
            "grown_vs_fixed_initial_delta": round(grown_vs_fixed_init, 4),
            "seed_spread": round(spread, 4),
            "growth_events": round(_mean([float(x) for x in events_list]), 3),
            "growth_events_per_seed": list(events_list),
            "grown_final_widths": list(grown_widths),
            "w_init": w_init,
            "w_final": w_final,
            "param_count_grown": list(param_grown),
            "param_count_fixed_final": list(param_fixed_final),
            "param_count_fixed_initial": list(param_fixed_init),
            "capacity_matched": capacity_matched,
            "growth_helps_over_matched_capacity": bool(growth_helps),
            "seeds": list(seeds),
            "null_supported": null,
        }
