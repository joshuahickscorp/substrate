
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from ..devices import DeviceInfo
from ..diagnostics.substrate_ablation import frozen_random_projection
from ..seeding import seed_everything
from ..shell.heads import ClassHead
from ..substrate.datasets import make_task_stream
from .base import Experiment, _mean, _std


def _trapz_area(xs: list[float], up: list[float], down: list[float]) -> float:
    diff = [u - d for u, d in zip(up, down, strict=True)]
    area = 0.0
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        area += 0.5 * (diff[i] + diff[i + 1]) * dx
    return area


class Y4(Experiment):
    id = "y4_hysteresis"
    metric = ("hysteresis_area", "path_dependent", "up_vs_down_gap")
    baseline = (
        "the down-sweep retention curve compared against the up-sweep curve at each matched parameter value"
    )
    ablation = (
        "sweep direction (interference intensity swept up then back down) on the same carried-forward "
        "shell state, real substrate vs a frozen_random_projection substrate"
    )
    null_hypothesis = (
        "retention is a single-valued function of the swept schedule parameter: sweeping it up then back "
        "down traces the same curve (no hysteresis, no path dependence); the shell has no memory of its "
        "trajectory beyond its current parameter value"
    )
    tier = "cpu-now"

    def _sweep(
        self,
        anchor_x: torch.Tensor,
        anchor_y: torch.Tensor,
        interfere_tasks: list,
        grid: list[float],
        dim: int,
        nc: int,
        base_lr: float,
        anchor_epochs: int,
        interfere_epochs: int,
        seed: int,
    ) -> tuple[list[float], list[float]]:
        cut = int(anchor_x.shape[0] * 0.7)
        xtr, ytr, xte, yte = anchor_x[:cut], anchor_y[:cut], anchor_x[cut:], anchor_y[cut:]

        seed_everything(seed)
        head = ClassHead(dim, nc, depth=0)  # linear anchor head, the whole trainable shell state
        opt = torch.optim.Adam(head.parameters(), lr=base_lr)
        for _ in range(anchor_epochs):
            opt.zero_grad()
            F.cross_entropy(head(xtr), ytr).backward()
            opt.step()

        def retention() -> float:
            with torch.no_grad():
                return float((head(xte).argmax(-1) == yte).float().mean())

        def interfere(intensity: float, task_idx: int) -> None:
            task = interfere_tasks[task_idx % len(interfere_tasks)]
            xi, yi = task.x, task.y
            for grp in opt.param_groups:
                grp["lr"] = intensity
            for _ in range(interfere_epochs):
                opt.zero_grad()
                F.cross_entropy(head(xi), yi).backward()
                opt.step()

        up = []
        step = 0
        for intensity in grid:
            interfere(intensity, step)
            up.append(retention())
            step += 1
        down_rev = []
        for intensity in reversed(grid):
            interfere(intensity, step)
            down_rev.append(retention())
            step += 1
        down = list(reversed(down_rev))  # realign to ascending grid so up[i] and down[i] share a value
        return up, down

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, nc = int(e.dim), int(e.n_classes)
        samples = int(e.samples)
        separation = float(e.separation)
        base_lr = float(e.base_lr)
        anchor_epochs = int(e.anchor_epochs)
        interfere_epochs = int(e.interfere_epochs)
        n_interfere = int(e.n_interfere_tasks)
        grid = [float(v) for v in e.interference_grid]

        real_areas, real_gaps = [], []
        fr_areas, fr_gaps = [], []
        for s in seeds:
            seed_everything(s)
            anchor = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=samples,
                separation=separation,
                seed=s,
            )[0]
            interfere_tasks = make_task_stream(
                n_tasks=n_interfere,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=samples,
                separation=separation,
                incremental="domain",
                seed=s + 101,
            )

            up_r, down_r = self._sweep(
                anchor.x,
                anchor.y,
                interfere_tasks,
                grid,
                dim,
                nc,
                base_lr,
                anchor_epochs,
                interfere_epochs,
                s,
            )
            real_areas.append(_trapz_area(grid, up_r, down_r))
            real_gaps.append(_mean([u - d for u, d in zip(up_r, down_r, strict=True)]))

            fr_anchor_x = frozen_random_projection(anchor.x, seed=s)
            fr_interfere = []
            for t in interfere_tasks:
                ft = type(t)(
                    name=t.name,
                    x=frozen_random_projection(t.x, seed=s),
                    y=t.y,
                    n_classes=t.n_classes,
                    task_id=t.task_id,
                )
                fr_interfere.append(ft)
            up_f, down_f = self._sweep(
                fr_anchor_x,
                anchor.y,
                fr_interfere,
                grid,
                dim,
                nc,
                base_lr,
                anchor_epochs,
                interfere_epochs,
                s,
            )
            fr_areas.append(_trapz_area(grid, up_f, down_f))
            fr_gaps.append(_mean([u - d for u, d in zip(up_f, down_f, strict=True)]))

        span = max(grid) - min(grid) if len(grid) > 1 else 1.0

        real_area = _mean(real_areas)
        real_area_std = _std(real_areas)
        real_gap = _mean(real_gaps)
        fr_area = _mean(fr_areas)
        fr_gap = _mean(fr_gaps)

        corrected_area = real_area - fr_area
        seed_spread = max(real_area_std, float(e.area_std_floor))
        area_beats_zero = abs(real_area) > float(e.spread_k) * seed_spread
        loop_survives_control = abs(corrected_area) > float(e.control_margin) * abs(span)

        path_dependent = bool(area_beats_zero and loop_survives_control)
        null = bool((not area_beats_zero) or (not loop_survives_control))

        return {
            "hysteresis_area": round(real_area, 5),
            "hysteresis_area_std": round(real_area_std, 5),
            "frozen_random_hysteresis_area": round(fr_area, 5),
            "substrate_corrected_area": round(corrected_area, 5),
            "up_vs_down_gap": round(real_gap, 5),
            "frozen_random_up_vs_down_gap": round(fr_gap, 5),
            "path_dependent": path_dependent,
            "area_beats_seed_spread": bool(area_beats_zero),
            "loop_survives_frozen_random": bool(loop_survives_control),
            "parameter_span": round(float(span), 5),
            "interference_grid": list(grid),
            "seeds": list(seeds),
            "null_supported": null,
        }
