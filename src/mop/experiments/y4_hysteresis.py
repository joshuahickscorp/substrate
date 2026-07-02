"""Y4: hysteresis / phase transition in retention. A dynamical-systems moldability signature that the
Y-series never built: PATH DEPENDENCE. The question is whether the shell's retention of an anchor task
depends on the TRAJECTORY the schedule took, not just on where the schedule parameter currently sits.

Mechanism. One shell (a linear anchor head) learns an anchor task, then rides a slow sweep of a single
schedule parameter that plausibly drives a forgetting transition (here the interference intensity, the
learning rate spent on an interfering domain-incremental distractor). The parameter is swept UP (weak to
strong interference) and then back DOWN (strong to weak) on the SAME carried-forward shell state (weights
never reset between steps, that is the whole point). At each parameter value on both legs we measure
anchor retention (held-out anchor accuracy). If retention were a single-valued function of the parameter,
the up-leg and down-leg curves would coincide (no loop). A gap between them (the down-leg staying low
after strong interference has already erased the anchor, because weak interference on the way back does
not restore it) is a hysteresis loop: memory of the trajectory beyond the current parameter value.

hysteresis_area is the signed area between the up-sweep and down-sweep retention curves over the shared
parameter grid (0 = single-valued, no path dependence; large = a strong loop). up_vs_down_gap is the
mean retention difference at matched parameter values. Standing control: the identical sweep on a
frozen_random_projection substrate, so a loop that ALSO appears under frozen-random is a generic
optimizer / SGD-trajectory artifact of continual training, not a substrate property.

null_supported = the hysteresis area is within the seed spread of zero (no path dependence), OR the same
loop appears under frozen-random (the substrate-corrected area is within seed spread of zero). We report
path_dependent honestly from the numbers.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only). No sentience or
agency language. Honest nulls only.
"""

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
from .base import Experiment


def _mean(v: list[float]) -> float:
    return sum(v) / max(1, len(v))


def _std(v: list[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return (sum((a - m) ** 2 for a in v) / (len(v) - 1)) ** 0.5


def _trapz_area(xs: list[float], up: list[float], down: list[float]) -> float:
    """Signed area between the up-leg and down-leg retention curves over a shared parameter grid xs.
    Trapezoidal integral of (up - down) with respect to the parameter. Positive means the up-leg
    retention sat above the down-leg (the down-leg stayed forgotten), the expected loop orientation."""
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
        """Run one up-then-down interference sweep on a single carried-forward head. Returns the
        (up_retention, down_retention) curves aligned to `grid`. The head is trained once on the anchor,
        then at every parameter value it takes `interfere_epochs` steps of interference at that intensity
        (the swept parameter is the interference learning rate), and anchor retention is read out. Weights
        are never reset across the sweep: the state carried into each step is the state the trajectory
        produced, which is what makes a loop possible."""
        cut = int(anchor_x.shape[0] * 0.7)
        xtr, ytr, xte, yte = anchor_x[:cut], anchor_y[:cut], anchor_x[cut:], anchor_y[cut:]

        seed_everything(seed)
        head = ClassHead(dim, nc, depth=0)  # linear anchor head, the whole trainable shell state
        opt = torch.optim.Adam(head.parameters(), lr=base_lr)
        for _ in range(anchor_epochs):
            opt.zero_grad()
            F.cross_entropy(head(xtr), ytr).backward()
            opt.step()

        # a rotating pool of interfering domains (shared label space, different geometry): training on
        # them at a high learning rate overwrites the anchor mapping, the forgetting driver.
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

        # UP leg: weak to strong interference, carried forward.
        up = []
        step = 0
        for intensity in grid:
            interfere(intensity, step)
            up.append(retention())
            step += 1
        # DOWN leg: strong to weak interference, continuing from the SAME state (no reset).
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
            # anchor task: a single well-separated classification task the shell must retain.
            anchor = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=samples,
                separation=separation,
                seed=s,
            )[0]
            # interfering domains: same label space, independent geometry (the reliable forgetting regime).
            interfere_tasks = make_task_stream(
                n_tasks=n_interfere,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=samples,
                separation=separation,
                incremental="domain",
                seed=s + 101,
            )

            # REAL substrate sweep.
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

            # FROZEN-RANDOM control: identical sweep on a fixed random projection of every input. A loop
            # that survives here is a generic continual-SGD trajectory artifact, not a substrate property.
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

        # parameter span, used to normalize the area into a mean-retention-gap scale (per unit parameter
        # -> per parameter, so the area and the mean gap are comparable magnitudes).
        span = max(grid) - min(grid) if len(grid) > 1 else 1.0

        real_area = _mean(real_areas)
        real_area_std = _std(real_areas)
        real_gap = _mean(real_gaps)
        fr_area = _mean(fr_areas)
        fr_gap = _mean(fr_gaps)

        # substrate-corrected loop: the real area beyond what frozen-random already produces.
        corrected_area = real_area - fr_area
        # seed spread of zero: an area is "within seed spread of zero" if its magnitude does not clear a
        # few times the cross-seed standard deviation (a loop must be larger than seed noise to count).
        seed_spread = max(real_area_std, float(e.area_std_floor))
        area_beats_zero = abs(real_area) > float(e.spread_k) * seed_spread
        # the loop must ALSO exceed the frozen-random loop by a margin to be a substrate property.
        loop_survives_control = abs(corrected_area) > float(e.control_margin) * abs(span)

        path_dependent = bool(area_beats_zero and loop_survives_control)
        # null: area within seed spread of zero (no path dependence), OR the same loop appears under
        # frozen-random (the substrate-corrected area does not survive the control).
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
