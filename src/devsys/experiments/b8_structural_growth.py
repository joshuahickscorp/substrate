"""B8: morphogenetic structural growth vs fixed final capacity. The only catalog mechanism that tests
STRUCTURAL change (adding units), not just reweighting: a direct moldability probe. Question: does a
shell that GROWS hidden units in response to experience (add units when learning plateaus) beat simply
starting at the final size?

Three arms at matched data/seed/budget:
  GROWN: starts at a small width, adds hidden units whenever validation loss plateaus over a patience
         window, up to a final width W_final.
  FIXED-FINAL: a shell of width W_final from the start. THIS is the matched-final-capacity control, the
         only one that isolates the growth PROCESS from just-more-capacity (the b2/EX4 confound).
  FIXED-INITIAL: a shell that stays at the small initial width (shows growing does SOMETHING vs never).

grown_vs_fixed_final_delta = final_acc(grown) minus final_acc(fixed-final): the ONLY number that tests
the null, since it holds capacity constant. We confirm capacity_matched via param_count (grown and
fixed-final end at the same trainable parameter count). growth_events counts how many times it grew.

null_supported = grown ties fixed-final within seed spread (growth-as-process adds nothing over just
being that size), which the corpus capacity-confound experience (b2/EX4) predicts is likely. The grown
arm must NOT win merely by ending larger or training longer, that is the exact confound controlled out:
all three arms train for the SAME number of epochs on the SAME data, and grown/fixed-final end at the
SAME width.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only). No sentience or
agency language. Honest nulls only (null_supported reflects the real toy outcome).
"""

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
from .base import Experiment


def _mean(v: list[float]) -> float:
    return sum(v) / max(1, len(v))


def _spread(v: list[float]) -> float:
    """Seed spread: half the (max - min) range, the band a gain must clear to count."""
    return (max(v) - min(v)) / 2.0 if len(v) > 1 else 0.0


class _GrowableHead(nn.Module):
    """A one-hidden-layer classifier (dim -> width -> nc) whose hidden width can GROW in place. Growth
    adds fresh hidden units: new rows in the input weight (in_proj) and new columns in the output weight
    (out_proj). Existing units keep their learned weights, so growth is structural addition, not a reset.
    New in_proj rows are randomly initialized; new out_proj columns are zeroed so that adding a unit does
    NOT perturb the current function (the network output is unchanged at the growth step, the new unit
    then learns from zero contribution). GELU nonlinearity matches the shell.predictor.mlp block.
    """

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
        """Append `add` hidden units. in_proj gains `add` random rows (new features), out_proj gains
        `add` zeroed columns (so the function is preserved at the growth step). Returns nothing."""
        if add <= 0:
            return
        w_old = self.width
        # new in_proj: [w_old+add, dim], first w_old rows copied, new rows randomly initialized.
        new_in = nn.Linear(self.dim, w_old + add)
        new_in.weight.zero_()
        new_in.bias.zero_()
        new_in.weight[:w_old].copy_(self.in_proj.weight)
        new_in.bias[:w_old].copy_(self.in_proj.bias)
        # small random init for the fresh rows (Kaiming-ish scale for GELU), reproducible via self.gen.
        scale = (2.0 / self.dim) ** 0.5
        new_in.weight[w_old:].copy_(torch.randn(add, self.dim, generator=self.gen) * scale)
        # new out_proj: [nc, w_old+add], first w_old columns copied, new columns ZEROED (function preserved).
        new_out = nn.Linear(w_old + add, self.nc)
        new_out.weight.zero_()
        new_out.bias.copy_(self.out_proj.bias)
        new_out.weight[:, :w_old].copy_(self.out_proj.weight)
        self.in_proj = new_in
        self.out_proj = new_out


def _split(x: torch.Tensor, y: torch.Tensor, frac: float = 0.7):
    cut = int(x.shape[0] * frac)
    return x[:cut], y[:cut], x[cut:], y[cut:]


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
        """Train `head` for a fixed number of epochs; return the per-epoch held-out (val) loss trace."""
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
        """A plateau: over the last `patience` epochs the best val loss did not improve by min_delta vs
        the best seen BEFORE that window (learning stalled)."""
        if len(trace) <= patience:
            return False
        best_before = min(trace[:-patience])
        best_recent = min(trace[-patience:])
        return (best_before - best_recent) < min_delta

    def _run_grown(self, xtr, ytr, xva, yva, dim, nc, e, gen, lr) -> tuple[float, int, int, nn.Module]:
        """GROWN arm: start at w_init, train epoch-by-epoch, and add `grow_add` units whenever the val
        loss plateaus, up to w_final. Trains for the SAME total epochs as the fixed arms (grow_budget).
        Returns (final val accuracy, growth_events, final width, head)."""
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
            # growth trigger: on a plateau, add units (up to the final width) and rebuild the optimizer
            # so it tracks the new parameters. Reset the plateau window after growing.
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
        """A fixed-width arm: build a head at `width`, train the SAME total epochs, no growth."""
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
            # a multi-class stream with a modest separation so the small (w_init) shell is genuinely
            # capacity-starved and growing has room to help IF the process (not just the size) matters.
            task = make_task_stream(
                n_tasks=1,
                dim=dim,
                classes_per_task=nc,
                samples_per_task=int(e.samples),
                separation=float(e.separation),
                seed=s,
            )[0]
            xtr, ytr, xva, yva = _split(task.x, task.y)

            # one shared generator per seed drives every arm's fresh-unit init, so growth randomness is
            # controlled and reproducible. All three arms see the SAME data split and epoch budget.
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

        # capacity match: the grown arm and the fixed-final arm must END at the same trainable param
        # count. Both are dim->w_final->nc heads once grown, so this holds exactly when every seed grew
        # all the way to w_final. Report it honestly per the actual param counts.
        widths_all_final = all(w == w_final for w in grown_widths)
        capacity_matched = bool(param_grown == param_fixed_final) and widths_all_final

        # the grown arm BEATS fixed-final only if it clears the seed spread. That is the only way to
        # reject the null, since capacity is held constant.
        growth_helps = capacity_matched and (grown_vs_fixed_final > spread)
        # null: growth-as-process adds nothing over just being the final size (grown ties fixed-final
        # within seed spread), OR the capacity was not actually matched (then the delta is confounded and
        # cannot reject the null anyway). The b2/EX4 capacity-confound experience predicts this is true.
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
            # null: grown ties fixed-final within seed spread (process adds nothing over size), OR
            # capacity was not matched (delta is confounded).
            "null_supported": null,
        }
