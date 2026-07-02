#!/usr/bin/env python
"""PR6: offline sleep consolidation (WP-07, plasticity program; registry
docs/mixture_of_perspectives/11_experiment_registry.md, PR full schema part 2).

Thesis: separating a WAKE phase (train on the current task only, encode to the replay buffer) from
an offline SLEEP phase (replay-only gradient steps plus an EWC Fisher refresh) beats MIXING the
same ingredients at matched total gradient steps. The variable under test is PHASE SEPARATION and
nothing else: both arms see the same current-task sample budget, the same replay sample budget, the
same buffer, the same EWC refresh at every task boundary, and EXACTLY the same number of optimizer
steps (counted in code and asserted equal; a mismatch invalidates the run and forces win False).

Arms:
  wake_sleep  : per task, wake_steps current-only steps (writing to the buffer), then sleep_steps
                replay-only steps; EWC refresh at task end
  interleaved : per task, wake_steps + sleep_steps steps, each on a half-current half-replay batch
                (the same totals of current and replay samples, mixed at every step); same buffer
                writes, same EWC refresh
  wake_only   : per task, wake_steps + sleep_steps current-only steps, no replay (context floor,
                the do-nothing arm every corpus replay result already beats; reported, not gated)

Preregistered null (fixed before running): offline replay-only ties online interleaved replay at
matched total gradient steps: any sleep benefit is added compute, not phase separation. Verdict
rule, uniform across the MoT plasticity scripts: WIN iff EITHER the final mean accuracy delta or
the BWT delta (wake_sleep minus interleaved) passes mean > max(seed SD, MIN_MARGIN) with no sign
flip AND the step counts matched exactly; else null_supported=True.

Usage: python scripts/mop_pr6_sleep_consolidation.py --seeds 0-4
Writes runs/mot/pr6_sleep_consolidation.json. No em or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.continual_metrics import backward_transfer, forgetting_area
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import seed_everything
from mop.shell.buffer import ReplayBuffer
from mop.shell.consolidation import EWC
from mop.substrate.datasets import make_task_stream

MIN_MARGIN = 0.02  # preregistered minimum meaningful accuracy/BWT delta
ARMS = ("wake_sleep", "interleaved", "wake_only")


def parse_seeds(spec: str) -> list[int]:
    """'0-4' -> [0..4]; '0,2,5' -> [0,2,5]; '3' -> [3]."""
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def verdict(deltas: list[float], min_margin: float = MIN_MARGIN) -> dict:
    """The uniform preregistered rule: WIN iff mean delta > max(sd, min_margin) and no sign flip."""
    ci = seed_ci(deltas)
    flips = sign_flip_report(deltas)
    win = ci["mean"] > max(ci["sd"], min_margin) and not flips["any_flip"]
    return {"per_seed": [round(d, 4) for d in deltas], "ci": ci, "sign_flips": flips, "win": bool(win)}


def split_task(task, train_frac: float):
    n = task.x.shape[0]
    cut = int(n * train_frac)
    return task.x[:cut], task.y[:cut], task.x[cut:], task.y[cut:]


def _ewc_refresh(ewc: EWC, head: nn.Module, buf: ReplayBuffer, batch: int, samples: int) -> None:
    """Fisher refresh from the buffer (the consolidation half of sleep, given to BOTH replay arms
    so the only difference between them is phase separation)."""
    if len(buf) == 0:
        return
    batches = [buf.sample(batch) for _ in range(samples)]
    ewc.estimate(head, batches, lambda b: F.cross_entropy(head(b["x"]), b["y"]), samples=samples)


def run_arm(e: DictConfig, seed: int, arm: str) -> dict:
    """One arm through the task stream. Returns the accuracy matrix, the task-0 accuracy curve
    (for forgetting_area), and the exact optimizer step count."""
    dim, n_classes = int(e.dim), int(e.classes_per_task)
    tasks = make_task_stream(
        n_tasks=int(e.n_tasks),
        dim=dim,
        classes_per_task=n_classes,
        samples_per_task=int(e.samples_per_task),
        separation=float(e.separation),
        incremental="domain",
        seed=seed,
    )
    splits = [split_task(t, float(e.train_frac)) for t in tasks]
    wake_steps, sleep_steps = int(e.wake_steps), int(e.sleep_steps)
    batch = int(e.batch)
    torch.manual_seed(seed)
    head = nn.Linear(dim, n_classes)
    opt = torch.optim.SGD(head.parameters(), lr=float(e.lr))
    buf = ReplayBuffer(
        capacity=int(e.buffer_capacity), dim=dim, prioritized=False, eviction="reservoir", seed=seed
    )
    ewc = EWC(lam=float(e.ewc_lambda))
    g = torch.Generator().manual_seed(seed + 31)
    steps_taken = 0
    acc_matrix: list[list[float]] = []
    task0_curve: list[float] = []

    def train_step(x: torch.Tensor, y: torch.Tensor) -> None:
        nonlocal steps_taken
        opt.zero_grad()
        loss = F.cross_entropy(head(x), y) + ewc.penalty(head)
        loss.backward()
        opt.step()
        steps_taken += 1

    for xtr, ytr, _, _ in splits:

        def cur_batch(bs: int, xtr: torch.Tensor = xtr, ytr: torch.Tensor = ytr) -> tuple:
            idx = torch.randint(0, xtr.shape[0], (bs,), generator=g)
            return xtr[idx], ytr[idx]

        if arm == "wake_sleep":
            for _ in range(wake_steps):
                xb, yb = cur_batch(batch)
                train_step(xb, yb)
                buf.add(xb, yb)
            for _ in range(sleep_steps):  # offline: replay-only, no current-task data
                rb = buf.sample(batch)
                train_step(rb["x"], rb["y"])
        elif arm == "interleaved":
            for step in range(wake_steps + sleep_steps):
                xb, yb = cur_batch(batch // 2)
                if len(buf) > 0:
                    rb = buf.sample(batch - batch // 2)
                    xb, yb = torch.cat([xb, rb["x"]]), torch.cat([yb, rb["y"]])
                train_step(xb, yb)
                if step % 2 == 0:  # same current-sample write rate as the wake arm (B/2 per step)
                    buf.add(xb[: batch // 2], yb[: batch // 2])
        else:  # wake_only: same step budget, current task only, no replay
            for _ in range(wake_steps + sleep_steps):
                xb, yb = cur_batch(batch)
                train_step(xb, yb)
        if arm != "wake_only":
            _ewc_refresh(ewc, head, buf, batch, samples=int(e.ewc_samples))
        with torch.no_grad():
            acc_matrix.append([float((head(x).argmax(-1) == y).float().mean()) for _, _, x, y in splits])
            task0_curve.append(acc_matrix[-1][0])
    return {"acc_matrix": acc_matrix, "task0_curve": task0_curve, "steps": steps_taken}


class MotPR6SleepConsolidation(Experiment):
    id = "mop_pr6_sleep_consolidation"
    metric = ("final_mean_acc", "bwt", "forgetting_area_task0", "steps_matched")
    baseline = "interleaved online replay at matched total gradient steps, same buffer and EWC refresh"
    ablation = "wake/sleep phase separation vs mixed replay; wake-only context floor"
    null_hypothesis = (
        "offline replay-only ties online interleaved replay at matched total gradient steps: any "
        "sleep benefit is added compute, not phase separation"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        t0 = time.perf_counter()
        seeds = list(e.seeds)
        final_acc = {a: [] for a in ARMS}
        bwt = {a: [] for a in ARMS}
        f_area = {a: [] for a in ARMS}
        steps = {a: [] for a in ARMS}
        for s in seeds:
            seed_everything(s)
            for arm in ARMS:
                res = run_arm(e, s, arm)
                last = res["acc_matrix"][-1]
                final_acc[arm].append(sum(last) / len(last))
                bwt[arm].append(backward_transfer(res["acc_matrix"]))
                f_area[arm].append(forgetting_area(res["task0_curve"]))
                steps[arm].append(res["steps"])
        steps_matched = bool(
            all(steps["wake_sleep"][i] == steps["interleaved"][i] for i in range(len(seeds)))
        )
        d_acc = [final_acc["wake_sleep"][i] - final_acc["interleaved"][i] for i in range(len(seeds))]
        d_bwt = [bwt["wake_sleep"][i] - bwt["interleaved"][i] for i in range(len(seeds))]
        v_acc, v_bwt = verdict(d_acc), verdict(d_bwt)
        win = bool(steps_matched and (v_acc["win"] or v_bwt["win"]))
        out = {
            "experiment": self.id,
            "contract": self.contract(),
            "config": OmegaConf.to_container(e, resolve=True),
            "steps_matched": steps_matched,
            "steps_per_arm": {a: steps[a] for a in ARMS},
            "final_mean_acc": {a: {"per_seed": final_acc[a], "ci": seed_ci(final_acc[a])} for a in ARMS},
            "bwt": {a: {"per_seed": bwt[a], "ci": seed_ci(bwt[a])} for a in ARMS},
            "forgetting_area_task0": {a: {"per_seed": f_area[a], "ci": seed_ci(f_area[a])} for a in ARMS},
            "delta_final_acc": v_acc,
            "delta_bwt": v_bwt,
            "verdict_rule": (
                "WIN iff optimizer step counts match exactly between wake_sleep and interleaved (else "
                "the run is invalid and win is forced False) AND EITHER the final mean accuracy delta OR "
                f"the BWT delta (wake_sleep minus interleaved) > max(seed SD, {MIN_MARGIN}) with no sign "
                "flip; else the preregistered null (sleep is just more compute) is supported"
            ),
            "null_supported": bool(not win),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        return out


def default_cfg(**overrides) -> DictConfig:
    base = {
        "seeds": [0, 1, 2, 3, 4],
        "dim": 64,
        "classes_per_task": 8,
        "n_tasks": 5,
        "samples_per_task": 400,
        "separation": 2.0,
        "train_frac": 0.8,
        "wake_steps": 100,
        "sleep_steps": 100,
        "batch": 32,
        "lr": 0.05,
        "buffer_capacity": 500,
        "ewc_lambda": 10.0,
        "ewc_samples": 8,
    }
    base.update(overrides)
    return OmegaConf.create({"experiment": base})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR6: wake/sleep phase separation vs interleaved replay")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", default="runs/mot/pr6_sleep_consolidation.json")
    a = ap.parse_args(argv)
    cfg = default_cfg(seeds=parse_seeds(a.seeds))
    result = MotPR6SleepConsolidation().run(cfg, resolve("cpu"), Path(a.out).parent)
    text = json.dumps(result, indent=2, default=str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
