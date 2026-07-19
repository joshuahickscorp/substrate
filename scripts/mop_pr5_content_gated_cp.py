#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.continual_metrics import (
    LRIntegralAccumulator,
    adaptation_speed,
    backward_transfer,
    forward_transfer,
)
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import parse_seeds, seed_everything
from mop.shell.neuromod import RunningStat
from mop.shell.plasticity import PlasticityController
from mop.substrate.datasets import make_task_stream

MIN_MARGIN = 0.02  # preregistered minimum meaningful BWT delta
MIN_STEPS = 1.0  # preregistered minimum meaningful adaptation-steps delta (eval units)
ARMS = ("content", "cosine", "constant", "content_shuffled")
BASELINES = ("cosine", "constant")


def verdict(deltas: list[float], min_margin: float) -> dict:
    ci = seed_ci(deltas)
    flips = sign_flip_report(deltas)
    win = ci["mean"] > max(ci["sd"], min_margin) and not flips["any_flip"]
    return {"per_seed": [round(d, 4) for d in deltas], "ci": ci, "sign_flips": flips, "win": bool(win)}


def split_task(task, train_frac: float):
    n = task.x.shape[0]
    cut = int(n * train_frac)
    return task.x[:cut], task.y[:cut], task.x[cut:], task.y[cut:]


class ContentSchedule:

    def __init__(self, e: DictConfig, seed: int):
        self.ctl = PlasticityController(
            OmegaConf.create(
                {"schedule": "soft", "lr": float(e.base_lr), "rigidity": 0.0, "pnn_fraction": 0.0}
            ),
            seed=seed,
        )
        self.ctl.reopen_threshold = float(e.reopen_threshold)
        self.stat = RunningStat(momentum=float(e.surprise_momentum))

    def lr(self, progress: float, batch_loss: float) -> float:
        z = self.stat.update(batch_loss)
        return self.ctl.lr(progress, signal=max(0.0, z))


def run_arm(e: DictConfig, seed: int, arm: str, task_order: list[int], budget: float | None) -> dict:
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
    tasks = [tasks[i] for i in task_order]
    splits = [split_task(t, float(e.train_frac)) for t in tasks]
    steps_per_task = int(e.steps_per_task)
    n_steps = steps_per_task * len(tasks)
    torch.manual_seed(seed)
    head = nn.Linear(dim, n_classes)
    opt = torch.optim.SGD(head.parameters(), lr=float(e.base_lr))
    schedule = ContentSchedule(e, seed) if arm.startswith("content") else None
    if arm == "cosine":
        weights = [0.5 * (1 + math.cos(math.pi * t / max(1, n_steps - 1))) for t in range(n_steps)]
        amp = budget / sum(weights)
    lrint = LRIntegralAccumulator()
    g = torch.Generator().manual_seed(seed + 31)
    acc_matrix: list[list[float]] = []
    curves: list[list[float]] = []
    step = 0
    for ti, (xtr, ytr, _, _) in enumerate(splits):
        curve: list[float] = []
        for k in range(steps_per_task):
            idx = torch.randint(0, xtr.shape[0], (int(e.batch),), generator=g)
            loss = F.cross_entropy(head(xtr[idx]), ytr[idx])
            progress = step / max(1, n_steps - 1)
            if arm.startswith("content"):
                lr = schedule.lr(progress, float(loss))
            elif arm == "cosine":
                lr = amp * weights[step]
            else:  # constant, tuned to the matched budget
                lr = budget / n_steps
            for group in opt.param_groups:
                group["lr"] = lr
            opt.zero_grad()
            loss.backward()
            opt.step()
            lrint.add(lr, steps=1, partition=f"task{ti}")
            step += 1
            if (k + 1) % int(e.eval_every) == 0:
                with torch.no_grad():
                    xte, yte = splits[ti][2], splits[ti][3]
                    curve.append(float((head(xte).argmax(-1) == yte).float().mean()))
        curves.append(curve)
        with torch.no_grad():
            acc_matrix.append([float((head(x).argmax(-1) == y).float().mean()) for _, _, x, y in splits])
    return {"acc_matrix": acc_matrix, "curves": curves, "lrint": lrint}


class MotPR5ContentGatedCP(Experiment):
    id = "mop_pr5_content_gated_cp"
    metric = ("bwt", "fwt", "adaptation_steps", "lr_integral_matched")
    baseline = "cosine decay and tuned constant LR, both solved to the content arm's exact LR-integral"
    ablation = "surprise-triggered reopening vs clock-gated schedules; shuffled task order control"
    null_hypothesis = (
        "at matched LR-integral the content-gated schedule shows no retention or reopening advantage "
        "over cosine decay or tuned constant LR: pure LR annealing (the e3/d6 negative)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        t0 = time.perf_counter()
        seeds = list(e.seeds)
        n_tasks = int(e.n_tasks)
        chance = 1.0 / int(e.classes_per_task)
        bwt = {a: [] for a in ARMS}
        fwt = {a: [] for a in ARMS}
        adapt = {a: [] for a in ARMS}
        matched_all = []
        for s in seeds:
            seed_everything(s)
            order = list(range(n_tasks))
            content = run_arm(e, s, "content", order, budget=None)
            budget = content["lrint"].total()
            cosine = run_arm(e, s, "cosine", order, budget=budget)
            constant = run_arm(e, s, "constant", order, budget=budget)
            shuf_order = torch.randperm(n_tasks, generator=torch.Generator().manual_seed(s + 53)).tolist()
            shuffled = run_arm(e, s, "content_shuffled", shuf_order, budget=None)
            matched = bool(
                content["lrint"].matched(cosine["lrint"]) and content["lrint"].matched(constant["lrint"])
            )
            matched_all.append(matched)
            for arm, res in (
                ("content", content),
                ("cosine", cosine),
                ("constant", constant),
                ("content_shuffled", shuffled),
            ):
                bwt[arm].append(backward_transfer(res["acc_matrix"]))
                fwt[arm].append(forward_transfer(res["acc_matrix"], [chance] * n_tasks))
                later = [adaptation_speed(c, target_frac=0.9)["steps"] for c in res["curves"][1:]] or [0.0]
                adapt[arm].append(sum(later) / len(later))
        d_ret = [bwt["content"][i] - max(bwt[b][i] for b in BASELINES) for i in range(len(seeds))]
        d_reo = [min(adapt[b][i] for b in BASELINES) - adapt["content"][i] for i in range(len(seeds))]
        v_ret = verdict(d_ret, MIN_MARGIN)
        v_reo = verdict(d_reo, MIN_STEPS)
        all_matched = bool(all(matched_all))
        win = bool(all_matched and (v_ret["win"] or v_reo["win"]))
        out = {
            "experiment": self.id,
            "contract": self.contract(),
            "config": OmegaConf.to_container(e, resolve=True),
            "lr_integral_matched": all_matched,
            "lr_integral_matched_per_seed": matched_all,
            "bwt": {a: {"per_seed": bwt[a], "ci": seed_ci(bwt[a])} for a in ARMS},
            "fwt": {a: {"per_seed": fwt[a], "ci": seed_ci(fwt[a])} for a in ARMS},
            "adaptation_steps_later_tasks": {
                a: {"per_seed": adapt[a], "ci": seed_ci(adapt[a])} for a in ARMS
            },
            "retention_delta": v_ret,
            "reopening_delta_steps": v_reo,
            "verdict_rule": (
                "WIN iff LR-integrals match (tol 0.02, LRIntegralAccumulator.matched, else the run is "
                "invalid and win is forced False) AND EITHER the retention delta (content BWT minus best "
                f"baseline BWT) > max(seed SD, {MIN_MARGIN}) OR the reopening delta (best baseline "
                f"adaptation steps minus content steps) > max(seed SD, {MIN_STEPS}), each with no sign "
                "flip; else the preregistered null (pure LR annealing, the e3/d6 negative) is supported"
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
        "steps_per_task": 150,
        "batch": 32,
        "base_lr": 0.05,
        "reopen_threshold": 1.0,
        "surprise_momentum": 0.95,
        "eval_every": 10,
    }
    base.update(overrides)
    return OmegaConf.create({"experiment": base})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR5: content-gated critical period at matched LR-integral")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", default="runs/mot/pr5_content_gated_cp.json")
    a = ap.parse_args(argv)
    cfg = default_cfg(seeds=parse_seeds(a.seeds))
    result = MotPR5ContentGatedCP().run(cfg, resolve("cpu"), Path(a.out).parent)
    text = json.dumps(result, indent=2, default=str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
