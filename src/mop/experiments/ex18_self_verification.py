
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from ..devices import DeviceInfo
from ..diagnostics.compute import matched_within, refiner_flops
from ..seeding import seed_everything
from ..shell.refine import IterativeRefiner, Verifier
from .base import Experiment


def _fit(refiner: nn.Module, verifier: Verifier, head: nn.Module, x, y, epochs: int, lr: float) -> None:
    opt = torch.optim.Adam([*refiner.parameters(), *head.parameters(), *verifier.parameters()], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        z, _ = refiner(x)
        logits = head(z)
        task_loss = F.cross_entropy(logits, y)
        with torch.no_grad():
            per_sample_err = F.cross_entropy(logits, y, reduction="none").detach()
        verifier_loss = F.mse_loss(verifier.score(z), per_sample_err)
        (task_loss + verifier_loss).backward()
        opt.step()


@torch.no_grad()
def _single_shot(refiner: IterativeRefiner, head: nn.Module, x, y, steps: int) -> tuple[float, int]:
    z, _ = refiner(x, max_steps=steps)
    acc = float((head(z).argmax(-1) == y).float().mean())
    total_steps = steps * x.shape[0]
    return acc, total_steps


@torch.no_grad()
def _verify_revise(
    refiner: IterativeRefiner,
    verifier: Verifier,
    head: nn.Module,
    x,
    y,
    base_steps: int,
    budget: int,
    threshold: float,
) -> tuple[float, int, list[int]]:
    z = x
    used = torch.zeros(x.shape[0], dtype=torch.long)
    for _ in range(base_steps):
        z = z + refiner._update(z)
    used += base_steps
    active = torch.ones(x.shape[0], dtype=torch.bool)
    while True:
        conf_err = verifier.score(z)  # predicted error, high == low confidence
        low_conf = active & (conf_err > threshold) & (used < budget)
        if not low_conf.any():
            break
        z = torch.where(low_conf.unsqueeze(-1), z + refiner._update(z), z)
        used = torch.where(low_conf, used + 1, used)
        active = used < budget
        if not active.any():
            break
    acc = float((head(z).argmax(-1) == y).float().mean())
    total_steps = int(used.sum())
    halting_steps = used.tolist()
    return acc, total_steps, halting_steps


def _shuffle_verifier(dim: int, hidden: int) -> Verifier:
    return Verifier(dim, hidden=hidden)


class EX18(Experiment):
    id = "ex18_self_verification"
    metric = ("accuracy", "self_correction_gain", "halting_step_distribution")
    baseline = "single-shot fixed-N refiner, no verification, no revision, matched forward-step budget"
    ablation = (
        "verify-revise (trained verifier gates extra steps) vs shuffled-verifier control, same step budget"
    )
    null_hypothesis = (
        "verify-revise ties single-shot at matched compute; the verifier carries no usable correction signal"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from ..substrate.datasets import make_task_stream

        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden = int(e.dim), int(e.hidden)
        base_steps, budget = int(e.base_steps), int(e.step_budget)
        threshold = float(e.verify_threshold)
        margin = float(e.margin)
        epochs, lr = int(e.epochs), float(e.lr)

        single_acc, verify_acc, shuffled_acc = [], [], []
        single_steps, verify_steps, shuffled_steps = [], [], []
        all_halts: list[int] = []

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

            verifier_hidden = max(32, hidden // 2)
            refiner = IterativeRefiner(dim, hidden, budget)
            verifier = Verifier(dim, hidden=verifier_hidden)
            head = nn.Linear(dim, nc)
            _fit(refiner, verifier, head, xtr, ytr, epochs, lr)

            acc_s, steps_s = _single_shot(refiner, head, xte, yte, base_steps)
            acc_v, steps_v, halts_v = _verify_revise(
                refiner, verifier, head, xte, yte, base_steps, budget, threshold
            )
            shuffled = _shuffle_verifier(dim, verifier_hidden)
            acc_sh, steps_sh, _ = _verify_revise(
                refiner, shuffled, head, xte, yte, base_steps, budget, threshold
            )

            single_acc.append(acc_s)
            verify_acc.append(acc_v)
            shuffled_acc.append(acc_sh)
            single_steps.append(steps_s)
            verify_steps.append(steps_v)
            shuffled_steps.append(steps_sh)
            all_halts.extend(halts_v)

        n = len(seeds)
        sm = sum(single_acc) / n
        vm = sum(verify_acc) / n
        shm = sum(shuffled_acc) / n
        gain = vm - sm
        shuffled_gain = shm - sm

        flops_single = refiner_flops(dim, hidden, base_steps)
        flops_verify = refiner_flops(dim, hidden, budget)  # upper bound: revise never exceeds budget
        compute = matched_within(flops_single, flops_verify)

        halts_sorted = sorted(all_halts)
        halting_hist = {str(k): int(halts_sorted.count(k)) for k in range(base_steps, budget + 1)}

        seed_spread = (max(verify_acc) - min(verify_acc)) if n > 1 else 0.0

        gain_within_margin = bool(gain <= margin)
        shuffled_matches_trained = bool(abs(gain - shuffled_gain) <= margin)
        null_supported = bool(gain_within_margin or shuffled_matches_trained)

        out = {
            "single_shot_acc_mean": round(sm, 4),
            "verify_revise_acc_mean": round(vm, 4),
            "shuffled_verifier_acc_mean": round(shm, 4),
            "accuracy": round(vm, 4),
            "self_correction_gain": round(gain, 4),
            "shuffled_verifier_gain": round(shuffled_gain, 4),
            "margin": margin,
            "seed_spread": round(seed_spread, 4),
            "compute_matched": compute,
            "mean_steps_single_shot": round(sum(single_steps) / n / xte.shape[0], 3),
            "mean_steps_verify_revise": round(sum(verify_steps) / n / xte.shape[0], 3),
            "mean_steps_shuffled_verifier": round(sum(shuffled_steps) / n / xte.shape[0], 3),
            "halting_step_distribution": halting_hist,
            "seeds": list(seeds),
            "null_supported": null_supported,
            "verifier_wins_at_equal_or_less_compute": bool(gain > margin and not shuffled_matches_trained),
        }
        return out
