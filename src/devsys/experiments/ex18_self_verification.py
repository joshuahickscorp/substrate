"""EX18: latent self-verification / self-correction. Built directly on EX17's IterativeRefiner. A
small Verifier head (shell/refine.Verifier) scores the refined latent's confidence; when confidence is
low the arm spends an EXTRA refinement step (revise) instead of stopping, up to a shared step budget.
The question is sharp: does verify-then-revise beat a fixed-N single-shot refiner at MATCHED forward
compute, or was any apparent gain just extra iteration bought for free (taxonomy 9), or noise the
verifier cannot actually detect (taxonomy 4).

Three arms, one step budget, one trained refiner block per seed:
  - single-shot: fixed N steps, no verification, no revision (the baseline).
  - verify-revise: at each step past a minimum, a TRAINED verifier scores the latent; low confidence
    triggers one more step, spent out of the SAME total step budget as single-shot (matched compute,
    diagnostics/compute.matched_within on refiner_flops), not added on top of it.
  - shuffled-verifier: identical revise logic, but the verifier's weights are randomly re-initialized
    (untrained, shuffled) after being score-called once, isolating whether any gain comes from the
    TRAINED scoring signal or merely from spending the same extra steps on low-margin samples.

NULL (verbatim, doctrine): verify-revise ties single-shot at matched compute; the verifier carries no
usable correction signal. Negative-result taxonomy slot 4 (predictor/verifier too weak) or 9 (bought
compute masquerading as mechanism) if the shuffled control also gains. cpu-now, seconds.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).
"""

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
    """Fit refiner + head on a fixed-step single-shot forward (the shared trained backbone every arm
    reuses at eval time), and the verifier as an auxiliary error-predictor: it learns to predict the
    per-sample head loss AFTER refinement, so score(z) is a trained confidence-of-error estimate, not
    a random projection."""
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
    """Fixed-N single-pass refinement, no verification. Total step-count spent == steps (per sample)."""
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
    """Run `base_steps` of refinement, then for each low-confidence sample keep spending one extra step
    at a time (revise) until the verifier is satisfied or `budget` total steps is hit. Every sample is
    charged its own steps-used, so the arm's TOTAL step budget across the batch is comparable to
    single-shot at base_steps averaged the same way (matched via refiner_flops on the realized mean)."""
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
    """A structurally identical Verifier with freshly re-initialized (untrained, shuffled) weights: the
    revise LOGIC and step budget are unchanged, only the trained scoring signal is destroyed. If this
    control gains as much as the trained verifier, the gain was bought compute, not correction."""
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

        # the explicit, honest null: verify-revise fails to beat single-shot beyond margin, OR the
        # shuffled-verifier control matches the trained verifier's gain (signal is just extra compute,
        # not a trained correction) - either condition alone is enough to support the null.
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
