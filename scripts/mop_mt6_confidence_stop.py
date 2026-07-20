#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from mop_mt5_adaptive_halting import (  # the WP-03 shared halting harness
    ENTROPY_FLOOR_BITS,
    FLOP_TOL,
    default_cfg,
    make_graded_split,
    parse_seeds,
    resolve_out,
    step_entropy_bits,
    train_refiner_and_halt,
)
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.compute import matched_within, refiner_flops
from mop.diagnostics.riskcov import auroc, seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import seed_everything
from mop.shell.refine import IterativeRefiner


@torch.no_grad()
def norm_rule_eval(
    refiner: IterativeRefiner, head: nn.Module, x, y, threshold: float, max_steps: int
) -> tuple[float, float, torch.Tensor]:
    z = x.clone()
    active = torch.ones(x.shape[0], dtype=torch.bool)
    used = torch.zeros(x.shape[0])
    for _ in range(int(max_steps)):
        u = refiner._update(z)
        z = torch.where(active.unsqueeze(-1), z + u, z)
        used = used + active.float()
        rel = u.norm(dim=-1) / (z.norm(dim=-1) + 1e-9)
        active = active & (rel >= threshold)
        if not active.any():
            break
    acc = float((head(z).argmax(-1) == y).float().mean())
    return acc, float(used.mean()), used


@torch.no_grad()
def tune_norm_threshold(
    refiner: IterativeRefiner, head: nn.Module, xtr, ytr, target_mean: float, max_steps: int
) -> float:
    lo, hi = 0.0, 10.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        _, mean_mid, _ = norm_rule_eval(refiner, head, xtr, ytr, mid, max_steps)
        if mean_mid > target_mean:
            lo = mid  # halting too late, raise the threshold
        else:
            hi = mid
    return 0.5 * (lo + hi)


@torch.no_grad()
def signals_at_depth(
    refiner: IterativeRefiner, head: nn.Module, x, y, depth: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = x.clone()
    u = torch.zeros_like(z)
    for _ in range(max(1, int(depth))):
        u = refiner._update(z)
        z = z + u
    correct = (head(z).argmax(-1) == y).float()
    assert refiner.halt_head is not None
    trained_sig = torch.sigmoid(refiner.halt_head(z)).squeeze(-1)
    free_sig = -(u.norm(dim=-1) / (z.norm(dim=-1) + 1e-9))
    return trained_sig, free_sig, correct, z


def shuffled_label_auroc(z: torch.Tensor, correct: torch.Tensor, seed: int, epochs: int, lr: float) -> float:
    seed_everything(seed + 13)
    g = torch.Generator().manual_seed(seed + 13)
    perm_labels = correct[torch.randperm(correct.shape[0], generator=g)]
    head = nn.Linear(z.shape[1], 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    zd = z.detach()
    for _ in range(epochs):
        opt.zero_grad()
        F.binary_cross_entropy(torch.sigmoid(head(zd)).squeeze(-1), perm_labels).backward()
        opt.step()
    with torch.no_grad():
        scores = torch.sigmoid(head(zd)).squeeze(-1)
    return auroc(scores, correct)


class MT6ConfidenceStop(Experiment):
    id = "mop_mt6_confidence_stop"
    metric = ("trained_halt_acc", "free_rule_acc", "delta", "auroc_trained", "auroc_free")
    baseline = "free update-norm early-exit rule, threshold tuned on train to the same mean steps"
    ablation = "trained halt head vs free rule at matched mean FLOPs; shuffled-halt-labels control"
    null_hypothesis = (
        "the trained halt head ties the free update-norm rule at matched mean FLOPs; confidence is "
        "just the latent ceasing to move"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden = int(e.dim), int(e.hidden)
        max_steps, epochs, lr = int(e.max_steps), int(e.epochs), float(e.lr)
        nc = int(e.n_classes)

        per_seed: list[dict] = []
        deltas, entropies, matched_flags = [], [], []
        aur_tr, aur_fr, aur_sh = [], [], []
        for s in seeds:
            x, y, _easy = make_graded_split(
                int(e.samples), dim, nc, float(e.sep_easy), float(e.sep_hard), float(e.easy_frac), s
            )
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]

            refiner, head = train_refiner_and_halt(
                xtr,
                ytr,
                nc,
                dim,
                hidden,
                max_steps,
                epochs,
                lr,
                float(e.tau),
                float(e.halt_threshold),
                s,
            )
            with torch.no_grad():
                z, used = refiner(xte)
                trained_acc = float((head(z).argmax(-1) == yte).float().mean())
            trained_mean = float(used.float().mean())
            entropy = step_entropy_bits(used)

            thr = tune_norm_threshold(refiner, head, xtr, ytr, trained_mean, max_steps)
            free_acc, free_mean, _ = norm_rule_eval(refiner, head, xte, yte, thr, max_steps)

            per_step = refiner_flops(dim, hidden, 1)
            halt_cost = 2 * dim
            compute = matched_within(
                int(trained_mean * (per_step + halt_cost)), int(free_mean * per_step), tol=FLOP_TOL
            )

            depth = max(1, round(trained_mean))
            tr_sig, fr_sig, correct, z_depth = signals_at_depth(refiner, head, xte, yte, depth)
            a_tr = auroc(tr_sig, correct)
            a_fr = auroc(fr_sig, correct)
            a_sh = shuffled_label_auroc(z_depth, correct, s, epochs, lr)

            delta = trained_acc - free_acc
            deltas.append(delta)
            entropies.append(entropy)
            matched_flags.append(bool(compute["matched"]))
            aur_tr.append(a_tr)
            aur_fr.append(a_fr)
            aur_sh.append(a_sh)
            per_seed.append(
                {
                    "seed": s,
                    "trained_halt_acc": round(trained_acc, 4),
                    "free_rule_acc": round(free_acc, 4),
                    "delta": round(delta, 4),
                    "trained_mean_steps": round(trained_mean, 3),
                    "free_mean_steps": round(free_mean, 3),
                    "tuned_norm_threshold": round(thr, 6),
                    "halt_entropy_bits": round(entropy, 4),
                    "auroc_trained": round(a_tr, 4),
                    "auroc_free": round(a_fr, 4),
                    "auroc_shuffled_labels": round(a_sh, 4),
                    "compute": compute,
                }
            )

        ci = seed_ci(deltas)
        flips = sign_flip_report(deltas)
        entropy_mean = sum(entropies) / len(entropies)
        halt_collapsed = entropy_mean <= ENTROPY_FLOOR_BITS
        matched_all = all(matched_flags)
        auroc_gap = sum(aur_tr) / len(aur_tr) - sum(aur_fr) / len(aur_fr)

        win = bool(
            matched_all
            and not halt_collapsed
            and ci["lo"] > 0
            and flips["consistent_sign"] == 1
            and auroc_gap > 0
        )
        if halt_collapsed:
            verdict = "NULL (automatic): halt head collapsed to a constant step count"
        elif win:
            verdict = "WIN: the trained halt head beats the tuned free update-norm rule at matched mean FLOPs"
        else:
            verdict = "NULL: trained confidence ties the free update-norm rule (latent stopped moving)"

        return {
            "experiment": self.id,
            "config": OmegaConf.to_container(cfg.experiment),
            "per_seed": per_seed,
            "delta_ci": ci,
            "sign_flips": flips,
            "auroc_trained_mean": round(sum(aur_tr) / len(aur_tr), 4),
            "auroc_free_mean": round(sum(aur_fr) / len(aur_fr), 4),
            "auroc_shuffled_labels_mean": round(sum(aur_sh) / len(aur_sh), 4),
            "auroc_gap": round(auroc_gap, 4),
            "halt_entropy_bits_mean": round(entropy_mean, 4),
            "halt_collapsed": bool(halt_collapsed),
            "compute_matched_all_seeds": bool(matched_all),
            "preregistered": {
                "entropy_floor_bits": ENTROPY_FLOOR_BITS,
                "flop_tol": FLOP_TOL,
                "win_rule": "delta CI excludes 0, consistent sign, matched mean FLOPs, entropy above "
                "floor, and trained AUROC above free AUROC",
            },
            "verdict": verdict,
            "null_supported": bool(not win),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=MT6ConfidenceStop.__doc__)
    ap.add_argument("--seeds", default="0-4", help="seed range 0-4 or list 0,1,2")
    ap.add_argument("--out", default="runs/mot/mt6_confidence_stop.json")
    ap.add_argument("--rerun", action="store_true", help="Q4.1 10-seed rerun naming (_seeds10)")
    args = ap.parse_args(argv)
    out = resolve_out(args.out, args.rerun)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_cfg(parse_seeds(args.seeds))
    t0 = time.time()
    result = MT6ConfidenceStop().run(cfg, resolve("cpu"), out.parent)
    result["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"experiment": result["experiment"], "verdict": result["verdict"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
