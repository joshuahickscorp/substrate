#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from mop_mt5_adaptive_halting import make_graded_split, parse_seeds, resolve_out
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.compute import matched_within, mlp_flops, refiner_flops
from mop.diagnostics.difficulty_calibration import reference_separation
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import seed_everything
from mop.shell.refine import IterativeRefiner, Verifier

FLOP_TOL = 0.10  # preregistered: every revise arm matched to single-shot on TOTAL FLOPs within this


def default_cfg(seeds: list[int], **overrides) -> DictConfig:
    exp = {
        "seeds": list(seeds),
        "dim": 64,
        "hidden": 128,
        "verifier_hidden": 64,
        "n_classes": 6,
        "samples": 3000,
        "n_single": 4,
        "n_start": 2,
        "n_max": 8,
        "epochs": 200,
        "lr": 1e-3,
        "sep_easy": 3.0,
        "sep_hard": 0.9,
        "easy_frac": 0.5,
    }
    exp.update(overrides)
    return OmegaConf.create({"experiment": exp})


def train_backbone(
    x, y, n_classes: int, dim: int, hidden: int, vhidden: int, n_max: int, epochs: int, lr: float, seed: int
) -> tuple[IterativeRefiner, nn.Linear, Verifier]:
    seed_everything(seed)
    refiner = IterativeRefiner(dim, hidden, n_max)
    head = nn.Linear(dim, n_classes)
    verifier = Verifier(dim, hidden=vhidden)
    opt = torch.optim.Adam([*refiner.parameters(), *head.parameters(), *verifier.parameters()], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        z = x
        task_losses, ver_losses = [], []
        for _ in range(n_max):
            z = z + refiner._update(z)
            logits = head(z)
            task_losses.append(F.cross_entropy(logits, y))
            with torch.no_grad():
                per_sample = F.cross_entropy(logits, y, reduction="none")
            ver_losses.append(F.mse_loss(verifier.score(z), per_sample))
        (torch.stack(task_losses).mean() + torch.stack(ver_losses).mean()).backward()
        opt.step()
    return refiner, head, verifier


@torch.no_grad()
def revise_loop(
    refiner: IterativeRefiner,
    head: nn.Module,
    x,
    y,
    score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    threshold: float,
    n_start: int,
    n_max: int,
) -> tuple[float, float, float]:
    z = x.clone()
    u = torch.zeros_like(z)
    for _ in range(int(n_start)):
        u = refiner._update(z)
        z = z + u
    used = torch.full((x.shape[0],), float(n_start))
    rel = u.norm(dim=-1) / (z.norm(dim=-1) + 1e-9)
    evals = torch.zeros(x.shape[0])
    while True:
        eligible = used < n_max
        if not eligible.any():
            break
        scores = score_fn(z, rel)
        evals = evals + eligible.float()
        need = eligible & (scores > threshold)
        if not need.any():
            break
        u2 = refiner._update(z)
        z = torch.where(need.unsqueeze(-1), z + u2, z)
        rel = torch.where(need, u2.norm(dim=-1) / (z.norm(dim=-1) + 1e-9), rel)
        used = torch.where(need, used + 1, used)
    acc = float((head(z).argmax(-1) == y).float().mean())
    return acc, float(used.mean()), float(evals.mean())


def tune_threshold_to_budget(
    refiner: IterativeRefiner,
    head: nn.Module,
    xtr,
    ytr,
    score_fn,
    n_start: int,
    n_max: int,
    per_step: int,
    per_eval: int,
    target_flops: float,
    lo: float = 0.0,
    hi: float = 10.0,
) -> float:
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        _, steps, evals = revise_loop(refiner, head, xtr, ytr, score_fn, mid, n_start, n_max)
        if steps * per_step + evals * per_eval > target_flops:
            lo = mid  # revising too much, raise the threshold
        else:
            hi = mid
    return 0.5 * (lo + hi)


class DR9VerifyRevise(Experiment):
    id = "mop_dr9_verify_revise"
    metric = ("single_shot_acc", "verify_revise_acc", "shuffled_verifier_acc", "free_norm_acc")
    baseline = "single-shot fixed-N refinement plus the free update-norm revise rule, both FLOP-matched"
    ablation = "trained verifier vs row-shuffled verifier scores in the identical revise loop"
    null_hypothesis = (
        "verify-revise ties single-shot at matched total FLOPs and the trained verifier ties the "
        "shuffled one (the ex18 result): no correction-relevant signal"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, vhidden = int(e.dim), int(e.hidden), int(e.verifier_hidden)
        n_single, n_start, n_max = int(e.n_single), int(e.n_start), int(e.n_max)
        epochs, lr, nc = int(e.epochs), float(e.lr), int(e.n_classes)

        per_step = refiner_flops(dim, hidden, 1)
        per_eval = mlp_flops([dim, vhidden, 1])
        target_flops = float(n_single * per_step)

        per_seed: list[dict] = []
        gains, vs_shuffled, vs_free = [], [], []
        matched_flags, calibrated_flags = [], []
        for s in seeds:
            x, y, _easy = make_graded_split(
                int(e.samples), dim, nc, float(e.sep_easy), float(e.sep_hard), float(e.easy_frac), s
            )
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            d3 = reference_separation(xtr, ytr, seed=s)

            refiner, head, verifier = train_backbone(xtr, ytr, nc, dim, hidden, vhidden, n_max, epochs, lr, s)

            with torch.no_grad():
                z = xte
                for _ in range(n_single):
                    z = z + refiner._update(z)
                single_acc = float((head(z).argmax(-1) == yte).float().mean())

            def trained_score(z, rel, verifier=verifier):
                return verifier.score(z)

            g = torch.Generator().manual_seed(s + 31)

            def shuffled_score(z, rel, verifier=verifier, g=g):
                sc = verifier.score(z)
                return sc[torch.randperm(sc.shape[0], generator=g)]

            def free_score(z, rel):
                return rel

            arms = {}
            for name, fn, eval_cost in (
                ("verify_revise", trained_score, per_eval),
                ("shuffled_verifier", shuffled_score, per_eval),
                ("free_norm", free_score, 0),
            ):
                thr = tune_threshold_to_budget(
                    refiner, head, xtr, ytr, fn, n_start, n_max, per_step, eval_cost, target_flops
                )
                acc, steps, evals = revise_loop(refiner, head, xte, yte, fn, thr, n_start, n_max)
                flops = steps * per_step + evals * eval_cost
                arms[name] = {
                    "acc": round(acc, 4),
                    "mean_steps": round(steps, 3),
                    "mean_score_evals": round(evals, 3),
                    "threshold": round(thr, 6),
                    "compute": matched_within(int(flops), int(target_flops), tol=FLOP_TOL),
                }

            gain = arms["verify_revise"]["acc"] - single_acc
            d_shuf = arms["verify_revise"]["acc"] - arms["shuffled_verifier"]["acc"]
            d_free = arms["verify_revise"]["acc"] - arms["free_norm"]["acc"]
            gains.append(gain)
            vs_shuffled.append(d_shuf)
            vs_free.append(d_free)
            matched_flags.append(all(a["compute"]["matched"] for a in arms.values()))
            calibrated_flags.append(bool(d3["regime_calibrated"]))
            per_seed.append(
                {
                    "seed": s,
                    "single_shot_acc": round(single_acc, 4),
                    "arms": arms,
                    "gain_vs_single": round(gain, 4),
                    "delta_vs_shuffled": round(d_shuf, 4),
                    "delta_vs_free_norm": round(d_free, 4),
                    "d3": d3,
                }
            )

        ci_gain = seed_ci(gains)
        ci_shuf = seed_ci(vs_shuffled)
        ci_free = seed_ci(vs_free)
        flips = sign_flip_report(gains)
        matched_all = all(matched_flags)
        calibrated_all = all(calibrated_flags)

        win = bool(
            calibrated_all
            and matched_all
            and ci_gain["lo"] > 0
            and ci_shuf["lo"] > 0
            and ci_free["lo"] > 0
            and flips["consistent_sign"] == 1
        )
        if not calibrated_all:
            verdict = "UNREADABLE: regime not D3-calibrated"
        elif win:
            verdict = "WIN: the trained verifier carries a correction signal (overturns ex18)"
        elif ci_gain["lo"] <= 0:
            verdict = "NULL: verify-revise ties single-shot at matched total FLOPs (the ex18 result)"
        elif ci_shuf["lo"] <= 0:
            verdict = "NULL: trained verifier ties the shuffled verifier (reallocation, not correction)"
        else:
            verdict = "NULL: trained verifier ties the free update-norm signal (no learned scorer needed)"

        return {
            "experiment": self.id,
            "config": OmegaConf.to_container(cfg.experiment),
            "per_seed": per_seed,
            "gain_vs_single_ci": ci_gain,
            "delta_vs_shuffled_ci": ci_shuf,
            "delta_vs_free_norm_ci": ci_free,
            "sign_flips": flips,
            "compute_matched_all_seeds": bool(matched_all),
            "regime_calibrated_all_seeds": bool(calibrated_all),
            "preregistered": {
                "flop_tol": FLOP_TOL,
                "win_rule": "verify-revise beats single-shot AND shuffled AND free-norm, each seed CI "
                "excluding 0, consistent sign, matched total FLOPs, calibrated regime",
            },
            "verdict": verdict,
            "null_supported": bool(not win),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=DR9VerifyRevise.__doc__)
    ap.add_argument("--seeds", default="0-4", help="seed range 0-4 or list 0,1,2")
    ap.add_argument("--out", default="runs/mot/dr9_verify_revise.json")
    ap.add_argument("--rerun", action="store_true", help="Q4.1 10-seed rerun naming (_seeds10)")
    args = ap.parse_args(argv)
    out = resolve_out(args.out, args.rerun)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_cfg(parse_seeds(args.seeds))
    t0 = time.time()
    result = DR9VerifyRevise().run(cfg, resolve("cpu"), out.parent)
    result["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"experiment": result["experiment"], "verdict": result["verdict"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
