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
from mop.diagnostics.compute import matched_within, refiner_flops
from mop.diagnostics.difficulty_calibration import reference_separation
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import parse_seeds, seed_everything
from mop.shell.refine import IterativeRefiner

ENTROPY_FLOOR_BITS = 0.1  # halting entropy at or below this is an automatic null (constant halt head)
FLOP_TOL = 0.10  # halting rows are matched on MEAN FLOPs within this tolerance (manifest appendix)
GRADIENT_MARGIN = 0.05  # easy-vs-hard probe accuracy gap certifying a real hardness gradient (D3)


def resolve_out(path: str | Path, rerun: bool) -> Path:
    p = Path(path)
    return p.with_name(p.stem + "_seeds10.json") if rerun else p


def make_graded_split(
    n: int, dim: int, n_classes: int, sep_easy: float, sep_hard: float, easy_frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, dim, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    easy = torch.rand(n, generator=g) < easy_frac
    sep = torch.where(easy, torch.tensor(float(sep_easy)), torch.tensor(float(sep_hard)))
    x = centers[y] * sep.unsqueeze(-1) + torch.randn(n, dim, generator=g) * 0.5
    return x, y, easy


def train_refiner_and_halt(
    x: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    dim: int,
    hidden: int,
    max_steps: int,
    epochs: int,
    lr: float,
    tau: float,
    halt_threshold: float,
    seed: int,
) -> tuple[IterativeRefiner, nn.Linear]:
    seed_everything(seed)
    refiner = IterativeRefiner(dim, hidden, max_steps, halt=True, halt_threshold=halt_threshold)
    head = nn.Linear(dim, n_classes)
    opt = torch.optim.Adam(
        [*refiner.norm.parameters(), *refiner.block.parameters(), *head.parameters()], lr=lr
    )
    for _ in range(epochs):
        opt.zero_grad()
        z = x
        losses = []
        for _ in range(max_steps):
            z = z + refiner._update(z)
            losses.append(F.cross_entropy(head(z), y))
        torch.stack(losses).mean().backward()
        opt.step()
    with torch.no_grad():
        zs, correct = [], []
        z = x
        for _ in range(max_steps):
            z = z + refiner._update(z)
            zs.append(z)
            correct.append((head(z).argmax(-1) == y).float())
    assert refiner.halt_head is not None
    hopt = torch.optim.Adam(refiner.halt_head.parameters(), lr=lr)
    for _ in range(epochs):
        hopt.zero_grad()
        ps = [torch.sigmoid(refiner.halt_head(zt)).squeeze(-1) for zt in zs]
        bce = torch.stack([F.binary_cross_entropy(p, c) for p, c in zip(ps, correct, strict=True)]).mean()
        survive = torch.ones_like(ps[0])
        esteps = torch.zeros_like(ps[0])
        for p in ps:
            esteps = esteps + survive
            survive = survive * (1 - p)
        (bce + tau * esteps.mean() / max_steps).backward()
        hopt.step()
    return refiner, head


@torch.no_grad()
def fixed_depth_eval(refiner: IterativeRefiner, head: nn.Module, x, y, steps: int) -> float:
    z = x
    for _ in range(int(steps)):
        z = z + refiner._update(z)
    return float((head(z).argmax(-1) == y).float().mean())


@torch.no_grad()
def random_allocation_eval(
    refiner: IterativeRefiner, head: nn.Module, x, y, mean_steps: float, max_steps: int, seed: int
) -> tuple[float, float]:
    lo = int(math.floor(mean_steps))
    hi = min(lo + 1, int(max_steps))
    frac = mean_steps - lo
    g = torch.Generator().manual_seed(seed + 101)
    take_hi = torch.rand(x.shape[0], generator=g) < frac
    steps_per = torch.where(take_hi, torch.tensor(hi), torch.tensor(lo))
    z = x.clone()
    for t in range(int(steps_per.max().item())):
        active = steps_per > t
        z = torch.where(active.unsqueeze(-1), z + refiner._update(z), z)
    acc = float((head(z).argmax(-1) == y).float().mean())
    return acc, float(steps_per.float().mean())


def step_entropy_bits(used: torch.Tensor) -> float:
    _, counts = used.long().unique(return_counts=True)
    p = counts.float() / counts.sum()
    return float(-(p * p.log2()).sum())


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def hardness_gradient_certificate(
    xtr, ytr, xte, yte, easy_te: torch.Tensor, n_classes: int, seed: int
) -> dict:
    seed_everything(seed)
    probe = nn.Linear(xtr.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        F.cross_entropy(probe(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        correct = (probe(xte).argmax(-1) == yte).float()
    acc_easy = float(correct[easy_te].mean()) if easy_te.any() else 0.0
    acc_hard = float(correct[~easy_te].mean()) if (~easy_te).any() else 0.0
    gap = acc_easy - acc_hard
    return {
        "probe_acc_easy": round(acc_easy, 4),
        "probe_acc_hard": round(acc_hard, 4),
        "gap": round(gap, 4),
        "gradient_present": bool(gap > GRADIENT_MARGIN),
    }


def default_cfg(seeds: list[int], **overrides) -> DictConfig:
    exp = {
        "seeds": list(seeds),
        "dim": 64,
        "hidden": 128,
        "n_classes": 6,
        "samples": 3000,
        "max_steps": 8,
        "epochs": 200,
        "lr": 1e-3,
        "tau": 0.05,
        "halt_threshold": 0.9,
        "sep_easy": 3.0,
        "sep_hard": 0.9,
        "easy_frac": 0.5,
    }
    exp.update(overrides)
    return OmegaConf.create({"experiment": exp})


class MT5AdaptiveHalting(Experiment):
    id = "mop_mt5_adaptive_halting"
    metric = ("adaptive_acc", "matched_mean_control_acc", "delta", "mean_halt_steps", "halt_entropy_bits")
    baseline = (
        "random-allocation floor/ceil mixture of fixed depths at EXACTLY the adaptive arm's mean steps "
        "(matched average FLOPs), plus fixed-round(mean) for context"
    )
    ablation = "adaptive halt head vs random allocation of the same step budget; shuffled halt-step corr"
    null_hypothesis = (
        "at equal average FLOPs adaptive halting ties fixed depth (no exploitable per-sample hardness "
        "heterogeneity), or the halt head collapses to a constant N (halting entropy ~0, automatic null)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden = int(e.dim), int(e.hidden)
        max_steps, epochs, lr = int(e.max_steps), int(e.epochs), float(e.lr)
        nc = int(e.n_classes)

        per_seed: list[dict] = []
        deltas, entropies, corrs, shuf_corrs = [], [], [], []
        calibrated_flags, gradient_flags, matched_flags = [], [], []
        for s in seeds:
            x, y, easy = make_graded_split(
                int(e.samples), dim, nc, float(e.sep_easy), float(e.sep_hard), float(e.easy_frac), s
            )
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            easy_te = easy[cut:]

            d3 = reference_separation(xtr, ytr, seed=s)
            grad_cert = hardness_gradient_certificate(xtr, ytr, xte, yte, easy_te, nc, s)

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
                adaptive_acc = float((head(z).argmax(-1) == yte).float().mean())
            mean_steps = float(used.float().mean())
            entropy = step_entropy_bits(used)

            control_acc, control_mean = random_allocation_eval(
                refiner, head, xte, yte, mean_steps, max_steps, s
            )
            fixed_round_acc = fixed_depth_eval(refiner, head, xte, yte, max(1, round(mean_steps)))

            per_step = refiner_flops(dim, hidden, 1)
            halt_cost = 2 * dim  # one Linear(dim,1) score per applied step, charged to the adaptive arm
            compute = matched_within(
                int(mean_steps * (per_step + halt_cost)), int(control_mean * per_step), tol=FLOP_TOL
            )
            delta = adaptive_acc - control_acc

            corr = pearson(used.tolist(), (~easy_te).float().tolist())
            g = torch.Generator().manual_seed(s + 7)
            shuf = used[torch.randperm(used.shape[0], generator=g)]
            shuf_corr = pearson(shuf.tolist(), (~easy_te).float().tolist())

            deltas.append(delta)
            entropies.append(entropy)
            corrs.append(corr)
            shuf_corrs.append(shuf_corr)
            calibrated_flags.append(bool(d3["regime_calibrated"]))
            gradient_flags.append(bool(grad_cert["gradient_present"]))
            matched_flags.append(bool(compute["matched"]))
            per_seed.append(
                {
                    "seed": s,
                    "adaptive_acc": round(adaptive_acc, 4),
                    "matched_mean_control_acc": round(control_acc, 4),
                    "fixed_round_acc": round(fixed_round_acc, 4),
                    "delta": round(delta, 4),
                    "mean_halt_steps": round(mean_steps, 3),
                    "control_mean_steps": round(control_mean, 3),
                    "halt_entropy_bits": round(entropy, 4),
                    "halt_vs_hardness_corr": round(corr, 4),
                    "shuffled_corr": round(shuf_corr, 4),
                    "compute": compute,
                    "d3": d3,
                    "hardness_gradient": grad_cert,
                }
            )

        ci = seed_ci(deltas)
        flips = sign_flip_report(deltas)
        entropy_mean = sum(entropies) / len(entropies)
        halt_collapsed = entropy_mean <= ENTROPY_FLOOR_BITS
        matched_all = all(matched_flags)
        regime_readable = all(calibrated_flags) and all(gradient_flags)

        win = bool(
            regime_readable
            and matched_all
            and not halt_collapsed
            and ci["lo"] > 0
            and flips["consistent_sign"] == 1
        )
        if not regime_readable:
            verdict = "UNREADABLE: regime not calibrated or no hardness gradient (D3 failed)"
        elif halt_collapsed:
            verdict = "NULL (automatic): halt head collapsed to a constant step count"
        elif win:
            verdict = "WIN: adaptive allocation beats random allocation of the same mean step budget"
        else:
            verdict = "NULL: adaptive halting ties fixed depth at equal average FLOPs"

        return {
            "experiment": self.id,
            "config": OmegaConf.to_container(cfg.experiment),
            "per_seed": per_seed,
            "delta_ci": ci,
            "sign_flips": flips,
            "halt_entropy_bits_mean": round(entropy_mean, 4),
            "halt_collapsed": bool(halt_collapsed),
            "halt_vs_hardness_corr_mean": round(sum(corrs) / len(corrs), 4),
            "shuffled_corr_mean": round(sum(shuf_corrs) / len(shuf_corrs), 4),
            "compute_matched_all_seeds": bool(matched_all),
            "regime_readable": bool(regime_readable),
            "preregistered": {
                "entropy_floor_bits": ENTROPY_FLOOR_BITS,
                "flop_tol": FLOP_TOL,
                "gradient_margin": GRADIENT_MARGIN,
                "win_rule": "delta CI excludes 0, consistent sign, matched mean FLOPs, entropy above "
                "floor, calibrated regime with a hardness gradient",
            },
            "verdict": verdict,
            "null_supported": bool(not win),
        }


class MT5AdaptiveHaltingRecal(MT5AdaptiveHalting):

    id = "mop_mt5_adaptive_halting_recal"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        from mop.diagnostics.hardness import make_graded_slot_task

        e = cfg.experiment
        seeds = list(e.seeds)
        max_steps, epochs, lr = int(e.max_steps), int(e.epochs), float(e.lr)

        per_seed: list[dict] = []
        deltas, entropies, corrs, shuf_corrs = [], [], [], []
        calibrated_flags, gradient_flags, matched_flags = [], [], []
        for s in seeds:
            task = make_graded_slot_task(int(e.samples), noise=2.4, seed=s)
            x, y, easy = task.x, task.y, ~task.hard_mask
            dim, nc = task.dim, int(y.max()) + 1
            hidden = int(e.hidden)
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            easy_te = easy[cut:]

            d3 = reference_separation(xtr, ytr, seed=s)
            grad_cert = hardness_gradient_certificate(xtr, ytr, xte, yte, easy_te, nc, s)

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
                adaptive_acc = float((head(z).argmax(-1) == yte).float().mean())
            mean_steps = float(used.float().mean())
            entropy = step_entropy_bits(used)

            control_acc, control_mean = random_allocation_eval(
                refiner, head, xte, yte, mean_steps, max_steps, s
            )
            fixed_round_acc = fixed_depth_eval(refiner, head, xte, yte, max(1, round(mean_steps)))

            per_step = refiner_flops(dim, hidden, 1)
            halt_cost = 2 * dim
            compute = matched_within(
                int(mean_steps * (per_step + halt_cost)), int(control_mean * per_step), tol=FLOP_TOL
            )
            delta = adaptive_acc - control_acc

            corr = pearson(used.tolist(), (~easy_te).float().tolist())
            g = torch.Generator().manual_seed(s + 7)
            shuf = used[torch.randperm(used.shape[0], generator=g)]
            shuf_corr = pearson(shuf.tolist(), (~easy_te).float().tolist())

            deltas.append(delta)
            entropies.append(entropy)
            corrs.append(corr)
            shuf_corrs.append(shuf_corr)
            calibrated_flags.append(bool(d3["regime_calibrated"]))
            gradient_flags.append(bool(grad_cert["gradient_present"]))
            matched_flags.append(bool(compute["matched"]))
            per_seed.append(
                {
                    "seed": s,
                    "adaptive_acc": round(adaptive_acc, 4),
                    "matched_mean_control_acc": round(control_acc, 4),
                    "fixed_round_acc": round(fixed_round_acc, 4),
                    "delta": round(delta, 4),
                    "mean_halt_steps": round(mean_steps, 3),
                    "control_mean_steps": round(control_mean, 3),
                    "halt_entropy_bits": round(entropy, 4),
                    "halt_vs_hardness_corr": round(corr, 4),
                    "shuffled_corr": round(shuf_corr, 4),
                    "compute": compute,
                    "d3": d3,
                    "hardness_gradient": grad_cert,
                }
            )

        ci = seed_ci(deltas)
        flips = sign_flip_report(deltas)
        entropy_mean = sum(entropies) / len(entropies)
        halt_collapsed = entropy_mean <= ENTROPY_FLOOR_BITS
        matched_all = all(matched_flags)
        regime_readable = all(calibrated_flags) and all(gradient_flags)

        win = bool(
            regime_readable
            and matched_all
            and not halt_collapsed
            and ci["lo"] > 0
            and flips["consistent_sign"] == 1
        )
        if not regime_readable:
            verdict = "UNREADABLE: D3 slot regime not calibrated or no hardness gradient on some seed"
        elif halt_collapsed:
            verdict = "NULL (automatic): halt head collapsed to a constant step count on D3 regime"
        elif win:
            verdict = "WIN: adaptive allocation beats random allocation at equal FLOPs on the D3 regime"
        else:
            verdict = "NULL: adaptive halting ties fixed depth at equal average FLOPs on the D3 regime"

        return {
            "experiment": self.id,
            "recal": "D3 graded slot task (diagnostics/hardness.make_graded_slot_task)",
            "config": OmegaConf.to_container(cfg.experiment),
            "per_seed": per_seed,
            "delta_ci": ci,
            "sign_flips": flips,
            "halt_entropy_bits_mean": round(entropy_mean, 4),
            "halt_collapsed": bool(halt_collapsed),
            "halt_vs_hardness_corr_mean": round(sum(corrs) / len(corrs), 4),
            "shuffled_corr_mean": round(sum(shuf_corrs) / len(shuf_corrs), 4),
            "compute_matched_all_seeds": bool(matched_all),
            "regime_readable": bool(regime_readable),
            "preregistered": {
                "entropy_floor_bits": ENTROPY_FLOOR_BITS,
                "flop_tol": FLOP_TOL,
                "gradient_margin": GRADIENT_MARGIN,
                "win_rule": "delta CI excludes 0, consistent sign, matched mean FLOPs, entropy above "
                "floor, calibrated D3 regime with a certified hardness gradient",
            },
            "verdict": verdict,
            "null_supported": bool(not win),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=MT5AdaptiveHalting.__doc__)
    ap.add_argument("--seeds", default="0-4", help="seed range 0-4 or list 0,1,2")
    ap.add_argument("--out", default="runs/mot/mt5_adaptive_halting.json")
    ap.add_argument("--rerun", action="store_true", help="Q4.1 10-seed rerun naming (_seeds10)")
    ap.add_argument("--recal", action="store_true", help="A5: run on D3 graded task, write _recal.json")
    args = ap.parse_args(argv)
    if args.recal:
        out = Path("runs/mot/mt5_adaptive_halting_recal.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg = default_cfg(parse_seeds(args.seeds))
        t0 = time.time()
        result = MT5AdaptiveHaltingRecal().run(cfg, resolve("cpu"), out.parent)
        result["seconds"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(result, indent=2))
        print(json.dumps({"experiment": result["experiment"], "verdict": result["verdict"], "out": str(out)}))
        return 0
    out = resolve_out(args.out, args.rerun)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_cfg(parse_seeds(args.seeds))
    t0 = time.time()
    result = MT5AdaptiveHalting().run(cfg, resolve("cpu"), out.parent)
    result["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"experiment": result["experiment"], "verdict": result["verdict"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
