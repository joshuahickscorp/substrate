#!/usr/bin/env python
"""DR8: recurrent refinement, fixed point vs drift, per substrate arm (WP-03, EXECUTION_MANIFEST).

Thesis: the weight-tied refiner converges to an INPUT-DEPENDENT attractor on real V-JEPA latents, and
the property is substrate-specific (the typing arm reruns this script later on the random-init-ViT
cache, WP-11 Q3.3; decay on real AND random-init means geometry, not substrate). Mechanism: train an
IterativeRefiner + head on cached latents, then unroll(K=64) far past the trained horizon and read the
update-norm decay curve (diagnostics/convergence), the past-horizon loss curve, basin stability, and a
fixed-point collapse check (input-dependence: the spread of the K-step latents relative to the input
spread). A matched-depth UNTIED control (the ex17 control) is trained alongside so any refiner
accuracy is not read as more than depth.

Preregistered null (verbatim, manifest WP-03): no geometric decay and past-horizon loss rises (the
n9/y1 result): it is unrolled depth, not an attractor. The positive requires, in EVERY seed: the
convergence report classifies the dynamics as converging, past-horizon loss does not rise beyond
PAST_HORIZON_TOL, and the fixed points stay input-dependent (no collapse to a global attractor).
Substrate typing is cross-run: this script records ONE cache arm per invocation.

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).

Usage: .venv/bin/python scripts/mot_dr8_fixed_point.py --cache vjepa --seeds 0-4
       .venv/bin/python scripts/mot_dr8_fixed_point.py --cache randominit_vitl --seeds 0-4  (Q3.3)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from mot_mt5_adaptive_halting import parse_seeds, resolve_out
from omegaconf import DictConfig, OmegaConf
from torch import nn

from devsys.devices import DeviceInfo, resolve
from devsys.diagnostics.compute import matched_within, param_count, refiner_flops
from devsys.diagnostics.convergence import basin_stability, convergence_report
from devsys.diagnostics.riskcov import seed_ci
from devsys.experiments.base import Experiment
from devsys.experiments.ex17_latent_reasoning import _UntiedDepth
from devsys.seeding import seed_everything
from devsys.shell.refine import IterativeRefiner
from devsys.substrate.datasets import make_task_stream
from devsys.substrate.real_latent import open_real_store

# preregistered thresholds, fixed before any result exists
PAST_HORIZON_TOL = 0.05  # CE (nats) rise at K=64 over the trained horizon that counts as "loss rises"
COLLAPSE_RATIO_FLOOR = 0.1  # fixed-point spread / input spread below this == global-attractor collapse
UNROLL_STEPS = 64
CHECKPOINTS = (1, 2, 4, 8, 16, 32, 64)

CACHE_NAMES = {
    "vjepa": "vjepa2_vitl_fpc64_256_real",
    "randominit_vitl": "randominit_vitl_nuisance",  # written by WP-11 Q2.1, the typing arm
}


def default_cfg(seeds: list[int], **overrides) -> DictConfig:
    exp = {
        "seeds": list(seeds),
        "cache": "vjepa",
        "data_dir": "data/cache",
        "hidden": 128,
        "n_train_steps": 4,
        "epochs": 200,
        "lr": 1e-3,
        # synthetic fallback scale (tests and the no-cache smoke path)
        "dim": 64,
        "n_classes": 6,
        "samples": 600,
        "separation": 1.5,
    }
    exp.update(overrides)
    return OmegaConf.create({"experiment": exp})


def load_cache_xy(e: DictConfig) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Resolve the --cache arm to (latents, labels, tag). synthetic uses the Gaussian-cluster stream
    (tests, smoke); real arms open the LatentStore by its manifest cache name. Latents are
    standardized per dimension so refiner dynamics are comparable across substrates."""
    cache = str(e.cache)
    if cache == "synthetic":
        task = make_task_stream(
            n_tasks=1,
            dim=int(e.dim),
            classes_per_task=int(e.n_classes),
            samples_per_task=int(e.samples),
            separation=float(e.separation),
            seed=0,
        )[0]
        x, y = task.x, task.y
    else:
        if cache not in CACHE_NAMES:
            raise KeyError(f"unknown cache arm {cache!r}: expected one of {sorted(CACHE_NAMES)} or synthetic")
        store = open_real_store(CACHE_NAMES[cache], data_dir=str(e.data_dir))
        x = store.latents().float()
        y = store.labels()
        if y is None:
            raise ValueError(f"cache {CACHE_NAMES[cache]} has no labels; DR8 needs a supervised head")
        y = y.long()
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return x, y, cache


class DR8FixedPoint(Experiment):
    id = "mot_dr8_fixed_point"
    metric = ("contraction_factor", "past_horizon_ce_rise", "fixed_point_spread_ratio", "classification")
    baseline = "matched-depth untied control (ex17 _UntiedDepth) at equal block count and hidden width"
    ablation = "unroll K=64 past the trained horizon; rerun on the random-init-ViT cache types the property"
    null_hypothesis = (
        "update norms do not decay geometrically and unrolling past N worsens loss: no fixed point, "
        "it is unrolled depth (the n9/y1 result)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        hidden, n_train = int(e.hidden), int(e.n_train_steps)
        epochs, lr = int(e.epochs), float(e.lr)

        x_all, y_all, cache = load_cache_xy(e)
        dim = int(x_all.shape[1])
        nc = int(y_all.max()) + 1

        per_seed: list[dict] = []
        converged_flags, rises_flags, collapse_flags = [], [], []
        refiner_accs, control_accs, contraction_factors = [], [], []
        for s in seeds:
            seed_everything(s)
            perm = torch.randperm(x_all.shape[0], generator=torch.Generator().manual_seed(s))
            cut = int(x_all.shape[0] * 0.7)
            xtr, ytr = x_all[perm[:cut]], y_all[perm[:cut]]
            xte, yte = x_all[perm[cut:]], y_all[perm[cut:]]

            seed_everything(s)
            refiner = IterativeRefiner(dim, hidden, n_train)
            head = nn.Linear(dim, nc)
            opt = torch.optim.Adam([*refiner.parameters(), *head.parameters()], lr=lr)
            for _ in range(epochs):
                opt.zero_grad()
                z, _ = refiner(xtr)
                F.cross_entropy(head(z), ytr).backward()
                opt.step()
            with torch.no_grad():
                z, _ = refiner(xte)
                refiner_acc = float((head(z).argmax(-1) == yte).float().mean())

            seed_everything(s)  # same init seed, the fair matched-depth control (ex17 protocol)
            control = _UntiedDepth(dim, hidden, n_train)
            chead = nn.Linear(dim, nc)
            copt = torch.optim.Adam([*control.parameters(), *chead.parameters()], lr=lr)
            for _ in range(epochs):
                copt.zero_grad()
                F.cross_entropy(chead(control(xtr)), ytr).backward()
                copt.step()
            with torch.no_grad():
                control_acc = float((chead(control(xte)).argmax(-1) == yte).float().mean())

            report = convergence_report(refiner, xte, steps=UNROLL_STEPS)
            basin = basin_stability(refiner, xte, steps=UNROLL_STEPS, seed=s)

            # past-horizon loss curve: CE at each checkpoint depth of the SAME trained refiner.
            # The trained horizon is always a checkpoint (ce_rise reads it), even when n_train is
            # not one of the preregistered powers of two (tiny test configs).
            checkpoints = sorted(set(CHECKPOINTS) | {n_train})
            ce_curve: dict[str, float] = {}
            with torch.no_grad():
                z = xte.clone()
                step = 0
                for k in checkpoints:
                    while step < k:
                        z = z + refiner._update(z)
                        step += 1
                    ce_curve[str(k)] = round(float(F.cross_entropy(head(z), yte)), 4)
                z_final = z
            ce_rise = ce_curve[str(UNROLL_STEPS)] - ce_curve[str(n_train)]
            past_rises = bool(ce_rise > PAST_HORIZON_TOL)

            # input-dependence: did the fixed points collapse to one global attractor
            spread_in = float(torch.pdist(xte).mean()) if xte.shape[0] > 1 else 0.0
            spread_fp = float(torch.pdist(z_final).mean()) if z_final.shape[0] > 1 else 0.0
            spread_ratio = spread_fp / spread_in if spread_in > 0 else 0.0
            collapsed = bool(spread_ratio < COLLAPSE_RATIO_FLOOR)

            converged_flags.append(bool(report["classification"] == "converges"))
            rises_flags.append(past_rises)
            collapse_flags.append(collapsed)
            refiner_accs.append(refiner_acc)
            control_accs.append(control_acc)
            contraction_factors.append(float(report["contraction_factor"]))
            per_seed.append(
                {
                    "seed": s,
                    "refiner_acc": round(refiner_acc, 4),
                    "untied_control_acc": round(control_acc, 4),
                    "convergence": {k: v for k, v in report.items() if k != "update_norms"},
                    "update_norms_head": report["update_norms"][:8],
                    "update_norms_tail": report["update_norms"][-4:],
                    "basin": basin,
                    "ce_curve": ce_curve,
                    "past_horizon_ce_rise": round(ce_rise, 4),
                    "past_horizon_loss_rises": past_rises,
                    "fixed_point_spread_ratio": round(spread_ratio, 4),
                    "fixed_point_collapsed": collapsed,
                }
            )

        attractor = bool(all(converged_flags) and not any(rises_flags) and not any(collapse_flags))
        depth_compute = matched_within(
            refiner_flops(dim, hidden, n_train), refiner_flops(dim, hidden, n_train)
        )
        if attractor:
            verdict = (
                f"POSITIVE on {cache}: input-dependent fixed point (typing needs the random-init arm: "
                "decay on BOTH caches = geometry, not substrate)"
            )
        else:
            verdict = f"NULL on {cache}: no attractor, it is unrolled depth (the n9/y1 result)"

        return {
            "experiment": self.id,
            "cache": cache,
            "config": OmegaConf.to_container(cfg.experiment),
            "n_samples": int(x_all.shape[0]),
            "dim": dim,
            "per_seed": per_seed,
            "refiner_acc_ci": seed_ci(refiner_accs),
            "untied_control_acc_ci": seed_ci(control_accs),
            "contraction_factor_ci": seed_ci(contraction_factors),
            "all_seeds_converge": bool(all(converged_flags)),
            "any_past_horizon_rise": bool(any(rises_flags)),
            "any_fixed_point_collapse": bool(any(collapse_flags)),
            "params": {
                "refiner": param_count(IterativeRefiner(dim, hidden, n_train)),
                "untied_control": param_count(_UntiedDepth(dim, hidden, n_train)),
            },
            "depth_control_compute": depth_compute,
            "preregistered": {
                "past_horizon_tol_ce": PAST_HORIZON_TOL,
                "collapse_ratio_floor": COLLAPSE_RATIO_FLOOR,
                "unroll_steps": UNROLL_STEPS,
                "positive_rule": "every seed converges, no past-horizon CE rise, no fixed-point collapse",
                "typing_rule": "V-JEPA-specific only if the randominit_vitl rerun of this script nulls",
            },
            "verdict": verdict,
            "null_supported": bool(not attractor),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=DR8FixedPoint.__doc__)
    ap.add_argument("--cache", default="vjepa", choices=[*CACHE_NAMES, "synthetic"])
    ap.add_argument("--seeds", default="0-4", help="seed range 0-4 or list 0,1,2")
    ap.add_argument("--out", default=None, help="default runs/mot/dr8_fixed_point_<cache>.json")
    ap.add_argument("--rerun", action="store_true", help="Q4.1 10-seed rerun naming (_seeds10)")
    args = ap.parse_args(argv)
    out = resolve_out(args.out or f"runs/mot/dr8_fixed_point_{args.cache}.json", args.rerun)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_cfg(parse_seeds(args.seeds), cache=args.cache)
    t0 = time.time()
    result = DR8FixedPoint().run(cfg, resolve("cpu"), out.parent)
    result["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"experiment": result["experiment"], "verdict": result["verdict"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
