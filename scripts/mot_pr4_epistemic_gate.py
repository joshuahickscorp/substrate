#!/usr/bin/env python
"""PR4: epistemic gate vs noisy-TV (WP-07, plasticity program; registry
docs/mixture_of_thinking/11_experiment_registry.md, PR full schema part 1). Retries the strongest
corpus negative: e4 point-error gating chased irreducible noise 30/30.

Thesis: ensemble-disagreement (epistemic) uncertainty steers PLASTICITY toward reducible structure
and ignores aleatoric noise. The exact mechanism chain from the manifest:
Ensemble.mean_and_disagreement -> Neuromodulation.gate -> PlasticityController.lr_scale, scored on
the reducible-vs-noise LR-integral allocation through the WP-02 LRIntegralAccumulator.

Stream: fresh batches each step (no memorizable noise set) alternating between two partitions,
REDUCIBLE (a fixed rotation target, learnable) and NOISE (irreducible random targets, the noisy
TV). Every arm trains an identical ensemble with SGD; only the per-step learning-rate multiplier
differs. Each step logs lr into its partition, so the score is the fraction of total LR-integral
spent on the reducible partition.

Arms:
  gated    : disagreement -> neuromod uncertainty gate (gain in [floor, ceil]) -> controller
             lr_scale (gains above reopen_threshold reopen plasticity over the soft schedule)
  ungated  : the SAME soft schedule with signal 0 (allocates by the clock only, so its reducible
             fraction equals the partition mix by construction; the e4 conflation baseline)
  shuffled : the gated arm re-run with its own recorded signal sequence PERMUTED across steps (the
             registry's shuffled-disagreement random control)
  gated_perm : the gated arm under a permuted partition ORDER (the curriculum-permutation control;
             the allocation skew must replicate, else it is a schedule artifact)

Preregistered verdict (fixed before running): WIN iff the per-seed delta means of BOTH
(frac_gated - frac_ungated) and (frac_gated - frac_shuffled) exceed max(per-seed SD, MIN_MARGIN)
with no sign flip, AND the curriculum-permutation arm preserves the gated-over-ungated sign, AND
the diagnostics/noisy_tv three-boolean contract passes jointly (a guard failure overrides any
primary win, per the manifest appendix). Otherwise null_supported=True: the gate allocates no more
LR-integral to the reducible partition than ungated (the e4 conflation replicates).

Usage: python scripts/mot_pr4_epistemic_gate.py --seeds 0-4
Writes runs/mot/pr4_epistemic_gate.json. No em or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from devsys.devices import DeviceInfo, resolve
from devsys.diagnostics.continual_metrics import LRIntegralAccumulator
from devsys.diagnostics.noisy_tv import noisy_tv_diagnostic
from devsys.diagnostics.riskcov import seed_ci, sign_flip_report
from devsys.experiments.base import Experiment
from devsys.seeding import seed_everything
from devsys.shell.ensemble import Ensemble
from devsys.shell.neuromod import Neuromodulation
from devsys.shell.plasticity import PlasticityController
from devsys.shell.predictor import Predictor

MIN_MARGIN = 0.02  # preregistered minimum meaningful allocation-fraction delta
ARMS = ("gated", "ungated", "shuffled", "gated_perm")


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


def make_regions(seed: int, dim: int, noise_scale: float):
    """Fresh-sample generators for the two partitions (mirrors diagnostics/noisy_tv): REDUCIBLE has
    clustered inputs and a fixed learnable rotation target, NOISE has plain gaussian inputs and an
    irreducibly random target every call. The input distributions DIFFER, so an input-conditioned
    disagreement signal is able to tell the partitions apart (as in the noisy_tv diagnostic)."""
    g = torch.Generator().manual_seed(seed)
    rot = torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]
    centers = torch.randn(2, dim, generator=g)

    def reducible(bs: int):
        y = torch.randint(0, 2, (bs,), generator=g)
        x = centers[y] * 2.0 + 0.5 * torch.randn(bs, dim, generator=g)
        return x, x @ rot

    def noise(bs: int):
        x = torch.randn(bs, dim, generator=g)
        return x, torch.randn(bs, dim, generator=g) * noise_scale

    return {"reducible": reducible, "noise": noise}


def make_schedule(seed: int, n_steps: int, permute: bool) -> list[str]:
    """The partition order: strict alternation by default; the curriculum-permutation control
    shuffles the same multiset of labels."""
    order = ["reducible" if t % 2 == 0 else "noise" for t in range(n_steps)]
    if permute:
        perm = torch.randperm(n_steps, generator=torch.Generator().manual_seed(seed + 71))
        order = [order[int(i)] for i in perm]
    return order


def _neuromod(e: DictConfig) -> Neuromodulation:
    return Neuromodulation(
        OmegaConf.create(
            {
                "enabled": True,
                "surprise_gain": 1.0,
                "novelty_gain": 1.0,
                "uncertainty_gain": 1.0,
                "gate_floor": float(e.gate_floor),
                "gate_ceil": float(e.gate_ceil),
            }
        )
    )


def _controller(e: DictConfig, seed: int) -> PlasticityController:
    ctl = PlasticityController(
        OmegaConf.create({"schedule": "soft", "lr": float(e.base_lr), "rigidity": 0.0, "pnn_fraction": 0.0}),
        seed=seed,
    )
    ctl.reopen_threshold = float(e.reopen_threshold)
    return ctl


def run_arm(
    e: DictConfig,
    seed: int,
    mode: str,
    signal_override: list[float] | None = None,
    permute_order: bool = False,
) -> dict:
    """Train one arm and return its per-partition LR-integral, reducible fraction, and the per-step
    signal sequence (recorded so the shuffled control can replay it permuted). mode is 'gated' or
    'ungated'; signal_override replaces the live disagreement signal (shuffled control)."""
    dim, batch, n_steps = int(e.dim), int(e.batch), int(e.n_steps)
    regions = make_regions(seed, dim, float(e.noise_scale))
    order = make_schedule(seed, n_steps, permute=permute_order)
    torch.manual_seed(seed)
    ens = Ensemble(lambda: Predictor(dim, hidden=int(e.hidden), depth=2), size=int(e.ensemble_size))
    opt = torch.optim.SGD(ens.parameters(), lr=float(e.base_lr))
    neuromod = _neuromod(e)
    ctl = _controller(e, seed)
    lrint = LRIntegralAccumulator()
    signals: list[float] = []
    for t, part in enumerate(order):
        x, target = regions[part](batch)
        progress = t / max(1, n_steps - 1)
        if mode == "ungated":
            lr_mult = ctl.lr_scale(progress, 0.0)
            signals.append(0.0)
        else:
            with torch.no_grad():
                _, dis = ens.mean_and_disagreement(x)
            raw = float(dis.mean()) if signal_override is None else signal_override[t]
            gain = neuromod.gate("uncertainty", raw)
            lr_mult = ctl.lr_scale(progress, signal=gain)
            signals.append(raw)
        lr = float(e.base_lr) * lr_mult
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad()
        out = ens(x)
        F.mse_loss(out, target.unsqueeze(0).expand_as(out)).backward()
        opt.step()
        lrint.add(lr, steps=1, partition=part)
    total = lrint.total()
    return {
        "lr_integral": lrint.as_dict(),
        "reducible_fraction": lrint.total("reducible") / total if total > 0 else 0.0,
        "signals": signals,
    }


class MotPR4EpistemicGate(Experiment):
    id = "mot_pr4_epistemic_gate"
    metric = ("reducible_lr_fraction", "noisy_tv_guard", "delta_vs_ungated", "delta_vs_shuffled")
    baseline = "the same soft plasticity schedule ungated (clock-only allocation, the e4 baseline)"
    ablation = "live disagreement gate vs permuted-signal gate vs permuted curriculum"
    null_hypothesis = (
        "the disagreement gate allocates no more LR-integral to the reducible partition than the "
        "ungated arm (the e4 epistemic/aleatoric conflation), or the noisy-TV guard fails"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        t0 = time.perf_counter()
        seeds = list(e.seeds)
        frac = {a: [] for a in ARMS}
        lrints = []
        for s in seeds:
            seed_everything(s)
            gated = run_arm(e, s, "gated")
            ungated = run_arm(e, s, "ungated")
            perm = torch.randperm(int(e.n_steps), generator=torch.Generator().manual_seed(s + 17)).tolist()
            shuffled_signals = [gated["signals"][i] for i in perm]
            shuffled = run_arm(e, s, "gated", signal_override=shuffled_signals)
            gated_perm = run_arm(e, s, "gated", permute_order=True)
            for arm, res in (
                ("gated", gated),
                ("ungated", ungated),
                ("shuffled", shuffled),
                ("gated_perm", gated_perm),
            ):
                frac[arm].append(res["reducible_fraction"])
            lrints.append({a: r["lr_integral"] for a, r in (("gated", gated), ("ungated", ungated))})
        d_ungated = [frac["gated"][i] - frac["ungated"][i] for i in range(len(seeds))]
        d_shuffled = [frac["gated"][i] - frac["shuffled"][i] for i in range(len(seeds))]
        v_ungated, v_shuffled = verdict(d_ungated), verdict(d_shuffled)
        perm_sign_ok = bool(
            seed_ci([frac["gated_perm"][i] - frac["ungated"][i] for i in range(len(seeds))])["mean"] > 0
        )
        # the noisy-TV three-boolean contract (manifest appendix: a guard failure overrides any win)
        guard = noisy_tv_diagnostic(
            dim=int(e.guard_dim),
            ensemble_size=int(e.ensemble_size),
            steps=int(e.guard_steps),
            noise_scale=float(e.noise_scale),
            seed=int(seeds[0]),
        )
        guard_pass = bool(
            guard["noise_error_stays_high"]
            and guard["epistemic_collapses_on_noise"]
            and guard["learning_progress_separates"]
        )
        win = bool(v_ungated["win"] and v_shuffled["win"] and perm_sign_ok and guard_pass)
        out = {
            "experiment": self.id,
            "contract": self.contract(),
            "config": OmegaConf.to_container(e, resolve=True),
            "reducible_lr_fraction": {a: {"per_seed": frac[a], "ci": seed_ci(frac[a])} for a in ARMS},
            "lr_integrals_per_seed": lrints,
            "delta_vs_ungated": v_ungated,
            "delta_vs_shuffled": v_shuffled,
            "curriculum_permutation_sign_ok": perm_sign_ok,
            "noisy_tv_guard": {
                "raw_error": guard["raw_error"],
                "disagreement": guard["disagreement"],
                "learning_progress": guard["learning_progress"],
                "noise_error_stays_high": guard["noise_error_stays_high"],
                "epistemic_collapses_on_noise": guard["epistemic_collapses_on_noise"],
                "learning_progress_separates": guard["learning_progress_separates"],
                "pass": guard_pass,
            },
            "verdict_rule": (
                "WIN iff mean per-seed (frac_gated - frac_ungated) AND (frac_gated - frac_shuffled) both "
                f"> max(seed SD, {MIN_MARGIN}) with no sign flip, the curriculum-permutation arm keeps "
                "the gated-over-ungated sign, and all three noisy-TV booleans pass jointly; else the "
                "preregistered null (the e4 conflation) is supported"
            ),
            "null_supported": bool(not win),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        return out


def default_cfg(**overrides) -> DictConfig:
    base = {
        "seeds": [0, 1, 2, 3, 4],
        "dim": 32,
        "hidden": 64,
        "ensemble_size": 5,
        "batch": 64,
        "n_steps": 400,
        "noise_scale": 5.0,
        "base_lr": 0.05,
        "gate_floor": 0.5,
        "gate_ceil": 2.0,
        "reopen_threshold": 1.0,
        "guard_dim": 64,
        "guard_steps": 400,
    }
    base.update(overrides)
    return OmegaConf.create({"experiment": base})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR4: disagreement-gated plasticity vs the noisy TV")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", default="runs/mot/pr4_epistemic_gate.json")
    a = ap.parse_args(argv)
    cfg = default_cfg(seeds=parse_seeds(a.seeds))
    result = MotPR4EpistemicGate().run(cfg, resolve("cpu"), Path(a.out).parent)
    text = json.dumps(result, indent=2, default=str)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
