#!/usr/bin/env python
"""MP8: latent debate between seeded modules vs single module and plain ensemble (WP-04, MANIFEST).

Thesis: two seeded modules critiquing each other's latents through trained exchange wires, arbitrated
by a tiny referee, beat BOTH the single matched module and a plain ensemble at matched TOTAL FLOPs.
Mechanism: per seed, two independently initialized WP-03 backbones (refiner + deep-supervised head +
verifier, mop_dr9_verify_revise.train_backbone) are trained on the same regime, then frozen. The
debate arm adds zero-initialized cross-projection wires (each module's update is corrected by a linear
message from the OTHER module's LayerNormed latent, every round) and a referee (a tiny MLP over both
verifier scores and the inter-module latent distance emitting a per-sample blend weight); wires and
referee are trained in a second phase with the modules frozen. Zero-init means debate STARTS exactly
at the independent pair, so anything it learns is attributable to exchange.

Controls, all preregistered: (1) single matched module, the BETTER of the two backbones unrolled to
the FLOP-matched depth (inside its deep-supervised horizon, never past it); (2) plain ensemble, both
backbones unrolled independently at matched split FLOPs, logits averaged (the ex17 unrolled-ensemble
null made explicit); (3) self-debate, module A debating a second copy of ITSELF through freshly
trained wires (if this ties real debate, the exchange is self-generated depth, not cross-module
content); (4) shuffled-partner, the trained debate wires evaluated with the incoming partner latent
row-permuted every round (seed-permuted modules control: destroys per-sample cross-module information,
keeps marginals).

PR1 gate (manifest WP-04): the PR1 mode-error-disjointness verdict is READ AT RUN TIME from
runs/pre_studio/pr1_mode_error_disjointness.json (never rebuilt, via mop_dr12_disagreement's reader).
GREEN licenses a win as live; otherwise a win is demoted to PLAUSIBLE-BUT-UNVERIFIED (manifest
appendix), and nulls report against the PR1 context either way.

Preregistered null (verbatim, manifest WP-04): MP8 debate ties max(single, ensemble) (the ex17
unrolled-ensemble null). A win requires, at matched total FLOPs on a D3-calibrated regime: debate
beats max(single matched, plain ensemble) with a seed CI excluding zero and no sign flip, AND beats
the shuffled-partner arm AND the self-debate arm (attribution to genuine cross-module exchange).

Form per BLACKHOLE.md: no em dashes or en dashes (commas, colons, parentheses only).

Usage: .venv/bin/python scripts/mop_mt8_latent_debate.py --seeds 0-4
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from mop_dr9_verify_revise import train_backbone
from mop_dr12_disagreement import PR1_PATH, read_pr1_context
from mop_mt5_adaptive_halting import make_graded_split, parse_seeds, pearson, resolve_out
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.compute import matched_within, mlp_flops, refiner_flops
from mop.diagnostics.difficulty_calibration import reference_separation
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import seed_everything
from mop.shell.predictor import mlp
from mop.shell.refine import IterativeRefiner, Verifier

FLOP_TOL = 0.10  # preregistered: routing rows are matched on total FLOPs within this (manifest appendix)
REFEREE_HIDDEN = 8  # referee width, fixed a priori (3 features -> 8 -> 1, rounding-error FLOPs)


def default_cfg(seeds: list[int], **overrides) -> DictConfig:
    """The preregistered full-scale config. Tests shrink it via overrides (tiny tensors, seconds)."""
    exp = {
        "seeds": list(seeds),
        "dim": 64,
        "hidden": 128,
        "verifier_hidden": 64,
        "n_classes": 6,
        "samples": 3000,
        "rounds": 4,
        "epochs": 200,
        "lr": 1e-3,
        "sep_easy": 3.0,
        "sep_hard": 0.9,
        "easy_frac": 0.5,
        "pr1_path": PR1_PATH,
    }
    exp.update(overrides)
    return OmegaConf.create({"experiment": exp})


@dataclass
class ModuleShell:
    """One frozen debate participant: the WP-03 backbone triple."""

    refiner: IterativeRefiner
    head: nn.Linear
    verifier: Verifier

    def freeze(self) -> ModuleShell:
        for m in (self.refiner, self.head, self.verifier):
            for p in m.parameters():
                p.requires_grad_(False)
        return self


class DebateWires(nn.Module):
    """The trainable debate machinery: two cross-projection message wires (zero-initialized, so the
    debate starts exactly at the independent pair) and the referee (per-sample blend weight over
    [score_a, score_b, latent distance])."""

    def __init__(self, dim: int):
        super().__init__()
        self.msg_a = nn.Linear(dim, dim)  # partner latent -> correction for module A
        self.msg_b = nn.Linear(dim, dim)
        nn.init.zeros_(self.msg_a.weight)
        nn.init.zeros_(self.msg_a.bias)
        nn.init.zeros_(self.msg_b.weight)
        nn.init.zeros_(self.msg_b.bias)
        self.referee = mlp(3, 1, REFEREE_HIDDEN, depth=1, ln=True)


def debate_forward(
    mod_a: ModuleShell,
    mod_b: ModuleShell,
    wires: DebateWires,
    x: torch.Tensor,
    rounds: int,
    shuffle_partner: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """One debate: `rounds` alternating-free rounds where each module refines its own latent AND
    receives a linear message from the other's LayerNormed latent, then the referee blends the two
    heads' logits per sample. shuffle_partner row-permutes the incoming partner latent every round
    (the seed-permuted-modules control)."""
    dim = x.shape[-1]
    z_a, z_b = x, x
    for _ in range(int(rounds)):
        in_a, in_b = z_b, z_a  # messages cross modules
        if shuffle_partner:
            perm = torch.randperm(x.shape[0], generator=generator)
            in_a, in_b = in_a[perm], in_b[perm]
        u_a = mod_a.refiner._update(z_a) + wires.msg_a(F.layer_norm(in_a, (dim,)))
        u_b = mod_b.refiner._update(z_b) + wires.msg_b(F.layer_norm(in_b, (dim,)))
        z_a, z_b = z_a + u_a, z_b + u_b
    score_a = mod_a.verifier.score(z_a)
    score_b = mod_b.verifier.score(z_b)
    dist = (z_a - z_b).norm(dim=-1) / math.sqrt(dim)
    feats = torch.stack([score_a, score_b, dist], dim=-1)
    w = torch.sigmoid(wires.referee(feats)).squeeze(-1)
    logits = w.unsqueeze(-1) * mod_a.head(z_a) + (1 - w).unsqueeze(-1) * mod_b.head(z_b)
    return logits, {"w_mean": float(w.mean()), "dist_mean": float(dist.mean())}


def train_debate_wires(
    mod_a: ModuleShell, mod_b: ModuleShell, x, y, rounds: int, epochs: int, lr: float, seed: int
) -> DebateWires:
    """Phase 2: train wires + referee with both modules frozen (gradient flows through them, their
    parameters never step, mirroring the WP-03 two-phase convention)."""
    seed_everything(seed)
    wires = DebateWires(x.shape[-1])
    opt = torch.optim.Adam(wires.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logits, _ = debate_forward(mod_a, mod_b, wires, x, rounds)
        F.cross_entropy(logits, y).backward()
        opt.step()
    return wires


@torch.no_grad()
def unroll_logits(mod: ModuleShell, x: torch.Tensor, steps: int) -> torch.Tensor:
    z = x
    for _ in range(int(steps)):
        z = z + mod.refiner._update(z)
    return mod.head(z)


def debate_flop_budget(dim: int, hidden: int, vhidden: int, n_classes: int, rounds: int) -> dict:
    """The debate arm's TOTAL forward FLOPs, computed a priori from the config, and the FLOP-matched
    step counts for the single-module and plain-ensemble controls. All controls stay INSIDE the
    deep-supervised training horizon (n_max_train), never a past-horizon unroll."""
    per_step = refiner_flops(dim, hidden, 1)
    per_msg = mlp_flops([dim, dim])
    per_eval = mlp_flops([dim, vhidden, 1])
    head_flops = mlp_flops([dim, n_classes])
    referee_flops = mlp_flops([3, REFEREE_HIDDEN, 1])
    debate_total = 2 * rounds * (per_step + per_msg) + 2 * per_eval + referee_flops + 2 * head_flops
    steps_single = max(rounds, round((debate_total - head_flops) / per_step))
    steps_ensemble = max(rounds, round((debate_total - 2 * head_flops) / (2 * per_step)))
    return {
        "per_step": per_step,
        "per_msg": per_msg,
        "debate_total": int(debate_total),
        "steps_single": int(steps_single),
        "single_total": int(steps_single * per_step + head_flops),
        "steps_ensemble": int(steps_ensemble),
        "ensemble_total": int(2 * steps_ensemble * per_step + 2 * head_flops),
        "n_max_train": int(max(steps_single, steps_ensemble, rounds)),
    }


class MT8LatentDebate(Experiment):
    id = "mop_mt8_latent_debate"
    metric = ("debate_acc", "best_single_acc", "ensemble_acc", "delta_vs_max")
    baseline = (
        "max(single matched module, plain two-module ensemble) at matched total FLOPs, both unrolled "
        "inside their deep-supervised horizon"
    )
    ablation = "self-debate (module vs a copy of itself) and shuffled-partner (row-permuted exchange)"
    null_hypothesis = (
        "debate ties the single matched module AND ties a plain ensemble at matched total FLOPs: "
        "structured exchange is an unrolled ensemble (the ex17 result)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, vhidden = int(e.dim), int(e.hidden), int(e.verifier_hidden)
        nc, rounds = int(e.n_classes), int(e.rounds)
        epochs, lr = int(e.epochs), float(e.lr)

        pr1 = read_pr1_context(str(e.pr1_path))
        budget = debate_flop_budget(dim, hidden, vhidden, nc, rounds)
        match_single = matched_within(budget["debate_total"], budget["single_total"], tol=FLOP_TOL)
        match_ens = matched_within(budget["debate_total"], budget["ensemble_total"], tol=FLOP_TOL)

        per_seed: list[dict] = []
        d_max, d_shuf, d_self = [], [], []
        calibrated_flags = []
        err_corrs = []
        for s in seeds:
            x, y, _easy = make_graded_split(
                int(e.samples), dim, nc, float(e.sep_easy), float(e.sep_hard), float(e.easy_frac), s
            )
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            d3 = reference_separation(xtr, ytr, seed=s)

            n_max = budget["n_max_train"]
            mod_a = ModuleShell(
                *train_backbone(xtr, ytr, nc, dim, hidden, vhidden, n_max, epochs, lr, 1000 * s + 1)
            ).freeze()
            mod_b = ModuleShell(
                *train_backbone(xtr, ytr, nc, dim, hidden, vhidden, n_max, epochs, lr, 1000 * s + 2)
            ).freeze()

            # PR1-precondition diagnostic: are the two seeded modules actually decorrelated in errors
            with torch.no_grad():
                err_a = (unroll_logits(mod_a, xte, rounds).argmax(-1) != yte).float()
                err_b = (unroll_logits(mod_b, xte, rounds).argmax(-1) != yte).float()
            err_corr = pearson(err_a.tolist(), err_b.tolist())

            wires = train_debate_wires(mod_a, mod_b, xtr, ytr, rounds, epochs, lr, s)
            self_wires = train_debate_wires(mod_a, mod_a, xtr, ytr, rounds, epochs, lr, s + 500)

            with torch.no_grad():
                logits, info = debate_forward(mod_a, mod_b, wires, xte, rounds)
                debate_acc = float((logits.argmax(-1) == yte).float().mean())
                g = torch.Generator().manual_seed(s + 733)
                logits, _ = debate_forward(
                    mod_a, mod_b, wires, xte, rounds, shuffle_partner=True, generator=g
                )
                shuffled_acc = float((logits.argmax(-1) == yte).float().mean())
                logits, _ = debate_forward(mod_a, mod_a, self_wires, xte, rounds)
                self_acc = float((logits.argmax(-1) == yte).float().mean())

                single_a = float(
                    (unroll_logits(mod_a, xte, budget["steps_single"]).argmax(-1) == yte).float().mean()
                )
                single_b = float(
                    (unroll_logits(mod_b, xte, budget["steps_single"]).argmax(-1) == yte).float().mean()
                )
                best_single = max(single_a, single_b)
                ens_logits = (
                    unroll_logits(mod_a, xte, budget["steps_ensemble"])
                    + unroll_logits(mod_b, xte, budget["steps_ensemble"])
                ) / 2
                ensemble_acc = float((ens_logits.argmax(-1) == yte).float().mean())

            delta_max = debate_acc - max(best_single, ensemble_acc)
            d_max.append(delta_max)
            d_shuf.append(debate_acc - shuffled_acc)
            d_self.append(debate_acc - self_acc)
            err_corrs.append(err_corr)
            calibrated_flags.append(bool(d3["regime_calibrated"]))
            per_seed.append(
                {
                    "seed": s,
                    "debate_acc": round(debate_acc, 4),
                    "single_acc_a": round(single_a, 4),
                    "single_acc_b": round(single_b, 4),
                    "best_single_acc": round(best_single, 4),
                    "ensemble_acc": round(ensemble_acc, 4),
                    "self_debate_acc": round(self_acc, 4),
                    "shuffled_partner_acc": round(shuffled_acc, 4),
                    "delta_vs_max": round(delta_max, 4),
                    "delta_vs_shuffled": round(debate_acc - shuffled_acc, 4),
                    "delta_vs_self_debate": round(debate_acc - self_acc, 4),
                    "module_error_correlation": round(err_corr, 4),
                    "referee_weight_mean": round(info["w_mean"], 4),
                    "d3": d3,
                }
            )

        ci_max = seed_ci(d_max)
        ci_shuf = seed_ci(d_shuf)
        ci_self = seed_ci(d_self)
        flips = sign_flip_report(d_max)
        calibrated_all = all(calibrated_flags)
        matched_all = bool(match_single["matched"] and match_ens["matched"])

        win = bool(
            calibrated_all
            and matched_all
            and ci_max["lo"] > 0
            and ci_shuf["lo"] > 0
            and ci_self["lo"] > 0
            and flips["consistent_sign"] == 1
        )
        claim_strength = "verified" if (win and pr1["green"]) else ("plausible_unverified" if win else "n/a")
        if not calibrated_all:
            verdict = "UNREADABLE: regime not D3-calibrated"
        elif win and pr1["green"]:
            verdict = "WIN (PR1 GREEN, live): latent exchange adds over the single module and the ensemble"
        elif win:
            verdict = (
                "WIN but PLAUSIBLE-BUT-UNVERIFIED: PR1 gate is "
                f"{pr1['gate']}, pending the Studio regime (manifest appendix demotion)"
            )
        elif ci_max["lo"] <= 0 or flips["consistent_sign"] != 1:
            verdict = "NULL: debate ties max(single, ensemble) at matched total FLOPs (the ex17 result)"
        elif ci_shuf["lo"] <= 0:
            verdict = "NULL: debate ties the shuffled-partner arm (exchange content carries no signal)"
        else:
            verdict = "NULL: debate ties self-debate (self-generated depth, not cross-module exchange)"

        return {
            "experiment": self.id,
            "config": OmegaConf.to_container(cfg.experiment),
            "pr1": pr1,
            "per_seed": per_seed,
            "delta_vs_max_ci": ci_max,
            "delta_vs_shuffled_ci": ci_shuf,
            "delta_vs_self_debate_ci": ci_self,
            "sign_flips": flips,
            "module_error_correlation_mean": round(sum(err_corrs) / len(err_corrs), 4),
            "flop_budget": budget,
            "compute_vs_single": match_single,
            "compute_vs_ensemble": match_ens,
            "regime_calibrated_all_seeds": bool(calibrated_all),
            "preregistered": {
                "flop_tol": FLOP_TOL,
                "win_rule": "debate beats max(single, ensemble) AND shuffled-partner AND self-debate, "
                "each seed CI excluding 0, consistent sign, matched total FLOPs, calibrated regime; "
                "PR1 non-GREEN demotes a win to PLAUSIBLE-BUT-UNVERIFIED",
            },
            "claim_strength": claim_strength,
            "verdict": verdict,
            "null_supported": bool(not win),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=MT8LatentDebate.__doc__)
    ap.add_argument("--seeds", default="0-4", help="seed range 0-4 or list 0,1,2")
    ap.add_argument("--out", default="runs/mot/mt8_latent_debate.json")
    ap.add_argument("--pr1", default=PR1_PATH, help="PR1 verdict json (read only, never rebuilt)")
    ap.add_argument("--rerun", action="store_true", help="Q4.1 10-seed rerun naming (_seeds10)")
    args = ap.parse_args(argv)
    out = resolve_out(args.out, args.rerun)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_cfg(parse_seeds(args.seeds), pr1_path=args.pr1)
    t0 = time.time()
    result = MT8LatentDebate().run(cfg, resolve("cpu"), out.parent)
    result["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"experiment": result["experiment"], "verdict": result["verdict"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
