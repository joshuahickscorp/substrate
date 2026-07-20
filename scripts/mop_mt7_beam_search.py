#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from mop_dr9_verify_revise import train_backbone
from mop_mt5_adaptive_halting import parse_seeds, resolve_out
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.compute import matched_within, mlp_flops, refiner_flops
from mop.diagnostics.difficulty_calibration import reference_separation
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import seed_everything
from mop.shell.refine import IterativeRefiner, Verifier

FLOP_TOL = 0.10  # preregistered: search rows are matched on TOTAL FLOPs within this (manifest appendix)


def default_cfg(seeds: list[int], **overrides) -> DictConfig:
    exp = {
        "seeds": list(seeds),
        "dim": 64,
        "hidden": 128,
        "verifier_hidden": 64,
        "n_classes": 6,
        "samples": 3000,
        "beam_width": 4,
        "expansions": 2,
        "rounds": 6,
        "branch_sigma": 0.5,
        "epochs": 200,
        "lr": 1e-3,
        "sep": 3.0,
        "ambig_frac": 0.5,
    }
    exp.update(overrides)
    return OmegaConf.create({"experiment": exp})


def make_ambiguous_split(
    n: int, dim: int, n_classes: int, sep: float, ambig_frac: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, dim, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    ambig = torch.rand(n, generator=g) < ambig_frac
    offset = torch.randint(1, n_classes, (n,), generator=g)
    distractor = (y + offset) % n_classes
    base = centers[y] * sep
    midpoint = 0.5 * (centers[y] + centers[distractor]) * sep
    x = torch.where(ambig.unsqueeze(-1), midpoint, base) + torch.randn(n, dim, generator=g) * 0.5
    return x, y, ambig


def beam_flop_schedule(rounds: int, beam_width: int, expansions: int, per_step: int, per_eval: int) -> dict:
    k, total, expanded, widths = 1, 0, 0, []
    for _ in range(int(rounds)):
        children = k * int(expansions)
        total += children * (per_step + per_eval)
        expanded += children
        k = min(int(beam_width), children)
        widths.append(k)
    return {"total_flops": int(total), "expansions": int(expanded), "kept_widths": widths}


@torch.no_grad()
def beam_search_eval(
    refiner: IterativeRefiner,
    head: nn.Module,
    verifier: Verifier,
    x: torch.Tensor,
    y: torch.Tensor,
    beam_width: int,
    expansions: int,
    rounds: int,
    sigma: float,
    score_mode: str,
    seed: int,
) -> float:
    if score_mode not in ("trained", "shuffled", "oracle"):
        raise ValueError(f"unknown score_mode {score_mode!r}")
    b, dim = x.shape
    g = torch.Generator().manual_seed(seed + 977)
    z = x.unsqueeze(1)  # [B, k=1, D]
    last_scores = torch.zeros(b, 1)
    for _ in range(int(rounds)):
        k = z.shape[1]
        u = refiner._update(z.reshape(b * k, dim)).reshape(b, k, dim)
        children = [z + u]
        for _ in range(int(expansions) - 1):
            noise = sigma * torch.randn(b, k, dim, generator=g)
            children.append(z + u + noise)
        cand = torch.cat(children, dim=1)  # [B, k*E, D]
        n_cand = cand.shape[1]
        if score_mode == "oracle":
            logits = head(cand.reshape(b * n_cand, dim)).reshape(b, n_cand, -1)
            scores = F.cross_entropy(
                logits.reshape(b * n_cand, -1), y.repeat_interleave(n_cand), reduction="none"
            ).reshape(b, n_cand)
        else:
            scores = verifier.score(cand.reshape(b * n_cand, dim)).reshape(b, n_cand)
            if score_mode == "shuffled":
                scores = scores[torch.randperm(b, generator=g)]
        keep = min(int(beam_width), n_cand)
        idx = scores.topk(keep, dim=1, largest=False).indices
        z = torch.gather(cand, 1, idx.unsqueeze(-1).expand(-1, -1, dim))
        last_scores = torch.gather(scores, 1, idx)
    best = last_scores.argmin(dim=1)
    z_best = z[torch.arange(b), best]
    return float((head(z_best).argmax(-1) == y).float().mean())


def train_greedy_chain(
    x: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    dim: int,
    hidden: int,
    depth: int,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[IterativeRefiner, nn.Linear]:
    seed_everything(seed)
    refiner = IterativeRefiner(dim, hidden, int(depth))
    head = nn.Linear(dim, n_classes)
    opt = torch.optim.Adam([*refiner.parameters(), *head.parameters()], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        z = x
        losses = []
        for _ in range(int(depth)):
            z = z + refiner._update(z)
            losses.append(F.cross_entropy(head(z), y))
        torch.stack(losses).mean().backward()
        opt.step()
    return refiner, head


@torch.no_grad()
def greedy_eval(refiner: IterativeRefiner, head: nn.Module, x, y, steps: int) -> float:
    z = x
    for _ in range(int(steps)):
        z = z + refiner._update(z)
    return float((head(z).argmax(-1) == y).float().mean())


class MT7BeamSearch(Experiment):
    id = "mop_mt7_beam_search"
    metric = ("beam_acc", "greedy_matched_acc", "delta", "oracle_gap")
    baseline = (
        "deeper greedy chain trained with deep supervision at the FLOP-matched depth (every expanded "
        "beam child charged one refiner step plus one verifier score, pruned work counted)"
    )
    ablation = "row-shuffled branch scorer in the identical beam loop; oracle-scored beam upper bound"
    null_hypothesis = (
        "at matched total FLOPs, search ties a deeper greedy chain because the scorer cannot outrank "
        "the refiner's own step"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim, hidden, vhidden = int(e.dim), int(e.hidden), int(e.verifier_hidden)
        nc, rounds = int(e.n_classes), int(e.rounds)
        beam_width, expansions = int(e.beam_width), int(e.expansions)
        sigma, epochs, lr = float(e.branch_sigma), int(e.epochs), float(e.lr)

        per_step = refiner_flops(dim, hidden, 1)
        per_eval = mlp_flops([dim, vhidden, 1])
        schedule = beam_flop_schedule(rounds, beam_width, expansions, per_step, per_eval)
        greedy_depth = max(rounds, round(schedule["total_flops"] / per_step))
        compute = matched_within(schedule["total_flops"], greedy_depth * per_step, tol=FLOP_TOL)

        per_seed: list[dict] = []
        d_greedy, d_shuffled, oracle_gaps = [], [], []
        calibrated_flags, headroom_flags = [], []
        for s in seeds:
            x, y, _ambig = make_ambiguous_split(int(e.samples), dim, nc, float(e.sep), float(e.ambig_frac), s)
            cut = int(x.shape[0] * 0.7)
            xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
            d3 = reference_separation(xtr, ytr, seed=s)

            refiner, head, verifier = train_backbone(
                xtr, ytr, nc, dim, hidden, vhidden, rounds, epochs, lr, s
            )
            beam_acc = beam_search_eval(
                refiner, head, verifier, xte, yte, beam_width, expansions, rounds, sigma, "trained", s
            )
            shuffled_acc = beam_search_eval(
                refiner, head, verifier, xte, yte, beam_width, expansions, rounds, sigma, "shuffled", s
            )
            oracle_acc = beam_search_eval(
                refiner, head, verifier, xte, yte, beam_width, expansions, rounds, sigma, "oracle", s
            )
            greedy_horizon_acc = greedy_eval(refiner, head, xte, yte, rounds)  # unmatched, context only

            g_refiner, g_head = train_greedy_chain(xtr, ytr, nc, dim, hidden, greedy_depth, epochs, lr, s)
            greedy_matched_acc = greedy_eval(g_refiner, g_head, xte, yte, greedy_depth)

            delta = beam_acc - greedy_matched_acc
            delta_shuf = beam_acc - shuffled_acc
            headroom = bool(oracle_acc > greedy_matched_acc)
            d_greedy.append(delta)
            d_shuffled.append(delta_shuf)
            oracle_gaps.append(oracle_acc - beam_acc)
            calibrated_flags.append(bool(d3["regime_calibrated"]))
            headroom_flags.append(headroom)
            per_seed.append(
                {
                    "seed": s,
                    "beam_acc": round(beam_acc, 4),
                    "greedy_matched_acc": round(greedy_matched_acc, 4),
                    "greedy_horizon_acc_unmatched": round(greedy_horizon_acc, 4),
                    "shuffled_scorer_acc": round(shuffled_acc, 4),
                    "oracle_beam_acc": round(oracle_acc, 4),
                    "delta_vs_greedy": round(delta, 4),
                    "delta_vs_shuffled": round(delta_shuf, 4),
                    "oracle_headroom_over_greedy": headroom,
                    "d3": d3,
                }
            )

        ci_greedy = seed_ci(d_greedy)
        ci_shuf = seed_ci(d_shuffled)
        flips = sign_flip_report(d_greedy)
        calibrated_all = all(calibrated_flags)
        headroom_all = all(headroom_flags)

        win = bool(
            calibrated_all
            and compute["matched"]
            and ci_greedy["lo"] > 0
            and ci_shuf["lo"] > 0
            and flips["consistent_sign"] == 1
        )
        if not calibrated_all:
            verdict = "UNREADABLE: regime not D3-calibrated"
        elif win:
            verdict = (
                "WIN: scored search beats the matched deeper greedy chain (the verifier outranks the "
                "refiner's own step)"
            )
        elif ci_greedy["lo"] <= 0 or flips["consistent_sign"] != 1:
            verdict = "NULL: search ties deeper greedy at matched total FLOPs, pruned work counted"
            if not headroom_all:
                verdict += " (weakly informative: no oracle headroom over greedy in every seed)"
        else:
            verdict = "NULL: trained scorer ties the shuffled branch scorer (exploration, not scoring)"

        return {
            "experiment": self.id,
            "config": OmegaConf.to_container(cfg.experiment),
            "per_seed": per_seed,
            "delta_vs_greedy_ci": ci_greedy,
            "delta_vs_shuffled_ci": ci_shuf,
            "oracle_gap_ci": seed_ci(oracle_gaps),
            "sign_flips": flips,
            "flop_schedule": schedule,
            "greedy_matched_depth": int(greedy_depth),
            "compute": compute,
            "regime_calibrated_all_seeds": bool(calibrated_all),
            "oracle_headroom_all_seeds": bool(headroom_all),
            "preregistered": {
                "flop_tol": FLOP_TOL,
                "win_rule": "beam beats matched greedy AND the shuffled-scorer beam, each seed CI "
                "excluding 0, consistent sign, matched total FLOPs incl. pruned work, calibrated regime",
                "pruned_work_rule": "every expanded child charged one refiner step plus one verifier "
                "score, kept or pruned",
            },
            "verdict": verdict,
            "null_supported": bool(not win),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=MT7BeamSearch.__doc__)
    ap.add_argument("--seeds", default="0-4", help="seed range 0-4 or list 0,1,2")
    ap.add_argument("--out", default="runs/mot/mt7_beam_search.json")
    ap.add_argument("--rerun", action="store_true", help="Q4.1 10-seed rerun naming (_seeds10)")
    args = ap.parse_args(argv)
    out = resolve_out(args.out, args.rerun)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_cfg(parse_seeds(args.seeds))
    t0 = time.time()
    result = MT7BeamSearch().run(cfg, resolve("cpu"), out.parent)
    result["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"experiment": result["experiment"], "verdict": result["verdict"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
