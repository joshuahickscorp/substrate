#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from mop_mt5_adaptive_halting import parse_seeds, resolve_out
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.compute import matched_within, mlp_flops, refiner_flops
from mop.diagnostics.hardness import (
    SLOT_CARD,
    SLOT_ORDER,
    hardness_gradient_certificate,
    make_graded_slot_task,
)
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.seeding import seed_everything
from mop.shell.refine import IterativeRefiner
from mop.shell.verifier_exec import ExecutableVerifier

FLOP_TOL = 0.10  # every arm matched to the feedforward budget on TOTAL FLOPs within this


class SlotHeads(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.slots = nn.ModuleList([nn.Linear(dim, SLOT_CARD[name]) for name in SLOT_ORDER])
        self.label = nn.Linear(dim, 2)

    def decode_slots(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(z).argmax(-1) for h in self.slots], dim=1)

    def slot_logits(self, z: torch.Tensor) -> list[torch.Tensor]:
        return [h(z) for h in self.slots]


def train_backbone(task, dim, hidden, n_max, epochs, lr, seed):
    seed_everything(seed)
    refiner = IterativeRefiner(dim, hidden, n_max)
    heads = SlotHeads(dim)
    opt = torch.optim.Adam([*refiner.parameters(), *heads.parameters()], lr=lr)
    x, slots, y = task.x, task.slots, task.y
    for _ in range(epochs):
        opt.zero_grad()
        z = x
        losses = []
        for _ in range(n_max):
            z = z + refiner._update(z)
            for i, logit in enumerate(heads.slot_logits(z)):
                losses.append(F.cross_entropy(logit, slots[:, i]))
            losses.append(F.cross_entropy(heads.label(z), y))
        torch.stack(losses).mean().backward()
        opt.step()
    return refiner, heads


@torch.no_grad()
def guided_loop(refiner, heads, verifier, x, n_start, n_max, shuffle_seed=None):
    z = x.clone()
    n = x.shape[0]
    for _ in range(int(n_start)):
        z = z + refiner._update(z)
    committed = torch.full((n,), -1, dtype=torch.long)
    used = torch.full((n,), float(n_start))
    checks = torch.zeros(n)
    g = torch.Generator().manual_seed(shuffle_seed) if shuffle_seed is not None else None
    while True:
        active = committed < 0
        if not active.any() or float(used[active].max()) >= n_max:
            break
        cand_slots = heads.decode_slots(z)
        cand_label = heads.label(z).argmax(-1)
        ok = verifier.consistent(cand_slots, cand_label)  # pure code execution
        executed = verifier.execute(cand_slots)
        checks = checks + active.float()
        if g is not None:  # shuffled control: destroy per-sample alignment of the verdict
            perm = torch.randperm(n, generator=g)
            ok = ok[perm]
        verify_now = active & ok
        committed = torch.where(verify_now, executed, committed)
        revise = active & (~ok) & (used < n_max)
        if not revise.any():
            break
        upd = refiner._update(z)
        z = torch.where(revise.unsqueeze(-1), z + upd, z)
        used = torch.where(revise, used + 1, used)
    fallback = heads.label(z).argmax(-1)
    pred = torch.where(committed >= 0, committed, fallback)
    return pred, float(used.mean()), float(checks.mean())


@torch.no_grad()
def matched_feedforward_eval(refiner, heads, x, mean_steps, n_max, seed):
    import math

    lo = int(math.floor(mean_steps))
    hi = min(lo + 1, int(n_max))
    frac = mean_steps - lo
    g = torch.Generator().manual_seed(seed + 4242)
    take_hi = torch.rand(x.shape[0], generator=g) < frac
    steps_per = torch.where(take_hi, torch.tensor(hi), torch.tensor(lo)).clamp(min=1)
    z = x.clone()
    for t in range(int(steps_per.max().item())):
        active = steps_per > t
        z = torch.where(active.unsqueeze(-1), z + refiner._update(z), z)
    return heads.label(z).argmax(-1), float(steps_per.float().mean())


class D3VerifierReasoning(Experiment):
    id = "mop_d3_verifier_reasoning"
    metric = ("feedforward_hard_acc", "verifier_hard_acc", "delta_hard", "shuffled_hard_acc")
    baseline = "matched-FLOP fixed-N feedforward refinement reading the label head, no verifier"
    ablation = "executable DSL verifier vs row-shuffled verdicts in the same guided loop; hard-bin grading"
    null_hypothesis = (
        "verifier-guided test-time compute ties matched-FLOP feedforward on the hard bin even with a "
        "perfect executable oracle checker (kill-switch: test-time compute is dead at this substrate)"
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        e = cfg.experiment
        seeds = list(e.seeds)
        dim_slot = int(e.slot_dim)
        hidden = int(e.hidden)
        n_max = int(e.n_max)
        epochs, lr = int(e.epochs), float(e.lr)

        verifier = ExecutableVerifier()
        per_seed: list[dict] = []
        deltas_hard, deltas_all, vs_shuffled_hard = [], [], []
        calibrated_flags, gradient_flags, matched_flags = [], [], []
        best_mode_accs = []
        for s in seeds:
            task = make_graded_slot_task(
                int(e.samples),
                slot_dim=dim_slot,
                noise=float(e.noise),
                hard_frac=float(e.hard_frac),
                hard_threshold=int(e.hard_threshold),
                seed=s,
            )
            dim = task.dim
            n = task.x.shape[0]
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(s))
            cut = int(n * 0.7)
            tr, te = perm[:cut], perm[cut:]
            xte, yte = task.x[te], task.y[te]
            hard_te = task.hard_mask[te]

            cert = hardness_gradient_certificate(task, seed=s, margin=float(e.gradient_margin))

            refiner, heads = train_backbone(task_slice(task, tr), dim, hidden, n_max, epochs, lr, s)

            per_step = refiner_flops(dim, hidden, 1)
            per_decode = mlp_flops([dim, 2]) + sum(mlp_flops([dim, SLOT_CARD[nm]]) for nm in SLOT_ORDER)
            per_check = verifier.flops_per_check(1)

            n_start = int(e.n_start)
            g_pred, g_steps, g_checks = guided_loop(refiner, heads, verifier, xte, n_start, n_max)
            sh_pred, sh_steps, sh_checks = guided_loop(
                refiner, heads, verifier, xte, n_start, n_max, shuffle_seed=s + 991
            )
            g_flops = g_steps * per_step + g_checks * (per_decode + per_check)
            ff_extra_decode = per_decode  # feedforward's single label read, charged symmetrically
            g_total = g_flops + per_decode  # guided's final fallback read, charged too
            ff_mean_steps = max(1.0, (g_total - ff_extra_decode) / per_step)

            ff_pred, ff_realized = matched_feedforward_eval(refiner, heads, xte, ff_mean_steps, n_max, s)
            ff_flops = ff_realized * per_step + ff_extra_decode

            def acc_on(pred, mask, target):
                if not mask.any():
                    return 0.0
                return float((pred[mask] == target[mask]).float().mean())

            ff_hard = acc_on(ff_pred, hard_te, yte)
            g_hard = acc_on(g_pred, hard_te, yte)
            sh_hard = acc_on(sh_pred, hard_te, yte)
            ff_all = float((ff_pred == yte).float().mean())
            g_all = float((g_pred == yte).float().mean())

            compute = matched_within(int(g_total), int(ff_flops), tol=FLOP_TOL)
            d_hard = g_hard - ff_hard
            d_all = g_all - ff_all
            d_shuf_hard = g_hard - sh_hard

            deltas_hard.append(d_hard)
            deltas_all.append(d_all)
            vs_shuffled_hard.append(d_shuf_hard)
            matched_flags.append(bool(compute["matched"]))
            calibrated_flags.append(bool(cert["regime_calibrated"]))
            gradient_flags.append(bool(cert["gradient_present"]))
            best_mode_accs.append(max(ff_all, g_all))
            per_seed.append(
                {
                    "seed": s,
                    "n_hard_test": int(hard_te.sum()),
                    "feedforward_hard_acc": round(ff_hard, 4),
                    "verifier_hard_acc": round(g_hard, 4),
                    "shuffled_hard_acc": round(sh_hard, 4),
                    "delta_hard": round(d_hard, 4),
                    "delta_vs_shuffled_hard": round(d_shuf_hard, 4),
                    "feedforward_all_acc": round(ff_all, 4),
                    "verifier_all_acc": round(g_all, 4),
                    "delta_all": round(d_all, 4),
                    "guided_mean_steps": round(g_steps, 3),
                    "guided_mean_checks": round(g_checks, 3),
                    "guided_n_start": n_start,
                    "feedforward_matched_steps": round(ff_realized, 3),
                    "compute": compute,
                    "hardness_gradient": cert,
                }
            )

        ci_hard = seed_ci(deltas_hard)
        ci_all = seed_ci(deltas_all)
        ci_shuf = seed_ci(vs_shuffled_hard)
        flips_hard = sign_flip_report(deltas_hard)
        matched_all = all(matched_flags)
        regime_readable = all(calibrated_flags) and all(gradient_flags)
        best_mode_mean = sum(best_mode_accs) / len(best_mode_accs)
        in_target_band = bool(0.55 <= best_mode_mean <= 0.85)

        win = bool(
            regime_readable
            and matched_all
            and ci_hard["lo"] > 0
            and flips_hard["consistent_sign"] == 1
            and ci_shuf["lo"] > 0
        )
        if not regime_readable:
            verdict = "UNREADABLE: regime not D3-calibrated or no hardness gradient (cannot grade)"
        elif not matched_all:
            verdict = "UNREADABLE: guided arm not matched to feedforward on total FLOPs"
        elif win:
            verdict = (
                "WIN: the executable DSL verifier carries a correction signal, verifier-guided iteration "
                "beats matched-FLOP feedforward on the HARD bin and beats the shuffled control"
            )
        elif ci_hard["lo"] <= 0:
            verdict = (
                "NULL (kill-switch): verifier-guided ties matched-FLOP feedforward on the hard bin even "
                "with a perfect executable oracle, test-time compute is dead at this substrate"
            )
        else:
            verdict = (
                "NULL: verifier-guided beats feedforward but ties the shuffled control on the hard bin "
                "(reallocation, not DSL alignment)"
            )

        return {
            "experiment": self.id,
            "config": OmegaConf.to_container(cfg.experiment),
            "per_seed": per_seed,
            "delta_hard_ci": ci_hard,
            "delta_all_ci": ci_all,
            "delta_vs_shuffled_hard_ci": ci_shuf,
            "sign_flips_hard": flips_hard,
            "compute_matched_all_seeds": bool(matched_all),
            "regime_readable": bool(regime_readable),
            "best_mode_acc_mean": round(best_mode_mean, 4),
            "best_mode_in_target_band": in_target_band,
            "preregistered": {
                "flop_tol": FLOP_TOL,
                "target_band": [0.55, 0.85],
                "win_rule": "hard-bin delta CI excludes 0, consistent sign, beats shuffled, matched "
                "FLOPs, calibrated regime with a hardness gradient",
                "killswitch_rule": "hard-bin delta CI includes 0 with the executable oracle == "
                "test-time compute dead at this substrate",
            },
            "verdict": verdict,
            "null_supported": bool(not win),
        }


def task_slice(task, idx):
    from mop.diagnostics.hardness import GradedTask

    return GradedTask(
        x=task.x[idx],
        slots=task.slots[idx],
        y=task.y[idx],
        hardness=task.hardness[idx],
        hard_mask=task.hard_mask[idx],
        slot_dim=task.slot_dim,
        dim=task.dim,
    )


def default_cfg(seeds: list[int], **overrides) -> DictConfig:
    exp = {
        "seeds": list(seeds),
        "slot_dim": 16,
        "hidden": 128,
        "samples": 3000,
        "noise": 4.5,  # tuned so best-mode accuracy lands in the 0.55 to 0.85 target band (non-ceiling)
        "hard_frac": 0.45,
        "hard_threshold": 2,
        "n_start": 2,
        "n_max": 8,
        "epochs": 200,
        "lr": 3e-3,
        "gradient_margin": 0.05,
    }
    exp.update(overrides)
    return OmegaConf.create({"experiment": exp})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=D3VerifierReasoning.__doc__)
    ap.add_argument("--seeds", default="0-9", help="seed range 0-9 or list 0,1,2")
    ap.add_argument("--out", default="runs/mot/d3_verifier_reasoning.json")
    ap.add_argument("--rerun", action="store_true", help="Q4.1 10-seed rerun naming (_seeds10)")
    args = ap.parse_args(argv)
    out = resolve_out(args.out, args.rerun)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_cfg(parse_seeds(args.seeds))
    t0 = time.time()
    result = D3VerifierReasoning().run(cfg, resolve("cpu"), out.parent)
    result["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"experiment": result["experiment"], "verdict": result["verdict"], "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
