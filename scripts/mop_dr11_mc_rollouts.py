#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from mop.devices import DeviceInfo, resolve
from mop.diagnostics.compute import matched_within, mlp_flops
from mop.diagnostics.riskcov import seed_ci, sign_flip_report
from mop.experiments.base import Experiment
from mop.experiments.ex2_latent_planning import (
    _DynamicsModel,
    _r2,
    _sample_actions,
    _step_true,
    _train_dynamics,
    _true_dynamics_params,
)
from mop.seeding import parse_seeds, seed_everything

MC_MARGIN = 0.05  # MC must beat the matched deterministic arm's mean true-env distance by this
NOISE_REGIME_MARGIN = 0.05  # any MC "advantage" in the noise-dominant regime above this fails the guard
DIVERSITY_FLOOR = 1e-3  # normalized terminal spread below this = samples collapsed to the mean
R2_FLOOR = 0.5  # the learnable regime's model must clear this, the noise regime must document its own
FLOP_TOL = 0.10
OUT_PATH = Path("runs/mot/dr11_mc_rollouts.json")


def default_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "seeds": [0, 1, 2, 3, 4],
            "dim": 24,
            "action_dim": 6,
            "hidden": 256,
            "epochs": 300,
            "lr": 1e-3,
            "n_train_transitions": 4000,
            "nonlinearity": 0.5,
            "noise_learnable": 0.3,  # moderate aleatoric process noise: stochastic but structured
            "noise_dominant": 3.0,  # process noise dwarfs the O(1) action effect: irreducible
            "n_trials": 40,
            "horizon": 6,
            "n_cand": 60,
            "k_mc": 8,
            "n_exec": 16,  # true-env noise realizations averaged when scoring the selected plan
        }
    )


def stochastic_rollout(
    model: _DynamicsModel, z0: torch.Tensor, seqs: torch.Tensor, resid_std: float, g: torch.Generator
) -> torch.Tensor:
    n, horizon, _ = seqs.shape
    z = z0.unsqueeze(0).expand(n, -1).clone()
    with torch.no_grad():
        for t in range(horizon):
            z = model(z, seqs[:, t, :]) + resid_std * torch.randn(n, z.shape[1], generator=g)
    return z


def deterministic_rollout(model: _DynamicsModel, z0: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
    n, horizon, _ = seqs.shape
    z = z0.unsqueeze(0).expand(n, -1).clone()
    with torch.no_grad():
        for t in range(horizon):
            z = model(z, seqs[:, t, :])
    return z


def execute_true_stochastic(
    seq: torch.Tensor,
    z0: torch.Tensor,
    goal: torch.Tensor,
    params,
    nonlin: float,
    noise: float,
    n_exec: int,
    g: torch.Generator,
) -> float:
    z = z0.unsqueeze(0).expand(n_exec, -1).clone()
    with torch.no_grad():
        for t in range(seq.shape[0]):
            a = seq[t].unsqueeze(0).expand(n_exec, -1)
            z = _step_true(z, a, params, nonlin, noise, g)
    return float((z - goal.unsqueeze(0)).norm(dim=-1).mean())


def _run_regime(cfg: DictConfig, seed: int, noise: float, g: torch.Generator) -> dict:
    dim, adim, hid = int(cfg.dim), int(cfg.action_dim), int(cfg.hidden)
    horizon, n_cand, k_mc = int(cfg.horizon), int(cfg.n_cand), int(cfg.k_mc)
    nonlin = float(cfg.nonlinearity)

    params = _true_dynamics_params(dim, adim, nonlin, g)
    z_tr = torch.randn(int(cfg.n_train_transitions), dim, generator=g)
    a_tr = _sample_actions(int(cfg.n_train_transitions), adim, g)
    zn_tr = _step_true(z_tr, a_tr, params, nonlin, noise, g)
    model = _DynamicsModel(dim, adim, hid)
    _train_dynamics(model, z_tr, a_tr, zn_tr, int(cfg.epochs), float(cfg.lr))

    n_val = max(64, int(cfg.n_train_transitions) // 4)
    z_val = torch.randn(n_val, dim, generator=g)
    a_val = _sample_actions(n_val, adim, g)
    zn_val = _step_true(z_val, a_val, params, nonlin, noise, g)
    with torch.no_grad():
        pred_val = model(z_val, a_val)
    one_step_r2 = _r2(pred_val, zn_val)
    resid_std = float((pred_val - zn_val).std())  # the preregistered MC noise scale, no tuning

    mc_d, det_d, diversities = [], [], []
    for _ in range(int(cfg.n_trials)):
        z0 = torch.randn(dim, generator=g)
        goal = torch.randn(dim, generator=g)

        cand = torch.randn(n_cand, horizon, adim, generator=g).clamp(-2.0, 2.0)
        rep = cand.repeat_interleave(k_mc, dim=0)  # [n_cand * k_mc, H, A]
        term = stochastic_rollout(model, z0, rep, resid_std, g).view(n_cand, k_mc, dim)
        dist = (term - goal.view(1, 1, dim)).norm(dim=-1)  # [n_cand, k_mc]
        mc_seq = cand[int(torch.argmin(dist.mean(dim=1)))]
        spread = term.std(dim=1).mean()
        scale = term.norm(dim=-1).mean().clamp(min=1e-8)
        diversities.append(float(spread / scale))

        dcand = torch.randn(n_cand * k_mc, horizon, adim, generator=g).clamp(-2.0, 2.0)
        dterm = deterministic_rollout(model, z0, dcand)
        det_seq = dcand[int(torch.argmin((dterm - goal.unsqueeze(0)).norm(dim=-1)))]

        mc_d.append(execute_true_stochastic(mc_seq, z0, goal, params, nonlin, noise, int(cfg.n_exec), g))
        det_d.append(execute_true_stochastic(det_seq, z0, goal, params, nonlin, noise, int(cfg.n_exec), g))

    fwd_flops = mlp_flops([dim + adim, hid, hid, dim])
    mc_fwds = n_cand * k_mc * horizon
    det_fwds = (n_cand * k_mc) * horizon
    mean_mc = sum(mc_d) / max(1, len(mc_d))
    mean_det = sum(det_d) / max(1, len(det_d))
    return {
        "process_noise": noise,
        "one_step_r2": round(one_step_r2, 4),
        "resid_std": round(resid_std, 4),
        "mean_terminal_dist_true": {"mc": round(mean_mc, 4), "deterministic": round(mean_det, 4)},
        "mc_gain": round(mean_det - mean_mc, 4),  # positive = MC closer to goal
        "diversity": round(sum(diversities) / max(1, len(diversities)), 6),
        "compute": {
            "dyn_forward_flops": fwd_flops,
            "mc_forwards_per_trial": mc_fwds,
            "det_forwards_per_trial": det_fwds,
            "matched": matched_within(mc_fwds * fwd_flops, det_fwds * fwd_flops, tol=FLOP_TOL),
        },
    }


def _run_seed(cfg: DictConfig, seed: int) -> dict:
    seed_everything(seed)
    g = torch.Generator().manual_seed(seed)
    learnable = _run_regime(cfg, seed, float(cfg.noise_learnable), g)
    noise_dom = _run_regime(cfg, seed, float(cfg.noise_dominant), g)
    mc_wins = bool(learnable["mc_gain"] > MC_MARGIN)
    guard_ok = bool(noise_dom["mc_gain"] <= NOISE_REGIME_MARGIN)
    diversity_ok = bool(learnable["diversity"] > DIVERSITY_FLOOR)
    return {
        "seed": seed,
        "learnable_regime": learnable,
        "noise_dominant_regime": noise_dom,
        "mc_beats_matched_deterministic": mc_wins,
        "noisy_tv_guard_ok": guard_ok,
        "diversity_above_floor": diversity_ok,
        "regime_model_r2_ok": bool(learnable["one_step_r2"] > R2_FLOOR),
    }


class DR11MCRollouts(Experiment):
    id = "mop_dr11_mc_rollouts"
    metric = ("true_env_terminal_distance", "mc_gain", "rollout_diversity")
    baseline = "deterministic rollouts over n_cand * k_mc candidates at identical total FLOPs"
    ablation = "noise-dominant regime (irreducible aleatoric outcomes): the noisy-TV guard arm"
    null_hypothesis = (
        "MC ties the matched deterministic budget (samples collapse to the mean) OR wins only by "
        "exploiting irreducible aleatoric noise (the advantage also appears in the noise-dominant "
        "regime, failing the noisy-TV guard)."
    )
    tier = "cpu-now"

    def run(self, cfg: DictConfig, device: DeviceInfo, run_dir: Path) -> dict:
        t0 = time.time()
        per_seed = [_run_seed(cfg, int(s)) for s in cfg.seeds]
        gains = [r["learnable_regime"]["mc_gain"] for r in per_seed]
        n_seeds = len(per_seed)
        n_wins = sum(1 for r in per_seed if r["mc_beats_matched_deterministic"])
        n_guard = sum(1 for r in per_seed if r["noisy_tv_guard_ok"])
        n_div = sum(1 for r in per_seed if r["diversity_above_floor"])
        n_r2 = sum(1 for r in per_seed if r["regime_model_r2_ok"])
        survives = n_wins == n_seeds and n_guard == n_seeds and n_div == n_seeds and n_r2 == n_seeds
        out = {
            "contract": self.contract(),
            "preregistered": {
                "mc_margin": MC_MARGIN,
                "noise_regime_margin": NOISE_REGIME_MARGIN,
                "diversity_floor": DIVERSITY_FLOOR,
                "r2_floor": R2_FLOOR,
                "flop_tol": FLOP_TOL,
                "survival_rule": "MC beats the matched deterministic arm on EVERY seed in the "
                "learnable regime, shows NO advantage in the noise-dominant regime "
                "on any seed, keeps sample diversity above the floor, and the "
                "learnable-regime model clears the R2 floor",
            },
            "cfg": OmegaConf.to_container(cfg),
            "per_seed": per_seed,
            "mc_gain_ci": seed_ci(gains),
            "mc_gain_sign_flips": sign_flip_report(gains),
            "n_seeds_mc_wins": n_wins,
            "n_seeds_guard_ok": n_guard,
            "n_seeds_diversity_ok": n_div,
            "n_seeds_r2_ok": n_r2,
            "n_seeds": n_seeds,
            "survives": survives,
            "null_supported": not survives,
            "wall_clock_s": round(time.time() - t0, 3),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "dr11_mc_rollouts.json").write_text(json.dumps(out, indent=2))
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description="DR11 MC rollouts (noisy-TV guarded, matched FLOPs)")
    ap.add_argument("--seeds", type=str, default="0-4")
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()
    cfg = default_cfg()
    cfg.seeds = parse_seeds(args.seeds)
    out_path = Path(args.out)
    result = DR11MCRollouts().run(cfg, resolve("cpu"), out_path.parent)
    written = out_path.parent / "dr11_mc_rollouts.json"
    if out_path != written:
        written.rename(out_path)
    print(f"[dr11] wrote {out_path}", file=sys.stderr)
    print(json.dumps({k: result[k] for k in ("survives", "null_supported", "mc_gain_ci")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
