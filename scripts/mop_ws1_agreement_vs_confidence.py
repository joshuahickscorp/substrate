#!/usr/bin/env python
"""WS1 (WP-12, the central workspace gate): does cross-substrate AGREEMENT between two genuinely
different frozen encoders (zA = V-JEPA pooled, zB = DINOv2-S pooled, same clips) predict per-input
correctness better than the best single substrate's calibrated CONFIDENCE? A positive argues two frozen
encoders carry complementary correctness information (the workspace is not decoration) and gates WS3.

Signals, scored as predictors of the best single source's per-sample correctness on held-out data:
  confidence = max softmax probability of the best single source's head,
  agreement  = sum_k pA_k * pB_k (probability two independently sampled predictions coincide).
Headline delta = AUROC(agreement) - AUROC(confidence); AURC (risk-coverage) reported alongside, plus
the corr(errA, errB) complementarity headline.

ALL FIVE PREREGISTERED CONTROLS SHIP IN THIS FILE (thresholds fixed before any result exists):
  1. invertible-remap vacuity guard: rerun the pipeline with source B replaced by a ridge linear remap
     of A; guard passes iff the real delta beats the remap arm's delta on every seed.
  2. shuffled-cross-source: agreement with B's rows permuted; guard passes iff the real agreement AUROC
     beats the shuffled one on every seed.
  3. noisy-TV: the three-boolean contract from diagnostics/noisy_tv.py must pass jointly on every seed.
  4. corr(errA, errB) reported as the complementarity headline (not a gate).
  5. 5-seed CI with sign flips on the headline delta.

PREREGISTERED NULL (verbatim, registry WS1): agreement AUROC minus single-substrate-confidence AUROC
<= 0 within seed CI, OR the win fails the invertible-remap vacuity guard, the shuffled-cross-source
control, or noisy-TV.
PREREGISTERED VERDICT RULE: WS1 is positive iff the delta CI excludes zero from below with no sign
flip, guards 1-3 all pass, and the task is non-ceiling (best single-source accuracy <= 0.97, the D3
requirement). Anything else supports the null. WS3 reads `ws1_positive` from this run's json.

Depends at run time on the WP-11 V-JEPA cache and the WP-01 DINOv2-S cache; never loads an encoder.

Usage: python scripts/mop_ws1_agreement_vs_confidence.py --seeds 0-4

No em dashes or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cache_randominit_vitl_features import load_feature_cache  # noqa: E402

from mop.diagnostics.noisy_tv import noisy_tv_diagnostic  # noqa: E402
from mop.diagnostics.riskcov import auroc, risk_coverage, seed_ci, sign_flip_report  # noqa: E402
from mop.experiments.base import Experiment  # noqa: E402
from mop.seeding import parse_seeds as _parse_seeds  # noqa: E402

DEFAULTS = {
    "cache_a": "data/cache/vjepa2_vitl_nuisance",
    "cache_b": "data/cache/dinov2s_nuisance_real",
    "seeds": (0, 1, 2, 3, 4),
    "epochs": 300,
    "lr": 1e-2,
    "test_frac": 0.3,
    "ceiling": 0.97,
    "ridge_lam_scale": 1e-2,
    "noisy_tv_steps": 400,
    "noisy_tv_dim": 32,
}


def _get(cfg, key, default):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_dual_source(cache_a: str, cache_b: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the two pooled-feature caches over the SAME clips; returns (za, zb, labels). Raises if the
    clip populations disagree (the triplet-store identity requirement)."""
    a, b = load_feature_cache(cache_a), load_feature_cache(cache_b)
    if a["features"].shape[0] != b["features"].shape[0]:
        raise ValueError("dual-source caches have different clip counts: not the same population")
    ya, yb = a["labels_shape"], b["labels_shape"]
    if ya is None and yb is None:
        raise ValueError("neither cache carries labels_shape")
    if ya is not None and yb is not None and not torch.equal(ya, yb):
        raise ValueError("dual-source caches disagree on labels: clip identity rule violated")
    return a["features"], b["features"], ya if ya is not None else yb


def split_train_test(n: int, test_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    cut = int(n * (1 - test_frac))
    return perm[:cut], perm[cut:]


def zscore_by(train: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    mu = train.mean(0, keepdim=True)
    sd = train.std(0, keepdim=True) + 1e-6
    return (x - mu) / sd


def train_softmax_head(
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    n_classes: int,
    *,
    epochs: int,
    lr: float,
    seed: int,
    model: nn.Module | None = None,
    weight_decay: float = 0.0,
) -> nn.Module:
    """Full-batch Adam training of a classification head (linear by default). Seeded init so arms with
    the same seed start identically."""
    torch.manual_seed(seed)
    head = model if model is not None else nn.Linear(xtr.shape[1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(xtr), ytr).backward()
        opt.step()
    head.eval()
    return head


@torch.no_grad()
def predict_probs(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return F.softmax(head(x), dim=-1)


def _ridge_map(xa: torch.Tensor, xb: torch.Tensor, lam_scale: float) -> torch.Tensor:
    """Closed-form ridge regression map A -> B on z-scored train features (the invertible-remap guard's
    'B is just a linear function of A' hypothesis, fit as well as a linear map can)."""
    d = xa.shape[1]
    lam = lam_scale * d
    return torch.linalg.solve(xa.T @ xa + lam * torch.eye(d), xa.T @ xb)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if denom > 0 else 0.0


def _signals(pa: torch.Tensor, pb: torch.Tensor, y: torch.Tensor) -> dict:
    """Best-single-source correctness target plus the two competing predictors of it."""
    acc_a = float((pa.argmax(-1) == y).float().mean())
    acc_b = float((pb.argmax(-1) == y).float().mean())
    best, p_best = ("a", pa) if acc_a >= acc_b else ("b", pb)
    correct = (p_best.argmax(-1) == y).float()
    return {
        "best": best,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "correct": correct,
        "confidence": p_best.max(-1).values,
        "agreement": (pa * pb).sum(-1),
    }


def run_seed(za: torch.Tensor, zb: torch.Tensor, y: torch.Tensor, seed: int, cfg) -> dict:
    epochs = int(_get(cfg, "epochs", DEFAULTS["epochs"]))
    lr = float(_get(cfg, "lr", DEFAULTS["lr"]))
    n_classes = int(y.max()) + 1
    tr, te = split_train_test(len(y), float(_get(cfg, "test_frac", DEFAULTS["test_frac"])), seed)
    xa, xb = zscore_by(za[tr], za), zscore_by(zb[tr], zb)
    head_a = train_softmax_head(xa[tr], y[tr], n_classes, epochs=epochs, lr=lr, seed=seed)
    head_b = train_softmax_head(xb[tr], y[tr], n_classes, epochs=epochs, lr=lr, seed=seed + 1)
    pa, pb = predict_probs(head_a, xa[te]), predict_probs(head_b, xb[te])
    sig = _signals(pa, pb, y[te])
    auroc_conf = auroc(sig["confidence"], sig["correct"])
    auroc_agree = auroc(sig["agreement"], sig["correct"])
    delta = auroc_agree - auroc_conf
    err_a, err_b = (pa.argmax(-1) != y[te]).float(), (pb.argmax(-1) != y[te]).float()

    # control 2: shuffled-cross-source (break the pairing, agreement should lose its information)
    g = torch.Generator().manual_seed(seed + 7)
    perm = torch.randperm(len(te), generator=g)
    auroc_shuf = auroc((pa * pb[perm]).sum(-1), sig["correct"])

    # control 1: invertible-remap vacuity guard (replace B with a ridge linear remap of A end to end)
    w = _ridge_map(xa[tr], xb[tr], float(_get(cfg, "ridge_lam_scale", DEFAULTS["ridge_lam_scale"])))
    xb_hat = xa @ w
    head_r = train_softmax_head(xb_hat[tr], y[tr], n_classes, epochs=epochs, lr=lr, seed=seed + 2)
    pr = predict_probs(head_r, xb_hat[te])
    sig_r = _signals(pa, pr, y[te])
    delta_remap = auroc(sig_r["agreement"], sig_r["correct"]) - auroc(sig_r["confidence"], sig_r["correct"])

    # control 3: noisy-TV three-boolean contract (agreement/disagreement must be epistemic)
    ntv = noisy_tv_diagnostic(
        dim=int(_get(cfg, "noisy_tv_dim", DEFAULTS["noisy_tv_dim"])),
        ensemble_size=3,
        steps=int(_get(cfg, "noisy_tv_steps", DEFAULTS["noisy_tv_steps"])),
        batch=64,
        seed=seed,
    )
    ntv_pass = bool(
        ntv["noise_error_stays_high"]
        and ntv["epistemic_collapses_on_noise"]
        and ntv["learning_progress_separates"]
    )
    return {
        "seed": seed,
        "best_source": sig["best"],
        "acc_a": round(sig["acc_a"], 4),
        "acc_b": round(sig["acc_b"], 4),
        "auroc_confidence": round(auroc_conf, 4),
        "auroc_agreement": round(auroc_agree, 4),
        "delta": round(delta, 4),
        "aurc_confidence": risk_coverage(sig["confidence"], sig["correct"])["aurc"],
        "aurc_agreement": risk_coverage(sig["agreement"], sig["correct"])["aurc"],
        "err_corr": round(_pearson(err_a, err_b), 4),
        "shuffle_margin": round(auroc_agree - auroc_shuf, 4),
        "remap_margin": round(delta - delta_remap, 4),
        "noisy_tv_pass": ntv_pass,
        "noisy_tv": {k: v for k, v in ntv.items() if isinstance(v, bool)},
    }


def run(cfg=None, device=None, run_dir: Path | None = None) -> dict:
    za, zb, y = load_dual_source(
        _get(cfg, "cache_a", DEFAULTS["cache_a"]), _get(cfg, "cache_b", DEFAULTS["cache_b"])
    )
    seeds = list(_get(cfg, "seeds", DEFAULTS["seeds"]))
    ceiling = float(_get(cfg, "ceiling", DEFAULTS["ceiling"]))
    t0 = time.perf_counter()
    per_seed = [run_seed(za, zb, y, s, cfg) for s in seeds]
    deltas = [r["delta"] for r in per_seed]
    ci = seed_ci(deltas)
    flips = sign_flip_report(deltas)
    guards = {
        "remap_pass": sign_flip_report([r["remap_margin"] for r in per_seed])["consistent_sign"] == 1,
        "shuffle_pass": sign_flip_report([r["shuffle_margin"] for r in per_seed])["consistent_sign"] == 1,
        "noisy_tv_pass": all(r["noisy_tv_pass"] for r in per_seed),
    }
    best_acc = max(
        sum(r["acc_a"] for r in per_seed) / len(per_seed),
        sum(r["acc_b"] for r in per_seed) / len(per_seed),
    )
    non_ceiling = best_acc <= ceiling
    positive = bool(ci["lo"] > 0 and flips["consistent_sign"] == 1 and all(guards.values()) and non_ceiling)
    out = {
        "experiment": WS1Experiment.id,
        "contract": WS1Experiment().contract(),
        "seeds": seeds,
        "per_seed": per_seed,
        "delta_ci": ci,
        "sign_flips": flips,
        "guards": guards,
        "err_corr_mean": round(sum(r["err_corr"] for r in per_seed) / len(per_seed), 4),
        "best_single_acc_mean": round(best_acc, 4),
        "non_ceiling": non_ceiling,
        "ws1_positive": positive,
        "null_supported": not positive,
        "seconds": round(time.perf_counter() - t0, 1),
        "verdict": (
            "WS1 POSITIVE: cross-substrate agreement beats single-source confidence with all guards "
            "passing; two frozen encoders carry complementary correctness info, WS3 unlocked"
            if positive
            else "NULL SUPPORTED: agreement does not beat confidence (or a guard/ceiling condition "
            "failed); one substrate's confidence suffices, the workspace reads as decoration"
        ),
    }
    if not non_ceiling:
        out["ceiling_note"] = "best single source above 0.97: test-too-easy, tie unreadable without AT4"
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    return out


class WS1Experiment(Experiment):
    id = "ws1_agreement_vs_confidence"
    metric = ("auroc_agreement_minus_confidence", "aurc", "err_corr")
    baseline = "best single substrate's calibrated confidence as the correctness predictor"
    ablation = "shuffled-cross-source and invertible-remap arms remove the genuine second substrate"
    null_hypothesis = (
        "Agreement AUROC minus single-substrate-confidence AUROC <= 0 within seed CI, OR the win fails "
        "the invertible-remap vacuity guard (B is a linear remap of A), the shuffled-cross-source "
        "control, or noisy-TV"
    )
    tier = "cpu-now"

    def run(self, cfg, device, run_dir: Path) -> dict:
        return run(cfg, device, run_dir)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WS1 agreement vs confidence (the central workspace gate)")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--cache-a", default=DEFAULTS["cache_a"])
    ap.add_argument("--cache-b", default=DEFAULTS["cache_b"])
    ap.add_argument("--rerun", action="store_true", help="stage-4 rerun naming (_seeds10 suffix)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = {"seeds": _parse_seeds(a.seeds), "cache_a": a.cache_a, "cache_b": a.cache_b}
    out_path = a.out or (
        "runs/mot/ws1_agreement_vs_confidence_seeds10.json"
        if a.rerun
        else "runs/mot/ws1_agreement_vs_confidence.json"
    )
    result = run(cfg, "cpu", None)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("delta_ci", "guards", "ws1_positive", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
