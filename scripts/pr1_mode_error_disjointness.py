#!/usr/bin/env python
"""PR1: mode-error disjointness, the cheap precondition for the whole Mixture-of-Thinking router line
(preregistered in docs/mixture_of_thinking/11_experiment_registry.md, PR full schema part 1).

Thesis: per-sample errors of DISTINCT reasoning modes are decorrelated enough that an ORACLE router
(pick the best mode per sample) beats the single best mode. If not, no learned router can help and
MT1/MT4 are not worth building. Pure diagnostic: five heterogeneous heads over the SAME synthetic
latents, per-sample 0/1 correctness on a SHARED test set, pairwise phi correlations, oracle bound.

Task: a preregistered MIXTURE of three equal sub-populations over the same 1024-d latent space, each
with a DIFFERENT Bayes-optimal hypothesis class, so complementarity is expressible at all (on plain
equal-covariance blobs the Bayes rule IS nearest-centroid and is linear, so every mode estimates the
same boundary and any oracle gain is pure estimation noise, which the H4 control subtracts):
  (1) linear_blobs: the established Gaussian-cluster generator (make_task_stream), linear Bayes rule,
  (2) antipodal: each class is a pair of antipodal blobs (+mu_c and -mu_c), class means collapse to
      the origin, so prototype and linear fail while nonlinear function classes (mlp, knn) succeed,
  (3) fine_grained_local: each class is several scattered low-separation sub-clusters, so local
      neighborhood structure (knn, the memory mode) wins over global parametric boundaries.
All three share the label space and the single separation difficulty knob.

The mandatory control (MT hypothesis H4 at diagnostic cost): oracle gain grows mechanically with the
number of arms even from pure seed noise, so we ALSO compute the oracle over K retrained seed-copies
of the single best mode (same arm count). Its baseline is the MEAN copy accuracy: the copies are
exchangeable, so max(copy_accs) carries a best-of-K winner's curse on the test set that the
well-separated heterogeneous arms do not, deflating hom_gain by ~0.01 and biasing toward GREEN.
For the deterministic modes (knn, prototype) fresh-seed retraining is a no-op, so their copies are
built on MILDLY SUBSAMPLED train sets (without replacement), with the fraction calibrated per seed
so the copy-accuracy spread matches the within-seed spread of genuine retrained copies of the best
TRAINED mode, and the control is VALID only if the copies' mean accuracy stays within that spread of
the real arm (full-size bootstrap failed this: it produced arms 6-10 points weaker than the real
prototype arm, so its "homogeneous" oracle measured bagging over degraded arms, not seed noise, and
forced NULL structurally). The verdict gates on the MAX over the valid homogeneous gains (the
genuine retrained-copies control is always computed and always valid; the subsample control joins it
only when calibration succeeds), conservative against false GREEN; an invalid subsample control is
reported as advisory only.

Modes (each distinct on a named MoT axis, all tiny heads on cached/synthetic 1024-d latents):
  reactive_linear (direct feedforward), mlp (different function class), knn (memory system,
  nonparametric), prototype (nearest class centroid, different inductive bias), recurrent_refiner
  (the existing IterativeRefiner from src/devsys/shell/refine.py driving a linear head).

Difficulty is CALIBRATED before the main run: the separation knob is swept until the best CHEAP
reference mode (max over linear, knn, prototype; on the mixture no single mode is representative)
lands mid-range (target 0.55-0.85 best-mode accuracy; on ceilinged content everyone is correct
everywhere and disjointness is undefined). Verdict thresholds are fixed below, before running.

Preregistered verdict logic (fixed in code before running):
  GREEN  if het_oracle_gain > hom_oracle_gain + seed_SD AND het_oracle_gain > 0.05
  NULL   otherwise (errors too correlated, or the gain is matched by seed-copy arms, or too small)
  DEGENERATE if best mode accuracy > 0.95 or < chance + 0.1 (recalibrate difficulty, no verdict)

Usage: PYTHONPATH=. python scripts/pr1_mode_error_disjointness.py --seeds 5 --n-train 2000 \
    --n-test 600 --out runs/pre_studio/pr1_mode_error_disjointness.json

No em dashes or en dashes (BLACKHOLE.md). No encoder, no weights, latents are synthetic.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from devsys.shell.predictor import mlp
from devsys.shell.refine import IterativeRefiner
from devsys.substrate import make_task_stream

MODES = ["reactive_linear", "mlp", "knn", "prototype", "recurrent_refiner"]
DETERMINISTIC_MODES = {"knn", "prototype"}
SUBPOPS = ["linear_blobs", "antipodal", "fine_grained_local"]

# preregistered verdict thresholds, fixed BEFORE any run
GREEN_MIN_GAIN = 0.05
DEGENERATE_HIGH = 0.95
DEGENERATE_LOW_MARGIN = 0.10
TARGET_BAND = (0.55, 0.85)
TARGET_CENTER = 0.70
CALIBRATION_GRID = [0.01, 0.02, 0.03, 0.045, 0.07, 0.12, 0.2, 0.35]
CALIBRATION_SEED = 1234  # disjoint from the experiment seeds 0..seeds-1
CAL_REFERENCE_MODES = ["reactive_linear", "knn", "prototype"]  # cheap references only
# deterministic-mode seed-copies: mild subsample-without-replacement fractions, calibrated per seed
SUBSAMPLE_FRACTIONS = [0.99, 0.97, 0.95, 0.90, 0.80]
MIN_COPY_TOL = 0.01  # floor on the copy-mean-accuracy validity tolerance


def make_dataset(
    seed: int, n_train: int, n_test: int, n_classes: int, dim: int, separation: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The preregistered MIXTURE task: three equal sub-populations over the same latent space and
    label space, each with a different Bayes-optimal hypothesis class (see module docstring), all
    driven by the single separation knob. Noise scale 0.5 mirrors the established _clusters
    convention (centers ~ N(0,1) scaled by separation). Returns train/test plus the test-set
    sub-population ids (for the per-subpop diagnostic only, never shown to any mode)."""
    n_total = n_train + n_test
    n_a = n_total // 3
    n_b = n_total // 3
    n_c = n_total - n_a - n_b
    # subpop 1, linear_blobs: the established generator, Bayes rule = nearest centroid = linear
    task = make_task_stream(
        n_tasks=1,
        dim=dim,
        classes_per_task=n_classes,
        samples_per_task=n_a,
        separation=separation,
        incremental="class",
        seed=seed,
    )[0]
    xa, ya = task.x, task.y
    g = torch.Generator().manual_seed(seed + 101)
    # subpop 2, antipodal: each class is +mu_c and -mu_c, class mean ~ 0, not linearly separable
    cb = torch.randn(n_classes, dim, generator=g)
    yb = torch.randint(0, n_classes, (n_b,), generator=g)
    signs = (torch.randint(0, 2, (n_b, 1), generator=g).float() * 2.0) - 1.0
    xb = cb[yb] * separation * signs + torch.randn(n_b, dim, generator=g) * 0.5
    # subpop 3, fine_grained_local: 4 scattered sub-clusters per class at 0.6x separation, so local
    # neighborhood structure (knn) wins over any global parametric boundary
    m_sub = 4
    cc = torch.randn(n_classes, m_sub, dim, generator=g)
    yc = torch.randint(0, n_classes, (n_c,), generator=g)
    which = torch.randint(0, m_sub, (n_c,), generator=g)
    xc = cc[yc, which] * (separation * 0.6) + torch.randn(n_c, dim, generator=g) * 0.5
    x = torch.cat([xa, xb, xc])
    y = torch.cat([ya, yb, yc])
    sub = torch.cat([torch.zeros(n_a), torch.ones(n_b), torch.full((n_c,), 2.0)]).long()
    gp = torch.Generator().manual_seed(seed + 7)
    perm = torch.randperm(n_total, generator=gp)
    x, y, sub = x[perm], y[perm], sub[perm]
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:], sub[n_train:]


def _train(model: torch.nn.Module, xtr: torch.Tensor, ytr: torch.Tensor, epochs: int, lr: float, seed: int):
    g = torch.Generator().manual_seed(seed + 31)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n, batch = xtr.shape[0], 256
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            F.cross_entropy(model(xtr[idx]), ytr[idx]).backward()
            opt.step()


class RefinerHead(torch.nn.Module):
    """The recurrent mode: the existing IterativeRefiner (weight-tied residual recurrence over the
    latent) followed by a linear readout, trained jointly. Tiny: hidden=64, steps=3."""

    def __init__(self, dim: int, n_classes: int, hidden: int = 64, steps: int = 3):
        super().__init__()
        self.refiner = IterativeRefiner(dim, hidden=hidden, steps=steps)
        self.head = torch.nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.refiner(x)
        return self.head(z)


def predict_knn(xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor, k: int = 15) -> torch.Tensor:
    idx = torch.cdist(xte, xtr).topk(min(k, xtr.shape[0]), largest=False).indices
    return ytr[idx].mode(dim=1).values


def predict_prototype(
    xtr: torch.Tensor, ytr: torch.Tensor, xte: torch.Tensor, n_classes: int
) -> torch.Tensor:
    cents = torch.stack(
        [xtr[ytr == c].mean(0) if bool((ytr == c).any()) else xtr.mean(0) for c in range(n_classes)]
    )
    return torch.cdist(xte, cents).argmin(-1)


def run_mode(
    mode: str,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xte: torch.Tensor,
    n_classes: int,
    epochs: int,
    seed: int,
    subsample_frac: float | None = None,
) -> torch.Tensor:
    """Fit one mode and return its test-set predictions. subsample_frac keeps that fraction of the
    training set (WITHOUT replacement), used ONLY for seed-copies of the deterministic modes; the
    fraction is calibrated per seed so the copies match the real arm (full-size bootstrap made
    copies systematically weaker and more diverse than the real arm, invalidating the H4 control)."""
    if subsample_frac is not None:
        g = torch.Generator().manual_seed(seed + 97)
        keep = max(1, int(round(subsample_frac * xtr.shape[0])))
        idx = torch.randperm(xtr.shape[0], generator=g)[:keep]
        xtr, ytr = xtr[idx], ytr[idx]
    dim = xtr.shape[1]
    if mode == "knn":
        return predict_knn(xtr, ytr, xte)
    if mode == "prototype":
        return predict_prototype(xtr, ytr, xte, n_classes)
    torch.manual_seed(seed)
    if mode == "reactive_linear":
        model: torch.nn.Module = torch.nn.Linear(dim, n_classes)
        lr = 0.01
    elif mode == "mlp":
        model = mlp(dim, n_classes, hidden=128, depth=1, ln=True)
        lr = 1e-3
    elif mode == "recurrent_refiner":
        model = RefinerHead(dim, n_classes)
        lr = 1e-3
    else:
        raise ValueError(f"unknown mode {mode!r}")
    _train(model, xtr, ytr, epochs, lr, seed)
    with torch.no_grad():
        return model(xte).argmax(-1)


def phi_coefficient(e1: torch.Tensor, e2: torch.Tensor) -> float:
    """Phi coefficient between two 0/1 error indicator vectors (0.0 when a margin is degenerate)."""
    n11 = float(((e1 == 1) & (e2 == 1)).sum())
    n10 = float(((e1 == 1) & (e2 == 0)).sum())
    n01 = float(((e1 == 0) & (e2 == 1)).sum())
    n00 = float(((e1 == 0) & (e2 == 0)).sum())
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return 0.0 if den == 0 else (n11 * n00 - n10 * n01) / den


def calibrate_separation(n_train: int, n_test: int, n_classes: int, dim: int, epochs: int) -> dict:
    """Sweep the separation knob until the best CHEAP reference mode (max over linear, knn,
    prototype; on the mixture no single mode is representative of best-mode accuracy) lands in the
    target band (closest to 0.70). Run once, on a seed disjoint from the experiment seeds, BEFORE
    the modes."""
    table = []
    for sep in CALIBRATION_GRID:
        xtr, ytr, xte, yte, _ = make_dataset(CALIBRATION_SEED, n_train, n_test, n_classes, dim, sep)
        accs = {}
        for m in CAL_REFERENCE_MODES:
            pred = run_mode(m, xtr, ytr, xte, n_classes, epochs, CALIBRATION_SEED)
            accs[m] = float((pred == yte).float().mean())
        ref = max(accs.values())
        row = {"separation": sep, "ref_acc": round(ref, 4)}
        row.update({f"{m}_acc": round(a, 4) for m, a in accs.items()})
        table.append(row)
        print(f"calibration sep={sep} ref_acc={ref:.3f} ({accs})", flush=True)
    in_band = [r for r in table if TARGET_BAND[0] <= r["ref_acc"] <= TARGET_BAND[1]]
    pool = in_band or table
    best = min(pool, key=lambda r: abs(r["ref_acc"] - TARGET_CENTER))
    return {"chosen_separation": best["separation"], "in_band": bool(in_band), "table": table}


def hom_copies(
    mode: str,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
    n_classes: int,
    epochs: int,
    seed_base: int,
    frac: float | None = None,
) -> tuple[list[float], float, float]:
    """K seed-copies of one mode (K = arm count of the heterogeneous oracle). Returns the copy
    accuracies, the homogeneous oracle accuracy, and the homogeneous oracle GAIN over the MEAN copy
    accuracy. The mean baseline is bias-free in expectation: the copies are exchangeable, so
    max(copy_accs) carries the full best-of-K winner's curse on the test set (~+0.009 at n_test=600,
    K=5) that the well-separated heterogeneous arms' max does not, which would deflate hom_gain and
    bias the verdict toward GREEN."""
    preds = [
        run_mode(mode, xtr, ytr, xte, n_classes, epochs, seed=seed_base + j, subsample_frac=frac)
        for j in range(len(MODES))
    ]
    correct = torch.stack([(p == yte).long() for p in preds])
    accs = [float(c.float().mean()) for c in correct]
    oracle = float(correct.amax(0).float().mean())
    return accs, oracle, oracle - sum(accs) / len(accs)


def verdict_of(het_gain: float, hom_gain: float, seed_sd: float, best_acc: float, chance: float) -> str:
    """The preregistered verdict logic, fixed before running."""
    if best_acc > DEGENERATE_HIGH or best_acc < chance + DEGENERATE_LOW_MARGIN:
        return "DEGENERATE: best mode accuracy outside the calibrated band, recalibrate difficulty"
    if het_gain > hom_gain + seed_sd and het_gain > GREEN_MIN_GAIN:
        return (
            "GREEN: heterogeneous oracle gain beats the homogeneous seed-copy oracle gain by more "
            "than the seed spread and exceeds 0.05, modes are complementary, a router is worth building"
        )
    return (
        "NULL: heterogeneous oracle gain is matched by seed-copy arms (or too small), errors are too "
        "correlated for any router to help even on a mixture task built to express complementarity, "
        "stop the MT router line"
    )


def run(seeds: int, n_train: int, n_test: int, n_classes: int, dim: int, epochs: int, sep: float | None):
    t0 = time.perf_counter()
    chance = 1.0 / n_classes
    cal: dict = {"skipped": True, "chosen_separation": sep}
    if sep is None:
        cal = calibrate_separation(n_train, n_test, n_classes, dim, epochs)
        sep = float(cal["chosen_separation"])

    per_mode_acc: dict[str, list[float]] = {m: [] for m in MODES}
    per_subpop_acc: dict[str, dict[str, list[float]]] = {m: {sp: [] for sp in SUBPOPS} for m in MODES}
    phi_sum = torch.zeros(len(MODES), len(MODES))
    best_accs, het_oracles, hom_oracles, het_gains, hom_gains, best_modes = [], [], [], [], [], []
    hom_control_all = []
    for s in range(seeds):
        xtr, ytr, xte, yte, sub_te = make_dataset(s, n_train, n_test, n_classes, dim, sep)
        errors, accs = {}, {}
        for m in MODES:
            pred = run_mode(m, xtr, ytr, xte, n_classes, epochs, seed=s)
            correct = (pred == yte).long()
            errors[m] = 1 - correct
            accs[m] = float(correct.float().mean())
            per_mode_acc[m].append(accs[m])
            for k, sp in enumerate(SUBPOPS):
                per_subpop_acc[m][sp].append(float(correct[sub_te == k].float().mean()))
        for i, mi in enumerate(MODES):
            for j, mj in enumerate(MODES):
                phi_sum[i, j] += phi_coefficient(errors[mi], errors[mj])
        best_mode = max(MODES, key=lambda m: accs[m])
        best_acc = accs[best_mode]
        het_oracle = float(torch.stack([1 - errors[m] for m in MODES]).amax(0).float().mean())
        # homogeneous H4 control (same arm count K = len(MODES)), TWO variants, and the verdict
        # gates on the MAX over the VALID gains (conservative against false GREEN):
        #   (a) genuine fresh-seed retrained copies of the best TRAINED mode, always computed and
        #       always valid (this is real seed noise, the thing the control is meant to measure),
        #   (b) only if the overall best mode is deterministic (retraining is a no-op): copies on
        #       mildly subsampled train sets, fraction calibrated so the copy-accuracy spread
        #       matches (a)'s spread, VALID only if the copies' mean accuracy stays within that
        #       spread of the real arm (full-size bootstrap failed this and forced NULL).
        best_trained = max((m for m in MODES if m not in DETERMINISTIC_MODES), key=lambda m: accs[m])
        tr_accs, tr_oracle, tr_gain = hom_copies(
            best_trained, xtr, ytr, xte, yte, n_classes, epochs, seed_base=s * 1000 + 1
        )
        control: dict = {
            "best_mode": best_mode,
            "trained_mode": best_trained,
            "trained_copy_accs": [round(a, 4) for a in tr_accs],
            "trained_oracle": round(tr_oracle, 4),
            "trained_gain": round(tr_gain, 4),
            "subsample": None,
        }
        hom_oracle, hom_gain = tr_oracle, tr_gain
        if best_mode in DETERMINISTIC_MODES:
            sd_trained = float(torch.tensor(tr_accs).std(unbiased=False))
            tol = max(sd_trained, MIN_COPY_TOL)
            cands = []
            for f in SUBSAMPLE_FRACTIONS:
                d_accs, d_oracle, d_gain = hom_copies(
                    best_mode, xtr, ytr, xte, yte, n_classes, epochs, seed_base=s * 1000 + 501, frac=f
                )
                d_mean = sum(d_accs) / len(d_accs)
                d_sd = float(torch.tensor(d_accs).std(unbiased=False))
                cands.append(
                    {
                        "frac": f,
                        "copy_accs": [round(a, 4) for a in d_accs],
                        "mean_acc": round(d_mean, 4),
                        "sd_acc": round(d_sd, 4),
                        "oracle": round(d_oracle, 4),
                        "gain": round(d_gain, 4),
                        "valid": bool(d_mean >= best_acc - tol),
                    }
                )
            valid = [c for c in cands if c["valid"]]
            chosen = min(valid, key=lambda c: abs(c["sd_acc"] - sd_trained)) if valid else None
            control["subsample"] = {
                "tol": round(tol, 4),
                "target_sd": round(sd_trained, 4),
                "chosen_frac": None if chosen is None else chosen["frac"],
                "valid": chosen is not None,
                "candidates": cands,
            }
            if chosen is not None and chosen["gain"] > hom_gain:
                hom_oracle, hom_gain = float(chosen["oracle"]), float(chosen["gain"])
        control["gate_oracle"] = round(hom_oracle, 4)
        control["gate_gain"] = round(hom_gain, 4)
        hom_control_all.append(control)
        best_modes.append(best_mode)
        best_accs.append(best_acc)
        het_oracles.append(het_oracle)
        hom_oracles.append(hom_oracle)
        het_gains.append(het_oracle - best_acc)
        hom_gains.append(hom_gain)
        print(
            f"seed {s}: best={best_mode} acc={best_acc:.3f} het_oracle={het_oracle:.3f} "
            f"hom_oracle={hom_oracle:.3f} ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )

    def stat(v: list[float]) -> dict:
        t = torch.tensor(v)
        return {"mean": round(float(t.mean()), 4), "sd": round(float(t.std(unbiased=False)), 4)}

    seed_sd = stat(best_accs)["sd"]
    het_gain_mean = stat(het_gains)["mean"]
    hom_gain_mean = stat(hom_gains)["mean"]
    best_acc_mean = stat(best_accs)["mean"]
    phi_mean = (phi_sum / seeds).tolist()
    return {
        "experiment": "pr1_mode_error_disjointness",
        "config": {
            "seeds": seeds,
            "n_train": n_train,
            "n_test": n_test,
            "n_classes": n_classes,
            "dim": dim,
            "epochs": epochs,
            "separation": sep,
            "chance": chance,
            "modes": MODES,
            "subpops": SUBPOPS,
            "deterministic_copy_rule": (
                "knn/prototype seed-copies use mild subsample-without-replacement train sets, "
                "fraction calibrated per seed to match the trained-copy accuracy spread, valid only "
                "if copy mean accuracy is within that spread of the real arm; the verdict gates on "
                "the max over valid homogeneous gains (trained-copy control always included)"
            ),
            "hom_baseline_rule": (
                "homogeneous oracle gain uses the MEAN copy accuracy as baseline (copies are "
                "exchangeable, max would carry a best-of-K winner's curse absent from the "
                "heterogeneous baseline)"
            ),
        },
        "calibration": cal,
        "per_mode_accuracy": {m: {**stat(per_mode_acc[m]), "per_seed": per_mode_acc[m]} for m in MODES},
        "per_mode_subpop_accuracy_mean": {
            m: {sp: stat(per_subpop_acc[m][sp])["mean"] for sp in SUBPOPS} for m in MODES
        },
        "phi_error_correlation_mean": phi_mean,
        "best_mode_per_seed": best_modes,
        "best_single_mode_acc": {**stat(best_accs), "per_seed": best_accs},
        "seed_sd_of_best_mode": seed_sd,
        "het_oracle_acc": {**stat(het_oracles), "per_seed": het_oracles},
        "hom_oracle_acc": {**stat(hom_oracles), "per_seed": hom_oracles},
        "hom_control_per_seed": hom_control_all,
        "het_oracle_gain": {**stat(het_gains), "per_seed": het_gains},
        "hom_oracle_gain": {**stat(hom_gains), "per_seed": hom_gains},
        "verdict_rule": (
            "GREEN iff het_gain > hom_gain + seed_SD and het_gain > 0.05; DEGENERATE iff best acc "
            "> 0.95 or < chance + 0.1; else NULL (thresholds fixed in code before running; hom_gain "
            "is the max over valid homogeneous controls with a mean-copy baseline, see config)"
        ),
        "verdict": verdict_of(het_gain_mean, hom_gain_mean, seed_sd, best_acc_mean, chance),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PR1: mode-error disjointness with homogeneous-oracle control")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--separation", type=float, default=None, help="skip calibration, use this value")
    ap.add_argument("--out", default="runs/pre_studio/pr1_mode_error_disjointness.json")
    a = ap.parse_args(argv)
    result = run(a.seeds, a.n_train, a.n_test, a.n_classes, a.dim, a.epochs, a.separation)
    text = json.dumps(result, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
