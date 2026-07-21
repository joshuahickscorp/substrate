"""The E2 cell vocabulary, the staged reduction, and one training path shared by every cell.

The full Cartesian product of architecture, capacity, readout, horizon, reset, history and bed is not run and
was never going to be. What is run is a set of sweeps, each holding the others at one preregistered reference
configuration, plus the interaction cells that H6 and the bed interactions actually need. A cell that no
hypothesis needs is not run, and the list of what is not run is part of the record.

House style: no dashes.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from fastforge import engine as E
from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import witness as W

# the reference configuration every sweep varies one factor away from
REFERENCE = {"family": "gru", "tier": "small", "readout": "linear", "reset": "none", "history_k": 1}

LR = 3e-3
BATCH = 64
STEPS = 1200
HORIZON_FRACTIONS = (1, 2, 5, 10, 20, 45, 90, "full")
HISTORY_K = (1, 2, 5, 10, 20, "full_window", "pooled_summary")


def reset_schedule(kind: str, sp: dict, seed: int) -> tuple[list, dict]:
    """Return the reset indices and the alignment witness for one declared schedule."""
    T, seg = sp["sequence_length"], sp["segment_length"]
    rng = np.random.default_rng(9000 + seed)
    periods = W.coprime_periods(T, seg, 2, lo=max(5, seg // 2), hi=int(seg * 1.4))
    if kind == "none":
        idx = []
    elif kind == "every_observation":
        idx = W.reset_indices_for("every_observation", T, seg)
    elif kind == "misaligned_a":
        idx = W.reset_indices_for("fixed_period", T, seg, period=periods[0])
    elif kind == "misaligned_b":
        idx = W.reset_indices_for("fixed_period", T, seg, period=periods[1])
    elif kind == "random_rate_matched":
        rate = len(W.reset_indices_for("fixed_period", T, seg, period=periods[0])) / T
        idx = W.reset_indices_for("random_rate_matched", T, seg, rate=rate, rng=rng)
    elif kind == "block_shuffled":
        idx = W.reset_indices_for("block_shuffled", T, seg, period=periods[0], rng=rng)
    elif kind == "true_boundary":
        idx = W.reset_indices_for("true_boundary", T, seg)
    elif kind == "wrong_boundary":
        idx = W.reset_indices_for("wrong_boundary", T, seg)
    elif kind.startswith("horizon_"):
        h = kind.split("_", 1)[1]
        idx = [] if h == "full" else W.reset_indices_for("fixed_period", T, seg, period=int(h))
    else:
        raise ValueError(kind)
    wit = W.reset_alignment(idx, sp["boundaries"], T)
    wit["kind"] = kind
    wit["periods_used"] = periods
    return idx, wit


RESET_KINDS = ("none", "every_observation", "misaligned_a", "misaligned_b", "random_rate_matched",
               "block_shuffled", "true_boundary", "wrong_boundary")


def resolve_history_k(k, sp: dict) -> int:
    if k == "full_window":
        return sp["sequence_length"]
    if k == "pooled_summary":
        return sp["sequence_length"]
    return int(k)


def build_cell(sp: dict, *, family: str, tier: str, readout: str, reset: str, history_k, seed: int):
    idx, wit = reset_schedule(reset, sp, seed)
    k = resolve_history_k(history_k, sp)
    fam = "pooled" if history_k == "pooled_summary" and family == "histmlp" else family
    m = A.build(family=fam, ch=sp["channels"], classes=sp["classes"], tier=tier, readout=readout,
                history_k=k, reset=idx)
    hist = A.history_profile(fam, history_k=k, reset=idx, sequence_length=sp["sequence_length"])
    return m, wit, hist


def cell_name(*, family: str, tier: str, readout: str, reset: str, history_k) -> str:
    return f"{family}|{tier}|{readout}|{reset}|h{history_k}"


def run_cell(sp: dict, spec: dict, seed: int, eval_on: str, steps: int = STEPS) -> dict:
    """One trained cell. The only training path in this program, so no arm can differ by accident."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, wit, hist = build_cell(sp, seed=seed, **spec)
    X, Y = sp["main"]
    t0 = time.time()
    rec = E.fit(m, None, X, Y, train_groups=["core", "readout"], steps=steps, lr=LR, rng=rng, batch=BATCH)
    Xe, Ye, ue = (sp["tune"][0], sp["tune"][1], sp["tune_units"]) if eval_on == "tune" else (
        sp["test"][0], sp["test"][1], sp["test_units"])
    acc = float(E.evaluate(m, None, Xe, Ye))
    with torch.no_grad():
        pred = torch.cat([m(Xe[i : i + 256], None)[0].argmax(1) for i in range(0, len(Xe), 256)])
    correct = (pred == Ye).numpy()
    u = np.asarray(ue)
    per_unit = {str(x): float(correct[u == x].mean()) for x in np.unique(u) if (u == x).sum() >= 5}
    counts = A.count(m)
    return {
        "cell": cell_name(**spec),
        "bed": sp["bed"],
        "seed": seed,
        "eval_on": eval_on,
        "spec": spec,
        "accuracy": round(acc, 5),
        "per_unit_accuracy": per_unit,
        "params": counts,
        "steps": steps,
        "updates": rec["updates"],
        "undeclared_changes": rec["undeclared_changes"],
        "checkpoint_sha_after": rec["checkpoint_sha_after"],
        "reset_witness": {k: v for k, v in wit.items() if k not in ("reset_indices", "distance_to_nearest_boundary")},
        "history_profile": hist,
        "prediction_concentration": float(np.bincount(pred.numpy(), minlength=sp["classes"]).max() / len(pred)),
        "wall_seconds": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------- the staged cell set


def sweep_cells() -> dict:
    """Every cell the eight hypotheses need, grouped by the sweep that needs it."""
    ref = REFERENCE
    out: dict[str, list[dict]] = {}

    out["architecture"] = [dict(ref, family=f) for f in A.FAMILIES]

    out["capacity"] = [dict(ref, family=f, tier=t)
                       for f in ("gru", "lstm", "mgu", "pooled", "histmlp", "tcn")
                       for t in A.CAPACITY_TIERS]

    out["readout"] = [dict(ref, family=f, readout=r)
                      for f in ("gru", "pooled", "histmlp")
                      for r in A.READOUTS]

    out["horizon"] = [dict(ref, family=f, reset=f"horizon_{h}")
                      for f in ("gru", "mgu")
                      for h in HORIZON_FRACTIONS]

    out["reset"] = [dict(ref, reset=k) for k in RESET_KINDS]

    out["history"] = [dict(ref, family="histmlp", history_k=k) for k in HISTORY_K] + [
        dict(ref, family="tcn", history_k=k) for k in (5, 20, "full_window")]

    # H6 needs capacity crossed with horizon, not each alone
    out["capacity_by_horizon"] = [dict(ref, tier=t, reset=f"horizon_{h}")
                                  for t in ("micro", "small", "medium", "large")
                                  for h in (5, 45, "full")]

    # H3 versus H2 needs a strong readout on the stateless families at large capacity
    out["capacity_by_readout"] = [dict(ref, family=f, tier="large", readout="mlp_strong")
                                  for f in ("pooled", "histmlp", "tcn", "gru")]

    seen, uniq = set(), {}
    for group, cells in out.items():
        keep = []
        for c in cells:
            n = cell_name(**c)
            if n in seen:
                continue
            seen.add(n)
            keep.append(c)
        uniq[group] = keep
    uniq["_all"] = [c for g, cells in uniq.items() if not g.startswith("_") for c in cells]
    return uniq


def not_run() -> dict:
    """What the staged reduction deliberately leaves out, and why. Silence here would read as coverage."""
    full = (len(A.FAMILIES) * len(A.CAPACITY_TIERS) * len(A.READOUTS) * len(RESET_KINDS)
            * len(HISTORY_K))
    run = len(sweep_cells()["_all"])
    return {
        "full_cartesian_cells": full,
        "cells_run": run,
        "fraction_run": round(run / full, 5),
        "omitted": [
            "capacity crossed with readout below the large tier, because the readout sweep is flat at small",
            "history crossed with capacity for the recurrent families, which have no history parameter",
            "reset crossed with readout, because reset acts on the core and the readout sweep is flat",
            "every cell of a family at a tier its width search cannot reach",
        ],
        "rule": "a cell that no hypothesis needs is not run, and what is not run is listed here",
    }
