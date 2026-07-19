#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.featurize_programmatic import CACHE_ROOT, CLIPSET  # noqa: E402
from scripts.mop_at4_programmatic_ceiling import FACTORS, load_column, parse_seeds  # noqa: E402

from mop.diagnostics.linear_probe import linear_probe  # noqa: E402
from mop.diagnostics.nonlinear_probe import nonlinear_probe  # noqa: E402
from mop.diagnostics.riskcov import seed_ci  # noqa: E402
from mop.seeding import seed_everything  # noqa: E402

CONTRACT = {
    "id": "AT5",
    "metric": "per-cell decodability verdict across probe classes (linear, nonlinear_gain, mlp)",
    "baseline": "the linear probe verdict (the class every other atlas experiment defaults to)",
    "ablation": "the two higher-capacity probe classes on identical splits and seeds",
    "null_hypothesis": (
        "Every cell's verdict is invariant to probe class, so probe choice never changes decodability "
        "and probe-specific is an empty verdict"
    ),
    "tier": "cpu-now",
}

COLUMNS = [
    "programmatic_reference",
    "handcrafted_descriptors",
    "vjepa2_vitl_nuisance_real",
    "vjepa2_vitl_singleframe",
    "randominit_vitl_nuisance",
    "dinov2s_nuisance_real",
    "dinov2s_nuisance_randominit",
    "qwen05b_textified_real",
    "qwen05b_textified_randominit",
    "wav2vec2_sonified_real",
    "wav2vec2_sonified_randominit",
]
PROBE_CLASSES = ["linear", "nonlinear_gain", "mlp"]
DECODABLE_MARGIN = 0.1  # the linear_probe/nonlinear_probe convention, shared so verdicts compare


def nonlinear_gain_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 150,
    lr: float = 0.02,
    test_frac: float = 0.3,
    seed: int = 0,
) -> dict:
    seed_everything(seed)
    n = x.shape[0]
    perm = torch.randperm(n)
    x, y = x[perm].float(), y[perm]
    cut = int(n * (1 - test_frac))
    xtr, xte, ytr, yte = x[:cut], x[cut:], y[:cut], y[cut:]
    nc = int(y.max()) + 1
    d = x.shape[1]
    gain = torch.nn.Parameter(torch.ones(d))
    bias = torch.nn.Parameter(torch.zeros(d))
    head = torch.nn.Linear(d, nc)
    opt = torch.optim.Adam([gain, bias, *head.parameters()], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(head(F.gelu(xtr * gain + bias)), ytr).backward()
        opt.step()
    with torch.no_grad():
        acc = float((head(F.gelu(xte * gain + bias)).argmax(-1) == yte).float().mean())
    return {
        "metric": "accuracy",
        "score": acc,
        "chance": 1.0 / nc,
        "decodable": acc > 1.0 / nc + DECODABLE_MARGIN,
    }


def probe_cell(x: torch.Tensor, y: torch.Tensor, seeds: list[int], epochs: dict[str, int]) -> dict:
    accs: dict[str, list[float]] = {c: [] for c in PROBE_CLASSES}
    for s in seeds:
        accs["linear"].append(
            round(linear_probe(x, y, classification=True, epochs=epochs["linear"], seed=s)["score"], 4)
        )
        accs["nonlinear_gain"].append(
            round(nonlinear_gain_probe(x, y, epochs=epochs["nonlinear_gain"], seed=s)["score"], 4)
        )
        accs["mlp"].append(round(nonlinear_probe(x, y, epochs=epochs["mlp"], seed=s)["score"], 4))
    return accs


def cell_verdict(accs: dict[str, list[float]], chance: float) -> dict:
    decodable = {c: seed_ci(accs[c])["mean"] > chance + DECODABLE_MARGIN for c in PROBE_CLASSES}
    n_dec = sum(decodable.values())
    if n_dec == len(PROBE_CLASSES):
        verdict = "probe-invariant-decodable"
    elif n_dec == 0:
        verdict = "probe-invariant-undecodable"
    else:
        present = sorted(c for c, d in decodable.items() if d)
        verdict = f"probe-specific (decodable under {present} only)"
    return {
        "verdict": verdict,
        "probe_specific": 0 < n_dec < len(PROBE_CLASSES),
        "decodable_under": decodable,
        "ci": {c: seed_ci(accs[c]) for c in PROBE_CLASSES},
        "chance": round(chance, 4),
    }


def run(
    cache_root: Path,
    seeds: list[int],
    columns: list[str] | None = None,
    probe_epochs: dict[str, int] | None = None,
) -> dict:
    t0 = time.perf_counter()
    columns = COLUMNS if columns is None else columns
    epochs = {"linear": 300, "nonlinear_gain": 200, "mlp": 150}
    if probe_epochs:
        epochs.update(probe_epochs)
    cells: list[dict] = []
    skipped: list[str] = []
    for name in columns:
        loaded = load_column(Path(cache_root) / name)
        if loaded is None:
            skipped.append(name)
            print(f"[skip] {name}: store or factors sidecar absent", flush=True)
            continue
        for factor in FACTORS:
            if factor not in loaded["factors"]:
                continue
            y = loaded["factors"][factor]
            chance = 1.0 / float(int(y.max()) + 1)
            accs = probe_cell(loaded["x"], y, seeds, epochs)
            v = cell_verdict(accs, chance)
            cells.append({"column": name, "factor": factor, "per_seed": accs, **v})
            print(f"[{name}] {factor}: {v['verdict']}", flush=True)

    specific = [f"{c['column']}/{c['factor']}" for c in cells if c["probe_specific"]]
    if not cells:
        verdict = "NO CELLS: no labeled store with a factors sidecar is on disk yet, nothing swept"
        null_supported = None  # not evaluable: nothing was tested, never a rejected null
    elif specific:
        verdict = (
            f"PROBE-SPECIFIC cells caught: {specific}. These cells' atlas verdicts depend on the probe "
            "class and must not be trusted under a single probe (the guard fired)"
        )
        null_supported = False
    else:
        verdict = (
            "NULL: every cell's verdict is invariant to probe class (linear, nonlinear_gain, mlp), "
            "probe choice never changed decodability on this clip set"
        )
        null_supported = True
    return {
        "experiment": "mop_at5_probe_class_sweep",
        "contract": CONTRACT,
        "clipset": CLIPSET,
        "seeds": seeds,
        "probe_epochs": epochs,
        "probe_classes": PROBE_CLASSES,
        "decodable_rule": f"mean accuracy over seeds > chance + {DECODABLE_MARGIN} (fixed in code "
        "before running)",
        "cells": cells,
        "skipped_columns": skipped,
        "probe_specific_cells": specific,
        "verdict": verdict,
        "null_supported": null_supported,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AT5: every atlas cell under three probe classes")
    ap.add_argument("--seeds", default="0-4", help="seed spec, e.g. 0-4 or 0,2,5")
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--out", default="runs/mot/at5_probe_class_sweep.json")
    ap.add_argument("--rerun", action="store_true", help="Stage 4 rerun marker, recorded in the output")
    a = ap.parse_args(argv)
    result = run(Path(a.cache_root), parse_seeds(a.seeds))
    result["rerun"] = a.rerun
    text = json.dumps(result, indent=2)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    print(json.dumps({k: result[k] for k in ("verdict", "null_supported", "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
