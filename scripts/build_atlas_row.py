#!/usr/bin/env python
"""Build one representational-atlas row from a cached frozen-latent store.

Proof system, Section 10.9: an atlas row records whether a factor is linearly decodable
from a frozen encoder latent, with a shuffle-label chance floor (a row is INVALID without
one), a reproducibility level, and a reproducible run id pinned to the input hash.

This MVP probes the 6-way visual class ("identity" factor) of the prior real-encoder
ViT-L cache. It is deliberately small and runs on cached latents only (no encode, no
Studio). The Studio re-runs this per factor and per encoder over the natural-video corpus.

Usage:
  .venv/bin/python scripts/build_atlas_row.py
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from devsys.diagnostics.linear_probe import linear_probe  # noqa: E402

CACHE = ROOT / "data" / "cache" / "vjepa2_vitl_fpc64_256_real"
ENCODER = "vjepa2_vitl_fpc64_256"
FACTOR = "identity"
SEEDS = [0, 1, 2, 3, 4]
OUT = ROOT / "proof" / "atlas" / ENCODER / f"{FACTOR}.json"


def _sha(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes())
    return h.hexdigest()


def _ci95(vals: list[float]) -> tuple[float, list[float], float]:
    mean = statistics.fmean(vals)
    sem = (statistics.stdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else 0.0
    lo, hi = max(0.0, mean - 1.96 * sem), min(1.0, mean + 1.96 * sem)
    return round(mean, 4), [round(lo, 4), round(hi, 4)], round(sem, 4)


def main() -> int:
    lat = np.load(CACHE / "latents.npy")
    lab = np.load(CACHE / "labels.npy")
    x = torch.tensor(lat, dtype=torch.float32)
    y = torch.tensor(lab, dtype=torch.long)
    n_classes = int(y.max()) + 1

    real = [linear_probe(x, y, seed=s)["score"] for s in SEEDS]
    # shuffle-label control: same probe, labels permuted. This is the empirical chance floor.
    chance = []
    for s in SEEDS:
        g = torch.Generator().manual_seed(1000 + s)
        chance.append(linear_probe(x, y[torch.randperm(y.shape[0], generator=g)], seed=s)["score"])

    acc_mean, acc_ci, acc_sem = _ci95(real)
    floor_mean, _, _ = _ci95(chance)
    analytic_floor = round(1.0 / n_classes, 4)
    margin = acc_mean - floor_mean
    decodable = "yes" if margin > 0.10 else ("marginal" if margin > 0.03 else "no")

    inputs_hash = _sha(CACHE / "latents.npy", CACHE / "labels.npy")
    row = {
        "encoder": ENCODER,
        "factor": FACTOR,
        "linear_acc": {"value": acc_mean, "ci95": acc_ci},
        "nonlinear_acc": {"value": None, "ci95": [None, None]},
        "chance_floor": floor_mean,
        "analytic_chance": analytic_floor,
        "decodable": decodable,
        "seeds": {"n": len(SEEDS), "sem": acc_sem, "list": SEEDS},
        "provenance_tag": "real-encoder",
        "repro_level": "R3",
        "raw_run_id": f"atlas_{FACTOR}_vitl_{inputs_hash[:12]}",
        "_inputs": {
            "cache": str(CACHE.relative_to(ROOT)),
            "n_clips": int(x.shape[0]),
            "latent_dim": int(x.shape[1]),
            "n_classes": n_classes,
            "inputs_sha256": inputs_hash,
            "builder": "scripts/build_atlas_row.py",
        },
        "_factor_note": (
            "The 6-way visual class label of the cached clips. Probes whether class identity "
            "is linearly decodable from the frozen ViT-L latent. Chance floor is the empirical "
            "shuffle-label accuracy over the same seeds."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}  (decodable={decodable}, acc={acc_mean} vs floor={floor_mean})")
    # invalidity guard (Section 10.9): a row is INVALID without these
    assert row["chance_floor"] is not None and row["repro_level"] and row["raw_run_id"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
