#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.diagnostics import geometry_report, linear_probe, readout_contribution
from mop.diagnostics.held_out_combo import held_out_combination
from mop.diagnostics.substrate_ablation import frozen_random_projection
from mop.substrate import factorized_arrays, factors_meta, open_real_store


def run(store_name: str, data_dir: str, seed: int) -> dict:
    store = open_real_store(store_name, data_dir)
    x = store.latents().flatten(1).float()
    y = store.labels()
    if y is None:
        raise ValueError(f"store {store_name} has no labels; cannot probe")
    y = y.long()
    n, dim = x.shape
    fm = factors_meta(store)

    out: dict = {
        "store": store_name,
        "n_latents": int(n),
        "dim": int(dim),
        "n_classes": int(len(set(y.tolist()))),
        "factorized": fm is not None,
        "seed": seed,
        "note": "real-encoder geometry vs frozen-random projection; leads with nonlinear/compositional tests",
    }

    lp_real = linear_probe(x, y, classification=True, epochs=300, seed=seed)
    lp_fr = linear_probe(frozen_random_projection(x, seed), y, classification=True, epochs=300, seed=seed)
    out["linear_probe"] = {
        "real_acc": round(lp_real["score"], 4),
        "frozen_random_acc": round(lp_fr["score"], 4),
        "chance": round(lp_real["chance"], 4),
        "delta_real_minus_fr": round(lp_real["score"] - lp_fr["score"], 4),
        "projection_invariant": bool(abs(lp_real["score"] - lp_fr["score"]) < 0.05),
    }

    rc = readout_contribution(x, y, hidden=min(128, max(32, dim // 4)), seed=seed)
    out["readout_contribution_p10"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rc.items()}

    if fm is not None:
        xf, ya, yb = factorized_arrays(store)
        hoc_real = held_out_combination(xf, ya, yb, seed=seed)
        hoc_fr = held_out_combination(frozen_random_projection(xf, seed), ya, yb, seed=seed)
        out["held_out_combination_c1"] = {
            "real": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in hoc_real.items()},
            "frozen_random": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in hoc_fr.items()},
            "heldout_delta_real_minus_fr": round(hoc_real["heldout_acc"] - hoc_fr["heldout_acc"], 4),
            "real_composes_above_fr": bool(hoc_real["heldout_acc"] - hoc_fr["heldout_acc"] > 0.05),
        }

    geo = geometry_report(x)
    out["geometry"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in geo.items()}

    real_diverges = bool(
        out["readout_contribution_p10"].get("substrate_carries_nonlinear_structure", False)
        or (fm is not None and out["held_out_combination_c1"]["real_composes_above_fr"])
    )
    out["real_substrate_beats_frozen_random"] = real_diverges
    out["headline"] = (
        "real encoder geometry carries structure a frozen-random projection does not (nonlinear or "
        "compositional)"
        if real_diverges
        else "on this cache, real geometry is not distinguishable from a frozen-random projection on any "
        "tested nonlinear/compositional probe (projection-invariant at this scale/difficulty)"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="real-latent replication of the doctrine-load-bearing probes")
    ap.add_argument("--store", required=True, help="cache name under data/cache")
    ap.add_argument("--data-dir", default="data/cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the results JSON here")
    a = ap.parse_args(argv)

    result = run(a.store, a.data_dir, a.seed)
    text = json.dumps(result, indent=2, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
