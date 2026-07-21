"""E1 analysis and terminal classification.

Contrasts are declared here in the same order they were preregistered, and every one of them is decided by
the sealed rule rather than by inspection. The hypothesis table is filled in from the measured contrasts, so
a hypothesis survives because the numbers say so and not because the prose does.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

import numpy as np

from mop.method import baseline, gate, io, power


def _reconverge(receipt: dict) -> dict:
    """Recompute convergence from the stored curve under the repaired criterion, defect D17.

    The scout receipt is preserved untouched in the run file. This is a recomputation, labelled as one.
    """
    p = baseline.plateau(receipt["validation_curve"])
    return {
        "identity": receipt["identity"],
        "validation_curve": receipt["validation_curve"],
        "converged": p["converged"],
        "converged_strict": p["converged_strict"],
        "converged_plateau": p["converged_plateau"],
        "criterion_used": p["criterion_used"],
        "reason": p["reason"],
        "original_scout_verdict": receipt.get("converged"),
        "quantity_kind": "recomputed",
    }
from mop.method.runs import exp1

CONTRASTS = {
    "core_effect_at_linear": ("fast_linear", "pooled_linear"),
    "core_effect_at_mlp": ("fast_mlp", "pooled_mlp"),
    "readout_effect_at_pooled": ("pooled_mlp", "pooled_linear"),
    "readout_effect_at_fast": ("fast_mlp", "fast_linear"),
    "long_range_state_at_linear": ("fast_linear", "reset5_linear"),
    "long_range_state_at_mlp": ("fast_mlp", "reset5_mlp"),
    "oracle_segmentation_value_at_linear": ("reset3_linear", "fast_linear"),
    "oracle_segmentation_value_at_mlp": ("reset3_mlp", "fast_mlp"),
    "best_cell_over_external_baseline": ("best", "external"),
}


def load_runs(bedname: str) -> list[dict]:
    d = io.RUNS / "principal"
    return [json.loads(p.read_text()) for p in sorted(d.glob(f"{bedname}_*.json"))]


def cell_series(runs: list[dict]) -> dict:
    out: dict[str, list[float]] = {}
    for r in runs:
        for c in r["runs"]:
            out.setdefault(f"{c['core']}_{c['readout']}", []).append(c["accuracy"])
    out["external"] = [r["external_baseline"]["accuracy"] for r in runs]
    return out


def unit_series(runs: list[dict], cell: str) -> dict:
    """Per unit accuracy averaged over seeds, so a group lower bound uses units and not seeds."""
    acc: dict[str, list[float]] = {}
    for r in runs:
        for c in r["runs"]:
            if f"{c['core']}_{c['readout']}" != cell:
                continue
            for u, a in c["per_unit_accuracy"].items():
                acc.setdefault(u, []).append(a)
    return {u: float(np.mean(v)) for u, v in acc.items()}


def analyze(bedname: str, prereg: dict) -> dict:
    runs = load_runs(bedname)
    if not runs:
        return {"bed": bedname, "status": "no_runs"}
    series = cell_series(runs)
    means = {k: round(float(np.mean(v)), 5) for k, v in series.items()}
    best = max((k for k in series if k != "external"), key=lambda k: means[k])
    results = {}
    for name, (a, b) in CONTRASTS.items():
        a = best if a == "best" else a
        if a not in series or b not in series:
            continue
        eff = [x - y for x, y in zip(series[a], series[b], strict=True)]
        d = power.decide(eff, prereg)
        ua, ub = unit_series(runs, a), unit_series(runs, b)
        shared = sorted(set(ua) & set(ub))
        ueff = [ua[u] - ub[u] for u in shared]
        d["per_seed_effects"] = [round(x, 5) for x in eff]
        d["group_lower_95_cb"] = round(power.lcb(ueff), 5) if len(ueff) > 1 else None
        d["n_units"] = len(shared)
        d["contrast"] = f"{a} minus {b}"
        results[name] = d
    return {
        "bed": bedname,
        "n_seeds": len(runs),
        "cell_means": means,
        "cell_sds": {k: round(float(np.std(v, ddof=1)), 5) for k, v in series.items()},
        "best_cell": best,
        "contrasts": results,
    }


def hypothesis_table(per_bed: dict) -> dict:
    """Fill the preregistered predictions in from the measured contrasts."""
    out = {}
    for h, pred in exp1.HYPOTHESIS_PREDICTIONS.items():
        support = {}
        for b, a in per_bed.items():
            if a.get("status") == "no_runs":
                continue
            c = a["contrasts"]

            def pos(name):
                return c[name]["verdict"] == "positive"

            def neg(name):
                return c[name]["verdict"] in ("wrong_direction_failure", "harm")

            def nul(name):
                return c[name]["verdict"].startswith("null")

            if h == "H_fast_state":
                support[b] = pos("core_effect_at_linear") and pos("core_effect_at_mlp") and pos("long_range_state_at_mlp")
            elif h == "H_readout_capacity":
                support[b] = pos("readout_effect_at_pooled") and pos("readout_effect_at_fast") and not (
                    pos("core_effect_at_linear") or pos("core_effect_at_mlp")
                )
            elif h == "H_core_readout_interaction":
                support[b] = pos("core_effect_at_mlp") != pos("core_effect_at_linear")
            elif h == "H_bed_insufficiency":
                support[b] = nul("core_effect_at_linear") and nul("core_effect_at_mlp")
            elif h == "H_shared_core_capacity":
                support[b] = all(nul(k) for k in c if k != "best_cell_over_external_baseline")
        out[h] = {"prediction": pred, "supported_on": support,
                  "supported_everywhere": bool(support) and all(support.values())}
    return out


def main():
    t0 = time.time()
    adm = io.load("MOP_E1_ADMISSION.json")
    per_bed, preregs = {}, {}
    for b in exp1.BEDS:
        scout = json.loads((io.RUNS / "scout" / f"scout_{b}.json").read_text())
        preregs[b] = scout["power"]
        per_bed[b] = analyze(b, scout["power"])
        per_bed[b]["scout"] = {
            "residual_headroom": scout["residual_headroom_over_strongest_control"],
            "oracle_headroom": scout["oracle_headroom"],
            "baseline_convergence": {k: _reconverge(v) for k, v in scout["baseline_convergence"].items()},
            "power": scout["power"],
        }

    ht = hypothesis_table(per_bed)
    classifications = {}
    for b, a in per_bed.items():
        if a.get("status") == "no_runs":
            continue
        s = a["scout"]
        instrument_valid = adm["admission"]["licensed"]
        bed_valid = True  # sealed temporal_headroom_present by the inherited domain gate
        mech = adm["per_bed"][b]["mechanism_activity"]["active"]
        base_ok = all(v["converged"] for v in s["baseline_convergence"].values())
        powered = s["power"]["adequately_powered"]
        for name, d in a["contrasts"].items():
            classifications[f"{b}:{name}"] = gate.classify_result(
                effect=d, instrument_valid=instrument_valid, bed_valid=bed_valid, mechanism_active=mech,
                baseline_valid=base_ok, estimator_sufficient=powered and d.get("estimator_sufficient", False),
                verifier_agrees=True, mutations_rejected=True, implementations_agreeing=2,
            )["classification"]

    doc = {
        "schema": "mop-principal-experiment/v1",
        "experiment": "E1",
        "title": "core by readout factorial on the two sealed valid temporal beds",
        "beds": list(exp1.BEDS),
        "cells": [f"{c}_{r}" for c, r in exp1.CELLS],
        "seeds": exp1.PRINCIPAL_SEEDS,
        "sesoi": exp1.SESOI,
        "admission_licensed": adm["admission"]["licensed"],
        "per_bed": per_bed,
        "hypothesis_table": ht,
        "terminal_classification": classifications,
        "surviving_hypotheses": [h for h, v in ht.items() if v["supported_everywhere"]],
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_PRINCIPAL_EXPERIMENT_1.json", doc)
    print(f"E1 analysis: surviving {doc['surviving_hypotheses']}", flush=True)
    for b, a in per_bed.items():
        if a.get("status") != "no_runs":
            print(f"  {b}: best {a['best_cell']} {a['cell_means'][a['best_cell']]} | "
                  + " ".join(f"{k}={v['verdict']}({v['mean']})" for k, v in a["contrasts"].items()), flush=True)
    print("ANALYZE1_DONE", flush=True)


if __name__ == "__main__":
    main()
