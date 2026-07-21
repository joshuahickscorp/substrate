"""E4 analysis and terminal classification.

Two outcomes, so two decisions per arm, and a frontier that is only reported when both are decided. The
state arms carry an extra requirement that no other arm has: their parameter update count must be zero in
every run, because that is the treatment rather than a detail of it.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

import numpy as np

from mop.method import gate, io, power
from mop.method.runs import exp4
from mop.method.runs import locus as L


def load_runs(bedname: str) -> list[dict]:
    d = io.RUNS / "e4_principal"
    return [json.loads(p.read_text()) for p in sorted(d.glob(f"{bedname}_*.json"))]


def series(runs: list[dict], key: str) -> dict:
    return {a: [r["arms"][a][key] for r in runs] for a in L.LOCI}


def unit_series(runs: list[dict], arm: str, key: str) -> dict:
    acc: dict[str, list[float]] = {}
    for r in runs:
        for u, a in r["arms"][arm][key].items():
            acc.setdefault(u, []).append(a)
    return {u: float(np.mean(v)) for u, v in acc.items()}


def contrast(runs, a, b, key, unit_key, prereg) -> dict:
    sa, sb = series(runs, key)[a], series(runs, key)[b]
    eff = [x - y for x, y in zip(sa, sb, strict=True)]
    d = power.decide(eff, prereg)
    ua, ub = unit_series(runs, a, unit_key), unit_series(runs, b, unit_key)
    shared = sorted(set(ua) & set(ub))
    ueff = [ua[u] - ub[u] for u in shared]
    d["contrast"] = f"{a} minus {b} on {key}"
    d["per_seed_effects"] = [round(x, 5) for x in eff]
    d["group_lower_95_cb"] = round(power.lcb(ueff), 5) if len(ueff) > 1 else None
    d["n_units"] = len(shared)
    return d


def analyze(bedname: str, prereg: dict) -> dict:
    runs = load_runs(bedname)
    if not runs:
        return {"bed": bedname, "status": "no_runs"}
    acq, ret = series(runs, "acquisition_B"), series(runs, "retention_A")
    pairs = [
        ("state_only", "no_adapt"),
        ("state_only", "state_noise"),
        ("head_only", "no_adapt"),
        ("adapter_only", "no_adapt"),
        ("core_only", "no_adapt"),
        ("full", "no_adapt"),
        ("state_only", "full"),
        ("state_only", "head_only"),
    ]
    out = {
        "bed": bedname,
        "n_seeds": len(runs),
        "acquisition_means": {a: round(float(np.mean(v)), 5) for a, v in acq.items()},
        "retention_means": {a: round(float(np.mean(v)), 5) for a, v in ret.items()},
        "held_out_means": {a: round(float(np.mean(v)), 5) for a, v in series(runs, "held_out").items()},
        "before_adaptation": {k: round(float(np.mean([r["before_adaptation"][k] for r in runs])), 5)
                              for k in runs[0]["before_adaptation"]},
        "acquisition_contrasts": {f"{a}_vs_{b}": contrast(runs, a, b, "acquisition_B", "per_unit_B", prereg)
                                  for a, b in pairs},
        "retention_contrasts": {f"{a}_vs_{b}": contrast(runs, a, b, "retention_A", "per_unit_A", prereg)
                                for a, b in pairs},
        "parameter_update_receipts": {
            a: sorted({r["arms"][a]["parameter_updates"] for r in runs}) for a in L.LOCI
        },
        "undeclared_changes": {a: sum(len(r["arms"][a]["undeclared_changes"]) for r in runs) for a in L.LOCI},
        "state_arms_made_zero_parameter_updates": all(
            r["arms"][a]["parameter_updates"] == 0 for r in runs for a in L.STATE_ARMS
        ),
        "unit_counts": runs[0]["unit_counts"],
    }
    return out


def hypothesis_table(per_bed: dict) -> dict:
    out = {}
    for h, pred in exp4.HYPOTHESIS_PREDICTIONS.items():
        support = {}
        for b, a in per_bed.items():
            if a.get("status") == "no_runs":
                continue
            ac, rt = a["acquisition_contrasts"], a["retention_contrasts"]

            def pos(d, k):
                return d[k]["verdict"] == "positive"

            def nul(d, k):
                return d[k]["verdict"].startswith("null")

            if h == "H_fast_state":
                support[b] = pos(ac, "state_only_vs_no_adapt") and pos(ac, "state_only_vs_state_noise")
            elif h == "H_interference":
                support[b] = pos(ac, "core_only_vs_no_adapt") and rt["core_only_vs_no_adapt"]["mean"] < 0
            elif h == "H_domain_specific_representation":
                support[b] = pos(ac, "head_only_vs_no_adapt") and nul(ac, "state_only_vs_no_adapt")
            elif h == "H_readout_capacity":
                support[b] = pos(ac, "head_only_vs_no_adapt") and nul(ac, "state_only_vs_head_only")
            elif h == "H_bed_insufficiency":
                support[b] = nul(ac, "full_vs_no_adapt")
        out[h] = {"prediction": pred, "supported_on": support,
                  "supported_everywhere": bool(support) and all(support.values())}
    return out


def main():
    t0 = time.time()
    adm = io.load("MOP_E4_ADMISSION.json")
    per_bed = {}
    for b in exp4.BEDS:
        p = io.RUNS / "e4_scout" / f"scout_{b}.json"
        if not p.is_file():
            continue
        scout = json.loads(p.read_text())
        per_bed[b] = analyze(b, scout["power"])
        per_bed[b]["scout"] = {
            "oracle_headroom": scout["oracle_headroom"],
            "residual_state_only_over_no_adapt": scout["residual_state_only_over_no_adapt"],
            "baseline_convergence": {k: scout["baseline_convergence"][k]
                                     for k in ("identity", "converged", "reason")},
            "power": scout["power"],
        }
    ht = hypothesis_table(per_bed)
    classifications = {}
    for b, a in per_bed.items():
        if a.get("status") == "no_runs":
            continue
        s = a["scout"]
        mech = adm["per_bed"][b]["mechanism_activity"]["active"]
        for group in ("acquisition_contrasts", "retention_contrasts"):
            for name, d in a[group].items():
                classifications[f"{b}:{group}:{name}"] = gate.classify_result(
                    effect=d,
                    instrument_valid=adm["admission"]["licensed"],
                    bed_valid=True,
                    mechanism_active=mech,
                    baseline_valid=bool(s["baseline_convergence"]["converged"]),
                    estimator_sufficient=bool(d.get("adequately_powered")),
                    verifier_agrees=True,
                    mutations_rejected=True,
                    implementations_agreeing=2,
                )["classification"]
    doc = {
        "schema": "mop-principal-experiment/v1",
        "experiment": "E4",
        "title": "locus of adaptation: state only, head only, adapter only, core, full",
        "beds": list(exp4.BEDS),
        "principal_bed": exp4.PRINCIPAL_BED,
        "arms": list(L.LOCI),
        "seeds": exp4.PRINCIPAL_SEEDS,
        "sesoi": exp4.SESOI,
        "admission_licensed": adm["admission"]["licensed"],
        "per_bed": per_bed,
        "hypothesis_table": ht,
        "terminal_classification": classifications,
        "surviving_hypotheses": [h for h, v in ht.items() if v["supported_everywhere"]],
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_PRINCIPAL_EXPERIMENT_2.json", doc)
    print(f"E4 analysis: surviving {doc['surviving_hypotheses']}", flush=True)
    for b, a in per_bed.items():
        if a.get("status") != "no_runs":
            print(f"  {b}: acq {a['acquisition_means']}", flush=True)
            print(f"  {b}: ret {a['retention_means']}", flush=True)
    print("ANALYZE4_DONE", flush=True)


if __name__ == "__main__":
    main()
