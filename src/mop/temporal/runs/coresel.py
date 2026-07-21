"""Select the smallest configuration that preserves the evidence, or refuse to select one.

Selection order is fixed before the numbers arrive: best valid region, then everything statistically
equivalent to it inside the sealed margin, then the lowest parameter count, then the lowest compute, then the
shortest sufficient horizon, then the simplest readout, then the simplest architecture.

Nothing here may select a configuration whose evidence does not survive the primary success rules.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from fastforge import engine as E
from mop.method import power
from mop.temporal import analysis as AN
from mop.temporal import arch as A
from mop.temporal import io
from mop.temporal.runs import e2
from mop.temporal.runs.analyze import load_runs

SIMPLICITY = {"pooled": 0, "histmlp": 1, "tcn": 2, "mgu": 3, "gru": 4, "lstm": 5, "ff_gru": 6}
READOUT_SIMPLICITY = {"linear": 0, "mlp1": 1, "mlp_strong": 2}


def factorial_evidence_gates(authority: dict) -> dict[str, bool]:
    principal = authority.get("principal_beds") or {}
    expected = set(e2.B.PRINCIPAL)
    return {
        "all_principal_beds_valid": authority.get("all_principal_beds_valid") is True,
        "exact_principal_bed_checks": (
            isinstance(principal, dict)
            and set(principal) == expected
            and all(isinstance(principal[bed], dict) for bed in expected)
            and all((principal[bed].get("checks") or {}).get("all_pass") is True for bed in expected)
        ),
    }


def third_bed_licensed(preflight: dict, replication: dict, result: dict) -> bool:
    """A secondary domain is licensed only by admission and consistent replication artifacts."""
    selected = preflight.get("selected")
    return (
        isinstance(selected, list)
        and "harth_stream" in selected
        and replication.get("third_bed_classification") == "replicated"
        and result.get("bed") == "harth_stream"
        and result.get("classification") == "replicated"
    )


def package_checkpoints(selected: dict, beds: list[str]) -> dict:
    """Retrain the sealed minimal cell for packaging; these runs are not new principal evidence."""
    out = {}
    spec = dict(selected["spec"])
    spec["history_k"] = int(spec["history_k"]) if str(spec["history_k"]).isdigit() else spec["history_k"]
    root = io.PROOF / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    for bed in beds:
        seed = 0
        sp = e2.B.splits(bed, seed)
        torch.manual_seed(seed)
        model, reset_witness, history = e2.Fx.build_cell(sp, seed=seed, **spec)
        receipt = E.fit(model, None, sp["main"][0], sp["main"][1], train_groups=["core", "readout"],
                        steps=e2.Fx.STEPS, lr=e2.Fx.LR, rng=np.random.default_rng(seed),
                        batch=e2.Fx.BATCH)
        score = E.evaluate(model, None, sp["test"][0], sp["test"][1])
        path = root / f"owned_temporal_core_v1_{bed}.pt"
        torch.save({"schema": "mop-owned-temporal-core-checkpoint/v1", "bed": bed, "seed": seed,
                    "spec": spec, "state_dict": model.state_dict(), "params": A.count(model),
                    "test_accuracy": score, "training_receipt": receipt,
                    "reset_witness": reset_witness, "history_profile": history,
                    "source_commit": io.commit()}, path)
        out[bed] = {"path": path.relative_to(io.ROOT).as_posix(), "sha256": io.sha_file(path),
                    "test_accuracy": round(float(score), 5), "params": A.count(model),
                    "checkpoint_sha": E.checkpoint_sha(model),
                    "not_additional_principal_evidence": True}
    return out


def parse(cell: str) -> dict:
    f, t, r, reset, h = cell.split("|")
    return {"family": f, "tier": t, "readout": r, "reset": reset, "history_k": h[1:]}


def horizon_of(spec: dict, sp_len: int) -> int:
    reset = spec["reset"]
    if reset in ("none", "horizon_full"):
        return sp_len
    if reset.startswith("horizon_"):
        return int(reset.split("_")[1])
    return sp_len


def equivalence_to_best(candidate: str, best: str, beds: list[str], principal: dict) -> dict:
    """Paired seed and independent-unit equivalence on every production bed."""
    per, raw_available = {}, True
    for bed in beds:
        runs = load_runs(bed)
        series, units = e2._series(runs) if runs else ({}, {})
        if candidate not in series or best not in series:
            raw_available = False
            continue
        d = AN.contrast(series, candidate, best, e2.PREREG, units)
        group_equivalent = (d.get("group_lower_95_cb") is not None
                            and d["group_lower_95_cb"] >= -io.EQUIVALENCE_MARGIN
                            and d.get("group_upper_95_cb") is not None
                            and d["group_upper_95_cb"] <= io.EQUIVALENCE_MARGIN)
        per[bed] = {"seed_equivalent": AN.equivalent(d), "group_equivalent": group_equivalent,
                    "mean": d.get("mean"), "lower_95_cb": d.get("lower_95_cb"),
                    "group_lower_95_cb": d.get("group_lower_95_cb"),
                    "group_upper_95_cb": d.get("group_upper_95_cb")}
    if raw_available and len(per) == len(beds):
        return {"passes": all(v["seed_equivalent"] and v["group_equivalent"] for v in per.values()),
                "per_bed": per, "source": "paired raw seed and independent-unit effects"}
    # Synthetic unit tests and callers without run custody retain a transparent summary-only fallback.
    gaps = {b: principal["per_bed"][b]["cell_means"][best]
            - principal["per_bed"][b]["cell_means"][candidate] for b in beds}
    return {"passes": all(abs(v) <= io.EQUIVALENCE_MARGIN for v in gaps.values()),
            "per_bed_mean_gap": gaps, "source": "summary fallback; forbidden in production evidence"}


def select(principal: dict) -> dict:
    """principal is the sealed E2 result. Returns the selection or a refusal with a reason."""
    beds = [b for b in principal["principal_beds"] if principal["per_bed"][b].get("status") != "no_runs"]
    if len(beds) < 2:
        return {"selected": None, "reason": "fewer than two valid principal beds carry results"}

    # a configuration is eligible only where the recurrence result survived on both beds
    fold = principal["hypothesis_fold"]["hypotheses"]
    if fold["H1_recurrence"]["state"] not in ("supported", "mixed"):
        return {"selected": None,
                "reason": f"H1 recurrence is {fold['H1_recurrence']['state']}, so no temporal core is licensed",
                "hypothesis_state": {k: v["state"] for k, v in fold.items()}}

    shared = set(principal["per_bed"][beds[0]]["cell_means"])
    for b in beds[1:]:
        shared &= set(principal["per_bed"][b]["cell_means"])
    shared = {c for c in shared if (parse(c)["reset"] in ("none", "horizon_full")
                                    or parse(c)["reset"].startswith("horizon_"))}
    if all("configs" in principal["per_bed"][b].get("convergence", {}) for b in beds):
        shared = {c for c in shared if all(
            principal["per_bed"][b]["convergence"]["configs"].get(c, {}).get("classification")
            == "converged" for b in beds)}
    if not shared:
        return {"selected": None,
                "reason": "no deployment valid recurrent cell has converged on both principal beds"}
    # the region is judged by the worst bed, so a configuration that only works on one bed cannot win
    worst = {c: min(principal["per_bed"][b]["cell_means"][c] for b in beds) for c in shared}
    best_cell = max(worst, key=worst.get)
    best_score = worst[best_cell]

    equivalent, equivalence = [], {}
    for c, m in worst.items():
        equivalence[c] = equivalence_to_best(c, best_cell, beds, principal)
        if equivalence[c]["passes"]:
            equivalent.append(c)

    rows = []
    for c in equivalent:
        spec = parse(c)
        params = principal["per_bed"][beds[0]]["cell_params"].get(c, {})
        wall = [principal["per_bed"][b].get("metric_panel", {}).get(
            "latency_wall_seconds_mean", {}).get(c, float("inf")) for b in beds]
        sp_len = 192
        rows.append({
            "cell": c,
            "spec": spec,
            "worst_bed_mean": round(worst[c], 5),
            "gap_to_best": round(best_score - worst[c], 5),
            "params_total": params.get("total", 10**9),
            "params_core": params.get("core", 10**9),
            "measured_compute_seconds": round(max(wall), 5) if all(np.isfinite(wall)) else float("inf"),
            "horizon": horizon_of(spec, sp_len),
            "readout_rank": READOUT_SIMPLICITY.get(spec["readout"], 9),
            "architecture_rank": SIMPLICITY.get(spec["family"], 9),
            "is_recurrent": spec["family"] in A.RECURRENT,
        })
    # a selected core must actually own state, otherwise it is not a temporal core
    recurrent_rows = [r for r in rows if r["is_recurrent"] and r["spec"]["family"] in ("gru", "mgu")]
    if not recurrent_rows:
        return {"selected": None,
                "reason": ("no independently replicated GRU or MGU configuration is inside the equivalence "
                           "region of the best cell"),
                "equivalent_region": sorted(equivalent)}
    ordered = sorted(recurrent_rows, key=lambda r: (r["params_total"], r["measured_compute_seconds"], r["horizon"],
                                                    r["readout_rank"], r["architecture_rank"]))
    return {
        "selected": ordered[0],
        "best_cell": best_cell,
        "best_worst_bed_mean": round(best_score, 5),
        "equivalence_margin": io.EQUIVALENCE_MARGIN,
        "equivalent_region": sorted(equivalent),
        "equivalence_evidence": {c: equivalence[c] for c in sorted(equivalence)},
        "ordered_candidates": ordered[:12],
        "selection_order": ["best valid region", "statistically equivalent inside the sealed margin",
                            "lowest parameter count", "lowest compute", "shortest sufficient horizon",
                            "simplest readout", "simplest architecture"],
    }


def main():
    t0 = time.time()
    principal = io.load("MOP_E2_PRINCIPAL_RESULT.json")
    factorial = (io.load("MOP_E2_FACTORIAL_AUTHORITY.json")
                 if io.exists("MOP_E2_FACTORIAL_AUTHORITY.json") else {})
    evidence = {
        "independent_replication": (io.load("MOP_E2_INDEPENDENT_REPLICATION.json").get("all_pass")
                                    if io.exists("MOP_E2_INDEPENDENT_REPLICATION.json") else False),
        "independent_verification": (io.load("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json").get("all_pass")
                                     if io.exists("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json") else False),
        "mutations": (io.load("MOP_TEMPORAL_CORE_MUTATION_REPORT.json").get("all_rejected")
                      if io.exists("MOP_TEMPORAL_CORE_MUTATION_REPORT.json") else False),
        **factorial_evidence_gates(factorial),
    }
    sel = select(principal) if all(evidence.values()) else {
        "selected": None,
        "reason": "load bearing evidence gates are not all green",
        "evidence_gates": evidence,
    }
    beds = [b for b in principal["principal_beds"] if principal["per_bed"][b].get("status") != "no_runs"]
    replication = (io.load("MOP_E2_INDEPENDENT_REPLICATION.json")
                   if io.exists("MOP_E2_INDEPENDENT_REPLICATION.json") else {})
    preflight = (io.load("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json")
                 if io.exists("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") else {})
    third_result = (io.load("MOP_THIRD_TEMPORAL_BED_RESULT.json")
                    if io.exists("MOP_THIRD_TEMPORAL_BED_RESULT.json") else {})
    if third_bed_licensed(preflight, replication, third_result):
        beds.append("harth_stream")
    doc = {
        "schema": "mop-owned-temporal-core-v1/v1",
        "selected": bool(sel.get("selected")),
        "selection": sel,
        "evidence_gates": evidence,
        "evidence_ceiling": (
            "this is a substrate component with evidence on the beds named here. It does not establish a "
            "complete substrate architecture, continual plasticity, cross domain transfer, functional "
            "reorganization or activation"
        ),
        "activation": False,
        "wall_seconds": round(time.time() - t0, 1),
    }
    if sel.get("selected"):
        s = sel["selected"]
        doc["core"] = {
            "architecture": s["spec"]["family"],
            "capacity_tier": s["spec"]["tier"],
            "owned_parameters": s["params_total"],
            "core_parameters": s["params_core"],
            "owned_state": "one recurrent hidden state vector, carried across timesteps within the horizon",
            "readout": s["spec"]["readout"],
            "horizon": s["horizon"],
            "reset_semantics": s["spec"]["reset"],
            "training_rule": f"Adam at {e2.Fx.LR}, batch {e2.Fx.BATCH}, {e2.Fx.STEPS} updates, "
                             f"all declared groups trainable",
            "adaptation_rule": "none is selected. The E4 state only rule is explicitly not reused",
            "memory_cost_bytes": s["params_total"] * 4,
            "compute_cost": {"principal_wall_seconds_worst_bed": s["measured_compute_seconds"],
                             "resource_benchmark": "MOP_TEMPORAL_CORE_RESOURCE_REPORT.json"},
            "valid_domains": beds,
            "strongest_baseline_comparison": {
                b: (replication.get("per_bed", {}).get(b, {}).get("effects", {}).get(
                    "explicit_mgu_vs_full_history" if s["spec"]["family"] == "mgu"
                    else "torch_gru_vs_full_history")) for b in beds},
        }
        doc["core"]["checkpoints"] = package_checkpoints(s, beds)
        rows = "\n".join(
            f"| {r['cell']} | {r['worst_bed_mean']} | {r['gap_to_best']} | {r['params_total']} | "
            f"{r['measured_compute_seconds']} | {r['horizon']} |" for r in sel["ordered_candidates"])
        io.seal_md("MOP_OWNED_TEMPORAL_CORE_V1.md", f"""# Owned Temporal Core v1

Selected: `{s['cell']}`.

| field | value |
|---|---|
| architecture | {doc['core']['architecture']} |
| capacity tier | {doc['core']['capacity_tier']} |
| owned parameters | {doc['core']['owned_parameters']} |
| owned state | {doc['core']['owned_state']} |
| readout | {doc['core']['readout']} |
| horizon | {doc['core']['horizon']} |
| reset semantics | {doc['core']['reset_semantics']} |
| measured compute seconds | {doc['core']['compute_cost']['principal_wall_seconds_worst_bed']} |
| memory bytes | {doc['core']['memory_cost_bytes']} |
| valid domains | {', '.join(beds)} |

## The equivalence region

Selection took the smallest recurrent configuration whose worst bed score sits within
{io.EQUIVALENCE_MARGIN} of the best cell on its worst bed.

| cell | worst bed mean | gap to best | parameters | compute seconds | horizon |
|---|---|---|---|---|---|
{rows}

## Evidence ceiling

{doc['evidence_ceiling']}
""")
    else:
        io.seal_md("MOP_OWNED_TEMPORAL_CORE_V1.md", f"""# Owned Temporal Core v1

Not selected.

{sel.get('reason')}

A core is selected only when a minimal configuration survives two valid beds, two capable implementations,
matched strong alternatives, independent verification and full positive mutation testing. When it does not,
the honest artifact is this one.
""")
    io.seal("MOP_OWNED_TEMPORAL_CORE_V1.json", doc)
    print(f"core selection: selected={doc['selected']} "
          f"{sel.get('selected', {}).get('cell') if sel.get('selected') else sel.get('reason')}", flush=True)
    print("CORESEL_DONE", flush=True)


if __name__ == "__main__":
    main()
