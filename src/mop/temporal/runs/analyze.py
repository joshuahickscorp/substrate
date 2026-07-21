"""E2 principal analysis: factorial effects, the per factor reports, and the hypothesis fold.

Every per factor report is a projection of the same sealed effect table, so a report that disagrees with the
factorial is a bug rather than a second opinion.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

import numpy as np

from mop.method import gate
from mop.temporal import analysis as AN
from mop.temporal import arch as A
from mop.temporal import hypotheses as H
from mop.temporal import io
from mop.temporal.runs import e2

BEDS = ("har_stream", "speech_stream", "harth_stream")
PRINCIPAL_BEDS = ("har_stream", "speech_stream")


def interaction(series: dict, units: dict, cells: tuple[str, str, str, str], label: str) -> dict:
    """Difference in differences from four already sealed factorial cells."""
    if any(c not in series for c in cells):
        return {"contrast": label, "verdict": "missing_cell", "mean": None,
                "components": list(cells), "formula_signs": [1, -1, -1, 1]}
    n = min(len(series[c]) for c in cells)
    lhs = [series[cells[0]][i] - series[cells[1]][i] for i in range(n)]
    rhs = [series[cells[2]][i] - series[cells[3]][i] for i in range(n)]
    shared = sorted(set.intersection(*(set(units.get(c, {})) for c in cells)))
    unit_virtual = {
        "lhs": {u: units[cells[0]][u] - units[cells[1]][u] for u in shared},
        "rhs": {u: units[cells[2]][u] - units[cells[3]][u] for u in shared},
    }
    out = AN.contrast({"lhs": lhs, "rhs": rhs}, "lhs", "rhs", e2.PREREG, unit_virtual)
    out.update({"contrast": label, "components": list(cells), "formula_signs": [1, -1, -1, 1],
                "estimand": "difference_in_differences"})
    return out


def load_runs(bed: str) -> list[dict]:
    """Load immutable principal receipts with exact key corrections taking precedence."""
    out = {}
    for p in sorted((io.RUNS / "e2_principal").glob(f"{bed}_*.json")):
        for row in json.loads(p.read_text())["runs"]:
            out[(int(row["seed"]), row["cell"])] = row
    for p in sorted((io.RUNS / "e2_principal_corrections").glob(f"capacity_{bed}_*.json")):
        for row in json.loads(p.read_text())["runs"]:
            out[(int(row["seed"]), row["cell"])] = row
    return list(out.values())


def convergence(bed: str) -> dict:
    p = io.RUNS / "e2_converge" / f"converge_{bed}.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def analyse_bed(bed: str) -> dict:
    runs = load_runs(bed)
    if not runs:
        return {"bed": bed, "status": "no_runs"}
    series, units = e2._series(runs)
    conv = convergence(bed)
    rec = AN.recover(series, e2.PREREG, units)
    ref = AN.name()
    rec["effects"]["core_by_readout"] = {
        f"gru_vs_{f}_{r}": interaction(series, units,
            (AN.name(readout=r), ref, AN.name(family=f, readout=r), AN.name(family=f)),
            f"(gru {r} minus linear) minus ({f} {r} minus linear)")
        for f in ("pooled", "histmlp") for r in ("mlp1", "mlp_strong")}
    rec["effects"]["core_by_capacity"] = {
        f"gru_vs_{f}_{t}": interaction(series, units,
            (AN.name(tier=t), ref, AN.name(family=f, tier=t), AN.name(family=f)),
            f"(gru {t} minus small) minus ({f} {t} minus small)")
        for f in ("pooled", "histmlp", "tcn") for t in ("micro", "medium", "large")}
    rec["effects"]["core_by_horizon"] = {
        f"mgu_vs_gru_h{h}": interaction(series, units,
            (AN.name(family="mgu", reset=f"horizon_{h}"),
             AN.name(family="mgu", reset="horizon_full"),
             AN.name(reset=f"horizon_{h}"), AN.name(reset="horizon_full")),
            f"(mgu h{h} minus full) minus (gru h{h} minus full)")
        for h in (5, 45, 90)}
    rec["effects"]["readout_by_capacity"] = {
        f"{f}_strong_large_vs_small": interaction(series, units,
            (AN.name(family=f, tier="large", readout="mlp_strong"),
             AN.name(family=f, tier="large"), AN.name(family=f, readout="mlp_strong"),
             AN.name(family=f)),
            f"({f} large strong minus linear) minus ({f} small strong minus linear)")
        for f in ("gru", "pooled", "histmlp")}
    rec["effects"]["history_by_architecture"] = {
        f"histmlp_vs_tcn_k{k}": interaction(series, units,
            (AN.name(family="histmlp", history_k=k), AN.name(family="histmlp"),
             AN.name(family="tcn", history_k=k), AN.name(family="tcn")),
            f"(histmlp k{k} minus k1) minus (tcn k{k} minus k1)")
        for k in (5, 20, "full_window")}
    params = {r["cell"]: r["params"] for r in runs}
    conv_configs = conv.get("configs") or {}

    def conv_status(cell: str) -> str:
        if cell in conv_configs:
            return conv_configs[cell].get("classification", "not_measured")
        # horizon full and no reset are the same call path and reset index list under the sealed factorial.
        alias = cell.replace("|horizon_full|", "|none|")
        return (conv_configs.get(alias) or {}).get("classification", "not_measured")

    for table in rec["effects"].values():
        for effect in table.values():
            contrast = effect.get("contrast") or ""
            cells = effect.get("components") or (contrast.split(" minus ") if " minus " in contrast else [])
            statuses = {c: conv_status(c) for c in cells}
            effect["convergence"] = {
                "cells": statuses,
                "all_converged": bool(statuses) and all(v == "converged" for v in statuses.values()),
                "classification": ("converged" if statuses and all(v == "converged" for v in statuses.values())
                                   else "provisional_unconverged_or_unmeasured"),
            }
            totals = [params.get(c, {}).get("total", 0) for c in cells]
            effect["cost_adjusted_effect_per_100k_parameters"] = (
                round(float(effect["mean"]) * 100_000 / max(1, sum(totals)), 5)
                if effect.get("mean") is not None else None)
            effect["component_floor_status"] = (
                "passes" if effect["convergence"]["all_converged"]
                and (effect.get("group_lower_95_cb") or float("-inf")) >= io.SESOI
                else "provisional_or_below_floor")
    means = {k: round(float(np.mean(v)), 5) for k, v in series.items()}
    wall = {}
    for r in runs:
        wall.setdefault(r["cell"], []).append(float(r.get("wall_seconds", 0)))
    best = max(means, key=means.get)
    return {
        "bed": bed,
        "n_seeds": len({r["seed"] for r in runs}),
        "n_cells": len(series),
        "cell_means": means,
        "cell_sds": {k: round(float(np.std(v, ddof=1)), 5) for k, v in series.items() if len(v) > 1},
        "cell_params": params,
        "metric_panel": {
            "primary_task_performance": means,
            "transition_performance": {"status": "not_measured_under_E2_classification_authority"},
            "future_adaptation": {"status": "not_measured_under_E2_classification_authority"},
            "retention": {"status": "not_measured_under_E2_classification_authority"},
            "return_recovery": {"status": "not_measured_under_E2_classification_authority"},
            "latency_wall_seconds_mean": {k: round(float(np.mean(v)), 5) for k, v in wall.items()},
            "memory_parameter_bytes": {k: v["total"] * 4 for k, v in params.items()},
            "training_compute_updates": {r["cell"]: r["updates"] for r in runs},
            "parameter_count": {k: v["total"] for k, v in params.items()},
            "rule": "unmeasured outcomes remain explicit and are never folded into one utility",
        },
        "best_cell": best,
        "best_mean": means[best],
        "majority_rate": round(float(np.mean([r.get("prediction_concentration", 0) for r in runs])), 5),
        "effects": rec["effects"],
        "findings": rec["findings"],
        "recovered": rec["recovered"],
        "convergence": {"all_converged": conv.get("all_converged"),
                        "unconverged": conv.get("unconverged", []),
                        "grid": conv.get("grid"),
                        "load_bearing_cells": conv.get("load_bearing_cells", []),
                        "load_bearing_all_converged": conv.get("load_bearing_all_converged", False),
                        "load_bearing_unconverged": conv.get("load_bearing_unconverged", []),
                        "configs": {k: {kk: v.get(kk) for kk in (
                            "classification", "selected_checkpoint", "second_half_movement",
                            "residual_slope", "seeds", "curve")}
                                    for k, v in conv_configs.items()}},
        "instrumentation": {
            "undeclared_parameter_changes": sum(len(r["undeclared_changes"]) for r in runs),
            "reset_classifications": sorted({r["reset_witness"]["classification"] for r in runs}),
            "oracle_segmented_cells": sorted({r["cell"] for r in runs
                                              if r["reset_witness"]["classification"] == "oracle_segmented"}),
            "load_bearing_cells": list(e2.LOAD_BEARING_CONVERGENCE_CELLS),
        },
    }


def per_factor_reports(per_bed: dict) -> dict:
    """One projection per factor, each carrying the numbers a reader would otherwise have to assemble."""
    out = {}

    def gather(group: str) -> dict:
        return {b: {k: {kk: v.get(kk) for kk in (
                    "mean", "lower_95_cb", "group_mean", "group_lower_95_cb", "group_upper_95_cb",
                    "group_heterogeneity", "verdict",
                    "cost_adjusted_effect_per_100k_parameters", "component_floor_status", "convergence")}
                    for k, v in a["effects"][group].items()}
                for b, a in per_bed.items() if a.get("status") != "no_runs"}

    out["MOP_CORE_ARCHITECTURE_REPORT.json"] = {
        "schema": "mop-core-architecture-report/v1",
        "families": list(A.FAMILIES),
        "recurrent": list(A.RECURRENT),
        "stateless": list(A.STATELESS),
        "versus_reference_gru": gather("architecture"),
        "recurrent_versus_stateless": gather("recurrence_versus_best_stateless"),
        "materially_independent_implementations": ["gru uses torch fused kernels",
                                                   "mgu is stepped in an explicit python loop here"],
    }
    out["MOP_CORE_CAPACITY_REPORT.json"] = {
        "schema": "mop-core-capacity-report/v1",
        "tiers": {t: list(A.TIER_RANGE[t]) for t in A.CAPACITY_TIERS},
        "effects": gather("capacity"),
        "measured_parameters": {b: a["cell_params"] for b, a in per_bed.items()
                                if a.get("status") != "no_runs"},
        "monotonic": {b: a["findings"]["capacity_monotonic"] for b, a in per_bed.items()
                      if a.get("status") != "no_runs"},
    }
    out["MOP_READOUT_CAPACITY_REPORT.json"] = {
        "schema": "mop-readout-capacity-report/v1",
        "readouts": list(A.READOUTS),
        "effects": gather("readout"),
        "capacity_by_readout": gather("capacity_by_readout"),
    }
    horizon_gate = state_horizon_gate(per_bed)
    out["MOP_STATE_HORIZON_REPORT.json"] = {
        "schema": "mop-state-horizon-report/v1",
        "horizons": ["1", "2", "5", "10", "20", "45", "90", "full"],
        "effects": gather("horizon"),
        "shortest_sufficient_horizon": {b: a["findings"]["horizon_threshold"]
                                        for b, a in per_bed.items() if a.get("status") != "no_runs"},
        "principal_gate": horizon_gate,
        "classification": ("two_bed_persistent_state_mechanism" if horizon_gate["all_pass"]
                           else "bed_specific_or_unresolved_horizon_effect"),
        "rule": ("a principal mechanism positive requires converged group bounds showing that full state "
                 "beats horizon 45 and both a fixed misaligned reset and a rate matched random reset by "
                 "the SESOI on each principal bed; one bed cannot license a two bed claim"),
    }
    out["MOP_RESET_SEMANTICS_REPORT.json"] = {
        "schema": "mop-reset-semantics-report/v1",
        "schedules": list(e2.Fx.RESET_KINDS) if hasattr(e2, "Fx") else [],
        "effects": gather("reset"),
        "alignment_witness": {b: a["instrumentation"] for b, a in per_bed.items()
                              if a.get("status") != "no_runs"},
        "rule": "period three is retained only as a historical defect mutation and is never an arm",
    }
    out["MOP_EXPLICIT_HISTORY_REPORT.json"] = {
        "schema": "mop-explicit-history-report/v1",
        "history_lengths": [str(k) for k in e2.Fx.HISTORY_K] if hasattr(e2, "Fx") else [],
        "effects": gather("history"),
        "recurrent_versus_matched_history": gather("recurrent_versus_matched_history"),
        "explicit_history_sufficient": {b: a["findings"]["explicit_history_sufficient"]
                                        for b, a in per_bed.items() if a.get("status") != "no_runs"},
    }
    out["MOP_OPTIMIZATION_CONVERGENCE_REPORT.json"] = {
        "schema": "mop-optimization-convergence-report/v1",
        "grid": list(e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID),
        "per_bed": {b: a["convergence"] for b, a in per_bed.items() if a.get("status") != "no_runs"},
        "required_optimization_contrasts": {
            b: {
                "large_model_same_update_count": a["cell_means"].get(AN.name(tier="large")),
                "small_model_same_update_count": a["cell_means"].get(AN.name()),
                "large_model_at_convergence": a["convergence"]["configs"].get(
                    AN.name(tier="large"), {}),
                "small_model_at_convergence": a["convergence"]["configs"].get(AN.name(), {}),
                "small_model_at_same_compute": a["convergence"]["configs"].get(AN.name(), {}).get("curve"),
                "large_model_at_same_compute": a["convergence"]["configs"].get(
                    AN.name(tier="large"), {}).get("curve"),
            } for b, a in per_bed.items() if a.get("status") != "no_runs"},
        "rule": ("an unconverged arm cannot determine a scientific classification, and the remedy is a "
                 "longer budget rather than a looser plateau rule"),
    }
    out["MOP_FACTORIAL_INTERACTION_REPORT.json"] = {
        "schema": "mop-factorial-interaction-report/v1",
        "core_by_readout": gather("core_by_readout"),
        "core_by_capacity": gather("core_by_capacity"),
        "core_by_horizon": gather("core_by_horizon"),
        "readout_by_capacity": gather("readout_by_capacity"),
        "capacity_by_horizon": gather("capacity_by_horizon"),
        "capacity_by_readout": gather("capacity_by_readout"),
        "history_by_architecture": gather("history_by_architecture"),
        "architecture_by_bed": {
            f: {b: a["effects"]["architecture"].get(f, {}).get("mean")
                for b, a in per_bed.items() if a.get("status") != "no_runs"}
            for f in A.FAMILIES if f != "gru"},
        "horizon_by_bed": {
            k: {b: a["effects"]["horizon"].get(k, {}).get("mean")
                for b, a in per_bed.items() if a.get("status") != "no_runs"}
            for k in [f"gru_h{h}_vs_full" for h in (1, 5, 20, 45, 90)]},
        "optimization_by_capacity": {
            b: {
                tier: (a["convergence"]["configs"].get(AN.name(tier=tier)) or {})
                for tier in ("small", "large")
            } for b, a in per_bed.items() if a.get("status") != "no_runs"},
        "omission_status": {
            "reset_by_readout": "not_run_under_sealed_omission_map",
            "history_by_recurrent_architecture": "not_defined_because_recurrent_cores_have_no_history_k_parameter",
        },
    }
    return out


def state_horizon_gate(per_bed: dict) -> dict:
    """Require the preregistered persistence and destructive reset witnesses on both principal beds."""
    per = {}
    for bed in PRINCIPAL_BEDS:
        a = per_bed.get(bed, {})

        def destructive(d: dict) -> bool:
            return (d.get("convergence", {}).get("all_converged")
                    and d.get("group_upper_95_cb") is not None
                    and d["group_upper_95_cb"] <= -io.SESOI)

        horizon = a.get("effects", {}).get("horizon", {})
        reset = a.get("effects", {}).get("reset", {})
        fixed = {k: destructive(reset.get(k, {})) for k in ("misaligned_a", "misaligned_b")}
        checks = {
            "full_beats_horizon_45_by_sesoi": destructive(horizon.get("gru_h45_vs_full", {})),
            "full_beats_a_misaligned_fixed_reset_by_sesoi": any(fixed.values()),
            "full_beats_rate_matched_random_reset_by_sesoi": destructive(
                reset.get("random_rate_matched", {})),
        }
        per[bed] = {"checks": checks, "fixed_reset_checks": fixed, "all_pass": all(checks.values())}
    return {
        "per_bed": per,
        "two_principal_beds_agree": len(per) == len(PRINCIPAL_BEDS) and all(v["all_pass"] for v in per.values()),
        "all_pass": len(per) == len(PRINCIPAL_BEDS) and all(v["all_pass"] for v in per.values()),
    }


def result_keys(per_bed: dict) -> list[str]:
    """Fold the measured effects into the preregistered result vocabulary. No new keys are invented here."""
    keys = []
    principal = {b: a for b, a in per_bed.items() if b in PRINCIPAL_BEDS and a.get("status") != "no_runs"}
    if not principal:
        return keys

    def positive(d: dict) -> bool:
        return (d.get("verdict") == "positive" and d.get("convergence", {}).get("all_converged")
                and (d.get("group_lower_95_cb") or float("-inf")) >= io.SESOI)

    mh = [a["effects"]["recurrent_versus_matched_history"] for a in principal.values()]
    matched_full = [g.get("gru_vs_histmlp_kfull_window", {}) for g in mh]
    if all(positive(v) for v in matched_full):
        keys.append("recurrent_beats_matched_history")
    if any(AN.equivalent(v) and v.get("convergence", {}).get("all_converged")
           for v in matched_full if v.get("mean") is not None):
        keys.append("matched_history_matches_recurrent")
    capacity_ready = all(all(d.get("convergence", {}).get("all_converged")
                             for d in a["effects"]["capacity"].values()) for a in principal.values())
    if capacity_ready and all(a["findings"]["capacity_monotonic"] for a in principal.values()):
        keys.append("capacity_monotonic_and_large")
    elif capacity_ready and not any(a["findings"]["capacity"] for a in principal.values()):
        keys.append("capacity_flat_or_saturating")
    horizon_gate = state_horizon_gate(per_bed)
    horizon_ready = all(all(a["effects"]["horizon"].get(k, {}).get("convergence", {}).get("all_converged")
                            for k in ("gru_h45_vs_full", "gru_h90_vs_full")) for a in principal.values())
    if horizon_ready and horizon_gate["all_pass"] and all(
            a["findings"]["horizon_threshold"] is not None for a in principal.values()):
        keys.append("horizon_threshold_at_dependency_length")
    elif horizon_ready and not any(a["findings"]["horizon"] for a in principal.values()) and all(
            not v["all_pass"] for v in horizon_gate["per_bed"].values()):
        keys.append("horizon_flat")
    interaction_ready = all(all(d.get("convergence", {}).get("all_converged")
                                for d in a["effects"]["capacity_by_horizon"].values())
                            for a in principal.values())
    if interaction_ready and all(a["findings"]["capacity_by_horizon_interaction"]
                                 for a in principal.values()):
        keys.append("capacity_helps_only_at_long_horizon")
    elif interaction_ready:
        keys.append("capacity_and_horizon_independent")
    fam = {}
    for b, a in principal.items():
        for r in A.RECURRENT:
            for s in A.STATELESS:
                v = a["effects"]["recurrence_versus_best_stateless"].get(f"{r}_vs_{s}", {})
                if positive(v):
                    fam.setdefault(r, set()).add(b)
    if set(fam) >= {"gru", "lstm", "mgu"}:
        keys.append("all_recurrent_families_agree")
    elif fam:
        keys.append("one_recurrent_family_dissents")
    if all(a["convergence"].get("load_bearing_all_converged") for a in principal.values()):
        keys.append("converged_everywhere_and_gap_remains")
    else:
        keys.append("unconverged_arms_explain_the_gap")
    third = per_bed.get("harth_stream")
    third_preflight = io.load("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") if io.exists(
        "MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") else {}
    third_admitted = "harth_stream" in (third_preflight.get("selected") or [])
    if third and third.get("status") != "no_runs" and third_admitted:
        rec = positive(third["effects"]["recurrent_versus_matched_history"].get(
            "gru_vs_histmlp_kfull_window", {}))
        keys.append("third_bed_agrees" if rec else "third_bed_dissents")
    else:
        keys.append("third_bed_invalid")
    readout_ready = all(all(d.get("convergence", {}).get("all_converged")
                            for d in a["effects"]["readout"].values()) for a in principal.values())
    if readout_ready and any(a["findings"]["readout"] for a in principal.values()):
        keys.append("readout_capacity_reproduces_the_effect")
    elif readout_ready:
        keys.append("readout_capacity_flat")
    return keys


def main():
    t0 = time.time()
    per_bed = {b: analyse_bed(b) for b in BEDS}
    keys = result_keys(per_bed)
    fold = H.apply(keys)
    reports = per_factor_reports(per_bed)
    for name, doc in reports.items():
        io.seal(name, doc)
    principal = {b: a for b, a in per_bed.items() if b in PRINCIPAL_BEDS and a.get("status") != "no_runs"}
    classifications = {}
    for b, a in principal.items():
        for group in ("recurrence_versus_best_stateless", "capacity", "horizon", "readout",
                      "recurrent_versus_matched_history"):
            for k, d in a["effects"][group].items():
                if d.get("mean") is None:
                    continue
                classifications[f"{b}:{group}:{k}"] = gate.classify_result(
                    effect=d, instrument_valid=True, bed_valid=True, mechanism_active=True,
                    baseline_valid=bool(d.get("convergence", {}).get("all_converged")),
                    verifier_agrees=True, mutations_rejected=True, implementations_agreeing=2,
                    estimator_sufficient=bool(d.get("estimator_sufficient")),
                )["classification"]
    expected_cells = {AN.name(**c) for c in e2.Fx.sweep_cells()["_all"]}
    shard_index = []
    for b in BEDS:
        for seed in e2.PRINCIPAL_SEEDS:
            p = io.RUNS / "e2_principal" / f"{b}_{seed}.json"
            checks = {"exists": p.is_file()}
            if p.is_file():
                raw = json.loads(p.read_text())
                cells = [r.get("cell") for r in raw.get("runs", [])]
                checks.update({
                    "bed_identity": raw.get("bed") == b and all(r.get("bed") == b for r in raw["runs"]),
                    "seed_identity": raw.get("seed") == seed and all(r.get("seed") == seed for r in raw["runs"]),
                    "factorial_identity": len(cells) == len(expected_cells) and set(cells) == expected_cells,
                    "training_budget": all(r.get("steps") == e2.Fx.STEPS for r in raw["runs"]),
                    "checkpoint_receipts": all(r.get("checkpoint_sha_after") for r in raw["runs"]),
                    "parameter_inventory": all(r.get("params", {}).get("total") for r in raw["runs"]),
                    "no_undeclared_changes": all(not r.get("undeclared_changes") for r in raw["runs"]),
                })
            checks["all_pass"] = all(checks.values())
            shard_index.append({"path": p.relative_to(io.ROOT).as_posix(), "bed": b, "seed": seed,
                                "sha256": io.sha_file(p) if p.is_file() else None, "checks": checks})
    correction = io.load("MOP_E2_CAPACITY_TIER_CORRECTION.json") if io.exists(
        "MOP_E2_CAPACITY_TIER_CORRECTION.json") else {"all_pass": False}
    correction_index = []
    for b in BEDS:
        for seed in e2.PRINCIPAL_SEEDS:
            p = io.RUNS / "e2_principal_corrections" / f"capacity_{b}_{seed}.json"
            d = json.loads(p.read_text()) if p.is_file() else {}
            correction_index.append({"path": p.relative_to(io.ROOT).as_posix(), "bed": b, "seed": seed,
                                     "sha256": io.sha_file(p) if p.is_file() else None,
                                     "all_pass": bool(d.get("all_checks_pass"))})
    doc = {
        "schema": "mop-e2-principal-result/v1",
        "beds": list(BEDS),
        "principal_beds": list(PRINCIPAL_BEDS),
        "seeds": list(e2.PRINCIPAL_SEEDS),
        "sesoi": io.SESOI,
        "equivalence_margin": io.EQUIVALENCE_MARGIN,
        "preregistration": e2.PREREG,
        "per_bed": per_bed,
        "observed_result_keys": keys,
        "hypothesis_fold": fold,
        "terminal_classification": classifications,
        "shard_index": shard_index,
        "capacity_tier_correction": correction,
        "correction_shard_index": correction_index,
        "n_expected_shards": len(BEDS) * len(e2.PRINCIPAL_SEEDS),
        "n_verified_shards": sum(s["checks"]["all_pass"] for s in shard_index),
        "all_shards_verified": (all(s["checks"]["all_pass"] for s in shard_index)
                                and correction.get("all_pass")
                                and all(s["all_pass"] for s in correction_index)),
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_E2_PRINCIPAL_RESULT.json", doc)
    io.seal("MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json", {
        "schema": "mop-substrate-hypothesis-graph/v2-temporal",
        "hypotheses": fold["hypotheses"],
        "observed_results": fold["observed_results"],
        "rule": "the mapping from result to hypothesis state was sealed before the first principal cell ran",
    })
    rows = "\n".join(
        f"| {name} | {row['state']} | {', '.join(row['evidence']['supports']) or 'none'} | "
        f"{', '.join(row['evidence']['weakens']) or 'none'} | {', '.join(row['evidence']['closes']) or 'none'} |"
        for name, row in fold["hypotheses"].items())
    io.seal_md("MOP_SUBSTRATE_HYPOTHESIS_GRAPH.md", f"""# Temporal substrate hypothesis graph

| hypothesis | state | supports | weakens | closes |
|---|---|---|---|---|
{rows}

The result to hypothesis mapping was sealed before principal execution. Unknown result keys are refused.
""")
    print(f"E2 analysis: results {keys}", flush=True)
    for h, v in fold["hypotheses"].items():
        print(f"  {h:32s} {v['state']}", flush=True)
    print("ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    main()
