"""Conditional successor gates.

E3, E5 and the hybrid adaptation experiment each have an opening condition that is checked against sealed
evidence rather than against enthusiasm. A successor that does not open still produces a gate artifact, so a
reader can see what was not run and why.

House style: no dashes.
"""

from __future__ import annotations

import time

from mop.method import power
from mop.temporal import factorial as Fx
from mop.temporal import io
from mop.temporal.runs import e2 as E2


def _e2():
    return io.load("MOP_E2_PRINCIPAL_RESULT.json") if io.exists("MOP_E2_PRINCIPAL_RESULT.json") else {}


def _core():
    return io.load("MOP_OWNED_TEMPORAL_CORE_V1.json") if io.exists("MOP_OWNED_TEMPORAL_CORE_V1.json") else {}


def _capacity_forgetting(e2: dict) -> dict:
    """A capacity proxy is not a forgetting result. Require a measured retention or interference contrast."""
    measured = e2.get("capacity_retention_or_interference") or {}
    effects = measured.get("per_independent_unit_effects") or []
    permitted = measured.get("estimand") in (
        "retention_loss_large_minus_small", "interference_cost_large_minus_small")
    valid = bool(measured.get("measured") is True and permitted and len(effects) >= 2)
    lower = power.lcb(effects) if valid else None
    return {
        "measured": valid,
        "increases_forgetting": bool(valid and lower is not None and lower >= io.SESOI),
        "estimand": measured.get("estimand"),
        "mean": round(sum(effects) / len(effects), 5) if valid else None,
        "lower_95_cb": round(lower, 5) if lower is not None else None,
        "n_independent_units": len(effects) if valid else 0,
        "source": "MOP_E2_PRINCIPAL_RESULT.json#/capacity_retention_or_interference",
        "why_unmeasured": (None if valid else
                           "E2 did not measure retention or interference as a function of capacity"),
    }


def _ranking(voi: float, cost: int, evidence: list[dict]) -> dict:
    complete = bool(voi >= 0 and cost > 0 and evidence and all(e.get("artifact") and e.get("field")
                                                               for e in evidence))
    return {
        "value_of_information": round(float(voi), 8),
        "estimated_parameter_update_cost": int(cost),
        "priority_score": float(voi / cost) if complete else None,
        "evidence": evidence,
        "complete": complete,
    }


def gates() -> dict:
    e2, core = _e2(), _core()
    fold = (e2.get("hypothesis_fold") or {}).get("hypotheses", {})
    beds = [b for b in e2.get("principal_beds", []) if e2.get("per_bed", {}).get(b, {}).get("status") != "no_runs"]
    core_positive = bool(core.get("selected")) and fold.get("H1_recurrence", {}).get("state") == "supported"

    capacity_measurement = _capacity_forgetting(e2)
    capacity_increases_forgetting = capacity_measurement["increases_forgetting"]
    interaction_report = (io.load("MOP_FACTORIAL_INTERACTION_REPORT.json")
                          if io.exists("MOP_FACTORIAL_INTERACTION_REPORT.json") else {})
    bed_dids = interaction_report.get("architecture_by_bed") or {}
    bounded = [max(float(d.get("group_lower_95_cb") or 0.0),
                   -float(d.get("group_upper_95_cb") or 0.0), 0.0)
               for d in bed_dids.values() if d.get("mean") is not None]
    heterogeneity_magnitude = max(bounded, default=0.0)
    bed_heterogeneity = heterogeneity_magnitude >= io.SESOI

    residual_headroom_for_self_supervision = False

    third_preflight = (io.load("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json")
                       if io.exists("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") else {})
    third_result = (io.load("MOP_THIRD_TEMPORAL_BED_RESULT.json")
                    if io.exists("MOP_THIRD_TEMPORAL_BED_RESULT.json") else {})
    selected = (core.get("selection") or {}).get("selected") or {}
    owned_params = int(selected.get("params_total") or (core.get("core") or {}).get("owned_parameters") or 1)
    e3_voi = max(heterogeneity_magnitude,
                 float(capacity_measurement.get("lower_95_cb") or 0.0))
    e3_cost = 2 * len(E2.PRINCIPAL_SEEDS) * 2 * Fx.STEPS * owned_params
    hybrid_voi = float(e2.get("sesoi") or io.SESOI)
    hybrid_cost = len(beds or (1,)) * len(E2.PRINCIPAL_SEEDS) * 3 * (Fx.STEPS // 4) * owned_params
    third_voi = float((third_result.get("effect") or {}).get("group_lower_95_cb") or (
        e2.get("sesoi") or io.SESOI if third_result.get("classification") == "replicated" else 0.0))
    third_cost = len(E2.PRINCIPAL_SEEDS) * Fx.STEPS * owned_params
    e3_rank = _ranking(e3_voi, e3_cost, [
        {"artifact": "MOP_FACTORIAL_INTERACTION_REPORT.json", "field": "/architecture_by_bed",
         "value": round(heterogeneity_magnitude, 8)},
        {"artifact": "MOP_E2_PRINCIPAL_RESULT.json",
         "field": "/capacity_retention_or_interference/lower_95_cb",
         "value": capacity_measurement.get("lower_95_cb")},
        {"artifact": "MOP_OWNED_TEMPORAL_CORE_V1.json", "field": "/core/owned_parameters",
         "value": owned_params},
    ])
    hybrid_rank = _ranking(hybrid_voi, hybrid_cost, [
        {"artifact": "MOP_E2_PRINCIPAL_RESULT.json", "field": "/sesoi", "value": hybrid_voi},
        {"artifact": "MOP_OWNED_TEMPORAL_CORE_V1.json", "field": "/core/owned_parameters",
         "value": owned_params},
    ])
    third_rank = _ranking(third_voi, third_cost, [
        {"artifact": "MOP_THIRD_TEMPORAL_BED_RESULT.json", "field": "/effect/group_lower_95_cb",
         "value": third_voi},
        {"artifact": "MOP_OWNED_TEMPORAL_CORE_V1.json", "field": "/core/owned_parameters",
         "value": owned_params},
    ])
    return {
        "E3_shared_versus_local": {
            "condition": ("an E2 core positive exists and either capacity increases forgetting or bed "
                          "heterogeneity indicates domain specific representation"),
            "core_positive": core_positive,
            "capacity_increases_forgetting": capacity_increases_forgetting,
            "capacity_forgetting_measurement": capacity_measurement,
            "bed_heterogeneity": bed_heterogeneity,
            "bed_heterogeneity_magnitude": round(heterogeneity_magnitude, 8),
            "opens": bool(core_positive and (capacity_increases_forgetting or bed_heterogeneity)),
            "ranking": e3_rank,
        },
        "E5_self_supervised": {
            "condition": ("a robust E2 core positive remains, capacity, explicit history and optimization do "
                          "not explain all value, and self supervision has measurable residual oracle headroom"),
            "core_positive": core_positive,
            "residual_oracle_headroom_measured": residual_headroom_for_self_supervision,
            "opens": bool(core_positive and residual_headroom_for_self_supervision),
            "why_not": ("no residual oracle headroom for self supervision has been measured, and the failed "
                        "Architecture F premise may not be repeated on a guess"),
            "ranking": _ranking(0.0, e3_cost, [
                {"artifact": "MOP_E2_PRINCIPAL_RESULT.json", "field": "/residual_oracle_headroom",
                 "value": None}]),
        },
        "hybrid_adaptation": {
            "condition": "a minimal temporal core is selected",
            "core_selected": bool(core.get("selected")),
            "opens": bool(core.get("selected")),
            "constraint": ("the E4 state only rule is not reused unchanged. A hybrid must introduce a new "
                           "causal premise, and head plus state is only interesting if the state arm carries "
                           "zero hidden core updates"),
            "ranking": hybrid_rank,
        },
        "third_bed_replication": {
            "condition": "a third valid secondary bed exists and independently reproduces the core effect",
            "bed_admitted": "harth_stream" in (third_preflight.get("selected") or []),
            "terminal_result": third_result.get("classification"),
            "opens": ("harth_stream" in (third_preflight.get("selected") or [])
                      and third_result.get("classification") == "replicated"),
            "why_not": (None if third_result.get("classification") == "replicated" else
                        "the terminal third bed result did not independently reproduce the core effect"),
            "ranking": third_rank,
        },
    }


def ranked_successors(g: dict) -> list[str]:
    opened = [k for k, v in g.items() if v.get("opens")]
    legacy = {"third_bed_replication": 0, "E3_shared_versus_local": 1,
              "hybrid_adaptation": 2, "E5_self_supervised": 3}

    def rank(k):
        score = (g[k].get("ranking") or {}).get("priority_score")
        score = float(score) if isinstance(score, (int, float)) else float("-inf")
        return -score, legacy.get(k, 9), k

    return sorted(opened, key=rank)


def main():
    t0 = time.time()
    g = gates()
    opened = [k for k, v in g.items() if v.get("opens")]
    ranked = ranked_successors(g)
    licensed = ranked[:2]
    for name, key in (("MOP_E3_SHARED_LOCAL_RESULT.json", "E3_shared_versus_local"),
                      ("MOP_E5_SELF_SUPERVISED_RESULT.json", "E5_self_supervised"),
                      ("MOP_HYBRID_ADAPTATION_RESULT.json", "hybrid_adaptation")):
        existing = io.load(name) if io.exists(name) else {}
        actually_executed = bool(existing.get("experiment_terminal"))
        io.seal(name, {
            "schema": f"mop-successor-gate/{key}",
            "gate": g[key],
            "opened": key in licensed,
            "status": ("executed" if key in licensed and actually_executed else
                       "licensed_pending_execution" if key in licensed else
                       "opened_not_licensed" if g[key].get("opens") else "gate_closed"),
            "experiment_terminal": actually_executed,
            "result": existing.get("result") if actually_executed else None,
            "mutation_checks": (existing.get("mutation_checks")
                                or (existing.get("result") or {}).get("mutation_checks")
                                if actually_executed else {}),
            "rule": "a conditional experiment that does not open still produces this artifact",
        })
    io.seal("MOP_EXPERIMENT_VALUE_QUEUE.json", {
        "schema": "mop-experiment-value-queue/v2-temporal",
        "gates": g,
        "opened": opened,
        "licensed_top_two": licensed,
        "numerical_ranking": {k: g[k].get("ranking") for k in ranked},
        "rule": "execute the top two licensed successors, not every branch that is listed",
    })
    print(f"successor gates: opened {opened}, licensed {licensed}", flush=True)
    print("SUCCESSORS_DONE", flush=True)


if __name__ == "__main__":
    main()
