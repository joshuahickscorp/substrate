"""Conditional successor gates.

E3, E5 and the hybrid adaptation experiment each have an opening condition that is checked against sealed
evidence rather than against enthusiasm. A successor that does not open still produces a gate artifact, so a
reader can see what was not run and why.

House style: no dashes.
"""

from __future__ import annotations

import time

from mop.temporal import io


def _e2():
    return io.load("MOP_E2_PRINCIPAL_RESULT.json") if io.exists("MOP_E2_PRINCIPAL_RESULT.json") else {}


def _core():
    return io.load("MOP_OWNED_TEMPORAL_CORE_V1.json") if io.exists("MOP_OWNED_TEMPORAL_CORE_V1.json") else {}


def gates() -> dict:
    e2, core = _e2(), _core()
    fold = (e2.get("hypothesis_fold") or {}).get("hypotheses", {})
    beds = [b for b in e2.get("principal_beds", []) if e2.get("per_bed", {}).get(b, {}).get("status") != "no_runs"]
    core_positive = bool(core.get("selected")) and fold.get("H1_recurrence", {}).get("state") == "supported"

    capacity_increases_forgetting = False
    bed_heterogeneity = False
    if len(beds) >= 2:
        arch = {b: e2["per_bed"][b]["effects"]["architecture"] for b in beds}
        signs = []
        for f in set(arch[beds[0]]) & set(arch[beds[1]]):
            a, c = arch[beds[0]][f].get("mean"), arch[beds[1]][f].get("mean")
            if a is not None and c is not None:
                signs.append((a > 0) == (c > 0))
        bed_heterogeneity = bool(signs) and not all(signs)
        caps = [e2["per_bed"][b]["findings"].get("capacity_monotonic") for b in beds]
        capacity_increases_forgetting = any(c is False for c in caps)

    residual_headroom_for_self_supervision = False

    third_preflight = (io.load("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json")
                       if io.exists("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json") else {})
    third_result = (io.load("MOP_THIRD_TEMPORAL_BED_RESULT.json")
                    if io.exists("MOP_THIRD_TEMPORAL_BED_RESULT.json") else {})
    return {
        "E3_shared_versus_local": {
            "condition": ("an E2 core positive exists and either capacity increases forgetting or bed "
                          "heterogeneity indicates domain specific representation"),
            "core_positive": core_positive,
            "capacity_increases_forgetting": capacity_increases_forgetting,
            "bed_heterogeneity": bed_heterogeneity,
            "opens": bool(core_positive and (capacity_increases_forgetting or bed_heterogeneity)),
        },
        "E5_self_supervised": {
            "condition": ("a robust E2 core positive remains, capacity, explicit history and optimization do "
                          "not explain all value, and self supervision has measurable residual oracle headroom"),
            "core_positive": core_positive,
            "residual_oracle_headroom_measured": residual_headroom_for_self_supervision,
            "opens": bool(core_positive and residual_headroom_for_self_supervision),
            "why_not": ("no residual oracle headroom for self supervision has been measured, and the failed "
                        "Architecture F premise may not be repeated on a guess"),
        },
        "hybrid_adaptation": {
            "condition": "a minimal temporal core is selected",
            "core_selected": bool(core.get("selected")),
            "opens": bool(core.get("selected")),
            "constraint": ("the E4 state only rule is not reused unchanged. A hybrid must introduce a new "
                           "causal premise, and head plus state is only interesting if the state arm carries "
                           "zero hidden core updates"),
        },
        "third_bed_replication": {
            "condition": "a third valid secondary bed exists and its replication experiment is terminal",
            "bed_admitted": "harth_stream" in (third_preflight.get("selected") or []),
            "terminal_result": third_result.get("classification"),
            "opens": ("harth_stream" in (third_preflight.get("selected") or [])
                      and bool(third_result.get("classification"))),
        },
    }


def ranked_successors(g: dict) -> list[str]:
    opened = [k for k, v in g.items() if v.get("opens")]
    return sorted(opened, key=lambda k: {"third_bed_replication": 0, "E3_shared_versus_local": 1,
                                         "hybrid_adaptation": 2, "E5_self_supervised": 3}.get(k, 9))


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
        "rule": "execute the top two licensed successors, not every branch that is listed",
    })
    print(f"successor gates: opened {opened}, licensed {licensed}", flush=True)
    print("SUCCESSORS_DONE", flush=True)


if __name__ == "__main__":
    main()
