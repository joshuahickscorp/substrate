"""Terminal synthesis, durable state, ledger and scorecard."""

from __future__ import annotations

import json
import time

from mop.temporal import io


def L(n, d=None):
    return io.load(n) if io.exists(n) else (d if d is not None else {})


def answers() -> dict:
    e2 = L("MOP_E2_PRINCIPAL_RESULT.json")
    core = L("MOP_OWNED_TEMPORAL_CORE_V1.json")
    ver = L("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json")
    mut = L("MOP_TEMPORAL_CORE_MUTATION_REPORT.json")
    cov = L("MOP_TEMPORAL_CORE_COVERAGE_REPORT.json")
    cust = L("MOP_DATA_CUSTODY_AUTHORITY.json")
    codel = L("MOP_ACTIVE_CODE_ACCOUNTING.json")
    froz = L("MOP_FROZEN_REPRODUCIBILITY_ACCOUNTING.json")
    gates = L("MOP_EXPERIMENT_VALUE_QUEUE.json")
    third = L("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json")
    fold = (e2.get("hypothesis_fold") or {}).get("hypotheses", {})
    per = e2.get("per_bed", {})

    def bed(b, path, default=None):
        cur = per.get(b, {})
        for k in path:
            cur = cur.get(k, {}) if isinstance(cur, dict) else {}
        return cur if cur != {} else default

    return {
        "1 was the E1 temporal core effect reproduced": {
            b: bed(b, ["findings", "recurrence"]) for b in per if per[b].get("status") != "no_runs"},
        "2 was the magnitude comparable": {
            b: bed(b, ["findings", "base_effect_size"]) for b in per if per[b].get("status") != "no_runs"},
        "3 did larger capacity increase the effect": {
            b: bed(b, ["findings", "capacity"]) for b in per if per[b].get("status") != "no_runs"},
        "4 did it saturate": {b: bed(b, ["findings", "capacity_monotonic"]) for b in per
                              if per[b].get("status") != "no_runs"},
        "6 what was the smallest useful core": (core.get("core") or {}).get("owned_parameters"),
        "7 did a strong readout reproduce the effect": {
            b: bed(b, ["findings", "readout"]) for b in per if per[b].get("status") != "no_runs"},
        "8 did explicit history reproduce the effect": {
            b: bed(b, ["findings", "explicit_history_sufficient"]) for b in per
            if per[b].get("status") != "no_runs"},
        "9 was recurrence independently necessary": fold.get("H1_recurrence", {}).get("state"),
        "11 what was the shortest useful horizon": {
            b: bed(b, ["findings", "horizon_threshold"]) for b in per if per[b].get("status") != "no_runs"},
        "16 were all load bearing baselines converged": {
            b: bed(b, ["convergence", "all_converged"]) for b in per if per[b].get("status") != "no_runs"},
        "17 did optimization explain any apparent architecture effect": fold.get(
            "H5_optimization", {}).get("state"),
        "19 did both valid beds agree": len([b for b in e2.get("principal_beds", [])
                                             if per.get(b, {}).get("status") != "no_runs"]) == 2,
        "20 was a third valid bed found": third.get("selected"),
        "22 which hypotheses survived": [k for k, v in fold.items() if v["state"] in ("supported", "mixed")],
        "23 which hypotheses closed": [k for k, v in fold.items() if v["state"] == "closed"],
        "24 was Owned Temporal Core v1 selected": bool(core.get("selected")),
        "25 which architecture does it use": (core.get("core") or {}).get("architecture"),
        "26 how many parameters does it have": (core.get("core") or {}).get("owned_parameters"),
        "27 what state does it own": (core.get("core") or {}).get("owned_state"),
        "28 what horizon does it retain": (core.get("core") or {}).get("horizon"),
        "31 did E3 open": (gates.get("gates") or {}).get("E3_shared_versus_local", {}).get("opens"),
        "32 did E5 open": (gates.get("gates") or {}).get("E5_self_supervised", {}).get("opens"),
        "33 did hybrid adaptation open": (gates.get("gates") or {}).get("hybrid_adaptation", {}).get("opens"),
        "34 did data custody pass": cust.get("guard_mutations_all_pass"),
        "35 can worktree deletion remove a unique corpus": False,
        "36 what code remains active": codel.get("active_runtime_plus_substrate"),
        "37 what code remains frozen for reproducibility": (froz.get("frozen_reproducibility") or {}).get("loc"),
        "38 did independent replication pass": fold.get("H7_architecture_family", {}).get("state"),
        "39 did every required mutation fail": mut.get("all_rejected"),
        "40 what scientific evidence ceiling was reached": core.get("evidence_ceiling"),
        "41 is a useful owned substrate component now evidenced": bool(core.get("selected")),
        "42 is a complete substrate architecture selected": False,
        "43 is plasticity evidenced": False,
        "44 is cross domain moldability evidenced": False,
        "45 is activation licensed": False,
        "46 what exact experiment has the highest information value next": (
            (gates.get("licensed_top_two") or ["E3 shared versus local temporal representation"])[0]),
        "verification": {"role_b": (ver.get("role_b") or {}).get("all_pass"),
                         "role_c": (ver.get("role_c") or {}).get("all_pass")},
        "coverage": {"method_kernel": (cov.get("method_kernel_gate") or {}).get("met"),
                     "active_critical_path": (cov.get("active_critical_path_gate") or {}).get("met")},
    }


def main():
    t0 = time.time()
    a = answers()
    e2 = L("MOP_E2_PRINCIPAL_RESULT.json")
    core = L("MOP_OWNED_TEMPORAL_CORE_V1.json")
    stages = {
        "start_authority": io.exists("MOP_TEMPORAL_CORE_START_AUTHORITY.json"),
        "binding_results": io.exists("MOP_TEMPORAL_CORE_BINDING_RESULTS.json"),
        "data_custody": io.exists("MOP_DATA_CUSTODY_AUTHORITY.json"),
        "code_lifecycle": io.exists("MOP_CODE_LIFECYCLE_AUTHORITY.json"),
        "method_extension": io.exists("MOP_TEMPORAL_METHOD_EXTENSION.json"),
        "e2_calibration": bool(L("MOP_E2_CALIBRATION.json").get("all_pass")),
        "bed_validity": io.exists("MOP_E2_FACTORIAL_AUTHORITY.json"),
        "third_bed_preflight": io.exists("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json"),
        "e2_principal": io.exists("MOP_E2_PRINCIPAL_RESULT.json"),
        "core_selection": io.exists("MOP_OWNED_TEMPORAL_CORE_V1.json"),
        "successor_gates": io.exists("MOP_EXPERIMENT_VALUE_QUEUE.json"),
        "mutations": bool(L("MOP_TEMPORAL_CORE_MUTATION_REPORT.json").get("all_rejected")),
        "verification": bool(L("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json").get("all_pass")),
        "reports": io.exists("MOP_TEMPORAL_CORE_COVERAGE_REPORT.json"),
        "evidence_fabric": io.exists("MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json"),
    }
    io.seal("MOP_TEMPORAL_CORE_STATE.json", {
        "schema": "mop-temporal-core-state/v1", "program_id": io.PROGRAM,
        "branch": "agent/mop-temporal-core-mechanism", "stop_switch": str(io.STOP),
        "stages": stages, "stages_green": sum(1 for v in stages.values() if v), "n_stages": len(stages),
        "resume": "python3.12 -m mop.temporal.runs.supervisor resumes from shard files on disk",
        "activation": False})
    io.seal("MOP_TEMPORAL_CORE_SYNTHESIS.json", {
        "schema": "mop-temporal-core-synthesis/v1", "terminal_questions": a,
        "activation": False,
        "forbidden_claims": ["a complete substrate architecture is selected", "plasticity is evidenced",
                             "cross domain moldability is evidenced", "activation is licensed",
                             "the E4 state only adaptation rule is competitive"],
        "wall_seconds": round(time.time() - t0, 1)})
    scorecard = {
        "schema": "mop-temporal-core-scorecard/v1",
        "dimensions": {
            "data_custody": 100 if a["34 did data custody pass"] else 0,
            "method_extension": 100 if io.exists("MOP_TEMPORAL_METHOD_EXTENSION.json") else 0,
            "calibration": 100 if stages["e2_calibration"] else 0,
            "bed_validity": 100 if stages["bed_validity"] else 0,
            "principal_factorial": 100 if stages["e2_principal"] else 0,
            "independent_replication": 100 if a["38 did independent replication pass"] else 0,
            "verification": 100 if stages["verification"] else 0,
            "mutations": 100 if stages["mutations"] else 0,
            "coverage": 100 if (a["coverage"] or {}).get("method_kernel") else 0,
            "code_lifecycle": 100 if stages["code_lifecycle"] else 0,
        },
        "stages_green": stages, "activation": False}
    io.seal("MOP_TEMPORAL_CORE_SCORECARD.json", scorecard)
    io.seal("MOP_TEMPORAL_CORE_NEXT_FRONTIER.json", {
        "schema": "mop-temporal-core-next-frontier/v1",
        "next": a["46 what exact experiment has the highest information value next"],
        "hypotheses_open": a["22 which hypotheses survived"],
        "hypotheses_closed": a["23 which hypotheses closed"],
        "activation": False})
    rows = "\n".join(f"| {k} | {json.dumps(v)[:150]} |" for k, v in a.items())
    io.seal_md("MOP_TEMPORAL_CORE_SYNTHESIS.md", f"""# Temporal core synthesis

| question | answer |
|---|---|
{rows}

## Activation

False, and never separately granted.
""")
    srows = "\n".join(f"| {k} | {v} |" for k, v in scorecard["dimensions"].items())
    io.seal_md("MOP_TEMPORAL_CORE_SCORECARD.md", f"""# Temporal core scorecard

| dimension | score |
|---|---|
{srows}
""")
    io.seal_md("MOP_TEMPORAL_CORE_LEDGER.md", f"""# Temporal core ledger

Stages green: {sum(1 for v in stages.values() if v)} of {len(stages)}.

{json.dumps(stages, indent=1)}

## Selected core

{json.dumps(core.get("core"), indent=1) if core.get("selected") else "not selected: " + str((core.get("selection") or {}).get("reason"))}

## Resume

`python3.12 -m mop.temporal.runs.supervisor` resumes from the shard files on disk without replanning.
""")
    print(f"synthesis: {sum(1 for v in stages.values() if v)}/{len(stages)} stages green, "
          f"core selected {bool(core.get('selected'))}", flush=True)
    print("SYNTHESIS_DONE", flush=True)


if __name__ == "__main__":
    main()
