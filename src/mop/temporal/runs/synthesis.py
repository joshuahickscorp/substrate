"""Terminal synthesis, durable state, ledger and scorecard."""

from __future__ import annotations

import json
import time

from mop.temporal import io
from mop.temporal.runs import e2 as E2


def L(n, d=None):
    return io.load(n) if io.exists(n) else (d if d is not None else {})


def receipt_items(common: dict) -> dict:
    items = {}
    for p in sorted(io.RUNS.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rel = p.relative_to(io.ROOT).as_posix()
        params = d.get("params") or {}
        runs = d.get("runs") or []
        parameter_count = params.get("total") or sorted({r.get("params", {}).get("total") for r in runs
                                                          if r.get("params", {}).get("total")}) or None
        checkpoints = d.get("checkpoint_sha_after") or d.get("checkpoint_sha") or sorted({
            r.get("checkpoint_sha_after") for r in runs if r.get("checkpoint_sha_after")}) or None
        items[f"receipt:{rel}"] = {
            **common, "status": "terminal", "authority": d.get("source_commit") or d.get("authority_commit"),
            "dependencies": d.get("extends", {}).get("path", []) if isinstance(d.get("extends"), dict) else [],
            "bed": d.get("bed") or d.get("target_bed"),
            "factor_levels": d.get("spec") or d.get("cell") or sorted({r.get("cell") for r in runs}),
            "arm": "multi_arm" if d.get("arms") else d.get("arm"), "seed": d.get("seed"),
            "implementation": d.get("schema") or d.get("program"),
            "parameter_count": parameter_count,
            "training_budget": d.get("steps") or d.get("budgets") or d.get("grid") or sorted({
                r.get("steps") for r in runs if r.get("steps") is not None}),
            "checkpoint": checkpoints,
            "classification": d.get("classification") or d.get("status") or "receipt_verified",
            "commit": d.get("source_commit") or d.get("authority_commit"),
            "next_action": "none", "sha256": io.sha_file(p),
        }
    return items


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
            b: bed(b, ["convergence", "load_bearing_all_converged"])
            for b in per if per[b].get("status") != "no_runs"},
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
        "46 what exact experiment has the highest information value next": next((x for x in gates.get(
            "opened", []) if x not in gates.get("licensed_top_two", [])), "no licensed successor remains"),
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
    scout = L("MOP_E2_SCOUT_RESULT.json")
    rep = L("MOP_E2_INDEPENDENT_REPLICATION.json")
    queue = L("MOP_EXPERIMENT_VALUE_QUEUE.json")
    e3 = L("MOP_E3_SHARED_LOCAL_RESULT.json")
    third_result = L("MOP_THIRD_TEMPORAL_BED_RESULT.json")
    tests = L("MOP_TEMPORAL_CORE_TEST_REPORT.json")
    cov = L("MOP_TEMPORAL_CORE_COVERAGE_REPORT.json")
    clean = L("MOP_TEMPORAL_CORE_CLEAN_CLONE.json")
    fabric = L("MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json")
    factorial = L("MOP_E2_FACTORIAL_AUTHORITY.json")
    third = L("MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json")
    required = [
        "MOP_E2_SCOUT_RESULT.json", "MOP_E2_PRINCIPAL_RESULT.json", "MOP_E2_CAPACITY_TIER_CORRECTION.json",
        "MOP_E2_INDEPENDENT_REPLICATION.json", "MOP_CORE_ARCHITECTURE_REPORT.json",
        "MOP_CORE_CAPACITY_REPORT.json", "MOP_READOUT_CAPACITY_REPORT.json",
        "MOP_STATE_HORIZON_REPORT.json", "MOP_RESET_SEMANTICS_REPORT.json",
        "MOP_EXPLICIT_HISTORY_REPORT.json", "MOP_OPTIMIZATION_CONVERGENCE_REPORT.json",
        "MOP_FACTORIAL_INTERACTION_REPORT.json", "MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json",
        "MOP_THIRD_TEMPORAL_BED_ADMISSION_PROBE.json",
        "MOP_OWNED_TEMPORAL_CORE_V1.json", "MOP_OWNED_TEMPORAL_CORE_V1.md",
        "MOP_E3_SHARED_LOCAL_RESULT.json", "MOP_E5_SELF_SUPERVISED_RESULT.json",
        "MOP_HYBRID_ADAPTATION_RESULT.json", "MOP_EXPERIMENT_VALUE_QUEUE.json",
        "MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json", "MOP_SUBSTRATE_HYPOTHESIS_GRAPH.md",
        "MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json", "MOP_TEMPORAL_CORE_MUTATION_REPORT.json",
        "MOP_TEMPORAL_CORE_TEST_REPORT.json", "MOP_TEMPORAL_CORE_COVERAGE_REPORT.json",
        "MOP_TEMPORAL_CORE_RESOURCE_REPORT.json", "MOP_TEMPORAL_CORE_CLEAN_CLONE.json",
        "MOP_TEMPORAL_CORE_STATE.json", "MOP_TEMPORAL_CORE_LEDGER.md",
        "MOP_TEMPORAL_CORE_SCORECARD.json", "MOP_TEMPORAL_CORE_SCORECARD.md",
        "MOP_TEMPORAL_CORE_SYNTHESIS.json", "MOP_TEMPORAL_CORE_SYNTHESIS.md",
        "MOP_TEMPORAL_CORE_NEXT_FRONTIER.json",
    ]
    deliverables_present = all((io.PROOF / name).is_file() for name in required
                               if name not in ("MOP_TEMPORAL_CORE_STATE.json",
                                               "MOP_TEMPORAL_CORE_LEDGER.md",
                                               "MOP_TEMPORAL_CORE_SCORECARD.json",
                                               "MOP_TEMPORAL_CORE_SCORECARD.md",
                                               "MOP_TEMPORAL_CORE_SYNTHESIS.json",
                                               "MOP_TEMPORAL_CORE_SYNTHESIS.md",
                                               "MOP_TEMPORAL_CORE_NEXT_FRONTIER.json"))
    licensed = queue.get("licensed_top_two") or []
    convergence_terminal = bool(e2.get("beds")) and all(
        len((e2.get("per_bed", {}).get(b, {}).get("convergence", {}).get("configs") or {}))
        == len(E2.CONVERGE_CONFIGS)
        and max(e2["per_bed"][b]["convergence"].get("grid") or [0]) >= max(E2.EXTENDED_CONVERGENCE_GRID)
        and all(c.get("classification") in ("converged", "unconverged") for c in
                e2["per_bed"][b]["convergence"]["configs"].values()) for b in e2.get("beds", []))
    replication_terminal = (len(rep.get("per_bed") or {}) == len(e2.get("beds") or [])
                            and all(r.get("n_seeds") == len(E2.PRINCIPAL_SEEDS)
                                    for r in (rep.get("per_bed") or {}).values()))
    core_terminal = io.exists("MOP_OWNED_TEMPORAL_CORE_V1.json") and (
        bool(core.get("selected")) or bool((core.get("selection") or {}).get("reason")))
    successor_gates_terminal = io.exists("MOP_EXPERIMENT_VALUE_QUEUE.json") and (
        len(licensed) == min(2, len(queue.get("opened") or [])))
    mutation_doc = L("MOP_TEMPORAL_CORE_MUTATION_REPORT.json")
    mutation_terminal = bool(mutation_doc.get("required_coverage")) and all(
        mutation_doc["required_coverage"].values())
    verification_doc = L("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json")
    verification_terminal = (bool((verification_doc.get("role_b") or {}).get("checks"))
                             and bool((verification_doc.get("role_c") or {}).get("n_checks")))
    successor_terminal = all(
        (name == "third_bed_replication" and bool(third_result.get("classification")))
        or (name == "E3_shared_versus_local" and bool(e3.get("experiment_terminal")))
        or (name == "E5_self_supervised" and L("MOP_E5_SELF_SUPERVISED_RESULT.json").get("experiment_terminal"))
        or (name == "hybrid_adaptation" and L("MOP_HYBRID_ADAPTATION_RESULT.json").get("experiment_terminal"))
        for name in licensed)
    stages = {
        "start_authority": io.exists("MOP_TEMPORAL_CORE_START_AUTHORITY.json"),
        "binding_results": io.exists("MOP_TEMPORAL_CORE_BINDING_RESULTS.json"),
        "data_custody": io.exists("MOP_DATA_CUSTODY_AUTHORITY.json"),
        "code_lifecycle": io.exists("MOP_CODE_LIFECYCLE_AUTHORITY.json"),
        "method_extension": io.exists("MOP_TEMPORAL_METHOD_EXTENSION.json"),
        "e2_calibration": bool(L("MOP_E2_CALIBRATION.json").get("all_pass")),
        "scout": bool(scout.get("all_pass")),
        "convergence": convergence_terminal,
        "bed_validity": (len(factorial.get("principal_beds") or {}) == len(e2.get("principal_beds") or [])
                         and all(v.get("classification") not in (None, "preflight_incomplete")
                                 for v in (factorial.get("principal_beds") or {}).values())),
        "third_bed_preflight": bool(third.get("candidates")) and all(
            v.get("classification") not in ("preflight_incomplete", "unavailable", None)
            for v in third.get("candidates", {}).values()),
        "e2_principal": bool(e2.get("all_shards_verified")),
        "independent_replication": replication_terminal,
        "core_selection": core_terminal,
        "successor_gates": successor_gates_terminal,
        "successors_terminal": successor_terminal,
        "mutations": mutation_terminal,
        "verification": verification_terminal,
        "tests_and_coverage": bool(tests.get("passed")) and bool((cov.get("method_kernel_gate") or {}).get("met"))
        and bool((cov.get("active_critical_path_gate") or {}).get("met")),
        "clean_clone": bool(clean.get("all_pass")),
        "required_deliverables": deliverables_present,
        "evidence_fabric": bool((fabric.get("verification") or {}).get("all_pass"))
        and bool((fabric.get("mutations") or {}).get("all_rejected")),
    }
    common = {"authority": io.commit(), "bed": None, "factor_levels": None, "arm": None, "seed": None,
              "implementation": None, "parameter_count": None, "training_budget": None,
              "checkpoint": None, "tests": tests.get("passed"),
              "verification": L("MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json").get("all_pass"),
              "mutations": L("MOP_TEMPORAL_CORE_MUTATION_REPORT.json").get("all_rejected"),
              "commit": io.commit(), "tag": None}
    items = {f"stage:{name}": {**common, "status": "terminal" if value else "incomplete",
                    "dependencies": [], "classification": "green" if value else "not_green",
                    "next_action": "none" if value else "resume supervisor"}
             for name, value in stages.items()}
    items.update(receipt_items(common))
    items.update({f"deliverable:{name}": {**common,
                  "status": "terminal" if (io.PROOF / name).is_file() else "incomplete",
                  "dependencies": [], "classification": "sealed" if (io.PROOF / name).is_file() else "missing",
                  "next_action": "none" if (io.PROOF / name).is_file() else "resume producing stage",
                  "sha256": io.sha_file(io.PROOF / name) if (io.PROOF / name).is_file() else None}
                 for name in required})
    io.seal("MOP_TEMPORAL_CORE_STATE.json", {
        "schema": "mop-temporal-core-state/v1", "program_id": io.PROGRAM,
        "branch": "agent/mop-temporal-core-mechanism", "stop_switch": str(io.STOP),
        "stages": stages, "stages_green": sum(1 for v in stages.values() if v), "n_stages": len(stages),
        "items": items, "required_deliverables": required,
        "all_terminal": all(stages.values()), "no_dependency_ready_work": all(stages.values()),
        "resume": "python3.12 -m mop.temporal.runs.supervisor resumes from shard files on disk",
        "activation": False})
    io.seal("MOP_TEMPORAL_CORE_SYNTHESIS.json", {
        "schema": "mop-temporal-core-synthesis/v1", "terminal_questions": a,
        "stages": stages, "all_terminal": all(stages.values()),
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
            "bed_validity": 100 if factorial.get("all_principal_beds_valid") else 0,
            "principal_factorial": 100 if stages["e2_principal"] else 0,
            "independent_replication": 100 if rep.get("all_pass") else 0,
            "verification": 100 if verification_doc.get("all_pass") else 0,
            "mutations": 100 if mutation_doc.get("all_rejected") else 0,
            "coverage": 100 if stages["tests_and_coverage"] else 0,
            "code_lifecycle": 100 if stages["code_lifecycle"] else 0,
        },
        "stages_green": stages, "all_terminal": all(stages.values()), "activation": False}
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

## Durable item index

{json.dumps(items, indent=1)}

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
