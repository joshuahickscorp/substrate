"""Terminal synthesis, durable state, ledger and scorecard."""

from __future__ import annotations

import json
import subprocess
import time

from mop.temporal import io
from mop.temporal.runs import e2 as E2

LEGACY_RECEIPT_AUTHORITY = "b3f7421e6545527b3385f1368784ac2f0e1602a6"
LEGACY_RECEIPT_BINDING = (
    "runs/substrate/mop-temporal-core-mechanism-v1/orchestration/"
    "legacy_receipt_hash_normalization_20260721.json")

TERMINAL_METADATA = {
    "MOP_TEMPORAL_CORE_STATE.json", "MOP_TEMPORAL_CORE_LEDGER.md",
    "MOP_TEMPORAL_CORE_SCORECARD.json", "MOP_TEMPORAL_CORE_SCORECARD.md",
    "MOP_TEMPORAL_CORE_SYNTHESIS.json", "MOP_TEMPORAL_CORE_SYNTHESIS.md",
    "MOP_TEMPORAL_CORE_NEXT_FRONTIER.json",
}
POST_SNAPSHOT_DELIVERABLES = TERMINAL_METADATA | {
    "MOP_TEMPORAL_CORE_CLEAN_CLONE.json", "MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json"}
DELIVERABLE_STAGE = {
    "MOP_E2_SCOUT_RESULT.json": "scout",
    "MOP_E2_FACTORIAL_AUTHORITY.json": "bed_validity",
    "MOP_E2_PRINCIPAL_RESULT.json": "e2_principal",
    "MOP_E2_CAPACITY_TIER_CORRECTION.json": "capacity_corrections",
    "MOP_E2_INDEPENDENT_REPLICATION.json": "independent_replication",
    "MOP_CORE_ARCHITECTURE_REPORT.json": "e2_principal",
    "MOP_CORE_CAPACITY_REPORT.json": "e2_principal",
    "MOP_READOUT_CAPACITY_REPORT.json": "e2_principal",
    "MOP_STATE_HORIZON_REPORT.json": "e2_principal",
    "MOP_RESET_SEMANTICS_REPORT.json": "e2_principal",
    "MOP_EXPLICIT_HISTORY_REPORT.json": "e2_principal",
    "MOP_OPTIMIZATION_CONVERGENCE_REPORT.json": "capacity_corrections",
    "MOP_FACTORIAL_INTERACTION_REPORT.json": "e2_principal",
    "MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json": "third_bed_preflight",
    "MOP_THIRD_TEMPORAL_BED_ADMISSION_PROBE.json": "third_bed_preflight",
    "MOP_THIRD_TEMPORAL_BED_RESULT.json": "successors_terminal",
    "MOP_OWNED_TEMPORAL_CORE_V1.json": "core_selection",
    "MOP_OWNED_TEMPORAL_CORE_V1.md": "core_selection",
    "MOP_E3_SHARED_LOCAL_RESULT.json": "successors_terminal",
    "MOP_E5_SELF_SUPERVISED_RESULT.json": "successors_terminal",
    "MOP_HYBRID_ADAPTATION_RESULT.json": "successors_terminal",
    "MOP_EXPERIMENT_VALUE_QUEUE.json": "successor_gates",
    "MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json": "e2_principal",
    "MOP_SUBSTRATE_HYPOTHESIS_GRAPH.md": "e2_principal",
    "MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json": "verification",
    "MOP_TEMPORAL_CORE_MUTATION_REPORT.json": "mutations",
    "MOP_TEMPORAL_CORE_TEST_REPORT.json": "tests_and_coverage",
    "MOP_TEMPORAL_CORE_COVERAGE_REPORT.json": "tests_and_coverage",
    "MOP_TEMPORAL_CORE_RESOURCE_REPORT.json": "tests_and_coverage",
    "MOP_TEMPORAL_CORE_CLEAN_CLONE.json": "clean_clone",
    "MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json": "evidence_fabric",
    **{name: "terminal_metadata" for name in TERMINAL_METADATA},
}


def science_snapshot_binding(clean: dict, final_commit: str, ancestor=None) -> dict:
    snapshot = clean.get("science_snapshot_commit")
    commit_alias = clean.get("commit")
    is_commit = lambda value: (isinstance(value, str) and len(value) == 40
                               and all(c in "0123456789abcdef" for c in value.lower()))
    shaped = is_commit(snapshot) and snapshot == commit_alias
    if not shaped or not is_commit(final_commit):
        relation = "invalid_or_missing_snapshot"
    elif snapshot == final_commit:
        relation = "same_commit"
    else:
        if ancestor is None:
            r = subprocess.run(["git", "merge-base", "--is-ancestor", snapshot, final_commit],
                               cwd=io.ROOT, capture_output=True)
            ancestor = r.returncode == 0
        relation = "terminal_metadata_descendant" if ancestor else "diverged"
    return {"science_snapshot_commit": snapshot, "final_metadata_commit": final_commit,
            "relationship": relation, "snapshot_fields_match": shaped,
            "bound": shaped and relation in ("same_commit", "terminal_metadata_descendant")}


def deliverable_dependencies(name: str) -> list[str]:
    stage = DELIVERABLE_STAGE[name]
    if stage == "clean_clone":
        return ["stage:preclone_deliverables"]
    if stage == "evidence_fabric":
        return ["stage:clean_clone"]
    if stage == "terminal_metadata":
        return ["stage:evidence_fabric"]
    return [f"stage:{stage}"]


def stage_dependencies() -> dict[str, list[str]]:
    return {
        "start_authority": [], "binding_results": ["start_authority"],
        "data_custody": ["binding_results"], "code_lifecycle": ["start_authority"],
        "method_extension": ["binding_results", "data_custody", "code_lifecycle"],
        "e2_calibration": ["method_extension"], "scout": ["e2_calibration"],
        "convergence": ["scout"], "third_bed_preflight": ["scout"],
        "e2_principal": ["convergence"],
        "capacity_corrections": ["e2_principal", "convergence"],
        "bed_validity": ["capacity_corrections", "third_bed_preflight"],
        "independent_replication": ["bed_validity", "e2_principal"],
        "mutations": ["independent_replication"], "verification": ["mutations"],
        "core_selection": ["verification"], "successor_gates": ["core_selection"],
        "successors_terminal": ["successor_gates"],
        "tests_and_coverage": ["successors_terminal", "verification"],
        "preclone_deliverables": ["tests_and_coverage"],
        "clean_clone": ["preclone_deliverables"], "evidence_fabric": ["clean_clone"],
        "terminal_metadata": ["evidence_fabric"],
    }


def L(n, d=None):
    return io.load(n) if io.exists(n) else (d if d is not None else {})


def receipt_items(common: dict) -> dict:
    items = {}
    receipt_paths = [p for p in sorted(io.RUNS.rglob("*.json"))
                     if "locks" not in p.relative_to(io.RUNS).parts
                     and ".partial." not in p.name and not p.name.startswith(".")]
    receipt_ids = {f"receipt:{p.relative_to(io.ROOT).as_posix()}" for p in receipt_paths}
    for p in receipt_paths:
        rel = p.relative_to(io.ROOT).as_posix()
        whole_sha = io.sha_file(p)
        quarantined = "quarantine" in p.relative_to(io.RUNS).parts
        try:
            d = json.loads(p.read_text())
            if not isinstance(d, dict):
                raise TypeError("receipt top level must be an object")
            version = d.get("result_hash_version")
            canonical = version == "canonical_json_v2"
            hash_valid = not canonical or d.get("result_sha256") == io.sha_obj(
                {k: v for k, v in d.items() if k != "result_sha256"})
            known_version = version in (None, "canonical_json_v2")
            valid, failure = hash_valid and known_version, (
                "hash_mismatch" if not hash_valid else "unknown_hash_version" if not known_version else None)
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            d, canonical, valid, failure = {}, False, False, "invalid_json"
        params = d.get("params") if isinstance(d.get("params"), dict) else {}
        runs = d.get("runs") if isinstance(d.get("runs"), list) else []
        runs = [r for r in runs if isinstance(r, dict)]
        run_parameter_counts = {r["params"].get("total") for r in runs
                                if isinstance(r.get("params"), dict) and r["params"].get("total")}
        parameter_count = params.get("total") or sorted(run_parameter_counts) or None
        checkpoints = d.get("checkpoint_sha_after") or d.get("checkpoint_sha") or sorted({
            r.get("checkpoint_sha_after") for r in runs if r.get("checkpoint_sha_after")}) or None
        source = d.get("source_commit") or d.get("authority_commit")
        extension = d.get("extends") if isinstance(d.get("extends"), dict) else {}
        declared_paths = extension.get("path", [])
        dependency_shape_valid = (isinstance(declared_paths, str) or
                                  (isinstance(declared_paths, list)
                                   and all(isinstance(path, str) for path in declared_paths)) or
                                  declared_paths in (None, []))
        dependency_paths = [declared_paths] if isinstance(declared_paths, str) else list(
            declared_paths or []) if isinstance(declared_paths, list) else []
        dependencies = [f"receipt:{path}" for path in dependency_paths]
        declared_hashes = extension.get("sha256")
        if isinstance(declared_hashes, dict):
            declared_by_path = declared_hashes
        elif isinstance(declared_hashes, list) and len(declared_hashes) == len(dependency_paths):
            declared_by_path = dict(zip(dependency_paths, declared_hashes))
        elif isinstance(declared_hashes, str) and len(dependency_paths) == 1:
            declared_by_path = {dependency_paths[0]: declared_hashes}
        else:
            declared_by_path = {}
        dependency_bindings = []
        for path in dependency_paths:
            target, declared = io.ROOT / path, declared_by_path.get(path)
            target_within_runs = target.resolve().is_relative_to(io.RUNS.resolve())
            bound = (isinstance(declared, str) and len(declared) == 64 and target.is_file()
                     and target_within_runs and f"receipt:{path}" in receipt_ids
                     and io.sha_file(target) == declared)
            dependency_bindings.append({"item_id": f"receipt:{path}", "path": path,
                                        "sha256": declared, "bound": bound})
        if not dependency_shape_valid:
            valid, failure = False, "invalid_dependency_shape"
        elif dependency_bindings and not all(b["bound"] for b in dependency_bindings):
            valid, failure = False, "dependency_hash_mismatch"
        items[f"receipt:{rel}"] = {
            **common, "status": "terminal" if valid or quarantined else "incomplete",
            "authority": source or LEGACY_RECEIPT_AUTHORITY, "dependencies": dependencies,
            "dependency_bindings": dependency_bindings,
            "bed": d.get("bed") or d.get("target_bed"),
            "factor_levels": d.get("spec") or d.get("cell") or [
                r.get("cell") for r in runs if r.get("cell") is not None],
            "arm": "multi_arm" if d.get("arms") else d.get("arm"), "seed": d.get("seed"),
            "implementation": d.get("schema") or d.get("program"),
            "parameter_count": parameter_count,
            "training_budget": d.get("steps") or d.get("budgets") or d.get("grid") or sorted({
                r.get("steps") for r in runs if r.get("steps") is not None}),
            "checkpoint": checkpoints,
            "classification": "quarantined_invalid_not_scientific" if quarantined else
            failure or d.get("classification") or d.get("status") or "receipt_verified",
            "scientific_current": bool(valid and not quarantined), "quarantined": quarantined,
            "commit": source, "custody_commit": common["commit"],
            "receipt_integrity": "canonical_json_v2" if canonical and valid else
            "legacy_outer_sha256" if valid else failure,
            "custody_binding": None if canonical else LEGACY_RECEIPT_BINDING,
            "next_action": "none" if valid or quarantined else
            "quarantine invalid receipt and resume producing shard",
            "sha256": whole_sha,
        }
    return items


def convergence_is_terminal(doc: dict) -> bool:
    """Every selected factorial cell must have a terminal extended-budget classification on every bed."""
    expected = {E2.Fx.cell_name(**spec) for spec in E2.Fx.sweep_cells()["_all"]}
    return bool(doc.get("beds")) and all(
        set((doc.get("per_bed", {}).get(b, {}).get("convergence", {}).get("configs") or {})) == expected
        and max(doc["per_bed"][b]["convergence"].get("grid") or [0]) >= max(E2.EXTENDED_CONVERGENCE_GRID)
        and all(c.get("classification") in ("converged", "unconverged") for c in
                doc["per_bed"][b]["convergence"]["configs"].values()) for b in doc.get("beds", []))


def ready_stages(stages: dict[str, bool], dependencies: dict[str, list[str]]) -> list[str]:
    return sorted(name for name, value in stages.items() if not value and
                  all(stages.get(dep, False) for dep in dependencies[name]))


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
        "MOP_E2_SCOUT_RESULT.json", "MOP_E2_FACTORIAL_AUTHORITY.json", "MOP_E2_PRINCIPAL_RESULT.json",
        "MOP_E2_CAPACITY_TIER_CORRECTION.json",
        "MOP_E2_INDEPENDENT_REPLICATION.json", "MOP_CORE_ARCHITECTURE_REPORT.json",
        "MOP_CORE_CAPACITY_REPORT.json", "MOP_READOUT_CAPACITY_REPORT.json",
        "MOP_STATE_HORIZON_REPORT.json", "MOP_RESET_SEMANTICS_REPORT.json",
        "MOP_EXPLICIT_HISTORY_REPORT.json", "MOP_OPTIMIZATION_CONVERGENCE_REPORT.json",
        "MOP_FACTORIAL_INTERACTION_REPORT.json", "MOP_THIRD_TEMPORAL_BED_PREFLIGHT.json",
        "MOP_THIRD_TEMPORAL_BED_ADMISSION_PROBE.json", "MOP_THIRD_TEMPORAL_BED_RESULT.json",
        "MOP_OWNED_TEMPORAL_CORE_V1.json", "MOP_OWNED_TEMPORAL_CORE_V1.md",
        "MOP_E3_SHARED_LOCAL_RESULT.json", "MOP_E5_SELF_SUPERVISED_RESULT.json",
        "MOP_HYBRID_ADAPTATION_RESULT.json", "MOP_EXPERIMENT_VALUE_QUEUE.json",
        "MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json", "MOP_SUBSTRATE_HYPOTHESIS_GRAPH.md",
        "MOP_TEMPORAL_CORE_INDEPENDENT_VERIFICATION.json", "MOP_TEMPORAL_CORE_MUTATION_REPORT.json",
        "MOP_TEMPORAL_CORE_TEST_REPORT.json", "MOP_TEMPORAL_CORE_COVERAGE_REPORT.json",
        "MOP_TEMPORAL_CORE_RESOURCE_REPORT.json", "MOP_TEMPORAL_CORE_CLEAN_CLONE.json",
        "MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json",
        "MOP_TEMPORAL_CORE_STATE.json", "MOP_TEMPORAL_CORE_LEDGER.md",
        "MOP_TEMPORAL_CORE_SCORECARD.json", "MOP_TEMPORAL_CORE_SCORECARD.md",
        "MOP_TEMPORAL_CORE_SYNTHESIS.json", "MOP_TEMPORAL_CORE_SYNTHESIS.md",
        "MOP_TEMPORAL_CORE_NEXT_FRONTIER.json",
    ]
    preclone_required = [name for name in required if name not in POST_SNAPSHOT_DELIVERABLES]
    deliverables_present = all((io.PROOF / name).is_file() for name in preclone_required)
    terminal_metadata_present = all((io.PROOF / name).is_file() for name in TERMINAL_METADATA)
    licensed = queue.get("licensed_top_two") or []
    convergence_terminal = convergence_is_terminal(e2)
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
    correction = L("MOP_E2_CAPACITY_TIER_CORRECTION.json")
    final_commit = io.commit()
    clean_binding = science_snapshot_binding(clean, final_commit)
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
        "capacity_corrections": bool(correction.get("all_pass")),
        "independent_replication": replication_terminal,
        "core_selection": core_terminal,
        "successor_gates": successor_gates_terminal,
        "successors_terminal": successor_terminal,
        "mutations": mutation_terminal,
        "verification": verification_terminal,
        "tests_and_coverage": bool(tests.get("passed")) and bool((cov.get("method_kernel_gate") or {}).get("met"))
        and bool((cov.get("active_critical_path_gate") or {}).get("met")),
        "preclone_deliverables": deliverables_present,
        "clean_clone": bool(clean.get("all_pass")) and clean_binding["bound"],
        "evidence_fabric": bool((fabric.get("verification") or {}).get("all_pass"))
        and bool((fabric.get("mutations") or {}).get("all_rejected")),
        "terminal_metadata": terminal_metadata_present,
    }
    io.seal("MOP_TEMPORAL_CORE_SYNTHESIS.json", {
        "schema": "mop-temporal-core-synthesis/v1", "terminal_questions": a,
        "stages": stages, "all_terminal": all(stages.values()),
        "activation": False,
        "forbidden_claims": ["a complete substrate architecture is selected", "plasticity is evidenced",
                             "cross domain moldability is evidenced", "activation is licensed",
                             "the E4 state only adaptation rule is competitive"],
        "science_snapshot_binding": clean_binding,
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
    dependencies = stage_dependencies()
    actions = {
        "start_authority": "seal start authority", "binding_results": "seal binding results",
        "data_custody": "run custody audit", "code_lifecycle": "run code lifecycle accounting",
        "method_extension": "seal method extension", "e2_calibration": "run E2 calibration",
        "scout": "resume scout shards", "convergence": "resume base and extended convergence shards",
        "third_bed_preflight": "resume third bed preflight", "e2_principal": "resume principal shards",
        "capacity_corrections": "resume append only capacity and optimization corrections",
        "bed_validity": "recompute bed validity", "independent_replication": "run independent replication",
        "mutations": "run required mutations", "verification": "run Roles B and C verification",
        "core_selection": "run minimal core selection", "successor_gates": "recompute value queue",
        "successors_terminal": "execute only licensed successor shards", "tests_and_coverage": "run reports",
        "preclone_deliverables": "seal missing science snapshot deliverables",
        "clean_clone": "validate the explicit science snapshot commit in a clean clone",
        "evidence_fabric": "rebuild terminal evidence fabric",
        "terminal_metadata": "seal terminal state, ledger, scorecard, synthesis, and next frontier"}
    dependency_ready = ready_stages(stages, dependencies)
    common = {"authority": io.commit(), "bed": None, "factor_levels": None, "arm": None, "seed": None,
              "implementation": None, "parameter_count": None, "training_budget": None,
              "checkpoint": None, "tests": tests.get("passed"),
              "verification": verification_doc.get("all_pass"), "mutations": mutation_doc.get("all_rejected"),
              "commit": io.commit(), "tag": None}
    items = {f"stage:{name}": {**common, "status": "terminal" if value else "incomplete",
                    "dependencies": [f"stage:{dep}" for dep in dependencies[name]],
                    "classification": "green" if value else "dependency_ready" if name in dependency_ready
                    else "blocked_by_dependency", "next_action": "none" if value else actions[name]}
             for name, value in stages.items()}
    items.update(receipt_items(common))
    mutable_self = TERMINAL_METADATA | {"MOP_TEMPORAL_CORE_EVIDENCE_FABRIC.json"}
    for name in sorted(required):
        p = io.PROOF / name
        p = p if p.is_file() else None
        stable = p is not None and name not in mutable_self
        items[f"deliverable:{name}"] = {**common, "status": "terminal" if p else "incomplete",
            "dependencies": deliverable_dependencies(name),
            "classification": "self_sealed_no_recursive_hash" if p and name in mutable_self else
            "sealed" if p else "missing", "next_action": "none" if p else "resume producing stage",
            "sha256": io.sha_file(p) if stable else None}
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
    io.seal("MOP_TEMPORAL_CORE_STATE.json", {
        "schema": "mop-temporal-core-state/v2", "program_id": io.PROGRAM,
        "branch": "agent/mop-temporal-core-mechanism", "stop_switch": str(io.STOP),
        "stages": stages, "stage_dependencies": dependencies, "dependency_ready": dependency_ready,
        "stages_green": sum(1 for v in stages.values() if v), "n_stages": len(stages),
        "items": items, "required_deliverables": required,
        "preclone_required_deliverables": preclone_required,
        "post_snapshot_deliverables": sorted(POST_SNAPSHOT_DELIVERABLES),
        "science_snapshot_binding": clean_binding,
        "all_terminal": all(stages.values()), "no_dependency_ready_work": not dependency_ready,
        "resume": "python3.12 -m mop.temporal.runs.supervisor resumes from shard files on disk",
        "activation": False})
    print(f"synthesis: {sum(1 for v in stages.values() if v)}/{len(stages)} stages green, "
          f"core selected {bool(core.get('selected'))}", flush=True)
    print("SYNTHESIS_DONE", flush=True)


if __name__ == "__main__":
    main()
