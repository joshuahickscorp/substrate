"""Independent recomputation and mutation attacks on the Substrate program.

Two jobs, and neither trusts the module it is checking.

Recomputation reads the sealed artifacts as bytes and recomputes their headline numbers by a different
route, from a different starting point, without importing the code that produced them. If a module is
quietly reporting what it wishes were true, the arithmetic done here from its own sealed output will not
agree with it.

Mutation attacks go the other way. Each mutation breaks one guard on purpose and then runs the test that
is supposed to catch it, in a fresh process. A mutation that survives is a guard that does not work, and
it is reported as a survivor rather than folded into a pass rate. The suite is only meaningful if every
mutation dies.

"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from substrate import evidence as io
from substrate import historical

PY = sys.executable
SELECTED_MUTATION_WORKERS = 12


# ---------------------------------------------------------------- independent recomputation


def _sealed(name: str) -> dict:
    return json.loads((io.PROOF / name).read_text())


def _recompute_sx5(doc: dict) -> dict:
    """A second, independent derivation of SX5 straight from the temporal program's sealed receipts."""
    import statistics

    runs = historical.root("temporal_receipts")

    def pairs(bed: str) -> dict:
        cfg = json.loads((runs / "e2_converge" / f"converge_{bed}.json").read_text())["configs"]
        by_cell: dict[str, list] = {}
        for path in sorted(Path(runs / "e2_principal").glob(f"{bed}_*.json")):
            for row in json.loads(path.read_text())["runs"]:
                by_cell.setdefault(row["cell"], []).append(float(row["accuracy"]))
        out = {}
        for cell, scores in by_cell.items():
            entry = cfg.get(cell)
            if not entry:
                continue
            curve, sel = entry["curve"], entry["selected_checkpoint"]
            predicted = curve.get(str(sel), curve.get(sel))
            if predicted is not None:
                out[cell] = (float(predicted), sum(scores) / len(scores))
        return out

    checks = {}
    for key, reported in doc["principal"]["directions"].items():
        fit_bed, test_bed = key.split("_to_")
        fit, held = pairs(fit_bed), pairs(test_bed)
        offset = statistics.fmean(p - a for p, a in fit.values())
        prior = statistics.fmean(a for _, a in fit.values())
        families = sorted({c.split("|", 1)[0] for c in held})
        gains = []
        for f in families:
            rows = [(p, a) for c, (p, a) in held.items() if c.split("|", 1)[0] == f]
            naive = statistics.fmean(abs(p - a) for p, a in rows)
            fixed = statistics.fmean(abs(prior - a) for _, a in rows)
            upd = statistics.fmean(abs((p - offset) - a) for p, a in rows)
            gains.append(min(naive, fixed) - upd)
        effect = round(statistics.fmean(gains), 6)
        half = 1.96 * statistics.stdev(gains) / len(gains) ** 0.5
        checks[f"sx5_effect_{key}"] = {
            "reported": reported["effect_best_baseline_minus_updating"],
            "recomputed": effect,
            "agrees": reported["effect_best_baseline_minus_updating"] == effect,
            "route": "rebuilt the pairing and refitted the offset from the temporal receipts",
        }
        checks[f"sx5_support_{key}"] = {
            "reported": reported["supports"],
            "recomputed": bool(effect - half > 0.05),
            "agrees": reported["supports"] == bool(effect - half > 0.05),
            "route": "reapplied the SESOI rule to the independently derived interval",
        }
    return checks


def _seal_intact(doc: dict) -> bool:
    return doc.get("sha256") == io.sha_obj({k: v for k, v in doc.items() if k != "sha256"})


def recompute() -> dict:
    """Recompute each headline number from sealed bytes, by a route the producing module does not share."""
    checks: dict[str, dict] = {}

    def check(name: str, reported, recomputed, how: str):
        checks[name] = {
            "reported": reported,
            "recomputed": recomputed,
            "agrees": reported == recomputed,
            "route": how,
        }

    ws = _sealed("SUBSTRATE_WORKSPACE.json")
    globally_writable = sum(1 for r in ws["regions"] if r["writers"] == ["*"])
    check(
        "workspace_no_region_is_globally_writable",
        0,
        globally_writable,
        "counted writer sets in the sealed region table, not by importing the workspace",
    )
    check(
        "workspace_every_region_globally_readable",
        len(ws["regions"]),
        sum(1 for r in ws["regions"] if r["readers"] == ["*"]),
        "counted reader sets in the sealed region table",
    )

    ps = _sealed("SUBSTRATE_PERSPECTIVE_SYSTEM.json")
    check(
        "perspective_families_are_distinct",
        len(ps["catalog"]),
        len({p["family"] for p in ps["catalog"]}),
        "counted distinct families in the sealed catalog",
    )
    check(
        "perspective_family_accounting_closes",
        ps["families_declared"],
        len(ps["families_implemented"]) + len(ps["families_without_an_implementation"]),
        "added the implemented and missing family lists from the sealed artifact",
    )

    mem = _sealed("SUBSTRATE_MEMORY_SYSTEM.json")
    selections = mem["consolidation"]["comparison_on_a_probe_stream"]["selected"]
    check(
        "consolidation_selections_are_distinct",
        len(selections),
        len({tuple(v) for v in selections.values()}),
        "hashed the sealed selection lists rather than rerunning the policies",
    )
    oracle_only = [p["name"] for p in mem["consolidation"]["policies"] if not p["available_at_decision_time"]]
    check(
        "only_the_oracle_policy_uses_future_information",
        ["oracle"],
        oracle_only,
        "filtered the sealed policy table on availability at decision time",
    )

    world = _sealed("SUBSTRATE_WORLD_MODEL.json")
    tests, group = world["battery"]["tests"], world["test_to_distinction"]
    for distinction, reported in world["battery"]["distinctions"].items():
        members = [tests[t] for t, d in group.items() if d == distinction]
        again = round(sum(members) / len(members), 4)
        check(
            f"world_distinction_{distinction}",
            reported,
            again,
            "averaged the sealed per test scores back into their declared distinction",
        )

    mk = _sealed("SUBSTRATE_METACOGNITION.json")
    h = mk["oracle_headroom"]
    check(
        "metacognition_residual_headroom",
        h["residual"],
        round(h["oracle"] - max(h["per_simple_policy"].values()), 6),
        "recomputed oracle minus the strongest simple policy from the sealed rows",
    )
    check(
        "learned_metacognition_licensing",
        mk["learned_currently_licensed"],
        h["residual"] > 0.05,
        "reapplied the SESOI rule to the sealed residual",
    )

    cont = _sealed("SUBSTRATE_CONTINUITY_BATTERY.json")
    contested = [k for k, r in cont["surfaces"].items() if r["probed"] and r["replay_full"]]
    check(
        "continuity_contested_surfaces",
        sorted(cont["contested_surfaces"]),
        sorted(contested),
        "rederived the contested set from the sealed per surface rows",
    )
    check(
        "continuity_support_needs_a_capable_control",
        cont["supported"],
        bool(contested) and cont["contested_margin"] > 0.05 and cont["control_is_capable"],
        "reapplied the support rule to the sealed margins",
    )

    state = _sealed("SUBSTRATE_STATE.json")
    card = _sealed("SUBSTRATE_PROGRESS_SCORECARD.json")
    for category, row in card["categories"].items():
        if not row["items"]:
            continue
        earned = [i for i in row["items"] if (state["items"][i].get("result") or {}).get("scientific") is True]
        check(
            f"scorecard_evidence_{category}",
            row["evidence_pct"],
            round(100 * len(earned) / len(row["items"])),
            "recounted classified results in the sealed state file",
        )

    # SX5 is recomputed from the temporal program's own receipts, not from the SX5 receipt. The route
    # shares no code with the experiment: it re reads the sealed convergence and principal files, rebuilds
    # the pairing, refits the offset and re derives the per family errors from scratch.
    sx5 = io.RUNS / "experiments" / "SX5_decision.json"
    if sx5.is_file():
        doc = json.loads(sx5.read_text())
        if doc.get("licensed"):
            for name, row in _recompute_sx5(doc).items():
                checks[name] = row

    seals = {name: _seal_intact(_sealed(name)) for name in sorted(p.name for p in io.PROOF.glob("SUBSTRATE_*.json"))}
    disagreements = sorted(k for k, v in checks.items() if not v["agrees"])
    broken_seals = sorted(k for k, v in seals.items() if not v)
    return {
        "schema": "substrate-independent-verification/v1",
        "role": ("recomputation from sealed bytes by a second route, without importing the modules that produced them"),
        "checks": checks,
        "seal_integrity": seals,
        "disagreements": disagreements,
        "broken_seals": broken_seals,
        "all_pass": not disagreements and not broken_seals,
    }


# ---------------------------------------------------------------- mutation attacks

# each entry breaks one guard and names the test that must notice
MUTATIONS = (
    (
        "workspace_ignores_writer_sets",
        "substrate.workspace",
        "Workspace._permitted = lambda self, allowed, who: True",
        "tests/substrate/test_workspace.py::test_a_write_outside_the_declared_writers_is_refused",
    ),
    (
        "workspace_drops_provenance_requirement",
        "substrate.workspace",
        "REGIONS = tuple(type(r)(**{**vars(r), 'provenance': False}) for r in REGIONS)\nBY_NAME = {r.name: r for r in REGIONS}",
        "tests/substrate/test_workspace.py::test_provenance_and_confidence_are_mandatory_where_declared",
    ),
    (
        "untyped_control_becomes_typed",
        "substrate.workspace",
        "UntypedWorkspace.typed = True",
        "tests/substrate/test_workspace.py::test_the_untyped_control_removes_typing_and_nothing_else",
    ),
    (
        "learned_selector_opens_freely",
        "substrate.perspectives",
        "select = lambda strategy, catalog, **kw: catalog[:2]",
        "tests/substrate/test_perspectives.py::test_learned_selector_stays_closed_without_headroom",
    ),
    (
        "arbitration_discards_the_minority",
        "substrate.perspectives",
        "_arbitrate = arbitrate\n"
        "arbitrate = lambda outputs, **kw: {**_arbitrate(outputs, **kw), 'alternative_hypotheses': [],\n"
        "                                   'minority_preserved': 0}",
        "tests/substrate/test_perspectives.py::test_minority_hypothesis_survives_arbitration",
    ),
    (
        "generated_episodes_promote_on_a_flag",
        "substrate.memory",
        "EpisodicMemory.promote_to_training = lambda self, eid: self.store[eid]",
        "tests/substrate/test_memory.py::test_generated_episode_cannot_be_promoted_without_verification",
    ),
    (
        "procedures_transfer_on_their_own_episodes",
        "substrate.memory",
        "ProceduralMemory.transfer_test = lambda self, pid, evaluated_on, score, baseline: "
        "self.store[pid].__setattr__('transfer', {'improves': True, 'held_out': True, "
        "'evaluated_on': evaluated_on, 'score': score, 'baseline': baseline}) or self.store[pid].transfer",
        "tests/substrate/test_memory.py::test_procedure_requires_transfer_beyond_source_episodes",
    ),
    (
        "hygiene_deletes_audit_records",
        "substrate.memory",
        "hygiene = lambda records, *, audit_required, requests: {"
        "'applied': [{'record': r, 'action': a} for r, a in requests], 'refused': [],"
        "'audit_preserved': True, 'nothing_audit_required_was_deleted': True}",
        "tests/substrate/test_memory.py::test_hygiene_never_destroys_audit_required_records",
    ),
    (
        "intervening_becomes_conditioning",
        "substrate.world",
        "WorldModel.intervene = lambda self, state, do: self.predict({**state, **do})",
        "tests/substrate/test_world.py::test_intervening_is_not_the_same_operation_as_observing",
    ),
    (
        "self_facts_need_no_source",
        "substrate.selfmodel",
        "SelfFact.violations = lambda self: []",
        "tests/substrate/test_selfmodel.py::test_a_self_fact_without_a_source_is_refused",
    ),
    (
        "fast_adaptation_touches_shared_parameters",
        "substrate.plasticity",
        "LEVELS = tuple(type(l)(**{**vars(l), 'touches_shared_parameters': False, 'speed': 'fast'}) for l in LEVELS)\nBY_LEVEL = {l.name: l for l in LEVELS}",
        "tests/substrate/test_plasticity.py::test_fast_adaptation_does_not_touch_shared_parameters",
    ),
    (
        "slow_adaptation_skips_its_evidence_bar",
        "substrate.plasticity",
        "slow_adapt = lambda adaptation, **kw: {'level': adaptation.level, 'applied': True, "
        "'refusals': [], 'repetitions': 0, 'held_out': {}, 'retention': {}, 'rollback': ''}",
        "tests/substrate/test_plasticity.py::test_slow_adaptation_requires_repeated_evidence_and_rollback",
    ),
    # the first version of this mutation appended the forbidden list to the permitted one, which changed
    # nothing because the forbidden branch runs first. An equivalent mutant proves nothing, so this one
    # removes the branch instead.
    (
        "reorganize_drops_the_forbidden_branch",
        "substrate.plasticity",
        "reorganize = lambda change, *, measured, cost, sesoi=0.05: {'change': change, 'permitted': True, 'applied': True, 'reason': ''}",
        "tests/substrate/test_plasticity.py::test_forbidden_reorganizations_are_refused",
    ),
    (
        "forbidden_reorganization_list_is_emptied",
        "substrate.safety",
        "FORBIDDEN_REORGANIZATIONS = ()",
        "tests/substrate/test_plasticity.py::test_forbidden_reorganizations_are_refused",
    ),
    (
        "protected_surfaces_become_removable",
        "substrate.safety",
        "PROTECTED_SURFACES = ()",
        "tests/substrate/test_safety.py::test_protected_surfaces_cannot_be_removed_by_adaptation",
    ),
    (
        "forbidden_claims_pass",
        "substrate.safety",
        "check_claim = lambda text: []",
        "tests/substrate/test_safety.py::test_forbidden_claim_vocabulary_is_refused",
    ),
    (
        "integrity_stops_failing_closed",
        "substrate.safety",
        "integrity_report = lambda observations: {'schema': 'x', 'surfaces': {}, 'failed_surfaces': [], 'all_pass': True, 'fails_closed': False}",
        "tests/substrate/test_safety.py::test_integrity_violation_is_detected_and_fails_closed",
    ),
    (
        "latency_counts_as_thinking",
        "substrate.batteries",
        "NOT_EVIDENCE = ()\nALTERNATIVES = ALTERNATIVES + ('latency', 'hidden_activations')",
        "tests/substrate/test_batteries.py::test_thinking_requires_a_declared_alternative_to_beat",
    ),
    (
        "reflective_reports_answer_without_provenance",
        "substrate.batteries",
        "_rr = reflective_report\nreflective_report = lambda e, b: {**_rr(e, b), 'answered': True, 'failed_closed': False}",
        "tests/substrate/test_batteries.py::test_reflective_report_fails_closed_without_provenance",
    ),
    (
        "evidence_counts_on_presence_alone",
        "substrate.program",
        "evidence_state = lambda ref: {'reference': ref, 'present': True, 'counts': True, 'reason': ''}",
        "tests/substrate/test_program.py::test_evidence_that_is_stale_or_failing_does_not_count",
    ),
    (
        "implementation_raises_evidence",
        "substrate.program",
        "scorecard = lambda st=None: {'categories': {'working_memory': {'items': ['M1'], "
        "'implementation_pct': 100, 'evidence_pct': 100, 'items_with_earned_evidence': ['M1']}}}",
        "tests/substrate/test_program.py::test_evidence_never_rises_from_implementation_alone",
    ),
    (
        "admission_stops_requiring_completeness",
        "substrate.admission",
        "completeness = lambda prereg: []",
        "tests/substrate/test_admission.py::test_substrate_experiment_cannot_reach_principal_unproven",
    ),
    (
        "a_structurally_guaranteed_effect_is_licensed",
        "substrate.experiments",
        "_d = sx1_decision\nsx1_decision = lambda: {**_d(), 'licensed': True, 'classification': 'licensed'}",
        "tests/substrate/test_experiments.py::test_sx1_is_refused_because_its_effect_is_true_by_construction",
    ),
    (
        "a_refusal_is_filed_as_a_null",
        "substrate.deliverables",
        "methodological_refusals = lambda: []",
        "tests/substrate/test_experiments.py::test_a_refusal_is_not_recorded_as_a_null",
    ),
    (
        "an_underpowered_design_reaches_the_test_split",
        "substrate.experiments",
        "_e = sx1b_evidence\nsx1b_evidence = lambda: {**_e(), 'power': {'mde': 0.001, 'n_test_units': 7}}",
        "tests/substrate/test_experiments.py::test_sx1b_is_refused_on_power_and_never_touches_the_test_split",
    ),
    (
        "the_sesoi_is_lowered_until_a_bed_qualifies",
        "substrate.experiments",
        "BED_SCREEN_RULE = {**BED_SCREEN_RULE, 'sesoi': 0.001}",
        "tests/substrate/test_experiments.py::test_no_bed_under_custody_can_test_the_typed_workspace_hypothesis",
    ),
    (
        "a_measurement_boundary_closes_its_descendants",
        "substrate.deliverables",
        "_hg = hypothesis_graph\n"
        "def hypothesis_graph(st):\n"
        "    g = _hg(st)\n"
        "    for h in g['hypotheses']:\n"
        "        if h['id'] == 'H_arbitration_minority':\n"
        "            h['state'] = 'closed'\n"
        "            h['blocking_null'] = 'upstream measurement boundary'\n"
        "    return g",
        "tests/substrate/test_experiments.py::test_a_measurement_boundary_closes_nothing_downstream",
    ),
    (
        "the_loop_skips_a_stage_silently",
        "substrate.runtime",
        "CycleTrace.skip = lambda self, stage, reason: self.stages.__setitem__(stage, {'ran': True})",
        "tests/substrate/test_runtime.py::test_a_skipped_stage_says_why_rather_than_looking_like_one_that_ran",
    ),
    (
        "the_loop_restores_a_tampered_checkpoint",
        "substrate.runtime",
        "_r = Substrate.restore\n"
        "def restore(self, snapshot):\n"
        "    try:\n"
        "        return _r(self, snapshot)\n"
        "    except Refused:\n"
        "        return self\n"
        "Substrate.restore = restore",
        "tests/substrate/test_runtime.py::test_a_tampered_checkpoint_is_refused_rather_than_silently_restored",
    ),
    (
        "sx5_reports_support_below_the_sesoi",
        "substrate.experiments",
        "SESOI = 0.001",
        "tests/substrate/test_experiments.py::test_sx5_is_a_null_at_the_declared_effect_size",
    ),
    (
        "sx5_drops_the_second_direction",
        "substrate.experiments",
        "_r = sx5_run\n"
        "def sx5_run():\n"
        "    o = _r()\n"
        "    if o.get('licensed'):\n"
        "        d = o['principal']['directions']\n"
        "        o['principal']['directions'] = {k: v for k, v in list(d.items())[:1]}\n"
        "        o['principal']['both_directions_support'] = True\n"
        "    return o",
        "tests/substrate/test_experiments.py::test_sx5_requires_both_directions",
    ),
    (
        "the_ceiling_below_sesoi_is_blamed_on_the_unit_count",
        "substrate.experiments",
        "_d = _sx1b_diagnosis\n_sx1b_diagnosis = lambda ev: {**_d(ev), 'more_units_would_help': True, 'decisive_number': 'mde'}",
        "tests/substrate/test_experiments.py::test_the_sx1b_diagnosis_names_the_number_that_actually_decides",
    ),
)

RUNNER = """
import importlib, sys
import pytest
mod = importlib.import_module({module!r})
exec({patch!r}, mod.__dict__)
sys.exit(pytest.main(["-q", "--no-header", "-p", "no:cacheprovider", {node!r}]))
"""


def run_mutation(name: str, module: str, patch: str, node: str) -> dict:
    code = RUNNER.format(module=module, patch=patch, node=node)
    env = {**__import__("os").environ, "PYTHONPATH": str(io.ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([PY, "-c", code], cwd=io.ROOT, capture_output=True, text=True, env=env)
    rejected = r.returncode != 0
    return {
        "mutation": name,
        "module": module,
        "guard": node,
        "rejected": rejected,
        "survivor": not rejected,
        "detail": "" if rejected else (r.stdout or r.stderr).strip().splitlines()[-1:],
    }


def mutation_report(only: list[str] | None = None, workers: int = SELECTED_MUTATION_WORKERS) -> dict:
    selected = [m for m in MUTATIONS if only is None or m[0] in only]
    if workers < 1:
        raise ValueError("mutation workers must be positive")
    # Each attack still receives a fresh Python process and a fresh module graph.  The persistent
    # bounded thread pool only supervises those isolated children and preserves declaration order.
    with ThreadPoolExecutor(max_workers=min(workers, len(selected) or 1), thread_name_prefix="substrate-mutation") as pool:
        rows = list(pool.map(lambda mutation: run_mutation(*mutation), selected))
    survivors = [r["mutation"] for r in rows if r["survivor"]]
    return {
        "schema": "substrate-mutation-report/v1",
        "rule": (
            "every mutation must be rejected by the guard named beside it. A survivor is a guard "
            "that does not work and is reported as a survivor, never folded into a pass rate"
        ),
        "mutations": rows,
        "total": len(rows),
        "rejected": len(rows) - len(survivors),
        "survivors": survivors,
        "all_rejected": not survivors,
        "execution": {
            "model": "bounded persistent supervisor pool over isolated mutation subprocesses",
            "workers": min(workers, len(selected) or 1),
            "isolation": "fresh interpreter and module graph per mutation",
            "ordering": "declaration order retained",
        },
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "all"
    if command in ("recompute", "all"):
        doc = recompute()
        io.seal("SUBSTRATE_INDEPENDENT_VERIFICATION.json", doc)
        print(
            f"substrate recomputation: {len(doc['checks'])} checks, {len(doc['disagreements'])} disagreements, {len(doc['broken_seals'])} broken seals",
            flush=True,
        )
    if command in ("mutate", "all"):
        doc = mutation_report()
        io.seal("SUBSTRATE_MUTATION_REPORT.json", doc)
        print(
            f"substrate mutations: {doc['rejected']}/{doc['total']} rejected, survivors {doc['survivors']}",
            flush=True,
        )
    if command not in ("recompute", "mutate", "all"):
        raise ValueError(argv)


if __name__ == "__main__":
    main()
