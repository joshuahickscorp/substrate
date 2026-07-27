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

House style: no dashes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from mop.cognition import io

PY = sys.executable


# ---------------------------------------------------------------- independent recomputation


def _sealed(name: str) -> dict:
    return json.loads((io.PROOF / name).read_text())


def _seal_intact(doc: dict) -> bool:
    return doc.get("sha256") == io.sha_obj({k: v for k, v in doc.items() if k != "sha256"})


def recompute() -> dict:
    """Recompute each headline number from sealed bytes, by a route the producing module does not share."""
    checks: dict[str, dict] = {}

    def check(name: str, reported, recomputed, how: str):
        checks[name] = {"reported": reported, "recomputed": recomputed,
                        "agrees": reported == recomputed, "route": how}

    ws = _sealed("SUBSTRATE_WORKSPACE.json")
    globally_writable = sum(1 for r in ws["regions"] if r["writers"] == ["*"])
    check("workspace_no_region_is_globally_writable", 0, globally_writable,
          "counted writer sets in the sealed region table, not by importing the workspace")
    check("workspace_every_region_globally_readable", len(ws["regions"]),
          sum(1 for r in ws["regions"] if r["readers"] == ["*"]),
          "counted reader sets in the sealed region table")

    ps = _sealed("SUBSTRATE_PERSPECTIVE_SYSTEM.json")
    check("perspective_families_are_distinct", len(ps["catalog"]),
          len({p["family"] for p in ps["catalog"]}),
          "counted distinct families in the sealed catalog")
    check("perspective_family_accounting_closes", ps["families_declared"],
          len(ps["families_implemented"]) + len(ps["families_without_an_implementation"]),
          "added the implemented and missing family lists from the sealed artifact")

    mem = _sealed("SUBSTRATE_MEMORY_SYSTEM.json")
    selections = mem["consolidation"]["comparison_on_a_probe_stream"]["selected"]
    check("consolidation_selections_are_distinct", len(selections),
          len({tuple(v) for v in selections.values()}),
          "hashed the sealed selection lists rather than rerunning the policies")
    oracle_only = [p["name"] for p in mem["consolidation"]["policies"]
                   if not p["available_at_decision_time"]]
    check("only_the_oracle_policy_uses_future_information", ["oracle"], oracle_only,
          "filtered the sealed policy table on availability at decision time")

    world = _sealed("SUBSTRATE_WORLD_MODEL.json")
    tests, group = world["battery"]["tests"], world["test_to_distinction"]
    for distinction, reported in world["battery"]["distinctions"].items():
        members = [tests[t] for t, d in group.items() if d == distinction]
        again = round(sum(members) / len(members), 4)
        check(f"world_distinction_{distinction}", reported, again,
              "averaged the sealed per test scores back into their declared distinction")

    mk = _sealed("SUBSTRATE_METACOGNITION.json")
    h = mk["oracle_headroom"]
    check("metacognition_residual_headroom", h["residual"],
          round(h["oracle"] - max(h["per_simple_policy"].values()), 6),
          "recomputed oracle minus the strongest simple policy from the sealed rows")
    check("learned_metacognition_licensing", mk["learned_currently_licensed"], h["residual"] > 0.05,
          "reapplied the SESOI rule to the sealed residual")

    cont = _sealed("SUBSTRATE_CONTINUITY_BATTERY.json")
    contested = [k for k, r in cont["surfaces"].items() if r["probed"] and r["replay_full"]]
    check("continuity_contested_surfaces", sorted(cont["contested_surfaces"]), sorted(contested),
          "rederived the contested set from the sealed per surface rows")
    check("continuity_support_needs_a_capable_control", cont["supported"],
          bool(contested) and cont["contested_margin"] > 0.05 and cont["control_is_capable"],
          "reapplied the support rule to the sealed margins")

    state = _sealed("SUBSTRATE_STATE.json")
    card = _sealed("SUBSTRATE_PROGRESS_SCORECARD.json")
    for category, row in card["categories"].items():
        if not row["items"]:
            continue
        earned = [i for i in row["items"]
                  if (state["items"][i].get("result") or {}).get("scientific") is True]
        check(f"scorecard_evidence_{category}", row["evidence_pct"],
              round(100 * len(earned) / len(row["items"])),
              "recounted classified results in the sealed state file")

    seals = {name: _seal_intact(_sealed(name)) for name in sorted(
        p.name for p in io.PROOF.glob("SUBSTRATE_*.json"))}
    disagreements = sorted(k for k, v in checks.items() if not v["agrees"])
    broken_seals = sorted(k for k, v in seals.items() if not v)
    return {
        "schema": "substrate-independent-verification/v1",
        "role": ("recomputation from sealed bytes by a second route, without importing the modules that "
                 "produced them"),
        "checks": checks,
        "seal_integrity": seals,
        "disagreements": disagreements,
        "broken_seals": broken_seals,
        "all_pass": not disagreements and not broken_seals,
    }


# ---------------------------------------------------------------- mutation attacks

# each entry breaks one guard and names the test that must notice
MUTATIONS = (
    ("workspace_ignores_writer_sets", "mop.cognition.workspace",
     "Workspace._permitted = lambda self, allowed, who: True",
     "tests/cognition/test_workspace.py::test_a_write_outside_the_declared_writers_is_refused"),
    ("workspace_drops_provenance_requirement", "mop.cognition.workspace",
     "REGIONS = tuple(type(r)(**{**vars(r), 'provenance': False}) for r in REGIONS)\n"
     "BY_NAME = {r.name: r for r in REGIONS}",
     "tests/cognition/test_workspace.py::test_provenance_and_confidence_are_mandatory_where_declared"),
    ("untyped_control_becomes_typed", "mop.cognition.workspace",
     "UntypedWorkspace.typed = True",
     "tests/cognition/test_workspace.py::test_the_untyped_control_removes_typing_and_nothing_else"),
    ("learned_selector_opens_freely", "mop.cognition.perspectives",
     "select = lambda strategy, catalog, **kw: catalog[:2]",
     "tests/cognition/test_perspectives.py::test_learned_selector_stays_closed_without_headroom"),
    ("arbitration_discards_the_minority", "mop.cognition.perspectives",
     "_arbitrate = arbitrate\n"
     "arbitrate = lambda outputs, **kw: {**_arbitrate(outputs, **kw), 'alternative_hypotheses': [],\n"
     "                                   'minority_preserved': 0}",
     "tests/cognition/test_perspectives.py::test_minority_hypothesis_survives_arbitration"),
    ("generated_episodes_promote_on_a_flag", "mop.cognition.memory",
     "EpisodicMemory.promote_to_training = lambda self, eid: self.store[eid]",
     "tests/cognition/test_memory.py::test_generated_episode_cannot_be_promoted_without_verification"),
    ("procedures_transfer_on_their_own_episodes", "mop.cognition.memory",
     "ProceduralMemory.transfer_test = lambda self, pid, evaluated_on, score, baseline: "
     "self.store[pid].__setattr__('transfer', {'improves': True, 'held_out': True, "
     "'evaluated_on': evaluated_on, 'score': score, 'baseline': baseline}) or self.store[pid].transfer",
     "tests/cognition/test_memory.py::test_procedure_requires_transfer_beyond_source_episodes"),
    ("hygiene_deletes_audit_records", "mop.cognition.memory",
     "hygiene = lambda records, *, audit_required, requests: {"
     "'applied': [{'record': r, 'action': a} for r, a in requests], 'refused': [],"
     "'audit_preserved': True, 'nothing_audit_required_was_deleted': True}",
     "tests/cognition/test_memory.py::test_hygiene_never_destroys_audit_required_records"),
    ("intervening_becomes_conditioning", "mop.cognition.world",
     "WorldModel.intervene = lambda self, state, do: self.predict({**state, **do})",
     "tests/cognition/test_world.py::test_intervening_is_not_the_same_operation_as_observing"),
    ("self_facts_need_no_source", "mop.cognition.selfmodel",
     "SelfFact.violations = lambda self: []",
     "tests/cognition/test_selfmodel.py::test_a_self_fact_without_a_source_is_refused"),
    ("fast_adaptation_touches_shared_parameters", "mop.cognition.plasticity",
     "LEVELS = tuple(type(l)(**{**vars(l), 'touches_shared_parameters': False, 'speed': 'fast'}) "
     "for l in LEVELS)\nBY_LEVEL = {l.name: l for l in LEVELS}",
     "tests/cognition/test_plasticity.py::test_fast_adaptation_does_not_touch_shared_parameters"),
    ("slow_adaptation_skips_its_evidence_bar", "mop.cognition.plasticity",
     "slow_adapt = lambda adaptation, **kw: {'level': adaptation.level, 'applied': True, "
     "'refusals': [], 'repetitions': 0, 'held_out': {}, 'retention': {}, 'rollback': ''}",
     "tests/cognition/test_plasticity.py::test_slow_adaptation_requires_repeated_evidence_and_rollback"),
    # the first version of this mutation appended the forbidden list to the permitted one, which changed
    # nothing because the forbidden branch runs first. An equivalent mutant proves nothing, so this one
    # removes the branch instead.
    ("reorganize_drops_the_forbidden_branch", "mop.cognition.plasticity",
     "reorganize = lambda change, *, measured, cost, sesoi=0.05: "
     "{'change': change, 'permitted': True, 'applied': True, 'reason': ''}",
     "tests/cognition/test_plasticity.py::test_forbidden_reorganizations_are_refused"),
    ("forbidden_reorganization_list_is_emptied", "mop.cognition.safety",
     "FORBIDDEN_REORGANIZATIONS = ()",
     "tests/cognition/test_plasticity.py::test_forbidden_reorganizations_are_refused"),
    ("protected_surfaces_become_removable", "mop.cognition.safety",
     "PROTECTED_SURFACES = ()",
     "tests/cognition/test_safety.py::test_protected_surfaces_cannot_be_removed_by_adaptation"),
    ("forbidden_claims_pass", "mop.cognition.safety",
     "check_claim = lambda text: []",
     "tests/cognition/test_safety.py::test_forbidden_claim_vocabulary_is_refused"),
    ("integrity_stops_failing_closed", "mop.cognition.safety",
     "integrity_report = lambda observations: {'schema': 'x', 'surfaces': {}, 'failed_surfaces': [], "
     "'all_pass': True, 'fails_closed': False}",
     "tests/cognition/test_safety.py::test_integrity_violation_is_detected_and_fails_closed"),
    ("latency_counts_as_thinking", "mop.cognition.batteries",
     "NOT_EVIDENCE = ()\nALTERNATIVES = ALTERNATIVES + ('latency', 'hidden_activations')",
     "tests/cognition/test_batteries.py::test_thinking_requires_a_declared_alternative_to_beat"),
    ("reflective_reports_answer_without_provenance", "mop.cognition.batteries",
     "_rr = reflective_report\n"
     "reflective_report = lambda e, b: {**_rr(e, b), 'answered': True, 'failed_closed': False}",
     "tests/cognition/test_batteries.py::test_reflective_report_fails_closed_without_provenance"),
    ("evidence_counts_on_presence_alone", "mop.cognition.program",
     "evidence_state = lambda ref: {'reference': ref, 'present': True, 'counts': True, 'reason': ''}",
     "tests/cognition/test_program.py::test_evidence_that_is_stale_or_failing_does_not_count"),
    ("implementation_raises_evidence", "mop.cognition.program",
     "scorecard = lambda st=None: {'categories': {'working_memory': {'items': ['M1'], "
     "'implementation_pct': 100, 'evidence_pct': 100, 'items_with_earned_evidence': ['M1']}}}",
     "tests/cognition/test_program.py::test_evidence_never_rises_from_implementation_alone"),
    ("admission_stops_requiring_completeness", "mop.cognition.admission",
     "completeness = lambda prereg: []",
     "tests/cognition/test_admission.py::test_substrate_experiment_cannot_reach_principal_unproven"),
    ("a_structurally_guaranteed_effect_is_licensed", "mop.cognition.experiments",
     "_d = sx1_decision\n"
     "sx1_decision = lambda: {**_d(), 'licensed': True, 'classification': 'licensed'}",
     "tests/cognition/test_experiments.py::"
     "test_sx1_is_refused_because_its_effect_is_true_by_construction"),
    ("a_refusal_is_filed_as_a_null", "mop.cognition.deliverables",
     "methodological_refusals = lambda: []",
     "tests/cognition/test_experiments.py::test_a_refusal_is_not_recorded_as_a_null"),
    ("an_underpowered_design_reaches_the_test_split", "mop.cognition.experiments",
     "_e = sx1b_evidence\n"
     "sx1b_evidence = lambda: {**_e(), 'power': {'mde': 0.001, 'n_test_units': 7}}",
     "tests/cognition/test_experiments.py::"
     "test_sx1b_is_refused_on_power_and_never_touches_the_test_split"),
    ("the_ceiling_below_sesoi_is_blamed_on_the_unit_count", "mop.cognition.experiments",
     "_d = _sx1b_diagnosis\n"
     "_sx1b_diagnosis = lambda ev: {**_d(ev), 'more_units_would_help': True, "
     "'decisive_number': 'mde'}",
     "tests/cognition/test_experiments.py::"
     "test_the_sx1b_diagnosis_names_the_number_that_actually_decides"),
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
    return {"mutation": name, "module": module, "guard": node, "rejected": rejected,
            "survivor": not rejected,
            "detail": "" if rejected else (r.stdout or r.stderr).strip().splitlines()[-1:]}


def mutation_report(only: list[str] | None = None) -> dict:
    rows = [run_mutation(*m) for m in MUTATIONS if only is None or m[0] in only]
    survivors = [r["mutation"] for r in rows if r["survivor"]]
    return {
        "schema": "substrate-mutation-report/v1",
        "rule": ("every mutation must be rejected by the guard named beside it. A survivor is a guard "
                 "that does not work and is reported as a survivor, never folded into a pass rate"),
        "mutations": rows,
        "total": len(rows),
        "rejected": len(rows) - len(survivors),
        "survivors": survivors,
        "all_rejected": not survivors,
    }


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "all"
    if command in ("recompute", "all"):
        doc = recompute()
        io.seal("SUBSTRATE_INDEPENDENT_VERIFICATION.json", doc)
        print(f"substrate recomputation: {len(doc['checks'])} checks, "
              f"{len(doc['disagreements'])} disagreements, {len(doc['broken_seals'])} broken seals",
              flush=True)
    if command in ("mutate", "all"):
        doc = mutation_report()
        io.seal("SUBSTRATE_MUTATION_REPORT.json", doc)
        print(f"substrate mutations: {doc['rejected']}/{doc['total']} rejected, "
              f"survivors {doc['survivors']}", flush=True)
    if command not in ("recompute", "mutate", "all"):
        raise ValueError(argv)


if __name__ == "__main__":
    main()
