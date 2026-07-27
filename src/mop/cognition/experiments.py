"""Substrate experiments, and the first one the gate refuses.

SX1 is the obvious first experiment: does a typed workspace beat an untyped one of matched capacity. It is
also the first place a Substrate result could be manufactured, so it is worth being precise about why it
cannot run as first designed.

On a synthetic bed the comparison has a closed form. Typing refuses writes from an undeclared component, so
the typed arm reports the declared writer's value and the untyped arm reports whatever wrote last. Which
arm wins follows from the two reliabilities the bed generator was given. Nothing is measured; the answer
was set when the bed was written. The historical defect ledger already carries this failure under its own
name, analytic values reported as measured, and the causal graph validator refuses a claim that is broader
than any measured relation.

So SX1 is preregistered honestly, with the treatment to outcome edge labelled as the structural guarantee
it is, and the gate refuses it. That refusal is a methodological result and not a scientific null, and this
module records the difference.

SX1b is the version that could run: the same question on a bed where the undeclared writer's reliability is
measured from held out data rather than chosen by the generator, which turns a closed form into a real
question about whether a write restriction fitted on training units survives on units it has never seen.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys

from mop.cognition import admission as A
from mop.cognition import io
from mop.method import contracts as C
from mop.method import gate
from mop.method import graph as G

# ---------------------------------------------------------------- SX1, as honestly declared

SX1_GRAPH = {
    "nodes": [
        {"id": "typing", "type": "mechanism", "label": "region write typing",
         "implementation": "mop.cognition.workspace.Workspace._permitted"},
        {"id": "restrict", "type": "intervention", "label": "refuse an undeclared write",
         "implementation": "mop.cognition.workspace.Workspace.write"},
        {"id": "typed", "type": "treatment", "label": "typed workspace",
         "implementation": "mop.cognition.workspace.Workspace"},
        {"id": "untyped", "type": "control", "label": "untyped workspace of matched capacity",
         "removes": "reader and writer sets",
         "implementation": "mop.cognition.workspace.UntypedWorkspace"},
        {"id": "reliabilities", "type": "hidden_information", "label": "the generator's two accuracies",
         "time": "decision_time", "declared": True},
        {"id": "unit", "type": "independent_unit", "label": "a held out source group"},
        {"id": "answer", "type": "primary_outcome", "label": "task accuracy"},
        {"id": "claim", "type": "claim", "label": "typing improves cross perspective accuracy",
         "requires": ["answer"]},
    ],
    "edges": [
        {"src": "typing", "dst": "restrict", "type": "implemented_causal_path"},
        {"src": "restrict", "dst": "typed", "type": "implemented_causal_path"},
        # the honest label. On a synthetic bed this edge is fixed by the generator, not measured.
        {"src": "typed", "dst": "answer", "type": "structural_guarantee"},
        {"src": "untyped", "dst": "answer", "type": "implemented_causal_path"},
        {"src": "reliabilities", "dst": "answer", "type": "analytic_relation"},
        {"src": "unit", "dst": "answer", "type": "analytic_relation"},
    ],
}


def sx1() -> gate.Preregistration:
    """The full preregistration. Every mandatory contract is present, so the refusal is about the science."""
    return gate.Preregistration(
        experiment_id="SX1",
        title="does a typed workspace beat an untyped one of matched capacity on a synthetic bed",
        mechanism_activity={"active": True, "classification": "active",
                            "failed": [],
                            "evidence": "an undeclared write is refused and the region is unchanged"},
        contracts=[
            C.ExperimentQuestion(name="q", declared={
                "question": "does refusing undeclared writes raise cross perspective accuracy",
                "hypotheses": ["H_typed_workspace", "H_capacity_only"],
                "predictions": {"H_typed_workspace": {"typed_beats_untyped": True},
                                "H_capacity_only": {"typed_beats_untyped": False}}}),
            C.CausalModel(name="g", declared={"graph": SX1_GRAPH}),
            C.MeasurementModel(name="m", declared={"outcomes": {
                "accuracy": {"estimator": "mean over held out source groups",
                             "unit": "held out source group"}}}),
            C.InstrumentContract(name="i", evidence={"calibration": {
                "recovers_a_planted_corruption": True, "reports_none_when_none_planted": True,
                "all_pass": True}}),
            C.ArmContract(name="typed", evidence={"distinctness": {"untyped": "distinct"}}),
            C.ControlContract(name="untyped", evidence={"semantic": {
                "typing_absent": True, "capacity_matched": True, "cost_accounting_retained": True,
                "all_pass": True}}),
            C.DatasetContract(name="bed", evidence={"bed_validity": {
                "classification": "valid_principal_bed"}}),
            C.IndependentUnitContract(name="units", evidence={"units": {
                "group_disjoint": True, "n_units": 8, "test_touched": False}}),
            C.BaselineContract(name="base", declared={"identity": "untyped workspace"},
                               evidence={"convergence": {"converged": True, "resource_matched": True,
                                                         "identity": "untyped workspace"}}),
            C.OracleContract(name="oracle", evidence={"headroom": {
                "n_seeds": 8, "residual_lower_95_cb": 0.22}}),
            C.PowerContract(name="power",
                            declared={"sesoi": 0.05, "seeds": 8, "futility": 0.01, "harm": -0.02},
                            evidence={"power": {"mde": 0.03}}),
        ])


def sx1_decision() -> dict:
    """Admit SX1 and record why it is refused, in the vocabulary the method kernel already uses."""
    prereg = sx1()
    report = A.admit(prereg, "principal")
    graph_violations = G.validate(SX1_GRAPH)
    return {
        "schema": "substrate-experiment-decision/v1",
        "experiment_id": "SX1",
        "title": prereg.title,
        # named explicitly so the hypothesis graph can attach the refusal without guessing from prose
        "hypotheses": sorted({h for c in prereg.contracts if c.kind == "ExperimentQuestion"
                              for h in c.declared.get("hypotheses", [])}),
        "admission": report,
        "causal_graph_violations": graph_violations,
        "licensed": report["licensed"],
        "classification": "methodological_refusal" if not report["licensed"] else "licensed",
        "why": ("on a synthetic bed the comparison has a closed form. Typing refuses writes from an "
                "undeclared component, so which arm wins follows from the two reliabilities the generator "
                "was given. The treatment to outcome edge is a structural guarantee, no measured relation "
                "reaches the outcome, and the claim is therefore broader than the measured path"),
        "not_a_scientific_null": ("a methodological failure is not a scientific null. H_typed_workspace is "
                                  "untested, not refuted, and stays open in the hypothesis graph"),
        "successor": "SX1b",
        "activation": False,
    }


# ---------------------------------------------------------------- SX1b, the version that could run

SX1B_DESIGN = {
    "schema": "substrate-experiment-design/v1",
    "experiment_id": "SX1b",
    "question": ("does a write restriction fitted on training units still help on source groups it has "
                 "never seen"),
    "why_it_is_not_a_closed_form": ("the reliability of each writer is measured on training units rather "
                                    "than set by a generator, so the restriction can be fitted to a "
                                    "reliability ordering that does not hold on held out groups. The "
                                    "answer depends on how stable that ordering is across units, which "
                                    "no closed form supplies"),
    "arms": {
        "typed_fitted": "writers restricted to the perspective that was more reliable on training units",
        "typed_oracle": "writers restricted using the held out ordering, an upper bound",
        "untyped": "the matched capacity control, no restriction",
        "typed_wrong": "restricted to the less reliable training writer, the harm direction",
    },
    "bed_requirements": ["a real corpus already under data custody",
                         "source groups that make train and test disjoint",
                         "at least two perspectives whose reliabilities differ by group",
                         "a measured, not declared, reliability ordering"],
    "candidate_beds": ["harth_stream", "pamap2_stream"],
    "primary_outcome": "accuracy per held out source group",
    "sesoi": 0.05,
    "predictions": {
        "H_typed_workspace": {"typed_fitted_beats_untyped": True},
        "H_reliability_instability": {"typed_fitted_beats_untyped": False,
                                      "typed_oracle_beats_untyped": True},
        "H_capacity_only": {"typed_fitted_beats_untyped": False, "typed_oracle_beats_untyped": False},
    },
    "blocked_on": ("principal compute. The temporal core factorial holds every worker slot, and this "
                   "design is queued behind it rather than run alongside it"),
    "activation": False,
}


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    decision = sx1_decision()
    io.run_json("SX1_decision.json", decision, "experiments")
    io.run_json("SX1b_design.json", SX1B_DESIGN, "experiments")
    print(json.dumps({"SX1_licensed": decision["licensed"],
                      "classification": decision["classification"],
                      "graph_violations": decision["causal_graph_violations"],
                      "successor": decision["successor"]}, indent=2))


if __name__ == "__main__":
    main()
