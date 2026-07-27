"""No Substrate experiment reaches principal compute unproven.

The load bearing case is the third one. The experiment validity kernel walks its stage sequence and a stage
with no contracts has no violations, so an experiment that simply never declares oracle headroom passes the
kernel. Substrate refuses it. If that ever stops being true, this test fails.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from mop.cognition import admission as A
from mop.method import contracts as C
from mop.method import gate

GRAPH = {
    "nodes": [
        {"id": "M", "type": "mechanism", "label": "typed workspace",
         "implementation": "mop.cognition.workspace"},
        {"id": "I", "type": "intervention", "label": "enable region typing",
         "implementation": "mop.cognition.workspace.Workspace"},
        {"id": "T", "type": "treatment", "label": "typed regions",
         "implementation": "mop.cognition.workspace.Workspace"},
        {"id": "K", "type": "control", "label": "untyped shared state", "removes": "region typing",
         "implementation": "mop.cognition.workspace.UntypedWorkspace"},
        {"id": "U", "type": "independent_unit", "label": "held out unit"},
        {"id": "O", "type": "primary_outcome", "label": "task accuracy"},
        {"id": "CL", "type": "claim", "label": "typing helps", "requires": ["O"]},
    ],
    "edges": [
        {"src": "M", "dst": "I", "type": "implemented_causal_path"},
        {"src": "I", "dst": "T", "type": "implemented_causal_path"},
        {"src": "T", "dst": "O", "type": "implemented_causal_path"},
        {"src": "K", "dst": "O", "type": "implemented_causal_path"},
        {"src": "U", "dst": "O", "type": "measured_relation"},
    ],
}


def _contracts() -> list[C.Contract]:
    return [
        C.ExperimentQuestion(name="q", declared={
            "question": "does region typing beat one untyped state of matched capacity",
            "hypotheses": ["H_typed_workspace", "H_capacity_only"],
            "predictions": {"H_typed_workspace": {"typed_beats_untyped": True},
                            "H_capacity_only": {"typed_beats_untyped": False}}}),
        C.CausalModel(name="g", declared={"graph": GRAPH}),
        C.MeasurementModel(name="m", declared={"outcomes": {
            "accuracy": {"estimator": "mean over held out units", "unit": "held out unit"}}}),
        C.InstrumentContract(name="i", evidence={"calibration": {"recovers_known_effect": True,
                                                                 "rejects_absent_effect": True,
                                                                 "all_pass": True}}),
        C.ArmContract(name="typed", evidence={"distinctness": {"untyped": "distinct"}}),
        C.ControlContract(name="untyped", evidence={"semantic": {"typing_absent": True,
                                                                 "capacity_matched": True,
                                                                 "all_pass": True}}),
        C.DatasetContract(name="bed", evidence={"bed_validity": {
            "classification": "valid_principal_bed"}}),
        C.IndependentUnitContract(name="units", evidence={"units": {
            "group_disjoint": True, "n_units": 8, "test_touched": False}}),
        C.BaselineContract(name="base", declared={"identity": "untyped shared state"},
                           evidence={"convergence": {"converged": True, "resource_matched": True,
                                                     "identity": "untyped shared state"}}),
        C.OracleContract(name="oracle", evidence={"headroom": {
            "n_seeds": 8, "residual_lower_95_cb": 0.041}}),
        C.PowerContract(name="power",
                        declared={"sesoi": 0.05, "seeds": 8, "futility": 0.01, "harm": -0.02},
                        evidence={"power": {"mde": 0.031}}),
    ]


def _prereg(contracts=None, activity=None) -> gate.Preregistration:
    return gate.Preregistration(
        experiment_id="SX1", title="typed workspace against one untyped state",
        contracts=_contracts() if contracts is None else contracts,
        mechanism_activity={"active": True, "classification": "active", "failed": []}
        if activity is None else activity)


def test_a_complete_preregistration_is_licensed():
    report = A.admit(_prereg())
    assert report["licensed"] is True, report["blocking_violations"]
    assert report["completeness_violations"] == []
    assert report["substrate_admission"] == "licensed"


def test_substrate_experiment_cannot_reach_principal_unproven():
    # 1. a missing arm contract is caught by the kernel itself
    without_arms = [c for c in _contracts() if c.kind != "ArmContract"]
    report = A.admit(_prereg(without_arms))
    assert report["licensed"] is False
    assert report["blocked_at"] == "arm_distinctness"

    # 2. a missing oracle contract is invisible to the kernel because an empty stage has no violations
    without_oracle = [c for c in _contracts() if c.kind != "OracleContract"]
    kernel_only = _prereg(without_oracle).admit("principal")
    assert kernel_only["licensed"] is True, "the kernel changed; this test guards the gap it leaves"

    # 3. Substrate refuses it anyway
    report = A.admit(_prereg(without_oracle))
    assert report["licensed"] is False
    assert any("no OracleContract declared" in v for v in report["blocking_violations"])

    # 4. an unmeasured mechanism is refused
    report = A.admit(_prereg(activity=None if False else None))
    assert report["licensed"] is True  # the fixture supplies activity by default
    report = A.admit(gate.Preregistration(experiment_id="SX2", title="t", contracts=_contracts()))
    assert report["licensed"] is False
    assert any("mechanism activity never measured" in v for v in report["blocking_violations"])


def test_an_inactive_mechanism_blocks_before_principal():
    report = A.admit(_prereg(activity={"active": False, "classification": "inactive_mechanism",
                                       "failed": ["no measurable causal effect"]}))
    assert report["licensed"] is False
    assert report["blocked_at"] == "control_semantics"


def test_requirements_authority_lists_every_mandatory_kind():
    doc = A.requirements_authority()
    assert set(doc["mandatory_contract_kinds"]) <= set(C.CONTRACT_TYPES)
    assert len(doc["mandatory_contract_kinds"]) == 11
    assert "principal" not in doc["pre_principal_stages"]


@pytest.mark.parametrize("dropped", [c.kind for c in _contracts()])
def test_dropping_any_mandatory_contract_refuses_admission(dropped):
    kept = [c for c in _contracts() if c.kind != dropped]
    assert A.admit(_prereg(kept))["licensed"] is False
