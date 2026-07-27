"""Unit coverage of the validity kernel.

The defect regression file proves the kernel catches the failures that already happened. This file proves
the rest of the kernel does what it says on the paths those defects never touched, including the paths that
must accept a good experiment. A gate that rejects everything is as useless as one that accepts everything.

House style: no dashes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mop.method import (
    arms,
    baseline,
    bed,
    contracts,
    controls,
    defects,
    gate,
    graph,
    hypothesis,
    io,
    mechanism,
    power,
    report,
    voi,
)

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------- contracts


def test_quantity_helpers_and_kinds():
    assert contracts.measured(1, "a.json#/x").kind == "measured"
    assert contracts.analytic(1, "closed form").kind == "analytic"
    assert contracts.structural(0, "partition").kind == "structurally_guaranteed"
    assert contracts.Quantity(1, "invented", "s").violations("q")
    assert contracts.measured(1, "s").as_dict()["value"] == 1


def test_contract_base_requires_declared_evidence():
    c = contracts.Contract(name="x", declared={"required_evidence": ["a"]})
    assert c.violations() and not c.passed
    c.evidence["a"] = 1
    assert c.passed
    assert c.as_dict()["kind"] == "Contract"


def test_experiment_question_needs_discriminating_predictions():
    bad = contracts.ExperimentQuestion(name="q", declared={
        "question": "x", "hypotheses": ["a", "b"], "predictions": {"a": "same", "b": "same"}})
    assert any("no discrimination" in v for v in bad.violations())
    missing = contracts.ExperimentQuestion(name="q", declared={
        "question": "x", "hypotheses": ["a", "b"], "predictions": {"a": "one"}})
    assert any("makes no prediction" in v for v in missing.violations())
    good = contracts.ExperimentQuestion(name="q", declared={
        "question": "x", "hypotheses": ["a", "b"], "predictions": {"a": "one", "b": "two"}})
    assert good.passed


def test_measurement_model_requires_estimator_and_unit():
    assert contracts.MeasurementModel(name="m", declared={}).violations()
    assert contracts.MeasurementModel(name="m", declared={"outcomes": {"acc": {"estimator": "x"}}}).violations()
    assert contracts.MeasurementModel(
        name="m", declared={"outcomes": {"acc": {"estimator": "x", "unit": "subject"}}}).passed


def test_instrument_contract_needs_calibration():
    assert contracts.InstrumentContract(name="i").violations()
    assert contracts.InstrumentContract(name="i", evidence={"calibration": {"a": False, "all_pass": False}}).violations()
    assert contracts.InstrumentContract(name="i", evidence={"calibration": {"a": True, "all_pass": True}}).passed


def test_arm_and_control_contracts_need_evidence():
    assert contracts.ArmContract(name="a").violations()
    assert contracts.ArmContract(name="a", evidence={"distinctness": {"b": "distinct"}}).passed
    assert contracts.ControlContract(name="c").violations()
    assert contracts.ControlContract(name="c", evidence={"semantic": {"x": True, "all_pass": True}}).passed


def test_baseline_contract_checks_identity_and_resources():
    ev = {"convergence": {"converged": True, "identity": "lstm", "resource_matched": True}}
    assert contracts.BaselineContract(name="b", declared={"identity": "lstm"}, evidence=ev).passed
    assert contracts.BaselineContract(name="b", declared={"identity": "gru"}, evidence=ev).violations()
    ev2 = {"convergence": {"converged": False, "reason": "still rising", "identity": "lstm"}}
    assert contracts.BaselineContract(name="b", evidence=ev2).violations()
    ev3 = {"convergence": {"converged": True, "identity": "lstm", "resource_matched": False}}
    assert any("resource" in v for v in contracts.BaselineContract(name="b", evidence=ev3).violations())
    assert contracts.BaselineContract(name="b").violations()


def test_dataset_and_unit_contracts():
    assert contracts.DatasetContract(name="d").violations()
    assert contracts.DatasetContract(name="d", evidence={"bed_validity": {"classification": "invalid_no_headroom"}}).violations()
    assert contracts.DatasetContract(name="d", evidence={"bed_validity": {"classification": "valid_secondary_bed"}}).passed
    assert contracts.IndependentUnitContract(name="u").violations()
    ok = {"units": {"group_disjoint": True, "n_units": 5, "test_touched": False}}
    assert contracts.IndependentUnitContract(name="u", evidence=ok).passed
    bad = {"units": {"group_disjoint": True, "n_units": 1, "test_touched": True}}
    assert len(contracts.IndependentUnitContract(name="u", evidence=bad).violations()) == 2


def test_execution_verification_mutation_result_claim_contracts():
    assert contracts.ExecutionContract(name="e").violations()
    assert contracts.ExecutionContract(name="e", evidence={"execution": {"undeclared_changes": ["w"], "seeds": [0]}}).violations()
    assert contracts.ExecutionContract(name="e", evidence={"execution": {"undeclared_changes": [], "seeds": [0]}}).passed
    assert contracts.VerificationContract(name="v").violations()
    assert contracts.VerificationContract(name="v", evidence={"roles": {"B": True, "C": False}}).violations()
    assert contracts.VerificationContract(name="v", evidence={"roles": {"B": True, "C": True}}).passed
    assert contracts.MutationContract(name="m").violations()
    assert contracts.MutationContract(name="m", evidence={"mutations": {"a": True}}).passed
    r = contracts.ResultContract(name="r", declared={"classification": "null"},
                                 evidence={"quantities": {"acc": contracts.measured(0.5, "s")}})
    assert r.passed
    assert contracts.ResultContract(name="r", declared={}).violations()
    c = contracts.ClaimContract(name="c", declared={"claims": [{"text": "t", "requires": ["a", "b"]}]},
                                evidence={"measured_paths": ["a"]})
    assert any("unmeasured paths" in v for v in c.violations())
    assert contracts.CONTRACT_TYPES["ArmContract"] is contracts.ArmContract


def test_oracle_and_power_contract_paths():
    assert contracts.OracleContract(name="o").violations()
    assert contracts.OracleContract(name="o", evidence={"headroom": {"n_seeds": 5}}).violations()
    p = contracts.PowerContract(name="p", declared={"sesoi": None, "seeds": 5, "futility": 0, "harm": 0})
    assert any("sesoi not preregistered" in v for v in p.violations())


def test_causal_model_contract_delegates_to_the_validator():
    assert contracts.CausalModel(name="c").violations()
    assert contracts.CausalModel(name="c", declared={"graph": {"nodes": [], "edges": []}}).violations()


# ---------------------------------------------------------------- graph


def _good_graph():
    return {
        "nodes": [
            {"id": "m", "type": "mechanism", "implementation": "pkg.m"},
            {"id": "i", "type": "intervention", "implementation": "pkg.i"},
            {"id": "rep", "type": "mediator"},
            {"id": "o", "type": "primary_outcome"},
            {"id": "v", "type": "verdict", "requires": ["o"]},
        ],
        "edges": [
            {"src": "m", "dst": "i", "type": "implemented_causal_path"},
            {"src": "i", "dst": "rep", "type": "implemented_causal_path"},
            {"src": "rep", "dst": "o", "type": "measured_relation"},
        ],
    }


def test_a_well_formed_graph_is_admitted():
    assert graph.validate(_good_graph()) == []


def test_unknown_node_and_edge_types_are_rejected():
    g = _good_graph()
    g["nodes"].append({"id": "z", "type": "nonsense"})
    g["edges"].append({"src": "z", "dst": "missing", "type": "telepathy"})
    v = graph.validate(g)
    assert any("unknown type" in x for x in v) and any("unknown node" in x for x in v)


def test_outcome_without_a_measured_relation_is_rejected():
    g = _good_graph()
    g["edges"] = [e for e in g["edges"] if e["type"] != "measured_relation"]
    assert any("no measured relation" in x for x in graph.validate(g))


def test_control_without_a_declared_removal_is_rejected():
    g = _good_graph()
    g["nodes"].append({"id": "c", "type": "control", "implementation": "pkg.c"})
    g["edges"].append({"src": "c", "dst": "rep", "type": "implemented_causal_path"})
    assert any("does not declare what it removes" in x for x in graph.validate(g))


def test_control_with_failing_semantics_is_rejected_when_evidence_is_supplied():
    g = _good_graph()
    g["nodes"].append({"id": "c", "type": "control", "implementation": "pkg.c", "removes": "order"})
    g["edges"].append({"src": "c", "dst": "rep", "type": "implemented_causal_path"})
    v = graph.validate(g, evidence={"c": {"all_pass": False, "permutation": False}})
    assert any("still retains" in x for x in v)


def test_realized_forbidden_path_is_rejected():
    g = _good_graph()
    g["nodes"].append({"id": "f", "type": "hidden_information", "time": "future", "declared": True})
    g["edges"].append({"src": "f", "dst": "i", "type": "forbidden_information_path", "realized": True})
    assert any("is realized" in x for x in graph.validate(g))


def test_verdict_requiring_nothing_is_rejected_and_summary_counts():
    g = _good_graph()
    g["nodes"].append({"id": "v2", "type": "verdict"})
    assert any("requires nothing" in x for x in graph.validate(g))
    s = graph.summarize(g)
    assert s["nodes"] == len(g["nodes"]) and s["by_node_type"]["mechanism"] == 1


def test_schema_lists_every_rejection_rule():
    assert len(graph.SCHEMA["rejections"]) == 8


# ---------------------------------------------------------------- controls


def test_no_replay_and_random_and_shuffled_and_wrong_time_and_frozen():
    assert controls.no_replay({"replayed_items": 0, "buffer_reads": 0, "cached_batches": 0, "buffer_size": 0})["all_pass"]
    assert not controls.no_replay({"replayed_items": 3, "buffer_reads": 0, "cached_batches": 0, "buffer_size": 0})["all_pass"]
    real = {"intervention_rate": 0.5, "updates": 100, "samples_seen": 640}
    ctrl = {"intervention_rate": 0.52, "updates": 100, "samples_seen": 640, "seed": 0, "signal_target_corr": 0.01}
    assert controls.random_control(real, ctrl)["all_pass"]
    ctrl_bad = dict(ctrl, intervention_rate=0.9, seed=None)
    assert not controls.random_control(real, ctrl_bad)["all_pass"]
    before, after = [1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 3.0, 2.0]
    assert controls.shuffled_control(before, after, before, before)["marginals_preserved"]
    assert not controls.shuffled_control(before, [9.0, 9.0, 9.0, 9.0], before, before)["marginals_preserved"]
    r = controls.wrong_time_control({"intervention_kind": "freeze", "interventions": 4, "times": [1, 2]},
                                    {"intervention_kind": "freeze", "interventions": 4, "times": [7, 8]},
                                    [7, 8])
    assert r["all_pass"]
    ex = {"changed_params": ["head.weight"]}
    assert controls.frozen_control(ex, ["core"], {"core": ["core.w"], "head": ["head.weight"]})["all_pass"]
    assert not controls.frozen_control(ex, ["head"], {"head": ["head.weight"]})["all_pass"]
    assert not controls.frozen_control(ex, ["absent"], {"core": ["core.w"]})["all_pass"]


def test_order_free_declared_exception_is_honoured():
    import torch.nn as nn

    class Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv1d(2, 2, 3, padding=1)

        def forward(self, x):
            return self.c(x.transpose(1, 2)).mean(-1)

    m = Conv().eval()
    r = controls.order_free(lambda t: m(t), torch.randn(2, 8, 2), module=m,
                            declared_exceptions=("no_temporal_convolution",))
    assert "no_temporal_convolution" not in r


def test_structural_scan_finds_position_parameters():
    import torch.nn as nn

    class Pos(nn.Module):
        def __init__(self):
            super().__init__()
            self.position_embed = nn.Parameter(torch.zeros(4))
            self.register_buffer("hidden_state", torch.zeros(4))

        def forward(self, x):
            return x

    s = controls.structural_temporal_scan(Pos())
    assert not s["no_position_encoding"] and not s["no_carried_state_buffer"]


def test_controls_registry_is_complete():
    assert set(controls.REGISTRY) >= {"order_free", "no_replay", "random", "shuffled", "wrong_time", "frozen"}


# ---------------------------------------------------------------- arms


def _r(name, tag):
    return arms.record(name, source=lambda: tag, config={"c": tag}, call_graph=[tag], state_transitions=[tag],
                       param_delta={"p": tag}, memory={"m": tag}, resources={"r": tag}, outputs=[tag])


def test_source_sha_falls_back_for_uninspectable_objects():
    assert arms.source_sha(object())


def test_compare_reports_the_differing_fields():
    c = arms.compare(_r("a", "x"), _r("b", "y"))
    assert not c["aliased"] and "config_sha" in c["differing_fields"]


def test_branch_activity_accepts_an_executed_branch():
    assert arms.branch_activity([_r("a", "x")], {"a": "x"})["all_active"]


def test_mutations_need_two_arms():
    with pytest.raises(ValueError):
        arms.mutations([_r("a", "x")])


# ---------------------------------------------------------------- bed


def test_context_boundary_requires_both_signatures():
    r = bed.context_boundary(0.5, 0.7, 0.65, 0.66)
    assert r["classification"] == "context_boundary_crossed"
    assert bed.context_boundary(0.69, 0.68, 0.73, 0.73)["classification"] == "invalid_no_context_boundary"
    assert bed.context_boundary(0.5, 0.7, 0.65, 0.75)["classification"] == "invalid_no_context_boundary"


def test_leakage_audit_and_headroom_helpers():
    assert bed.leakage_audit([1], [2], "train_only")["clean"]
    assert not bed.leakage_audit([1], [1], "train_and_test")["clean"]
    assert bed.order_necessity(0.9, 0.7) == 0.2
    assert bed.residual_headroom(0.9, 0.8) == 0.1


def test_bed_without_intervention_and_without_headroom():
    m = {"construct_valid": True, "units": {"group_disjoint": True, "test_touched": False, "n_units": 10},
         "leakage": {"clean": True}, "oracle_headroom": 0.2, "residual_headroom_lcb": 0.05,
         "baseline_converged": True, "order_necessity": 0.2, "intervention_possible": False,
         "seed_stability": 0.01}
    assert bed.classify(m)["classification"] == "invalid_no_intervention"
    m["intervention_possible"] = True
    m["baseline_converged"] = False
    assert bed.classify(m)["classification"] == "invalid_unconverged_baseline"
    m["baseline_converged"] = True
    m["seed_stability"] = 0.9
    assert bed.classify(m)["classification"] == "valid_secondary_bed"


# ---------------------------------------------------------------- baseline and power


def test_plateau_needs_enough_points():
    assert not baseline.plateau([0.1, 0.2])["converged"]
    assert baseline.plateau([0.8, 0.8, 0.8, 0.8, 0.8])["converged"]


def test_receipt_records_group_variance_when_supplied():
    r = baseline.receipt("b", identity="i", model="m", parameters=1, updates=1, data_exposure=1, memory=0,
                         compute_seconds=1.0, validation_curve=[0.8] * 5, selected_checkpoint="best",
                         seed_scores=[0.8, 0.82], group_scores=[0.7, 0.9])
    assert r["group_variance"] is not None and r["seed_variance"] > 0


def test_power_helpers():
    assert power.t95(3) == 2.920 and power.t95(50) == 1.729
    assert power.lcb([]) == 0.0 and power.lcb([0.5]) == 0.5
    assert power.mde(0.1, 1) == float("inf")
    pre = power.preregistration(name="p", independent_unit="u", expected_sd=0.01, sesoi=0.05, seeds=8,
                                units=8, max_seeds=8, futility=0.01, harm=0.05)
    assert pre["adequately_powered"]
    assert power.decide([0.1], pre)["verdict"] == "insufficient_power"
    assert power.stage_open("scout", None)["open"] is False
    assert power.stage_open("nonsense", None)["open"] is False
    assert power.stage_open("scout", {"passed": True})["open"] is True


# ---------------------------------------------------------------- report


def test_resolve_handles_lists_and_escapes_and_failures():
    doc = {"a/b": [{"x": 1}]}
    assert report.resolve(doc, "/a~1b/0/x") == 1
    assert report.resolve(doc, "") is doc
    with pytest.raises(report.ReportFieldError):
        report.resolve(doc, "/a~1b/9/x")
    with pytest.raises(report.ReportFieldError):
        report.resolve({"a": 1}, "/a/b")


def test_bind_type_and_missing_artifact_failures():
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "a.json").write_text(json.dumps({"x": "text"}))
        with pytest.raises(report.ReportFieldError):
            report.bind(Path(td), "missing.json", "/x")
        with pytest.raises(report.ReportFieldError):
            report.bind(Path(td), "a.json", "/x", expect=int)
        b = report.bind(Path(td), "a.json", "/x", transform=str.upper)
        assert b["value"] == "TEXT" and b["transformation"] == "upper"


def test_render_reports_wording_failures_too():
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "a.json").write_text(json.dumps({"x": 1}))
        spec = {"q": {"artifact": "a.json", "pointer": "/x"}}
        assert report.render(Path(td), spec, {"p": ("a plain null", "mechanism_null")})["all_resolved"]
        with pytest.raises(report.ReportFieldError):
            report.render(Path(td), spec, {"p": ("it confirms the mechanism", "mechanism_null")})


def test_validate_spec_accepts_a_matching_schema_and_flags_a_missing_pointer():
    assert report.validate_spec({"q": {"pointer": "/a/b"}}, {"a": {"b": 1}})["passes"]
    assert not report.validate_spec({"q": {}}, {"a": 1})["passes"]


def test_verdict_class_mapping():
    assert report.verdict_class("invalid_no_temporal_headroom") == "invalid"
    assert report.verdict_class("mechanism_null") == "null"
    assert report.verdict_class("anything else") == "null"


# ---------------------------------------------------------------- voi and hypothesis


def _cand(cid, **over):
    s = {d: 0.5 for d in voi.DIMENSIONS}
    s["risk_of_repeating_a_closed_premise"] = 0.1
    s.update(over)
    return {"id": cid, "title": cid, "question": "q", "hypotheses_separated": ["H1"], "scores": s}


def test_voi_scores_and_refuses():
    assert voi.score(_cand("a"))["status"] == "eligible"
    assert voi.score(_cand("b", risk_of_repeating_a_closed_premise=0.9))["status"] == "refused_closed_premise"
    assert voi.score(_cand("c", oracle_headroom=0.0))["status"] == "refused_closed_premise"
    with pytest.raises(ValueError):
        voi.score({"id": "d", "title": "d", "question": "q", "scores": {}})


def test_voi_queue_selects_and_reports_coverage():
    q = voi.queue([_cand("a", expected_information_gain=0.9), _cand("b"), _cand("c", oracle_headroom=0.0)])
    assert q["selected"][0] == "a" and "c" in q["refused"]
    assert q["hypotheses_covered_by_selection"] == ["H1"]


def _hyp(hid, dependents=(), requires=(), state="unopened"):
    return {"id": hid, "premise": "p", "predecessor": None, "support": [], "contradictions": [],
            "required_bed": "b", "required_headroom": "h", "strongest_baseline": "s",
            "cheapest_falsifier": "f", "dependent_hypotheses": list(dependents),
            "requires_premise_of": list(requires), "state": state}


def test_hypothesis_validation_catches_missing_fields_and_unknown_links():
    assert hypothesis.validate([{"id": "a"}])
    assert hypothesis.validate([_hyp("a", dependents=["ghost"])])
    assert hypothesis.validate([dict(_hyp("a"), state="imaginary")])
    assert hypothesis.validate([]) == []


def test_a_null_closes_only_dependents_that_require_the_premise():
    nodes = [_hyp("a", dependents=["b", "c"]), _hyp("b", requires=["a"]), _hyp("c")]
    r = hypothesis.propagate(nodes, {"hypothesis": "a", "verdict": "null"})
    assert r["closed_descendants"] == ["b"] and r["independent_descendants_left_open"] == ["c"]
    assert hypothesis.summarize(nodes)["open"] == ["c"]
    with pytest.raises(KeyError):
        hypothesis.propagate(nodes, {"hypothesis": "ghost", "verdict": "null"})


def test_a_supported_result_closes_nothing():
    nodes = [_hyp("a", dependents=["b"]), _hyp("b", requires=["a"])]
    r = hypothesis.propagate(nodes, {"hypothesis": "a", "verdict": "supported"})
    assert r["closed_descendants"] == []


# ---------------------------------------------------------------- gate


def _pre(**over):
    cs = [
        contracts.ArmContract(name="a", evidence={"distinctness": {"b": "distinct"}}),
        contracts.ExperimentQuestion(name="q", declared={"question": "x", "hypotheses": ["a", "b"],
                                                         "predictions": {"a": "one", "b": "two"}}),
        contracts.CausalModel(name="c", declared={"graph": _good_graph()}),
        contracts.MeasurementModel(name="m", declared={"outcomes": {"acc": {"estimator": "e", "unit": "u"}}}),
    ]
    return gate.Preregistration(experiment_id="X", title="t", contracts=cs, **over)


def test_admission_passes_a_sound_preregistration_and_names_the_blocking_stage():
    a = _pre().admit(stage="arm_distinctness")
    assert a["licensed"] and a["blocked_at"] is None
    blocked = gate.Preregistration(experiment_id="X", title="t", contracts=[]).admit(stage="arm_distinctness")
    assert blocked["blocked_at"] == "arm_distinctness"
    inactive = _pre(mechanism_activity={"active": False, "classification": "inactive_instrumentation",
                                        "failed": ["counterfactual_difference"]})
    assert inactive.admit(stage="control_semantics")["blocked_at"] == "control_semantics"


def test_admission_as_dict_and_stage_map():
    d = _pre().as_dict()
    assert d["experiment_id"] == "X" and d["admission"]["requested_stage"] == "principal"
    assert gate.CONTRACT_STAGE["ArmContract"] == "arm_distinctness"
    assert "principal" not in gate.PRE_PRINCIPAL


def test_classify_result_covers_every_gate():
    base = dict(instrument_valid=True, bed_valid=True, mechanism_active=True, baseline_valid=True,
                estimator_sufficient=True, verifier_agrees=True, mutations_rejected=True,
                implementations_agreeing=2)
    assert gate.classify_result(effect={"verdict": "null"}, **{**base, "instrument_valid": False})["classification"] == "invalid_instrument"
    assert gate.classify_result(effect={"verdict": "null"}, **{**base, "estimator_sufficient": False})["classification"] == "insufficient_power"
    assert gate.classify_result(effect={"verdict": "harm"}, **base)["classification"] == "harm"
    assert gate.classify_result(effect={"verdict": "wrong_direction_failure"}, **base)["classification"] == "failure_wrong_direction"
    assert gate.classify_result(effect={"verdict": "null"}, **{**base, "verifier_agrees": False})["classification"] == "scientifically_unresolved"
    assert gate.classify_result(effect={"verdict": "positive"}, **{**base, "mutations_rejected": False})["classification"] == "provisional_positive"
    assert gate.classify_result(effect={"verdict": "positive"}, **{**base, "cost_adjusted_pass": False})["classification"] == "provisional_positive"
    assert gate.classify_result(effect={"verdict": "null"}, **base)["classification"] == "mechanism_null"


# ---------------------------------------------------------------- io and mechanism


def test_io_roundtrip_and_hashing():
    p = io.seal("MOP_KERNEL_SELFTEST.json", {"schema": "test", "value": 1}, subdir="selftest")
    d = io.load("MOP_KERNEL_SELFTEST.json", "selftest")
    assert d["value"] == 1 and io.exists("MOP_KERNEL_SELFTEST.json", "selftest")
    assert io.sha_obj({k: v for k, v in d.items() if k != "sha256"}) == d["sha256"]
    io.seal_md("selftest.md", "# t", subdir="selftest")
    io.run_json("selftest.json", {"a": 1}, "selftest")
    assert (io.RUNS / "selftest" / "selftest.json").is_file()
    p.unlink()
    (io.PROOF / "selftest" / "selftest.md").unlink()
    (io.RUNS / "selftest" / "selftest.json").unlink()
    (io.PROOF / "selftest").rmdir()
    (io.RUNS / "selftest").rmdir()
    assert io.commit()


def test_mechanism_forcing_and_shuffled_paths():
    on = {"intervention_count": 10, "intervention_timing": [1], "affected_samples": 10,
          "affected_parameter_groups": ["g"], "affected_state": [], "affected_memory": [],
          "counterfactual_difference": 0.1, "downstream_path": ["a"], "cost": 1.0}
    off = {k: 0 for k in mechanism.REQUIRED_MEASUREMENTS}
    r = mechanism.activity({"enabled": on, "disabled": off, "forced_active": dict(on, intervention_count=20),
                            "randomized": on, "shuffled": dict(on, counterfactual_difference=0.02)})
    assert r["active"] and r["checks"]["forcing_increases_intervention"]
    weak = mechanism.activity({"enabled": dict(on, counterfactual_difference=0.0), "disabled": off})
    assert not weak["active"] and "counterfactual_difference" in weak["failed"]


def test_defect_followups_and_ledger_ids():
    assert len(defects.required_followups("D1")) == 6
    assert defects.MUTATIONS and len(defects.BY_ID) == len(defects.LEDGER)
    assert not defects.substantiated({"path": "x"})


def test_power_is_a_precondition_for_a_null_not_for_a_positive():
    pre = power.preregistration(name="p", independent_unit="u", expected_sd=0.02, sesoi=0.05, seeds=16,
                                units=16, max_seeds=16, futility=0.01, harm=0.05)
    big = power.decide([0.27 + 0.11 * ((i % 5) - 2) for i in range(16)], pre)
    assert big["verdict"] == "positive"
    assert not big["adequately_powered"] and big["estimator_sufficient"]
    small = power.decide([0.002 + 0.001 * ((i % 3) - 1) for i in range(16)], pre)
    assert small["verdict"].startswith("null") and small["estimator_sufficient"]


def test_wording_check_does_not_flag_a_negated_term():
    assert report.wording_check("activation is not licensed and no architecture is selected", "invalid")["passes"]
    assert report.wording_check("the mechanism never improves retention", "mechanism_null")["passes"]
    assert not report.wording_check("activation is licensed", "invalid")["passes"]
    assert not report.wording_check("the mechanism improves retention", "mechanism_null")["passes"]
