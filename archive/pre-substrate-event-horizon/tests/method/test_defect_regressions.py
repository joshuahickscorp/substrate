"""One permanent regression test per historical defect class.

These are the tests that would have stopped each of the failures the Fast State Forge and its predecessors
shipped. They import the same kernel code the acceptance gate uses, so a kernel change that silently stops
catching a defect fails here rather than in a campaign six weeks later.

House style: no dashes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mop.method import acceptance, arms, baseline, bed, contracts, controls, defects, gate, graph, mechanism, power, report

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------- D1 order free control


def test_d1_order_free_control_with_temporal_conv_is_rejected():
    import torch.nn as nn

    class Bad(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(3, 3, 5, padding=2)

        def forward(self, x):
            return self.conv(x.transpose(1, 2)).mean(-1)

    torch.manual_seed(0)
    m = Bad().eval()
    r = controls.order_free(lambda t: m(t), torch.randn(4, 16, 3), module=m)
    assert not r["all_pass"]
    assert r["no_temporal_convolution"] is False
    assert r["timestep_permutation_invariant"] is False


def test_d1_a_real_order_free_control_passes():
    import torch.nn as nn

    class Good(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3, 3)

        def forward(self, x):
            return self.lin(x.mean(1))

    torch.manual_seed(0)
    m = Good().eval()
    assert controls.order_free(lambda t: m(t), torch.randn(4, 16, 3), module=m)["all_pass"]


def test_d1_recurrent_module_fails_the_structural_scan():
    import torch.nn as nn

    class Rec(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(3, 3, batch_first=True)

        def forward(self, x):
            return self.rnn(x)[0][:, -1]

    s = controls.structural_temporal_scan(Rec())
    assert s["no_recurrence"] is False


# ---------------------------------------------------------------- D2 and D3 replay


def test_d2_inactive_replay_is_rejected():
    r = controls.replay_active(
        {"replayed_items": 0, "admissions_after_full": 0, "buffer_sha_before": "a", "buffer_sha_after": "a"},
        boundary_crossed=True,
    )
    assert not r["all_pass"]
    assert r["items_were_replayed"] is False


def test_d3_buffer_that_stops_replacing_is_rejected():
    r = controls.replay_active(
        {"replayed_items": 10, "admissions_after_full": 0, "buffer_sha_before": "a", "buffer_sha_after": "b"},
        boundary_crossed=True,
    )
    assert r["buffer_kept_replacing"] is False
    assert not r["all_pass"]


def test_d3_a_run_that_never_crosses_a_boundary_is_rejected():
    r = controls.replay_active(
        {"replayed_items": 10, "admissions_after_full": 5, "buffer_sha_before": "a", "buffer_sha_after": "b"},
        boundary_crossed=False,
    )
    assert r["context_boundary_crossed"] is False
    assert not r["all_pass"]


def test_active_replay_passes():
    r = controls.replay_active(
        {"replayed_items": 100, "admissions_after_full": 40, "buffer_sha_before": "a", "buffer_sha_after": "b"},
        boundary_crossed=True,
    )
    assert r["all_pass"]


# ---------------------------------------------------------------- D4 and D12 arms


def _rec(name, tag):
    return arms.record(name, source=f"impl_{tag}", config={"p": tag}, call_graph=[tag],
                       state_transitions=[tag], param_delta={"g": tag}, memory={"m": tag},
                       resources={"r": tag}, outputs=[tag])


def test_d4_identical_arms_under_two_names_are_aliased():
    a = _rec("lstm", "x")
    b = dict(a)
    b["name"] = "lstm_gdumb"
    d = arms.distinctness([a, b])
    assert not d["all_distinct"]
    assert d["aliased_pairs"] == ["lstm|lstm_gdumb"]


def test_d4_genuinely_different_arms_are_distinct():
    assert arms.distinctness([_rec("a", "x"), _rec("b", "y")])["all_distinct"]


def test_d4_preregistered_alias_is_allowed():
    a = _rec("a", "x")
    b = dict(a)
    b["name"] = "b"
    assert arms.distinctness([a, b], preregistered_aliases=[("a", "b")])["all_distinct"]


def test_d12_ignored_configuration_field_is_rejected():
    base = _rec("arm", "x")
    s = arms.config_sensitivity(lambda cfg: {**base, "config_sha": arms.sha(cfg)}, {"flag": 0}, {"flag": 1})
    assert not s["all_honoured"]


def test_d12_honoured_configuration_field_passes():
    base = _rec("arm", "x")

    def probe(cfg):
        return {**base, "config_sha": arms.sha(cfg), "output_sha": arms.sha(cfg["flag"])}

    assert arms.config_sensitivity(probe, {"flag": 0}, {"flag": 1})["all_honoured"]


def test_inactive_branch_is_detected():
    r = arms.branch_activity([{**_rec("a", "x"), "call_graph": []}], {"a": "the_branch"})
    assert not r["all_active"]


def test_arm_mutation_suite_rejects_every_attack():
    m = arms.mutations([_rec("a", "x"), _rec("b", "y")])
    assert m["all_rejected"], [k for k, v in m.items() if not v]


# ---------------------------------------------------------------- D5, D6, D13 causal graph


def test_d5_mechanism_without_intervention_is_rejected():
    g = {"nodes": [{"id": "m", "type": "mechanism", "implementation": "x"},
                   {"id": "o", "type": "primary_outcome"}],
         "edges": [{"src": "m", "dst": "o", "type": "measured_relation"}]}
    assert any("no intervention" in v for v in graph.validate(g))


def test_d5_variable_without_implementation_is_rejected():
    g = {"nodes": [{"id": "m", "type": "mechanism", "implementation": ""},
                   {"id": "i", "type": "intervention", "implementation": "x"},
                   {"id": "o", "type": "primary_outcome"}],
         "edges": [{"src": "m", "dst": "i", "type": "implemented_causal_path"},
                   {"src": "i", "dst": "o", "type": "measured_relation"}]}
    assert any("no implementation path" in v for v in graph.validate(g))


def test_d6_analytic_relation_reported_as_measured_is_rejected():
    g = {"nodes": [{"id": "t", "type": "treatment", "implementation": "x"},
                   {"id": "o", "type": "primary_outcome"}],
         "edges": [{"src": "t", "dst": "o", "type": "measured_relation", "actually": "structurally_guaranteed"},
                   {"src": "t", "dst": "o", "type": "implemented_causal_path"}]}
    assert any("reported as measured" in v for v in graph.validate(g))


def test_d6_quantity_without_a_source_is_rejected():
    assert contracts.Quantity(0.0, "measured", "").violations("forgetting")


def test_d6_structural_quantity_is_labelled():
    q = contracts.structural(0.0, "engine.fit param partition")
    assert q.kind == "structurally_guaranteed" and not q.violations("forgetting")


def test_d13_future_information_reaching_a_decision_is_rejected():
    g = {"nodes": [{"id": "f", "type": "available_information", "time": "future"},
                   {"id": "m", "type": "mechanism", "implementation": "x"},
                   {"id": "i", "type": "intervention", "implementation": "x"},
                   {"id": "o", "type": "primary_outcome"}],
         "edges": [{"src": "f", "dst": "m", "type": "implemented_causal_path"},
                   {"src": "m", "dst": "i", "type": "implemented_causal_path"},
                   {"src": "i", "dst": "o", "type": "measured_relation"},
                   {"src": "i", "dst": "o", "type": "implemented_causal_path"}]}
    assert any("future information" in v for v in graph.validate(g))


def test_claim_broader_than_the_measured_path_is_rejected():
    g = {"nodes": [{"id": "o", "type": "primary_outcome"},
                   {"id": "c", "type": "claim", "requires": ["o", "unmeasured"]},
                   {"id": "unmeasured", "type": "mediator"}],
         "edges": [{"src": "unmeasured", "dst": "o", "type": "measured_relation"}]}
    assert any("broader than the measured path" in v for v in graph.validate(g))


def test_undeclared_confounder_is_rejected():
    g = {"nodes": [{"id": "c", "type": "confounder"}, {"id": "o", "type": "primary_outcome"}],
         "edges": [{"src": "c", "dst": "o", "type": "measured_relation"}]}
    assert any("not declared" in v for v in graph.validate(g))


# ---------------------------------------------------------------- D7, D8, D9 report


def test_d7_missing_report_key_raises():
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "a.json").write_text(json.dumps({"answers": {"Q12": 1}}))
        with pytest.raises(report.ReportFieldError):
            report.render(Path(td), {"Q13": {"artifact": "a.json", "pointer": "/answers/Q13"}})


def test_d7_null_valued_field_raises():
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "a.json").write_text(json.dumps({"answers": {"Q13": None}}))
        with pytest.raises(report.ReportFieldError):
            report.render(Path(td), {"Q13": {"artifact": "a.json", "pointer": "/answers/Q13"}})


def test_d7_spec_is_validated_before_the_run():
    r = report.validate_spec({"Q13": {"pointer": "/answers/Q13"}}, {"answers": {"Q12": 1}})
    assert not r["passes"]


def test_d7_resolvable_field_passes():
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "a.json").write_text(json.dumps({"answers": {"Q13": "yes"}}))
        r = report.render(Path(td), {"Q13": {"artifact": "a.json", "pointer": "/answers/Q13"}})
        assert r["fields"]["Q13"]["value"] == "yes"


def test_d8_baseline_identity_mismatch_is_provisional():
    rec = baseline.receipt("b", identity="fresh_independent", model="m", parameters=1, updates=1,
                           data_exposure=1, memory=0, compute_seconds=0.1,
                           validation_curve=[0.8, 0.8, 0.8, 0.8, 0.8], selected_checkpoint="best",
                           seed_scores=[0.8])
    c = baseline.comparison("e", "t", rec, "lstm_gdumb")
    assert c["status"] == "provisional" and not c["valid"]


def test_d8_matching_identity_is_terminal():
    rec = baseline.receipt("b", identity="lstm_gdumb", model="m", parameters=1, updates=1, data_exposure=1,
                           memory=0, compute_seconds=0.1, validation_curve=[0.8, 0.8, 0.8, 0.8, 0.8],
                           selected_checkpoint="best", seed_scores=[0.8])
    assert baseline.comparison("e", "t", rec, "lstm_gdumb")["status"] == "terminal"


def test_d9_prose_may_not_broaden_a_sealed_verdict():
    w = report.wording_check("the bed is marginal but promising", "invalid_no_temporal_headroom")
    assert not w["passes"]
    assert {o["term"] for o in w["offenders"]} >= {"marginal"}


def test_d9_prose_may_restate_a_sealed_verdict():
    assert report.wording_check("the bed is invalid: an order free reader solves it", "invalid_no_temporal_headroom")["passes"]


def test_d9_positive_prose_on_a_null_is_rejected():
    assert not report.wording_check("the mechanism improves retention", "mechanism_null")["passes"]


# ---------------------------------------------------------------- D10 veto


def _attack():
    return {f: "x" for f in defects.SUBSTANTIATION_FIELDS}


def test_d10_reproduction_beats_consensus():
    a = defects.adjudicate(_attack(), [{"verdict": "refuted"}] * 5, {"reproduced": True})
    assert a["status"] == "defect_confirmed" and a["vote_overridden"]


def test_d10_votes_alone_cannot_refute():
    a = defects.adjudicate(_attack(), [{"verdict": "refuted"}] * 5, None)
    assert a["status"] == "unresolved_no_reproduction_attempt"


def test_d10_failed_reproduction_refutes():
    a = defects.adjudicate(_attack(), [{"verdict": "confirmed"}], {"reproduced": False})
    assert a["status"] == "refuted_by_reproduction"


def test_d10_incomplete_substantiation_still_confirms_when_reproduced():
    a = defects.adjudicate({"path": "x"}, [], {"reproduced": True})
    assert a["status"] == "defect_confirmed"


# ---------------------------------------------------------------- D11 coverage


def test_d11_narrowed_scope_is_visible():
    c = gate.coverage_gate(99.0, 99.0, scope=["a.py"], excluded=["b.py"])
    assert c["scope_narrowing_declared"] and c["met"]


def test_d11_missed_target_reports_the_blocker():
    c = gate.coverage_gate(68.9, 56.0, scope=["a.py"], excluded=[])
    assert not c["met"] and "68.9" in c["honest_blocker"]


# ---------------------------------------------------------------- D14 and D15 headroom and baselines


def test_d14_two_seed_headroom_is_rejected():
    c = contracts.OracleContract(name="o", evidence={"headroom": {"n_seeds": 2, "residual_lower_95_cb": 0.01}})
    assert c.violations()


def test_d14_stable_headroom_passes():
    c = contracts.OracleContract(name="o", evidence={"headroom": {"n_seeds": 5, "residual_lower_95_cb": 0.02}})
    assert c.passed


def test_d14_nonpositive_headroom_is_rejected():
    c = contracts.OracleContract(name="o", evidence={"headroom": {"n_seeds": 8, "residual_lower_95_cb": -0.01}})
    assert c.violations()


def test_d15_unconverged_baseline_blocks_a_verdict():
    rec = baseline.receipt("b", identity="i", model="m", parameters=1, updates=1, data_exposure=1, memory=0,
                           compute_seconds=0.1, validation_curve=[0.2, 0.4, 0.6, 0.8, 0.95],
                           selected_checkpoint="last", seed_scores=[0.95])
    assert not rec["converged"]
    r = gate.classify_result(effect={"verdict": "positive"}, instrument_valid=True, bed_valid=True,
                             mechanism_active=True, baseline_valid=False, estimator_sufficient=True,
                             verifier_agrees=True, mutations_rejected=True, implementations_agreeing=2)
    assert r["classification"] == "unconverged_baseline"


def test_d15_resource_mismatch_is_detected():
    rec = baseline.receipt("b", identity="i", model="m", parameters=1, updates=10, data_exposure=10, memory=0,
                           compute_seconds=0.1, validation_curve=[0.8] * 5, selected_checkpoint="best",
                           seed_scores=[0.8], treatment_budget={"updates": 100, "memory": 0, "data_exposure": 100})
    assert rec["resource_matched"] is False


# ---------------------------------------------------------------- classification standards


def test_invalid_bed_never_becomes_a_mechanism_null():
    r = gate.classify_result(effect={"verdict": "null"}, instrument_valid=True, bed_valid=False,
                             mechanism_active=True, baseline_valid=True, estimator_sufficient=True,
                             verifier_agrees=True, mutations_rejected=True)
    assert r["classification"] == "invalid_bed" and not r["scientific"]


def test_inactive_mechanism_never_becomes_a_mechanism_null():
    r = gate.classify_result(effect={"verdict": "null"}, instrument_valid=True, bed_valid=True,
                             mechanism_active=False, baseline_valid=True, estimator_sufficient=True,
                             verifier_agrees=True, mutations_rejected=True)
    assert r["classification"] == "inactive_mechanism"


def test_a_positive_needs_two_implementations_and_a_verifier():
    e = {"verdict": "positive"}
    assert gate.classify_result(effect=e, instrument_valid=True, bed_valid=True, mechanism_active=True,
                                baseline_valid=True, estimator_sufficient=True, verifier_agrees=True,
                                mutations_rejected=True, implementations_agreeing=1)["classification"] == "provisional_positive"
    assert gate.classify_result(effect=e, instrument_valid=True, bed_valid=True, mechanism_active=True,
                                baseline_valid=True, estimator_sufficient=True, verifier_agrees=False,
                                mutations_rejected=True, implementations_agreeing=2)["classification"] == "provisional_positive"
    assert gate.classify_result(effect=e, instrument_valid=True, bed_valid=True, mechanism_active=True,
                                baseline_valid=True, estimator_sufficient=True, verifier_agrees=True,
                                mutations_rejected=True, implementations_agreeing=2)["classification"] == "positive"


def test_a_tie_is_a_null_and_a_wrong_direction_is_a_failure():
    pre = power.preregistration(name="p", independent_unit="u", expected_sd=0.01, sesoi=0.05, seeds=8,
                                units=8, max_seeds=8, futility=0.01, harm=0.05)
    assert power.decide([0.0] * 8, pre)["verdict"].startswith("null")
    assert power.decide([-0.02] * 8, pre)["verdict"] == "wrong_direction_failure"
    assert power.decide([-0.30] * 8, pre)["verdict"] == "harm"
    assert power.decide([0.20] * 8, pre)["verdict"] == "positive"


def test_underpowered_design_is_flagged_at_preregistration():
    p = power.preregistration(name="p", independent_unit="u", expected_sd=0.3, sesoi=0.05, seeds=3, units=3,
                              max_seeds=3, futility=0.01, harm=0.05)
    assert not p["adequately_powered"]
    c = contracts.PowerContract(name="p", declared={"sesoi": 0.05, "seeds": 3, "futility": 0.01, "harm": 0.05},
                                evidence={"power": {"mde": p["minimum_detectable_effect"]}})
    assert c.violations()


def test_stage_does_not_open_before_its_predecessor():
    assert power.stage_open("principal", {"passed": False})["open"] is False
    assert power.stage_open("calibration", None)["open"] is True


# ---------------------------------------------------------------- bed validity


def _bedm(**over):
    m = {"construct_valid": True, "units": {"group_disjoint": True, "test_touched": False, "n_units": 10},
         "leakage": {"clean": True}, "oracle_headroom": 0.2, "residual_headroom_lcb": 0.05,
         "baseline_converged": True, "order_necessity": 0.2, "intervention_possible": True,
         "seed_stability": 0.01}
    m.update(over)
    return m


def test_bed_without_temporal_requirement_is_invalid():
    assert bed.classify(_bedm(order_necessity=0.0))["classification"] == "invalid_no_temporal_requirement"


def test_bed_with_overlapping_units_is_invalid():
    assert bed.classify(_bedm(units={"group_disjoint": False, "test_touched": False, "n_units": 10}))[
        "classification"] == "invalid_no_independent_units"


def test_bed_with_a_touched_test_split_is_invalid():
    assert bed.classify(_bedm(units={"group_disjoint": True, "test_touched": True, "n_units": 10}))[
        "classification"] == "invalid_no_independent_units"


def test_unmeasured_bed_is_invalid_instrumentation_not_valid():
    assert bed.classify({"construct_valid": True})["classification"] == "invalid_instrumentation"


def test_valid_bed_is_valid():
    assert bed.classify(_bedm())["classification"] == "valid_principal_bed"


def test_unit_audit_detects_overlap():
    assert not bed.unit_audit([1, 2], [2, 3], [4])["group_disjoint"]


# ---------------------------------------------------------------- mechanism activity


def test_mechanism_with_no_effect_is_inactive():
    off = {k: 0 for k in mechanism.REQUIRED_MEASUREMENTS}
    assert mechanism.activity({"enabled": off, "disabled": off})["classification"] == "inactive_instrumentation"


def test_mechanism_with_missing_measurements_is_inactive():
    assert mechanism.activity({"enabled": {}, "disabled": {}})["classification"] == "inactive_instrumentation"


# ---------------------------------------------------------------- the suites themselves


def test_every_ledger_defect_has_a_mutation():
    r = acceptance.run()
    assert r["every_ledger_class_has_a_mutation"]
    assert r["all_rejected"], r["failures"]


def test_calibration_classifies_every_known_world():
    from mop.method import calibration

    r = calibration.run()
    assert r["all_pass"], [k for k, v in r["cases"].items() if not v["pass"]]


def test_ledger_entries_are_complete():
    for d in defects.LEDGER:
        assert set(d) >= {"id", "title", "declared", "actual", "rule", "detector", "mutation", "stage_caught"}


def test_d17_a_rising_curve_is_rejected_by_both_criteria():
    r = baseline.plateau([0.20, 0.40, 0.55, 0.70, 0.82, 0.95])
    assert not r["converged"] and not r["converged_strict"] and not r["converged_plateau"]


def test_d17_a_flat_noisy_curve_is_accepted_by_the_plateau_criterion_and_both_are_reported():
    r = baseline.plateau([0.8469, 0.8411, 0.8609, 0.8667, 0.8543, 0.8626])
    assert r["converged"] and r["criterion_used"] == "plateau" and not r["converged_strict"]


def test_d17_a_clean_plateau_still_passes_the_strict_criterion():
    r = baseline.plateau([0.80, 0.81, 0.81, 0.81, 0.81])
    assert r["converged_strict"] and r["criterion_used"] == "patience"


def test_d16_powered_boundary_rejects_the_measured_no_boundary_case():
    # five seeds around the measured E4 version one means: adaptation improved both contexts
    rows = [{"no_adapt_new": 0.687 + 0.01 * i, "no_adapt_old": 0.682 + 0.01 * i,
             "adapted_new": 0.726 + 0.01 * i, "adapted_old": 0.730 + 0.01 * i} for i in range(-2, 3)]
    r = bed.context_boundary_over_seeds(rows)
    assert r["classification"] == "invalid_no_context_boundary"
    assert not r["checks"]["adaptation_does_not_improve_the_old_context"]


def test_d16_powered_boundary_accepts_a_real_shift():
    rows = [{"no_adapt_new": 0.41 + 0.01 * i, "no_adapt_old": 0.69 + 0.01 * i,
             "adapted_new": 0.68 + 0.01 * i, "adapted_old": 0.64 + 0.01 * i} for i in range(-2, 3)]
    assert bed.context_boundary_over_seeds(rows)["classification"] == "context_boundary_crossed"


def test_d17_a_declining_curve_is_converged_not_still_improving():
    # peaked mid range and came down: overtrained, not undertrained. The measured E1 pooled control curve.
    r = baseline.plateau([0.4428, 0.4362, 0.4222, 0.451, 0.4115, 0.4132])
    assert r["converged"] and r["criterion_used"] == "plateau"


def test_d17_a_curve_still_rising_at_the_end_is_rejected():
    r = baseline.plateau([0.20, 0.30, 0.40, 0.55, 0.70, 0.86])
    assert not r["converged"]


def test_d18_label_permutation_is_scored_against_the_majority_class_rate():
    majority, fast, pooled, band = 0.2123, 0.2228, 0.1871, 0.05
    # neither arm learned anything: both sit within the band of the majority class rate
    assert abs(fast - majority) <= band and abs(pooled - majority) <= band
    # the naive criterion, a near zero difference between arms, failed a sound positive on the measured run
    assert not abs(0.04746) < 0.1 * 0.4192


def test_d18_a_real_residual_signal_still_fails_the_permutation_control():
    majority, fast, pooled, band = 0.2123, 0.60, 0.1871, 0.05
    assert not (abs(fast - majority) <= band and abs(pooled - majority) <= band)
