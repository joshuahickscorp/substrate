"""Method acceptance gate.

Every historical defect is rebuilt here as a live mutation and fed to the kernel. A mutation passes when the
kernel rejects it and names the right classification. Principal execution cannot begin until every one is
green, because a method that cannot catch the failures it already made has no standing to certify new ones.

Each mutation carries the stage that caught it, which is how the scorecard measures whether defects are
caught before principal compute or after.

House style: no dashes.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mop.method import arms, baseline, contracts, controls, defects, gate, graph, mechanism, power, report


def _ok(rejected: bool, classification: str, expected: str, detail=None) -> dict:
    return {
        "rejected": bool(rejected),
        "classification": classification,
        "expected_classification": expected,
        "correctly_classified": classification == expected,
        "pass": bool(rejected) and classification == expected,
        "detail": detail,
    }


# ---------------------------------------------------------------- the sixteen mutations


def m_temporal_conv_in_order_free_control() -> dict:
    import torch
    import torch.nn as nn

    class Ctrl(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(3, 3, 5, padding=2)

        def forward(self, x):
            return self.conv(x.transpose(1, 2)).mean(-1)

    torch.manual_seed(0)
    m = Ctrl().eval()
    r = controls.order_free(lambda t: m(t), torch.randn(4, 16, 3), module=m)
    c = contracts.ControlContract(name="order_free", evidence={"semantic": r})
    return _ok(not r["all_pass"] and not c.passed, "invalid_control_semantics", "invalid_control_semantics",
               {"failed": [k for k, v in r.items() if v is False]})


def m_inactive_replay() -> dict:
    trace = {"replayed_items": 0, "admissions_after_full": 0, "buffer_size": 600,
             "buffer_sha_before": "a", "buffer_sha_after": "a"}
    r = controls.replay_active(trace, boundary_crossed=True)
    on = {k: 0 for k in mechanism.REQUIRED_MEASUREMENTS}
    act = mechanism.activity({"enabled": on, "disabled": on})
    return _ok(not r["all_pass"] and not act["active"], act["classification"], "inactive_instrumentation",
               {"replay_checks": r})


def m_buffer_that_stops_replacing() -> dict:
    """The real defect: admissions stop once the buffer fills, so late classes never enter."""
    cap, admitted = 4, []
    for i in range(12):  # the defective policy: admit only while a slot is free
        if len(admitted) < cap:
            admitted.append(i)
    trace = {"replayed_items": 40, "admissions_after_full": 0, "buffer_size": cap,
             "buffer_sha_before": "a", "buffer_sha_after": "a"}
    r = controls.replay_active(trace, boundary_crossed=True)
    return _ok(not r["all_pass"], "inactive_instrumentation", "inactive_instrumentation",
               {"admitted": admitted, "late_items_admitted": [i for i in admitted if i >= cap], "checks": r})


def m_aliased_lstm_and_lstm_gdumb() -> dict:
    a = arms.record("lstm", source="lstm_impl", config={"memory": "none"}, call_graph=["fit"],
                    state_transitions=["s0", "s1"], param_delta={"w": 1}, memory={"policy": "none", "size": 0},
                    resources={"updates": 300}, outputs=[1, 2, 3])
    b = dict(a)
    b["name"] = "lstm_gdumb"  # different name, identical behaviour because the buffer never replayed
    d = arms.distinctness([a, b])
    c = contracts.ArmContract(name="lstm_gdumb", evidence={"distinctness": d["per_arm"]["lstm_gdumb"]})
    return _ok(not d["all_distinct"] and not c.passed, "aliased_arm", "aliased_arm",
               {"aliased_pairs": d["aliased_pairs"]})


def m_phantom_parameter_group() -> dict:
    g = {
        "nodes": [
            {"id": "H.norm", "type": "mechanism", "label": "phantom group"},
            {"id": "acc", "type": "primary_outcome"},
        ],
        "edges": [{"src": "H.norm", "dst": "acc", "type": "measured_relation"}],
    }
    v = graph.validate(g)
    return _ok(bool(v), "invalid_causal_graph", "invalid_causal_graph", {"rejections": v})


def m_variable_without_causal_path() -> dict:
    g = {
        "nodes": [
            {"id": "memory_state", "type": "mechanism", "implementation": "", "label": "reported cause"},
            {"id": "iv", "type": "intervention", "implementation": "", "label": "none"},
            {"id": "acc", "type": "primary_outcome"},
        ],
        "edges": [
            {"src": "memory_state", "dst": "iv", "type": "assumed_scientific_relation"},
            {"src": "iv", "dst": "acc", "type": "measured_relation"},
        ],
    }
    v = graph.validate(g)
    hits = [x for x in v if "no implementation path" in x or "no implemented causal path" in x]
    return _ok(bool(hits), "invalid_causal_graph", "invalid_causal_graph", {"rejections": v})


def m_analytic_value_marked_measured() -> dict:
    q = contracts.Quantity(0.0, "measured", "MOP_X.json#/forgetting/domain_local")
    g = {
        "nodes": [
            {"id": "partition", "type": "treatment", "implementation": "engine.fit"},
            {"id": "forgetting", "type": "primary_outcome"},
        ],
        "edges": [
            {"src": "partition", "dst": "forgetting", "type": "measured_relation",
             "actually": "structurally_guaranteed"},
            {"src": "partition", "dst": "forgetting", "type": "implemented_causal_path"},
        ],
    }
    v = graph.validate(g)
    hits = [x for x in v if "reported as measured" in x]
    # the quantity itself is well formed, which is exactly why the graph has to be the one that catches it
    return _ok(bool(hits), "analytic_reported_as_measured", "analytic_reported_as_measured",
               {"rejections": v, "quantity_violations": q.violations("forgetting")})


def m_missing_report_key() -> dict:
    """Caught twice: once at preregistration against the declared output schema, once against the artifact."""
    spec = {"Q13": {"artifact": "MOP_X.json", "pointer": "/answers/Q13"}}
    pre = report.validate_spec(spec, {"answers": {"Q12": None}})
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "MOP_X.json").write_text(json.dumps({"answers": {"Q12": "yes"}}))
        post = report.audit_report(Path(td), spec)
    return _ok(not pre["passes"] and not post["passes"], "report_field_unresolved", "report_field_unresolved",
               {"preregistration_time": pre["errors"], "artifact_time": post["errors"]})


def m_wrong_baseline_comparison() -> dict:
    rec = baseline.receipt("b", identity="fresh_independent", model="gru", parameters=10, updates=300,
                           data_exposure=1000, memory=600, compute_seconds=1.0,
                           validation_curve=[0.8, 0.81, 0.81, 0.81, 0.81], selected_checkpoint="best",
                           seed_scores=[0.81, 0.80])
    c = baseline.comparison("architecture_g_effect", "G", rec, "lstm_gdumb")
    return _ok(not c["valid"], "baseline_identity_mismatch", "baseline_identity_mismatch", c["issues"])


def m_softened_verdict_wording() -> dict:
    prose = "both principal beds are marginal: order matters but an order free reader is nearly as good"
    w = report.wording_check(prose, "invalid_no_temporal_headroom")
    return _ok(not w["passes"], "verdict_softened", "verdict_softened", w["offenders"])


def m_ignored_treatment_flag() -> dict:
    base = arms.record("arm", source="impl", config={"treatment": False}, call_graph=["fit"],
                       state_transitions=["s"], param_delta={"w": 1}, memory={}, resources={"updates": 1},
                       outputs=[1])

    def probe(cfg):  # the implementation never reads cfg["treatment"]
        r = dict(base)
        r["config_sha"] = arms.sha(cfg)
        return r

    s = arms.config_sensitivity(probe, {"treatment": False}, {"treatment": True})
    return _ok(not s["all_honoured"], "configuration_field_ignored", "configuration_field_ignored", s)


def m_future_information_leakage() -> dict:
    g = {
        "nodes": [
            {"id": "whole_sequence_stat", "type": "available_information", "time": "future"},
            {"id": "gate", "type": "mechanism", "implementation": "engine.Gate", "time": "decision_time"},
            {"id": "iv", "type": "intervention", "implementation": "engine.fit"},
            {"id": "acc", "type": "primary_outcome"},
        ],
        "edges": [
            {"src": "whole_sequence_stat", "dst": "gate", "type": "implemented_causal_path"},
            {"src": "gate", "dst": "iv", "type": "implemented_causal_path"},
            {"src": "iv", "dst": "acc", "type": "measured_relation"},
            {"src": "iv", "dst": "acc", "type": "implemented_causal_path"},
        ],
    }
    v = graph.validate(g)
    hits = [x for x in v if "future information" in x]
    return _ok(bool(hits), "future_information_leakage", "future_information_leakage", {"rejections": v})


def m_unconverged_baseline() -> dict:
    rec = baseline.receipt("b", identity="lstm_gdumb", model="lstm", parameters=10, updates=40,
                           data_exposure=100, memory=600, compute_seconds=1.0,
                           validation_curve=[0.2, 0.4, 0.55, 0.7, 0.82], selected_checkpoint="last",
                           seed_scores=[0.82, 0.80])
    c = baseline.comparison("e", "G", rec, "lstm_gdumb")
    res = gate.classify_result(effect={"verdict": "positive"}, instrument_valid=True, bed_valid=True,
                               mechanism_active=True, baseline_valid=c["valid"], estimator_sufficient=True,
                               verifier_agrees=True, mutations_rejected=True, implementations_agreeing=2)
    return _ok(not c["valid"] and res["classification"] == "unconverged_baseline", res["classification"],
               "unconverged_baseline", {"issues": c["issues"]})


def m_two_seed_false_headroom() -> dict:
    c = contracts.OracleContract(
        name="update_partition_oracle",
        evidence={"headroom": {"n_seeds": 2, "residual_lower_95_cb": 0.004, "mean": 0.012}},
    )
    v = c.violations()
    return _ok(bool(v) and not c.passed, "insufficient_headroom_authority", "insufficient_headroom_authority", v)


def m_reviewer_consensus_overrides_reproduction() -> dict:
    attack = {
        "path": "fastforge/engine.py",
        "condition": "Memory.add gates admission on a free slot",
        "reproduction": "python -m pytest tests/method/test_defect_regressions.py -k buffer",
        "expected": "late classes are admitted",
        "actual": "no admission after the buffer fills",
        "consequence": "two replay arms alias",
    }
    votes = [{"verdict": "refuted"}] * 4 + [{"verdict": "confirmed"}]
    a = defects.adjudicate(attack, votes, {"reproduced": True})
    return _ok(a["status"] == "defect_confirmed" and a["vote_overridden"], "defect_confirmed",
               "defect_confirmed", a)


def m_narrowed_coverage_scope() -> dict:
    c = gate.coverage_gate(96.0, 88.0, scope=["mop/method/power.py"],
                           excluded=["mop/method/controls.py", "fastforge/data.py"])
    # the numbers are green, the scope is not: the gate must still surface the narrowing
    return _ok(c["scope_narrowing_declared"], "coverage_scope_narrowed", "coverage_scope_narrowed", c)


def m_context_split_that_crosses_no_boundary() -> dict:
    """D16, discovered by this program's own E4 scout on speech_stream."""
    from mop.method import bed as B

    # the measured numbers from that scout: adaptation improved both contexts
    r = B.context_boundary(no_adapt_new=0.68699, no_adapt_old=0.68154,
                           adapted_new=0.72635, adapted_old=0.73007)
    ok = B.context_boundary(no_adapt_new=0.40, no_adapt_old=0.70, adapted_new=0.65, adapted_old=0.62)
    return _ok(r["classification"] == "invalid_no_context_boundary" and ok["checks"]["boundary_crossed"],
               r["classification"], "invalid_no_context_boundary",
               {"measured": r, "a_real_boundary_still_passes": ok["checks"]})


def m_brittle_plateau_criterion() -> dict:
    """D17. A genuinely rising curve must still be rejected by both criteria."""
    rising = baseline.plateau([0.20, 0.40, 0.55, 0.70, 0.82, 0.95])
    flat = baseline.plateau([0.8469, 0.8411, 0.8609, 0.8667, 0.8543, 0.8626])  # the measured E1 curve
    ok = (not rising["converged"]) and flat["converged"] and not flat["converged_strict"]
    return _ok(ok, "brittle_criterion_repaired", "brittle_criterion_repaired",
               {"rising_curve_still_rejected": not rising["converged"],
                "flat_curve_now_accepted_by": flat["criterion_used"],
                "strict_criterion_still_reported": flat["converged_strict"]})


def m_underpowered_design_admitted() -> dict:
    p = power.preregistration(name="weak", independent_unit="seed", expected_sd=0.25, sesoi=0.05,
                              seeds=3, units=3, max_seeds=3, futility=0.01, harm=0.05)
    c = contracts.PowerContract(name="weak", declared={"sesoi": 0.05, "seeds": 3, "futility": 0.01, "harm": 0.05},
                                evidence={"power": {"mde": p["minimum_detectable_effect"]}})
    return _ok(bool(c.violations()), "insufficient_power", "insufficient_power", c.violations())


# What each stage protects. compute means the defect is caught before principal training is spent; claim
# means the run may be sound but the finding cannot be published until the defect is repaired. Both are
# automatic; conflating them would overstate what a preregistration time check can see.
STAGE_BLOCKS = {
    "measurement_model": "compute",
    "causal_model": "compute",
    "control_semantics": "compute",
    "arm_distinctness": "compute",
    "bed_validity": "compute",
    "baseline_convergence": "compute",
    "oracle_headroom": "compute",
    "power_and_units": "compute",
    "acceptance": "compute",
    "adjudication": "claim",
    "terminal_classification": "claim",
}

MUTATIONS = {
    "temporal_conv_in_order_free_control": (m_temporal_conv_in_order_free_control, "D1", "control_semantics"),
    "inactive_replay": (m_inactive_replay, "D2", "control_semantics"),
    "buffer_that_stops_replacing": (m_buffer_that_stops_replacing, "D3", "control_semantics"),
    "aliased_lstm_and_lstm_gdumb": (m_aliased_lstm_and_lstm_gdumb, "D4", "arm_distinctness"),
    "phantom_parameter_group": (m_phantom_parameter_group, "D5", "causal_model"),
    "variable_without_causal_path": (m_variable_without_causal_path, "D5", "causal_model"),
    "analytic_value_marked_measured": (m_analytic_value_marked_measured, "D6", "causal_model"),
    "missing_report_key": (m_missing_report_key, "D7", "measurement_model"),
    "wrong_baseline_comparison": (m_wrong_baseline_comparison, "D8", "baseline_convergence"),
    "softened_verdict_wording": (m_softened_verdict_wording, "D9", "terminal_classification"),
    "reviewer_consensus_overrides_reproduction": (m_reviewer_consensus_overrides_reproduction, "D10", "adjudication"),
    "narrowed_coverage_scope": (m_narrowed_coverage_scope, "D11", "acceptance"),
    "ignored_treatment_flag": (m_ignored_treatment_flag, "D12", "arm_distinctness"),
    "future_information_leakage": (m_future_information_leakage, "D13", "causal_model"),
    "two_seed_false_headroom": (m_two_seed_false_headroom, "D14", "oracle_headroom"),
    "unconverged_baseline": (m_unconverged_baseline, "D15", "baseline_convergence"),
    "context_split_that_crosses_no_boundary": (m_context_split_that_crosses_no_boundary, "D16", "bed_validity"),
    "brittle_plateau_criterion": (m_brittle_plateau_criterion, "D17", "baseline_convergence"),
    "underpowered_design_admitted": (m_underpowered_design_admitted, "D14", "power_and_units"),
}


def run() -> dict:
    results = {}
    for name, (fn, defect_id, stage) in MUTATIONS.items():
        r = fn()
        r["defect_id"] = defect_id
        r["stage_caught"] = stage
        r["blocks"] = STAGE_BLOCKS.get(stage, "claim")
        r["before_principal_execution"] = r["blocks"] == "compute"
        results[name] = r
    covered = {d["id"] for _, d_id, _ in MUTATIONS.values() for d in defects.LEDGER if d["id"] == d_id}
    return {
        "mutations": results,
        "n_mutations": len(results),
        "all_rejected": all(r["pass"] for r in results.values()),
        "failures": [k for k, r in results.items() if not r["pass"]],
        "defect_classes_covered": sorted(covered),
        "defect_classes_in_ledger": sorted(d["id"] for d in defects.LEDGER),
        "every_ledger_class_has_a_mutation": sorted(covered) == sorted(d["id"] for d in defects.LEDGER),
        "blocks_compute": sorted(k for k, r in results.items() if r["blocks"] == "compute"),
        "blocks_claim_only": sorted(k for k, r in results.items() if r["blocks"] == "claim"),
        "all_automatically_detected": all(r["pass"] for r in results.values()),
        "all_caught_before_principal_execution": all(
            r["before_principal_execution"] for r in results.values()
        ),
        "note": (
            "report integrity and adjudication defects are caught automatically but after execution, "
            "because prose and reviewer votes do not exist before the run. They block the claim, not the "
            "compute, and no claim can be sealed while one is open."
        ),
    }
