from __future__ import annotations

import pytest

from mop.experiments.expansion_harness import CLAIM_SCOPE as HARNESS_CLAIM_SCOPE
from mop.mechanisms.stage5_session_disjoint_validity_scaffold import (
    CLAIM_SCOPE,
    NEGATIVE_CONTROLS,
    PRIOR_NULL,
    VALIDITY_AXES,
    AxisOutcome,
    ControlOutcome,
    ExternalValidityGate,
    MatchedCostBudget,
    MeasuredEfficiencyContract,
    MeasuredResource,
    SessionDisjointValidityContract,
    SessionDisjointValidityRefusal,
    build_activation_gate,
    build_measured_efficiency,
    build_session_disjoint_contract,
    synthesize_axis_evidence,
)


def test_claim_scope_matches_harness_byte_for_byte() -> None:
    assert CLAIM_SCOPE == HARNESS_CLAIM_SCOPE
    assert CLAIM_SCOPE == "deterministic programmatic mechanics only; no capability or natural-data claim"


def test_prior_null_and_capability_flag() -> None:
    from mop.mechanisms.stage5_session_disjoint_validity_scaffold import SCIENTIFIC_CAPABILITY_CLAIM

    assert PRIOR_NULL == "session-bound-leakage"
    assert SCIENTIFIC_CAPABILITY_CLAIM is False


def test_axis_outcome_digest_is_stable_and_deterministic() -> None:
    ev = synthesize_axis_evidence(7, "fresh-session")
    first = AxisOutcome(
        axis="fresh-session", passed=True, replications=3, disjoint_from_calibration=True, evidence_sha256=ev
    )
    second = AxisOutcome(
        axis="fresh-session", passed=True, replications=3, disjoint_from_calibration=True, evidence_sha256=ev
    )
    assert first.digest() == second.digest()
    assert len(first.digest()) == 64


def test_axis_outcome_rejects_passing_but_leaky_axis() -> None:
    ev = synthesize_axis_evidence(1, "new-seeds")
    with pytest.raises(SessionDisjointValidityRefusal, match="disjoint from the calibration session"):
        AxisOutcome(
            axis="new-seeds", passed=True, replications=2, disjoint_from_calibration=False, evidence_sha256=ev
        )


def test_axis_outcome_requires_two_replications() -> None:
    ev = synthesize_axis_evidence(1, "lesions")
    with pytest.raises(SessionDisjointValidityRefusal, match="two independent replications"):
        AxisOutcome(
            axis="lesions", passed=False, replications=1, disjoint_from_calibration=False, evidence_sha256=ev
        )


def test_axis_outcome_rejects_unknown_axis() -> None:
    ev = synthesize_axis_evidence(1, "fresh-session")
    with pytest.raises(SessionDisjointValidityRefusal, match="unsupported validity axis"):
        AxisOutcome(
            axis="cherry-picked",
            passed=False,
            replications=2,
            disjoint_from_calibration=False,
            evidence_sha256=ev,
        )


def test_axis_outcome_rejects_widened_claim_scope() -> None:
    ev = synthesize_axis_evidence(1, "fresh-session")
    with pytest.raises(SessionDisjointValidityRefusal, match="claim scope cannot be widened"):
        AxisOutcome(
            axis="fresh-session",
            passed=False,
            replications=2,
            disjoint_from_calibration=False,
            evidence_sha256=ev,
            claim_scope="a capability was demonstrated",
        )


def test_axis_outcome_rejects_bad_evidence_digest() -> None:
    with pytest.raises(SessionDisjointValidityRefusal, match="SHA-256"):
        AxisOutcome(
            axis="fresh-session",
            passed=False,
            replications=2,
            disjoint_from_calibration=False,
            evidence_sha256="nothex",
        )


def test_control_outcome_rejects_unknown_control() -> None:
    ev = synthesize_axis_evidence(1, "fresh-session")
    with pytest.raises(SessionDisjointValidityRefusal, match="unsupported negative control"):
        ControlOutcome(control="hindsight", leaked=False, evidence_sha256=ev)


def test_matched_cost_budget_must_be_non_vacuous() -> None:
    with pytest.raises(SessionDisjointValidityRefusal, match="non-vacuous"):
        MatchedCostBudget(params=0, flops=1, wall_time_ms=1, peak_memory_bytes=1)


def test_measured_efficiency_valid_and_digest_stable() -> None:
    contract = build_measured_efficiency(4)
    assert contract.digest() == build_measured_efficiency(4).digest()
    assert contract.payload()["matched_cost_required"] is True


def test_measured_efficiency_refuses_declared_only_cost() -> None:
    budget = MatchedCostBudget(params=8, flops=8, wall_time_ms=8, peak_memory_bytes=8)
    resource = MeasuredResource(
        kind="wall_time_s",
        declared=1.0,
        measured=10.0,
        unit="seconds",
        measurement_source="perf_counter",
    )
    with pytest.raises(SessionDisjointValidityRefusal, match="declared-only efficiency claim refused"):
        MeasuredEfficiencyContract(
            efficiency_metric="acc_per_flop",
            resources=(resource,),
            matched=budget,
            rtol=0.05,
            matched_cost_required=True,
            baseline_measured=True,
        )


def test_measured_resource_refuses_declared_source() -> None:
    with pytest.raises(SessionDisjointValidityRefusal, match="real measurement source"):
        MeasuredResource(kind="flops", declared=1.0, measured=1.0, unit="flop", measurement_source="estimate")


def test_measured_efficiency_requires_measured_baseline() -> None:
    budget = MatchedCostBudget(params=8, flops=8, wall_time_ms=8, peak_memory_bytes=8)
    resource = MeasuredResource(
        kind="flops", declared=8.0, measured=8.0, unit="flop", measurement_source="counter"
    )
    with pytest.raises(SessionDisjointValidityRefusal, match="measured baseline cost"):
        MeasuredEfficiencyContract(
            efficiency_metric="acc_per_flop",
            resources=(resource,),
            matched=budget,
            rtol=0.05,
            matched_cost_required=True,
            baseline_measured=False,
        )


def test_measured_efficiency_requires_matched_cost() -> None:
    budget = MatchedCostBudget(params=8, flops=8, wall_time_ms=8, peak_memory_bytes=8)
    resource = MeasuredResource(
        kind="flops", declared=8.0, measured=8.0, unit="flop", measurement_source="counter"
    )
    with pytest.raises(SessionDisjointValidityRefusal, match="matched full-system cost"):
        MeasuredEfficiencyContract(
            efficiency_metric="acc_per_flop",
            resources=(resource,),
            matched=budget,
            rtol=0.05,
            matched_cost_required=False,
            baseline_measured=True,
        )


def test_measured_efficiency_needs_a_compute_resource() -> None:
    budget = MatchedCostBudget(params=8, flops=8, wall_time_ms=8, peak_memory_bytes=8)
    resource = MeasuredResource(
        kind="peak_memory_bytes", declared=8.0, measured=8.0, unit="bytes", measurement_source="rss sampler"
    )
    with pytest.raises(SessionDisjointValidityRefusal, match="measured compute cost"):
        MeasuredEfficiencyContract(
            efficiency_metric="acc_per_flop",
            resources=(resource,),
            matched=budget,
            rtol=0.05,
            matched_cost_required=True,
            baseline_measured=True,
        )


def test_contract_is_deterministic_under_seed() -> None:
    first = build_session_disjoint_contract(11)
    second = build_session_disjoint_contract(11)
    assert first.digest() == second.digest()
    assert build_session_disjoint_contract(12).digest() != first.digest()


def test_contract_axis_completeness_fails_closed_on_reorder() -> None:
    base = build_session_disjoint_contract(3)
    reordered = (base.axes[1], base.axes[0]) + base.axes[2:]
    with pytest.raises(SessionDisjointValidityRefusal, match="out of canonical order"):
        SessionDisjointValidityContract(axes=reordered, controls=base.controls, efficiency=base.efficiency)


def test_contract_control_completeness_fails_closed_on_missing() -> None:
    base = build_session_disjoint_contract(3)
    with pytest.raises(SessionDisjointValidityRefusal, match="control coverage"):
        SessionDisjointValidityContract(
            axes=base.axes, controls=base.controls[:2], efficiency=base.efficiency
        )


def test_contract_rejects_reframed_prior_null() -> None:
    base = build_session_disjoint_contract(3)
    with pytest.raises(SessionDisjointValidityRefusal, match="prior null cannot be reframed"):
        SessionDisjointValidityContract(
            axes=base.axes, controls=base.controls, efficiency=base.efficiency, prior_null="measurement-noise"
        )


def test_contract_reports_unmet_axes_and_leaks() -> None:
    failing = build_session_disjoint_contract(5, all_axes_pass=False, controls_clean=False)
    assert set(failing.unmet_axes()) == set(VALIDITY_AXES)
    assert set(failing.leaked_controls()) == set(NEGATIVE_CONTROLS)
    assert not failing.all_axes_pass()
    assert failing.any_control_leaked()


def test_gate_is_off_by_default_and_refuses() -> None:
    gate = build_activation_gate()
    contract = build_session_disjoint_contract(1)
    with pytest.raises(SessionDisjointValidityRefusal, match="activation not earned"):
        gate.expected_receipt(contract)


def test_gate_cannot_be_forced_on_at_construction() -> None:
    with pytest.raises(SessionDisjointValidityRefusal, match="off by default"):
        ExternalValidityGate(default_state="on")


def test_gate_certifies_only_licensed_matching_and_passing() -> None:
    gate = build_activation_gate("disjoint-reviewer-license")
    contract = build_session_disjoint_contract(9)
    receipt = gate.expected_receipt(contract)
    certificate = gate.certify_generality(contract, receipt)
    assert certificate["certified"] is True
    assert certificate["contract_digest"] == contract.digest()


def test_gate_refuses_wrong_receipt() -> None:
    gate = build_activation_gate("license-a")
    contract = build_session_disjoint_contract(9)
    other = gate.expected_receipt(build_session_disjoint_contract(10))
    with pytest.raises(SessionDisjointValidityRefusal, match="receipt does not match"):
        gate.certify_generality(contract, other)


def test_gate_refuses_when_an_axis_did_not_pass() -> None:
    gate = build_activation_gate("license-a")
    contract = build_session_disjoint_contract(9, all_axes_pass=False)
    receipt = gate.expected_receipt(contract)
    with pytest.raises(SessionDisjointValidityRefusal, match="axes did not pass"):
        gate.certify_generality(contract, receipt)


def test_gate_refuses_when_a_control_leaked() -> None:
    gate = build_activation_gate("license-a")
    contract = build_session_disjoint_contract(9, controls_clean=False)
    receipt = gate.expected_receipt(contract)
    with pytest.raises(SessionDisjointValidityRefusal, match="negative controls leaked"):
        gate.certify_generality(contract, receipt)
