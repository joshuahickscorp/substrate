"""Unit tests for the cross-seed cognitive-trace stability scaffold (epoch G1-C0).

These tests exercise the contracts, the deterministic agreement mechanics, the fail-closed verdict,
and the mechanism license gate. They assert determinism and refusal behavior. No capability is
claimed anywhere.
"""

from __future__ import annotations

import pytest

from mop.mechanisms.trace_stability_scaffold import (
    CLAIM_SCOPE,
    MIN_SEEDS,
    REQUIRED_CONTROLS,
    SCIENTIFIC_CAPABILITY_CLAIM,
    STABILITY_METRICS,
    TRACE_STABILITY_SCHEMA,
    ControlOutcome,
    LicenseReceipt,
    MatchedMeasurementBudget,
    MechanismLicenseGate,
    StabilityVerdict,
    TraceRecord,
    TraceStabilityContract,
    TraceStabilityRefusal,
    assert_controls_complete,
    build_default_contract,
    coverage,
    cross_seed_agreement,
    dead_control_outcomes,
    synthesize_trace_records,
)


def _verdict(
    contract: TraceStabilityContract,
    records: tuple[TraceRecord, ...],
    outcomes: tuple[ControlOutcome, ...],
) -> StabilityVerdict:
    return StabilityVerdict(
        schema=TRACE_STABILITY_SCHEMA,
        contract=contract,
        records=records,
        control_outcomes=outcomes,
    )


# ---------------------------------------------------------------------------
# Claim scope and module-level invariants.
# ---------------------------------------------------------------------------


def test_claim_scope_constant_value() -> None:
    assert CLAIM_SCOPE == "deterministic programmatic mechanics only; no capability or natural-data claim"
    assert SCIENTIFIC_CAPABILITY_CLAIM is False


def test_stability_metric_vocabulary_is_fixed() -> None:
    assert STABILITY_METRICS == ("rank-correlation", "sign-agreement", "effect-direction")
    assert MIN_SEEDS >= 8


# ---------------------------------------------------------------------------
# Contract bar.
# ---------------------------------------------------------------------------


def test_contract_digest_is_stable() -> None:
    first = build_default_contract().sha256
    second = build_default_contract().sha256
    assert first == second
    assert len(first) == 64


def test_contract_requires_min_seeds() -> None:
    with pytest.raises(ValueError, match="at least 8 seeds"):
        TraceStabilityContract(
            schema=TRACE_STABILITY_SCHEMA,
            trace_id="trace.candidate",
            stability_metric="sign-agreement",
            min_seeds=4,
            agreement_threshold=0.9,
            controls=REQUIRED_CONTROLS,
            matched_cost_required=True,
            matched=MatchedMeasurementBudget(seeds=8, sessions=2, samples_per_seed=8, compute_units=8),
        )


def test_contract_rejects_undeclared_metric() -> None:
    with pytest.raises(ValueError, match="undeclared or off vocabulary"):
        TraceStabilityContract(
            schema=TRACE_STABILITY_SCHEMA,
            trace_id="trace.candidate",
            stability_metric="p-value",
            min_seeds=8,
            agreement_threshold=0.9,
            controls=REQUIRED_CONTROLS,
            matched_cost_required=True,
            matched=MatchedMeasurementBudget(seeds=8, sessions=2, samples_per_seed=8, compute_units=8),
        )


def test_contract_requires_matched_cost() -> None:
    with pytest.raises(ValueError, match="matched measurement cost"):
        TraceStabilityContract(
            schema=TRACE_STABILITY_SCHEMA,
            trace_id="trace.candidate",
            stability_metric="sign-agreement",
            min_seeds=8,
            agreement_threshold=0.9,
            controls=REQUIRED_CONTROLS,
            matched_cost_required=False,
            matched=MatchedMeasurementBudget(seeds=8, sessions=2, samples_per_seed=8, compute_units=8),
        )


def test_contract_rejects_control_drift() -> None:
    with pytest.raises(ValueError, match="membership or order drift"):
        TraceStabilityContract(
            schema=TRACE_STABILITY_SCHEMA,
            trace_id="trace.candidate",
            stability_metric="sign-agreement",
            min_seeds=8,
            agreement_threshold=0.9,
            controls=("shuffled-seed", "single-seed", "permuted-trace", "label-shuffled"),
            matched_cost_required=True,
            matched=MatchedMeasurementBudget(seeds=8, sessions=2, samples_per_seed=8, compute_units=8),
        )


def test_contract_rejects_widened_claim_scope() -> None:
    with pytest.raises(ValueError, match="claim scope cannot be widened"):
        TraceStabilityContract(
            schema=TRACE_STABILITY_SCHEMA,
            trace_id="trace.candidate",
            stability_metric="sign-agreement",
            min_seeds=8,
            agreement_threshold=0.9,
            controls=REQUIRED_CONTROLS,
            matched_cost_required=True,
            matched=MatchedMeasurementBudget(seeds=8, sessions=2, samples_per_seed=8, compute_units=8),
            claim_scope="a stable trace was demonstrated",
        )


def test_matched_budget_must_be_non_vacuous() -> None:
    with pytest.raises(ValueError, match="non-vacuous"):
        MatchedMeasurementBudget(seeds=0, sessions=2, samples_per_seed=8, compute_units=8)


def test_controls_completeness_helper() -> None:
    assert_controls_complete(REQUIRED_CONTROLS)
    with pytest.raises(ValueError, match="membership or order drift"):
        assert_controls_complete(("single-seed",))


# ---------------------------------------------------------------------------
# Records and agreement mechanics.
# ---------------------------------------------------------------------------


def test_synthesize_is_deterministic_under_seed() -> None:
    first = synthesize_trace_records(trace_id="trace.candidate", seeds=range(8))
    second = synthesize_trace_records(trace_id="trace.candidate", seeds=range(8))
    assert tuple(r.digest() for r in first) == tuple(r.digest() for r in second)


def test_trace_record_rejects_bad_ranking() -> None:
    with pytest.raises(ValueError, match="permutation of range"):
        TraceRecord(
            trace_id="trace.candidate",
            seed=0,
            session_id="s0",
            effect=0.5,
            trace_sha256="a" * 64,
            ranking=(0, 0, 1),
        )


def test_trace_record_rejects_bad_digest() -> None:
    with pytest.raises(ValueError, match="SHA-256 digest"):
        TraceRecord(
            trace_id="trace.candidate",
            seed=0,
            session_id="s0",
            effect=0.5,
            trace_sha256="not-a-digest",
        )


def test_sign_agreement_is_unanimous_for_stable_trace() -> None:
    records = synthesize_trace_records(trace_id="trace.candidate", seeds=range(8))
    assert cross_seed_agreement(records, "sign-agreement") == 1.0


def test_agreement_refuses_undeclared_metric() -> None:
    records = synthesize_trace_records(trace_id="trace.candidate", seeds=range(8))
    with pytest.raises(ValueError, match="undeclared stability metric"):
        cross_seed_agreement(records, "z-score")


def test_agreement_is_identity_bound() -> None:
    a = synthesize_trace_records(trace_id="trace.one", seeds=range(4))
    b = synthesize_trace_records(trace_id="trace.two", seeds=range(4))
    with pytest.raises(ValueError, match="identity-bound"):
        cross_seed_agreement(a + b, "sign-agreement")


def test_rank_correlation_agreement_for_identical_rankings() -> None:
    ranking = (0, 1, 2, 3)
    records = tuple(
        TraceRecord(
            trace_id="trace.ranked",
            seed=seed,
            session_id="s0",
            effect=0.5,
            trace_sha256="b" * 64,
            ranking=ranking,
        )
        for seed in range(3)
    )
    assert cross_seed_agreement(records, "rank-correlation") == 1.0


# ---------------------------------------------------------------------------
# Verdict: fail-closed stability.
# ---------------------------------------------------------------------------


def test_verdict_stable_when_controls_dead() -> None:
    contract = build_default_contract()
    records = synthesize_trace_records(trace_id=contract.trace_id, seeds=range(8))
    verdict = _verdict(contract, records, dead_control_outcomes())
    assert verdict.decide() == "stable"
    verdict.assert_stable()
    assert len(verdict.sha256) == 64


def test_verdict_null_when_a_control_reproduces() -> None:
    contract = build_default_contract()
    records = synthesize_trace_records(trace_id=contract.trace_id, seeds=range(8))
    outcomes = (
        ControlOutcome(control="single-seed", agreement=0.95, reproduced=True),
        ControlOutcome(control="shuffled-seed", agreement=0.5, reproduced=False),
        ControlOutcome(control="permuted-trace", agreement=0.5, reproduced=False),
        ControlOutcome(control="label-shuffled", agreement=0.5, reproduced=False),
    )
    verdict = _verdict(contract, records, outcomes)
    assert verdict.decide() == "null"
    with pytest.raises(ValueError, match="not stable"):
        verdict.assert_stable()


def test_verdict_fails_closed_below_min_seeds() -> None:
    contract = build_default_contract()
    records = synthesize_trace_records(trace_id=contract.trace_id, seeds=range(4))
    with pytest.raises(ValueError, match="distinct seeds"):
        _verdict(contract, records, dead_control_outcomes())


def test_verdict_rejects_control_flag_inconsistency() -> None:
    contract = build_default_contract()
    records = synthesize_trace_records(trace_id=contract.trace_id, seeds=range(8))
    outcomes = (
        ControlOutcome(control="single-seed", agreement=0.95, reproduced=False),
        ControlOutcome(control="shuffled-seed", agreement=0.5, reproduced=False),
        ControlOutcome(control="permuted-trace", agreement=0.5, reproduced=False),
        ControlOutcome(control="label-shuffled", agreement=0.5, reproduced=False),
    )
    with pytest.raises(ValueError, match="disagrees with its agreement"):
        _verdict(contract, records, outcomes)


def test_verdict_rejects_foreign_record_identity() -> None:
    contract = build_default_contract()
    records = synthesize_trace_records(trace_id="trace.other", seeds=range(8))
    with pytest.raises(ValueError, match="bind to the contract trace identity"):
        _verdict(contract, records, dead_control_outcomes())


# ---------------------------------------------------------------------------
# Activation gate.
# ---------------------------------------------------------------------------


def test_gate_refuses_by_default() -> None:
    gate = MechanismLicenseGate()
    with pytest.raises(TraceStabilityRefusal, match="no stability receipt"):
        gate.authorize()


def test_gate_rejects_pre_granted_construction() -> None:
    with pytest.raises(ValueError, match="never pre-granted"):
        MechanismLicenseGate(license_granted=True)


def test_gate_refuses_null_verdict_receipt() -> None:
    gate = MechanismLicenseGate()
    receipt = LicenseReceipt(
        verdict_sha256="c" * 64,
        verdict_label="null",
        independent_confirmations=3,
        replication_min=2,
    )
    with pytest.raises(ValueError, match="null verdict cannot license"):
        gate.authorize(receipt)


def test_gate_refuses_underreplicated_receipt() -> None:
    gate = MechanismLicenseGate()
    receipt = LicenseReceipt(
        verdict_sha256="d" * 64,
        verdict_label="stable",
        independent_confirmations=1,
        replication_min=2,
    )
    with pytest.raises(ValueError, match="not independently replicated"):
        gate.authorize(receipt)


def test_gate_grants_on_stable_replicated_receipt() -> None:
    contract = build_default_contract()
    records = synthesize_trace_records(trace_id=contract.trace_id, seeds=range(8))
    verdict = _verdict(contract, records, dead_control_outcomes())
    receipt = LicenseReceipt(
        verdict_sha256=verdict.sha256,
        verdict_label=verdict.decide(),
        independent_confirmations=2,
        replication_min=2,
    )
    MechanismLicenseGate().authorize(receipt)  # does not raise


# ---------------------------------------------------------------------------
# Coverage record.
# ---------------------------------------------------------------------------


def test_coverage_lists_all_subquestions() -> None:
    cov = coverage()
    assert set(cov) == {"seed-stability", "session-stability", "control-death", "license-gate"}
    for bullets in cov.values():
        assert len(bullets) >= 2
