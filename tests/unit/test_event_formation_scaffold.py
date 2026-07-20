from __future__ import annotations

import pytest

from mop.mechanisms.event_formation_scaffold import (
    CLAIM_SCOPE,
    REQUIRED_CONTROLS,
    SCIENTIFIC_CAPABILITY_CLAIM,
    UNTRAINED_CONTROLS,
    ActivationReceipt,
    ControlLedgerContract,
    EventFormationActivationGate,
    EventFormationRefusal,
    EventUtilityVerdict,
    MatchedBudget,
    OracleHeadroomContract,
    RelationalEventContract,
    TemporalEventBindingContract,
    assert_control_ledger,
    build_hypothetical_useful_verdict,
    build_x0_strong_null_verdict,
    default_matched_budget,
    mint_receipt,
    synthesize_relational_episode,
)

EVENT_FORMATION_SCHEMA = "mop-event-formation/v1"


def test_module_declares_no_capability_claim() -> None:
    assert SCIENTIFIC_CAPABILITY_CLAIM is False
    assert CLAIM_SCOPE == "deterministic programmatic mechanics only; no capability or natural-data claim"


def test_control_ledger_digest_is_stable() -> None:
    ledger = ControlLedgerContract()
    assert ledger.digest() == ControlLedgerContract().digest()
    assert len(ledger.digest()) == 64


def test_control_ledger_rejects_membership_or_order_drift() -> None:
    with pytest.raises(ValueError, match="membership or order drift"):
        ControlLedgerContract(
            controls=("wrong-event", "wrong-time", "appearance-only", "stateless-delayed-trigger")
        )
    with pytest.raises(EventFormationRefusal):
        assert_control_ledger(("wrong-time", "wrong-event"))


def test_control_ledger_rejects_widened_claim_scope() -> None:
    with pytest.raises(ValueError, match="claim scope cannot be widened"):
        ControlLedgerContract(claim_scope="a useful event was demonstrated")


def test_relational_event_valid_and_digest_stable() -> None:
    contract = RelationalEventContract(
        event_id="event.rel.1",
        relation="supports",
        entity_refs=("entity.a", "entity.b"),
    )
    assert contract.digest() == contract.digest()
    assert contract.payload()["scalar_only"] is False


def test_relational_event_refuses_scalar_only() -> None:
    with pytest.raises(ValueError, match="cannot be a single scalar"):
        RelationalEventContract(
            event_id="event.rel.2",
            relation="supports",
            entity_refs=("entity.a", "entity.b"),
            scalar_only=True,
        )


def test_relational_event_needs_two_distinct_entities() -> None:
    with pytest.raises(ValueError, match="at least two entities"):
        RelationalEventContract(event_id="event.rel.3", relation="contains", entity_refs=("entity.a",))
    with pytest.raises(ValueError, match="must be unique"):
        RelationalEventContract(
            event_id="event.rel.4", relation="contains", entity_refs=("entity.a", "entity.a")
        )


def test_temporal_binding_valid() -> None:
    contract = TemporalEventBindingContract(
        event_id="event.rel.1",
        clock_id="clock.1",
        window_start_tick=10,
        window_end_tick=20,
        wrong_time_tick=40,
        wrong_event_id="event.decoy",
    )
    assert contract.window_ticks == 10
    assert contract.controls == ("wrong-time", "wrong-event")


def test_temporal_binding_refuses_wrong_time_inside_window() -> None:
    with pytest.raises(ValueError, match="outside the binding window"):
        TemporalEventBindingContract(
            event_id="event.rel.1",
            clock_id="clock.1",
            window_start_tick=10,
            window_end_tick=20,
            wrong_time_tick=15,
            wrong_event_id="event.decoy",
        )


def test_temporal_binding_refuses_same_wrong_event() -> None:
    with pytest.raises(ValueError, match="different event"):
        TemporalEventBindingContract(
            event_id="event.rel.1",
            clock_id="clock.1",
            window_start_tick=10,
            window_end_tick=20,
            wrong_time_tick=40,
            wrong_event_id="event.rel.1",
        )


def test_temporal_binding_refuses_empty_window() -> None:
    with pytest.raises(ValueError, match="end strictly after"):
        TemporalEventBindingContract(
            event_id="event.rel.1",
            clock_id="clock.1",
            window_start_tick=20,
            window_end_tick=20,
            wrong_time_tick=40,
            wrong_event_id="event.decoy",
        )


def test_oracle_headroom_valid_and_measures_savings() -> None:
    oracle = OracleHeadroomContract(
        oracle_id="oracle.x",
        oracle_utility=1.0,
        baseline_utility=0.3,
        full_charged_compute=1.0,
        oracle_charged_compute=0.27,
        headroom_min=0.2,
        measured=True,
    )
    assert oracle.utility_headroom() == pytest.approx(0.7)
    assert oracle.oracle_savings_fraction() == pytest.approx(0.73)


def test_oracle_headroom_refuses_unmeasured() -> None:
    with pytest.raises(ValueError, match="must be measured"):
        OracleHeadroomContract(
            oracle_id="oracle.x",
            oracle_utility=1.0,
            baseline_utility=0.3,
            full_charged_compute=1.0,
            oracle_charged_compute=0.27,
            headroom_min=0.2,
            measured=False,
        )


def test_oracle_headroom_refuses_absent_headroom() -> None:
    with pytest.raises(ValueError, match="no real headroom"):
        OracleHeadroomContract(
            oracle_id="oracle.x",
            oracle_utility=0.35,
            baseline_utility=0.30,
            full_charged_compute=1.0,
            oracle_charged_compute=0.27,
            headroom_min=0.20,
            measured=True,
        )


def test_matched_budget_must_be_non_vacuous() -> None:
    with pytest.raises(ValueError, match="non-vacuous"):
        MatchedBudget(relational_ops=0, temporal_ops=1, trigger_evals=1, memory_bytes=1)


def test_x0_strong_null_never_claims_useful_event() -> None:
    verdict = build_x0_strong_null_verdict()
    assert verdict.claims_useful_event() is False
    assert verdict.refutes_x0_strong_null() is False
    assert verdict.utility_preserved() is False
    assert verdict.beats_untrained_utility() is False


def test_verdict_requires_matched_cost() -> None:
    oracle = OracleHeadroomContract(
        oracle_id="oracle.x",
        oracle_utility=1.0,
        baseline_utility=0.3,
        full_charged_compute=1.0,
        oracle_charged_compute=0.27,
        headroom_min=0.2,
        measured=True,
    )
    with pytest.raises(ValueError, match="must require matched charged-compute cost"):
        EventUtilityVerdict(
            candidate_id="candidate.x",
            oracle=oracle,
            utility_candidate=0.9,
            utility_floor=0.9,
            utility_by_control={"appearance-only": 0.4, "stateless-delayed-trigger": 0.35},
            charged_compute_candidate=0.3,
            charged_compute_by_control={"appearance-only": 0.8, "stateless-delayed-trigger": 0.45},
            budget=default_matched_budget(),
            matched_cost_required=False,
        )


def test_verdict_refuses_incomplete_control_coverage() -> None:
    oracle = OracleHeadroomContract(
        oracle_id="oracle.x",
        oracle_utility=1.0,
        baseline_utility=0.3,
        full_charged_compute=1.0,
        oracle_charged_compute=0.27,
        headroom_min=0.2,
        measured=True,
    )
    with pytest.raises(ValueError, match="must cover exactly the untrained controls"):
        EventUtilityVerdict(
            candidate_id="candidate.x",
            oracle=oracle,
            utility_candidate=0.9,
            utility_floor=0.9,
            utility_by_control={"appearance-only": 0.4},
            charged_compute_candidate=0.3,
            charged_compute_by_control={"appearance-only": 0.8, "stateless-delayed-trigger": 0.45},
            budget=default_matched_budget(),
        )


def test_verdict_fails_when_compute_not_cut_vs_both_controls() -> None:
    verdict = build_hypothetical_useful_verdict()
    assert verdict.claims_useful_event() is True
    broken = EventUtilityVerdict(
        candidate_id=verdict.candidate_id,
        oracle=verdict.oracle,
        utility_candidate=verdict.utility_candidate,
        utility_floor=verdict.utility_floor,
        utility_by_control=dict(verdict.utility_by_control),
        charged_compute_candidate=verdict.charged_compute_candidate,
        charged_compute_by_control={"appearance-only": 0.80, "stateless-delayed-trigger": 0.20},
        budget=verdict.budget,
    )
    assert broken.compute_cut_vs_both_untrained() is False
    assert broken.claims_useful_event() is False


def test_hypothetical_useful_verdict_passes_all_gates() -> None:
    verdict = build_hypothetical_useful_verdict()
    assert verdict.utility_preserved() is True
    assert verdict.beats_untrained_utility() is True
    assert verdict.compute_cut_vs_both_untrained() is True
    assert verdict.claims_useful_event() is True
    assert verdict.refutes_x0_strong_null() is True


def test_verdict_widened_claim_scope_rejected() -> None:
    verdict = build_x0_strong_null_verdict()
    with pytest.raises(ValueError, match="claim scope cannot be widened"):
        EventUtilityVerdict(
            candidate_id="candidate.x",
            oracle=verdict.oracle,
            utility_candidate=0.9,
            utility_floor=0.9,
            utility_by_control={"appearance-only": 0.4, "stateless-delayed-trigger": 0.35},
            charged_compute_candidate=0.3,
            charged_compute_by_control={"appearance-only": 0.8, "stateless-delayed-trigger": 0.45},
            budget=default_matched_budget(),
            claim_scope="a useful event was demonstrated",
        )


def test_activation_gate_refuses_by_default() -> None:
    gate = EventFormationActivationGate()
    with pytest.raises(EventFormationRefusal):
        gate.authorize()
    with pytest.raises(EventFormationRefusal):
        gate.authorize_local()


def test_activation_gate_rejects_self_permission() -> None:
    with pytest.raises(ValueError, match="cannot be self-permitted"):
        EventFormationActivationGate(activation_permitted=True)


def test_activation_gate_refuses_x0_null_receipt() -> None:
    gate = EventFormationActivationGate()
    receipt = mint_receipt(build_x0_strong_null_verdict(), license_id="lic.x0", independent_replications=5)
    assert receipt.claims_useful_event is False
    with pytest.raises(ValueError, match="does not claim a useful event"):
        gate.authorize(receipt)


def test_activation_gate_refuses_under_replicated_receipt() -> None:
    gate = EventFormationActivationGate()
    receipt = mint_receipt(
        build_hypothetical_useful_verdict(), license_id="lic.ok", independent_replications=1
    )
    assert receipt.claims_useful_event is True
    with pytest.raises(ValueError, match="fewer than the required independent replications"):
        gate.authorize(receipt)


def test_activation_gate_opens_for_earned_replicated_receipt() -> None:
    gate = EventFormationActivationGate()
    receipt = mint_receipt(
        build_hypothetical_useful_verdict(), license_id="lic.ok", independent_replications=3
    )
    gate.authorize(receipt)  # must not raise


def test_episode_is_deterministic_under_seed() -> None:
    first = synthesize_relational_episode(7)
    second = synthesize_relational_episode(7)
    assert first.digest() == second.digest()


def test_episode_differs_across_seeds() -> None:
    assert synthesize_relational_episode(1).digest() != synthesize_relational_episode(2).digest()


def test_episode_refuses_bad_arguments() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        synthesize_relational_episode(-1)
    with pytest.raises(ValueError, match="at least two entities"):
        synthesize_relational_episode(0, num_entities=1)


def test_receipt_digest_stable_and_scope_locked() -> None:
    receipt = ActivationReceipt(
        license_id="lic.a",
        verdict_digest="0" * 64,
        claims_useful_event=False,
        independent_replications=2,
    )
    assert receipt.digest() == receipt.digest()
    with pytest.raises(ValueError, match="claim scope cannot be widened"):
        ActivationReceipt(
            license_id="lic.a",
            verdict_digest="0" * 64,
            claims_useful_event=False,
            independent_replications=2,
            claim_scope="widened",
        )


def test_required_controls_partition_into_untrained() -> None:
    assert set(UNTRAINED_CONTROLS) <= set(REQUIRED_CONTROLS)
    assert REQUIRED_CONTROLS[-2:] == UNTRAINED_CONTROLS
