from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from mop.escs.accounting import FACTUAL_BRANCH, LifecycleLedger, WorkVector
from mop.escs.causal_resource_monitor import (
    MAX_HARD_EVENT_ENTRIES,
    MONITOR_CLAIM_SCHEMA,
    MONITOR_ID,
    MonitorConfig,
    MonitorContractError,
    MonitorControl,
    analyze_snapshots,
)
from mop.escs.events import (
    EpistemicStatus,
    HypothesisEvent,
    HypothesisOrigin,
    ObservationEvent,
)
from mop.escs.ledger import EventLedger
from mop.escs.messages import (
    ClaimMessage,
    ClaimValidationContext,
    EventClaimEvidence,
    SchemaRegistry,
    validate_claim,
)
from mop.substrate.events import BranchRef, canonical_sha256


def _pair_contract_sha256(pair_id: str) -> str:
    return canonical_sha256(
        {
            "schema": "mop-escs-causal-resource-pair/v1",
            "pair_id": pair_id,
            "arms": ["observational", "resource_control"],
        }
    )


def _observation(index: int) -> ObservationEvent:
    return ObservationEvent.create(
        raw_packet_or_delta_refs=(f"packet:test/{index}",),
        adapter_version="test-v1",
        sensor_scope={"sensor": f"resource-{index}"},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=1),
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance={"source": "unit-test"},
        measured_creation_cost=WorkVector(event_formation=1),
    )


def _hypothesis(
    observation: ObservationEvent,
    *,
    branch: BranchRef = FACTUAL_BRANCH,
    pair_id: str = "pair:test/0",
) -> HypothesisEvent:
    simulated = branch != FACTUAL_BRANCH
    return HypothesisEvent.create(
        origin=HypothesisOrigin.VERIFIER,
        epistemic_status=(EpistemicStatus.SIMULATED if simulated else EpistemicStatus.INFERRED),
        referent_hypotheses={"candidates": ["resource:escs-lifecycle"]},
        factor_change_distribution={"resource_pressure": {"low": 0.5, "high": 0.5}},
        decision_relevance_distribution={"relevant": 0.5, "irrelevant": 0.5},
        reducibility_distribution={"reducible": 0.5, "irreducible": 0.5},
        supporting_event_ids=(observation.event_id,),
        calibrated_confidence=0.0,
        abstention_reason=None,
        predicted_value_of_further_computation=0.0,
        causal_parent_ids=(observation.event_id,),
        counterfactual_branch_id=branch,
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance={
            "source": "unit-test",
            "causal_resource_monitor_pair": {
                "pair_id": pair_id,
                "arm": "resource_control" if simulated else "observational",
                "contract_sha256": _pair_contract_sha256(pair_id),
            },
        },
        measured_creation_cost=WorkVector(event_formation=1),
    )


def _snapshots(*, same_parent: bool = True, matched_pair: bool = True) -> tuple[dict, dict]:
    events = EventLedger()
    observation = _observation(0)
    events.append(observation)
    alternate_parent = observation
    if not same_parent:
        alternate_parent = _observation(1)
        events.append(alternate_parent)
    factual = _hypothesis(observation)
    simulated = _hypothesis(
        alternate_parent,
        branch=BranchRef("branch:simulated-resource-control"),
        pair_id="pair:test/0" if matched_pair else "pair:test/unrelated",
    )
    events.append(factual)
    events.append(simulated)

    lifecycle = LifecycleLedger()
    lifecycle.charge(
        owner="fixture",
        reason="factual-low-work",
        work=WorkVector(indexing_and_graph_maintenance=3),
        start_tick=1,
        end_tick=2,
        causal_event_ids=(factual.event_id,),
    )
    lifecycle.charge(
        owner="fixture",
        reason="simulated-high-work",
        work=WorkVector(indexing_and_graph_maintenance=15),
        start_tick=1,
        end_tick=2,
        branch_id=simulated.branch_id,
        causal_event_ids=(simulated.event_id,),
    )
    lifecycle.charge(
        owner="fixture",
        reason="factual-retained-anomaly",
        work=WorkVector(indexing_and_graph_maintenance=20, retained_byte_time=256),
        start_tick=2,
        end_tick=3,
        causal_event_ids=(factual.event_id,),
    )
    return events.payload(), lifecycle.payload()


def _claim_payload(message) -> dict:
    return json.loads(message.payload_bytes)


def test_monitor_is_pure_deterministic_and_authority_free() -> None:
    event_snapshot, lifecycle_snapshot = _snapshots()
    before_events = copy.deepcopy(event_snapshot)
    before_lifecycle = copy.deepcopy(lifecycle_snapshot)

    first = analyze_snapshots(event_snapshot, lifecycle_snapshot, observed_through_tick=4)
    second = analyze_snapshots(event_snapshot, lifecycle_snapshot, observed_through_tick=4)

    assert first.payload() == second.payload()
    assert event_snapshot == before_events
    assert lifecycle_snapshot == before_lifecycle
    assert first.result_sha256 == canonical_sha256(first.payload(include_digest=False))
    assert {message.header.claim_type for message in first.claims} == {
        "resource_anomaly",
        "same_parent_resource_contrast",
    }
    assert all(message.integrity_valid() for message in first.claims)
    assert all(message.header.created_tick == first.observed_through_tick for message in first.claims)
    payload = first.payload()
    assert payload["activation_enabled"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert payload["monitor_work_charge_applied"] is False
    assert payload["payload_semantics_read"] is False
    assert payload["official_run"] is False
    assert payload["energy_measured"] is False
    assert set(payload["authority"].values()) == {False}
    for name in ("dispatch", "commit", "mutate", "trigger", "activate"):
        assert not hasattr(first, name)
    for message in first.claims:
        claim = _claim_payload(message)
        assert set(claim["authority"].values()) == {False}
        assert claim["activation_enabled"] is False
        assert claim["scientific_promotion_allowed"] is False
        assert claim["payload_semantics_read"] is False
        assert message.header.predicted_utility == ()
        assert message.header.calibrated_confidence == 0.0


def test_monitor_claims_validate_with_shared_message_contract() -> None:
    event_snapshot, lifecycle_snapshot = _snapshots()
    result = analyze_snapshots(event_snapshot, lifecycle_snapshot, observed_through_tick=4)
    ledger = EventLedger.from_payload(event_snapshot)
    evidence = tuple(
        EventClaimEvidence(
            event_id=str(event.event_id),
            event_kind=event.kind.value,
            evidence_class=event.evidence_class,
            branch_id=str(event.branch_id),
            epistemic_status=(
                event.epistemic_status
                if isinstance(event, HypothesisEvent)
                else EpistemicStatus.OBSERVED_CANDIDATE
            ),
            created_tick=event.envelope.clock_end_tick,
        )
        for event in ledger.events
    )
    schemas = SchemaRegistry((MONITOR_CLAIM_SCHEMA,))
    for message in result.claims:
        context = ClaimValidationContext(
            now_tick=message.header.created_tick,
            branch_id=message.header.branch_id,
            factual_branch_id=str(FACTUAL_BRANCH),
            allowed_referents=frozenset({"resource:escs-lifecycle"}),
            allowed_factor_scopes=frozenset(
                {"resource:abstract-work", "resource:retained-byte-time"}
            ),
            event_evidence=evidence,
            accepted_producer_state_versions=(
                (MONITOR_ID, (message.header.producer_state_version,)),
            ),
        )
        assert validate_claim(message, schemas=schemas, context=context).accepted is True


def test_configuration_cannot_grant_authority_or_escape_hard_bounds() -> None:
    with pytest.raises(MonitorContractError, match="activation"):
        MonitorConfig(activation_enabled=True)
    with pytest.raises(MonitorContractError, match="scientific promotion"):
        MonitorConfig(scientific_promotion_allowed=True)
    with pytest.raises(MonitorContractError, match="hard bound"):
        MonitorConfig(max_event_entries=MAX_HARD_EVENT_ENTRIES + 1)
    with pytest.raises(MonitorContractError, match="bounded analysis horizon"):
        MonitorConfig(window_ticks=1, max_windows=1, stale_ticks=2)

    event_snapshot, lifecycle_snapshot = _snapshots()
    result = analyze_snapshots(event_snapshot, lifecycle_snapshot, observed_through_tick=4)
    with pytest.raises(MonitorContractError, match="activation-disabled"):
        replace(result, activation_enabled=True)
    with pytest.raises(MonitorContractError, match="observation-only accounting bucket"):
        replace(result, monitor_work=WorkVector(dispatch_and_exploration=1))

    original = result.claims[0]
    header = original.header
    foreign_producer = ClaimMessage.create(
        schema=MONITOR_CLAIM_SCHEMA,
        source_hypothesis_event_ids=header.source_hypothesis_event_ids,
        referent_hypotheses=header.referent_hypotheses,
        branch_id=header.branch_id,
        factor_scope=header.factor_scope,
        claim_type=header.claim_type,
        epistemic_status=header.epistemic_status,
        supporting_event_ids=header.supporting_event_ids,
        producer_actor_id="monitor:foreign/v1",
        producer_state_version=header.producer_state_version,
        calibrated_confidence=header.calibrated_confidence,
        created_tick=header.created_tick,
        expiry_tick=header.expiry_tick,
        predicted_utility=header.predicted_utility,
        producer_operations=header.producer_operations,
        payload_form=header.payload_form,
        payload_bytes=original.payload_bytes,
        evidence_class=header.evidence_class,
    )
    with pytest.raises(MonitorContractError, match="producer drift"):
        replace(result, claims=(foreign_producer,))


def test_future_evaluator_and_bounded_work_fail_closed() -> None:
    event_snapshot, lifecycle_snapshot = _snapshots()
    leaked = copy.deepcopy(event_snapshot)
    leaked["entries"][0]["event"]["body"]["sensor_scope"]["value"]["evaluator_label"] = "fail"
    with pytest.raises(MonitorContractError, match="evaluator/future"):
        analyze_snapshots(leaked, lifecycle_snapshot, observed_through_tick=4)

    with pytest.raises(MonitorContractError, match="future-dated"):
        analyze_snapshots(event_snapshot, lifecycle_snapshot, observed_through_tick=1)
    with pytest.raises(MonitorContractError, match="entry cap"):
        analyze_snapshots(
            event_snapshot,
            lifecycle_snapshot,
            observed_through_tick=4,
            config=MonitorConfig(max_event_entries=1),
        )
    with pytest.raises(MonitorContractError, match="work-unit cap"):
        analyze_snapshots(
            event_snapshot,
            lifecycle_snapshot,
            observed_through_tick=4,
            config=MonitorConfig(max_work_units=1),
        )


def test_noisy_poison_stale_and_shuffle_controls_are_exact() -> None:
    event_snapshot, lifecycle_snapshot = _snapshots()
    results = {
        control: analyze_snapshots(
            event_snapshot,
            lifecycle_snapshot,
            observed_through_tick=4,
            control=control,
        )
        for control in MonitorControl
    }

    clean = results[MonitorControl.CLEAN]
    assert all(row.source_charge_sha256 == row.target_charge_sha256 for row in clean.control_trace)
    assert all(row.effective_total_work == row.raw_total_work for row in clean.control_trace)

    noisy = results[MonitorControl.NOISY]
    for row in noisy.control_trace:
        expected = 1 + int(row.target_charge_sha256[:8], 16) % 2
        assert row.transformation == f"noisy:+{expected}"
        assert row.effective_total_work == row.raw_total_work + expected
        assert row.source_charge_sha256 == row.target_charge_sha256

    poison = results[MonitorControl.POISON]
    poisoned = [row for row in poison.control_trace if row.transformation.startswith("poison:+")]
    assert len(poisoned) == 1
    assert poisoned[0].effective_total_work == poisoned[0].raw_total_work + 16
    assert poisoned[0].target_charge_sha256 == min(
        row.target_charge_sha256 for row in poison.control_trace
    )

    stale = results[MonitorControl.STALE]
    assert stale.analysis_window == (0, 0)
    assert stale.claims == ()
    assert all(row.included is False for row in stale.control_trace)
    assert all(row.transformation == "stale:excluded-current" for row in stale.control_trace)

    shuffled = results[MonitorControl.SHUFFLE]
    target_digests = [row.target_charge_sha256 for row in shuffled.control_trace]
    assert [row.source_charge_sha256 for row in shuffled.control_trace] == (
        target_digests[1:] + target_digests[:1]
    )
    raw_by_digest = {row.target_charge_sha256: row.raw_total_work for row in shuffled.control_trace}
    retained_by_digest = {
        row.target_charge_sha256: row.raw_retained_byte_time for row in shuffled.control_trace
    }
    assert all(
        row.effective_total_work == raw_by_digest[row.source_charge_sha256]
        for row in shuffled.control_trace
    )
    assert all(
        row.effective_retained_byte_time == retained_by_digest[row.source_charge_sha256]
        for row in shuffled.control_trace
    )
    assert all(row.transformation == "shuffle:rotate-left-1" for row in shuffled.control_trace)

    for control, result in results.items():
        assert result.control is control
        assert result.payload()["control"] == control.value
        assert set(result.payload()["authority"].values()) == {False}
        assert all(_claim_payload(message)["control"] == control.value for message in result.claims)


def test_poisoned_or_reordered_snapshot_bytes_are_rejected_before_claims() -> None:
    event_snapshot, lifecycle_snapshot = _snapshots()
    poisoned = copy.deepcopy(lifecycle_snapshot)
    poisoned["entries"][0]["work"]["indexing_and_graph_maintenance"] += 1
    with pytest.raises(MonitorContractError, match="snapshot replay failed"):
        analyze_snapshots(event_snapshot, poisoned, observed_through_tick=4)

    shuffled = copy.deepcopy(lifecycle_snapshot)
    shuffled["entries"][0], shuffled["entries"][1] = (
        shuffled["entries"][1],
        shuffled["entries"][0],
    )
    with pytest.raises(MonitorContractError, match="snapshot replay failed"):
        analyze_snapshots(event_snapshot, shuffled, observed_through_tick=4)


def test_cross_branch_or_pre_event_lifecycle_provenance_is_rejected() -> None:
    events = EventLedger()
    observation = _observation(0)
    events.append(observation)
    simulated = _hypothesis(
        observation,
        branch=BranchRef("branch:simulated-resource-control"),
    )
    events.append(simulated)

    wrong_branch = LifecycleLedger()
    wrong_branch.charge(
        owner="fixture",
        reason="cross-branch",
        work=WorkVector(indexing_and_graph_maintenance=20),
        start_tick=1,
        end_tick=2,
        causal_event_ids=(simulated.event_id,),
    )
    with pytest.raises(MonitorContractError, match="unauthorized event branch"):
        analyze_snapshots(events.payload(), wrong_branch.payload(), observed_through_tick=2)

    early = LifecycleLedger()
    early.charge(
        owner="fixture",
        reason="pre-event",
        work=WorkVector(indexing_and_graph_maintenance=20),
        start_tick=0,
        end_tick=1,
        branch_id=simulated.branch_id,
        causal_event_ids=(simulated.event_id,),
    )
    with pytest.raises(MonitorContractError, match="predates its causal event"):
        analyze_snapshots(events.payload(), early.payload(), observed_through_tick=1)


def test_same_parent_contrast_is_required_for_causal_claim() -> None:
    event_snapshot, lifecycle_snapshot = _snapshots(same_parent=False)
    result = analyze_snapshots(event_snapshot, lifecycle_snapshot, observed_through_tick=4)
    assert "resource_anomaly" in {message.header.claim_type for message in result.claims}
    assert "same_parent_resource_contrast" not in {
        message.header.claim_type for message in result.claims
    }

    same_parent, lifecycle = _snapshots(matched_pair=False)
    unrelated = analyze_snapshots(same_parent, lifecycle, observed_through_tick=4)
    assert "same_parent_resource_contrast" not in {
        message.header.claim_type for message in unrelated.claims
    }


def test_high_work_without_hypothesis_provenance_abstains() -> None:
    event_ledger = EventLedger()
    observation = _observation(0)
    event_ledger.append(observation)
    lifecycle = LifecycleLedger()
    lifecycle.charge(
        owner="fixture",
        reason="observation-only-high-work",
        work=WorkVector(indexing_and_graph_maintenance=100),
        start_tick=0,
        end_tick=1,
        causal_event_ids=(observation.event_id,),
    )
    result = analyze_snapshots(
        event_ledger.payload(),
        lifecycle.payload(),
        observed_through_tick=1,
    )
    assert result.claims == ()
    assert result.abstentions == ("resource-anomaly-without-same-branch-hypothesis",)


def test_shuffle_requires_a_nonidentity_pair() -> None:
    event_ledger = EventLedger()
    observation = _observation(0)
    event_ledger.append(observation)
    hypothesis = _hypothesis(observation)
    event_ledger.append(hypothesis)
    lifecycle = LifecycleLedger()
    lifecycle.charge(
        owner="fixture",
        reason="single-row",
        work=WorkVector(indexing_and_graph_maintenance=20),
        start_tick=1,
        end_tick=1,
        causal_event_ids=(hypothesis.event_id,),
    )
    with pytest.raises(MonitorContractError, match="at least two"):
        analyze_snapshots(
            event_ledger.payload(),
            lifecycle.payload(),
            observed_through_tick=1,
            control=MonitorControl.SHUFFLE,
        )
