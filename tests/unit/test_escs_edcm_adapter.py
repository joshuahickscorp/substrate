from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mop.escs.accounting import FACTUAL_BRANCH, LifecycleLedger
from mop.escs.edcm_adapter import (
    PROPOSAL_CLAIM_SCHEMA,
    VERIFICATION_CLAIM_SCHEMA,
    AdapterActivationError,
    AdapterConfig,
    AdapterContractError,
    EDCMToESCSAdapter,
    VerifiedEDCMAuthority,
    load_verified_edcm_authority,
)
from mop.escs.events import EpistemicStatus, EvidenceClass
from mop.escs.ledger import EventLedger
from mop.escs.messages import (
    ClaimValidationContext,
    EventClaimEvidence,
    SchemaRegistry,
    validate_claim,
)
from mop.studies import edcm1_event_triggered_coalition as edcm


def observation(tick: int = 0, *, world_id: str = "unit-world") -> edcm.VisibleObservation:
    return edcm.VisibleObservation(
        world_id=world_id,
        event_id=edcm.canonical_sha256({"world_id": world_id, "tick": tick}),
        tick=tick,
        local_blocked=(0, 0, 0, 1),
        relative_goal=(2 - tick, 1),
        previous_action=0,
        previous_reward=-0.01,
        novelty_channels=(0, 0, 0, 0),
    )


def prepared_resolution(
    source: edcm.VisibleObservation,
    *,
    mode: str = "event_triggered",
) -> tuple[dict, edcm.CoalitionController, edcm.PreparedDecision, edcm.Resolution]:
    config = edcm.load_config()
    controller = edcm.CoalitionController(config, mode)
    prepared = controller.prepare(source)
    resolution = controller.resolve(prepared)
    return config, controller, prepared, resolution


def verified_authority(*, passed: bool = True) -> VerifiedEDCMAuthority:
    producer = {"path": "proof/producer.json", "bytes": 1, "sha256": "1" * 64}
    verification = {"path": "proof/verifier.json", "bytes": 1, "sha256": "2" * 64}
    return VerifiedEDCMAuthority._create(
        producer_file=producer,
        verification_file=verification,
        producer_receipt_sha256="3" * 64,
        verification_artifact_sha256="4" * 64,
        config_authority_sha256="5" * 64,
        implementation_authority_sha256="6" * 64,
        complementarity_gate_sha256="7" * 64,
        complementarity_passed=passed,
        terminal_execution_status="complete",
        verifier_mode=edcm.OFFICIAL_VERIFIER_MODE,
    )


def test_observation_translation_is_deterministic_factual_quiescent_and_fully_charged() -> None:
    adapter = EDCMToESCSAdapter()
    source = observation()
    source_work = edcm.AbstractWork(scalar_ops=2, bytes_hashed=3, bytes_serialized=4)

    first = adapter.translate_observation(source, source_work=source_work)
    repeated = adapter.translate_observation(source, source_work=source_work)

    assert first == repeated
    assert first.observation_event.branch_id == first.hypothesis_event.branch_id == FACTUAL_BRANCH
    assert first.hypothesis_event.epistemic_status is EpistemicStatus.OBSERVED_CANDIDATE
    assert first.hypothesis_event.calibrated_confidence == 0.0
    assert first.payload()["activation_enabled"] is False
    accounting = first.accounting
    assert accounting.source_payload_bytes > 0 and accounting.target_payload_bytes > 0
    assert accounting.adapter_bytes_serialized == (
        accounting.source_payload_bytes + accounting.target_payload_bytes
    )
    assert accounting.adapter_bytes_hashed == accounting.adapter_bytes_serialized
    assert accounting.work.total_work == accounting.source_work_units + accounting.adapter_operations
    ledger = LifecycleLedger()
    accounting.charge(
        ledger,
        start_tick=0,
        end_tick=0,
        causal_event_ids=(first.observation_event.event_id, first.hypothesis_event.event_id),
    )
    assert ledger.total == accounting.work
    with pytest.raises(ValueError, match="translation self-hash mismatch"):
        replace(first, translation_sha256="0" * 64)
    with pytest.raises(ValueError, match="serialization bytes are incomplete"):
        replace(accounting, target_payload_bytes=accounting.target_payload_bytes + 1)


def test_observation_rejects_future_evaluator_fields_and_silent_identity_mutation() -> None:
    adapter = EDCMToESCSAdapter()
    payload = dataclasses.asdict(observation())
    payload["hidden_change"] = True
    with pytest.raises(AdapterContractError, match="future/evaluator-only"):
        adapter.translate_observation(payload)

    mutated = replace(observation(), event_id="0" * 64)
    with pytest.raises(AdapterContractError, match="event identity mismatch"):
        adapter.translate_observation(mutated)

    nan_reward = replace(observation(), previous_reward=float("nan"))
    with pytest.raises(AdapterContractError, match="previous reward must be finite"):
        adapter.translate_observation(nan_reward)


def test_clean_specialist_resolution_maps_to_valid_claims_action_and_commitment() -> None:
    source = observation()
    _, _, prepared, resolution = prepared_resolution(source)
    adapter = EDCMToESCSAdapter()
    observed = adapter.translate_observation(source)

    translated = adapter.translate_decision(observed, prepared, resolution)
    repeated = adapter.translate_decision(observed, prepared, resolution)

    assert translated == repeated
    assert translated.commitment_event.branch_id == FACTUAL_BRANCH
    assert translated.action_intent.branch_id == str(FACTUAL_BRANCH)
    assert translated.action_intent.integrity_valid()
    assert all(message.integrity_valid() for message in translated.proposal_claims)
    assert all(
        message.header.expiry_tick == message.header.created_tick for message in translated.proposal_claims
    )
    assert [message.header.producer_actor_id for message in translated.proposal_claims] == [
        f"actor:edcm/{proposal.specialist_id}" for proposal in prepared.proposals
    ]
    assert translated.accounting.work.total_work == (
        translated.accounting.source_work_units + translated.accounting.adapter_operations
    )
    assert translated.accounting.work.actor_execution > 0
    assert translated.accounting.work.messages > 0
    assert set(dict(translated.accounting.source_work_buckets)) == {"actor_execution", "messages"}
    first_proposal = prepared.proposals[0]
    first_claim = translated.proposal_claims[0]
    context = ClaimValidationContext(
        now_tick=source.tick,
        branch_id=str(FACTUAL_BRANCH),
        factual_branch_id=str(FACTUAL_BRANCH),
        allowed_referents=frozenset({first_proposal.referent_id}),
        allowed_factor_scopes=frozenset(first_claim.header.factor_scope),
        event_evidence=(
            EventClaimEvidence(
                str(observed.observation_event.event_id),
                "observation",
                EvidenceClass.SCRIPTED_MECHANICS,
                str(FACTUAL_BRANCH),
                EpistemicStatus.OBSERVED_CANDIDATE,
                source.tick,
            ),
            EventClaimEvidence(
                str(observed.hypothesis_event.event_id),
                "hypothesis",
                EvidenceClass.SCRIPTED_MECHANICS,
                str(FACTUAL_BRANCH),
                EpistemicStatus.OBSERVED_CANDIDATE,
                source.tick,
            ),
        ),
        accepted_producer_state_versions=(
            (first_claim.header.producer_actor_id, (first_proposal.provenance.state_digest,)),
        ),
    )
    validation = validate_claim(
        first_claim,
        schemas=SchemaRegistry((PROPOSAL_CLAIM_SCHEMA,)),
        context=context,
    )
    assert validation.accepted


def test_verifier_round_maps_to_typed_claim_without_unbounded_rounds() -> None:
    source = observation()
    _, _, prepared, resolution = prepared_resolution(source, mode="always_on")
    adapter = EDCMToESCSAdapter()

    translated = adapter.translate_decision(adapter.translate_observation(source), prepared, resolution)

    assert prepared.activation.extra_round == (edcm.VERIFIER_ID,)
    assert translated.verification_claim is not None
    assert translated.verification_claim.integrity_valid()
    assert translated.verification_claim.header.claim_schema_digest == VERIFICATION_CLAIM_SCHEMA.digest
    with pytest.raises(AdapterContractError, match="exactly two bounded rounds"):
        AdapterConfig(max_reasoning_rounds=3)

    duplicate = edcm.ActivationRecord(
        ("reactive_spatial", "reactive_spatial"),
        (),
        "event",
        ("duplicate",),
    )
    bad_prepared = replace(
        prepared,
        activation=duplicate,
        active_ids=duplicate.initial,
        proposals=(prepared.proposals[0], prepared.proposals[0]),
    )
    with pytest.raises(AdapterContractError, match="duplicated, or noncanonical"):
        adapter.translate_decision(adapter.translate_observation(source), bad_prepared, resolution)


def test_message_mutation_and_lesion_resolution_cannot_cross_factual_adapter() -> None:
    source = observation()
    _, _, prepared, resolution = prepared_resolution(source)
    adapter = EDCMToESCSAdapter()
    observed = adapter.translate_observation(source)

    mutated_proposal = replace(
        prepared.proposals[0],
        proposed_action=(prepared.proposals[0].proposed_action + 1) % 4,
    )
    mutated_prepared = replace(
        prepared,
        proposals=(mutated_proposal, *prepared.proposals[1:]),
    )
    with pytest.raises(AdapterContractError, match="integrity mismatch"):
        adapter.translate_decision(observed, mutated_prepared, resolution)

    lesion = replace(
        resolution,
        delivered=(),
        chosen_message_id=None,
        verification=None,
        message_bytes=0,
        verifier_executed=False,
    )
    with pytest.raises(AdapterContractError, match="rejects lesion"):
        adapter.translate_decision(observed, prepared, lesion)


def test_commitment_then_visible_consequence_replays_in_escs_ledgers() -> None:
    source = observation()
    _, controller, prepared, resolution = prepared_resolution(source)
    adapter = EDCMToESCSAdapter()
    observed = adapter.translate_observation(source)
    decision = adapter.translate_decision(observed, prepared, resolution)
    feedback = edcm.PublicFeedback(source.event_id, 0, resolution.action, -0.01, False, False)
    after = replace(
        observation(1),
        previous_action=resolution.action,
        previous_reward=feedback.reward,
    )
    transition = edcm.VisibleTransition(source, resolution.action, feedback, after, False)
    update_work = controller.update(transition)
    successor = adapter.translate_observation(after)

    consequence = adapter.translate_consequence(
        decision,
        transition,
        update_work=update_work,
        successor_observation=successor,
    )

    events = EventLedger()
    events.append_batch(
        (
            observed.observation_event,
            observed.hypothesis_event,
            decision.commitment_event,
            successor.observation_event,
            consequence.consequence_event,
        )
    )
    assert events.verify() == []
    assert consequence.consequence_event.commitment_event_id == decision.commitment_event.event_id
    assert consequence.consequence_event.observed_outcome.value()["reward"] == feedback.reward
    work = LifecycleLedger()
    observed.accounting.charge(
        work,
        start_tick=0,
        end_tick=0,
        causal_event_ids=(observed.observation_event.event_id, observed.hypothesis_event.event_id),
    )
    decision.accounting.charge(
        work,
        start_tick=0,
        end_tick=0,
        causal_event_ids=(decision.commitment_event.event_id,),
    )
    consequence.accounting.charge(
        work,
        start_tick=1,
        end_tick=1,
        causal_event_ids=(consequence.consequence_event.event_id,),
    )
    expected = observed.accounting.work + decision.accounting.work + consequence.accounting.work
    assert work.total == expected
    assert work.verify(event_ids=set(events.event_ids)) == []


def test_evaluator_transition_wrong_action_and_world_identity_are_rejected() -> None:
    source = observation()
    _, _, prepared, resolution = prepared_resolution(source)
    adapter = EDCMToESCSAdapter()
    observed = adapter.translate_observation(source)
    decision = adapter.translate_decision(observed, prepared, resolution)
    feedback = edcm.PublicFeedback(source.event_id, 0, resolution.action, -0.01, False, False)
    after = replace(
        observation(1),
        previous_action=resolution.action,
        previous_reward=feedback.reward,
    )
    visible = edcm.VisibleTransition(source, resolution.action, feedback, after, False)
    evaluator = edcm.EvaluatorTransition(visible, True, 1, 2, "oracle-niche", True)
    with pytest.raises(AdapterContractError, match="evaluator-only"):
        adapter.translate_consequence(  # type: ignore[arg-type]
            decision,
            evaluator,
            update_work=edcm.AbstractWork(),
        )

    wrong_action = replace(visible, action=(resolution.action + 1) % 4)
    with pytest.raises(AdapterContractError, match="action/commitment mismatch"):
        adapter.translate_consequence(decision, wrong_action, update_work=edcm.AbstractWork())

    other_world = replace(
        observation(1, world_id="other-world"),
        previous_action=resolution.action,
        previous_reward=feedback.reward,
    )
    wrong_world = replace(visible, after=other_world)
    with pytest.raises(AdapterContractError, match="world identity"):
        adapter.translate_consequence(decision, wrong_world, update_work=edcm.AbstractWork())


def test_activation_requires_verified_current_authority_and_remains_disabled() -> None:
    no_authority = EDCMToESCSAdapter()
    assessment = no_authority.assess_activation()
    assert assessment.activation_enabled is False
    assert "verified-current-edcm-authority-missing" in assessment.blockers
    with pytest.raises(AdapterActivationError, match="verified EDCM result"):
        no_authority.require_activation()

    failed_gate = EDCMToESCSAdapter(verified_authority=verified_authority(passed=False))
    with pytest.raises(AdapterActivationError, match="complementarity gate"):
        failed_gate.require_activation()

    loaded_authority = verified_authority(passed=True)
    with pytest.raises(AdapterContractError, match="current-authority loader"):
        replace(loaded_authority, _validation_token=object())
    verified = EDCMToESCSAdapter(verified_authority=loaded_authority)
    assert verified.assess_activation().verified_current_authority is True
    with pytest.raises(AdapterActivationError, match="activation is disabled"):
        verified.activate()


def test_authority_loader_rejects_incompatible_producer_schema(tmp_path: Path) -> None:
    producer = tmp_path / "producer.json"
    verification = tmp_path / "verification.json"
    producer.write_text(json.dumps({"schema": "wrong"}))
    verification.write_text(json.dumps({"schema": "wrong"}))

    with pytest.raises(AdapterContractError, match="fields mismatch"):
        load_verified_edcm_authority(producer, verification)


def test_authority_loader_rejects_ambiguous_json_and_symlink_sources(tmp_path: Path) -> None:
    producer = tmp_path / "producer.json"
    verification = tmp_path / "verification.json"
    verification.write_text('{"schema":"wrong"}')

    producer.write_text('{"schema":"wrong","schema":"also-wrong"}')
    with pytest.raises(AdapterContractError, match="duplicate JSON field 'schema'"):
        load_verified_edcm_authority(producer, verification)

    producer.write_text('{"schema":NaN}')
    with pytest.raises(AdapterContractError, match="nonfinite JSON constant 'NaN'"):
        load_verified_edcm_authority(producer, verification)

    producer.write_text('{"schema":"wrong"}')
    producer_link = tmp_path / "producer-link.json"
    producer_link.symlink_to(producer)
    with pytest.raises(AdapterContractError, match="regular non-symlink file"):
        load_verified_edcm_authority(producer_link, verification)
