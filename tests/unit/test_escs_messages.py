from __future__ import annotations

from dataclasses import replace

import pytest

from mop.escs.messages import (
    ClaimFault,
    ClaimMessage,
    ClaimSchema,
    ClaimValidationContext,
    EpistemicStatus,
    EventClaimEvidence,
    EvidenceClass,
    SchemaRegistry,
    validate_claim,
)

STATE_V1 = "1" * 64


@pytest.fixture
def schema() -> ClaimSchema:
    return ClaimSchema(
        schema_id="mop.test.factor-claim",
        version=1,
        claim_types=frozenset({"factor_distribution", "action_distribution"}),
        payload_forms=frozenset({"probability-table", "spatial-graph", "motor-program"}),
        max_payload_bytes=1024,
    )


def make_message(
    schema: ClaimSchema,
    *,
    epistemic_status: EpistemicStatus = EpistemicStatus.INFERRED,
    branch_id: str = "branch:factual",
    payload_form: str = "probability-table",
    payload_bytes: bytes = b"{left:0.75,right:0.25}",
) -> ClaimMessage:
    return ClaimMessage.create(
        schema=schema,
        source_hypothesis_event_ids=("event:hypothesis/1",),
        referent_hypotheses=("referent:object/7",),
        branch_id=branch_id,
        factor_scope=("factor:motion",),
        claim_type="factor_distribution",
        epistemic_status=epistemic_status,
        supporting_event_ids=("event:observation/1",),
        producer_actor_id="actor:alpha",
        producer_state_version=STATE_V1,
        calibrated_confidence=0.7,
        created_tick=4,
        expiry_tick=8,
        predicted_utility=(0.2, -0.1),
        producer_operations=19,
        payload_form=payload_form,
        payload_bytes=payload_bytes,
    )


def make_context(
    *,
    now_tick: int = 5,
    branch_id: str = "branch:factual",
    hypothesis_status: EpistemicStatus = EpistemicStatus.INFERRED,
) -> ClaimValidationContext:
    return ClaimValidationContext(
        now_tick=now_tick,
        branch_id=branch_id,
        factual_branch_id="branch:factual",
        allowed_referents=frozenset({"referent:object/7"}),
        allowed_factor_scopes=frozenset({"factor:motion"}),
        event_evidence=(
            EventClaimEvidence(
                "event:hypothesis/1",
                "hypothesis",
                EvidenceClass.SCRIPTED_MECHANICS,
                branch_id,
                hypothesis_status,
                3,
            ),
            EventClaimEvidence(
                "event:observation/1",
                "observation",
                EvidenceClass.SCRIPTED_MECHANICS,
                "branch:factual",
                EpistemicStatus.OBSERVED_CANDIDATE,
                2,
            ),
        ),
        accepted_producer_state_versions=(("actor:alpha", (STATE_V1,)),),
    )


def test_claim_is_content_addressed_and_payload_representation_neutral(schema: ClaimSchema):
    first = make_message(schema, payload_form="spatial-graph", payload_bytes=b"graph-bytes")
    repeated = make_message(schema, payload_form="spatial-graph", payload_bytes=b"graph-bytes")
    other_form = make_message(schema, payload_form="motor-program", payload_bytes=b"program-bytes")

    assert first == repeated
    assert first.header.message_id != other_form.header.message_id
    assert first.integrity_valid() and other_form.integrity_valid()
    assert first.encoded_bytes == len(
        __import__("json").dumps(first.wire_payload(), sort_keys=True, separators=(",", ":")).encode()
    )


def test_receiver_validates_schema_digest_branch_referent_state_and_expiry(schema: ClaimSchema):
    message = make_message(schema)
    registry = SchemaRegistry((schema,))
    assert validate_claim(message, schemas=registry, context=make_context()).accepted

    mutated = {
        ClaimFault.INTEGRITY: replace(message, payload_bytes=b"corrupted"),
        ClaimFault.SCHEMA_DIGEST: replace(
            message, header=replace(message.header, claim_schema_digest="f" * 64)
        ),
        ClaimFault.BRANCH: replace(message, header=replace(message.header, branch_id="branch:unrelated")),
        ClaimFault.REFERENT: replace(
            message,
            header=replace(message.header, referent_hypotheses=("referent:unknown",)),
        ),
        ClaimFault.PRODUCER_STATE: replace(
            message, header=replace(message.header, producer_state_version="2" * 64)
        ),
        ClaimFault.FACTOR_SCOPE: replace(
            message, header=replace(message.header, factor_scope=("factor:unknown",))
        ),
    }
    for expected_fault, candidate in mutated.items():
        result = validate_claim(candidate, schemas=registry, context=make_context())
        assert not result.accepted
        assert expected_fault in result.faults

    expired = validate_claim(message, schemas=registry, context=make_context(now_tick=9))
    assert ClaimFault.EXPIRED in expired.faults


def test_missing_provenance_and_unknown_schemas_fail_closed(schema: ClaimSchema):
    message = make_message(schema)
    no_source_context = replace(
        make_context(),
        event_evidence=(
            EventClaimEvidence(
                "event:observation/1",
                "observation",
                EvidenceClass.SCRIPTED_MECHANICS,
                "branch:factual",
                EpistemicStatus.OBSERVED_CANDIDATE,
                2,
            ),
        ),
    )
    result = validate_claim(message, schemas=SchemaRegistry((schema,)), context=no_source_context)
    assert not result.accepted and ClaimFault.UNKNOWN_EVENT in result.faults

    unknown = validate_claim(message, schemas=SchemaRegistry(()), context=make_context())
    assert not unknown.accepted and ClaimFault.UNKNOWN_SCHEMA in unknown.faults
    with pytest.raises(ValueError, match="duplicate claim schema"):
        SchemaRegistry((schema, schema))

    wrong_kind = replace(
        make_context(),
        event_evidence=(
            EventClaimEvidence(
                "event:hypothesis/1",
                "observation",
                EvidenceClass.SCRIPTED_MECHANICS,
                "branch:factual",
                EpistemicStatus.INFERRED,
                3,
            ),
            make_context().event_evidence[1],
        ),
    )
    result = validate_claim(message, schemas=SchemaRegistry((schema,)), context=wrong_kind)
    assert ClaimFault.SOURCE_NOT_HYPOTHESIS in result.faults


def test_epistemic_laundering_and_factual_simulation_are_rejected(schema: ClaimSchema):
    inferred_as_observed = make_message(schema, epistemic_status=EpistemicStatus.OBSERVED_CANDIDATE)
    result = validate_claim(
        inferred_as_observed,
        schemas=SchemaRegistry((schema,)),
        context=make_context(hypothesis_status=EpistemicStatus.INFERRED),
    )
    assert ClaimFault.EPISTEMIC_LAUNDERING in result.faults

    simulated_source_as_inferred = make_message(schema, epistemic_status=EpistemicStatus.INFERRED)
    result = validate_claim(
        simulated_source_as_inferred,
        schemas=SchemaRegistry((schema,)),
        context=make_context(hypothesis_status=EpistemicStatus.SIMULATED),
    )
    assert ClaimFault.EPISTEMIC_LAUNDERING in result.faults

    simulated_on_fact = make_message(schema, epistemic_status=EpistemicStatus.SIMULATED)
    result = validate_claim(
        simulated_on_fact,
        schemas=SchemaRegistry((schema,)),
        context=make_context(hypothesis_status=EpistemicStatus.INFERRED),
    )
    assert ClaimFault.SIMULATION_ON_FACTUAL_BRANCH in result.faults


def test_counterfactual_claim_may_use_factual_evidence_but_not_an_unrelated_branch(
    schema: ClaimSchema,
):
    message = make_message(
        schema,
        epistemic_status=EpistemicStatus.SIMULATED,
        branch_id="branch:counterfactual/1",
    )
    context = make_context(
        branch_id="branch:counterfactual/1",
        hypothesis_status=EpistemicStatus.SIMULATED,
    )
    assert validate_claim(message, schemas=SchemaRegistry((schema,)), context=context).accepted

    unrelated = replace(
        context,
        event_evidence=(
            EventClaimEvidence(
                "event:hypothesis/1",
                "hypothesis",
                EvidenceClass.SCRIPTED_MECHANICS,
                "branch:counterfactual/2",
                EpistemicStatus.SIMULATED,
                3,
            ),
            context.event_evidence[1],
        ),
    )
    result = validate_claim(message, schemas=SchemaRegistry((schema,)), context=unrelated)
    assert ClaimFault.EVENT_BRANCH in result.faults


def test_oracle_evidence_taint_cannot_vanish_at_a_message_boundary(schema: ClaimSchema):
    message = make_message(schema)
    context = make_context()
    oracle_context = replace(
        context,
        event_evidence=(
            replace(context.event_evidence[0], evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE),
            context.event_evidence[1],
        ),
    )
    result = validate_claim(message, schemas=SchemaRegistry((schema,)), context=oracle_context)
    assert ClaimFault.EVIDENCE_CLASS_DOWNGRADE in result.faults

    oracle_message = ClaimMessage.create(
        schema=schema,
        source_hypothesis_event_ids=("event:hypothesis/1",),
        referent_hypotheses=("referent:object/7",),
        branch_id="branch:factual",
        factor_scope=("factor:motion",),
        claim_type="factor_distribution",
        epistemic_status=EpistemicStatus.INFERRED,
        supporting_event_ids=("event:observation/1",),
        producer_actor_id="actor:alpha",
        producer_state_version=STATE_V1,
        calibrated_confidence=0.5,
        created_tick=4,
        expiry_tick=8,
        predicted_utility=(0.1,),
        producer_operations=2,
        payload_form="probability-table",
        payload_bytes=b"oracle-tainted",
        evidence_class=EvidenceClass.ORACLE_NONPROMOTABLE,
    )
    assert validate_claim(
        oracle_message,
        schemas=SchemaRegistry((schema,)),
        context=oracle_context,
    ).accepted
