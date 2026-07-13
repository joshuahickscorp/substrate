from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mop.config import REPO_ROOT
from mop.escs.accounting import WorkVector
from mop.escs.perspective_registry import (
    PerspectiveCandidateRegistry,
    load_perspective_candidate_registry,
)
from mop.escs.topology_grammar import (
    ActorSlot,
    ConstructionMutation,
    GrammarStatus,
    MutationKind,
    OperatorPrimitive,
    StatePrimitive,
    TopologyGrammar,
    TopologyMutation,
    TopologySnapshot,
    TopologyTransaction,
    apply_topology_mutation,
    assess_topology_transaction,
    load_topology_grammar,
    verify_candidate_registry,
    verify_freeze_authority,
)
from mop.substrate.events import canonical_sha256

GRAMMAR_PATH = REPO_ROOT / "configs/experiment/escs_g0_topology_grammar.json"
REGISTRY_PATH = REPO_ROOT / "configs/experiment/escs_perspective_candidates.json"
STATE_A = "a" * 64
STATE_B = "b" * 64


def _mutation(kind: MutationKind, **parameters: object) -> TopologyMutation:
    return TopologyMutation.create(
        kind=kind,
        parameters=parameters,
        declared_work=WorkVector(indexing_and_graph_maintenance=7),
        retained_state_bytes_delta=32,
    )


def _base() -> TopologySnapshot:
    return TopologySnapshot(
        actors=(ActorSlot("actor:primary", "visual_reasoning", STATE_A, True),),
        routing_subscriptions=(("actor:primary", "vision"),),
        factor_scopes=(("actor:primary", "object"),),
    )


def _registry() -> PerspectiveCandidateRegistry:
    return load_perspective_candidate_registry(REGISTRY_PATH)


def _rehash(payload: dict[str, object]) -> dict[str, object]:
    core = dict(payload)
    core.pop("grammar_sha256", None)
    payload["grammar_sha256"] = canonical_sha256(core)
    return payload


def _frozen_grammar(authority_sha256: str) -> TopologyGrammar:
    payload = json.loads(GRAMMAR_PATH.read_text())
    payload["status"] = "frozen"
    payload["activation_enabled"] = True
    payload["construction_language"]["implementation_complete"] = True
    payload["operators"][0]["enabled"] = True
    payload["freeze_authority"] = {
        "artifact_path": "authority.json",
        "artifact_sha256": authority_sha256,
        "artifact_schema": "mop-test-g0-freeze-authority/v1",
    }
    return TopologyGrammar.from_payload(_rehash(payload))


def test_committed_g0_is_a_complete_disabled_nonpromoting_scaffold() -> None:
    grammar = load_topology_grammar(GRAMMAR_PATH)

    assert grammar.status is GrammarStatus.SCAFFOLD
    assert grammar.activation_enabled is False
    assert grammar.scientific_promotion_allowed is False
    assert grammar.construction_language.implementation_complete is False
    assert set(grammar.construction_language.state_primitives) == set(StatePrimitive)
    assert set(grammar.construction_language.operator_primitives) == set(OperatorPrimitive)
    assert set(grammar.construction_language.allowed_mutations) == set(ConstructionMutation)
    assert grammar.freeze_authority is None
    assert {rule.kind for rule in grammar.operators} == set(MutationKind)
    assert all(rule.enabled is False for rule in grammar.operators)
    assert grammar.grammar_sha256 == canonical_sha256(grammar.payload(include_digest=False))


def test_grammar_rejects_digest_tampering_and_enabled_scaffold(tmp_path: Path) -> None:
    payload = json.loads(GRAMMAR_PATH.read_text())
    payload["caps"]["max_mutations_per_transaction"] += 1
    with pytest.raises(ValueError, match="self-hash mismatch"):
        TopologyGrammar.from_payload(payload)

    payload = json.loads(GRAMMAR_PATH.read_text())
    payload["operators"][0]["enabled"] = True
    _rehash(payload)
    with pytest.raises(ValueError, match="scaffold mutation operators must be disabled"):
        TopologyGrammar.from_payload(payload)


def test_disabled_grammar_can_describe_shadow_state_but_cannot_authorize_it() -> None:
    grammar = load_topology_grammar(GRAMMAR_PATH)
    mutation = _mutation(
        MutationKind.ADD_ACTOR_SLOT,
        actor_id="actor:spare",
        candidate_id="visual_reasoning",
        state_version=STATE_B,
        active=False,
        spare_for="actor:primary",
    )

    transaction, proposed = TopologyTransaction.propose(
        grammar=grammar,
        base=_base(),
        mutations=(mutation,),
    )
    assessment = assess_topology_transaction(
        grammar,
        transaction,
        (mutation,),
        base=_base(),
        proposed=proposed,
        candidate_registry=_registry(),
    )

    assert proposed.actors[-1].spare_for == "actor:primary"
    assert transaction.proposed_topology_sha256 == proposed.sha256
    assert assessment.structurally_valid is True
    assert assessment.shadow_authorized is False
    assert assessment.factual_commitment_authorized is False
    assert assessment.blockers == (
        "grammar-activation-disabled",
        "grammar-not-frozen",
        "operator-disabled:add_actor_slot",
    )


def test_swap_to_spare_is_pure_and_rewires_declared_topology() -> None:
    base = TopologySnapshot(
        actors=(
            ActorSlot("actor:primary", "visual_reasoning", STATE_A, True),
            ActorSlot(
                "actor:spare",
                "visual_reasoning",
                STATE_B,
                False,
                spare_for="actor:primary",
            ),
        ),
        routing_subscriptions=(("actor:primary", "vision"),),
        factor_scopes=(("actor:primary", "object"),),
    )
    mutation = _mutation(
        MutationKind.SWAP_TO_SPARE,
        failed_actor_id="actor:primary",
        spare_actor_id="actor:spare",
    )

    result = apply_topology_mutation(base, mutation)

    by_id = {row.actor_id: row for row in result.actors}
    assert by_id["actor:primary"].active is False
    assert by_id["actor:spare"].active is True
    assert result.routing_subscriptions == (("actor:spare", "vision"),)
    assert result.factor_scopes == (("actor:spare", "object"),)
    assert base.routing_subscriptions == (("actor:primary", "vision"),)


def test_mutations_reject_unknown_parameters_zero_work_and_identity_splice() -> None:
    with pytest.raises(ValueError, match="fields mismatch"):
        _mutation(
            MutationKind.RETIRE_ACTOR_SLOT,
            actor_id="actor:primary",
            evaluator_label="leak",
        )
    with pytest.raises(ValueError, match="nonzero lifecycle work"):
        TopologyMutation.create(
            kind=MutationKind.RETIRE_ACTOR_SLOT,
            parameters={"actor_id": "actor:primary"},
            declared_work=WorkVector.zero(),
        )

    row = _mutation(MutationKind.RETIRE_ACTOR_SLOT, actor_id="actor:primary")
    with pytest.raises(ValueError, match="identity mismatch"):
        TopologyMutation(
            mutation_id="0" * 64,
            kind=row.kind,
            parameters=row.parameters,
            declared_work=row.declared_work,
            retained_state_bytes_delta=row.retained_state_bytes_delta,
        )


def test_transaction_authority_splice_is_structurally_invalid() -> None:
    grammar = load_topology_grammar(GRAMMAR_PATH)
    mutation = _mutation(MutationKind.RETIRE_ACTOR_SLOT, actor_id="actor:primary")
    transaction, proposed = TopologyTransaction.propose(
        grammar=grammar,
        base=_base(),
        mutations=(mutation,),
    )

    assessment = assess_topology_transaction(
        grammar,
        transaction,
        (),
        base=_base(),
        proposed=proposed,
        candidate_registry=_registry(),
    )

    assert assessment.structurally_valid is False
    assert "mutation-authority-mismatch" in assessment.blockers


def test_frozen_grammar_requires_a_separately_verified_authority(tmp_path: Path) -> None:
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps({"schema": "mop-test-g0-freeze-authority/v1"}))
    authority_sha256 = hashlib.sha256(authority_path.read_bytes()).hexdigest()
    grammar = _frozen_grammar(authority_sha256)
    mutation = _mutation(
        MutationKind.ADD_ACTOR_SLOT,
        actor_id="actor:spare",
        candidate_id="visual_reasoning",
        state_version=STATE_B,
        active=False,
        spare_for="actor:primary",
    )
    transaction, proposed = TopologyTransaction.propose(
        grammar=grammar,
        base=_base(),
        mutations=(mutation,),
    )

    unverified = assess_topology_transaction(
        grammar,
        transaction,
        (mutation,),
        base=_base(),
        proposed=proposed,
        candidate_registry=_registry(),
    )

    assert unverified.shadow_authorized is False
    assert "freeze-authority-unverified" in unverified.blockers
    assert verify_freeze_authority(grammar, tmp_path) == ()
    verified_registry, registry_problems = verify_candidate_registry(grammar, REPO_ROOT)
    assert registry_problems == ()
    assert verified_registry == _registry()

    verified = assess_topology_transaction(
        grammar,
        transaction,
        (mutation,),
        base=_base(),
        proposed=proposed,
        candidate_registry=_registry(),
        freeze_authority_verified=True,
    )
    assert verified.shadow_authorized is True
    assert verified.factual_commitment_authorized is False


def test_snapshot_rejects_edges_to_inactive_or_missing_actors() -> None:
    with pytest.raises(ValueError, match="subscriptions require an active actor"):
        TopologySnapshot(
            actors=(ActorSlot("actor:primary", "visual_reasoning", STATE_A, False),),
            routing_subscriptions=(("actor:primary", "vision"),),
        )
