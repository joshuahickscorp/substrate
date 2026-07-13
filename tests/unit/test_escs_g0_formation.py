from __future__ import annotations

from dataclasses import dataclass

import pytest

from mop.config import REPO_ROOT
from mop.escs.g0_genotype import G0ActorGenotype, G0OperatorNode, G0StateSlot
from mop.escs.perspective_registry import load_perspective_candidate_registry
from mop.escs.topology_grammar import OperatorPrimitive, StatePrimitive, load_topology_grammar
from mop.studies.escs_g0_construction import (
    G0ConstructionOperation,
    G0ConstructionRequest,
    G0ConstructionSnapshot,
    G0ConstructionStatus,
    attempt_g0_construction,
)
from mop.studies.escs_g0_formation import (
    G0CandidateBundle,
    G0FormationAttempt,
    G0FormationCosts,
    G0FormationLedger,
    G0FormationStatus,
    G0ObjectiveVector,
    G0ParetoArchive,
    G0ParetoDecisionStatus,
    G0TraceAssessment,
    build_g0_pareto_archive,
    g0_objective_dominates,
    verify_g0_formation_attempt,
    verify_g0_pareto_archive,
)
from mop.studies.escs_g0_shadow_coalition import (
    G0ShadowActorState,
    G0ShadowCaps,
    G0ShadowEpisode,
    G0ShadowPortBinding,
    G0ShadowSeed,
    G0ShadowTrace,
    execute_g0_shadow_coalition,
)
from mop.substrate.events import canonical_sha256

GRAMMAR_PATH = REPO_ROOT / "configs/experiment/escs_g0_topology_grammar.json"
REGISTRY_PATH = REPO_ROOT / "configs/experiment/escs_perspective_candidates.json"


def _context():
    return load_topology_grammar(GRAMMAR_PATH), load_perspective_candidate_registry(REGISTRY_PATH)


def _actor(*, capacity_bytes: int) -> G0ActorGenotype:
    state = G0StateSlot(
        slot_id="memory",
        primitive=StatePrimitive.BOUNDED_RECURRENT_STATE,
        schema_id="memory-v1",
        capacity_bytes=capacity_bytes,
    )
    update = G0OperatorNode.create(
        node_id="update",
        operator=OperatorPrimitive.AFFINE_OR_NONLINEAR_LOCAL_UPDATE,
        state_slot_ids=(state.slot_id,),
        parameters={
            "weights": [[1.0, 1.0]],
            "bias": [0.0],
            "activation": "identity",
            "state_mode": "concatenate",
        },
        declared_operations=4,
        max_output_bytes=64,
    )
    return G0ActorGenotype.create(
        candidate_id="planning",
        state_slots=(state,),
        operator_nodes=(update,),
        output_node_ids=(update.node_id,),
    )


def _snapshot(*, capacity_bytes: int) -> G0ConstructionSnapshot:
    actor = _actor(capacity_bytes=capacity_bytes)
    return G0ConstructionSnapshot.create(
        actors=(actor,),
        factor_scopes={actor.candidate_id: ("scope.planning",)},
    )


def _episode_and_trace(
    source: G0ConstructionSnapshot,
    *,
    episode_id: str,
) -> tuple[G0ShadowEpisode, G0ShadowTrace]:
    grammar, registry = _context()
    actor = source.actors[0]
    episode = G0ShadowEpisode.create(
        episode_id=episode_id,
        source=source,
        grammar=grammar,
        candidate_registry=registry,
        ports=(
            G0ShadowPortBinding(
                actor_id=actor.candidate_id,
                genotype_sha256=actor.genotype_sha256,
                root_node_id="update",
                schema_id="stimulus-v1",
                payload_form="numeric-vector",
                max_encoded_bytes=64,
            ),
        ),
        initial_states=(G0ShadowActorState.create(actor.candidate_id, {"memory": [0.0]}),),
        seeds=(
            G0ShadowSeed.create(
                seed_id="root",
                actor_id=actor.candidate_id,
                schema_id="stimulus-v1",
                payload_form="numeric-vector",
                payload=[1.0],
            ),
        ),
        caps=G0ShadowCaps(
            max_rounds=4,
            max_activations=4,
            max_queue_depth=4,
            max_messages=4,
            max_actor_operations=64,
            max_routed_payload_bytes=256,
            max_retained_state_bytes=source.retained_state_bytes,
        ),
    )
    trace = execute_g0_shadow_coalition(
        source,
        episode,
        grammar=grammar,
        candidate_registry=registry,
    )
    return episode, trace


def _assessment(
    *,
    task_authority_sha256: str,
    episode: G0ShadowEpisode,
    trace: G0ShadowTrace,
    scorer_sha256: str,
    quality: int,
) -> G0TraceAssessment:
    return G0TraceAssessment.create(
        task_id="task.shared",
        task_authority_sha256=task_authority_sha256,
        episode=episode,
        trace=trace,
        scorer_sha256=scorer_sha256,
        quality_microunits=quality,
        robustness_microunits=quality,
        diversity_microunits=quality,
    )


@dataclass(frozen=True)
class _Pair:
    base: G0CandidateBundle
    derived: G0CandidateBundle
    request: G0ConstructionRequest
    construction: object
    base_episode: G0ShadowEpisode
    base_trace: G0ShadowTrace
    derived_episode: G0ShadowEpisode
    derived_trace: G0ShadowTrace
    base_assessment: G0TraceAssessment
    derived_assessment: G0TraceAssessment
    base_attempt: G0FormationAttempt
    derived_attempt: G0FormationAttempt


def _pair(
    *,
    derived_task_authority_sha256: str | None = None,
    derived_scorer_sha256: str | None = None,
) -> _Pair:
    grammar, registry = _context()
    source = _snapshot(capacity_bytes=64)
    base = G0CandidateBundle.create_base(
        candidate_id="candidate.base",
        snapshot=source,
        grammar=grammar,
        candidate_registry=registry,
    )
    request = G0ConstructionRequest.create(
        attempt_id="mutation.capacity",
        source_snapshot_sha256=source.snapshot_sha256,
        operation=G0ConstructionOperation.ADJUST_STATE_CAPACITY,
        parameters={"actor_id": "planning", "slot_id": "memory", "capacity_bytes": 128},
        declared_work=101,
    )
    construction = attempt_g0_construction(
        request,
        source=source,
        grammar=grammar,
        candidate_registry=registry,
    )
    derived = G0CandidateBundle.create_derived(
        candidate_id="candidate.derived",
        parent=base,
        request=request,
        construction_attempt=construction,
    )
    base_episode, base_trace = _episode_and_trace(base.snapshot, episode_id="task.base")
    derived_episode, derived_trace = _episode_and_trace(
        derived.snapshot,
        episode_id="task.derived",
    )
    task_authority = canonical_sha256({"task": "shared", "version": 1})
    scorer = canonical_sha256({"scorer": "exact", "version": 1})
    base_assessment = _assessment(
        task_authority_sha256=task_authority,
        episode=base_episode,
        trace=base_trace,
        scorer_sha256=scorer,
        quality=20,
    )
    derived_assessment = _assessment(
        task_authority_sha256=derived_task_authority_sha256 or task_authority,
        episode=derived_episode,
        trace=derived_trace,
        scorer_sha256=derived_scorer_sha256 or scorer,
        quality=10,
    )
    base_attempt = G0FormationAttempt.create(
        formation_attempt_id="formation.base",
        candidate=base,
        assessments=(base_assessment,),
    )
    derived_attempt = G0FormationAttempt.create(
        formation_attempt_id="formation.derived",
        candidate=derived,
        construction_request=request,
        construction_attempt=construction,
        assessments=(derived_assessment,),
    )
    return _Pair(
        base=base,
        derived=derived,
        request=request,
        construction=construction,
        base_episode=base_episode,
        base_trace=base_trace,
        derived_episode=derived_episode,
        derived_trace=derived_trace,
        base_assessment=base_assessment,
        derived_assessment=derived_assessment,
        base_attempt=base_attempt,
        derived_attempt=derived_attempt,
    )


def _ledger(pair: _Pair) -> G0FormationLedger:
    return G0FormationLedger.empty().append(pair.base_attempt).append(pair.derived_attempt)


def test_candidate_specific_episodes_share_only_explicit_task_and_scorer_cohort() -> None:
    pair = _pair()

    assert pair.base_episode.episode_sha256 != pair.derived_episode.episode_sha256
    assert pair.base.snapshot.snapshot_sha256 != pair.derived.snapshot.snapshot_sha256
    assert pair.base_attempt.objective is not None
    assert pair.derived_attempt.objective is not None
    assert (
        pair.base_attempt.objective.evaluation_cohort_sha256
        == pair.derived_attempt.objective.evaluation_cohort_sha256
    )
    assert g0_objective_dominates(
        pair.base_attempt.objective,
        pair.derived_attempt.objective,
    )
    assert (
        pair.base_attempt.objective.retained_state_byte_rounds
        == pair.base_assessment.retained_state_byte_rounds
    )

    archive = build_g0_pareto_archive(_ledger(pair))
    statuses = {row.attempt_sha256: row.status for row in archive.decisions}
    assert statuses[pair.base_attempt.attempt_sha256] is G0ParetoDecisionStatus.RETAINED
    assert statuses[pair.derived_attempt.attempt_sha256] is G0ParetoDecisionStatus.DOMINATED


@pytest.mark.parametrize(
    ("task_authority", "scorer"),
    [
        (canonical_sha256({"task": "changed"}), None),
        (None, canonical_sha256({"scorer": "changed"})),
    ],
)
def test_changed_task_or_scorer_authority_prevents_cross_cohort_dominance(
    task_authority: str | None,
    scorer: str | None,
) -> None:
    pair = _pair(
        derived_task_authority_sha256=task_authority,
        derived_scorer_sha256=scorer,
    )
    assert pair.base_attempt.objective is not None
    assert pair.derived_attempt.objective is not None
    assert (
        pair.base_attempt.objective.evaluation_cohort_sha256
        != pair.derived_attempt.objective.evaluation_cohort_sha256
    )
    assert not g0_objective_dominates(
        pair.base_attempt.objective,
        pair.derived_attempt.objective,
    )
    archive = build_g0_pareto_archive(_ledger(pair))
    assert {row.status for row in archive.decisions} == {G0ParetoDecisionStatus.RETAINED}


def test_formation_ledger_and_archive_round_trip_and_replay_exactly() -> None:
    pair = _pair()
    ledger = _ledger(pair)
    archive = build_g0_pareto_archive(ledger)
    grammar, registry = _context()

    assert G0FormationAttempt.from_payload(pair.derived_attempt.payload()) == pair.derived_attempt
    assert G0FormationLedger.from_payload(ledger.payload()) == ledger
    assert G0ParetoArchive.from_payload(archive.payload()) == archive
    assert ledger.verify() == ()
    assert verify_g0_pareto_archive(archive, ledger)
    assert (
        verify_g0_formation_attempt(
            pair.derived_attempt,
            grammar=grammar,
            candidate_registry=registry,
            parents={pair.base.snapshot.snapshot_sha256: pair.base},
            episodes={pair.derived_episode.episode_sha256: pair.derived_episode},
            traces={pair.derived_trace.trace_sha256: pair.derived_trace},
        )
        == ()
    )
    assert pair.derived_attempt.costs.construction_work == pair.request.declared_work
    assert pair.derived_attempt.counterfactual_only is True
    assert not any(
        (
            pair.derived_attempt.activation_enabled,
            pair.derived_attempt.shadow_execution_authorized,
            pair.derived_attempt.factual_effects,
            pair.derived_attempt.factual_mutation_authorized,
            pair.derived_attempt.scientific_promotion_allowed,
        )
    )


def test_derived_candidate_cannot_enter_formation_without_exact_authority_pair() -> None:
    pair = _pair()
    with pytest.raises(ValueError, match="requires construction authorities"):
        G0FormationAttempt.create(
            formation_attempt_id="formation.missing",
            candidate=pair.derived,
            assessments=(pair.derived_assessment,),
        )
    with pytest.raises(ValueError, match="present as a pair"):
        G0FormationAttempt.create(
            formation_attempt_id="formation.incomplete",
            candidate=pair.base,
            construction_request=pair.request,
        )
    with pytest.raises(ValueError, match="parent candidate is absent"):
        G0FormationLedger.empty().append(pair.derived_attempt)


def test_refused_construction_is_charged_but_cannot_enter_pareto_archive() -> None:
    pair = _pair()
    grammar, registry = _context()
    request = G0ConstructionRequest.create(
        attempt_id="mutation.refused",
        source_snapshot_sha256=pair.base.snapshot.snapshot_sha256,
        operation=G0ConstructionOperation.ADJUST_STATE_CAPACITY,
        parameters={"actor_id": "planning", "slot_id": "memory", "capacity_bytes": 0},
        declared_work=77,
    )
    construction = attempt_g0_construction(
        request,
        source=pair.base.snapshot,
        grammar=grammar,
        candidate_registry=registry,
    )
    attempt = G0FormationAttempt.create(
        formation_attempt_id="formation.refused",
        candidate=None,
        construction_request=request,
        construction_attempt=construction,
    )

    assert construction.status is G0ConstructionStatus.REFUSED
    assert attempt.status is G0FormationStatus.REFUSED
    assert attempt.costs.construction_work == 77
    assert attempt.objective is None
    archive = build_g0_pareto_archive(G0FormationLedger.empty().append(attempt))
    assert archive.decisions[0].status is G0ParetoDecisionStatus.INELIGIBLE


def test_aggregate_costs_are_recomputed_even_if_all_nested_hashes_are_valid() -> None:
    pair = _pair()
    payload = pair.derived_attempt.payload()
    payload["costs"] = G0FormationCosts.create(
        construction_work=0,
        retained_state_bytes=pair.derived.snapshot.retained_state_bytes,
        assessments=pair.derived_attempt.assessments,
    ).payload()
    unhashed = dict(payload)
    unhashed.pop("attempt_sha256")
    payload["attempt_sha256"] = canonical_sha256(unhashed)

    with pytest.raises(ValueError, match="aggregate costs mismatch"):
        G0FormationAttempt.from_payload(payload)


def test_duplicate_task_evaluations_and_negative_construction_work_are_refused() -> None:
    pair = _pair()
    duplicate = G0TraceAssessment.create(
        task_id=pair.base_assessment.task_id,
        task_authority_sha256=canonical_sha256({"task": "duplicate"}),
        episode=pair.base_episode,
        trace=pair.base_trace,
        scorer_sha256=pair.base_assessment.scorer_sha256,
        quality_microunits=1,
        robustness_microunits=1,
        diversity_microunits=1,
    )
    with pytest.raises(ValueError, match="task_id duplicated"):
        G0FormationAttempt.create(
            formation_attempt_id="formation.duplicate",
            candidate=pair.base,
            assessments=(pair.base_assessment, duplicate),
        )
    with pytest.raises(ValueError, match="construction_work must be nonnegative"):
        G0ObjectiveVector.create(
            construction_work=-1,
            retained_state_bytes=pair.base.snapshot.retained_state_bytes,
            assessments=(pair.base_assessment,),
        )
