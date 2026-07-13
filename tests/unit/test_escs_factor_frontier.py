from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, replace

import pytest

import mop.escs.factor_frontier as factor_frontier_module
from mop.escs.accounting import FACTUAL_BRANCH, WorkVector
from mop.escs.events import (
    EpistemicStatus,
    ESCSEvent,
    HypothesisEvent,
    HypothesisOrigin,
    ObservationEvent,
    state_version_for_parents,
)
from mop.escs.factor_frontier import (
    FactorFrontierCapExceeded,
    FactorFrontierCaps,
    FactorFrontierError,
    FactorFrontierQuery,
    FactorInvalidationPlan,
    FrontierProjectionReceipt,
    FrontierQueryReceipt,
    ShadowCausalFactorFrontier,
    ShadowFactorNode,
    plan_shadow_invalidation,
    project_shadow_factor_frontier,
    query_shadow_factor_frontier,
    verify_shadow_factor_frontier,
)
from mop.escs.ledger import EventLedger
from mop.substrate.events import BranchRef, EventRef, canonical_bytes


@dataclass(frozen=True)
class FrontierFixture:
    ledger: EventLedger
    observation_a: ObservationEvent
    observation_b: ObservationEvent
    base: HypothesisEvent
    competitor: HypothesisEvent
    independent: HypothesisEvent
    revision: HypothesisEvent
    counterfactual: HypothesisEvent
    counterfactual_child: HypothesisEvent
    branch: BranchRef


class SpoofedLedger(EventLedger):
    @property
    def entry_count(self) -> int:
        raise AssertionError("a spoofed ledger view was read before exact-type rejection")


def _observation(label: str) -> ObservationEvent:
    return ObservationEvent.create(
        raw_packet_or_delta_refs=(f"packet:{label}",),
        adapter_version="factor-frontier-test/v1",
        sensor_scope={"sensor": label},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=1),
        clock_start_tick=0,
        clock_end_tick=0,
        source_and_provenance={"source": f"fixture:{label}"},
        measured_creation_cost=WorkVector(event_formation=1),
    )


def _hypothesis(
    *,
    label: str,
    parents: tuple,
    referents: dict,
    factors: dict,
    tick: int,
    status: EpistemicStatus = EpistemicStatus.INFERRED,
    branch: BranchRef = FACTUAL_BRANCH,
    supersedes: tuple = (),
) -> HypothesisEvent:
    return HypothesisEvent.create(
        origin=HypothesisOrigin.ACTOR,
        epistemic_status=status,
        referent_hypotheses=referents,
        factor_change_distribution=factors,
        decision_relevance_distribution={"action:test": 0.5},
        reducibility_distribution={"factor:test": 0.5},
        supporting_event_ids=parents,
        calibrated_confidence=0.5,
        abstention_reason=None,
        predicted_value_of_further_computation=0.25,
        causal_parent_ids=parents,
        counterfactual_branch_id=branch,
        clock_start_tick=tick,
        clock_end_tick=tick,
        source_and_provenance={"source": f"fixture:{label}"},
        measured_creation_cost=WorkVector(indexing_and_graph_maintenance=1),
        supersedes_event_ids=supersedes,
    )


def _fixture() -> FrontierFixture:
    ledger = EventLedger()
    observation_a = _observation("a")
    observation_b = _observation("b")
    ledger.append(observation_a)
    ledger.append(observation_b)

    base = _hypothesis(
        label="base",
        parents=(observation_a.event_id,),
        referents={"referent:a": 0.8},
        factors={
            "factor:common": {"payload_form": "spatial-graph"},
            "factor:spatial": {"opaque_secret": "BASE_FACTOR_VALUE_MUST_NOT_LEAK"},
        },
        tick=1,
    )
    competitor = _hypothesis(
        label="competitor",
        parents=(observation_a.event_id,),
        referents={"referent:a": 0.7},
        factors={
            "factor:common": {"payload_form": "symbolic-constraint"},
            "factor:spatial": {"opaque_secret": "COMPETING_PRIVATE_VALUE"},
        },
        tick=1,
    )
    independent = _hypothesis(
        label="independent",
        parents=(observation_b.event_id,),
        referents={"referent:b": 0.9},
        factors={"factor:motor": {"opaque_program": [1, 0, 1]}},
        tick=1,
    )
    for event in (base, competitor, independent):
        ledger.append(event)

    revision = _hypothesis(
        label="revision",
        parents=(base.event_id,),
        referents={"referent:a": 0.85},
        factors={
            "factor:common": {"payload_form": "trajectory"},
            "factor:spatial": {"revision": 1},
        },
        tick=2,
        supersedes=(base.event_id,),
    )
    ledger.append(revision)

    branch = BranchRef("branch:factor-frontier-counterfactual")
    counterfactual = _hypothesis(
        label="counterfactual",
        parents=(competitor.event_id,),
        referents={"referent:a": 0.7},
        factors={"factor:causal": {"intervention": "turn-left"}},
        tick=2,
        status=EpistemicStatus.SIMULATED,
        branch=branch,
    )
    ledger.append(counterfactual)
    counterfactual_child = _hypothesis(
        label="counterfactual-child",
        parents=(counterfactual.event_id,),
        referents={"referent:a": 0.6},
        factors={"factor:causal": {"outcome": "simulated-only"}},
        tick=3,
        status=EpistemicStatus.SIMULATED,
        branch=branch,
    )
    ledger.append(counterfactual_child)
    return FrontierFixture(
        ledger=ledger,
        observation_a=observation_a,
        observation_b=observation_b,
        base=base,
        competitor=competitor,
        independent=independent,
        revision=revision,
        counterfactual=counterfactual,
        counterfactual_child=counterfactual_child,
        branch=branch,
    )


def _by_event(snapshot: ShadowCausalFactorFrontier) -> dict[str, ShadowFactorNode]:
    return {node.source_hypothesis_event_id: node for node in snapshot.nodes}


def _prefix(ledger: EventLedger, length: int) -> EventLedger:
    prefix = EventLedger()
    for entry in ledger.entries[:length]:
        prefix.append(entry.event)
    return prefix


def test_projection_is_exact_self_hashed_opaque_and_non_authoritative() -> None:
    fixture = _fixture()
    caps = FactorFrontierCaps()
    receipt = project_shadow_factor_frontier(fixture.ledger, caps=caps)
    snapshot = receipt.snapshot

    assert verify_shadow_factor_frontier(snapshot, ledger=fixture.ledger, caps=caps) == ()
    assert ShadowCausalFactorFrontier.from_payload(snapshot.payload()).payload() == snapshot.payload()
    assert snapshot.retained_state_bytes == len(canonical_bytes(snapshot.payload()))
    assert receipt.receipt_sha256
    assert receipt.events_examined == fixture.ledger.entry_count
    assert receipt.nodes_materialized == 6
    assert receipt.work.indexing_and_graph_maintenance > 0
    assert receipt.work.total_work <= caps.max_work_units
    assert receipt.accounting_applied is False

    serialized = canonical_bytes(snapshot.payload())
    assert b"BASE_FACTOR_VALUE_MUST_NOT_LEAK" not in serialized
    assert b"COMPETING_PRIVATE_VALUE" not in serialized
    assert snapshot.activation_enabled is False
    assert snapshot.runtime_consumable is False
    assert snapshot.factual_write_authorized is False
    assert snapshot.scientific_promotion_allowed is False


def test_competing_hypotheses_coexist_and_only_explicit_revision_retires() -> None:
    fixture = _fixture()
    snapshot = project_shadow_factor_frontier(fixture.ledger, caps=FactorFrontierCaps()).snapshot
    by_event = _by_event(snapshot)
    base = by_event[str(fixture.base.event_id)]
    competitor = by_event[str(fixture.competitor.event_id)]
    revision = by_event[str(fixture.revision.event_id)]

    assert base.factor_id in snapshot.superseded_factor_ids
    assert base.factor_id not in snapshot.active_factor_ids
    assert competitor.factor_id in snapshot.active_factor_ids
    assert revision.factor_id in snapshot.active_factor_ids
    assert revision.supersedes_factor_ids == (base.factor_id,)
    assert competitor.supersedes_factor_ids == ()

    active = query_shadow_factor_frontier(
        snapshot,
        FactorFrontierQuery(
            branch_id=str(FACTUAL_BRANCH),
            referent_any=("referent:a",),
            factor_scope_any=("factor:spatial",),
            max_results=8,
        ),
        ledger=fixture.ledger,
        caps=FactorFrontierCaps(),
    )
    assert set(active.matched_factor_ids) == {competitor.factor_id, revision.factor_id}
    assert active.runtime_consumable is False
    assert active.accounting_applied is False

    historical = query_shadow_factor_frontier(
        snapshot,
        FactorFrontierQuery(
            branch_id=str(FACTUAL_BRANCH),
            referent_any=("referent:a",),
            factor_scope_any=("factor:spatial",),
            max_results=8,
            include_superseded=True,
        ),
        ledger=fixture.ledger,
        caps=FactorFrontierCaps(),
    )
    assert set(historical.matched_factor_ids) == {
        base.factor_id,
        competitor.factor_id,
        revision.factor_id,
    }


def test_counterfactual_projection_and_query_are_one_way_branch_isolated() -> None:
    fixture = _fixture()
    snapshot = project_shadow_factor_frontier(fixture.ledger, caps=FactorFrontierCaps()).snapshot
    by_event = _by_event(snapshot)
    competitor = by_event[str(fixture.competitor.event_id)]
    counterfactual = by_event[str(fixture.counterfactual.event_id)]
    child = by_event[str(fixture.counterfactual_child.event_id)]

    assert counterfactual.parent_factor_ids == (competitor.factor_id,)
    assert child.parent_factor_ids == (counterfactual.factor_id,)
    assert counterfactual.epistemic_status is EpistemicStatus.SIMULATED

    counterfactual_result = query_shadow_factor_frontier(
        snapshot,
        FactorFrontierQuery(
            branch_id=str(fixture.branch),
            referent_any=("referent:a",),
            factor_scope_any=("factor:causal",),
            max_results=8,
        ),
        ledger=fixture.ledger,
        caps=FactorFrontierCaps(),
    )
    assert set(counterfactual_result.matched_factor_ids) == {
        counterfactual.factor_id,
        child.factor_id,
    }
    factual_result = query_shadow_factor_frontier(
        snapshot,
        FactorFrontierQuery(
            branch_id=str(FACTUAL_BRANCH),
            referent_any=("referent:a",),
            factor_scope_any=("factor:causal",),
            max_results=8,
        ),
        ledger=fixture.ledger,
        caps=FactorFrontierCaps(),
    )
    assert factual_result.matched_factor_ids == ()


def test_previous_prefix_is_verified_but_final_projection_is_batch_independent() -> None:
    fixture = _fixture()
    caps = FactorFrontierCaps()
    prefix = _prefix(fixture.ledger, 5)
    previous = project_shadow_factor_frontier(prefix, caps=caps).snapshot
    incremental = project_shadow_factor_frontier(fixture.ledger, caps=caps, previous=previous)
    full = project_shadow_factor_frontier(fixture.ledger, caps=caps)

    assert incremental.snapshot.payload() == full.snapshot.payload()
    assert incremental.previous_snapshot_sha256 == previous.snapshot_sha256
    assert incremental.events_examined == fixture.ledger.entry_count

    unrelated = EventLedger()
    unrelated.append(_observation("unrelated"))
    with pytest.raises(FactorFrontierError, match="previous snapshot"):
        project_shadow_factor_frontier(unrelated, caps=caps, previous=previous)


def test_snapshot_tamper_and_authority_enablement_fail_closed() -> None:
    fixture = _fixture()
    snapshot = project_shadow_factor_frontier(fixture.ledger, caps=FactorFrontierCaps()).snapshot

    tampered_node = copy.deepcopy(snapshot.payload())
    tampered_node["nodes"][0]["source_payload_digest"] = "f" * 64
    with pytest.raises(FactorFrontierError, match="factor-node self-hash mismatch"):
        ShadowCausalFactorFrontier.from_payload(tampered_node)

    enabled = copy.deepcopy(snapshot.payload())
    enabled["activation_enabled"] = True
    with pytest.raises(FactorFrontierError, match="activation_enabled must be the boolean false"):
        ShadowCausalFactorFrontier.from_payload(enabled)


def test_projection_and_snapshot_caps_fail_closed() -> None:
    fixture = _fixture()
    defaults = FactorFrontierCaps()
    with pytest.raises(FactorFrontierCapExceeded, match="event cap"):
        project_shadow_factor_frontier(
            fixture.ledger,
            caps=replace(defaults, max_events=1, max_nodes=1, max_query_results=1),
        )
    with pytest.raises(FactorFrontierCapExceeded, match="factor nodes"):
        project_shadow_factor_frontier(
            fixture.ledger,
            caps=replace(defaults, max_nodes=2, max_query_results=2),
        )
    with pytest.raises(FactorFrontierCapExceeded, match="outer keys"):
        project_shadow_factor_frontier(
            fixture.ledger,
            caps=replace(defaults, max_scopes_per_node=1),
        )
    with pytest.raises(FactorFrontierCapExceeded, match="snapshot exceeds"):
        project_shadow_factor_frontier(
            fixture.ledger,
            caps=replace(defaults, max_snapshot_bytes=128),
        )
    with pytest.raises(FactorFrontierCapExceeded, match="declared work cap"):
        project_shadow_factor_frontier(
            fixture.ledger,
            caps=replace(defaults, max_work_units=1),
        )


def test_factor_node_hard_key_caps_cover_factory_and_direct_reconstruction() -> None:
    fixture = _fixture()
    node = project_shadow_factor_frontier(fixture.ledger, caps=FactorFrontierCaps()).snapshot.nodes[0]
    too_many = tuple(sorted(f"referent:hard-cap-{index}" for index in range(65)))

    with pytest.raises(FactorFrontierCapExceeded, match="pre-scan item cap"):
        ShadowFactorNode.create(
            source_hypothesis_event_id=node.source_hypothesis_event_id,
            branch_id=node.branch_id,
            causal_event_ids=node.causal_event_ids,
            parent_factor_ids=node.parent_factor_ids,
            supersedes_factor_ids=node.supersedes_factor_ids,
            supporting_event_ids=node.supporting_event_ids,
            referent_hypotheses=too_many,
            factor_scopes=node.factor_scopes,
            epistemic_status=node.epistemic_status,
            evidence_class=node.evidence_class,
            source_payload_digest=node.source_payload_digest,
            producer_state_version=node.producer_state_version,
            clock_start_tick=node.clock_start_tick,
            clock_end_tick=node.clock_end_tick,
            clock_uncertainty=node.clock_uncertainty,
        )

    payload = node.payload()
    payload["referent_hypotheses"] = list(too_many)
    with pytest.raises(FactorFrontierCapExceeded, match="pre-scan item cap"):
        ShadowFactorNode.from_payload(payload)


def test_declared_node_caps_are_rechecked_by_query_verifier_and_invalidation() -> None:
    fixture = _fixture()
    defaults = FactorFrontierCaps()
    snapshot = project_shadow_factor_frontier(fixture.ledger, caps=defaults).snapshot
    narrow = replace(defaults, max_scopes_per_node=1)
    query = FactorFrontierQuery(
        branch_id=str(FACTUAL_BRANCH),
        referent_any=("referent:a",),
        factor_scope_any=("factor:spatial",),
        max_results=8,
    )

    with pytest.raises(FactorFrontierCapExceeded, match="per-node cap"):
        query_shadow_factor_frontier(snapshot, query, ledger=fixture.ledger, caps=narrow)
    assert verify_shadow_factor_frontier(snapshot, ledger=fixture.ledger, caps=narrow) == (
        "frontier scopes exceed the per-node cap",
    )
    with pytest.raises(FactorFrontierError, match="snapshot authority failed"):
        plan_shadow_invalidation(
            snapshot,
            erased_event_ids=(str(fixture.observation_a.event_id),),
            ledger=fixture.ledger,
            caps=narrow,
        )


def test_global_source_event_edge_cap_is_charged_and_fails_before_projection() -> None:
    ledger = EventLedger()
    first = _observation("edge-chain-0")
    ledger.append(first)
    second = ObservationEvent.create(
        raw_packet_or_delta_refs=("packet:edge-chain-1",),
        adapter_version="factor-frontier-test/v1",
        sensor_scope={"sensor": "edge-chain-1"},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=1),
        causal_parent_ids=(first.event_id,),
        clock_start_tick=1,
        clock_end_tick=1,
        source_and_provenance={"source": "fixture:edge-chain-1"},
    )
    ledger.append(second)
    third = ObservationEvent.create(
        raw_packet_or_delta_refs=("packet:edge-chain-2",),
        adapter_version="factor-frontier-test/v1",
        sensor_scope={"sensor": "edge-chain-2"},
        transport_and_detection_cost=WorkVector(raw_transport_and_adapters=1),
        causal_parent_ids=(second.event_id,),
        clock_start_tick=2,
        clock_end_tick=2,
        source_and_provenance={"source": "fixture:edge-chain-2"},
    )
    ledger.append(third)

    with pytest.raises(FactorFrontierCapExceeded, match="source-ledger causal edges"):
        project_shadow_factor_frontier(
            ledger,
            caps=replace(FactorFrontierCaps(), max_edges=1),
        )

    receipt = project_shadow_factor_frontier(ledger, caps=FactorFrontierCaps())
    assert receipt.source_event_edges_examined == 2
    assert receipt.edges_materialized == 0
    assert receipt.work.indexing_and_graph_maintenance == 1 + 3 + 0 + 2 + 0 + 0


def test_snapshot_hard_sequence_lineage_state_and_temporal_invariants() -> None:
    fixture = _fixture()
    snapshot = project_shadow_factor_frontier(fixture.ledger, caps=FactorFrontierCaps()).snapshot
    by_event = _by_event(snapshot)
    parent = by_event[str(fixture.base.event_id)]
    child = by_event[str(fixture.revision.event_id)]

    with pytest.raises(FactorFrontierCapExceeded, match="hard event cap"):
        replace(snapshot, through_sequence=2_049)
    with pytest.raises(FactorFrontierError, match="more factor nodes than source events"):
        replace(snapshot, through_sequence=1)
    with pytest.raises(FactorFrontierError, match="producer state"):
        replace(child, producer_state_version="f" * 64)

    disconnected = ShadowFactorNode.create(
        source_hypothesis_event_id=child.source_hypothesis_event_id,
        branch_id=child.branch_id,
        causal_event_ids=(),
        parent_factor_ids=(parent.factor_id,),
        supersedes_factor_ids=(),
        supporting_event_ids=(),
        referent_hypotheses=child.referent_hypotheses,
        factor_scopes=child.factor_scopes,
        epistemic_status=child.epistemic_status,
        evidence_class=child.evidence_class,
        source_payload_digest=child.source_payload_digest,
        producer_state_version=state_version_for_parents(()),
        clock_start_tick=child.clock_start_tick,
        clock_end_tick=child.clock_end_tick,
        clock_uncertainty=child.clock_uncertainty,
    )
    with pytest.raises(FactorFrontierError, match="causal hypothesis-event parents"):
        ShadowCausalFactorFrontier.create(
            source_ledger_sha256=snapshot.source_ledger_sha256,
            source_ledger_head_sha256=snapshot.source_ledger_head_sha256,
            through_sequence=2,
            nodes=(parent, disconnected),
        )

    too_early = ShadowFactorNode.create(
        source_hypothesis_event_id=child.source_hypothesis_event_id,
        branch_id=child.branch_id,
        causal_event_ids=(parent.source_hypothesis_event_id,),
        parent_factor_ids=(parent.factor_id,),
        supersedes_factor_ids=(),
        supporting_event_ids=(parent.source_hypothesis_event_id,),
        referent_hypotheses=child.referent_hypotheses,
        factor_scopes=child.factor_scopes,
        epistemic_status=child.epistemic_status,
        evidence_class=child.evidence_class,
        source_payload_digest=child.source_payload_digest,
        producer_state_version=state_version_for_parents((EventRef(parent.source_hypothesis_event_id),)),
        clock_start_tick=0,
        clock_end_tick=0,
        clock_uncertainty=0,
    )
    with pytest.raises(FactorFrontierError, match="before its parent factor ends"):
        ShadowCausalFactorFrontier.create(
            source_ledger_sha256=snapshot.source_ledger_sha256,
            source_ledger_head_sha256=snapshot.source_ledger_head_sha256,
            through_sequence=2,
            nodes=(parent, too_early),
        )


def test_query_is_capped_deterministic_and_reports_saturation() -> None:
    fixture = _fixture()
    snapshot = project_shadow_factor_frontier(fixture.ledger, caps=FactorFrontierCaps()).snapshot
    caps = replace(FactorFrontierCaps(), max_query_results=1)
    query = FactorFrontierQuery(
        branch_id=str(FACTUAL_BRANCH),
        referent_any=("referent:a",),
        factor_scope_any=("factor:spatial",),
        max_results=1,
    )
    first = query_shadow_factor_frontier(snapshot, query, ledger=fixture.ledger, caps=caps)
    second = query_shadow_factor_frontier(snapshot, query, ledger=fixture.ledger, caps=caps)
    assert first.payload() == second.payload()
    assert len(first.matched_factor_ids) == 1
    assert first.total_match_count == 2
    assert first.saturated is True
    assert first.work.dispatch_and_exploration > 0
    assert first.work.indexing_and_graph_maintenance > 0
    assert first.source_replay_verified is True
    assert first.source_ledger_sha256 == fixture.ledger.sha256

    oversized = replace(query, max_results=2)
    with pytest.raises(FactorFrontierCapExceeded, match="result limit"):
        query_shadow_factor_frontier(snapshot, oversized, ledger=fixture.ledger, caps=caps)


def test_query_charges_every_present_or_absent_index_key_probe() -> None:
    fixture = _fixture()
    caps = FactorFrontierCaps()
    snapshot = project_shadow_factor_frontier(fixture.ledger, caps=caps).snapshot
    referents = tuple(sorted(f"referent:missing-{index}" for index in range(64)))
    scopes = tuple(sorted(f"factor:missing-{index}" for index in range(64)))
    receipt = query_shadow_factor_frontier(
        snapshot,
        FactorFrontierQuery(
            branch_id=str(FACTUAL_BRANCH),
            referent_any=referents,
            factor_scope_any=scopes,
            max_results=1,
        ),
        ledger=fixture.ledger,
        caps=caps,
    )

    assert receipt.index_keys_probed == 128
    assert receipt.index_postings_touched == 0
    assert receipt.matched_factor_ids == ()
    assert receipt.work.dispatch_and_exploration == 129


def test_exact_ledger_type_and_query_source_authority_fail_closed() -> None:
    fixture = _fixture()
    caps = FactorFrontierCaps()
    snapshot = project_shadow_factor_frontier(fixture.ledger, caps=caps).snapshot
    query = FactorFrontierQuery(
        branch_id=str(FACTUAL_BRANCH),
        referent_any=("referent:a",),
        factor_scope_any=("factor:spatial",),
        max_results=8,
    )

    with pytest.raises(FactorFrontierError, match="exact EventLedger"):
        project_shadow_factor_frontier(SpoofedLedger(), caps=caps)

    unrelated = EventLedger()
    unrelated.append(_observation("wrong-query-authority"))
    with pytest.raises(FactorFrontierError, match="differs from exact source-ledger"):
        query_shadow_factor_frontier(snapshot, query, ledger=unrelated, caps=caps)


def test_valid_observed_candidate_child_of_inferred_parent_is_preserved() -> None:
    ledger = EventLedger()
    observation = _observation("epistemic-status")
    ledger.append(observation)
    inferred = _hypothesis(
        label="epistemic-inferred",
        parents=(observation.event_id,),
        referents={"referent:epistemic": 0.5},
        factors={"factor:epistemic": {"state": "inferred"}},
        tick=1,
    )
    ledger.append(inferred)
    observed_child = _hypothesis(
        label="epistemic-observed-child",
        parents=(inferred.event_id,),
        referents={"referent:epistemic": 0.6},
        factors={"factor:epistemic": {"state": "observed-candidate"}},
        tick=2,
        status=EpistemicStatus.OBSERVED_CANDIDATE,
    )
    ledger.append(observed_child)

    snapshot = project_shadow_factor_frontier(ledger, caps=FactorFrontierCaps()).snapshot
    child_node = _by_event(snapshot)[str(observed_child.event_id)]
    assert child_node.epistemic_status is EpistemicStatus.OBSERVED_CANDIDATE
    assert child_node.parent_factor_ids


def test_atomic_source_capture_does_not_adopt_concurrent_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture()
    before_count = fixture.ledger.entry_count
    before_sha256 = fixture.ledger.sha256
    append_started = threading.Event()
    append_attempting = threading.Event()
    original = factor_frontier_module._precheck_event_shape

    def instrumented_precheck(event: ESCSEvent, caps: FactorFrontierCaps) -> None:
        if not append_started.is_set():
            append_started.set()
            assert append_attempting.wait(timeout=2)
        original(event, caps)

    monkeypatch.setattr(factor_frontier_module, "_precheck_event_shape", instrumented_precheck)
    concurrent = _observation("concurrent-append")

    def append_concurrently() -> None:
        assert append_started.wait(timeout=2)
        append_attempting.set()
        fixture.ledger.append(concurrent)

    worker = threading.Thread(target=append_concurrently)
    worker.start()
    receipt = project_shadow_factor_frontier(fixture.ledger, caps=FactorFrontierCaps())
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert fixture.ledger.entry_count == before_count + 1
    assert fixture.ledger.sha256 != before_sha256
    assert receipt.events_examined == before_count
    assert receipt.snapshot.source_ledger_sha256 == before_sha256


def test_receipt_and_invalidation_issuance_cannot_be_reconstructed_or_forged() -> None:
    fixture = _fixture()
    caps = FactorFrontierCaps()
    projection = project_shadow_factor_frontier(fixture.ledger, caps=caps)
    query = query_shadow_factor_frontier(
        projection.snapshot,
        FactorFrontierQuery(
            branch_id=str(FACTUAL_BRANCH),
            referent_any=("referent:a",),
            factor_scope_any=("factor:spatial",),
            max_results=8,
        ),
        ledger=fixture.ledger,
        caps=caps,
    )
    plan = plan_shadow_invalidation(
        projection.snapshot,
        erased_event_ids=(str(fixture.observation_a.event_id),),
        ledger=fixture.ledger,
        caps=caps,
    )

    for artifact_type in (
        FrontierProjectionReceipt,
        FrontierQueryReceipt,
        FactorInvalidationPlan,
    ):
        assert not hasattr(artifact_type, "create")
        assert not hasattr(artifact_type, "from_payload")

    with pytest.raises(FactorFrontierError, match="issued only after exact source-ledger replay"):
        replace(projection)
    with pytest.raises(FactorFrontierError, match="issued only after exact source-ledger replay"):
        replace(query)
    with pytest.raises(FactorFrontierError, match="issued only after exact source-ledger replay"):
        replace(plan)


def test_invalidation_is_conservative_nonapplying_and_branch_safe() -> None:
    fixture = _fixture()
    caps = FactorFrontierCaps()
    snapshot = project_shadow_factor_frontier(fixture.ledger, caps=caps).snapshot
    by_event = _by_event(snapshot)
    plan = plan_shadow_invalidation(
        snapshot,
        erased_event_ids=(str(fixture.observation_a.event_id),),
        ledger=fixture.ledger,
        caps=caps,
    )

    affected_source_ids = {
        fixture.base.event_id,
        fixture.competitor.event_id,
        fixture.revision.event_id,
        fixture.counterfactual.event_id,
        fixture.counterfactual_child.event_id,
    }
    affected_factor_ids = {by_event[str(event_id)].factor_id for event_id in affected_source_ids}
    independent_id = by_event[str(fixture.independent.event_id)].factor_id
    base_id = by_event[str(fixture.base.event_id)].factor_id
    assert set(plan.affected_factor_ids) == affected_factor_ids
    assert independent_id not in plan.affected_factor_ids
    assert base_id not in plan.invalidated_active_factor_ids
    assert set(plan.invalidated_active_factor_ids) == affected_factor_ids - {base_id}
    assert plan.archive_deletion_verified is False
    assert plan.application_authorized is False
    assert plan.accounting_applied is False
    assert plan.activation_enabled is False
    assert plan.runtime_consumable is False
    assert plan.factual_write_authorized is False
    assert plan.scientific_promotion_allowed is False
    assert plan.plan_sha256

    with pytest.raises(FactorFrontierError, match="outside the source ledger"):
        plan_shadow_invalidation(
            snapshot,
            erased_event_ids=("event:unknown",),
            ledger=fixture.ledger,
            caps=caps,
        )


def test_empty_key_query_remains_bounded_and_returns_only_requested_branch() -> None:
    fixture = _fixture()
    caps = FactorFrontierCaps()
    snapshot = project_shadow_factor_frontier(fixture.ledger, caps=caps).snapshot
    receipt = query_shadow_factor_frontier(
        snapshot,
        FactorFrontierQuery(
            branch_id=str(fixture.branch),
            referent_any=(),
            factor_scope_any=(),
            max_results=8,
        ),
        ledger=fixture.ledger,
        caps=caps,
    )
    assert receipt.total_match_count == 2
    assert all(
        next(node for node in snapshot.nodes if node.factor_id == factor_id).branch_id == str(fixture.branch)
        for factor_id in receipt.matched_factor_ids
    )
