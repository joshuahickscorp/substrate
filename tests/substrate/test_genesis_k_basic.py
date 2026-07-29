"""Tests for genesis basic cognitive materials K1, K2, K3, K5."""

from __future__ import annotations

import inspect

import pytest

import substrate.genesis_k_basic as k_basic
from substrate.genesis_material import (
    Observation,
    Probe,
    ResourceExhausted,
    Verdict,
    build,
    distinctness_report,
    equal_opportunity,
    registered,
)

MATERIALS = (
    "K1_monolithic_plastic_field",
    "K2_graph_plastic_field",
    "K3_cellular_plastic_field",
    "K5_recurrent_state_space_plastic_field",
)


def _opportunity(*, operation_budget: int = 50_000, durable_write_budget: int = 64):
    observations = (
        Observation(index=0, channel="sight", payload=(1, 0, -1, 2), teaching=True),
        Observation(index=1, channel="touch", payload=(0, 1, 1, -2), teaching=False),
    )
    return equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=("sight", "touch"),
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
    )


def _make(name: str, **options):
    return build(name, _opportunity(), **options)


def _make_with_opp(name: str, opportunity, **options):
    return build(name, opportunity, **options)


def _obs(index: int = 0) -> Observation:
    return Observation(index=index, channel="sight", payload=(1, -1, 2, 0, 1), teaching=index % 2 == 0)


def _probe(index: int = 0) -> Probe:
    return Probe(index=index, family="identity", channel="sight", probe=(1, 0, -1, 2), arity=2)


def _admit_once(material):
    material.observe(_obs())
    material.answer(_probe())
    proposals = material.propose()
    assert proposals, f"{material.name} emitted no proposals"
    before = material.durable_state_digest()
    receipts = material.apply([Verdict(p.proposal_id, True, 0.5, 0.5) for p in proposals])
    after = material.durable_state_digest()
    return before, after, receipts


@pytest.mark.parametrize("name", MATERIALS)
def test_registered(name):
    assert name in registered()


@pytest.mark.parametrize("name", MATERIALS)
def test_observe_and_answer_leave_durable_unchanged(name):
    material = _make(name)
    before = material.durable_state_digest()
    material.observe(_obs())
    mid = material.durable_state_digest()
    material.answer(_probe())
    after = material.durable_state_digest()
    assert mid == before
    assert after == before


@pytest.mark.parametrize("name", MATERIALS)
def test_refused_verdict_leaves_durable_unchanged_admitted_changes(name):
    material = _make(name)
    material.observe(_obs(0))
    material.observe(_obs(1))
    proposals = material.propose()
    assert proposals
    before = material.durable_state_digest()
    refused = material.apply([Verdict(proposals[0].proposal_id, False, 0.0, 0.0)])
    assert not refused[0].committed
    assert material.durable_state_digest() == before

    material.observe(_obs(2))
    proposals = material.propose()
    assert proposals
    before = material.durable_state_digest()
    admitted = material.apply([Verdict(proposals[0].proposal_id, True, 0.4, 0.6)])
    assert admitted[0].committed
    assert material.durable_state_digest() != before


@pytest.mark.parametrize("name", MATERIALS)
def test_checkpoint_restore_round_trip(name):
    material = _make(name)
    _admit_once(material)
    material.observe(_obs(3))
    material.answer(_probe(3))
    digest_before = material.durable_state_digest()
    active_before = material.active_state_digest()
    checkpoint = material.checkpoint()
    assert checkpoint["activation"] is False

    # Mutate then restore.
    _admit_once(material)
    assert material.durable_state_digest() != digest_before
    material.restore(checkpoint)
    assert material.durable_state_digest() == digest_before
    assert material.active_state_digest() == active_before


@pytest.mark.parametrize("name", MATERIALS)
def test_rollback_restores_prior_digest(name):
    material = _make(name)
    material.observe(_obs())
    proposals = material.propose()
    assert proposals
    receipts = material.apply([Verdict(proposals[0].proposal_id, True, 0.3, 0.7)])
    receipt = receipts[0]
    assert receipt.committed
    assert material.durable_state_digest() == receipt.durable_state_digest_after
    material.rollback(receipt)
    assert material.durable_state_digest() == receipt.durable_state_digest_before


def test_distinctness_report_passes_for_all_materials():
    materials = [_make(name) for name in MATERIALS]
    observations = [_obs(i) for i in range(3)]
    probes = [_probe(i) for i in range(2)]
    report = distinctness_report(materials, observations, probes)
    assert report["activation"] is False
    assert report["checks"]["distinct_mechanism_identifiers"]
    assert report["checks"]["distinct_durable_state_under_identical_probe"]
    assert report["checks"]["no_colliding_pair"]
    assert report["all_pass"], report
    digests = {name: row["durable_state_digest"] for name, row in report["traces"].items()}
    assert len(set(digests.values())) == len(MATERIALS)


def test_resource_exhausted_when_operation_budget_exceeded():
    opportunity = _opportunity(operation_budget=2, durable_write_budget=8)
    material = _make_with_opp("K1_monolithic_plastic_field", opportunity)
    # observe spends 1 in MaterialBase plus field_dim in the transition; budget 2 is exceeded honestly.
    with pytest.raises(ResourceExhausted):
        material.observe(_obs(0))


def test_no_material_exposes_expected_label_channel():
    for name in MATERIALS:
        cls = getattr(k_basic, name)
        for attr_name, _value in vars(cls).items():
            assert "expected" not in attr_name.lower(), f"{name}.{attr_name} looks like a label channel"
        for attr_name, member in inspect.getmembers(cls):
            if attr_name.startswith("__"):
                continue
            assert "expected" not in attr_name.lower(), f"{name}.{attr_name} looks like a label channel"
            if callable(member) and attr_name.startswith("_"):
                source = inspect.getsource(member)
                # Methods must not accept or store an expected label.
                assert "expected_label" not in source
                assert "held_out" not in source


def test_k1_freeze_structural_mechanism_is_noop_for_durable_change():
    material = _make("K1_monolithic_plastic_field")
    material.observe(_obs())
    proposals = material.propose()
    material.apply([Verdict(p.proposal_id, True, 1.0, 1.0) for p in proposals])
    before = material.durable_state_digest()

    # K1 never owned or used graph/cellular structure; freezing them must not alter its law.
    material.freeze_mechanism("typed_per_edge_plastic_value_scope_and_precision")
    material.freeze_mechanism("bounded_radius_local_neighbourhood_rule")
    material.freeze_mechanism("input_dependent_bounded_recurrence")

    material.observe(_obs(1))
    proposals = material.propose()
    material.apply([Verdict(p.proposal_id, True, 1.0, 1.0) for p in proposals])
    after = material.durable_state_digest()
    # Plasticity still works — freeze of foreign mechanisms is a no-op, not a freeze of K1 itself.
    assert after != before


def test_k2_k3_locality_separation():
    """Rewiring a non-neighbour edge changes K2 durable state and leaves K3 unchanged."""
    k2 = _make("K2_graph_plastic_field")
    k3 = _make("K3_cellular_plastic_field")

    for material in (k2, k3):
        material.observe(_obs(0))
        material.observe(_obs(1))
        proposals = material.propose()
        material.apply([Verdict(p.proposal_id, True, 1.0, 1.0) for p in proposals])

    # K2: every edge is part of durable topology; rewiring changes the digest.
    assert len(k2._edges) >= 2
    before_k2 = k2.durable_state_digest()
    edge = k2._edges[0]
    new_dst = (edge.dst + 1) % k2._n_nodes
    if new_dst == edge.dst:
        new_dst = (edge.dst + 2) % k2._n_nodes
    k2.rewire_edge(0, new_dst)
    assert k2.durable_state_digest() != before_k2

    # K3: non-local rewiring must not appear in durable state (local rule only).
    before_k3 = k3.durable_state_digest()
    src, dst = k3.find_non_neighbour_pair()
    assert k3._chebyshev(src, dst) > k3._radius
    k3.rewire_non_neighbour(src, dst)
    assert k3.durable_state_digest() == before_k3
    # Sanity: changing a local cell does change durable state.
    k3._cells[0] = (k3._cells[0] + 1) if k3._cells[0] < 2 else (k3._cells[0] - 1)
    k3._resize()
    assert k3.durable_state_digest() != before_k3


def test_each_material_has_unique_mechanism_string():
    mechanisms = []
    for name in MATERIALS:
        material = _make(name)
        assert material.name == name
        mechanisms.append(material.mechanism)
        assert material.mechanism != name
    assert len(set(mechanisms)) == len(mechanisms)


def test_admitted_change_assertion_is_load_bearing():
    """If commit were a silent no-op, the admitted-changes check would fail."""
    material = _make("K1_monolithic_plastic_field")
    material.observe(_obs())
    proposals = material.propose()
    assert proposals
    material._commit = lambda _proposal: None  # type: ignore[method-assign]
    before = material.durable_state_digest()
    material.apply([Verdict(proposals[0].proposal_id, True, 1.0, 1.0)])
    after = material.durable_state_digest()
    # This is the condition the real admitted-changes test asserts against.
    assert after == before, "sabotaged commit must leave durable state unchanged"
    # And the non-sabotaged path must differ (re-run on a fresh material).
    healthy = _make("K1_monolithic_plastic_field")
    healthy.observe(_obs())
    proposals = healthy.propose()
    before = healthy.durable_state_digest()
    healthy.apply([Verdict(proposals[0].proposal_id, True, 1.0, 1.0)])
    assert healthy.durable_state_digest() != before


def test_resident_bytes_track_packed_state_not_a_constant():
    material = _make("K1_monolithic_plastic_field")
    first = material.cost()["memory"]
    assert first > 0
    # Growing compiled procedures increases shell estimate after resize.
    material._compiled_procedures.append({"name": "proc", "body": "x" * 128})
    material._resize()
    second = material.cost()["memory"]
    assert second > first


def test_activation_always_false_in_durable_state():
    for name in MATERIALS:
        material = _make(name)
        _admit_once(material)
        durable = material._durable_state()
        assert durable.get("activation") is False
        checkpoint = material.checkpoint()
        assert checkpoint.get("activation") is False
