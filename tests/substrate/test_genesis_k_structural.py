"""Contract tests for structural genesis materials K4, K6, K7, K8."""

from __future__ import annotations

import inspect

import pytest

from substrate import genesis_config as C
from substrate.genesis_k_structural import (
    MECH_K4,
    MECH_K6,
    MECH_K7,
    MECH_K8,
    K4_continuous_time_plastic_field,
    K6_adaptive_topology_field,
    K7_native_mixed_radix_field,
    K8_event_sourced_plastic_field,
)
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

MATERIAL_NAMES = (
    "K4_continuous_time_plastic_field",
    "K6_adaptive_topology_field",
    "K7_native_mixed_radix_field",
    "K8_event_sourced_plastic_field",
)

MECHANISMS = {
    "K4_continuous_time_plastic_field": MECH_K4,
    "K6_adaptive_topology_field": MECH_K6,
    "K7_native_mixed_radix_field": MECH_K7,
    "K8_event_sourced_plastic_field": MECH_K8,
}


def _observations() -> tuple[Observation, ...]:
    return (
        Observation(index=0, channel="vision", payload=(1, -1, 1, 0, 2), elapsed_ms=10, teaching=True),
        Observation(index=1, channel="proprio", payload=(-2, 1, 0, 1, -1), elapsed_ms=5, teaching=False),
        Observation(index=2, channel="vision", payload=(1, 1, -1, 0, 1), elapsed_ms=0, teaching=False),
    )


def _probes() -> tuple[Probe, ...]:
    return (
        Probe(index=0, family="unseen_concept_acquisition", channel="vision", probe=(1, 0, -1, 1), arity=2),
        Probe(index=1, family="causal_system_induction", channel="proprio", probe=(-1, 1, 0, 0), arity=2),
    )


def _opportunity(*, operation_budget: int = 10_000, durable_write_budget: int = 10_000):
    return equal_opportunity(
        envelope="512MB",
        observations=_observations(),
        sensor_channels=("vision", "proprio"),
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
    )


def _make(name: str, **kwargs):
    return build(name, _opportunity(**kwargs))


def _admit_all(material):
    proposals = material.propose()
    if not proposals:
        return ()
    verdicts = [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals]
    return material.apply(verdicts)


def _refuse_all(material):
    proposals = material.propose()
    if not proposals:
        return ()
    verdicts = [Verdict(proposal_id=p.proposal_id, admitted=False, improvement=0.0, retention=0.0) for p in proposals]
    return material.apply(verdicts)


@pytest.fixture(autouse=True)
def _import_registers_materials():
    import substrate.genesis_k_structural as _mod  # noqa: F401

    for name in MATERIAL_NAMES:
        assert name in registered()


# --------------------------------------------------------------------------
# observe / answer leave durable digest unchanged
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_observe_does_not_change_durable_digest(name: str) -> None:
    material = _make(name)
    before = material.durable_state_digest()
    for observation in _observations():
        material.observe(observation)
    assert material.durable_state_digest() == before


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_answer_does_not_change_durable_digest(name: str) -> None:
    material = _make(name)
    for observation in _observations():
        material.observe(observation)
    before = material.durable_state_digest()
    for probe in _probes():
        material.answer(probe)
    assert material.durable_state_digest() == before


# --------------------------------------------------------------------------
# refused / admitted verdicts
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_refused_verdict_leaves_durable_unchanged(name: str) -> None:
    material = _make(name)
    for observation in _observations():
        material.observe(observation)
    before = material.durable_state_digest()
    receipts = _refuse_all(material)
    assert receipts, f"{name} emitted no proposals to refuse"
    assert material.durable_state_digest() == before
    assert all(not receipt.committed for receipt in receipts)


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_admitted_verdict_changes_durable_state(name: str) -> None:
    material = _make(name)
    for observation in _observations():
        material.observe(observation)
    before = material.durable_state_digest()
    receipts = _admit_all(material)
    assert receipts, f"{name} emitted no proposals to admit"
    assert material.durable_state_digest() != before
    assert any(receipt.committed for receipt in receipts)


# --------------------------------------------------------------------------
# checkpoint / restore and rollback
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_checkpoint_restore_round_trip(name: str) -> None:
    material = _make(name)
    for observation in _observations():
        material.observe(observation)
    _admit_all(material)
    if name == "K4_continuous_time_plastic_field":
        material.advance(120)
    checkpoint = material.checkpoint()
    digest = material.durable_state_digest()
    active = material.active_state_digest()
    # Mutate after checkpoint.
    for observation in _observations():
        material.observe(observation)
    _admit_all(material)
    material.restore(checkpoint)
    assert material.durable_state_digest() == digest
    assert material.active_state_digest() == active
    assert material.observations_seen == checkpoint["observations_seen"]
    assert material.elapsed_ms == checkpoint["elapsed_ms"]


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_rollback_restores_prior_digest(name: str) -> None:
    material = _make(name)
    for observation in _observations():
        material.observe(observation)
    before = material.durable_state_digest()
    receipts = _admit_all(material)
    assert receipts
    committed = [receipt for receipt in receipts if receipt.committed]
    assert committed
    # Roll back in reverse order so nested undos restore the initial digest.
    for receipt in reversed(committed):
        material.rollback(receipt)
    assert material.durable_state_digest() == before
    assert material.durable_state_digest() == committed[0].durable_state_digest_before


# --------------------------------------------------------------------------
# Distinctness
# --------------------------------------------------------------------------


def test_distinctness_report_passes_for_all_structural_materials() -> None:
    materials = [_make(name) for name in MATERIAL_NAMES]
    report = distinctness_report(materials, _observations(), _probes())
    assert report["checks"]["distinct_mechanism_identifiers"]
    assert report["checks"]["distinct_durable_state_under_identical_probe"]
    assert report["checks"]["no_colliding_pair"]
    assert report["all_pass"]
    assert report["activation"] is False
    for name in MATERIAL_NAMES:
        assert report["traces"][name]["mechanism"] == MECHANISMS[name]


# --------------------------------------------------------------------------
# Resource budget
# --------------------------------------------------------------------------


def test_resource_exhausted_when_operation_budget_exceeded() -> None:
    material = _make("K4_continuous_time_plastic_field", operation_budget=2, durable_write_budget=100)
    material.observe(_observations()[0])
    material.observe(_observations()[1])
    with pytest.raises(ResourceExhausted):
        material.observe(_observations()[2])


# --------------------------------------------------------------------------
# No expected-label back channel
# --------------------------------------------------------------------------


def test_no_material_exposes_expected_label_channel() -> None:
    for name in MATERIAL_NAMES:
        material = _make(name)
        class_dict = type(material).__dict__
        assert "expected" not in class_dict
        for attr_name in dir(material):
            if attr_name.startswith("_") and attr_name not in {"_opportunity"}:
                continue
            assert "expected" not in attr_name.lower() or attr_name == "expected_value"
        # expected_value lives on Proposal, never as material state.
        assert not hasattr(material, "expected")
        source = inspect.getsource(type(material))
        # Methods must not accept or store an evaluator/expected label.
        assert "held_out" not in source
        assert "answer_key" not in source


# --------------------------------------------------------------------------
# K4 continuous time
# --------------------------------------------------------------------------


def test_k4_advance_changes_state_without_observation() -> None:
    k4 = _make("K4_continuous_time_plastic_field")
    # Seed durable plastic so decay is visible, then advance alone.
    for observation in _observations():
        k4.observe(observation)
    _admit_all(k4)
    before_durable = k4.durable_state_digest()
    before_active = k4.active_state_digest()
    k4.advance(200)
    assert k4.durable_state_digest() != before_durable or k4.active_state_digest() != before_active
    assert k4.durable_state_digest() != before_durable


def test_only_k4_changes_under_advance() -> None:
    materials = {name: _make(name) for name in MATERIAL_NAMES}
    for material in materials.values():
        for observation in _observations():
            material.observe(observation)
        _admit_all(material)
    before = {name: material.durable_state_digest() for name, material in materials.items()}
    materials["K4_continuous_time_plastic_field"].advance(150)
    assert materials["K4_continuous_time_plastic_field"].durable_state_digest() != before["K4_continuous_time_plastic_field"]
    for name in ("K6_adaptive_topology_field", "K7_native_mixed_radix_field", "K8_event_sourced_plastic_field"):
        assert not hasattr(materials[name], "advance")
        assert materials[name].durable_state_digest() == before[name]


def test_k4_does_not_self_advance_clock() -> None:
    source = inspect.getsource(K4_continuous_time_plastic_field)
    assert "time.time" not in source
    assert "time.monotonic" not in source
    assert "perf_counter" not in source


# --------------------------------------------------------------------------
# K6 topology rent
# --------------------------------------------------------------------------


def test_topology_growth_without_verified_value_is_pruned() -> None:
    material = _make("K6_adaptive_topology_field", durable_write_budget=10_000)
    # Drive demand high enough to allocate.
    for index in range(4):
        material.observe(
            Observation(
                index=index,
                channel="vision",
                payload=(2, 2, 2, 2, 1, -1, 1, 1),
                elapsed_ms=0,
                teaching=True,
            )
        )
    proposals = material.propose()
    alloc = [p for p in proposals if p.kind == "topology_allocate"]
    assert alloc, "expected an allocate proposal under high demand"
    # Admit everything with zero improvement: growth without verified value.
    verdicts = [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=0.0, retention=0.0) for p in proposals]
    material.apply(verdicts)
    assert material._nodes, "structure should exist immediately after allocate"
    allocated_ids = set(material._nodes)
    # Run a full audit window of zero-value commits.
    for step in range(C.PRECISION_AUDIT_WINDOW + 2):
        material.observe(
            Observation(
                index=100 + step,
                channel="vision",
                payload=(1, -1, 1, 0),
                elapsed_ms=0,
            )
        )
        proposals = material.propose()
        if not proposals:
            continue
        material.apply(
            [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=0.0, retention=0.0) for p in proposals]
        )
    # Original unpaid nodes must be gone (pruned/archived).
    survivors = allocated_ids & set(material._nodes)
    assert not survivors, f"unpaid growth survived rent audit: {survivors}"
    assert any(entry.get("reason") == "rent_default" for entry in material._archive) or not material._nodes


def test_k6_growth_survives_when_value_is_verified() -> None:
    material = _make("K6_adaptive_topology_field", durable_write_budget=10_000)
    for index in range(4):
        material.observe(
            Observation(index=index, channel="vision", payload=(2, 2, 2, 2, 1, 1, 1, 1), elapsed_ms=0, teaching=True)
        )
    proposals = material.propose()
    material.apply([Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals])
    assert material._nodes
    kept = set(material._nodes)
    for step in range(C.PRECISION_AUDIT_WINDOW + 2):
        material.observe(Observation(index=200 + step, channel="vision", payload=(1, 1, -1, 0), elapsed_ms=0))
        proposals = material.propose()
        material.apply(
            [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals]
        )
    assert kept & set(material._nodes), "verified structure should survive rent"


# --------------------------------------------------------------------------
# K7 promote / demote
# --------------------------------------------------------------------------


def test_k7_promotion_and_demotion_under_rent() -> None:
    material = _make("K7_native_mixed_radix_field", durable_write_budget=10_000)
    assert isinstance(material, K7_native_mixed_radix_field)
    baseline = dict(material._precision_map)
    # Build demand for promotion.
    for index in range(6):
        material.observe(
            Observation(
                index=index,
                channel="vision",
                payload=(2, 2, -2, 2, 1, 1, -1, 2),
                elapsed_ms=0,
                teaching=True,
            )
        )
    promoted = False
    for _ in range(8):
        proposals = material.propose()
        promote = [p for p in proposals if p.kind == "precision_promote"]
        if promote:
            material.apply(
                [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals]
            )
            promoted = True
            break
        material.apply(
            [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals]
        )
        material.observe(
            Observation(index=50 + _, channel="vision", payload=(2, 2, 2, 2, 2, 2, 2, 2), elapsed_ms=0)
        )
    assert promoted, "expected at least one precision promotion"
    assert material._precision_map != baseline
    higher_regions = {
        name: precision
        for name, precision in material._precision_map.items()
        if REGION_RANK(precision) > REGION_RANK(baseline[name])
    }
    assert higher_regions
    # With sustained verified utility, promotion holds.
    for step in range(C.PRECISION_AUDIT_WINDOW + 2):
        material.observe(Observation(index=300 + step, channel="vision", payload=(1, 1, 1, 1), elapsed_ms=0))
        proposals = material.propose()
        material.apply(
            [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=1.0, retention=1.0) for p in proposals]
        )
    for name, precision in higher_regions.items():
        assert REGION_RANK(material._precision_map[name]) >= REGION_RANK(precision)

    # Fresh material: promote then starve of utility → automatic demotion.
    starved = _make("K7_native_mixed_radix_field", durable_write_budget=10_000)
    for index in range(6):
        starved.observe(
            Observation(index=index, channel="vision", payload=(2, 2, -2, 2, 1, 1, -1, 2), elapsed_ms=0, teaching=True)
        )
    before_promote = dict(starved._precision_map)
    promoted_targets: set[str] = set()
    for _ in range(8):
        proposals = starved.propose()
        promote = [p for p in proposals if p.kind == "precision_promote"]
        if promote:
            promoted_targets.update(p.target for p in promote)
            starved.apply(
                [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=0.0, retention=0.0) for p in proposals]
            )
            break
        starved.apply(
            [Verdict(proposal_id=p.proposal_id, admitted=True, improvement=0.0, retention=0.0) for p in proposals]
        )
        starved.observe(Observation(index=80 + _, channel="vision", payload=(2, 2, 2, 2, 2, 2), elapsed_ms=0))
    assert promoted_targets, "expected at least one precision promotion under starvation path"
    after_promote = {name: starved._precision_map[name] for name in promoted_targets}
    assert any(after_promote[name] != before_promote[name] for name in promoted_targets)
    for step in range(C.PRECISION_AUDIT_WINDOW + 4):
        starved.observe(Observation(index=400 + step, channel="vision", payload=(1, 0, -1, 0), elapsed_ms=0))
        proposals = starved.propose()
        # Refuse further promotions so rent demotion is not immediately undone.
        verdicts = [
            Verdict(
                proposal_id=p.proposal_id,
                admitted=p.kind != "precision_promote",
                improvement=0.0,
                retention=0.0,
            )
            for p in proposals
        ]
        starved.apply(verdicts)
    for name in promoted_targets:
        assert starved._precision_map[name] == before_promote[name], (
            f"{name} should demote back to {before_promote[name]}, got {starved._precision_map[name]}"
        )
        assert name not in starved._precision_rent


def REGION_RANK(precision: str) -> int:
    ladder = ("binary", "ternary", "quinary", "4_bit", "8_bit")
    return ladder.index(precision)


# --------------------------------------------------------------------------
# K8 event-sourced replay
# --------------------------------------------------------------------------


def test_k8_replay_determinism_from_archive_alone() -> None:
    material = _make("K8_event_sourced_plastic_field")
    assert isinstance(material, K8_event_sourced_plastic_field)
    for observation in _observations():
        material.observe(observation)
    _admit_all(material)
    for observation in _observations():
        material.observe(observation)
    _admit_all(material)
    digest = material.durable_state_digest()
    rebuilt = material.rebuild_projection()
    live = material.project_from_archive(material._archive)
    assert rebuilt == live
    # Durable digest is a pure function of the archive (and its projection).
    clone = _make("K8_event_sourced_plastic_field")
    assert isinstance(clone, K8_event_sourced_plastic_field)
    clone._archive = list(material._archive)
    clone._projection = clone.project_from_archive(clone._archive)
    clone._precision_map = dict(clone._projection["precision_map"])
    clone._compiled_procedures = list(clone._projection["compiled_procedures"])
    assert clone.durable_state_digest() == digest


# --------------------------------------------------------------------------
# freeze_mechanism
# --------------------------------------------------------------------------


def test_material_cannot_freeze_its_own_mechanism() -> None:
    for name, mechanism in MECHANISMS.items():
        material = _make(name)
        with pytest.raises(ValueError):
            material.freeze_mechanism(mechanism)


def test_freezing_foreign_mechanism_is_recorded() -> None:
    material = _make("K4_continuous_time_plastic_field")
    material.freeze_mechanism(MECH_K6)
    assert MECH_K6 in material.frozen_mechanisms


# --------------------------------------------------------------------------
# registration and identity
# --------------------------------------------------------------------------


def test_names_and_mechanisms_match_constitution() -> None:
    for name in MATERIAL_NAMES:
        material = _make(name)
        assert material.name == name
        assert name in C.CANDIDATES
        assert material.mechanism == C.EXCLUSIVE_MECHANISMS[name][0]
        assert material.mechanism == MECHANISMS[name]


def test_activation_never_true_in_durable_state() -> None:
    for name in MATERIAL_NAMES:
        material = _make(name)
        for observation in _observations():
            material.observe(observation)
        _admit_all(material)
        durable = material._durable_state()
        assert durable.get("activation") is False
        checkpoint = material.checkpoint()
        assert checkpoint["activation"] is False


# --------------------------------------------------------------------------
# Meta: tests fail if the covered code is broken
# --------------------------------------------------------------------------


def test_broken_observe_durable_write_is_detected() -> None:
    """If a material writes durable state during observe, MaterialBase must raise."""

    class Broken(K6_adaptive_topology_field):
        def _transition(self, observation: Observation) -> None:
            super()._transition(observation)
            self._plastic[0] = 1  # durable write during observe

    broken = Broken(
        name="K6_adaptive_topology_field",
        mechanism=MECH_K6,
        _opportunity=_opportunity(),
    )
    with pytest.raises(RuntimeError, match="wrote durable state during observation"):
        broken.observe(_observations()[0])


# --------------------------------------------------------------------------
# Proposal parity: multi-attempt emission from each material's own mechanism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_propose_emits_multiple_unique_proposals_when_licensed(name: str) -> None:
    from substrate import genesis_k_structural as structural

    material = _make(name)
    for observation in _observations():
        material.observe(observation)
    if name == "K4_continuous_time_plastic_field":
        material.advance(80)
    proposals = material.propose()
    assert len(proposals) > 1, f"{name} emitted only {len(proposals)} proposal(s)"
    ids = [p.proposal_id for p in proposals]
    assert len(ids) == len(set(ids))
    assert len(proposals) <= material.proposals_per_cycle()
    assert material.proposals_per_cycle() == structural.PROPOSALS_PER_CYCLE


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_each_proposal_is_individually_committable_and_rollable(name: str) -> None:
    material = _make(name)
    for observation in _observations():
        material.observe(observation)
    if name == "K4_continuous_time_plastic_field":
        material.advance(80)
    proposals = material.propose()
    assert len(proposals) > 1
    for proposal in proposals:
        before = material.durable_state_digest()
        receipts = material.apply(
            [Verdict(proposal_id=proposal.proposal_id, admitted=True, improvement=0.5, retention=0.5)]
        )
        receipt = receipts[0]
        assert receipt.committed
        assert material.durable_state_digest() != before
        material.rollback(receipt)
        assert material.durable_state_digest() == receipt.durable_state_digest_before
        assert material.durable_state_digest() == before


def test_deprived_plasticity_emits_no_proposals() -> None:
    opportunity = equal_opportunity(
        envelope="512MB",
        observations=_observations(),
        sensor_channels=("vision", "proprio"),
        operation_budget=10_000,
        durable_write_budget=10_000,
        deprived=("plasticity",),
    )
    for name in MATERIAL_NAMES:
        material = build(name, opportunity)
        for observation in _observations():
            material.observe(observation)
        assert material.propose() == ()


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_no_proposal_is_a_durable_noop(name: str) -> None:
    material = _make(name)
    for observation in _observations():
        material.observe(observation)
    if name == "K4_continuous_time_plastic_field":
        material.advance(80)
    proposals = material.propose()
    assert proposals
    for proposal in proposals:
        before = material.durable_state_digest()
        receipts = material.apply(
            [Verdict(proposal_id=proposal.proposal_id, admitted=True, improvement=0.5, retention=0.5)]
        )
        assert material.durable_state_digest() != before, f"{name} proposal {proposal.proposal_id} was a durable no-op"
        material.rollback(receipts[0])
        assert material.durable_state_digest() == before
