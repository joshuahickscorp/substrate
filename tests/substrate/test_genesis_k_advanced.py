"""Contract tests for advanced genesis materials K9, K10 and K11."""

from __future__ import annotations

import contextlib
import inspect

import pytest

import substrate.genesis_k_advanced as advanced
from substrate.genesis_k_advanced import (
    K10_COMPOSED_MECHANISMS,
    K9_predictive_plastic_field,
    K10_integrated_plastic_field,
    K11_interference_gated_sparse_fiber_field,
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
    "K9_predictive_plastic_field",
    "K10_integrated_plastic_field",
    "K11_interference_gated_sparse_fiber_field",
)

MATERIAL_TYPES = (
    K9_predictive_plastic_field,
    K10_integrated_plastic_field,
    K11_interference_gated_sparse_fiber_field,
)


def _opportunity(*, operation_budget: int = 10_000, durable_write_budget: int = 64, envelope: str = "512MB"):
    observations = (
        Observation(0, "vision", (1, -1, 1, 0), teaching=True),
        Observation(1, "vision", (1, 1, -1, 1), teaching=False),
    )
    return equal_opportunity(
        envelope=envelope,
        observations=observations,
        sensor_channels=("vision", "proprio"),
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
    )


def _material(name: str, **options):
    return build(name, _opportunity(**options.pop("opp_kwargs", {})), **options)


def _observation(index: int = 0, payload: tuple[int, ...] = (1, -1, 1, 1)) -> Observation:
    return Observation(index=index, channel="vision", payload=payload, elapsed_ms=5, teaching=True)


def _probe(index: int = 0, cue: tuple[int, ...] = (1, 0, -1, 1), arity: int = 3) -> Probe:
    return Probe(index=index, family="recall", channel="vision", probe=cue, arity=arity)


def _drive_to_proposal(material):
    """Observe until the material emits at least one proposal (bounded)."""
    payloads = (
        (1, -1, 1, 1),
        (-1, 1, -1, 1),
        (1, 1, 1, -1),
        (-1, -1, 1, 1),
        (1, -1, -1, 1),
        (2, -2, 1, 0),
        (-2, 2, -1, 1),
        (1, 0, 1, -1),
    )
    for index, payload in enumerate(payloads):
        material.observe(_observation(index=index, payload=payload))
        material.answer(_probe(index=index, cue=payload))
        if isinstance(material, K11_interference_gated_sparse_fiber_field) and material._interference_residual < material.tau:
            material.force_interference(material.tau + 2)
        proposals = material.propose()
        if proposals:
            return proposals
    if isinstance(material, K11_interference_gated_sparse_fiber_field):
        material.force_interference(material.tau + 4)
        proposals = material.propose()
        if proposals:
            return proposals
    raise AssertionError(f"{material.name} never emitted a proposal under the drive sequence")


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_registered(name: str) -> None:
    assert name in registered()


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_observe_and_answer_leave_durable_unchanged(name: str) -> None:
    material = _material(name)
    before = material.durable_state_digest()
    material.observe(_observation())
    after_observe = material.durable_state_digest()
    material.answer(_probe())
    after_answer = material.durable_state_digest()
    assert after_observe == before
    assert after_answer == before


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_refused_verdict_leaves_durable_unchanged_admitted_changes(name: str) -> None:
    material = _material(name)
    proposals = _drive_to_proposal(material)
    before = material.durable_state_digest()
    refused = [Verdict(proposal.proposal_id, False, 0.0, 0.0) for proposal in proposals]
    material.apply(refused)
    assert material.durable_state_digest() == before

    proposals = _drive_to_proposal(material)
    before = material.durable_state_digest()
    admitted = [Verdict(proposal.proposal_id, True, 1.0, 1.0) for proposal in proposals]
    receipts = material.apply(admitted)
    assert any(receipt.committed for receipt in receipts)
    assert material.durable_state_digest() != before


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_checkpoint_restore_round_trip(name: str) -> None:
    material = _material(name)
    material.observe(_observation())
    material.answer(_probe())
    proposals = _drive_to_proposal(material)
    material.apply([Verdict(p.proposal_id, True, 1.0, 0.9) for p in proposals])
    digest = material.durable_state_digest()
    active = material.active_state_digest()
    checkpoint = material.checkpoint()
    assert checkpoint["activation"] is False

    material.observe(_observation(index=99, payload=(-1, 1, 0, 1)))
    _drive_to_proposal(material)
    material.restore(checkpoint)
    assert material.durable_state_digest() == digest
    assert material.active_state_digest() == active


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_rollback_restores_prior_digest(name: str) -> None:
    material = _material(name)
    proposals = _drive_to_proposal(material)
    before = material.durable_state_digest()
    receipts = material.apply([Verdict(p.proposal_id, True, 1.0, 1.0) for p in proposals])
    committed = [receipt for receipt in receipts if receipt.committed]
    assert committed
    assert material.durable_state_digest() != before
    for receipt in reversed(committed):
        material.rollback(receipt)
    assert material.durable_state_digest() == before


def test_distinctness_report_passes() -> None:
    materials = [_material(name) for name in MATERIAL_NAMES]
    observations = [
        _observation(index=0, payload=(1, -1, 1, 0)),
        _observation(index=1, payload=(-1, 1, -1, 1)),
        _observation(index=2, payload=(1, 1, 0, -1)),
    ]
    probes = [
        _probe(index=0, cue=(1, 0, -1, 1)),
        _probe(index=1, cue=(-1, 1, 1, 0)),
    ]
    # Ensure K11 can propose under the identical probe sequence by lowering tau.
    for material in materials:
        if isinstance(material, K11_interference_gated_sparse_fiber_field):
            material.tau = 0
    report = distinctness_report(materials, observations, probes)
    assert report["activation"] is False
    assert report["all_pass"], report


def test_resource_exhausted_on_operation_budget() -> None:
    material = _material("K9_predictive_plastic_field", opp_kwargs={"operation_budget": 4, "durable_write_budget": 10})
    with pytest.raises(ResourceExhausted):
        for index in range(8):
            material.observe(_observation(index=index, payload=(1, -1, 1, index % 3 - 1)))


def test_no_expected_label_surface() -> None:
    for cls in MATERIAL_TYPES:
        names = set(dir(cls))
        names |= set(getattr(cls, "__annotations__", {}))
        names |= set(getattr(cls, "__dataclass_fields__", {}))
        for name, member in inspect.getmembers(cls):
            names.add(name)
            if callable(member):
                with contextlib.suppress(TypeError, ValueError):
                    names.update(inspect.signature(member).parameters)
        assert "expected" not in names
        source = inspect.getsource(cls)
        assert "expected" not in source or "expected_value" in source
        # expected_value is the Proposal field for self-assessed value, not a label.
        assert "held_out" not in source
        assert "answer_key" not in source


def test_k10_ablations_constructible_and_mechanically_different() -> None:
    ablations = K10_integrated_plastic_field.ablations()
    assert set(ablations) == set(K10_COMPOSED_MECHANISMS)
    full = _material("K10_integrated_plastic_field")
    observations = [
        _observation(index=0, payload=(1, -1, 1, 1)),
        _observation(index=1, payload=(-1, 1, -1, 0)),
    ]
    probes = [_probe(index=0), _probe(index=1, cue=(-1, 1, 0, 1))]

    def run(material):
        for observation in observations:
            material.observe(observation)
        for probe in probes:
            material.answer(probe)
        proposals = material.propose()
        if proposals:
            material.apply([Verdict(p.proposal_id, True, 1.0, 1.0) for p in proposals])
        return material.durable_state_digest()

    full_digest = run(full)
    for name, factory in ablations.items():
        ablated = factory(_opportunity())
        assert isinstance(ablated, K10_integrated_plastic_field)
        assert name in ablated.frozen_mechanisms or ablated._enabled.get(name) is False
        ablated_digest = run(ablated)
        assert ablated_digest != full_digest, f"ablation {name} collided with full K10"


def test_k11_rebind_refused_below_tau_admitted_above() -> None:
    material = _material("K11_interference_gated_sparse_fiber_field", tau=5)
    material.observe(_observation(payload=(1, -1, 1, 0)))
    material.force_interference(0)
    assert material.propose() == ()

    material.force_interference(5)
    proposals = material.propose()
    assert proposals
    assert all(p.trigger == "interference_residual" for p in proposals)
    before = material.durable_state_digest()
    material.apply([Verdict(p.proposal_id, False, 0.0, 0.0) for p in proposals])
    assert material.durable_state_digest() == before

    material.force_interference(6)
    proposals = material.propose()
    assert proposals
    before = material.durable_state_digest()
    material.apply([Verdict(p.proposal_id, True, 0.5, 0.5) for p in proposals])
    assert material.durable_state_digest() != before


def test_k9_emits_no_proposal_without_prediction_error() -> None:
    material = _material("K9_predictive_plastic_field", error_threshold=1)
    # No observations → zero error energy → no proposals.
    assert material.propose() == ()


def test_k9_prediction_error_gate_required_for_proposals() -> None:
    material = _material("K9_predictive_plastic_field")
    material.observe(_observation(payload=(1, -1, 1, 1)))
    assert material._error_energy >= 1
    proposals = material.propose()
    assert proposals
    assert all(p.trigger == "prediction_error" for p in proposals)


def test_activation_always_false_in_state() -> None:
    for name in MATERIAL_NAMES:
        material = _material(name)
        material.observe(_observation())
        durable = material._durable_state()
        active = material._active_state()
        assert durable.get("activation") is False
        assert active.get("activation") is False


def test_mechanisms_are_unique_and_named() -> None:
    mechanisms = {build(name, _opportunity()).mechanism for name in MATERIAL_NAMES}
    assert len(mechanisms) == len(MATERIAL_NAMES)
    assert "prediction_error_gated_durable_rewrite" in mechanisms
    assert "integrated_k1_to_k9_composed_shell" in mechanisms
    assert "interference_gated_sparse_fiber_rebind" in mechanisms


def test_module_exports() -> None:
    assert advanced.K9_predictive_plastic_field is K9_predictive_plastic_field
    assert advanced.K11_interference_gated_sparse_fiber_field is K11_interference_gated_sparse_fiber_field


# --------------------------------------------------------------------------
# Proposal parity: multi-attempt emission from each material's own mechanism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_propose_emits_multiple_unique_proposals_when_licensed(name: str) -> None:
    material = _material(name)
    proposals = _drive_to_proposal(material)
    # Drive once more so the mechanism has richer active state for multi-proposal emission.
    material.observe(_observation(index=20, payload=(-1, 1, -1, 1)))
    material.answer(_probe(index=20, cue=(-1, 1, -1, 1)))
    if isinstance(material, K11_interference_gated_sparse_fiber_field):
        material.force_interference(material.tau + 4)
    proposals = material.propose()
    assert len(proposals) > 1, f"{name} emitted only {len(proposals)} proposal(s)"
    ids = [p.proposal_id for p in proposals]
    assert len(ids) == len(set(ids))
    assert len(proposals) <= material.proposals_per_cycle()
    assert material.proposals_per_cycle() == advanced.PROPOSALS_PER_CYCLE


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_each_proposal_is_individually_committable_and_rollable(name: str) -> None:
    material = _material(name)
    material.observe(_observation(payload=(1, -1, 1, 1)))
    material.answer(_probe(cue=(1, -1, 1, 1)))
    if isinstance(material, K11_interference_gated_sparse_fiber_field):
        material.force_interference(material.tau + 4)
    proposals = material.propose()
    if len(proposals) <= 1:
        proposals = _drive_to_proposal(material)
        material.observe(_observation(index=30, payload=(-2, 2, 1, -1)))
        if isinstance(material, K11_interference_gated_sparse_fiber_field):
            material.force_interference(material.tau + 4)
        proposals = material.propose()
    assert len(proposals) > 1
    for proposal in proposals:
        before = material.durable_state_digest()
        receipts = material.apply([Verdict(proposal.proposal_id, True, 0.5, 0.5)])
        receipt = receipts[0]
        assert receipt.committed
        assert material.durable_state_digest() != before
        material.rollback(receipt)
        assert material.durable_state_digest() == receipt.durable_state_digest_before
        assert material.durable_state_digest() == before


def test_deprived_plasticity_emits_no_proposals() -> None:
    opportunity = equal_opportunity(
        envelope="512MB",
        observations=(
            Observation(0, "vision", (1, -1, 1, 0), teaching=True),
            Observation(1, "vision", (1, 1, -1, 1), teaching=False),
        ),
        sensor_channels=("vision", "proprio"),
        operation_budget=10_000,
        durable_write_budget=64,
        deprived=("plasticity",),
    )
    for name in MATERIAL_NAMES:
        material = build(name, opportunity)
        material.observe(_observation())
        if isinstance(material, K11_interference_gated_sparse_fiber_field):
            material.force_interference(material.tau + 4)
        assert material.propose() == ()


@pytest.mark.parametrize("name", MATERIAL_NAMES)
def test_no_proposal_is_a_durable_noop(name: str) -> None:
    material = _material(name)
    material.observe(_observation(payload=(1, -1, 1, 1)))
    material.answer(_probe(cue=(1, -1, 1, 1)))
    if isinstance(material, K11_interference_gated_sparse_fiber_field):
        material.force_interference(material.tau + 4)
    proposals = material.propose()
    if not proposals:
        proposals = _drive_to_proposal(material)
    assert proposals
    for proposal in proposals:
        before = material.durable_state_digest()
        receipts = material.apply([Verdict(proposal.proposal_id, True, 0.5, 0.5)])
        assert material.durable_state_digest() != before, f"{name} proposal {proposal.proposal_id} was a durable no-op"
        material.rollback(receipts[0])
        assert material.durable_state_digest() == before
