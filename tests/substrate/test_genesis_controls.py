"""Tests for genesis controls and deprivation baselines."""

from __future__ import annotations

from substrate import genesis_config as C
from substrate import genesis_controls  # noqa: F401 — registration side effect
from substrate.genesis_material import (
    Observation,
    Probe,
    Verdict,
    build,
    distinctness_report,
    equal_opportunity,
    registered,
)


def _observations() -> tuple[Observation, ...]:
    return (
        Observation(index=0, channel="scene", payload=(1, 2, 3, 4), teaching=False, modality="symbolic"),
        Observation(index=1, channel="field:color", payload=(7, 8), teaching=True, modality="symbolic"),
        Observation(index=2, channel="scene", payload=(1, 2, 9, 10), teaching=False, modality="symbolic"),
        Observation(index=3, channel="label:shape", payload=(3, 3, 3), teaching=True, modality="symbolic"),
        Observation(index=4, channel="tool", payload=(5, 6), teaching=False, modality="symbolic"),
    )


def _probes() -> tuple[Probe, ...]:
    return (
        Probe(index=0, family="unseen_concept_acquisition", channel="scene", probe=(1, 2), arity=2),
        Probe(index=1, family="category_boundary_revision", channel="field:color", probe=(), arity=2),
        Probe(index=2, family="tool_acquisition", channel="tool", probe=(5,), arity=1),
    )


def _opportunity(name: str, *, operation_budget: int = 10_000, durable_write_budget: int = 32):
    return equal_opportunity(
        envelope="512MB",
        observations=_observations(),
        sensor_channels=("scene", "field:color", "label:shape", "tool"),
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
        deprived=C.BASELINE_DEPRIVATION[name],
    )


def test_every_baseline_registered_under_exact_constitution_identifier() -> None:
    names = set(registered())
    for baseline in C.BASELINES:
        assert baseline in names, f"{baseline} is not registered"
        material = build(baseline, _opportunity(baseline))
        assert material.name == baseline


def test_baseline_deprivation_static_frozen_cannot_change_durable_state() -> None:
    name = "static_frozen_field"
    assert C.BASELINE_DEPRIVATION[name] == ("plasticity",)
    opportunity = _opportunity(name, durable_write_budget=16)
    assert opportunity.ledger.durable_write_budget == 0
    assert opportunity.plasticity_enabled is False
    material = build(name, opportunity)
    before = material.durable_state_digest()
    for observation in _observations():
        material.observe(observation)
    proposals = material.propose()
    assert proposals == ()
    # A positive verdict stream is a no-op when plasticity is constitutionally off.
    material.apply(())
    assert material.durable_state_digest() == before
    assert material.cost()["plasticity"] == 0


def test_deprivation_arms_cannot_spend_what_they_lack() -> None:
    for name, deprived in C.BASELINE_DEPRIVATION.items():
        opportunity = _opportunity(name, durable_write_budget=8)
        if "plasticity" in deprived:
            assert opportunity.ledger.durable_write_budget == 0
            assert opportunity.plasticity_enabled is False
            material = build(name, opportunity)
            for observation in _observations():
                material.observe(observation)
            assert material.propose() == ()
            before = material.durable_state_digest()
            material.apply(())
            assert material.durable_state_digest() == before


def test_s2_and_fr_deprived_of_nothing_and_can_spend_full_budget() -> None:
    for name in (C.CANONICAL_S2_ID, "FR_selected_kernel"):
        assert C.BASELINE_DEPRIVATION[name] == ()
        write_budget = 6
        opportunity = _opportunity(name, durable_write_budget=write_budget)
        assert opportunity.plasticity_enabled is True
        assert opportunity.persistence_enabled is True
        assert opportunity.ledger.durable_write_budget == write_budget
        assert opportunity.deprived == ()
        material = build(name, opportunity)
        for observation in _observations():
            material.observe(observation)
        # Drain the full durable write budget across propose/apply rounds.
        writes = 0
        safety = 0
        while writes < write_budget and safety < 20:
            safety += 1
            proposals = material.propose()
            if not proposals:
                # Feed another observation to stage more candidates.
                material.observe(
                    Observation(
                        index=100 + safety,
                        channel="scene",
                        payload=(safety, safety + 1, safety + 2, safety + 3),
                        teaching=True,
                    )
                )
                proposals = material.propose()
            if not proposals:
                break
            batch = proposals[: write_budget - writes]
            material.apply([Verdict(proposal.proposal_id, True, 1.0, 1.0) for proposal in batch])
            writes += len(batch)
        assert writes == write_budget
        assert material.cost()["plasticity"] == write_budget
        # Full operation budget is available (not zeroed by deprivation).
        assert opportunity.ledger.operation_budget >= write_budget
        # Persistence channel is open.
        checkpoint = material.checkpoint()
        assert checkpoint["activation"] is False
        material.restore(checkpoint)
        assert material.cost()["persistence"] >= 2


def test_s2_mechanism_differs_from_every_candidate() -> None:
    material = build(C.CANONICAL_S2_ID, _opportunity(C.CANONICAL_S2_ID))
    candidate_mechanisms = {meta["distinct_mechanism"] for meta in C.CANDIDATES.values()}
    candidate_forms = {meta["form"] for meta in C.CANDIDATES.values()}
    candidate_names = set(C.CANDIDATES)
    exclusive = {item for values in C.EXCLUSIVE_MECHANISMS.values() for item in values}
    assert material.mechanism not in candidate_mechanisms
    assert material.mechanism not in candidate_forms
    assert material.mechanism not in candidate_names
    assert material.mechanism not in exclusive
    assert material.mechanism != "K1_monolithic_plastic_field"


def test_distinctness_report_passes_across_controls_and_baselines() -> None:
    observations = _observations()
    probes = _probes()
    materials = []
    for name in C.BASELINES:
        # Distinctness uses identical opportunities; each arm still owns its mechanism.
        opportunity = equal_opportunity(
            envelope="512MB",
            observations=observations,
            sensor_channels=("scene", "field:color", "label:shape", "tool"),
            operation_budget=50_000,
            durable_write_budget=64,
            deprived=(),
        )
        materials.append(build(name, opportunity))
    report = distinctness_report(materials, observations, probes)
    assert report["checks"]["distinct_mechanism_identifiers"] is True
    assert report["checks"]["distinct_durable_state_under_identical_probe"] is True
    assert report["checks"]["no_colliding_pair"] is True
    assert report["all_pass"] is True
    assert report["activation"] is False
    assert report["collisions"] == []
    mechanisms = {row["mechanism"] for row in report["traces"].values()}
    assert len(mechanisms) == len(C.BASELINES)


def test_record_store_null_cannot_answer_unseen_labelled_field() -> None:
    opportunity = _opportunity("record_store_null")
    material = build("record_store_null", opportunity)
    for observation in _observations():
        material.observe(observation)
    proposals = material.propose()
    material.apply([Verdict(proposal.proposal_id, True, 0.0, 0.0) for proposal in proposals])
    # Known labelled fields from teaching can be echoed.
    known = material.answer(Probe(index=10, family="storage", channel="field:color", probe=(), arity=2))
    assert known.abstained is False
    assert known.value == (7, 8)
    # A probe whose target never appeared as a labelled field must abstain.
    unseen = material.answer(
        Probe(index=11, family="unseen_concept_acquisition", channel="held_out_entity_xyz", probe=(1, 2, 3), arity=2)
    )
    assert unseen.abstained is True
    # family-as-target that was never labelled also abstains when channel is novel.
    unseen_family = material.answer(
        Probe(index=12, family="never_labelled_family", channel="also_never_seen_channel", probe=(), arity=1)
    )
    assert unseen_family.abstained is True


def test_wrong_and_shuffled_do_not_reorder_inputs() -> None:
    """History order controls accept observe() order as given; they do not unshuffle."""
    stream = (
        Observation(index=0, channel="a", payload=(1,)),
        Observation(index=1, channel="b", payload=(2,)),
        Observation(index=2, channel="c", payload=(3,)),
    )
    for name in ("wrong_history_plastic", "shuffled_history_plastic"):
        material = build(name, _opportunity(name))
        for observation in stream:
            material.observe(observation)
        active = material._active_state()  # noqa: SLF001 — intentional mechanism check
        channels = [row["channel"] for row in active["buffer"]]
        assert channels == ["a", "b", "c"]


def test_activation_false_everywhere_in_checkpoints() -> None:
    for name in C.BASELINES:
        material = build(name, _opportunity(name))
        material.observe(_observations()[0])
        checkpoint = material.checkpoint()
        assert checkpoint["activation"] is False
        durable = material._durable_state()  # noqa: SLF001
        if isinstance(durable, dict):
            assert durable.get("activation", False) is False
