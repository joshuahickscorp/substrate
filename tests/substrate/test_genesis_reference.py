"""Tests for the genesis reference learner and solvability diagnostic."""

from __future__ import annotations

import math
import random

import pytest

from substrate import genesis_config as C
from substrate import genesis_reference  # noqa: F401 — registration side effect
from substrate.genesis_challenge import CHANCE_LEVEL, SealedAnswers, generate
from substrate.genesis_material import Answer, Observation, build, equal_opportunity
from substrate.genesis_reference import (
    CHANCE_Z_CEILING,
    ORDER_DEPENDENT_FAMILIES,
    run_reference_on_unit,
    solvability_report,
)

SEED = "genesis-reference-test-namespace"
REPORT_UNITS = 200


def _opportunity(observations):
    channels = sorted({observation.channel for observation in observations}) or ("symbolic",)
    return equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=tuple(channels),
        operation_budget=100_000,
        durable_write_budget=64,
    )


def test_reference_learner_never_holds_sealed_state() -> None:
    unit = generate("tool_acquisition", "principal", 0, seed_namespace=SEED)
    public = unit.public()
    material = build("reference_learner", _opportunity(public.observations))
    for observation in public.observations:
        material.observe(observation)
    for probe in public.probes:
        material.answer(probe)

    # No attribute may hold a SealedAnswers instance.
    for name in dir(material):
        if name.startswith("__"):
            continue
        try:
            value = getattr(material, name)
        except Exception:
            continue
        assert not isinstance(value, SealedAnswers), f"attribute {name!r} holds SealedAnswers"
        if isinstance(value, dict):
            for item in value.values():
                assert not isinstance(item, SealedAnswers)
        if isinstance(value, (list, tuple, set)):
            for item in value:
                assert not isinstance(item, SealedAnswers)

    durable = material._durable_state()
    active = material._active_state()
    assert_no_sealed(durable)
    assert_no_sealed(active)


def assert_no_sealed(obj) -> None:
    if isinstance(obj, SealedAnswers):
        raise AssertionError("SealedAnswers present in material state")
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert "sealed" not in str(key).lower() or key in {"activation"}
            assert_no_sealed(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            assert_no_sealed(item)


def test_reference_learner_runs_on_public_payload_only() -> None:
    unit = generate("exception_after_rule", "principal", 3, seed_namespace=SEED)
    public = unit.public()
    material = build("reference_learner", _opportunity(public.observations))
    for observation in public.observations:
        material.observe(observation)
    answers = [material.answer(probe) for probe in public.probes]
    assert all(isinstance(answer, Answer) for answer in answers)
    # Scoring uses the sealed object outside the material, as the evaluator does.
    score = unit.sealed.score(answers)
    assert 0.0 <= score <= 1.0


def test_observe_and_answer_do_not_write_durable_state() -> None:
    unit = generate("causal_system_induction", "principal", 1, seed_namespace=SEED)
    public = unit.public()
    material = build("reference_learner", _opportunity(public.observations))
    before = material.durable_state_digest()
    for observation in public.observations:
        material.observe(observation)
        assert material.durable_state_digest() == before
    for probe in public.probes:
        material.answer(probe)
        assert material.durable_state_digest() == before
    assert material.propose() == ()


def test_solvable_families_beat_chance_and_unsolvable_do_not() -> None:
    report = solvability_report(units_per_family=REPORT_UNITS, seed_namespace=SEED)
    assert report["n_families"] == 14
    assert report["n_solvable"] + report["n_unsolvable"] == 14

    standard_error = math.sqrt(CHANCE_LEVEL * (1.0 - CHANCE_LEVEL) / REPORT_UNITS)
    for family, row in report["families"].items():
        assert row["units"] >= 200
        assert pytest.approx(row["chance_level"]) == CHANCE_LEVEL
        if row["solvable"]:
            assert row["z"] > CHANCE_Z_CEILING, f"{family}: claimed solvable but z={row['z']:.2f}"
            assert row["mean_score"] > CHANCE_LEVEL
            scores = [run_reference_on_unit(generate(family, "principal", unit_id, seed_namespace=SEED + "-check")) for unit_id in range(REPORT_UNITS)]
            mean_score = sum(scores) / len(scores)
            z = (mean_score - CHANCE_LEVEL) / standard_error
            assert z > CHANCE_Z_CEILING, f"{family}: re-score mean={mean_score:.4f} z={z:.2f}"
        else:
            assert row["z"] <= CHANCE_Z_CEILING, f"{family}: claimed unsolvable but z={row['z']:.2f}"
            diagnosis = row.get("diagnosis", "")
            assert isinstance(diagnosis, str) and diagnosis.strip(), f"{family}: empty diagnosis"
            scores = [run_reference_on_unit(generate(family, "principal", unit_id, seed_namespace=SEED + "-check")) for unit_id in range(REPORT_UNITS)]
            mean_score = sum(scores) / len(scores)
            z = (mean_score - CHANCE_LEVEL) / standard_error
            assert z <= CHANCE_Z_CEILING, f"{family}: unsolvable family beat chance on re-score z={z:.2f}"


def test_order_dependent_families_read_structure_not_absolute_position() -> None:
    """Order-dependent families must use structural fields, not fixed stream indices.

    Full-stream shuffle destroys absolute positions but keeps channel payloads.
    The reference learner must still beat chance after shuffle. A fixed-index
    exploit must fall to chance on families with non-degenerate answer entropy.

    Note: teaching_sequence's fold is modular addition over a fixed multiset, so
    arrival order alone cannot change the answer; step/channel structure is what
    the reference uses. category_boundary_revision's sealed answer is constant 0
    under the current generator, so the position-exploit arm is only asserted on
    families with full answer-alphabet entropy.
    """
    standard_error = math.sqrt(CHANCE_LEVEL * (1.0 - CHANCE_LEVEL) / REPORT_UNITS)
    full_alphabet = ("teaching_sequence_following", "contradiction_reopening")
    for family in ORDER_DEPENDENT_FAMILIES:
        assert family in C.CHALLENGE_FAMILIES
        structure_scores: list[float] = []
        position_scores: list[float] = []
        for unit_id in range(REPORT_UNITS):
            unit = generate(family, "principal", unit_id, seed_namespace=SEED + "-shuffle")
            public = unit.public()
            observations = list(public.observations)
            if len(observations) > 1:
                rng = random.Random(unit_id + 17)
                rng.shuffle(observations)
                if observations == list(public.observations):
                    observations = list(reversed(observations))
            material = build("reference_learner", _opportunity(observations))
            for observation in observations:
                material.observe(observation)
            answers = [material.answer(probe) for probe in public.probes]
            structure_scores.append(unit.sealed.score(answers))

            if len(observations) > 4 and observations[4].payload:
                guess = sum(int(x) for x in observations[4].payload) % 8
            else:
                guess = unit_id % 8
            position_scores.append(unit.sealed.score([Answer(public.probes[0].index, (guess,))]))

        structure_mean = sum(structure_scores) / len(structure_scores)
        structure_z = (structure_mean - CHANCE_LEVEL) / standard_error
        assert structure_z > CHANCE_Z_CEILING, (
            f"{family}: after shuffle structure-reader mean={structure_mean:.4f} z={structure_z:.2f}; expected to survive shuffle by reading channel structure"
        )

        if family in full_alphabet:
            position_mean = sum(position_scores) / len(position_scores)
            position_z = (position_mean - CHANCE_LEVEL) / standard_error
            assert position_z <= CHANCE_Z_CEILING, (
                f"{family}: fixed-index exploit mean={position_mean:.4f} z={position_z:.2f}; expected at chance after shuffle"
            )


def test_shuffled_observations_preserve_observation_objects() -> None:
    """Sanity: shuffle test feeds real Observation instances, not rebuilt payloads."""
    unit = generate("teaching_sequence_following", "principal", 0, seed_namespace=SEED)
    observations = list(unit.public().observations)
    random.Random(0).shuffle(observations)
    assert all(isinstance(row, Observation) for row in observations)
    material = build("reference_learner", _opportunity(observations))
    for observation in observations:
        material.observe(observation)
    material.answer(unit.public().probes[0])
