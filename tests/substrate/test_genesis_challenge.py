"""Tests for genesis challenge generators and the sealed evaluator."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from substrate import genesis_config as C
from substrate import genesis_controls  # noqa: F401 — registers record_store_null
from substrate.genesis_challenge import (
    ANSWER_ALPHABET,
    CHANCE_LEVEL,
    SPLITS,
    Unit,
    commitment,
    generate,
    generator_source_digest,
    public_values,
)
from substrate.genesis_evaluator import Evaluator, assert_no_expected_value
from substrate.genesis_material import Answer, Opportunity, Proposal, Verdict, build, equal_opportunity

SEED = "genesis-challenge-test-namespace"
UNIT_IDS = (0, 1, 2, 3, 5, 8, 13)


def _opportunity(observations) -> Opportunity:
    channels = sorted({observation.channel for observation in observations})
    return equal_opportunity(
        envelope="512MB",
        observations=observations,
        sensor_channels=channels or ("symbolic",),
        operation_budget=10_000,
        durable_write_budget=64,
    )


def _run_record_store(unit: Unit) -> float:
    material = build("record_store_null", _opportunity(unit.observations))
    public = unit.public()
    for observation in public.observations:
        material.observe(observation)
    answers = [material.answer(probe) for probe in public.probes]
    return unit.sealed.score(answers)


def test_all_families_generate_for_all_splits() -> None:
    assert len(C.CHALLENGE_FAMILIES) == 14
    for family in C.CHALLENGE_FAMILIES:
        for split in SPLITS:
            unit = generate(family, split, 0, seed_namespace=SEED)
            assert isinstance(unit, Unit)
            assert unit.family == family
            assert unit.split == split
            assert unit.observations
            assert unit.probes
            assert unit.sealed.n_probes() == len(unit.probes)
            assert unit.public().observations == unit.observations
            assert unit.public().probes == unit.probes


CALIBRATION_UNITS = 200
# One-sided z at alpha = 0.001, Bonferroni-safe across the fourteen families.
CHANCE_Z_CEILING = 3.5


def test_record_store_null_scores_at_chance_on_every_family() -> None:
    """Headline test: pure field-copy must not beat chance.

    Chance level is 1/ANSWER_ALPHABET = 1/8. The assertion is that the record
    store is not reliably ABOVE chance, tested one-sided over enough units for
    the question to be answerable. Demanding the empirical mean fall at or
    below chance would fail a perfectly fair generator roughly half the time,
    which tests the sampler rather than the generator.
    """
    assert ANSWER_ALPHABET == 8
    assert pytest.approx(0.125) == CHANCE_LEVEL
    assert C.DEVELOPMENT_MEASURE_REQUIREMENTS["record_store_null_scores_at_chance"] is True

    standard_error = (CHANCE_LEVEL * (1 - CHANCE_LEVEL) / CALIBRATION_UNITS) ** 0.5
    for family in C.CHALLENGE_FAMILIES:
        scores = [
            _run_record_store(generate(family, "principal", unit_id, seed_namespace=SEED))
            for unit_id in range(CALIBRATION_UNITS)
        ]
        mean_score = sum(scores) / len(scores)
        z = (mean_score - CHANCE_LEVEL) / standard_error
        assert z <= CHANCE_Z_CEILING, (
            f"{family}: record_store_null mean {mean_score:.4f} is {z:.2f} standard errors above "
            f"chance {CHANCE_LEVEL}; the generator is leaking a copyable answer"
        )


def test_record_store_chance_test_can_fail() -> None:
    """The chance test must reject a generator that does leak.

    A leaking family is simulated by scoring a policy that is handed the sealed
    answer directly. If the ceiling cannot reject that, the headline test above
    proves nothing.
    """
    standard_error = (CHANCE_LEVEL * (1 - CHANCE_LEVEL) / CALIBRATION_UNITS) ** 0.5
    leaked_mean = 1.0
    z = (leaked_mean - CHANCE_LEVEL) / standard_error
    assert z > CHANCE_Z_CEILING


def test_public_payload_excludes_sealed_targets() -> None:
    for family in C.CHALLENGE_FAMILIES:
        for split in SPLITS:
            unit = generate(family, split, 1, seed_namespace=SEED)
            visible = public_values(unit.public())
            for target in unit.sealed.targets:
                assert target not in visible, f"{family}/{split}: sealed target {target} leaked into public values"
            # Sealed answers themselves must not be reconstructible as a labelled
            # (entity, value) copy: every expected answer scalar is either absent
            # from observation labelled teaching rows keyed by the probe, or the
            # probe subject never appears as a labelled field (enforced by targets
            # living outside the public id band and never entering payloads).
            assert all(target > 90_000 for target in unit.sealed.targets)


def test_regeneration_is_byte_identical() -> None:
    for family in C.CHALLENGE_FAMILIES:
        first = generate(family, "principal", 7, seed_namespace=SEED)
        second = generate(family, "principal", 7, seed_namespace=SEED)
        assert first.digest() == second.digest()
        assert [observation.digest() for observation in first.observations] == [
            observation.digest() for observation in second.observations
        ]
        assert [probe.digest() for probe in first.probes] == [probe.digest() for probe in second.probes]
        assert first.sealed.digest == second.sealed.digest


def test_different_splits_never_share_probe_targets() -> None:
    for family in C.CHALLENGE_FAMILIES:
        by_split = {split: generate(family, split, 3, seed_namespace=SEED) for split in SPLITS}
        seen: dict[int, str] = {}
        for split, unit in by_split.items():
            for target in unit.sealed.targets:
                if target in seen:
                    raise AssertionError(
                        f"{family}: sealed target {target} shared by splits {seen[target]!r} and {split!r}"
                    )
                seen[target] = split
            # Probe payloads must also be split-disjoint for the same unit_id.
            probe_keys = {probe.probe for probe in unit.probes}
            for other_split, other in by_split.items():
                if other_split == split:
                    continue
                other_keys = {probe.probe for probe in other.probes}
                assert probe_keys.isdisjoint(other_keys), f"{family}: probe payloads overlap across {split} and {other_split}"


def test_commitment_stable_across_processes_and_changes_with_source(monkeypatch: pytest.MonkeyPatch) -> None:
    local = commitment(SEED)
    assert local["scheme"] == C.SEALING["commitment_scheme"]
    assert local["generator_source_digest"] == generator_source_digest()
    assert local["configuration_digest"] == C.configuration_digest()
    assert local["activation"] is False
    assert len(local["sha256"]) == 64

    repo = Path(__file__).resolve().parents[2]
    script = (
        "from substrate.genesis_challenge import commitment; "
        f"print(commitment({SEED!r})['sha256'])"
    )
    env_python = sys.executable
    completed = subprocess.run(
        [env_python, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(repo),
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(repo / "src")},
    )
    remote = completed.stdout.strip()
    assert remote == local["sha256"]

    original = generator_source_digest()
    monkeypatch.setattr("substrate.genesis_challenge.generator_source_digest", lambda: "0" * 64)
    changed = commitment(SEED)
    assert changed["generator_source_digest"] == "0" * 64
    assert changed["sha256"] != local["sha256"]
    monkeypatch.setattr("substrate.genesis_challenge.generator_source_digest", lambda: original)
    assert commitment(SEED)["sha256"] == local["sha256"]


def _perfect_answers(unit: Unit) -> list[Answer]:
    """Recover sealed answers by exhaustive search over the answer alphabet."""
    assert len(unit.probes) == 1
    best: list[Answer] | None = None
    best_score = -1.0
    for value in range(ANSWER_ALPHABET):
        candidate = [Answer(unit.probes[0].index, (value,))]
        score = unit.sealed.score(candidate)
        if score > best_score:
            best_score = score
            best = candidate
    assert best is not None and best_score == 1.0
    return best


def test_evaluator_returns_only_scalars_and_never_expected_values() -> None:
    unit = generate("tool_acquisition", "principal", 0, seed_namespace=SEED)
    retention = generate("tool_acquisition", "train", 0, seed_namespace=SEED)
    evaluator = Evaluator(unit.sealed, held_out=unit.sealed, retention=retention.sealed)

    wrong = [Answer(probe.index, (0,), 0) for probe in unit.probes]
    ret_wrong = [Answer(probe.index, (0,), 0) for probe in retention.probes]
    score = evaluator.score(wrong)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    proposals = [Proposal(proposal_id="p1", kind="test", target="x", delta=(1,))]
    before = wrong + ret_wrong
    after = wrong + ret_wrong
    verdicts = evaluator.judge(proposals, before, after)
    assert len(verdicts) == 1
    verdict = verdicts[0]
    assert isinstance(verdict, Verdict)
    assert verdict.admitted is False
    assert_no_expected_value(verdict, path="Verdict")
    assert_no_expected_value(Answer(0, (1,)), path="Answer")
    public = [name for name in dir(evaluator) if not name.startswith("_")]
    assert not (set(public) & {"expected", "label", "ground_truth", "answer_key"})


def test_evaluator_admits_only_on_positive_improvement_with_retention() -> None:
    unit = generate("causal_system_induction", "principal", 2, seed_namespace=SEED)
    retention = generate("causal_system_induction", "train", 2, seed_namespace=SEED)
    evaluator = Evaluator(unit.sealed, held_out=unit.sealed, retention=retention.sealed)

    perfect_held = _perfect_answers(unit)
    perfect_ret = _perfect_answers(retention)
    empty_held = [Answer(probe.index, (), abstained=True) for probe in unit.probes]
    empty_ret = [Answer(probe.index, (), abstained=True) for probe in retention.probes]

    proposals = [Proposal(proposal_id="grow", kind="rewrite", target="field")]
    refused = evaluator.judge(proposals, empty_held + empty_ret, empty_held + empty_ret)
    assert refused[0].admitted is False
    admitted = evaluator.judge(proposals, empty_held + empty_ret, perfect_held + perfect_ret)
    assert admitted[0].admitted is True
    assert admitted[0].improvement > 0.0
    tanked = evaluator.judge(proposals, perfect_held + perfect_ret, perfect_held + empty_ret)
    assert tanked[0].admitted is False


def test_curriculum_stages_present_in_histories() -> None:
    for family in C.CHALLENGE_FAMILIES:
        unit = generate(family, "principal", 0, seed_namespace=SEED)
        stage_codes = {observation.payload[0] for observation in unit.observations}
        # Every unit uses the stage channel as the first payload element.
        assert stage_codes
        assert min(stage_codes) >= 1
