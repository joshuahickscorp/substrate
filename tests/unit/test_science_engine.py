"""Parity, authority, and mutation tests for the selected scientific record interpreter."""

from __future__ import annotations

import copy

import pytest

from mop.beds.starss23.experiments import COUNTING, RECORDS
from mop.science import RecordRefused, render_report, run_experiment, seal_record, verify_artifact
from mop.substrate.events import canonical_sha256


def _value(arm: str, seed: int, direction: str) -> float:
    offset = seed / 1000
    values = {"candidate": 0.40, "rate_matched_random": 0.50, "always_on": 0.60,
              "never_update": 0.70}
    return values[arm] + offset if direction == "lower" else 1 - values[arm] + offset


def _runner(record):
    def run(arm, seed, inputs):
        return {"arm": arm, "seed": seed,
                "metric_value": _value(arm, seed, record["metric"]["direction"]),
                "receipt": {"provider": "deterministic_fixture", "arm": arm, "seed": seed}}
    return run


def _independent(rows, record):
    """Separate graded recomputation: no producer decision helper is imported or called."""

    candidate = {row["seed"]: row["metric_value"] for row in rows["candidate"]}
    primary = record["controls"]["primary"]
    control = {row["seed"]: row["metric_value"] for row in rows[primary]}
    wins = losses = 0
    for seed in sorted(candidate):
        delta = (control[seed] - candidate[seed] if record["metric"]["direction"] == "lower"
                 else candidate[seed] - control[seed])
        wins += delta > 0
        losses += delta < 0
    survived = wins >= record["stop"]["min_reproductions"] and losses == 0
    return {"verdict": "reproduced_effect" if survived else "null_or_inconclusive"}


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: record["id"])
def test_required_starss23_axes_execute_deterministically_and_verify(record):
    first = run_experiment(record, _runner(record))
    second = run_experiment(record, _runner(record))
    assert first == second
    assert first["verdict"] == "reproduced_effect"
    assert first["decision"]["favorable"] == 5
    assert first["decision"]["p_value"] == 1 / 32
    assert verify_artifact(first, record, _independent)["verified"] is True


def test_tie_is_a_null():
    def tied(arm, seed, inputs):
        return {"arm": arm, "seed": seed, "metric_value": 0.5, "receipt": {}}

    artifact = run_experiment(COUNTING, tied)
    assert artifact["decision"]["favorable"] == 0
    assert artifact["decision"]["ties"] == 5
    assert artifact["verdict"] == "null_or_inconclusive"


@pytest.mark.parametrize(
    "field",
    [
        "schema", "experiment_id", "stage", "question", "null_hypothesis", "source", "split", "unit",
        "providers", "treatments", "controls", "metric", "sesoi", "multiplicity", "budget", "stop",
        "claims", "verification", "seeds", "metric_value", "promotion",
    ],
)
def test_resealed_lifecycle_and_graded_mutations_are_refused(field):
    artifact = run_experiment(COUNTING, _runner(COUNTING))
    changed = copy.deepcopy(artifact)
    if field == "metric_value":
        changed["results"]["candidate"][0]["metric_value"] = 9.0
    elif field == "promotion":
        changed["scientific_promotion"] = True
    elif field in ("schema", "experiment_id", "question", "null_hypothesis"):
        changed[field] = "wrong"
    elif field == "stage":
        changed[field] = 99
    elif field == "seeds":
        changed[field] = [999]
    else:
        changed[field] = {"mutated": True}
    changed["seal"] = canonical_sha256({key: value for key, value in changed.items() if key != "seal"})

    with pytest.raises(RecordRefused):
        verify_artifact(changed, COUNTING, _independent)


def test_record_authority_drift_is_refused_before_provider_execution():
    changed = copy.deepcopy(COUNTING)
    changed["split"] = {"rule": "not_room_disjoint"}
    called = False

    def provider(arm, seed, inputs):
        nonlocal called
        called = True
        return {}

    with pytest.raises(RecordRefused, match="authority has drifted"):
        run_experiment(changed, provider)
    assert called is False
    assert seal_record(changed)["record_sha256"] != COUNTING["record_sha256"]


def test_independent_verifier_disagreement_is_refused():
    artifact = run_experiment(COUNTING, _runner(COUNTING))

    with pytest.raises(RecordRefused, match="independent verifier disagrees"):
        verify_artifact(artifact, COUNTING, lambda rows, record: {"verdict": "null_or_inconclusive"})


def test_report_is_a_small_safe_projection():
    report = render_report(run_experiment(COUNTING, _runner(COUNTING)))
    assert "reproduced_effect" in report
    assert "activation_allowed: false" in report
