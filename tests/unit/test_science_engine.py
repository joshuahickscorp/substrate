"""Parity, authority, and mutation tests for the selected scientific record interpreter."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from mop.beds.starss23.experiments import COUNTING, RECORDS
from mop.science import (
    MATCHED_BUDGET_WALL_NOTE,
    RecordRefused,
    artifact_envelope,
    demonstration_receipt,
    finalize_artifact,
    render_report,
    run_experiment,
    safety_flags,
    seal_record,
    verify_artifact,
)
from mop.substrate.events import canonical_sha256


def _value(arm: str, seed: int, direction: str) -> float:
    offset = seed / 1000
    values = {"candidate": 0.40, "rate_matched_random": 0.50, "always_on": 0.60,
              "never_update": 0.70}
    return values[arm] + offset if direction == "lower" else 1 - values[arm] + offset


def _runner(record):
    def run(arm, seed, inputs):
        scale = max(1.0, 2.0 * record["sesoi"]["value"] / 0.1)
        return {"arm": arm, "seed": seed,
                "metric_value": scale * _value(arm, seed, record["metric"]["direction"]),
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


def test_shared_producer_finalization_is_canonical_and_nonmutating():
    body = {"schema": "fixture/v1", "value": 3}
    result = finalize_artifact(
        body,
        verdict="null",
        detail={"measured": True},
        prereg={"canonical_sha256": "registered"},
        receipt_payload={"evidence_digest": "receipt"},
    )
    assert body == {"schema": "fixture/v1", "value": 3}
    assert result.artifact == {**body, "seal": canonical_sha256(body)}
    assert result.seal == result.artifact["seal"]
    assert result.prereg == {"canonical_sha256": "registered"}
    assert result.receipt_payload == {"evidence_digest": "receipt"}
    with pytest.raises(RecordRefused, match="unsealed artifact body"):
        finalize_artifact(result.artifact, verdict="null")


def test_shared_demonstration_receipt_binds_the_exact_evidence_projection():
    evidence = {"per_seed": [{"seed": 0}], "flags": {"activation_allowed": False}}
    receipt = demonstration_receipt(
        mechanism_id="fixture",
        controls_cleared=("rate_matched_random", "always_on"),
        evidence=evidence,
        verdict="null",
        detail={"source_kind": "synthetic"},
    )
    assert receipt["evidence_digest"] == canonical_sha256(evidence)
    assert receipt["requirement_id"] == "stage3.confirmed_useful_mechanism"
    assert receipt["detail"] == {"source_kind": "synthetic"}


def test_safety_flags_returns_a_fresh_closed_boundary():
    first = safety_flags()
    first["activation_allowed"] = True
    assert safety_flags() == {
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }


def test_artifact_envelope_has_one_closed_matched_budget_authority():
    report = SimpleNamespace(
        policy=SimpleNamespace(bed_id="fixture_bed", claim_scope="fixture_scope"),
        source_kind="real",
        payload=lambda: {"report": "payload"},
        matched_budget=SimpleNamespace(payload=lambda: {"flops": 12}),
        break_even=SimpleNamespace(payload=lambda: {"n_star": 34}),
    )
    shared = {
        "schema": "fixture/v1",
        "report": report,
        "seeds": (1, 2),
        "per_seed": [{"seed": 1}],
        "stats": {"effect": 0.5},
        "controls": {"random": "cleared"},
        "flags": {"activation_allowed": False},
        "verdict": "null",
        "featurizer": {"n_params": 0},
        "gate": {"params": 3},
        "receipt_payload": {"evidence_digest": "digest"},
    }
    body = artifact_envelope(**shared, extra={"producer": "fixture"})
    assert set(body) == set(
        "schema stage bed_id claim_scope source_kind rights_clean reproductions seeds per_seed stats "
        "controls flags verdict harness featurizer gate demonstration_receipt matched_budget "
        "matched_budget_wall_note break_even producer".split()
    )
    assert body["seeds"] == [1, 2]
    assert body["harness"] == {"report": "payload"}
    assert body["matched_budget"] == {"flops": 12}
    assert body["break_even"] == {"n_star": 34}
    assert body["matched_budget_wall_note"] == MATCHED_BUDGET_WALL_NOTE
    with pytest.raises(RecordRefused, match="overlap the shared envelope"):
        artifact_envelope(**shared, extra={"schema": "drifted/v1"})
