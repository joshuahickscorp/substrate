"""Fixture parity + mutation tests for the shared science engine (architecture A).

Proves, on deterministic fixtures, that the shared lifecycle reproduces a paired-arm experiment, seals it
canonically, is deterministic, treats ties as nulls, keeps producer and verifier graded logic separate, and
refuses every semantic mutation. This is the engine-level parity the STARSS23 migration builds on.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json

import pytest

from mop.science import ExperimentSpec, MetricSpec, run_experiment, verify_artifact
from mop.science.engine import paired_sign_flip_one_sided
from mop.science.spec import ArmSeedResult
from mop.science.verify import VerificationRefused

# A deterministic fixture family: candidate MAE is lower than the controls by a fixed per-seed margin.
_FIXTURE_MAE = {
    "candidate": {1: 0.40, 2: 0.42, 3: 0.39, 4: 0.41, 5: 0.38},
    "rate_matched_random": {1: 0.50, 2: 0.51, 3: 0.49, 4: 0.52, 5: 0.50},
    "always_on": {1: 0.60, 2: 0.61, 3: 0.59, 4: 0.62, 5: 0.60},
}

SPEC = ExperimentSpec(
    experiment_id="fixture_counting_bed",
    schema="mop-science-fixture/v1",
    stage=3,
    question="does the candidate re-estimation gate reduce coasted count MAE versus its rate-matched control",
    null_hypothesis="the candidate does not reduce MAE versus the rate-matched random control",
    metric=MetricSpec(name="coasted count MAE", direction="lower", sesoi=0.02),
    seeds=(1, 2, 3, 4, 5),
    arms=("candidate", "rate_matched_random", "always_on"),
    primary_control="rate_matched_random",
    decision_rule="paired_sign_flip_one_sided",
    min_reproductions=3,
    claim_ceiling="deterministic programmatic mechanics only; no capability or natural-data claim",
    forbidden_claim_verbs=("proves", "demonstrates capability"),
)


def _arm_runner(arm, seed, inputs):
    return ArmSeedResult(arm=arm, seed=seed, metric_value=_FIXTURE_MAE[arm][seed],
                         receipt={"source": "fixture", "arm": arm, "seed": seed})


def _independent_graded_recompute(arms, spec):
    """Structurally separate recompute: recompute improvements and verdict from raw, no engine import."""
    cand = {r.seed: r.metric_value for r in arms["candidate"]}
    ctrl = {r.seed: r.metric_value for r in arms[spec.primary_control]}
    seeds = sorted(set(cand) & set(ctrl))
    wins = 0
    losses = 0
    for s in seeds:
        diff = ctrl[s] - cand[s] if spec.metric.direction == "lower" else cand[s] - ctrl[s]
        if diff > 0:
            wins += 1
        elif diff < 0:
            losses += 1
    reproduced = wins >= spec.min_reproductions and losses == 0
    return {"verdict": "reproduced_effect" if reproduced else "null_or_inconclusive",
            "wins": wins, "losses": losses}


def test_engine_runs_seals_and_is_deterministic():
    a1 = run_experiment(SPEC, _arm_runner, inputs=None)
    a2 = run_experiment(SPEC, _arm_runner, inputs=None)
    assert a1 == a2  # deterministic
    assert a1["seal"] and a1["verdict"] == "reproduced_effect"
    assert a1["decision"]["favorable"] == 5 and a1["decision"]["against"] == 0


def test_independent_verifier_agrees_and_is_separate():
    art = run_experiment(SPEC, _arm_runner, inputs=None)
    receipt = verify_artifact(art, SPEC, _independent_graded_recompute)
    assert receipt["verified"] is True
    assert receipt["seal_reproduced"] is True
    assert receipt["independent_scientific_confirmation"] is False


def test_tie_is_a_null():
    tie = dict(_FIXTURE_MAE)
    tie = {**_FIXTURE_MAE, "rate_matched_random": dict(_FIXTURE_MAE["candidate"])}  # control == candidate
    runner = lambda arm, seed, inputs: ArmSeedResult(  # noqa: E731
        arm=arm, seed=seed, metric_value=tie[arm][seed], receipt={})
    art = run_experiment(SPEC, runner, inputs=None)
    assert art["decision"]["favorable"] == 0
    assert art["decision"]["ties"] == 5
    assert art["verdict"] == "null_or_inconclusive"


@pytest.mark.parametrize("mutate", ["verdict", "metric_value", "seal", "schema", "promotion"])
def test_mutations_are_refused(mutate):
    art = run_experiment(SPEC, _arm_runner, inputs=None)
    tampered = json.loads(json.dumps(art))
    if mutate == "verdict":
        tampered["verdict"] = "reproduced_effect" if art["verdict"] != "reproduced_effect" else "null_or_inconclusive"
        # reseal so the seal check passes and only the graded disagreement trips
        from mop.substrate.events import canonical_sha256
        body = {k: v for k, v in tampered.items() if k != "seal"}
        tampered["seal"] = canonical_sha256(body)
    elif mutate == "metric_value":
        tampered["results"]["candidate"][0]["metric_value"] = 0.99  # candidate now loses seed 1
        from mop.substrate.events import canonical_sha256
        body = {k: v for k, v in tampered.items() if k != "seal"}
        tampered["seal"] = canonical_sha256(body)
    elif mutate == "seal":
        tampered["seal"] = "0" * 64
    elif mutate == "schema":
        tampered["schema"] = "wrong/v9"
    elif mutate == "promotion":
        tampered["scientific_promotion"] = True
        from mop.substrate.events import canonical_sha256
        body = {k: v for k, v in tampered.items() if k != "seal"}
        tampered["seal"] = canonical_sha256(body)
    with pytest.raises(VerificationRefused):
        verify_artifact(tampered, SPEC, _independent_graded_recompute)


def test_sign_flip_exact_p_value():
    # 5 favorable of 5 decisive: one-sided binomial tail at p=0.5 is 1/32
    r = paired_sign_flip_one_sided([0.1, 0.1, 0.1, 0.1, 0.1])
    assert r["favorable"] == 5 and r["decisive_pairs"] == 5
    assert abs(r["p_value"] - (1 / 32)) < 1e-12
