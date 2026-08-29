"""The world model battery, and the separation it exists to enforce.

"""

from __future__ import annotations

import pytest

from substrate import world as W


@pytest.fixture(scope="module")
def fitted():
    bed = W.synthetic_world()
    return W.WorldModel(bed["parents"]).fit(bed["train"]), bed


def test_the_battery_covers_every_declared_test_and_distinction(fitted):
    model, bed = fitted
    report = W.evaluate(model, bed)
    assert set(report["tests"]) == set(W.TESTS)
    assert set(report["distinctions"]) == set(W.DISTINCTIONS)
    assert set(W.TEST_GROUP.values()) == set(W.DISTINCTIONS)


def test_prediction_recovers_the_known_generative_truth(fitted):
    model, bed = fitted
    report = W.evaluate(model, bed)
    assert report["tests"]["next_state"] > 0.8
    assert report["tests"]["missing_observation"] > 0.8


def test_intervening_is_not_the_same_operation_as_observing(fitted):
    """do(road=slick) holds the road slick and mismatches the summer tyre. Observing it does not."""
    model, _ = fitted
    dry = {
        "season": 0,
        "weather": "dry",
        "road": "grip",
        "tyre": "summer",
        "speed": "fast",
        "arrive": "ontime",
    }
    done = model.intervene(dry, {"road": "slick"})
    assert done["road"] == "slick", "an intervened variable keeps its forced value"
    assert done["speed"] == "slow", "and its children respond to the forced value"
    # under observation the parent still drives the child, so road returns to what weather implies
    seen = model.observe_set(dry, {"road": "slick"})
    assert seen["road"] == "grip", "conditioning must not be silently promoted to intervening"


def test_a_null_counterfactual_reproduces_the_factual_prediction(fitted):
    model, bed = fitted
    for state, _ in bed["test"][:20]:
        assert model.counterfactual(state, {}) == model.predict(state)


def test_a_model_that_only_predicts_is_reported_as_a_limited_instrument():
    """The section 8 verdict is computed, not left to prose, and both branches are exercised."""
    assert (
        W.limited_instrument(
            {
                "predictive_accuracy": 0.97,
                "decision_usefulness": 0.0,
                "causal_validity": 1.0,
                "simulation_reliability": 1.0,
            }
        )
        is True
    )
    assert (
        W.limited_instrument(
            {
                "predictive_accuracy": 0.97,
                "decision_usefulness": 0.3,
                "causal_validity": 1.0,
                "simulation_reliability": 1.0,
            }
        )
        is False
    )
    # a weak predictor is not called a limited instrument either; it is simply a weak predictor
    assert (
        W.limited_instrument(
            {
                "predictive_accuracy": 0.4,
                "decision_usefulness": 0.0,
                "causal_validity": 0.5,
                "simulation_reliability": 0.5,
            }
        )
        is False
    )


def test_the_four_distinctions_separate_on_the_calibration_bed(fitted):
    """A single number would hide all of this. Four do not."""
    model, bed = fitted
    d = W.evaluate(model, bed)["distinctions"]
    assert d["predictive_accuracy"] > 0.85
    assert d["causal_validity"] > 0.95
    assert d["simulation_reliability"] > 0.85
    # the decision arm is the one that has to be earned, and on this bed it is
    assert d["decision_usefulness"] > 0.1, "no fixed action is good everywhere, so the model must pay off"


def test_the_decision_arm_is_measured_against_the_best_fixed_action(fitted):
    """If a constant policy could match it, the world model earned nothing."""
    model, bed = fitted
    gain = W._decision_gain(model, bed)
    assert gain > 0.0
    # the baseline really is the strongest state free policy, so the gain cannot come from a weak control
    costs = {
        a: W._mean([0.0 if model.rollout(model.intervene(s, {"tyre": a}), 2)[-1]["arrive"] == "ontime" else 1.0 for s, _ in bed["test"]])
        for a in bed["actions"]["tyre"]
    }
    assert min(costs.values()) > 0.0, "some fixed action would have been perfect, so the test is empty"


def test_every_distinction_is_recomputable_from_the_published_test_scores(fitted):
    """Correction C_DISTINCTION_ROUNDING: a sealed report must be recomputable from its own figures.

    The first version averaged unrounded scores and published rounded ones, so an independent verifier
    recomputing a distinction from the artifact got a different number than the artifact stated.
    """
    model, bed = fitted
    report = W.evaluate(model, bed)
    for distinction, reported in report["distinctions"].items():
        members = [report["tests"][t] for t, d in W.TEST_GROUP.items() if d == distinction]
        assert reported == round(sum(members) / len(members), 4), distinction
    assert report["distinctions_are_recomputable_from_tests"] is True


def test_an_unfitted_model_persists_rather_than_inventing_a_value():
    model = W.WorldModel({"a": (), "b": ("a",)})
    state = {"a": 1, "b": 2}
    assert model.predict(state) == state, "with no evidence the model must not invent a transition"


def test_long_horizon_rollout_is_scored_against_the_observed_support(fitted):
    model, bed = fitted
    traj = model.rollout(bed["test"][0][0], 20)
    assert len(traj) == 20
    report = W.evaluate(model, bed)
    assert 0.0 <= report["tests"]["long_horizon_consistency"] <= 1.0


def test_the_declaration_reports_the_battery_it_actually_ran():
    doc = W.declaration()
    assert doc["battery"]["n_test_transitions"] > 0
    assert set(doc["represents"]) == set(W.REPRESENTED)
    assert doc["calibration_bed"]["kind"].startswith("synthetic world")
