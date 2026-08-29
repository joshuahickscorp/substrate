"""The self model is measured, not narrated.

"""

from __future__ import annotations

import pytest

from substrate import selfmodel as S


def test_a_self_fact_without_a_source_is_refused():
    sm = S.SelfModel()
    with pytest.raises(S.Refused):
        sm.record(S.SelfFact("recent_errors", ["a"], source=""))
    with pytest.raises(S.Refused):
        sm.record(S.SelfFact("vibes", 1, source="proof/x.json"))
    sm.record(S.SelfFact("recent_errors", ["a"], source="runs/.../failure_events"))
    assert sm.fact("recent_errors").source
    assert len(sm.missing_facts()) == len(S.FACT_KINDS) - 1


def test_calibration_needs_a_prediction_paired_to_an_outcome():
    sm = S.SelfModel()
    empty = sm.calibration()
    assert empty["per_kind"]["accuracy"]["n"] == 0
    assert empty["per_kind"]["accuracy"]["calibrated"] is False
    assert set(empty["kinds_without_evidence"]) == set(S.PREDICTION_KINDS)

    for actual in (0.9, 0.88, 0.91, 0.9, 0.9):
        sm.observe(sm.predict("accuracy"), actual)
    row = sm.calibration()["per_kind"]["accuracy"]
    assert row["n"] == 5
    assert row["mean_absolute_error"] < 0.3
    assert "accuracy" in sm.calibration()["kinds_with_evidence"]


def test_an_unknown_prediction_kind_is_refused():
    with pytest.raises(S.Refused):
        S.SelfModel().predict("how_it_feels")


def test_updating_beats_the_fixed_prior_only_when_the_prior_is_wrong():
    # a badly placed prior: updating should measurably help
    wrong = S.compare_against_fixed_prior({"accuracy": [0.95] * 12}, prior={"accuracy": 0.2})
    assert wrong["improves_calibration"] is True
    assert wrong["calibration_gain"]["accuracy"] > 0

    # an already correct prior: updating has nothing to earn, and the report says so
    right = S.compare_against_fixed_prior({"accuracy": [0.5] * 12}, prior={"accuracy": 0.5})
    assert right["calibration_gain"]["accuracy"] == 0.0
    assert right["improves_calibration"] is False


def test_usefulness_is_only_claimed_where_a_comparison_was_run():
    report = S.usefulness_report({"decisions": {"with_model": 0.8, "without_model": 0.6}})
    assert report["uses"]["decisions"]["improves"] is True
    assert report["earned_uses"] == ["decisions"]
    for absent in ("calibration", "recovery", "planning", "adaptation"):
        assert report["uses"][absent]["measured"] is False
        assert report["uses"][absent]["improves"] is False

    nothing = S.usefulness_report({})
    assert nothing["any_use_earned"] is False
    assert nothing["earned_uses"] == []

    # a comparison that does not improve is recorded as not improving, not dropped
    worse = S.usefulness_report({"planning": {"with_model": 0.4, "without_model": 0.7}})
    assert worse["uses"]["planning"]["measured"] is True
    assert worse["uses"]["planning"]["improves"] is False
    assert worse["uses"]["planning"]["margin"] < 0


def test_the_declaration_lists_what_is_missing():
    doc = S.declaration()
    assert set(doc["fact_kinds"]) == set(S.FACT_KINDS)
    assert doc["facts_missing"] == list(S.FACT_KINDS), "an empty self model reports every fact missing"
    assert doc["calibration"]["kinds_with_evidence"] == []
