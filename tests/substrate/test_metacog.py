"""Metacognition and endogenous attention.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from substrate import metacog as K


def _stream(n=60):
    return K._probe(n)


def test_the_eight_measures_are_computed_from_paired_decisions_and_outcomes():
    report = K.evaluate("confidence_threshold", _stream())
    assert set(report) >= set(K.MEASURES)
    assert 0.0 <= report["decision_quality"] <= 1.0
    # transfer stays null because it needs a second bed, and a number would be invented
    assert report["transfer"] is None


def test_the_tradeoff_a_flattering_report_would_hide_is_visible():
    always = K.evaluate("always_verify", _stream())
    never = K.evaluate("never_verify", _stream())
    assert always["unnecessary_thought"] > never["unnecessary_thought"]
    assert always["missed_verification"] < never["missed_verification"]
    assert never["compute"] == 0.0 and always["compute"] > 0.0


def test_a_budget_forces_the_policy_to_stop_verifying():
    rich = K.evaluate("always_verify", _stream(), budget=1000.0)
    poor = K.evaluate("always_verify", _stream(), budget=5.0)
    assert poor["compute"] <= 5.0 < rich["compute"]
    assert poor["missed_verification"] > rich["missed_verification"]


def test_learned_metacognition_requires_measured_oracle_headroom():
    for simple in K.SIMPLE:
        assert K.select_policy(simple.name).name == simple.name
    with pytest.raises(K.Refused):
        K.select_policy("learned")
    with pytest.raises(K.Refused):
        K.select_policy("learned", oracle_headroom={"residual": 0.01})
    assert K.select_policy("learned", oracle_headroom={"residual": 0.2}).learned is True
    with pytest.raises(K.Refused):
        K.select_policy("intuition")


def test_the_oracle_policy_is_an_upper_bound_not_a_candidate():
    h = K.headroom(_stream())
    assert h["oracle"] >= h["best_simple"]
    assert K.BY_POLICY["oracle"].information_used == frozenset({"outcome"})
    for p in K.SIMPLE:
        assert "outcome" not in p.information_used
    # on this stream the strongest simple policy already matches the oracle, so nothing is licensed
    assert h["residual"] >= 0.0
    if h["residual"] <= K.SESOI:
        with pytest.raises(K.Refused):
            K.select_policy("learned", oracle_headroom=h)


def test_attention_ranks_by_declared_drivers_under_budget():
    candidates = [
        {
            "id": "urgent",
            "goal_relevance": 1.0,
            "uncertainty": 0.9,
            "risk": 0.8,
            "expected_value": 0.9,
            "novelty": 0.2,
            "contradiction": 1.0,
            "cost": 1.0,
        },
        {
            "id": "idle",
            "goal_relevance": 0.1,
            "uncertainty": 0.1,
            "risk": 0.0,
            "expected_value": 0.1,
            "novelty": 0.1,
            "contradiction": 0.0,
            "cost": 1.0,
        },
        {
            "id": "expensive",
            "goal_relevance": 0.9,
            "uncertainty": 0.9,
            "risk": 0.9,
            "expected_value": 0.9,
            "novelty": 0.9,
            "contradiction": 1.0,
            "cost": 99.0,
        },
    ]
    out = K.attend(candidates, budget=2.0)
    assert out["ranked"][0]["id"] == "expensive", "ranking is by score, before any budget is applied"
    assert "expensive" in out["dropped_for_budget"], "and the resource limit still excludes it"
    assert out["attended"] == ["urgent", "idle"]
    assert out["spent"] <= 2.0

    # a candidate that omits a driver is flagged rather than scored as if it declared zero silently
    thin = K.attend([{"id": "thin", "goal_relevance": 1.0, "cost": 1.0}], budget=5.0)
    assert thin["ranked"][0]["undeclared_drivers"]
    with pytest.raises(K.Refused):
        K.attend(candidates, weights={"vibes": 1.0}, budget=5.0)


def test_the_declaration_reports_the_headroom_it_measured():
    doc = K.declaration()
    assert set(doc["governs"]) == set(K.ACTIONS)
    assert doc["learned_currently_licensed"] == (doc["oracle_headroom"]["residual"] > K.SESOI)
    assert all(row["transfer"] is None for row in doc["measured_on_a_probe_stream"].values())
