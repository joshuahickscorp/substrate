"""Perspectives as declared processes, a selection ladder that refuses to skip rungs, and arbitration
that keeps the minority.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from substrate import perspectives as PS
from substrate import workspace as W


def _out(name, value, confidence, cost=1.0):
    return PS.Output(name, value, confidence, [f"{name}:perceptual"], cost)


def test_every_catalogued_perspective_is_fully_declared():
    for p in PS.CATALOG:
        assert p.spec.violations() == [], p.spec.violations()
    doc = PS.declaration(PS.CATALOG)
    assert doc["catalog_fully_declared"] is True
    # diversity is by construction: no two catalogued perspectives share a family
    families = [p.spec.family for p in PS.CATALOG]
    assert len(families) == len(set(families))
    # the gap between declared and implemented families is stated, not hidden
    assert doc["families_without_an_implementation"]
    assert len(doc["families_implemented"]) + len(doc["families_without_an_implementation"]) == doc["families_declared"]


def test_a_perspective_cannot_read_what_it_did_not_declare():
    ws = W.Workspace()
    ws.write("perceptual", "sensor", {"label": "a", "label_confidence": 0.8}, provenance="camera")
    ws.write("world", "world_model", {"secret": True}, provenance="measured", confidence=0.9)

    direct = next(p for p in PS.CATALOG if p.spec.name == "direct")
    assert direct.run(ws).value == "a"
    assert "world" not in direct.view(ws)

    greedy = PS.Perspective(
        PS._spec("greedy", "direct_prediction", ("perceptual",), "peek", "label", 1.0, ("peeks",), "n/a"),
        lambda seen: (seen.get("world"), 1.0),
    )
    # declaring one region and reading another is refused by the workspace, not by convention
    object.__setattr__(greedy.spec, "inputs", ("perceptual", "world"))
    restricted = W.Workspace(
        (
            W.BY_NAME["perceptual"],
            W.RegionSpec(
                "world",
                "graph",
                "persistent",
                "slow",
                ("world_model",),
                ("world_model",),
                True,
                True,
                0.4,
                "persistent",
                "guarded",
            ),
        )
    )
    restricted.write("perceptual", "sensor", {"label": "a"}, provenance="camera")
    out = greedy.run(restricted)
    assert out.refused and out.value is None


def test_learned_selector_stays_closed_without_headroom():
    catalog = PS.CATALOG
    # every simple rung opens without ceremony
    for strategy in PS.SIMPLE_STRATEGIES:
        assert select_ok(strategy, catalog)
    # the learned rung does not
    with pytest.raises(W.Refused):
        PS.select("learned", catalog, reliability={"direct": 0.9})
    with pytest.raises(W.Refused):
        PS.select("learned", catalog, headroom={"residual_lower_95_cb": 0.01})  # below the SESOI
    opened = PS.select("learned", catalog, headroom={"residual_lower_95_cb": 0.09}, reliability={"critic": 0.99})
    assert opened[0].spec.name == "critic"
    # and an oracle selector without oracle scores is refused rather than guessed
    with pytest.raises(W.Refused):
        PS.select("oracle", catalog)


def select_ok(strategy, catalog):
    """A simple rung opens and returns at most k. A label rule may legitimately match fewer than k."""
    chosen = PS.select(
        strategy,
        catalog,
        task="predict",
        context={"regions": ["temporal"]},
        reliability={"direct": 0.7},
        oracle={"direct": 1.0},
    )
    return 1 <= len(chosen) <= 2


def test_selection_headroom_lower_bound_is_what_gates_the_learned_rung():
    tight = PS.selection_headroom(0.90, 0.88, spread=0.02, n_seeds=8)
    assert tight["residual_lower_95_cb"] < PS.SESOI
    wide = PS.selection_headroom(0.95, 0.80, spread=0.02, n_seeds=8)
    assert wide["residual_lower_95_cb"] > PS.SESOI
    with pytest.raises(W.Refused):
        PS.select("learned", PS.CATALOG, headroom=tight)
    assert PS.select("learned", PS.CATALOG, headroom=wide)


def test_minority_hypothesis_survives_arbitration():
    outputs = [_out("a", "left", 0.9), _out("b", "left", 0.8), _out("c", "right", 0.85)]
    report = PS.arbitrate(outputs)
    assert report["dominant_hypothesis"]["value"] == "left"
    assert report["minority_preserved"] == 1
    alt = report["alternative_hypotheses"][0]
    assert alt["value"] == "right" and alt["support"] == ["c"]
    assert alt["provenance"] == ["c:perceptual"], "the minority keeps its own provenance"
    assert report["unresolved_contradictions"][0]["alternative"] == "right"


def test_arbitration_defers_inside_the_sesoi_and_says_what_would_settle_it():
    outputs = [_out("a", "left", 0.51), _out("b", "right", 0.50)]
    report = PS.arbitrate(outputs)
    assert report["deferred"] is True
    assert report["decision"] is None
    assert report["provisional_belief"] == "left"
    assert report["required_evidence"] and "separates" in report["required_evidence"][0]
    assert report["confidence_interval"] == [0.5, 0.51]

    # with an affordable verifier the arbiter spends compute instead of deferring
    resolved = PS.arbitrate(outputs, budget=5.0, verifier=lambda v: v == "left")
    assert resolved["verification_ran"] is True
    assert resolved["deferred"] is False and resolved["decision"] == "left"
    assert resolved["compute_allocated"] > 0


def test_arbitration_combines_compatible_results_without_calling_them_a_contradiction():
    outputs = [_out("a", {"x": 1}, 0.9), _out("b", {"y": 2}, 0.8), _out("c", {"x": 9}, 0.7)]
    report = PS.arbitrate(outputs)
    assert report["dominant_hypothesis"]["value"] == {"x": 1, "y": 2}
    assert report["unresolved_contradictions"], "a conflicting key is still a contradiction"


def test_arbitration_over_only_refused_perspectives_defers_rather_than_inventing_an_answer():
    refused = PS.Output("blocked", None, 0.0, [], 1.0, refused="not a declared reader of world")
    report = PS.arbitrate([refused])
    assert report["deferred"] is True and report["decision"] is None
    assert report["refused"][0]["perspective"] == "blocked"


def test_historical_reliability_can_overturn_raw_confidence():
    outputs = [_out("loud", "wrong", 0.95), _out("quiet", "right", 0.60)]
    naive = PS.arbitrate(outputs)
    assert naive["dominant_hypothesis"]["value"] == "wrong"
    weighted = PS.arbitrate(outputs, reliability={"loud": 0.2, "quiet": 0.95})
    assert weighted["dominant_hypothesis"]["value"] == "right"
    assert weighted["minority_preserved"] == 1
