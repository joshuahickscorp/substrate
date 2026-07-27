"""SX1 is refused, and the refusal is classified as a method failure rather than a scientific null.

House style: no dashes.
"""

from __future__ import annotations

from substrate import deliverables as D
from substrate import experiments as X
from substrate import program as P
from substrate.method import graph as G


def test_sx1_declares_every_mandatory_contract():
    """The refusal has to be about the science, not about a missing form."""
    from substrate import admission as A

    prereg = X.sx1()
    assert A.completeness(prereg) == [], "SX1 is refused for its causal graph, not for a missing contract"


def test_sx1_is_refused_because_its_effect_is_true_by_construction():
    decision = X.sx1_decision()
    assert decision["licensed"] is False
    assert decision["classification"] == "methodological_refusal"
    violations = decision["causal_graph_violations"]
    assert any("no measured relation" in v for v in violations)
    assert any("broader than the measured path" in v for v in violations)
    assert decision["admission"]["blocked_at"] == "causal_model"


def test_relabelling_the_structural_edge_as_measured_does_not_rescue_it():
    """The obvious way to make SX1 pass is the exact defect the ledger already names."""
    cheated = {
        **X.SX1_GRAPH,
        "edges": [
            {**e, "type": "measured_relation", "actually": "structurally_guaranteed"} if e["src"] == "typed" and e["dst"] == "answer" else e
            for e in X.SX1_GRAPH["edges"]
        ],
    }
    violations = G.validate(cheated)
    assert any("reported as measured but is structurally_guaranteed" in v for v in violations)


def test_a_refusal_is_not_recorded_as_a_null():
    refusals = D.methodological_refusals()
    assert any(r["experiment_id"] == "SX1" for r in refusals), "run substrate.experiments seal first"
    doc = D.null_map(P.state())
    assert doc["refused_count"] >= 1
    assert "not a scientific null" in doc["refusal_rule"]
    # and the refused experiment does not appear among the nulls
    assert all(r.get("experiment_id") != "SX1" for r in doc["substrate_native_nulls"].values())


def test_the_refused_hypothesis_stays_open():
    graph = D.hypothesis_graph(P.state())
    typed = next(h for h in graph["hypotheses"] if h["id"] == "H_typed_workspace")
    assert typed["state"] == "instrument_pending"
    # two attempts refused for two different reasons, and the hypothesis is still untested by either
    assert typed["refused_attempts"] == ["SX1", "SX1b"]
    assert typed["still_open"] is True
    assert typed["blocking_null"] is None, "a refusal closes nothing downstream"


def test_sx1b_is_refused_on_power_and_never_touches_the_test_split():
    """The successor is a real experiment on a real bed, and it is still refused before principal."""
    out = X.sx1b_run()
    assert out["licensed"] is False
    assert out["admission"]["blocked_at"] == "power_and_units"
    assert out["causal_graph_violations"] == [], "the graph is sound; the block is the design's power"
    assert "principal" not in out, "a refused experiment never reaches the held out units"
    assert out["preprincipal_evidence"]["test_split_touched"] is False
    # the pre principal evidence is measured, not declared
    ev = out["preprincipal_evidence"]
    assert ev["units_disjoint"] is True and ev["arms_distinct"] is True
    assert ev["fitted_writer"] in ("static", "dynamic")
    assert 0.0 < ev["measured_reliability_on_train"][ev["fitted_writer"]] < 1.0


def test_the_sx1b_diagnosis_names_the_number_that_actually_decides():
    """The power block invites finding more units. On this bed that would not help, and it says so."""
    out = X.sx1b_run()
    d = out["diagnosis"]
    assert d["blocking_contract"] == "power"
    assert d["reported_mde"] > d["sesoi"]
    # the ceiling is what decides, and it does not depend on the unit count
    assert d["oracle_residual"] <= d["sesoi"]
    assert d["decisive_number"] == "oracle_residual"
    assert d["more_units_would_help"] is False
    assert d["classification"] == "bed_cannot_answer_the_question_at_this_effect_size"
    assert "untested on this bed, not refuted" in d["not_a_null"]


def test_the_bed_screen_rule_is_declared_before_its_outcome():
    """Searching beds until one clears the SESOI is the same defect as searching arms until one does."""
    rule = X.BED_SCREEN_RULE
    assert rule["sesoi"] == 0.05
    assert set(rule["candidates"]) == {"harth_stream", "pamap2_stream"}
    # the beds that are excluded say why, rather than being silently absent
    assert set(rule["excluded"]) == {"har_stream", "speech_stream"}
    assert all("invalid principal bed" in why for why in rule["excluded"].values())
    # and the outcome if nothing clears is fixed in advance, including not moving the SESOI
    assert "not lowered" in rule["outcome_if_none_clears"]
    assert "carried forward unchanged" in rule["prior_measurement"]


def test_no_bed_under_custody_can_test_the_typed_workspace_hypothesis():
    out = X.bed_screen()
    assert [r["bed"] for r in out["screened"]] == X.BED_SCREEN_RULE["candidates"]
    assert all(r["available"] for r in out["screened"]), "both caches are under custody"
    assert out["any_candidate"] is False
    assert out["classification"] == "no_bed_can_answer_at_this_effect_size"
    for row in out["screened"]:
        assert row["oracle_ceiling_lower_95_cb"] <= X.BED_SCREEN_RULE["sesoi"]
    assert "untested, not refuted" in out["not_a_null"]


def test_a_measurement_boundary_closes_nothing_downstream():
    graph = D.hypothesis_graph(P.state())
    typed = next(h for h in graph["hypotheses"] if h["id"] == "H_typed_workspace")
    boundary = typed["measurement_boundary"]
    assert boundary["closes_descendants"] is False
    assert boundary["best_ceiling_lower_95_cb"] <= boundary["sesoi"]
    assert typed["still_open"] is True and typed["blocking_null"] is None
    # the descendant is untouched
    dependent = next(h for h in graph["hypotheses"] if h["id"] == "H_arbitration_minority")
    assert dependent["state"] == "unopened" and dependent["blocking_null"] is None
    # and the boundary is filed apart from the nulls
    nulls = D.null_map(P.state())
    assert nulls["measurement_boundaries"]
    assert nulls["substrate_native_nulls"] == {}


def test_the_value_queue_refuses_what_the_evidence_already_closed():
    """Section 17's selection, computed by the queue the method reformation already sealed."""
    q = X.voi_queue()
    # the bed screen measured no oracle headroom for the typed workspace, so its successor is refused
    assert "SX1c" in q["refused"]
    sx1c = next(c for c in q["candidates"] if c["id"] == "SX1c")
    assert sx1c["refusal_reason"] == "no oracle headroom" and sx1c["priority"] == 0.0
    # and the learned plasticity policy is refused on the inherited null, not rerun
    sx7 = next(c for c in q["candidates"] if c["id"] == "SX7")
    assert "closed premise" in sx7["refusal_reason"]
    # two selections that between them separate more than one hypothesis
    assert len(q["selected"]) == 2
    assert len(q["hypotheses_covered_by_selection"]) >= 3
    # every candidate justifies its scores rather than asserting them
    assert all(c["justification"] for c in q["candidates"])
    assert "measures the generator" in q["program_lesson"]


def test_the_successor_design_is_not_a_closed_form():
    design = X.SX1B_DESIGN
    assert design["experiment_id"] == "SX1b"
    assert (
        "measured on training units" in design["why_it_is_not_a_closed_form"]
        or "measured on training" in design["why_it_is_not_a_closed_form"]
        or "measured on" in design["why_it_is_not_a_closed_form"]
    )
    # the arms include an oracle upper bound and a harm direction, not only the flattering one
    assert "typed_oracle" in design["arms"] and "typed_wrong" in design["arms"]
    # the three hypotheses do not all predict the same thing
    predictions = {tuple(sorted(v.items())) for v in design["predictions"].values()}
    assert len(predictions) == len(design["predictions"])
