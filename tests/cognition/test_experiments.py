"""SX1 is refused, and the refusal is classified as a method failure rather than a scientific null.

House style: no dashes.
"""

from __future__ import annotations

from mop.cognition import deliverables as D
from mop.cognition import experiments as X
from mop.cognition import program as P
from mop.method import graph as G


def test_sx1_declares_every_mandatory_contract():
    """The refusal has to be about the science, not about a missing form."""
    from mop.cognition import admission as A
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
    cheated = {**X.SX1_GRAPH, "edges": [
        {**e, "type": "measured_relation", "actually": "structurally_guaranteed"}
        if e["src"] == "typed" and e["dst"] == "answer" else e
        for e in X.SX1_GRAPH["edges"]]}
    violations = G.validate(cheated)
    assert any("reported as measured but is structurally_guaranteed" in v for v in violations)


def test_a_refusal_is_not_recorded_as_a_null():
    refusals = D.methodological_refusals()
    assert any(r["experiment_id"] == "SX1" for r in refusals), "run mop.cognition.experiments seal first"
    doc = D.null_map(P.state())
    assert doc["refused_count"] >= 1
    assert "not a scientific null" in doc["refusal_rule"]
    # and the refused experiment does not appear among the nulls
    assert all(r.get("experiment_id") != "SX1" for r in doc["substrate_native_nulls"].values())


def test_the_refused_hypothesis_stays_open():
    graph = D.hypothesis_graph(P.state())
    typed = next(h for h in graph["hypotheses"] if h["id"] == "H_typed_workspace")
    assert typed["state"] == "instrument_pending"
    assert typed["refused_attempts"] == ["SX1"]
    assert typed["still_open"] is True
    assert typed["blocking_null"] is None, "a refusal closes nothing downstream"


def test_the_successor_design_is_not_a_closed_form():
    design = X.SX1B_DESIGN
    assert design["experiment_id"] == "SX1b"
    assert "measured on training units" in design["why_it_is_not_a_closed_form"] or \
        "measured on training" in design["why_it_is_not_a_closed_form"] or \
        "measured on" in design["why_it_is_not_a_closed_form"]
    # the arms include an oracle upper bound and a harm direction, not only the flattering one
    assert "typed_oracle" in design["arms"] and "typed_wrong" in design["arms"]
    # the three hypotheses do not all predict the same thing
    predictions = {tuple(sorted(v.items())) for v in design["predictions"].values()}
    assert len(predictions) == len(design["predictions"])
