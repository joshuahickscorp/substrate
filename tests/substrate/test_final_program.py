"""The final autonomous program: graph, authority, temporal link, bodies, sessions, goals, grounding,
divergence.

"""

from __future__ import annotations

import pytest

from substrate import authority as AU
from substrate import bodies as B
from substrate import divergence as DV
from substrate import evidence as io
from substrate import goals as GO
from substrate import graph as G
from substrate import grounding as GR
from substrate import sessions as S
from substrate import temporal_link as TL

# ---------------------------------------------------------------- section 5, the graph


def test_the_graph_is_valid_and_declares_every_required_field():
    doc = G.declaration()
    assert doc["valid"] is True, doc["declaration_violations"]
    assert set(G.NODE_FIELDS) <= set(doc["nodes"][0])
    for node in G.NODES:
        assert node.violations() == [], node.violations()
        assert node.exit_gate, f"{node.identity} has no exit gate"


def test_no_future_wave_exists_only_as_prose():
    """Section 5: the world model bed, the body adapters and the real session authority are nodes."""
    ids = set(G.BY_ID)
    for required in (
        "world_model_bed",
        "world_model_in_loop",
        "body_adapter_compact",
        "body_adapter_general",
        "body_adapter_tool",
        "real_session_authority",
    ):
        assert required in ids, f"{required} is not a node"
        assert G.BY_ID[required].module, f"{required} has nothing that can run it"
    assert G.declaration()["no_prose_waves"] is True


def test_a_buildable_blocker_is_work_and_only_an_external_one_is_terminal():
    doc = G.declaration()
    for row in doc["buildable_prerequisites"]:
        assert row["classification"] == "buildable_prerequisite"
        assert G.BY_ID[row["node"]].external_blocker == ""
    for row in doc["externally_blocked"]:
        assert row["blocker"], "an externally blocked node must say what is unavailable"


def test_an_exit_gate_cannot_be_passed_by_declaring_it_passed():
    ghost = G.Node(
        "ghost",
        "implementation",
        module="x",
        exit_gate="something",
        produces=("SUBSTRATE_DOES_NOT_EXIST.json",),
    )
    assert G.exit_passed(ghost)["passed"] is False
    assert "SUBSTRATE_DOES_NOT_EXIST.json" in G.exit_passed(ghost)["missing_artifacts"]


# ---------------------------------------------------------------- section 4, the authority


def test_every_requirement_carries_a_rollback():
    from substrate import program as P

    rows = AU.requirement_rows(P.state())
    assert rows and all(r["rollback"] for r in rows)
    assert all(
        set(r)
        >= {
            "id",
            "category",
            "status",
            "authority",
            "dependencies",
            "implementation",
            "tests",
            "experiment",
            "evidence",
            "classification",
            "commit",
            "rollback",
            "next_action",
        }
        for r in rows
    )


def test_the_authority_binds_the_final_plan_and_the_inherited_programs():
    from substrate import program as P

    doc = AU.master_authority(P.state())
    assert doc["plans"]["final"]["resolved"] is True
    assert doc["plans"]["final"]["sha256"]
    assert doc["source"]["pull_request"] == 35
    assert len(doc["inherited_authorities"]) >= 8
    assert all(a["resolved"] for a in doc["inherited_authorities"])
    assert "No conversation is required" in doc["resume_without_history"]


def test_the_ancestry_records_that_no_temporal_core_was_licensed():
    doc = AU.ancestry()
    verdict = doc["temporal_core_verdict"]
    assert verdict["terminal"] is True and verdict["licensed"] is False
    assert "unconverged" in verdict["why"]


# ---------------------------------------------------------------- section 9, temporal integration


def test_no_licensed_core_exists_and_the_control_says_so():
    with pytest.raises(TL.Refused):
        TL.LicensedCore()
    core = TL.resolve_core()
    assert isinstance(core, TL.DeclaredControl)
    assert core.is_control is True
    assert "no temporal core was scientifically licensed" in core.limitation


def test_the_five_information_sources_are_never_collapsed():
    core = TL.resolve_core()
    for v in (0.1, 0.5, 0.2):
        core.observe(v)
    view = TL.merged_view(core, observation={"x": 1}, retrieved=None, predicted=None)
    assert set(view["readings"]) == set(TL.SOURCES)
    assert view["collapsed"] is False
    # core state and explicit history are different sources even though both are temporal
    assert view["readings"]["temporal_core_state"]["source"] == "temporal_core_state"
    assert view["readings"]["explicit_history"]["source"] == "explicit_history"
    with pytest.raises(TL.Refused):
        TL.Reading(1, "vibes", 1.0)


def test_the_temporal_checkpoint_refuses_a_tampered_restore():
    core = TL.resolve_core()
    core.observe(0.5)
    snap = core.checkpoint()
    snap["state"] = [9.9]
    with pytest.raises(TL.Refused):
        TL.DeclaredControl().restore(snap)
    with pytest.raises(TL.Refused):
        core.reset(by="whoever_feels_like_it")


# ---------------------------------------------------------------- section 14, bodies


def test_three_body_classes_conform_through_one_interface():
    for klass in ("compact", "general", "tool"):
        doc = B.conformance(klass)
        assert doc["conformance"]["conforms"] is True, doc["conformance"]["missing"]
        assert doc["all_messages_valid"] is True, doc["message_gaps"]
        assert doc["n_parameters"] > 0
    # they are genuinely different bodies, not one body named three times
    sizes = {k: B.conformance(k)["n_parameters"] for k in ("compact", "general", "tool")}
    assert len(set(sizes.values())) == 3, sizes


def test_the_frontier_body_is_recorded_as_externally_blocked_not_substituted():
    doc = B.conformance("general")
    assert doc["external_blocker"], "the larger general body must say what it is not"
    assert "no frontier model weights" in doc["external_blocker"]


# ---------------------------------------------------------------- section 8 and 29, sessions


def test_the_session_authority_is_real_and_certified():
    doc = S.build()
    cert = S.certify(doc)
    assert cert["certified"] is True, cert["failed"]
    assert doc["event_count"] >= 50
    assert doc["provenance"]["every_event_cites_its_file"] is True
    assert doc["limitations"], "a one program session says so"


def test_divergence_has_a_working_control():
    doc = DV.run()
    assert doc["control_shows_no_divergence"] is True, doc["control_same_history"]
    assert doc["verdict"] != "invalid_control"
    assert set(doc["dimensions"]) == set(DV.DIMENSIONS)
    # a zero divergence result is reported as no divergence, not as divergence without value
    if not doc["diverged_dimensions"]:
        assert doc["verdict"] == "no_divergence"
        assert "not evidence of individuality" in doc["reading"]


# ---------------------------------------------------------------- sections 20, 23, 24


def test_grounding_refuses_a_symbol_with_no_referent():
    doc = GR.run()
    assert doc["verbal_definition_is_not_evidence"] is True
    row = doc["results"]["symbol_with_no_referent"]
    assert row["in_corpus"] is False and row["passes"] is True
    assert doc["limitation"], "grounding scope is stated"


def test_a_goal_cannot_authorize_itself_or_widen_its_parent():
    gs = GO.GoalSystem()
    good = dict(
        origin="operator",
        scope="s",
        priority=1.0,
        constraints=("activation stays false",),
        resources="local",
        progress_measure="p",
        termination="t",
        rollback="r",
        authority="pending",
    )
    with pytest.raises(GO.Refused):
        gs.authorize(GO.Goal("g", **good), external_authority="substrate.goals")
    root = gs.authorize(GO.Goal("root", **good), external_authority="SUBSTRATE_FINAL_AUTONOMOUS_PROGRAM.md")
    assert root.authority.endswith(".md")
    with pytest.raises(GO.Refused):
        gs.decompose("root", GO.Goal("wide", **{**good, "constraints": ()}))
    child = gs.decompose("root", GO.Goal("narrow", **{**good, "priority": 2.0}))
    assert child.priority <= root.priority, "a subgoal cannot outrank its parent"
    assert "activation stays false" in gs.active_constraints("narrow")


def test_valuation_is_authorized_and_refuses_to_be_fitted():
    scored = GO.value({"task_utility": 1.0, "risk": 0.2})
    assert scored["permitted"] is True and scored["weights_authority"]
    harmful = GO.value({"task_utility": 100.0, "harm_constraints": 1.0})
    assert harmful["permitted"] is False and harmful["score"] == float("-inf")
    with pytest.raises(GO.Refused):
        GO.value({"vibes": 1.0})
    with pytest.raises(GO.Refused):
        GO.fit_weights_from_preference({"anything": 1})


def test_every_sealed_artifact_has_exactly_one_producer():
    """Two producers for one filename means the last one to run decides what the evidence says.

    The clean clone found three of these at once: developmental history, the model body interface and
    the temporal core record were each written by two modules with different content.
    """
    import collections
    import re

    producers = collections.defaultdict(set)
    for f in sorted((io.ROOT / "src/substrate").glob("*.py")):
        for m in re.finditer(r"io\.seal(?:_md)?\(\s*[\"']([A-Z_0-9]+\.(?:json|md))[\"']", f.read_text()):
            producers[m.group(1)].add(f.stem)
    duplicated = {k: sorted(v) for k, v in producers.items() if len(v) > 1}
    assert duplicated == {}, duplicated
    assert len(producers) >= 25, "the scan found suspiciously few sealed artifacts"
