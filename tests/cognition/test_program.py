"""Regressions for the Substrate master program.

Each of these guards one property the master plan states in a way that a later edit could quietly break:
historical names stay valid, deliverables bind to something real, batch selection stays dependency ready
and independent, and evidence never rises because code was written.

House style: no dashes.
"""

from __future__ import annotations

import copy

import pytest

from mop.cognition import deliverables as D
from mop.cognition import io
from mop.cognition import program as P


def test_naming_authority_preserves_historical_programs():
    """Section 2: historical identities stay valid and nothing was mass renamed."""
    assert D.NAMING["renamed_nothing"] is True
    assert set(D.NAMING["map"]) == {"MOP", "Mixture of Perspectives", "Mixture of Thinking"}
    # the historical proof roots are still where their sealed receipts say they are
    for root in ("proof/method/mop-experimental-method-reformation-v1",
                 "proof/substrate/mop-fast-state-plasticity-forge-v1",
                 "proof/substrate/mop-temporal-core-mechanism-v1"):
        assert (io.ROOT / root).is_dir(), f"historical authority root vanished: {root}"


def test_every_deliverable_binds_to_a_real_path():
    """Section 21: artifacts bind to implementation, evidence or a terminal gate. No empty placeholders."""
    for item in P.ITEMS:
        if item.evidence:
            assert item.impl, f"{item.id} declares evidence with no implementation that could produce it"
        for ref in item.evidence:
            root, _, name = ref.rpartition(":")
            assert root in D.ROOTS, f"{item.id} names an unknown proof root {root!r}"
            assert name.endswith((".json", ".md")), f"{item.id} names a malformed artifact {name!r}"
    # nothing sealed into this program's proof root is undeclared
    declared = {e.rpartition(":")[2] for item in P.ITEMS for e in item.evidence}
    declared |= {"SUBSTRATE_STATE.json", "SUBSTRATE_LEDGER.md", "SUBSTRATE_HYPOTHESIS_GRAPH.json",
                 "SUBSTRATE_NULL_MAP.json", "SUBSTRATE_NEXT_FRONTIER.json",
                 "SUBSTRATE_FINAL_LEDGER.md"}
    if io.PROOF.is_dir():
        for path in io.PROOF.glob("SUBSTRATE_*"):
            assert path.name in declared, f"undeclared artifact sealed into the proof root: {path.name}"


def test_batch_selection_is_dependency_ready_and_independent():
    """Section 17: one primary and one independent secondary, both able to start now."""
    st = P.state()
    frontier = P.next_batches(st)
    primary, secondary = frontier["primary"], frontier["secondary"]
    assert primary, "no dependency ready work was found, which cannot be true while items are unstarted"
    assert not st["items"][primary["id"]]["unmet_dependencies"]
    if secondary:
        a, b = st["items"][primary["id"]], st["items"][secondary["id"]]
        assert not b["unmet_dependencies"]
        assert a["id"] not in b["dependencies"] and b["id"] not in a["dependencies"]
        assert not set(a["dependencies"]) & set(b["dependencies"])


def test_evidence_never_rises_from_implementation_alone():
    """Section 20: code existing raises implementation and nothing else."""
    st = copy.deepcopy(P.state())
    for row in st["items"].values():
        row["level"] = "measured"  # every file present, every test green, every artifact sealed
        row["result"] = None  # and not one scientific classification recorded
    card = P.scorecard(st)
    scored = [c for c in card["categories"].values() if c["items"]]
    assert scored, "no category has members"
    assert all(c["evidence_pct"] == 0 for c in scored), "evidence rose without a classification"
    assert all(c["implementation_pct"] == 100 for c in scored)

    # and a classification in one category must not move another
    one = next(r for r in st["items"].values() if r["category"] == "working_memory")
    one["result"] = {"classification": "positive", "scientific": True}
    moved = P.scorecard(st)["categories"]
    assert moved["working_memory"]["evidence_pct"] > 0
    assert all(v["evidence_pct"] == 0 for k, v in moved.items()
               if k != "working_memory" and v["items"])


def test_evidence_that_is_stale_or_failing_does_not_count(tmp_path, monkeypatch):
    """Correction C_EVIDENCE_PRESENCE: an artifact existing is not an artifact passing.

    The first derivation counted any file that existed. The temporal core verification receipt exists,
    reports all_pass false, and was sealed at a commit that is not an ancestor of this branch head, and
    item C1 read as measured because of it. All three refusals are checked here.
    """
    import json as _json

    root = tmp_path / "proof"
    root.mkdir()
    monkeypatch.setitem(P.PROOF_ROOTS, "tmp", root)
    monkeypatch.setattr(P, "_REACHABLE", {}, raising=False)
    live = P.io.commit()

    (root / "good.json").write_text(_json.dumps({"all_pass": True, "source_commit": live}))
    (root / "failing.json").write_text(_json.dumps({"all_pass": False, "source_commit": live}))
    (root / "stale.json").write_text(_json.dumps({"all_pass": True, "source_commit": "0" * 40}))
    (root / "unstamped.json").write_text(_json.dumps({"all_pass": True}))

    assert P.evidence_state("tmp:good.json")["counts"] is True
    failing = P.evidence_state("tmp:failing.json")
    assert failing["present"] is True and failing["counts"] is False
    assert "all_pass" in failing["reason"]
    stale = P.evidence_state("tmp:stale.json")
    assert stale["present"] is True and stale["counts"] is False
    assert "ancestor" in stale["reason"]
    assert P.evidence_state("tmp:unstamped.json")["counts"] is False
    assert P.evidence_state("tmp:absent.json")["present"] is False

    # and an item whose only evidence is refused cannot climb past tested
    item = P.Item("Z8", "0", "stale evidence", "an item bound to a failing receipt",
                  impl=("src/mop/cognition/program.py",),
                  tests=("tests/cognition/test_program.py::test_status_is_derived_not_asserted",),
                  evidence=("tmp:failing.json",))
    row = P.item_status(item, {item.tests[0]: True}, {}, {})
    assert row["level"] == "tested"
    assert row["evidence"]["present_but_refused"]


def test_an_authority_is_terminal_once_sealed_and_tested(tmp_path, monkeypatch):
    """Correction C_AUTHORITY_TERMINALITY: an authority is not an experiment.

    Before this, every authority parked at measured forever waiting for a scientific classification it
    can never have, and outranked genuinely unstarted work in the selection queue.
    """
    import json as _json

    root = tmp_path / "proof"
    root.mkdir()
    monkeypatch.setitem(P.PROOF_ROOTS, "tmp", root)
    monkeypatch.setattr(P, "_REACHABLE", {}, raising=False)
    (root / "auth.json").write_text(_json.dumps({"source_commit": P.io.commit()}))

    common = dict(impl=("src/mop/cognition/program.py",),
                  tests=("tests/cognition/test_program.py::test_status_is_derived_not_asserted",),
                  evidence=("tmp:auth.json",))
    green = {common["tests"][0]: True}

    for kind in ("authority", "boundary"):
        row = P.item_status(P.Item("Z1", "0", "an authority", "declared", kind=kind, **common),
                            green, {}, {})
        assert row["level"] == "terminal", kind

    # an empirical item is not let through on the same evidence
    row = P.item_status(P.Item("Z2", "0", "an experiment", "declared", kind="implementation", **common),
                        green, {}, {})
    assert row["level"] == "measured"
    assert "classify" in row["next_action"]


def test_state_reports_whether_the_source_tree_was_clean():
    """Correction C_DIRTY_SRC_HALTS_SUPERVISOR: a state file written from a dirty tree must say so."""
    tree = P.source_tree_state()
    assert set(tree) == {"clean", "dirty_paths", "why_it_matters"}
    assert isinstance(tree["clean"], bool)
    assert tree["clean"] == (not tree["dirty_paths"])
    assert "supervisor" in tree["why_it_matters"]
    assert P.state()["source_tree"]["clean"] == tree["clean"]


def test_corrections_are_append_only():
    recorded = {c["correction_id"] for c in P.corrections()}
    assert {"C_EVIDENCE_PRESENCE", "C_AUTHORITY_TERMINALITY"} <= recorded
    for c in P.corrections():
        assert c["regression_test"], f"{c['correction_id']} has no permanent regression test"
        assert c["reproduced_by"], f"{c['correction_id']} does not say how it was reproduced"
    # rewriting an existing correction with different content is refused
    first = P.corrections()[0]
    with pytest.raises(ValueError):
        P.record_correction(first["correction_id"], "a different defect", "a different fix",
                            "a different test", "a different observation")


def test_status_is_derived_not_asserted():
    """An item whose declared file is absent can never report implemented."""
    ghost = P.Item("Z9", "0", "ghost", "a requirement with no implementation on disk",
                   impl=("src/mop/cognition/does_not_exist.py",), tests=(), evidence=())
    row = P.item_status(ghost, {}, {}, {})
    assert row["level"] == "not_started"
    assert row["next_action"].startswith("implement")
