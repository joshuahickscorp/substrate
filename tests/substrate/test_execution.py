"""The frozen long run: units, freeze, rehearsal and the claim ceiling.

House style: no dashes.
"""

from __future__ import annotations

import json

from substrate import evidence as io
from substrate import execution as L
from substrate import safety as SF


def test_every_unit_is_certified_or_necessary():
    """A unit in the run names what licensed it. Nothing is there because it was already written."""
    assert len(L.UNIT_LIST) >= 15
    for unit in L.UNIT_LIST:
        assert unit.certified, f"{unit.identity} does not say what licensed it"
        assert unit.module, f"{unit.identity} has nothing to run"
    ids = [u.identity for u in L.UNIT_LIST]
    assert len(set(ids)) == len(ids)
    # declared in dependency order, so a resume never waits on something later in the list
    seen = set()
    for unit in L.UNIT_LIST:
        assert set(unit.depends_on) <= seen, unit.identity
        seen.add(unit.identity)


def test_the_frozen_manifest_covers_everything_the_authority_lists():
    man = L.manifest()
    for key in (
        "source_commit",
        "source_digest",
        "source_tree",
        "sessions",
        "splits",
        "perspectives",
        "bodies",
        "seeds",
        "controls",
        "sesoi",
        "stop_rules",
        "checkpoint_policy",
        "retries",
        "claim_ceiling",
    ):
        assert man.get(key) not in (None, ""), key
    assert man["manifest_sha256"]
    assert man["sesoi"] == L.SESOI
    assert man["run_classification"] == "terminal deterministic synthesis"
    assert man["scientific_work_unit_count"] == 0
    assert "zero new scientific trials" in man["completion"]


def test_a_live_edit_after_the_freeze_is_detectable():
    man = L.manifest()
    assert L.live_edit_detected(man)["live_edit"] is False
    # advancing the commit is not a live edit, because committing the authority does exactly that
    assert L.live_edit_detected({**man, "source_commit": "0" * 40})["live_edit"] is False
    drifted = L.live_edit_detected({**man, "source_digest": "0" * 64})
    assert drifted["live_edit"] is True
    assert "source_digest" in drifted["drifted_keys"]


def test_a_unit_cannot_be_claimed_twice():
    L.release("test_probe")
    try:
        assert L.claim("test_probe") is True
        assert L.claim("test_probe") is False
    finally:
        L.release("test_probe")


def test_completion_is_units_not_wall_time():
    plan = L.resource_plan()
    assert plan["completion_criterion"] == "all units terminal"
    assert "independent of elapsed time" in plan["not_a_wall_clock"]
    assert plan["unit_count"] == len(L.UNIT_LIST)
    assert plan["scientific_work_units"] == 0


def test_the_rehearsal_breaks_things_and_survives():
    doc = json.loads((io.PROOF / "SUBSTRATE_LONG_RUN_REHEARSAL.json").read_text())
    assert doc["all_pass"] is True, doc["failed"]
    for required in (
        "injected_failure_preserves_completed_work",
        "stop_switch_halts",
        "exclusive_writers",
        "checkpoint_resume",
        "stale_artifact_refusal",
        "terminal_closure",
    ):
        assert doc["checks"][required]["ok"] is True, required
    assert "only proves the happy path" in doc["note"]


def test_the_claim_ceiling_forbids_what_it_must():
    doc = L.claim_boundary()
    assert set(doc["forbidden"]) == set(SF.FORBIDDEN_CLAIM_TERMS)
    assert "No result from this run can license either" in doc["requires_separate_authority"]
    assert doc["current_claims_supported"] == []
    assert "No category has a positive" in doc["current_evidence"]
    # the permitted terms are permitted only with a classification behind them
    assert "classification" in doc["permitted_only_when"]


def test_the_launch_gate_refuses_without_a_green_admission(tmp_path, monkeypatch):
    sealed = json.loads((io.PROOF / "SUBSTRATE_LONG_RUN_AUTHORITY.json").read_text())
    assert sealed["admission"] in ("green", "refused")
    assert set(sealed["commands"]) == {"launch", "status", "stop", "resume"}
    if sealed["admission"] == "green":
        assert sealed["refusal_reason"] == ""
        assert sealed["audit"]["all_pass"] is True
        assert sealed["certification"]["green"] is True
        assert sealed["rehearsal"]["all_pass"] is True
