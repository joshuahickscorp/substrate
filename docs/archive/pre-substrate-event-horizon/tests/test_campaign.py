"""The campaign driver: resume from disk, stop on a repeated failure, and end rather than spin.

House style: no dashes.
"""

from __future__ import annotations

import json

import pytest

from mop.cognition import campaign as CP


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point the driver at a throwaway receipt root so the real campaign state is untouched."""
    receipts, holds = tmp_path / "stages", tmp_path / "failure_holds"
    receipts.mkdir()
    holds.mkdir()
    monkeypatch.setattr(CP, "RECEIPTS", receipts)
    monkeypatch.setattr(CP, "HOLDS", holds)
    monkeypatch.setattr(CP, "wave_index", lambda: 1)
    monkeypatch.setattr(CP.io, "run_json",
                        lambda name, obj, subdir="": (tmp_path / subdir).mkdir(exist_ok=True)
                        or (tmp_path / subdir / name).write_text(json.dumps(obj)))
    return tmp_path


def _mark(root, stage, ok=True, wave=1):
    (root / "stages" / f"{stage}.json").write_text(
        json.dumps({"stage": stage, "wave": wave, "ok": ok}))


def test_the_graph_is_acyclic_and_every_dependency_exists():
    for stage in CP.STAGES:
        for dep in stage.deps:
            assert dep in CP.BY_NAME, f"{stage.name} depends on an undeclared stage {dep}"
    seen = set()
    for stage in CP.STAGES:  # declared in dependency order, so every dep precedes its dependant
        assert set(stage.deps) <= seen, f"{stage.name} is declared before its dependency"
        seen.add(stage.name)


def test_a_stage_is_done_because_its_receipt_exists(isolated):
    assert CP.done("declarations") is False
    _mark(isolated, "declarations")
    assert CP.done("declarations") is True
    # a receipt from a different wave does not count as done in this one
    _mark(isolated, "tests", wave=99)
    assert CP.done("tests") is False
    # nor does a receipt that records a failure
    _mark(isolated, "experiments", ok=False)
    assert CP.done("experiments") is False


def test_a_stage_does_not_open_before_its_dependencies(isolated):
    assert [s.name for s in CP.ready()] == ["declarations"]
    _mark(isolated, "declarations")
    assert [s.name for s in CP.ready()] == ["tests"]
    _mark(isolated, "tests")
    assert [s.name for s in CP.ready()] == ["experiments"]


def test_a_repeated_failure_stops_being_retried(isolated):
    first = CP.record_failure("tests", "exit 1")
    assert first["attempts"] == 1 and first["state"] == "retry_allowed"
    assert CP.blocked("tests") is False

    second = CP.record_failure("tests", "exit 1")
    assert second["attempts"] == 2 and second["state"] == "implementation_change_required"
    assert CP.blocked("tests") is True
    # a blocked stage is not offered as ready, so the run cannot loop on it
    _mark(isolated, "declarations")
    assert "tests" not in [s.name for s in CP.ready()]


def test_a_hold_from_a_different_implementation_is_released(isolated, monkeypatch):
    CP.record_failure("tests", "exit 1")
    CP.record_failure("tests", "exit 1")
    assert CP.blocked("tests") is True
    # change the implementation authority, which is what a code change does
    monkeypatch.setattr(CP, "implementation_authority",
                        lambda: {"source_commit": "0" * 40, "source_tree_oid": "1" * 40})
    assert CP.holds("tests") == []
    assert CP.blocked("tests") is False


def test_the_run_ends_rather_than_spinning(isolated, monkeypatch):
    monkeypatch.setattr(CP, "run_stage",
                        lambda stage: (_mark(isolated, stage.name), {"ok": True})[1])
    out = CP.run(max_waves=1)
    wave = out["waves"][0]
    assert wave["terminal"] is True
    assert wave["unfinished"] == []
    assert [r["stage"] for r in wave["ran"]] == [s.name for s in CP.STAGES]


def test_a_failing_stage_halts_the_wave_instead_of_carrying_on(isolated, monkeypatch):
    def half(stage):
        ok = stage.name == "declarations"
        _mark(isolated, stage.name, ok=ok)
        return {"ok": ok}

    monkeypatch.setattr(CP, "run_stage", half)
    wave = CP.run(max_waves=1)["waves"][0]
    assert [r["stage"] for r in wave["ran"]] == ["declarations", "tests"]
    assert wave["terminal"] is False
    assert "tests" in wave["unfinished"]


def test_the_stop_switch_ends_the_run(isolated, monkeypatch):
    switch = isolated / "stop"
    switch.write_text("")
    monkeypatch.setattr(CP.io, "STOP", switch)
    out = CP.run(max_waves=3)
    assert out["waves"][0]["stopped"] == "stop switch"
    assert len(out["waves"]) == 1


def test_the_plan_says_which_waves_wait_on_something_that_does_not_exist():
    doc = CP.plan()
    assert len(doc["waves"]) == 6
    blocked = [w["wave"] for w in doc["waves"] if w["blocked_on_absent_prerequisite"]]
    assert blocked == ["W4_world_model_in_the_loop", "W5_model_body",
                       "W6_entity_batteries_on_real_sessions"]
    for wave in doc["waves"]:
        assert wave["entry"] and wave["exit"], f"{wave['wave']} has no entry or exit condition"
    assert "activation" in doc["explicitly_not_planned"][0]
    assert any("SESOI" in item for item in doc["explicitly_not_planned"])
    assert "running out of ideas is not a terminal condition" in \
        doc["termination"]["what_would_not_end_it"]
