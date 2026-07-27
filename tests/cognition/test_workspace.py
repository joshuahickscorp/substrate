"""The typed workspace: broad reads, narrow writes, and a control that removes exactly one capability.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from mop.cognition import workspace as W


def test_every_region_declares_all_ten_fields():
    for spec in W.REGIONS:
        assert spec.violations() == [], spec.violations()
    assert len(W.REGIONS) == 12  # section 10 of the final authority adds ontological and epistemic
    assert W.declaration()["all_regions_fully_declared"] is True


def test_a_write_outside_the_declared_writers_is_refused():
    ws = W.Workspace()
    ws.write("perceptual", "sensor", {"label": "a"}, provenance="camera")
    assert ws.read("perceptual", "any_reader") == {"label": "a"}
    with pytest.raises(W.Refused):
        ws.write("perceptual", "critic", {"label": "b"}, provenance="opinion")
    # the refused write changed nothing
    assert ws.read("perceptual", "any_reader") == {"label": "a"}
    assert ws.refusals == ["write perceptual by critic"]


def test_a_read_outside_the_declared_readers_is_refused():
    spec = W.RegionSpec("private", "scalar", "episode", "fast", ("owner",), ("owner",), True, False,
                        0.1, "episode", "owner only")
    ws = W.Workspace((spec,))
    ws.write("private", "owner", 1, provenance="self")
    assert ws.read("private", "owner") == 1
    with pytest.raises(W.Refused):
        ws.read("private", "stranger")


def test_provenance_and_confidence_are_mandatory_where_declared():
    ws = W.Workspace()
    with pytest.raises(W.Refused):
        ws.write("world", "world_model", {"x": 1}, provenance="", confidence=0.9)
    with pytest.raises(W.Refused):
        ws.write("world", "world_model", {"x": 1}, provenance="measured", confidence=None)
    ws.write("world", "world_model", {"x": 1}, provenance="measured", confidence=0.9)
    assert ws.broadcast()["world"]["provenance"] == "measured"


def test_global_availability_does_not_widen_write_access():
    """The whole point of section 6.2: information is globally readable, not globally writable."""
    ws = W.Workspace()
    ws.write("temporal", "temporal_core", {"history": [0.1, 0.4]}, provenance="core state")
    view = ws.broadcast()
    assert "temporal" in view and view["temporal"]["writer"] == "temporal_core"
    for spec in W.REGIONS:
        assert spec.readers == ("*",), "every region is globally readable"
        assert spec.writers != ("*",), f"{spec.name} would be globally writable"
    with pytest.raises(W.Refused):
        ws.write("temporal", "critic", {"history": [9.9]}, provenance="forged")


def test_reset_clears_only_regions_at_or_below_the_trigger():
    ws = W.Workspace()
    ws.write("perceptual", "sensor", 1, provenance="p")
    ws.write("temporal", "temporal_core", 2, provenance="p")
    ws.write("world", "world_model", 3, provenance="p", confidence=0.5)
    assert set(ws.reset("step")) == {"perceptual"}
    assert ws.read("temporal", "x") == 2 and ws.read("world", "x") == 3
    assert set(ws.reset("episode")) >= {"temporal"}
    assert ws.read("world", "x") == 3, "a persistent region must survive an episode reset"


def test_checkpoint_and_restore_recover_the_exact_state():
    ws = W.Workspace()
    ws.write("goal", "goal_authority", ["finish batch two"], provenance="operator")
    snap = ws.checkpoint()
    ws.write("goal", "goal_authority", ["drifted"], provenance="drift")
    assert ws.read("goal", "x") == ["drifted"]
    ws.restore(snap)
    assert ws.read("goal", "x") == ["finish batch two"]
    assert ws.writes == snap["writes"]


def test_the_untyped_control_removes_typing_and_nothing_else():
    typed, untyped = W.Workspace(), W.UntypedWorkspace()
    assert W.capacity(typed) == W.capacity(untyped), "the arms must be capacity matched"
    assert set(typed.specs) == set(untyped.specs)
    # the control retains cost accounting
    untyped.write("perceptual", "anyone_at_all", 1)
    assert untyped.spent == typed.specs["perceptual"].cost
    # and it is genuinely a control: the capability under test is gone
    untyped.write("world", "critic", {"forged": True})
    assert untyped.read("world", "critic") == {"forged": True}
    with pytest.raises(W.Refused):
        typed.write("world", "critic", {"forged": True}, provenance="p", confidence=0.5)


def test_the_budget_refuses_a_write_it_cannot_afford():
    ws = W.Workspace(budget=0.15)
    ws.write("perceptual", "sensor", 1, provenance="p")  # 0.1
    with pytest.raises(W.Refused):
        ws.write("temporal", "temporal_core", 2, provenance="p")  # 0.2 would exceed
    assert ws.spent == pytest.approx(0.1)
