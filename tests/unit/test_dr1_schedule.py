import pytest

from mop.studio.dr1_schedule import build_dr1_schedule_plan, daemon_plan_from_dr1_schedule_plan


def _schedule(**overrides):
    schedule = {
        "ok_to_launch": True,
        "blocked_reasons": [],
        "effective_clips": 10,
        "profile": {"name": "studio-m1ultra"},
        "winner": {"device": "cpu", "wall_s_per_clip": 12.0, "workers": 16},
        "checkpoint": {"every_clips": 4, "every_min": 30},
        "thermal_pacing": {"mode": "monitor", "heartbeat_every_min": 5, "pause_s": 0},
    }
    schedule.update(overrides)
    return schedule


def test_dr1_schedule_plan_splits_by_checkpoint_and_carries_device():
    plan = build_dr1_schedule_plan(
        _schedule(),
        source="/data/dr1",
        factors=("object", "action"),
        min_per_cell=8,
    )
    assert plan["ok_to_launch"] is True
    assert plan["ranges"] == [[0, 4], [4, 8], [8, 10]]
    assert plan["summary"]["encode_legs"] == 3
    leg = next(job for job in plan["jobs"] if job["id"] == "dr1_encode_000000_000004")
    assert leg["cmd"][-2:] == ["--device", "cpu"]
    assert leg["estimated_wall_min"] == 0.8


def test_dr1_schedule_plan_emits_daemon_jobs_in_execution_order():
    plan = build_dr1_schedule_plan(_schedule(effective_clips=5), source="/data/dr1")
    daemon = daemon_plan_from_dr1_schedule_plan(plan)
    assert daemon["schema"] == "mop-long-run-daemon/v1"
    assert [job["id"] for job in daemon["jobs"]] == [
        "dr1_caption_gate",
        "dr1_encode_000000_000004",
        "dr1_encode_000004_000005",
        "dr1_merge",
        "dr1_a6_guard",
    ]
    merge = next(job for job in daemon["jobs"] if job["id"] == "dr1_merge")
    assert "--source" in merge["cmd"]
    assert daemon["jobs"][0]["kind"] == "verdict-gate"
    assert daemon["jobs"][-1]["kind"] == "verdict-gate"


def test_blocked_schedule_has_no_jobs_and_no_daemon_plan():
    plan = build_dr1_schedule_plan(
        _schedule(
            ok_to_launch=False, winner=None, effective_clips=0, checkpoint={}, blocked_reasons=["no speed"]
        ),
        source="/data/dr1",
    )
    assert plan["ok_to_launch"] is False
    assert plan["jobs"] == []
    assert any("schedule blocked" in reason for reason in plan["blocked_reasons"])
    with pytest.raises(ValueError, match="blocked DR1 schedule"):
        daemon_plan_from_dr1_schedule_plan(plan)
