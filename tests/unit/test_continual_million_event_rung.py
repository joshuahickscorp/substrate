from __future__ import annotations

import json

import pytest
from scripts.continual_million_event_rung import (
    PROGRESS_SCHEMA,
    _progress,
    build_plan,
    load_config,
)


def test_progressive_config_supports_exact_three_rungs_and_full_replication_matrix():
    config = load_config()
    assert config["null_hypothesis"].startswith("replay does not improve preregistered retention")
    assert config["replication"]["rungs"] == [10_000, 100_000, 1_000_000]
    for rung in config["replication"]["rungs"]:
        plan = build_plan(config, rung)
        assert plan["expected_cells"] == 2 * 3 * 5
        assert plan["schedules"] == ["abrupt", "gradual"]
        assert plan["arms"] == ["replay", "no-replay", "fresh-init"]
        assert len(plan["seeds"]) == 5
        assert plan["profile"] == {
            "checkpoint_every": 1250,
            "replay_capacity": 128,
            "future_window_events": 48,
            "threshold_window_events": 16,
            "future_accuracy_threshold": 0.5,
            "matched_updates_per_event": 2,
        }


def test_resource_probe_is_only_the_first_canonical_10k_cell():
    config = load_config()
    plan = build_plan(
        config,
        10_000,
        resource_probe=True,
        seed_count=1,
        schedules=("abrupt",),
        arms=("replay",),
    )
    assert plan["mode"] == "resource-probe"
    assert plan["cells"] == [{"seed": 20260710, "schedule": "abrupt", "arm": "replay"}]
    with pytest.raises(ValueError, match="exactly 10k"):
        build_plan(
            config,
            100_000,
            resource_probe=True,
            seed_count=1,
            schedules=("abrupt",),
            arms=("replay",),
        )


def test_full_rung_refuses_underreplication_or_missing_control():
    config = load_config()
    with pytest.raises(ValueError, match="at least five seeds"):
        build_plan(config, 10_000, seed_count=4)
    with pytest.raises(ValueError, match="every schedule and arm"):
        build_plan(config, 10_000, arms=("replay", "no-replay"))


def test_progress_receipt_resumes_only_under_exact_identity(tmp_path):
    path = tmp_path / "progress.json"
    identity = {"rung": 10_000, "mode": "resource-probe"}
    first, resumed = _progress(path, identity)
    assert resumed is False
    assert first["schema"] == PROGRESS_SCHEMA
    first["cells"]["seed_20260710/abrupt/replay"] = {"all_mechanics_ok": True}
    path.write_text(json.dumps(first))
    second, resumed = _progress(path, identity)
    assert resumed is True
    assert second["cells"] == first["cells"]
    with pytest.raises(ValueError, match="identity drift"):
        _progress(path, {"rung": 100_000, "mode": "replication"})
