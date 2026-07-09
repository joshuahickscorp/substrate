import json

from scripts.studio import pr9_continual_backprop as pr9


def test_pr9_run_state_receipt_records_resume_inventory(tmp_path):
    legs = tmp_path / "result.json.legs"
    legs.mkdir()
    (legs / "seed0_plain_lr0.01_rate0.json").write_text("{}\n")
    out = tmp_path / "result.json"
    state = tmp_path / "result.state.json"
    cfg = {
        "cache": "data/cache/dr1",
        "seeds": (0, 1),
        "cbp_rate_grid": (0.001, 0.005),
    }

    pr9._write_run_state(
        state,
        cfg,
        legs,
        out,
        status="running",
        stage="plain_seed_0",
        extra={"well_tuned_baseline_lr": 0.01},
    )

    receipt = json.loads(state.read_text())
    assert receipt["schema"] == "mop-pr9-run-state/v1"
    assert receipt["expected_leg_count"] == 6
    assert receipt["completed_leg_count"] == 1
    assert "loaded and skipped" in receipt["resume_behavior"]
