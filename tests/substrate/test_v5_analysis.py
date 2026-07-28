from __future__ import annotations

from substrate import v5analysis


def test_v5_matched_history_analysis_clears_frozen_endpoints() -> None:
    table = v5analysis.evaluate_histories("moderate_pilot", range(8_000, 8_016))
    report = v5analysis.effects(table)
    assert len(report["effects"]) == 15
    assert report["all_pass"]
    assert all(
        row["mean"] >= row["sesoi"] and row["bootstrap_95_ci"][0] > 0
        for row in report["effects"].values()
    )
    assert report["activation"] is False
