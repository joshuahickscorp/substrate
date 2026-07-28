from __future__ import annotations

import pytest

from substrate import v5stats


def test_v5_paired_effect_is_deterministic_and_keeps_raw_units() -> None:
    first = v5stats.paired_effect([0.11, 0.12, 0.13, 0.14], "endpoint", sesoi=0.05)
    second = v5stats.paired_effect([0.11, 0.12, 0.13, 0.14], "endpoint", sesoi=0.05)
    assert first == second
    assert first["raw_paired_effects"] == [0.11, 0.12, 0.13, 0.14]
    assert first["independent_unit"] == "developmental_history"
    assert first["clears_sesoi"]
    assert first["activation"] is False


def test_v5_paired_contrast_uses_strongest_control_per_history() -> None:
    result = v5stats.paired_contrast(
        {1: 0.9, 2: 0.8, 3: 0.85, 4: 0.88},
        {
            "fixed": {1: 0.7, 2: 0.6, 3: 0.65, 4: 0.66},
            "large": {1: 0.72, 2: 0.64, 3: 0.63, 4: 0.67},
        },
        "integrated",
        sesoi=0.05,
    )
    assert result["strongest_control_by_history"] == [
        "large",
        "large",
        "fixed",
        "large",
    ]
    assert result["mean"] > 0.15


def test_v5_statistics_refuse_unmatched_or_invalid_inputs() -> None:
    with pytest.raises(v5stats.Refused):
        v5stats.paired_effect([], "empty", sesoi=0.05)
    with pytest.raises(v5stats.Refused):
        v5stats.paired_contrast(
            {1: 1.0},
            {"bad": {2: 0.0}},
            "unmatched",
            sesoi=0.05,
        )
    with pytest.raises(v5stats.Refused):
        v5stats.holm({"bad": 2.0})


def test_v5_holm_is_step_down() -> None:
    report = v5stats.holm({"a": 0.001, "b": 0.01, "c": 0.2})
    assert report["rows"]["a"]["reject_zero"]
    assert report["rows"]["b"]["reject_zero"]
    assert not report["rows"]["c"]["reject_zero"]
    assert report["activation"] is False
