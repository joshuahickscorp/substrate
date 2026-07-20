from __future__ import annotations

import copy

import pytest

from mop.mechanisms.construction_search_bed import CHEAP_CONTROLS
from mop.studies import generation1_c3_construction as construction


def _ranges(count: int = 2) -> list[dict[str, object]]:
    return [
        {
            "name": "primary",
            "role": "pilot-primary",
            "start": 20276001,
            "count": count,
            "source": construction.FRESH_SOURCE,
        },
        {
            "name": "challenge",
            "role": "pilot-challenge",
            "start": 20277001,
            "count": count,
            "source": construction.FRESH_SOURCE,
        },
    ]


def _config(**overrides: object) -> dict:
    return construction.build_pilot_config(seed_ranges=_ranges(), **overrides)


def _reseal(value: dict, field: str) -> None:
    value[field] = construction.canonical_sha256(
        {key: item for key, item in value.items() if key != field}
    )


def test_fresh_multirange_pilot_is_sealed_and_mechanics_only() -> None:
    config = _config()
    result = construction.run_pilot(config)
    construction.validate_pilot_result(result)

    assert result["overall"]["seed_count"] == 4
    assert result["overall"]["discrimination_fraction"] == 1.0
    assert result["decision"]["verdict"] == "mechanics-pilot-favorable"
    assert result["decision"]["mechanics_pattern_observed"] is True
    assert result["activation_allowed"] is False
    assert result["scientific_promotion"] is False
    assert result["independent_verification_complete"] is False
    assert {row["range_name"] for row in result["rows"]} == {"primary", "challenge"}
    assert all(row["mechanics_receipt"]["is_confirmation"] is False for row in result["rows"])


def test_all_charged_controls_and_oracle_headroom_are_aggregated() -> None:
    result = construction.run_pilot(_config())
    assert list(result["overall"]["favorable_charged_margin_by_control"]) == list(CHEAP_CONTROLS)
    for control in CHEAP_CONTROLS:
        assert result["overall"]["favorable_charged_margin_by_control"][control]["mean"] > 0.0
    assert result["overall"]["oracle_headroom"]["mean_gap"] >= 0.0
    assert result["overall"]["oracle_headroom"]["interpretation"] == (
        "uncharged exhaustive diagnostic only"
    )


def test_pilot_is_exactly_deterministic() -> None:
    config = _config()
    first = construction.run_pilot(config)
    second = construction.run_pilot(config)
    assert first == second
    assert first["result_sha256"] == second["result_sha256"]


def test_flat_favorable_regime_fails_closed_to_null() -> None:
    result = construction.run_pilot(_config(synergy_bonus=0.0))
    assert result["decision"]["verdict"] == "mechanics-pilot-null"
    assert result["decision"]["mechanics_pattern_observed"] is False
    assert result["overall"]["discrimination_fraction"] == 0.0
    assert result["activation_allowed"] is False
    assert result["scientific_promotion"] is False


def test_high_charged_cost_fails_closed_to_null() -> None:
    result = construction.run_pilot(_config(per_eval_cost=0.01))
    assert result["decision"]["verdict"] == "mechanics-pilot-null"
    assert result["decision"]["conditions"][
        "positive_mean_charged_margin_over_every_control"
    ] is False


def test_overlapping_or_prior_seed_ranges_are_rejected() -> None:
    overlap = _ranges()
    overlap[1]["start"] = 20276002
    with pytest.raises(construction.Generation1C3ConstructionRefusal, match="overlap"):
        construction.build_pilot_config(seed_ranges=overlap)

    prior = _ranges()
    prior[0]["start"] = construction.C2_MAX_SEED
    with pytest.raises(construction.Generation1C3ConstructionRefusal, match="prior C1/C2"):
        construction.build_pilot_config(seed_ranges=prior)


def test_config_self_seal_rejects_mutation() -> None:
    config = _config()
    config["bed"]["random_samples"] += 1
    with pytest.raises(construction.Generation1C3ConstructionRefusal, match="self-seal"):
        construction.validate_pilot_config(config)


def test_result_rejects_resealed_arithmetic_mutation() -> None:
    result = construction.run_pilot(_config())
    mutated = copy.deepcopy(result)
    row = mutated["rows"][0]
    row["favorable"]["charged_margins"]["greedy-only"] += 0.5
    _reseal(row, "row_sha256")
    _reseal(mutated, "result_sha256")
    with pytest.raises(construction.Generation1C3ConstructionRefusal, match="charged margin"):
        construction.validate_pilot_result(mutated)


def test_result_rejects_scientific_or_activation_promotion() -> None:
    result = construction.run_pilot(_config())
    for field in ("scientific_promotion", "activation_allowed"):
        mutated = copy.deepcopy(result)
        mutated[field] = True
        _reseal(mutated, "result_sha256")
        with pytest.raises(construction.Generation1C3ConstructionRefusal, match="cannot"):
            construction.validate_pilot_result(mutated)
