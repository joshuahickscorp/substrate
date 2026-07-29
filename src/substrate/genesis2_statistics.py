"""Frozen family-history-cell analysis for Cognitive Material Genesis II."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from substrate import genesis2_config as C2
from substrate import genesis_statistics as G1S

UnitKey = tuple[str, int]


class AnalysisRefused(RuntimeError):
    pass


def history_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[UnitKey, float]]:
    """One score per independently generated family-history cell."""
    cells: dict[tuple[str, UnitKey], list[float]] = {}
    for row in rows:
        unit = (str(row["family"]), int(row["history_id"]))
        cells.setdefault((str(row["arm"]), unit), []).append(float(row["score"]))
    result: dict[str, dict[UnitKey, float]] = {}
    for (arm, unit), values in cells.items():
        result.setdefault(arm, {})[unit] = sum(values) / len(values)
    return result


def mean_by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    scores = history_scores(rows)
    return {arm: sum(per_history.values()) / len(per_history) for arm, per_history in sorted(scores.items()) if per_history}


def resolve_comparator(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    parity_passed: Mapping[str, bool],
) -> dict[str, Any]:
    """Resolve the strongest eligible plastic monolith before principal."""
    means = mean_by_arm(rows)
    eligible_names = (
        C2.CANONICAL_S2_ID,
        C2.S2_LOW_BIT_ID,
        "FR_selected_kernel",
        "L1_associative_monolith",
    )
    eligible: dict[str, float] = {}
    excluded: dict[str, str] = {}
    for arm in eligible_names:
        if arm == candidate:
            excluded[arm] = "selected material itself"
        elif arm not in means:
            excluded[arm] = "not run"
        elif arm in C2.CONTROLS and not C2.CONTROLS[arm]["eligible_decisive_comparator"]:
            excluded[arm] = "diagnostic representation ablation only"
        elif not parity_passed.get(arm, False):
            excluded[arm] = "parity audit failed"
        elif C2.BASELINE_DEPRIVATION.get(arm, ()):
            excluded[arm] = "deprived control"
        else:
            eligible[arm] = means[arm]
    if not eligible:
        raise AnalysisRefused(f"no eligible monolithic comparator: {excluded}")
    comparator = max(sorted(eligible), key=lambda arm: eligible[arm])
    return {
        "comparator": comparator,
        "eligible": eligible,
        "excluded": excluded,
        "rule": C2.DECISIVE_COMPARATOR_RULE,
        "resolved_before_principal": True,
        "activation": False,
    }


def _paired(
    scores: Mapping[str, Mapping[UnitKey, float]],
    left: str,
    right: str,
) -> list[float]:
    if left not in scores or right not in scores:
        raise AnalysisRefused(f"missing arm for paired contrast: {left!r}, {right!r}")
    left_ids = set(scores[left])
    right_ids = set(scores[right])
    if left_ids != right_ids:
        raise AnalysisRefused(
            f"unpaired family-history cells for {left!r} versus {right!r}: left_only={sorted(left_ids - right_ids)}, right_only={sorted(right_ids - left_ids)}"
        )
    return [float(scores[left][unit]) - float(scores[right][unit]) for unit in sorted(left_ids)]


def paired_differences(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    comparator: str,
) -> list[float]:
    """Return the frozen paired-cell contrast for design and diagnostics."""
    return _paired(history_scores(rows), candidate, comparator)


def decisive_analysis(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    comparator: str,
    resamples: int | None = None,
) -> dict[str, Any]:
    scores = history_scores(rows)
    differences = paired_differences(rows, candidate=candidate, comparator=comparator)
    interval = G1S.bootstrap_interval(
        differences,
        resamples=resamples or cast(int, C2.STATISTICS["bootstrap_resamples"]),
        confidence=C2.CONFIDENCE,
    )
    p_value = G1S.paired_permutation_p(
        differences,
        resamples=resamples or cast(int, C2.STATISTICS["bootstrap_resamples"]),
    )
    oracle_headroom = sum(_paired(scores, "oracle", candidate)) / len(differences)
    unique_history_ids = len({int(row["history_id"]) for row in rows})
    primary = {
        "effect_at_least_sesoi": interval["effect"] >= C2.SESOI,
        "lower_bound_above_zero": interval["lower"] > 0.0,
        "oracle_headroom_at_least_minimum": oracle_headroom >= C2.MINIMUM_ORACLE_HEADROOM,
    }
    robust = {
        "lower_bound_at_least_sesoi": interval["lower"] >= C2.SESOI,
        "oracle_headroom_at_least_preferred": oracle_headroom >= C2.PREFERRED_ORACLE_HEADROOM,
    }
    return {
        "candidate": candidate,
        "comparator": comparator,
        "histories": unique_history_ids,
        "independent_units": len(differences),
        "unique_history_ids": unique_history_ids,
        "effect": interval["effect"],
        "confidence_lower": interval["lower"],
        "confidence_upper": interval["upper"],
        "p_value": p_value,
        "oracle_headroom": oracle_headroom,
        "primary_gate": primary,
        "primary_gate_pass": all(primary.values()),
        "robust_gate": robust,
        "robust_gate_pass": all(robust.values()),
        "independent_unit": C2.STATISTICS["independent_unit"],
        "confidence_method": C2.STATISTICS["confidence_method"],
        "activation": False,
    }


def family_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    """Exploratory only; never used to select or classify."""
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(str(row["family"]), []).append(row)
    effects: dict[str, Any] = {}
    for family, family_rows in sorted(by_family.items()):
        scores = history_scores(family_rows)
        differences = _paired(scores, candidate, comparator)
        effects[family] = {
            "effect": sum(differences) / len(differences),
            "histories": len(differences),
        }
    return {"effects": effects, "exploratory_only": True, "activation": False}


def demo() -> None:
    rows: list[dict[str, Any]] = []
    for history_id in range(32):
        rows.extend(
            (
                {"family": "a", "history_id": history_id, "arm": "field", "score": 0.6},
                {"family": "a", "history_id": history_id, "arm": "monolith", "score": 0.5},
                {"family": "a", "history_id": history_id, "arm": "oracle", "score": 1.0},
            )
        )
    report = decisive_analysis(rows, candidate="field", comparator="monolith", resamples=1_000)
    assert report["effect"] > C2.SESOI
    assert report["primary_gate_pass"]
    print("genesis2 statistics self-check passed")


if __name__ == "__main__":
    demo()
