"""Frozen analysis for Substrate Cognitive Material Genesis.

The estimator, the resampling unit, the interval method and the multiplicity
correction are fixed here before any principal instance exists. The decisive
comparator is resolved by a rule over eligible controls, never by choosing the
arm that happens to make the effect look best.

The independent unit is the developmental history. Episodes inside one history
are not independent, so every interval resamples histories, never episodes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from substrate import genesis_config as C


class AnalysisRefused(RuntimeError):
    """The analysis was asked to do something the frozen constitution forbids."""


@dataclass(frozen=True, slots=True)
class HistoryScore:
    """One scalar per arm per developmental history. The only analysis input."""

    history_id: int
    arm: str
    score: float


def _paired(scores: Sequence[HistoryScore], candidate: str, comparator: str) -> list[float]:
    """Differences on shared histories only, in ascending history order."""
    by_arm: dict[str, dict[int, float]] = {}
    for row in scores:
        by_arm.setdefault(row.arm, {})[row.history_id] = row.score
    if candidate not in by_arm:
        raise AnalysisRefused(f"no scores for candidate {candidate!r}")
    if comparator not in by_arm:
        raise AnalysisRefused(f"no scores for comparator {comparator!r}")
    shared = sorted(set(by_arm[candidate]) & set(by_arm[comparator]))
    if not shared:
        raise AnalysisRefused("candidate and comparator share no developmental history")
    unpaired = (set(by_arm[candidate]) | set(by_arm[comparator])) - set(shared)
    if unpaired:
        raise AnalysisRefused(f"unpaired histories would bias the estimate: {sorted(unpaired)}")
    return [by_arm[candidate][history] - by_arm[comparator][history] for history in shared]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _normal_quantile(p: float) -> float:
    """Acklam's inverse normal CDF. Deterministic, no scipy."""
    if not 0.0 < p < 1.0:
        raise AnalysisRefused("quantile outside the open unit interval")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    low, high = 0.02425, 1 - 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _resample_indices(count: int, replicate: int, seed: int) -> list[int]:
    """Deterministic resampling. A fixed seed makes the interval reproducible."""
    state = (seed * 0x9E3779B97F4A7C15 + replicate * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    indices = []
    for _ in range(count):
        state ^= (state >> 30) & 0xFFFFFFFFFFFFFFFF
        state = (state * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        state ^= (state >> 27) & 0xFFFFFFFFFFFFFFFF
        state = (state * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        state ^= (state >> 31) & 0xFFFFFFFFFFFFFFFF
        indices.append(state % count)
    return indices


def bootstrap_interval(
    differences: Sequence[float],
    *,
    resamples: int | None = None,
    confidence: float | None = None,
    seed: int = 0x53_55_42_5F,
) -> dict[str, float]:
    """Bias-corrected and accelerated interval over the resampling unit.

    Resamples are drawn over developmental histories, which is the only unit
    the constitution treats as independent.
    """
    resamples = resamples if resamples is not None else int(C.STATISTICS["bootstrap_resamples"])
    confidence = confidence if confidence is not None else C.CONFIDENCE
    count = len(differences)
    if count < 2:
        raise AnalysisRefused("a confidence interval needs at least two developmental histories")
    observed = _mean(differences)

    replicates = []
    for replicate in range(resamples):
        indices = _resample_indices(count, replicate, seed)
        replicates.append(_mean([differences[index] for index in indices]))
    replicates.sort()

    below = sum(1 for value in replicates if value < observed)
    if below in (0, resamples):
        # Degenerate bootstrap distribution: fall back to the percentile
        # interval rather than producing an undefined correction.
        bias = 0.0
    else:
        bias = _normal_quantile(below / resamples)

    jackknife = []
    for omitted in range(count):
        rest = differences[:omitted] + differences[omitted + 1 :]
        jackknife.append(_mean(rest))
    jackknife_mean = _mean(jackknife)
    numerator = sum((jackknife_mean - value) ** 3 for value in jackknife)
    denominator = 6.0 * (sum((jackknife_mean - value) ** 2 for value in jackknife) ** 1.5)
    acceleration = numerator / denominator if denominator else 0.0

    alpha = (1.0 - confidence) / 2.0

    def adjusted(probability: float) -> float:
        z = _normal_quantile(probability)
        adjustment = bias + (bias + z) / max(1e-12, 1.0 - acceleration * (bias + z))
        return min(max(_normal_cdf(adjustment), 0.0), 1.0)

    def percentile(probability: float) -> float:
        position = probability * (resamples - 1)
        lower = int(math.floor(position))
        upper = min(lower + 1, resamples - 1)
        weight = position - lower
        return replicates[lower] * (1 - weight) + replicates[upper] * weight

    return {
        "effect": observed,
        "lower": percentile(adjusted(alpha)),
        "upper": percentile(adjusted(1.0 - alpha)),
        "histories": float(count),
        "resamples": float(resamples),
        "bias_correction": bias,
        "acceleration": acceleration,
    }


def paired_permutation_p(differences: Sequence[float], *, resamples: int = 10_000, seed: int = 0x53_55_42_5F) -> float:
    """Two-sided sign-flip p value. The pairing is the history."""
    count = len(differences)
    observed = abs(_mean(differences))
    extreme = 0
    for replicate in range(resamples):
        state = (seed ^ (replicate * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
        total = 0.0
        for value in differences:
            state ^= (state >> 33) & 0xFFFFFFFFFFFFFFFF
            state = (state * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
            total += value if state & 1 else -value
        if abs(total / count) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (resamples + 1)


def holm(p_values: Mapping[str, float], *, alpha: float | None = None) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni over the family of primary claims."""
    alpha = alpha if alpha is not None else float(C.STATISTICS["decisive_claim_alpha"])
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    result: dict[str, dict[str, Any]] = {}
    rejected_so_far = True
    for rank, (name, value) in enumerate(ordered):
        threshold = alpha / (total - rank)
        rejected = rejected_so_far and value <= threshold
        rejected_so_far = rejected
        result[name] = {"p_value": value, "threshold": threshold, "rank": rank + 1, "rejected": rejected}
    return result


def resolve_decisive_comparator(
    scores: Sequence[HistoryScore],
    *,
    parity_passed: Mapping[str, bool],
    separate_implementation: Mapping[str, bool],
) -> dict[str, Any]:
    """Pick the arm the decisive claim must beat, by rule and before unblinding.

    Eligible arms are plastic, deprived of nothing, parity-audited and
    separately implemented. Among those, the highest scoring one is chosen,
    which is the hardest comparator rather than a convenient one.
    """
    eligible: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for arm in sorted({row.arm for row in scores}):
        canonical = C.S2_ALIASES.get(arm, arm)
        control = C.CONTROLS.get(canonical)
        if control is None:
            reasons[arm] = "not a registered control"
            continue
        if not control.get("plastic"):
            reasons[arm] = "not plastic"
            continue
        if C.BASELINE_DEPRIVATION.get(canonical, ("unknown",)):
            reasons[arm] = "deprived of an opportunity"
            continue
        if not parity_passed.get(arm, False):
            reasons[arm] = "parity audit did not pass"
            continue
        if not separate_implementation.get(arm, False):
            reasons[arm] = "not a separate implementation"
            continue
        values = [row.score for row in scores if row.arm == arm]
        eligible[arm] = _mean(values)
    if not eligible:
        raise AnalysisRefused(f"no eligible decisive comparator: {reasons}")
    chosen = max(sorted(eligible), key=lambda arm: eligible[arm])
    return {
        "comparator": chosen,
        "canonical": C.S2_ALIASES.get(chosen, chosen),
        "eligible": eligible,
        "excluded": reasons,
        "rule": C.DECISIVE_COMPARATOR_RULE,
    }


def decisive_analysis(
    scores: Sequence[HistoryScore],
    *,
    candidate: str,
    comparator: str,
    oracle: str | None = None,
) -> dict[str, Any]:
    """The frozen P10 analysis. Both the primary and the stricter gate."""
    differences = _paired(scores, candidate, comparator)
    interval = bootstrap_interval(differences)
    p_value = paired_permutation_p(differences)

    headroom = None
    if oracle is not None:
        oracle_differences = _paired(scores, oracle, candidate)
        headroom = _mean(oracle_differences)

    primary = {
        "effect_at_least_sesoi": interval["effect"] >= C.SESOI,
        "lower_bound_above_zero": interval["lower"] > 0.0,
    }
    robust = {
        "lower_bound_at_least_sesoi": interval["lower"] >= C.SESOI,
        "oracle_headroom_at_least_preferred": (headroom is not None and headroom >= C.PREFERRED_ORACLE_HEADROOM),
    }
    return {
        "candidate": candidate,
        "comparator": comparator,
        "comparator_canonical": C.S2_ALIASES.get(comparator, comparator),
        "histories": len(differences),
        "estimator": C.STATISTICS["primary_estimator"],
        "confidence_method": C.STATISTICS["confidence_method"],
        "effect": interval["effect"],
        "confidence_lower": interval["lower"],
        "confidence_upper": interval["upper"],
        "p_value": p_value,
        "oracle_headroom": headroom,
        "oracle_headroom_at_least_minimum": (headroom is not None and headroom >= C.MINIMUM_ORACLE_HEADROOM),
        "primary_gate": primary,
        "primary_gate_pass": all(primary.values()),
        "robust_gate": robust,
        "robust_gate_pass": all(robust.values()),
        "activation": False,
    }


def demo() -> None:
    """Runnable self-check: the analysis must call a true null a null."""
    null_scores = [HistoryScore(index, "K1", 0.5) for index in range(32)]
    null_scores += [HistoryScore(index, "S2", 0.5) for index in range(32)]
    null = decisive_analysis(null_scores, candidate="K1", comparator="S2")
    assert null["effect"] == 0.0, null
    assert not null["primary_gate_pass"], null

    # A real effect of 0.08 with low noise must clear the primary gate.
    real_scores = []
    for index in range(32):
        wobble = ((index * 37) % 11 - 5) / 500.0
        real_scores.append(HistoryScore(index, "K1", 0.58 + wobble))
        real_scores.append(HistoryScore(index, "S2", 0.50 + wobble))
    real = decisive_analysis(real_scores, candidate="K1", comparator="S2")
    assert abs(real["effect"] - 0.08) < 1e-9, real
    assert real["primary_gate_pass"], real
    assert real["confidence_lower"] > 0.0, real

    # A borderline effect whose interval reaches below the SESOI must clear the
    # primary gate but fail the stricter one.
    borderline = []
    for index in range(32):
        wobble = ((index * 53) % 17 - 8) / 100.0
        borderline.append(HistoryScore(index, "K1", 0.5575 + wobble))
        borderline.append(HistoryScore(index, "S2", 0.50))
    edge = decisive_analysis(borderline, candidate="K1", comparator="S2")
    assert edge["effect"] >= C.SESOI, edge
    assert not edge["robust_gate"]["lower_bound_at_least_sesoi"], edge

    corrected = holm({"P1": 0.001, "P2": 0.04, "P10": 0.20})
    assert corrected["P1"]["rejected"], corrected
    assert not corrected["P10"]["rejected"], corrected
    print("genesis statistics self-check passed")


if __name__ == "__main__":
    demo()
