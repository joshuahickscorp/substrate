"""Shared producer-side paired statistics; independent verifiers reimplement decisive math separately."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mop.substrate.events import canonical_sha256

STATS_SCHEMA = "mop-starss23-escs-stats/v1"
PROVISIONAL_SESOI_F1 = 0.05
DEFAULT_ALPHA = 0.05
FORBIDDEN_CLAIM_VERBS = (
    "demonstrates", "shows", "proves", "establishes", "significant", "confirms",
)
BOUNDED_CLAIM_VERB = "consistent with"
DEFAULT_CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"
CLAIM_SCOPE = DEFAULT_CLAIM_SCOPE
_MAX_ENUMERATION_EXPONENT = 20
_TIE_EPS = 1e-9


class StatsRefusal(ValueError):
    """A statistic, SESOI, unit count, or exact-enumeration request is malformed."""


def _finite(values: Sequence[float], label: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise StatsRefusal(f"{label} must be a sequence of numbers")
    result = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StatsRefusal(f"{label}[{index}] must be a real number")
        number = float(value)
        if not math.isfinite(number):
            raise StatsRefusal(f"{label}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _sealed(value: float) -> float:
    return round(float(value), 12)


def paired_deltas(candidate: Sequence[float], control: Sequence[float]) -> tuple[float, ...]:
    """Return paired candidate-minus-control effects in the caller's fixed seed order."""

    candidate_values = _finite(candidate, "candidate_f1_by_seed")
    control_values = _finite(control, "control_f1_by_seed")
    if len(candidate_values) != len(control_values):
        raise StatsRefusal("candidate and control must cover the same paired seeds")
    if not candidate_values:
        raise StatsRefusal("paired deltas need at least one paired seed")
    return tuple(a - b for a, b in zip(candidate_values, control_values, strict=True))


def two_sided_alpha_reachable(n: int, alpha: float = DEFAULT_ALPHA) -> bool:
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise StatsRefusal("n must be a positive integer")
    return 2.0 / 2**n <= alpha


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    n: int
    permutations: int
    t_observed: float
    mean_delta: float
    count_ge_one_sided: int
    count_ge_two_sided: int
    one_sided_p: float
    two_sided_p: float
    min_one_sided_p: float
    two_sided_floor: float
    alpha: float
    one_sided_significant: bool
    two_sided_alpha_reachable: bool
    phipson_smyth_applied: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "n": self.n, "permutations": self.permutations, "t_observed": _sealed(self.t_observed),
            "mean_delta": _sealed(self.mean_delta), "count_ge_one_sided": self.count_ge_one_sided,
            "count_ge_two_sided": self.count_ge_two_sided, "one_sided_p": _sealed(self.one_sided_p),
            "two_sided_p": _sealed(self.two_sided_p), "min_one_sided_p": _sealed(self.min_one_sided_p),
            "two_sided_floor": _sealed(self.two_sided_floor), "alpha": _sealed(self.alpha),
            "one_sided_significant": self.one_sided_significant,
            "two_sided_alpha_reachable": self.two_sided_alpha_reachable,
            "phipson_smyth_applied": self.phipson_smyth_applied,
        }


def sign_flip_payload(
    result: SignFlipResult,
    deltas: Sequence[float],
    *,
    sesoi_key: str,
    sesoi: float,
    exceeds_sesoi: bool,
    claim_verb: str = BOUNDED_CLAIM_VERB,
    experimental_unit: str = "clip",
    provisional: bool | None = None,
    prereg_digest: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the common paired sign-flip audit block without changing the decisive statistic."""

    if not sesoi_key.startswith("sesoi_"):
        raise StatsRefusal("sesoi_key must name the artifact SESOI field")
    payload: dict[str, Any] = {
        "deltas": [float(value) for value in deltas],
        "t_obs": float(result.mean_delta),
        "one_sided_p": float(result.one_sided_p),
        "n_permutations": int(result.permutations),
        "two_sided_005_reachable": bool(result.two_sided_alpha_reachable),
        sesoi_key: float(sesoi),
        "mean_delta_exceeds_sesoi": bool(exceeds_sesoi),
        "claim_verb": claim_verb,
        "experimental_unit": experimental_unit,
        "frame_or_clip_bootstrap_allowed": False,
    }
    if provisional is not None:
        payload["sesoi_provisional"] = bool(provisional)
    if prereg_digest is not None:
        payload["prereg_canonical_sha256"] = prereg_digest
    additions = dict(extra or {})
    if payload.keys() & additions.keys():
        raise StatsRefusal("extra sign-flip fields overlap the shared projection")
    payload.update(additions)
    return payload


def count_sign_flip_payload(
    result: SignFlipResult,
    deltas: Sequence[float],
    *,
    sesoi: float,
    exceeds_sesoi: bool,
    mean_candidate_minus_control: float,
    prereg_digest: str,
    metric: str = "coasted-count-MAE",
    delta_definition: str = (
        "delta_i = MAE_rate_matched_random(i) - MAE_candidate(i); positive = candidate lower error"
    ),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the shared counting statistic with its lower-is-better delta convention."""

    additions = {
        "metric": metric,
        "delta_definition": delta_definition,
        "mean_delta_control_minus_candidate": float(result.mean_delta),
        "mean_delta_candidate_minus_control": float(mean_candidate_minus_control),
        **dict(extra or {}),
    }
    return sign_flip_payload(
        result,
        deltas,
        sesoi_key="sesoi_mae",
        sesoi=sesoi,
        exceeds_sesoi=exceeds_sesoi,
        provisional=False,
        prereg_digest=prereg_digest,
        extra=additions,
    )


def exact_sign_flip(deltas: Sequence[float], alpha: float = DEFAULT_ALPHA) -> SignFlipResult:
    """Enumerate every sign assignment and return exact one- and two-sided tails."""

    values = _finite(deltas, "deltas")
    n = len(values)
    if not n:
        raise StatsRefusal("sign-flip permutation needs at least one paired delta")
    if n > _MAX_ENUMERATION_EXPONENT:
        raise StatsRefusal(f"exact enumeration is capped at n={_MAX_ENUMERATION_EXPONENT}; got n={n}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < float(alpha) < 1:
        raise StatsRefusal("alpha must be a probability strictly between 0 and 1")
    alpha = float(alpha)
    permutations = 2**n
    observed = math.fsum(values)
    one = two = 0
    for mask in range(permutations):
        statistic = math.fsum(values[index] if mask >> index & 1 else -values[index] for index in range(n))
        one += statistic >= observed - _TIE_EPS
        two += abs(statistic) >= abs(observed) - _TIE_EPS
    one_p, two_p = one / permutations, two / permutations
    return SignFlipResult(
        n, permutations, observed, observed / n, one, two, one_p, two_p, 1 / permutations,
        2 / permutations, alpha, one_p <= alpha, 2 / permutations <= alpha, False,
    )


@dataclass(frozen=True, slots=True)
class SesoiCheck:
    sesoi_f1: float
    provisional: bool
    observed_effect: float
    exceeds_sesoi: bool

    def payload(self) -> dict[str, Any]:
        return {"sesoi_f1": _sealed(self.sesoi_f1), "provisional": self.provisional,
                "observed_effect": _sealed(self.observed_effect), "exceeds_sesoi": self.exceeds_sesoi}


def sesoi_check(
    observed_effect: float, sesoi_f1: float = PROVISIONAL_SESOI_F1, provisional: bool = True,
) -> SesoiCheck:
    effect = _finite([observed_effect], "observed_effect")[0]
    if isinstance(sesoi_f1, bool) or not isinstance(sesoi_f1, (int, float)):
        raise StatsRefusal("sesoi_f1 must be a real number")
    threshold = float(sesoi_f1)
    if not math.isfinite(threshold) or threshold <= 0:
        raise StatsRefusal("sesoi_f1 must be a positive finite number")
    return SesoiCheck(threshold, bool(provisional), effect, effect >= threshold)


@dataclass(frozen=True, slots=True)
class ClaimCeiling:
    experimental_unit: str
    n_experimental_units: int
    n_seeds: int
    claim_verb: str
    forbidden_verbs: tuple[str, ...]
    frame_or_clip_bootstrap_allowed: bool
    rationale: str

    def payload(self) -> dict[str, Any]:
        return {"experimental_unit": self.experimental_unit,
                "n_experimental_units": self.n_experimental_units, "n_seeds": self.n_seeds,
                "claim_verb": self.claim_verb, "forbidden_verbs": list(self.forbidden_verbs),
                "frame_or_clip_bootstrap_allowed": self.frame_or_clip_bootstrap_allowed,
                "rationale": self.rationale}


def claim_ceiling(n_experimental_units: int, n_seeds: int) -> ClaimCeiling:
    for name, value in (("n_experimental_units", n_experimental_units), ("n_seeds", n_seeds)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise StatsRefusal(f"{name} must be a positive integer")
    rationale = (
        "the clip is the experimental unit; frames are correlated sub-samples, so a frame or clip "
        "bootstrap is refused and the claim verb is bounded to 'consistent with', never 'demonstrates'"
    )
    return ClaimCeiling(
        "clip", n_experimental_units, n_seeds, BOUNDED_CLAIM_VERB, FORBIDDEN_CLAIM_VERBS, False, rationale,
    )


@dataclass(frozen=True, slots=True)
class PairedSeedStats:
    schema: str
    deltas: tuple[float, ...]
    sign_flip: SignFlipResult
    sesoi: SesoiCheck
    claim: ClaimCeiling
    meets_statistical_bar: bool
    promotable: bool
    scientific_promotion: bool
    independent_scientific_confirmation: bool
    claim_scope: str

    def payload(self) -> dict[str, Any]:
        return {"schema": self.schema, "deltas": [_sealed(value) for value in self.deltas],
                "sign_flip": self.sign_flip.payload(), "sesoi": self.sesoi.payload(),
                "claim": self.claim.payload(), "meets_statistical_bar": self.meets_statistical_bar,
                "promotable": self.promotable, "scientific_promotion": self.scientific_promotion,
                "independent_scientific_confirmation": self.independent_scientific_confirmation,
                "claim_scope": self.claim_scope}

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def analyze_paired_seeds(
    deltas: Sequence[float], *, n_experimental_units: int,
    sesoi_f1: float = PROVISIONAL_SESOI_F1, provisional_sesoi: bool = True,
    alpha: float = DEFAULT_ALPHA, claim_scope: str = DEFAULT_CLAIM_SCOPE,
) -> PairedSeedStats:
    values = _finite(deltas, "deltas")
    sign_flip = exact_sign_flip(values, alpha)
    sesoi = sesoi_check(sign_flip.mean_delta, sesoi_f1, provisional_sesoi)
    claim = claim_ceiling(n_experimental_units, len(values))
    passes = sesoi.exceeds_sesoi and sign_flip.one_sided_significant
    return PairedSeedStats(
        STATS_SCHEMA, values, sign_flip, sesoi, claim, passes, passes, False, False, claim_scope,
    )


__all__ = [
    "STATS_SCHEMA", "PROVISIONAL_SESOI_F1", "DEFAULT_ALPHA", "FORBIDDEN_CLAIM_VERBS",
    "BOUNDED_CLAIM_VERB", "StatsRefusal", "SignFlipResult", "SesoiCheck", "ClaimCeiling",
    "PairedSeedStats", "paired_deltas", "count_sign_flip_payload", "exact_sign_flip",
    "sign_flip_payload", "sesoi_check", "claim_ceiling",
    "two_sided_alpha_reachable", "analyze_paired_seeds",
    "CLAIM_SCOPE",
]
