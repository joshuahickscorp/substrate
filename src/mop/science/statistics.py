from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_ALPHA = 0.05
BOUNDED_CLAIM_VERB = "consistent with"
FORBIDDEN_CLAIM_VERBS = (
    "demonstrates",
    "shows",
    "proves",
    "establishes",
    "significant",
    "confirms",
)
_MAX_EXACT_SEEDS = 20
_TIE_EPS = 1e-9


class StatsRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    mean_delta: float
    one_sided_p: float
    permutations: int
    one_sided_significant: bool
    two_sided_alpha_reachable: bool


def _finite(values: Sequence[float], label: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise StatsRefusal(f"{label} must be a sequence of numbers")
    result = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StatsRefusal(f"{label}[{index}] must be a real number")
        if not math.isfinite(number := float(value)):
            raise StatsRefusal(f"{label}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def exact_sign_flip(deltas: Sequence[float], alpha: float = DEFAULT_ALPHA) -> SignFlipResult:
    values = _finite(deltas, "deltas")
    n = len(values)
    if not n:
        raise StatsRefusal("sign-flip permutation needs at least one paired delta")
    if n > _MAX_EXACT_SEEDS:
        raise StatsRefusal(f"exact enumeration is capped at n={_MAX_EXACT_SEEDS}; got n={n}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < float(alpha) < 1:
        raise StatsRefusal("alpha must be a probability strictly between 0 and 1")
    alpha = float(alpha)
    permutations = 2**n
    observed = math.fsum(values)
    count = sum(
        math.fsum(values[index] if mask >> index & 1 else -values[index] for index in range(n))
        >= observed - _TIE_EPS
        for mask in range(permutations)
    )
    one_sided_p = count / permutations
    return SignFlipResult(
        observed / n,
        one_sided_p,
        permutations,
        one_sided_p <= alpha,
        2 / permutations <= alpha,
    )


def sesoi_exceeded(observed_effect: float, sesoi: float) -> bool:
    effect = _finite([observed_effect], "observed_effect")[0]
    if isinstance(sesoi, bool) or not isinstance(sesoi, (int, float)):
        raise StatsRefusal("sesoi_f1 must be a real number")
    threshold = float(sesoi)
    if not math.isfinite(threshold) or threshold <= 0:
        raise StatsRefusal("sesoi_f1 must be a positive finite number")
    return effect >= threshold


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
    payload: dict[str, Any] = {
        "deltas": [float(value) for value in deltas],
        "t_obs": float(result.mean_delta),
        "one_sided_p": float(result.one_sided_p),
        "n_permutations": int(result.permutations),
        "two_sided_005_reachable": bool(result.two_sided_alpha_reachable),
        "sesoi_mae": float(sesoi),
        "mean_delta_exceeds_sesoi": bool(exceeds_sesoi),
        "claim_verb": BOUNDED_CLAIM_VERB,
        "experimental_unit": "clip",
        "frame_or_clip_bootstrap_allowed": False,
        "sesoi_provisional": False,
        "prereg_canonical_sha256": prereg_digest,
    }
    additions = {
        "metric": metric,
        "delta_definition": delta_definition,
        "mean_delta_control_minus_candidate": float(result.mean_delta),
        "mean_delta_candidate_minus_control": float(mean_candidate_minus_control),
        **dict(extra or {}),
    }
    if payload.keys() & additions.keys():
        raise StatsRefusal("extra sign-flip fields overlap the shared projection")
    payload.update(additions)
    return payload


__all__ = [
    "BOUNDED_CLAIM_VERB",
    "FORBIDDEN_CLAIM_VERBS",
    "count_sign_flip_payload",
    "exact_sign_flip",
    "sesoi_exceeded",
]
