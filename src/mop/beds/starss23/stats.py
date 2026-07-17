"""STARSS23 ESCS bed, component 7: honest small-n paired statistics.

This module implements the statistics ceiling from docs/ESCS_DEEP_RESEARCH.md (Q5b, Q5c) for the
STARSS23 event-formation bed. It carries no experiment and no capability claim. It takes the paired
per-seed F1 deltas the harness produces and answers three questions, each bounded by the pseudo
replication reality of a clip-limited bed:

1. Exact sign-flip permutation. With n paired seeds there are exactly 2**n sign assignments. The one
   sided p value is the fraction of assignments whose statistic reaches the observed statistic, so its
   floor is 1/2**n. At n=5 that floor is 1/32 = 0.03125 one-sided and 2/32 = 0.0625 two-sided, which
   means a two-sided alpha of 0.05 is unreachable. The Phipson and Smyth (b+1)/(m+1) correction is
   deliberately NOT applied: it is for randomly drawn Monte Carlo permutations, not a full 2**n
   enumeration, and mixing the conventions would be wrong.

2. A registered, compute-normalized SESOI check. The smallest effect size of interest on F1 is fixed
   before data collection (provisional SESOI_F1 = 0.05). A win below the SESOI is not promotable even
   when the sign-flip p value clears alpha, so a +0.3 percent blip cannot be promoted.

3. A claim ceiling keyed to the experimental unit. With few clips the clip is the experimental unit,
   frames are correlated sub-samples, and a frame or clip bootstrap is refused. The claim verb is
   bounded to "consistent with"; "demonstrates" and "significant" are forbidden.

Nothing here can promote a result. scientific_promotion stays False and the independent verifier is
the only thing that may ever set independent_scientific_confirmation. House style: no em dashes and no
en dashes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mop.substrate.events import canonical_sha256

from . import CLAIM_SCOPE

__all__ = [
    "STATS_SCHEMA",
    "PROVISIONAL_SESOI_F1",
    "DEFAULT_ALPHA",
    "FORBIDDEN_CLAIM_VERBS",
    "BOUNDED_CLAIM_VERB",
    "StatsRefusal",
    "SignFlipResult",
    "SesoiCheck",
    "ClaimCeiling",
    "PairedSeedStats",
    "paired_deltas",
    "exact_sign_flip",
    "sesoi_check",
    "claim_ceiling",
    "two_sided_alpha_reachable",
    "analyze_paired_seeds",
]

STATS_SCHEMA = "mop-starss23-escs-stats/v1"

# Preregistered smallest effect size of interest on the onset F1 scale. Provisional per the recipe
# (Q5a): a compute-normalized delta, fixed before data collection, below which a win is not promotable.
PROVISIONAL_SESOI_F1 = 0.05

# The alpha the discrete permutation floor is checked against. Two-sided 0.05 is unreachable at n=5.
DEFAULT_ALPHA = 0.05

# The bed is clip-limited and synthetic, so its claims are pseudoreplication-bounded. These verbs are
# never permitted; the claim verb is bounded to "consistent with".
FORBIDDEN_CLAIM_VERBS: tuple[str, ...] = (
    "demonstrates",
    "shows",
    "proves",
    "establishes",
    "significant",
    "confirms",
)
BOUNDED_CLAIM_VERB = "consistent with"

# Enumerating 2**n sign vectors is only sane for a small paired-seed count. The bed uses five seeds.
_MAX_ENUMERATION_EXPONENT = 20

# Absolute tolerance for tie counting when comparing floating-point statistics. The observed all
# positive-sign configuration is summed in the same index order as the observed statistic, so it is
# exactly equal; this tolerance only guards genuine ties between distinct sign assignments. It is far
# below the smallest meaningful gap between distinct sums of F1-scale deltas.
_TIE_EPS = 1e-9


class StatsRefusal(ValueError):
    """Raised when a statistics input is malformed or a boundary invariant would be violated."""


def _require_finite_floats(values: Sequence[float], label: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise StatsRefusal(f"{label} must be a sequence of numbers")
    out: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StatsRefusal(f"{label}[{index}] must be a real number")
        number = float(value)
        if not math.isfinite(number):
            raise StatsRefusal(f"{label}[{index}] must be finite")
        out.append(number)
    return tuple(out)


def _seal_float(value: float) -> float:
    """Round a statistic to a fixed precision so the sealed digest is byte-reproducible."""

    return round(float(value), 12)


def paired_deltas(
    candidate_f1_by_seed: Sequence[float], control_f1_by_seed: Sequence[float]
) -> tuple[float, ...]:
    """Return the paired per-seed deltas Delta_i = F1_candidate(i) - F1_control(i).

    Both inputs must be the candidate and control F1 on the same paired seeds in the same seed order.
    """

    candidate = _require_finite_floats(candidate_f1_by_seed, "candidate_f1_by_seed")
    control = _require_finite_floats(control_f1_by_seed, "control_f1_by_seed")
    if len(candidate) != len(control):
        raise StatsRefusal("candidate and control must cover the same paired seeds")
    if not candidate:
        raise StatsRefusal("paired deltas need at least one paired seed")
    return tuple(c - k for c, k in zip(candidate, control, strict=True))


def two_sided_alpha_reachable(n: int, alpha: float = DEFAULT_ALPHA) -> bool:
    """Return whether the discrete two-sided floor 2/2**n can reach alpha at n paired seeds."""

    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise StatsRefusal("n must be a positive integer")
    return (2.0 / float(2**n)) <= alpha


@dataclass(frozen=True, slots=True)
class SignFlipResult:
    """The exact sign-flip permutation result over all 2**n sign assignments of the paired deltas."""

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
            "n": self.n,
            "permutations": self.permutations,
            "t_observed": _seal_float(self.t_observed),
            "mean_delta": _seal_float(self.mean_delta),
            "count_ge_one_sided": self.count_ge_one_sided,
            "count_ge_two_sided": self.count_ge_two_sided,
            "one_sided_p": _seal_float(self.one_sided_p),
            "two_sided_p": _seal_float(self.two_sided_p),
            "min_one_sided_p": _seal_float(self.min_one_sided_p),
            "two_sided_floor": _seal_float(self.two_sided_floor),
            "alpha": _seal_float(self.alpha),
            "one_sided_significant": self.one_sided_significant,
            "two_sided_alpha_reachable": self.two_sided_alpha_reachable,
            "phipson_smyth_applied": self.phipson_smyth_applied,
        }


def exact_sign_flip(deltas: Sequence[float], alpha: float = DEFAULT_ALPHA) -> SignFlipResult:
    """Enumerate all 2**n sign assignments of the paired deltas and return the exact permutation test.

    The statistic is T(s) = sum_i s_i * delta_i with s_i in {+1, -1}. The observed statistic uses the
    observed signs, T_obs = sum_i delta_i. The one-sided (upper-tail) p value is the fraction of the
    2**n assignments with T(s) >= T_obs; because the observed assignment always satisfies that, the p
    value is never zero and its floor is 1/2**n. The two-sided p value uses |T(s)| >= |T_obs| and its
    floor is 2/2**n. No Phipson and Smyth correction is applied: this is a full enumeration, not a
    Monte Carlo draw.
    """

    values = _require_finite_floats(deltas, "deltas")
    n = len(values)
    if n == 0:
        raise StatsRefusal("sign-flip permutation needs at least one paired delta")
    if n > _MAX_ENUMERATION_EXPONENT:
        raise StatsRefusal(
            f"exact enumeration is capped at n={_MAX_ENUMERATION_EXPONENT}; got n={n}"
        )
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0.0 < float(alpha) < 1.0:
        raise StatsRefusal("alpha must be a probability strictly between 0 and 1")
    alpha = float(alpha)

    permutations = 2**n
    t_observed = math.fsum(values)
    abs_observed = abs(t_observed)

    count_ge_one = 0
    count_ge_two = 0
    for mask in range(permutations):
        terms = [values[i] if (mask >> i) & 1 else -values[i] for i in range(n)]
        statistic = math.fsum(terms)
        if statistic >= t_observed - _TIE_EPS:
            count_ge_one += 1
        if abs(statistic) >= abs_observed - _TIE_EPS:
            count_ge_two += 1

    one_sided_p = count_ge_one / permutations
    two_sided_p = count_ge_two / permutations
    min_one_sided_p = 1.0 / permutations
    two_sided_floor = 2.0 / permutations
    return SignFlipResult(
        n=n,
        permutations=permutations,
        t_observed=t_observed,
        mean_delta=t_observed / n,
        count_ge_one_sided=count_ge_one,
        count_ge_two_sided=count_ge_two,
        one_sided_p=one_sided_p,
        two_sided_p=two_sided_p,
        min_one_sided_p=min_one_sided_p,
        two_sided_floor=two_sided_floor,
        alpha=alpha,
        one_sided_significant=one_sided_p <= alpha,
        two_sided_alpha_reachable=two_sided_floor <= alpha,
        phipson_smyth_applied=False,
    )


@dataclass(frozen=True, slots=True)
class SesoiCheck:
    """A preregistered, compute-normalized smallest-effect-of-interest check on the F1 scale."""

    sesoi_f1: float
    provisional: bool
    observed_effect: float
    exceeds_sesoi: bool

    def payload(self) -> dict[str, Any]:
        return {
            "sesoi_f1": _seal_float(self.sesoi_f1),
            "provisional": self.provisional,
            "observed_effect": _seal_float(self.observed_effect),
            "exceeds_sesoi": self.exceeds_sesoi,
        }


def sesoi_check(
    observed_effect: float,
    sesoi_f1: float = PROVISIONAL_SESOI_F1,
    provisional: bool = True,
) -> SesoiCheck:
    """Compare an observed mean F1 delta against the registered SESOI. Meeting it is necessary."""

    effect = _require_finite_floats([observed_effect], "observed_effect")[0]
    if not isinstance(sesoi_f1, (int, float)) or isinstance(sesoi_f1, bool):
        raise StatsRefusal("sesoi_f1 must be a real number")
    threshold = float(sesoi_f1)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise StatsRefusal("sesoi_f1 must be a positive finite number")
    return SesoiCheck(
        sesoi_f1=threshold,
        provisional=bool(provisional),
        observed_effect=effect,
        exceeds_sesoi=effect >= threshold,
    )


@dataclass(frozen=True, slots=True)
class ClaimCeiling:
    """The pseudoreplication-bounded claim ceiling keyed to the number of experimental units."""

    experimental_unit: str
    n_experimental_units: int
    n_seeds: int
    claim_verb: str
    forbidden_verbs: tuple[str, ...]
    frame_or_clip_bootstrap_allowed: bool
    rationale: str

    def payload(self) -> dict[str, Any]:
        return {
            "experimental_unit": self.experimental_unit,
            "n_experimental_units": self.n_experimental_units,
            "n_seeds": self.n_seeds,
            "claim_verb": self.claim_verb,
            "forbidden_verbs": list(self.forbidden_verbs),
            "frame_or_clip_bootstrap_allowed": self.frame_or_clip_bootstrap_allowed,
            "rationale": self.rationale,
        }


def claim_ceiling(n_experimental_units: int, n_seeds: int) -> ClaimCeiling:
    """Return the claim ceiling. On this clip-limited bed the verb is bounded to "consistent with"."""

    for name, value in (("n_experimental_units", n_experimental_units), ("n_seeds", n_seeds)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise StatsRefusal(f"{name} must be a positive integer")
    rationale = (
        "the clip is the experimental unit; frames are correlated sub-samples, so a frame or clip "
        "bootstrap is refused and the claim verb is bounded to 'consistent with', never 'demonstrates'"
    )
    return ClaimCeiling(
        experimental_unit="clip",
        n_experimental_units=n_experimental_units,
        n_seeds=n_seeds,
        claim_verb=BOUNDED_CLAIM_VERB,
        forbidden_verbs=FORBIDDEN_CLAIM_VERBS,
        frame_or_clip_bootstrap_allowed=False,
        rationale=rationale,
    )


@dataclass(frozen=True, slots=True)
class PairedSeedStats:
    """The assembled honest paired-seed statistics for one candidate-versus-control comparison."""

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
        return {
            "schema": self.schema,
            "deltas": [_seal_float(value) for value in self.deltas],
            "sign_flip": self.sign_flip.payload(),
            "sesoi": self.sesoi.payload(),
            "claim": self.claim.payload(),
            "meets_statistical_bar": self.meets_statistical_bar,
            "promotable": self.promotable,
            "scientific_promotion": self.scientific_promotion,
            "independent_scientific_confirmation": self.independent_scientific_confirmation,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def analyze_paired_seeds(
    deltas: Sequence[float],
    *,
    n_experimental_units: int,
    sesoi_f1: float = PROVISIONAL_SESOI_F1,
    provisional_sesoi: bool = True,
    alpha: float = DEFAULT_ALPHA,
) -> PairedSeedStats:
    """Run the full honest paired-seed analysis: sign-flip, SESOI, and the claim ceiling.

    ``promotable`` requires both the registered SESOI to be exceeded and the one-sided sign-flip test
    to clear alpha. It is a statistical-bar gate only: ``scientific_promotion`` stays False because a
    synthetic, clip-limited bed can never promote, and ``independent_scientific_confirmation`` can only
    ever be set by the separately authored verifier, never here.
    """

    values = _require_finite_floats(deltas, "deltas")
    n = len(values)
    sign_flip = exact_sign_flip(values, alpha=alpha)
    sesoi = sesoi_check(sign_flip.mean_delta, sesoi_f1=sesoi_f1, provisional=provisional_sesoi)
    claim = claim_ceiling(n_experimental_units=n_experimental_units, n_seeds=n)
    meets_statistical_bar = sesoi.exceeds_sesoi and sign_flip.one_sided_significant
    return PairedSeedStats(
        schema=STATS_SCHEMA,
        deltas=values,
        sign_flip=sign_flip,
        sesoi=sesoi,
        claim=claim,
        meets_statistical_bar=meets_statistical_bar,
        promotable=meets_statistical_bar,
        scientific_promotion=False,
        independent_scientific_confirmation=False,
        claim_scope=CLAIM_SCOPE,
    )
