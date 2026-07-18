"""Reusable neutral scientific framework for campaign science nodes.

This is the shared, task-agnostic infrastructure the mandate asks for: deterministic seeding, control
construction, independent experimental units, matched-budget lifecycle accounting, SESOI handling, an
exact sign-flip interface, and canonical sealing. It sits beside the sealed STARSS23 beds and does not edit
them. Science node runners build a new question mostly from a declarative spec plus their genuinely
task-specific estimator, labels, and scoring geometry, reusing everything here.

The decisive verifier logic is kept OUT of this producer-side framework: a verifier node re-derives results
independently rather than importing the same grading path.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def derive_seed(*parts: Any) -> int:
    """A deterministic 32-bit seed from any parts, so nodes are byte-reproducible without a wall clock."""

    payload = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def rng(*parts: Any) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts))


# ---------------------------------------------------------------------------
# Independent experimental units and the exact sign-flip (the beds' primary test).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnitEstimate:
    """One independent unit's paired delta (control minus candidate; positive favors candidate)."""

    unit_id: str
    delta: float


def exact_sign_flip_one_sided(deltas: Sequence[float], max_enum: int = 22) -> dict[str, Any]:
    """Exact one-sided upper-tail sign-flip p over independent-unit deltas. Enumerates 2**n sign flips.

    The statistic is sum_i s_i * delta_i with the observed all-plus assignment; p is the fraction of sign
    assignments with T(s) >= T_obs. Refuses above ``max_enum`` units rather than silently approximating.
    """

    values = [float(v) for v in deltas]
    n = len(values)
    if n == 0:
        raise ValueError("sign-flip needs at least one unit")
    if n > max_enum:
        raise ValueError(f"exact enumeration capped at {max_enum} units; got {n}")
    t_obs = math.fsum(values)
    count = 0
    for mask in range(1 << n):
        total = 0.0
        for i, v in enumerate(values):
            total += v if (mask >> i) & 1 else -v
        if total >= t_obs - 1e-12:
            count += 1
    p = count / (1 << n)
    n_fav = sum(1 for v in values if v > 0)
    return {
        "n_units": n,
        "permutations": 1 << n,
        "t_observed": round(t_obs, 12),
        "mean_delta": round(t_obs / n, 12),
        "n_units_favorable": n_fav,
        "one_sided_p": round(p, 12),
        "min_one_sided_p": round(1.0 / (1 << n), 12),
    }


# ---------------------------------------------------------------------------
# SESOI and stopping.
# ---------------------------------------------------------------------------


def clears_sesoi(mean_delta: float, sesoi: float) -> bool:
    """A tie or a below-SESOI effect is a null. Positive means the candidate is better by the metric sign."""

    return mean_delta >= sesoi and sesoi > 0.0


def verdict_from(mean_delta: float, one_sided_p: float, sesoi: float, alpha: float = 0.05) -> str:
    """The bed-neutral verdict: 'survives' iff point estimate favors candidate, clears SESOI, and p<=alpha."""

    if mean_delta > 0 and clears_sesoi(mean_delta, sesoi) and one_sided_p <= alpha:
        return "survives"
    return "null"


# ---------------------------------------------------------------------------
# Matched-budget lifecycle accounting.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LifecycleCost:
    """Full-lifecycle compute for one arm: training charged to the candidate only, inference to every arm."""

    train_flops: int
    inference_flops: int

    @property
    def total(self) -> int:
        return self.train_flops + self.inference_flops


def assert_matched_budget(costs: dict[str, LifecycleCost], ceiling: int) -> dict[str, Any]:
    """Return the per-arm totals and whether all arms are within a shared FLOP ceiling."""

    totals = {arm: c.total for arm, c in costs.items()}
    return {
        "per_arm_total_flops": totals,
        "ceiling": ceiling,
        "within_ceiling": all(t <= ceiling for t in totals.values()),
    }


# ---------------------------------------------------------------------------
# Honest artifact envelope every science node shares.
# ---------------------------------------------------------------------------


def honest_envelope(node_id: str, schema: str, coverage: dict[str, str]) -> dict[str, Any]:
    """The fixed honesty boundary carried by every science artifact: no activation, promotion, or
    confirmation from one run, and the coverage coordinates for the operator view."""

    return {
        "schema": schema,
        "node_id": node_id,
        "coverage": coverage,
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
        "claim_verb": "consistent with",
    }
