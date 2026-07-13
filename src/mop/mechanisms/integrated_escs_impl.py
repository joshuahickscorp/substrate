"""Deterministic seeded organizations for the integrated ESCS advantage frontier.

This module is the runnable reference for the integrated ESCS workstream. It builds the integrated
organization and the four matched baseline organizations, each of which turns a task (an EscsRegime)
into one labeled (quality, compute) point on the shared matched-cost frontier, plus the ordered
ablation ladder whose rungs each carry a marginal quality gain and a marginal compute.

The model is deliberately small and transparent. Quality is a base plus, for every mechanism the
organization employs, a per-mechanism gain scaled by how strongly the task demands that mechanism:

- the integrated organization employs all four mechanisms;
- baseline family i ablates exactly mechanism i, so its quality is the integrated quality minus the
  gain for the one mechanism it drops.

Every organization is held to one matched full-system cost, so Pareto dominance at matched cost
reduces to strictly higher quality. When a mechanism is not demanded, the baseline that drops it ties
the integrated organization and the matching ablation rung earns no gain, so it fails to justify its
marginal compute. Claim scope: deterministic programmatic mechanics only; no capability claim.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .integrated_escs_bed import EscsRegime
from .integrated_escs_scaffold import (
    MECHANISM_LADDER,
    REQUIRED_BASELINES,
    CostVector,
    FrontierPoint,
    IntegratedEscsRefusal,
)

# Base quality and the per-mechanism quality gain. Kept small so total quality stays inside [0, 1].
Q0 = 0.50
GAIN = 0.05

# The marginal compute (FLOPs) each mechanism rung adds on the ablation ladder, and the minimum
# efficiency (quality gain per FLOP) a rung must reach to justify its marginal compute.
MECHANISM_FLOPS: tuple[int, ...] = (1024, 1024, 1024, 1024)
MIN_EFFICIENCY = 1e-9

# The single matched full-system cost every organization point is held to (five ordered cost axes).
MATCHED_COST = CostVector(params=1000, flops=2000, memory_bytes=4000, wall_ticks=100, energy_units=50)


def _quality(demands: tuple[float, ...], skip: int | None) -> float:
    """Return base quality plus the gain for every mechanism employed. ``skip`` drops one mechanism."""

    total = 0.0
    for index, demand in enumerate(demands):
        if index == skip:
            continue
        total += GAIN * demand
    return round(Q0 + total, 9)


def integrated_point(regime: EscsRegime) -> FrontierPoint:
    """Return the integrated organization point: it employs every mechanism at matched cost."""

    return FrontierPoint(
        label=f"integrated.escs.{regime.name}",
        quality=_quality(regime.demands, None),
        cost=MATCHED_COST,
    )


def baseline_point(index: int, regime: EscsRegime) -> FrontierPoint:
    """Return the matched baseline point for family ``index``, which ablates mechanism ``index``."""

    if not 0 <= index < len(REQUIRED_BASELINES):
        raise IntegratedEscsRefusal("baseline index is out of range")
    family = REQUIRED_BASELINES[index]
    return FrontierPoint(
        label=f"baseline.{family}.{regime.name}",
        quality=_quality(regime.demands, index),
        cost=MATCHED_COST,
    )


def baseline_points(regime: EscsRegime) -> tuple[FrontierPoint, ...]:
    """Return every matched baseline point in the canonical baseline family order."""

    return tuple(baseline_point(index, regime) for index in range(len(REQUIRED_BASELINES)))


@dataclass(frozen=True, slots=True)
class LadderRung:
    """One rung of the ablation ladder: a mechanism, its marginal quality gain, and marginal FLOPs.

    A rung is justified only when it earns strictly positive gain and its efficiency reaches the
    declared minimum. Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    mechanism: str
    marginal_quality_gain: float
    marginal_flops: int
    min_efficiency: float

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISM_LADDER:
            raise IntegratedEscsRefusal(f"unsupported ladder mechanism {self.mechanism!r}")
        if self.marginal_flops <= 0:
            raise IntegratedEscsRefusal("rung marginal FLOPs must be positive (non-vacuous)")
        if self.min_efficiency <= 0.0:
            raise IntegratedEscsRefusal("rung minimum efficiency must be positive")
        if self.marginal_quality_gain < 0.0:
            raise IntegratedEscsRefusal("rung marginal quality gain must be nonnegative")

    @property
    def efficiency(self) -> float:
        return self.marginal_quality_gain / float(self.marginal_flops)

    @property
    def justified(self) -> bool:
        return self.marginal_quality_gain > 0.0 and self.efficiency >= self.min_efficiency

    def payload(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "marginal_quality_gain": self.marginal_quality_gain,
            "marginal_flops": self.marginal_flops,
            "min_efficiency": self.min_efficiency,
            "efficiency": self.efficiency,
            "justified": self.justified,
        }


def ablation_ladder(regime: EscsRegime) -> tuple[LadderRung, ...]:
    """Return the ordered ablation ladder; each rung's gain scales with how strongly it is demanded."""

    rungs: list[LadderRung] = []
    for index, mechanism in enumerate(MECHANISM_LADDER):
        gain = round(GAIN * regime.demands[index], 9)
        rungs.append(
            LadderRung(
                mechanism=mechanism,
                marginal_quality_gain=gain,
                marginal_flops=MECHANISM_FLOPS[index],
                min_efficiency=MIN_EFFICIENCY,
            )
        )
    return tuple(rungs)
