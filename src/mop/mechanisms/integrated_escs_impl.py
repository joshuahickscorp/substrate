
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

Q0 = 0.50
GAIN = 0.05

MECHANISM_FLOPS: tuple[int, ...] = (1024, 1024, 1024, 1024)
MIN_EFFICIENCY = 1e-9

MATCHED_COST = CostVector(params=1000, flops=2000, memory_bytes=4000, wall_ticks=100, energy_units=50)


def _quality(demands: tuple[float, ...], skip: int | None) -> float:

    total = 0.0
    for index, demand in enumerate(demands):
        if index == skip:
            continue
        total += GAIN * demand
    return round(Q0 + total, 9)


def integrated_point(regime: EscsRegime) -> FrontierPoint:

    return FrontierPoint(
        label=f"integrated.escs.{regime.name}",
        quality=_quality(regime.demands, None),
        cost=MATCHED_COST,
    )


def baseline_point(index: int, regime: EscsRegime) -> FrontierPoint:

    if not 0 <= index < len(REQUIRED_BASELINES):
        raise IntegratedEscsRefusal("baseline index is out of range")
    family = REQUIRED_BASELINES[index]
    return FrontierPoint(
        label=f"baseline.{family}.{regime.name}",
        quality=_quality(regime.demands, index),
        cost=MATCHED_COST,
    )


def baseline_points(regime: EscsRegime) -> tuple[FrontierPoint, ...]:

    return tuple(baseline_point(index, regime) for index in range(len(REQUIRED_BASELINES)))


@dataclass(frozen=True, slots=True)
class LadderRung:

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
