
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256
from .stage4_integration_bed import (
    BASELINE_ARMS,
    IntegrationPoint,
    RegimeSample,
    seeded_unit,
)

STAGE4_IMPL_SCHEMA = "mop-stage4-integration-impl/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

MIN_ABLATION_EFFICIENCY = 1e-9


class Stage4ImplRefusal(ValueError):
    pass




@dataclass(frozen=True, slots=True)
class AblationRung:

    mechanism_id: str
    marginal_quality_gain: float
    marginal_compute: int
    min_efficiency: float = MIN_ABLATION_EFFICIENCY
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4ImplRefusal("ablation rung claim scope cannot be widened")
        if not self.marginal_quality_gain > 0.0:
            raise Stage4ImplRefusal("ablation rung requires a strictly positive marginal quality gain")
        if (
            not isinstance(self.marginal_compute, int)
            or isinstance(self.marginal_compute, bool)
            or self.marginal_compute <= 0
        ):
            raise Stage4ImplRefusal("ablation rung marginal compute must be a positive integer")
        if not self.min_efficiency > 0.0:
            raise Stage4ImplRefusal("ablation rung minimum efficiency must be positive")

    @property
    def efficiency(self) -> float:
        return self.marginal_quality_gain / float(self.marginal_compute)

    def justified(self) -> bool:
        return self.efficiency >= self.min_efficiency

    def payload(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "marginal_quality_gain": self.marginal_quality_gain,
            "marginal_compute": self.marginal_compute,
            "min_efficiency": self.min_efficiency,
            "justified": self.justified(),
        }


@dataclass(frozen=True, slots=True)
class AblationLadder:

    rungs: tuple[AblationRung, ...]
    schema: str = STAGE4_IMPL_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE4_IMPL_SCHEMA:
            raise Stage4ImplRefusal(f"unsupported ablation ladder schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4ImplRefusal("ablation ladder claim scope cannot be widened")
        if not self.rungs:
            raise Stage4ImplRefusal("an ablation ladder needs at least one rung")
        ids = [rung.mechanism_id for rung in self.rungs]
        if len(set(ids)) != len(ids):
            raise Stage4ImplRefusal("ablation ladder mechanism ids must be unique")
        for rung in self.rungs:
            if not rung.justified():
                raise Stage4ImplRefusal(
                    "one or more ablation rungs do not justify their marginal compute at the declared minimum"
                )

    def total_marginal_gain(self) -> float:
        return sum((rung.marginal_quality_gain for rung in self.rungs), 0.0)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "rungs": [rung.payload() for rung in self.rungs],
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())




@dataclass(frozen=True, slots=True)
class Composition:

    regime: str
    seed: int
    joint: IntegrationPoint
    baselines: tuple[tuple[str, IntegrationPoint], ...]
    ablation: AblationLadder
    schema: str = STAGE4_IMPL_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE4_IMPL_SCHEMA:
            raise Stage4ImplRefusal(f"unsupported composition schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4ImplRefusal("composition claim scope cannot be widened")
        arms = tuple(arm for arm, _ in self.baselines)
        if arms != BASELINE_ARMS:
            raise Stage4ImplRefusal("composition baseline arms are incomplete or out of canonical order")

    def dominated_baselines(self) -> tuple[str, ...]:

        return tuple(arm for arm, point in self.baselines if self.joint.dominates(point))

    def dominates_all(self) -> bool:

        return all(self.joint.dominates(point) for _, point in self.baselines)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "regime": self.regime,
            "seed": self.seed,
            "joint": self.joint.payload(),
            "baselines": [{"arm": arm, "point": point.payload()} for arm, point in self.baselines],
            "ablation": self.ablation.payload(),
            "dominated_baselines": list(self.dominated_baselines()),
            "dominates_all": self.dominates_all(),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _build_baselines(sample: RegimeSample, best_quality: float) -> tuple[tuple[str, IntegrationPoint], ...]:

    compute = sample.matched_compute
    first = sample.mechanism_samples[0]
    points = {
        "single-mechanism": IntegrationPoint(
            label="baseline.single-mechanism", quality=first.quality, compute=compute
        ),
        "static-composition": IntegrationPoint(
            label="baseline.static-composition", quality=best_quality, compute=compute
        ),
        "best-single": IntegrationPoint(
            label="baseline.best-single", quality=best_quality, compute=compute
        ),
    }
    return tuple((arm, points[arm]) for arm in BASELINE_ARMS)


def _build_ablation(sample: RegimeSample, seed: int) -> AblationLadder:

    rungs = tuple(
        AblationRung(
            mechanism_id=s.mechanism_id,
            marginal_quality_gain=0.01 + 0.04 * seeded_unit(seed, f"rung.{s.mechanism_id}"),
            marginal_compute=s.marginal_compute,
        )
        for s in sample.mechanism_samples
    )
    return AblationLadder(rungs=rungs)


def integrate(sample: RegimeSample, seed: int) -> Composition:

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise Stage4ImplRefusal("seed must be a nonnegative integer")
    qualities = [s.quality for s in sample.mechanism_samples]
    best_quality = max(qualities)
    joint_quality = min(1.0, best_quality + sample.synergy_quality_bonus)
    joint = IntegrationPoint(
        label="joint.integrated",
        quality=joint_quality,
        compute=sample.joint_compute,
    )
    baselines = _build_baselines(sample, best_quality)
    ablation = _build_ablation(sample, seed)
    return Composition(
        regime=sample.regime,
        seed=seed,
        joint=joint,
        baselines=baselines,
        ablation=ablation,
    )
