
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256

STAGE4_BED_SCHEMA = "mop-stage4-integration-bed/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

MECHANISM_ID = "stage4_integration"

BASELINE_ARMS: tuple[str, ...] = ("single-mechanism", "static-composition", "best-single")

REGIME_NULL = "null"
REGIME_FAVORABLE = "favorable"
ALLOWED_REGIMES: frozenset[str] = frozenset({REGIME_NULL, REGIME_FAVORABLE})

DEFAULT_MECHANISM_COUNT = 3
MIN_MECHANISM_COUNT = 2

MATCHED_COMPUTE = 4096
FAVORABLE_SYNERGY_BONUS = 0.15
NULL_EXTRA_COMPUTE = MATCHED_COMPUTE // 2

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")


class Stage4BedRefusal(ValueError):
    pass


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise Stage4BedRefusal(f"{label} must use stable lowercase characters")


def _require_finite_unit(value: float, label: str) -> None:
    if value != value or value in (float("inf"), float("-inf")):
        raise Stage4BedRefusal(f"{label} must be a finite number")
    if not 0.0 <= value <= 1.0:
        raise Stage4BedRefusal(f"{label} must lie in [0.0, 1.0]")


def seeded_unit(seed: int, salt: str) -> float:

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise Stage4BedRefusal("seed must be a nonnegative integer")
    digest = canonical_sha256({"seed": seed, "salt": salt})
    return int(digest[:8], 16) / float(0xFFFFFFFF)


@dataclass(frozen=True, slots=True)
class IntegrationPoint:

    label: str
    quality: float
    compute: int
    claim_scope: str = CLAIM_SCOPE
    schema: str = STAGE4_BED_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STAGE4_BED_SCHEMA:
            raise Stage4BedRefusal(f"unsupported integration point schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4BedRefusal("integration point claim scope cannot be widened")
        _require_id(self.label, "IntegrationPoint.label")
        _require_finite_unit(self.quality, "IntegrationPoint.quality")
        if not isinstance(self.compute, int) or isinstance(self.compute, bool) or self.compute <= 0:
            raise Stage4BedRefusal("IntegrationPoint.compute must be a positive integer")

    def dominates(self, other: IntegrationPoint) -> bool:

        quality_no_worse = self.quality >= other.quality
        compute_no_worse = self.compute <= other.compute
        if not (quality_no_worse and compute_no_worse):
            return False
        return self.quality > other.quality or self.compute < other.compute

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "label": self.label,
            "quality": self.quality,
            "compute": self.compute,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MechanismSample:

    mechanism_id: str
    quality: float
    marginal_compute: int
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        _require_id(self.mechanism_id, "MechanismSample.mechanism_id")
        _require_finite_unit(self.quality, "MechanismSample.quality")
        if (
            not isinstance(self.marginal_compute, int)
            or isinstance(self.marginal_compute, bool)
            or self.marginal_compute <= 0
        ):
            raise Stage4BedRefusal("MechanismSample.marginal_compute must be a positive integer")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4BedRefusal("mechanism sample claim scope cannot be widened")

    def point(self, compute: int) -> IntegrationPoint:

        return IntegrationPoint(label=f"single.{self.mechanism_id}", quality=self.quality, compute=compute)

    def payload(self) -> dict[str, Any]:
        return {
            "mechanism_id": self.mechanism_id,
            "quality": self.quality,
            "marginal_compute": self.marginal_compute,
            "claim_scope": self.claim_scope,
        }


@dataclass(frozen=True, slots=True)
class RegimeSample:

    regime: str
    mechanism_samples: tuple[MechanismSample, ...]
    matched_compute: int
    joint_compute: int
    synergy_quality_bonus: float
    seed: int
    schema: str = STAGE4_BED_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != STAGE4_BED_SCHEMA:
            raise Stage4BedRefusal(f"unsupported regime sample schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4BedRefusal("regime sample claim scope cannot be widened")
        if self.regime not in ALLOWED_REGIMES:
            raise Stage4BedRefusal(f"unsupported regime {self.regime!r}")
        if len(self.mechanism_samples) < MIN_MECHANISM_COUNT:
            raise Stage4BedRefusal(f"a regime needs at least {MIN_MECHANISM_COUNT} mechanism samples")
        ids = [sample.mechanism_id for sample in self.mechanism_samples]
        if len(set(ids)) != len(ids):
            raise Stage4BedRefusal("regime mechanism samples must reference distinct mechanisms")
        for name, value in (("matched_compute", self.matched_compute), ("joint_compute", self.joint_compute)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise Stage4BedRefusal(f"regime {name} must be a positive integer")
        if self.synergy_quality_bonus != self.synergy_quality_bonus or self.synergy_quality_bonus < 0.0:
            raise Stage4BedRefusal("regime synergy quality bonus must be a nonnegative finite number")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise Stage4BedRefusal("regime seed must be a nonnegative integer")

    def mechanism_ids(self) -> tuple[str, ...]:
        return tuple(sample.mechanism_id for sample in self.mechanism_samples)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "regime": self.regime,
            "mechanism_samples": [sample.payload() for sample in self.mechanism_samples],
            "matched_compute": self.matched_compute,
            "joint_compute": self.joint_compute,
            "synergy_quality_bonus": self.synergy_quality_bonus,
            "seed": self.seed,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class Stage4IntegrationBed:

    mechanism_id: str = MECHANISM_ID
    mechanism_count: int = DEFAULT_MECHANISM_COUNT
    matched_compute: int = MATCHED_COMPUTE
    schema: str = field(default=STAGE4_BED_SCHEMA)
    claim_scope: str = field(default=CLAIM_SCOPE)

    def __post_init__(self) -> None:
        if self.schema != STAGE4_BED_SCHEMA:
            raise Stage4BedRefusal(f"unsupported bed schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise Stage4BedRefusal("bed claim scope cannot be widened")
        _require_id(self.mechanism_id, "Stage4IntegrationBed.mechanism_id")
        if self.mechanism_count < MIN_MECHANISM_COUNT:
            raise Stage4BedRefusal(f"the bed needs at least {MIN_MECHANISM_COUNT} mechanisms")
        if (
            not isinstance(self.matched_compute, int)
            or isinstance(self.matched_compute, bool)
            or self.matched_compute <= 0
        ):
            raise Stage4BedRefusal("bed matched compute must be a positive integer")

    def controls(self) -> tuple[str, ...]:

        return BASELINE_ARMS

    def matched_cost(self) -> MatchedBudget:

        return MatchedBudget(
            params=1_048_576,
            flops=self.matched_compute,
            wall_ns=1_000_000,
            seeds=8,
        )

    def _mechanism_samples(self, seed: int) -> tuple[MechanismSample, ...]:

        samples: list[MechanismSample] = []
        for index in range(self.mechanism_count):
            mechanism_id = f"{self.mechanism_id}.m{index}"
            quality = 0.40 + 0.28 * seeded_unit(seed, f"{mechanism_id}.quality")
            marginal_compute = 128 + int(round(128 * seeded_unit(seed, f"{mechanism_id}.compute")))
            samples.append(
                MechanismSample(
                    mechanism_id=mechanism_id,
                    quality=quality,
                    marginal_compute=marginal_compute,
                )
            )
        return tuple(samples)

    def null_regime(self, seed: int) -> RegimeSample:

        return RegimeSample(
            regime=REGIME_NULL,
            mechanism_samples=self._mechanism_samples(seed),
            matched_compute=self.matched_compute,
            joint_compute=self.matched_compute + NULL_EXTRA_COMPUTE,
            synergy_quality_bonus=0.0,
            seed=seed,
        )

    def favorable_regime(self, seed: int) -> RegimeSample:

        return RegimeSample(
            regime=REGIME_FAVORABLE,
            mechanism_samples=self._mechanism_samples(seed),
            matched_compute=self.matched_compute,
            joint_compute=self.matched_compute,
            synergy_quality_bonus=FAVORABLE_SYNERGY_BONUS,
            seed=seed,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "mechanism_id": self.mechanism_id,
            "mechanism_count": self.mechanism_count,
            "matched_compute": self.matched_compute,
            "controls": list(self.controls()),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def build_default_bed() -> Stage4IntegrationBed:

    return Stage4IntegrationBed()
