
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    RunReceipt,
    mint_demonstration,
)
from ..ladder.stage_ladder import FIRST_ACTIVATION_STAGE
from ..substrate.events import canonical_sha256
from .stability_plasticity_r2_bed import REGIME_FAVORABLE, REGIME_NULL, REGIMES
from .stability_plasticity_r2_impl import MECHANISM_ARM, run_all
from .stability_plasticity_r2_scaffold import (
    DUAL_AXES,
    PRIOR_NULL,
    REQUIRED_CONTROLS,
    DualMetricReading,
)

MECHANISM_ID = "stability_plasticity_r2"
REQUIREMENT_ID = "s3.stability_plasticity_r2"
RUNNER_SCHEMA = "mop-stability-plasticity-r2-run/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"


class RunnerRefusal(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunResult:

    regime: str
    seed: int
    mechanism_reading: DualMetricReading
    control_readings: tuple[tuple[str, DualMetricReading], ...]
    schema: str = RUNNER_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != RUNNER_SCHEMA:
            raise RunnerRefusal(f"unsupported run result schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise RunnerRefusal("run result claim scope cannot be widened")
        if self.regime not in REGIMES:
            raise RunnerRefusal(f"unknown regime {self.regime!r}")
        if self.seed < 0:
            raise RunnerRefusal("run result seed must be nonnegative")
        names = tuple(name for name, _ in self.control_readings)
        if not names:
            raise RunnerRefusal("a run result needs at least one control to compare against")
        if len(set(names)) != len(names):
            raise RunnerRefusal("control readings must be unique")
        for name in names:
            if name not in REQUIRED_CONTROLS:
                raise RunnerRefusal(f"unsupported control {name!r}")

    @property
    def retention_margin(self) -> float:
        return min(
            self.mechanism_reading.retention - reading.retention
            for _, reading in self.control_readings
        )

    @property
    def plasticity_margin(self) -> float:
        return min(
            self.mechanism_reading.future_learnability - reading.future_learnability
            for _, reading in self.control_readings
        )

    @property
    def both_axes_win(self) -> bool:

        return self.retention_margin > 0.0 and self.plasticity_margin > 0.0

    @property
    def controls_beaten_both(self) -> tuple[str, ...]:

        beaten = {
            name
            for name, reading in self.control_readings
            if self.mechanism_reading.retention > reading.retention
            and self.mechanism_reading.future_learnability > reading.future_learnability
        }
        return tuple(control for control in REQUIRED_CONTROLS if control in beaten)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_id": MECHANISM_ID,
            "regime": self.regime,
            "seed": self.seed,
            "axes": list(DUAL_AXES),
            "mechanism": self.mechanism_reading.payload(),
            "controls": [
                {"control": name, "reading": reading.payload()}
                for name, reading in self.control_readings
            ],
            "retention_margin": self.retention_margin,
            "plasticity_margin": self.plasticity_margin,
            "both_axes_win": self.both_axes_win,
            "prior_null": PRIOR_NULL,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class StabilityPlasticityR2Runner:

    mechanism_id: str = MECHANISM_ID
    schema: str = RUNNER_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.mechanism_id != MECHANISM_ID:
            raise RunnerRefusal("runner mechanism_id drift")
        if self.schema != RUNNER_SCHEMA:
            raise RunnerRefusal(f"unsupported runner schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise RunnerRefusal("runner claim scope cannot be widened")

    def run(self, bed: Bed, seed: int, regime: str = REGIME_FAVORABLE) -> RunResult:

        if bed.mechanism_id != MECHANISM_ID:
            raise RunnerRefusal("runner and bed mechanism_id mismatch")
        if regime == REGIME_NULL:
            stream = bed.null_regime(seed)
        elif regime == REGIME_FAVORABLE:
            stream = bed.favorable_regime(seed)
        else:
            raise RunnerRefusal(f"unknown regime {regime!r}")
        readings = run_all(stream)
        control_readings = tuple(
            (control, readings[control]) for control in bed.controls() if control in REQUIRED_CONTROLS
        )
        return RunResult(
            regime=regime,
            seed=seed,
            mechanism_reading=readings[MECHANISM_ARM],
            control_readings=control_readings,
        )

    def mint(self, results: RunResult) -> RunReceipt:

        if results.regime == REGIME_FAVORABLE and results.both_axes_win:
            verdict = VERDICT_MECHANICS_OK
        else:
            verdict = VERDICT_NULL
        detail: dict[str, Any] = {
            "regime": results.regime,
            "seed": results.seed,
            "axes": list(DUAL_AXES),
            "controls": list(REQUIRED_CONTROLS),
            "retention_margin": results.retention_margin,
            "future_learnability_margin": results.plasticity_margin,
            "both_axes_win": results.both_axes_win,
            "prior_null": PRIOR_NULL,
        }
        return mint_demonstration(
            mechanism_id=MECHANISM_ID,
            stage=FIRST_ACTIVATION_STAGE,
            requirement_id=REQUIREMENT_ID,
            controls_cleared=results.controls_beaten_both,
            evidence_digest=results.digest(),
            verdict=verdict,
            detail=detail,
        )
