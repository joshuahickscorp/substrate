from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from ..ladder.ladder_contracts import (
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    RunReceipt,
    mint_demonstration,
)
from ..ladder.stage_ladder import FIRST_ACTIVATION_STAGE
from ..substrate.events import canonical_sha256

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"


@dataclass(frozen=True, slots=True)
class JointAxisSpec:
    mechanism_id: str
    requirement_id: str
    schema: str
    axes: tuple[str, str]
    controls: tuple[str, ...]
    prior_null: str
    mechanism_arm: str
    result_margin_keys: tuple[str, str]
    receipt_margin_keys: tuple[str, str]
    regimes: tuple[str, str] = ("null", "favorable")


@dataclass(frozen=True, slots=True)
class JointAxisResult:
    regime: str
    seed: int
    mechanism_reading: Any
    control_readings: tuple[tuple[str, Any], ...]
    schema: str
    claim_scope: str

    spec: ClassVar[JointAxisSpec]
    refusal: ClassVar[type[ValueError]]

    def __post_init__(self) -> None:
        if self.schema != self.spec.schema:
            raise self.refusal(f"unsupported run result schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("run result claim scope cannot be widened")
        if self.regime not in self.spec.regimes:
            raise self.refusal(f"unknown regime {self.regime!r}")
        if self.seed < 0:
            raise self.refusal("run result seed must be nonnegative")
        names = tuple(name for name, _ in self.control_readings)
        if not names:
            raise self.refusal("a run result needs at least one control to compare against")
        if len(set(names)) != len(names):
            raise self.refusal("control readings must be unique")
        for name in names:
            if name not in self.spec.controls:
                raise self.refusal(f"unsupported control {name!r}")

    def margin(self, axis: int) -> float:
        attribute = self.spec.axes[axis]
        candidate = getattr(self.mechanism_reading, attribute)
        return min(candidate - getattr(reading, attribute) for _, reading in self.control_readings)

    @property
    def both_axes_win(self) -> bool:
        return self.margin(0) > 0.0 and self.margin(1) > 0.0

    @property
    def controls_beaten_both(self) -> tuple[str, ...]:
        first, second = self.spec.axes
        beaten = {
            name
            for name, reading in self.control_readings
            if getattr(self.mechanism_reading, first) > getattr(reading, first)
            and getattr(self.mechanism_reading, second) > getattr(reading, second)
        }
        return tuple(control for control in self.spec.controls if control in beaten)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_id": self.spec.mechanism_id,
            "regime": self.regime,
            "seed": self.seed,
            "axes": list(self.spec.axes),
            "mechanism": self.mechanism_reading.payload(),
            "controls": [
                {"control": name, "reading": reading.payload()}
                for name, reading in self.control_readings
            ],
            self.spec.result_margin_keys[0]: self.margin(0),
            self.spec.result_margin_keys[1]: self.margin(1),
            "both_axes_win": self.both_axes_win,
            "prior_null": self.spec.prior_null,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class JointAxisRunner:
    mechanism_id: str
    schema: str
    claim_scope: str

    spec: ClassVar[JointAxisSpec]
    refusal: ClassVar[type[ValueError]]
    result_type: ClassVar[type[JointAxisResult]]
    run_all: ClassVar[Callable[[Any], Mapping[str, Any]]]

    def __post_init__(self) -> None:
        if self.mechanism_id != self.spec.mechanism_id:
            raise self.refusal("runner mechanism_id drift")
        if self.schema != self.spec.schema:
            raise self.refusal(f"unsupported runner schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise self.refusal("runner claim scope cannot be widened")

    def run(self, bed: Bed, seed: int, regime: str = "favorable") -> JointAxisResult:
        if bed.mechanism_id != self.spec.mechanism_id:
            raise self.refusal("runner and bed mechanism_id mismatch")
        if regime == self.spec.regimes[0]:
            inputs = bed.null_regime(seed)
        elif regime == self.spec.regimes[1]:
            inputs = bed.favorable_regime(seed)
        else:
            raise self.refusal(f"unknown regime {regime!r}")
        readings = type(self).run_all(inputs)
        control_readings = tuple(
            (control, readings[control])
            for control in bed.controls()
            if control in self.spec.controls
        )
        return self.result_type(
            regime=regime,
            seed=seed,
            mechanism_reading=readings[self.spec.mechanism_arm],
            control_readings=control_readings,
        )

    def mint(self, results: JointAxisResult) -> RunReceipt:
        verdict = (
            VERDICT_MECHANICS_OK
            if results.regime == self.spec.regimes[1] and results.both_axes_win
            else VERDICT_NULL
        )
        detail = {
            "regime": results.regime,
            "seed": results.seed,
            "axes": list(self.spec.axes),
            "controls": list(self.spec.controls),
            self.spec.receipt_margin_keys[0]: results.margin(0),
            self.spec.receipt_margin_keys[1]: results.margin(1),
            "both_axes_win": results.both_axes_win,
            "prior_null": self.spec.prior_null,
        }
        return mint_demonstration(
            mechanism_id=self.spec.mechanism_id,
            stage=FIRST_ACTIVATION_STAGE,
            requirement_id=self.spec.requirement_id,
            controls_cleared=results.controls_beaten_both,
            evidence_digest=results.digest(),
            verdict=verdict,
            detail=detail,
        )
