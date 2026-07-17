"""MechanismRunner for the calibrated uncertainty bed: measure both axes, mint a demonstration.

This module wires the bed and the policies into the ladder run-receipt contract. It runs the
mechanism and every declared control on a chosen regime, measures selective risk reduction and
decision utility for each, and mints a mechanics-demonstration RunReceipt whose verdict is
fail-closed on the decoupled confidence null:

- On the NULL regime the verdict is always ``null``: the confidence signal is decoupled noise held
  below the answer bar, so calibrated thresholding collapses onto the frozen-uniform abstainer and
  the measurement confirms the mechanism cannot beat every control on both axes there.
- On the FAVORABLE regime the verdict is ``mechanics-ok`` ONLY when the mechanism strictly improves
  BOTH selective risk reduction AND decision utility over EVERY control. Improving only one axis is
  exactly the decoupled null, so it mints ``null``, never a joint win.

The receipt is a mechanics-demonstration and can never be a scientific confirmation: it carries no
matched-cost attestation, opens no stage gate, and is_confirmation is always False. Claim scope:
deterministic programmatic mechanics only; no capability or natural-data claim.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

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
from .calibrated_uncertainty_bed import REGIME_FAVORABLE, REGIME_NULL, REGIMES
from .calibrated_uncertainty_impl import MECHANISM_ARM, run_all
from .calibrated_uncertainty_scaffold import (
    DUAL_AXES,
    PRIOR_NULL,
    REQUIRED_CONTROLS,
    DualMetricReading,
)

MECHANISM_ID = "calibrated_uncertainty"
REQUIREMENT_ID = "s3.calibrated_uncertainty"
RUNNER_SCHEMA = "mop-calibrated-uncertainty-run/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"


class RunnerRefusal(ValueError):
    """Raised when a run result is malformed or a comparison is attempted without controls."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """The measured readings for one regime: the mechanism vs every control on both axes.

    ``risk_margin`` and ``utility_margin`` are the mechanism's WORST margin over the control family
    on each axis, so a positive margin means the mechanism beat every control on that axis.
    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

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
    def risk_margin(self) -> float:
        return min(
            self.mechanism_reading.selective_risk_reduction - reading.selective_risk_reduction
            for _, reading in self.control_readings
        )

    @property
    def utility_margin(self) -> float:
        return min(
            self.mechanism_reading.decision_utility - reading.decision_utility
            for _, reading in self.control_readings
        )

    @property
    def both_axes_win(self) -> bool:
        """Strictly beats every control on selective risk reduction AND decision utility. The bar."""

        return self.risk_margin > 0.0 and self.utility_margin > 0.0

    @property
    def controls_beaten_both(self) -> tuple[str, ...]:
        """The controls the mechanism strictly beat on both axes, in declared order."""

        beaten = {
            name
            for name, reading in self.control_readings
            if self.mechanism_reading.selective_risk_reduction > reading.selective_risk_reduction
            and self.mechanism_reading.decision_utility > reading.decision_utility
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
            "risk_margin": self.risk_margin,
            "utility_margin": self.utility_margin,
            "both_axes_win": self.both_axes_win,
            "prior_null": PRIOR_NULL,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class CalibratedUncertaintyRunner:
    """Runs the calibrated uncertainty mechanism and controls, and mints a demonstration receipt.

    Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
    """

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
        """Measure selective risk reduction and decision utility for every arm on a regime."""

        if bed.mechanism_id != MECHANISM_ID:
            raise RunnerRefusal("runner and bed mechanism_id mismatch")
        if regime == REGIME_NULL:
            batch = bed.null_regime(seed)
        elif regime == REGIME_FAVORABLE:
            batch = bed.favorable_regime(seed)
        else:
            raise RunnerRefusal(f"unknown regime {regime!r}")
        readings = run_all(batch)
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
        """Mint a mechanics-demonstration receipt. mechanics-ok only for a favorable both-axes win."""

        if results.regime == REGIME_FAVORABLE and results.both_axes_win:
            verdict = VERDICT_MECHANICS_OK
        else:
            verdict = VERDICT_NULL
        detail: dict[str, Any] = {
            "regime": results.regime,
            "seed": results.seed,
            "axes": list(DUAL_AXES),
            "controls": list(REQUIRED_CONTROLS),
            "selective_risk_reduction_margin": results.risk_margin,
            "decision_utility_margin": results.utility_margin,
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
