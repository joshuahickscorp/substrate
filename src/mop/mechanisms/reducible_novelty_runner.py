"""MechanismRunner for the reducible novelty bed: measure both axes, mint a demonstration.

This module wires the bed and the allocators into the ladder run-receipt contract. It runs the
mechanism and every declared control on a chosen regime, measures learning progress and allocation
efficiency for each, and mints a mechanics-demonstration RunReceipt whose verdict is fail-closed on
the irreducible noise trap:

- On the NULL regime the verdict is always ``null``: every source is irreducible noise by
  construction, so every arm scores zero on both axes and the mechanism cannot strictly beat
  uniform_allocation there.
- On the FAVORABLE regime the verdict is ``mechanics-ok`` ONLY when the mechanism strictly improves
  BOTH learning progress AND allocation efficiency over EVERY control. Improving only one axis is
  exactly the trap, so it mints ``null``, never a joint win.

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
from .reducible_novelty_bed import REGIME_FAVORABLE, REGIME_NULL, REGIMES
from .reducible_novelty_impl import MECHANISM_ARM, run_all
from .reducible_novelty_scaffold import DUAL_AXES, PRIOR_NULL, REQUIRED_CONTROLS, DualMetricReading

MECHANISM_ID = "reducible_novelty"
REQUIREMENT_ID = "s3.reducible_novelty"
RUNNER_SCHEMA = "mop-reducible-novelty-run/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"


class RunnerRefusal(ValueError):
    """Raised when a run result is malformed or a comparison is attempted without controls."""


@dataclass(frozen=True, slots=True)
class RunResult:
    """The measured readings for one regime: the mechanism vs every control on both axes.

    ``progress_margin`` and ``efficiency_margin`` are the mechanism's WORST margin over the control
    family on each axis, so a positive margin means the mechanism beat every control on that axis.
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
    def progress_margin(self) -> float:
        return min(
            self.mechanism_reading.learning_progress - reading.learning_progress
            for _, reading in self.control_readings
        )

    @property
    def efficiency_margin(self) -> float:
        return min(
            self.mechanism_reading.allocation_efficiency - reading.allocation_efficiency
            for _, reading in self.control_readings
        )

    @property
    def both_axes_win(self) -> bool:
        """Strictly beats every control on learning progress AND allocation efficiency. The joint bar."""

        return self.progress_margin > 0.0 and self.efficiency_margin > 0.0

    @property
    def controls_beaten_both(self) -> tuple[str, ...]:
        """The controls the mechanism strictly beat on both axes, in declared order."""

        beaten = {
            name
            for name, reading in self.control_readings
            if self.mechanism_reading.learning_progress > reading.learning_progress
            and self.mechanism_reading.allocation_efficiency > reading.allocation_efficiency
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
            "progress_margin": self.progress_margin,
            "efficiency_margin": self.efficiency_margin,
            "both_axes_win": self.both_axes_win,
            "prior_null": PRIOR_NULL,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ReducibleNoveltyRunner:
    """Runs the reducible novelty mechanism and controls, and mints a demonstration receipt.

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
        """Measure learning progress and allocation efficiency for every arm on a regime."""

        if bed.mechanism_id != MECHANISM_ID:
            raise RunnerRefusal("runner and bed mechanism_id mismatch")
        if regime == REGIME_NULL:
            panel = bed.null_regime(seed)
        elif regime == REGIME_FAVORABLE:
            panel = bed.favorable_regime(seed)
        else:
            raise RunnerRefusal(f"unknown regime {regime!r}")
        readings = run_all(panel)
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
            "learning_progress_margin": results.progress_margin,
            "allocation_efficiency_margin": results.efficiency_margin,
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
