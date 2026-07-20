
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import (
    FIRST_ACTIVATION_STAGE,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    RunReceipt,
    mint_demonstration,
)
from ..substrate.events import canonical_sha256
from .event_formation_impl import (
    EVENT_FORMER_ID,
    FAVORABLE_REGIME,
    NULL_REGIME,
    UTILITY_FLOOR,
    RegimeMeasurement,
    measure_regime,
)
from .event_formation_scaffold import (
    REQUIRED_CONTROLS,
    UNTRAINED_CONTROLS,
    EventFormationRefusal,
    EventUtilityVerdict,
    OracleHeadroomContract,
    default_matched_budget,
)

RUN_SCHEMA = "mop-event-formation-run/v1"
REQUIREMENT_ID = "s3.event_formation"


def _reference_oracle() -> OracleHeadroomContract:

    return OracleHeadroomContract(
        oracle_id="oracle.event-formation",
        oracle_utility=1.0,
        baseline_utility=0.3,
        full_charged_compute=1.0,
        oracle_charged_compute=0.3,
        headroom_min=0.2,
        measured=True,
    )


def _verdict_for(measurement: RegimeMeasurement) -> EventUtilityVerdict:

    return EventUtilityVerdict(
        candidate_id="candidate.relational-temporal-former",
        oracle=_reference_oracle(),
        utility_candidate=measurement.utility[EVENT_FORMER_ID],
        utility_floor=UTILITY_FLOOR,
        utility_by_control={control: measurement.utility[control] for control in UNTRAINED_CONTROLS},
        charged_compute_candidate=measurement.charged_compute[EVENT_FORMER_ID],
        charged_compute_by_control={
            control: measurement.charged_compute[control] for control in UNTRAINED_CONTROLS
        },
        budget=default_matched_budget(),
    )


@dataclass(frozen=True, slots=True)
class EventFormationRunResults:

    seed: int
    null: RegimeMeasurement
    favorable: RegimeMeasurement

    def measurement(self, regime: str) -> RegimeMeasurement:
        if regime == NULL_REGIME:
            return self.null
        if regime == FAVORABLE_REGIME:
            return self.favorable
        raise EventFormationRefusal(f"unknown regime {regime!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": RUN_SCHEMA,
            "seed": self.seed,
            "null": self.null.payload(),
            "favorable": self.favorable.payload(),
        }


@dataclass(frozen=True, slots=True)
class EventFormationRunner:

    mechanism_id: str = "event_formation"

    def run(self, bed: Bed, seed: int) -> EventFormationRunResults:

        null_ticks = bed.null_regime(seed)
        favorable_ticks = bed.favorable_regime(seed)
        return EventFormationRunResults(
            seed=seed,
            null=measure_regime(null_ticks, NULL_REGIME, seed),
            favorable=measure_regime(favorable_ticks, FAVORABLE_REGIME, seed),
        )

    def _controls_cleared(self, measurement: RegimeMeasurement) -> tuple[str, ...]:
        former_utility = measurement.utility[EVENT_FORMER_ID]
        return tuple(
            control for control in REQUIRED_CONTROLS if former_utility > measurement.utility[control]
        )

    def _margins(
        self,
        results: EventFormationRunResults,
        regime_focus: str,
        favorable_ok: bool,
        null_holds: bool,
    ) -> dict[str, Any]:
        favorable = results.favorable
        former_utility = favorable.utility[EVENT_FORMER_ID]
        former_compute = favorable.charged_compute[EVENT_FORMER_ID]
        return {
            "schema": RUN_SCHEMA,
            "regime_focus": regime_focus,
            "utility_floor": UTILITY_FLOOR,
            "former_utility_favorable": former_utility,
            "former_utility_null": results.null.utility[EVENT_FORMER_ID],
            "utility_margin_vs_floor": former_utility - UTILITY_FLOOR,
            "utility_margin_vs_untrained": {
                control: former_utility - favorable.utility[control] for control in UNTRAINED_CONTROLS
            },
            "compute_saved_vs_untrained": {
                control: favorable.charged_compute[control] - former_compute
                for control in UNTRAINED_CONTROLS
            },
            "favorable_claims_useful": favorable_ok,
            "null_holds": null_holds,
            "controls": list(REQUIRED_CONTROLS),
            "untrained_controls": list(UNTRAINED_CONTROLS),
        }

    def _mint(
        self,
        results: EventFormationRunResults,
        regime_focus: str,
        verdict: str,
        controls_cleared: tuple[str, ...],
        favorable_ok: bool,
        null_holds: bool,
    ) -> RunReceipt:
        detail = self._margins(results, regime_focus, favorable_ok, null_holds)
        evidence_digest = canonical_sha256(
            {"receipt_focus": regime_focus, "verdict": verdict, "run": results.payload()}
        )
        return mint_demonstration(
            mechanism_id=self.mechanism_id,
            stage=FIRST_ACTIVATION_STAGE,
            requirement_id=REQUIREMENT_ID,
            controls_cleared=controls_cleared,
            evidence_digest=evidence_digest,
            verdict=verdict,
            detail=detail,
        )

    def mint(self, results: EventFormationRunResults) -> RunReceipt:

        favorable_ok = _verdict_for(results.favorable).claims_useful_event()
        null_holds = not _verdict_for(results.null).claims_useful_event()
        mechanics_ok = favorable_ok and null_holds
        verdict = VERDICT_MECHANICS_OK if mechanics_ok else VERDICT_NULL
        controls_cleared = self._controls_cleared(results.favorable) if mechanics_ok else ()
        return self._mint(results, "both", verdict, controls_cleared, favorable_ok, null_holds)

    def mint_regime(self, results: EventFormationRunResults, regime: str) -> RunReceipt:

        measurement = results.measurement(regime)
        claims_useful = _verdict_for(measurement).claims_useful_event()
        verdict = VERDICT_MECHANICS_OK if claims_useful else VERDICT_NULL
        controls_cleared = self._controls_cleared(measurement) if claims_useful else ()
        favorable_ok = _verdict_for(results.favorable).claims_useful_event()
        null_holds = not _verdict_for(results.null).claims_useful_event()
        return self._mint(results, regime, verdict, controls_cleared, favorable_ok, null_holds)
