"""A MechanismRunner for niche_dispatch that mints a mechanics-only demonstration receipt.

The runner computes value-of-compute dispatch vs the four declared matched controls on both regimes
of a niche_dispatch bed and mints a single RunReceipt. It mints a mechanics-demonstration only, never
a scientific confirmation and never a cleared verdict. The verdict is mechanics-ok exactly when the
named prior null holds on the null regime (dispatch does not beat every control there) AND the
dispatcher strictly beats every matched control on the favorable regime. If any favorable control
ties or wins, the run fails closed to the null verdict. No capability or natural-data claim is made.

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
from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256
from .niche_dispatch_impl import RegimeData, control_accuracies, dispatch_accuracy
from .niche_dispatch_scaffold import DISPATCH_CONTROLS

MECHANISM_ID = "niche_dispatch"
STAGE = 3
REQUIREMENT_ID = "s3.niche_dispatch"


def _round(value: float) -> float:
    return round(value, 6)


@dataclass(frozen=True, slots=True)
class RegimeReport:
    """The scored outcome for one regime: dispatch accuracy, control scores, per-control margins."""

    regime: str
    dispatch_accuracy: float
    control_accuracies: tuple[tuple[str, float], ...]
    margins: tuple[tuple[str, float], ...]
    beats: tuple[tuple[str, bool], ...]

    @property
    def beats_every_control(self) -> bool:
        return all(beaten for _, beaten in self.beats)

    @property
    def controls_beaten(self) -> tuple[str, ...]:
        """The controls strictly beaten by dispatch, in canonical control-family order."""

        beaten = {name for name, ok in self.beats if ok}
        return tuple(name for name in DISPATCH_CONTROLS if name in beaten)

    def payload(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "dispatch_accuracy": self.dispatch_accuracy,
            "control_accuracies": [[name, score] for name, score in self.control_accuracies],
            "margins": [[name, margin] for name, margin in self.margins],
            "beats": [[name, beaten] for name, beaten in self.beats],
        }


def _evaluate_regime(data: RegimeData, seed: int) -> RegimeReport:
    dispatch = dispatch_accuracy(data)
    controls = control_accuracies(data, seed)
    ordered = tuple((name, controls[name]) for name in DISPATCH_CONTROLS)
    margins = tuple((name, _round(dispatch - score)) for name, score in ordered)
    beats = tuple((name, dispatch > score) for name, score in ordered)
    return RegimeReport(
        regime=data.regime,
        dispatch_accuracy=dispatch,
        control_accuracies=ordered,
        margins=margins,
        beats=beats,
    )


@dataclass(frozen=True, slots=True)
class NicheDispatchRunResult:
    """Everything one run measured: config, matched budget, and both regime reports."""

    mechanism_id: str
    seed: int
    n_perspectives: int
    contexts_per_cell: int
    matched: MatchedBudget
    null_report: RegimeReport
    favorable_report: RegimeReport

    @property
    def null_holds(self) -> bool:
        """The named prior null holds when dispatch does not beat every control on the null regime."""

        return not self.null_report.beats_every_control

    @property
    def favorable_win(self) -> bool:
        return self.favorable_report.beats_every_control

    def evidence_payload(self) -> dict[str, Any]:
        return {
            "schema": "mop-niche-dispatch-run/v1",
            "mechanism_id": self.mechanism_id,
            "seed": self.seed,
            "n_perspectives": self.n_perspectives,
            "contexts_per_cell": self.contexts_per_cell,
            "matched": self.matched.payload(),
            "null_report": self.null_report.payload(),
            "favorable_report": self.favorable_report.payload(),
        }

    def evidence_digest(self) -> str:
        return canonical_sha256(self.evidence_payload())


@dataclass(frozen=True, slots=True)
class NicheDispatchRunner:
    """Runs the niche_dispatch bed and mints a mechanics-only demonstration receipt.

    Claim scope: deterministic programmatic mechanics only; no capability claim. This runner never
    mints a scientific confirmation and never emits a cleared verdict.
    """

    mechanism_id: str = MECHANISM_ID

    def run(self, bed: Bed, seed: int) -> NicheDispatchRunResult:
        """Score dispatch vs every matched control on both regimes of the bed."""

        null_data = bed.null_regime(seed)
        favorable_data = bed.favorable_regime(seed)
        matched = bed.matched_cost()
        return NicheDispatchRunResult(
            mechanism_id=self.mechanism_id,
            seed=seed,
            n_perspectives=null_data.n_perspectives,
            contexts_per_cell=null_data.n_contexts // null_data.n_perspectives,
            matched=matched,
            null_report=_evaluate_regime(null_data, seed),
            favorable_report=_evaluate_regime(favorable_data, seed),
        )

    def mint(self, results: NicheDispatchRunResult) -> RunReceipt:
        """Mint a mechanics-demonstration receipt; mechanics-ok only on a genuine favorable win."""

        mechanics_ok = results.null_holds and results.favorable_win
        verdict = VERDICT_MECHANICS_OK if mechanics_ok else VERDICT_NULL
        controls_cleared = results.favorable_report.controls_beaten
        detail: dict[str, Any] = {
            "null_holds": results.null_holds,
            "favorable_win": results.favorable_win,
            "null_margins": [[name, margin] for name, margin in results.null_report.margins],
            "favorable_margins": [
                [name, margin] for name, margin in results.favorable_report.margins
            ],
            "null_dispatch_accuracy": results.null_report.dispatch_accuracy,
            "favorable_dispatch_accuracy": results.favorable_report.dispatch_accuracy,
        }
        return mint_demonstration(
            mechanism_id=self.mechanism_id,
            stage=STAGE,
            requirement_id=REQUIREMENT_ID,
            controls_cleared=controls_cleared,
            evidence_digest=results.evidence_digest(),
            verdict=verdict,
            detail=detail,
        )
