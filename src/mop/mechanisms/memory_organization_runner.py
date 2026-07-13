"""MechanismRunner for the memory-organization epoch. Scores future decisions and mints a receipt.

A run scores the future-decision value of the organized memory and each of the four controls on both
the null regime and the favorable regime, and it checks the episodic-harm guard by comparing the
organized memory against the memory-free no-memory arm. The receipt it mints is always a mechanics
demonstration, never a scientific confirmation, so it can never open a stage gate.

The receipt verdict is null unless three things all hold on the favorable regime: the organized
memory beats every control on future-decision value, it does not measurably harm relative to
no-memory, and the named prior null still holds on the null regime. Only then is the verdict
mechanics-ok, and even then it stays a mechanics demonstration. A prediction-only improvement that
raises recall accuracy without moving any future decision does not clear the bar, because the win is
measured on decisions and not on prediction.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.

House style: no em or en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import (
    FIRST_ACTIVATION_STAGE,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    RunReceipt,
    mint_demonstration,
)
from ..substrate.events import canonical_sha256
from .memory_organization_bed import (
    ARMS,
    MECHANISM_ID,
    ORGANIZED_ARM,
    DecisionStream,
    action_of,
)
from .memory_organization_impl import recall
from .memory_organization_scaffold import (
    FUTURE_DECISION_CONTROLS,
    MemoryOrganizationRefusal,
    default_episodic_harm_guard,
)

REQUIREMENT_ID = "s3.memory_organization"
NO_MEMORY_ARM = "no-memory"


@dataclass(frozen=True, slots=True)
class RegimeScore:
    """Per-arm future-decision and prediction counts for one regime, sealed by the stream digest."""

    regime: str
    total: int
    decision_correct: dict[str, int]
    prediction_correct: dict[str, int]
    stream_digest: str

    def decision_value(self, arm: str) -> float:
        return self.decision_correct[arm] / self.total

    def prediction_value(self, arm: str) -> float:
        return self.prediction_correct[arm] / self.total


@dataclass(frozen=True, slots=True)
class MemoryRunResult:
    """The full result of one run: both regimes scored plus the episodic-harm verdict."""

    seed: int
    null: RegimeScore
    favorable: RegimeScore
    harmed: bool
    harm_value: float


def _score_regime(stream: DecisionStream) -> RegimeScore:
    decision_correct: dict[str, int] = {arm: 0 for arm in ARMS}
    prediction_correct: dict[str, int] = {arm: 0 for arm in ARMS}
    for query in stream.queries:
        for arm in ARMS:
            value = recall(arm, stream.ops, query.key, query.default_feature)
            if action_of(value) == query.optimal_action:
                decision_correct[arm] += 1
            if value == query.true_value:
                prediction_correct[arm] += 1
    return RegimeScore(
        regime=stream.regime,
        total=len(stream.queries),
        decision_correct=decision_correct,
        prediction_correct=prediction_correct,
        stream_digest=stream.digest(),
    )


@dataclass(frozen=True, slots=True)
class MemoryOrganizationRunner:
    """Runs the memory-organization bed and mints a mechanics demonstration receipt.

    Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
    """

    mechanism_id: str = MECHANISM_ID

    def run(self, bed: Any, seed: int) -> MemoryRunResult:
        """Score both regimes and check the episodic-harm guard on the favorable regime."""

        null = _score_regime(bed.null_regime(seed))
        favorable = _score_regime(bed.favorable_regime(seed))
        guard = default_episodic_harm_guard()
        no_memory_value = favorable.decision_value(NO_MEMORY_ARM)
        organized_value = favorable.decision_value(ORGANIZED_ARM)
        try:
            harm_value = guard.assert_no_episodic_harm(
                no_memory_value=no_memory_value, episodic_value=organized_value
            )
            harmed = False
        except MemoryOrganizationRefusal:
            harmed = True
            harm_value = no_memory_value - organized_value
        return MemoryRunResult(
            seed=seed, null=null, favorable=favorable, harmed=harmed, harm_value=harm_value
        )

    def mint(self, results: MemoryRunResult) -> RunReceipt:
        """Mint a demonstration receipt: mechanics-ok only on a clean, non-harming favorable win."""

        favorable = results.favorable
        null = results.null
        organized_value = favorable.decision_value(ORGANIZED_ARM)

        favorable_margins = {
            control: organized_value - favorable.decision_value(control)
            for control in FUTURE_DECISION_CONTROLS
        }
        null_margins = {
            control: null.decision_value(ORGANIZED_ARM) - null.decision_value(control)
            for control in FUTURE_DECISION_CONTROLS
        }
        prediction_margins = {
            control: favorable.prediction_value(ORGANIZED_ARM) - favorable.prediction_value(control)
            for control in FUTURE_DECISION_CONTROLS
        }

        favorable_win = all(margin > 0.0 for margin in favorable_margins.values())
        null_holds = not all(margin > 0.0 for margin in null_margins.values())
        mechanics_ok = favorable_win and null_holds and not results.harmed
        verdict = VERDICT_MECHANICS_OK if mechanics_ok else VERDICT_NULL

        cleared = tuple(
            control for control in FUTURE_DECISION_CONTROLS if favorable_margins[control] > 0.0
        )
        controls_cleared = FUTURE_DECISION_CONTROLS if mechanics_ok else cleared

        evidence = {
            "mechanism_id": MECHANISM_ID,
            "requirement_id": REQUIREMENT_ID,
            "verdict": verdict,
            "null": {
                "total": null.total,
                "decision_correct": null.decision_correct,
                "stream_digest": null.stream_digest,
            },
            "favorable": {
                "total": favorable.total,
                "decision_correct": favorable.decision_correct,
                "prediction_correct": favorable.prediction_correct,
                "stream_digest": favorable.stream_digest,
            },
            "harmed": results.harmed,
        }
        evidence_digest = canonical_sha256(evidence)

        detail: dict[str, Any] = {
            "favorable_margins": favorable_margins,
            "null_margins": null_margins,
            "favorable_prediction_margins": prediction_margins,
            "favorable_win": favorable_win,
            "null_holds": null_holds,
            "harmed": results.harmed,
            "harm_value": results.harm_value,
            "organized_favorable_decision_value": organized_value,
            "regime_digests": {
                "null": null.stream_digest,
                "favorable": favorable.stream_digest,
            },
        }

        return mint_demonstration(
            mechanism_id=MECHANISM_ID,
            stage=FIRST_ACTIVATION_STAGE,
            requirement_id=REQUIREMENT_ID,
            controls_cleared=controls_cleared,
            evidence_digest=evidence_digest,
            verdict=verdict,
            detail=detail,
        )
