"""The mechanism runner for the messaging_repair Stage 3 workstream (epoch messaging_repair).

The runner scores the mechanism against every declared control on both regimes at the matched action
budget, then mints a mechanics-demonstration RunReceipt. The receipt is honest by construction:

- On the null regime the mechanism must NOT beat every control (the no-message floor ties it), which
  is what keeps the named prior null intact.
- The receipt carries the "mechanics-ok" verdict ONLY when the mechanism strictly beats EVERY control
  on the favorable regime at matched cost AND the null still holds on the null regime. Otherwise the
  verdict is "null". A single leaking control that ties or beats the mechanism blocks mechanics-ok,
  fail-closed.

The receipt is always a mechanics-demonstration: it can never carry a confirmation or a cleared
verdict, and it can never open a stage gate. It records the requirement id "s3.messaging_repair", the
controls the mechanism cleared, the canonical evidence digest of the scored result, and the per-control
margins on both regimes.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
from .messaging_repair_bed import MECHANISM_ID, REQUIREMENT_ID
from .messaging_repair_impl import (
    CLAIM_SCOPE,
    CONTROL_POLICIES,
    MECHANISM_POLICY,
    MatchedCost,
    PolicyFn,
    Regime,
)

MESSAGING_REPAIR_RUNNER_SCHEMA = "mop-messaging-repair-runner/v1"


class MessagingRepairRunnerError(ValueError):
    """Raised when a declared control has no policy. The runner fails closed rather than skip it."""


@dataclass(frozen=True, slots=True)
class RegimeScore:
    """The mechanism-versus-controls scoring of one regime, with per-control matched-cost margins.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    kind: str
    regime_digest: str
    mechanism_improvement: int
    control_improvements: tuple[tuple[str, int], ...]
    margins: tuple[tuple[str, int], ...]
    mechanism_beats_all: bool

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "regime_digest": self.regime_digest,
            "mechanism_improvement": self.mechanism_improvement,
            "control_improvements": [[name, value] for name, value in self.control_improvements],
            "margins": [[name, value] for name, value in self.margins],
            "mechanism_beats_all": self.mechanism_beats_all,
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    """The full scored result of one run: the null regime score and the favorable regime score.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    schema: str
    mechanism_id: str
    seed: int
    null: RegimeScore
    favorable: RegimeScore
    claim_scope: str = CLAIM_SCOPE

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_id": self.mechanism_id,
            "seed": self.seed,
            "null": self.null.payload(),
            "favorable": self.favorable.payload(),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MessagingRepairRunner:
    """Scores the mechanism against the declared controls and mints a mechanics-demonstration receipt.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    mechanism_id: str = MECHANISM_ID
    mechanism_policy: PolicyFn = MECHANISM_POLICY
    control_policies: Mapping[str, PolicyFn] = field(default_factory=lambda: dict(CONTROL_POLICIES))

    def _score(self, regime: Regime, controls: tuple[str, ...]) -> RegimeScore:
        budget = MatchedCost(action_budget=regime.action_budget)
        mechanism = self.mechanism_policy(regime, budget).improvement
        control_improvements: list[tuple[str, int]] = []
        margins: list[tuple[str, int]] = []
        for name in controls:
            policy = self.control_policies.get(name)
            if policy is None:
                raise MessagingRepairRunnerError(f"declared control {name!r} has no policy")
            improvement = policy(regime, budget).improvement
            control_improvements.append((name, improvement))
            margins.append((name, mechanism - improvement))
        beats_all = bool(controls) and all(margin > 0 for _, margin in margins)
        return RegimeScore(
            kind=regime.kind,
            regime_digest=regime.digest(),
            mechanism_improvement=mechanism,
            control_improvements=tuple(control_improvements),
            margins=tuple(margins),
            mechanism_beats_all=beats_all,
        )

    def run(self, bed: Bed, seed: int) -> RunResult:
        """Score the mechanism against every declared control on both regimes at matched cost."""

        controls = tuple(bed.controls())
        null = self._score(bed.null_regime(seed), controls)
        favorable = self._score(bed.favorable_regime(seed), controls)
        return RunResult(
            schema=MESSAGING_REPAIR_RUNNER_SCHEMA,
            mechanism_id=self.mechanism_id,
            seed=seed,
            null=null,
            favorable=favorable,
        )

    def mint(self, results: RunResult) -> RunReceipt:
        """Mint the mechanics-demonstration receipt. Never a confirmation and never cleared."""

        favorable_win = results.favorable.mechanism_beats_all
        null_holds = not results.null.mechanism_beats_all
        cleared = tuple(name for name, margin in results.favorable.margins if margin > 0)
        mechanics_ok = favorable_win and null_holds
        verdict = VERDICT_MECHANICS_OK if mechanics_ok else VERDICT_NULL
        controls_cleared = cleared if mechanics_ok else ()
        detail: dict[str, Any] = {
            "schema": MESSAGING_REPAIR_RUNNER_SCHEMA,
            "seed": results.seed,
            "favorable_win": favorable_win,
            "null_holds": null_holds,
            "mechanism_improvement_favorable": results.favorable.mechanism_improvement,
            "mechanism_improvement_null": results.null.mechanism_improvement,
            "favorable_margins": {name: margin for name, margin in results.favorable.margins},
            "null_margins": {name: margin for name, margin in results.null.margins},
        }
        return mint_demonstration(
            mechanism_id=self.mechanism_id,
            stage=FIRST_ACTIVATION_STAGE,
            requirement_id=REQUIREMENT_ID,
            controls_cleared=controls_cleared,
            evidence_digest=results.digest(),
            verdict=verdict,
            detail=detail,
        )


def build_default_runner() -> MessagingRepairRunner:
    """Return the canonical messaging_repair runner with the default mechanism and control policies."""

    return MessagingRepairRunner()
