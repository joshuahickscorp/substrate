"""MechanismRunner for the integrated ESCS advantage frontier (epoch integrated_escs).

The runner computes, for both regimes and the ablation ladder, the (quality, compute) frontier of the
integrated organization and every matched baseline, then mints a mechanics-demonstration RunReceipt.
It never mints a scientific confirmation and can never carry the ``cleared`` verdict.

The question this epoch guards: does an integrated ESCS beat simpler organizations on the
quality-vs-compute Pareto frontier under matched baselines. The named prior null, which holds on the
null regime, is that there is no integrated advantage: the integrated system fails to dominate at
matched cost. The demonstration verdict is ``mechanics-ok`` ONLY when the integrated organization
Pareto-dominates every matched baseline on the favorable regime and every added mechanism justifies
its marginal compute, while the null regime still holds null. Otherwise the verdict is ``null``.

Claim scope: deterministic programmatic mechanics only; no capability claim.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import (
    FIRST_ACTIVATION_STAGE,
    VERDICT_MECHANICS_OK,
    VERDICT_NULL,
    Bed,
    MatchedBudget,
    RunReceipt,
    mint_demonstration,
)
from ..substrate.events import canonical_sha256
from .integrated_escs_bed import EscsRegime
from .integrated_escs_impl import (
    LadderRung,
    ablation_ladder,
    baseline_points,
    integrated_point,
)
from .integrated_escs_scaffold import REQUIRED_BASELINES, FrontierPoint

REQUIREMENT_ID = "s3.integrated_escs"


@dataclass(frozen=True, slots=True)
class RegimeReport:
    """The frontier and ablation result for one regime: which baselines the integrated point dominates.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    regime_name: str
    seed: int
    integrated: FrontierPoint
    baselines: tuple[FrontierPoint, ...]
    dominated: tuple[str, ...]
    rungs: tuple[LadderRung, ...]

    def dominates_all(self) -> bool:
        return len(self.dominated) == len(self.baselines)

    def ablation_all_justified(self) -> bool:
        return all(rung.justified for rung in self.rungs)

    def earned(self) -> bool:
        """Mechanics are earned on this regime only when dominance and marginal justification both hold."""

        return self.dominates_all() and self.ablation_all_justified()

    def payload(self) -> dict[str, Any]:
        return {
            "regime_name": self.regime_name,
            "seed": self.seed,
            "integrated": self.integrated.payload(),
            "baselines": [point.payload() for point in self.baselines],
            "dominated": list(self.dominated),
            "dominates_all": self.dominates_all(),
            "ablation": [rung.payload() for rung in self.rungs],
            "ablation_all_justified": self.ablation_all_justified(),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def evaluate_regime(regime: EscsRegime, seed: int) -> RegimeReport:
    """Compute the frontier and ablation ladder for one regime under one seed."""

    integrated = integrated_point(regime)
    baselines = baseline_points(regime)
    dominated = tuple(
        REQUIRED_BASELINES[index]
        for index, point in enumerate(baselines)
        if integrated.dominates(point)
    )
    return RegimeReport(
        regime_name=regime.name,
        seed=seed,
        integrated=integrated,
        baselines=baselines,
        dominated=dominated,
        rungs=ablation_ladder(regime),
    )


@dataclass(frozen=True, slots=True)
class RunResult:
    """The full run: the frontier and ablation for the null and favorable regimes at one matched budget.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    seed: int
    null_report: RegimeReport
    favorable_report: RegimeReport
    matched: MatchedBudget

    def payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "null": self.null_report.payload(),
            "favorable": self.favorable_report.payload(),
            "matched": self.matched.payload(),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class IntegratedEscsRunner:
    """Runs the integrated ESCS organizations against a bed and mints a mechanics demonstration.

    Claim scope: deterministic programmatic mechanics only; no capability claim.
    """

    mechanism_id: str = "integrated_escs"
    stage: int = FIRST_ACTIVATION_STAGE
    requirement_id: str = REQUIREMENT_ID

    def run(self, bed: Bed, seed: int) -> RunResult:
        """Compute the frontier and ablation ladder on both regimes at the bed's matched budget."""

        null_regime = bed.null_regime(seed)
        favorable_regime = bed.favorable_regime(seed)
        return RunResult(
            seed=seed,
            null_report=evaluate_regime(null_regime, seed),
            favorable_report=evaluate_regime(favorable_regime, seed),
            matched=bed.matched_cost(),
        )

    def demonstration_for(self, report: RegimeReport) -> RunReceipt:
        """Mint the per-regime demonstration: mechanics-ok only when this regime earns it, else null."""

        verdict = VERDICT_MECHANICS_OK if report.earned() else VERDICT_NULL
        detail = report.payload()
        return mint_demonstration(
            mechanism_id=self.mechanism_id,
            stage=self.stage,
            requirement_id=self.requirement_id,
            controls_cleared=report.dominated,
            evidence_digest=canonical_sha256(detail),
            verdict=verdict,
            detail=detail,
        )

    def mint(self, results: RunResult) -> RunReceipt:
        """Mint the run demonstration: mechanics-ok only when favorable earns and the null still holds."""

        favorable = results.favorable_report
        null = results.null_report
        earned = favorable.earned() and not null.dominates_all()
        verdict = VERDICT_MECHANICS_OK if earned else VERDICT_NULL
        detail: dict[str, Any] = {
            "seed": results.seed,
            "favorable": favorable.payload(),
            "null": null.payload(),
            "favorable_dominates_all": favorable.dominates_all(),
            "favorable_ablation_all_justified": favorable.ablation_all_justified(),
            "null_holds": not null.dominates_all(),
            "matched": results.matched.payload(),
        }
        return mint_demonstration(
            mechanism_id=self.mechanism_id,
            stage=self.stage,
            requirement_id=self.requirement_id,
            controls_cleared=favorable.dominated,
            evidence_digest=canonical_sha256(detail),
            verdict=verdict,
            detail=detail,
        )


def build_default_runner() -> IntegratedEscsRunner:
    """Return the canonical integrated ESCS runner."""

    return IntegratedEscsRunner()
