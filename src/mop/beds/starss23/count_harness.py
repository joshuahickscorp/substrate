"""Concurrent-source-counting bed, component 6: the matched-budget harness for count-MAE.

The metric-agnostic full-lifecycle FLOP accounting is reused BY IMPORT from the sealed ``harness.py``
(never edited): ``FlopModel``, ``featurize_run_flops``, ``gate_infer_run_flops``, ``gate_train_flops``,
``per_query_saving_vs_always_on``, ``break_even_queries``, ``BreakEven``, and the ladder ``MatchedBudget``.
Only the metric-carrying pieces are re-implemented here, because ``harness.Arm`` and ``harness.ArmSeedResult``
force the score into [0, 1] and the Pareto analysis MAXIMIZES it, whereas count-MAE is unbounded and
MINIMIZED (lower is better).

The honest test (saving-at-parity): at a matched re-estimation budget (byte-equal inference FLOPs, same K
per seed), does the trained candidate reach a strictly LOWER pooled count-MAE than the rate-matched-random
control at every budget point. A tie is a null. Every arm's full-lifecycle FLOPs are asserted under the
~6e10 ceiling, and the candidate alone charges its amortized training cost C_train in full. The report is
mechanics-only: ``activation_allowed``, ``scientific_promotion``, and
``independent_scientific_confirmation`` are hardcoded false, so it can never open a stage gate.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mop.ladder.stage_ladder import MatchedBudget
from mop.substrate.events import canonical_sha256

from . import CLAIM_SCOPE, FLOP_CEILING
from .harness import (  # reused unchanged from the sealed onset matched-budget harness
    BreakEven,
    FlopModel,
    break_even_queries,
    per_query_saving_vs_always_on,
)

COUNT_HARNESS_SCHEMA = "mop-starss23-count-harness/v1"

# This counting bed is a DIFFERENT question from the seven sealed onset-localization nulls, so it carries
# its own bed id. CLAIM_SCOPE and FLOP_CEILING are reused byte-identically from the package contract.
COUNT_BED_ID = "starss23_escs_source_counting"

ARM_CANDIDATE = "candidate"
ARM_RATE_MATCHED_RANDOM = "rate_matched_random"
ARM_ALWAYS_ON = "always_on"
ARM_NEVER_UPDATE = "never_update"
CONTROL_ARM_KINDS: tuple[str, ...] = (ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE)
ALL_ARM_KINDS: tuple[str, ...] = (ARM_CANDIDATE, *CONTROL_ARM_KINDS)

MAX_GATE_PARAMS = 4096


class CountHarnessRefusal(ValueError):
    """Raised when a count-harness input is malformed or a budget invariant would be violated."""


class CountBudgetMismatch(CountHarnessRefusal):
    """Raised when the candidate and a matched control do not share byte-equal inference FLOPs."""


class CountUnchargedTraining(CountHarnessRefusal):
    """Raised when a candidate comparison omits the amortized training cost C_train."""


class CountCeilingExceeded(CountHarnessRefusal):
    """Raised when an arm's full-lifecycle FLOPs exceed the matched budget ceiling."""


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CountHarnessRefusal(f"{label} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CountHarnessRefusal(f"{label} must be a positive integer")
    return value


def _require_mae(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CountHarnessRefusal(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise CountHarnessRefusal(f"{label} must be a finite nonnegative MAE")
    return number


def _seal_float(value: float) -> float:
    return round(float(value), 12)


@dataclass(frozen=True, slots=True)
class CountArmSeedResult:
    """One arm's result on one paired seed: the pooled count-MAE and the re-estimation count K on it."""

    seed: int
    mae: float
    reestimations: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.seed, "CountArmSeedResult.seed")
        object.__setattr__(self, "mae", _require_mae(self.mae, "CountArmSeedResult.mae"))
        _require_nonnegative_int(self.reestimations, "CountArmSeedResult.reestimations")

    def payload(self) -> dict[str, Any]:
        return {"seed": self.seed, "mae": _seal_float(self.mae), "reestimations": self.reestimations}


@dataclass(frozen=True, slots=True)
class CountArm:
    """One arm of the counting bed: kind, trained-parameter count, FLOP model, and per-seed results."""

    name: str
    kind: str
    total_frames: int
    params: int
    flop_model: FlopModel
    seed_results: tuple[CountArmSeedResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise CountHarnessRefusal("CountArm.name must be a non-empty string")
        if self.kind not in ALL_ARM_KINDS:
            raise CountHarnessRefusal(f"CountArm.kind {self.kind!r} is not one of {ALL_ARM_KINDS}")
        _require_positive_int(self.total_frames, "CountArm.total_frames")
        _require_nonnegative_int(self.params, "CountArm.params")
        if not isinstance(self.flop_model, FlopModel):
            raise CountHarnessRefusal("CountArm.flop_model must be a FlopModel")
        if not self.seed_results:
            raise CountHarnessRefusal("CountArm.seed_results must not be empty")
        object.__setattr__(self, "seed_results", tuple(self.seed_results))
        seeds = [result.seed for result in self.seed_results]
        if len(set(seeds)) != len(seeds):
            raise CountHarnessRefusal("CountArm.seed_results must have unique seeds")
        if seeds != sorted(seeds):
            raise CountHarnessRefusal("CountArm.seed_results must be in ascending seed order")
        for result in self.seed_results:
            if result.reestimations > self.total_frames:
                raise CountHarnessRefusal("a re-estimation count cannot exceed the total frame count")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(result.seed for result in self.seed_results)

    def result_for_seed(self, seed: int) -> CountArmSeedResult:
        for result in self.seed_results:
            if result.seed == seed:
                return result
        raise CountHarnessRefusal(f"arm {self.name!r} has no result for seed {seed}")

    def mean_mae(self) -> float:
        return math.fsum(result.mae for result in self.seed_results) / len(self.seed_results)

    def mean_lifecycle_flops(self) -> float:
        totals = [self.flop_model.lifecycle_flops(result.reestimations) for result in self.seed_results]
        return math.fsum(totals) / len(totals)

    def max_lifecycle_flops(self) -> int:
        return max(self.flop_model.lifecycle_flops(result.reestimations) for result in self.seed_results)

    def mean_reestimations(self) -> float:
        return math.fsum(result.reestimations for result in self.seed_results) / len(self.seed_results)

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "total_frames": self.total_frames,
            "params": self.params,
            "flop_model": self.flop_model.payload(),
            "seed_results": [result.payload() for result in self.seed_results],
            "mean_mae": _seal_float(self.mean_mae()),
            "mean_lifecycle_flops": _seal_float(self.mean_lifecycle_flops()),
            "max_lifecycle_flops": self.max_lifecycle_flops(),
            "mean_reestimations": _seal_float(self.mean_reestimations()),
        }


def assert_within_ceiling(arm: CountArm, ceiling: int = FLOP_CEILING) -> None:
    """Refuse unless every paired-seed run of the arm stays under the full-lifecycle FLOP ceiling."""

    ceiling = _require_positive_int(ceiling, "ceiling")
    for result in arm.seed_results:
        total = arm.flop_model.lifecycle_flops(result.reestimations)
        if total > ceiling:
            raise CountCeilingExceeded(
                f"arm {arm.name!r} seed {result.seed} lifecycle FLOPs {total} exceed ceiling {ceiling}"
            )


def assert_matched_ex_training(candidate: CountArm, control: CountArm) -> None:
    """Refuse unless the candidate and the matched control share byte-equal inference FLOPs.

    Same seeds, same total_frames; the candidate charges C_train > 0 and the control charges 0; and per
    seed the re-estimation counts K match AND the inference run FLOPs (featurize + gate_infer + K x C_reest)
    match byte-for-byte, so any MAE gap is WHERE re-estimations land, not how many.
    """

    if candidate.kind != ARM_CANDIDATE:
        raise CountHarnessRefusal("assert_matched_ex_training expects the candidate arm first")
    if candidate.seeds != control.seeds:
        raise CountBudgetMismatch("candidate and control do not share the same paired seeds")
    if candidate.total_frames != control.total_frames:
        raise CountBudgetMismatch("candidate and control cover a different number of frames")
    if candidate.flop_model.train_flops <= 0:
        raise CountUnchargedTraining(
            "the candidate must charge its amortized training cost C_train in full-lifecycle accounting"
        )
    if control.flop_model.train_flops != 0:
        raise CountBudgetMismatch("a control must not charge a training cost; it learns nothing")
    for seed in candidate.seeds:
        candidate_result = candidate.result_for_seed(seed)
        control_result = control.result_for_seed(seed)
        if candidate_result.reestimations != control_result.reestimations:
            raise CountBudgetMismatch(
                f"seed {seed}: control re-estimation count {control_result.reestimations} does not match "
                f"the candidate re-estimation count {candidate_result.reestimations}"
            )
        candidate_run = candidate.flop_model.run_flops(candidate_result.reestimations)
        control_run = control.flop_model.run_flops(control_result.reestimations)
        if candidate_run != control_run:
            raise CountBudgetMismatch(
                f"seed {seed}: control inference FLOPs {control_run} do not match the candidate "
                f"inference FLOPs {candidate_run}"
            )


def paired_deltas(candidate: CountArm, control: CountArm) -> tuple[float, ...]:
    """Return delta_i = MAE_control(i) - MAE_candidate(i). Positive delta = candidate LOWERS the error."""

    if candidate.seeds != control.seeds:
        raise CountHarnessRefusal("paired deltas require the same paired seeds on both arms")
    return tuple(
        control.result_for_seed(seed).mae - candidate.result_for_seed(seed).mae
        for seed in candidate.seeds
    )


@dataclass(frozen=True, slots=True)
class CountComputePoint:
    """A single (compute, MAE) point on the sweep. Compute is reported as a set of indicators."""

    budget_id: str
    arm_kind: str
    flops: float
    mae: float
    params: int
    reestimations: float

    def payload(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "arm_kind": self.arm_kind,
            "flops": _seal_float(self.flops),
            "mae": _seal_float(self.mae),
            "params": self.params,
            "reestimations": _seal_float(self.reestimations),
        }


def _compute_point(budget_id: str, arm: CountArm) -> CountComputePoint:
    return CountComputePoint(
        budget_id=budget_id,
        arm_kind=arm.kind,
        flops=arm.mean_lifecycle_flops(),
        mae=arm.mean_mae(),
        params=arm.params,
        reestimations=arm.mean_reestimations(),
    )


def pareto_frontier(points: Sequence[CountComputePoint]) -> tuple[CountComputePoint, ...]:
    """Return the non-dominated set: minimize FLOPs AND minimize MAE (both lower is better).

    A point is dominated when another has FLOPs less than or equal and MAE less than or equal, with at
    least one of the two strict. The frontier is returned sorted by ascending FLOPs.
    """

    unique = list(points)
    frontier: list[CountComputePoint] = []
    for candidate in unique:
        dominated = False
        for other in unique:
            if other is candidate:
                continue
            no_worse = other.flops <= candidate.flops and other.mae <= candidate.mae
            strictly_better = other.flops < candidate.flops or other.mae < candidate.mae
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda point: (point.flops, point.mae))
    return tuple(frontier)


@dataclass(frozen=True, slots=True)
class CountBudgetPoint:
    """One re-estimation-budget operating point: the candidate plus its three controls at a shared budget."""

    budget_id: str
    candidate: CountArm
    rate_matched_random: CountArm
    always_on: CountArm
    never_update: CountArm

    def __post_init__(self) -> None:
        if not isinstance(self.budget_id, str) or not self.budget_id.strip():
            raise CountHarnessRefusal("CountBudgetPoint.budget_id must be a non-empty string")
        for arm, expected in (
            (self.candidate, ARM_CANDIDATE),
            (self.rate_matched_random, ARM_RATE_MATCHED_RANDOM),
            (self.always_on, ARM_ALWAYS_ON),
            (self.never_update, ARM_NEVER_UPDATE),
        ):
            if not isinstance(arm, CountArm):
                raise CountHarnessRefusal("CountBudgetPoint arms must be CountArm instances")
            if arm.kind != expected:
                raise CountHarnessRefusal(f"CountBudgetPoint arm {arm.name!r} must have kind {expected!r}")
        seeds = self.candidate.seeds
        for arm in self.arms():
            if arm.seeds != seeds:
                raise CountHarnessRefusal("all arms in a budget point must share the same paired seeds")
            if arm.total_frames != self.candidate.total_frames:
                raise CountHarnessRefusal("all arms in a budget point must cover the same frame count")

    def arms(self) -> tuple[CountArm, ...]:
        return (self.candidate, self.rate_matched_random, self.always_on, self.never_update)

    def certify(self, ceiling: int = FLOP_CEILING) -> None:
        for arm in self.arms():
            assert_within_ceiling(arm, ceiling)
            if arm.kind in CONTROL_ARM_KINDS and arm.flop_model.train_flops != 0:
                raise CountBudgetMismatch(f"control arm {arm.name!r} must not charge a training cost")
        assert_matched_ex_training(self.candidate, self.rate_matched_random)

    def candidate_beats_rate_matched_random(self) -> bool:
        """Lower MAE is better, so the candidate wins iff its mean MAE is strictly below the control's."""

        return self.candidate.mean_mae() < self.rate_matched_random.mean_mae()


@dataclass(frozen=True, slots=True)
class CountHarnessReport:
    """The assembled matched-budget analysis. Mechanics-only: it can never clear a stage gate."""

    schema: str
    bed_id: str
    source_kind: str
    flop_ceiling: int
    seeds: tuple[int, ...]
    arm_summaries: tuple[dict[str, Any], ...]
    pareto: tuple[CountComputePoint, ...]
    per_budget_candidate_vs_rate_matched_random: tuple[dict[str, Any], ...]
    candidate_strictly_dominates_rate_matched_random: bool
    break_even: BreakEven
    matched_budget: MatchedBudget
    verdict: str
    activation_allowed: bool
    scientific_promotion: bool
    independent_scientific_confirmation: bool
    claim_scope: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "bed_id": self.bed_id,
            "source_kind": self.source_kind,
            "flop_ceiling": self.flop_ceiling,
            "seeds": list(self.seeds),
            "arm_summaries": [dict(summary) for summary in self.arm_summaries],
            "pareto": [point.payload() for point in self.pareto],
            "per_budget_candidate_vs_rate_matched_random": [
                dict(row) for row in self.per_budget_candidate_vs_rate_matched_random
            ],
            "candidate_strictly_dominates_rate_matched_random": (
                self.candidate_strictly_dominates_rate_matched_random
            ),
            "break_even": self.break_even.payload(),
            "matched_budget": self.matched_budget.payload(),
            "verdict": self.verdict,
            "activation_allowed": self.activation_allowed,
            "scientific_promotion": self.scientific_promotion,
            "independent_scientific_confirmation": self.independent_scientific_confirmation,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _matched_budget_from_points(
    budget_points: Sequence[CountBudgetPoint], wall_ns: int, ceiling: int
) -> MatchedBudget:
    wall_ns = _require_positive_int(wall_ns, "wall_ns")
    candidate = budget_points[0].candidate
    binding_flops = max(point.candidate.max_lifecycle_flops() for point in budget_points)
    if binding_flops > ceiling:
        raise CountCeilingExceeded(f"binding candidate FLOPs {binding_flops} exceed ceiling {ceiling}")
    return MatchedBudget(
        params=candidate.params,
        flops=binding_flops,
        wall_ns=wall_ns,
        seeds=len(candidate.seeds),
    )


def run_matched_budget(
    budget_points: Sequence[CountBudgetPoint],
    *,
    wall_ns: int,
    operating_budget_id: str | None = None,
    source_kind: str = "synthetic",
    ceiling: int = FLOP_CEILING,
) -> CountHarnessReport:
    """Run the matched-budget analysis over a re-estimation-budget sweep and assemble the mechanics report.

    Each budget point is certified (every arm within the ceiling, the matched control matched ex-training
    and charging no training, the candidate charging C_train). The candidate strictly dominates the
    rate-matched-random control only when its pooled mean count-MAE is strictly LOWER at every budget point,
    at matched inference cost. A tie is a null. The verdict is capped at mechanics-ok and the boundary flags
    are hardcoded off, so the report can never open a stage gate.
    """

    points = list(budget_points)
    if not points:
        raise CountHarnessRefusal("run_matched_budget needs at least one budget point")
    if source_kind not in ("synthetic", "real"):
        raise CountHarnessRefusal("source_kind must be 'synthetic' or 'real'")

    seeds = points[0].candidate.seeds
    for point in points:
        if point.candidate.seeds != seeds:
            raise CountHarnessRefusal("all budget points must share the same paired seeds")
        point.certify(ceiling)

    budget_ids = [point.budget_id for point in points]
    if len(set(budget_ids)) != len(budget_ids):
        raise CountHarnessRefusal("budget point ids must be unique")

    compute_points: list[CountComputePoint] = []
    per_budget_rows: list[dict[str, Any]] = []
    dominates_everywhere = True
    for point in points:
        for arm in point.arms():
            compute_points.append(_compute_point(point.budget_id, arm))
        candidate_mae = point.candidate.mean_mae()
        rmr_mae = point.rate_matched_random.mean_mae()
        wins = candidate_mae < rmr_mae
        dominates_everywhere = dominates_everywhere and wins
        per_budget_rows.append(
            {
                "budget_id": point.budget_id,
                "candidate_mean_mae": _seal_float(candidate_mae),
                "rate_matched_random_mean_mae": _seal_float(rmr_mae),
                "delta_mean_mae_control_minus_candidate": _seal_float(rmr_mae - candidate_mae),
                "matched_inference_flops": point.candidate.flop_model.run_flops(
                    point.candidate.result_for_seed(seeds[0]).reestimations
                ),
                "candidate_strictly_beats_rate_matched_random": wins,
            }
        )

    operating = points[0]
    if operating_budget_id is not None:
        matches = [point for point in points if point.budget_id == operating_budget_id]
        if not matches:
            raise CountHarnessRefusal(f"operating_budget_id {operating_budget_id!r} is not in the sweep")
        operating = matches[0]
    operating_saving = per_query_saving_vs_always_on(
        operating.candidate.total_frames,
        round(operating.candidate.mean_reestimations()),
        operating.candidate.flop_model.downstream_flops_per_firing,
    )
    break_even = break_even_queries(operating.candidate.flop_model.train_flops, operating_saving)

    matched_budget = _matched_budget_from_points(points, wall_ns=wall_ns, ceiling=ceiling)

    verdict = "mechanics-ok" if dominates_everywhere else "null"

    arm_summaries = tuple(arm.payload() for point in points for arm in point.arms())
    return CountHarnessReport(
        schema=COUNT_HARNESS_SCHEMA,
        bed_id=COUNT_BED_ID,
        source_kind=source_kind,
        flop_ceiling=ceiling,
        seeds=seeds,
        arm_summaries=arm_summaries,
        pareto=pareto_frontier(compute_points),
        per_budget_candidate_vs_rate_matched_random=tuple(per_budget_rows),
        candidate_strictly_dominates_rate_matched_random=dominates_everywhere,
        break_even=break_even,
        matched_budget=matched_budget,
        verdict=verdict,
        activation_allowed=False,
        scientific_promotion=False,
        independent_scientific_confirmation=False,
        claim_scope=CLAIM_SCOPE,
    )
