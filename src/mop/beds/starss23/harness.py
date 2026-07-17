"""STARSS23 ESCS bed, component 6: the matched-budget harness.

This module implements the matched full-system budget discipline from docs/ESCS_DEEP_RESEARCH.md
(Q2a, Q2e) for the STARSS23 event-formation bed. It carries no experiment and no capability claim. It
takes the per-arm, per-seed results the arms produce (candidate gate plus the three controls) and does
the full-lifecycle FLOP accounting, matching, and Pareto analysis the recipe requires:

- Per-arm full-lifecycle FLOP ledger. For a run with K firings the arm FLOPs are
  F_arm = F_featurize + F_gate_infer + K * C_down, and the candidate additionally charges its amortized
  training cost C_train. Every arm total is asserted to stay under the ~6e10 FLOP ceiling.

- Matched budget, ex-training. The candidate and the rate-matched-random control must have byte-equal
  inference FLOPs at every firing budget and paired seed, so any accuracy gap is attributable to WHERE
  compute is spent, not how much. The comparison refuses unless the inference budgets match and the
  candidate actually charges C_train (the honest full-lifecycle rule of Green AI, Schwartz et al. 2020).

- Accuracy-versus-total-compute Pareto and strict dominance. Sweeping the firing budget traces
  (compute, accuracy) points; compute is reported as a set of indicators (params, FLOPs, wall clock)
  because they disagree (The Efficiency Misnomer, Dehghani et al. ICLR 2022). The headline verdict is
  whether the candidate strictly dominates the rate-matched-random control at matched inference cost.

- Break-even. The amortized training cost is only worth paying after N* = C_train / per_query_saving
  queries, where per_query_saving = (F - K) / F * C_down measured against the always-on reference.

Because the data is synthetic, the harness caps its verdict at mechanics-ok and never clears a gate.
Every report hardcodes activation_allowed=False, scientific_promotion=False, and
independent_scientific_confirmation=False. House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mop.ladder.stage_ladder import MatchedBudget
from mop.substrate.events import canonical_sha256

from . import BED_ID, CLAIM_SCOPE, FLOP_CEILING

__all__ = [
    "HARNESS_SCHEMA",
    "ARM_CANDIDATE",
    "ARM_RATE_MATCHED_RANDOM",
    "ARM_ALWAYS_ON",
    "ARM_BEST_SINGLE",
    "CONTROL_ARM_KINDS",
    "ALL_ARM_KINDS",
    "MAX_GATE_PARAMS",
    "FEATURIZE_FLOPS_PER_FRAME",
    "GATE_INFER_FLOPS_PER_FRAME",
    "GATE_PARAMS",
    "TRAIN_BACKWARD_MULTIPLIER",
    "HarnessRefusal",
    "BudgetMismatch",
    "UnchargedTraining",
    "CeilingExceeded",
    "FlopModel",
    "ArmSeedResult",
    "Arm",
    "BudgetPoint",
    "ComputePoint",
    "BreakEven",
    "HarnessReport",
    "featurize_run_flops",
    "gate_infer_run_flops",
    "gate_train_flops",
    "paired_deltas",
    "per_query_saving_vs_always_on",
    "break_even_queries",
    "assert_within_ceiling",
    "assert_matched_ex_training",
    "pareto_frontier",
    "run_matched_budget",
]

HARNESS_SCHEMA = "mop-starss23-escs-harness/v1"

# The four arms of the bed. The candidate is the only trained module; the three controls isolate the
# three honest questions (where-not-how-much, max-compute and non-learned reference, irreducible noise).
ARM_CANDIDATE = "candidate"
ARM_RATE_MATCHED_RANDOM = "rate_matched_random"
ARM_ALWAYS_ON = "always_on"
ARM_BEST_SINGLE = "best_single"
CONTROL_ARM_KINDS: tuple[str, ...] = (ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_BEST_SINGLE)
ALL_ARM_KINDS: tuple[str, ...] = (ARM_CANDIDATE, *CONTROL_ARM_KINDS)

# The hard ESCS parameter ceiling for the only trained module.
MAX_GATE_PARAMS = 4096

# Canonical FLOP-count anchors from the bed specification. These mirror the featurizer and gate FLOP
# formulas so the harness is usable and testable without importing those sibling modules; in the real
# run the authoritative counts flow in from featurizer.FrozenFeaturizer.flops_for_frames and
# gate.inference_flops / gate.training_flops through FlopModel. The harness never relies on these
# constants for correctness: it operates on whatever integers it is given.
FEATURIZE_FLOPS_PER_FRAME = 1_121_340
GATE_INFER_FLOPS_PER_FRAME = 6_385
GATE_PARAMS = 3_193
# A training step is one forward plus a backward pass, charged at three times the forward FLOPs.
TRAIN_BACKWARD_MULTIPLIER = 3


class HarnessRefusal(ValueError):
    """Raised when a harness input is malformed or a budget invariant would be violated."""


class BudgetMismatch(HarnessRefusal):
    """Raised when the candidate and a matched control do not share byte-equal inference FLOPs."""


class UnchargedTraining(HarnessRefusal):
    """Raised when a candidate comparison omits the amortized training cost C_train."""


class CeilingExceeded(HarnessRefusal):
    """Raised when an arm's full-lifecycle FLOPs exceed the matched budget ceiling."""


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarnessRefusal(f"{label} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HarnessRefusal(f"{label} must be a positive integer")
    return value


def _require_unit_interval(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessRefusal(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise HarnessRefusal(f"{label} must be a finite F1 score in [0, 1]")
    return number


def _seal_float(value: float) -> float:
    """Round a reported statistic so the sealed digest is byte-reproducible."""

    return round(float(value), 12)


# ---------------------------------------------------------------------------
# FLOP-count anchors mirroring the featurizer and gate specifications.
# ---------------------------------------------------------------------------


def featurize_run_flops(total_frames: int) -> int:
    """Return the frozen featurizer FLOPs for a run of ``total_frames`` 100 ms frames."""

    total_frames = _require_nonnegative_int(total_frames, "total_frames")
    return FEATURIZE_FLOPS_PER_FRAME * total_frames


def gate_infer_run_flops(total_frames: int) -> int:
    """Return the candidate gate inference FLOPs for a run of ``total_frames`` frames."""

    total_frames = _require_nonnegative_int(total_frames, "total_frames")
    return GATE_INFER_FLOPS_PER_FRAME * total_frames


def gate_train_flops(
    epochs: int, train_frames: int, infer_flops_per_frame: int = GATE_INFER_FLOPS_PER_FRAME
) -> int:
    """Return the amortized gate training cost C_train charged in full-lifecycle accounting.

    C_train = epochs * train_frames * TRAIN_BACKWARD_MULTIPLIER * per-frame forward FLOPs.
    """

    epochs = _require_positive_int(epochs, "epochs")
    train_frames = _require_positive_int(train_frames, "train_frames")
    infer_flops_per_frame = _require_positive_int(infer_flops_per_frame, "infer_flops_per_frame")
    return epochs * train_frames * TRAIN_BACKWARD_MULTIPLIER * infer_flops_per_frame


# ---------------------------------------------------------------------------
# Per-arm FLOP model and results.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FlopModel:
    """The full-lifecycle FLOP structure of one arm. Deterministic and shared across paired seeds.

    ``train_flops`` is the amortized training cost C_train, charged only by the trained candidate. A
    control that runs an equal-cost inference stub still carries the same ``gate_infer_flops`` but zero
    ``train_flops``, because it learns nothing.
    """

    featurize_flops: int
    gate_infer_flops: int
    downstream_flops_per_firing: int
    train_flops: int = 0

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.featurize_flops, "FlopModel.featurize_flops")
        _require_nonnegative_int(self.gate_infer_flops, "FlopModel.gate_infer_flops")
        _require_nonnegative_int(self.downstream_flops_per_firing, "FlopModel.downstream_flops_per_firing")
        _require_nonnegative_int(self.train_flops, "FlopModel.train_flops")

    def run_flops(self, firings: int) -> int:
        """Inference FLOPs for a run with ``firings`` firings: featurize + gate infer + K * C_down."""

        firings = _require_nonnegative_int(firings, "firings")
        return self.featurize_flops + self.gate_infer_flops + firings * self.downstream_flops_per_firing

    def lifecycle_flops(self, firings: int) -> int:
        """Full-lifecycle FLOPs: the inference run plus the amortized training cost C_train."""

        return self.run_flops(firings) + self.train_flops

    def payload(self) -> dict[str, int]:
        return {
            "featurize_flops": self.featurize_flops,
            "gate_infer_flops": self.gate_infer_flops,
            "downstream_flops_per_firing": self.downstream_flops_per_firing,
            "train_flops": self.train_flops,
        }


@dataclass(frozen=True, slots=True)
class ArmSeedResult:
    """One arm's result on one paired seed: the referee F1 and the firing count K on that seed."""

    seed: int
    f1: float
    firings: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.seed, "ArmSeedResult.seed")
        object.__setattr__(self, "f1", _require_unit_interval(self.f1, "ArmSeedResult.f1"))
        _require_nonnegative_int(self.firings, "ArmSeedResult.firings")

    def payload(self) -> dict[str, Any]:
        return {"seed": self.seed, "f1": _seal_float(self.f1), "firings": self.firings}


@dataclass(frozen=True, slots=True)
class Arm:
    """One arm of the bed: its kind, trained-parameter count, FLOP model, and per-seed results."""

    name: str
    kind: str
    total_frames: int
    params: int
    flop_model: FlopModel
    seed_results: tuple[ArmSeedResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise HarnessRefusal("Arm.name must be a non-empty string")
        if self.kind not in ALL_ARM_KINDS:
            raise HarnessRefusal(f"Arm.kind {self.kind!r} is not one of {ALL_ARM_KINDS}")
        _require_positive_int(self.total_frames, "Arm.total_frames")
        _require_nonnegative_int(self.params, "Arm.params")
        if not isinstance(self.flop_model, FlopModel):
            raise HarnessRefusal("Arm.flop_model must be a FlopModel")
        if not self.seed_results:
            raise HarnessRefusal("Arm.seed_results must not be empty")
        object.__setattr__(self, "seed_results", tuple(self.seed_results))
        seeds = [result.seed for result in self.seed_results]
        if len(set(seeds)) != len(seeds):
            raise HarnessRefusal("Arm.seed_results must have unique seeds")
        if seeds != sorted(seeds):
            raise HarnessRefusal("Arm.seed_results must be in ascending seed order")
        for result in self.seed_results:
            if result.firings > self.total_frames:
                raise HarnessRefusal("a firing count cannot exceed the total frame count")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(result.seed for result in self.seed_results)

    def result_for_seed(self, seed: int) -> ArmSeedResult:
        for result in self.seed_results:
            if result.seed == seed:
                return result
        raise HarnessRefusal(f"arm {self.name!r} has no result for seed {seed}")

    def mean_f1(self) -> float:
        return math.fsum(result.f1 for result in self.seed_results) / len(self.seed_results)

    def mean_lifecycle_flops(self) -> float:
        totals = [self.flop_model.lifecycle_flops(result.firings) for result in self.seed_results]
        return math.fsum(totals) / len(totals)

    def max_lifecycle_flops(self) -> int:
        return max(self.flop_model.lifecycle_flops(result.firings) for result in self.seed_results)

    def mean_firings(self) -> float:
        return math.fsum(result.firings for result in self.seed_results) / len(self.seed_results)

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "total_frames": self.total_frames,
            "params": self.params,
            "flop_model": self.flop_model.payload(),
            "seed_results": [result.payload() for result in self.seed_results],
            "mean_f1": _seal_float(self.mean_f1()),
            "mean_lifecycle_flops": _seal_float(self.mean_lifecycle_flops()),
            "max_lifecycle_flops": self.max_lifecycle_flops(),
        }


# ---------------------------------------------------------------------------
# Full-lifecycle accounting checks.
# ---------------------------------------------------------------------------


def assert_within_ceiling(arm: Arm, ceiling: int = FLOP_CEILING) -> None:
    """Refuse unless every paired-seed run of the arm stays under the full-lifecycle FLOP ceiling."""

    ceiling = _require_positive_int(ceiling, "ceiling")
    for result in arm.seed_results:
        total = arm.flop_model.lifecycle_flops(result.firings)
        if total > ceiling:
            raise CeilingExceeded(
                f"arm {arm.name!r} seed {result.seed} lifecycle FLOPs {total} exceed ceiling {ceiling}"
            )


def assert_matched_ex_training(candidate: Arm, control: Arm) -> None:
    """Refuse unless the candidate and the matched control share byte-equal inference FLOPs.

    The candidate and the rate-matched-random control must fire the same count on each paired seed and
    have byte-equal inference FLOPs, so any accuracy gap is WHERE compute is spent, not how much. The
    candidate must charge its amortized training cost C_train; the control must charge none. Any
    violation refuses the comparison.
    """

    if candidate.kind != ARM_CANDIDATE:
        raise HarnessRefusal("assert_matched_ex_training expects the candidate arm first")
    if candidate.seeds != control.seeds:
        raise BudgetMismatch("candidate and control do not share the same paired seeds")
    if candidate.total_frames != control.total_frames:
        raise BudgetMismatch("candidate and control cover a different number of frames")
    if candidate.flop_model.train_flops <= 0:
        raise UnchargedTraining(
            "the candidate must charge its amortized training cost C_train in full-lifecycle accounting"
        )
    if control.flop_model.train_flops != 0:
        raise BudgetMismatch("a control must not charge a training cost; it learns nothing")
    for seed in candidate.seeds:
        candidate_result = candidate.result_for_seed(seed)
        control_result = control.result_for_seed(seed)
        if candidate_result.firings != control_result.firings:
            raise BudgetMismatch(
                f"seed {seed}: control firing count {control_result.firings} does not match the "
                f"candidate firing count {candidate_result.firings}"
            )
        candidate_run = candidate.flop_model.run_flops(candidate_result.firings)
        control_run = control.flop_model.run_flops(control_result.firings)
        if candidate_run != control_run:
            raise BudgetMismatch(
                f"seed {seed}: control inference FLOPs {control_run} do not match the candidate "
                f"inference FLOPs {candidate_run}"
            )


def paired_deltas(candidate: Arm, control: Arm) -> tuple[float, ...]:
    """Return the paired per-seed F1 deltas Delta_i = F1_candidate(i) - F1_control(i)."""

    if candidate.seeds != control.seeds:
        raise HarnessRefusal("paired deltas require the same paired seeds on both arms")
    return tuple(
        candidate.result_for_seed(seed).f1 - control.result_for_seed(seed).f1 for seed in candidate.seeds
    )


# ---------------------------------------------------------------------------
# Break-even accounting.
# ---------------------------------------------------------------------------


def per_query_saving_vs_always_on(
    total_frames: int, firings: int, downstream_flops_per_firing: int
) -> float:
    """Return the average per-frame downstream FLOPs saved against the always-on reference.

    The always-on reference fires on every one of the ``total_frames`` frames. A gate that fires only
    ``firings`` times spends downstream compute on ``firings`` frames, so the average per-frame saving
    is (F - K) / F * C_down.
    """

    total_frames = _require_positive_int(total_frames, "total_frames")
    firings = _require_nonnegative_int(firings, "firings")
    downstream_flops_per_firing = _require_nonnegative_int(
        downstream_flops_per_firing, "downstream_flops_per_firing"
    )
    if firings > total_frames:
        raise HarnessRefusal("firings cannot exceed the total frame count")
    return (total_frames - firings) / total_frames * downstream_flops_per_firing


@dataclass(frozen=True, slots=True)
class BreakEven:
    """The full-lifecycle break-even: how many queries recover the amortized training cost C_train."""

    train_flops: int
    per_query_saving: float
    n_star_frames: int | None
    amortizable: bool

    def payload(self) -> dict[str, Any]:
        return {
            "train_flops": self.train_flops,
            "per_query_saving": _seal_float(self.per_query_saving),
            "n_star_frames": self.n_star_frames,
            "amortizable": self.amortizable,
        }


def break_even_queries(train_flops: int, per_query_saving: float) -> BreakEven:
    """Return N* = ceil(C_train / per_query_saving), or an unamortizable marker if no saving exists."""

    train_flops = _require_nonnegative_int(train_flops, "train_flops")
    if isinstance(per_query_saving, bool) or not isinstance(per_query_saving, (int, float)):
        raise HarnessRefusal("per_query_saving must be a real number")
    saving = float(per_query_saving)
    if not math.isfinite(saving):
        raise HarnessRefusal("per_query_saving must be finite")
    if saving <= 0.0:
        return BreakEven(
            train_flops=train_flops, per_query_saving=saving, n_star_frames=None, amortizable=False
        )
    n_star = math.ceil(train_flops / saving)
    return BreakEven(
        train_flops=train_flops, per_query_saving=saving, n_star_frames=n_star, amortizable=True
    )


# ---------------------------------------------------------------------------
# Budget points, Pareto, and strict dominance.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    """One firing-budget operating point: the candidate plus its three controls at a shared budget."""

    budget_id: str
    candidate: Arm
    rate_matched_random: Arm
    always_on: Arm
    best_single: Arm

    def __post_init__(self) -> None:
        if not isinstance(self.budget_id, str) or not self.budget_id.strip():
            raise HarnessRefusal("BudgetPoint.budget_id must be a non-empty string")
        for arm, expected in (
            (self.candidate, ARM_CANDIDATE),
            (self.rate_matched_random, ARM_RATE_MATCHED_RANDOM),
            (self.always_on, ARM_ALWAYS_ON),
            (self.best_single, ARM_BEST_SINGLE),
        ):
            if not isinstance(arm, Arm):
                raise HarnessRefusal("BudgetPoint arms must be Arm instances")
            if arm.kind != expected:
                raise HarnessRefusal(f"BudgetPoint arm {arm.name!r} must have kind {expected!r}")
        seeds = self.candidate.seeds
        for arm in self.arms():
            if arm.seeds != seeds:
                raise HarnessRefusal("all arms in a budget point must share the same paired seeds")
            if arm.total_frames != self.candidate.total_frames:
                raise HarnessRefusal("all arms in a budget point must cover the same frame count")

    def arms(self) -> tuple[Arm, ...]:
        return (self.candidate, self.rate_matched_random, self.always_on, self.best_single)

    def certify(self, ceiling: int = FLOP_CEILING) -> None:
        """Refuse unless every arm is within the ceiling and the matched control matches ex-training."""

        for arm in self.arms():
            assert_within_ceiling(arm, ceiling)
            if arm.kind in CONTROL_ARM_KINDS and arm.flop_model.train_flops != 0:
                raise BudgetMismatch(f"control arm {arm.name!r} must not charge a training cost")
        assert_matched_ex_training(self.candidate, self.rate_matched_random)

    def candidate_beats_rate_matched_random(self) -> bool:
        return self.candidate.mean_f1() > self.rate_matched_random.mean_f1()


@dataclass(frozen=True, slots=True)
class ComputePoint:
    """A single (compute, accuracy) point on the sweep. Compute is reported as a set of indicators."""

    budget_id: str
    arm_kind: str
    flops: float
    f1: float
    params: int
    firings: float

    def payload(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "arm_kind": self.arm_kind,
            "flops": _seal_float(self.flops),
            "f1": _seal_float(self.f1),
            "params": self.params,
            "firings": _seal_float(self.firings),
        }


def _compute_point(budget_id: str, arm: Arm) -> ComputePoint:
    return ComputePoint(
        budget_id=budget_id,
        arm_kind=arm.kind,
        flops=arm.mean_lifecycle_flops(),
        f1=arm.mean_f1(),
        params=arm.params,
        firings=arm.mean_firings(),
    )


def pareto_frontier(points: Sequence[ComputePoint]) -> tuple[ComputePoint, ...]:
    """Return the non-dominated set: minimize FLOPs and maximize F1.

    A point is dominated when another point has FLOPs less than or equal and F1 greater than or equal,
    with at least one of the two strict. The frontier is returned sorted by ascending FLOPs.
    """

    unique = list(points)
    frontier: list[ComputePoint] = []
    for candidate in unique:
        dominated = False
        for other in unique:
            if other is candidate:
                continue
            no_worse = other.flops <= candidate.flops and other.f1 >= candidate.f1
            strictly_better = other.flops < candidate.flops or other.f1 > candidate.f1
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda point: (point.flops, -point.f1))
    return tuple(frontier)


# ---------------------------------------------------------------------------
# The assembled matched-budget report.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HarnessReport:
    """The assembled matched-budget analysis. Mechanics-only: it can never clear a stage gate."""

    schema: str
    bed_id: str
    source_kind: str
    flop_ceiling: int
    seeds: tuple[int, ...]
    arm_summaries: tuple[dict[str, Any], ...]
    pareto: tuple[ComputePoint, ...]
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
    budget_points: Sequence[BudgetPoint], wall_ns: int, ceiling: int
) -> MatchedBudget:
    """Build the ladder MatchedBudget: candidate params, binding lifecycle FLOPs, wall, and seed count."""

    wall_ns = _require_positive_int(wall_ns, "wall_ns")
    candidate = budget_points[0].candidate
    binding_flops = max(point.candidate.max_lifecycle_flops() for point in budget_points)
    if binding_flops > ceiling:
        raise CeilingExceeded(f"binding candidate FLOPs {binding_flops} exceed ceiling {ceiling}")
    return MatchedBudget(
        params=candidate.params,
        flops=binding_flops,
        wall_ns=wall_ns,
        seeds=len(candidate.seeds),
    )


def run_matched_budget(
    budget_points: Sequence[BudgetPoint],
    *,
    wall_ns: int,
    operating_budget_id: str | None = None,
    source_kind: str = "synthetic",
    ceiling: int = FLOP_CEILING,
) -> HarnessReport:
    """Run the matched-budget analysis over a firing-budget sweep and assemble the mechanics report.

    Each budget point is certified (every arm within the ceiling, the matched control matched
    ex-training and charging no training, the candidate charging C_train). The candidate strictly
    dominates the rate-matched-random control only when it wins on mean F1 at every budget point, at
    matched inference cost. Because the data is synthetic the verdict is capped at mechanics-ok and the
    boundary flags are hardcoded off, so the report can never open a stage gate.
    """

    points = list(budget_points)
    if not points:
        raise HarnessRefusal("run_matched_budget needs at least one budget point")
    if source_kind not in ("synthetic", "real"):
        raise HarnessRefusal("source_kind must be 'synthetic' or 'real'")

    seeds = points[0].candidate.seeds
    for point in points:
        if point.candidate.seeds != seeds:
            raise HarnessRefusal("all budget points must share the same paired seeds")
        point.certify(ceiling)

    budget_ids = [point.budget_id for point in points]
    if len(set(budget_ids)) != len(budget_ids):
        raise HarnessRefusal("budget point ids must be unique")

    compute_points: list[ComputePoint] = []
    per_budget_rows: list[dict[str, Any]] = []
    dominates_everywhere = True
    for point in points:
        for arm in point.arms():
            compute_points.append(_compute_point(point.budget_id, arm))
        candidate_f1 = point.candidate.mean_f1()
        rmr_f1 = point.rate_matched_random.mean_f1()
        wins = candidate_f1 > rmr_f1
        dominates_everywhere = dominates_everywhere and wins
        per_budget_rows.append(
            {
                "budget_id": point.budget_id,
                "candidate_mean_f1": _seal_float(candidate_f1),
                "rate_matched_random_mean_f1": _seal_float(rmr_f1),
                "delta_mean_f1": _seal_float(candidate_f1 - rmr_f1),
                "matched_inference_flops": point.candidate.flop_model.run_flops(
                    point.candidate.result_for_seed(seeds[0]).firings
                ),
                "candidate_strictly_beats_rate_matched_random": wins,
            }
        )

    # The break-even is evaluated at the chosen operating budget (default: the first budget point).
    operating = points[0]
    if operating_budget_id is not None:
        matches = [point for point in points if point.budget_id == operating_budget_id]
        if not matches:
            raise HarnessRefusal(f"operating_budget_id {operating_budget_id!r} is not in the sweep")
        operating = matches[0]
    operating_saving = per_query_saving_vs_always_on(
        operating.candidate.total_frames,
        round(operating.candidate.mean_firings()),
        operating.candidate.flop_model.downstream_flops_per_firing,
    )
    break_even = break_even_queries(operating.candidate.flop_model.train_flops, operating_saving)

    matched_budget = _matched_budget_from_points(points, wall_ns=wall_ns, ceiling=ceiling)

    # Synthetic data caps the verdict at mechanics-ok; a real source would still need the independent
    # verifier before any promotion. The verdict is mechanics-ok when the candidate strictly dominates
    # the rate-matched-random control at matched cost, and null otherwise.
    verdict = "mechanics-ok" if dominates_everywhere else "null"

    arm_summaries = tuple(arm.payload() for point in points for arm in point.arms())
    return HarnessReport(
        schema=HARNESS_SCHEMA,
        bed_id=BED_ID,
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
