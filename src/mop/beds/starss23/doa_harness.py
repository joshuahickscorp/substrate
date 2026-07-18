"""Direction-of-arrival bed, component 7: the dual-architecture matched-budget harness.

This is the second hardening this bed builds in from day one: gate-architecture invariance is checked IN
THE SAME RUN, not bolted on after a positive the way the counting bed's ``count_repro_gate_arch`` reproduction
was. The FLOP model (``FlopModel``) is reused unchanged BY IMPORT from the sealed ``harness.py``, since it
is metric- and architecture-agnostic. ``assert_within_ceiling`` and ``assert_matched_ex_training`` are
REIMPLEMENTED here (not imported from ``harness.py``) for the same reason ``count_harness.py`` reimplements
them rather than importing them unchanged: the sealed ``harness.Arm``/``ArmSeedResult`` hardcode a
``.firings`` field name and an F1-in-[0,1]-maximized score, whereas this bed's arms carry ``.reestimations``
and a MINIMIZED, unbounded-below great-circle MAE in degrees, exactly the metric-shape mismatch that already
forced ``count_harness.py`` off the literal by-import path for these two functions.

Per architecture X in ``{arch_a_264_12_1, arch_b_264_6_6_1}``, the matched-budget analysis (certify every
arm within the ~6e10 FLOP ceiling, the matched rate-matched-random control matched ex-training and
charging no training, the candidate charging its architecture's own C_train) runs independently, on that
architecture's OWN paired seeds and OWN rate-matched-random draws. The candidate "dominates" for
architecture X only when its clip-macro mean angular error is strictly lower than the rate-matched-random
control's at every swept budget point, at matched inference cost. A tie is a null for that architecture.

The bed-level verdict from this module alone is a coarse dominance signal only (mechanics-ok / null per
architecture, folded into mechanics-ok / architecture-fragile / null at the bed level); it does NOT include
the clip-level sign-flip, the SESOI, or the room-majority discipline, which are statistical tests computed
downstream by ``doa_referee.py`` and assembled into the full SURVIVES(X) conjunction by ``doa_producer.py``.

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
from .doa_gate import ARCH_A_ID, ARCH_B_ID, ARCHITECTURES
from .harness import (  # reused unchanged: metric- and architecture-agnostic
    BreakEven,
    FlopModel,
    break_even_queries,
    per_query_saving_vs_always_on,
)

DOA_HARNESS_SCHEMA = "mop-starss23-doa-harness/v1"

# Defined locally rather than added to __init__.py, following count_harness.COUNT_BED_ID's own precedent.
DOA_BED_ID = "starss23_escs_direction_of_arrival"

ARM_CANDIDATE = "candidate"
ARM_RATE_MATCHED_RANDOM = "rate_matched_random"
ARM_ALWAYS_ON = "always_on"
ARM_NEVER_UPDATE = "never_update"
CONTROL_ARM_KINDS: tuple[str, ...] = (ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE)
ALL_ARM_KINDS: tuple[str, ...] = (ARM_CANDIDATE, *CONTROL_ARM_KINDS)

MAX_GATE_PARAMS = 4096

VERDICT_MECHANICS_OK = "mechanics-ok"
VERDICT_ARCHITECTURE_FRAGILE = "architecture-fragile"
VERDICT_NULL = "null"


class DoaHarnessRefusal(ValueError):
    """Raised when a DoA harness input is malformed or a budget invariant would be violated."""


class DoaBudgetMismatch(DoaHarnessRefusal):
    """Raised when the candidate and a matched control do not share byte-equal inference FLOPs."""


class DoaUnchargedTraining(DoaHarnessRefusal):
    """Raised when a candidate comparison omits the amortized training cost C_train."""


class DoaCeilingExceeded(DoaHarnessRefusal):
    """Raised when an arm's full-lifecycle FLOPs exceed the matched budget ceiling."""


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DoaHarnessRefusal(f"{label} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DoaHarnessRefusal(f"{label} must be a positive integer")
    return value


def _require_mae_deg(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DoaHarnessRefusal(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 180.0:
        raise DoaHarnessRefusal(f"{label} must be a finite great-circle MAE in [0, 180] degrees")
    return number


def _seal_float(value: float) -> float:
    return round(float(value), 12)


def _require_architecture(value: str) -> str:
    if value not in ARCHITECTURES:
        raise DoaHarnessRefusal(f"architecture {value!r} is not one of {ARCHITECTURES}")
    return value


# ---------------------------------------------------------------------------
# Per-arm, per-seed results and one arm.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoaArmSeedResult:
    """One arm's result on one paired seed: the clip-macro MAE (degrees) and the re-estimation count K."""

    seed: int
    mae_deg: float
    reestimations: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.seed, "DoaArmSeedResult.seed")
        object.__setattr__(self, "mae_deg", _require_mae_deg(self.mae_deg, "DoaArmSeedResult.mae_deg"))
        _require_nonnegative_int(self.reestimations, "DoaArmSeedResult.reestimations")

    def payload(self) -> dict[str, Any]:
        return {"seed": self.seed, "mae_deg": _seal_float(self.mae_deg), "reestimations": self.reestimations}


@dataclass(frozen=True, slots=True)
class DoaArm:
    """One arm of the DoA bed: kind, architecture, trained-parameter count, FLOP model, per-seed results."""

    name: str
    kind: str
    architecture: str
    total_frames: int
    params: int
    flop_model: FlopModel
    seed_results: tuple[DoaArmSeedResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DoaHarnessRefusal("DoaArm.name must be a non-empty string")
        if self.kind not in ALL_ARM_KINDS:
            raise DoaHarnessRefusal(f"DoaArm.kind {self.kind!r} is not one of {ALL_ARM_KINDS}")
        _require_architecture(self.architecture)
        _require_positive_int(self.total_frames, "DoaArm.total_frames")
        _require_nonnegative_int(self.params, "DoaArm.params")
        if not isinstance(self.flop_model, FlopModel):
            raise DoaHarnessRefusal("DoaArm.flop_model must be a FlopModel")
        if not self.seed_results:
            raise DoaHarnessRefusal("DoaArm.seed_results must not be empty")
        object.__setattr__(self, "seed_results", tuple(self.seed_results))
        seeds = [result.seed for result in self.seed_results]
        if len(set(seeds)) != len(seeds):
            raise DoaHarnessRefusal("DoaArm.seed_results must have unique seeds")
        if seeds != sorted(seeds):
            raise DoaHarnessRefusal("DoaArm.seed_results must be in ascending seed order")
        for result in self.seed_results:
            if result.reestimations > self.total_frames:
                raise DoaHarnessRefusal("a re-estimation count cannot exceed the total frame count")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(result.seed for result in self.seed_results)

    def result_for_seed(self, seed: int) -> DoaArmSeedResult:
        for result in self.seed_results:
            if result.seed == seed:
                return result
        raise DoaHarnessRefusal(f"arm {self.name!r} has no result for seed {seed}")

    def mean_mae_deg(self) -> float:
        return math.fsum(result.mae_deg for result in self.seed_results) / len(self.seed_results)

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
            "architecture": self.architecture,
            "total_frames": self.total_frames,
            "params": self.params,
            "flop_model": self.flop_model.payload(),
            "seed_results": [result.payload() for result in self.seed_results],
            "mean_mae_deg": _seal_float(self.mean_mae_deg()),
            "mean_lifecycle_flops": _seal_float(self.mean_lifecycle_flops()),
            "max_lifecycle_flops": self.max_lifecycle_flops(),
            "mean_reestimations": _seal_float(self.mean_reestimations()),
        }


def assert_within_ceiling(arm: DoaArm, ceiling: int = FLOP_CEILING) -> None:
    """Refuse unless every paired-seed run of the arm stays under the full-lifecycle FLOP ceiling."""

    ceiling = _require_positive_int(ceiling, "ceiling")
    for result in arm.seed_results:
        total = arm.flop_model.lifecycle_flops(result.reestimations)
        if total > ceiling:
            raise DoaCeilingExceeded(
                f"arm {arm.name!r} ({arm.architecture}) seed {result.seed} lifecycle FLOPs {total} "
                f"exceed ceiling {ceiling}"
            )


def assert_matched_ex_training(candidate: DoaArm, control: DoaArm) -> None:
    """Refuse unless the candidate and the matched control share byte-equal inference FLOPs.

    Same architecture, same seeds, same total_frames; the candidate charges C_train > 0 and the control
    charges 0; per seed the re-estimation counts K match AND the run FLOPs match byte-for-byte, so any MAE
    gap is WHERE re-estimations land, not how many.
    """

    if candidate.kind != ARM_CANDIDATE:
        raise DoaHarnessRefusal("assert_matched_ex_training expects the candidate arm first")
    if candidate.architecture != control.architecture:
        raise DoaBudgetMismatch("candidate and control must share the same architecture")
    if candidate.seeds != control.seeds:
        raise DoaBudgetMismatch("candidate and control do not share the same paired seeds")
    if candidate.total_frames != control.total_frames:
        raise DoaBudgetMismatch("candidate and control cover a different number of frames")
    if candidate.flop_model.train_flops <= 0:
        raise DoaUnchargedTraining(
            "the candidate must charge its amortized training cost C_train in full-lifecycle accounting"
        )
    if control.flop_model.train_flops != 0:
        raise DoaBudgetMismatch("a control must not charge a training cost; it learns nothing")
    for seed in candidate.seeds:
        candidate_result = candidate.result_for_seed(seed)
        control_result = control.result_for_seed(seed)
        if candidate_result.reestimations != control_result.reestimations:
            raise DoaBudgetMismatch(
                f"seed {seed}: control re-estimation count {control_result.reestimations} does not match "
                f"the candidate re-estimation count {candidate_result.reestimations}"
            )
        candidate_run = candidate.flop_model.run_flops(candidate_result.reestimations)
        control_run = control.flop_model.run_flops(control_result.reestimations)
        if candidate_run != control_run:
            raise DoaBudgetMismatch(
                f"seed {seed}: control inference FLOPs {control_run} do not match the candidate "
                f"inference FLOPs {candidate_run}"
            )


def paired_deltas(candidate: DoaArm, control: DoaArm) -> tuple[float, ...]:
    """Return delta_i = MAE_control(i) - MAE_candidate(i). Positive delta = candidate LOWERS the error."""

    if candidate.seeds != control.seeds:
        raise DoaHarnessRefusal("paired deltas require the same paired seeds on both arms")
    return tuple(
        control.result_for_seed(seed).mae_deg - candidate.result_for_seed(seed).mae_deg
        for seed in candidate.seeds
    )


# ---------------------------------------------------------------------------
# Compute points and the per-architecture Pareto frontier.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoaComputePoint:
    """A single (compute, MAE) point on the sweep. Compute is reported as a set of indicators."""

    budget_id: str
    architecture: str
    arm_kind: str
    flops: float
    mae_deg: float
    params: int
    reestimations: float

    def payload(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "architecture": self.architecture,
            "arm_kind": self.arm_kind,
            "flops": _seal_float(self.flops),
            "mae_deg": _seal_float(self.mae_deg),
            "params": self.params,
            "reestimations": _seal_float(self.reestimations),
        }


def _compute_point(budget_id: str, arm: DoaArm) -> DoaComputePoint:
    return DoaComputePoint(
        budget_id=budget_id,
        architecture=arm.architecture,
        arm_kind=arm.kind,
        flops=arm.mean_lifecycle_flops(),
        mae_deg=arm.mean_mae_deg(),
        params=arm.params,
        reestimations=arm.mean_reestimations(),
    )


def pareto_frontier(points: Sequence[DoaComputePoint]) -> tuple[DoaComputePoint, ...]:
    """Return the non-dominated set: minimize FLOPs AND minimize MAE (both lower is better)."""

    unique = list(points)
    frontier: list[DoaComputePoint] = []
    for candidate in unique:
        dominated = False
        for other in unique:
            if other is candidate:
                continue
            no_worse = other.flops <= candidate.flops and other.mae_deg <= candidate.mae_deg
            strictly_better = other.flops < candidate.flops or other.mae_deg < candidate.mae_deg
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier.sort(key=lambda point: (point.flops, point.mae_deg))
    return tuple(frontier)


# ---------------------------------------------------------------------------
# One budget point: the candidate plus its three controls, at one architecture, one re-estimation budget.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoaBudgetPoint:
    """One re-estimation-budget operating point for ONE architecture: candidate plus its three controls."""

    budget_id: str
    architecture: str
    candidate: DoaArm
    rate_matched_random: DoaArm
    always_on: DoaArm
    never_update: DoaArm

    def __post_init__(self) -> None:
        if not isinstance(self.budget_id, str) or not self.budget_id.strip():
            raise DoaHarnessRefusal("DoaBudgetPoint.budget_id must be a non-empty string")
        _require_architecture(self.architecture)
        for arm, expected in (
            (self.candidate, ARM_CANDIDATE),
            (self.rate_matched_random, ARM_RATE_MATCHED_RANDOM),
            (self.always_on, ARM_ALWAYS_ON),
            (self.never_update, ARM_NEVER_UPDATE),
        ):
            if not isinstance(arm, DoaArm):
                raise DoaHarnessRefusal("DoaBudgetPoint arms must be DoaArm instances")
            if arm.kind != expected:
                raise DoaHarnessRefusal(f"DoaBudgetPoint arm {arm.name!r} must have kind {expected!r}")
            if arm.architecture != self.architecture:
                raise DoaHarnessRefusal(f"DoaBudgetPoint arm {arm.name!r} architecture must match the point")
        seeds = self.candidate.seeds
        for arm in self.arms():
            if arm.seeds != seeds:
                raise DoaHarnessRefusal("all arms in a budget point must share the same paired seeds")
            if arm.total_frames != self.candidate.total_frames:
                raise DoaHarnessRefusal("all arms in a budget point must cover the same frame count")

    def arms(self) -> tuple[DoaArm, ...]:
        return (self.candidate, self.rate_matched_random, self.always_on, self.never_update)

    def certify(self, ceiling: int = FLOP_CEILING) -> None:
        for arm in self.arms():
            assert_within_ceiling(arm, ceiling)
            if arm.kind in CONTROL_ARM_KINDS and arm.flop_model.train_flops != 0:
                raise DoaBudgetMismatch(f"control arm {arm.name!r} must not charge a training cost")
        assert_matched_ex_training(self.candidate, self.rate_matched_random)

    def candidate_beats_rate_matched_random(self) -> bool:
        """Lower MAE is better, so the candidate wins iff its mean MAE is strictly below the control's."""

        return self.candidate.mean_mae_deg() < self.rate_matched_random.mean_mae_deg()


# ---------------------------------------------------------------------------
# The per-architecture report and the dual-architecture assembly.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DoaArchitectureReport:
    """The matched-budget analysis for ONE architecture.

    A coarse dominance signal, not the full survive rule.
    """

    schema: str
    architecture: str
    source_kind: str
    flop_ceiling: int
    seeds: tuple[int, ...]
    arm_summaries: tuple[dict[str, Any], ...]
    pareto: tuple[DoaComputePoint, ...]
    per_budget_candidate_vs_rate_matched_random: tuple[dict[str, Any], ...]
    candidate_strictly_dominates_rate_matched_random: bool
    break_even: BreakEven
    matched_budget: MatchedBudget
    dominance_verdict: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "architecture": self.architecture,
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
            "dominance_verdict": self.dominance_verdict,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _matched_budget_from_points(
    budget_points: Sequence[DoaBudgetPoint], wall_ns: int, ceiling: int
) -> MatchedBudget:
    wall_ns = _require_positive_int(wall_ns, "wall_ns")
    candidate = budget_points[0].candidate
    binding_flops = max(point.candidate.max_lifecycle_flops() for point in budget_points)
    if binding_flops > ceiling:
        raise DoaCeilingExceeded(f"binding candidate FLOPs {binding_flops} exceed ceiling {ceiling}")
    return MatchedBudget(
        params=candidate.params, flops=binding_flops, wall_ns=wall_ns, seeds=len(candidate.seeds)
    )


def run_matched_budget_one_architecture(
    budget_points: Sequence[DoaBudgetPoint],
    *,
    architecture: str,
    wall_ns: int,
    operating_budget_id: str | None = None,
    source_kind: str = "real",
    ceiling: int = FLOP_CEILING,
) -> DoaArchitectureReport:
    """Run the matched-budget analysis over a re-estimation-budget sweep, for one fixed architecture."""

    points = list(budget_points)
    if not points:
        raise DoaHarnessRefusal("run_matched_budget_one_architecture needs at least one budget point")
    if source_kind not in ("synthetic", "real"):
        raise DoaHarnessRefusal("source_kind must be 'synthetic' or 'real'")
    _require_architecture(architecture)
    for point in points:
        if point.architecture != architecture:
            raise DoaHarnessRefusal(
                f"budget point {point.budget_id!r} does not match architecture {architecture!r}"
            )

    seeds = points[0].candidate.seeds
    for point in points:
        if point.candidate.seeds != seeds:
            raise DoaHarnessRefusal("all budget points must share the same paired seeds")
        point.certify(ceiling)

    budget_ids = [point.budget_id for point in points]
    if len(set(budget_ids)) != len(budget_ids):
        raise DoaHarnessRefusal("budget point ids must be unique")

    compute_points: list[DoaComputePoint] = []
    per_budget_rows: list[dict[str, Any]] = []
    dominates_everywhere = True
    for point in points:
        for arm in point.arms():
            compute_points.append(_compute_point(point.budget_id, arm))
        candidate_mae = point.candidate.mean_mae_deg()
        rmr_mae = point.rate_matched_random.mean_mae_deg()
        wins = candidate_mae < rmr_mae
        dominates_everywhere = dominates_everywhere and wins
        per_budget_rows.append(
            {
                "budget_id": point.budget_id,
                "candidate_mean_mae_deg": _seal_float(candidate_mae),
                "rate_matched_random_mean_mae_deg": _seal_float(rmr_mae),
                "delta_mean_mae_deg_control_minus_candidate": _seal_float(rmr_mae - candidate_mae),
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
            raise DoaHarnessRefusal(f"operating_budget_id {operating_budget_id!r} is not in the sweep")
        operating = matches[0]
    operating_saving = per_query_saving_vs_always_on(
        operating.candidate.total_frames,
        round(operating.candidate.mean_reestimations()),
        operating.candidate.flop_model.downstream_flops_per_firing,
    )
    break_even = break_even_queries(operating.candidate.flop_model.train_flops, operating_saving)

    matched_budget = _matched_budget_from_points(points, wall_ns=wall_ns, ceiling=ceiling)
    dominance_verdict = VERDICT_MECHANICS_OK if dominates_everywhere else VERDICT_NULL

    arm_summaries = tuple(arm.payload() for point in points for arm in point.arms())
    return DoaArchitectureReport(
        schema=DOA_HARNESS_SCHEMA,
        architecture=architecture,
        source_kind=source_kind,
        flop_ceiling=ceiling,
        seeds=seeds,
        arm_summaries=arm_summaries,
        pareto=pareto_frontier(compute_points),
        per_budget_candidate_vs_rate_matched_random=tuple(per_budget_rows),
        candidate_strictly_dominates_rate_matched_random=dominates_everywhere,
        break_even=break_even,
        matched_budget=matched_budget,
        dominance_verdict=dominance_verdict,
    )


@dataclass(frozen=True, slots=True)
class DoaHarnessReport:
    """The dual-architecture matched-budget analysis. Mechanics-only: it can never clear a stage gate."""

    schema: str
    bed_id: str
    source_kind: str
    flop_ceiling: int
    per_architecture: dict[str, DoaArchitectureReport]
    candidate_strictly_dominates_rate_matched_random_arch_a: bool
    candidate_strictly_dominates_rate_matched_random_arch_b: bool
    both_architectures_dominate: bool
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
            "per_architecture": {arch: report.payload() for arch, report in self.per_architecture.items()},
            "candidate_strictly_dominates_rate_matched_random_arch_a": (
                self.candidate_strictly_dominates_rate_matched_random_arch_a
            ),
            "candidate_strictly_dominates_rate_matched_random_arch_b": (
                self.candidate_strictly_dominates_rate_matched_random_arch_b
            ),
            "both_architectures_dominate": self.both_architectures_dominate,
            "verdict": self.verdict,
            "activation_allowed": self.activation_allowed,
            "scientific_promotion": self.scientific_promotion,
            "independent_scientific_confirmation": self.independent_scientific_confirmation,
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def run_matched_budget_dual_architecture(
    budget_points_a: Sequence[DoaBudgetPoint],
    budget_points_b: Sequence[DoaBudgetPoint],
    *,
    wall_ns: int,
    source_kind: str = "real",
    ceiling: int = FLOP_CEILING,
    operating_budget_id_a: str | None = None,
    operating_budget_id_b: str | None = None,
) -> DoaHarnessReport:
    """Run the matched-budget certify + Pareto machinery independently per architecture, then fold.

    The ceiling applies PER ARM PER SEED PER ARCHITECTURE, the existing "per-arm total FLOPs" contract, not
    a combined budget across architectures. ``verdict`` here is the coarse dominance-only fold
    (mechanics-ok / architecture-fragile / null); the full SURVIVES(X) statistical conjunction (SESOI,
    clip-level sign-flip, room-majority) is computed by ``doa_producer.py`` on top of this.
    """

    report_a = run_matched_budget_one_architecture(
        budget_points_a,
        architecture=ARCH_A_ID,
        wall_ns=wall_ns,
        operating_budget_id=operating_budget_id_a,
        source_kind=source_kind,
        ceiling=ceiling,
    )
    report_b = run_matched_budget_one_architecture(
        budget_points_b,
        architecture=ARCH_B_ID,
        wall_ns=wall_ns,
        operating_budget_id=operating_budget_id_b,
        source_kind=source_kind,
        ceiling=ceiling,
    )
    dominates_a = report_a.candidate_strictly_dominates_rate_matched_random
    dominates_b = report_b.candidate_strictly_dominates_rate_matched_random
    both = dominates_a and dominates_b
    if both:
        verdict = VERDICT_MECHANICS_OK
    elif dominates_a or dominates_b:
        verdict = VERDICT_ARCHITECTURE_FRAGILE
    else:
        verdict = VERDICT_NULL

    return DoaHarnessReport(
        schema=DOA_HARNESS_SCHEMA,
        bed_id=DOA_BED_ID,
        source_kind=source_kind,
        flop_ceiling=ceiling,
        per_architecture={ARCH_A_ID: report_a, ARCH_B_ID: report_b},
        candidate_strictly_dominates_rate_matched_random_arch_a=dominates_a,
        candidate_strictly_dominates_rate_matched_random_arch_b=dominates_b,
        both_architectures_dominate=both,
        verdict=verdict,
        activation_allowed=False,
        scientific_promotion=False,
        independent_scientific_confirmation=False,
        claim_scope=CLAIM_SCOPE,
    )
