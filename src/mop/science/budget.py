"""One matched-budget engine for experiment metrics, controls, and architecture folds."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mop.ladder.stage_ladder import MatchedBudget
from mop.substrate.events import canonical_sha256

ARM_CANDIDATE = "candidate"
ARM_RATE_MATCHED_RANDOM = "rate_matched_random"
ARM_ALWAYS_ON = "always_on"
ARM_BEST_SINGLE = "best_single"
ARM_NEVER_UPDATE = "never_update"
TRAIN_BACKWARD_MULTIPLIER = 3


class BudgetRefusal(ValueError):
    """A matched-budget input or declared policy is malformed."""


class BudgetMismatch(BudgetRefusal):
    """Candidate and matched control inference budgets differ."""


class UnchargedTraining(BudgetRefusal):
    """A trained candidate omitted its amortized training charge."""


class CeilingExceeded(BudgetRefusal):
    """An arm exceeds the declared full-lifecycle FLOP ceiling."""


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BudgetRefusal(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BudgetRefusal(f"{label} must be a positive integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BudgetRefusal(f"{label} must be a finite real number")
    return float(value)


def _sealed(value: float) -> float:
    return round(float(value), 12)


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """The complete experiment-specific data consumed by the generic budget lifecycle."""

    schema: str
    bed_id: str
    metric_key: str
    action_key: str
    direction: str
    reference_kind: str
    claim_scope: str
    flop_ceiling: int
    metric_min: float = 0.0
    metric_max: float | None = None
    architectures: tuple[str, ...] = ()
    include_mean_actions: bool = True
    delta_key: str = ""

    def __post_init__(self) -> None:
        for field in ("schema", "bed_id", "metric_key", "action_key", "reference_kind", "claim_scope"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise BudgetRefusal(f"BudgetPolicy.{field} must be a non-empty string")
        if self.direction not in ("higher", "lower"):
            raise BudgetRefusal("BudgetPolicy.direction must be higher or lower")
        if self.reference_kind not in (ARM_BEST_SINGLE, ARM_NEVER_UPDATE):
            raise BudgetRefusal("BudgetPolicy.reference_kind is not a supported control")
        _positive_int(self.flop_ceiling, "BudgetPolicy.flop_ceiling")
        if self.metric_max is not None and self.metric_max < self.metric_min:
            raise BudgetRefusal("BudgetPolicy metric bounds are reversed")
        if len(set(self.architectures)) != len(self.architectures):
            raise BudgetRefusal("BudgetPolicy.architectures must be unique")

    @property
    def controls(self) -> tuple[str, ...]:
        return (ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, self.reference_kind)

    @property
    def all_arms(self) -> tuple[str, ...]:
        return (ARM_CANDIDATE, *self.controls)

    def validate_metric(self, value: object) -> float:
        number = _finite(value, self.metric_key)
        if number < self.metric_min or (self.metric_max is not None and number > self.metric_max):
            interval = f"[{self.metric_min}, {self.metric_max}]"
            raise BudgetRefusal(f"{self.metric_key} must be in {interval}")
        return number

    def improvement(self, candidate: float, control: float) -> float:
        return candidate - control if self.direction == "higher" else control - candidate


@dataclass(frozen=True, slots=True)
class FlopModel:
    featurize_flops: int
    gate_infer_flops: int
    downstream_flops_per_firing: int
    train_flops: int = 0

    def __post_init__(self) -> None:
        for field in (
            "featurize_flops", "gate_infer_flops", "downstream_flops_per_firing", "train_flops"
        ):
            _nonnegative_int(getattr(self, field), f"FlopModel.{field}")

    def run_flops(self, actions: int) -> int:
        return (self.featurize_flops + self.gate_infer_flops
                + _nonnegative_int(actions, "actions") * self.downstream_flops_per_firing)

    def lifecycle_flops(self, actions: int) -> int:
        return self.run_flops(actions) + self.train_flops

    def payload(self) -> dict[str, int]:
        return {"featurize_flops": self.featurize_flops, "gate_infer_flops": self.gate_infer_flops,
                "downstream_flops_per_firing": self.downstream_flops_per_firing,
                "train_flops": self.train_flops}


def arm_flop_model(
    kind: str,
    total_frames: int,
    *,
    featurize_per_frame: int,
    gate_infer_per_frame: int,
    downstream_flops_per_firing: int,
    candidate_train_flops: Callable[[], int],
) -> FlopModel:
    """Project provider-specific costs onto the conventional candidate and control arms."""

    runs_gate = kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM)
    return FlopModel(
        featurize_flops=featurize_per_frame * total_frames,
        gate_infer_flops=gate_infer_per_frame * total_frames if runs_gate else 0,
        downstream_flops_per_firing=downstream_flops_per_firing,
        train_flops=candidate_train_flops() if kind == ARM_CANDIDATE else 0,
    )


@dataclass(frozen=True, slots=True)
class SeedResult:
    seed: int
    metric_value: float
    actions: int

    def __post_init__(self) -> None:
        _nonnegative_int(self.seed, "SeedResult.seed")
        object.__setattr__(self, "metric_value", _finite(self.metric_value, "SeedResult.metric_value"))
        _nonnegative_int(self.actions, "SeedResult.actions")

    def payload(self, policy: BudgetPolicy) -> dict[str, Any]:
        return {"seed": self.seed, policy.metric_key: _sealed(self.metric_value),
                policy.action_key: self.actions}


@dataclass(frozen=True, slots=True)
class BudgetSeedRun:
    """The conventional per-seed producer record consumed by budget-point projection."""

    seed: int
    total_frames: int
    train_frames: int
    gate_params: int
    per_budget: dict[str, dict[str, Any]]
    operating_budget_id: str
    per_seed_block: dict[str, Any]
    noisy_tv: dict[str, Any]


def noise_control_summary(
    policy: BudgetPolicy,
    seed_runs: Sequence[Any],
    *,
    at_chance: bool,
    mean_noise_rate: float,
    mean_base_rate: float,
    rate_key: str,
) -> dict[str, Any]:
    """Project the shared noisy-TV and control-arm audit block."""

    if rate_key not in ("mean_firing_rate_on_noise", "mean_reestimate_rate_on_noise"):
        raise BudgetRefusal("noise-control rate_key is not declared")
    return {
        "noisy_tv_at_chance": bool(at_chance),
        rate_key: _sealed(mean_noise_rate),
        "mean_base_rate": _sealed(mean_base_rate),
        "per_seed_noisy_tv": [run.noisy_tv for run in seed_runs],
        "primary_control": ARM_RATE_MATCHED_RANDOM,
        "control_arms": [*policy.controls, "noisy_tv"],
    }


@dataclass(frozen=True, slots=True)
class Arm:
    policy: BudgetPolicy
    name: str
    kind: str
    total_frames: int
    params: int
    flop_model: FlopModel
    seed_results: tuple[SeedResult, ...]
    architecture: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy, BudgetPolicy):
            raise BudgetRefusal("Arm.policy must be a BudgetPolicy")
        if not isinstance(self.name, str) or not self.name.strip():
            raise BudgetRefusal("Arm.name must be a non-empty string")
        if self.kind not in self.policy.all_arms:
            raise BudgetRefusal(f"Arm.kind {self.kind!r} is not one of {self.policy.all_arms}")
        _positive_int(self.total_frames, "Arm.total_frames")
        _nonnegative_int(self.params, "Arm.params")
        if not isinstance(self.flop_model, FlopModel):
            raise BudgetRefusal("Arm.flop_model must be a FlopModel")
        if self.policy.architectures:
            if self.architecture not in self.policy.architectures:
                raise BudgetRefusal(f"architecture {self.architecture!r} is not declared")
        elif self.architecture is not None:
            raise BudgetRefusal("an architecture is not valid for this policy")
        if not self.seed_results:
            raise BudgetRefusal("Arm.seed_results must not be empty")
        object.__setattr__(self, "seed_results", tuple(self.seed_results))
        seeds = [result.seed for result in self.seed_results]
        if len(set(seeds)) != len(seeds) or seeds != sorted(seeds):
            raise BudgetRefusal("Arm.seed_results must have unique ascending seeds")
        for result in self.seed_results:
            self.policy.validate_metric(result.metric_value)
            if result.actions > self.total_frames:
                raise BudgetRefusal("an action count cannot exceed the total frame count")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(result.seed for result in self.seed_results)

    def result_for_seed(self, seed: int) -> SeedResult:
        for result in self.seed_results:
            if result.seed == seed:
                return result
        raise BudgetRefusal(f"arm {self.name!r} has no result for seed {seed}")

    def mean_metric(self) -> float:
        return math.fsum(result.metric_value for result in self.seed_results) / len(self.seed_results)

    def mean_lifecycle_flops(self) -> float:
        values = [self.flop_model.lifecycle_flops(result.actions) for result in self.seed_results]
        return math.fsum(values) / len(values)

    def max_lifecycle_flops(self) -> int:
        return max(self.flop_model.lifecycle_flops(result.actions) for result in self.seed_results)

    def mean_actions(self) -> float:
        return math.fsum(result.actions for result in self.seed_results) / len(self.seed_results)

    def payload(self) -> dict[str, Any]:
        result = {"name": self.name, "kind": self.kind}
        if self.architecture is not None:
            result["architecture"] = self.architecture
        result.update({"total_frames": self.total_frames, "params": self.params,
                       "flop_model": self.flop_model.payload(),
                       "seed_results": [row.payload(self.policy) for row in self.seed_results],
                       f"mean_{self.policy.metric_key}": _sealed(self.mean_metric()),
                       "mean_lifecycle_flops": _sealed(self.mean_lifecycle_flops()),
                       "max_lifecycle_flops": self.max_lifecycle_flops()})
        if self.policy.include_mean_actions:
            result[f"mean_{self.policy.action_key}"] = _sealed(self.mean_actions())
        return result


def assert_within_ceiling(arm: Arm, ceiling: int | None = None) -> None:
    limit = arm.policy.flop_ceiling if ceiling is None else _positive_int(ceiling, "ceiling")
    for result in arm.seed_results:
        total = arm.flop_model.lifecycle_flops(result.actions)
        if total > limit:
            suffix = f" ({arm.architecture})" if arm.architecture else ""
            raise CeilingExceeded(
                f"arm {arm.name!r}{suffix} seed {result.seed} lifecycle FLOPs {total} exceed ceiling {limit}"
            )


def assert_matched_ex_training(candidate: Arm, control: Arm) -> None:
    if candidate.policy != control.policy:
        raise BudgetMismatch("candidate and control use different budget policies")
    if candidate.kind != ARM_CANDIDATE:
        raise BudgetRefusal("assert_matched_ex_training expects the candidate arm first")
    if candidate.architecture != control.architecture:
        raise BudgetMismatch("candidate and control must share the same architecture")
    if candidate.seeds != control.seeds:
        raise BudgetMismatch("candidate and control do not share the same paired seeds")
    if candidate.total_frames != control.total_frames:
        raise BudgetMismatch("candidate and control cover a different number of frames")
    if candidate.flop_model.train_flops <= 0:
        raise UnchargedTraining("the candidate must charge its amortized training cost C_train")
    if control.flop_model.train_flops != 0:
        raise BudgetMismatch("a control must not charge a training cost; it learns nothing")
    for seed in candidate.seeds:
        candidate_result = candidate.result_for_seed(seed)
        control_result = control.result_for_seed(seed)
        if candidate_result.actions != control_result.actions:
            raise BudgetMismatch(f"seed {seed}: matched control action count differs from the candidate")
        if (candidate.flop_model.run_flops(candidate_result.actions)
                != control.flop_model.run_flops(control_result.actions)):
            raise BudgetMismatch(f"seed {seed}: matched control inference FLOPs differ from the candidate")


def paired_deltas(candidate: Arm, control: Arm) -> tuple[float, ...]:
    if candidate.policy != control.policy or candidate.seeds != control.seeds:
        raise BudgetRefusal("paired deltas require one policy and the same paired seeds")
    return tuple(candidate.policy.improvement(
        candidate.result_for_seed(seed).metric_value, control.result_for_seed(seed).metric_value,
    ) for seed in candidate.seeds)


def featurize_run_flops(total_frames: int, flops_per_frame: int) -> int:
    return _nonnegative_int(total_frames, "total_frames") * _nonnegative_int(
        flops_per_frame, "flops_per_frame"
    )


def gate_infer_run_flops(total_frames: int, flops_per_frame: int) -> int:
    return featurize_run_flops(total_frames, flops_per_frame)


def gate_train_flops(epochs: int, train_frames: int, infer_flops_per_frame: int) -> int:
    return (_positive_int(epochs, "epochs") * _positive_int(train_frames, "train_frames")
            * TRAIN_BACKWARD_MULTIPLIER
            * _positive_int(infer_flops_per_frame, "infer_flops_per_frame"))


def per_query_saving_vs_always_on(
    total_frames: int, actions: int, downstream_flops_per_action: int,
) -> float:
    frames = _positive_int(total_frames, "total_frames")
    action_count = _nonnegative_int(actions, "actions")
    if action_count > frames:
        raise BudgetRefusal("actions cannot exceed the total frame count")
    downstream = _nonnegative_int(downstream_flops_per_action, "downstream_flops_per_action")
    return (frames - action_count) / frames * downstream


@dataclass(frozen=True, slots=True)
class BreakEven:
    train_flops: int
    per_query_saving: float
    n_star_frames: int | None
    amortizable: bool

    def payload(self) -> dict[str, Any]:
        return {"train_flops": self.train_flops, "per_query_saving": _sealed(self.per_query_saving),
                "n_star_frames": self.n_star_frames, "amortizable": self.amortizable}


def break_even_queries(train_flops: int, per_query_saving: float) -> BreakEven:
    training = _nonnegative_int(train_flops, "train_flops")
    saving = _finite(per_query_saving, "per_query_saving")
    if saving <= 0.0:
        return BreakEven(training, saving, None, False)
    return BreakEven(training, saving, math.ceil(training / saving), True)


@dataclass(frozen=True, slots=True)
class BudgetPoint:
    policy: BudgetPolicy
    budget_id: str
    candidate: Arm
    rate_matched_random: Arm
    always_on: Arm
    reference: Arm
    architecture: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.budget_id, str) or not self.budget_id.strip():
            raise BudgetRefusal("BudgetPoint.budget_id must be a non-empty string")
        if self.architecture not in ((None,) if not self.policy.architectures else self.policy.architectures):
            raise BudgetRefusal(f"architecture {self.architecture!r} is not declared")
        expected = (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, self.policy.reference_kind)
        for arm, kind in zip(self.arms(), expected, strict=True):
            if not isinstance(arm, Arm) or arm.policy != self.policy:
                raise BudgetRefusal("BudgetPoint arms must use its declared policy")
            if arm.kind != kind:
                raise BudgetRefusal(f"BudgetPoint arm {arm.name!r} must have kind {kind!r}")
            if arm.architecture != self.architecture:
                raise BudgetRefusal("BudgetPoint arm architecture must match the point")
            if arm.seeds != self.candidate.seeds or arm.total_frames != self.candidate.total_frames:
                raise BudgetRefusal("all arms must share paired seeds and total frame count")

    def arms(self) -> tuple[Arm, ...]:
        return self.candidate, self.rate_matched_random, self.always_on, self.reference

    def certify(self, ceiling: int | None = None) -> None:
        for arm in self.arms():
            assert_within_ceiling(arm, ceiling)
            if arm.kind in self.policy.controls and arm.flop_model.train_flops != 0:
                raise BudgetMismatch(f"control arm {arm.name!r} must not charge a training cost")
        assert_matched_ex_training(self.candidate, self.rate_matched_random)

    def candidate_beats_rate_matched_random(self) -> bool:
        return self.policy.improvement(
            self.candidate.mean_metric(), self.rate_matched_random.mean_metric()
        ) > 0


def build_budget_points(
    policy: BudgetPolicy,
    seed_runs: Sequence[Any],
    *,
    score_group: str,
    score_field: str,
    action_group: str,
    flop_model: Callable[[str], FlopModel],
    architecture: str | None = None,
) -> list[BudgetPoint]:
    """Project conventional per-seed producer records into one certified budget sweep."""

    runs = list(seed_runs)
    if not runs:
        raise BudgetRefusal("budget-point projection needs at least one seed run")
    first = runs[0]
    points = []
    for budget_id in first.per_budget:
        arms = {}
        for kind in policy.all_arms:
            results = tuple(
                SeedResult(
                    run.seed,
                    run.per_budget[budget_id][score_group][kind][score_field],
                    run.per_budget[budget_id][action_group][kind],
                )
                for run in runs
            )
            arm_name = f"{kind}@{architecture}@{budget_id}" if architecture else f"{kind}@{budget_id}"
            arms[kind] = Arm(
                policy, arm_name, kind, first.total_frames,
                first.gate_params if kind == ARM_CANDIDATE else 0,
                flop_model(kind), results, architecture,
            )
        points.append(BudgetPoint(
            policy, budget_id, arms[ARM_CANDIDATE], arms[ARM_RATE_MATCHED_RANDOM],
            arms[ARM_ALWAYS_ON], arms[policy.reference_kind], architecture,
        ))
    return points


@dataclass(frozen=True, slots=True)
class ComputePoint:
    policy: BudgetPolicy
    budget_id: str
    arm_kind: str
    flops: float
    metric_value: float
    params: int
    actions: float
    architecture: str | None = None

    def payload(self) -> dict[str, Any]:
        result = {"budget_id": self.budget_id}
        if self.architecture is not None:
            result["architecture"] = self.architecture
        result.update({"arm_kind": self.arm_kind, "flops": _sealed(self.flops),
                       self.policy.metric_key: _sealed(self.metric_value), "params": self.params,
                       self.policy.action_key: _sealed(self.actions)})
        return result


def _compute_point(point: BudgetPoint, arm: Arm) -> ComputePoint:
    return ComputePoint(point.policy, point.budget_id, arm.kind, arm.mean_lifecycle_flops(),
                        arm.mean_metric(), arm.params, arm.mean_actions(), point.architecture)


def pareto_frontier(points: Sequence[ComputePoint]) -> tuple[ComputePoint, ...]:
    candidates = list(points)
    frontier = []
    for candidate in candidates:
        direction = candidate.policy.direction
        for other in candidates:
            metric_no_worse = (other.metric_value >= candidate.metric_value if direction == "higher"
                               else other.metric_value <= candidate.metric_value)
            metric_better = (other.metric_value > candidate.metric_value if direction == "higher"
                             else other.metric_value < candidate.metric_value)
            if other is not candidate and other.flops <= candidate.flops and metric_no_worse and (
                other.flops < candidate.flops or metric_better
            ):
                break
        else:
            frontier.append(candidate)
    reverse_metric = -1 if points and points[0].policy.direction == "higher" else 1
    frontier.sort(key=lambda row: (row.flops, reverse_metric * row.metric_value))
    return tuple(frontier)


@dataclass(frozen=True, slots=True)
class BudgetReport:
    policy: BudgetPolicy
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
    activation_allowed: bool = False
    scientific_promotion: bool = False
    independent_scientific_confirmation: bool = False

    def payload(self) -> dict[str, Any]:
        return {"schema": self.policy.schema, "bed_id": self.policy.bed_id,
                "source_kind": self.source_kind, "flop_ceiling": self.flop_ceiling,
                "seeds": list(self.seeds), "arm_summaries": [dict(row) for row in self.arm_summaries],
                "pareto": [row.payload() for row in self.pareto],
                "per_budget_candidate_vs_rate_matched_random": [
                    dict(row) for row in self.per_budget_candidate_vs_rate_matched_random
                ], "candidate_strictly_dominates_rate_matched_random": (
                    self.candidate_strictly_dominates_rate_matched_random
                ), "break_even": self.break_even.payload(), "matched_budget": self.matched_budget.payload(),
                "verdict": self.verdict, "activation_allowed": self.activation_allowed,
                "scientific_promotion": self.scientific_promotion,
                "independent_scientific_confirmation": self.independent_scientific_confirmation,
                "claim_scope": self.policy.claim_scope}

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ArchitectureReport:
    policy: BudgetPolicy
    architecture: str
    source_kind: str
    flop_ceiling: int
    seeds: tuple[int, ...]
    arm_summaries: tuple[dict[str, Any], ...]
    pareto: tuple[ComputePoint, ...]
    per_budget_candidate_vs_rate_matched_random: tuple[dict[str, Any], ...]
    candidate_strictly_dominates_rate_matched_random: bool
    break_even: BreakEven
    matched_budget: MatchedBudget
    dominance_verdict: str

    def payload(self) -> dict[str, Any]:
        return {"schema": self.policy.schema, "architecture": self.architecture,
                "source_kind": self.source_kind, "flop_ceiling": self.flop_ceiling,
                "seeds": list(self.seeds), "arm_summaries": [dict(row) for row in self.arm_summaries],
                "pareto": [row.payload() for row in self.pareto],
                "per_budget_candidate_vs_rate_matched_random": [
                    dict(row) for row in self.per_budget_candidate_vs_rate_matched_random
                ], "candidate_strictly_dominates_rate_matched_random": (
                    self.candidate_strictly_dominates_rate_matched_random
                ), "break_even": self.break_even.payload(), "matched_budget": self.matched_budget.payload(),
                "dominance_verdict": self.dominance_verdict}

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _analyze(
    budget_points: Sequence[BudgetPoint], *, wall_ns: int, source_kind: str,
    ceiling: int | None, operating_budget_id: str | None,
) -> tuple[BudgetPolicy, int, tuple[int, ...], tuple[dict[str, Any], ...], tuple[ComputePoint, ...],
           tuple[dict[str, Any], ...], bool, BreakEven, MatchedBudget]:
    points = list(budget_points)
    if not points:
        raise BudgetRefusal("matched-budget analysis needs at least one budget point")
    if source_kind not in ("synthetic", "real"):
        raise BudgetRefusal("source_kind must be 'synthetic' or 'real'")
    policy = points[0].policy
    limit = policy.flop_ceiling if ceiling is None else _positive_int(ceiling, "ceiling")
    seeds = points[0].candidate.seeds
    ids = []
    compute_points = []
    rows = []
    dominates = True
    for point in points:
        if point.policy != policy or point.candidate.seeds != seeds:
            raise BudgetRefusal("all budget points must share one policy and paired seeds")
        point.certify(limit)
        ids.append(point.budget_id)
        compute_points.extend(_compute_point(point, arm) for arm in point.arms())
        candidate = point.candidate.mean_metric()
        control = point.rate_matched_random.mean_metric()
        delta = policy.improvement(candidate, control)
        wins = delta > 0
        dominates = dominates and wins
        delta_key = policy.delta_key or f"delta_mean_{policy.metric_key}"
        rows.append({"budget_id": point.budget_id,
                     f"candidate_mean_{policy.metric_key}": _sealed(candidate),
                     f"rate_matched_random_mean_{policy.metric_key}": _sealed(control),
                     delta_key: _sealed(delta),
                     "matched_inference_flops": point.candidate.flop_model.run_flops(
                         point.candidate.result_for_seed(seeds[0]).actions
                     ), "candidate_strictly_beats_rate_matched_random": wins})
    if len(set(ids)) != len(ids):
        raise BudgetRefusal("budget point ids must be unique")
    operating = points[0]
    if operating_budget_id is not None:
        selected = [point for point in points if point.budget_id == operating_budget_id]
        if not selected:
            raise BudgetRefusal(f"operating_budget_id {operating_budget_id!r} is not in the sweep")
        operating = selected[0]
    saving = per_query_saving_vs_always_on(
        operating.candidate.total_frames, round(operating.candidate.mean_actions()),
        operating.candidate.flop_model.downstream_flops_per_firing,
    )
    break_even = break_even_queries(operating.candidate.flop_model.train_flops, saving)
    binding = max(point.candidate.max_lifecycle_flops() for point in points)
    if binding > limit:
        raise CeilingExceeded(f"binding candidate FLOPs {binding} exceed ceiling {limit}")
    matched = MatchedBudget(
        points[0].candidate.params, binding, _positive_int(wall_ns, "wall_ns"), len(seeds)
    )
    summaries = tuple(arm.payload() for point in points for arm in point.arms())
    return (policy, limit, seeds, summaries, pareto_frontier(compute_points), tuple(rows), dominates,
            break_even, matched)


def run_matched_budget(
    budget_points: Sequence[BudgetPoint], *, wall_ns: int, operating_budget_id: str | None = None,
    source_kind: str = "synthetic", ceiling: int | None = None,
) -> BudgetReport:
    values = _analyze(budget_points, wall_ns=wall_ns, source_kind=source_kind, ceiling=ceiling,
                      operating_budget_id=operating_budget_id)
    policy, limit, seeds, summaries, pareto, rows, dominates, break_even, matched = values
    return BudgetReport(policy, source_kind, limit, seeds, summaries, pareto, rows, dominates,
                        break_even, matched, "mechanics-ok" if dominates else "null")


def _run_architecture(
    points: Sequence[BudgetPoint], architecture: str, *, wall_ns: int, source_kind: str,
    ceiling: int | None, operating_budget_id: str | None,
) -> ArchitectureReport:
    if not points or any(point.architecture != architecture for point in points):
        raise BudgetRefusal(f"every budget point must declare architecture {architecture!r}")
    values = _analyze(points, wall_ns=wall_ns, source_kind=source_kind, ceiling=ceiling,
                      operating_budget_id=operating_budget_id)
    policy, limit, seeds, summaries, pareto, rows, dominates, break_even, matched = values
    return ArchitectureReport(policy, architecture, source_kind, limit, seeds, summaries, pareto, rows,
                              dominates, break_even, matched, "mechanics-ok" if dominates else "null")


@dataclass(frozen=True, slots=True)
class DualBudgetReport:
    policy: BudgetPolicy
    source_kind: str
    flop_ceiling: int
    per_architecture: dict[str, ArchitectureReport]
    candidate_strictly_dominates_rate_matched_random_arch_a: bool
    candidate_strictly_dominates_rate_matched_random_arch_b: bool
    both_architectures_dominate: bool
    verdict: str
    activation_allowed: bool = False
    scientific_promotion: bool = False
    independent_scientific_confirmation: bool = False

    def payload(self) -> dict[str, Any]:
        return {"schema": self.policy.schema, "bed_id": self.policy.bed_id,
                "source_kind": self.source_kind, "flop_ceiling": self.flop_ceiling,
                "per_architecture": {key: value.payload() for key, value in self.per_architecture.items()},
                "candidate_strictly_dominates_rate_matched_random_arch_a": (
                    self.candidate_strictly_dominates_rate_matched_random_arch_a
                ), "candidate_strictly_dominates_rate_matched_random_arch_b": (
                    self.candidate_strictly_dominates_rate_matched_random_arch_b
                ), "both_architectures_dominate": self.both_architectures_dominate,
                "verdict": self.verdict, "activation_allowed": self.activation_allowed,
                "scientific_promotion": self.scientific_promotion,
                "independent_scientific_confirmation": self.independent_scientific_confirmation,
                "claim_scope": self.policy.claim_scope}

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def run_dual_architecture(
    budget_points_a: Sequence[BudgetPoint], budget_points_b: Sequence[BudgetPoint], *, wall_ns: int,
    source_kind: str = "real", ceiling: int | None = None,
    operating_budget_id_a: str | None = None, operating_budget_id_b: str | None = None,
) -> DualBudgetReport:
    points_a, points_b = list(budget_points_a), list(budget_points_b)
    if not points_a or not points_b or points_a[0].policy != points_b[0].policy:
        raise BudgetRefusal("dual architecture analysis needs two sweeps under one policy")
    policy = points_a[0].policy
    if len(policy.architectures) != 2:
        raise BudgetRefusal("dual architecture policy must declare exactly two architectures")
    arch_a, arch_b = policy.architectures
    report_a = _run_architecture(points_a, arch_a, wall_ns=wall_ns, source_kind=source_kind,
                                 ceiling=ceiling, operating_budget_id=operating_budget_id_a)
    report_b = _run_architecture(points_b, arch_b, wall_ns=wall_ns, source_kind=source_kind,
                                 ceiling=ceiling, operating_budget_id=operating_budget_id_b)
    dominates_a = report_a.candidate_strictly_dominates_rate_matched_random
    dominates_b = report_b.candidate_strictly_dominates_rate_matched_random
    both = dominates_a and dominates_b
    verdict = "mechanics-ok" if both else ("architecture-fragile" if dominates_a or dominates_b else "null")
    limit = policy.flop_ceiling if ceiling is None else ceiling
    return DualBudgetReport(policy, source_kind, limit, {arch_a: report_a, arch_b: report_b},
                            dominates_a, dominates_b, both, verdict)


__all__ = [
    "ARM_ALWAYS_ON", "ARM_BEST_SINGLE", "ARM_CANDIDATE", "ARM_NEVER_UPDATE",
    "ARM_RATE_MATCHED_RANDOM", "TRAIN_BACKWARD_MULTIPLIER", "ArchitectureReport", "Arm",
    "BreakEven", "BudgetMismatch", "BudgetPoint", "BudgetPolicy", "BudgetRefusal", "BudgetReport",
    "BudgetSeedRun",
    "CeilingExceeded", "ComputePoint", "DualBudgetReport", "FlopModel", "SeedResult",
    "UnchargedTraining", "assert_matched_ex_training", "assert_within_ceiling", "break_even_queries",
    "build_budget_points", "featurize_run_flops", "gate_infer_run_flops", "gate_train_flops", "paired_deltas",
    "noise_control_summary", "pareto_frontier", "per_query_saving_vs_always_on", "run_dual_architecture",
    "run_matched_budget",
]
