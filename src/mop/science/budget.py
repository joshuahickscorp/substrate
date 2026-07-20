from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

ARM_CANDIDATE = "candidate"
ARM_RATE_MATCHED_RANDOM = "rate_matched_random"
ARM_ALWAYS_ON = "always_on"
ARM_BEST_SINGLE = "best_single"
ARM_NEVER_UPDATE = "never_update"


class BudgetRefusal(ValueError):
    pass


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise BudgetRefusal(f"{label} must be a {qualifier} integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BudgetRefusal(f"{label} must be a finite real number")
    return float(value)


def _sealed(value: float) -> float:
    return round(float(value), 12)


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
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
        required = ("schema", "bed_id", "metric_key", "action_key", "reference_kind", "claim_scope")
        if any(not isinstance(getattr(self, field), str) or not getattr(self, field) for field in required):
            raise BudgetRefusal("BudgetPolicy identity fields must be non-empty strings")
        if self.direction not in ("higher", "lower"):
            raise BudgetRefusal("BudgetPolicy.direction must be higher or lower")
        if self.reference_kind not in (ARM_BEST_SINGLE, ARM_NEVER_UPDATE):
            raise BudgetRefusal("BudgetPolicy.reference_kind is not a supported control")
        _integer(self.flop_ceiling, "BudgetPolicy.flop_ceiling", positive=True)
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

    def metric(self, value: object) -> float:
        number = _finite(value, self.metric_key)
        if number < self.metric_min or (self.metric_max is not None and number > self.metric_max):
            raise BudgetRefusal(f"{self.metric_key} must be in [{self.metric_min}, {self.metric_max}]")
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
        for field in ("featurize_flops", "gate_infer_flops", "downstream_flops_per_firing", "train_flops"):
            _integer(getattr(self, field), f"FlopModel.{field}")

    def run_flops(self, actions: int) -> int:
        return (
            self.featurize_flops
            + self.gate_infer_flops
            + _integer(actions, "actions") * self.downstream_flops_per_firing
        )

    def lifecycle_flops(self, actions: int) -> int:
        return self.run_flops(actions) + self.train_flops

    def payload(self) -> dict[str, int]:
        return {
            "featurize_flops": self.featurize_flops,
            "gate_infer_flops": self.gate_infer_flops,
            "downstream_flops_per_firing": self.downstream_flops_per_firing,
            "train_flops": self.train_flops,
        }


def arm_flop_model(
    kind: str,
    total_frames: int,
    *,
    featurize_per_frame: int,
    gate_infer_per_frame: int,
    downstream_flops_per_firing: int,
    candidate_train_flops: Callable[[], int],
) -> FlopModel:
    runs_gate = kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM)
    return FlopModel(
        featurize_per_frame * total_frames,
        gate_infer_per_frame * total_frames if runs_gate else 0,
        downstream_flops_per_firing,
        candidate_train_flops() if kind == ARM_CANDIDATE else 0,
    )


@dataclass(frozen=True, slots=True)
class BudgetSeedRun:
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
class _Arm:
    policy: BudgetPolicy
    name: str
    kind: str
    total_frames: int
    params: int
    flop_model: FlopModel
    results: tuple[tuple[int, float, int], ...]
    architecture: str | None

    def __post_init__(self) -> None:
        _integer(self.total_frames, "Arm.total_frames", positive=True)
        _integer(self.params, "Arm.params")
        if self.kind not in self.policy.all_arms or not self.results:
            raise BudgetRefusal("arm kind and seed results must satisfy the policy")
        if self.architecture not in ((None,) if not self.policy.architectures else self.policy.architectures):
            raise BudgetRefusal(f"architecture {self.architecture!r} is not declared")
        seeds = [row[0] for row in self.results]
        if seeds != sorted(set(seeds)):
            raise BudgetRefusal("arm seeds must be unique and ascending")
        for seed, metric, actions in self.results:
            _integer(seed, "seed")
            self.policy.metric(metric)
            if _integer(actions, "actions") > self.total_frames:
                raise BudgetRefusal("an action count cannot exceed the total frame count")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(row[0] for row in self.results)

    def row(self, seed: int) -> tuple[int, float, int]:
        try:
            return next(row for row in self.results if row[0] == seed)
        except StopIteration as error:
            raise BudgetRefusal(f"arm {self.name!r} has no result for seed {seed}") from error

    def mean_metric(self) -> float:
        return math.fsum(row[1] for row in self.results) / len(self.results)

    def mean_actions(self) -> float:
        return math.fsum(row[2] for row in self.results) / len(self.results)

    def mean_flops(self) -> float:
        return math.fsum(self.flop_model.lifecycle_flops(row[2]) for row in self.results) / len(self.results)

    def max_flops(self) -> int:
        return max(self.flop_model.lifecycle_flops(row[2]) for row in self.results)

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.architecture is not None:
            payload["architecture"] = self.architecture
        payload.update(
            total_frames=self.total_frames,
            params=self.params,
            flop_model=self.flop_model.payload(),
            seed_results=[
                {"seed": seed, self.policy.metric_key: _sealed(metric), self.policy.action_key: actions}
                for seed, metric, actions in self.results
            ],
            **{
                f"mean_{self.policy.metric_key}": _sealed(self.mean_metric()),
                "mean_lifecycle_flops": _sealed(self.mean_flops()),
                "max_lifecycle_flops": self.max_flops(),
            },
        )
        if self.policy.include_mean_actions:
            payload[f"mean_{self.policy.action_key}"] = _sealed(self.mean_actions())
        return payload


@dataclass(frozen=True, slots=True)
class _Point:
    policy: BudgetPolicy
    budget_id: str
    arms: tuple[_Arm, _Arm, _Arm, _Arm]
    architecture: str | None

    @property
    def candidate(self) -> _Arm:
        return self.arms[0]

    @property
    def random(self) -> _Arm:
        return self.arms[1]

    def certify(self, ceiling: int) -> None:
        expected = (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, self.policy.reference_kind)
        if not self.budget_id or tuple(arm.kind for arm in self.arms) != expected:
            raise BudgetRefusal("budget point identity or arm order is invalid")
        for arm in self.arms:
            if (
                arm.policy != self.policy
                or arm.seeds != self.candidate.seeds
                or arm.total_frames != self.candidate.total_frames
            ):
                raise BudgetRefusal("all arms must share one policy, paired seeds, and total frame count")
            if arm.max_flops() > ceiling:
                raise BudgetRefusal(f"arm {arm.name!r} exceeds lifecycle FLOP ceiling {ceiling}")
            if arm.kind in self.policy.controls and arm.flop_model.train_flops:
                raise BudgetRefusal(f"control arm {arm.name!r} must not charge a training cost")
        if self.candidate.flop_model.train_flops <= 0:
            raise BudgetRefusal("the candidate must charge its amortized training cost C_train")
        for seed in self.candidate.seeds:
            candidate, control = self.candidate.row(seed), self.random.row(seed)
            if candidate[2] != control[2]:
                raise BudgetRefusal(f"seed {seed}: matched control action count differs from the candidate")
            if self.candidate.flop_model.run_flops(candidate[2]) != self.random.flop_model.run_flops(
                control[2]
            ):
                raise BudgetRefusal(f"seed {seed}: matched control inference FLOPs differ from the candidate")


@dataclass(frozen=True, slots=True)
class _ComputePoint:
    policy: BudgetPolicy
    budget_id: str
    arm_kind: str
    flops: float
    metric_value: float
    params: int
    actions: float
    architecture: str | None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"budget_id": self.budget_id}
        if self.architecture is not None:
            payload["architecture"] = self.architecture
        payload.update(
            arm_kind=self.arm_kind,
            flops=_sealed(self.flops),
            **{
                self.policy.metric_key: _sealed(self.metric_value),
                "params": self.params,
                self.policy.action_key: _sealed(self.actions),
            },
        )
        return payload


def _pareto(points: list[_ComputePoint]) -> tuple[_ComputePoint, ...]:
    frontier = []
    for candidate in points:
        higher = candidate.policy.direction == "higher"
        for other in points:
            metric_better = (
                other.metric_value > candidate.metric_value
                if higher
                else other.metric_value < candidate.metric_value
            )
            metric_no_worse = (
                other.metric_value >= candidate.metric_value
                if higher
                else other.metric_value <= candidate.metric_value
            )
            if (
                other.flops <= candidate.flops
                and metric_no_worse
                and (other.flops < candidate.flops or metric_better)
            ):
                break
        else:
            frontier.append(candidate)
    sign = -1 if points and points[0].policy.direction == "higher" else 1
    return tuple(sorted(frontier, key=lambda row: (row.flops, sign * row.metric_value)))


@dataclass(frozen=True, slots=True)
class BudgetReport:
    policy: BudgetPolicy
    source_kind: str
    flop_ceiling: int
    seeds: tuple[int, ...]
    arm_summaries: tuple[dict[str, Any], ...]
    pareto: tuple[_ComputePoint, ...]
    rows: tuple[dict[str, Any], ...]
    candidate_strictly_dominates_rate_matched_random: bool
    break_even: dict[str, Any]
    matched_budget: dict[str, int]
    verdict: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.policy.schema,
            "bed_id": self.policy.bed_id,
            "source_kind": self.source_kind,
            "flop_ceiling": self.flop_ceiling,
            "seeds": list(self.seeds),
            "arm_summaries": [dict(row) for row in self.arm_summaries],
            "pareto": [row.payload() for row in self.pareto],
            "per_budget_candidate_vs_rate_matched_random": [dict(row) for row in self.rows],
            "candidate_strictly_dominates_rate_matched_random": (
                self.candidate_strictly_dominates_rate_matched_random
            ),
            "break_even": dict(self.break_even),
            "matched_budget": dict(self.matched_budget),
            "verdict": self.verdict,
            "activation_allowed": False,
            "scientific_promotion": False,
            "independent_scientific_confirmation": False,
            "claim_scope": self.policy.claim_scope,
        }


def run_matched_budget(
    policy: BudgetPolicy,
    seed_runs: Sequence[Any],
    *,
    score_group: str,
    score_field: str,
    action_group: str,
    flop_model: Callable[[str], FlopModel],
    operating_budget_id: str | None = None,
    source_kind: str = "synthetic",
    ceiling: int | None = None,
    architecture: str | None = None,
    wall_ns: int | None = None,
) -> BudgetReport:
    runs = list(seed_runs)
    if not runs or source_kind not in ("synthetic", "real"):
        raise BudgetRefusal("matched-budget analysis requires seed runs and a real or synthetic source")
    if not isinstance(policy, BudgetPolicy):
        raise BudgetRefusal("matched-budget analysis requires a BudgetPolicy")
    first = runs[0]
    limit = policy.flop_ceiling if ceiling is None else _integer(ceiling, "ceiling", positive=True)
    points: list[_Point] = []
    for budget_id in first.per_budget:
        arms = []
        for kind in policy.all_arms:
            model = flop_model(kind)
            if not isinstance(model, FlopModel):
                raise BudgetRefusal("flop_model must return FlopModel")
            name = f"{kind}@{architecture}@{budget_id}" if architecture else f"{kind}@{budget_id}"
            results = tuple(
                (
                    run.seed,
                    run.per_budget[budget_id][score_group][kind][score_field],
                    run.per_budget[budget_id][action_group][kind],
                )
                for run in runs
            )
            arms.append(
                _Arm(
                    policy,
                    name,
                    kind,
                    first.total_frames,
                    first.gate_params if kind == ARM_CANDIDATE else 0,
                    model,
                    results,
                    architecture,
                )
            )
        point = _Point(policy, budget_id, tuple(arms), architecture)  # type: ignore[arg-type]
        point.certify(limit)
        points.append(point)
    if not points:
        raise BudgetRefusal("budget-point projection needs at least one budget")
    if len({point.budget_id for point in points}) != len(points):
        raise BudgetRefusal("budget point ids must be unique")

    compute: list[_ComputePoint] = []
    rows = []
    dominates = True
    for point in points:
        for arm in point.arms:
            compute.append(
                _ComputePoint(
                    policy,
                    point.budget_id,
                    arm.kind,
                    arm.mean_flops(),
                    arm.mean_metric(),
                    arm.params,
                    arm.mean_actions(),
                    architecture,
                )
            )
        candidate, control = point.candidate.mean_metric(), point.random.mean_metric()
        delta = policy.improvement(candidate, control)
        wins = delta > 0
        dominates &= wins
        rows.append(
            {
                "budget_id": point.budget_id,
                f"candidate_mean_{policy.metric_key}": _sealed(candidate),
                f"rate_matched_random_mean_{policy.metric_key}": _sealed(control),
                policy.delta_key or f"delta_mean_{policy.metric_key}": _sealed(delta),
                "matched_inference_flops": point.candidate.flop_model.run_flops(
                    point.candidate.results[0][2]
                ),
                "candidate_strictly_beats_rate_matched_random": wins,
            }
        )

    operating = points[0]
    if operating_budget_id is not None:
        try:
            operating = next(point for point in points if point.budget_id == operating_budget_id)
        except StopIteration as error:
            raise BudgetRefusal(f"operating_budget_id {operating_budget_id!r} is not in the sweep") from error
    frames = operating.candidate.total_frames
    actions = round(operating.candidate.mean_actions())
    saving = (frames - actions) / frames * operating.candidate.flop_model.downstream_flops_per_firing
    training = operating.candidate.flop_model.train_flops
    break_even = {
        "train_flops": training,
        "per_query_saving": _sealed(saving),
        "n_star_frames": math.ceil(training / saving) if saving > 0 else None,
        "amortizable": saving > 0,
    }
    binding = max(point.candidate.max_flops() for point in points)
    if binding > limit:
        raise BudgetRefusal(f"binding candidate FLOPs {binding} exceed ceiling {limit}")
    matched = {
        "params": points[0].candidate.params,
        "flops": binding,
        "wall_ns": binding if wall_ns is None else _integer(wall_ns, "wall_ns", positive=True),
        "seeds": len(points[0].candidate.seeds),
    }
    return BudgetReport(
        policy,
        source_kind,
        limit,
        points[0].candidate.seeds,
        tuple(arm.payload() for point in points for arm in point.arms),
        _pareto(compute),
        tuple(rows),
        dominates,
        break_even,
        matched,
        "mechanics-ok" if dominates else "null",
    )


__all__ = [
    "ARM_ALWAYS_ON",
    "ARM_BEST_SINGLE",
    "ARM_CANDIDATE",
    "ARM_NEVER_UPDATE",
    "ARM_RATE_MATCHED_RANDOM",
    "BudgetPolicy",
    "BudgetRefusal",
    "BudgetReport",
    "BudgetSeedRun",
    "FlopModel",
    "arm_flop_model",
    "noise_control_summary",
    "run_matched_budget",
]
