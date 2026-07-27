from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

ARM_CANDIDATE = "candidate"
ARM_RATE_MATCHED_RANDOM = "rate_matched_random"
ARM_ALWAYS_ON = "always_on"
ARM_NEVER_UPDATE = "never_update"
_ARMS = (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE)


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
    claim_scope: str
    flop_ceiling: int

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value for value in (self.schema, self.bed_id, self.claim_scope)
        ):
            raise BudgetRefusal("BudgetPolicy identity fields must be non-empty strings")
        _integer(self.flop_ceiling, "BudgetPolicy.flop_ceiling", positive=True)


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
    return FlopModel(
        featurize_per_frame * total_frames,
        gate_infer_per_frame * total_frames if kind in (ARM_CANDIDATE, ARM_RATE_MATCHED_RANDOM) else 0,
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
    seed_runs: Sequence[Any],
    *,
    at_chance: bool,
    mean_noise_rate: float,
    mean_base_rate: float,
) -> dict[str, Any]:
    return {
        "noisy_tv_at_chance": bool(at_chance),
        "mean_reestimate_rate_on_noise": _sealed(mean_noise_rate),
        "mean_base_rate": _sealed(mean_base_rate),
        "per_seed_noisy_tv": [run.noisy_tv for run in seed_runs],
        "primary_control": ARM_RATE_MATCHED_RANDOM,
        "control_arms": [ARM_RATE_MATCHED_RANDOM, ARM_ALWAYS_ON, ARM_NEVER_UPDATE, "noisy_tv"],
    }


@dataclass(frozen=True, slots=True)
class _Arm:
    name: str
    kind: str
    total_frames: int
    params: int
    flop_model: FlopModel
    results: tuple[tuple[int, float, int], ...]

    def __post_init__(self) -> None:
        _integer(self.total_frames, "Arm.total_frames", positive=True)
        _integer(self.params, "Arm.params")
        if self.kind not in _ARMS or not self.results:
            raise BudgetRefusal("arm kind and seed results must satisfy the policy")
        seeds = [row[0] for row in self.results]
        if seeds != sorted(set(seeds)):
            raise BudgetRefusal("arm seeds must be unique and ascending")
        for seed, metric, actions in self.results:
            _integer(seed, "seed")
            if _finite(metric, "mae") < 0:
                raise BudgetRefusal("mae must be in [0.0, None]")
            if _integer(actions, "actions") > self.total_frames:
                raise BudgetRefusal("an action count cannot exceed the total frame count")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(row[0] for row in self.results)

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
        payload.update(
            total_frames=self.total_frames,
            params=self.params,
            flop_model=self.flop_model.payload(),
            seed_results=[
                {"seed": seed, "mae": _sealed(metric), "reestimations": actions}
                for seed, metric, actions in self.results
            ],
            mean_mae=_sealed(self.mean_metric()),
            mean_lifecycle_flops=_sealed(self.mean_flops()),
            max_lifecycle_flops=self.max_flops(),
            mean_reestimations=_sealed(self.mean_actions()),
        )
        return payload


def _certify_point(budget_id: str, arms: tuple[_Arm, ...], ceiling: int) -> None:
    if not budget_id or tuple(arm.kind for arm in arms) != _ARMS:
        raise BudgetRefusal("budget point identity or arm order is invalid")
    candidate, random = arms[:2]
    for arm in arms:
        if arm.seeds != candidate.seeds or arm.total_frames != candidate.total_frames:
            raise BudgetRefusal("all arms must share one policy, paired seeds, and total frame count")
        if arm.max_flops() > ceiling:
            raise BudgetRefusal(f"arm {arm.name!r} exceeds lifecycle FLOP ceiling {ceiling}")
        if arm.kind != ARM_CANDIDATE and arm.flop_model.train_flops:
            raise BudgetRefusal(f"control arm {arm.name!r} must not charge a training cost")
    if candidate.flop_model.train_flops <= 0:
        raise BudgetRefusal("the candidate must charge its amortized training cost C_train")
    for candidate_row, control_row in zip(candidate.results, random.results, strict=True):
        seed = candidate_row[0]
        if candidate_row[2] != control_row[2]:
            raise BudgetRefusal(f"seed {seed}: matched control action count differs from the candidate")
        if candidate.flop_model.run_flops(candidate_row[2]) != random.flop_model.run_flops(control_row[2]):
            raise BudgetRefusal(f"seed {seed}: matched control inference FLOPs differ from the candidate")


@dataclass(frozen=True, slots=True)
class _ComputePoint:
    budget_id: str
    arm_kind: str
    flops: float
    mae: float
    params: int
    actions: float

    def payload(self) -> dict[str, Any]:
        return {
            "budget_id": self.budget_id,
            "arm_kind": self.arm_kind,
            "flops": _sealed(self.flops),
            "mae": _sealed(self.mae),
            "params": self.params,
            "reestimations": _sealed(self.actions),
        }


def _pareto(points: list[_ComputePoint]) -> tuple[_ComputePoint, ...]:
    frontier = []
    for candidate in points:
        if not any(
            other.flops <= candidate.flops
            and other.mae <= candidate.mae
            and (other.flops < candidate.flops or other.mae < candidate.mae)
            for other in points
        ):
            frontier.append(candidate)
    return tuple(sorted(frontier, key=lambda row: (row.flops, row.mae)))


def run_matched_budget(
    policy: BudgetPolicy,
    seed_runs: Sequence[Any],
    *,
    flop_model: Callable[[str], FlopModel],
    operating_budget_id: str | None = None,
    ceiling: int | None = None,
) -> dict[str, Any]:
    runs = list(seed_runs)
    if not runs:
        raise BudgetRefusal("matched-budget analysis requires seed runs and a real source")
    if not isinstance(policy, BudgetPolicy):
        raise BudgetRefusal("matched-budget analysis requires a BudgetPolicy")
    first = runs[0]
    limit = policy.flop_ceiling if ceiling is None else _integer(ceiling, "ceiling", positive=True)
    points: list[tuple[str, tuple[_Arm, ...]]] = []
    for budget_id in first.per_budget:
        arms = tuple(
            _Arm(
                f"{kind}@{budget_id}",
                kind,
                first.total_frames,
                first.gate_params if kind == ARM_CANDIDATE else 0,
                model,
                tuple(
                    (
                        run.seed,
                        run.per_budget[budget_id]["arm_scores"][kind]["mae"],
                        run.per_budget[budget_id]["reestimations"][kind],
                    )
                    for run in runs
                ),
            )
            for kind in _ARMS
            if isinstance(model := flop_model(kind), FlopModel)
        )
        if len(arms) != len(_ARMS):
            raise BudgetRefusal("flop_model must return FlopModel")
        _certify_point(budget_id, arms, limit)
        points.append((budget_id, arms))
    if not points:
        raise BudgetRefusal("budget-point projection needs at least one budget")

    compute: list[_ComputePoint] = []
    rows = []
    dominates = True
    for budget_id, arms in points:
        for arm in arms:
            compute.append(
                _ComputePoint(
                    budget_id,
                    arm.kind,
                    arm.mean_flops(),
                    arm.mean_metric(),
                    arm.params,
                    arm.mean_actions(),
                )
            )
        candidate, random = arms[:2]
        candidate_mae, random_mae = candidate.mean_metric(), random.mean_metric()
        delta = random_mae - candidate_mae
        wins = delta > 0
        dominates &= wins
        rows.append(
            {
                "budget_id": budget_id,
                "candidate_mean_mae": _sealed(candidate_mae),
                "rate_matched_random_mean_mae": _sealed(random_mae),
                "delta_mean_mae_control_minus_candidate": _sealed(delta),
                "matched_inference_flops": candidate.flop_model.run_flops(candidate.results[0][2]),
                "candidate_strictly_beats_rate_matched_random": wins,
            }
        )

    operating = points[0]
    if operating_budget_id is not None:
        try:
            operating = next(point for point in points if point[0] == operating_budget_id)
        except StopIteration as error:
            raise BudgetRefusal(f"operating_budget_id {operating_budget_id!r} is not in the sweep") from error
    candidate = operating[1][0]
    frames, actions = candidate.total_frames, round(candidate.mean_actions())
    saving = (frames - actions) / frames * candidate.flop_model.downstream_flops_per_firing
    training = candidate.flop_model.train_flops
    break_even = {
        "train_flops": training,
        "per_query_saving": _sealed(saving),
        "n_star_frames": math.ceil(training / saving) if saving > 0 else None,
        "amortizable": saving > 0,
    }
    binding = max(arms[0].max_flops() for _, arms in points)
    if binding > limit:
        raise BudgetRefusal(f"binding candidate FLOPs {binding} exceed ceiling {limit}")
    first_candidate = points[0][1][0]
    return {
        "schema": policy.schema,
        "bed_id": policy.bed_id,
        "source_kind": "real",
        "flop_ceiling": limit,
        "seeds": list(first_candidate.seeds),
        "arm_summaries": [arm.payload() for _, arms in points for arm in arms],
        "pareto": [row.payload() for row in _pareto(compute)],
        "per_budget_candidate_vs_rate_matched_random": rows,
        "candidate_strictly_dominates_rate_matched_random": dominates,
        "break_even": break_even,
        "matched_budget": {
            "params": first_candidate.params,
            "flops": binding,
            "wall_ns": binding,
            "seeds": len(first_candidate.seeds),
        },
        "verdict": "mechanics-ok" if dominates else "null",
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
        "claim_scope": policy.claim_scope,
    }


__all__ = [
    "ARM_ALWAYS_ON",
    "ARM_CANDIDATE",
    "ARM_NEVER_UPDATE",
    "ARM_RATE_MATCHED_RANDOM",
    "BudgetPolicy",
    "BudgetRefusal",
    "BudgetSeedRun",
    "FlopModel",
    "arm_flop_model",
    "noise_control_summary",
    "run_matched_budget",
]
