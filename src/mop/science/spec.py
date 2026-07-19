"""One typed scientific experiment contract (spec section 10).

An ExperimentSpec carries only the scientific DECLARATION of an experiment: identity, the null, the metric
and its direction, the sensitivity threshold, the paired seeds, the arms and their primary control, the
budget and parameter ceilings, the decision rule name, the reproduction floor, and the claim ceiling. All the
lifecycle and integrity machinery lives in ``mop.science.engine``; all the unique mathematics lives in
per-experiment providers. A family that used to hand-expand a prereg + producer + gate + harness per axis
becomes one ExperimentSpec plus its math providers.

Nothing here computes science. This module is declaration types only, so it stays tiny and auditable.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """A scientific metric and the direction that is better. ``sesoi`` is the smallest effect of interest."""

    name: str
    direction: str  # "lower" or "higher"
    sesoi: float

    def __post_init__(self) -> None:
        if self.direction not in ("lower", "higher"):
            raise ValueError("metric direction must be 'lower' or 'higher'")
        if not (self.sesoi > 0):
            raise ValueError("sesoi must be positive")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """The complete declaration of one experiment. Pure data; the engine supplies the lifecycle."""

    experiment_id: str
    schema: str
    stage: int
    question: str
    null_hypothesis: str
    metric: MetricSpec
    seeds: tuple[int, ...]
    arms: tuple[str, ...]
    primary_control: str
    decision_rule: str  # e.g. "paired_sign_flip_one_sided"
    min_reproductions: int
    claim_ceiling: str
    budget_flop_ceiling: float | None = None
    param_ceiling: int | None = None
    allowed_claim_verbs: tuple[str, ...] = ()
    forbidden_claim_verbs: tuple[str, ...] = ()
    extra: dict[str, object] = field(default_factory=dict)  # axis-specific declaration values

    def __post_init__(self) -> None:
        if self.primary_control not in self.arms:
            raise ValueError("primary_control must be one of arms")
        if "candidate" not in self.arms:
            raise ValueError("arms must include 'candidate'")
        if self.min_reproductions < 1:
            raise ValueError("min_reproductions must be >= 1")


@dataclass(frozen=True, slots=True)
class ArmSeedResult:
    """One arm at one seed: the sealed scalar metric value plus an opaque receipt of how it was produced."""

    arm: str
    seed: int
    metric_value: float
    receipt: dict[str, object]


@runtime_checkable
class ArmRunner(Protocol):
    """Unique-math boundary: produce one arm's metric value at one seed from deterministic inputs.

    Implementations live in the experiment family (the preserved science). The engine never computes the
    metric itself; it only orchestrates, pairs, decides, and seals.
    """

    def __call__(self, arm: str, seed: int, inputs: object) -> ArmSeedResult: ...


@runtime_checkable
class GradedRecompute(Protocol):
    """Independent verifier boundary: recompute the decision from raw arm results, structurally separate.

    This lives in the family's verifier module, never in the engine, so producer and verifier graded logic
    never share an implementation (spec section 9).
    """

    def __call__(self, arms: dict[str, list[ArmSeedResult]], spec: ExperimentSpec) -> dict[str, object]: ...


__all__ = ["MetricSpec", "ExperimentSpec", "ArmSeedResult", "ArmRunner", "GradedRecompute"]
