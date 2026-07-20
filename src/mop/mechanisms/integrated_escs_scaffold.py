
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..substrate.events import canonical_sha256

INTEGRATED_ESCS_SCHEMA = "mop-integrated-escs/v1"

CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

SCIENTIFIC_CAPABILITY_CLAIM = False

COST_AXES: tuple[str, ...] = ("params", "flops", "memory_bytes", "wall_ticks", "energy_units")

REQUIRED_BASELINES: tuple[str, ...] = (
    "single-perspective",
    "static-ensemble",
    "no-memory",
    "sparse-only",
)

MECHANISM_LADDER: tuple[str, ...] = (
    "multi-perspective",
    "dynamic-composition",
    "working-memory",
    "dense-mixing",
)

NULL_VERDICT = "no-integrated-advantage-at-matched-cost"
ADVANTAGE_VERDICT = "integrated-advantage-earned-at-matched-cost"

_ID_RE = re.compile(r"^[a-z][a-z0-9._:-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class IntegratedEscsRefusal(ValueError):
    pass


def _require_id(value: str, label: str) -> None:
    if _ID_RE.fullmatch(value) is None:
        raise IntegratedEscsRefusal(f"{label} must use stable lowercase characters")


def _require_sha256(value: str, label: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise IntegratedEscsRefusal(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class CostVector:

    params: int
    flops: int
    memory_bytes: int
    wall_ticks: int
    energy_units: int

    def __post_init__(self) -> None:
        for name, value in (
            ("params", self.params),
            ("flops", self.flops),
            ("memory_bytes", self.memory_bytes),
            ("wall_ticks", self.wall_ticks),
            ("energy_units", self.energy_units),
        ):
            if value <= 0:
                raise IntegratedEscsRefusal(f"cost axis {name} must be positive (non-vacuous)")

    def as_mapping(self) -> dict[str, int]:
        return {
            "params": self.params,
            "flops": self.flops,
            "memory_bytes": self.memory_bytes,
            "wall_ticks": self.wall_ticks,
            "energy_units": self.energy_units,
        }

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (self.params, self.flops, self.memory_bytes, self.wall_ticks, self.energy_units)


@dataclass(frozen=True, slots=True)
class FrontierPoint:

    label: str
    quality: float
    cost: CostVector
    claim_scope: str = CLAIM_SCOPE
    schema: str = INTEGRATED_ESCS_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != INTEGRATED_ESCS_SCHEMA:
            raise IntegratedEscsRefusal(f"unsupported frontier point schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise IntegratedEscsRefusal("frontier point claim scope cannot be widened")
        _require_id(self.label, "FrontierPoint.label")
        if self.quality != self.quality or self.quality in (float("inf"), float("-inf")):
            raise IntegratedEscsRefusal("frontier point quality must be a finite number")
        if not 0.0 <= self.quality <= 1.0:
            raise IntegratedEscsRefusal("frontier point quality must lie in [0.0, 1.0]")

    def dominates(self, other: FrontierPoint) -> bool:

        mine = self.cost.as_tuple()
        theirs = other.cost.as_tuple()
        quality_no_worse = self.quality >= other.quality
        costs_no_worse = all(a <= b for a, b in zip(mine, theirs, strict=True))
        if not (quality_no_worse and costs_no_worse):
            return False
        strictly_better_quality = self.quality > other.quality
        strictly_cheaper = any(a < b for a, b in zip(mine, theirs, strict=True))
        return strictly_better_quality or strictly_cheaper

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "label": self.label,
            "quality": self.quality,
            "cost": self.cost.as_mapping(),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class BaselineDeclaration:

    family: str
    point: FrontierPoint
    rationale: str
    matched: bool

    def __post_init__(self) -> None:
        if self.family not in REQUIRED_BASELINES:
            raise IntegratedEscsRefusal(f"unsupported baseline family {self.family!r}")
        if not self.rationale:
            raise IntegratedEscsRefusal("baseline declaration requires a nonempty rationale")
        if not self.matched:
            raise IntegratedEscsRefusal(
                "baseline declaration must be declared cost-matched to the integrated system"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "point": self.point.payload(),
            "rationale": self.rationale,
            "matched": self.matched,
        }


@dataclass(frozen=True, slots=True)
class BaselineSet:

    schema: str
    declarations: tuple[BaselineDeclaration, ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != INTEGRATED_ESCS_SCHEMA:
            raise IntegratedEscsRefusal(f"unsupported baseline set schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise IntegratedEscsRefusal("baseline set claim scope cannot be widened")
        families = tuple(row.family for row in self.declarations)
        if families != REQUIRED_BASELINES:
            raise IntegratedEscsRefusal("baseline set membership or order drift is refused")

    def assert_all_cost_matched(self, cost: CostVector) -> None:

        for row in self.declarations:
            if row.point.cost.as_tuple() != cost.as_tuple():
                raise IntegratedEscsRefusal(
                    f"baseline {row.family!r} cost is not matched to the integrated full-system cost"
                )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "declarations": [row.payload() for row in self.declarations],
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


_MATCHED_COST = CostVector(params=1000, flops=2000, memory_bytes=4000, wall_ticks=100, energy_units=50)


def build_default_baseline_set() -> BaselineSet:

    qualities = {
        "single-perspective": 0.50,
        "static-ensemble": 0.55,
        "no-memory": 0.60,
        "sparse-only": 0.65,
    }
    declarations = tuple(
        BaselineDeclaration(
            family=family,
            point=FrontierPoint(
                label=f"baseline.{family}",
                quality=qualities[family],
                cost=_MATCHED_COST,
            ),
            rationale=f"matched {family} control held to the integrated full-system cost",
            matched=True,
        )
        for family in REQUIRED_BASELINES
    )
    return BaselineSet(schema=INTEGRATED_ESCS_SCHEMA, declarations=declarations)


@dataclass(frozen=True, slots=True)
class IntegratedAdvantageContract:

    schema: str
    integrated: FrontierPoint
    baselines: BaselineSet
    matched_cost_required: bool
    dominance_required: bool
    replication_min: int
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != INTEGRATED_ESCS_SCHEMA:
            raise IntegratedEscsRefusal(f"unsupported advantage contract schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise IntegratedEscsRefusal("advantage contract claim scope cannot be widened")
        if not self.matched_cost_required:
            raise IntegratedEscsRefusal(
                "integrated advantage requires matched full-system cost against every baseline"
            )
        if not self.dominance_required:
            raise IntegratedEscsRefusal(
                "integrated advantage requires strict Pareto dominance over every baseline"
            )
        if self.replication_min < 2:
            raise IntegratedEscsRefusal(
                "integrated advantage requires at least two independent replications"
            )
        self.baselines.assert_all_cost_matched(self.integrated.cost)
        for row in self.baselines.declarations:
            if not self.integrated.dominates(row.point):
                raise IntegratedEscsRefusal(
                    "integrated point fails strict Pareto dominance over a matched baseline"
                )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "integrated": self.integrated.payload(),
            "baselines": self.baselines.payload(),
            "matched_cost_required": self.matched_cost_required,
            "dominance_required": self.dominance_required,
            "replication_min": self.replication_min,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class MechanismRung:

    mechanism: str
    marginal_quality_gain: float
    marginal_cost: CostVector
    min_efficiency: float

    def __post_init__(self) -> None:
        _require_id(self.mechanism, "MechanismRung.mechanism")
        if not self.marginal_quality_gain > 0.0:
            raise IntegratedEscsRefusal("rung requires strictly positive marginal quality gain")
        if not self.min_efficiency > 0.0:
            raise IntegratedEscsRefusal("rung minimum efficiency must be positive")

    @property
    def efficiency(self) -> float:
        return self.marginal_quality_gain / float(self.marginal_cost.flops)

    def justified(self) -> bool:
        return self.efficiency >= self.min_efficiency

    def payload(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "marginal_quality_gain": self.marginal_quality_gain,
            "marginal_cost": self.marginal_cost.as_mapping(),
            "min_efficiency": self.min_efficiency,
        }


@dataclass(frozen=True, slots=True)
class AblationLadderContract:

    schema: str
    rungs: tuple[MechanismRung, ...]
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != INTEGRATED_ESCS_SCHEMA:
            raise IntegratedEscsRefusal(f"unsupported ablation ladder schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise IntegratedEscsRefusal("ablation ladder claim scope cannot be widened")
        if tuple(row.mechanism for row in self.rungs) != MECHANISM_LADDER:
            raise IntegratedEscsRefusal("ablation ladder membership or order drift is refused")
        for row in self.rungs:
            if not row.justified():
                raise IntegratedEscsRefusal(
                    "one or more rungs do not justify their marginal compute at the declared minimum"
                )

    def total_marginal_gain(self) -> float:
        return sum((row.marginal_quality_gain for row in self.rungs), 0.0)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "rungs": [row.payload() for row in self.rungs],
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def build_default_ablation_ladder() -> AblationLadderContract:

    rungs = tuple(
        MechanismRung(
            mechanism=name,
            marginal_quality_gain=0.05,
            marginal_cost=CostVector(
                params=64, flops=1024, memory_bytes=256, wall_ticks=8, energy_units=4
            ),
            min_efficiency=1e-9,
        )
        for name in MECHANISM_LADDER
    )
    return AblationLadderContract(schema=INTEGRATED_ESCS_SCHEMA, rungs=rungs)


@dataclass(frozen=True, slots=True)
class FrontierVerdict:

    schema: str
    integrated: FrontierPoint
    baselines: BaselineSet
    matched_cost_confirmed: bool
    replications: int
    seed: int = 0
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != INTEGRATED_ESCS_SCHEMA:
            raise IntegratedEscsRefusal(f"unsupported frontier verdict schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise IntegratedEscsRefusal("frontier verdict claim scope cannot be widened")
        if self.seed < 0:
            raise IntegratedEscsRefusal("frontier verdict seed must be nonnegative")
        if self.replications < 1:
            raise IntegratedEscsRefusal("frontier verdict requires at least one replication")
        if not self.matched_cost_confirmed:
            raise IntegratedEscsRefusal(
                "frontier verdict fails closed because matched cost was not confirmed"
            )
        self.baselines.assert_all_cost_matched(self.integrated.cost)

    def _dominates_all(self) -> bool:
        return all(self.integrated.dominates(row.point) for row in self.baselines.declarations)

    def verdict(self) -> str:
        if self._dominates_all() and self.matched_cost_confirmed and self.replications >= 2:
            return ADVANTAGE_VERDICT
        return NULL_VERDICT

    def assert_advantage_earned(self) -> None:
        if self.verdict() != ADVANTAGE_VERDICT:
            raise IntegratedEscsRefusal(
                "integrated advantage is unearned; the prior null holds at matched cost"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "integrated": self.integrated.payload(),
            "baselines": self.baselines.payload(),
            "matched_cost_confirmed": self.matched_cost_confirmed,
            "replications": self.replications,
            "seed": self.seed,
            "verdict": self.verdict(),
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def build_null_frontier_verdict(seed: int = 0) -> FrontierVerdict:

    integrated = FrontierPoint(label="integrated.escs.null", quality=0.30, cost=_MATCHED_COST)
    return FrontierVerdict(
        schema=INTEGRATED_ESCS_SCHEMA,
        integrated=integrated,
        baselines=build_default_baseline_set(),
        matched_cost_confirmed=True,
        replications=2,
        seed=seed,
    )


def build_dominating_frontier_verdict(seed: int = 0) -> FrontierVerdict:

    integrated = FrontierPoint(label="integrated.escs.dominating", quality=0.95, cost=_MATCHED_COST)
    return FrontierVerdict(
        schema=INTEGRATED_ESCS_SCHEMA,
        integrated=integrated,
        baselines=build_default_baseline_set(),
        matched_cost_confirmed=True,
        replications=3,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class ActivationReceipt:

    verdict_digest: str
    replications: int
    independent_auditor: str
    preregistration_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.verdict_digest, "ActivationReceipt.verdict_digest")
        _require_sha256(self.preregistration_sha256, "ActivationReceipt.preregistration_sha256")
        if not self.independent_auditor.strip():
            raise IntegratedEscsRefusal("activation receipt requires a named independent auditor")
        if self.replications < 2:
            raise IntegratedEscsRefusal(
                "activation receipt requires at least two independent replications"
            )


@dataclass(frozen=True, slots=True)
class IntegratedActivationGate:

    activation_required: bool = True
    local_activation_permitted: bool = False

    def __post_init__(self) -> None:
        if not self.activation_required:
            raise IntegratedEscsRefusal("the activation gate only guards work that requires activation")
        if self.local_activation_permitted:
            raise IntegratedEscsRefusal("local activation of integration is never permitted by default")

    def authorize(
        self,
        receipt: ActivationReceipt | None,
        *,
        expected_verdict_digest: str,
    ) -> ActivationReceipt:

        _require_sha256(expected_verdict_digest, "IntegratedActivationGate.expected_verdict_digest")
        if receipt is None:
            raise IntegratedEscsRefusal(
                "integrated activation is not earned yet; supply a valid activation receipt from an "
                "external authority with its own randomization and held-out evaluation"
            )
        if receipt.verdict_digest != expected_verdict_digest:
            raise IntegratedEscsRefusal(
                "activation receipt does not bind the expected verdict digest"
            )
        return receipt


def coverage() -> dict[str, tuple[str, ...]]:

    return {
        "matched-baseline-frontier": (
            "matched baselines via BaselineSet over single-perspective, static-ensemble, no-memory, "
            "and sparse-only, refusing any membership or order drift",
            "matched full-system cost enforced by assert_all_cost_matched before any comparison",
        ),
        "marginal-mechanism-justification": (
            "an ordered ablation ladder via AblationLadderContract over the fixed mechanism ladder",
            "each rung must justify its marginal compute per FLOP or the ladder fails closed",
        ),
        "pareto-dominance-or-null": (
            "strict Pareto dominance over every matched baseline via FrontierPoint.dominates",
            "the prior null (no integrated advantage at matched cost) holds by default and must be "
            "earned through a dominating verdict and an external activation receipt",
        ),
    }
