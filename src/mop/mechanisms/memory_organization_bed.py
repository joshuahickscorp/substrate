"""Deterministic toy bed for the memory-organization epoch (mechanism_id memory_organization).

This module builds the task environment, not a result. It supplies a null regime where there is no
reusable structure across decisions, so any memory organization buys nothing and the named prior
null holds, and a favorable regime where there is reusable structure across future decisions that an
organized memory can exploit through revision, provenance, deletion, and action-conditioned recall.
Every regime is a stream of memory ops followed by future-decision queries, and every arm is scored
on the FUTURE decisions it makes, not only on the values it predicts.

The favorable regime has four selectable structure modes. The reusable mode carries genuine reusable
structure that lets an organized memory beat every control. The absent mode strips that structure so
the favorable regime degenerates to the null and no arm can win. The prediction-only mode carries
structure that lets an organized memory recall values more accurately without changing any decision,
which must never count as a future-decision win. The harmful mode injects misleading memory so a
faithful organized recall does worse than ignoring memory, which the episodic-harm guard must catch.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim. The
streams are toy op logs and the scores are byte-exact test vectors, not evidence about any memory.

House style: no em or en dashes. Engineering vocabulary only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.ladder_contracts import MatchedBudget
from ..substrate.events import canonical_sha256
from .memory_organization_scaffold import FUTURE_DECISION_CONTROLS

MECHANISM_ID = "memory_organization"

# The treatment arm and its four declared controls, reusing the scaffold's canonical control tuple.
ORGANIZED_ARM = "organized-memory"
ARMS: tuple[str, ...] = (ORGANIZED_ARM,) + FUTURE_DECISION_CONTROLS

# The action space. action_of buckets a recalled value into a decision. Values that share a bucket
# yield the same decision, which is what makes prediction-only improvement possible without any
# future-decision gain.
N_ACTIONS = 4

REGIME_NULL = "null"
REGIME_FAVORABLE = "favorable"

# Favorable regime structure modes. Only the reusable mode should ever yield a future-decision win.
STRUCTURE_REUSABLE = "reusable-structure"
STRUCTURE_ABSENT = "no-reusable-structure"
STRUCTURE_PREDICTION_ONLY = "prediction-only"
STRUCTURE_HARMFUL = "memory-harms"
FAVORABLE_STRUCTURES: tuple[str, ...] = (
    STRUCTURE_REUSABLE,
    STRUCTURE_ABSENT,
    STRUCTURE_PREDICTION_ONLY,
    STRUCTURE_HARMFUL,
)


class MemoryBedRefusal(ValueError):
    """Raised when a bed request is malformed. The bed fails closed."""


def action_of(value: int) -> int:
    """Bucket a recalled value into a decision. Deterministic and total over the integers."""

    return value % N_ACTIONS


@dataclass(frozen=True, slots=True)
class MemoryOp:
    """One memory operation arriving at a step: a keyed value with provenance and a retract flag."""

    step: int
    key: int
    value: int
    reliable: bool
    retract: bool

    def payload(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "key": self.key,
            "value": self.value,
            "reliable": self.reliable,
            "retract": self.retract,
        }


@dataclass(frozen=True, slots=True)
class DecisionQuery:
    """One future decision: which key it is about, the observable default, and the ground truth."""

    key: int
    default_feature: int
    optimal_action: int
    true_value: int

    def payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "default_feature": self.default_feature,
            "optimal_action": self.optimal_action,
            "true_value": self.true_value,
        }


@dataclass(frozen=True, slots=True)
class DecisionStream:
    """One regime instance: the memory ops that arrived and the future decisions to be made."""

    regime: str
    structure: str
    seed: int
    ops: tuple[MemoryOp, ...]
    queries: tuple[DecisionQuery, ...]

    def __post_init__(self) -> None:
        if not self.queries:
            raise MemoryBedRefusal("a decision stream must carry at least one future decision")

    def payload(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "structure": self.structure,
            "seed": self.seed,
            "ops": [op.payload() for op in self.ops],
            "queries": [query.payload() for query in self.queries],
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


# ---------------------------------------------------------------------------
# Scenario builders. Each scenario uses one unique key and is designed so that a specific subset of
# arms recalls the correct current value and the others do not. Values are fixed so the win pattern
# holds for every seed; the seed only offsets keys and salts the pseudo-random null targets.
# ---------------------------------------------------------------------------

# Fixed value labels. action_of(0)=0, action_of(1)=1, action_of(2)=2, action_of(3)=3.
_A = 1
_B = 2
_C = 3
_W = 2
_DEFAULT = 0
# Prediction-only labels: _PA and _PB share an action bucket but differ in value.
_PA = 1
_PB = 1 + N_ACTIONS
_PDEFAULT = 1


def _revision_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:
    """Late genuine revision. Organized and flat recall the new value; replay and stale do not."""

    ops = [
        MemoryOp(step=0, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=2, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=3, key=key, value=_B, reliable=True, retract=False),
    ]
    query = DecisionQuery(
        key=key, default_feature=_DEFAULT, optimal_action=action_of(_B), true_value=_B
    )
    return ops, query


def _provenance_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:
    """Late unreliable noise. Organized, replay, and stale keep the trusted value; flat is fooled."""

    ops = [
        MemoryOp(step=0, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=2, key=key, value=_C, reliable=False, retract=False),
    ]
    query = DecisionQuery(
        key=key, default_feature=_DEFAULT, optimal_action=action_of(_A), true_value=_A
    )
    return ops, query


def _deletion_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:
    """A retraction. Organized and no-memory fall back to the default; the rest keep a dead value."""

    ops = [
        MemoryOp(step=0, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_A, reliable=True, retract=True),
    ]
    query = DecisionQuery(
        key=key, default_feature=_DEFAULT, optimal_action=action_of(_DEFAULT), true_value=_DEFAULT
    )
    return ops, query


def _organized_only_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:
    """Revision plus provenance so only the fully organized memory recalls the current value."""

    ops = [
        MemoryOp(step=0, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_A, reliable=True, retract=False),
        MemoryOp(step=2, key=key, value=_B, reliable=True, retract=False),
        MemoryOp(step=3, key=key, value=_C, reliable=False, retract=False),
    ]
    query = DecisionQuery(
        key=key, default_feature=_DEFAULT, optimal_action=action_of(_B), true_value=_B
    )
    return ops, query


def _misleading_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:
    """Misleading memory. Every arm that trusts memory picks a worse action than no-memory does."""

    ops = [
        MemoryOp(step=0, key=key, value=_W, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_W, reliable=True, retract=False),
    ]
    query = DecisionQuery(
        key=key, default_feature=_DEFAULT, optimal_action=action_of(_DEFAULT), true_value=_DEFAULT
    )
    return ops, query


def _prediction_only_scenario(key: int) -> tuple[list[MemoryOp], DecisionQuery]:
    """A revision whose old and new values share an action bucket: better recall, same decision."""

    ops = [
        MemoryOp(step=0, key=key, value=_PA, reliable=True, retract=False),
        MemoryOp(step=1, key=key, value=_PA, reliable=True, retract=False),
        MemoryOp(step=2, key=key, value=_PA, reliable=True, retract=False),
        MemoryOp(step=3, key=key, value=_PB, reliable=True, retract=False),
    ]
    query = DecisionQuery(
        key=key, default_feature=_PDEFAULT, optimal_action=action_of(_PB), true_value=_PB
    )
    return ops, query


def _structureless_scenario(seed: int, key: int) -> tuple[list[MemoryOp], DecisionQuery]:
    """No reusable structure: memory only echoes the default, so every arm ties at the same score."""

    optimal = int(canonical_sha256({"seed": seed, "key": key})[:8], 16) % N_ACTIONS
    ops = [MemoryOp(step=0, key=key, value=_PDEFAULT, reliable=True, retract=False)]
    query = DecisionQuery(
        key=key, default_feature=_PDEFAULT, optimal_action=optimal, true_value=_PDEFAULT
    )
    return ops, query


def _assemble(
    regime: str,
    structure: str,
    seed: int,
    scenarios: list[tuple[list[MemoryOp], DecisionQuery]],
) -> DecisionStream:
    ops: list[MemoryOp] = []
    queries: list[DecisionQuery] = []
    for scenario_ops, query in scenarios:
        ops.extend(scenario_ops)
        queries.append(query)
    return DecisionStream(
        regime=regime,
        structure=structure,
        seed=seed,
        ops=tuple(ops),
        queries=tuple(queries),
    )


def _require_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise MemoryBedRefusal("regime seed must be a nonnegative integer")


def _key_base(seed: int) -> int:
    return seed * 1000


@dataclass(frozen=True, slots=True)
class MemoryOrganizationBed:
    """A deterministic memory-organization bed with a null and a favorable regime.

    Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
    """

    structure: str = STRUCTURE_REUSABLE
    mechanism_id: str = MECHANISM_ID

    def __post_init__(self) -> None:
        if self.structure not in FAVORABLE_STRUCTURES:
            raise MemoryBedRefusal(f"unsupported favorable structure {self.structure!r}")
        if self.mechanism_id != MECHANISM_ID:
            raise MemoryBedRefusal("memory-organization bed mechanism id cannot be renamed")

    def controls(self) -> tuple[str, ...]:
        return FUTURE_DECISION_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        """A non-vacuous matched retrieval budget every arm is held to before comparison."""

        return MatchedBudget(params=256, flops=4096, wall_ns=1024, seeds=8)

    def null_regime(self, seed: int) -> DecisionStream:
        """No reusable structure across decisions, so an organized memory can buy nothing here."""

        _require_seed(seed)
        base = _key_base(seed)
        scenarios = [_structureless_scenario(seed, base + index) for index in range(6)]
        return _assemble(REGIME_NULL, STRUCTURE_ABSENT, seed, scenarios)

    def favorable_regime(self, seed: int) -> DecisionStream:
        """Reusable structure across future decisions, selected by this bed's structure mode."""

        _require_seed(seed)
        base = _key_base(seed)
        if self.structure == STRUCTURE_REUSABLE:
            scenarios = [_revision_scenario(base + index) for index in range(3)]
            scenarios += [_provenance_scenario(base + 3 + index) for index in range(3)]
            scenarios += [_deletion_scenario(base + 6 + index) for index in range(3)]
        elif self.structure == STRUCTURE_ABSENT:
            scenarios = [_structureless_scenario(seed, base + index) for index in range(6)]
        elif self.structure == STRUCTURE_PREDICTION_ONLY:
            scenarios = [_prediction_only_scenario(base + index) for index in range(6)]
        else:
            scenarios = [_misleading_scenario(base + index) for index in range(4)]
            scenarios += [_organized_only_scenario(base + 4 + index) for index in range(2)]
        return _assemble(REGIME_FAVORABLE, self.structure, seed, scenarios)


def favorable_bed() -> MemoryOrganizationBed:
    """The default bed whose favorable regime carries genuine reusable structure."""

    return MemoryOrganizationBed(structure=STRUCTURE_REUSABLE)


def structureless_bed() -> MemoryOrganizationBed:
    """A bed whose favorable regime is stripped of reusable structure, so no arm can win."""

    return MemoryOrganizationBed(structure=STRUCTURE_ABSENT)


def prediction_only_bed() -> MemoryOrganizationBed:
    """A bed whose favorable regime improves recall accuracy but changes no future decision."""

    return MemoryOrganizationBed(structure=STRUCTURE_PREDICTION_ONLY)


def harmful_bed() -> MemoryOrganizationBed:
    """A bed whose favorable regime injects misleading memory that harms future decisions."""

    return MemoryOrganizationBed(structure=STRUCTURE_HARMFUL)
