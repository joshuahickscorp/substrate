from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256
from .calibrated_uncertainty_scaffold import REQUIRED_CONTROLS
from .joint_axis_runner import MechanismBedBase, MechanismBedSpec, seeded_unit

MECHANISM_ID = "calibrated_uncertainty"
BED_SCHEMA = "mop-calibrated-uncertainty-bed/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

TASK_COUNT = 32
INCORRECT_COUNT = 8
_INCORRECT_BASE: tuple[int, ...] = (2, 5, 10, 13, 18, 21, 26, 29)

ANSWER_THRESHOLD = 0.5

CORRECT_CONF_BASE = 0.70
INCORRECT_CONF_BASE = 0.05
CONF_SPAN = 0.25

NULL_CONF_BASE = 0.05
NULL_CONF_SPAN = 0.40

REGIME_NULL = "null"
REGIME_FAVORABLE = "favorable"
REGIMES: tuple[str, ...] = (REGIME_NULL, REGIME_FAVORABLE)


class BedRefusal(ValueError):
    pass


def _unit(seed: int, label: str) -> float:
    return seeded_unit(seed, label, BedRefusal)


def _rotation(seed: int) -> int:

    digest = canonical_sha256({"seed": seed, "label": "cu.rotation"})
    return 2 * (int(digest[:8], 16) % (TASK_COUNT // 2))


def _correctness(seed: int) -> tuple[int, ...]:

    rotation = _rotation(seed)
    incorrect = {(base + rotation) % TASK_COUNT for base in _INCORRECT_BASE}
    return tuple(0 if index in incorrect else 1 for index in range(TASK_COUNT))


@dataclass(frozen=True, slots=True)
class TaskBatch:
    regime: str
    seed: int
    correctness: tuple[int, ...]
    confidence: tuple[float, ...]
    task_count: int = TASK_COUNT
    schema: str = BED_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != BED_SCHEMA:
            raise BedRefusal(f"unsupported task batch schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise BedRefusal("task batch claim scope cannot be widened")
        if self.regime not in REGIMES:
            raise BedRefusal(f"unknown regime {self.regime!r}")
        if self.task_count < 2:
            raise BedRefusal("a task batch needs at least two tasks")
        if len(self.correctness) != self.task_count or len(self.confidence) != self.task_count:
            raise BedRefusal("every task must carry exactly one correctness bit and one confidence")
        for bit in self.correctness:
            if bit not in (0, 1):
                raise BedRefusal("correctness bits must be 0 or 1")
        for value in self.confidence:
            if not 0.0 <= value <= 1.0:
                raise BedRefusal("every confidence must lie in the unit interval [0, 1]")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "regime": self.regime,
            "seed": self.seed,
            "task_count": self.task_count,
            "correctness": list(self.correctness),
            "confidence": list(self.confidence),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


def _favorable_confidence(seed: int, correctness: tuple[int, ...]) -> tuple[float, ...]:

    values: list[float] = []
    for index, bit in enumerate(correctness):
        base = CORRECT_CONF_BASE if bit == 1 else INCORRECT_CONF_BASE
        values.append(base + CONF_SPAN * _unit(seed, f"fav.conf.{index}"))
    return tuple(values)


def _null_confidence(seed: int) -> tuple[float, ...]:

    return tuple(
        NULL_CONF_BASE + NULL_CONF_SPAN * _unit(seed, f"null.conf.{index}") for index in range(TASK_COUNT)
    )


@dataclass(frozen=True, slots=True)
class CalibratedUncertaintyBed(MechanismBedBase):
    mechanism_id: str = MECHANISM_ID
    schema: str = BED_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    spec = MechanismBedSpec(
        MECHANISM_ID,
        BED_SCHEMA,
        REQUIRED_CONTROLS,
        MatchedBudget(params=TASK_COUNT, flops=1_048_576, wall_ns=1_000_000, seeds=8),
    )
    refusal = BedRefusal

    def null_regime(self, seed: int) -> TaskBatch:

        return TaskBatch(
            regime=REGIME_NULL,
            seed=seed,
            correctness=_correctness(seed),
            confidence=_null_confidence(seed),
        )

    def favorable_regime(self, seed: int) -> TaskBatch:

        correctness = _correctness(seed)
        return TaskBatch(
            regime=REGIME_FAVORABLE,
            seed=seed,
            correctness=correctness,
            confidence=_favorable_confidence(seed, correctness),
        )

    def configuration_payload(self) -> dict[str, Any]:
        return {
            "task_count": TASK_COUNT,
            "incorrect_count": INCORRECT_COUNT,
            "answer_threshold": ANSWER_THRESHOLD,
        }
