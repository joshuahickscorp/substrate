"""Deterministic task-batch bed for the calibrated uncertainty mechanism (lane calibrated_uncertainty).

This module is runnable machinery, not a scaffold and not evidence. It builds the two task-batch
regimes a calibrated uncertainty mechanism must be measured on:

- NULL regime: the confidence signal is decoupled from correctness by construction. Every task's
  confidence is pure seeded noise confined strictly below the answer threshold, so any policy that
  thresholds confidence collapses onto the frozen-uniform abstainer and can never strictly beat it
  on either axis. The joint win is impossible by construction; the decoupled null holds.
- FAVORABLE regime: an honest confidence signal correlates with correctness by construction. Every
  correct task's confidence lands in a high band strictly above the answer threshold, every incorrect
  task's confidence lands in a low band strictly below it, so a calibrated thresholder answers exactly
  the correct tasks and abstains on exactly the incorrect ones, at fixed known margins.

Each batch carries TASK_COUNT tasks with a fixed correct vs incorrect composition. The incorrect
tasks are placed on a seed-rotated pattern that keeps both index parity classes at the same
composition, so the parity-masked random-abstain control answers a subset with a known error count
for EVERY seed. That is what makes the favorable joint win provable by construction, not sampled.

The regimes are mechanics only: seeded, byte-reproducible tuples with no capability claim and no
natural data. The bed owns the task geometry; the policies live in the impl module and the
measurement and verdict live in the runner module.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ladder.stage_ladder import MatchedBudget
from ..substrate.events import canonical_sha256
from .calibrated_uncertainty_scaffold import REQUIRED_CONTROLS

MECHANISM_ID = "calibrated_uncertainty"
BED_SCHEMA = "mop-calibrated-uncertainty-bed/v1"
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

# Geometry of the task batch. TASK_COUNT tasks, exactly INCORRECT_COUNT of them incorrect. The base
# incorrect positions hold four even and four odd indices; the seed rotates them by an even offset,
# which preserves the per-parity composition, so every parity half holds exactly four incorrect and
# twelve correct tasks for every seed.
TASK_COUNT = 32
INCORRECT_COUNT = 8
_INCORRECT_BASE: tuple[int, ...] = (2, 5, 10, 13, 18, 21, 26, 29)

# The answer bar every thresholding policy uses: answer strictly above, abstain at or below.
ANSWER_THRESHOLD = 0.5

# Favorable confidence bands. Correct tasks land in [0.70, 0.95), incorrect in [0.05, 0.30); both
# bands clear the threshold with comfortable slack, so the separation never depends on the seed.
CORRECT_CONF_BASE = 0.70
INCORRECT_CONF_BASE = 0.05
CONF_SPAN = 0.25

# Null confidence band. Pure seeded noise in [0.05, 0.45), decoupled from correctness and wholly
# below the answer threshold, so thresholding it is behaviorally identical to frozen abstention.
NULL_CONF_BASE = 0.05
NULL_CONF_SPAN = 0.40

REGIME_NULL = "null"
REGIME_FAVORABLE = "favorable"
REGIMES: tuple[str, ...] = (REGIME_NULL, REGIME_FAVORABLE)


class BedRefusal(ValueError):
    """Raised when a bed regime request is malformed or a declaration is widened."""


def _unit(seed: int, label: str) -> float:
    """A deterministic value in [0, 1) from a seeded digest; no wall clock, no rng."""

    if seed < 0:
        raise BedRefusal("bed seed must be nonnegative")
    digest = canonical_sha256({"seed": seed, "label": label})
    return int(digest[:8], 16) / 0x1_0000_0000


def _rotation(seed: int) -> int:
    """An even, seed-determined rotation of the incorrect positions; parity is preserved."""

    digest = canonical_sha256({"seed": seed, "label": "cu.rotation"})
    return 2 * (int(digest[:8], 16) % (TASK_COUNT // 2))


def _correctness(seed: int) -> tuple[int, ...]:
    """The 0/1 correctness pattern for a seed: 1 correct, 0 incorrect; composition is fixed."""

    rotation = _rotation(seed)
    incorrect = {(base + rotation) % TASK_COUNT for base in _INCORRECT_BASE}
    return tuple(0 if index in incorrect else 1 for index in range(TASK_COUNT))


@dataclass(frozen=True, slots=True)
class TaskBatch:
    """One regime rendered as concrete seeded tasks: a correctness bit and a confidence per task.

    ``correctness`` marks which tasks an answer would be correct on; ``confidence`` is the signal a
    policy may threshold. Claim scope: deterministic programmatic mechanics only.
    """

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
    """Honest confidence: a high band for correct tasks, a low band for incorrect ones."""

    values: list[float] = []
    for index, bit in enumerate(correctness):
        base = CORRECT_CONF_BASE if bit == 1 else INCORRECT_CONF_BASE
        values.append(base + CONF_SPAN * _unit(seed, f"fav.conf.{index}"))
    return tuple(values)


def _null_confidence(seed: int) -> tuple[float, ...]:
    """Decoupled confidence: pure seeded noise, wholly below the answer threshold."""

    return tuple(
        NULL_CONF_BASE + NULL_CONF_SPAN * _unit(seed, f"null.conf.{index}")
        for index in range(TASK_COUNT)
    )


@dataclass(frozen=True, slots=True)
class CalibratedUncertaintyBed:
    """The concrete calibrated uncertainty bed. mechanism_id matches the scaffold and the runner.

    Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
    """

    mechanism_id: str = MECHANISM_ID
    schema: str = BED_SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.mechanism_id != MECHANISM_ID:
            raise BedRefusal("bed mechanism_id drift")
        if self.schema != BED_SCHEMA:
            raise BedRefusal(f"unsupported bed schema {self.schema!r}")
        if self.claim_scope != CLAIM_SCOPE:
            raise BedRefusal("bed claim scope cannot be widened")

    def controls(self) -> tuple[str, ...]:
        """The declared control family: always_answer, random_abstain, overconfident_score,
        frozen_uniform."""

        return REQUIRED_CONTROLS

    def matched_cost(self) -> MatchedBudget:
        """A non-vacuous matched full-system budget every arm is held to. Positive on every axis."""

        return MatchedBudget(params=TASK_COUNT, flops=1_048_576, wall_ns=1_000_000, seeds=8)

    def null_regime(self, seed: int) -> TaskBatch:
        """A batch whose confidence is decoupled noise below the bar: the null holds by construction."""

        return TaskBatch(
            regime=REGIME_NULL,
            seed=seed,
            correctness=_correctness(seed),
            confidence=_null_confidence(seed),
        )

    def favorable_regime(self, seed: int) -> TaskBatch:
        """A batch whose confidence honestly separates correct from incorrect tasks by construction."""

        correctness = _correctness(seed)
        return TaskBatch(
            regime=REGIME_FAVORABLE,
            seed=seed,
            correctness=correctness,
            confidence=_favorable_confidence(seed, correctness),
        )

    def regime(self, name: str, seed: int) -> TaskBatch:
        """Dispatch to a regime by name. Fails closed on an unknown regime."""

        if name == REGIME_NULL:
            return self.null_regime(seed)
        if name == REGIME_FAVORABLE:
            return self.favorable_regime(seed)
        raise BedRefusal(f"unknown regime {name!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_id": self.mechanism_id,
            "task_count": TASK_COUNT,
            "incorrect_count": INCORRECT_COUNT,
            "answer_threshold": ANSWER_THRESHOLD,
            "controls": list(self.controls()),
            "matched_cost": self.matched_cost().payload(),
            "claim_scope": self.claim_scope,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())
