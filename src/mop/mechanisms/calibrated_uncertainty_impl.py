"""Deterministic policies for the calibrated uncertainty bed: the mechanism and its control policies.

This module is runnable machinery. It implements a tiny, fully deterministic selective-answering
dynamics (threshold a per-task confidence, answer above the bar, abstain at or below it) and five
arms measured on a task batch:

- ``mechanism`` (calibrated selective answering): answer exactly the tasks whose calibrated
  confidence clears the bar. On an honest signal this answers the correct tasks and abstains on the
  incorrect ones; on a decoupled signal it degenerates to whatever the noise dictates.
- ``always_answer``: answer every task regardless of confidence. Maximal coverage, pays every error.
- ``random_abstain``: abstain on a seeded parity half of the tasks, independent of confidence. Its
  answered half has a fixed composition by bed construction, so it forfeits utility without buying
  selective risk.
- ``overconfident_score``: squash the confidence toward certainty before thresholding, so every task
  clears the bar and the arm behaves like always_answer while claiming calibration.
- ``frozen_uniform``: hold a frozen uniform confidence exactly at the bar, which never clears it, so
  the arm abstains on everything and forfeits all utility beyond the abstention floor.

Selective risk reduction is one minus the error rate among answered tasks (zero when nothing is
answered; an empty answer set reduces no risk). Decision utility is the mean per-task payoff, +1 for
a correct answer, -PENALTY for a wrong one, 0 for an abstention, mapped to the unit interval. Both
feed the scaffold's DualMetricReading directly.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim. These
readings are arithmetic over a seeded fixture, never a measurement of a real system.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..substrate.events import canonical_sha256
from .calibrated_uncertainty_bed import ANSWER_THRESHOLD, TaskBatch
from .calibrated_uncertainty_scaffold import REQUIRED_CONTROLS, DualMetricReading

PolicyFn = Callable[[TaskBatch], DualMetricReading]

# Payoff of a wrong answer relative to a correct one. PENALTY = 1.0 makes the raw mean payoff live in
# [-1, 1]; the unit mapping below is exact in binary arithmetic for the batch sizes the bed uses.
PENALTY = 1.0

# The overconfident squash: conf' = OVER_BASE + OVER_SPAN * conf lands strictly above the answer
# threshold for every conf in [0, 1], so the arm answers everything while reporting near-certainty.
OVER_BASE = 0.55
OVER_SPAN = 0.45

# The frozen uniform confidence sits exactly at the bar; the strict threshold never clears it.
FROZEN_CONF = 0.5

MECHANISM_ARM = "mechanism"
ARMS: tuple[str, ...] = (MECHANISM_ARM, *REQUIRED_CONTROLS)


class ImplRefusal(ValueError):
    """Raised when a policy is asked to act on a malformed batch or an unknown arm."""


def _reading_from_answers(batch: TaskBatch, answers: Sequence[bool]) -> DualMetricReading:
    """Score one arm's answer decisions on both axes. Pure arithmetic over the batch."""

    if len(answers) != batch.task_count:
        raise ImplRefusal("every task needs exactly one answer decision")
    answered = sum(1 for flag in answers if flag)
    correct_answered = sum(
        1 for flag, bit in zip(answers, batch.correctness, strict=True) if flag and bit == 1
    )
    errors = answered - correct_answered
    if answered == 0:
        risk_reduction = 0.0
    else:
        risk_reduction = 1.0 - errors / answered
    raw_utility = (correct_answered - PENALTY * errors) / batch.task_count
    utility = (raw_utility + PENALTY) / (1.0 + PENALTY)
    return DualMetricReading(
        selective_risk_reduction=min(1.0, max(0.0, risk_reduction)),
        decision_utility=min(1.0, max(0.0, utility)),
    )


# ---------------------------------------------------------------------------
# The mechanism: calibrated selective answering over the honest confidence signal.
# ---------------------------------------------------------------------------


def run_mechanism(batch: TaskBatch) -> DualMetricReading:
    """Answer exactly the tasks whose confidence strictly clears the bar; abstain otherwise."""

    answers = tuple(conf > ANSWER_THRESHOLD for conf in batch.confidence)
    return _reading_from_answers(batch, answers)


# ---------------------------------------------------------------------------
# The control policies. Each fails at least one axis of the decoupled confidence null.
# ---------------------------------------------------------------------------


def run_always_answer(batch: TaskBatch) -> DualMetricReading:
    """Answer every task regardless of confidence. Pays the full base error rate."""

    answers = tuple(True for _ in range(batch.task_count))
    return _reading_from_answers(batch, answers)


def run_random_abstain(batch: TaskBatch) -> DualMetricReading:
    """Abstain on a seeded parity half of the tasks, ignoring confidence entirely."""

    digest = canonical_sha256({"seed": batch.seed, "label": "cu.random_abstain.offset"})
    offset = int(digest[:8], 16) % 2
    answers = tuple((index + offset) % 2 == 1 for index in range(batch.task_count))
    return _reading_from_answers(batch, answers)


def run_overconfident_score(batch: TaskBatch) -> DualMetricReading:
    """Squash confidence toward certainty before thresholding; every task then clears the bar."""

    answers = tuple(OVER_BASE + OVER_SPAN * conf > ANSWER_THRESHOLD for conf in batch.confidence)
    return _reading_from_answers(batch, answers)


def run_frozen_uniform(batch: TaskBatch) -> DualMetricReading:
    """Hold a frozen uniform confidence at the bar; the strict threshold abstains on everything."""

    answers = tuple(FROZEN_CONF > ANSWER_THRESHOLD for _ in range(batch.task_count))
    return _reading_from_answers(batch, answers)


_CONTROL_POLICIES: dict[str, PolicyFn] = {
    "always_answer": run_always_answer,
    "random_abstain": run_random_abstain,
    "overconfident_score": run_overconfident_score,
    "frozen_uniform": run_frozen_uniform,
}


def run_control(control: str, batch: TaskBatch) -> DualMetricReading:
    """Run one named control policy against a batch. Fails closed on an unknown control."""

    policy = _CONTROL_POLICIES.get(control)
    if policy is None:
        raise ImplRefusal(f"unknown control {control!r}")
    return policy(batch)


def run_all(batch: TaskBatch) -> dict[str, DualMetricReading]:
    """Run the mechanism and every declared control against a batch; return readings by arm name."""

    readings: dict[str, DualMetricReading] = {MECHANISM_ARM: run_mechanism(batch)}
    for control in REQUIRED_CONTROLS:
        readings[control] = run_control(control, batch)
    return readings
