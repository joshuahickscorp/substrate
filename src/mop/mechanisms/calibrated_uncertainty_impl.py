
from __future__ import annotations

from collections.abc import Callable, Sequence

from ..substrate.events import canonical_sha256
from .calibrated_uncertainty_bed import ANSWER_THRESHOLD, TaskBatch
from .calibrated_uncertainty_scaffold import REQUIRED_CONTROLS, DualMetricReading

PolicyFn = Callable[[TaskBatch], DualMetricReading]

PENALTY = 1.0

OVER_BASE = 0.55
OVER_SPAN = 0.45

FROZEN_CONF = 0.5

MECHANISM_ARM = "mechanism"
ARMS: tuple[str, ...] = (MECHANISM_ARM, *REQUIRED_CONTROLS)


class ImplRefusal(ValueError):
    pass


def _reading_from_answers(batch: TaskBatch, answers: Sequence[bool]) -> DualMetricReading:

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




def run_mechanism(batch: TaskBatch) -> DualMetricReading:

    answers = tuple(conf > ANSWER_THRESHOLD for conf in batch.confidence)
    return _reading_from_answers(batch, answers)




def run_always_answer(batch: TaskBatch) -> DualMetricReading:

    answers = tuple(True for _ in range(batch.task_count))
    return _reading_from_answers(batch, answers)


def run_random_abstain(batch: TaskBatch) -> DualMetricReading:

    digest = canonical_sha256({"seed": batch.seed, "label": "cu.random_abstain.offset"})
    offset = int(digest[:8], 16) % 2
    answers = tuple((index + offset) % 2 == 1 for index in range(batch.task_count))
    return _reading_from_answers(batch, answers)


def run_overconfident_score(batch: TaskBatch) -> DualMetricReading:

    answers = tuple(OVER_BASE + OVER_SPAN * conf > ANSWER_THRESHOLD for conf in batch.confidence)
    return _reading_from_answers(batch, answers)


def run_frozen_uniform(batch: TaskBatch) -> DualMetricReading:

    answers = tuple(FROZEN_CONF > ANSWER_THRESHOLD for _ in range(batch.task_count))
    return _reading_from_answers(batch, answers)


_CONTROL_POLICIES: dict[str, PolicyFn] = {
    "always_answer": run_always_answer,
    "random_abstain": run_random_abstain,
    "overconfident_score": run_overconfident_score,
    "frozen_uniform": run_frozen_uniform,
}


def run_control(control: str, batch: TaskBatch) -> DualMetricReading:

    policy = _CONTROL_POLICIES.get(control)
    if policy is None:
        raise ImplRefusal(f"unknown control {control!r}")
    return policy(batch)


def run_all(batch: TaskBatch) -> dict[str, DualMetricReading]:

    readings: dict[str, DualMetricReading] = {MECHANISM_ARM: run_mechanism(batch)}
    for control in REQUIRED_CONTROLS:
        readings[control] = run_control(control, batch)
    return readings
