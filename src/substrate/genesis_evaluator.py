"""Sealed evaluator for Substrate Cognitive Material Genesis.

The evaluator holds sealed answers and returns only scalar verdicts. No public
method, ``Verdict``, or ``Answer`` carries an expected value, case identity or
held-out label.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from substrate.genesis_challenge import SealedAnswers
from substrate.genesis_material import Answer, Proposal, Verdict

# Names that would constitute a label leak if present on a public surface.
_FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "expected",
        "expected_value",
        "expected_answer",
        "ground_truth",
        "label",
        "answer_key",
        "sealed_answer",
        "target_answer",
        "correct_answer",
    }
)


def assert_no_expected_value(obj: Any, *, path: str = "root") -> None:
    """Raise AssertionError if ``obj`` carries an expected-label field.

    Applied to ``Verdict``, ``Answer`` and the evaluator's public call surface.
    """
    if obj is None:
        return
    if is_dataclass(obj):
        for field in fields(obj):
            name = field.name
            if name in _FORBIDDEN_LABEL_KEYS or name.lstrip("_") in _FORBIDDEN_LABEL_KEYS:
                # Private sealed storage on SealedAnswers is the single exception
                # and is not reachable through Verdict/Answer/Evaluator public API.
                if path.startswith("SealedAnswers"):
                    continue
                raise AssertionError(f"{path}.{name} carries a forbidden expected-label field")
            if name.startswith("_"):
                continue
            assert_no_expected_value(getattr(obj, name), path=f"{path}.{name}")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_text = str(key)
            if key_text in _FORBIDDEN_LABEL_KEYS:
                raise AssertionError(f"{path}[{key_text!r}] carries a forbidden expected-label field")
            assert_no_expected_value(value, path=f"{path}[{key_text!r}]")
        return
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            assert_no_expected_value(value, path=f"{path}[{index}]")


def _score_delta(before: float, after: float) -> float:
    return float(after) - float(before)


class Evaluator:
    """Holds sealed answers and returns only scalars.

    ``held_out`` is the sealed batch used for improvement. ``retention`` is an
    earlier-split sealed batch that must not degrade when a proposal is admitted.
    """

    def __init__(
        self,
        sealed: SealedAnswers,
        *,
        held_out: SealedAnswers | None = None,
        retention: SealedAnswers | None = None,
    ) -> None:
        if not isinstance(sealed, SealedAnswers):
            raise TypeError("sealed must be a SealedAnswers instance")
        self._sealed = sealed
        self._held_out = held_out if held_out is not None else sealed
        self._retention = retention
        # Structural guarantee: public attribute names do not include label keys.
        public_names = [name for name in dir(self) if not name.startswith("_")]
        leaked = sorted(set(public_names) & _FORBIDDEN_LABEL_KEYS)
        if leaked:
            raise AssertionError(f"Evaluator public surface leaks label keys: {leaked}")
        assert_no_expected_value(Answer(0, (0,)), path="Answer")
        assert_no_expected_value(Verdict("id", False, 0.0, 0.0), path="Verdict")

    def score(self, answers: Sequence[Answer]) -> float:
        """Score answers against the primary sealed batch. Returns a scalar only."""
        return float(self._sealed.score(answers))

    def score_held_out(self, answers: Sequence[Answer]) -> float:
        return float(self._held_out.score(answers))

    def score_retention(self, answers: Sequence[Answer]) -> float:
        if self._retention is None:
            raise RuntimeError("no retention sealed batch configured")
        return float(self._retention.score(answers))

    def judge(
        self,
        proposals: Sequence[Proposal],
        before_answers: Sequence[Answer],
        after_answers: Sequence[Answer],
    ) -> tuple[Verdict, ...]:
        """Admit only when held-out improvement is positive and retention does not fall.

        ``before_answers`` / ``after_answers`` are concatenated
        ``held_out || retention`` answer sequences when a retention batch is
        configured; otherwise they are scored entirely against the held-out
        sealed batch and retention is reported as 1.0 (no earlier competence to
        lose).
        """
        held_n = self._held_out.n_probes()
        before = list(before_answers)
        after = list(after_answers)

        if self._retention is None:
            held_before = before
            held_after = after
            retention_before = 1.0
            retention_after = 1.0
        else:
            ret_n = self._retention.n_probes()
            if len(before) < held_n + ret_n or len(after) < held_n + ret_n:
                raise ValueError("before/after answers must cover held_out and retention probes")
            held_before = before[:held_n]
            held_after = after[:held_n]
            ret_before = before[held_n : held_n + ret_n]
            ret_after = after[held_n : held_n + ret_n]
            retention_before = self._retention.score(ret_before)
            retention_after = self._retention.score(ret_after)

        score_before = self._held_out.score(held_before)
        score_after = self._held_out.score(held_after)
        improvement = _score_delta(score_before, score_after)
        admitted = improvement > 0.0 and retention_after + 1e-15 >= retention_before

        verdicts: list[Verdict] = []
        for proposal in proposals:
            verdict = Verdict(
                proposal_id=proposal.proposal_id,
                admitted=admitted,
                improvement=float(improvement),
                retention=float(retention_after),
            )
            assert_no_expected_value(verdict, path="Verdict")
            verdicts.append(verdict)
        return tuple(verdicts)


__all__ = [
    "Evaluator",
    "assert_no_expected_value",
]
