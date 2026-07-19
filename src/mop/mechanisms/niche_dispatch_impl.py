
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from .niche_dispatch_scaffold import DISPATCH_CONTROLS

LABEL_TRUE = 0
LABEL_WRONG = 1
LABEL_ABSTAIN = -1

CONTROL_ALL_PERSPECTIVES = "all-perspectives"
CONTROL_RANDOM_DISPATCH = "random-dispatch"
CONTROL_SINGLE_BEST = "single-best"
CONTROL_MAJORITY_VOTE = "majority-vote"


class NicheDispatchImplError(ValueError):
    pass


def _perspective_id(index: int) -> str:
    return f"perspective.{index:02d}"


def _cell_id(index: int) -> str:
    return f"cell.{index:03d}"


def _round(value: float) -> float:

    return round(value, 6)


@dataclass(frozen=True, slots=True)
class RegimeData:

    regime: str
    perspective_ids: tuple[str, ...]
    context_cells: tuple[str, ...]
    true_labels: tuple[int, ...]
    owner_index_per_context: tuple[int, ...]
    predictions: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        n_contexts = len(self.context_cells)
        if n_contexts == 0:
            raise NicheDispatchImplError("a regime needs at least one context")
        if len(self.true_labels) != n_contexts:
            raise NicheDispatchImplError("true labels must align with contexts")
        if len(self.owner_index_per_context) != n_contexts:
            raise NicheDispatchImplError("owner indices must align with contexts")
        n_perspectives = len(self.perspective_ids)
        if n_perspectives < 3:
            raise NicheDispatchImplError("a niche-dispatch regime needs at least three perspectives")
        if len(self.predictions) != n_perspectives:
            raise NicheDispatchImplError("predictions must align with perspectives")
        for row in self.predictions:
            if len(row) != n_contexts:
                raise NicheDispatchImplError("each perspective must predict on every context")
        for owner in self.owner_index_per_context:
            if not 0 <= owner < n_perspectives:
                raise NicheDispatchImplError("owner index out of range for the perspective set")

    @property
    def n_contexts(self) -> int:
        return len(self.context_cells)

    @property
    def n_perspectives(self) -> int:
        return len(self.perspective_ids)


def _make_regime(
    *,
    regime: str,
    n_perspectives: int,
    contexts_per_cell: int,
    disjoint: bool,
) -> RegimeData:

    if n_perspectives < 3:
        raise NicheDispatchImplError("a niche-dispatch regime needs at least three perspectives")
    if contexts_per_cell < 1:
        raise NicheDispatchImplError("each cell needs at least one context")
    perspective_ids = tuple(_perspective_id(i) for i in range(n_perspectives))
    context_cells: list[str] = []
    true_labels: list[int] = []
    owner_index_per_context: list[int] = []
    for cell in range(n_perspectives):
        for _ in range(contexts_per_cell):
            context_cells.append(_cell_id(cell))
            true_labels.append(LABEL_TRUE)
            owner_index_per_context.append(cell if disjoint else 0)
    predictions: list[tuple[int, ...]] = []
    for j in range(n_perspectives):
        row: list[int] = []
        for context_index in range(len(context_cells)):
            if disjoint:
                owner = owner_index_per_context[context_index]
                row.append(LABEL_TRUE if j == owner else LABEL_WRONG)
            else:
                row.append(LABEL_WRONG if j == n_perspectives - 1 else LABEL_TRUE)
        predictions.append(tuple(row))
    return RegimeData(
        regime=regime,
        perspective_ids=perspective_ids,
        context_cells=tuple(context_cells),
        true_labels=tuple(true_labels),
        owner_index_per_context=tuple(owner_index_per_context),
        predictions=tuple(predictions),
    )


def build_disjoint_regime(regime: str, *, n_perspectives: int, contexts_per_cell: int) -> RegimeData:

    return _make_regime(
        regime=regime,
        n_perspectives=n_perspectives,
        contexts_per_cell=contexts_per_cell,
        disjoint=True,
    )


def build_overlapping_regime(regime: str, *, n_perspectives: int, contexts_per_cell: int) -> RegimeData:

    return _make_regime(
        regime=regime,
        n_perspectives=n_perspectives,
        contexts_per_cell=contexts_per_cell,
        disjoint=False,
    )


def _accuracy(labels: Sequence[int], true_labels: Sequence[int]) -> float:
    correct = sum(1 for predicted, truth in zip(labels, true_labels, strict=True) if predicted == truth)
    return _round(correct / len(true_labels))


def _argmax_index(scores: Sequence[float]) -> int:

    best_index = 0
    best_score = scores[0]
    for index in range(1, len(scores)):
        if scores[index] > best_score:
            best_index = index
            best_score = scores[index]
    return best_index


def dispatch_labels(data: RegimeData) -> tuple[int, ...]:

    return tuple(
        data.predictions[data.owner_index_per_context[i]][i] for i in range(data.n_contexts)
    )


def _all_perspectives_labels(data: RegimeData) -> tuple[int, ...]:

    labels: list[int] = []
    for i in range(data.n_contexts):
        column = {row[i] for row in data.predictions}
        labels.append(next(iter(column)) if len(column) == 1 else LABEL_ABSTAIN)
    return tuple(labels)


def _random_dispatch_labels(data: RegimeData, seed: int) -> tuple[int, ...]:

    labels: list[int] = []
    for i in range(data.n_contexts):
        token = f"{seed}:{data.regime}:{i}".encode("ascii")
        draw = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
        chosen = draw % data.n_perspectives
        labels.append(data.predictions[chosen][i])
    return tuple(labels)


def _single_best_labels(data: RegimeData) -> tuple[int, ...]:

    per_perspective = [_accuracy(row, data.true_labels) for row in data.predictions]
    best = _argmax_index(per_perspective)
    return data.predictions[best]


def _majority_vote_labels(data: RegimeData) -> tuple[int, ...]:

    labels: list[int] = []
    for i in range(data.n_contexts):
        counts: dict[int, int] = {}
        for row in data.predictions:
            counts[row[i]] = counts.get(row[i], 0) + 1
        top = max(counts.values())
        labels.append(min(label for label, count in counts.items() if count == top))
    return tuple(labels)


def dispatch_accuracy(data: RegimeData) -> float:
    return _accuracy(dispatch_labels(data), data.true_labels)


def control_accuracies(data: RegimeData, seed: int) -> dict[str, float]:

    scores = {
        CONTROL_ALL_PERSPECTIVES: _accuracy(_all_perspectives_labels(data), data.true_labels),
        CONTROL_RANDOM_DISPATCH: _accuracy(_random_dispatch_labels(data, seed), data.true_labels),
        CONTROL_SINGLE_BEST: _accuracy(_single_best_labels(data), data.true_labels),
        CONTROL_MAJORITY_VOTE: _accuracy(_majority_vote_labels(data), data.true_labels),
    }
    if set(scores) != set(DISPATCH_CONTROLS):
        raise NicheDispatchImplError("control scores must cover exactly the declared control family")
    return scores
