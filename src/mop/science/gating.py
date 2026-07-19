"""Provider-neutral causal gate lifecycle shared by experiment families."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def assemble_causal_inputs(
    features: np.ndarray, state_factory: Callable[[], Any]
) -> np.ndarray:
    """Append a label-free causal state vector to each frozen feature frame."""

    state = state_factory()
    rows: list[np.ndarray] = []
    for frame in range(features.shape[0]):
        rows.append(np.concatenate([features[frame], state.to_vector()]))
        state = state.update(features[frame], 0.0, False)
    return np.asarray(rows, dtype=np.float64)


def causal_gate_trace(
    gate: Any,
    features: np.ndarray,
    theta: float,
    state_factory: Callable[[], Any],
) -> tuple[list[int], np.ndarray]:
    """Run one gate causally and return event frames plus its probability trace."""

    state = state_factory()
    events: list[int] = []
    probabilities = np.empty(features.shape[0], dtype=np.float64)
    for frame in range(features.shape[0]):
        probability = gate.infer(features[frame], state)
        probabilities[frame] = probability
        event = probability >= theta
        if event:
            events.append(frame)
        state = state.update(features[frame], probability, event)
    return events, probabilities


__all__ = ["assemble_causal_inputs", "causal_gate_trace"]
