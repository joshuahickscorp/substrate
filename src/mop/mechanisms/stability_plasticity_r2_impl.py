from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from .joint_axis_runner import run_control_policy, run_policy_family
from .stability_plasticity_r2_bed import CORE_DIM, DIM, TaskStream
from .stability_plasticity_r2_scaffold import REQUIRED_CONTROLS, DualMetricReading

Vector = tuple[float, ...]
PolicyFn = Callable[[TaskStream], DualMetricReading]

FIT_BASE = 0.5
FIT_BUDGET = 13.0

SIMILARITY_BETA = 2.4

CORE_DIMS: tuple[int, ...] = tuple(range(CORE_DIM))
ADAPTER_DIMS: tuple[int, ...] = tuple(range(CORE_DIM, DIM))
ALL_DIMS: tuple[int, ...] = tuple(range(DIM))

MECHANISM_ARM = "mechanism"
ARMS: tuple[str, ...] = (MECHANISM_ARM, *REQUIRED_CONTROLS)

ZEROS: Vector = tuple(0.0 for _ in range(DIM))


class ImplRefusal(ValueError):
    pass


def _residual(active_count: int) -> float:

    if active_count <= 0:
        return 1.0
    return float(FIT_BASE ** (FIT_BUDGET / active_count))


def _gd(start: Vector, target: Vector, active: Sequence[int], residual: float) -> Vector:

    values = list(start)
    for dim in active:
        values[dim] = target[dim] + residual * (start[dim] - target[dim])
    return tuple(values)


def _combine(core_source: Vector, adapter_source: Vector) -> Vector:

    return tuple(core_source[dim] if dim < CORE_DIM else adapter_source[dim] for dim in range(DIM))


def _mean_vectors(vectors: Iterable[Vector]) -> Vector:

    collected = list(vectors)
    if not collected:
        raise ImplRefusal("cannot average an empty set of vectors")
    count = float(len(collected))
    return tuple(sum(vector[dim] for vector in collected) / count for dim in range(DIM))


def _mse(left: Vector, right: Vector) -> float:
    return sum((left[dim] - right[dim]) ** 2 for dim in range(DIM)) / float(DIM)


def _score(reconstruction: Vector, target: Vector) -> float:

    return math.exp(-SIMILARITY_BETA * _mse(reconstruction, target))


def _retention(reconstructions: Sequence[Vector], stream: TaskStream) -> float:
    scores = [_score(reconstructions[index], task) for index, task in enumerate(stream.history)]
    return sum(scores) / float(len(scores))


def _reading(retention: float, future_learnability: float) -> DualMetricReading:
    return DualMetricReading(
        retention=min(1.0, max(0.0, retention)),
        future_learnability=min(1.0, max(0.0, future_learnability)),
    )


def run_mechanism(stream: TaskStream) -> DualMetricReading:

    core_state = _gd(ZEROS, stream.history[0], CORE_DIMS, _residual(len(CORE_DIMS)))
    adapter_residual = _residual(len(ADAPTER_DIMS))
    adapter_memory: list[Vector] = []
    for index, task in enumerate(stream.history):
        fitted = _gd(ZEROS, task, ADAPTER_DIMS, adapter_residual)
        if stream.recurrence_flags[index]:
            fitted = _gd(fitted, task, ADAPTER_DIMS, adapter_residual)
        adapter_memory.append(fitted)
    reconstructions = [_combine(core_state, adapter_memory[index]) for index in range(len(stream.history))]
    retention = _retention(reconstructions, stream)
    if stream.future_recurrence_index >= 0:
        warm_start = adapter_memory[stream.future_recurrence_index]
        future_adapter = _gd(warm_start, stream.future, ADAPTER_DIMS, adapter_residual)
    else:
        future_adapter = _gd(ZEROS, stream.future, ADAPTER_DIMS, adapter_residual)
    future_reconstruction = _combine(core_state, future_adapter)
    future = _score(future_reconstruction, stream.future)
    return _reading(retention, future)


def run_fresh_init(stream: TaskStream) -> DualMetricReading:

    end = ZEROS
    for task in stream.history:
        end = _gd(ZEROS, task, ALL_DIMS, _residual(len(ALL_DIMS)))
    reconstructions = [end for _ in stream.history]
    retention = _retention(reconstructions, stream)
    future_state = _gd(end, stream.future, ALL_DIMS, _residual(len(ALL_DIMS)))
    future = _score(future_state, stream.future)
    return _reading(retention, future)


def run_frozen_core(stream: TaskStream) -> DualMetricReading:

    state = _gd(ZEROS, stream.history[0], ALL_DIMS, _residual(len(ALL_DIMS)))
    reconstructions = [state for _ in stream.history]
    retention = _retention(reconstructions, stream)
    future = _score(state, stream.future)
    return _reading(retention, future)


def run_full_retrain(stream: TaskStream) -> DualMetricReading:

    history_mean = _mean_vectors(stream.history)
    state = _gd(ZEROS, history_mean, ALL_DIMS, _residual(len(ALL_DIMS)))
    reconstructions = [state for _ in stream.history]
    retention = _retention(reconstructions, stream)
    all_mean = _mean_vectors((*stream.history, stream.future))
    future_state = _gd(ZEROS, all_mean, ALL_DIMS, _residual(len(ALL_DIMS)))
    future = _score(future_state, stream.future)
    return _reading(retention, future)


def run_no_replay(stream: TaskStream) -> DualMetricReading:

    state = ZEROS
    for task in stream.history:
        state = _gd(state, task, ALL_DIMS, _residual(len(ALL_DIMS)))
    reconstructions = [state for _ in stream.history]
    retention = _retention(reconstructions, stream)
    future_state = _gd(state, stream.future, ALL_DIMS, _residual(len(ALL_DIMS)))
    future = _score(future_state, stream.future)
    return _reading(retention, future)


_CONTROL_POLICIES: dict[str, PolicyFn] = {
    "fresh_init": run_fresh_init,
    "frozen_core": run_frozen_core,
    "full_retrain": run_full_retrain,
    "no_replay": run_no_replay,
}


def run_control(control: str, stream: TaskStream) -> DualMetricReading:
    return run_control_policy(control, stream, _CONTROL_POLICIES, ImplRefusal)


def run_all(stream: TaskStream) -> dict[str, DualMetricReading]:
    return run_policy_family(stream, run_mechanism, REQUIRED_CONTROLS, run_control)
