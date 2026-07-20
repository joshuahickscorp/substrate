from __future__ import annotations

from collections.abc import Callable

from ..substrate.events import canonical_sha256
from .joint_axis_runner import run_control_policy, run_policy_family
from .reducible_novelty_bed import PILOT_PROBES_PER_SOURCE, SourcePanel
from .reducible_novelty_scaffold import REQUIRED_CONTROLS, DualMetricReading

Allocation = tuple[float, ...]
PolicyFn = Callable[[SourcePanel], Allocation]

RANDOM_WEIGHT_SPAN = 0.5

STATIC_PRIOR_SLOTS: tuple[int, ...] = (0, 1)

PROGRESS_DETECTION_THRESHOLD = 1e-12

BUDGET_TOLERANCE = 1e-6

MECHANISM_ARM = "mechanism"
ARMS: tuple[str, ...] = (MECHANISM_ARM, *REQUIRED_CONTROLS)


class ImplRefusal(ValueError):
    pass


def _unit(seed: int, label: str) -> float:

    if seed < 0:
        raise ImplRefusal("allocator seed must be nonnegative")
    digest = canonical_sha256({"seed": seed, "label": label})
    return int(digest[:8], 16) / 0x1_0000_0000


def _check_allocation(panel: SourcePanel, allocation: Allocation) -> Allocation:

    if len(allocation) != panel.source_count:
        raise ImplRefusal("allocation must cover every source exactly once")
    for value in allocation:
        if value < 0.0:
            raise ImplRefusal("allocation entries must be nonnegative")
    if abs(sum(allocation) - panel.probe_budget) > BUDGET_TOLERANCE:
        raise ImplRefusal("allocation does not spend the matched probe budget exactly")
    return allocation


def _source_progress(panel: SourcePanel, index: int, probes: float) -> float:

    if probes <= 0.0:
        return 0.0
    return panel.signals[index] * (1.0 - panel.decays[index] ** probes)


def total_progress(panel: SourcePanel, allocation: Allocation) -> float:

    return sum(_source_progress(panel, index, allocation[index]) for index in range(panel.source_count))


def learning_progress(panel: SourcePanel, allocation: Allocation) -> float:

    removable = sum(panel.signals)
    if removable <= 0.0:
        return 0.0
    ratio = total_progress(panel, allocation) / removable
    return min(1.0, max(0.0, ratio))


def allocation_efficiency(panel: SourcePanel, allocation: Allocation) -> float:

    productive = sum(allocation[index] for index in panel.reducible_sources)
    ratio = productive / panel.probe_budget
    return min(1.0, max(0.0, ratio))


def allocate_mechanism(panel: SourcePanel) -> Allocation:

    pilot_total = PILOT_PROBES_PER_SOURCE * panel.source_count
    if pilot_total >= panel.probe_budget:
        raise ImplRefusal("panel budget too small to pilot every source")
    pilot_gains = tuple(
        _source_progress(panel, index, PILOT_PROBES_PER_SOURCE) for index in range(panel.source_count)
    )
    detected = tuple(
        index for index in range(panel.source_count) if pilot_gains[index] > PROGRESS_DETECTION_THRESHOLD
    )
    remainder = panel.probe_budget - pilot_total
    values = [PILOT_PROBES_PER_SOURCE for _ in range(panel.source_count)]
    if detected:
        per_source = remainder / float(len(detected))
        for index in detected:
            values[index] += per_source
    else:
        per_source = remainder / float(panel.source_count)
        for index in range(panel.source_count):
            values[index] += per_source
    return _check_allocation(panel, tuple(values))


def allocate_uniform(panel: SourcePanel) -> Allocation:

    share = panel.probe_budget / float(panel.source_count)
    return _check_allocation(panel, tuple(share for _ in range(panel.source_count)))


def allocate_random(panel: SourcePanel) -> Allocation:

    weights = tuple(
        1.0 + RANDOM_WEIGHT_SPAN * _unit(panel.seed, f"random.weight.{index}")
        for index in range(panel.source_count)
    )
    total = sum(weights)
    return _check_allocation(panel, tuple(panel.probe_budget * weight / total for weight in weights))


def allocate_novelty_chaser(panel: SourcePanel) -> Allocation:

    novelties = panel.novelties
    total = sum(novelties)
    if total <= 0.0:
        raise ImplRefusal("novelty chaser needs a panel with positive raw novelty")
    return _check_allocation(panel, tuple(panel.probe_budget * novelty / total for novelty in novelties))


def allocate_static_prior(panel: SourcePanel) -> Allocation:

    for slot in STATIC_PRIOR_SLOTS:
        if slot >= panel.source_count:
            raise ImplRefusal("static prior slot outside the panel")
    share = panel.probe_budget / float(len(STATIC_PRIOR_SLOTS))
    values = [0.0 for _ in range(panel.source_count)]
    for slot in STATIC_PRIOR_SLOTS:
        values[slot] = share
    return _check_allocation(panel, tuple(values))


_CONTROL_POLICIES: dict[str, PolicyFn] = {
    "uniform_allocation": allocate_uniform,
    "random_allocation": allocate_random,
    "novelty_chaser": allocate_novelty_chaser,
    "static_prior": allocate_static_prior,
}


def _reading(panel: SourcePanel, allocation: Allocation) -> DualMetricReading:
    return DualMetricReading(
        learning_progress=learning_progress(panel, allocation),
        allocation_efficiency=allocation_efficiency(panel, allocation),
    )


def run_mechanism(panel: SourcePanel) -> DualMetricReading:

    return _reading(panel, allocate_mechanism(panel))


def run_control(control: str, panel: SourcePanel) -> DualMetricReading:
    allocation = run_control_policy(control, panel, _CONTROL_POLICIES, ImplRefusal)
    return _reading(panel, allocation)


def run_all(panel: SourcePanel) -> dict[str, DualMetricReading]:
    return run_policy_family(panel, run_mechanism, REQUIRED_CONTROLS, run_control)
