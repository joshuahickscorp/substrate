"""Deterministic allocators for the reducible novelty bed: the mechanism and its control policies.

This module is runnable machinery. It implements a tiny, fully deterministic probe economy and five
arms measured on a source panel:

- ``mechanism`` (reducible novelty detector): spend one pilot probe on every source, observe the
  per-probe learning progress each pilot yields, then commit the remaining budget EQUALLY across the
  sources whose pilot showed strictly positive progress. Pure noise yields exactly zero pilot
  progress, so the mechanism walks away from it; if nothing shows progress the remainder falls back
  to a uniform spread (curiosity finds nothing reducible and says so).
- ``uniform_allocation``: split the budget equally across all sources, structure or not.
- ``random_allocation``: split the budget by seeded arbitrary weights in [1.0, 1.5) per source. The
  band is part of the control's declaration: it is an arbitrary allocator, not an adversarial one,
  and the band keeps its per-source spend provably below the mechanism's committed spend.
- ``novelty_chaser``: split the budget proportional to raw novelty. Raw novelty includes the loud
  irreducible noise floors, so on a mixed panel this control is pulled toward the unlearnable
  sources; this is the irreducible noise trap made concrete.
- ``static_prior``: ignore the panel and spend the whole budget on a fixed pair of favorite slots.

Learning progress of an allocation is the total reducible error actually removed, normalized by the
total removable error; allocation efficiency is the fraction of the budget spent on sources that a
probe can actually improve. Both live in the unit interval, so they feed the scaffold's
DualMetricReading directly. On a null panel the removable error is zero, so every arm scores zero
progress and zero efficiency and a strict joint win is impossible by construction.

Margin arithmetic for the favorable regime (the by-construction guarantee, for every nonnegative
seed): the mechanism commits 1 + (40 - 8) / 4 = 9.0 probes to every reducible source and 36 of 40
probes overall to reducible sources (efficiency 0.9 exactly). Per reducible source the controls
spend strictly less: uniform 5.0; random below 40 * 1.5 / 8.5 = 7.06; novelty chaser below
40 * 1.35 / 12.65 = 4.27 (reducible novelty tops out at 1.35 while each of the four noise floors is
at least 2.0). Progress per source is monotone in probes at a fixed decay, so the mechanism strictly
wins learning progress against those three, and against static_prior it wins on the aggregate:
static_prior reaches at most one reducible source, capping its progress at 1.2 / 4.2 = 0.286, while
the mechanism's nine probes per source guarantee at least 1 - 0.65 ** 9 = 0.979. Efficiency of the
controls is capped at 0.5 (uniform), 6 / 10 = 0.6 (random band), 5.4 / 13.4 = 0.403 (chaser), and
0.5 (static pair), all comfortably below 0.9.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim. These
readings are arithmetic over a seeded fixture, never a measurement of a real system.

House style: no em dashes and no en dashes. Use commas, semicolons, or "vs".
"""

from __future__ import annotations

from collections.abc import Callable

from ..substrate.events import canonical_sha256
from .reducible_novelty_bed import PILOT_PROBES_PER_SOURCE, SourcePanel
from .reducible_novelty_scaffold import REQUIRED_CONTROLS, DualMetricReading

Allocation = tuple[float, ...]
PolicyFn = Callable[[SourcePanel], Allocation]

# Seeded band of the random_allocation control weights: 1.0 + RANDOM_WEIGHT_SPAN * unit, unit in
# [0, 1). The span is load-bearing for the margin arithmetic in the module docstring.
RANDOM_WEIGHT_SPAN = 0.5

# The fixed favorite slots of the static_prior control. Chosen once, blind to any panel.
STATIC_PRIOR_SLOTS: tuple[int, ...] = (0, 1)

# Pilot progress below this threshold counts as "nothing reducible here". Pure noise yields exactly
# zero, so the threshold only guards against pathological float dust.
PROGRESS_DETECTION_THRESHOLD = 1e-12

# Tolerance for the matched-spend check: every arm must spend the panel budget exactly, up to float
# accumulation error.
BUDGET_TOLERANCE = 1e-6

MECHANISM_ARM = "mechanism"
ARMS: tuple[str, ...] = (MECHANISM_ARM, *REQUIRED_CONTROLS)


class ImplRefusal(ValueError):
    """Raised when an allocator is asked to act on a malformed panel or an unknown arm."""


def _unit(seed: int, label: str) -> float:
    """A deterministic value in [0, 1) from a seeded digest; no wall clock, no rng."""

    if seed < 0:
        raise ImplRefusal("allocator seed must be nonnegative")
    digest = canonical_sha256({"seed": seed, "label": label})
    return int(digest[:8], 16) / 0x1_0000_0000


def _check_allocation(panel: SourcePanel, allocation: Allocation) -> Allocation:
    """Fail closed unless the allocation is nonnegative and spends the panel budget exactly."""

    if len(allocation) != panel.source_count:
        raise ImplRefusal("allocation must cover every source exactly once")
    for value in allocation:
        if value < 0.0:
            raise ImplRefusal("allocation entries must be nonnegative")
    if abs(sum(allocation) - panel.probe_budget) > BUDGET_TOLERANCE:
        raise ImplRefusal("allocation does not spend the matched probe budget exactly")
    return allocation


def _source_progress(panel: SourcePanel, index: int, probes: float) -> float:
    """Reducible error removed from one source by ``probes`` probes: signal * (1 - decay ** probes)."""

    if probes <= 0.0:
        return 0.0
    return panel.signals[index] * (1.0 - panel.decays[index] ** probes)


def total_progress(panel: SourcePanel, allocation: Allocation) -> float:
    """Total reducible error removed across the panel by an allocation."""

    return sum(
        _source_progress(panel, index, allocation[index]) for index in range(panel.source_count)
    )


def learning_progress(panel: SourcePanel, allocation: Allocation) -> float:
    """Removed reducible error over removable reducible error, in [0, 1). Zero on a pure-noise panel."""

    removable = sum(panel.signals)
    if removable <= 0.0:
        return 0.0
    ratio = total_progress(panel, allocation) / removable
    return min(1.0, max(0.0, ratio))


def allocation_efficiency(panel: SourcePanel, allocation: Allocation) -> float:
    """Budget fraction spent on sources a probe can improve, in [0, 1]. Zero on a pure-noise panel."""

    productive = sum(allocation[index] for index in panel.reducible_sources)
    ratio = productive / panel.probe_budget
    return min(1.0, max(0.0, ratio))


# ---------------------------------------------------------------------------
# The mechanism: pilot every source, then commit the remainder to observed learning progress.
# ---------------------------------------------------------------------------


def allocate_mechanism(panel: SourcePanel) -> Allocation:
    """Pilot each source once, then split the remaining budget equally over the progressing sources."""

    pilot_total = PILOT_PROBES_PER_SOURCE * panel.source_count
    if pilot_total >= panel.probe_budget:
        raise ImplRefusal("panel budget too small to pilot every source")
    pilot_gains = tuple(
        _source_progress(panel, index, PILOT_PROBES_PER_SOURCE)
        for index in range(panel.source_count)
    )
    detected = tuple(
        index
        for index in range(panel.source_count)
        if pilot_gains[index] > PROGRESS_DETECTION_THRESHOLD
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


# ---------------------------------------------------------------------------
# The control policies. Each falls into one side of the irreducible noise trap.
# ---------------------------------------------------------------------------


def allocate_uniform(panel: SourcePanel) -> Allocation:
    """Split the budget equally across all sources, learnable or not."""

    share = panel.probe_budget / float(panel.source_count)
    return _check_allocation(panel, tuple(share for _ in range(panel.source_count)))


def allocate_random(panel: SourcePanel) -> Allocation:
    """Split the budget by seeded arbitrary weights in the declared [1.0, 1.5) band."""

    weights = tuple(
        1.0 + RANDOM_WEIGHT_SPAN * _unit(panel.seed, f"random.weight.{index}")
        for index in range(panel.source_count)
    )
    total = sum(weights)
    return _check_allocation(
        panel, tuple(panel.probe_budget * weight / total for weight in weights)
    )


def allocate_novelty_chaser(panel: SourcePanel) -> Allocation:
    """Split the budget proportional to raw novelty, irreducible noise included."""

    novelties = panel.novelties
    total = sum(novelties)
    if total <= 0.0:
        raise ImplRefusal("novelty chaser needs a panel with positive raw novelty")
    return _check_allocation(
        panel, tuple(panel.probe_budget * novelty / total for novelty in novelties)
    )


def allocate_static_prior(panel: SourcePanel) -> Allocation:
    """Spend the whole budget on the fixed favorite slots, blind to the panel."""

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
    """Measure the mechanism arm on a panel: both axes from its pilot-then-commit allocation."""

    return _reading(panel, allocate_mechanism(panel))


def run_control(control: str, panel: SourcePanel) -> DualMetricReading:
    """Run one named control policy against a panel. Fails closed on an unknown control."""

    policy = _CONTROL_POLICIES.get(control)
    if policy is None:
        raise ImplRefusal(f"unknown control {control!r}")
    return _reading(panel, policy(panel))


def run_all(panel: SourcePanel) -> dict[str, DualMetricReading]:
    """Run the mechanism and every declared control against a panel; return readings by arm name."""

    readings: dict[str, DualMetricReading] = {MECHANISM_ARM: run_mechanism(panel)}
    for control in REQUIRED_CONTROLS:
        readings[control] = run_control(control, panel)
    return readings
