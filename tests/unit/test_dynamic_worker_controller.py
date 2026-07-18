"""Behavior and safety tests for the dynamic worker controller (advisory sizer).

The worker count is receipt-invariant, so these tests assert the OPERATIONAL control law, not
any evidence property: the pool backs off to a tiny reserve under Hawking, ramps to the measured
throughput-peak ceiling of 20 when the host is idle, stays bounded by cores and by memory at the
right values, never returns 24, is deterministic given the host state, does not oscillate on a
jittery load trace, and always carries a priority/nice recommendation that yields under Hawking.
The decision core is pure, so every property is checked with synthetic host states and no psutil.
"""

from __future__ import annotations

import pytest

from mop.studio.dynamic_worker_controller import (
    DEFAULT_POLICY,
    HAWKING_RESERVE_WORKERS,
    MEASURED_AGGREGATE_MHS,
    WORKER_CEILING,
    WORKER_FLOOR,
    HostState,
    PriorityAdvice,
    WorkerControllerRefused,
    WorkerPolicy,
    recommended_priority,
    recommended_workers,
    worker_bounds,
)


def idle_state(**overrides: object) -> HostState:
    """A comfortable, fully idle host: all cores free, memory ample, on AC, thermal normal."""

    base: dict[str, object] = dict(
        free_p_cores=28,
        hawking_active=False,
        mem_available_gb=90.0,
        thermal_ok=True,
        on_ac=True,
        current_load=1.0,
        current_workers=0,
    )
    base.update(overrides)
    return HostState(**base)  # type: ignore[arg-type]


def settle(state_fn, *, start: int = 0, ticks: int = 60, policy: WorkerPolicy = DEFAULT_POLICY):
    """Thread current_workers through recommended_workers, returning the whole output trace.

    state_fn(tick, current) -> HostState so a test can drive a jittery input while the controller
    carries its own hysteresis state through current_workers.
    """

    trace: list[int] = []
    current = start
    for tick in range(ticks):
        current = recommended_workers(state_fn(tick, current), policy)
        trace.append(current)
    return trace


# ----------------------------------------------------------------------------------
# Hawking back off
# ----------------------------------------------------------------------------------


def test_backs_off_to_reserve_under_hawking() -> None:
    # A full pool collapses to the reserve in a single tick (fast backoff), from any size.
    for current in (0, 1, 8, 16, 20):
        state = idle_state(hawking_active=True, current_workers=current)
        assert recommended_workers(state) == HAWKING_RESERVE_WORKERS == 2


def test_hawking_backoff_is_immediate_not_ramped() -> None:
    # Even from the ceiling, one tick reaches the reserve; it does not step down one per tick.
    state = idle_state(hawking_active=True, current_workers=WORKER_CEILING)
    assert recommended_workers(state) == HAWKING_RESERVE_WORKERS


# ----------------------------------------------------------------------------------
# Idle ramp to the ceiling
# ----------------------------------------------------------------------------------


def test_ramps_to_ceiling_when_idle() -> None:
    trace = settle(lambda tick, current: idle_state(current_workers=current), start=0, ticks=60)
    assert trace[-1] == WORKER_CEILING == 20
    # Ramp UP is slow: never more than one worker added per tick.
    for previous, nxt in zip(trace, trace[1:], strict=False):
        assert nxt - previous <= DEFAULT_POLICY.ramp_up_step
    # It climbs monotonically to the ceiling and then holds there.
    assert trace == sorted(trace)
    assert trace[20:] == [20] * len(trace[20:])


def test_ceiling_holds_and_never_returns_twenty_four() -> None:
    # Absurdly large host: the ceiling binds; the pool never exceeds 20 and never reaches 24.
    for current in range(0, 25):
        state = idle_state(free_p_cores=200, mem_available_gb=10_000.0, current_workers=current)
        assert recommended_workers(state) <= WORKER_CEILING == 20
    assert recommended_workers(idle_state(free_p_cores=200, current_workers=19)) == 20
    assert recommended_workers(idle_state(free_p_cores=200, current_workers=24)) == 20


def test_measured_peak_is_twenty_not_twenty_four() -> None:
    assert MEASURED_AGGREGATE_MHS[20] > MEASURED_AGGREGATE_MHS[24]
    assert max(MEASURED_AGGREGATE_MHS, key=MEASURED_AGGREGATE_MHS.get) == WORKER_CEILING


# ----------------------------------------------------------------------------------
# Core and memory bounds
# ----------------------------------------------------------------------------------


def test_core_bounded_at_the_right_value() -> None:
    # 8 free cores minus the 4-core reserve = 4 workers; memory and ceiling are slack.
    state = idle_state(free_p_cores=8, mem_available_gb=90.0, current_workers=20)
    assert recommended_workers(state) == 4
    bounds = worker_bounds(state)
    assert bounds["core_bound"] == 4
    assert bounds["binding_constraint"] == "core_bound"
    assert bounds["comfortable_target"] == 4


def test_memory_bounded_at_the_right_value() -> None:
    # 3.0 GB available / 0.75 GB per worker = 4 workers; cores and ceiling are slack.
    state = idle_state(free_p_cores=28, mem_available_gb=3.0, current_workers=20)
    assert recommended_workers(state) == 4
    bounds = worker_bounds(state)
    assert bounds["memory_bound"] == 4
    assert bounds["binding_constraint"] == "memory_bound"


def test_floor_is_honored_when_resources_are_starved() -> None:
    # Memory near zero would compute a zero bound; the floor of 1 still holds (throttle owns OOM).
    state = idle_state(free_p_cores=28, mem_available_gb=0.1, current_workers=20)
    assert recommended_workers(state) == WORKER_FLOOR == 1


# ----------------------------------------------------------------------------------
# Hysteresis: no oscillation on a jittery load trace
# ----------------------------------------------------------------------------------


def test_hysteresis_prevents_oscillation_on_jittery_load() -> None:
    # Free cores jitter every tick between 10 and 12, so the raw comfortable target alternates
    # between 6 and 8. A naive sizer would flap 6 <-> 8 forever; the deadband must settle it.
    def jitter(tick: int, current: int) -> HostState:
        free = 12 if tick % 2 == 0 else 10
        return idle_state(free_p_cores=free, current_workers=current)

    trace = settle(jitter, start=0, ticks=60)
    raw = [worker_bounds(jitter(tick, 0))["comfortable_target"] for tick in range(60)]

    # The input really is jittery.
    assert set(raw) == {6, 8}
    # The settled tail is a single constant value: no oscillation.
    tail = trace[-10:]
    assert len(set(tail)) == 1
    assert tail[0] == 8  # it holds at the top of the jitter band, absorbing the dips to 6


def test_small_dips_are_held_but_real_drops_back_off_fast() -> None:
    # Within the deadband: a 2-worker comfortable dip from a settled pool is absorbed (held).
    held = recommended_workers(idle_state(free_p_cores=14, current_workers=12))
    assert held == 12  # comfortable = 14 - 4 = 10, and 10 >= 12 - deadband(2), so hold
    # Past the deadband: a large comfortable drop backs off fully in one tick (fast).
    dropped = recommended_workers(idle_state(free_p_cores=8, current_workers=12))
    assert dropped == 4  # comfortable = 8 - 4 = 4, which is > deadband below 12, so drop to 4


# ----------------------------------------------------------------------------------
# CPU / thermal ceiling holds (never sheds for a non-OOM signal)
# ----------------------------------------------------------------------------------


def test_thermal_pressure_holds_not_sheds() -> None:
    state = idle_state(thermal_ok=False, current_workers=12)
    # Hold: keep the current pool, do not shed and do not grow.
    assert recommended_workers(state) == 12


def test_load_oversubscription_holds() -> None:
    # Load per reference core over the 0.9 ceiling (28 * 0.9 = 25.2): hold at current.
    state = idle_state(current_load=27.0, current_workers=10)
    assert recommended_workers(state) == 10


def test_battery_sheds_to_floor() -> None:
    state = idle_state(on_ac=False, current_workers=18)
    assert recommended_workers(state) == WORKER_FLOOR == 1


# ----------------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------------


def test_deterministic_given_host_state() -> None:
    states = [
        idle_state(current_workers=5),
        idle_state(hawking_active=True, current_workers=20),
        idle_state(free_p_cores=8, current_workers=20),
        idle_state(mem_available_gb=3.0, current_workers=20),
        idle_state(thermal_ok=False, current_workers=12),
        idle_state(current_load=27.0, current_workers=10),
    ]
    for state in states:
        first = recommended_workers(state)
        for _ in range(5):
            assert recommended_workers(state) == first


# ----------------------------------------------------------------------------------
# Priority / nice lever
# ----------------------------------------------------------------------------------


def test_priority_recommendation_present_and_yields_under_hawking() -> None:
    hawking = recommended_priority(idle_state(hawking_active=True, current_workers=20))
    idle = recommended_priority(idle_state(current_workers=20))
    assert isinstance(hawking, PriorityAdvice)
    assert isinstance(idle, PriorityAdvice)
    # Under Hawking the pool yields by priority: gentler QoS class and a higher nice level.
    assert hawking.yield_to_hawking is True
    assert idle.yield_to_hawking is False
    assert hawking.taskpolicy_class == "background"
    assert idle.taskpolicy_class == "utility"
    assert hawking.nice_level > idle.nice_level


def test_priority_pressure_tier_is_gentle() -> None:
    pressure = recommended_priority(idle_state(thermal_ok=False, current_workers=12))
    idle = recommended_priority(idle_state(current_workers=12))
    assert pressure.nice_level > idle.nice_level
    assert pressure.taskpolicy_class == "background"


# ----------------------------------------------------------------------------------
# Validation (fail closed)
# ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"free_p_cores": -1},
        {"free_p_cores": 1.5},
        {"mem_available_gb": -0.5},
        {"hawking_active": "yes"},
        {"thermal_ok": 1},
        {"current_load": float("inf")},
        {"current_workers": -3},
    ],
)
def test_host_state_fails_closed(overrides: dict[str, object]) -> None:
    with pytest.raises(WorkerControllerRefused):
        idle_state(**overrides)


def test_policy_ceiling_cannot_exceed_measured_peak() -> None:
    with pytest.raises(WorkerControllerRefused):
        WorkerPolicy(ceiling=24)


def test_policy_claim_scope_cannot_be_widened() -> None:
    with pytest.raises(WorkerControllerRefused):
        WorkerPolicy(claim_scope="capability claim")
