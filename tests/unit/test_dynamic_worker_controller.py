
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mop.studio import dynamic_worker_controller as controller
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
    sample_host_state,
    worker_bounds,
)


def idle_state(**overrides: object) -> HostState:

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

    trace: list[int] = []
    current = start
    for tick in range(ticks):
        current = recommended_workers(state_fn(tick, current), policy)
        trace.append(current)
    return trace


def test_backs_off_to_reserve_under_hawking() -> None:
    for current in (0, 1, 8, 16, 20):
        state = idle_state(hawking_active=True, current_workers=current)
        assert recommended_workers(state) == HAWKING_RESERVE_WORKERS == 2


def test_hawking_backoff_is_immediate_not_ramped() -> None:
    state = idle_state(hawking_active=True, current_workers=WORKER_CEILING)
    assert recommended_workers(state) == HAWKING_RESERVE_WORKERS


def test_ramps_to_ceiling_when_idle() -> None:
    trace = settle(lambda tick, current: idle_state(current_workers=current), start=0, ticks=60)
    assert trace[-1] == WORKER_CEILING == 20
    for previous, nxt in zip(trace, trace[1:], strict=False):
        assert nxt - previous <= DEFAULT_POLICY.ramp_up_step
    assert trace == sorted(trace)
    assert trace[20:] == [20] * len(trace[20:])


def test_ceiling_holds_and_never_returns_twenty_four() -> None:
    for current in range(0, 25):
        state = idle_state(free_p_cores=200, mem_available_gb=10_000.0, current_workers=current)
        assert recommended_workers(state) <= WORKER_CEILING == 20
    assert recommended_workers(idle_state(free_p_cores=200, current_workers=19)) == 20
    assert recommended_workers(idle_state(free_p_cores=200, current_workers=24)) == 20


def test_measured_peak_is_twenty_not_twenty_four() -> None:
    assert MEASURED_AGGREGATE_MHS[20] > MEASURED_AGGREGATE_MHS[24]
    assert max(MEASURED_AGGREGATE_MHS, key=MEASURED_AGGREGATE_MHS.get) == WORKER_CEILING


def test_core_bounded_at_the_right_value() -> None:
    state = idle_state(free_p_cores=8, mem_available_gb=90.0, current_workers=20)
    assert recommended_workers(state) == 4
    bounds = worker_bounds(state)
    assert bounds["core_bound"] == 4
    assert bounds["binding_constraint"] == "core_bound"
    assert bounds["comfortable_target"] == 4


def test_memory_bounded_at_the_right_value() -> None:
    state = idle_state(free_p_cores=28, mem_available_gb=3.0, current_workers=20)
    assert recommended_workers(state) == 4
    bounds = worker_bounds(state)
    assert bounds["memory_bound"] == 4
    assert bounds["binding_constraint"] == "memory_bound"


def test_floor_is_honored_when_resources_are_starved() -> None:
    state = idle_state(free_p_cores=28, mem_available_gb=0.1, current_workers=20)
    assert recommended_workers(state) == WORKER_FLOOR == 1


def test_sample_host_state_uses_instantaneous_cpu_not_laggy_load_average(monkeypatch) -> None:

    monkeypatch.setattr(controller.psutil, "cpu_count", lambda logical=True: 28)
    monkeypatch.setattr(controller.psutil, "cpu_percent", lambda interval=None, percpu=False: 10.0)
    monkeypatch.setattr(
        controller.psutil, "virtual_memory", lambda: SimpleNamespace(available=90e9, total=96e9)
    )
    monkeypatch.setattr(controller.os, "getloadavg", lambda: (20.0, 15.0, 12.0))
    monkeypatch.setattr(controller, "detect_hawking", lambda exclude_pids=None: (False, []))
    monkeypatch.setattr(controller, "_thermal_ok", lambda: True)
    monkeypatch.setattr(controller, "_on_ac", lambda: True)

    sample = sample_host_state()
    assert sample.state.free_p_cores == 25
    assert sample.bounds["comfortable_target"] == WORKER_CEILING
    assert sample.state.current_load == 20.0
    assert sample.telemetry["load_1m"] == 20.0


def test_sample_host_state_cpu_percent_uses_the_tuned_sample_window(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_cpu_percent(interval=None, percpu=False):
        captured["interval"] = interval
        return 0.0

    monkeypatch.setattr(controller.psutil, "cpu_count", lambda logical=True: 28)
    monkeypatch.setattr(controller.psutil, "cpu_percent", fake_cpu_percent)
    monkeypatch.setattr(
        controller.psutil, "virtual_memory", lambda: SimpleNamespace(available=90e9, total=96e9)
    )
    monkeypatch.setattr(controller.os, "getloadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(controller, "detect_hawking", lambda exclude_pids=None: (False, []))
    monkeypatch.setattr(controller, "_thermal_ok", lambda: True)
    monkeypatch.setattr(controller, "_on_ac", lambda: True)

    sample_host_state()
    assert captured["interval"] == controller.CPU_SAMPLE_INTERVAL_SECONDS


def test_hysteresis_prevents_oscillation_on_jittery_load() -> None:
    def jitter(tick: int, current: int) -> HostState:
        free = 12 if tick % 2 == 0 else 10
        return idle_state(free_p_cores=free, current_workers=current)

    trace = settle(jitter, start=0, ticks=60)
    raw = [worker_bounds(jitter(tick, 0))["comfortable_target"] for tick in range(60)]

    assert set(raw) == {6, 8}
    tail = trace[-10:]
    assert len(set(tail)) == 1
    assert tail[0] == 8  # it holds at the top of the jitter band, absorbing the dips to 6


def test_small_dips_are_held_but_real_drops_back_off_fast() -> None:
    held = recommended_workers(idle_state(free_p_cores=14, current_workers=12))
    assert held == 12  # comfortable = 14 - 4 = 10, and 10 >= 12 - deadband(2), so hold
    dropped = recommended_workers(idle_state(free_p_cores=8, current_workers=12))
    assert dropped == 4  # comfortable = 8 - 4 = 4, which is > deadband below 12, so drop to 4


def test_thermal_pressure_holds_not_sheds() -> None:
    state = idle_state(thermal_ok=False, current_workers=12)
    assert recommended_workers(state) == 12


def test_load_oversubscription_holds() -> None:
    state = idle_state(current_load=27.0, current_workers=10)
    assert recommended_workers(state) == 10


def test_battery_sheds_to_floor() -> None:
    state = idle_state(on_ac=False, current_workers=18)
    assert recommended_workers(state) == WORKER_FLOOR == 1


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


def test_priority_recommendation_present_and_yields_under_hawking() -> None:
    hawking = recommended_priority(idle_state(hawking_active=True, current_workers=20))
    idle = recommended_priority(idle_state(current_workers=20))
    assert isinstance(hawking, PriorityAdvice)
    assert isinstance(idle, PriorityAdvice)
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
