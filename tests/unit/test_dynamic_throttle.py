
from __future__ import annotations

import pytest

from mop.ladder.dynamic_throttle import (
    CLAIM_SCOPE,
    GIB,
    DynamicThrottleController,
    HostSample,
    ThrottleDecision,
    ThrottlePolicy,
    ThrottleRefusal,
)


def comfortable_sample(**overrides: object) -> HostSample:
    base: dict[str, object] = dict(
        total_memory_bytes=100 * GIB,
        available_memory_bytes=80 * GIB,
        swap_used_bytes=0,
        logical_cpus=28,
        load_1m_per_core=0.1,
        cpu_utilization=0.1,
        thermal_normal=True,
        unmanaged_heavy_rss_bytes=0,
        own_workers_rss_bytes=0,
    )
    base.update(overrides)
    return HostSample(**base)  # type: ignore[arg-type]


def default_policy(**overrides: object) -> ThrottlePolicy:
    base: dict[str, object] = dict(
        oom_floor_bytes=8 * GIB,
        per_worker_peak_rss_bytes=4 * GIB,
        max_workers=28,
        load_ceiling_per_core=0.90,
        utilization_ceiling=0.92,
        swap_backoff_bytes=256 * 1024 * 1024,
    )
    base.update(overrides)
    return ThrottlePolicy(**base)  # type: ignore[arg-type]


def test_policy_rejects_widened_claim_scope() -> None:
    with pytest.raises(ThrottleRefusal):
        default_policy(claim_scope="a capability was demonstrated")


def test_policy_rejects_nonpositive_floor() -> None:
    with pytest.raises(ThrottleRefusal):
        default_policy(oom_floor_bytes=0)


def test_policy_rejects_headroom_below_one() -> None:
    with pytest.raises(ThrottleRefusal):
        default_policy(headroom_factor=0.9)


def test_policy_rejects_decrease_factor_out_of_range() -> None:
    with pytest.raises(ThrottleRefusal):
        default_policy(decrease_factor=1.0)


def test_policy_rejects_min_above_max() -> None:
    with pytest.raises(ThrottleRefusal):
        default_policy(min_workers=40, max_workers=8)


def test_sample_rejects_available_over_total() -> None:
    with pytest.raises(ThrottleRefusal):
        comfortable_sample(available_memory_bytes=200 * GIB)


def test_sample_rejects_utilization_out_of_range() -> None:
    with pytest.raises(ThrottleRefusal):
        comfortable_sample(cpu_utilization=1.5)


def test_policy_digest_is_stable() -> None:
    assert default_policy().digest() == default_policy().digest()
    assert len(default_policy().digest()) == 64


def test_never_admits_when_projection_would_cross_floor() -> None:
    controller = DynamicThrottleController(default_policy())
    sample = comfortable_sample(available_memory_bytes=10 * GIB)
    decision = controller.decide(sample, running=0)
    assert decision.admit is False
    assert decision.projected_available_after_admit_bytes < default_policy().oom_floor_bytes


def test_admits_when_projection_stays_above_floor() -> None:
    controller = DynamicThrottleController(default_policy())
    sample = comfortable_sample(available_memory_bytes=80 * GIB)
    decision = controller.decide(sample, running=0)
    assert decision.admit is True
    assert decision.projected_available_after_admit_bytes >= default_policy().oom_floor_bytes


def test_below_floor_sheds_own_workers_and_refuses() -> None:
    controller = DynamicThrottleController(default_policy())
    sample = comfortable_sample(available_memory_bytes=6 * GIB, own_workers_rss_bytes=20 * GIB)
    decision = controller.decide(sample, running=5)
    assert decision.admit is False
    assert decision.must_shed >= 1
    assert decision.reason.startswith("below oom floor")


def test_shed_count_never_exceeds_running() -> None:
    controller = DynamicThrottleController(default_policy(oom_floor_bytes=90 * GIB))
    sample = comfortable_sample(available_memory_bytes=1 * GIB, own_workers_rss_bytes=8 * GIB)
    decision = controller.decide(sample, running=2)
    assert 0 <= decision.must_shed <= 2


def test_swap_pressure_backs_off_and_refuses() -> None:
    controller = DynamicThrottleController(default_policy())
    for _ in range(5):
        controller.decide(comfortable_sample(), running=1)
    grown = controller.target
    sample = comfortable_sample(swap_used_bytes=2 * GIB)
    decision = controller.decide(sample, running=6)
    assert decision.admit is False
    assert decision.target_workers <= grown
    assert decision.reason.startswith("swap pressure")


def test_cpu_ceiling_holds_without_shedding() -> None:
    controller = DynamicThrottleController(default_policy())
    sample = comfortable_sample(load_1m_per_core=1.5)
    decision = controller.decide(sample, running=4)
    assert decision.admit is False
    assert decision.must_shed == 0
    assert "cpu or thermal" in decision.reason


def test_utilization_ceiling_holds() -> None:
    controller = DynamicThrottleController(default_policy())
    decision = controller.decide(comfortable_sample(cpu_utilization=0.99), running=2)
    assert decision.admit is False


def test_thermal_abnormal_holds() -> None:
    controller = DynamicThrottleController(default_policy())
    decision = controller.decide(comfortable_sample(thermal_normal=False), running=2)
    assert decision.admit is False
    assert decision.must_shed == 0


def test_target_grows_toward_cap_under_comfort() -> None:
    controller = DynamicThrottleController(default_policy(max_workers=10))
    targets = []
    running = 0
    for _ in range(15):
        decision = controller.decide(comfortable_sample(logical_cpus=28), running=running)
        targets.append(decision.target_workers)
        if decision.admit:
            running += 1
    assert targets[-1] > targets[0]
    assert max(targets) <= 10  # never exceeds max_workers


def test_target_never_exceeds_logical_cpus() -> None:
    controller = DynamicThrottleController(default_policy(max_workers=64))
    running = 0
    for _ in range(40):
        decision = controller.decide(comfortable_sample(logical_cpus=8), running=running)
        assert decision.target_workers <= 8
        if decision.admit:
            running = min(running + 1, 8)


def test_memory_cap_bounds_target_below_cpu_cap() -> None:
    per_worker = 8 * GIB
    controller = DynamicThrottleController(
        default_policy(max_workers=28, per_worker_peak_rss_bytes=per_worker)
    )
    total = 100 * GIB
    running = 0
    for _ in range(30):
        used_by_workers = running * per_worker
        decision = controller.decide(
            comfortable_sample(
                total_memory_bytes=total,
                logical_cpus=28,
                available_memory_bytes=40 * GIB - used_by_workers,
                own_workers_rss_bytes=used_by_workers,
            ),
            running=running,
        )
        assert decision.memory_capped_workers <= 28
        assert (40 * GIB - running * per_worker) >= default_policy().oom_floor_bytes
        if decision.admit:
            running += 1
    assert 1 <= running <= 4


def test_learned_per_worker_rss_is_monotone_non_decreasing() -> None:
    controller = DynamicThrottleController(default_policy())
    seen = []
    for rss in [4 * GIB, 12 * GIB, 6 * GIB, 20 * GIB, 5 * GIB]:
        controller.decide(comfortable_sample(own_workers_rss_bytes=rss), running=2)
        seen.append(controller.learned_per_worker_rss_bytes)
    assert seen == sorted(seen)


def test_decisions_are_deterministic_for_equal_state() -> None:
    a = DynamicThrottleController(default_policy())
    b = DynamicThrottleController(default_policy())
    sample = comfortable_sample(available_memory_bytes=50 * GIB)
    assert a.decide(sample, running=1).payload() == b.decide(sample, running=1).payload()


def test_decision_digest_stable() -> None:
    controller = DynamicThrottleController(default_policy())
    decision = controller.decide(comfortable_sample(), running=0)
    assert isinstance(decision, ThrottleDecision)
    assert len(decision.digest()) == 64


def test_autoscaled_policy_sizes_floor_and_cap() -> None:
    sample = comfortable_sample(total_memory_bytes=100 * GIB, logical_cpus=28)
    policy = ThrottlePolicy.autoscaled(sample, per_worker_peak_gb=4.0)
    assert policy.max_workers == 28
    assert policy.oom_floor_bytes >= 8 * GIB
    assert policy.claim_scope == CLAIM_SCOPE


def test_autoscaled_floor_is_at_least_twelve_percent() -> None:
    sample = comfortable_sample(total_memory_bytes=256 * GIB, logical_cpus=20)
    policy = ThrottlePolicy.autoscaled(sample)
    assert policy.oom_floor_bytes >= int(256 * GIB * 0.12)
