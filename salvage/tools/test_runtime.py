"""Phase 3 runtime tests: scheduler receipt-invariance/no-dup/resume, retry escalation, stall, notifications.

Proves the repaired runtime behaves as required, on the real stopped-run DAG where relevant.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtime"))
import reliability as rel  # noqa: E402
import scheduler as sch  # noqa: E402

DAG = Path("/Users/scammermike/Downloads/mop/salvage/runtime/stopped_dag.json")


def test_scheduler_receipt_invariant_across_widths():
    dag = sch.load_dag(DAG)
    ids = None
    for w in (1, 2, 4, 8, 20):
        s = sch.simulate(dag, width=w)
        if ids is None:
            ids = s.completed_identities
        assert s.completed_identities == ids, f"receipt drift at width {w}"


def test_scheduler_no_duplicate_execution_and_no_dup_real():
    caps = [sch.Capsule("a", (), 0.0, "small_cpu"), sch.Capsule("b", ("a",), 0.0, "small_cpu"),
            sch.Capsule("c", ("a",), 0.0, "small_cpu"), sch.Capsule("d", ("b", "c"), 0.0, "small_cpu")]
    seen = []
    r = sch.execute(caps, width=4, run_fn=lambda c: seen.append(c.id))
    assert sorted(seen) == ["a", "b", "c", "d"], seen
    assert len(seen) == len(set(seen)), "duplicate execution"
    assert r["no_duplicate_execution"]


def test_scheduler_resume_from_sealed():
    dag = sch.load_dag(DAG)
    full = sch.simulate(dag, width=8)
    # mark half as already completed (sealed) and resume
    half = set(list(dag)[: len(dag) // 2])
    resumed = sch.simulate(dag, width=8, completed=half)
    assert resumed.wall_seconds <= full.wall_seconds
    # every capsule still accounted for (resumed completes the rest)
    assert len(resumed.per_capsule) == len(dag) - len(half)


def test_retry_escalation_ladder():
    led = rel.RetryLedger()
    d1 = led.record("deterministic_wall_overrun", "verify:catwave")
    d2 = led.record("deterministic_wall_overrun", "verify:catwave")
    d3 = led.record("deterministic_wall_overrun", "verify:catwave")
    assert d1["decision"] == "bounded_retry_or_resume"
    assert d2["decision"] == "failure_hold_plus_notification"
    assert d3["decision"] == "prohibited"
    # third with operator override is permitted
    d3b = led.record("deterministic_wall_overrun", "verify:catwave", operator_override=True)
    assert "changed_authority_or_operator_override" in d3b["decision"]
    # history is append-only
    assert len(led.attempts) == 4
    assert led.notify_required() is False  # last was override, not a failure_hold


def test_retry_transient_not_escalated():
    led = rel.RetryLedger()
    for _ in range(5):
        e = led.record("transient_resource_pressure", "mem")
        assert e["decision"] == "bounded_retry_or_resume"


def test_stall_detects_zero_progress_but_not_slow_valid():
    det = rel.StallDetector(allowed_interval_seconds=600, ready_capsule_count=3)
    now = 10_000.0
    stalled = det.evaluate(now=now, last_capsule_finish=now - 4000, last_checkpoint_advance=now - 4000,
                           last_output_change=now - 4000, cpu_active=True, concurrency=1,
                           repeated_same_wall_boundary=True)
    assert stalled["stalled"] is True
    assert "same_wall_boundary_repeats" in stalled["triggers"]
    assert "concurrency_one_while_multiple_ready" in stalled["triggers"]
    # slow but valid: a checkpoint advanced recently -> not a stall even if a capsule is slow
    slow = det.evaluate(now=now, last_capsule_finish=now - 4000, last_checkpoint_advance=now - 60,
                        last_output_change=now - 60, cpu_active=True, concurrency=4,
                        repeated_same_wall_boundary=False)
    assert slow["stalled"] is False and slow["slow_but_valid"] is True


def test_notifications_fire_and_isolate_failure():
    n = rel.Notifier()
    for ev in rel.NOTIFY_EVENTS:
        assert n.notify(ev, {"k": 1}) is True
    assert len(n.sent) == len(rel.NOTIFY_EVENTS)
    # transport failure is isolated: notify returns False, never raises, scientific state untouched
    n._transport_fails = True
    assert n.notify("terminal_result") is False
    assert n.delivery_failures == 1


def test_profiles_shared_authority_prevents_oversubscription():
    # 6 vectorized_construction x internal width 4 = 24 <= 28 ok; add 2 more -> 32 > 28 refused
    assert rel.shared_authority_ok({"vectorized_construction": 6}) is True
    assert rel.shared_authority_ok({"vectorized_construction": 8}) is False


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} runtime tests passed")
    raise SystemExit(0 if passed == len(tests) else 1)
