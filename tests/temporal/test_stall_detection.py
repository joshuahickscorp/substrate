"""Stall detection regression.

Defect this closes: a shard can run for hours past any reasonable ceiling with
no automated signal — the har_stream_18 extended shard ran 42h before it was
noticed by manually reading `ps`. stalled_workers() is detection only (never
kills or reschedules); it exists so status()/reconcile_live_state() surface a
stuck shard instead of requiring a human to go looking.
"""

import json
import os
import time

from mop.temporal.runs import supervisor


def write_lock(locks, tag, pid, age_seconds):
    locks.mkdir(parents=True, exist_ok=True)
    path = supervisor._lock_path(tag)
    path.write_text(json.dumps({"pid": pid, "tag": tag, "state": "active"}))
    os.utime(path, (time.time() - age_seconds, time.time() - age_seconds))
    return path


def test_extended_shard_running_far_past_its_own_base_wall_is_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, "LOCKS", tmp_path / "locks")
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path / "runs")
    base_dir = tmp_path / "runs" / "e2_converge"
    base_dir.mkdir(parents=True)
    # base convergence for this config took 3600s; 4x ceiling is 14400s
    (base_dir / "cshard_har_stream_18.json").write_text(json.dumps({"wall_seconds": 3600}))
    tag = "x:xshard_har_stream_18"
    write_lock(supervisor.LOCKS, tag, os.getpid(), age_seconds=20000)  # past the 14400s ceiling
    stalled = supervisor.stalled_workers()
    assert len(stalled) == 1
    assert stalled[0]["tag"] == tag
    assert stalled[0]["threshold_seconds"] == 14400
    assert stalled[0]["elapsed_seconds"] >= 20000


def test_extended_shard_within_its_threshold_is_not_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, "LOCKS", tmp_path / "locks")
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path / "runs")
    base_dir = tmp_path / "runs" / "e2_converge"
    base_dir.mkdir(parents=True)
    (base_dir / "cshard_har_stream_18.json").write_text(json.dumps({"wall_seconds": 3600}))
    write_lock(supervisor.LOCKS, "x:xshard_har_stream_18", os.getpid(), age_seconds=5000)
    assert supervisor.stalled_workers() == []


def test_tag_with_no_base_wall_reference_uses_the_flat_default_ceiling(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, "LOCKS", tmp_path / "locks")
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path / "runs")
    tag = "c:cshard_har_stream_0"
    write_lock(supervisor.LOCKS, tag, os.getpid(), age_seconds=supervisor.STALL_DEFAULT_SECONDS + 1)
    stalled = supervisor.stalled_workers()
    assert len(stalled) == 1 and stalled[0]["threshold_seconds"] == supervisor.STALL_DEFAULT_SECONDS


def test_a_dead_pids_lock_is_never_reported_as_stalled(monkeypatch, tmp_path):
    """A dead worker is a reclaim case (lock_active handles it), not a stall —
    stalled_workers must only ever look at pids that are still actually alive."""
    monkeypatch.setattr(supervisor, "LOCKS", tmp_path / "locks")
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path / "runs")
    write_lock(supervisor.LOCKS, "p:har_stream_0", pid=999999, age_seconds=999999)
    assert supervisor.stalled_workers() == []


def test_stall_detection_is_reported_through_status_and_reconcile(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, "LOCKS", tmp_path / "locks")
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    write_lock(supervisor.LOCKS, "c:cshard_har_stream_0", os.getpid(),
               age_seconds=supervisor.STALL_DEFAULT_SECONDS + 1)
    row = supervisor.status()
    assert len(row["stalled_workers"]) == 1

    monkeypatch.setattr(supervisor.io, "exists", lambda name: False)
    sealed = {}
    monkeypatch.setattr(supervisor.io, "seal", lambda name, doc: sealed.update({name: doc}))
    doc = supervisor.reconcile_live_state("test", row)
    assert len(doc["stalled_workers"]) == 1
    assert doc["stalled_workers"][0]["tag"] == "c:cshard_har_stream_0"
