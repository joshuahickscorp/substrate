"""Repeat-failure detection regression.

Defect this closes: a persistently-broken shard is relaunched identically
forever — a quarantined invalid receipt or stale partial just makes the tag
"missing" again, so the main loop tries it again with no memory of the prior
attempts. There is nothing distinguishing "this failed once, bad luck" from
"this has failed ten times and will keep failing." repeatedly_failing_shards()
reads the existing quarantine incident trail (already durable, already
timestamped) to surface that pattern. Detection only — never blocks a retry.
"""

import json
import os
import time

from mop.temporal.runs import supervisor


def write_incident(orch_dir, stage, identity, kind, age_seconds, suffix=""):
    orch_dir.mkdir(parents=True, exist_ok=True)
    path = orch_dir / f"quarantine_{kind}_{stage}_{identity}{suffix}.json"
    path.write_text(json.dumps({"schema": "mop-temporal-receipt-quarantine/v2",
                                "stage": stage, "identity": identity, "classification": kind}))
    t = time.time() - age_seconds
    os.utime(path, (t, t))
    return path


def test_below_threshold_is_not_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    orch = tmp_path / "orchestration"
    write_incident(orch, "e2_principal", "har_stream_0", "invalid_receipts", age_seconds=100, suffix="_a")
    write_incident(orch, "e2_principal", "har_stream_0", "invalid_receipts", age_seconds=200, suffix="_b")
    assert supervisor.repeatedly_failing_shards(threshold=3) == []


def test_at_threshold_is_flagged_with_the_right_count(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    orch = tmp_path / "orchestration"
    for i, age in enumerate((100, 200, 300)):
        write_incident(orch, "e2_principal", "har_stream_0", "invalid_receipts", age_seconds=age, suffix=f"_{i}")
    flagged = supervisor.repeatedly_failing_shards(threshold=3)
    assert len(flagged) == 1
    assert flagged[0] == {"stage": "e2_principal", "identity": "har_stream_0",
                          "quarantine_count": 3, "most_recent_seconds_ago": 100}


def test_incidents_outside_the_window_do_not_count(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    orch = tmp_path / "orchestration"
    write_incident(orch, "e2_principal", "har_stream_0", "invalid_receipts", age_seconds=100, suffix="_a")
    write_incident(orch, "e2_principal", "har_stream_0", "invalid_receipts", age_seconds=200, suffix="_b")
    write_incident(orch, "e2_principal", "har_stream_0", "invalid_receipts",
                    age_seconds=999999, suffix="_old")  # far outside any reasonable window
    assert supervisor.repeatedly_failing_shards(threshold=3, window_seconds=3600) == []


def test_different_stage_identity_pairs_are_tracked_independently(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    orch = tmp_path / "orchestration"
    for i in range(3):
        write_incident(orch, "e2_principal", "har_stream_0", "invalid_receipts", age_seconds=i, suffix=f"_p{i}")
    for i in range(2):
        write_incident(orch, "e2_converge", "speech_stream_5", "stale_partials", age_seconds=i, suffix=f"_c{i}")
    flagged = supervisor.repeatedly_failing_shards(threshold=3)
    assert len(flagged) == 1 and flagged[0]["identity"] == "har_stream_0"


def test_surfaced_through_status_and_reconcile(monkeypatch, tmp_path):
    monkeypatch.setattr(supervisor, "LOCKS", tmp_path / "locks")
    monkeypatch.setattr(supervisor.io, "RUNS", tmp_path)
    orch = tmp_path / "orchestration"
    for i in range(3):
        write_incident(orch, "e2_principal", "har_stream_0", "invalid_receipts", age_seconds=i, suffix=f"_{i}")
    row = supervisor.status()
    assert len(row["repeatedly_failing_shards"]) == 1

    monkeypatch.setattr(supervisor.io, "exists", lambda name: False)
    sealed = {}
    monkeypatch.setattr(supervisor.io, "seal", lambda name, doc: sealed.update({name: doc}))
    doc = supervisor.reconcile_live_state("test", row)
    assert len(doc["repeatedly_failing_shards"]) == 1
    assert doc["repeatedly_failing_shards"][0]["identity"] == "har_stream_0"
