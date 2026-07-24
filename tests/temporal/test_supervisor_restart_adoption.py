"""Restart safety and worker re-adoption regressions.

These pin the invariants that stop a supervisor restart from ever costing
in-flight compute again:

* every shard is launched detached (``start_new_session=True``), so it lives
  in its own process group and a restart of the supervisor's process group
  does not signal it;
* a restarted supervisor that finds a still-running shard (its lock names a
  live pid) never launches a duplicate writer for that tag;
* once a worker dies, its stale lock is reclaimed so the tag can resume.
"""

import json
import os
import subprocess

import pytest

from mop.temporal.runs import supervisor


@pytest.fixture
def launch_sandbox(monkeypatch, tmp_path):
    """Isolate launch() onto a temp tree with a captured, fake Popen."""
    env = {"MOP_LAUNCH_SOURCE_COMMIT": "c" * 40, "MOP_LAUNCH_SOURCE_TREE_OID": "d" * 40}
    binding = {"source_commit": "c" * 40, "source_tree_oid": "d" * 40}
    monkeypatch.setattr(supervisor, "LOCKS", tmp_path / "locks")
    monkeypatch.setattr(supervisor, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(supervisor, "launch_environment", lambda: (env, binding))
    calls = []

    class Child:
        pid = 4242

    def popen(args, **kwargs):
        calls.append(kwargs)
        return Child()

    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    return calls


def test_workers_are_launched_detached_into_their_own_session(launch_sandbox):
    assert supervisor.launch(["converge_shard", "har_stream", "0"], "w.log", "c:cshard_har_stream_0")
    assert len(launch_sandbox) == 1
    # start_new_session=True is the whole mechanism: it setsid()s the child into
    # a fresh process group, so restarting the supervisor's group never kills it.
    assert launch_sandbox[0].get("start_new_session") is True


def test_restart_does_not_relaunch_a_tag_whose_worker_is_still_alive(launch_sandbox):
    tag = "x:xshard_har_stream_0"
    supervisor.LOCKS.mkdir(parents=True, exist_ok=True)
    # A live worker == a lock naming a pid that is actually alive (our own).
    supervisor._lock_path(tag).write_text(json.dumps(
        {"pid": os.getpid(), "tag": tag, "state": "active"}))
    assert supervisor.lock_active(tag) is True
    # A restarted supervisor calls launch() for every pending tag; the live one
    # must be skipped, never double-written.
    assert supervisor.launch(["extend_converge_shard", "har_stream", "0"], "w.log", tag) is False
    assert launch_sandbox == []  # no second writer spawned


def test_dead_workers_lock_is_reclaimed_so_the_tag_can_resume(launch_sandbox, monkeypatch):
    tag = "p:har_stream_0"
    supervisor.LOCKS.mkdir(parents=True, exist_ok=True)
    supervisor._lock_path(tag).write_text(json.dumps(
        {"pid": 999999, "tag": tag, "state": "active"}))
    monkeypatch.setattr(supervisor, "_pid_alive", lambda pid: False)
    assert supervisor.lock_active(tag) is False  # stale lock cleared
    assert supervisor.launch(["principal", "har_stream", "0"], "w.log", tag) is True
    assert len(launch_sandbox) == 1  # exactly one fresh worker, no duplicate


def test_adoption_liveness_probe_is_process_group_independent():
    """Re-adoption keys on pid liveness (os.kill(pid, 0)), not the caller's
    process group, so a fresh supervisor counts detached survivors it did not
    itself spawn — the basis for counting without double-launching them."""
    assert supervisor._pid_alive(os.getpid()) is True
    assert supervisor._pid_alive(999999) is False
