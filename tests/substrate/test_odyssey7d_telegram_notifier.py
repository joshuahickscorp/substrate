"""Focused fail-closed health and delivery-ledger tests for Odyssey Telegram."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _notifier() -> object:
    source = Path(__file__).parents[2] / "tools/odyssey7d_telegram_notifier.py"
    specification = importlib.util.spec_from_file_location("odyssey7d_telegram_notifier_test", source)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    # ``RunContext`` is a dataclass with postponed annotations; register the
    # dynamically loaded module before its class decorator resolves them.
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _sealed(notifier: object, value: dict[str, object]) -> dict[str, object]:
    document = dict(value)
    document["sha256"] = notifier._digest(document)
    return document


def _write(path: Path, value: dict[str, object], *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    if mode is not None:
        path.chmod(mode)


def _configured_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[object, Path, dict[str, object], object]:
    notifier = _notifier()
    root = tmp_path / "workspace"
    run_root = root / "runs/substrate/odyssey7d/v1"
    authority_path = root / "plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json"
    state_path = root / "runs/substrate/odyssey7d/notifier-state.json"
    monkeypatch.setattr(notifier, "ROOT", root)
    monkeypatch.setattr(notifier, "RUNS", run_root)
    monkeypatch.setattr(notifier, "AUTHORITY", authority_path)
    monkeypatch.setattr(notifier, "STATE", state_path)
    frozen = {"sha256": "f" * 64}
    monkeypatch.setattr(notifier, "_current_frozen_build", lambda: frozen)
    authority = _sealed(
        notifier,
        {
            "schema": notifier.AUTHORITY_SCHEMA,
            "activation": False,
            "external_activation": False,
            "status": "sealed_admitted",
            "run_id": "odyssey-notifier-fixture",
            "program": {"id": notifier.PROGRAM, "launch_allowed": True},
            "launch_allowed": True,
            "seal": {"status": "sealed", "frozen_build_sha256": frozen["sha256"]},
            "frozen_build_sha256": frozen["sha256"],
            "worker": {
                "run_root": "runs/substrate/odyssey7d/v1",
                "argv": [sys.executable, "-m", "substrate.odyssey_worker", "run"],
            },
        },
    )
    _write(authority_path, authority)
    context = notifier._sealed_run_context()
    assert context is not None
    return notifier, run_root, authority, context


def _live_state(notifier: object, context: object, *, now: float | None = None) -> dict[str, object]:
    now = time.time() if now is None else now
    return _sealed(
        notifier,
        {
            "schema": notifier.SUPERVISOR_STATE_SCHEMA,
            "activation": False,
            "authority_sha256": context.authority_sha256,
            "run_id": context.run_id,
            "day": 1,
            "elapsed_seconds": 30.0,
            "completion_percent": 0.01,
            "microcycles_complete": 0,
            "resident_memory": 1,
            "free_storage": 2,
            "storage_guard": 3,
            "broker_action": "admit_or_resume",
            "next_boundary": 1800,
            "cpu_time_deltas": {"active_cores_equivalent": 0.5, "logical_cores_available": 8},
            "sampled_at_epoch": now,
            "pids": {"supervisor": os.getpid(), "worker": os.getpid(), "worker_tree": [os.getpid()]},
            "run_active": True,
            "status": "worker_running",
        },
    )


def _terminal_state(notifier: object, context: object, *, now: float | None = None) -> dict[str, object]:
    now = time.time() if now is None else now
    state = _live_state(notifier, context, now=now)
    state.pop("sha256")
    state.update(
        {
            "run_active": False,
            "status": "terminal_safe_hold",
            "terminal_reason": "memory_cap_safe_hold",
            "terminal_at_epoch": now,
            "pids": {"supervisor": os.getpid(), "worker": None, "worker_tree": []},
            "restart_lineage": {
                "abnormal_restart_count": 0,
                "max_abnormal_restarts": 3,
                "terminal_status": "memory_cap_safe_hold",
            },
        }
    )
    return _sealed(notifier, state)


def _postflight_receipt(notifier: object, context: object) -> dict[str, object]:
    return _sealed(
        notifier,
        {
            "schema": notifier.POSTFLIGHT_RECEIPT_SCHEMA,
            "activation": False,
            "authority_sha256": context.authority_sha256,
            "run_id": context.run_id,
            "outcome": "worker_trace_locked_waiting_for_independent_evaluation",
            "scientific_results_included": False,
            "worker_state": {"sha256": "1" * 64},
            "trace_lock": {"sha256": "2" * 64},
            "evaluator_release_request": {"sha256": "3" * 64, "worker_accessed_evaluator_answers": False},
        },
    )


def _write_lease(notifier: object, run_root: Path, context: object, *, now: float | None = None) -> Path:
    now = time.time() if now is None else now
    lease = _sealed(
        notifier,
        {
            "schema": notifier.SUPERVISOR_LEASE_SCHEMA,
            "activation": False,
            "authority_sha256": context.authority_sha256,
            "run_id": context.run_id,
            "supervisor_pid": os.getpid(),
            "attempt": 1,
            "worker_argv_sha256": context.worker_argv_sha256,
            "issued_at_epoch": now - 1,
        },
    )
    path = run_root / "leases/attempt-001.json"
    _write(path, lease, mode=0o600)
    return path


def test_latest_health_accepts_only_the_authority_bound_canonical_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notifier, run_root, _authority, context = _configured_run(tmp_path, monkeypatch)
    state_path = run_root / "SUPERVISOR_STATE.json"
    _write_lease(notifier, run_root, context)
    _write(state_path, _live_state(notifier, context))
    # Even a freshly written nested receipt can only be rehearsal/history; it
    # is not a candidate for a live notifier report.
    nested = run_root / "rehearsal/SUPERVISOR_STATE.json"
    _write(nested, {"not": "a live receipt"})

    observed = notifier.latest_health()

    assert observed is not None
    path, health = observed
    assert path == state_path
    assert health["run_id"] == context.run_id
    assert health["status"] == "worker_running"


def test_latest_health_rejects_stale_and_tampered_state_before_delivery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notifier, run_root, _authority, context = _configured_run(tmp_path, monkeypatch)
    now = time.time()
    _write_lease(notifier, run_root, context, now=now)
    stale = _live_state(notifier, context, now=now - notifier.LIVE_FRESHNESS_SECONDS - 1)
    _write(run_root / "SUPERVISOR_STATE.json", stale)

    with pytest.raises(notifier.NotifierError, match="stale"):
        notifier.latest_health()

    fresh = _live_state(notifier, context, now=now)
    fresh["sha256"] = "0" * 64
    _write(run_root / "SUPERVISOR_STATE.json", fresh)
    with pytest.raises(notifier.NotifierError, match="self-digest"):
        notifier.latest_health()


def test_latest_health_rejects_lease_that_does_not_bind_current_supervisor_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notifier, run_root, _authority, context = _configured_run(tmp_path, monkeypatch)
    now = time.time()
    _write(run_root / "SUPERVISOR_STATE.json", _live_state(notifier, context, now=now))
    lease_path = _write_lease(notifier, run_root, context, now=now)
    lease_path.chmod(0o644)
    with pytest.raises(notifier.NotifierError, match="mode 0600"):
        notifier.latest_health()
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease.pop("sha256")
    lease["supervisor_pid"] = os.getpid() + 1
    _write(lease_path, _sealed(notifier, lease), mode=0o600)

    with pytest.raises(notifier.NotifierError, match="does not bind"):
        notifier.latest_health()


def test_terminal_notification_is_single_redacted_and_atomically_ledgared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notifier, run_root, _authority, context = _configured_run(tmp_path, monkeypatch)
    _write(run_root / "SUPERVISOR_STATE.json", _terminal_state(notifier, context))
    delivered: list[str] = []
    monkeypatch.setattr(notifier, "send", lambda text: delivered.append(text) or 73)

    first = notifier.tick(deliver=True)
    second = notifier.tick(deliver=True)

    assert first["state"] == "delivered"
    assert first["source"] == "canonical_supervisor_state"
    assert second["state"] == "already_delivered"
    assert len(delivered) == 1
    assert "memory_cap_safe_hold" not in delivered[0]
    assert context.run_id not in delivered[0]
    ledger = json.loads(notifier.STATE.read_text(encoding="utf-8"))
    assert ledger["schema"] == notifier.LEDGER_SCHEMA
    assert len(ledger["sent"]) == 1
    assert not list(notifier.STATE.parent.glob(f".{notifier.STATE.name}.*.tmp"))


def test_worker_complete_notification_requires_the_bound_postflight_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notifier, run_root, _authority, context = _configured_run(tmp_path, monkeypatch)
    state = _terminal_state(notifier, context)
    state.pop("sha256")
    state.update({"status": "worker_complete", "terminal_reason": "worker_complete"})
    _write(run_root / "SUPERVISOR_STATE.json", _sealed(notifier, state))

    with pytest.raises(notifier.NotifierError, match="postflight receipt"):
        notifier.latest_health()

    receipt = _postflight_receipt(notifier, context)
    state["postflight_receipt_sha256"] = receipt["sha256"]
    _write(run_root / notifier.POSTFLIGHT_RECEIPT_NAME, receipt)
    _write(run_root / "SUPERVISOR_STATE.json", _sealed(notifier, state))

    observed = notifier.latest_health()

    assert observed is not None
    assert observed[1]["status"] == "worker_complete"


def test_busy_ledger_lock_prevents_a_second_delivery_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notifier, run_root, _authority, context = _configured_run(tmp_path, monkeypatch)
    _write_lease(notifier, run_root, context)
    _write(run_root / "SUPERVISOR_STATE.json", _live_state(notifier, context))
    monkeypatch.setattr(notifier, "_acquire_ledger_lock", lambda: None)
    monkeypatch.setattr(notifier, "send", lambda _text: pytest.fail("busy notifier must not send"))

    result = notifier.tick(deliver=True)

    assert result == {"state": "notifier_busy", "delivered": False}


def test_preflight_capacity_forecast_keeps_the_max_cap_distinct_from_g07(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notifier = _notifier()
    root = tmp_path / "workspace"
    reserve = root / "plans/substrate/tangible_next_launch/ODYSSEY_SHARED_STORAGE_RESERVE.draft.json"
    _write(
        reserve,
        {
            "shared_post_r2_capacity_policy": {
                "model_ladder_reservation_bytes_decimal": 120_000_000_000,
                "private_write_cap_bytes": 120 * 1024**3,
                "largest_transient_window_bytes": 8 * 1024**3,
                "terminal_allowance_bytes": 4 * 1024**3,
            }
        },
    )
    monkeypatch.setattr(notifier, "ROOT", root)
    monkeypatch.setattr(notifier, "SHARED_STORAGE_RESERVE", reserve)
    monkeypatch.setattr(notifier.shutil, "disk_usage", lambda _path: SimpleNamespace(free=350 * 1024**3))

    forecast = notifier._preflight_capacity_forecast()

    assert "max-cap preview" in forecast
    assert "G07 measurement pending" in forecast
    assert "margin" in forecast


def test_preflight_freeze_drift_suppresses_transition_and_gate_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notifier = _notifier()
    root = tmp_path / "workspace"
    monkeypatch.setattr(notifier, "ROOT", root)
    monkeypatch.setattr(notifier, "_current_frozen_build", lambda: (_ for _ in ()).throw(notifier.NotifierError("drift")))
    monkeypatch.setattr(notifier.shutil, "disk_usage", lambda _path: SimpleNamespace(free=350 * 1024**3))

    event_id, text = notifier.preflight_payload()

    assert event_id.startswith("odyssey-preflight/")
    assert "source/freeze verification pending" in text
    assert "0/15 trusted gates" in text
    assert "R2:" not in text
    assert "transition:" not in text


def test_preflight_delivery_is_limited_to_one_message_per_bucket_when_gates_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notifier = _notifier()
    root = tmp_path / "workspace"
    monkeypatch.setattr(notifier, "ROOT", root)
    monkeypatch.setattr(notifier, "STATE", root / "runs/substrate/odyssey7d/notifier-state.json")
    snapshots = iter(
        (
            {"source_freeze_valid": True, "transition_state": "odyssey_preflight_authorized", "gates_passed": 1, "gates_total": 15, "completion_percent": 6.7, "r2_status": "verified", "free_storage": 1, "storage_guard": 1, "blockers": []},
            {"source_freeze_valid": True, "transition_state": "odyssey_preflight_authorized", "gates_passed": 5, "gates_total": 15, "completion_percent": 33.3, "r2_status": "verified", "free_storage": 1, "storage_guard": 1, "blockers": []},
        )
    )
    monkeypatch.setattr(notifier, "preflight_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(notifier, "_preflight_capacity_forecast", lambda: "forecast")
    sent: list[str] = []
    monkeypatch.setattr(notifier, "send", lambda text: sent.append(text) or 77)

    first = notifier.tick(deliver=True, phase="preflight")
    second = notifier.tick(deliver=True, phase="preflight")

    assert first["state"] == "delivered"
    assert second["state"] == "already_delivered"
    assert len(sent) == 1


def test_live_notifier_refuses_a_source_freeze_drift_before_reading_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notifier, run_root, _authority, _context = _configured_run(tmp_path, monkeypatch)
    monkeypatch.setattr(notifier, "_current_frozen_build", lambda: (_ for _ in ()).throw(notifier.NotifierError("drift")))
    _write(run_root / "SUPERVISOR_STATE.json", {"not": "trusted"})

    with pytest.raises(notifier.NotifierError, match="drift"):
        notifier.latest_health()


def test_launchd_jobs_use_a_private_log_umask() -> None:
    notifier = _notifier()

    assert notifier.launchd_job()["Umask"] == 0o077
    assert notifier.preflight_launchd_job()["Umask"] == 0o077
