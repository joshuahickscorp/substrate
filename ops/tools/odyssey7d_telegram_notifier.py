#!/usr/bin/env python3
"""Read-only, launchd-friendly Telegram health notifier for a sealed Odyssey."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import plistlib
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/substrate/odyssey7d/v1"
STATE = ROOT / "runs/substrate/odyssey7d/notifier-state.json"
AUTHORITY = ROOT / "docs/plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json"
SHARED_STORAGE_RESERVE = ROOT / "docs/plans/substrate/tangible_next_launch/ODYSSEY_SHARED_STORAGE_RESERVE.draft.json"
LABEL = "org.substrate.odyssey7d.telegram"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
PREFLIGHT_LABEL = "org.substrate.odyssey-preflight.telegram"
PREFLIGHT_PLIST = ROOT / "docs/plans/substrate/tangible_next_launch/ODYSSEY_PREFLIGHT.telegram.launchd.plist"
PREFLIGHT_RUNS = ROOT / "runs/substrate/odyssey_transition"
TOKEN_SERVICE = "com.hawking.doctorv5.telegram.bot-token"
CHAT_SERVICE = "com.hawking.doctorv5.telegram.chat-id"
ACCOUNT = "hawking"
INTERVAL = 120
BUCKET = 1800
PROGRAM = "substrate-odyssey-7d-v1"
AUTHORITY_SCHEMA = "SUBSTRATE_ODYSSEY_7D_AUTHORITY/v1"
SUPERVISOR_STATE_SCHEMA = "SUBSTRATE_ODYSSEY_SUPERVISOR_STATE/v2"
SUPERVISOR_LEASE_SCHEMA = "SUBSTRATE_ODYSSEY_SUPERVISOR_RUNTIME_LEASE/v1"
POSTFLIGHT_RECEIPT_SCHEMA = "SUBSTRATE_ODYSSEY_POSTFLIGHT_RECEIPT/v1"
POSTFLIGHT_RECEIPT_NAME = "POSTFLIGHT_RECEIPT.json"
LEDGER_SCHEMA = "SUBSTRATE_ODYSSEY_TELEGRAM_LEDGER/v1"
MAX_RECEIPT_BYTES = 4 * 1024 * 1024
LIVE_FRESHNESS_SECONDS = max(5 * INTERVAL, 600)
TERMINAL_FRESHNESS_SECONDS = 24 * 60 * 60
CLOCK_SKEW_SECONDS = 120
GIB = 1024**3
BASE_PROTECTED_FLOOR_BYTES = 50 * GIB
LIVE_BROKER_ACTIONS = frozenset(
    {
        "admit_or_resume",
        "deny_new_work",
        "checkpoint_reduce_p2",
        "pause_p1_checkpoint_p2",
        "safe_hold_non_p0",
    }
)
TERMINAL_REASONS = frozenset(
    {
        "worker_complete",
        "memory_cap_safe_hold",
        "supervisor_observation_safe_hold",
        "restart_budget_exhausted",
        "orphaned_worker_detected_safe_hold",
        "postflight_verification_failed_safe_hold",
    }
)


class NotifierError(RuntimeError):
    """A status source or delivery ledger failed a fail-closed check."""


@dataclass(frozen=True)
class RunContext:
    """The one sealed authority-selected run that may produce notifications."""

    run_root: Path
    authority_sha256: str
    run_id: str
    worker_argv_sha256: str


def _digest(value: dict[str, Any]) -> str:
    """Match the canonical self-digest used by Odyssey control-plane receipts."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    """Read a bounded regular JSON object without following a receipt symlink."""
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_RECEIPT_BYTES:
            raise NotifierError(f"{label} is absent or unsafe")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NotifierError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise NotifierError(f"{label} is not an object")
    return value


def read_json(path: Path) -> dict[str, Any]:
    """Compatibility wrapper for read-only preflight inputs."""
    return _read_json(path, label="JSON input")


def _self_digest(value: dict[str, Any], *, label: str) -> str:
    claimed = value.get("sha256")
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    if not isinstance(claimed, str) or len(claimed) != 64 or any(character not in "0123456789abcdef" for character in claimed) or claimed != _digest(unsigned):
        raise NotifierError(f"{label} self-digest is invalid")
    return claimed


def _positive_pid(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise NotifierError(f"{label} is invalid")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise NotifierError(f"{label} is invalid")
    return value


def _nonnegative_number(value: Any, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
        raise NotifierError(f"{label} is invalid")
    return float(value)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _inside(parent: Path, child: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _current_frozen_build() -> dict[str, Any]:
    """Return only a source-current frozen build, without mutating it.

    The notifier must never turn a previously valid transition receipt into a
    current status claim after either its protocol inputs or its implementation
    drifted.  The authority module owns the canonical source-map validation;
    this small wrapper intentionally reduces every validation/import failure
    to one non-sensitive, fail-closed notifier condition.
    """
    try:
        from substrate import odyssey_authority

        frozen = odyssey_authority.validate_current_frozen_build(ROOT)
    except (ImportError, OSError, RuntimeError) as error:
        raise NotifierError("Odyssey source/freeze verification is pending") from error
    if not isinstance(frozen, dict) or not isinstance(frozen.get("sha256"), str):
        raise NotifierError("Odyssey source/freeze verification is pending")
    return frozen


def _sealed_run_context() -> RunContext | None:
    """Return the only run root eligible for live notification.

    The notifier deliberately does not search historical, rehearsal, or
    run-id-nested roots.  The sealed authority names one root, and that exact
    root must also match this notifier's configured canonical run root.
    """
    frozen = _current_frozen_build()
    if not AUTHORITY.exists():
        return None
    authority = _read_json(AUTHORITY, label="sealed Odyssey authority")
    authority_sha256 = _self_digest(authority, label="sealed Odyssey authority")
    program = authority.get("program")
    worker = authority.get("worker")
    seal = authority.get("seal")
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("activation") is not False
        or authority.get("external_activation") is not False
        or authority.get("status") != "sealed_admitted"
        or not isinstance(program, dict)
        or program.get("id") != PROGRAM
        or program.get("launch_allowed") is not True
        or authority.get("launch_allowed") is not True
        or not isinstance(seal, dict)
        or seal.get("status") != "sealed"
        or seal.get("frozen_build_sha256") != frozen["sha256"]
        or authority.get("frozen_build_sha256") != frozen["sha256"]
        or not isinstance(worker, dict)
    ):
        raise NotifierError("sealed Odyssey authority is malformed")
    run_id = authority.get("run_id")
    run_root_value = worker.get("run_root")
    argv = worker.get("argv")
    if (
        not isinstance(run_id, str)
        or not run_id.strip()
        or not isinstance(run_root_value, str)
        or not run_root_value
        or Path(run_root_value).is_absolute()
        or ".." in Path(run_root_value).parts
        or not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item and "\n" not in item for item in argv)
    ):
        raise NotifierError("sealed Odyssey authority run binding is malformed")
    run_root = (ROOT / run_root_value).resolve()
    runs_root = (ROOT / "runs").resolve()
    if run_root == runs_root or not _inside(runs_root, run_root) or run_root != RUNS.resolve():
        raise NotifierError("sealed authority does not bind the notifier canonical run root")
    return RunContext(
        run_root=run_root,
        authority_sha256=authority_sha256,
        run_id=run_id,
        worker_argv_sha256=_digest({"argv": argv}),
    )


def _current_lease(context: RunContext, state: dict[str, Any]) -> dict[str, Any]:
    """Validate the latest canonical runtime lease for an active state."""
    leases = context.run_root / "leases"
    if not leases.is_dir() or leases.is_symlink():
        raise NotifierError("active Odyssey state has no canonical lease directory")
    candidates: list[tuple[int, Path]] = []
    for path in leases.glob("attempt-*.json"):
        if path.is_symlink() or not path.is_file():
            continue
        stem = path.stem
        suffix = stem.removeprefix("attempt-")
        if suffix.isdigit() and len(suffix) == 3:
            candidates.append((int(suffix), path))
    if not candidates:
        raise NotifierError("active Odyssey state has no runtime lease")
    attempt, path = max(candidates)
    try:
        metadata = path.stat()
    except OSError as error:
        raise NotifierError("active Odyssey runtime lease metadata is unreadable") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise NotifierError("active Odyssey runtime lease is not mode 0600")
    lease = _read_json(path, label="Odyssey runtime lease")
    _self_digest(lease, label="Odyssey runtime lease")
    pids = state["pids"]
    if (
        lease.get("schema") != SUPERVISOR_LEASE_SCHEMA
        or lease.get("activation") is not False
        or lease.get("authority_sha256") != context.authority_sha256
        or lease.get("run_id") != context.run_id
        or lease.get("supervisor_pid") != pids["supervisor"]
        or lease.get("attempt") != attempt
        or lease.get("worker_argv_sha256") != context.worker_argv_sha256
    ):
        raise NotifierError("active Odyssey runtime lease does not bind the live state")
    issued_at = _nonnegative_number(lease.get("issued_at_epoch"), label="runtime lease issued_at_epoch")
    sampled_at = _nonnegative_number(state.get("sampled_at_epoch"), label="supervisor state sampled_at_epoch")
    if issued_at > sampled_at + CLOCK_SKEW_SECONDS:
        raise NotifierError("runtime lease was issued after its observed state")
    return lease


def _assert_fresh(value: Any, *, now: float, maximum_age: int, label: str) -> float:
    observed = _nonnegative_number(value, label=label)
    if observed > now + CLOCK_SKEW_SECONDS or now - observed > maximum_age:
        raise NotifierError(f"{label} is stale")
    return observed


def _validate_health_fields(health: dict[str, Any]) -> None:
    _nonnegative_int(health.get("day"), label="supervisor state day")
    if not 1 <= health["day"] <= 7:
        raise NotifierError("supervisor state day is outside the Odyssey schedule")
    _nonnegative_number(health.get("elapsed_seconds"), label="supervisor state elapsed_seconds")
    completion = _nonnegative_number(health.get("completion_percent"), label="supervisor state completion_percent")
    if completion > 100:
        raise NotifierError("supervisor state completion percent is invalid")
    _nonnegative_int(health.get("microcycles_complete"), label="supervisor state microcycles_complete")
    _nonnegative_int(health.get("resident_memory"), label="supervisor state resident_memory")
    _nonnegative_int(health.get("free_storage"), label="supervisor state free_storage")
    _nonnegative_int(health.get("storage_guard"), label="supervisor state storage_guard")
    _nonnegative_int(health.get("next_boundary"), label="supervisor state next_boundary")
    cpu = health.get("cpu_time_deltas")
    if not isinstance(cpu, dict):
        raise NotifierError("supervisor state CPU telemetry is invalid")
    _nonnegative_number(cpu.get("active_cores_equivalent"), label="supervisor state active cores")
    logical = cpu.get("logical_cores_available")
    if not isinstance(logical, int) or isinstance(logical, bool) or logical < 1:
        raise NotifierError("supervisor state logical core count is invalid")
    if health.get("broker_action") not in LIVE_BROKER_ACTIONS:
        raise NotifierError("supervisor state broker action is invalid")


def _validate_completed_postflight(context: RunContext, state: dict[str, Any]) -> None:
    """Require the supervisor's custody-only receipt before saying complete.

    Telegram intentionally validates a compact receipt rather than parsing a
    candidate/control trace or an evaluator namespace.  Its purpose is only to
    prevent a bare zero worker exit from becoming an externally visible
    completion claim.
    """
    expected_digest = state.get("postflight_receipt_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise NotifierError("completed Odyssey state lacks a postflight receipt binding")
    receipt_path = context.run_root / POSTFLIGHT_RECEIPT_NAME
    receipt = _read_json(receipt_path, label="Odyssey postflight receipt")
    observed_digest = _self_digest(receipt, label="Odyssey postflight receipt")
    if (
        observed_digest != expected_digest
        or receipt.get("schema") != POSTFLIGHT_RECEIPT_SCHEMA
        or receipt.get("activation") is not False
        or receipt.get("authority_sha256") != context.authority_sha256
        or receipt.get("run_id") != context.run_id
        or receipt.get("outcome") != "worker_trace_locked_waiting_for_independent_evaluation"
        or receipt.get("scientific_results_included") is not False
    ):
        raise NotifierError("completed Odyssey postflight receipt is malformed or unbound")
    state_ref = receipt.get("worker_state")
    trace_ref = receipt.get("trace_lock")
    release_ref = receipt.get("evaluator_release_request")
    if (
        not isinstance(state_ref, dict)
        or not isinstance(trace_ref, dict)
        or not isinstance(release_ref, dict)
        or release_ref.get("worker_accessed_evaluator_answers") is not False
        or not all(isinstance(reference.get("sha256"), str) and len(reference["sha256"]) == 64 for reference in (state_ref, trace_ref, release_ref))
    ):
        raise NotifierError("completed Odyssey postflight receipt lacks custody bindings")


def _validate_supervisor_state(path: Path, context: RunContext, *, now: float) -> dict[str, Any]:
    """Validate a current v2 supervisor state before it can reach Telegram."""
    state = _read_json(path, label="canonical Odyssey supervisor state")
    _self_digest(state, label="canonical Odyssey supervisor state")
    if (
        state.get("schema") != SUPERVISOR_STATE_SCHEMA
        or state.get("activation") is not False
        or state.get("authority_sha256") != context.authority_sha256
        or state.get("run_id") != context.run_id
    ):
        raise NotifierError("canonical Odyssey supervisor state is not bound to the sealed run")
    status = state.get("status")
    pids = state.get("pids")
    if not isinstance(pids, dict):
        raise NotifierError("canonical Odyssey supervisor state has no PID binding")
    supervisor_pid = _positive_pid(pids.get("supervisor"), label="supervisor PID")
    worker_value = pids.get("worker")
    if status == "worker_running":
        if state.get("run_active") is not True:
            raise NotifierError("running Odyssey state is not marked active")
        worker_pid = _positive_pid(worker_value, label="worker PID")
        if not _pid_is_alive(supervisor_pid) or not _pid_is_alive(worker_pid):
            raise NotifierError("running Odyssey state has a dead PID")
        _assert_fresh(state.get("sampled_at_epoch"), now=now, maximum_age=LIVE_FRESHNESS_SECONDS, label="supervisor state sampled_at_epoch")
        try:
            modified = path.stat().st_mtime
        except OSError as error:
            raise NotifierError("canonical Odyssey supervisor state metadata is unreadable") from error
        _assert_fresh(modified, now=now, maximum_age=LIVE_FRESHNESS_SECONDS, label="supervisor state mtime")
        _validate_health_fields(state)
        _current_lease(context, state)
    elif status in {"worker_complete", "terminal_safe_hold"}:
        if state.get("run_active") is not False:
            raise NotifierError("terminal Odyssey state is still marked active")
        if worker_value is not None:
            _positive_pid(worker_value, label="terminal worker PID")
        reason = state.get("terminal_reason")
        if reason not in TERMINAL_REASONS or (status == "worker_complete" and reason != "worker_complete"):
            raise NotifierError("terminal Odyssey state has an unknown terminal reason")
        _assert_fresh(state.get("terminal_at_epoch"), now=now, maximum_age=TERMINAL_FRESHNESS_SECONDS, label="terminal Odyssey timestamp")
        lineage = state.get("restart_lineage")
        if not isinstance(lineage, dict):
            raise NotifierError("terminal Odyssey state has no restart lineage")
        _nonnegative_int(lineage.get("abnormal_restart_count"), label="terminal restart count")
        _nonnegative_int(lineage.get("max_abnormal_restarts"), label="terminal restart maximum")
        terminal_status = lineage.get("terminal_status")
        if terminal_status is not None and not isinstance(terminal_status, str):
            raise NotifierError("terminal Odyssey restart lineage is malformed")
        _validate_health_fields(state)
        if worker_value is not None and _pid_is_alive(_positive_pid(worker_value, label="terminal worker PID")):
            raise NotifierError("terminal Odyssey state still names a live worker")
        if status == "worker_complete":
            _validate_completed_postflight(context, state)
    else:
        raise NotifierError("canonical Odyssey supervisor state has an unsupported status")
    return state


def keychain(service: str) -> str | None:
    try:
        result = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-a", ACCOUNT, "-s", service, "-w"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def latest_health() -> tuple[Path, dict[str, Any]] | None:
    """Return only a fresh, sealed, canonical supervisor state.

    In particular, an old nested rehearsal receipt can never become a live
    health report merely because it has a newer modification time.
    """
    context = _sealed_run_context()
    if context is None:
        return None
    path = context.run_root / "SUPERVISOR_STATE.json"
    if not path.exists():
        return None
    return path, _validate_supervisor_state(path, context, now=time.time())


def payload(health: dict[str, Any]) -> tuple[str, str]:
    status = health.get("status")
    binding = _digest({"authority_sha256": health.get("authority_sha256"), "run_id": health.get("run_id")})[:24]
    if status in {"worker_complete", "terminal_safe_hold"}:
        event_id = f"odyssey-terminal/{binding}"
        terminal = "worker complete" if status == "worker_complete" else "safe hold"
        return (
            event_id,
            f"🧭 Substrate Odyssey 7D — terminal status: {terminal}. One terminal notification was recorded; inspect the local sealed state for details.",
        )
    if status != "worker_running":
        raise NotifierError("unsupported Odyssey notification state")
    fields = (
        "day",
        "elapsed_seconds",
        "completion_percent",
        "microcycles_complete",
        "resident_memory",
        "free_storage",
        "storage_guard",
        "broker_action",
        "next_boundary",
        "cpu_time_deltas",
    )
    missing = [field for field in fields if field not in health]
    if missing:
        raise RuntimeError(f"missing Odyssey health fields: {missing}")
    bucket = int(time.time() // BUCKET)
    event_id = f"odyssey-health/{binding}/{bucket}"
    cpu = health["cpu_time_deltas"]
    active_cores = float(cpu["active_cores_equivalent"])
    logical_cores = int(cpu["logical_cores_available"])
    text = (
        f"📊 Substrate Odyssey 7D — day {health['day']}; {float(health['completion_percent']):.1f}%; "
        f"microcycles: {health['microcycles_complete']}; RAM: {_human_bytes(health['resident_memory'])}; "
        f"cores: {active_cores:.2f}/{logical_cores}; "
        f"storage: {_human_bytes(health['free_storage'])} / guard {_human_bytes(health['storage_guard'])}; "
        f"broker: {health['broker_action']}; next: {health['next_boundary']}."
    )
    return event_id, text


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, RuntimeError):
        return None


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "pending"
    gib = value / (1024**3)
    return f"{gib:.1f} GiB"


def _preflight_capacity_forecast() -> str:
    """Render the conservative, model-reserved pre-rehearsal envelope.

    This is intentionally an upper-bound preview only.  G07 remains the sole
    authority for a measured launch guard, so an unreadable draft or a missing
    measurement becomes an explicit pending state rather than a guess.
    """
    try:
        reserve = _read_json(SHARED_STORAGE_RESERVE, label="Odyssey shared-storage reserve")
        policy = reserve.get("shared_post_r2_capacity_policy")
        if not isinstance(policy, dict):
            raise NotifierError("Odyssey shared-storage reserve policy is absent")
        model = policy.get("model_ladder_reservation_bytes_decimal")
        private_cap = policy.get("private_write_cap_bytes")
        transient = policy.get("largest_transient_window_bytes")
        terminal = policy.get("terminal_allowance_bytes")
        values = (model, private_cap, transient, terminal)
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values):
            raise NotifierError("Odyssey shared-storage reserve values are invalid")
        required = BASE_PROTECTED_FLOOR_BYTES + model + private_cap + 2 * transient + terminal
        free = shutil.disk_usage(ROOT).free
        margin = free - required
        return (
            f"max-cap preview: {_human_bytes(free)} free; {_human_bytes(required)} envelope "
            f"({_human_bytes(BASE_PROTECTED_FLOOR_BYTES)} base + {_human_bytes(model)} model + "
            f"{_human_bytes(private_cap)} private + 2×{_human_bytes(transient)} transient + {_human_bytes(terminal)} terminal); "
            f"{_human_bytes(abs(margin))} {'margin' if margin >= 0 else 'shortfall'}; G07 measurement pending"
        )
    except NotifierError:
        return "max-cap preview unavailable; G07 measurement pending"


def _readiness_gates() -> dict[str, str]:
    """Read the authority-validated preflight status without mutating it."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "substrate.odyssey7d", "readiness", "--root", str(ROOT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {}
    gates = result.get("gates") if isinstance(result, dict) else None
    if not isinstance(gates, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in gates.items()):
        return {}
    return gates


def preflight_snapshot() -> dict[str, Any]:
    """Build a non-scientific, read-only status view for the preflight phase."""
    try:
        _current_frozen_build()
    except NotifierError:
        # Do not derive a reassuring status from a transition receipt, design,
        # R2 result, or gate ledger until the exact frozen source map validates.
        # Disk free space is a live OS fact and is the sole safe preview here.
        return {
            "source_freeze_valid": False,
            "transition_state": "source_freeze_verification_pending",
            "preflight_authorized": False,
            "r2_status": "source/freeze verification pending",
            "r2_checks": {},
            "gates_passed": 0,
            "gates_total": 15,
            "completion_percent": 0.0,
            "free_storage": shutil.disk_usage(ROOT).free,
            "storage_guard": None,
            "blockers": ["source/freeze verification pending"],
        }
    waiting_transition = _optional_json(PREFLIGHT_RUNS / "TRANSITION_STATE.json") or {}
    verified_transition = _optional_json(PREFLIGHT_RUNS / "R2_VERIFIED_ODYSSEY_PREFLIGHT_AUTHORIZATION.json") or {}
    frozen = _optional_json(ROOT / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json") or {}
    transition = (
        verified_transition
        if (verified_transition.get("state") == "odyssey_preflight_authorized" and verified_transition.get("frozen_build_sha256") == frozen.get("sha256"))
        else waiting_transition
    )
    design = _optional_json(ROOT / "docs/plans/substrate/tangible_next_launch/ODYSSEY_7D.hardened.draft.json") or {}
    result = _optional_json(ROOT / "evidence/substrate/tangible_sandbox/SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json") or {}
    continuity = _optional_json(ROOT / "evidence/substrate/tangible_sandbox/SUBSTRATE_SANDBOX_R2_CONTINUITY_VERIFICATION.json") or {}
    provenance = _optional_json(ROOT / "evidence/substrate/tangible_sandbox/SUBSTRATE_SANDBOX_R2_PROVENANCE_VERIFICATION.json") or {}
    gates = design.get("launch_gates", [])
    passed = sum(1 for gate in gates if gate.get("status") == "pass")
    r2_checks = {
        "result": (
            result.get("scientific_status") == "complete" and result.get("continuity_passing") is True and float(result.get("actual_wall_hours", 0)) >= 24
        ),
        "continuity": (continuity.get("scientific_status") == "pass" and continuity.get("independently_verified") is True),
        "source_provenance": (provenance.get("scientific_status") == "pass" and provenance.get("independently_verified") is True),
    }
    if all(r2_checks.values()):
        r2_status = "verified"
    elif r2_checks["result"] and r2_checks["continuity"]:
        r2_status = "continuity verified; source provenance pending"
    else:
        r2_status = "verification pending"
    if transition.get("state") == "odyssey_preflight_authorized":
        passed = max(passed, 1)
    validated_gates = _readiness_gates()
    if validated_gates:
        passed = sum(1 for status in validated_gates.values() if status == "pass")
    guard = design.get("storage", {}).get("measured_guard_bytes")
    blockers = [str(gate.get("name", gate.get("id", "unnamed"))) for gate in gates if validated_gates.get(str(gate.get("id")), gate.get("status")) != "pass"]
    if transition.get("state") == "waiting_for_verified_r2":
        reason = transition.get("details", {}).get("reason", "R2 verification pending")
        blockers.insert(0, reason.replace("_", " "))
    return {
        "source_freeze_valid": True,
        "transition_state": transition.get("state", "not_started"),
        "preflight_authorized": transition.get("preflight_authorized") is True,
        "r2_status": r2_status,
        "r2_checks": r2_checks,
        "gates_passed": passed,
        "gates_total": len(gates),
        "completion_percent": round((passed / len(gates) * 100) if gates else 0.0, 1),
        "free_storage": shutil.disk_usage(ROOT).free,
        "storage_guard": guard,
        "blockers": blockers,
    }


def preflight_payload() -> tuple[str, str]:
    snapshot = preflight_snapshot()
    bucket = int(dt.datetime.now(dt.UTC).timestamp() // BUCKET)
    # The machine ticks every two minutes, but its user-facing preflight
    # contract is exactly one status update per thirty-minute bucket.  Gate
    # changes remain visible in the next bucket instead of creating bursts of
    # otherwise-identical notifications while the owner is away.
    event_id = f"odyssey-preflight/{bucket}"
    if snapshot["source_freeze_valid"] is False:
        return (
            event_id,
            f"🧭 Substrate Odyssey preflight — source/freeze verification pending; "
            f"0/{snapshot['gates_total']} trusted gates; no Odyssey worker admitted; "
            f"cores: no Odyssey worker admitted / {os.cpu_count() or 1} logical available; "
            f"free: {_human_bytes(snapshot['free_storage'])}; storage guard: pending.",
        )
    blockers = snapshot["blockers"][:3]
    blocker_text = ", ".join(blockers) if blockers else "none"
    text = (
        f"🧭 Substrate Odyssey preflight — {snapshot['completion_percent']:.1f}% "
        f"({snapshot['gates_passed']}/{snapshot['gates_total']} gates); "
        f"R2: {snapshot['r2_status']}; transition: {snapshot['transition_state']}; "
        f"cores: no Odyssey worker admitted / {os.cpu_count() or 1} logical available; "
        f"free: {_human_bytes(snapshot['free_storage'])}; "
        f"storage guard: {_human_bytes(snapshot['storage_guard'])}; "
        f"{_preflight_capacity_forecast()}; blockers: {blocker_text}."
    )
    return event_id, text


def send(text: str) -> int:
    token, chat_id = keychain(TOKEN_SERVICE), keychain(CHAT_SERVICE)
    if not token or not chat_id:
        raise NotifierError("Telegram Keychain credentials unavailable")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=urllib.parse.urlencode({"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": "true"}).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read())
    except (OSError, TimeoutError, URLError, json.JSONDecodeError) as error:
        raise NotifierError(f"Telegram transport error: {type(error).__name__}") from error
    result = body.get("result") if isinstance(body, dict) and body.get("ok") is True else None
    message_id = result.get("message_id") if isinstance(result, dict) else None
    if not isinstance(message_id, int) or isinstance(message_id, bool):
        raise NotifierError("Telegram rejected notification")
    return message_id


def probe(frozen_build_sha256: str) -> dict[str, Any]:
    """Send one explicit, non-scientific acknowledgement probe."""
    if len(frozen_build_sha256) != 64 or any(char not in "0123456789abcdef" for char in frozen_build_sha256):
        raise RuntimeError("invalid frozen-build digest for Telegram probe")
    message_id = send(f"🧭 Substrate Odyssey Telegram probe acknowledged — frozen build {frozen_build_sha256[:12]}; preflight notifications are armed.")
    return {
        "state": "probe_delivered",
        "delivered": True,
        "message_id": message_id,
        "frozen_build_sha256": frozen_build_sha256,
    }


def _ledger_lock_path() -> Path:
    return STATE.with_name(f".{STATE.name}.lock")


def _new_ledger() -> dict[str, Any]:
    return {"schema": LEDGER_SCHEMA, "sent": {}}


def _read_ledger() -> dict[str, Any]:
    if not STATE.exists():
        return _new_ledger()
    ledger = _read_json(STATE, label="Odyssey notifier delivery ledger")
    # Migrate the short-lived pre-hardening shape only when it contains an
    # unambiguous map of Telegram integer message IDs.  Any other unknown
    # ledger is refused rather than risking a historical replay.
    if "schema" not in ledger and set(ledger) == {"sent"} and isinstance(ledger.get("sent"), dict):
        legacy = ledger["sent"]
        if not all(isinstance(event, str) and isinstance(message_id, int) and not isinstance(message_id, bool) for event, message_id in legacy.items()):
            raise NotifierError("legacy Odyssey notifier ledger is malformed")
        return {
            "schema": LEDGER_SCHEMA,
            "sent": {event: {"message_id": message_id, "migrated": True} for event, message_id in legacy.items()},
        }
    sent = ledger.get("sent")
    if ledger.get("schema") != LEDGER_SCHEMA or not isinstance(sent, dict):
        raise NotifierError("Odyssey notifier delivery ledger is malformed")
    for event, record in sent.items():
        if not isinstance(event, str) or not event or not isinstance(record, dict):
            raise NotifierError("Odyssey notifier delivery ledger has an invalid event")
        message_id = record.get("message_id")
        if not isinstance(message_id, int) or isinstance(message_id, bool):
            raise NotifierError("Odyssey notifier delivery ledger has an invalid message ID")
    return ledger


def _atomic_write_ledger(ledger: dict[str, Any]) -> None:
    """Durably replace the small delivery ledger while its exclusive lock is held."""
    STATE.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{STATE.name}.", suffix=".tmp", dir=STATE.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(ledger, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, STATE)
        directory = os.open(STATE.parent, os.O_RDONLY)
        try:
            # Filesystems that cannot fsync directories still received the
            # atomic same-directory replacement above.
            with contextlib.suppress(OSError):
                os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as error:
        raise NotifierError("cannot atomically write Odyssey notifier ledger") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _acquire_ledger_lock() -> Any | None:
    lock_path = _ledger_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        os.chmod(lock_path, 0o600)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    except OSError as error:
        handle.close()
        raise NotifierError("cannot acquire Odyssey notifier ledger lock") from error
    return handle


def tick(deliver: bool, phase: str = "run") -> dict[str, Any]:
    if phase == "preflight":
        path, health = None, None
        event_id, text = preflight_payload()
    else:
        latest = latest_health()
        if latest is None:
            return {"state": "waiting_for_sealed_odyssey_run", "delivered": False}
        path, health = latest
        event_id, text = payload(health)
    lock = _acquire_ledger_lock()
    if lock is None:
        return {"state": "notifier_busy", "delivered": False}
    try:
        ledger = _read_ledger()
        sent = ledger["sent"]
        if event_id in sent:
            return {"state": "already_delivered", "event_id": event_id, "delivered": False}
        if not deliver:
            return {"state": "dry_run", "event_id": event_id, "text": text, "delivered": False}
        message_id = send(text)
        sent[event_id] = {
            "message_id": message_id,
            "delivered_at_epoch": round(time.time(), 3),
            "phase": phase,
            "terminal": health is not None and health.get("status") in {"worker_complete", "terminal_safe_hold"},
        }
        _atomic_write_ledger(ledger)
        result = {"state": "delivered", "event_id": event_id, "message_id": message_id, "delivered": True, "phase": phase}
        if path is not None:
            result["source"] = "canonical_supervisor_state"
        return result
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            lock.close()


def launchd_job() -> dict[str, Any]:
    return {
        "Label": LABEL,
        "ProgramArguments": [sys.executable, str(Path(__file__).resolve()), "tick", "--deliver"],
        "WorkingDirectory": str(ROOT),
        "StartInterval": INTERVAL,
        "RunAtLoad": False,
        "ProcessType": "Background",
        "Umask": 0o077,
        "StandardOutPath": str(ROOT / "runs/substrate/odyssey7d/telegram.stdout.log"),
        "StandardErrorPath": str(ROOT / "runs/substrate/odyssey7d/telegram.stderr.log"),
    }


def prepare() -> dict[str, Any]:
    path = ROOT / "docs/plans/substrate/tangible_next_launch/ODYSSEY_7D.telegram.launchd.template.plist"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        plistlib.dump(launchd_job(), handle, sort_keys=True)
    return {"prepared": str(path), "activation": False, "label": LABEL}


def preflight_launchd_job() -> dict[str, Any]:
    return {
        "Label": PREFLIGHT_LABEL,
        "ProgramArguments": [sys.executable, str(Path(__file__).resolve()), "tick", "--phase", "preflight", "--deliver"],
        "WorkingDirectory": str(ROOT),
        "StartInterval": INTERVAL,
        "RunAtLoad": True,
        "ProcessType": "Background",
        "ThrottleInterval": 60,
        "Umask": 0o077,
        "StandardOutPath": str(PREFLIGHT_RUNS / "preflight-telegram.stdout.log"),
        "StandardErrorPath": str(PREFLIGHT_RUNS / "preflight-telegram.stderr.log"),
    }


def prepare_preflight() -> dict[str, Any]:
    PREFLIGHT_PLIST.parent.mkdir(parents=True, exist_ok=True)
    with PREFLIGHT_PLIST.open("wb") as handle:
        plistlib.dump(preflight_launchd_job(), handle, sort_keys=True)
    return {"prepared": str(PREFLIGHT_PLIST), "activation": False, "label": PREFLIGHT_LABEL}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "prepare-preflight", "probe", "tick"))
    parser.add_argument("--deliver", action="store_true")
    parser.add_argument("--phase", choices=("run", "preflight"), default="run")
    parser.add_argument("--frozen-build")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare()
        elif args.command == "prepare-preflight":
            result = prepare_preflight()
        elif args.command == "probe":
            if args.frozen_build is None:
                raise RuntimeError("probe requires --frozen-build")
            result = probe(args.frozen_build)
        else:
            result = tick(args.deliver, args.phase)
    except RuntimeError as error:
        # Do not echo a transport URL, Keychain-derived value, or malformed
        # untrusted receipt text into a launchd log.
        print(json.dumps({"state": "delivery_retryable_failure", "error": "notifier_refused_or_delivery_failed", "delivered": False}))
        raise SystemExit(1) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
