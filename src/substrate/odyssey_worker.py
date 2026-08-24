"""Deterministic, custody-preserving executor for a sealed Substrate Odyssey.

This module owns execution only.  It neither materializes hidden answers nor
scores a scientific result.  A sealed authority supplies candidate-visible task
manifests and the two arm adapters; an independent evaluator is released only
after this worker locks the complete candidate/control trace.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from substrate import odyssey_task_bank as task_bank
from substrate import odyssey_transition

PROGRAM = "substrate-odyssey-7d-v1"
TEST_PROGRAM = "substrate-odyssey-7d-test-v1"
PHASES = ("retrieval", "exposure", "transfer", "repair_checkpoint")
FRONTIERS = tuple("ABCDEFGH")
GIB = 1024 ** 3
TELEMETRY_INTERVAL_SECONDS = 30.0


class Refused(RuntimeError):
    """Raised when an authority or execution boundary is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Refused(f"cannot read JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Refused(f"JSON object required at {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)
    # Replacing a synced temporary file is not enough to make the directory
    # entry durable across a restart.  This intentionally fails rather than
    # claiming a recoverable checkpoint on a filesystem that cannot flush it.
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _authority_digest(authority: dict[str, Any]) -> str:
    claimed = authority.get("sha256")
    unsigned = dict(authority)
    unsigned.pop("sha256", None)
    observed = _digest(unsigned)
    if not isinstance(claimed, str) or claimed != observed:
        raise Refused("sealed authority requires an exact self-digest")
    return observed


def _inside(root: Path, value: str, *, label: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        raise Refused(f"{label} must be root-relative")
    resolved_root = root.resolve()
    resolved = (root / raw).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise Refused(f"{label} escapes the repository root")
    return resolved


def _parse_cpu_seconds(value: str) -> float:
    """Parse the portable ``ps time`` shape (``[[days-]hh:]mm:ss``)."""
    text = value.strip()
    if not text:
        raise ValueError("empty process CPU time")
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        days = int(day_text)
    raw_parts = text.split(":")
    if len(raw_parts) < 2 or len(raw_parts) > 3:
        raise ValueError(f"unrecognised process CPU time: {value!r}")
    try:
        parts = [int(part) for part in raw_parts[:-1]] + [float(raw_parts[-1])]
    except ValueError as exc:
        raise ValueError(f"unrecognised process CPU time: {value!r}") from exc
    if len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
        valid_minutes = minutes >= 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        valid_minutes = 0 <= minutes < 60
    else:
        raise ValueError(f"unrecognised process CPU time: {value!r}")
    if hours < 0 or not valid_minutes or seconds < 0 or seconds >= 60:
        raise ValueError(f"invalid process CPU time: {value!r}")
    return float((((days * 24) + hours) * 60 + minutes) * 60 + seconds)


@dataclass(frozen=True)
class _Process:
    pid: int
    ppid: int
    rss_bytes: int
    cpu_seconds: float


@dataclass(frozen=True)
class _ProcessSample:
    monotonic_seconds: float
    processes: dict[int, _Process]


def _process_sample_via_ps() -> dict[int, _Process]:
    """Primary process-table sample via ``ps`` (full CPU time when available)."""
    completed = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,rss=,time="],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        message = completed.stderr.strip() or "ps exited nonzero"
        raise OSError(message)
    processes: dict[int, _Process] = {}
    malformed = 0
    for raw in completed.stdout.splitlines():
        fields = raw.split(maxsplit=3)
        if len(fields) != 4:
            malformed += 1
            continue
        try:
            pid, ppid, rss_kib = (int(fields[0]), int(fields[1]), int(fields[2]))
            cpu_seconds = _parse_cpu_seconds(fields[3])
        except ValueError:
            malformed += 1
            continue
        if pid > 0 and ppid >= 0 and rss_kib >= 0:
            processes[pid] = _Process(pid, ppid, rss_kib * 1024, cpu_seconds)
    if not processes:
        raise Refused("host telemetry sampling returned no usable processes")
    if malformed > len(processes):
        raise Refused("host telemetry sampling was materially malformed")
    return processes


def _process_sample_via_libproc() -> dict[int, _Process]:
    """macOS fallback when ``ps`` is blocked (seatbelt) but libproc remains usable.

    Returns observed RSS and PPID per PID via ``PROC_PIDTASKINFO`` /
    ``PROC_PIDTBSDINFO``.  CPU seconds are left at zero — the memory broker
    only needs RSS for admit/hold decisions.  Fail closed when no process
    rows are readable.
    """
    try:
        import ctypes
        import ctypes.util
    except ImportError as error:
        raise OSError("ctypes unavailable for libproc telemetry") from error
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        raise OSError("libc unavailable for libproc telemetry")
    libc = ctypes.CDLL(libc_name, use_errno=True)
    PROC_ALL_PIDS = 1
    PROC_PIDTBSDINFO = 3
    PROC_PIDTASKINFO = 4
    proc_listpids = libc.proc_listpids
    proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
    proc_listpids.restype = ctypes.c_int
    proc_pidinfo = libc.proc_pidinfo
    proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    proc_pidinfo.restype = ctypes.c_int

    size = proc_listpids(PROC_ALL_PIDS, 0, None, 0)
    if size <= 0:
        raise OSError("proc_listpids returned no space")
    buf = (ctypes.c_int * (size // ctypes.sizeof(ctypes.c_int)))()
    filled = proc_listpids(PROC_ALL_PIDS, 0, ctypes.byref(buf), size)
    if filled <= 0:
        raise OSError("proc_listpids failed to fill pid list")
    count = filled // ctypes.sizeof(ctypes.c_int)

    class proc_taskinfo(ctypes.Structure):
        _fields_ = [
            ("pti_virtual_size", ctypes.c_uint64),
            ("pti_resident_size", ctypes.c_uint64),
            ("pti_total_user", ctypes.c_uint64),
            ("pti_total_system", ctypes.c_uint64),
            ("pti_threads_user", ctypes.c_uint64),
            ("pti_threads_system", ctypes.c_uint64),
            ("pti_policy", ctypes.c_int32),
            ("pti_faults", ctypes.c_int32),
            ("pti_pageins", ctypes.c_int32),
            ("pti_cow_faults", ctypes.c_int32),
            ("pti_messages_sent", ctypes.c_int32),
            ("pti_messages_received", ctypes.c_int32),
            ("pti_syscalls_mach", ctypes.c_int32),
            ("pti_syscalls_unix", ctypes.c_int32),
            ("pti_csw", ctypes.c_int32),
            ("pti_threadnum", ctypes.c_int32),
            ("pti_numrunning", ctypes.c_int32),
            ("pti_priority", ctypes.c_int32),
        ]

    class proc_bsdinfo(ctypes.Structure):
        _fields_ = [
            ("pbi_flags", ctypes.c_uint32),
            ("pbi_status", ctypes.c_uint32),
            ("pbi_xstatus", ctypes.c_uint32),
            ("pbi_pid", ctypes.c_uint32),
            ("pbi_ppid", ctypes.c_uint32),
            ("pbi_uid", ctypes.c_uint32),
            ("pbi_gid", ctypes.c_uint32),
            ("pbi_ruid", ctypes.c_uint32),
            ("pbi_rgid", ctypes.c_uint32),
            ("pbi_svuid", ctypes.c_uint32),
            ("pbi_svgid", ctypes.c_uint32),
            ("rfu_1", ctypes.c_uint32),
            ("pbi_comm", ctypes.c_char * 16),
            ("pbi_name", ctypes.c_char * 32),
            ("pbi_nfiles", ctypes.c_uint32),
            ("pbi_pgid", ctypes.c_uint32),
            ("pbi_pjobc", ctypes.c_uint32),
            ("e_tdev", ctypes.c_uint32),
            ("e_tpgid", ctypes.c_uint32),
            ("pbi_nice", ctypes.c_int32),
            ("pbi_start_tvsec", ctypes.c_uint64),
            ("pbi_start_tvusec", ctypes.c_uint64),
        ]

    task_size = ctypes.sizeof(proc_taskinfo)
    bsd_size = ctypes.sizeof(proc_bsdinfo)
    processes: dict[int, _Process] = {}
    for index in range(count):
        pid = int(buf[index])
        if pid <= 0:
            continue
        task = proc_taskinfo()
        bsd = proc_bsdinfo()
        got_task = proc_pidinfo(pid, PROC_PIDTASKINFO, 0, ctypes.byref(task), task_size)
        got_bsd = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, ctypes.byref(bsd), bsd_size)
        if got_task < task_size or got_bsd < bsd_size:
            continue
        ppid = int(bsd.pbi_ppid)
        rss = int(task.pti_resident_size)
        if rss >= 0 and ppid >= 0:
            processes[pid] = _Process(pid, ppid, rss, 0.0)
    if not processes:
        raise Refused("host telemetry sampling returned no usable processes")
    return processes


def _process_sample(*, monotonic: Callable[[], float] = time.monotonic) -> _ProcessSample:
    """Capture one OS-provided process-table sample.

    ``rss`` is intentionally summed conservatively.  Shared pages can be
    counted more than once, so this is a guard input, not a claim about the
    physical-memory allocator.  It is nevertheless an observed value, unlike
    the old supervisor placeholders.

    Prefer ``ps``; when the host blocks it (seatbelt), fall back to libproc so
    memory-broker admission still observes real RSS rather than inventing zeros.
    """
    try:
        processes = _process_sample_via_ps()
    except (OSError, Refused, FileNotFoundError, PermissionError):
        try:
            processes = _process_sample_via_libproc()
        except (OSError, Refused, AttributeError, ValueError) as error:
            raise Refused(f"host telemetry sampling failed: {error}") from error
    return _ProcessSample(monotonic(), processes)


def _descendant_pids(processes: dict[int, _Process], root_pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for process in processes.values():
        children.setdefault(process.ppid, []).append(process.pid)
    found: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in found:
            continue
        found.add(pid)
        pending.extend(children.get(pid, ()))
    return found


def _broker_action_for_bytes(host_rss_bytes: int, resident_cap_bytes: int) -> str:
    if resident_cap_bytes != 85 * GIB:
        raise Refused("resident-memory cap must remain exactly 85 GiB")
    resident_gib = host_rss_bytes / GIB
    if resident_gib >= 85:
        return "safe_hold_non_p0"
    if resident_gib >= 82:
        return "pause_p1_checkpoint_p2"
    if resident_gib >= 80:
        return "checkpoint_reduce_p2"
    if resident_gib >= 75:
        return "deny_new_work"
    return "admit_or_resume"


def _telemetry_from_samples(
    *,
    current: _ProcessSample,
    previous: _ProcessSample | None,
    root_pid: int,
    resident_cap_bytes: int,
) -> dict[str, Any]:
    """Derive a transparent, interval-based telemetry record from ``ps``."""
    descendants = _descendant_pids(current.processes, root_pid)
    tree = [current.processes[pid] for pid in descendants if pid in current.processes]
    host_rss_bytes = sum(process.rss_bytes for process in current.processes.values())
    tree_rss_bytes = sum(process.rss_bytes for process in tree)
    cpu_delta: float | None = None
    interval_seconds: float | None = None
    equivalent_active_cores: float | None = None
    if previous is not None:
        interval_seconds = max(0.0, current.monotonic_seconds - previous.monotonic_seconds)
        # New descendants contribute their observed lifetime CPU; exited
        # descendants cannot be reconstructed, so the measurement is labelled
        # conservative rather than promoted to a calibration result.
        cpu_delta = sum(
            max(0.0, process.cpu_seconds - previous.processes[process.pid].cpu_seconds)
            if process.pid in previous.processes
            else process.cpu_seconds
            for process in tree
        )
        if interval_seconds > 0:
            equivalent_active_cores = round(cpu_delta / interval_seconds, 6)
    return {
        "sample_source": "ps -A -o pid,ppid,rss,time",
        "sampling_interval_target_seconds": TELEMETRY_INTERVAL_SECONDS,
        "sampling_interval_seconds": None if interval_seconds is None else round(interval_seconds, 6),
        "logical_cores_available": os.cpu_count() or 0,
        "sampled_process_count": len(current.processes),
        "worker_tree_process_count": len(tree),
        "worker_tree_pids": sorted(process.pid for process in tree),
        "worker_process_observed": root_pid in current.processes,
        "active_cores_equivalent": equivalent_active_cores,
        "worker_tree_cpu_seconds_delta": None if cpu_delta is None else round(cpu_delta, 6),
        "host_rss_bytes": host_rss_bytes,
        "worker_tree_rss_bytes": tree_rss_bytes,
        "resident_cap_bytes": resident_cap_bytes,
        "broker_action": _broker_action_for_bytes(host_rss_bytes, resident_cap_bytes),
        "memory_broker_certification": "observational_telemetry_only_not_G08_certification",
    }


def observed_runtime_telemetry(
    root_pid: int,
    *,
    resident_cap_bytes: int = 85 * GIB,
    previous: _ProcessSample | None = None,
) -> tuple[dict[str, Any], _ProcessSample]:
    """Return one non-invasive OS observation for a worker process tree.

    This helper is deliberately operational: callers must not treat it as the
    calibrated width-eight memory-broker evidence required by G08 or as the
    interruption/recovery evidence required by G09.
    """
    current = _process_sample()
    return (
        _telemetry_from_samples(
            current=current,
            previous=previous,
            root_pid=root_pid,
            resident_cap_bytes=resident_cap_bytes,
        ),
        current,
    )


@dataclass
class _TelemetryRecorder:
    """Write bounded, OS-observed execution telemetry without touching adapters.

    The recorder samples only the process table at a thirty-second cadence.  It
    never changes scheduler placement, model state, or task order.  A sampled
    cap crossing is a fail-closed admission signal, not a calibration receipt.
    """

    path: Path
    authority_sha256: str
    run_id: str
    resident_cap_bytes: int
    total_phase_count: int
    root_pid: int = field(default_factory=os.getpid)
    _previous: _ProcessSample | None = field(default=None, init=False)
    _context: dict[str, Any] = field(default_factory=dict, init=False)
    _last_payload: dict[str, Any] | None = field(default=None, init=False)
    _failure: str | None = field(default=None, init=False)
    _cap_crossed: bool = field(default=False, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def update_context(
        self,
        *,
        completed_phase_count: int,
        cycle: int | None,
        phase: str | None,
        phase_status: str,
        active_frontiers: list[str] | None = None,
    ) -> None:
        with self._lock:
            self._context = {
                "completed_phase_count": completed_phase_count,
                "total_phase_count": self.total_phase_count,
                "completion_percent": round(100 * completed_phase_count / self.total_phase_count, 6),
                "cycle": cycle,
                "phase": phase,
                "phase_status": phase_status,
                "active_frontiers": list(active_frontiers or ()),
            }

    def _payload(self, sample: _ProcessSample | None, *, error: str | None = None) -> dict[str, Any]:
        with self._lock:
            context = dict(self._context)
            previous = self._previous
        payload: dict[str, Any] = {
            "schema": "SUBSTRATE_ODYSSEY_LIVE_TELEMETRY/v1",
            "activation": False,
            "authority_sha256": self.authority_sha256,
            "run_id": self.run_id,
            "worker_pid": self.root_pid,
            "sampled_at_epoch": time.time(),
            "sampling_mode": "non_invasive_process_table_observation",
            "memory_broker_certification": "observational_telemetry_only_not_G08_certification",
            **context,
        }
        if sample is None:
            payload.update(
                {
                    "sample_status": "unavailable_fail_closed",
                    "telemetry_error": error,
                    "sampling_interval_target_seconds": TELEMETRY_INTERVAL_SECONDS,
                    "resident_cap_bytes": self.resident_cap_bytes,
                    "broker_action": "safe_hold_non_p0",
                }
            )
        else:
            payload.update(
                {
                    "sample_status": "observed",
                    **_telemetry_from_samples(
                        current=sample,
                        previous=previous,
                        root_pid=self.root_pid,
                        resident_cap_bytes=self.resident_cap_bytes,
                    ),
                }
            )
        payload["sha256"] = _digest(payload)
        return payload

    def sample(self) -> None:
        try:
            sample = _process_sample()
            payload = self._payload(sample)
            with self._lock:
                self._previous = sample
                self._last_payload = dict(payload)
                if payload["broker_action"] == "safe_hold_non_p0":
                    self._cap_crossed = True
        except (OSError, Refused, ValueError) as exc:
            message = str(exc)
            unavailable = self._payload(None, error=message)
            with self._lock:
                self._failure = message
                self._last_payload = unavailable
            try:
                _write_json(self.path, unavailable)
            except OSError as write_error:
                # The next foreground guard converts this into a refusal.  Do
                # not silently continue because the telemetry receipt itself
                # could not become durable.
                with self._lock:
                    self._failure = f"{message}; cannot durably write telemetry failure: {write_error}"
            return
        try:
            _write_json(self.path, payload)
        except OSError as exc:
            with self._lock:
                self._failure = f"cannot durably write live telemetry: {exc}"

    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._last_payload or {})

    def start(self) -> None:
        self.sample()
        thread = threading.Thread(target=self._loop, name="substrate-odyssey-telemetry", daemon=True)
        self._thread = thread
        thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(TELEMETRY_INTERVAL_SECONDS):
            self.sample()

    def assert_admissible(self) -> None:
        with self._lock:
            failure = self._failure
            cap_crossed = self._cap_crossed
        if failure:
            raise Refused(f"worker telemetry unavailable: {failure}")
        if cap_crossed:
            raise Refused("worker telemetry observed the 85 GiB resident-memory cap")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=TELEMETRY_INTERVAL_SECONDS + 5)
        # The final record makes a short test run observable too.  It remains
        # an observation, not a substitute for G08/G09 rehearsal evidence.
        self.sample()


def _sealed(authority: dict[str, Any]) -> bool:
    gates = authority.get("launch_gates")
    return (
        authority.get("program", {}).get("launch_allowed") is True
        and authority.get("seal", {}).get("status") == "sealed"
        and isinstance(gates, list)
        and bool(gates)
        and all(isinstance(gate, dict) and gate.get("status") == "pass" for gate in gates)
    )


def _assert_phase_within_budget(*, started: float, phase_seconds: int, monotonic: Callable[[], float]) -> None:
    """Fail rather than silently stretching a sealed active phase."""
    elapsed = monotonic() - started
    if elapsed > phase_seconds:
        raise Refused(
            "paired adapter dispatch exceeded the sealed phase budget "
            f"({elapsed:.3f}s > {phase_seconds}s)"
        )


def _full_worker_contract(authority: dict[str, Any], worker: dict[str, Any]) -> None:
    program = authority.get("program_config", authority.get("program", {}))
    if program.get("id") != PROGRAM:
        raise Refused("full worker requires the exact Odyssey program identity")
    if program.get("duration_seconds") != 7 * 24 * 3600:
        raise Refused("full Odyssey must retain exactly seven days")
    if worker.get("test_mode") is True:
        raise Refused("full Odyssey cannot use test mode")
    entries = worker.get("frontiers")
    frontier_ids = [item.get("id") for item in entries] if isinstance(entries, list) else []
    if frontier_ids != list(FRONTIERS):
        raise Refused("full Odyssey must retain all eight frontiers in order")
    if worker.get("microcycles_per_frontier") != 84:
        raise Refused("full Odyssey must retain 84 microcycles per frontier")
    if worker.get("phase_names") != list(PHASES) or worker.get("phase_seconds") != 1800:
        raise Refused("full Odyssey must retain four 30-minute phases")
    if worker.get("max_parallel_frontiers") != 8:
        raise Refused("full Odyssey requires the calibrated width-eight admission")
    checkpoint = worker.get("checkpoint")
    if checkpoint != {"delta_interval_seconds": 7200, "full_interval_seconds": 43200}:
        raise Refused("full Odyssey must retain its two-hour delta and twelve-hour full checkpoints")


def _test_worker_contract(authority: dict[str, Any], worker: dict[str, Any]) -> None:
    program = authority.get("program_config", authority.get("program", {}))
    if program.get("id") != TEST_PROGRAM or worker.get("test_mode") is not True:
        raise Refused("non-full worker identities are limited to an explicit test program")
    entries = worker.get("frontiers")
    frontier_ids = [item.get("id") for item in entries] if isinstance(entries, list) else []
    if not frontier_ids or any(item not in FRONTIERS for item in frontier_ids):
        raise Refused("test program has invalid frontier identifiers")
    if not worker.get("phase_names") or not isinstance(worker.get("phase_seconds"), int):
        raise Refused("test program has an invalid phase schedule")
    if int(worker.get("microcycles_per_frontier", 0)) < 1:
        raise Refused("test program requires at least one microcycle")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_exact_frozen_map(
    *,
    name: str,
    expected: Any,
    current: dict[str, Path],
) -> None:
    if not isinstance(expected, dict) or set(expected) != set(current):
        raise Refused(f"frozen {name} source map does not exactly match the transition controller")
    for key, path in current.items():
        digest = expected.get(key)
        if not _is_sha256(digest) or not path.is_file() or file_digest(path) != digest:
            raise Refused(f"frozen {name} source drift: {key}")


def _validate_full_frozen_build(root: Path, authority: dict[str, Any]) -> None:
    """Bind full execution to the current transition freeze, not a lookalike.

    This intentionally re-evaluates both maps at dispatch.  A sealed authority
    cannot continue after any implementation or static-input drift, including
    a changed worker module between the authority seal and process launch.
    """
    frozen_path = root / "plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json"
    frozen = _read_json(frozen_path)
    _require_self_digest(frozen, label="current frozen build")
    frozen_sha256 = frozen["sha256"]
    if (
        frozen.get("schema") != "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1"
        or frozen.get("program") != odyssey_transition.PROGRAM
        or frozen.get("activation") is not False
        or frozen.get("scientific_status") != "frozen_waiting_for_verified_r2"
    ):
        raise Refused("current frozen build is not the inactive Odyssey transition build")
    seal = authority.get("seal")
    if (
        not _is_sha256(frozen_sha256)
        or authority.get("frozen_build_sha256") != frozen_sha256
        or not isinstance(seal, dict)
        or seal.get("frozen_build_sha256") != frozen_sha256
    ):
        raise Refused("full authority is not bound to the current self-digested frozen build")
    _validate_exact_frozen_map(
        name="input",
        expected=frozen.get("input_sha256"),
        current=odyssey_transition.build_inputs(root),
    )
    _validate_exact_frozen_map(
        name="implementation",
        expected=frozen.get("implementation_sha256"),
        current=odyssey_transition.implementation_inputs(root),
    )


def _validate_runtime_lease(
    root: Path,
    *,
    worker: dict[str, Any],
    authority_sha256: str,
    run_id: str,
) -> None:
    """Require an exact one-attempt supervisor lease for full execution only."""
    path_value = os.environ.get("SUBSTRATE_ODYSSEY_RUNTIME_LEASE_PATH")
    expected_digest = os.environ.get("SUBSTRATE_ODYSSEY_RUNTIME_LEASE_SHA256")
    if not path_value or not _is_sha256(expected_digest):
        raise Refused("full Odyssey worker requires an exact supervisor runtime lease environment")
    raw_path = Path(path_value)
    run_root = _inside(root, str(worker.get("run_root", "")), label="worker run root")
    leases_root = run_root / "leases"
    # Do not resolve the requested path before checking it: ``Path.resolve``
    # would erase the evidence that a lease (or its containing directory) was
    # a symlink.  The supervisor writes this exact, direct child path.
    requested_path = raw_path if raw_path.is_absolute() else root / raw_path
    if leases_root.is_symlink() or requested_path.parent != leases_root or requested_path.is_symlink():
        raise Refused("runtime lease must be a direct non-symlink child of worker.run_root/leases")
    lease_path = requested_path
    try:
        metadata = lease_path.stat()
    except OSError as exc:
        raise Refused(f"runtime lease is not readable: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
        raise Refused("runtime lease must be a regular mode-0600 file")
    lease = _read_json(lease_path)
    _require_self_digest(lease, label="runtime lease")
    allowed = {
        "schema",
        "activation",
        "authority_sha256",
        "run_id",
        "supervisor_pid",
        "attempt",
        "worker_argv_sha256",
        "issued_at_epoch",
        "sha256",
    }
    if set(lease) != allowed:
        raise Refused("runtime lease fields are not exact")
    attempt = lease.get("attempt")
    expected_name = f"attempt-{attempt:03d}.json" if type(attempt) is int and attempt >= 1 else None
    argv = worker.get("argv")
    argv_valid = isinstance(argv, list) and bool(argv) and all(isinstance(item, str) and item for item in argv)
    checks = {
        "schema": lease.get("schema") == "SUBSTRATE_ODYSSEY_SUPERVISOR_RUNTIME_LEASE/v1",
        "inactive": lease.get("activation") is False,
        "authority": lease.get("authority_sha256") == authority_sha256,
        "run": lease.get("run_id") == run_id,
        "supervisor_pid": type(lease.get("supervisor_pid")) is int and lease["supervisor_pid"] > 0,
        "attempt": expected_name is not None and lease_path.name == expected_name,
        "argv": argv_valid and lease.get("worker_argv_sha256") == _digest({"argv": argv}),
        "issued_at": isinstance(lease.get("issued_at_epoch"), (int, float)) and not isinstance(lease.get("issued_at_epoch"), bool),
        "environment": lease.get("sha256") == expected_digest,
        "parent": os.getppid() == lease.get("supervisor_pid"),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise Refused(f"runtime lease is invalid: {failed}")


def validate_authority(root: Path, authority_path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Validate only executable, non-scientific authority conditions."""
    authority_path = authority_path.resolve()
    expected = (root / "plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json").resolve()
    if authority_path != expected and not authority_path.name.startswith("ODYSSEY_7D.test."):
        raise Refused("worker requires the canonical sealed authority path")
    authority = _read_json(authority_path)
    authority_sha256 = _authority_digest(authority)
    if not _sealed(authority):
        raise Refused("authority is not sealed and fully admitted")
    worker = authority.get("worker")
    if not isinstance(worker, dict):
        raise Refused("authority lacks an executable worker block")
    expected_source = authority.get("worker_source_sha256") or worker.get("source_sha256")
    if expected_source is None:
        for row in worker.get("source_files", []):
            if isinstance(row, dict) and row.get("path") == "src/substrate/odyssey_worker.py":
                expected_source = row.get("sha256")
                break
    if expected_source != file_digest(Path(__file__)):
        raise Refused("worker source drifted after authority sealing")
    if any("evaluator" in key.lower() or "answer" in key.lower() for key in worker):
        raise Refused("worker authority may not contain evaluator-only inputs")
    if authority.get("program_config", authority.get("program", {})).get("id") == PROGRAM:
        _full_worker_contract(authority, worker)
        _validate_full_frozen_build(root, authority)
    else:
        _test_worker_contract(authority, worker)
    return authority, worker, authority_sha256


def _validate_full_source_bundle(root: Path, manifest: dict[str, Any], *, frontier: str) -> None:
    """Recheck the G03 candidate source assets immediately before dispatch."""
    bundle = manifest.get("source_bundle")
    if not isinstance(bundle, dict):
        raise Refused(f"full candidate manifest lacks a source bundle for frontier {frontier}")
    assets = bundle.get("assets")
    if not isinstance(assets, list) or not assets:
        raise Refused(f"full candidate source bundle has no assets for frontier {frontier}")
    observed_paths: set[str] = set()
    observed_roles: set[str] = set()
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise Refused(f"full candidate source asset {frontier}[{index}] is malformed")
        relative = asset.get("path")
        digest = asset.get("sha256")
        role = asset.get("role")
        if not isinstance(relative, str) or not relative or not _is_sha256(digest) or not isinstance(role, str) or not role:
            raise Refused(f"full candidate source asset {frontier}[{index}] has invalid metadata")
        path = _inside(root, relative, label=f"candidate source asset {frontier}[{index}]")
        normalized = str(path.relative_to(root))
        prohibited = ("evaluator", "answer", "scorer")
        if any(token in part.casefold() for part in Path(normalized).parts for token in prohibited):
            raise Refused(f"full candidate source asset {frontier}[{index}] resolves into evaluator-only namespace")
        if asset.get("read_only") is not True or not path.is_file() or file_digest(path) != digest:
            raise Refused(f"full candidate source asset {frontier}[{index}] is missing, writable, or drifted")
        if normalized in observed_paths or role in observed_roles:
            raise Refused(f"full candidate source bundle repeats an asset path or role for frontier {frontier}")
        observed_paths.add(normalized)
        observed_roles.add(role)
    if bundle.get("selection_sha256") != _digest({"frontier": frontier, "assets": assets}):
        raise Refused(f"full candidate source bundle selection digest drifted for frontier {frontier}")


def _manifest_for_frontier(root: Path, frontier: dict[str, Any], *, full: bool, task_count: int) -> dict[str, Any]:
    required = {"id", "candidate_manifest", "candidate_manifest_sha256", "candidate_command", "control_command"}
    if not required.issubset(frontier):
        raise Refused(f"frontier worker entry is incomplete: {frontier.get('id')}")
    manifest_path = _inside(root, str(frontier["candidate_manifest"]), label="candidate manifest")
    if file_digest(manifest_path) != frontier["candidate_manifest_sha256"]:
        raise Refused(f"candidate manifest drifted for frontier {frontier['id']}")
    manifest = _read_json(manifest_path)
    if manifest.get("frontier") != frontier["id"] or not task_bank.candidate_is_structurally_safe(manifest):
        raise Refused(f"candidate manifest is invalid or structurally unsafe for frontier {frontier['id']}")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < task_count:
        raise Refused(f"candidate manifest lacks scheduled tasks for frontier {frontier['id']}")
    if full and len(tasks) != task_count:
        raise Refused(f"full Odyssey manifest must have exactly {task_count} tasks per frontier")
    if full:
        _validate_full_source_bundle(root, manifest, frontier=str(frontier["id"]))
    return manifest


def _require_self_digest(record: dict[str, Any], *, label: str) -> None:
    claimed = record.get("sha256")
    unsigned = dict(record)
    unsigned.pop("sha256", None)
    if not isinstance(claimed, str) or claimed != _digest(unsigned):
        raise Refused(f"{label} integrity digest is invalid")


@dataclass(frozen=True)
class _TraceInfo:
    completed_phase_count: int
    event_chain_sha256: str
    chain_at_phase_boundary: dict[int, str]


@dataclass(frozen=True)
class _CheckpointInfo:
    sha256: str
    cycle: int
    completed_phase_count: int


@dataclass(frozen=True)
class _ResumeState:
    completed_phase_count: int
    event_chain_sha256: str
    checkpoint_sha256: str | None
    checkpoint_count: int
    complete: bool
    elapsed_seconds: float | None
    broker_hold_seconds: float


def _read_trace(
    path: Path,
    *,
    authority_sha256: str,
    run_id: str,
    frontier_ids: list[str],
    phases: list[str],
    expected_phases: int,
) -> _TraceInfo:
    """Verify the append-only trace and derive every durable phase boundary."""
    if not path.exists():
        return _TraceInfo(0, "", {0: ""})
    if not path.is_file():
        raise Refused("event trace path is not a regular file")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return _TraceInfo(0, "", {0: ""})
    if len(lines) % len(frontier_ids):
        raise Refused("event trace ends inside a paired frontier phase")
    completed = len(lines) // len(frontier_ids)
    if completed > expected_phases:
        raise Refused("event trace exceeds the sealed phase schedule")
    chain = ""
    boundaries = {0: ""}
    for row_index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Refused("event trace contains invalid JSON") from exc
        if not isinstance(row, dict):
            raise Refused("event trace contains a non-object row")
        phase_index, frontier_index = divmod(row_index, len(frontier_ids))
        cycle, phase_index_in_cycle = divmod(phase_index, len(phases))
        digest = row.pop("event_sha256", None)
        checks = {
            "schema": row.get("schema") == "SUBSTRATE_ODYSSEY_PAIRED_EVENT/v1",
            "inactive": row.get("activation") is False,
            "authority": row.get("authority_sha256") == authority_sha256,
            "run": row.get("run_id") == run_id,
            "frontier": row.get("frontier") == frontier_ids[frontier_index],
            "cycle": row.get("cycle") == cycle,
            "phase": row.get("phase") == phases[phase_index_in_cycle],
            "parent": row.get("previous_event_sha256") == chain,
            "digest": isinstance(digest, str) and digest == _digest(row),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise Refused(f"event trace chain or schedule is invalid: {failed}")
        chain = digest
        if frontier_index == len(frontier_ids) - 1:
            boundaries[phase_index + 1] = chain
    return _TraceInfo(completed, chain, boundaries)


def _read_state(
    path: Path,
    *,
    authority_sha256: str,
    run_id: str,
    expected_phases: int,
    frontier_count: int,
) -> _ResumeState | None:
    if not path.exists():
        return None
    state = _read_json(path)
    _require_self_digest(state, label="existing run state")
    if (
        state.get("schema") != "SUBSTRATE_ODYSSEY_WORKER_STATE/v1"
        or state.get("activation") is not False
        or state.get("authority_sha256") != authority_sha256
        or state.get("run_id") != run_id
    ):
        raise Refused("existing run state is bound to a different or invalid authority")
    completed = state.get("completed_phase_count")
    total = state.get("total_phase_count")
    paired = state.get("completed_paired_events")
    chain = state.get("event_chain_sha256")
    checkpoint = state.get("checkpoint_sha256")
    checkpoint_count = state.get("checkpoint_count")
    complete = state.get("complete")
    elapsed = state.get("elapsed_seconds")
    broker_hold_seconds = state.get("broker_hold_seconds", 0.0)
    if (
        not isinstance(completed, int)
        or not 0 <= completed <= expected_phases
        or total != expected_phases
        or paired != completed * frontier_count
        or not isinstance(chain, str)
        or (checkpoint is not None and not isinstance(checkpoint, str))
        or not isinstance(checkpoint_count, int)
        or checkpoint_count < 0
        or not isinstance(complete, bool)
        or (elapsed is not None and not isinstance(elapsed, (int, float)))
        or not isinstance(broker_hold_seconds, (int, float))
        or isinstance(broker_hold_seconds, bool)
        or broker_hold_seconds < 0
    ):
        raise Refused("existing state is malformed")
    if (completed == 0 and chain) or (completed > 0 and len(chain) != 64):
        raise Refused("existing state has an invalid event-chain cursor")
    if (checkpoint is None and checkpoint_count) or (checkpoint is not None and len(checkpoint) != 64):
        raise Refused("existing state has an invalid checkpoint cursor")
    return _ResumeState(
        completed_phase_count=completed,
        event_chain_sha256=chain,
        checkpoint_sha256=checkpoint,
        checkpoint_count=checkpoint_count,
        complete=complete,
        elapsed_seconds=None if elapsed is None else float(elapsed),
        broker_hold_seconds=float(broker_hold_seconds),
    )


def _write_state(path: Path, state: dict[str, Any]) -> None:
    value = dict(state)
    value.pop("sha256", None)
    value["sha256"] = _digest(value)
    _write_json(path, value)


def _checkpoint_records(
    checkpoints: Path,
    *,
    authority_sha256: str,
    phases: list[str],
    frontier_count: int,
    trace: _TraceInfo,
) -> list[_CheckpointInfo]:
    if not checkpoints.exists():
        return []
    if not checkpoints.is_dir():
        raise Refused("checkpoint path is not a directory")
    raw: list[tuple[Path, dict[str, Any]]] = []
    for path in checkpoints.glob("*.json"):
        record = _read_json(path)
        _require_self_digest(record, label=f"checkpoint {path.name}")
        raw.append((path, record))
    raw.sort(key=lambda item: item[1].get("cycle", -1))
    records: list[_CheckpointInfo] = []
    parent = ""
    for ordinal, (path, record) in enumerate(raw):
        cycle = record.get("cycle")
        kind = record.get("kind")
        completed = record.get("completed_phase_count")
        expected_completed = (ordinal + 1) * len(phases)
        expected_cycle = ordinal
        expected_kind = "full" if (expected_cycle + 1) % 6 == 0 else "delta"
        expected_name = f"{expected_kind}-{expected_cycle + 1:03d}.json"
        checks = {
            "schema": record.get("schema") == "SUBSTRATE_ODYSSEY_CHECKPOINT/v1",
            "inactive": record.get("activation") is False,
            "authority": record.get("authority_sha256") == authority_sha256,
            "cycle": cycle == expected_cycle,
            "kind": kind == expected_kind,
            "name": path.name == expected_name,
            "completed": completed == expected_completed,
            "paired_events": record.get("completed_paired_events") == expected_completed * frontier_count,
            "event_chain": record.get("event_chain_sha256") == trace.chain_at_phase_boundary.get(expected_completed),
            "parent": record.get("parent_checkpoint_sha256") == parent,
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise Refused(f"checkpoint chain is invalid: {failed}")
        digest = record["sha256"]
        records.append(_CheckpointInfo(digest, cycle, completed))
        parent = digest
    return records


def _reconcile_checkpoints(
    checkpoints: Path,
    *,
    authority_sha256: str,
    phases: list[str],
    frontier_count: int,
    trace: _TraceInfo,
) -> list[_CheckpointInfo]:
    """Repair only a post-trace/pre-checkpoint crash window from valid trace data."""
    records = _checkpoint_records(
        checkpoints,
        authority_sha256=authority_sha256,
        phases=phases,
        frontier_count=frontier_count,
        trace=trace,
    )
    completed_cycles = trace.completed_phase_count // len(phases)
    parent = records[-1].sha256 if records else ""
    for cycle in range(len(records), completed_cycles):
        completed = (cycle + 1) * len(phases)
        kind = "full" if (cycle + 1) % 6 == 0 else "delta"
        digest = _write_checkpoint(
            checkpoints / f"{kind}-{cycle + 1:03d}.json",
            authority_sha256=authority_sha256,
            kind=kind,
            cycle=cycle,
            completed_phase_count=completed,
            completed_paired_events=completed * frontier_count,
            event_chain_sha256=trace.chain_at_phase_boundary[completed],
            parent_sha256=parent,
        )
        records.append(_CheckpointInfo(digest, cycle, completed))
        parent = digest
    return records


def _read_broker_observations(path: Path, *, authority_sha256: str, run_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    status = _read_json(path)
    _require_self_digest(status, label="broker status")
    if (
        status.get("schema") != "SUBSTRATE_ODYSSEY_BROKER_STATUS/v1"
        or status.get("activation") is not False
        or status.get("authority_sha256") != authority_sha256
        or status.get("run_id") != run_id
        or not isinstance(status.get("observations"), list)
    ):
        raise Refused("broker status is malformed or bound to a different run")
    observations: list[dict[str, Any]] = []
    parent = ""
    for sequence, observation in enumerate(status["observations"], start=1):
        if not isinstance(observation, dict):
            raise Refused("broker status contains a non-object observation")
        _require_self_digest(observation, label="broker observation")
        checks = {
            "schema": observation.get("schema") == "SUBSTRATE_ODYSSEY_BROKER_OBSERVATION/v1",
            "inactive": observation.get("activation") is False,
            "authority": observation.get("authority_sha256") == authority_sha256,
            "run": observation.get("run_id") == run_id,
            "sequence": observation.get("sequence") == sequence,
            "parent": observation.get("previous_observation_sha256") == parent,
            "action": observation.get("broker_action") in {
                "admit_or_resume",
                "deny_new_work",
                "checkpoint_reduce_p2",
                "pause_p1_checkpoint_p2",
                "safe_hold_non_p0",
            },
            "telemetry": _is_sha256(observation.get("observed_telemetry_sha256")),
            "hold_seconds": (
                "broker_hold_seconds" not in observation
                or (
                    isinstance(observation.get("broker_hold_seconds"), (int, float))
                    and not isinstance(observation.get("broker_hold_seconds"), bool)
                    and observation["broker_hold_seconds"] >= 0
                )
            ),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise Refused(f"broker observation chain is invalid: {failed}")
        parent = observation["sha256"]
        observations.append(observation)
    return observations


def _write_broker_observation(
    path: Path,
    *,
    observations: list[dict[str, Any]],
    authority_sha256: str,
    run_id: str,
    cycle: int,
    phase: str,
    completed_phase_count: int,
    observed: dict[str, Any],
    phase_boundary_status: str,
    implemented_action: str,
    no_new_adapter_work: bool,
    broker_hold_seconds: float | None = None,
) -> dict[str, Any]:
    previous = observations[-1]["sha256"] if observations else ""
    observation = {
        "schema": "SUBSTRATE_ODYSSEY_BROKER_OBSERVATION/v1",
        "activation": False,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "sequence": len(observations) + 1,
        "previous_observation_sha256": previous,
        "cycle": cycle,
        "phase": phase,
        "completed_phase_count": completed_phase_count,
        "observed_telemetry_sha256": observed.get("sha256"),
        "host_rss_bytes": observed.get("host_rss_bytes"),
        "resident_cap_bytes": observed.get("resident_cap_bytes"),
        "broker_action": observed.get("broker_action"),
        "phase_boundary_status": phase_boundary_status,
        "implemented_action": implemented_action,
        "no_new_adapter_work": no_new_adapter_work,
        "memory_broker_certification": "observational_telemetry_only_not_G08_certification",
    }
    if broker_hold_seconds is not None:
        if broker_hold_seconds < 0:
            raise Refused("broker hold duration cannot be negative")
        observation["broker_hold_seconds"] = round(broker_hold_seconds, 6)
    if not _is_sha256(observation["observed_telemetry_sha256"]):
        raise Refused("broker observation lacks a self-digested telemetry sample")
    if not isinstance(observation["host_rss_bytes"], int) or not isinstance(observation["resident_cap_bytes"], int):
        raise Refused("broker observation lacks measured resident-memory fields")
    observation["sha256"] = _digest(observation)
    updated = [*observations, observation]
    document = {
        "schema": "SUBSTRATE_ODYSSEY_BROKER_STATUS/v1",
        "activation": False,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "observations": updated,
    }
    document["sha256"] = _digest(document)
    _write_json(path, document)
    observations.append(observation)
    return observation


def _adapter(
    root: Path,
    *,
    authority_sha256: str,
    run_id: str,
    worker_root: Path,
    frontier: str,
    role: str,
    command: str | list[str],
    manifest_sha256: str,
    task: dict[str, Any],
    cycle: int,
    phase: str,
    abort_check: Callable[[], None] | None = None,
    pre_dispatch_check: Callable[[], None] | None = None,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    if deadline_monotonic is not None:
        if not isinstance(deadline_monotonic, (int, float)) or not math.isfinite(float(deadline_monotonic)):
            raise Refused("adapter deadline must be a finite monotonic timestamp")
        if time.monotonic() >= float(deadline_monotonic):
            raise Refused(f"{role} adapter reached the phase dispatch deadline before launch for {frontier}/{cycle}/{phase}")

    arm_root = worker_root / "arms" / frontier / role
    request_path = arm_root / "requests" / f"{cycle:03d}-{phase}.json"
    receipt_path = arm_root / "receipts" / f"{cycle:03d}-{phase}.json"
    request = {
        "schema": "SUBSTRATE_ODYSSEY_ADAPTER_REQUEST/v1",
        "activation": False,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "frontier": frontier,
        "role": role,
        "cycle": cycle,
        "phase": phase,
        "task": task,
        "candidate_manifest_sha256": manifest_sha256,
        "receipt_path": str(receipt_path.relative_to(root)),
    }
    request["request_sha256"] = _digest(request)
    _write_json(request_path, request)
    arguments = shlex.split(command) if isinstance(command, str) else list(command)
    if not arguments:
        raise Refused(f"{role} adapter command is empty")
    # Recheck immediately before every arm dispatch.  The full-program source
    # bundle is pinned anew here (rather than only at worker startup), and the
    # admission guard closes the candidate/control gap when a supervisor lease
    # is revoked or the memory guard changes.
    if pre_dispatch_check is not None:
        pre_dispatch_check()
    if abort_check is not None:
        abort_check()
    if deadline_monotonic is not None and time.monotonic() >= float(deadline_monotonic):
        raise Refused(f"{role} adapter reached the phase dispatch deadline before process launch for {frontier}/{cycle}/{phase}")
    process = subprocess.Popen(
        [*arguments, str(request_path)], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    adapter_deadline = time.monotonic() + 15 * 60
    phase_limited = deadline_monotonic is not None and float(deadline_monotonic) < adapter_deadline
    deadline = min(adapter_deadline, float(deadline_monotonic)) if deadline_monotonic is not None else adapter_deadline
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(arguments, 15 * 60)
            try:
                stdout, stderr = process.communicate(timeout=min(1.0, remaining))
                break
            except subprocess.TimeoutExpired:
                if abort_check is not None:
                    abort_check()
    except subprocess.TimeoutExpired as exc:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        if phase_limited:
            raise Refused(f"{role} adapter exceeded the phase dispatch deadline for {frontier}/{cycle}/{phase}") from exc
        raise Refused(f"{role} adapter timed out for {frontier}/{cycle}/{phase}") from exc
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise
    completed = subprocess.CompletedProcess(arguments, process.returncode, stdout, stderr)
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip() or "adapter exited nonzero"
        raise Refused(f"{role} adapter failed for {frontier}/{cycle}/{phase}: {message}")
    if not receipt_path.is_file():
        raise Refused(f"{role} adapter did not create its receipt")
    receipt = _read_json(receipt_path)
    checks = {
        "schema": receipt.get("schema") == "SUBSTRATE_ODYSSEY_ADAPTER_RECEIPT/v1",
        "inactive": receipt.get("activation") is False,
        "authority": receipt.get("authority_sha256") == authority_sha256,
        "run": receipt.get("run_id") == run_id,
        "frontier": receipt.get("frontier") == frontier,
        "role": receipt.get("role") == role,
        "cycle": receipt.get("cycle") == cycle,
        "phase": receipt.get("phase") == phase,
        "task": receipt.get("task_id") == task.get("task_id"),
        "manifest": receipt.get("candidate_manifest_sha256") == manifest_sha256,
        "request": receipt.get("request_sha256") == request["request_sha256"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise Refused(f"{role} receipt failed contract: {failed}")
    return {"receipt": receipt, "receipt_sha256": file_digest(receipt_path)}


def _dispatch_paired_frontier(
    root: Path,
    *,
    authority_sha256: str,
    run_id: str,
    worker_root: Path,
    frontier_entry: dict[str, Any],
    task: dict[str, Any],
    cycle: int,
    phase: str,
    full_source_guard: bool,
    task_count: int,
    abort_check: Callable[[], None] | None = None,
    phase_deadline_monotonic: float | None = None,
    dispatch_observer: Callable[[str, float], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run one production candidate/control lane without role-wave synchronization.

    The full worker and G06 both use this primitive.  A frontier's control may
    begin immediately after its candidate finishes while other frontier
    candidates are still running; callers must not add a global role barrier.
    ``phase_deadline_monotonic`` is an absolute deadline shared by all lanes
    in a phase and is enforced by the adapter subprocess watchdog.
    """
    frontier = frontier_entry.get("id")
    if not isinstance(frontier, str) or not frontier:
        raise Refused("paired dispatch frontier id is invalid")
    manifest_sha256 = frontier_entry.get("candidate_manifest_sha256")
    if not isinstance(manifest_sha256, str) or not _is_sha256(manifest_sha256):
        raise Refused(f"paired dispatch has an invalid candidate manifest digest for {frontier}")

    source_bundle_guard_calls = 0

    def source_bundle_dispatch_guard() -> None:
        nonlocal source_bundle_guard_calls
        if not full_source_guard:
            return
        source_bundle_guard_calls += 1
        current = _manifest_for_frontier(root, frontier_entry, full=True, task_count=task_count)
        tasks = current.get("tasks")
        if not isinstance(tasks, list) or not 0 <= cycle * len(PHASES) + PHASES.index(phase) < len(tasks):
            raise Refused(f"candidate manifest task index is invalid before dispatch for frontier {frontier}")
        current_task = tasks[cycle * len(PHASES) + PHASES.index(phase)]
        if not isinstance(current_task, dict) or current_task.get("task_id") != task.get("task_id"):
            raise Refused(f"candidate manifest task drifted before dispatch for frontier {frontier}")

    def observe(stage: str) -> None:
        if dispatch_observer is not None:
            dispatch_observer(stage, time.monotonic())

    observe("candidate_started")
    candidate = _adapter(
        root,
        authority_sha256=authority_sha256,
        run_id=run_id,
        worker_root=worker_root,
        frontier=frontier,
        role="candidate",
        command=frontier_entry["candidate_command"],
        manifest_sha256=manifest_sha256,
        task=task,
        cycle=cycle,
        phase=phase,
        abort_check=abort_check,
        pre_dispatch_check=source_bundle_dispatch_guard,
        deadline_monotonic=phase_deadline_monotonic,
    )
    observe("candidate_finished")
    observe("control_started")
    control = _adapter(
        root,
        authority_sha256=authority_sha256,
        run_id=run_id,
        worker_root=worker_root,
        frontier=frontier,
        role="control",
        command=frontier_entry["control_command"],
        manifest_sha256=manifest_sha256,
        task=task,
        cycle=cycle,
        phase=phase,
        abort_check=abort_check,
        pre_dispatch_check=source_bundle_dispatch_guard,
        deadline_monotonic=phase_deadline_monotonic,
    )
    observe("control_finished")
    return frontier, {
        "schema": "SUBSTRATE_ODYSSEY_PAIRED_EVENT/v1",
        "activation": False,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "frontier": frontier,
        "cycle": cycle,
        "phase": phase,
        "task_id": task.get("task_id"),
        "candidate_receipt_sha256": candidate["receipt_sha256"],
        "control_receipt_sha256": control["receipt_sha256"],
        "candidate_elapsed_seconds": candidate["receipt"].get("elapsed_seconds"),
        "control_elapsed_seconds": control["receipt"].get("elapsed_seconds"),
        "source_bundle_guard_calls": source_bundle_guard_calls,
    }


def _write_checkpoint(
    path: Path,
    *,
    authority_sha256: str,
    kind: str,
    cycle: int,
    completed_phase_count: int,
    completed_paired_events: int,
    event_chain_sha256: str,
    parent_sha256: str,
) -> str:
    record = {
        "schema": "SUBSTRATE_ODYSSEY_CHECKPOINT/v1",
        "activation": False,
        "authority_sha256": authority_sha256,
        "kind": kind,
        "cycle": cycle,
        "completed_phase_count": completed_phase_count,
        "completed_paired_events": completed_paired_events,
        "event_chain_sha256": event_chain_sha256,
        "parent_checkpoint_sha256": parent_sha256,
    }
    record["sha256"] = _digest(record)
    _write_json(path, record)
    return record["sha256"]


def run(
    root: Path,
    *,
    authority_file: Path,
    monotonic: Callable[[], float] = time.monotonic,
    epoch: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute or safely resume a sealed authority.  It never invokes an evaluator."""
    root = root.resolve()
    authority, worker, authority_sha256 = validate_authority(root, authority_file)
    full = authority.get("program_config", authority["program"])["id"] == PROGRAM
    run_id = authority.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise Refused("sealed authority has no run id")
    if full:
        # The supervisor intentionally calls validate_authority before it can
        # mint this lease.  The child consumes it at the first executable line
        # after validation, before manifests, storage work, or adapter launch.
        _validate_runtime_lease(root, worker=worker, authority_sha256=authority_sha256, run_id=run_id)
    entries = worker.get("frontiers")
    if not isinstance(entries, list):
        raise Refused("worker frontier entries are missing")
    frontier_ids = [item.get("id") for item in entries if isinstance(item, dict)]
    cycles = int(worker["microcycles_per_frontier"])
    phases = list(worker.get("phase_names", PHASES))
    phase_seconds = int(worker["phase_seconds"])
    max_parallel = int(worker.get("max_parallel_frontiers", 8 if full else 1))
    if max_parallel < 1 or max_parallel > len(frontier_ids):
        raise Refused("worker has an invalid parallel frontier width")
    expected_phases = cycles * len(phases)
    run_root = _inside(root, str(worker.get("run_root", "runs/substrate/odyssey7d/v1")), label="run root")
    if run_root == root:
        raise Refused("worker run root cannot be repository root")
    storage = authority.get("storage", worker.get("storage", {}))
    if (
        not isinstance(storage, dict)
        or not isinstance(storage.get("required_free_bytes"), int)
        or not isinstance(storage.get("launch_required_free_bytes"), int)
    ):
        raise Refused("worker storage guard is missing")
    required_free = int(storage["required_free_bytes"])
    launch_required_free = int(storage["launch_required_free_bytes"])
    if required_free <= 0 or launch_required_free < required_free:
        raise Refused("worker storage guard is invalid")
    if shutil.disk_usage(root).free < launch_required_free:
        raise Refused("worker lacks the sealed launch free-space admission")
    resident_cap = int(worker.get("resident_cap_bytes", 85 * GIB))
    if resident_cap != 85 * GIB:
        raise Refused("worker resident-memory cap must remain exactly 85 GiB")
    if [item.get("id") for item in entries if isinstance(item, dict)] != frontier_ids:
        raise Refused("worker frontier entries do not match its deterministic frontier order")
    manifests = {
        item["id"]: _manifest_for_frontier(root, item, full=full, task_count=expected_phases)
        for item in entries
    }
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / "worker.lock"
    lock_handle = lock_path.open("a+")
    telemetry: _TelemetryRecorder | None = None
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refused("another Odyssey worker owns this run") from exc
        trace_path = run_root / "EVENTS.jsonl"
        state_path = run_root / "STATE.json"
        checkpoints = run_root / "checkpoints"
        trace = _read_trace(
            trace_path,
            authority_sha256=authority_sha256,
            run_id=run_id,
            frontier_ids=frontier_ids,
            phases=phases,
            expected_phases=expected_phases,
        )
        checkpoint_records = _reconcile_checkpoints(
            checkpoints,
            authority_sha256=authority_sha256,
            phases=phases,
            frontier_count=len(frontier_ids),
            trace=trace,
        )
        resume = _read_state(
            state_path,
            authority_sha256=authority_sha256,
            run_id=run_id,
            expected_phases=expected_phases,
            frontier_count=len(frontier_ids),
        )
        if resume is not None:
            if resume.complete:
                raise Refused("sealed Odyssey run is already complete")
            if resume.completed_phase_count > trace.completed_phase_count:
                raise Refused("durable state is ahead of its event trace")
            if resume.event_chain_sha256 != trace.chain_at_phase_boundary[resume.completed_phase_count]:
                raise Refused("durable state does not match its event trace")
            state_checkpoint_count = resume.completed_phase_count // len(phases)
            state_checkpoint = checkpoint_records[state_checkpoint_count - 1].sha256 if state_checkpoint_count else None
            if (
                resume.checkpoint_count != state_checkpoint_count
                or resume.checkpoint_sha256 != state_checkpoint
            ):
                raise Refused("durable state does not match its checkpoint chain")
        completed_phases = trace.completed_phase_count
        chain = trace.event_chain_sha256
        previous_checkpoint = checkpoint_records[-1].sha256 if checkpoint_records else ""
        checkpoint_count = len(checkpoint_records)
        broker_hold_seconds = resume.broker_hold_seconds if resume is not None else 0.0
        telemetry_path = run_root / "LIVE_TELEMETRY.json"
        telemetry = _TelemetryRecorder(
            telemetry_path,
            authority_sha256,
            run_id,
            resident_cap,
            expected_phases,
        )
        if completed_phases < expected_phases:
            next_cycle, next_phase_index = divmod(completed_phases, len(phases))
            next_phase = phases[next_phase_index]
            next_cycle_display: int | None = next_cycle + 1
        else:
            next_phase = None
            next_cycle_display = None
        telemetry.update_context(
            completed_phase_count=completed_phases,
            cycle=next_cycle_display,
            phase=next_phase,
            phase_status="resume_validation",
        )
        telemetry.start()
        def runtime_admission_guard() -> None:
            telemetry.assert_admissible()
            if full:
                _validate_runtime_lease(root, worker=worker, authority_sha256=authority_sha256, run_id=run_id)

        runtime_admission_guard()
        last_adapter_latency: dict[str, Any] = {"status": "not_observed_yet"}
        broker_path = run_root / "BROKER_STATUS.json"
        broker_observations = _read_broker_observations(
            broker_path,
            authority_sha256=authority_sha256,
            run_id=run_id,
        )
        broker_ledger_hold_seconds = sum(
            float(observation.get("broker_hold_seconds", 0.0))
            for observation in broker_observations
        )
        # A broker restoration is independently durable before any following
        # adapter work.  If a process dies in the small gap before STATE.json
        # is replaced, recover that completed hold from the broker ledger.  A
        # state value ahead of the ledger, conversely, proves lost/tampered
        # durable broker history and must not be papered over.
        if resume is not None and resume.broker_hold_seconds > broker_ledger_hold_seconds + 0.001:
            raise Refused("durable state is ahead of its broker-hold ledger")
        broker_hold_seconds = max(broker_hold_seconds, broker_ledger_hold_seconds)

        def state_document(
            *,
            completed: int,
            elapsed_seconds: float,
            complete: bool,
            extra: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            observed = telemetry.latest()
            document: dict[str, Any] = {
                "schema": "SUBSTRATE_ODYSSEY_WORKER_STATE/v1",
                "activation": False,
                "authority_sha256": authority_sha256,
                "run_id": run_id,
                "completed_phase_count": completed,
                "total_phase_count": expected_phases,
                "completed_paired_events": completed * len(frontier_ids),
                "event_chain_sha256": chain,
                "elapsed_seconds": round(elapsed_seconds, 3),
                "target_seconds": cycles * len(phases) * phase_seconds,
                "completion_percent": round(100 * completed / expected_phases, 6),
                "broker_hold_seconds": round(broker_hold_seconds, 6),
                "timing_policy": "scheduled_active_time_preserved_broker_holds_extend_wall_schedule",
                "disk_free_bytes": shutil.disk_usage(root).free,
                "disk_guard_bytes": required_free,
                "telemetry_path": str(telemetry_path.relative_to(root)),
                "telemetry_sample_status": observed.get("sample_status"),
                "active_cores_equivalent": observed.get("active_cores_equivalent"),
                "host_rss_bytes": observed.get("host_rss_bytes"),
                "worker_tree_rss_bytes": observed.get("worker_tree_rss_bytes"),
                "resident_cap_bytes": resident_cap,
                "broker_action": observed.get("broker_action"),
                "memory_broker_certification": "observational_telemetry_only_not_G08_certification",
                "broker_observation_sha256": broker_observations[-1]["sha256"] if broker_observations else None,
                "adapter_latency_seconds": last_adapter_latency,
                "checkpoint_sha256": previous_checkpoint or None,
                "checkpoint_count": checkpoint_count,
                "heartbeat_at_epoch": epoch(),
                "complete": complete,
            }
            if extra:
                document.update(extra)
            return document

        # A valid trace plus a matching checkpoint chain is the recovery
        # source of truth when a crash happens after durable trace/checkpoint
        # writes but before the state replacement.  No task is replayed.
        if resume is None or resume.completed_phase_count != completed_phases:
            _write_state(
                state_path,
                state_document(
                    completed=completed_phases,
                    elapsed_seconds=(resume.elapsed_seconds if resume and resume.elapsed_seconds is not None else completed_phases * phase_seconds),
                    complete=False,
                    extra={"resume_recovered_from_trace": completed_phases > 0},
                ),
            )
        start_mono = monotonic() - completed_phases * phase_seconds

        def await_safe_phase_admission(*, phase_index: int, cycle: int, phase: str) -> float:
            """Hold *before* adapter dispatch until RSS drops below 75 GiB.

            The 80/82 policy labels describe the requested P2/P1 posture, but
            this worker has no hidden P1/P2 side channel to pretend it changed.
            Its concrete, stronger action is a durable all-adapter hold at the
            phase boundary until the observed broker returns safe admission.
            """
            prior_action = broker_observations[-1].get("broker_action") if broker_observations else None
            held = False
            hold_started: float | None = None
            hold_descriptions = {
                "deny_new_work": (
                    "holding_before_adapter_dispatch",
                    "hold_all_new_adapter_work_until_host_rss_below_75_gib",
                ),
                "checkpoint_reduce_p2": (
                    "holding_before_adapter_dispatch_with_durable_boundary",
                    "durable_boundary_hold_all_new_adapter_work_until_host_rss_below_75_gib",
                ),
                "pause_p1_checkpoint_p2": (
                    "holding_before_adapter_dispatch_with_durable_boundary",
                    "durable_boundary_hold_all_adapter_work_until_host_rss_below_75_gib",
                ),
            }
            while True:
                runtime_admission_guard()
                observed = telemetry.latest()
                action = observed.get("broker_action")
                if action == "admit_or_resume":
                    if held:
                        # ``held`` can become true only after the monotonic
                        # start marker is taken below.  Keep this explicit so
                        # a corrupt/control-flow regression cannot silently
                        # manufacture a zero-duration boundary hold.
                        if hold_started is None:
                            raise Refused("broker hold is missing its monotonic start marker")
                        hold_seconds = max(0.0, monotonic() - hold_started)
                        _write_broker_observation(
                            broker_path,
                            observations=broker_observations,
                            authority_sha256=authority_sha256,
                            run_id=run_id,
                            cycle=cycle,
                            phase=phase,
                            completed_phase_count=phase_index,
                            observed=observed,
                            phase_boundary_status="safe_admission_restored",
                            implemented_action="adapter_dispatch_permitted_after_observed_safe_admission",
                            no_new_adapter_work=False,
                            broker_hold_seconds=hold_seconds,
                        )
                        return hold_seconds
                    return 0.0
                if action not in hold_descriptions:
                    raise Refused(f"worker received an unknown broker action: {action!r}")
                status, implemented = hold_descriptions[action]
                if not held or action != prior_action:
                    _write_broker_observation(
                        broker_path,
                        observations=broker_observations,
                        authority_sha256=authority_sha256,
                        run_id=run_id,
                        cycle=cycle,
                        phase=phase,
                        completed_phase_count=phase_index,
                        observed=observed,
                        phase_boundary_status=status,
                        implemented_action=implemented,
                        no_new_adapter_work=True,
                    )
                    prior_action = action
                if not held:
                    hold_started = monotonic()
                held = True
                telemetry.update_context(
                    completed_phase_count=phase_index,
                    cycle=cycle + 1,
                    phase=phase,
                    phase_status=f"broker_hold_{action}",
                    active_frontiers=[],
                )
                telemetry.sample()
                runtime_admission_guard()
                sleep(TELEMETRY_INTERVAL_SECONDS)

        for phase_index in range(completed_phases, expected_phases):
            cycle, phase_index_in_cycle = divmod(phase_index, len(phases))
            phase = phases[phase_index_in_cycle]
            telemetry.update_context(
                completed_phase_count=phase_index,
                cycle=cycle + 1,
                phase=phase,
                phase_status="running",
                active_frontiers=frontier_ids,
            )
            telemetry.sample()
            runtime_admission_guard()
            free_bytes = shutil.disk_usage(root).free
            if free_bytes < required_free:
                raise Refused("worker crossed the sealed free-space guard")
            held_seconds = await_safe_phase_admission(phase_index=phase_index, cycle=cycle, phase=phase)
            if held_seconds:
                # A broker hold cannot silently eat into a scheduled 30-minute
                # phase.  Preserve active phase time and make the wall-clock
                # extension explicit in durable state/ledger records.
                broker_hold_seconds += held_seconds
                start_mono += held_seconds
            telemetry.update_context(
                completed_phase_count=phase_index,
                cycle=cycle + 1,
                phase=phase,
                phase_status="running_after_broker_admission",
                active_frontiers=frontier_ids,
            )
            phase_started = monotonic()
            task_index = phase_index
            phase_deadline = (phase_started + phase_seconds) if full else (phase_started + phase_seconds)
            # Density: refuse if the shared model gateway is not on the pinned
            # parallel-slot contract before any adapter work starts.  Test-mode
            # authorities stay on the same pin when a live gateway is present;
            # synthetic control-plane fixtures without a model service skip the
            # live check so G12 lock/admission mutations still exercise cleanly.
            try:
                from substrate.odyssey_density import DensityRefused, assert_ollama_num_parallel_pinned, order_frontier_entries_for_phase
            except ImportError as error:
                raise Refused(f"worker model-gateway pin module missing: {error}") from error
            try:
                assert_ollama_num_parallel_pinned(require_running=bool(full))
            except DensityRefused as error:
                if full:
                    raise Refused(f"worker model-gateway pin: {error}") from error
                # Non-full fixtures may run without a live ollama process.

            # Deadline-aware paired scheduling: order frontier pairs by slack,
            # estimated runtime, and checkpoint criticality.  Candidate and
            # control of a pair remain one unit (equal priority/queue class).
            checkpoint_criticality = 1 if phase_index_in_cycle == len(phases) - 1 else 0
            ordered_ids = order_frontier_entries_for_phase(
                [item["id"] for item in entries if isinstance(item, dict)],
                phase_deadline_monotonic=phase_deadline,
                checkpoint_criticality=checkpoint_criticality,
                now=phase_started,
            )
            entry_by_id = {item["id"]: item for item in entries if isinstance(item, dict)}
            ordered_entries = [entry_by_id[frontier_id] for frontier_id in ordered_ids if frontier_id in entry_by_id]

            def paired(
                item: dict[str, Any],
                *,
                selected_task_index: int = task_index,
                selected_cycle: int = cycle,
                selected_phase: str = phase,
                selected_phase_deadline: float | None = phase_deadline if full else None,
            ) -> tuple[str, dict[str, Any]]:
                frontier = item["id"]
                manifest = manifests[frontier]
                task = manifest["tasks"][selected_task_index]
                return _dispatch_paired_frontier(
                    root,
                    authority_sha256=authority_sha256,
                    run_id=run_id,
                    worker_root=run_root,
                    frontier_entry=item,
                    task=task,
                    cycle=selected_cycle,
                    phase=selected_phase,
                    full_source_guard=full,
                    task_count=expected_phases,
                    abort_check=runtime_admission_guard,
                    phase_deadline_monotonic=selected_phase_deadline,
                )

            # Cross-lane CPU/GPU overlap: ThreadPool runs pairs concurrently so
            # tool/CPU work on one lane can proceed while another lane holds the
            # model gateway.  Pairs are never split (fairness outranks throughput).
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
                results = list(pool.map(paired, ordered_entries))
            if full:
                _assert_phase_within_budget(
                    started=phase_started,
                    phase_seconds=phase_seconds,
                    monotonic=monotonic,
                )
            adapter_latency: dict[str, dict[str, float | None]] = {}
            for frontier, event in sorted(results):
                candidate_elapsed = event.get("candidate_elapsed_seconds")
                control_elapsed = event.get("control_elapsed_seconds")
                adapter_latency[frontier] = {
                    "candidate": float(candidate_elapsed)
                    if isinstance(candidate_elapsed, (int, float)) and candidate_elapsed >= 0
                    else None,
                    "control": float(control_elapsed)
                    if isinstance(control_elapsed, (int, float)) and control_elapsed >= 0
                    else None,
                }
            last_adapter_latency = adapter_latency
            with trace_path.open("a", encoding="utf-8") as trace:
                for _, row in sorted(results):
                    row["previous_event_sha256"] = chain
                    row["event_sha256"] = _digest(row)
                    chain = row["event_sha256"]
                    trace.write(json.dumps(row, sort_keys=True) + "\n")
                trace.flush()
                os.fsync(trace.fileno())
            completed_now = phase_index + 1
            telemetry.update_context(
                completed_phase_count=completed_now,
                cycle=cycle + 1,
                phase=phase,
                phase_status="boundary_durability",
                active_frontiers=frontier_ids,
            )
            telemetry.sample()
            runtime_admission_guard()
            free_bytes = shutil.disk_usage(root).free
            if free_bytes < required_free:
                raise Refused("worker crossed the sealed free-space guard")
            if phase_index_in_cycle == len(phases) - 1:
                kind = "full" if (cycle + 1) % 6 == 0 else "delta"
                checkpoint_path = checkpoints / (f"{kind}-{cycle + 1:03d}.json")
                previous_checkpoint = _write_checkpoint(
                    checkpoint_path, authority_sha256=authority_sha256, kind=kind, cycle=cycle,
                    completed_phase_count=completed_now,
                    completed_paired_events=completed_now * len(frontier_ids),
                    event_chain_sha256=chain, parent_sha256=previous_checkpoint,
                )
                checkpoint_count += 1
            elapsed = monotonic() - start_mono
            _write_state(
                state_path,
                state_document(
                    completed=completed_now,
                    elapsed_seconds=elapsed,
                    complete=False,
                    extra={"adapter_latency_seconds": adapter_latency},
                ),
            )
            deadline = start_mono + completed_now * phase_seconds
            while monotonic() < deadline:
                runtime_admission_guard()
                if shutil.disk_usage(root).free < required_free:
                    raise Refused("worker crossed the sealed free-space guard")
                sleep(min(30.0, deadline - monotonic()))
        telemetry.update_context(
            completed_phase_count=expected_phases,
            cycle=None,
            phase=None,
            phase_status="all_execution_phases_complete",
        )
        telemetry.sample()
        runtime_admission_guard()
        trace_lock = {
            "schema": "SUBSTRATE_ODYSSEY_TRACE_LOCK/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "trace": str(trace_path.relative_to(root)),
            "trace_sha256": file_digest(trace_path),
            "event_chain_sha256": chain,
            "paired_events": expected_phases * len(frontier_ids),
            "checkpoint_sha256": previous_checkpoint or None,
            "checkpoint_count": checkpoint_count,
            "locked_before_evaluator_release": True,
        }
        trace_lock["sha256"] = _digest(trace_lock)
        publication_root = _inside(root, str(worker.get("publication_root", "evidence/substrate/odyssey7d")), label="publication root")
        _write_json(publication_root / "TRACE_LOCK.json", trace_lock)
        release = {
            "schema": "SUBSTRATE_ODYSSEY_EVALUATOR_RELEASE_REQUEST/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "trace_lock_sha256": trace_lock["sha256"],
            "action": "independent_evaluator_may_now_receive_custodian_owned_answers",
            "worker_accessed_evaluator_answers": False,
        }
        release["sha256"] = _digest(release)
        _write_json(publication_root / "EVALUATOR_RELEASE_REQUEST.json", release)
        _write_state(
            state_path,
            state_document(
                completed=expected_phases,
                elapsed_seconds=monotonic() - start_mono,
                complete=True,
                extra={
                    "trace_lock_sha256": trace_lock["sha256"],
                    "evaluator_release_request_sha256": release["sha256"],
                    "completed_at_epoch": epoch(),
                },
            ),
        )
        return {
            "status": "trace_locked_waiting_for_independent_evaluation",
            "run_id": run_id,
            "paired_events": expected_phases * len(frontier_ids),
            "trace_lock": str((publication_root / "TRACE_LOCK.json").relative_to(root)),
            "activation": False,
        }
    except Exception as error:
        _write_json(
            run_root / "FAILURE.json",
            {
                "schema": "SUBSTRATE_ODYSSEY_WORKER_FAILURE/v1",
                "activation": False,
                "authority_sha256": authority_sha256,
                "run_id": authority.get("run_id"),
                "reason": str(error),
                "failed_at_epoch": epoch(),
            },
        )
        raise
    finally:
        if telemetry is not None:
            telemetry.stop()
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authority", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, authority_file=args.authority)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
