"""Dynamic worker controller: the standard adaptive worker sizer for idle-host compute.

This module answers one operational question for the local idle-host lane: given a live
host snapshot, how many single-threaded seeded-sha256 worker capsules should the pool run
right now, and at what OS priority should they run so they coexist with the external
Hawking CPU workload?

Boundary (hard):
- The worker count is RECEIPT-INVARIANT. Every capsule's result is a seeded sha256 over a
  fixed seed; running one worker or twenty changes wall time only, never a byte of any
  receipt. This controller therefore gates no evidence, changes no seed, threshold, or
  control, and touches no sealed artifact. It is advisory operational telemetry only.
- Ownership: the sampler only READS process telemetry. It never signals, pauses, or names
  any external process. Hawking is coexisted with by BACKING OFF our own pool and by
  LOWERING our own priority, never by touching Hawking.

Design:
- ``recommended_workers(host_state)`` is a PURE function: deterministic given the host
  state, with no wall clock and no randomness. Hysteresis is expressed through the
  ``current_workers`` field carried on the state, so the same state always yields the same
  next size. The live reads that build a host state live only in ``sample_host_state``.
- Control law: ramp UP slowly (bounded additive increase) and back off FAST. Under Hawking
  the pool sheds to a tiny reserve; the priority lever (nice level / taskpolicy QoS class)
  then lets the surviving pool yield cores to Hawking by PRIORITY rather than by dropping
  more workers, giving continuous core-sharing.
- Ceiling: the measured Hawking-idle aggregate throughput peaks at 20 workers on this host,
  so 20 is the hard ceiling; 24 regresses and is never recommended.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mop.studio import external_coexistence as coexistence

try:  # psutil is a hard dependency in this repo; guard only so the pure core imports anywhere.
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

SCHEMA = "mop-dynamic-worker-controller/v1"
CLAIM_SCOPE = (
    "advisory operational worker sizing only; the worker count is receipt-invariant (seeded "
    "sha256), so this gates no evidence and changes no receipt, seed, threshold, or control"
)

# Measured Hawking-idle aggregate throughput (millions of hashes per second) sweeping the
# pool size on this host (Apple M3 Ultra: 20 performance + 8 efficiency cores, 96 GiB unified
# memory). 24 workers regress below the 20-worker peak, so 20 is the throughput-peak ceiling.
MEASURED_AGGREGATE_MHS = {8: 20.5, 16: 33.0, 20: 35.6, 24: 35.2}
WORKER_CEILING = 20
WORKER_FLOOR = 1

# Host geometry (documentation constants; the pure core only reads the fields on HostState).
PERFORMANCE_CORES = 20
EFFICIENCY_CORES = 8
LOGICAL_CORES = 28
UNIFIED_MEMORY_GB = 96.0

# Under Hawking the worker COUNT drops to this tiny reserve; the priority lever does the rest.
HAWKING_RESERVE_WORKERS = 2
# Cores held back from the pool for the efficiency-core cluster, the OS, and Hawking headroom.
CORE_RESERVE = 4
# Conservative per-worker resident footprint. The seeded-sha256 capsule measured 0.21 to 0.56
# GB resident; 0.75 GB leaves memory non-binding on a 96 GiB host so cores are the real limit.
PER_WORKER_GB = 0.75
# Ramp UP by at most this many workers per tick (slow); backing off is never rate limited (fast).
RAMP_UP_STEP = 1
# Comfortable-target dips within this many workers of the current pool are treated as load
# jitter and absorbed (held), so the pool does not oscillate on a noisy load trace.
DOWN_DEADBAND = 2
# One-minute load per reference core above which we HOLD (stop growing) rather than shed.
MAX_LOAD_PER_CORE = 0.90
LOAD_REFERENCE_CORES = LOGICAL_CORES

# Recommended nice levels (higher yields more). Idle stays below interactive work; under
# Hawking we run at the gentlest tier so cores flow to Hawking by priority.
IDLE_NICE = 5
PRESSURE_NICE = 15
HAWKING_NICE = 20

# Hawking command markers. The serial-profile markers are reused verbatim from the reviewed
# external_coexistence profile; the v5 markers name the block-parallel quantizer and the
# doctor_v5 adapters that burn cores in the current live workload.
_SERIAL_MARKERS = tuple(relative for relative, _ in coexistence.LIVE_HAWKING_FILE_SHA256)
HAWKING_COMMAND_MARKERS = _SERIAL_MARKERS + (
    "quantize-model-block-parallel",
    "strand-block-parallel",
    "doctor_v5_",
)


class WorkerControllerRefused(ValueError):
    """Raised when a host state or policy is malformed. The controller fails closed."""


def _require_nonneg_int(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WorkerControllerRefused(f"{label} must be a nonnegative integer")


def _require_pos_int(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise WorkerControllerRefused(f"{label} must be a positive integer")


def _require_nonneg_number(value: float, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise WorkerControllerRefused(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise WorkerControllerRefused(f"{label} must be a nonnegative finite number")
    return number


def _require_bool(value: bool, label: str) -> None:
    if not isinstance(value, bool):
        raise WorkerControllerRefused(f"{label} must be a boolean")


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class WorkerPolicy:
    """Immutable sizing knobs. The controller never widens these at runtime."""

    ceiling: int = WORKER_CEILING
    floor: int = WORKER_FLOOR
    hawking_reserve: int = HAWKING_RESERVE_WORKERS
    core_reserve: int = CORE_RESERVE
    per_worker_gb: float = PER_WORKER_GB
    ramp_up_step: int = RAMP_UP_STEP
    down_deadband: int = DOWN_DEADBAND
    max_load_per_core: float = MAX_LOAD_PER_CORE
    load_reference_cores: int = LOAD_REFERENCE_CORES
    schema: str = SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise WorkerControllerRefused("unsupported dynamic-worker-controller schema")
        if self.claim_scope != CLAIM_SCOPE:
            raise WorkerControllerRefused("dynamic-worker-controller claim scope cannot be widened")
        _require_pos_int(self.ceiling, "ceiling")
        _require_pos_int(self.floor, "floor")
        _require_pos_int(self.hawking_reserve, "hawking_reserve")
        _require_nonneg_int(self.core_reserve, "core_reserve")
        _require_pos_int(self.ramp_up_step, "ramp_up_step")
        _require_nonneg_int(self.down_deadband, "down_deadband")
        _require_pos_int(self.load_reference_cores, "load_reference_cores")
        if self.floor > self.ceiling:
            raise WorkerControllerRefused("floor cannot exceed ceiling")
        if self.hawking_reserve > self.ceiling:
            raise WorkerControllerRefused("hawking_reserve cannot exceed ceiling")
        if self.ceiling > WORKER_CEILING:
            raise WorkerControllerRefused("ceiling cannot exceed the measured throughput peak of 20")
        cap = _require_nonneg_number(self.per_worker_gb, "per_worker_gb")
        if cap <= 0.0:
            raise WorkerControllerRefused("per_worker_gb must be positive")
        frac = _require_nonneg_number(self.max_load_per_core, "max_load_per_core")
        if frac <= 0.0:
            raise WorkerControllerRefused("max_load_per_core must be positive")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "ceiling": self.ceiling,
            "floor": self.floor,
            "hawking_reserve": self.hawking_reserve,
            "core_reserve": self.core_reserve,
            "per_worker_gb": float(self.per_worker_gb),
            "ramp_up_step": self.ramp_up_step,
            "down_deadband": self.down_deadband,
            "max_load_per_core": float(self.max_load_per_core),
            "load_reference_cores": self.load_reference_cores,
        }


DEFAULT_POLICY = WorkerPolicy()


@dataclass(frozen=True, slots=True)
class HostState:
    """One immutable host snapshot the pure control law reasons over.

    ``current_workers`` is the pool size right now; it is the only carrier of history, which
    is what makes the ramp-up-slow / back-off-fast hysteresis a pure function of the state.
    """

    free_p_cores: int  # free processor cores available to the pool (logical minus busy; up to 28)
    hawking_active: bool  # is the quantize-model-block-parallel / doctor_v5 workload burning cores
    mem_available_gb: float
    thermal_ok: bool
    on_ac: bool
    current_load: float  # one-minute load average, whole host
    current_workers: int = 0

    def __post_init__(self) -> None:
        _require_nonneg_int(self.free_p_cores, "free_p_cores")
        _require_bool(self.hawking_active, "hawking_active")
        _require_nonneg_number(self.mem_available_gb, "mem_available_gb")
        _require_bool(self.thermal_ok, "thermal_ok")
        _require_bool(self.on_ac, "on_ac")
        _require_nonneg_number(self.current_load, "current_load")
        _require_nonneg_int(self.current_workers, "current_workers")

    def payload(self) -> dict[str, Any]:
        return {
            "free_p_cores": self.free_p_cores,
            "hawking_active": self.hawking_active,
            "mem_available_gb": float(self.mem_available_gb),
            "thermal_ok": self.thermal_ok,
            "on_ac": self.on_ac,
            "current_load": float(self.current_load),
            "current_workers": self.current_workers,
        }


@dataclass(frozen=True, slots=True)
class PriorityAdvice:
    """The fractional/priority lever: how the surviving pool yields cores by priority."""

    nice_level: int
    taskpolicy_class: str  # darwin taskpolicy -c QoS class
    yield_to_hawking: bool
    rationale: str

    def payload(self) -> dict[str, Any]:
        return {
            "nice_level": self.nice_level,
            "taskpolicy_class": self.taskpolicy_class,
            "yield_to_hawking": self.yield_to_hawking,
            "rationale": self.rationale,
        }


def worker_bounds(state: HostState, policy: WorkerPolicy = DEFAULT_POLICY) -> dict[str, Any]:
    """Return the three comfortable-target bounds and which one binds (advisory transparency)."""

    core_bound = state.free_p_cores - policy.core_reserve
    mem_bound = math.floor(state.mem_available_gb / policy.per_worker_gb)
    candidates = (
        ("ceiling", policy.ceiling),
        ("core_bound", core_bound),
        ("memory_bound", mem_bound),
    )
    binding_constraint, bound = min(candidates, key=lambda pair: pair[1])
    comfortable = _clamp(bound, policy.floor, policy.ceiling)
    return {
        "ceiling": policy.ceiling,
        "core_bound": core_bound,
        "memory_bound": mem_bound,
        "binding_constraint": binding_constraint,
        "comfortable_target": comfortable,
        "clamped_to_floor": bound < policy.floor,
    }


def recommended_workers(state: HostState, policy: WorkerPolicy = DEFAULT_POLICY) -> int:
    """Pure adaptive worker size in [floor, ceiling], deterministic given the host state.

    Tiers, highest priority first:
      1a. On battery: this idle-host lane must not draw power. Shed to the floor (fast).
      1b. Hawking burning cores: back off the worker COUNT to a tiny reserve (fast). The
          priority lever then yields the remaining cores to Hawking continuously.
      2.  Transient CPU/thermal ceiling: HOLD. A non-OOM signal never sheds, it only stops
          growth; a genuine resource shrink is still honored by the comfortable cap.
      3.  Comfortable: ramp UP by at most ramp_up_step per tick; back off FAST once the
          comfortable target falls more than down_deadband below the current pool. The
          deadband absorbs load jitter so the pool does not oscillate.
    """

    floor = policy.floor
    ceiling = policy.ceiling
    current = state.current_workers

    core_bound = state.free_p_cores - policy.core_reserve
    mem_bound = math.floor(state.mem_available_gb / policy.per_worker_gb)
    comfortable = _clamp(min(ceiling, core_bound, mem_bound), floor, ceiling)

    # Tier 1a: battery.
    if not state.on_ac:
        return floor
    # Tier 1b: Hawking. Fast backoff of the worker count to the reserve.
    if state.hawking_active:
        return _clamp(policy.hawking_reserve, floor, ceiling)
    # Tier 2: transient CPU or thermal ceiling. Hold, never shed for a non-OOM signal.
    oversubscribed = state.current_load > policy.max_load_per_core * policy.load_reference_cores
    if (not state.thermal_ok) or oversubscribed:
        base = current if current > 0 else floor
        return _clamp(min(base, comfortable), floor, ceiling)
    # Tier 3: comfortable ramp with asymmetric smoothing.
    if comfortable >= current + policy.ramp_up_step:
        return _clamp(current + policy.ramp_up_step, floor, ceiling)  # slow ramp up
    if comfortable > current:
        return _clamp(comfortable, floor, ceiling)  # small step up to the target
    if comfortable >= current - policy.down_deadband:
        return _clamp(current if current > 0 else floor, floor, ceiling)  # jitter: hold (hysteresis)
    return _clamp(comfortable, floor, ceiling)  # real drop past the deadband: back off fast


def recommended_priority(state: HostState, policy: WorkerPolicy = DEFAULT_POLICY) -> PriorityAdvice:
    """Pure priority lever: how the pool should yield cores by priority given the host state.

    Under Hawking the pool runs at the gentlest darwin QoS class and the highest nice level so
    the surviving workers cede cores to Hawking continuously, sharing the machine by PRIORITY
    instead of by dropping still more workers.
    """

    if state.hawking_active:
        return PriorityAdvice(
            nice_level=HAWKING_NICE,
            taskpolicy_class="background",
            yield_to_hawking=True,
            rationale=(
                "Hawking active: keep the reserve pool but run it at background QoS and maximum "
                "nice so it yields cores to Hawking by priority (continuous core-sharing)"
            ),
        )
    if (not state.thermal_ok) or (not state.on_ac):
        return PriorityAdvice(
            nice_level=PRESSURE_NICE,
            taskpolicy_class="background",
            yield_to_hawking=False,
            rationale="thermal or power pressure: run the pool gently at background QoS",
        )
    return PriorityAdvice(
        nice_level=IDLE_NICE,
        taskpolicy_class="utility",
        yield_to_hawking=False,
        rationale="idle host: run the pool at utility QoS with a mild positive nice below interactive work",
    )


@dataclass(frozen=True, slots=True)
class HostSample:
    """A live host state plus the derived recommendation, timestamped for the operator."""

    state: HostState
    priority: PriorityAdvice
    recommended_workers: int
    bounds: dict[str, Any]
    hawking_processes: list[dict[str, Any]]
    telemetry: dict[str, Any]
    created_at: float
    schema: str = SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "created_at": self.created_at,
            "recommended_workers": self.recommended_workers,
            "state": self.state.payload(),
            "priority": self.priority.payload(),
            "bounds": self.bounds,
            "hawking_processes": self.hawking_processes,
            "telemetry": self.telemetry,
            "ownership": "observation-only; no external process is ever signaled",
            "activation_allowed": False,
            "scientific_promotion": False,
        }


# ----------------------------------------------------------------------------------
# Live sampler (isolated; the only place wall clock and psutil are read)
# ----------------------------------------------------------------------------------


def _capture(command: list[str], timeout: float = 5.0) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return False, ""
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def _thermal_ok() -> bool:
    ok, output = _capture(["pmset", "-g", "therm"])
    if not ok:
        return True  # absent a portable probe, treat unknown as normal; throttle owns the hard gate
    if "No thermal warning level has been recorded" in output:
        return True
    limits = [int(value) for value in re.findall(r"(?:CPU|GPU)_Speed_Limit\s*=\s*([0-9]+)", output)]
    return not limits or min(limits) >= 100


def _on_ac() -> bool:
    ok, output = _capture(["pmset", "-g", "batt"])
    if not ok or not output:
        return True  # desktop hosts report nothing useful; default to AC
    return "AC Power" in output or "Now drawing from 'AC Power'" in output or "Battery" not in output


def detect_hawking(exclude_pids: set[int] | None = None) -> tuple[bool, list[dict[str, Any]]]:
    """Read-only Hawking detection by command marker. Never signals any process.

    Reuses the reviewed external_coexistence serial markers and adds the live v5 markers
    (block-parallel quantizer, doctor_v5 adapters). Our own PID is always excluded.
    """

    if psutil is None:  # pragma: no cover
        return False, []
    exclude = set(exclude_pids or set()) | {os.getpid()}
    hits: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = int(proc.info.get("pid") or -1)
            if pid < 0 or pid in exclude:
                continue
            name = str(proc.info.get("name") or "")
            cmdline = " ".join(str(part) for part in (proc.info.get("cmdline") or []))
            haystack = f"{name} {cmdline}"
            marker = next((m for m in HAWKING_COMMAND_MARKERS if m in haystack), None)
            if marker is not None:
                hits.append({"pid": pid, "marker": marker})
        except (psutil.Error, OSError):
            continue
    return len(hits) > 0, sorted(hits, key=lambda row: row["pid"])


def sample_host_state(
    now_fn: Callable[[], float] = time.time,
    *,
    policy: WorkerPolicy = DEFAULT_POLICY,
    current_workers: int = 0,
    exclude_pids: set[int] | None = None,
) -> HostSample:
    """Read one live host state and derive the advisory recommendation.

    ``now_fn`` is injected so the sample can be timestamped without the pure core ever
    reading a clock. All live reads (psutil, pmset, process scan) are confined here.
    """

    if psutil is None:  # pragma: no cover
        raise WorkerControllerRefused("psutil is required to read a live host state")
    _require_nonneg_int(current_workers, "current_workers")

    logical = int(psutil.cpu_count(logical=True) or LOGICAL_CORES)
    try:
        load1 = float(os.getloadavg()[0])
    except OSError:  # pragma: no cover
        load1 = 0.0
    virtual = psutil.virtual_memory()
    mem_available_gb = float(virtual.available) / 1e9
    busy = min(float(logical), load1)
    free_p_cores = max(0, int(math.floor(logical - busy)))
    hawking_active, hawking_processes = detect_hawking(exclude_pids=exclude_pids)
    thermal_ok = _thermal_ok()
    on_ac = _on_ac()

    state = HostState(
        free_p_cores=free_p_cores,
        hawking_active=hawking_active,
        mem_available_gb=mem_available_gb,
        thermal_ok=thermal_ok,
        on_ac=on_ac,
        current_load=load1,
        current_workers=current_workers,
    )
    priority = recommended_priority(state, policy)
    workers = recommended_workers(state, policy)
    telemetry = {
        "logical_cpus": logical,
        "performance_cores": PERFORMANCE_CORES,
        "load_1m": load1,
        "mem_available_gb": mem_available_gb,
        "mem_total_gb": float(virtual.total) / 1e9,
    }
    return HostSample(
        state=state,
        priority=priority,
        recommended_workers=workers,
        bounds=worker_bounds(state, policy),
        hawking_processes=hawking_processes,
        telemetry=telemetry,
        created_at=float(now_fn()),
    )


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------


def _cmd_report(args: argparse.Namespace) -> int:
    sample = sample_host_state(current_workers=args.current_workers)
    print(json.dumps(sample.payload(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mop.studio.dynamic_worker_controller",
        description="Advisory adaptive worker sizer for idle-host compute (receipt-invariant).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    report = subparsers.add_parser("report", help="print the current recommendation from live host state")
    report.add_argument(
        "--current-workers",
        type=int,
        default=0,
        help="pool size running right now (for the ramp-up-slow / back-off-fast hysteresis)",
    )
    report.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.func
    return handler(args)


__all__ = [
    "CLAIM_SCOPE",
    "CORE_RESERVE",
    "HAWKING_COMMAND_MARKERS",
    "HAWKING_RESERVE_WORKERS",
    "MEASURED_AGGREGATE_MHS",
    "PER_WORKER_GB",
    "SCHEMA",
    "WORKER_CEILING",
    "WORKER_FLOOR",
    "HostSample",
    "HostState",
    "PriorityAdvice",
    "WorkerControllerRefused",
    "WorkerPolicy",
    "detect_hawking",
    "recommended_priority",
    "recommended_workers",
    "sample_host_state",
    "worker_bounds",
]


if __name__ == "__main__":
    raise SystemExit(main())
