"""Self-managing dynamic throttler for the chained ladder campaign.

The controller maximizes concurrent worker utilization (CPU and memory) while never crossing an
out-of-memory floor. It answers one question each tick: given a live host sample and how many of our
own workers are running, may we start one more worker, what is the current concurrency target, and how
many of our OWN newest workers (never anyone else's) must we shed to stay safe.

Design notes:
- The decision core (`DynamicThrottleController.decide`) is pure and deterministic given a `HostSample`.
  Telemetry reading (`read_host_sample`) is separated so the controller is fully testable with synthetic
  samples and no psutil dependency in the hot path.
- "Max": when memory and CPU headroom are comfortable the target grows by additive increase toward the
  smaller of the logical-CPU cap and the memory budget. "No OOM": admission requires that projected
  available memory after the new worker stays at or above the reserve floor; if the host is already below
  the floor the controller sheds its own newest workers; under swap pressure it backs off multiplicatively.
- The controller never signals, kills, or even names any process other than our own workers. Unmanaged
  heavy processes (for example an external quantize job) are already reflected in available memory and CPU
  load, so the same floor and ceiling logic keeps us out of their way without ever touching them.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

from ..substrate.events import canonical_sha256

try:  # psutil is a hard dependency in this repo; guard only so the pure core imports anywhere.
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

SCHEMA = "mop-dynamic-throttle/v1"
# Must stay byte-identical to experiments.expansion_harness.CLAIM_SCOPE. Duplicated here instead of
# imported so this module carries no capability-bearing import surface.
CLAIM_SCOPE = "deterministic programmatic mechanics only; no capability or natural-data claim"

GIB = 1024**3

# Known unmanaged-heavy command markers we account for in the budget but NEVER signal.
UNMANAGED_HEAVY_MARKERS = ("quantize-model", "strand-quant")


class ThrottleRefusal(ValueError):
    """Raised when a policy or sample is malformed. The controller fails closed."""


def _require_nonneg_int(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ThrottleRefusal(f"{label} must be a nonnegative integer")


def _require_pos_int(value: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ThrottleRefusal(f"{label} must be a positive integer")


def _require_fraction(value: float, label: str, *, upper: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ThrottleRefusal(f"{label} must be a number")
    if not (0.0 < float(value) <= upper):
        raise ThrottleRefusal(f"{label} must lie in (0, {upper}]")


@dataclass(frozen=True, slots=True)
class HostSample:
    """One immutable snapshot of the fields the controller reasons over."""

    total_memory_bytes: int
    available_memory_bytes: int
    swap_used_bytes: int
    logical_cpus: int
    load_1m_per_core: float
    cpu_utilization: float  # instantaneous fraction in [0, 1]
    thermal_normal: bool
    unmanaged_heavy_rss_bytes: int  # external heavy process RSS, for observability only
    own_workers_rss_bytes: int  # summed RSS of OUR running workers

    def __post_init__(self) -> None:
        _require_pos_int(self.total_memory_bytes, "total_memory_bytes")
        _require_nonneg_int(self.available_memory_bytes, "available_memory_bytes")
        _require_nonneg_int(self.swap_used_bytes, "swap_used_bytes")
        _require_pos_int(self.logical_cpus, "logical_cpus")
        _require_nonneg_int(self.unmanaged_heavy_rss_bytes, "unmanaged_heavy_rss_bytes")
        _require_nonneg_int(self.own_workers_rss_bytes, "own_workers_rss_bytes")
        if self.available_memory_bytes > self.total_memory_bytes:
            raise ThrottleRefusal("available_memory_bytes cannot exceed total_memory_bytes")
        if not isinstance(self.load_1m_per_core, (int, float)) or self.load_1m_per_core < 0:
            raise ThrottleRefusal("load_1m_per_core must be a nonnegative number")
        if not isinstance(self.cpu_utilization, (int, float)) or not (0.0 <= self.cpu_utilization <= 1.0):
            raise ThrottleRefusal("cpu_utilization must lie in [0, 1]")
        if not isinstance(self.thermal_normal, bool):
            raise ThrottleRefusal("thermal_normal must be a boolean")

    def payload(self) -> dict[str, Any]:
        return {
            "total_memory_bytes": self.total_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "logical_cpus": self.logical_cpus,
            "load_1m_per_core": float(self.load_1m_per_core),
            "cpu_utilization": float(self.cpu_utilization),
            "thermal_normal": self.thermal_normal,
            "unmanaged_heavy_rss_bytes": self.unmanaged_heavy_rss_bytes,
            "own_workers_rss_bytes": self.own_workers_rss_bytes,
        }


@dataclass(frozen=True, slots=True)
class ThrottlePolicy:
    """Safety knobs. All are hard: the controller never widens them at runtime."""

    oom_floor_bytes: int  # keep at least this much memory available at all times (hard floor)
    per_worker_peak_rss_bytes: int  # conservative initial per-worker peak; learned upward only
    max_workers: int  # absolute concurrency cap
    load_ceiling_per_core: float  # do not add work above this normalized one-minute load
    utilization_ceiling: float  # do not add work above this instantaneous CPU fraction
    swap_backoff_bytes: int  # swap use above this triggers multiplicative back off
    increase_step: int = 1  # additive increase per comfortable tick
    decrease_factor: float = 0.5  # multiplicative decrease under pressure
    min_workers: int = 1  # target floor
    headroom_factor: float = 1.15  # inflate the per-worker estimate when projecting, for margin
    schema: str = SCHEMA
    claim_scope: str = CLAIM_SCOPE

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ThrottleRefusal("unsupported dynamic-throttle schema")
        if self.claim_scope != CLAIM_SCOPE:
            raise ThrottleRefusal("dynamic-throttle claim scope cannot be widened")
        _require_pos_int(self.oom_floor_bytes, "oom_floor_bytes")
        _require_pos_int(self.per_worker_peak_rss_bytes, "per_worker_peak_rss_bytes")
        _require_pos_int(self.max_workers, "max_workers")
        _require_pos_int(self.min_workers, "min_workers")
        _require_pos_int(self.increase_step, "increase_step")
        _require_nonneg_int(self.swap_backoff_bytes, "swap_backoff_bytes")
        _require_fraction(self.load_ceiling_per_core, "load_ceiling_per_core", upper=8.0)
        _require_fraction(self.utilization_ceiling, "utilization_ceiling", upper=1.0)
        _require_fraction(self.decrease_factor, "decrease_factor", upper=0.999)
        if self.headroom_factor < 1.0:
            raise ThrottleRefusal("headroom_factor must be at least 1.0 (never underestimate a worker)")
        if self.min_workers > self.max_workers:
            raise ThrottleRefusal("min_workers cannot exceed max_workers")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "claim_scope": self.claim_scope,
            "oom_floor_bytes": self.oom_floor_bytes,
            "per_worker_peak_rss_bytes": self.per_worker_peak_rss_bytes,
            "max_workers": self.max_workers,
            "min_workers": self.min_workers,
            "load_ceiling_per_core": self.load_ceiling_per_core,
            "utilization_ceiling": self.utilization_ceiling,
            "swap_backoff_bytes": self.swap_backoff_bytes,
            "increase_step": self.increase_step,
            "decrease_factor": self.decrease_factor,
            "headroom_factor": self.headroom_factor,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())

    @classmethod
    def autoscaled(
        cls,
        sample: HostSample,
        *,
        per_worker_peak_gb: float = 4.0,
        oom_floor_gb: float | None = None,
        max_workers: int | None = None,
    ) -> ThrottlePolicy:
        """Build a host-sized policy: an 8 GiB or 12 percent floor, one worker per logical CPU cap."""

        floor = (
            int(oom_floor_gb * GIB)
            if oom_floor_gb is not None
            else max(8 * GIB, int(sample.total_memory_bytes * 0.12))
        )
        cap = max_workers if max_workers is not None else max(1, sample.logical_cpus)
        return cls(
            oom_floor_bytes=floor,
            per_worker_peak_rss_bytes=max(1, int(per_worker_peak_gb * GIB)),
            max_workers=cap,
            load_ceiling_per_core=0.90,
            utilization_ceiling=0.92,
            swap_backoff_bytes=256 * 1024 * 1024,
        )


@dataclass(frozen=True, slots=True)
class ThrottleDecision:
    """The pure output of one tick."""

    admit: bool
    target_workers: int
    must_shed: int
    reason: str
    projected_available_after_admit_bytes: int
    memory_capped_workers: int
    learned_per_worker_rss_bytes: int
    schema: str = SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "admit": self.admit,
            "target_workers": self.target_workers,
            "must_shed": self.must_shed,
            "reason": self.reason,
            "projected_available_after_admit_bytes": self.projected_available_after_admit_bytes,
            "memory_capped_workers": self.memory_capped_workers,
            "learned_per_worker_rss_bytes": self.learned_per_worker_rss_bytes,
        }

    def digest(self) -> str:
        return canonical_sha256(self.payload())


@dataclass
class DynamicThrottleController:
    """Stateful AIMD controller. One instance per campaign; call `decide` each tick."""

    policy: ThrottlePolicy
    learned_per_worker_rss_bytes: int = field(init=False)
    target: int = field(init=False)

    def __post_init__(self) -> None:
        self.learned_per_worker_rss_bytes = self.policy.per_worker_peak_rss_bytes
        self.target = self.policy.min_workers

    def observe(self, sample: HostSample, running: int) -> None:
        """Learn the per-worker peak RSS conservatively (monotone non-decreasing)."""

        _require_nonneg_int(running, "running")
        if running > 0 and sample.own_workers_rss_bytes > 0:
            measured = sample.own_workers_rss_bytes / running
            ewma = 0.7 * self.learned_per_worker_rss_bytes + 0.3 * measured
            self.learned_per_worker_rss_bytes = int(max(self.learned_per_worker_rss_bytes, ewma, measured))

    def _per_worker_projection(self) -> int:
        return max(1, int(self.learned_per_worker_rss_bytes * self.policy.headroom_factor))

    def decide(self, sample: HostSample, running: int) -> ThrottleDecision:
        """Pure given (learned state, sample, running). Fails closed on unsafe host conditions."""

        _require_nonneg_int(running, "running")
        self.observe(sample, running)
        per = self._per_worker_projection()
        free = sample.available_memory_bytes
        floor = self.policy.oom_floor_bytes
        hard_cap = min(self.policy.max_workers, sample.logical_cpus)
        # Model every worker at its learned peak rather than trusting live free to have caught up with
        # ramp-up. mem_room credits back our workers' current RSS (we recount them at `per`), so this is
        # an absolute cap on TOTAL concurrent workers that holds even if none have ramped yet.
        mem_room = free + sample.own_workers_rss_bytes - floor
        memory_capped = min(hard_cap, max(0, mem_room) // per)

        # 1. Critical: already below the OOM floor. Shed our own newest workers to recover.
        if free < floor:
            deficit = floor - free
            shed = min(running, max(1, math.ceil(deficit / per))) if running > 0 else 0
            self.target = max(self.policy.min_workers, running - shed)
            return ThrottleDecision(
                admit=False,
                target_workers=self.target,
                must_shed=shed,
                reason="below oom floor: shedding own newest workers",
                projected_available_after_admit_bytes=free,
                memory_capped_workers=memory_capped,
                learned_per_worker_rss_bytes=self.learned_per_worker_rss_bytes,
            )

        # 2. Swap pressure: over-committed. Back off multiplicatively and drain toward the lower target.
        if sample.swap_used_bytes > self.policy.swap_backoff_bytes:
            self.target = max(self.policy.min_workers, int(running * self.policy.decrease_factor))
            shed = max(0, running - self.target)
            return ThrottleDecision(
                admit=False,
                target_workers=self.target,
                must_shed=shed,
                reason="swap pressure: multiplicative back off",
                projected_available_after_admit_bytes=free,
                memory_capped_workers=memory_capped,
                learned_per_worker_rss_bytes=self.learned_per_worker_rss_bytes,
            )

        # 3. CPU or thermal ceiling: hold. This is not an OOM risk, so do not shed, just stop growing.
        if (
            sample.load_1m_per_core > self.policy.load_ceiling_per_core
            or sample.cpu_utilization > self.policy.utilization_ceiling
            or not sample.thermal_normal
        ):
            self.target = max(
                self.policy.min_workers, min(self.target, max(running, self.policy.min_workers))
            )
            return ThrottleDecision(
                admit=False,
                target_workers=self.target,
                must_shed=0,
                reason="cpu or thermal ceiling: holding concurrency",
                projected_available_after_admit_bytes=free - per,
                memory_capped_workers=memory_capped,
                learned_per_worker_rss_bytes=self.learned_per_worker_rss_bytes,
            )

        # 4. Comfortable: additive increase toward the smaller of the CPU cap and the memory cap.
        self.target = min(hard_cap, memory_capped, self.target + self.policy.increase_step)
        self.target = max(self.policy.min_workers, self.target)
        projected_after = free - per
        admit = running < self.target and running < memory_capped and projected_after >= floor
        return ThrottleDecision(
            admit=admit,
            target_workers=self.target,
            must_shed=max(0, running - self.target),
            reason="headroom: scaling toward cap" if admit else "at target or cap: holding",
            projected_available_after_admit_bytes=projected_after,
            memory_capped_workers=memory_capped,
            learned_per_worker_rss_bytes=self.learned_per_worker_rss_bytes,
        )


def _pid_rss(pid: int) -> int:
    if psutil is None:
        return 0
    try:
        return int(psutil.Process(pid).memory_info().rss)
    except Exception:
        return 0


def own_workers_rss_bytes(pids: list[int]) -> int:
    """Sum RSS of OUR worker PIDs. Missing or dead PIDs contribute zero."""

    return sum(_pid_rss(pid) for pid in pids)


def unmanaged_heavy_rss_bytes(exclude_pids: set[int] | None = None) -> int:
    """Sum RSS of external heavy processes we account for but never signal (observability only)."""

    if psutil is None:
        return 0
    exclude = exclude_pids or set()
    own = os.getpid()
    total = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            pid = int(proc.info.get("pid") or -1)
            if pid == own or pid in exclude:
                continue
            name = str(proc.info.get("name") or "")
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if any(marker in name or marker in cmdline for marker in UNMANAGED_HEAVY_MARKERS):
                info = proc.info.get("memory_info")
                if info is not None:
                    total += int(info.rss)
        except Exception:
            continue
    return total


def _thermal_normal() -> bool:
    # Absent a portable probe, treat unknown as normal; the memory and load guards are the hard limits.
    return True


def read_host_sample(worker_pids: list[int] | None = None) -> HostSample:
    """Read a live host sample via psutil. Kept thin and out of the tested decision path."""

    if psutil is None:  # pragma: no cover
        raise ThrottleRefusal("psutil is required to read a live host sample")
    worker_pids = worker_pids or []
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    logical = int(psutil.cpu_count(logical=True) or 1)
    try:
        load1 = os.getloadavg()[0]
    except OSError:  # pragma: no cover
        load1 = 0.0
    util = float(psutil.cpu_percent(interval=0.05) / 100.0)
    own_set = set(worker_pids)
    return HostSample(
        total_memory_bytes=int(virtual.total),
        available_memory_bytes=int(virtual.available),
        swap_used_bytes=int(swap.used),
        logical_cpus=logical,
        load_1m_per_core=load1 / logical if logical else load1,
        cpu_utilization=min(1.0, max(0.0, util)),
        thermal_normal=_thermal_normal(),
        unmanaged_heavy_rss_bytes=unmanaged_heavy_rss_bytes(exclude_pids=own_set),
        own_workers_rss_bytes=own_workers_rss_bytes(worker_pids),
    )
