"""Phase 3 reliability: retry escalation, stall detection, notifications, and frozen resource profiles.

These are the systems whose absence let the stopped run loop a deterministic wall overrun forever in silence.
Retry escalation classifies each attempt and forces deferral -> failure_hold -> prohibited on repeated
identical deterministic stops. Stall detection distinguishes slow valid work from repeated zero progress.
Notifications fire on the mandated events with delivery isolated from scientific state. Resource profiles are
per task class, declared, and frozen, with one shared resource authority to prevent nested oversubscription.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Retry escalation
# ---------------------------------------------------------------------------

ATTEMPT_CLASSES = ("transient_resource_pressure", "deterministic_wall_overrun", "scientific_failure",
                   "integrity_failure", "operator_interruption", "unknown_failure")
DETERMINISTIC = {"deterministic_wall_overrun", "scientific_failure", "integrity_failure"}


@dataclass
class RetryLedger:
    """Append-only attempt history + escalation decision. Never retries a deterministic failure indefinitely."""

    attempts: list[dict] = field(default_factory=list)

    def record(self, attempt_class: str, signature: str, *, changed_authority: bool = False,
               operator_override: bool = False) -> dict:
        if attempt_class not in ATTEMPT_CLASSES:
            raise ValueError(f"unknown attempt class {attempt_class!r}")
        # count prior IDENTICAL deterministic stops (same class + signature)
        identical = sum(1 for a in self.attempts
                        if a["attempt_class"] == attempt_class and a["signature"] == signature)
        if attempt_class in DETERMINISTIC:
            if identical == 0:
                decision = "bounded_retry_or_resume"
            elif identical == 1:
                decision = "failure_hold_plus_notification"
            else:
                decision = ("permitted_only_with_changed_authority_or_operator_override"
                            if (changed_authority or operator_override) else "prohibited")
        elif attempt_class == "transient_resource_pressure":
            decision = "bounded_retry_or_resume"  # transient may retry (with backoff), not escalated
        elif attempt_class == "operator_interruption":
            decision = "resume_from_checkpoint"
        else:  # unknown
            decision = "failure_hold_plus_notification" if identical >= 1 else "bounded_retry_or_resume"
        entry = {"attempt_class": attempt_class, "signature": signature,
                 "identical_prior": identical, "decision": decision,
                 "changed_authority": changed_authority, "operator_override": operator_override}
        self.attempts.append(entry)  # append-only
        return entry

    def notify_required(self) -> bool:
        return bool(self.attempts) and "failure_hold" in self.attempts[-1]["decision"]


# ---------------------------------------------------------------------------
# Stall detection
# ---------------------------------------------------------------------------

@dataclass
class StallDetector:
    """Distinguishes slow valid work (a progress signal is advancing) from repeated zero-progress execution."""

    allowed_interval_seconds: float
    ready_capsule_count: int = 0

    def evaluate(self, *, now: float, last_capsule_finish: float, last_checkpoint_advance: float,
                 last_output_change: float, cpu_active: bool, concurrency: int,
                 repeated_same_wall_boundary: bool) -> dict:
        triggers = []
        if now - last_capsule_finish > self.allowed_interval_seconds:
            triggers.append("no_capsule_finished_within_interval")
        if now - last_checkpoint_advance > self.allowed_interval_seconds:
            triggers.append("no_checkpoint_advance")
        if now - last_output_change > self.allowed_interval_seconds:
            triggers.append("logs_and_outputs_unchanged")
        if cpu_active and (now - last_checkpoint_advance) > self.allowed_interval_seconds:
            triggers.append("cpu_active_but_no_durable_progress")
        if concurrency == 1 and self.ready_capsule_count > 1:
            triggers.append("concurrency_one_while_multiple_ready")
        if repeated_same_wall_boundary:
            triggers.append("same_wall_boundary_repeats")
        # slow-but-valid: recent durable progress (a checkpoint advanced) -> not a stall even if slow
        slow_valid = (now - last_checkpoint_advance) <= self.allowed_interval_seconds
        return {"stalled": bool(triggers) and not slow_valid, "triggers": triggers, "slow_but_valid": slow_valid}


# ---------------------------------------------------------------------------
# Notifications (delivery isolated from scientific state)
# ---------------------------------------------------------------------------

NOTIFY_EVENTS = ("campaign_launch", "stage_start", "stage_complete", "progress_10pct",
                 "concurrency_collapse", "wall_boundary_stop", "repeated_retry", "stall",
                 "failure_hold", "integrity_hold", "operator_stop", "terminal_result")


@dataclass
class Notifier:
    """Fires on the mandated events. A delivery failure is swallowed so it can never change scientific state."""

    sent: list[dict] = field(default_factory=list)
    delivery_failures: int = 0
    _transport_fails: bool = False

    def notify(self, event: str, detail: dict | None = None) -> bool:
        if event not in NOTIFY_EVENTS:
            raise ValueError(f"unknown notify event {event!r}")
        try:
            if self._transport_fails:
                raise RuntimeError("telegram transport down")
            self.sent.append({"event": event, "detail": detail or {}})
            return True
        except Exception:
            self.delivery_failures += 1  # isolated: never propagates to the caller / scientific state
            return False


# ---------------------------------------------------------------------------
# Resource profiles (per task class, declared, frozen; one shared resource authority)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceProfile:
    task_class: str
    max_concurrency: int
    internal_worker_width: int
    memory_budget_gb: float
    wall_boundary_seconds: int
    checkpoint_interval_seconds: int
    retry_policy: str
    priority: int


# frozen defaults; the scheduler and internal pools draw from ONE authority (total logical demand <= cores)
LOGICAL_CORES = 28
PROFILES: dict[str, ResourceProfile] = {
    "small_cpu": ResourceProfile("small_cpu", 20, 1, 1.0, 900, 60, "bounded_retry", 500),
    "vectorized_construction": ResourceProfile("vectorized_construction", 6, 4, 8.0, 5400, 300, "resume", 700),
    "large_cpu": ResourceProfile("large_cpu", 4, 4, 16.0, 7200, 300, "resume", 600),
    "memory_heavy": ResourceProfile("memory_heavy", 2, 8, 48.0, 7200, 300, "resume", 650),
    "verification": ResourceProfile("verification", 8, 1, 4.0, 1800, 30, "resume_incremental", 900),
    "aggregation": ResourceProfile("aggregation", 1, 4, 16.0, 3600, 120, "resume", 800),
    "report": ResourceProfile("report", 1, 1, 2.0, 600, 60, "bounded_retry", 400),
}


def shared_authority_ok(active: dict[str, int]) -> bool:
    """One shared authority: total declared concurrent logical demand must not exceed the core budget."""
    demand = sum(active.get(tc, 0) * PROFILES[tc].internal_worker_width for tc in active if tc in PROFILES)
    return demand <= LOGICAL_CORES
