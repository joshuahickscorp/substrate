"""Pure, activation-disabled causal/resource monitoring over immutable ESCS snapshots.

This adapter is deliberately downstream of the event and lifecycle ledgers.  It validates bounded
snapshots, derives only provenance-bearing resource anomalies and same-parent simulated resource
contrasts, and returns immutable claim messages plus an uncommitted accounting observation.  It has
no event-admission, dispatch, commitment, mutation, effect, retry, relief, or trigger authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self

from mop.substrate.events import BranchRef, canonical_bytes, canonical_sha256

from .accounting import FACTUAL_BRANCH, LifecycleCharge, LifecycleLedger, WorkVector
from .events import EpistemicStatus, ESCSEvent, EvidenceClass, HypothesisEvent
from .ledger import EventLedger
from .messages import ClaimMessage, ClaimSchema

MONITOR_CONFIG_SCHEMA = "mop-escs-causal-resource-monitor-config/v1"
MONITOR_RESULT_SCHEMA = "mop-escs-causal-resource-monitor-result/v1"
MONITOR_CLAIM_PAYLOAD_SCHEMA = "mop-escs-causal-resource-monitor-claim/v1"
MONITOR_STATE_SCHEMA = "mop-escs-causal-resource-monitor-state/v1"
MONITOR_ID = "monitor:escs-causal-resource/v1"
CLAIM_SCOPE = (
    "bounded immutable-ledger mechanics only; anomaly and same-parent simulated resource contrasts; "
    "no causal-effect, failure, capability, efficiency, energy, activation, or scientific-promotion claim"
)

ACTIVATION_ENABLED = False
SCIENTIFIC_PROMOTION_ALLOWED = False
MAX_HARD_EVENT_ENTRIES = 512
MAX_HARD_LIFECYCLE_ENTRIES = 512
MAX_HARD_SNAPSHOT_BYTES = 1024 * 1024
MAX_HARD_WINDOW_TICKS = 4096
MAX_HARD_WINDOWS = 64
MAX_HARD_CLAIMS = 32
MAX_HARD_PROVENANCE_IDS = 32
MAX_HARD_PAIR_COMPARISONS = 512
MAX_HARD_WORK_UNITS = 32_768

MONITOR_CLAIM_SCHEMA = ClaimSchema(
    schema_id="mop.escs.causal-resource-monitor",
    version=1,
    claim_types=frozenset({"resource_anomaly", "same_parent_resource_contrast"}),
    payload_forms=frozenset({"escs-monitor-canonical-json"}),
    epistemic_statuses=frozenset({EpistemicStatus.INFERRED, EpistemicStatus.SIMULATED}),
    max_payload_bytes=16 * 1024,
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEYS = frozenset(
    {
        "canonical_referent_truth",
        "evaluator",
        "evaluator_label",
        "evaluator_truth",
        "future_outcome",
        "ground_truth",
        "ground_truth_label",
        "hidden_change_point",
        "hidden_shock",
        "oracle_label",
        "physical_failure_label",
    }
)
_FORBIDDEN_KEY_PREFIXES = (
    "evaluator",
    "future_",
    "ground_truth",
    "hidden_",
    "oracle_",
)
_AUTHORITY_FIELDS = (
    "activation",
    "dispatch",
    "commitment",
    "mutation",
    "effect",
    "retry",
    "resource_relief",
    "independent_trigger",
)


class MonitorContractError(ValueError):
    """The monitor input or requested control violates the bounded pure-monitor contract."""


class MonitorControl(StrEnum):
    """Exact deterministic controls; none grants runtime authority."""

    CLEAN = "clean"
    NOISY = "noisy"
    POISON = "poison"
    STALE = "stale"
    SHUFFLE = "shuffle"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MonitorContractError(message)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MonitorContractError(f"{label} must be a nonnegative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise MonitorContractError(f"{label} must be positive")
    return result


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise MonitorContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise MonitorContractError(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        raise MonitorContractError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _walk_keys(value: Any) -> set[str]:
    result: set[str] = set()
    stack = [value]
    while stack:
        nested = stack.pop()
        if isinstance(nested, Mapping):
            result.update(str(key) for key in nested)
            stack.extend(nested.values())
        elif isinstance(nested, (list, tuple)):
            stack.extend(nested)
    return result


def _reject_forbidden(value: Any, label: str) -> None:
    keys = _walk_keys(value)
    found = sorted(
        key
        for key in keys
        if key.casefold() in _FORBIDDEN_KEYS
        or key.casefold().startswith(_FORBIDDEN_KEY_PREFIXES)
    )
    if found:
        raise MonitorContractError(f"{label} contains evaluator/future-only fields: {found}")


def _authority_payload() -> dict[str, bool]:
    return {field: False for field in _AUTHORITY_FIELDS}


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    """Finite monitor authority; every field participates in ``sha256``."""

    window_ticks: int = 8
    max_windows: int = 4
    max_event_entries: int = 128
    max_lifecycle_entries: int = 128
    max_snapshot_bytes: int = MAX_HARD_SNAPSHOT_BYTES
    max_claims: int = 8
    max_provenance_ids: int = 16
    max_pair_comparisons: int = 128
    max_work_units: int = 4096
    work_anomaly_threshold: int = 12
    retained_byte_time_threshold: int = 128
    same_parent_work_delta: int = 4
    noise_ceiling: int = 2
    poison_increment: int = 16
    stale_ticks: int = 8
    activation_enabled: bool = ACTIVATION_ENABLED
    scientific_promotion_allowed: bool = SCIENTIFIC_PROMOTION_ALLOWED

    def __post_init__(self) -> None:
        for name in (
            "window_ticks",
            "max_windows",
            "max_event_entries",
            "max_lifecycle_entries",
            "max_snapshot_bytes",
            "max_claims",
            "max_provenance_ids",
            "max_pair_comparisons",
            "max_work_units",
            "work_anomaly_threshold",
            "retained_byte_time_threshold",
            "same_parent_work_delta",
            "poison_increment",
            "stale_ticks",
            "noise_ceiling",
        ):
            _positive_int(getattr(self, name), f"MonitorConfig.{name}")
        _require(self.max_event_entries <= MAX_HARD_EVENT_ENTRIES, "event-entry cap exceeds hard bound")
        _require(
            self.max_lifecycle_entries <= MAX_HARD_LIFECYCLE_ENTRIES,
            "lifecycle-entry cap exceeds hard bound",
        )
        _require(self.max_snapshot_bytes <= MAX_HARD_SNAPSHOT_BYTES, "snapshot byte cap exceeds hard bound")
        _require(self.window_ticks <= MAX_HARD_WINDOW_TICKS, "window tick cap exceeds hard bound")
        _require(self.max_windows <= MAX_HARD_WINDOWS, "window-count cap exceeds hard bound")
        _require(
            self.stale_ticks <= self.window_ticks * self.max_windows,
            "stale control must remain inside the bounded analysis horizon",
        )
        _require(self.max_claims <= MAX_HARD_CLAIMS, "claim cap exceeds hard bound")
        _require(self.max_provenance_ids <= MAX_HARD_PROVENANCE_IDS, "provenance cap exceeds hard bound")
        _require(
            self.max_pair_comparisons <= MAX_HARD_PAIR_COMPARISONS,
            "pair-comparison cap exceeds hard bound",
        )
        _require(self.max_work_units <= MAX_HARD_WORK_UNITS, "work cap exceeds hard bound")
        _require(self.activation_enabled is False, "causal/resource monitor activation must remain disabled")
        _require(
            self.scientific_promotion_allowed is False,
            "causal/resource monitor cannot grant scientific promotion",
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": MONITOR_CONFIG_SCHEMA,
            "claim_scope": CLAIM_SCOPE,
            "window_ticks": self.window_ticks,
            "max_windows": self.max_windows,
            "max_event_entries": self.max_event_entries,
            "max_lifecycle_entries": self.max_lifecycle_entries,
            "max_snapshot_bytes": self.max_snapshot_bytes,
            "max_claims": self.max_claims,
            "max_provenance_ids": self.max_provenance_ids,
            "max_pair_comparisons": self.max_pair_comparisons,
            "max_work_units": self.max_work_units,
            "work_anomaly_threshold": self.work_anomaly_threshold,
            "retained_byte_time_threshold": self.retained_byte_time_threshold,
            "same_parent_work_delta": self.same_parent_work_delta,
            "noise_ceiling": self.noise_ceiling,
            "poison_increment": self.poison_increment,
            "stale_ticks": self.stale_ticks,
            "authority": _authority_payload(),
            "activation_enabled": self.activation_enabled,
            "scientific_promotion_allowed": self.scientific_promotion_allowed,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class ControlTraceRow:
    """One exact mapping from an immutable charge to a controlled monitor sample."""

    target_sequence: int
    target_charge_sha256: str
    source_charge_sha256: str
    branch_id: str
    end_tick: int
    raw_total_work: int
    effective_total_work: int
    raw_retained_byte_time: int
    effective_retained_byte_time: int
    included: bool
    transformation: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.target_sequence, "target_sequence")
        _digest(self.target_charge_sha256, "target_charge_sha256")
        _digest(self.source_charge_sha256, "source_charge_sha256")
        BranchRef(self.branch_id)
        _nonnegative_int(self.end_tick, "end_tick")
        _nonnegative_int(self.raw_total_work, "raw_total_work")
        _nonnegative_int(self.effective_total_work, "effective_total_work")
        _nonnegative_int(self.raw_retained_byte_time, "raw_retained_byte_time")
        _nonnegative_int(self.effective_retained_byte_time, "effective_retained_byte_time")
        _require(isinstance(self.included, bool), "included must be a boolean")
        _require(bool(self.transformation), "control transformation must be named")

    def payload(self) -> dict[str, Any]:
        return {
            "target_sequence": self.target_sequence,
            "target_charge_sha256": self.target_charge_sha256,
            "source_charge_sha256": self.source_charge_sha256,
            "branch_id": self.branch_id,
            "end_tick": self.end_tick,
            "raw_total_work": self.raw_total_work,
            "effective_total_work": self.effective_total_work,
            "raw_retained_byte_time": self.raw_retained_byte_time,
            "effective_retained_byte_time": self.effective_retained_byte_time,
            "included": self.included,
            "transformation": self.transformation,
        }


def _monitor_result_payload(
    *,
    control: MonitorControl,
    config_sha256: str,
    event_snapshot_sha256: str,
    lifecycle_snapshot_sha256: str,
    observed_through_tick: int,
    analysis_window: tuple[int, int],
    event_entries_seen: int,
    lifecycle_entries_seen: int,
    pair_comparisons: int,
    control_trace: Sequence[ControlTraceRow],
    claims: Sequence[ClaimMessage],
    abstentions: Sequence[str],
    monitor_work: WorkVector,
    activation_enabled: bool,
    scientific_promotion_allowed: bool,
    result_sha256: str | None,
) -> dict[str, Any]:
    trace_payload = [row.payload() for row in control_trace]
    result: dict[str, Any] = {
        "schema": MONITOR_RESULT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "control": control.value,
        "config_sha256": config_sha256,
        "snapshot_authority": {
            "event_ledger_sha256": event_snapshot_sha256,
            "lifecycle_ledger_sha256": lifecycle_snapshot_sha256,
            "observed_through_tick": observed_through_tick,
        },
        "analysis_window": list(analysis_window),
        "bounds_observed": {
            "event_entries_seen": event_entries_seen,
            "lifecycle_entries_seen": lifecycle_entries_seen,
            "pair_comparisons": pair_comparisons,
            "control_rows": len(control_trace),
            "claims": len(claims),
            "monitor_work_units": monitor_work.indexing_and_graph_maintenance,
        },
        "control_trace": trace_payload,
        "control_trace_sha256": canonical_sha256(trace_payload),
        "claims": [message.wire_payload() for message in claims],
        "claim_ids": [message.header.message_id for message in claims],
        "abstentions": list(abstentions),
        "monitor_work": monitor_work.payload(),
        "monitor_work_charge_applied": False,
        "payload_semantics_read": False,
        "authority": _authority_payload(),
        "activation_enabled": activation_enabled,
        "scientific_promotion_allowed": scientific_promotion_allowed,
        "energy_measured": False,
        "official_run": False,
    }
    if result_sha256 is not None:
        result["result_sha256"] = result_sha256
    return result


@dataclass(frozen=True, slots=True)
class MonitorResult:
    """Self-sealed monitor result.  Claims are inert messages, never dispatch requests."""

    control: MonitorControl
    config_sha256: str
    event_snapshot_sha256: str
    lifecycle_snapshot_sha256: str
    observed_through_tick: int
    analysis_window: tuple[int, int]
    event_entries_seen: int
    lifecycle_entries_seen: int
    pair_comparisons: int
    control_trace: tuple[ControlTraceRow, ...]
    claims: tuple[ClaimMessage, ...]
    abstentions: tuple[str, ...]
    monitor_work: WorkVector
    activation_enabled: bool
    scientific_promotion_allowed: bool
    result_sha256: str

    def __post_init__(self) -> None:
        _require(isinstance(self.control, MonitorControl), "result control must be typed")
        for digest_value, label in (
            (self.config_sha256, "config_sha256"),
            (self.event_snapshot_sha256, "event_snapshot_sha256"),
            (self.lifecycle_snapshot_sha256, "lifecycle_snapshot_sha256"),
            (self.result_sha256, "result_sha256"),
        ):
            _digest(digest_value, label)
        _nonnegative_int(self.observed_through_tick, "observed_through_tick")
        _require(
            len(self.analysis_window) == 2
            and 0 <= self.analysis_window[0] <= self.analysis_window[1] <= self.observed_through_tick,
            "analysis window is invalid",
        )
        for count, label in (
            (self.event_entries_seen, "event_entries_seen"),
            (self.lifecycle_entries_seen, "lifecycle_entries_seen"),
            (self.pair_comparisons, "pair_comparisons"),
        ):
            _nonnegative_int(count, label)
        _require(isinstance(self.control_trace, tuple), "control trace must be immutable")
        _require(isinstance(self.claims, tuple), "claims must be immutable")
        _require(isinstance(self.abstentions, tuple), "abstentions must be immutable")
        _require(all(message.integrity_valid() for message in self.claims), "monitor claim integrity drift")
        _require(isinstance(self.monitor_work, WorkVector), "monitor work must be a WorkVector")
        _require(
            self.monitor_work.total_work == self.monitor_work.indexing_and_graph_maintenance
            and self.monitor_work.retained_byte_time == 0,
            "monitor work escaped the observation-only accounting bucket",
        )
        _require(self.event_entries_seen <= MAX_HARD_EVENT_ENTRIES, "result event count exceeds hard bound")
        _require(
            self.lifecycle_entries_seen <= MAX_HARD_LIFECYCLE_ENTRIES,
            "result lifecycle count exceeds hard bound",
        )
        _require(self.pair_comparisons <= MAX_HARD_PAIR_COMPARISONS, "result pair count exceeds hard bound")
        _require(len(self.claims) <= MAX_HARD_CLAIMS, "result claim count exceeds hard bound")
        _require(
            self.monitor_work.total_work <= MAX_HARD_WORK_UNITS,
            "result monitor work exceeds hard bound",
        )
        _require(self.activation_enabled is False, "monitor result escaped activation-disabled state")
        _require(
            self.scientific_promotion_allowed is False,
            "monitor result escaped mechanics-only claim boundary",
        )
        trace_sha = canonical_sha256([row.payload() for row in self.control_trace])
        expected_state_version = canonical_sha256(
            {
                "schema": MONITOR_STATE_SCHEMA,
                "config_sha256": self.config_sha256,
                "control": self.control.value,
                "event_snapshot_sha256": self.event_snapshot_sha256,
                "lifecycle_snapshot_sha256": self.lifecycle_snapshot_sha256,
                "control_trace_sha256": trace_sha,
            }
        )
        expected_claim_fields = {
            "schema",
            "claim_type",
            "claim_scope",
            "control",
            "control_contaminated",
            "config_sha256",
            "snapshot_authority",
            "analysis_window",
            "authority",
            "activation_enabled",
            "scientific_promotion_allowed",
            "payload_semantics_read",
            "causal_interpretation",
            "metrics",
            "provenance",
        }
        for message in self.claims:
            header = message.header
            _require(
                header.claim_schema_id == MONITOR_CLAIM_SCHEMA.schema_id
                and header.claim_schema_version == MONITOR_CLAIM_SCHEMA.version
                and header.claim_schema_digest == MONITOR_CLAIM_SCHEMA.digest,
                "result contains a foreign claim schema",
            )
            _require(
                header.claim_type in MONITOR_CLAIM_SCHEMA.claim_types,
                "result claim type escaped monitor",
            )
            _require(header.payload_form == "escs-monitor-canonical-json", "result claim payload form drift")
            _require(header.producer_actor_id == MONITOR_ID, "result claim producer drift")
            _require(header.producer_state_version == expected_state_version, "result claim state drift")
            _require(
                header.created_tick == self.observed_through_tick,
                "result claim was not created at the snapshot frontier",
            )
            _require(header.expiry_tick == header.created_tick, "result claim gained trigger lifetime")
            _require(header.calibrated_confidence == 0.0, "result claim gained unearned calibration")
            _require(header.predicted_utility == (), "result claim gained action utility")
            try:
                claim_payload = json.loads(message.payload_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MonitorContractError("result claim payload is not canonical JSON") from exc
            _exact_keys(claim_payload, expected_claim_fields, "monitor claim payload")
            _require(
                canonical_bytes(claim_payload) == message.payload_bytes,
                "result claim payload is not canonically encoded",
            )
            _require(
                claim_payload["schema"] == MONITOR_CLAIM_PAYLOAD_SCHEMA
                and claim_payload["claim_type"] == header.claim_type
                and claim_payload["claim_scope"] == CLAIM_SCOPE,
                "result claim identity fields drifted",
            )
            _require(claim_payload["control"] == self.control.value, "result claim control drift")
            _require(
                claim_payload["control_contaminated"] is (self.control is not MonitorControl.CLEAN),
                "result claim control contamination drifted",
            )
            _require(claim_payload["config_sha256"] == self.config_sha256, "result claim config drift")
            _require(
                claim_payload["snapshot_authority"]
                == {
                    "event_ledger_sha256": self.event_snapshot_sha256,
                    "lifecycle_ledger_sha256": self.lifecycle_snapshot_sha256,
                    "control_trace_sha256": trace_sha,
                },
                "result claim snapshot authority drifted",
            )
            _require(
                claim_payload["analysis_window"] == list(self.analysis_window),
                "result claim window drifted",
            )
            _require(claim_payload["authority"] == _authority_payload(), "result claim gained authority")
            _require(claim_payload["activation_enabled"] is False, "result claim gained activation")
            _require(
                claim_payload["scientific_promotion_allowed"] is False,
                "result claim gained scientific promotion",
            )
            _require(claim_payload["payload_semantics_read"] is False, "result claim read payload semantics")
        _require(
            canonical_sha256(self.payload(include_digest=False)) == self.result_sha256,
            "monitor result self-hash mismatch",
        )

    @classmethod
    def create(
        cls,
        *,
        control: MonitorControl,
        config_sha256: str,
        event_snapshot_sha256: str,
        lifecycle_snapshot_sha256: str,
        observed_through_tick: int,
        analysis_window: tuple[int, int],
        event_entries_seen: int,
        lifecycle_entries_seen: int,
        pair_comparisons: int,
        control_trace: tuple[ControlTraceRow, ...],
        claims: tuple[ClaimMessage, ...],
        abstentions: tuple[str, ...],
        monitor_work: WorkVector,
    ) -> Self:
        core = _monitor_result_payload(
            control=control,
            config_sha256=config_sha256,
            event_snapshot_sha256=event_snapshot_sha256,
            lifecycle_snapshot_sha256=lifecycle_snapshot_sha256,
            observed_through_tick=observed_through_tick,
            analysis_window=analysis_window,
            event_entries_seen=event_entries_seen,
            lifecycle_entries_seen=lifecycle_entries_seen,
            pair_comparisons=pair_comparisons,
            control_trace=control_trace,
            claims=claims,
            abstentions=abstentions,
            monitor_work=monitor_work,
            activation_enabled=False,
            scientific_promotion_allowed=False,
            result_sha256=None,
        )
        return cls(
            control=control,
            config_sha256=config_sha256,
            event_snapshot_sha256=event_snapshot_sha256,
            lifecycle_snapshot_sha256=lifecycle_snapshot_sha256,
            observed_through_tick=observed_through_tick,
            analysis_window=analysis_window,
            event_entries_seen=event_entries_seen,
            lifecycle_entries_seen=lifecycle_entries_seen,
            pair_comparisons=pair_comparisons,
            control_trace=control_trace,
            claims=claims,
            abstentions=abstentions,
            monitor_work=monitor_work,
            activation_enabled=False,
            scientific_promotion_allowed=False,
            result_sha256=canonical_sha256(core),
        )

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        return _monitor_result_payload(
            control=self.control,
            config_sha256=self.config_sha256,
            event_snapshot_sha256=self.event_snapshot_sha256,
            lifecycle_snapshot_sha256=self.lifecycle_snapshot_sha256,
            observed_through_tick=self.observed_through_tick,
            analysis_window=self.analysis_window,
            event_entries_seen=self.event_entries_seen,
            lifecycle_entries_seen=self.lifecycle_entries_seen,
            pair_comparisons=self.pair_comparisons,
            control_trace=self.control_trace,
            claims=self.claims,
            abstentions=self.abstentions,
            monitor_work=self.monitor_work,
            activation_enabled=self.activation_enabled,
            scientific_promotion_allowed=self.scientific_promotion_allowed,
            result_sha256=self.result_sha256 if include_digest else None,
        )


def _bounded_snapshot(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_keys: set[str],
    max_entries: int,
    max_bytes: int,
) -> tuple[dict[str, Any], int, str]:
    if not isinstance(payload, Mapping):
        raise MonitorContractError(f"{label} must be a mapping")
    try:
        encoded = canonical_bytes(payload)
    except (RecursionError, TypeError, ValueError) as exc:
        raise MonitorContractError(f"{label} is not strict canonical JSON") from exc
    if len(encoded) > max_bytes:
        raise MonitorContractError(f"{label} exceeds its byte cap")
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise MonitorContractError(f"{label} must encode a JSON object")
    _exact_keys(decoded, expected_keys, label)
    _reject_forbidden(decoded, label)
    entries = decoded.get("entries")
    if not isinstance(entries, list):
        raise MonitorContractError(f"{label}.entries must be a list")
    if len(entries) > max_entries:
        raise MonitorContractError(f"{label} exceeds its entry cap")
    return decoded, len(entries), hashlib.sha256(encoded).hexdigest()


def _parse_snapshots(
    event_snapshot: Mapping[str, Any],
    lifecycle_snapshot: Mapping[str, Any],
    *,
    observed_through_tick: int,
    config: MonitorConfig,
) -> tuple[EventLedger, LifecycleLedger, str, str]:
    event_copy, event_count, event_sha = _bounded_snapshot(
        event_snapshot,
        label="event snapshot",
        expected_keys={"schema", "entries", "head_sha256"},
        max_entries=config.max_event_entries,
        max_bytes=config.max_snapshot_bytes,
    )
    lifecycle_copy, lifecycle_count, lifecycle_sha = _bounded_snapshot(
        lifecycle_snapshot,
        label="lifecycle snapshot",
        expected_keys={"schema", "entries", "head_sha256", "total"},
        max_entries=config.max_lifecycle_entries,
        max_bytes=config.max_snapshot_bytes,
    )
    if event_count + 2 * lifecycle_count > config.max_work_units:
        raise MonitorContractError("snapshot replay exceeds the monitor work-unit cap")
    try:
        event_ledger = EventLedger.from_payload(event_copy)
        lifecycle_ledger = LifecycleLedger.from_payload(lifecycle_copy)
    except (TypeError, ValueError) as exc:
        raise MonitorContractError(f"snapshot replay failed: {exc}") from exc
    problems = lifecycle_ledger.verify(event_ids=set(event_ledger.event_ids))
    if problems:
        raise MonitorContractError("lifecycle/event provenance join failed: " + "; ".join(problems))
    for event in event_ledger.events:
        if event.envelope.clock_end_tick > observed_through_tick:
            raise MonitorContractError("event snapshot contains future-dated state")
    events_by_id = _event_map(event_ledger.events)
    for charge in lifecycle_ledger.entries:
        if charge.end_tick > observed_through_tick:
            raise MonitorContractError("lifecycle snapshot contains future-dated state")
        for event_id in charge.causal_event_ids:
            event = events_by_id[str(event_id)]
            if charge.start_tick < event.envelope.clock_end_tick:
                raise MonitorContractError("lifecycle charge predates its causal event")
            allowed_branches = {str(charge.branch_id)}
            if charge.branch_id != FACTUAL_BRANCH:
                allowed_branches.add(str(FACTUAL_BRANCH))
            if str(event.branch_id) not in allowed_branches:
                raise MonitorContractError("lifecycle charge crosses an unauthorized event branch")
    return event_ledger, lifecycle_ledger, event_sha, lifecycle_sha


def _analysis_window(
    observed_through_tick: int, control: MonitorControl, config: MonitorConfig
) -> tuple[int, int]:
    end_tick = observed_through_tick
    if control is MonitorControl.STALE:
        end_tick = max(0, observed_through_tick - config.stale_ticks)
    horizon = config.window_ticks * config.max_windows
    return max(0, end_tick - horizon), end_tick


def _control_trace(
    charges: Sequence[LifecycleCharge],
    *,
    control: MonitorControl,
    analysis_window: tuple[int, int],
    config: MonitorConfig,
) -> tuple[ControlTraceRow, ...]:
    start_tick, end_tick = analysis_window
    included_indices = [
        index for index, charge in enumerate(charges) if start_tick <= charge.end_tick <= end_tick
    ]
    included_set = set(included_indices)
    if control in {MonitorControl.POISON, MonitorControl.SHUFFLE} and not included_indices:
        raise MonitorContractError(f"{control.value} control requires at least one in-window charge")
    if control is MonitorControl.SHUFFLE and len(included_indices) < 2:
        raise MonitorContractError("shuffle control requires at least two in-window charges")
    poisoned_index = (
        min(included_indices, key=lambda index: charges[index].charge_sha256)
        if control is MonitorControl.POISON
        else None
    )
    shuffled_sources: dict[int, int] = {}
    if control is MonitorControl.SHUFFLE:
        shuffled_sources = {
            target: included_indices[(position + 1) % len(included_indices)]
            for position, target in enumerate(included_indices)
        }

    rows: list[ControlTraceRow] = []
    for index, charge in enumerate(charges):
        included = index in included_set
        source = charge
        effective = charge.work.total_work
        effective_retained = charge.work.retained_byte_time
        if not included:
            transformation = "outside-bounded-window"
            if control is MonitorControl.STALE and charge.end_tick > end_tick:
                transformation = "stale:excluded-current"
        elif control is MonitorControl.NOISY:
            noise = 1 + int(charge.charge_sha256[:8], 16) % config.noise_ceiling
            effective += noise
            transformation = f"noisy:+{noise}"
        elif control is MonitorControl.POISON and index == poisoned_index:
            effective += config.poison_increment
            transformation = f"poison:+{config.poison_increment}"
        elif control is MonitorControl.SHUFFLE:
            source = charges[shuffled_sources[index]]
            effective = source.work.total_work
            effective_retained = source.work.retained_byte_time
            transformation = "shuffle:rotate-left-1"
        else:
            transformation = f"{control.value}:identity"
        rows.append(
            ControlTraceRow(
                target_sequence=charge.sequence,
                target_charge_sha256=charge.charge_sha256,
                source_charge_sha256=source.charge_sha256,
                branch_id=str(charge.branch_id),
                end_tick=charge.end_tick,
                raw_total_work=charge.work.total_work,
                effective_total_work=effective,
                raw_retained_byte_time=charge.work.retained_byte_time,
                effective_retained_byte_time=effective_retained,
                included=included,
                transformation=transformation,
            )
        )
    return tuple(rows)


def _event_map(events: Sequence[ESCSEvent]) -> dict[str, ESCSEvent]:
    return {str(event.event_id): event for event in events}


def _pair_contract(event: HypothesisEvent) -> tuple[str, str, str] | None:
    provenance = event.envelope.source_and_provenance.value()
    if not isinstance(provenance, dict):
        raise MonitorContractError("hypothesis provenance must remain a mapping")
    contract = provenance.get("causal_resource_monitor_pair")
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise MonitorContractError("causal-resource pair contract must be a mapping")
    _exact_keys(contract, {"pair_id", "arm", "contract_sha256"}, "causal-resource pair contract")
    pair_id = contract["pair_id"]
    arm = contract["arm"]
    contract_sha256 = contract["contract_sha256"]
    _require(isinstance(pair_id, str) and bool(pair_id.strip()), "pair id must be nonempty text")
    _require(arm in {"observational", "resource_control"}, "pair arm is unsupported")
    _digest(contract_sha256, "pair contract sha256")
    expected = canonical_sha256(
        {
            "schema": "mop-escs-causal-resource-pair/v1",
            "pair_id": pair_id,
            "arms": ["observational", "resource_control"],
        }
    )
    _require(contract_sha256 == expected, "causal-resource pair contract self-hash mismatch")
    return pair_id, arm, contract_sha256


def _claim_evidence_class(event_ids: Sequence[str], events: Mapping[str, ESCSEvent]) -> EvidenceClass:
    rows = [events[event_id].evidence_class for event_id in event_ids]
    return max(rows, key=lambda value: value.taint_rank)


def _create_claim(
    *,
    claim_type: str,
    branch_id: str,
    source_hypothesis_ids: Sequence[str],
    supporting_event_ids: Sequence[str],
    created_tick: int,
    payload: Mapping[str, Any],
    events: Mapping[str, ESCSEvent],
    producer_state_version: str,
    producer_operations: int,
) -> ClaimMessage:
    payload_bytes = canonical_bytes(payload)
    if len(payload_bytes) > MONITOR_CLAIM_SCHEMA.max_payload_bytes:
        raise MonitorContractError("monitor claim payload exceeds its byte cap")
    evidence_ids = tuple(sorted(set(source_hypothesis_ids) | set(supporting_event_ids)))
    epistemic_status = (
        EpistemicStatus.INFERRED if branch_id == str(FACTUAL_BRANCH) else EpistemicStatus.SIMULATED
    )
    return ClaimMessage.create(
        schema=MONITOR_CLAIM_SCHEMA,
        source_hypothesis_event_ids=source_hypothesis_ids,
        referent_hypotheses=("resource:escs-lifecycle",),
        branch_id=branch_id,
        factor_scope=("resource:abstract-work", "resource:retained-byte-time"),
        claim_type=claim_type,
        epistemic_status=epistemic_status,
        supporting_event_ids=supporting_event_ids,
        producer_actor_id=MONITOR_ID,
        producer_state_version=producer_state_version,
        calibrated_confidence=0.0,
        created_tick=created_tick,
        expiry_tick=created_tick,
        predicted_utility=(),
        producer_operations=producer_operations,
        payload_form="escs-monitor-canonical-json",
        payload_bytes=payload_bytes,
        evidence_class=_claim_evidence_class(evidence_ids, events),
    )


def _claim_payload_base(
    *,
    claim_type: str,
    control: MonitorControl,
    config: MonitorConfig,
    event_snapshot_sha256: str,
    lifecycle_snapshot_sha256: str,
    control_trace_sha256: str,
    analysis_window: tuple[int, int],
) -> dict[str, Any]:
    return {
        "schema": MONITOR_CLAIM_PAYLOAD_SCHEMA,
        "claim_type": claim_type,
        "claim_scope": CLAIM_SCOPE,
        "control": control.value,
        "control_contaminated": control is not MonitorControl.CLEAN,
        "config_sha256": config.sha256,
        "snapshot_authority": {
            "event_ledger_sha256": event_snapshot_sha256,
            "lifecycle_ledger_sha256": lifecycle_snapshot_sha256,
            "control_trace_sha256": control_trace_sha256,
        },
        "analysis_window": list(analysis_window),
        "authority": _authority_payload(),
        "activation_enabled": False,
        "scientific_promotion_allowed": False,
        "payload_semantics_read": False,
    }


def analyze_snapshots(
    event_snapshot: Mapping[str, Any],
    lifecycle_snapshot: Mapping[str, Any],
    *,
    observed_through_tick: int,
    control: MonitorControl = MonitorControl.CLEAN,
    config: MonitorConfig = MonitorConfig(),
) -> MonitorResult:
    """Validate and monitor two immutable snapshots without retaining or mutating state."""

    _nonnegative_int(observed_through_tick, "observed_through_tick")
    if not isinstance(control, MonitorControl):
        raise MonitorContractError("control must be a MonitorControl")
    if not isinstance(config, MonitorConfig):
        raise MonitorContractError("config must be a MonitorConfig")
    event_ledger, lifecycle_ledger, event_sha, lifecycle_sha = _parse_snapshots(
        event_snapshot,
        lifecycle_snapshot,
        observed_through_tick=observed_through_tick,
        config=config,
    )
    window = _analysis_window(observed_through_tick, control, config)
    trace = _control_trace(
        lifecycle_ledger.entries,
        control=control,
        analysis_window=window,
        config=config,
    )
    trace_sha = canonical_sha256([row.payload() for row in trace])
    producer_state_version = canonical_sha256(
        {
            "schema": MONITOR_STATE_SCHEMA,
            "config_sha256": config.sha256,
            "control": control.value,
            "event_snapshot_sha256": event_sha,
            "lifecycle_snapshot_sha256": lifecycle_sha,
            "control_trace_sha256": trace_sha,
        }
    )
    events = _event_map(event_ledger.events)
    charges = {charge.sequence: charge for charge in lifecycle_ledger.entries}
    claims: list[ClaimMessage] = []
    abstentions: set[str] = set()
    work_units = len(event_ledger.entries) + len(lifecycle_ledger.entries) + len(trace)

    for row in trace:
        if not row.included:
            continue
        work_units += 1
        if (
            row.effective_total_work < config.work_anomaly_threshold
            and row.effective_retained_byte_time < config.retained_byte_time_threshold
        ):
            continue
        charge = charges[row.target_sequence]
        provenance_ids = tuple(sorted(str(event_id) for event_id in charge.causal_event_ids))
        if len(provenance_ids) > config.max_provenance_ids:
            abstentions.add("resource-anomaly-provenance-over-cap")
            continue
        source_ids = tuple(
            event_id
            for event_id in provenance_ids
            if isinstance(events[event_id], HypothesisEvent)
            and str(events[event_id].branch_id) == row.branch_id
        )
        if not source_ids:
            abstentions.add("resource-anomaly-without-same-branch-hypothesis")
            continue
        if len(claims) >= config.max_claims:
            abstentions.add("claim-cap-reached")
            continue
        payload = _claim_payload_base(
            claim_type="resource_anomaly",
            control=control,
            config=config,
            event_snapshot_sha256=event_sha,
            lifecycle_snapshot_sha256=lifecycle_sha,
            control_trace_sha256=trace_sha,
            analysis_window=window,
        )
        payload.update(
            {
                "causal_interpretation": "threshold anomaly only; causal effect not established",
                "metrics": {
                    "raw_total_work": row.raw_total_work,
                    "effective_total_work": row.effective_total_work,
                    "raw_retained_byte_time": row.raw_retained_byte_time,
                    "effective_retained_byte_time": row.effective_retained_byte_time,
                    "work_anomaly_threshold": config.work_anomaly_threshold,
                    "retained_byte_time_threshold": config.retained_byte_time_threshold,
                },
                "provenance": {
                    "target_charge_sequence": row.target_sequence,
                    "target_charge_sha256": row.target_charge_sha256,
                    "source_charge_sha256": row.source_charge_sha256,
                    "event_ids": list(provenance_ids),
                },
            }
        )
        claims.append(
            _create_claim(
                claim_type="resource_anomaly",
                branch_id=row.branch_id,
                source_hypothesis_ids=source_ids,
                supporting_event_ids=provenance_ids,
                created_tick=observed_through_tick,
                payload=payload,
                events=events,
                producer_state_version=producer_state_version,
                producer_operations=1 + len(provenance_ids),
            )
        )

    direct_work: dict[str, int] = {}
    for row in trace:
        if not row.included:
            continue
        charge = charges[row.target_sequence]
        if len(charge.causal_event_ids) == 1:
            event_id = str(charge.causal_event_ids[0])
            if (
                isinstance(events[event_id], HypothesisEvent)
                and str(events[event_id].branch_id) == row.branch_id
            ):
                direct_work[event_id] = direct_work.get(event_id, 0) + row.effective_total_work

    hypotheses = [
        event
        for event in event_ledger.events
        if isinstance(event, HypothesisEvent)
        and window[0] <= event.envelope.clock_end_tick <= window[1]
        and str(event.event_id) in direct_work
    ]
    groups: dict[tuple[tuple[str, ...], str, str], list[tuple[HypothesisEvent, str]]] = {}
    for event in hypotheses:
        parents = tuple(str(event_id) for event_id in event.envelope.causal_parent_ids)
        contract = _pair_contract(event)
        if parents and contract is not None:
            pair_id, arm, contract_sha256 = contract
            groups.setdefault((parents, pair_id, contract_sha256), []).append((event, arm))

    pair_comparisons = 0
    for group_key in sorted(groups):
        parent_ids, _, _ = group_key
        rows = groups[group_key]
        factual = sorted(
            (event for event, arm in rows if event.branch_id == FACTUAL_BRANCH and arm == "observational"),
            key=lambda event: str(event.event_id),
        )
        simulated = sorted(
            (
                event
                for event, arm in rows
                if event.branch_id != FACTUAL_BRANCH and arm == "resource_control"
            ),
            key=lambda event: str(event.event_id),
        )
        for factual_event in factual:
            for simulated_event in simulated:
                pair_comparisons += 1
                work_units += 1
                if pair_comparisons > config.max_pair_comparisons:
                    raise MonitorContractError("same-parent comparison cap exceeded")
                factual_id = str(factual_event.event_id)
                simulated_id = str(simulated_event.event_id)
                factual_work = direct_work[factual_id]
                simulated_work = direct_work[simulated_id]
                delta = simulated_work - factual_work
                if abs(delta) < config.same_parent_work_delta:
                    continue
                provenance_ids = tuple(sorted({*parent_ids, factual_id, simulated_id}))
                if len(provenance_ids) > config.max_provenance_ids:
                    abstentions.add("causal-contrast-provenance-over-cap")
                    continue
                if len(claims) >= config.max_claims:
                    abstentions.add("claim-cap-reached")
                    continue
                payload = _claim_payload_base(
                    claim_type="same_parent_resource_contrast",
                    control=control,
                    config=config,
                    event_snapshot_sha256=event_sha,
                    lifecycle_snapshot_sha256=lifecycle_sha,
                    control_trace_sha256=trace_sha,
                    analysis_window=window,
                )
                payload.update(
                    {
                        "causal_interpretation": (
                            "same-parent factual/simulated resource contrast; counterfactual candidate only, "
                            "not a realized causal effect"
                        ),
                        "metrics": {
                            "factual_total_work": factual_work,
                            "simulated_total_work": simulated_work,
                            "signed_simulated_minus_factual_work": delta,
                            "minimum_absolute_delta": config.same_parent_work_delta,
                        },
                        "provenance": {
                            "same_parent_event_ids": list(parent_ids),
                            "factual_hypothesis_event_id": factual_id,
                            "simulated_hypothesis_event_id": simulated_id,
                        },
                    }
                )
                claims.append(
                    _create_claim(
                        claim_type="same_parent_resource_contrast",
                        branch_id=str(simulated_event.branch_id),
                        source_hypothesis_ids=(factual_id, simulated_id),
                        supporting_event_ids=provenance_ids,
                        created_tick=observed_through_tick,
                        payload=payload,
                        events=events,
                        producer_state_version=producer_state_version,
                        producer_operations=1 + len(provenance_ids),
                    )
                )

    work_units += len(claims)
    if work_units > config.max_work_units:
        raise MonitorContractError("monitor work-unit cap exceeded")
    return MonitorResult.create(
        control=control,
        config_sha256=config.sha256,
        event_snapshot_sha256=event_sha,
        lifecycle_snapshot_sha256=lifecycle_sha,
        observed_through_tick=observed_through_tick,
        analysis_window=window,
        event_entries_seen=event_ledger.entry_count,
        lifecycle_entries_seen=lifecycle_ledger.entry_count,
        pair_comparisons=pair_comparisons,
        control_trace=trace,
        claims=tuple(claims),
        abstentions=tuple(sorted(abstentions)),
        monitor_work=WorkVector(indexing_and_graph_maintenance=work_units),
    )


__all__ = [
    "ACTIVATION_ENABLED",
    "CLAIM_SCOPE",
    "MONITOR_CLAIM_SCHEMA",
    "MONITOR_CONFIG_SCHEMA",
    "MONITOR_RESULT_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "ControlTraceRow",
    "MonitorConfig",
    "MonitorContractError",
    "MonitorControl",
    "MonitorResult",
    "analyze_snapshots",
]
