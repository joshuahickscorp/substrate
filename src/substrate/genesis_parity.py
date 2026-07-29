"""Measured equal-opportunity audit and capability-density frontier for genesis.

The previous campaign's equal-opportunity check built one dict for the candidate
and one for S2 from the same source and compared them, so it passed by
construction. This module measures what each arm actually consumed and spent,
and it can fail.

Resident memory for the frontier is taken from ``resource.getrusage``
(``ru_maxrss``), not from serialized payload length. On macOS ``ru_maxrss`` is
already in bytes; on Linux it is kilobytes and is converted by multiplying by
1024. The process-wide peak cannot isolate arms inside one process; that limit
is labelled in the output.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from substrate import genesis_config as C
from substrate.genesis_material import CognitiveMaterial, MaterialBase, Opportunity, ResourceLedger

SCHEMA_PARITY = "substrate-genesis-parity-audit/v1"
SCHEMA_FRONTIER = "substrate-genesis-capability-density/v1"


def measure_peak_resident_bytes() -> dict[str, Any]:
    """Read process peak RSS from ``ru_maxrss`` and convert to bytes.

    On macOS, ``resource.RUSAGE_SELF.ru_maxrss`` is already reported in bytes.
    On Linux it is reported in kilobytes; this function multiplies by 1024.
    """
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        peak_bytes = raw
        platform_unit = "bytes"
        conversion = "macos_ru_maxrss_already_bytes"
        conversion_note = "On macOS, resource.RUSAGE_SELF.ru_maxrss is already in bytes; no scale factor applied."
    else:
        peak_bytes = raw * 1024
        platform_unit = "kilobytes"
        conversion = "linux_ru_maxrss_kilobytes_times_1024"
        conversion_note = "On Linux, resource.RUSAGE_SELF.ru_maxrss is in kilobytes; converted to bytes by multiplying by 1024."
    return {
        "peak_resident_bytes": peak_bytes,
        "ru_maxrss_raw": raw,
        "ru_maxrss_platform_unit": platform_unit,
        "conversion": conversion,
        "conversion_note": conversion_note,
        "platform": sys.platform,
        "source": "resource.getrusage(resource.RUSAGE_SELF).ru_maxrss",
        "process_wide_peak": True,
        "per_arm_isolation": False,
        "activation": False,
    }


def _arm_name(arm: Any) -> str:
    if isinstance(arm, Mapping):
        for key in ("name", "arm", "id"):
            if key in arm and arm[key] is not None:
                return str(arm[key])
        return "unnamed"
    name = getattr(arm, "name", None)
    return str(name) if name is not None else "unnamed"


def _arm_deprived(arm: Any) -> tuple[str, ...]:
    if isinstance(arm, Mapping):
        if "deprived" in arm:
            return tuple(arm["deprived"])
        name = _arm_name(arm)
        return tuple(C.BASELINE_DEPRIVATION.get(name, ()))
    if hasattr(arm, "opportunity"):
        opportunity = arm.opportunity()
        if getattr(opportunity, "deprived", ()):
            return tuple(opportunity.deprived)
    name = getattr(arm, "name", "")
    return tuple(C.BASELINE_DEPRIVATION.get(name, ()))


def _mapping_channel(arm: Mapping[str, Any], channel: str) -> Any:
    if "measurements" in arm and isinstance(arm["measurements"], Mapping) and channel in arm["measurements"]:
        return arm["measurements"][channel]
    if channel in arm:
        return arm[channel]
    aliases = {
        "compute": ("operations", "operation_count", "transition_operations"),
        "plasticity": ("durable_writes", "admitted_durable_writes", "durable_write_budget"),
        "persistence": ("checkpoint_restore_count", "checkpoints_plus_restores"),
        "memory": ("peak_resident_bytes", "resident_bytes"),
        "information": ("observation_digest", "ordered_observation_digest"),
        "sensors": ("sensor_channels", "channel_set"),
        "teaching": ("teaching_digest", "teaching_event_digest"),
    }
    for alias in aliases.get(channel, ()):
        if alias in arm:
            return arm[alias]
        measurements = arm.get("measurements")
        if isinstance(measurements, Mapping) and alias in measurements:
            return measurements[alias]
    raise KeyError(f"arm {_arm_name(arm)!r} has no measurement for channel {channel!r}")


def _material_channel(arm: CognitiveMaterial | MaterialBase, channel: str) -> Any:
    opportunity: Opportunity = arm.opportunity()
    ledger: ResourceLedger = opportunity.ledger
    measured = arm.cost() if hasattr(arm, "cost") else ledger.measurement()
    if channel == "information":
        return opportunity.observation_digest
    if channel == "sensors":
        return tuple(opportunity.sensor_channels)
    if channel == "teaching":
        return opportunity.teaching_digest
    if channel == "compute":
        return int(measured.get("compute", ledger.operations))
    if channel == "plasticity":
        # Admitted durable writes actually committed under the shared budget.
        return int(measured.get("plasticity", ledger.durable_writes))
    if channel == "persistence":
        return int(measured.get("persistence", ledger.checkpoints + ledger.restores))
    if channel == "memory":
        return int(measured.get("memory", ledger.peak_resident_bytes))
    raise KeyError(f"unknown parity channel {channel!r}")


def extract_measurements(arm: Any) -> dict[str, Any]:
    """Pull the seven parity-channel values from a material or a plain record."""
    if isinstance(arm, Mapping):
        return {channel: _mapping_channel(arm, channel) for channel in C.PARITY_CHANNELS}
    if isinstance(arm, (MaterialBase, CognitiveMaterial)) or all(
        hasattr(arm, attr) for attr in ("opportunity", "cost", "name")
    ):
        return {channel: _material_channel(arm, channel) for channel in C.PARITY_CHANNELS}
    raise TypeError(f"unsupported arm type for parity audit: {type(arm)!r}")


def _normalize_value(channel: str, value: Any) -> Any:
    if channel == "sensors":
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return value
    if channel in ("compute", "plasticity", "persistence", "memory"):
        return int(value)
    return value


def _exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (bytes, bytearray)) or isinstance(right, (bytes, bytearray)):
        return bytes(left) == bytes(right)
    if isinstance(left, tuple) and isinstance(right, tuple):
        return left == right
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return tuple(left) == tuple(right)
    return left == right


def _within_relative_tolerance(values: Sequence[float | int], tolerance: float) -> bool:
    if not values:
        return True
    numeric = [float(value) for value in values]
    lo = min(numeric)
    hi = max(numeric)
    if hi == 0.0 and lo == 0.0:
        return True
    scale = max(abs(hi), abs(lo), 1.0)
    return (hi - lo) / scale <= tolerance


def parity_audit(arms: Sequence[Any], unit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Measure equality across ``PARITY_CHANNELS`` for arms that ran the same unit.

    An arm deprived of a channel (via ``opportunity.deprived`` or
    ``BASELINE_DEPRIVATION``) is recorded as ``exempt`` for that channel only,
    not as equal. Byte-exact channels must match exactly among non-exempt arms.
    Counted channels must agree within ``PARITY_RELATIVE_TOLERANCE``.
    """
    if len(arms) < 2:
        raise ValueError("parity_audit requires at least two arms")

    names = [_arm_name(arm) for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"parity_audit requires unique arm names, got {names}")

    deprived_by_arm = {name: _arm_deprived(arm) for name, arm in zip(names, arms, strict=True)}
    measured: dict[str, dict[str, Any]] = {}
    for name, arm in zip(names, arms, strict=True):
        raw = extract_measurements(arm)
        measured[name] = {channel: _normalize_value(channel, raw[channel]) for channel in C.PARITY_CHANNELS}

    channels: dict[str, Any] = {}
    channel_pass: dict[str, bool] = {}

    for channel in C.PARITY_CHANNELS:
        measurement_name = C.PARITY_MEASUREMENTS[channel]
        values = {name: measured[name][channel] for name in names}
        statuses: dict[str, str] = {}
        exempt_arms: list[str] = []
        compared: list[str] = []

        for name in names:
            if channel in deprived_by_arm[name]:
                statuses[name] = "exempt"
                exempt_arms.append(name)
            else:
                statuses[name] = "compared"
                compared.append(name)

        if not compared:
            passed = True
            detail = "all_arms_exempt"
            reference = None
        elif channel in C.PARITY_EXACT_CHANNELS:
            reference = values[compared[0]]
            mismatches = [name for name in compared if not _exact_equal(values[name], reference)]
            passed = not mismatches
            detail = "byte_identical" if passed else "byte_mismatch"
            for name in compared:
                statuses[name] = "equal" if name not in mismatches else "unequal"
        else:
            compared_values = [values[name] for name in compared]
            passed = _within_relative_tolerance(compared_values, C.PARITY_RELATIVE_TOLERANCE)
            detail = "within_relative_tolerance" if passed else "relative_tolerance_exceeded"
            reference = {
                "min": min(float(value) for value in compared_values),
                "max": max(float(value) for value in compared_values),
                "tolerance": C.PARITY_RELATIVE_TOLERANCE,
            }
            for name in compared:
                statuses[name] = "equal" if passed else "unequal"

        channels[channel] = {
            "measurement": measurement_name,
            "exact": channel in C.PARITY_EXACT_CHANNELS,
            "values": values,
            "status": statuses,
            "exempt_arms": exempt_arms,
            "compared_arms": compared,
            "reference": reference,
            "detail": detail,
            "pass": passed,
        }
        channel_pass[channel] = passed

    return {
        "schema": SCHEMA_PARITY,
        "unit": dict(unit) if unit is not None else None,
        "arms": names,
        "deprived": {name: list(deprived_by_arm[name]) for name in names},
        "measured": measured,
        "channels": channels,
        "channel_pass": channel_pass,
        "all_pass": all(channel_pass.values()),
        "parity_relative_tolerance": C.PARITY_RELATIVE_TOLERANCE,
        "parity_exact_channels": list(C.PARITY_EXACT_CHANNELS),
        "activation": False,
    }


def _as_envelope_map(value: Any, envelopes: Sequence[str], field_name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        missing = [envelope for envelope in envelopes if envelope not in value]
        if missing:
            # Allow a single scalar-like mapping that is not envelope-keyed by broadcasting a "default".
            if "default" in value:
                return {envelope: value.get(envelope, value["default"]) for envelope in envelopes}
            raise KeyError(f"{field_name} missing envelopes {missing}")
        return {envelope: value[envelope] for envelope in envelopes}
    return {envelope: value for envelope in envelopes}


def _pareto_points(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Non-dominated points under minimize resident_bytes, maximize capability."""
    frontier: list[dict[str, Any]] = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                float(other["resident_bytes"]) <= float(row["resident_bytes"])
                and float(other["capability"]) >= float(row["capability"])
                and (
                    float(other["resident_bytes"]) < float(row["resident_bytes"])
                    or float(other["capability"]) > float(row["capability"])
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(
                {
                    "arm": row["arm"],
                    "envelope": row["envelope"],
                    "resident_bytes": row["resident_bytes"],
                    "capability": row["capability"],
                }
            )
    return frontier


def capability_density_frontier(
    arms: Sequence[Mapping[str, Any]],
    *,
    envelopes: Sequence[str] | None = None,
    episode_count: int = 1,
    run_live: bool = False,
) -> dict[str, Any]:
    """Report capability versus measured size for every arm and envelope.

    Each arm is a mapping with at least ``name`` and either:

    - per-envelope fields ``capability`` / ``resident_bytes`` (and optional cost
      fields), or
    - a ``runner`` callable ``(envelope: str) -> Mapping`` returning those fields
      after a real run.

    Peak resident bytes for live runs come from ``measure_peak_resident_bytes``
    (``ru_maxrss``). Serialized payload length is never used as resident size.
    Energy is an operation-count proxy and is labelled as such. Size is a cost
    variable for the Pareto frontier, not a ranking objective.
    """
    if not arms:
        raise ValueError("capability_density_frontier requires at least one arm")
    env_list = list(envelopes) if envelopes is not None else list(C.MEMORY_ENVELOPES)
    if not env_list:
        raise ValueError("no memory envelopes to evaluate")

    process_rss = measure_peak_resident_bytes()
    rows: list[dict[str, Any]] = []

    for arm in arms:
        if not isinstance(arm, Mapping):
            raise TypeError("capability_density_frontier arms must be mappings")
        name = _arm_name(arm)
        runner: Callable[[str], Mapping[str, Any]] | None = arm.get("runner")
        provided_capability = arm.get("capability")
        provided_resident = arm.get("resident_bytes")
        provided_checkpoint = arm.get("checkpoint_bytes", 0)
        provided_disk = arm.get("disk_bytes", 0)
        provided_operations = arm.get("operations", arm.get("operation_count", 0))
        provided_wall = arm.get("wall_clock_seconds_per_episode", arm.get("wall_clock_seconds", None))
        provided_learning = arm.get("learning", arm.get("absolute_learning", None))

        capability_map = (
            _as_envelope_map(provided_capability, env_list, "capability") if provided_capability is not None else None
        )
        resident_map = (
            _as_envelope_map(provided_resident, env_list, "resident_bytes") if provided_resident is not None else None
        )
        checkpoint_map = _as_envelope_map(provided_checkpoint, env_list, "checkpoint_bytes")
        disk_map = _as_envelope_map(provided_disk, env_list, "disk_bytes")
        operations_map = _as_envelope_map(provided_operations, env_list, "operations")
        wall_map = (
            _as_envelope_map(provided_wall, env_list, "wall_clock_seconds_per_episode")
            if provided_wall is not None
            else None
        )
        learning_map = (
            _as_envelope_map(provided_learning, env_list, "learning") if provided_learning is not None else None
        )

        for envelope in env_list:
            live_metrics: dict[str, Any] = {}
            wall_clock: float | None = None
            rss_after: dict[str, Any] | None = None
            resident_source = "arm_provided"

            if runner is not None and (run_live or capability_map is None or resident_map is None):
                started = time.perf_counter()
                live_metrics = dict(runner(envelope))
                elapsed = time.perf_counter() - started
                rss_after = measure_peak_resident_bytes()
                episodes = int(live_metrics.get("episodes", episode_count)) or 1
                wall_clock = float(live_metrics.get("wall_clock_seconds_per_episode", elapsed / episodes))
                if "resident_bytes" in live_metrics:
                    resident_bytes = int(live_metrics["resident_bytes"])
                    resident_source = str(live_metrics.get("resident_bytes_source", "runner_provided"))
                else:
                    resident_bytes = int(rss_after["peak_resident_bytes"])
                    resident_source = "process_ru_maxrss"
                capability = float(live_metrics.get("capability", capability_map[envelope] if capability_map else 0.0))
                checkpoint_bytes = int(live_metrics.get("checkpoint_bytes", checkpoint_map[envelope]))
                disk_bytes = int(live_metrics.get("disk_bytes", disk_map[envelope]))
                operations = int(live_metrics.get("operations", operations_map[envelope]))
                learning = live_metrics.get("learning", learning_map[envelope] if learning_map else capability)
            else:
                if capability_map is None or resident_map is None:
                    raise KeyError(
                        f"arm {name!r} needs capability and resident_bytes for envelope {envelope!r}, or a runner"
                    )
                capability = float(capability_map[envelope])
                resident_bytes = int(resident_map[envelope])
                checkpoint_bytes = int(checkpoint_map[envelope])
                disk_bytes = int(disk_map[envelope])
                operations = int(operations_map[envelope])
                learning = float(learning_map[envelope]) if learning_map is not None else capability
                wall_clock = float(wall_map[envelope]) if wall_map is not None else None
                resident_source = str(arm.get("resident_bytes_source", "arm_provided"))

            envelope_limit = C.ENVELOPE_BYTES.get(envelope)
            fits = envelope_limit is None or resident_bytes <= envelope_limit
            row = {
                "arm": name,
                "envelope": envelope,
                "capability": capability if fits else 0.0,
                "capability_raw": capability,
                "fits_envelope": fits,
                "envelope_limit_bytes": envelope_limit,
                "resident_bytes": resident_bytes,
                "resident_bytes_source": resident_source,
                "resident_bytes_measurement": (
                    "resource.getrusage(RUSAGE_SELF).ru_maxrss"
                    if resident_source == "process_ru_maxrss"
                    else resident_source
                ),
                "checkpoint_bytes": checkpoint_bytes,
                "disk_bytes": disk_bytes,
                "wall_clock_seconds_per_episode": wall_clock,
                "wall_clock_is_environment_dependent": True,
                "operation_count_energy_proxy": operations,
                "energy_is_proxy": True,
                "energy_proxy_label": "operation_count",
                "learning": float(learning),
                "process_peak_resident_bytes": (rss_after or process_rss)["peak_resident_bytes"],
                "process_rss_conversion": (rss_after or process_rss)["conversion"],
                "process_rss_note": (
                    "ru_maxrss is a process-wide peak; it does not isolate per-arm residency inside one process."
                ),
                "full_architecture_residency_isolated": resident_source == "process_ru_maxrss" and runner is not None,
                "hardware_energy_measured": False,
                "activation": False,
            }
            if live_metrics:
                row["runner_metrics"] = {
                    key: value
                    for key, value in live_metrics.items()
                    if key
                    not in {
                        "capability",
                        "resident_bytes",
                        "checkpoint_bytes",
                        "disk_bytes",
                        "operations",
                        "learning",
                    }
                }
            rows.append(row)

    # Learning per added byte relative to the smallest fitting resident size in the same envelope.
    for envelope in env_list:
        envelope_rows = [row for row in rows if row["envelope"] == envelope and row["fits_envelope"]]
        if not envelope_rows:
            for row in rows:
                if row["envelope"] == envelope:
                    row["learning_per_added_byte"] = 0.0
                    row["added_bytes_vs_envelope_min"] = None
            continue
        min_resident = min(int(row["resident_bytes"]) for row in envelope_rows)
        for row in rows:
            if row["envelope"] != envelope:
                continue
            if not row["fits_envelope"]:
                row["learning_per_added_byte"] = 0.0
                row["added_bytes_vs_envelope_min"] = None
                continue
            added = max(int(row["resident_bytes"]) - min_resident, 0)
            row["added_bytes_vs_envelope_min"] = added
            if added == 0:
                row["learning_per_added_byte"] = None
                row["learning_per_added_byte_note"] = "baseline_size_in_envelope_no_added_bytes"
            else:
                row["learning_per_added_byte"] = float(row["learning"]) / float(added)

    per_envelope_frontier: dict[str, list[dict[str, Any]]] = {}
    arms_on_frontier: set[str] = set()
    for envelope in env_list:
        candidates = [row for row in rows if row["envelope"] == envelope and row["fits_envelope"]]
        frontier = _pareto_points(candidates)
        per_envelope_frontier[envelope] = frontier
        for point in frontier:
            arms_on_frontier.add(str(point["arm"]))

    global_candidates = [row for row in rows if row["fits_envelope"]]
    global_frontier = _pareto_points(global_candidates)
    for point in global_frontier:
        arms_on_frontier.add(str(point["arm"]))

    return {
        "schema": SCHEMA_FRONTIER,
        "envelopes": env_list,
        "envelope_bytes": {envelope: C.ENVELOPE_BYTES.get(envelope) for envelope in env_list},
        "arms": [_arm_name(arm) for arm in arms],
        "rows": rows,
        "pareto_frontier_by_envelope": per_envelope_frontier,
        "pareto_frontier": global_frontier,
        "arms_on_pareto_frontier": sorted(arms_on_frontier),
        "pareto_objectives": {
            "cost": "resident_bytes",
            "benefit": "capability",
            "cost_sense": "minimize",
            "benefit_sense": "maximize",
            "size_is_not_the_goal": True,
        },
        "process_peak_resident_bytes": process_rss,
        "resident_measurement_policy": {
            "preferred": "resource.getrusage(RUSAGE_SELF).ru_maxrss converted to bytes",
            "macos_unit": "bytes",
            "linux_unit": "kilobytes_times_1024",
            "serialized_payload_length_refused_as_resident_proxy": True,
            "process_wide_peak_limitation": (
                "ru_maxrss is cumulative for the process; per-arm isolation needs separate processes or cgroups."
            ),
        },
        "energy_measurement_policy": {
            "proxy": "operation_count",
            "hardware_energy_measured": False,
            "label": "operation_count_energy_proxy",
        },
        "activation": False,
    }


def audit_report_digest(report: Mapping[str, Any]) -> str:
    """Stable digest of a parity or frontier report for evidence chaining."""
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), default=str).encode()
    import hashlib

    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "audit_report_digest",
    "capability_density_frontier",
    "extract_measurements",
    "measure_peak_resident_bytes",
    "parity_audit",
]
