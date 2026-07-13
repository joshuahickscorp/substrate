"""Governed 16-worker resource canary for the Generation 1 corpus worker.

The canary is resource evidence only.  It runs one frozen sorted batch through the exact corpus
``worker`` entry point, records process-tree and host telemetry, and never grants a scientific
promotion.  Admission is fail-closed and the launcher can signal only process groups whose PID,
creation time, command, and canary-owned session identity all still match.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import psutil

from ..config import REPO_ROOT
from ..studio import local_throttle as throttle
from . import generation1_cognitive_corpus as corpus

CANARY_SCHEMA = "mop-generation1-resource-canary/v1"
CANARY_BATCH_SIZE = 16
CANARY_ANCHOR = "ex9_slot_attention"
CANARY_BATCH = (
    "ex14_memory_bakeoff",
    "ex15_rejuvenation",
    "ex16_codebook_sr",
    "ex17_latent_reasoning",
    "ex18_self_verification",
    "ex1_generative_replay",
    "ex2_latent_planning",
    "ex3_test_time_adaptation",
    "ex4_fast_weights",
    "ex5_local_rules_scale",
    "ex6_active_inference",
    "ex7_meta_learning",
    "ex8_curiosity_bakeoff",
    "ex9_slot_attention",
    "f10_intrinsic_form_curriculum",
    "f11_form_dream_replay",
)
DEFAULT_ROOT = REPO_ROOT / "runs/generation1/resource_canary"
DEFAULT_PROOF_OUT = REPO_ROOT / "proof/GENERATION1_RESOURCE_CANARY.json"
WORKER_SCRIPT = REPO_ROOT / "scripts/generation1_cognitive_corpus.py"
CANARY_SCRIPT = REPO_ROOT / "scripts/generation1_resource_canary.py"
CANARY_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TAIL_BYTES = 12_000
DECIMAL_GB = 1_000_000_000


class CanaryRefused(RuntimeError):
    """The canary cannot proceed without weakening an authority or resource gate."""


@dataclass(frozen=True, slots=True)
class CanaryThresholds:
    admission_minimum_available_gb: float = 48.0
    admission_minimum_available_percent: float = 45.0
    admission_minimum_pressure_free_percent: float = 70.0
    admission_maximum_swap_gb: float = 1.0
    admission_maximum_load_per_cpu: float = 0.45
    admission_maximum_cpu_fraction: float = 0.45
    minimum_disk_free_gb: float = 40.0
    runtime_abort_available_gb: float = 24.0
    runtime_abort_available_percent: float = 25.0
    runtime_abort_pressure_free_percent: float = 40.0
    runtime_abort_swap_gb: float = 4.0


@dataclass(slots=True)
class WorkerHandle:
    experiment_id: str
    process: subprocess.Popen[bytes]
    create_time: float | None
    command: list[str]
    run_dir: Path
    stdout_path: Path
    stderr_path: Path
    stdout_handle: BinaryIO
    stderr_handle: BinaryIO
    started_monotonic: float
    monitored_peak_rss_bytes: int = 0
    timed_out: bool = False
    resource_stopped: bool = False
    stop_problem: str | None = None


@dataclass(frozen=True, slots=True)
class CanaryPlan:
    config_path: Path
    config: dict[str, Any]
    outer_seed: int
    batch: tuple[str, ...]
    cell_authorities: dict[str, dict[str, Any]]
    source_snapshot: dict[str, Any]
    policy: Any
    preflight: dict[str, Any]


def _sealed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    core = dict(payload)
    core.pop(field, None)
    return {**core, field: corpus.canonical_sha256(core)}


def valid_seal(payload: Mapping[str, Any], field: str = "receipt_sha256") -> bool:
    core = dict(payload)
    declared = core.pop(field, None)
    return isinstance(declared, str) and declared == corpus.canonical_sha256(core)


def derive_exact_batch(config: dict[str, Any]) -> tuple[str, ...]:
    """Return the aligned sorted 16-class batch containing the frozen anchor."""

    eligible = corpus.eligible_experiment_ids(config)
    if CANARY_ANCHOR not in eligible:
        raise CanaryRefused(f"canary anchor {CANARY_ANCHOR} is not eligible")
    index = eligible.index(CANARY_ANCHOR)
    start = (index // CANARY_BATCH_SIZE) * CANARY_BATCH_SIZE
    batch = tuple(eligible[start : start + CANARY_BATCH_SIZE])
    if len(batch) != CANARY_BATCH_SIZE or batch != CANARY_BATCH:
        raise CanaryRefused(f"sorted canary batch drifted: {batch!r}")
    return batch


def _numeric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nested(payload: Mapping[str, Any], *keys: str) -> object:
    value: object = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _gate(name: str, observed: object, limit: object, ok: bool, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "limit": limit,
        "ok": bool(ok),
        "reason": reason,
    }


def evaluate_admission(
    telemetry: Mapping[str, Any],
    active_lanes: Sequence[Mapping[str, Any]],
    thresholds: CanaryThresholds = CanaryThresholds(),
) -> dict[str, Any]:
    """Apply conservative launch gates to one complete local-throttle snapshot."""

    available_gb = (_numeric(_nested(telemetry, "memory", "available_bytes")) or 0.0) / DECIMAL_GB
    available_percent = _numeric(_nested(telemetry, "memory", "available_percent"))
    pressure_percent = _numeric(_nested(telemetry, "memory", "pressure", "free_percent"))
    swap_gb = _numeric(_nested(telemetry, "swap", "used_gb"))
    load_per_cpu = _numeric(_nested(telemetry, "cpu", "load_1m_per_logical_cpu"))
    cpu_fraction = _numeric(_nested(telemetry, "cpu", "utilization_fraction"))
    disk_gb = _numeric(_nested(telemetry, "disk", "free_gb"))
    thermal = _nested(telemetry, "thermal", "status")
    power_on_ac = _nested(telemetry, "power", "on_ac")
    unmanaged = _nested(telemetry, "processes", "unmanaged_known_heavy")
    foreground = _nested(telemetry, "processes", "foreground_resource_processes")
    gates = [
        _gate(
            "required_telemetry",
            list(telemetry.get("missing_required_telemetry") or []),
            [],
            telemetry.get("all_required_available") is True,
            "all pressure, swap, thermal, power, CPU, process, and disk probes must be available",
        ),
        _gate(
            "active_lanes",
            len(active_lanes),
            0,
            not active_lanes,
            "the 16-worker canary requires an otherwise empty governed lane registry",
        ),
        _gate(
            "unmanaged_heavy",
            len(unmanaged) if isinstance(unmanaged, list) else None,
            0,
            isinstance(unmanaged, list) and not unmanaged,
            "known heavy work outside the canary must be absent",
        ),
        _gate(
            "foreground_resource_work",
            len(foreground) if isinstance(foreground, list) else None,
            0,
            isinstance(foreground, list) and not foreground,
            "foreground rendering or encoding work must be absent",
        ),
        _gate(
            "memory_available_gb",
            available_gb,
            thresholds.admission_minimum_available_gb,
            available_gb >= thresholds.admission_minimum_available_gb,
            "preserve a conservative unified-memory reserve before 16 concurrent imports",
        ),
        _gate(
            "memory_available_percent",
            available_percent,
            thresholds.admission_minimum_available_percent,
            available_percent is not None
            and available_percent >= thresholds.admission_minimum_available_percent,
            "available-memory percentage must clear the canary floor",
        ),
        _gate(
            "memory_pressure_free_percent",
            pressure_percent,
            thresholds.admission_minimum_pressure_free_percent,
            pressure_percent is not None
            and pressure_percent >= thresholds.admission_minimum_pressure_free_percent,
            "macOS memory pressure must report ample free capacity",
        ),
        _gate(
            "swap_used_gb",
            swap_gb,
            thresholds.admission_maximum_swap_gb,
            swap_gb is not None and swap_gb <= thresholds.admission_maximum_swap_gb,
            "existing swap indicates memory contention and blocks the canary",
        ),
        _gate(
            "load_per_logical_cpu",
            load_per_cpu,
            thresholds.admission_maximum_load_per_cpu,
            load_per_cpu is not None and load_per_cpu <= thresholds.admission_maximum_load_per_cpu,
            "pre-existing one-minute CPU load must leave room for the exact batch",
        ),
        _gate(
            "cpu_utilization_fraction",
            cpu_fraction,
            thresholds.admission_maximum_cpu_fraction,
            cpu_fraction is not None and cpu_fraction <= thresholds.admission_maximum_cpu_fraction,
            "instantaneous CPU pressure must be low before launch",
        ),
        _gate(
            "disk_free_gb",
            disk_gb,
            thresholds.minimum_disk_free_gb,
            disk_gb is not None and disk_gb >= thresholds.minimum_disk_free_gb,
            "the isolated output namespace must preserve the project disk floor",
        ),
        _gate("thermal", thermal, "normal", thermal == "normal", "thermal state must be normal"),
        _gate("power", power_on_ac, True, power_on_ac is True, "the Mac Studio must be on AC power"),
    ]
    problems = [row["name"] for row in gates if not row["ok"]]
    return {"safe": not problems, "gates": gates, "problems": problems}


def runtime_safety_problems(
    telemetry: Mapping[str, Any],
    active_lanes: Sequence[Mapping[str, Any]],
    thresholds: CanaryThresholds = CanaryThresholds(),
) -> list[str]:
    """Return only conditions serious enough to stop canary-owned workers."""

    problems: list[str] = []
    available_gb = (_numeric(_nested(telemetry, "memory", "available_bytes")) or 0.0) / DECIMAL_GB
    available_percent = _numeric(_nested(telemetry, "memory", "available_percent"))
    pressure = _numeric(_nested(telemetry, "memory", "pressure", "free_percent"))
    swap_gb = _numeric(_nested(telemetry, "swap", "used_gb"))
    if telemetry.get("all_required_available") is not True:
        problems.append("required host telemetry became unavailable")
    if available_gb < thresholds.runtime_abort_available_gb:
        problems.append("available unified memory crossed the runtime abort floor")
    if available_percent is None or available_percent < thresholds.runtime_abort_available_percent:
        problems.append("available-memory percentage crossed the runtime abort floor")
    if pressure is None or pressure < thresholds.runtime_abort_pressure_free_percent:
        problems.append("memory pressure crossed the runtime abort floor")
    if swap_gb is None or swap_gb > thresholds.runtime_abort_swap_gb:
        problems.append("swap crossed the runtime abort ceiling")
    if _nested(telemetry, "thermal", "status") != "normal":
        problems.append("thermal state is no longer normal")
    if _nested(telemetry, "power", "on_ac") is not True:
        problems.append("AC power is no longer confirmed")
    disk_gb = _numeric(_nested(telemetry, "disk", "free_gb"))
    if disk_gb is None or disk_gb < thresholds.minimum_disk_free_gb:
        problems.append("disk free space crossed the project floor")
    if active_lanes:
        problems.append("another governed lane became active")
    unmanaged = _nested(telemetry, "processes", "unmanaged_known_heavy")
    if not isinstance(unmanaged, list) or unmanaged:
        problems.append("unmanaged known-heavy work appeared")
    return problems


def empty_measurements() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "host_probe_count": 0,
        "aggregate_process_tree_peak_rss_bytes": 0,
        "worker_process_trees_peak_rss_bytes": 0,
        "minimum_memory_available_bytes": None,
        "minimum_memory_available_percent": None,
        "minimum_memory_pressure_free_percent": None,
        "maximum_swap_used_bytes": None,
        "maximum_swap_used_percent": None,
        "maximum_cpu_utilization_fraction": None,
        "maximum_load_1m": None,
        "maximum_load_1m_per_logical_cpu": None,
        "thermal_statuses": [],
        "power_sources": [],
        "power_on_ac_values": [],
        "runtime_safety_problems": [],
        "host_probes": [],
    }


def _extreme(summary: dict[str, Any], key: str, value: object, *, minimum: bool) -> None:
    number = _numeric(value)
    if number is None:
        return
    candidate: int | float = value if isinstance(value, int) and not isinstance(value, bool) else number
    prior_value = summary.get(key)
    prior = _numeric(prior_value)
    if prior is None or minimum and number < prior or not minimum and number > prior:
        summary[key] = candidate


def record_measurement(
    summary: dict[str, Any],
    *,
    fast_host: Mapping[str, Any],
    aggregate_process_tree_rss_bytes: int,
    worker_process_trees_rss_bytes: int,
    full_probe: Mapping[str, Any] | None = None,
) -> None:
    """Fold one fast sample and an optional complete host probe into exact extrema."""

    summary["sample_count"] += 1
    summary["aggregate_process_tree_peak_rss_bytes"] = max(
        int(summary["aggregate_process_tree_peak_rss_bytes"]), aggregate_process_tree_rss_bytes
    )
    summary["worker_process_trees_peak_rss_bytes"] = max(
        int(summary["worker_process_trees_peak_rss_bytes"]), worker_process_trees_rss_bytes
    )
    _extreme(summary, "minimum_memory_available_bytes", fast_host.get("memory_available_bytes"), minimum=True)
    _extreme(
        summary,
        "minimum_memory_available_percent",
        fast_host.get("memory_available_percent"),
        minimum=True,
    )
    _extreme(summary, "maximum_swap_used_bytes", fast_host.get("swap_used_bytes"), minimum=False)
    _extreme(summary, "maximum_swap_used_percent", fast_host.get("swap_used_percent"), minimum=False)
    _extreme(
        summary,
        "maximum_cpu_utilization_fraction",
        fast_host.get("cpu_utilization_fraction"),
        minimum=False,
    )
    _extreme(summary, "maximum_load_1m", fast_host.get("load_1m"), minimum=False)
    _extreme(
        summary,
        "maximum_load_1m_per_logical_cpu",
        fast_host.get("load_1m_per_logical_cpu"),
        minimum=False,
    )
    if full_probe is None:
        return
    summary["host_probe_count"] += 1
    pressure = _nested(full_probe, "memory", "pressure", "free_percent")
    _extreme(summary, "minimum_memory_pressure_free_percent", pressure, minimum=True)
    thermal = _nested(full_probe, "thermal", "status")
    if isinstance(thermal, str) and thermal not in summary["thermal_statuses"]:
        summary["thermal_statuses"].append(thermal)
    source = _nested(full_probe, "power", "source")
    if isinstance(source, str) and source not in summary["power_sources"]:
        summary["power_sources"].append(source)
    on_ac = _nested(full_probe, "power", "on_ac")
    if isinstance(on_ac, bool) and on_ac not in summary["power_on_ac_values"]:
        summary["power_on_ac_values"].append(on_ac)
    summary["host_probes"].append(
        {
            "created_at": full_probe.get("created_at"),
            "memory_available_bytes": _nested(full_probe, "memory", "available_bytes"),
            "memory_available_percent": _nested(full_probe, "memory", "available_percent"),
            "memory_pressure_free_percent": pressure,
            "swap_used_bytes": _nested(full_probe, "swap", "used_bytes"),
            "cpu_utilization_fraction": _nested(full_probe, "cpu", "utilization_fraction"),
            "load_1m": _nested(full_probe, "cpu", "load_1m"),
            "thermal_status": thermal,
            "power_source": source,
            "power_on_ac": on_ac,
            "missing_required_telemetry": list(full_probe.get("missing_required_telemetry") or []),
        }
    )


def recommend_resources(
    measurements: Mapping[str, Any],
    worker_rows: Sequence[Mapping[str, Any]],
    *,
    source_stable: bool,
) -> dict[str, Any]:
    """Recommend no more concurrency than was actually observed safely."""

    all_ok = len(worker_rows) == CANARY_BATCH_SIZE and all(row.get("outcome") == "ok" for row in worker_rows)
    runtime_safe = not measurements.get("runtime_safety_problems")
    eligible = bool(all_ok and runtime_safe and source_stable)
    peak = int(measurements.get("aggregate_process_tree_peak_rss_bytes") or 0)
    individual_peak = max((int(row.get("peak_rss_bytes") or 0) for row in worker_rows), default=0)
    if eligible and peak > 0:
        # 50% measured margin plus one decimal GB for the supervisor/telemetry envelope, rounded
        # upward in two-GB steps.  No concurrency extrapolation beyond the tested 16 is allowed.
        recommended_bytes = int(math.ceil((peak * 1.5 + DECIMAL_GB) / (2 * DECIMAL_GB)) * 2 * DECIMAL_GB)
        recommended_memory_gb: float | None = recommended_bytes / DECIMAL_GB
        recommended_workers: int | None = CANARY_BATCH_SIZE
    else:
        recommended_bytes = 0
        recommended_memory_gb = None
        recommended_workers = None
    return {
        "eligible": eligible,
        "recommended_max_workers": recommended_workers,
        "recommended_estimated_unified_memory_gb": recommended_memory_gb,
        "recommended_estimated_unified_memory_bytes": recommended_bytes or None,
        "observed_aggregate_process_tree_peak_rss_bytes": peak,
        "observed_max_individual_worker_rss_bytes": individual_peak,
        "margin_rule": "1.5x observed aggregate process-tree peak plus 1 GB, rounded up to 2 GB",
        "scaling_boundary": "16 workers maximum; no extrapolation beyond the measured batch",
        "scientific_promotion": False,
    }


def _snapshot_file(path: Path, repo_root: Path) -> dict[str, Any]:
    before = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise CanaryRefused(f"authority is not a regular non-symlink file: {path}")
    digest = corpus.sha256_file(path)
    after = path.lstat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise CanaryRefused(f"authority changed while being hashed: {path}")
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(repo_root.resolve())),
        "sha256": digest,
        "bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
    }


def source_snapshot(
    config_path: Path,
    config: dict[str, Any],
    batch: Sequence[str],
    outer_seed: int,
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Bind every file and effective-seed cell authority consumed by the exact batch."""

    cells = {
        experiment_id: corpus._expected_cell_authority(config, experiment_id, outer_seed)
        for experiment_id in batch
    }
    paths = {
        config_path.resolve(),
        CANARY_SCRIPT.resolve(),
        WORKER_SCRIPT.resolve(),
        Path(__file__).resolve(),
        Path(corpus.__file__).resolve(),
        Path(throttle.__file__).resolve(),
        throttle.DEFAULT_POLICY.resolve(),
        (repo_root / "src/mop/harness/runner.py").resolve(),
        (repo_root / "src/mop/experiments/__init__.py").resolve(),
    }
    for cell in cells.values():
        paths.add((repo_root / cell["experiment_config"]["path"]).resolve())
        for authority in cell["implementation_authorities"]:
            paths.add((repo_root / authority["path"]).resolve())
    files = [_snapshot_file(path, repo_root) for path in sorted(paths, key=str)]
    cell_digests = {
        experiment_id: corpus.canonical_sha256(cell) for experiment_id, cell in sorted(cells.items())
    }
    core = {
        "files": files,
        "cell_authority_sha256": cell_digests,
        "seed_policy_sha256": corpus.canonical_sha256(config["seed_authority"]),
    }
    return {**core, "aggregate_sha256": corpus.canonical_sha256(core)}, cells


def _adapter_problems(
    config: Mapping[str, Any], batch: Sequence[str], cells: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    problems: list[str] = []
    policy = config.get("seed_authority")
    if not isinstance(policy, Mapping):
        return ["final v2 seed_authority policy is absent"]
    if policy.get("schema") != corpus.SEED_POLICY_SCHEMA or policy.get("algorithm") != corpus.SEED_ALGORITHM:
        problems.append("seed-authority schema or algorithm drifted")
    for experiment_id in batch:
        cell = cells.get(experiment_id)
        authority = cell.get("seed_authority") if isinstance(cell, Mapping) else None
        if not isinstance(authority, Mapping):
            problems.append(f"{experiment_id}: effective seed authority is absent")
            continue
        if authority.get("schema") != corpus.SEED_AUTHORITY_SCHEMA:
            problems.append(f"{experiment_id}: effective seed authority schema drifted")
        if authority.get("outer_seed") != config["seeds"][0]:
            problems.append(f"{experiment_id}: effective seed authority outer seed drifted")
        if (
            authority.get("mode") == corpus.SEED_MODE_VARIED
            and len(authority.get("effective_overrides") or []) < 1
        ):
            problems.append(f"{experiment_id}: varied cell has no effective seed override")
    return problems


def _collect_full_telemetry(
    policy: Any,
    repo_root: Path,
    handles: Sequence[WorkerHandle] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    owned = {handle.process.pid for handle in handles if handle.process.poll() is None}
    telemetry = throttle.collect_host_telemetry(
        policy,
        disk_root=repo_root,
        excluded_pids=owned,
        excluded_process_groups=owned,
    )
    active = throttle.active_lanes(repo_root / "runs/local_throttle")
    return telemetry, active


def prepare_plan(
    config_path: Path = corpus.DEFAULT_CONFIG,
    *,
    repo_root: Path = REPO_ROOT,
    source_quiet_seconds: float = 60.0,
    thresholds: CanaryThresholds = CanaryThresholds(),
) -> tuple[CanaryPlan | None, dict[str, Any]]:
    """Rebind v2 seed authority and collect a fail-closed launch decision."""

    problems: list[str] = []
    config: dict[str, Any] | None = None
    batch: tuple[str, ...] = ()
    cells: dict[str, dict[str, Any]] = {}
    sources: dict[str, Any] | None = None
    policy: Any = None
    try:
        config = corpus.load_config(config_path)
        batch = derive_exact_batch(config)
        outer_seed = int(config["seeds"][0])
        sources, cells = source_snapshot(config_path, config, batch, outer_seed, repo_root=repo_root)
        problems.extend(_adapter_problems(config, batch, cells))
        minimum_age = min(time.time() - row["mtime_ns"] / 1e9 for row in sources["files"])
        if minimum_age < source_quiet_seconds:
            problems.append(
                f"source authority has been quiet for only {minimum_age:.3f}s; "
                f"requires {source_quiet_seconds:.3f}s"
            )
    except (OSError, ValueError, CanaryRefused) as exc:
        outer_seed = 0
        problems.append(f"source authority is not ready: {type(exc).__name__}: {exc}")
        minimum_age = None
    try:
        policy = throttle.load_policy()
        telemetry, active = _collect_full_telemetry(policy, repo_root)
        admission = evaluate_admission(telemetry, active, thresholds)
        problems.extend(f"resource gate failed: {name}" for name in admission["problems"])
    except (OSError, ValueError, throttle.ThrottleRefused) as exc:
        telemetry = {"all_required_available": False, "missing_required_telemetry": ["preflight"]}
        active = []
        admission = {"safe": False, "gates": [], "problems": ["telemetry"]}
        problems.append(f"resource telemetry is not ready: {type(exc).__name__}: {exc}")
    preflight_core = {
        "schema": "mop-generation1-resource-canary-preflight/v1",
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path.resolve().relative_to(repo_root.resolve())),
        "required_config_schema": corpus.CONFIG_SCHEMA,
        "observed_config_schema": config.get("schema") if config else None,
        "batch": list(batch),
        "batch_sha256": corpus.canonical_sha256(list(batch)) if batch else None,
        "outer_seed": outer_seed or None,
        "source_quiet_seconds_required": source_quiet_seconds,
        "source_quiet_seconds_observed": minimum_age,
        "source_snapshot": sources,
        "resource_thresholds": asdict(thresholds),
        "telemetry": telemetry,
        "active_lanes": active,
        "admission": admission,
        "problems": problems,
        "launch_authorized": not problems,
        "scientific_promotion": False,
    }
    preflight = _sealed(preflight_core, "preflight_sha256")
    if problems or config is None or sources is None or policy is None:
        return None, preflight
    return (
        CanaryPlan(
            config_path=config_path.resolve(),
            config=config,
            outer_seed=outer_seed,
            batch=batch,
            cell_authorities=cells,
            source_snapshot=sources,
            policy=policy,
            preflight=preflight,
        ),
        preflight,
    )


def wait_for_plan(
    config_path: Path = corpus.DEFAULT_CONFIG,
    *,
    repo_root: Path = REPO_ROOT,
    source_quiet_seconds: float = 60.0,
    maximum_wait_seconds: float = 600.0,
    poll_seconds: float = 5.0,
    thresholds: CanaryThresholds = CanaryThresholds(),
) -> tuple[CanaryPlan | None, dict[str, Any]]:
    """Recheck source and host gates until safe or a bounded deterministic deadline expires."""

    if maximum_wait_seconds < 0 or poll_seconds <= 0:
        raise CanaryRefused("launch wait must be nonnegative and polling must be positive")
    deadline = time.monotonic() + maximum_wait_seconds
    while True:
        plan, preflight = prepare_plan(
            config_path,
            repo_root=repo_root,
            source_quiet_seconds=source_quiet_seconds,
            thresholds=thresholds,
        )
        if plan is not None or time.monotonic() >= deadline:
            return plan, preflight
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def worker_command(
    experiment_id: str,
    *,
    outer_seed: int,
    run_dir: Path,
    result_tag: str,
    config_path: Path,
    python: str = sys.executable,
    worker_script: Path = WORKER_SCRIPT,
) -> list[str]:
    if experiment_id not in CANARY_BATCH:
        raise CanaryRefused(f"experiment is outside the frozen canary batch: {experiment_id}")
    return [
        python,
        str(worker_script),
        "worker",
        "--experiment",
        experiment_id,
        "--seed",
        str(outer_seed),
        "--run-dir",
        str(run_dir),
        "--result-tag",
        result_tag,
        "--config",
        str(config_path),
    ]


def _fast_host() -> dict[str, Any]:
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    load1, load5, load15 = os.getloadavg()
    logical = int(psutil.cpu_count(logical=True) or 0)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "memory_available_bytes": int(virtual.available),
        "memory_available_percent": float(100.0 - virtual.percent),
        "swap_used_bytes": int(swap.used),
        "swap_used_percent": float(swap.percent),
        "cpu_utilization_fraction": float(psutil.cpu_percent(interval=None) / 100.0),
        "load_1m": float(load1),
        "load_5m": float(load5),
        "load_15m": float(load15),
        "load_1m_per_logical_cpu": float(load1 / logical) if logical else None,
    }


def _tree_rss(pid: int, create_time: float | None) -> int | None:
    if create_time is None:
        return None
    try:
        process = psutil.Process(pid)
        if not math.isclose(process.create_time(), create_time, rel_tol=0.0, abs_tol=0.01):
            return None
        values = [int(process.memory_info().rss)]
        for child in process.children(recursive=True):
            try:
                values.append(int(child.memory_info().rss))
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        return sum(values)
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError):
        return 0 if psutil.pid_exists(pid) is False else None


def _owned_identity_verified(handle: WorkerHandle) -> bool:
    if handle.create_time is None or handle.process.poll() is not None:
        return False
    try:
        process = psutil.Process(handle.process.pid)
        command = process.cmdline()
        return bool(
            math.isclose(process.create_time(), handle.create_time, rel_tol=0.0, abs_tol=0.01)
            and process.ppid() == os.getpid()
            and os.getpgid(process.pid) == process.pid
            and command == handle.command
        )
    except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError):
        return False


def signal_owned_process_group(handle: WorkerHandle, requested_signal: int) -> bool:
    """Signal only the exact, still-live session created for this handle."""

    if requested_signal not in {signal.SIGTERM, signal.SIGKILL} or not _owned_identity_verified(handle):
        return False
    os.killpg(handle.process.pid, requested_signal)
    return True


def _stop_owned(handles: Sequence[WorkerHandle], reason: str, *, timeout: bool = False) -> list[str]:
    problems: list[str] = []
    live = [handle for handle in handles if handle.process.poll() is None]
    for handle in live:
        handle.timed_out = handle.timed_out or timeout
        handle.resource_stopped = handle.resource_stopped or not timeout
        handle.stop_problem = reason
        if not signal_owned_process_group(handle, signal.SIGTERM):
            problems.append(f"{handle.experiment_id}: refused SIGTERM because owned identity was not exact")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(handle.process.poll() is None for handle in live):
        time.sleep(0.05)
    for handle in live:
        if handle.process.poll() is None and not signal_owned_process_group(handle, signal.SIGKILL):
            problems.append(f"{handle.experiment_id}: refused SIGKILL because owned identity was not exact")
    return problems


def _launch_workers(plan: CanaryPlan, run_root: Path) -> list[WorkerHandle]:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
        }
    )
    handles: list[WorkerHandle] = []
    for experiment_id in plan.batch:
        worker_root = run_root / "workers" / experiment_id
        run_dir = worker_root / "attempt_001"
        worker_root.mkdir(parents=True, exist_ok=False)
        mpl = run_root / "mplconfig" / experiment_id
        mpl.mkdir(parents=True, exist_ok=False)
        worker_environment = {**environment, "MPLCONFIGDIR": str(mpl)}
        stdout_path = worker_root / "stdout.log"
        stderr_path = worker_root / "stderr.log"
        stdout_handle = stdout_path.open("wb")
        stderr_handle = stderr_path.open("wb")
        command = worker_command(
            experiment_id,
            outer_seed=plan.outer_seed,
            run_dir=run_dir,
            result_tag=str(plan.config["result_tag"]),
            config_path=plan.config_path,
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=worker_environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
                close_fds=True,
            )
            try:
                create_time = float(psutil.Process(process.pid).create_time())
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                create_time = None
            handles.append(
                WorkerHandle(
                    experiment_id=experiment_id,
                    process=process,
                    create_time=create_time,
                    command=command,
                    run_dir=run_dir,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    stdout_handle=stdout_handle,
                    stderr_handle=stderr_handle,
                    started_monotonic=time.monotonic(),
                )
            )
        except Exception:
            stdout_handle.close()
            stderr_handle.close()
            if handles:
                _stop_owned(handles, "later worker failed during launch")
            raise
    return handles


def _read_tail(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - TAIL_BYTES))
        return handle.read().decode("utf-8", errors="replace")


def _worker_report(stdout: str) -> dict[str, Any] | None:
    prefix = "GENERATION1_WORKER="
    for line in reversed(stdout.splitlines()):
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line.removeprefix(prefix))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_attempt_receipt(
    plan: CanaryPlan,
    handle: WorkerHandle,
    *,
    returncode: int | None,
    seconds: float,
    stdout_tail: str,
    stderr_tail: str,
    report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Write the supervisor-side v2 receipt omitted by the direct worker entry point."""

    if not handle.run_dir.is_dir():
        return None
    authority = plan.cell_authorities[handle.experiment_id]
    attempt = corpus.Attempt(
        experiment_id=handle.experiment_id,
        seed=plan.outer_seed,
        run_dir=corpus._repository_path(handle.run_dir),
        returncode=returncode,
        timed_out=handle.timed_out,
        seconds=round(seconds, 6),
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        evidence_class=str(authority["evidence_class"]),
        seed_mode=str(authority["seed_mode"]),
        seed_authority=dict(authority["seed_authority"]),
        experiment_config=dict(authority["experiment_config"]),
        implementation_authorities=list(authority["implementation_authorities"]),
        resolved_config=(
            dict(report["resolved_config"])
            if report and isinstance(report.get("resolved_config"), dict)
            else None
        ),
        manifest=(
            dict(report["manifest"])
            if report and isinstance(report.get("manifest"), dict)
            else None
        ),
        worker_report=report,
    )
    receipt = corpus._sealed(
        {
            "schema": corpus.ATTEMPT_SCHEMA,
            **asdict(attempt),
            "recorded_at": datetime.now(UTC).isoformat(),
        },
        "attempt_sha256",
    )
    path = handle.run_dir / "attempt_receipt.json"
    corpus._atomic_json(path, receipt)
    return {
        "path": corpus._repository_path(path),
        "sha256": corpus.sha256_file(path),
        "attempt_sha256": receipt["attempt_sha256"],
        "self_seal_valid": corpus._valid_seal(receipt, "attempt_sha256"),
    }


def _worker_rows(plan: CanaryPlan, handles: Sequence[WorkerHandle]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for handle in handles:
        handle.stdout_handle.close()
        handle.stderr_handle.close()
        returncode = handle.process.wait()
        stdout_tail = _read_tail(handle.stdout_path)
        stderr_tail = _read_tail(handle.stderr_path)
        report = _worker_report(stdout_tail)
        wall_seconds = time.monotonic() - handle.started_monotonic
        attempt_receipt = _write_attempt_receipt(
            plan,
            handle,
            returncode=returncode,
            seconds=wall_seconds,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            report=report,
        )
        manifest = handle.run_dir / "manifest.json"
        manifest_valid = corpus._manifest_ok(
            manifest,
            experiment_id=handle.experiment_id,
            seed=plan.outer_seed,
            result_tag=str(plan.config["result_tag"]),
            expected_cell_authority=plan.cell_authorities[handle.experiment_id],
        )
        reported_peak = int(report.get("maximum_rss_bytes") or 0) if report else 0
        peak = max(handle.monitored_peak_rss_bytes, reported_peak)
        if handle.timed_out:
            outcome = "timed_out"
        elif handle.resource_stopped:
            outcome = "resource_stopped"
        elif returncode == 0 and report is not None and manifest_valid:
            outcome = "ok"
        else:
            outcome = "failed"
        rows.append(
            {
                "experiment_id": handle.experiment_id,
                "outcome": outcome,
                "pid": handle.process.pid,
                "create_time": handle.create_time,
                "returncode": returncode,
                "timed_out": handle.timed_out,
                "resource_stopped": handle.resource_stopped,
                "stop_problem": handle.stop_problem,
                "wall_seconds": round(wall_seconds, 6),
                "monitored_process_tree_peak_rss_bytes": handle.monitored_peak_rss_bytes,
                "worker_reported_peak_rss_bytes": reported_peak,
                "peak_rss_bytes": peak,
                "worker_report": report,
                "attempt_receipt": attempt_receipt,
                "manifest": {
                    "path": str(manifest.relative_to(REPO_ROOT)) if manifest.is_file() else None,
                    "sha256": corpus.sha256_file(manifest) if manifest.is_file() else None,
                    "valid": manifest_valid,
                },
                "stdout": {
                    "path": str(handle.stdout_path.relative_to(REPO_ROOT)),
                    "sha256": corpus.sha256_file(handle.stdout_path),
                    "tail": stdout_tail,
                },
                "stderr": {
                    "path": str(handle.stderr_path.relative_to(REPO_ROOT)),
                    "sha256": corpus.sha256_file(handle.stderr_path),
                    "tail": stderr_tail,
                },
                "output_bytes": _directory_bytes(handle.run_dir) if handle.run_dir.is_dir() else 0,
                "command_sha256": corpus.canonical_sha256(handle.command),
            }
        )
    return sorted(rows, key=lambda row: str(row["experiment_id"]))


def run_canary(
    plan: CanaryPlan,
    *,
    run_id: str,
    repo_root: Path = REPO_ROOT,
    timeout_seconds: float = 900.0,
    sample_interval_seconds: float = 0.05,
    host_probe_interval_seconds: float = 1.0,
    thresholds: CanaryThresholds = CanaryThresholds(),
) -> tuple[Path, dict[str, Any]]:
    if CANARY_ID_RE.fullmatch(run_id) is None:
        raise CanaryRefused("run_id is invalid")
    if timeout_seconds <= 0 or sample_interval_seconds <= 0 or host_probe_interval_seconds <= 0:
        raise CanaryRefused("timeout and telemetry intervals must be positive")
    run_root = (repo_root / "runs/generation1/resource_canary" / run_id).resolve()
    if not run_root.is_relative_to((repo_root / "runs/generation1/resource_canary").resolve()):
        raise CanaryRefused("run root escaped the isolated resource_canary namespace")
    if run_root.exists():
        raise CanaryRefused(f"canary run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    receipt_path = run_root / "resource_canary.json"
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    measurements = empty_measurements()
    handles: list[WorkerHandle] = []
    orchestration_problems: list[str] = []
    source_stable = False
    psutil.cpu_percent(interval=None)
    try:
        current_sources, _ = source_snapshot(
            plan.config_path,
            plan.config,
            plan.batch,
            plan.outer_seed,
            repo_root=repo_root,
        )
        if current_sources != plan.source_snapshot:
            raise CanaryRefused("source authority changed between preflight and launch")
        handles = _launch_workers(plan, run_root)
        own = psutil.Process(os.getpid())
        own_create_time = float(own.create_time())
        last_probe = 0.0
        full_probe: dict[str, Any] | None = None
        while any(handle.process.poll() is None for handle in handles):
            now = time.monotonic()
            live = [handle for handle in handles if handle.process.poll() is None]
            timed_out = [handle for handle in live if now - handle.started_monotonic > timeout_seconds]
            if timed_out:
                orchestration_problems.extend(
                    _stop_owned(timed_out, f"worker exceeded {timeout_seconds:.3f}s", timeout=True)
                )
            worker_total = 0
            for handle in handles:
                rss = _tree_rss(handle.process.pid, handle.create_time)
                if rss is not None:
                    handle.monitored_peak_rss_bytes = max(handle.monitored_peak_rss_bytes, rss)
                    worker_total += rss
            aggregate = _tree_rss(os.getpid(), own_create_time) or worker_total
            if now - last_probe >= host_probe_interval_seconds:
                full_probe, active = _collect_full_telemetry(plan.policy, repo_root, handles)
                last_probe = now
                runtime_problems = runtime_safety_problems(full_probe, active, thresholds)
                try:
                    observed_sources, _ = source_snapshot(
                        plan.config_path,
                        plan.config,
                        plan.batch,
                        plan.outer_seed,
                        repo_root=repo_root,
                    )
                except (OSError, ValueError, CanaryRefused) as exc:
                    observed_sources = {}
                    runtime_problems.append(f"source authority revalidation failed: {exc}")
                if observed_sources != plan.source_snapshot:
                    runtime_problems.append("source authority changed during canary execution")
                for problem in runtime_problems:
                    if problem not in measurements["runtime_safety_problems"]:
                        measurements["runtime_safety_problems"].append(problem)
                if runtime_problems:
                    orchestration_problems.extend(_stop_owned(live, "; ".join(runtime_problems)))
            record_measurement(
                measurements,
                fast_host=_fast_host(),
                aggregate_process_tree_rss_bytes=aggregate,
                worker_process_trees_rss_bytes=worker_total,
                full_probe=full_probe if now == last_probe else None,
            )
            time.sleep(sample_interval_seconds)
    except Exception as exc:
        orchestration_problems.append(f"{type(exc).__name__}: {exc}")
        if handles:
            orchestration_problems.extend(_stop_owned(handles, "canary orchestration failure"))
    finally:
        for handle in handles:
            if handle.process.poll() is None:
                orchestration_problems.extend(_stop_owned([handle], "canary finalizer owned-child cleanup"))
    workers = _worker_rows(plan, handles) if handles else []
    try:
        final_config = corpus.load_config(plan.config_path)
        final_batch = derive_exact_batch(final_config)
        final_sources, _ = source_snapshot(
            plan.config_path,
            final_config,
            final_batch,
            plan.outer_seed,
            repo_root=repo_root,
        )
        source_stable = final_sources == plan.source_snapshot
    except (OSError, ValueError, CanaryRefused) as exc:
        final_sources = {"error": f"{type(exc).__name__}: {exc}"}
        orchestration_problems.append("final source authority revalidation failed")
    recommendation = recommend_resources(measurements, workers, source_stable=source_stable)
    outcomes = dict(sorted(Counter(str(row["outcome"]) for row in workers).items()))
    complete = bool(recommendation["eligible"] and not orchestration_problems)
    core = {
        "schema": CANARY_SCHEMA,
        "run_id": run_id,
        "claim_scope": "resource canary only; no capability, natural-world, or architecture claim",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": round(time.monotonic() - started, 6),
        "configuration": {
            "max_workers": CANARY_BATCH_SIZE,
            "batch": list(plan.batch),
            "batch_sha256": corpus.canonical_sha256(list(plan.batch)),
            "anchor": CANARY_ANCHOR,
            "outer_seed": plan.outer_seed,
            "result_tag": plan.config["result_tag"],
            "timeout_seconds": timeout_seconds,
            "sample_interval_seconds": sample_interval_seconds,
            "host_probe_interval_seconds": host_probe_interval_seconds,
            "thresholds": asdict(thresholds),
            "run_root": str(run_root.relative_to(repo_root)),
        },
        "preflight": plan.preflight,
        "source_authority": {
            "before": plan.source_snapshot,
            "after": final_sources,
            "stable": source_stable,
        },
        "workers": workers,
        "outcome_counts": outcomes,
        "measurements": measurements,
        "recommendation": recommendation,
        "orchestration_problems": orchestration_problems,
        "complete": complete,
        "scientific_promotion": False,
    }
    receipt = _sealed(core, "receipt_sha256")
    corpus._atomic_json(receipt_path, receipt)
    return receipt_path, receipt


def publish_canonical_proof(
    run_receipt_path: Path,
    receipt: Mapping[str, Any],
    proof_out: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Atomically publish a successful terminal receipt with exact byte equality."""

    expected = (repo_root / "proof/GENERATION1_RESOURCE_CANARY.json").resolve()
    destination = proof_out.resolve()
    if destination != expected:
        raise CanaryRefused(f"proof output must be the canonical path: {expected}")
    if receipt.get("complete") is not True or not valid_seal(receipt):
        raise CanaryRefused("only a complete self-sealed canary receipt may be published")
    if run_receipt_path.is_symlink() or not run_receipt_path.is_file():
        raise CanaryRefused("terminal run receipt is not a regular non-symlink file")
    try:
        disk_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CanaryRefused("terminal run receipt is not valid JSON") from exc
    if disk_receipt != dict(receipt) or not valid_seal(disk_receipt):
        raise CanaryRefused("terminal run receipt differs from the in-memory sealed result")
    run_bytes = run_receipt_path.read_bytes()
    if destination.is_symlink():
        raise CanaryRefused("canonical proof destination is a symlink")
    if destination.exists() and destination.read_bytes() != run_bytes:
        raise CanaryRefused("canonical proof already exists with different bytes")
    corpus._atomic_json(destination, disk_receipt)
    proof_bytes = destination.read_bytes()
    if proof_bytes != run_bytes:
        raise CanaryRefused("atomic proof publication did not preserve exact receipt bytes")
    return {
        "path": str(destination.relative_to(repo_root.resolve())),
        "sha256": corpus.sha256_file(destination),
        "bytes": len(proof_bytes),
        "byte_equal_to_run_receipt": True,
        "receipt_sha256": disk_receipt["receipt_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=corpus.DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--proof-out", type=Path)
    parser.add_argument("--source-quiet-seconds", type=float, default=60.0)
    parser.add_argument("--launch-wait-seconds", type=float, default=600.0)
    parser.add_argument("--launch-poll-seconds", type=float, default=5.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.05)
    parser.add_argument("--host-probe-interval-seconds", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.execute:
            plan, preflight = wait_for_plan(
                arguments.config.resolve(),
                source_quiet_seconds=arguments.source_quiet_seconds,
                maximum_wait_seconds=arguments.launch_wait_seconds,
                poll_seconds=arguments.launch_poll_seconds,
            )
        else:
            plan, preflight = prepare_plan(
                arguments.config.resolve(),
                source_quiet_seconds=arguments.source_quiet_seconds,
            )
        if not arguments.execute:
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return 0 if plan is not None else 2
        if plan is None:
            raise CanaryRefused("live preflight did not authorize the canary")
        run_id = arguments.run_id or datetime.now(UTC).strftime("canary-%Y%m%dT%H%M%SZ")
        path, receipt = run_canary(
            plan,
            run_id=run_id,
            timeout_seconds=arguments.timeout_seconds,
            sample_interval_seconds=arguments.sample_interval_seconds,
            host_probe_interval_seconds=arguments.host_probe_interval_seconds,
        )
        proof = (
            publish_canonical_proof(path, receipt, arguments.proof_out)
            if arguments.proof_out is not None
            else None
        )
        print(
            json.dumps(
                {
                    "complete": receipt["complete"],
                    "path": str(path),
                    "outcome_counts": receipt["outcome_counts"],
                    "recommendation": receipt["recommendation"],
                    "receipt_sha256": receipt["receipt_sha256"],
                    "canonical_proof": proof,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if receipt["complete"] else 1
    except (CanaryRefused, OSError, ValueError, throttle.ThrottleRefused) as exc:
        print(json.dumps({"complete": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


__all__ = [
    "CANARY_ANCHOR",
    "CANARY_BATCH",
    "CANARY_BATCH_SIZE",
    "CANARY_SCHEMA",
    "DEFAULT_PROOF_OUT",
    "CanaryPlan",
    "CanaryRefused",
    "CanaryThresholds",
    "WorkerHandle",
    "derive_exact_batch",
    "empty_measurements",
    "evaluate_admission",
    "main",
    "prepare_plan",
    "publish_canonical_proof",
    "recommend_resources",
    "record_measurement",
    "run_canary",
    "runtime_safety_problems",
    "signal_owned_process_group",
    "source_snapshot",
    "valid_seal",
    "wait_for_plan",
    "worker_command",
]
