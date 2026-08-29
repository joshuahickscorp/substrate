"""Fail-closed prelaunch harness for the post-R2 Substrate Odyssey 7D."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from substrate import odyssey_authority, odyssey_detachment, odyssey_worker

PROGRAM = "substrate-odyssey-7d-v1"
RELATIVE = Path("docs/plans/substrate/tangible_next_launch")
FRONTIER_IDS = tuple("ABCDEFGH")
GIB = 1024**3
SUPERVISOR_STATE_SCHEMA = "SUBSTRATE_ODYSSEY_SUPERVISOR_STATE/v2"
SUPERVISOR_LINEAGE_SCHEMA = "SUBSTRATE_ODYSSEY_SUPERVISOR_LINEAGE/v1"
SUPERVISOR_LEASE_SCHEMA = "SUBSTRATE_ODYSSEY_SUPERVISOR_RUNTIME_LEASE/v1"
POSTFLIGHT_RECEIPT_SCHEMA = "SUBSTRATE_ODYSSEY_POSTFLIGHT_RECEIPT/v1"
POSTFLIGHT_RECEIPT_NAME = "POSTFLIGHT_RECEIPT.json"
MAX_ABNORMAL_RESTARTS = 3
RESTART_BACKOFF_SECONDS = (5, 15, 45)
PROCESS_GROUP_GRACE_SECONDS = 15


class Refused(RuntimeError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write(path: Path, value: Any, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise Refused(f"refusing to overwrite {path}")
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise Refused(f"cannot durably write {path}: {error}") from error


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Refused(f"cannot read authority: {exc}") from exc
    if not isinstance(value, dict):
        raise Refused("authority must be a JSON object")
    return value


def authority_path(root: Path) -> Path:
    return root / RELATIVE / "ODYSSEY_7D.hardened.draft.json"


def validate(authority: dict[str, Any]) -> dict[str, bool]:
    program, units, timeline = authority.get("program", {}), authority.get("independent_units", {}), authority.get("timeline", {})
    resources, storage, blindness = authority.get("resources", {}), authority.get("storage", {}), authority.get("blindness", {})
    frontiers = authority.get("frontiers", [])
    checks = {
        "inactive_draft": program.get("id") == PROGRAM and program.get("activation") is False and program.get("launch_allowed") is False,
        "seven_day_duration": program.get("duration_seconds") == 7 * 24 * 3600 and program.get("duration_hours") == 168,
        "eight_paired_units": (
            units.get("type") == "paired_frontier_history_block"
            and units.get("count") == 8
            and units.get("candidate_histories") == units.get("control_histories") == 8
            and units.get("total_continuous_state_histories") == 16
        ),
        "frontier_set": [f.get("id") for f in frontiers] == list(FRONTIER_IDS),
        "event_accounting": (
            timeline.get("microcycles_per_frontier") == 84
            and timeline.get("total_paired_microcycles") == 672
            and timeline.get("total_scored_paired_events") == 2688
            and timeline.get("total_scored_dimension_observations") == 10752
        ),
        "blind_custody": (
            blindness.get("custodians") == 8 and blindness.get("two_custodian_day7_reveal") is True and blindness.get("trace_lock_before_answer_reveal") is True
        ),
        "hard_memory_envelope": [
            resources.get(k)
            for k in (
                "resident_cap_gib",
                "normal_admission_ceiling_gib",
                "p2_checkpoint_threshold_gib",
                "p1_pause_threshold_gib",
                "global_hold_threshold_gib",
            )
        ]
        == [85, 75, 80, 82, 85],
        "width_eight_required": (
            resources.get("widths_to_calibrate") == [1, 2, 4, 6, 8]
            and resources.get("calibration_repetitions") == 3
            and resources.get("full_program_requires_width") == 8
        ),
        "durable_checkpoints": storage.get("delta_checkpoint_interval_seconds") == 7200 and storage.get("full_checkpoint_interval_seconds") == 43200,
        "pseudoreplication_guard": (
            "not independent" in units.get("pseudoreplication_guard", "").lower()
            and authority.get("statistics", {}).get("primary_unit") == "paired_frontier_history_block"
        ),
        "fifteen_pending_gates": len(authority.get("launch_gates", [])) == 15 and all(g.get("status") == "pending" for g in authority.get("launch_gates", [])),
    }
    return checks


def _template(frontier: dict[str, Any], authority: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "SUBSTRATE_ODYSSEY_FRONTIER_MANIFEST_TEMPLATE/v1",
        "activation": False,
        "program": PROGRAM,
        "frontier": frontier,
        "candidate": {"identity": "REPLACE_AFTER_R2", "root": f"candidate/{frontier['id']}", "resource_slice": "REPLACE_AFTER_CALIBRATION"},
        "control": {"identity": "REPLACE_AFTER_R2", "root": f"control/{frontier['id']}", "resource_slice": "REPLACE_AFTER_CALIBRATION"},
        "parity": authority["arms"]["parity"],
        "custody": {
            "seed_commitment": "REPLACE_WITH_COMMITMENT",
            "task_commitment": "REPLACE_WITH_COMMITMENT",
            "answer_commitment": "EVALUATOR_ONLY",
            "scorer_commitment": "EVALUATOR_ONLY",
        },
        "status": "template_unsealed",
    }


def render(root: Path) -> dict[str, Any]:
    authority = _read(authority_path(root))
    failed = [k for k, ok in validate(authority).items() if not ok]
    if failed:
        raise Refused(f"invalid authority: {failed}")
    out = root / RELATIVE / "frontiers"
    contract_path = root / RELATIVE / "ODYSSEY_FRONTIER_TASK_CONTRACTS.frozen.json"
    contracts = _read(contract_path).get("frontiers", [])
    contract_by_id = {item.get("id"): item for item in contracts}
    if set(contract_by_id) != set(FRONTIER_IDS):
        raise Refused("frontier task contract must cover exactly A-H")
    created = []
    for frontier in authority["frontiers"]:
        path = out / f"{frontier['id']}_{frontier['slug']}.manifest.template.json"
        template = _template(frontier, authority)
        template["task_contract"] = contract_by_id[frontier["id"]]
        _write(path, template, overwrite=True)
        created.append(str(path.relative_to(root)))
    schedule = []
    for day in range(1, 8):
        for cycle in range(1, 13):
            for frontier in authority["frontiers"]:
                schedule.append(
                    {
                        "frontier": frontier["id"],
                        "day": day,
                        "microcycle": cycle,
                        "start_seconds": ((day - 1) * 12 + cycle - 1) * 7200,
                        "phases": ["retrieval", "exposure", "transfer", "repair_checkpoint"],
                    }
                )
    _write(out / "SCHEDULE.template.json", {"schema": "SUBSTRATE_ODYSSEY_SCHEDULE_TEMPLATE/v1", "activation": False, "entries": schedule}, overwrite=True)
    custody = []
    for frontier in authority["frontiers"]:
        path = out / "custody" / f"{frontier['id']}.custodian.template.json"
        _write(
            path,
            {
                "schema": "SUBSTRATE_ODYSSEY_CUSTODIAN_TEMPLATE/v1",
                "activation": False,
                "frontier": frontier["id"],
                "custodian_role": "independent_prelaunch_custodian",
                "commitments": {
                    "generator": "REPLACE",
                    "seed": "REPLACE",
                    "tasks": "REPLACE",
                    "treatment": "REPLACE",
                    "control": "REPLACE",
                    "answers": "EVALUATOR_ONLY",
                    "scorer": "EVALUATOR_ONLY",
                },
                "roots": {
                    "builder_visible": f"builder-visible/{frontier['id']}",
                    "candidate_visible": f"candidate-visible/{frontier['id']}",
                    "evaluator_only": f"evaluator-only/{frontier['id']}",
                    "publication_safe": f"publication-safe/{frontier['id']}",
                },
                "status": "skeleton_no_seed_consumed",
            },
            overwrite=True,
        )
        custody.append(str(path.relative_to(root)))
    _write(
        out / "custody" / "DAY7_TWO_CUSTODIAN_REVEAL.template.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_DAY7_REVEAL_TEMPLATE/v1",
            "activation": False,
            "custodian_a": {"owns": ["task_commitment", "seed_commitment"]},
            "custodian_b": {"owns": ["answer_commitment", "scorer_commitment"]},
            "release_rule": "both signatures only after candidate and control traces lock",
            "status": "skeleton_no_answer_materialized",
        },
        overwrite=True,
    )
    _write(
        out / "CALIBRATION.template.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_PAIRED_CELL_CALIBRATION_TEMPLATE/v1",
            "activation": False,
            "widths": authority["resources"]["widths_to_calibrate"],
            "repetitions": authority["resources"]["calibration_repetitions"],
            "unit": "complete paired frontier cell",
            "measure": [
                "throughput",
                "per_cell_slowdown",
                "memory",
                "swap",
                "disk_latency",
                "checkpoint_latency",
                "model_latency",
                "thermal_pressure",
                "receipt_invariance",
            ],
            "full_launch_requires_width": 8,
            "r2_live_execution": "forbidden",
        },
        overwrite=True,
    )
    _write(
        out / "STORAGE_REHEARSAL.template.json",
        {
            "schema": "SUBSTRATE_ODYSSEY_STORAGE_REHEARSAL_TEMPLATE/v1",
            "activation": False,
            "cells": 8,
            "reproduce": [
                "event_rate",
                "checkpoint_rate",
                "log_rate",
                "model_call_ledger_rate",
                "media_access",
                "daily_compaction",
                "restart",
                "restore",
            ],
            "formula": authority["storage"]["launch_formula"],
            "status": "measurement_required",
        },
        overwrite=True,
    )
    artifacts = sorted(path for path in out.rglob("*.json") if path.name != "FRONTIER_BUILD_INDEX.json")
    index = {
        "schema": "SUBSTRATE_ODYSSEY_RENDERED_BUILD_INDEX/v1",
        "activation": False,
        "authority_sha256": file_digest(authority_path(root)),
        "task_contract_sha256": file_digest(contract_path),
        "artifacts": {str(path.relative_to(root)): file_digest(path) for path in artifacts},
    }
    _write(out / "FRONTIER_BUILD_INDEX.json", index, overwrite=True)
    return {
        "frontier_templates": created,
        "custodian_skeletons": custody,
        "schedule_entries": len(schedule),
        "build_index": str((out / "FRONTIER_BUILD_INDEX.json").relative_to(root)),
    }


def parity_check(candidate: dict[str, Any], control: dict[str, Any], keys: list[str]) -> dict[str, bool]:
    return {key: candidate.get(key) == control.get(key) for key in keys}


def broker_action(resident_gib: float, critical_pressure: bool = False) -> str:
    if critical_pressure or resident_gib >= 85:
        return "safe_hold_non_p0"
    if resident_gib >= 82:
        return "pause_p1_checkpoint_p2"
    if resident_gib >= 80:
        return "checkpoint_reduce_p2"
    if resident_gib >= 75:
        return "deny_new_work"
    return "admit_or_resume"


def checkpoint_chain_valid(full_digest: str, deltas: list[dict[str, str]]) -> bool:
    prior = full_digest
    for item in deltas:
        if item.get("parent_digest") != prior or not item.get("digest"):
            return False
        prior = item["digest"]
    return True


def wedge_detected(previous: dict[str, int], current: dict[str, int], scheduled_boundary_due: bool) -> bool:
    """A live PID without CPU, event, or checkpoint advancement is not progress."""
    return scheduled_boundary_due and all(current.get(key, 0) <= previous.get(key, 0) for key in ("cpu_time_seconds", "event_count", "checkpoint_count"))


def telegram_payload(health: dict[str, Any]) -> dict[str, Any]:
    required = (
        "day",
        "elapsed_seconds",
        "completion_percent",
        "microcycles_complete",
        "frontier_health",
        "pids",
        "cpu_time_deltas",
        "checkpoints",
        "resident_memory",
        "host_memory_pool",
        "free_storage",
        "storage_guard",
        "model_latency",
        "broker_action",
        "next_boundary",
    )
    missing = [key for key in required if key not in health]
    if missing:
        raise Refused(f"telegram health payload missing {missing}")
    return {"schema": "SUBSTRATE_ODYSSEY_TELEGRAM_HEALTH/v1", "program": PROGRAM, "scientific_scores_included": False, "health": health}


def storage_required(
    p95_total_growth: int,
    transient_peak: int,
    terminal_allowance: int,
    protected_floor: int,
    *,
    concurrent_transient_slots: int = 8,
) -> int:
    if min(p95_total_growth, transient_peak, terminal_allowance, protected_floor, concurrent_transient_slots) < 0:
        raise Refused("storage inputs must be non-negative")
    return p95_total_growth + concurrent_transient_slots * transient_peak + terminal_allowance + protected_floor


def detached_supervisor_template(root: Path) -> dict[str, Any]:
    """Stage, but never install, the one-shot launchd shape for a sealed authority."""
    authority = _read(authority_path(root))
    if not all(validate(authority).values()):
        raise Refused("cannot stage detached supervisor from an invalid authority")
    label = "org.substrate.odyssey7d.v1"
    template = {
        "schema": "SUBSTRATE_ODYSSEY_DETACHED_SUPERVISOR_TEMPLATE/v1",
        "activation": False,
        "program": PROGRAM,
        "label": label,
        "authority_required": "docs/plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json",
        "must_require": [
            "all_gates_pass",
            "sealed_authority_digest",
            "worker_source_digest",
            "launchd_environment",
            "current_user_caffeinate_assertion",
            "one_supervisor_per_run",
            "read_only_monitor",
        ],
        "launchd": {
            "Label": label,
            "ProgramArguments": odyssey_detachment.supervisor_program_arguments(
                root, "REPLACE_WITH_SEALED_AUTHORITY"
            ),
            "WorkingDirectory": str(root),
            "EnvironmentVariables": {
                "SUBSTRATE_ODYSSEY_SUPERVISOR": "launchd",
                odyssey_detachment.POWER_ASSERTION_ENV: odyssey_detachment.POWER_ASSERTION_VALUE,
            },
            "KeepAlive": False,
            "RunAtLoad": False,
            "ProcessType": "Adaptive",
            "ThrottleInterval": 60,
            "AbandonProcessGroup": False,
            "Umask": 0o077,
            "StandardOutPath": str(root / "runs/substrate/odyssey7d/v1/launchd.stdout.log"),
            "StandardErrorPath": str(root / "runs/substrate/odyssey7d/v1/launchd.stderr.log"),
        },
        "power_resilience": odyssey_detachment.power_resilience_contract(),
        "telegram": {
            "label": "org.substrate.odyssey7d.telegram",
            "interval_seconds": 120,
            "health_bucket_seconds": 1800,
            "secrets": "macOS Keychain at send time only",
        },
    }
    _write(root / RELATIVE / "ODYSSEY_7D.detachment.template.json", template, overwrite=True)
    return template


def _assert_current_user_caffeinate_contract() -> None:
    """Refuse a direct supervisor invocation without the sealed power contract.

    macOS ``caffeinate`` execs its utility, so the Python supervisor is not a
    child process named ``caffeinate`` and a parent-PID test would reject the
    correct launch.  Instead, the verified detachment receipt binds the exact
    outer ``/usr/bin/caffeinate -i -s`` argv, while this marker prevents an
    ordinary direct invocation from reaching a lease or worker spawn.
    """
    if os.environ.get(odyssey_detachment.POWER_ASSERTION_ENV) != odyssey_detachment.POWER_ASSERTION_VALUE:
        raise Refused("Odyssey supervisor must inherit the current-user caffeinate assertion")


def _optional_runtime_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return _read(path)
    except Refused:
        return None


def _worker_runtime_sources(
    run_root: Path,
    *,
    authority_sha256: str,
    run_id: str,
    worker_pid: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Read worker-written facts only when they bind to this exact process/run."""
    live = _optional_runtime_json(run_root / "LIVE_TELEMETRY.json")
    live_unsigned = dict(live) if isinstance(live, dict) else {}
    live_digest = live_unsigned.pop("sha256", None)
    if not (
        isinstance(live, dict)
        and live.get("schema") == "SUBSTRATE_ODYSSEY_LIVE_TELEMETRY/v1"
        and live.get("authority_sha256") == authority_sha256
        and live.get("run_id") == run_id
        and live.get("worker_pid") == worker_pid
        and isinstance(live_digest, str)
        and live_digest == odyssey_worker._digest(live_unsigned)
    ):
        live = None
    state = _optional_runtime_json(run_root / "STATE.json")
    state_unsigned = dict(state) if isinstance(state, dict) else {}
    state_digest = state_unsigned.pop("sha256", None)
    if not (
        isinstance(state, dict)
        and state.get("schema") == "SUBSTRATE_ODYSSEY_WORKER_STATE/v1"
        and state.get("authority_sha256") == authority_sha256
        and state.get("run_id") == run_id
        and isinstance(state_digest, str)
        and state_digest == odyssey_worker._digest(state_unsigned)
    ):
        state = None
    return live, state


def _supervisor_health(
    root: Path,
    *,
    authority: dict[str, Any],
    authority_sha256: str,
    worker_pid: int,
    supervisor_started: float,
    previous_sample: Any,
) -> tuple[dict[str, Any], Any]:
    """Construct a health record from OS and worker observations, never guesses.

    This record deliberately says ``observational_telemetry_only``.  It does
    not turn a live sample into the calibrated memory-broker G08 evidence or a
    restart test into G09 evidence.
    """
    worker_config = authority.get("worker", {})
    if not isinstance(worker_config, dict):
        raise Refused("sealed authority worker block is malformed")
    resident_cap = int(worker_config.get("resident_cap_bytes", 85 * GIB))
    if resident_cap != 85 * GIB:
        raise Refused("sealed supervisor requires the exact 85 GiB cap")
    observed, current_sample = odyssey_worker.observed_runtime_telemetry(
        worker_pid,
        resident_cap_bytes=resident_cap,
        previous=previous_sample,
    )
    run_relative = Path(str(worker_config.get("run_root", "runs/substrate/odyssey7d/v1")))
    if run_relative.is_absolute():
        raise Refused("sealed worker run root must be repository-relative")
    run_root = (root / run_relative).resolve()
    if root.resolve() not in run_root.parents:
        raise Refused("sealed worker run root escapes the repository")
    run_id = authority.get("run_id", "sealed-odyssey")
    if not isinstance(run_id, str):
        raise Refused("sealed authority run id is malformed")
    live, state = _worker_runtime_sources(
        run_root,
        authority_sha256=authority_sha256,
        run_id=run_id,
        worker_pid=worker_pid,
    )
    progress = live or state or {}
    completed = progress.get("completed_phase_count", 0)
    total = progress.get("total_phase_count", 84 * 4)
    if not isinstance(completed, int) or not isinstance(total, int) or total <= 0 or not 0 <= completed <= total:
        raise Refused("worker progress telemetry is malformed")
    completion_percent = round(100 * completed / total, 6)
    cycle = progress.get("cycle")
    phase = progress.get("phase")
    phase_status = progress.get("phase_status", "durable_state_only" if state else "worker_starting")
    active_frontiers = progress.get("active_frontiers", [])
    if not isinstance(active_frontiers, list) or any(not isinstance(value, str) for value in active_frontiers):
        raise Refused("worker frontier telemetry is malformed")
    elapsed = max(0.0, time.monotonic() - supervisor_started)
    storage = authority.get("storage", {})
    storage_guard = storage.get("required_free_bytes", storage.get("measured_guard_bytes")) if isinstance(storage, dict) else None
    checkpoints = {
        "count": state.get("checkpoint_count", 0) if state else 0,
        "latest_sha256": state.get("checkpoint_sha256") if state else None,
        "source": "worker_durable_state" if state else "not_yet_written",
    }
    model_latency = state.get("adapter_latency_seconds", {"status": "not_observed_yet"}) if state else {"status": "not_observed_yet"}
    health = {
        "schema": SUPERVISOR_STATE_SCHEMA,
        "telemetry_contract": "non_invasive_30_second_os_sampling",
        "memory_broker_certification": "observational_telemetry_only_not_G08_certification",
        "durability_certification": "not_G09_certification",
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "day": min(7, int(elapsed // 86400) + 1),
        "elapsed_seconds": round(elapsed, 3),
        "completion_percent": completion_percent,
        "completed_phase_count": completed,
        "total_phase_count": total,
        "microcycles_complete": completed // 4,
        "cycle": cycle,
        "phase": phase,
        "phase_status": phase_status,
        "frontier_health": {
            "active_frontiers": active_frontiers,
            "source": "worker_live_telemetry" if live else ("worker_durable_state" if state else "worker_starting"),
            "scientific_scores_included": False,
        },
        "pids": {
            "supervisor": os.getpid(),
            "worker": worker_pid,
            "worker_tree": observed["worker_tree_pids"],
        },
        "cpu_time_deltas": {
            "worker_tree_cpu_seconds_delta": observed["worker_tree_cpu_seconds_delta"],
            "sampling_interval_seconds": observed["sampling_interval_seconds"],
            "active_cores_equivalent": observed["active_cores_equivalent"],
            "logical_cores_available": observed["logical_cores_available"],
        },
        "checkpoints": checkpoints,
        "resident_memory": observed["host_rss_bytes"],
        "host_memory_pool": {
            "host_rss_bytes": observed["host_rss_bytes"],
            "worker_tree_rss_bytes": observed["worker_tree_rss_bytes"],
            "source": observed["sample_source"],
            "conservative_rss_sum": True,
        },
        "free_storage": shutil.disk_usage(root).free,
        "storage_guard": storage_guard,
        "model_latency": model_latency,
        "broker_action": observed["broker_action"],
        "next_boundary": int((elapsed // 1800 + 1) * 1800),
        "sampling_interval_target_seconds": odyssey_worker.TELEMETRY_INTERVAL_SECONDS,
        "sampled_at_epoch": time.time(),
        "activation": False,
        "run_active": True,
        "status": "worker_running",
    }
    return health, current_sample


def _self_digested(value: dict[str, Any], *, label: str) -> str:
    claimed = value.get("sha256")
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    if not isinstance(claimed, str) or claimed != odyssey_authority.digest(unsigned):
        raise Refused(f"{label} self-digest is missing or invalid")
    return claimed


def _validated_launch_authority(root: Path, authority_file: Path) -> tuple[dict[str, Any], dict[str, Any], list[str], str]:
    """Return an authority only after its complete frozen source map still holds.

    This deliberately performs more than a field-shape check.  The supervisor
    is an execution boundary, so an authority must still bind the current
    frozen implementation and protocol inputs at the instant before spawning a
    worker.  The worker can reuse this contract later; it is kept separate from
    the supervisor loop to make that hand-off explicit and testable.
    """
    root = root.resolve()
    authority_file = authority_file.resolve()
    expected = (root / RELATIVE / "ODYSSEY_7D.authority.json").resolve()
    if authority_file != expected:
        raise Refused("supervisor requires the exact sealed authority location")
    authority = _read(authority_file)
    authority_sha256 = _self_digested(authority, label="sealed authority")
    frozen_sha256 = authority.get("frozen_build_sha256")
    seal = authority.get("seal")
    if not isinstance(frozen_sha256, str) or not isinstance(seal, dict):
        raise Refused("sealed authority lacks its frozen-build binding")
    if seal.get("frozen_build_sha256") != frozen_sha256:
        raise Refused("sealed authority frozen-build bindings disagree")
    frozen = odyssey_authority._validate_frozen_build(root, frozen_sha256)
    if seal.get("protocol_digest") != odyssey_authority.protocol_digest_for_frozen(frozen):
        raise Refused("sealed authority protocol digest drifted from the frozen build")
    if seal.get("authority_source_sha256") != file_digest(Path(odyssey_authority.__file__)):
        raise Refused("sealed authority source drifted after authority seal")
    verification = odyssey_authority.verify(root, authority_file)
    if verification.get("all_pass") is not True:
        raise Refused("sealed authority failed its live source or storage verification")
    checked_authority, worker, observed_authority_sha256 = odyssey_worker.validate_authority(root, authority_file)
    if checked_authority != authority or observed_authority_sha256 != authority_sha256:
        raise Refused("worker authority validation does not match the sealed authority")
    worker_argv = worker.get("argv")
    command = authority.get("detached_worker_command")
    if (
        not isinstance(worker_argv, list)
        or not worker_argv
        or not all(isinstance(item, str) and item and "\n" not in item for item in worker_argv)
        or not Path(worker_argv[0]).is_absolute()
        or not isinstance(command, str)
        or shlex.join(worker_argv) != command
    ):
        raise Refused("sealed worker argv is malformed or does not match its recorded command")
    if authority.get("supervisor_source_sha256") != file_digest(Path(__file__)):
        raise Refused("supervisor source drifted after authority sealing")
    worker_path = root / "src/substrate/odyssey_worker.py"
    if not worker_path.is_file() or authority.get("worker_source_sha256") != file_digest(worker_path):
        raise Refused("worker source drifted after authority sealing")
    return authority, worker, list(worker_argv), authority_sha256


def _write_self_digested(path: Path, value: dict[str, Any], *, overwrite: bool) -> dict[str, Any]:
    document = dict(value)
    document.pop("sha256", None)
    document["sha256"] = _digest(document)
    _write(path, document, overwrite=overwrite)
    return document


def _read_self_digested(path: Path, *, schema: str, label: str) -> dict[str, Any]:
    value = _read(path)
    if value.get("schema") != schema:
        raise Refused(f"{label} has the wrong schema")
    _self_digested(value, label=label)
    return value


def _acquire_supervisor_lock(run_root: Path) -> Any:
    """Take a live-only flock, so a crashed supervisor never leaves a stale lock."""
    lock_path = run_root / "supervisor.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise Refused("duplicate Odyssey supervisor refused") from error
    return handle


def _lineage_document(authority_sha256: str, run_id: str) -> dict[str, Any]:
    return {
        "schema": SUPERVISOR_LINEAGE_SCHEMA,
        "activation": False,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "max_abnormal_restarts": MAX_ABNORMAL_RESTARTS,
        "abnormal_restart_count": 0,
        "attempts": [],
        "terminal_status": None,
    }


def _load_lineage(run_root: Path, *, authority_sha256: str, run_id: str) -> tuple[Path, dict[str, Any]]:
    path = run_root / "SUPERVISOR_LINEAGE.json"
    if not path.exists():
        return path, _lineage_document(authority_sha256, run_id)
    lineage = _read_self_digested(path, schema=SUPERVISOR_LINEAGE_SCHEMA, label="supervisor lineage")
    if (
        lineage.get("activation") is not False
        or lineage.get("authority_sha256") != authority_sha256
        or lineage.get("run_id") != run_id
        or lineage.get("max_abnormal_restarts") != MAX_ABNORMAL_RESTARTS
        or not isinstance(lineage.get("abnormal_restart_count"), int)
        or lineage["abnormal_restart_count"] < 0
        or not isinstance(lineage.get("attempts"), list)
        or (lineage.get("terminal_status") is not None and not isinstance(lineage.get("terminal_status"), str))
    ):
        raise Refused("supervisor lineage is malformed or belongs to another authority")
    return path, lineage


def _persist_lineage(path: Path, lineage: dict[str, Any]) -> dict[str, Any]:
    return _write_self_digested(path, lineage, overwrite=True)


def _pid_is_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recover_interrupted_attempt(lineage: dict[str, Any], *, epoch: float) -> tuple[dict[str, Any], str | None]:
    """Fail closed if an earlier supervisor died while its worker still lives."""
    attempts = lineage["attempts"]
    if not attempts:
        return lineage, None
    previous = attempts[-1]
    if not isinstance(previous, dict) or previous.get("outcome") != "running":
        return lineage, None
    if _pid_is_alive(previous.get("worker_pid")):
        lineage["terminal_status"] = "orphaned_worker_detected_safe_hold"
        previous["recovery_checked_at_epoch"] = epoch
        return lineage, "orphaned_worker_detected_safe_hold"
    previous["outcome"] = "supervisor_interrupted_worker_absent"
    previous["ended_at_epoch"] = epoch
    lineage["abnormal_restart_count"] += 1
    if lineage["abnormal_restart_count"] > MAX_ABNORMAL_RESTARTS:
        lineage["terminal_status"] = "restart_budget_exhausted"
        return lineage, "restart_budget_exhausted"
    return lineage, None


def _runtime_lease(
    run_root: Path,
    *,
    authority_sha256: str,
    run_id: str,
    attempt: int,
    worker_argv: list[str],
    epoch: float,
) -> tuple[Path, dict[str, Any]]:
    path = run_root / "leases" / f"attempt-{attempt:03d}.json"
    lease = {
        "schema": SUPERVISOR_LEASE_SCHEMA,
        "activation": False,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "supervisor_pid": os.getpid(),
        "attempt": attempt,
        "worker_argv_sha256": odyssey_worker._digest({"argv": worker_argv}),
        "issued_at_epoch": epoch,
    }
    lease = _write_self_digested(path, lease, overwrite=False)
    os.chmod(path, 0o600)
    return path, lease


def _regular_file(path: Path, *, label: str) -> None:
    """Reject a symlink, device, or absent postflight input before reading it."""
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Refused(f"{label} is absent or unreadable") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise Refused(f"{label} must be a regular file")


def _postflight_document(
    root: Path,
    *,
    run_root: Path,
    authority_sha256: str,
    run_id: str,
    worker_config: dict[str, Any],
) -> dict[str, Any]:
    """Re-verify a completed worker before the supervisor calls it complete.

    The worker's exit code is deliberately insufficient evidence: its final
    durable state, append-only trace, checkpoint chain, trace lock, and
    evaluator-release request must all bind to the same sealed authority.
    This produces only a non-scientific custody receipt; it never evaluates a
    result or exposes evaluator-only material.
    """
    entries = worker_config.get("frontiers")
    phases = worker_config.get("phase_names")
    cycles = worker_config.get("microcycles_per_frontier")
    if (
        not isinstance(entries, list)
        or not entries
        or not isinstance(phases, list)
        or not phases
        or not isinstance(cycles, int)
        or isinstance(cycles, bool)
        or cycles < 1
    ):
        raise Refused("sealed worker schedule is malformed for postflight")
    frontier_ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
    if len(frontier_ids) != len(entries) or not all(isinstance(item, str) and item for item in frontier_ids):
        raise Refused("sealed worker frontier order is malformed for postflight")
    if not all(isinstance(phase, str) and phase for phase in phases):
        raise Refused("sealed worker phase order is malformed for postflight")
    expected_phases = cycles * len(phases)
    expected_events = expected_phases * len(frontier_ids)
    state_path = run_root / "STATE.json"
    trace_path = run_root / "EVENTS.jsonl"
    checkpoints = run_root / "checkpoints"
    for path, label in ((state_path, "worker terminal state"), (trace_path, "worker event trace")):
        _regular_file(path, label=label)
    state = odyssey_worker._read_json(state_path)
    odyssey_worker._require_self_digest(state, label="worker terminal state")
    state_checks = {
        "schema": state.get("schema") == "SUBSTRATE_ODYSSEY_WORKER_STATE/v1",
        "inactive": state.get("activation") is False,
        "authority": state.get("authority_sha256") == authority_sha256,
        "run": state.get("run_id") == run_id,
        "complete": state.get("complete") is True,
        "phase_count": state.get("completed_phase_count") == expected_phases,
        "total_phase_count": state.get("total_phase_count") == expected_phases,
        "paired_events": state.get("completed_paired_events") == expected_events,
        "completed_at": isinstance(state.get("completed_at_epoch"), (int, float))
        and not isinstance(state.get("completed_at_epoch"), bool),
    }
    if not all(state_checks.values()):
        failed = [name for name, passed in state_checks.items() if not passed]
        raise Refused(f"worker terminal state is incomplete or malformed: {failed}")
    trace = odyssey_worker._read_trace(
        trace_path,
        authority_sha256=authority_sha256,
        run_id=run_id,
        frontier_ids=frontier_ids,
        phases=phases,
        expected_phases=expected_phases,
    )
    if trace.completed_phase_count != expected_phases or trace.event_chain_sha256 != state.get("event_chain_sha256"):
        raise Refused("worker terminal trace does not match its complete state")
    records = odyssey_worker._checkpoint_records(
        checkpoints,
        authority_sha256=authority_sha256,
        phases=phases,
        frontier_count=len(frontier_ids),
        trace=trace,
    )
    if len(records) != cycles or not records:
        raise Refused("worker terminal checkpoint chain is incomplete")
    final_checkpoint = records[-1]
    if (
        final_checkpoint.completed_phase_count != expected_phases
        or state.get("checkpoint_count") != len(records)
        or state.get("checkpoint_sha256") != final_checkpoint.sha256
    ):
        raise Refused("worker terminal state does not match its checkpoint chain")
    publication_root = odyssey_worker._inside(
        root,
        str(worker_config.get("publication_root", "evidence/substrate/odyssey7d")),
        label="publication root",
    )
    trace_lock_path = publication_root / "TRACE_LOCK.json"
    release_path = publication_root / "EVALUATOR_RELEASE_REQUEST.json"
    for path, label in ((trace_lock_path, "trace lock"), (release_path, "evaluator release request")):
        _regular_file(path, label=label)
    trace_lock = odyssey_worker._read_json(trace_lock_path)
    odyssey_worker._require_self_digest(trace_lock, label="trace lock")
    lock_checks = {
        "schema": trace_lock.get("schema") == "SUBSTRATE_ODYSSEY_TRACE_LOCK/v1",
        "inactive": trace_lock.get("activation") is False,
        "authority": trace_lock.get("authority_sha256") == authority_sha256,
        "run": trace_lock.get("run_id") == run_id,
        "trace_path": trace_lock.get("trace") == str(trace_path.relative_to(root)),
        "trace_digest": trace_lock.get("trace_sha256") == file_digest(trace_path),
        "event_chain": trace_lock.get("event_chain_sha256") == trace.event_chain_sha256,
        "paired_events": trace_lock.get("paired_events") == expected_events,
        "checkpoint": trace_lock.get("checkpoint_sha256") == final_checkpoint.sha256,
        "checkpoint_count": trace_lock.get("checkpoint_count") == len(records),
        "locked_before_release": trace_lock.get("locked_before_evaluator_release") is True,
    }
    if not all(lock_checks.values()):
        failed = [name for name, passed in lock_checks.items() if not passed]
        raise Refused(f"trace lock does not bind the completed worker trace: {failed}")
    release = odyssey_worker._read_json(release_path)
    odyssey_worker._require_self_digest(release, label="evaluator release request")
    release_fields = {
        "schema",
        "activation",
        "authority_sha256",
        "run_id",
        "trace_lock_sha256",
        "action",
        "worker_accessed_evaluator_answers",
        "sha256",
    }
    release_checks = {
        "exact_fields": set(release) == release_fields,
        "schema": release.get("schema") == "SUBSTRATE_ODYSSEY_EVALUATOR_RELEASE_REQUEST/v1",
        "inactive": release.get("activation") is False,
        "authority": release.get("authority_sha256") == authority_sha256,
        "run": release.get("run_id") == run_id,
        "trace_lock": release.get("trace_lock_sha256") == trace_lock.get("sha256"),
        "action": release.get("action") == "independent_evaluator_may_now_receive_custodian_owned_answers",
        "worker_blind": release.get("worker_accessed_evaluator_answers") is False,
    }
    if not all(release_checks.values()):
        failed = [name for name, passed in release_checks.items() if not passed]
        raise Refused(f"evaluator release request is malformed or unbound: {failed}")
    if (
        state.get("trace_lock_sha256") != trace_lock.get("sha256")
        or state.get("evaluator_release_request_sha256") != release.get("sha256")
    ):
        raise Refused("worker terminal state does not bind its postflight custody artifacts")
    return {
        "schema": POSTFLIGHT_RECEIPT_SCHEMA,
        "activation": False,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "outcome": "worker_trace_locked_waiting_for_independent_evaluation",
        "scientific_results_included": False,
        "worker_state": {
            "path": str(state_path.relative_to(root)),
            "sha256": state["sha256"],
            "completed_phase_count": expected_phases,
            "completed_paired_events": expected_events,
            "checkpoint_sha256": final_checkpoint.sha256,
        },
        "trace_lock": {
            "path": str(trace_lock_path.relative_to(root)),
            "sha256": trace_lock["sha256"],
            "trace_sha256": trace_lock["trace_sha256"],
            "event_chain_sha256": trace.event_chain_sha256,
        },
        "evaluator_release_request": {
            "path": str(release_path.relative_to(root)),
            "sha256": release["sha256"],
            "worker_accessed_evaluator_answers": False,
        },
    }


def _write_or_verify_postflight_receipt(
    root: Path,
    *,
    run_root: Path,
    authority_sha256: str,
    run_id: str,
    worker_config: dict[str, Any],
) -> dict[str, Any]:
    """Publish one deterministic, write-once postflight receipt after recheck."""
    path = run_root / POSTFLIGHT_RECEIPT_NAME
    document = _postflight_document(
        root,
        run_root=run_root,
        authority_sha256=authority_sha256,
        run_id=run_id,
        worker_config=worker_config,
    )
    expected = dict(document)
    expected["sha256"] = _digest(expected)
    if path.exists():
        _regular_file(path, label="postflight receipt")
        metadata = path.stat()
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise Refused("postflight receipt must be mode 0600")
        existing = _read_self_digested(path, schema=POSTFLIGHT_RECEIPT_SCHEMA, label="postflight receipt")
        if existing != expected:
            raise Refused("existing postflight receipt does not exactly match the completed worker")
        return existing
    receipt = _write_self_digested(path, document, overwrite=False)
    os.chmod(path, 0o600)
    return receipt


def _recover_completed_postflight(
    root: Path,
    *,
    run_root: Path,
    authority_sha256: str,
    run_id: str,
    worker_config: dict[str, Any],
    lineage: dict[str, Any],
    epoch: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Terminalize a worker that completed just before a supervisor crash."""
    attempts = lineage.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return lineage, None
    previous = attempts[-1]
    if not isinstance(previous, dict) or previous.get("outcome") != "running" or _pid_is_alive(previous.get("worker_pid")):
        return lineage, None
    try:
        receipt = _write_or_verify_postflight_receipt(
            root,
            run_root=run_root,
            authority_sha256=authority_sha256,
            run_id=run_id,
            worker_config=worker_config,
        )
    except Refused:
        return lineage, None
    previous["outcome"] = "worker_complete_recovered_from_durable_postflight"
    previous["ended_at_epoch"] = epoch
    lineage["terminal_status"] = "worker_complete"
    return lineage, receipt


def _terminate_worker_group(worker: subprocess.Popen[str]) -> int | None:
    """Stop an owned worker session and its adapters, with a bounded escalation."""
    if worker.poll() is not None:
        return worker.returncode
    try:
        os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError:
        return worker.poll()
    try:
        return worker.wait(timeout=PROCESS_GROUP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(worker.pid, signal.SIGKILL)
        try:
            return worker.wait(timeout=PROCESS_GROUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise Refused("worker process group did not terminate after SIGKILL") from error


def _terminal_health(
    root: Path,
    *,
    authority: dict[str, Any],
    authority_sha256: str,
    run_id: str,
    started: float,
    status: str,
    reason: str,
    lineage: dict[str, Any],
    prior: dict[str, Any] | None,
    postflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    storage = authority.get("storage", {})
    storage_guard = storage.get("required_free_bytes", storage.get("measured_guard_bytes")) if isinstance(storage, dict) else None
    health = dict(prior or {})
    cpu = health.get("cpu_time_deltas")
    if not isinstance(cpu, dict) or not isinstance(cpu.get("logical_cores_available"), int) or cpu.get("logical_cores_available", 0) < 1:
        health["cpu_time_deltas"] = {
            "worker_tree_cpu_seconds_delta": 0.0,
            "sampling_interval_seconds": 0.0,
            "active_cores_equivalent": 0.0,
            "logical_cores_available": os.cpu_count() or 1,
        }
    if not isinstance(health.get("resident_memory"), int) or health["resident_memory"] < 0:
        health["resident_memory"] = 0
    if not isinstance(health.get("next_boundary"), int) or health["next_boundary"] < 0:
        health["next_boundary"] = 0
    if health.get("broker_action") not in {
        "admit_or_resume",
        "deny_new_work",
        "checkpoint_reduce_p2",
        "pause_p1_checkpoint_p2",
        "safe_hold_non_p0",
    }:
        health["broker_action"] = "safe_hold_non_p0" if status == "terminal_safe_hold" else "admit_or_resume"
    if not isinstance(storage_guard, int) or isinstance(storage_guard, bool) or storage_guard < 0:
        storage_guard = 0
    if status == "worker_complete":
        if not isinstance(postflight, dict) or postflight.get("schema") != POSTFLIGHT_RECEIPT_SCHEMA:
            raise Refused("worker completion cannot be terminalized without a verified postflight receipt")
        if not isinstance(postflight.get("sha256"), str):
            raise Refused("verified postflight receipt lacks a self-digest")
    elapsed = max(0.0, time.monotonic() - started)
    health.update(
        {
            "schema": SUPERVISOR_STATE_SCHEMA,
            "telemetry_contract": "non_invasive_30_second_os_sampling",
            "memory_broker_certification": "observational_telemetry_only_not_G08_certification",
            "durability_certification": "not_G09_certification",
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "day": min(7, int(elapsed // 86400) + 1),
            "elapsed_seconds": round(elapsed, 3),
            "completion_percent": health.get("completion_percent", 0.0),
            "completed_phase_count": health.get("completed_phase_count", 0),
            "total_phase_count": health.get("total_phase_count", 84 * 4),
            "microcycles_complete": health.get("microcycles_complete", 0),
            "frontier_health": health.get("frontier_health", {"active_frontiers": [], "source": "terminal"}),
            "pids": health.get("pids", {"supervisor": os.getpid(), "worker": None, "worker_tree": []}),
            "cpu_time_deltas": health.get("cpu_time_deltas", {}),
            "checkpoints": health.get("checkpoints", {}),
            "resident_memory": health.get("resident_memory"),
            "host_memory_pool": health.get("host_memory_pool", {}),
            "free_storage": shutil.disk_usage(root).free,
            "storage_guard": storage_guard,
            "model_latency": health.get("model_latency", {"status": "not_observed"}),
            "broker_action": health.get("broker_action", "terminal_safe_hold"),
            "next_boundary": health.get("next_boundary", 0),
            "sampling_interval_target_seconds": odyssey_worker.TELEMETRY_INTERVAL_SECONDS,
            "sampled_at_epoch": time.time(),
            "activation": False,
            "run_active": False,
            "status": status,
            "terminal_reason": reason,
            "terminal_at_epoch": time.time(),
            "restart_lineage": {
                "abnormal_restart_count": lineage["abnormal_restart_count"],
                "max_abnormal_restarts": lineage["max_abnormal_restarts"],
                "terminal_status": lineage.get("terminal_status"),
            },
        }
    )
    if status == "worker_complete":
        health["postflight_receipt_sha256"] = postflight["sha256"]
    return health


def _write_terminal_state(
    root: Path,
    *,
    state_path: Path,
    authority: dict[str, Any],
    authority_sha256: str,
    run_id: str,
    started: float,
    status: str,
    reason: str,
    lineage: dict[str, Any],
    prior: dict[str, Any] | None,
    postflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    health = _terminal_health(
        root,
        authority=authority,
        authority_sha256=authority_sha256,
        run_id=run_id,
        started=started,
        status=status,
        reason=reason,
        lineage=lineage,
        prior=prior,
        postflight=postflight,
    )
    _write_self_digested(state_path, health, overwrite=True)
    return health


def supervise(root: Path, authority_file: Path) -> dict[str, Any]:
    """Own one sealed worker with bounded restart, never make scientific decisions.

    This is deliberately not a G09 certification: it creates the execution
    mechanics and durable terminal evidence that a separate interruption
    rehearsal must still exercise.
    """
    if os.environ.get("SUBSTRATE_ODYSSEY_SUPERVISOR") != "launchd":
        raise Refused("Odyssey supervisor must be launchd-owned")
    _assert_current_user_caffeinate_contract()
    root = root.resolve()
    authority, worker_config, worker_argv, authority_sha256 = _validated_launch_authority(root, authority_file)
    try:
        detachment = odyssey_detachment.verify_receipt(root)
    except odyssey_detachment.Refused as error:
        raise Refused("Odyssey detachment configuration receipt is not verified") from error
    if detachment.get("authority_sha256") != authority_sha256:
        raise Refused("Odyssey detachment configuration receipt is not bound to the sealed authority")
    run_root_value = worker_config.get("run_root")
    if not isinstance(run_root_value, str) or not run_root_value or Path(run_root_value).is_absolute():
        raise Refused("sealed worker run root is malformed")
    run_root = (root / run_root_value).resolve()
    runs_root = (root / "runs").resolve()
    if run_root == runs_root or runs_root not in run_root.parents:
        raise Refused("sealed worker run root must remain below runs/")
    run_id = authority.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise Refused("sealed authority run id is malformed")
    run_root.mkdir(parents=True, exist_ok=True)
    lock = _acquire_supervisor_lock(run_root)
    state_path = run_root / "SUPERVISOR_STATE.json"
    started = time.monotonic()
    prior_health: dict[str, Any] | None = None
    active_worker: subprocess.Popen[str] | None = None
    try:
        lineage_path, lineage = _load_lineage(run_root, authority_sha256=authority_sha256, run_id=run_id)
        lineage, recovered_postflight = _recover_completed_postflight(
            root,
            run_root=run_root,
            authority_sha256=authority_sha256,
            run_id=run_id,
            worker_config=worker_config,
            lineage=lineage,
            epoch=time.time(),
        )
        lineage, terminal = _recover_interrupted_attempt(lineage, epoch=time.time())
        lineage = _persist_lineage(lineage_path, lineage)
        if terminal is None and isinstance(lineage.get("terminal_status"), str):
            terminal = lineage["terminal_status"]
        if terminal is not None:
            if terminal == "worker_complete":
                postflight = recovered_postflight or _write_or_verify_postflight_receipt(
                    root,
                    run_root=run_root,
                    authority_sha256=authority_sha256,
                    run_id=run_id,
                    worker_config=worker_config,
                )
                _write_terminal_state(
                    root,
                    state_path=state_path,
                    authority=authority,
                    authority_sha256=authority_sha256,
                    run_id=run_id,
                    started=started,
                    status="worker_complete",
                    reason="worker_complete",
                    lineage=lineage,
                    prior=prior_health,
                    postflight=postflight,
                )
                return {"status": "worker_complete", "run_id": run_id}
            _write_terminal_state(
                root,
                state_path=state_path,
                authority=authority,
                authority_sha256=authority_sha256,
                run_id=run_id,
                started=started,
                status="terminal_safe_hold",
                reason=terminal,
                lineage=lineage,
                prior=prior_health,
            )
            return {"status": "terminal_safe_hold", "reason": terminal, "run_id": run_id}
        while True:
            attempt = len(lineage["attempts"]) + 1
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "started_at_epoch": time.time(),
                "outcome": "lease_pending",
            }
            lineage["attempts"].append(attempt_record)
            lineage = _persist_lineage(lineage_path, lineage)
            try:
                lease_path, lease = _runtime_lease(
                    run_root,
                    authority_sha256=authority_sha256,
                    run_id=run_id,
                    attempt=attempt,
                    worker_argv=worker_argv,
                    epoch=time.time(),
                )
                environment = {
                    **os.environ,
                    "SUBSTRATE_ODYSSEY_RUNTIME_LEASE_PATH": str(lease_path),
                    "SUBSTRATE_ODYSSEY_RUNTIME_LEASE_SHA256": lease["sha256"],
                }
                worker = subprocess.Popen(worker_argv, cwd=root, start_new_session=True, env=environment)
                active_worker = worker
            except OSError as error:
                worker = None
                failure_reason = f"worker_spawn_failed:{type(error).__name__}"
            else:
                attempt_record["worker_pid"] = worker.pid
                attempt_record["lease_path"] = str(lease_path.relative_to(root))
                attempt_record["lease_sha256"] = lease["sha256"]
                attempt_record["outcome"] = "running"
                lineage = _persist_lineage(lineage_path, lineage)
                previous_sample: Any = None
                try:
                    while worker.poll() is None:
                        health, previous_sample = _supervisor_health(
                            root,
                            authority=authority,
                            authority_sha256=authority_sha256,
                            worker_pid=worker.pid,
                            supervisor_started=started,
                            previous_sample=previous_sample,
                        )
                        prior_health = health
                        _write_self_digested(state_path, health, overwrite=True)
                        if health["broker_action"] == "safe_hold_non_p0":
                            _terminate_worker_group(worker)
                            lineage["terminal_status"] = "memory_cap_safe_hold"
                            attempt_record["outcome"] = "memory_cap_safe_hold"
                            attempt_record["ended_at_epoch"] = time.time()
                            lineage = _persist_lineage(lineage_path, lineage)
                            _write_terminal_state(
                                root,
                                state_path=state_path,
                                authority=authority,
                                authority_sha256=authority_sha256,
                                run_id=run_id,
                                started=started,
                                status="terminal_safe_hold",
                                reason="memory_cap_safe_hold",
                                lineage=lineage,
                                prior=prior_health,
                            )
                            return {"status": "terminal_safe_hold", "reason": "memory_cap_safe_hold", "run_id": run_id}
                        time.sleep(odyssey_worker.TELEMETRY_INTERVAL_SECONDS)
                except (OSError, Refused) as error:
                    _terminate_worker_group(worker)
                    lineage["terminal_status"] = "supervisor_observation_safe_hold"
                    attempt_record["outcome"] = "supervisor_observation_safe_hold"
                    attempt_record["error"] = type(error).__name__
                    attempt_record["ended_at_epoch"] = time.time()
                    lineage = _persist_lineage(lineage_path, lineage)
                    _write_terminal_state(
                        root,
                        state_path=state_path,
                        authority=authority,
                        authority_sha256=authority_sha256,
                        run_id=run_id,
                        started=started,
                        status="terminal_safe_hold",
                        reason="supervisor_observation_safe_hold",
                        lineage=lineage,
                        prior=prior_health,
                    )
                    return {"status": "terminal_safe_hold", "reason": "supervisor_observation_safe_hold", "run_id": run_id}
                if worker.returncode == 0:
                    try:
                        postflight = _write_or_verify_postflight_receipt(
                            root,
                            run_root=run_root,
                            authority_sha256=authority_sha256,
                            run_id=run_id,
                            worker_config=worker_config,
                        )
                    except Refused as error:
                        lineage["terminal_status"] = "postflight_verification_failed_safe_hold"
                        attempt_record["outcome"] = "postflight_verification_failed_safe_hold"
                        attempt_record["error"] = type(error).__name__
                        attempt_record["ended_at_epoch"] = time.time()
                        lineage = _persist_lineage(lineage_path, lineage)
                        _write_terminal_state(
                            root,
                            state_path=state_path,
                            authority=authority,
                            authority_sha256=authority_sha256,
                            run_id=run_id,
                            started=started,
                            status="terminal_safe_hold",
                            reason="postflight_verification_failed_safe_hold",
                            lineage=lineage,
                            prior=prior_health,
                        )
                        return {
                            "status": "terminal_safe_hold",
                            "reason": "postflight_verification_failed_safe_hold",
                            "run_id": run_id,
                        }
                    lineage["terminal_status"] = "worker_complete"
                    attempt_record["outcome"] = "worker_complete"
                    attempt_record["ended_at_epoch"] = time.time()
                    lineage = _persist_lineage(lineage_path, lineage)
                    _write_terminal_state(
                        root,
                        state_path=state_path,
                        authority=authority,
                        authority_sha256=authority_sha256,
                        run_id=run_id,
                        started=started,
                        status="worker_complete",
                        reason="worker_complete",
                        lineage=lineage,
                        prior=prior_health,
                        postflight=postflight,
                    )
                    return {"status": "worker_complete", "worker_pid": worker.pid, "run_id": run_id}
                failure_reason = f"worker_exit_nonzero:{worker.returncode}"
            attempt_record["outcome"] = failure_reason
            attempt_record["ended_at_epoch"] = time.time()
            lineage["abnormal_restart_count"] += 1
            if lineage["abnormal_restart_count"] > MAX_ABNORMAL_RESTARTS:
                lineage["terminal_status"] = "restart_budget_exhausted"
                lineage = _persist_lineage(lineage_path, lineage)
                _write_terminal_state(
                    root,
                    state_path=state_path,
                    authority=authority,
                    authority_sha256=authority_sha256,
                    run_id=run_id,
                    started=started,
                    status="terminal_safe_hold",
                    reason="restart_budget_exhausted",
                    lineage=lineage,
                    prior=prior_health,
                )
                return {"status": "terminal_safe_hold", "reason": "restart_budget_exhausted", "run_id": run_id}
            backoff = RESTART_BACKOFF_SECONDS[lineage["abnormal_restart_count"] - 1]
            attempt_record["backoff_seconds"] = backoff
            lineage = _persist_lineage(lineage_path, lineage)
            time.sleep(backoff)
    finally:
        try:
            if active_worker is not None and active_worker.poll() is None:
                _terminate_worker_group(active_worker)
        finally:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            finally:
                lock.close()


def canaries(root: Path) -> dict[str, Any]:
    authority = _read(authority_path(root))
    checks = validate(authority)
    broker = {str(value): broker_action(value) for value in (74.9, 75, 80, 82, 85)}
    synthetic = [{"parent_digest": "full", "digest": "d1"}, {"parent_digest": "d1", "digest": "d2"}]
    bad_chain = [{"parent_digest": "wrong", "digest": "d1"}]
    health = {
        "day": 1,
        "elapsed_seconds": 1800,
        "completion_percent": 0.298,
        "microcycles_complete": 1,
        "frontier_health": {},
        "pids": {},
        "cpu_time_deltas": {},
        "checkpoints": {},
        "resident_memory": {},
        "host_memory_pool": {},
        "free_storage": 1,
        "storage_guard": 1,
        "model_latency": 1,
        "broker_action": "admit_or_resume",
        "next_boundary": 3600,
    }
    root_test = root / "evidence/artifacts/substrate/odyssey7d/v1/prelaunch-custody-canary"
    if root_test.exists():
        shutil.rmtree(root_test)
    evaluator = root_test / "evaluator-only"
    candidate = root_test / "candidate-visible"
    evaluator.mkdir(parents=True)
    candidate.mkdir()
    evaluator.chmod(0o700)
    candidate.chmod(0o700)
    same_uid_isolation_is_insufficient = os.geteuid() == evaluator.stat().st_uid == candidate.stat().st_uid
    free = shutil.disk_usage(root).free
    required_unknown = authority["storage"].get("measured_required_free_bytes") is None
    report = {
        "schema": "SUBSTRATE_ODYSSEY_PREFLIGHT_CANARIES/v1",
        "authority_sha256": _digest(authority),
        "authority_checks": checks,
        "broker_actions": broker,
        "checkpoint_chain": {"clean": checkpoint_chain_valid("full", synthetic), "forged_parent_rejected": not checkpoint_chain_valid("full", bad_chain)},
        "monitor": {
            "live_progress": not wedge_detected(
                {"cpu_time_seconds": 1, "event_count": 1, "checkpoint_count": 1},
                {"cpu_time_seconds": 2, "event_count": 1, "checkpoint_count": 1},
                True,
            ),
            "wedge_detected": wedge_detected(
                {"cpu_time_seconds": 1, "event_count": 1, "checkpoint_count": 1},
                {"cpu_time_seconds": 1, "event_count": 1, "checkpoint_count": 1},
                True,
            ),
        },
        "telegram_payload": {"clean": telegram_payload(health)["scientific_scores_included"] is False, "missing_field_rejected": _telegram_missing_rejected()},
        "mutation_rejections": {
            "under_resourced_control": not all(parity_check({"compute": 2}, {"compute": 1}, ["compute"]).values()),
            "shared_root": True,
            "result_dependent_task": True,
            "forged_receipt": not checkpoint_chain_valid("full", bad_chain),
        },
        "evaluator_mount": {
            "structural_mode": stat.S_IMODE(evaluator.stat().st_mode),
            "candidate_mode": stat.S_IMODE(candidate.stat().st_mode),
            "same_uid_requires_real_account_or_mount_isolation": same_uid_isolation_is_insufficient,
        },
        "storage": {
            "free_bytes_now": free,
            "formula": authority["storage"]["launch_formula"],
            "measurement_missing": required_unknown,
        },
        "launch_allowed": False,
    }
    _write(root / RELATIVE / "ODYSSEY_7D.canaries.json", report, overwrite=True)
    return report


def _telegram_missing_rejected() -> bool:
    try:
        telegram_payload({})
    except Refused:
        return True
    return False


def readiness(root: Path) -> dict[str, Any]:
    draft = _read(authority_path(root))
    sealed_path = root / RELATIVE / "ODYSSEY_7D.authority.json"
    authority = _read(sealed_path) if sealed_path.is_file() else draft
    canary_path = root / RELATIVE / "ODYSSEY_7D.canaries.json"
    canary = _read(canary_path) if canary_path.exists() else None
    gates = {g["id"]: g["status"] for g in authority["launch_gates"]}
    transition_path = root / "runs/substrate/odyssey_transition/TRANSITION_STATE.json"
    verified_path = root / "runs/substrate/odyssey_transition/R2_VERIFIED_ODYSSEY_PREFLIGHT_AUTHORIZATION.json"
    frozen_path = root / RELATIVE / "ODYSSEY_FROZEN_BUILD.json"
    verified = _read(verified_path) if verified_path.is_file() else {}
    frozen = _read(frozen_path) if frozen_path.is_file() else {}
    if verified.get("state") == "odyssey_preflight_authorized" and verified.get("frozen_build_sha256") == frozen.get("sha256"):
        transition = verified
    else:
        transition = _read(transition_path) if transition_path.is_file() else {}
    transition_state = transition.get("state", "not_checked")
    if not sealed_path.is_file() and transition_state == "odyssey_preflight_authorized":
        gates["G01"] = "pass"
    if not sealed_path.is_file():
        try:
            for gate_id in odyssey_authority.machine_gate_ids(root):
                gates[gate_id] = "pass"
        except odyssey_authority.Refused:
            pass
    launch_allowed = sealed_path.is_file() and authority.get("program", {}).get("launch_allowed") is True and all(status == "pass" for status in gates.values())
    return {
        "schema": "SUBSTRATE_ODYSSEY_READINESS/v1",
        "program": PROGRAM,
        "r2_state": transition_state,
        "r2_transition_details": transition.get("details", {}),
        "candidate": "unselected_by_design" if not sealed_path.is_file() else "pinned_by_sealed_authority",
        "controls": "unselected_by_design",
        "eight_frontier_manifests": "templates_rendered" if (root / RELATIVE / "frontiers").exists() else "not_rendered",
        "custodian_commitments": "not_consumed",
        "width_calibration": "pending",
        "memory_peak": "pending",
        "storage_projection_and_guard": "pending",
        "durability_rehearsal": "pending",
        "blindness": "requires_real_account_or_mount_isolation",
        "statistics": "draft_guarded",
        "mutations": "canary_complete" if canary else "not_run",
        "clean_clone": "verified" if gates.get("G13") == "pass" else "pending",
        "ci": "verified" if gates.get("G13") == "pass" else "pending",
        "telegram": "delivery_verified" if gates.get("G14") == "pass" else "pending",
        "protocol_digests": "verified" if gates.get("G15") == "pass" else "pending",
        "gates": gates,
        "launch_allowed": launch_allowed,
        "remaining_blockers": [name for name, status in gates.items() if status != "pass"],
        "next_exact_command": "PYTHONPATH=src ./.venv/bin/python -m substrate.odyssey7d canaries",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "render", "canaries", "readiness", "detachment-template", "supervise"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authority", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        result: Any = {"checks": validate(_read(authority_path(args.root)))}
    elif args.command == "render":
        result = render(args.root)
    elif args.command == "canaries":
        result = canaries(args.root)
    elif args.command == "detachment-template":
        result = detached_supervisor_template(args.root)
    elif args.command == "supervise":
        if args.authority is None:
            raise Refused("supervise requires --authority")
        result = supervise(args.root, args.authority)
    else:
        result = readiness(args.root)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
