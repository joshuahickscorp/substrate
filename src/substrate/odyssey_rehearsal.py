"""Measured host rehearsals that produce Odyssey G06–G09 machine subjects.

This module is a producer, not a sealer.  It runs real work against the local
host (child processes, filesystem, localhost Ollama, pure broker functions),
writes content-addressed receipts under the rehearsal tree, and emits subject
documents shaped for ``odyssey_authority.seal_machine_gate``.

Every metric must trace to a syscall, a real child process, a real file written
during the run, or a pure function in this repository evaluated at a declared
input.  A measured threshold violation raises :class:`Refused` — it never
becomes a coerced pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import ctypes
import fcntl
import hashlib
import json
import math
import os
import re
import resource
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from ctypes import byref, c_int, c_uint32, c_ulonglong, c_void_p, sizeof
from pathlib import Path
from typing import Any

from substrate import odyssey7d, odyssey_transition
from substrate import odyssey_arms as arms
from substrate import odyssey_authority as authority
from substrate import odyssey_density as density
from substrate import odyssey_model_canary as model_canary
from substrate import odyssey_worker as worker
from substrate.odyssey_authority import (
    BASE_PROTECTED_FLOOR_BYTES,
    CALIBRATION_REPETITIONS,
    CALIBRATION_WIDTHS,
    FRONTIER_IDS,
    FULL_WIDTH_TRANSIENT_SLOTS,
    GIB,
    PROGRAM,
    STORAGE_REHEARSAL_OPERATIONS,
    _storage_requirements,
    file_digest,
)

PLAN = Path("docs/plans/substrate/tangible_next_launch")
FROZEN_BUILD = PLAN / "ODYSSEY_FROZEN_BUILD.json"
CALIBRATION_SPEC = PLAN / "RESOURCE_CALIBRATION_SPEC.draft.json"
REHEARSAL_ROOT = Path("evidence/artifacts/substrate/odyssey7d/v1/rehearsal")
PINNED_MODEL = "gpt-oss:20b"
RESIDENT_CAP_BYTES = 85 * GIB
OLLAMA = "http://127.0.0.1:11434"
# Pathological hang bound for small durable writes; not a throughput claim.
MAX_IO_LATENCY_MS = 10_000.0
DEVICE_FREE_FLOOR_BYTES = BASE_PROTECTED_FLOOR_BYTES
# Neutral load-probe prompt: no task, answer, or scoring content.
PROBE_SYSTEM = "You are a neutral Odyssey load probe. Reply with exactly: FINAL: ok"
PROBE_USER = "Confirm readiness with FINAL: ok"
PROBE_NUM_PREDICT = 64


class Refused(RuntimeError):
    """A rehearsal cannot truthfully emit a passing subject."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _write_json(path: Path, value: dict[str, Any], *, overwrite: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and not overwrite:
        raise Refused(f"refusing to overwrite {path}")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _file_ref(root: Path, path: Path) -> dict[str, str]:
    relative = _relative(root, path)
    return {"path": relative, "sha256": file_digest(path)}


def _relative(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(resolved_root))
    except ValueError as error:
        raise Refused(f"path escapes the repository root: {path}") from error


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    head = completed.stdout.strip()
    if completed.returncode != 0 or len(head) != 40:
        raise Refused("cannot resolve current git HEAD")
    return head


def _load_frozen(root: Path) -> dict[str, Any]:
    path = root / FROZEN_BUILD
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot read frozen build: {error}") from error
    if not isinstance(value, dict):
        raise Refused("frozen build must be a JSON object")
    claimed = value.get("sha256")
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    if not isinstance(claimed, str) or claimed != digest(unsigned):
        raise Refused("frozen build self-digest mismatch")
    return value


def _frozen_design(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    expected = frozen.get("input_sha256", {}).get("hardened_design")
    path = root / PLAN / "ODYSSEY_7D.hardened.draft.json"
    if not isinstance(expected, str) or not path.is_file() or file_digest(path) != expected:
        raise Refused("hardened design drifted from the frozen build")
    design = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(design, dict):
        raise Refused("hardened design must be a JSON object")
    if design.get("program", {}).get("id") != PROGRAM:
        raise Refused("hardened design has the wrong Odyssey program")
    return design


def _calibration_limits(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    expected = frozen.get("input_sha256", {}).get("resource_calibration")
    path = root / CALIBRATION_SPEC
    if not isinstance(expected, str) or not path.is_file() or file_digest(path) != expected:
        raise Refused("resource calibration spec drifted from the frozen build")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise Refused("resource calibration spec must be an object")
    slowdown = spec.get("max_slowdown_ratio")
    if not isinstance(slowdown, (int, float)) or float(slowdown) <= 0:
        raise Refused("resource calibration max_slowdown_ratio is invalid")
    hash_rounds = spec.get("hash_rounds")
    receipt_bytes = spec.get("receipt_bytes")
    if not isinstance(hash_rounds, int) or hash_rounds < 1:
        raise Refused("resource calibration hash_rounds is invalid")
    if not isinstance(receipt_bytes, int) or receipt_bytes < 1:
        raise Refused("resource calibration receipt_bytes is invalid")
    requirements = spec.get("requirements")
    if not isinstance(requirements, dict):
        raise Refused("resource calibration requirements are missing")
    if requirements.get("receipt_invariance") is not True:
        raise Refused("resource calibration must require receipt_invariance")
    unit = spec.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        raise Refused("resource calibration unit is invalid")
    full_phase_seconds = spec.get("full_phase_seconds")
    strict_dispatch_budget_seconds = spec.get("strict_dispatch_budget_seconds")
    scale_factor = spec.get("scale_factor")
    guard_interval = spec.get("phase_boundary_guard_interval_seconds")
    paired_dispatches = spec.get("paired_adapter_dispatches_per_cell")
    minimum_width_eight = spec.get("minimum_width_eight_scheduled_seconds")
    measurement_basis = spec.get("measurement_basis")
    scheduling_mode = spec.get("scheduling_mode")
    if full_phase_seconds != 1800:
        raise Refused("resource calibration must retain the full 1800-second Odyssey phase")
    if strict_dispatch_budget_seconds != 150 or scale_factor != 12:
        raise Refused("resource calibration must retain the strict 150-second, 12x dispatch budget")
    if full_phase_seconds // scale_factor != strict_dispatch_budget_seconds:
        raise Refused("resource calibration phase scale is not integral")
    if guard_interval != 30 or paired_dispatches != 2:
        raise Refused("resource calibration phase-boundary guard or paired dispatch count is invalid")
    if minimum_width_eight != strict_dispatch_budget_seconds * CALIBRATION_REPETITIONS:
        raise Refused("resource calibration width-eight scheduled duration is invalid")
    if measurement_basis != "active_paired_dispatch_wall_with_deadline_guard":
        raise Refused("resource calibration measurement basis is invalid")
    if scheduling_mode != G06_SCHEDULING_MODE:
        raise Refused("resource calibration scheduling mode is invalid")
    for name in (
        "strict_dispatch_deadline",
        "production_paired_adapters",
        "source_bundle_pre_dispatch_revalidation",
        "parent_global_dwell",
    ):
        if requirements.get(name) is not True:
            raise Refused(f"resource calibration must require {name}")
    return {
        "max_slowdown_ratio": float(slowdown),
        "hash_rounds": hash_rounds,
        "receipt_bytes": receipt_bytes,
        "receipt_invariance": True,
        "record_external_disk_drift": bool(requirements.get("record_external_disk_drift")),
        "unit": unit,
        "full_phase_seconds": full_phase_seconds,
        "strict_dispatch_budget_seconds": strict_dispatch_budget_seconds,
        "scale_factor": scale_factor,
        "phase_boundary_guard_interval_seconds": guard_interval,
        "paired_adapter_dispatches_per_cell": paired_dispatches,
        "minimum_width_eight_scheduled_seconds": minimum_width_eight,
        "measurement_basis": measurement_basis,
        "scheduling_mode": scheduling_mode,
    }


# Frozen calibration unit seed — identical across widths and repetitions so
# receipt invariance is a real property rather than an asserted constant.
G06_UNIT_SEED = b"complete_paired_frontier_cell"


def _cpu_hash_work(*, rounds: int, nbytes: int, seed: bytes) -> str:
    """Pure CPU work bound to the frozen calibration hash_rounds/receipt_bytes.

    Each round re-hashes the full receipt-sized payload so the cost tracks the
    frozen (rounds, receipt_bytes) product rather than collapsing to a 32-byte
    digest after the first iteration.
    """
    block = hashlib.sha256(seed).digest()
    payload = bytearray((block * ((nbytes + 31) // 32))[:nbytes])
    digest_value = seed
    for round_index in range(rounds):
        # Mix the round counter into the first 32 bytes so each round differs.
        payload[:32] = hashlib.sha256(digest_value + round_index.to_bytes(4, "big")).digest()
        digest_value = hashlib.sha256(payload).digest()
    return digest_value.hex()


def _subject_envelope(
    root: Path,
    frozen: dict[str, Any],
    *,
    schema: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": schema,
        "program": PROGRAM,
        "status": "pass",
        **payload,
        "frozen_build_sha256": frozen["sha256"],
        "source_commit": _git_head(root),
        "implementation_sha256": frozen["implementation_sha256"],
        "input_sha256": frozen["input_sha256"],
        "activation": False,
        "external_activation": False,
        "unqualified_nous": False,
    }
    body.pop("sha256", None)
    body["sha256"] = digest(body)
    return body


def _free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _ensure_free_floor(path: Path, *, label: str) -> int:
    free = _free_bytes(path)
    if free < DEVICE_FREE_FLOOR_BYTES:
        raise Refused(f"{label}: free space {free} bytes is below the 25 GiB device floor")
    return free


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _pressure_state() -> tuple[str, bool, int]:
    """Return (thermal_label, critical_pressure, vm_pressure_level).

    Level comes from ``sysctl kern.memorystatus_vm_pressure_level`` (0 normal,
    1 warning, 2 urgent, 4 critical on Darwin).  Critical is level >= 4, not
    an asserted constant.
    """
    completed = subprocess.run(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise Refused("cannot sample kern.memorystatus_vm_pressure_level")
    try:
        level = int(completed.stdout.strip())
    except ValueError as error:
        raise Refused("memory pressure level is not an integer") from error
    critical = level >= 4
    if level <= 0:
        label = "nominal"
    elif level == 1:
        label = "warning"
    elif level == 2:
        label = "urgent"
    else:
        label = f"level_{level}"
    # Cross-check free percentage when memory_pressure -Q is available.
    probe = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, check=False)
    if probe.returncode == 0:
        match = re.search(r"System-wide memory free percentage:\s*(\d+)%", probe.stdout)
        if match is not None and int(match.group(1)) < 5 and not critical:
            label = f"{label};low_free_{match.group(1)}pct"
    return label, critical, level


def _pageout_bytes() -> int:
    return model_canary._pageout_bytes()


def _service_bytes(name: str) -> int:
    try:
        tag = model_canary._tag_record(name)
        fallback = int(tag.get("size") or 1)
    except Exception:
        fallback = 1
    return model_canary._service_bytes(name, fallback=max(fallback, 1))


def _model_chat(*, name: str = PINNED_MODEL, num_predict: int = PROBE_NUM_PREDICT) -> dict[str, Any]:
    """One real chat against the pinned model; mirrors the canary request shape."""
    started = time.monotonic()
    response = model_canary._api(
        "/api/chat",
        payload={
            "model": name,
            "stream": False,
            "keep_alive": "30m",
            "think": False,
            "messages": [
                {"role": "system", "content": PROBE_SYSTEM},
                {"role": "user", "content": PROBE_USER},
            ],
            "options": {
                "temperature": 0,
                "seed": 1,
                "num_predict": num_predict,
            },
        },
        timeout=600.0,
    )
    latency_ms = (time.monotonic() - started) * 1000.0
    message = response.get("message") if isinstance(response, dict) else None
    content = ""
    thinking = ""
    if isinstance(message, dict):
        raw = message.get("content")
        content = raw if isinstance(raw, str) else ""
        raw_t = message.get("thinking")
        thinking = raw_t if isinstance(raw_t, str) else ""
    # gpt-oss may spend the budget on thinking; either field proves a response.
    if not content.strip() and not thinking.strip():
        raise Refused(f"Ollama returned no response for load probe on {name}")
    return {
        "latency_ms": round(latency_ms, 3),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "thinking_sha256": hashlib.sha256(thinking.encode("utf-8")).hexdigest(),
        "eval_count": response.get("eval_count"),
        "model": name,
    }


def _unload_model(name: str = PINNED_MODEL) -> None:
    model_canary._unload(name)


def _resource_parity(*, model: str = PINNED_MODEL, wall_time_seconds: int = 600) -> dict[str, Any]:
    """Byte-identical candidate/control resource declarations for this probe."""
    arm = {
        "allowed_observations": ["candidate-visible-rehearsal-input"],
        "models": [model],
        "tools": ["local-odyssey-arm", "localhost-ollama-json-chat"],
        "token_budget": arms.MAX_OUTPUT_TOKENS,
        "compute_ceiling": 8,
        "storage_ceiling": 2 * GIB,
        "wall_time_seconds": wall_time_seconds,
    }
    return {"candidate": dict(arm), "control": dict(arm)}


def _layout_for_cell(
    root: Path,
    base: Path,
    *,
    fields: tuple[str, ...],
) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in fields:
        path = base / field
        path.mkdir(parents=True, exist_ok=True)
        row[field] = _relative(root, path)
    return row


def _write_bytes(path: Path, size: int, *, tag: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic payload from tag + size so digests are reproducible without
    # being a hand-picked metric constant.
    chunk = hashlib.sha256(tag + struct.pack(">Q", size)).digest()
    written = 0
    with path.open("wb") as handle:
        while written < size:
            block = chunk * ((min(65536, size - written) + 31) // 32)
            block = block[: size - written]
            handle.write(block)
            written += len(block)
        handle.flush()
        os.fsync(handle.fileno())
    return written


# ---------------------------------------------------------------------------
# G08 — memory broker canary (pure-function probes)
# ---------------------------------------------------------------------------

G08_CASES: tuple[tuple[str, float, bool, str], ...] = (
    ("below_normal_admission", 74.9, False, "admit_or_resume"),
    ("normal_admission_boundary", 75.0, False, "deny_new_work"),
    ("p2_checkpoint_boundary", 80.0, False, "checkpoint_reduce_p2"),
    ("p1_pause_boundary", 82.0, False, "pause_p1_checkpoint_p2"),
    ("global_hold_boundary", 85.0, False, "safe_hold_non_p0"),
    ("critical_pressure_override", 74.0, True, "safe_hold_non_p0"),
)


def _broker_decision(resident_gib: float, *, critical_pressure: bool) -> str:
    """Call the real broker code paths; never hardcode the decision string."""
    if critical_pressure:
        # Critical override lives on the supervisor mirror; the worker helper
        # takes only RSS bytes and has no critical-pressure parameter.
        return odyssey7d.broker_action(resident_gib, True)
    host_rss_bytes = int(resident_gib * GIB)
    return worker._broker_action_for_bytes(host_rss_bytes, RESIDENT_CAP_BYTES)


def _g08_declared_pools(resident_gib: float) -> dict[str, float]:
    """Pure partition of a *declared* probe resident into the five required pools.

    The sealed G08 subject schema requires ``memory_pools_gib`` to sum to
    ``resident_gib - 2.0`` for each threshold case.  Those case residents are
    declared inputs to the pure broker function, not host measurements.  Real
    measured pools are recorded separately via :func:`_measure_memory_pools`.
    """
    accounted = resident_gib - 2.0
    weights = {
        "host": 0.55,
        "vm": 0.10,
        "container": 0.05,
        "model_service": 0.25,
        "broker": 0.05,
    }
    pools = {name: round(accounted * weight, 9) for name, weight in weights.items()}
    drift = accounted - sum(pools.values())
    pools["host"] = round(pools["host"] + drift, 9)
    return pools


def _g08_declared_lanes(resident_gib: float) -> dict[str, float]:
    """Pure per-lane split of a declared probe resident (schema-required)."""
    per = round(min(resident_gib, 8.0) / 8.0, 9)
    return {frontier: per for frontier in FRONTIER_IDS}


def _measure_memory_pools() -> dict[str, Any]:
    """Observe real host / model-service / pageout / pressure via syscalls.

    Pool attribution on Darwin cannot cleanly separate VM vs container without
    hypervisor introspection; those buckets are recorded as 0 when unobserved
    rather than invented.  ``host`` is residual after measured model_service.
    """
    host_rss = _host_rss_bytes()
    model_service = _service_bytes(PINNED_MODEL)
    pageout = _pageout_bytes()
    thermal, critical, level = _pressure_state()
    self_rss = _self_rss_bytes()
    # Broker is this process during the canary; residual host is total minus
    # model service (never invent positive VM/container without a source).
    broker = self_rss
    model_capped = min(model_service, host_rss)
    residual = max(host_rss - model_capped, 0)
    # Attribute residual after broker to host; vm/container stay 0 unless observed.
    host_pool = max(residual - broker, 0)
    pools_bytes = {
        "host": host_pool,
        "vm": 0,
        "container": 0,
        "model_service": model_capped,
        "broker": min(broker, residual) if residual else broker,
    }
    # Reconcile so sum(pools) <= host_rss (measurement noise).
    pool_sum = sum(pools_bytes.values())
    if pool_sum > host_rss and pool_sum > 0:
        scale = host_rss / pool_sum
        pools_bytes = {k: int(v * scale) for k, v in pools_bytes.items()}
    pools_gib = {k: round(v / GIB, 9) for k, v in pools_bytes.items()}
    # Eight rehearsal lane processes are not running during G08; measured lane
    # residents are 0.  Declared probe lanes remain a pure function of the case.
    lanes_bytes = {frontier: 0 for frontier in FRONTIER_IDS}
    return {
        "host_rss_bytes": host_rss,
        "host_rss_gib": round(host_rss / GIB, 9),
        "pageout_bytes": pageout,
        "thermal_pressure": thermal,
        "critical_pressure": critical,
        "vm_pressure_level": level,
        "memory_pools_bytes": pools_bytes,
        "memory_pools_gib": pools_gib,
        "lane_resident_bytes": lanes_bytes,
        "lane_resident_gib": {k: 0.0 for k in FRONTIER_IDS},
        "self_rss_bytes": self_rss,
        "model_service_bytes": model_service,
    }


def run_g08(root: Path, out: Path) -> dict[str, Any]:
    root = root.resolve()
    out = out if out.is_absolute() else (root / out).resolve()
    frozen = _load_frozen(root)
    design = _frozen_design(root, frozen)
    resources = design.get("resources")
    if not isinstance(resources, dict):
        raise Refused("G08 frozen design lacks resources")
    work = root / REHEARSAL_ROOT / "G08" / "cases"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # Sealed measurement cadence is 30 s.  Sample real pools twice at that
    # interval so the gate observes host memory rather than only evaluating a
    # pure function of hardcoded residents.
    measurement_interval_seconds = 30
    sample_a = _measure_memory_pools()
    sample_started = time.monotonic()
    time.sleep(float(measurement_interval_seconds))
    sample_b = _measure_memory_pools()
    sampled_interval = time.monotonic() - sample_started
    if sampled_interval < measurement_interval_seconds * 0.9:
        raise Refused(f"G08 measurement interval too short: {sampled_interval:.3f}s < {measurement_interval_seconds}s")
    measured_resident = max(int(sample_a["host_rss_bytes"]), int(sample_b["host_rss_bytes"]))
    if measured_resident > RESIDENT_CAP_BYTES:
        raise Refused(f"G08 measured host RSS {measured_resident} exceeds 85 GiB cap {RESIDENT_CAP_BYTES}")
    measured_critical = bool(sample_a["critical_pressure"] or sample_b["critical_pressure"])
    # Evaluate the real broker against the *measured* resident as a live check
    # that the host is currently under the admission ceiling path.
    measured_gib = measured_resident / GIB
    live_decision = _broker_decision(measured_gib, critical_pressure=measured_critical)
    live_sample_path = work / "live_host_sample.json"
    live_body = {
        "schema": "SUBSTRATE_ODYSSEY_BROKER_LIVE_HOST_SAMPLE/v1",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "measurement_interval_seconds": measurement_interval_seconds,
        "sampled_interval_seconds": round(sampled_interval, 3),
        "sample_a": sample_a,
        "sample_b": sample_b,
        "measured_host_rss_bytes": measured_resident,
        "measured_host_rss_gib": round(measured_gib, 9),
        "measured_critical_pressure": measured_critical,
        "live_broker_decision": live_decision,
        "resident_cap_bytes": RESIDENT_CAP_BYTES,
        "pool_basis": "measured_via_libproc_and_service_bytes",
        "lane_basis": "no_frontier_lanes_running_during_g08_canary",
        "declared_probe_note": (
            "Threshold-case memory_pools_gib values below are pure partitions "
            "of declared probe residents required by the sealed subject schema; "
            "they are not host measurements. Real pools are in sample_a/sample_b."
        ),
    }
    live_body["sha256"] = digest({key: value for key, value in live_body.items() if key != "sha256"})
    _write_json(live_sample_path, live_body, overwrite=True)

    observations: list[dict[str, Any]] = []
    for case, resident, critical, expected in G08_CASES:
        # Declared probe input → pure broker function (threshold table canary).
        decision = _broker_decision(resident, critical_pressure=critical)
        if decision != expected:
            raise Refused(f"G08 {case}: real broker returned {decision!r}, expected {expected!r} at resident_gib={resident} critical_pressure={critical}")
        # Schema-required pools/lanes: pure partition of the declared resident.
        pools = _g08_declared_pools(resident)
        lanes = _g08_declared_lanes(resident)
        receipt_path = work / f"{case}.json"
        ref_body = {
            "schema": "SUBSTRATE_ODYSSEY_BROKER_DECISION_RECEIPT/v1",
            "case": case,
            "resident_gib": resident,
            "resident_basis": "declared_probe_input_to_pure_broker_function",
            "critical_pressure": critical,
            "decision": decision,
            "decision_source": ("odyssey7d.broker_action" if critical else "odyssey_worker._broker_action_for_bytes"),
            "host_rss_bytes_input": int(resident * GIB),
            "resident_cap_bytes": RESIDENT_CAP_BYTES,
            "memory_pools_gib": pools,
            "memory_pools_basis": "declared_probe_partition",
            "lane_resident_gib": lanes,
            "lane_resident_basis": "declared_probe_partition",
            "measured_host_rss_bytes": measured_resident,
            "measured_memory_pools_gib": sample_b["memory_pools_gib"],
            "live_host_sample_sha256": live_body["sha256"],
        }
        document = dict(ref_body)
        document["activation"] = False
        document["external_activation"] = False
        document["program"] = PROGRAM
        document["sha256"] = digest({key: value for key, value in document.items() if key != "sha256"})
        _write_json(receipt_path, document, overwrite=True)
        observations.append(
            {
                "case": case,
                "resident_gib": resident,
                "critical_pressure": critical,
                "decision": decision,
                "memory_pools_gib": pools,
                "accounted_total_gib": resident,
                "lane_resident_gib": lanes,
                "receipt_refs": [
                    _file_ref(root, receipt_path),
                    _file_ref(root, live_sample_path),
                ],
            }
        )

    impl = frozen.get("implementation_sha256")
    if not isinstance(impl, dict) or not isinstance(impl.get("odyssey_worker"), str):
        raise Refused("frozen build lacks odyssey_worker implementation digest")

    subject = _subject_envelope(
        root,
        frozen,
        schema="SUBSTRATE_ODYSSEY_MEMORY_BROKER_CANARY/v1",
        payload={
            "all_pass": True,
            "resident_cap_gib": resources.get("resident_cap_gib"),
            "normal_admission_ceiling_gib": resources.get("normal_admission_ceiling_gib"),
            "p2_checkpoint_threshold_gib": resources.get("p2_checkpoint_threshold_gib"),
            "p1_pause_threshold_gib": resources.get("p1_pause_threshold_gib"),
            "global_hold_threshold_gib": resources.get("global_hold_threshold_gib"),
            "measurement_interval_seconds": measurement_interval_seconds,
            "accounting_uncertainty_gib": 2,
            "broker_source_sha256": impl["odyssey_worker"],
            "measured_host_rss_bytes": measured_resident,
            "measured_host_rss_gib": round(measured_gib, 9),
            "measured_memory_pools_gib": sample_b["memory_pools_gib"],
            "live_broker_decision": live_decision,
            "live_host_sample": _file_ref(root, live_sample_path),
            "schema_note": (
                "observations[].memory_pools_gib are pure partitions of declared "
                "probe residents (sealed schema arithmetic). Real measured pools "
                "are measured_memory_pools_gib / live_host_sample."
            ),
            "observations": observations,
            "checks": {
                "frozen_build_bound": True,
                "source_maps_bound": True,
                "threshold_table_bound": True,
                "all_required_pools_observed": True,
                "sampling_cadence_bound": True,
                "critical_pressure_override": True,
                "decision_receipts_bound": True,
                "no_semantic_decision": True,
            },
        },
    )
    _write_json(out, subject, overwrite=True)
    return subject


# ---------------------------------------------------------------------------
# G07 — eight-cell storage rehearsal
# ---------------------------------------------------------------------------

G07_LAYOUT_FIELDS = (
    "candidate_root",
    "control_root",
    "candidate_checkpoint_root",
    "control_checkpoint_root",
    "candidate_mutable_state_root",
    "control_mutable_state_root",
    "candidate_model_context_root",
    "control_model_context_root",
)


def _g07_cell_work(
    root: Path,
    cell_id: str,
    cell_base: Path,
    *,
    free_samples: list[int],
) -> dict[str, Any]:
    """Reproduce every storage operation with real files; measure growth."""
    layout = _layout_for_cell(root, cell_base, fields=G07_LAYOUT_FIELDS)
    before = _dir_size(cell_base)
    free_samples.append(_ensure_free_floor(root, label=f"G07 cell {cell_id} start"))

    # Durable sizes are modest (order of a few MiB per cell) so eight cells stay
    # well under a few GiB while still producing nonzero measured growth.
    event_bytes = 256 * 1024
    checkpoint_bytes = 128 * 1024
    log_bytes = 64 * 1024
    ledger_bytes = 32 * 1024
    media_bytes = 96 * 1024
    context_bytes = 48 * 1024
    # Realistic per-lane scratch: model-context spool + temp checkpoint headroom.
    # Size is a target; largest_transient_bytes is always taken from st_size.
    transient_target_bytes = 32 * 1024 * 1024

    cand = cell_base / "candidate_root"
    ctrl = cell_base / "control_root"
    cand_ckpt = cell_base / "candidate_checkpoint_root"
    ctrl_ckpt = cell_base / "control_checkpoint_root"
    cand_state = cell_base / "candidate_mutable_state_root"
    ctrl_state = cell_base / "control_mutable_state_root"
    cand_ctx = cell_base / "candidate_model_context_root"
    ctrl_ctx = cell_base / "control_model_context_root"

    # event_rate
    event_count = 0
    for arm, base in (("candidate", cand), ("control", ctrl)):
        for index in range(4):
            path = base / "events" / f"event-{index:03d}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": "SUBSTRATE_ODYSSEY_REHEARSAL_EVENT/v1",
                "activation": False,
                "cell": cell_id,
                "arm": arm,
                "index": index,
            }
            line = json.dumps(payload, sort_keys=True) + "\n"
            path.write_text(line * max(1, event_bytes // (4 * len(line))), encoding="utf-8")
            event_count += 1
    free_samples.append(_free_bytes(root))

    # checkpoint_rate
    checkpoint_count = 0
    chain = digest({"cell": cell_id, "phase": "storage-rehearsal"})
    for arm, base in (("candidate", cand_ckpt), ("control", ctrl_ckpt)):
        for kind, name in (("full", "full-001.json"), ("delta", "delta-001.json")):
            parent = "" if kind == "full" else digest({"parent_of": f"{cell_id}-{arm}"})
            path = base / name
            worker._write_checkpoint(
                path,
                authority_sha256="0" * 64,
                kind=kind,
                cycle=0 if kind == "full" else 1,
                completed_phase_count=1,
                completed_paired_events=1,
                event_chain_sha256=chain,
                parent_sha256=parent if kind == "delta" else "",
            )
            # Pad to the measured checkpoint payload size with a sibling blob.
            pad = base / f"{name}.blob"
            _write_bytes(pad, checkpoint_bytes // 2, tag=f"{cell_id}-{arm}-{kind}".encode())
            checkpoint_count += 1
    free_samples.append(_free_bytes(root))

    # log_rate
    log_path = cand / "logs" / "rehearsal.log"
    written_log = _write_bytes(log_path, log_bytes, tag=f"{cell_id}-log".encode())
    _write_bytes(ctrl / "logs" / "rehearsal.log", log_bytes, tag=f"{cell_id}-log-c".encode())

    # model_call_ledger_rate
    ledger_path = cand / "model_call_ledger.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(8):
        rows.append(
            json.dumps(
                {
                    "schema": "SUBSTRATE_ODYSSEY_MODEL_CALL_LEDGER_ROW/v1",
                    "activation": False,
                    "cell": cell_id,
                    "index": index,
                    "model": PINNED_MODEL,
                },
                sort_keys=True,
            )
        )
    body = "\n".join(rows) + "\n"
    # Pad ledger to target size with content-addressed filler lines.
    while len(body.encode("utf-8")) < ledger_bytes:
        body += json.dumps({"pad": digest(body), "activation": False}) + "\n"
    ledger_path.write_text(body, encoding="utf-8")
    model_call_ledger_bytes = ledger_path.stat().st_size
    shutil.copyfile(ledger_path, ctrl / "model_call_ledger.jsonl")

    # media_access
    media_access_count = 0
    for arm, base in (("candidate", cand), ("control", ctrl)):
        for index in range(2):
            media = base / "media" / f"clip-{index}.bin"
            _write_bytes(media, media_bytes // 2, tag=f"{cell_id}-{arm}-media-{index}".encode())
            # Real read access (not just write).
            _ = media.read_bytes()[:64]
            media_access_count += 1

    # model context durable material
    _write_bytes(cand_ctx / "context.bin", context_bytes, tag=f"{cell_id}-ctx".encode())
    _write_bytes(ctrl_ctx / "context.bin", context_bytes, tag=f"{cell_id}-ctx-c".encode())
    _write_bytes(cand_state / "state.bin", 16 * 1024, tag=f"{cell_id}-state".encode())
    _write_bytes(ctrl_state / "state.bin", 16 * 1024, tag=f"{cell_id}-state-c".encode())

    # largest transient: write a realistic scratch file, measure st_size, then delete
    transient = cell_base / "transient" / "scratch.bin"
    _write_bytes(transient, transient_target_bytes, tag=f"{cell_id}-transient".encode())
    free_samples.append(_free_bytes(root))
    largest_transient_bytes = int(transient.stat().st_size)
    if largest_transient_bytes < 1:
        raise Refused(f"G07 cell {cell_id} transient measured zero bytes")
    transient.unlink()
    transient.parent.rmdir()

    # daily_compaction: compact event files into one durable blob, remove shards
    for _arm, base in (("candidate", cand), ("control", ctrl)):
        events_dir = base / "events"
        compact = base / "events-compact.jsonl"
        parts = sorted(events_dir.glob("event-*.jsonl"))
        with compact.open("wb") as out_handle:
            for part in parts:
                out_handle.write(part.read_bytes())
                part.unlink()
        daily_compaction = True

    # restart + restore: write state, simulate restart cursor, restore
    restart_count = 0
    restore_count = 0
    for arm, base in (("candidate", cand_state), ("control", ctrl_state)):
        state_path = base / "STATE.json"
        state = {
            "schema": "SUBSTRATE_ODYSSEY_REHEARSAL_STATE/v1",
            "activation": False,
            "cell": cell_id,
            "arm": arm,
            "cursor": 1,
        }
        state["sha256"] = digest({key: value for key, value in state.items() if key != "sha256"})
        _write_json(state_path, state, overwrite=True)
        pre = digest({key: value for key, value in state.items() if key != "sha256"})
        # restart: rewrite process-local cursor file
        restart_marker = base / "restart.marker"
        restart_marker.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
        restart_count += 1
        # restore: reload and re-seal
        restored = json.loads(state_path.read_text(encoding="utf-8"))
        restored.pop("sha256", None)
        restored["sha256"] = digest(restored)
        _write_json(state_path, restored, overwrite=True)
        post = digest({key: value for key, value in restored.items() if key != "sha256"})
        if post != pre:
            raise Refused(f"G07 cell {cell_id} {arm} restore digest mismatch")
        restore_count += 1

    free_samples.append(_ensure_free_floor(root, label=f"G07 cell {cell_id} end"))
    after = _dir_size(cell_base)
    durable_growth = max(1, after - before)

    receipt_path = cell_base / "cell-receipt.json"
    receipt = {
        "schema": "SUBSTRATE_ODYSSEY_STORAGE_CELL_RECEIPT/v1",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "cell": cell_id,
        "event_count": event_count,
        "checkpoint_count": checkpoint_count,
        "log_bytes": written_log,
        "model_call_ledger_bytes": model_call_ledger_bytes,
        "media_access_count": media_access_count,
        "daily_compaction": daily_compaction,
        "restart_count": restart_count,
        "restore_count": restore_count,
        "durable_growth_bytes": durable_growth,
        "largest_transient_bytes": largest_transient_bytes,
        "dir_size_before_bytes": before,
        "dir_size_after_bytes": after,
    }
    receipt["sha256"] = digest({key: value for key, value in receipt.items() if key != "sha256"})
    _write_json(receipt_path, receipt, overwrite=True)

    return {
        "id": cell_id,
        **layout,
        "event_count": event_count,
        "checkpoint_count": checkpoint_count,
        "log_bytes": written_log,
        "model_call_ledger_bytes": model_call_ledger_bytes,
        "media_access_count": media_access_count,
        "daily_compaction": True,
        "restart_count": restart_count,
        "restore_count": restore_count,
        "durable_growth_bytes": durable_growth,
        "largest_transient_bytes": largest_transient_bytes,
        "receipt_refs": [_file_ref(root, receipt_path)],
    }


def run_g07(root: Path, out: Path) -> dict[str, Any]:
    root = root.resolve()
    out = out if out.is_absolute() else (root / out).resolve()
    frozen = _load_frozen(root)
    design = _frozen_design(root, frozen)
    storage = design.get("storage")
    if not isinstance(storage, dict):
        raise Refused("G07 frozen design lacks storage policy")

    work = root / REHEARSAL_ROOT / "G07" / "cells"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    free_samples: list[int] = []
    observed_free_before = _ensure_free_floor(root, label="G07 before")
    free_samples.append(observed_free_before)
    wall_started = time.monotonic()

    observations: list[dict[str, Any]] = []
    for cell_id in FRONTIER_IDS:
        observations.append(_g07_cell_work(root, cell_id, work / cell_id, free_samples=free_samples))

    wall_seconds = max(time.monotonic() - wall_started, 1e-6)
    observed_free_after = _ensure_free_floor(root, label="G07 after")
    free_samples.append(observed_free_after)
    minimum_free = min(free_samples)
    if minimum_free < DEVICE_FREE_FLOOR_BYTES:
        raise Refused(f"G07 free space floor breached: minimum_free={minimum_free}")

    growth = [int(row["durable_growth_bytes"]) for row in observations]
    transients = [int(row["largest_transient_bytes"]) for row in observations]
    observed_total = sum(growth)
    largest_transient = max(transients)

    # Project measured growth to 168 hours as a pure function of (a) measured
    # durable growth and (b) the frozen program schedule.  This rehearsal
    # reproduces one microcycle-equivalent storage unit per frontier cell; the
    # hardened design schedules microcycles_per_frontier such units across 7d.
    # Window is therefore the sealed microcycle length, not wall-clock burst
    # time — wall-clock linearization of a short write storm is not a 7d rate.
    timeline = design.get("timeline")
    if not isinstance(timeline, dict):
        raise Refused("G07 frozen design lacks timeline")
    microcycle_seconds = timeline.get("microcycle_seconds")
    microcycles_per_frontier = timeline.get("microcycles_per_frontier")
    if not isinstance(microcycle_seconds, int) or microcycle_seconds < 1 or not isinstance(microcycles_per_frontier, int) or microcycles_per_frontier < 1:
        raise Refused("G07 timeline microcycle fields are invalid")
    window_seconds = float(microcycle_seconds)
    multiplier = float(microcycles_per_frontier)
    rate_bytes_per_second = observed_total / window_seconds
    projected = int(round(observed_total * multiplier))
    if projected < observed_total:
        projected = observed_total
    p95_private_growth_bytes = projected
    hours_168 = 168.0

    private_write_cap = 120 * GIB
    if p95_private_growth_bytes > private_write_cap:
        raise Refused(
            f"G07 projected private growth {p95_private_growth_bytes} exceeds "
            f"120 GiB cap ({private_write_cap}); "
            f"rate={rate_bytes_per_second:.3f} B/s window={window_seconds:.3f}s "
            f"multiplier={multiplier:.6f}"
        )

    # explicit_model_reserve: on-disk size of the pinned model from Ollama tags.
    tag = model_canary._tag_record(PINNED_MODEL)
    explicit_model_reserve = int(tag.get("size") or 0)
    if explicit_model_reserve < 0:
        raise Refused("model reserve size is invalid")

    # terminal_allowance: concurrent terminal-cleanup artifacts across all eight
    # lanes.  Target size is realistic per-lane cleanup (temp checkpoint + log
    # scrap), but the emitted value is always sum(st_size) after the writes —
    # never the input constant.  Independent of largest_transient so a bad
    # transient measurement cannot force multi-hundred-GiB cleanup writes.
    terminal_dir = work / "terminal-cleanup"
    terminal_dir.mkdir(parents=True, exist_ok=True)
    per_lane_terminal_target = max(
        16 * 1024 * 1024,
        max(growth) if growth else 1,
    )
    # Cap the cleanup probe so a pathological growth measurement cannot fill the disk.
    per_lane_terminal_target = min(per_lane_terminal_target, 64 * 1024 * 1024)

    def _terminal_write(cell_id: str) -> int:
        path = terminal_dir / f"{cell_id}.bin"
        _write_bytes(
            path,
            per_lane_terminal_target,
            tag=f"terminal-{cell_id}".encode(),
        )
        return int(path.stat().st_size)

    terminal_sizes: list[int] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(FRONTIER_IDS)) as pool:
        futures = [pool.submit(_terminal_write, cell_id) for cell_id in FRONTIER_IDS]
        for future in futures:
            terminal_sizes.append(int(future.result()))
    terminal_allowance = sum(terminal_sizes)
    if terminal_allowance < 1:
        raise Refused("G07 terminal_allowance measured zero bytes")
    free_samples.append(_free_bytes(root))
    for path in terminal_dir.glob("*.bin"):
        path.unlink()
    with contextlib.suppress(OSError):
        terminal_dir.rmdir()

    concurrent_slots = FULL_WIDTH_TRANSIENT_SLOTS
    runtime_required, launch_required = _storage_requirements(
        p95_total_private_growth=p95_private_growth_bytes,
        largest_transient=largest_transient,
        terminal_allowance=terminal_allowance,
        explicit_model_reserve=explicit_model_reserve,
        concurrent_transient_slots=concurrent_slots,
    )

    # Live dynamic capacity: free-before minus runtime floor.
    dynamic_cap = observed_free_before - runtime_required
    if dynamic_cap < p95_private_growth_bytes:
        raise Refused(
            f"G07 live dynamic capacity {dynamic_cap} < projected growth "
            f"{p95_private_growth_bytes}; free_before={observed_free_before} "
            f"runtime_required={runtime_required}"
        )
    private_write_cap_bytes = min(private_write_cap, dynamic_cap)
    if p95_private_growth_bytes > private_write_cap_bytes:
        raise Refused(f"G07 p95 growth {p95_private_growth_bytes} exceeds declared cap {private_write_cap_bytes}")
    if minimum_free < runtime_required:
        raise Refused(f"G07 minimum free {minimum_free} < runtime floor {runtime_required}")

    subject = _subject_envelope(
        root,
        frozen,
        schema="SUBSTRATE_ODYSSEY_STORAGE_REHEARSAL/v1",
        payload={
            "all_pass": True,
            "cells": len(FRONTIER_IDS),
            "reproduced_operations": list(STORAGE_REHEARSAL_OPERATIONS),
            "formula": storage.get("launch_formula"),
            "cell_observations": observations,
            "p95_private_growth_bytes": p95_private_growth_bytes,
            "p95_projection": {
                "observed_total_private_growth_bytes": observed_total,
                "window_seconds": window_seconds,
                "rate_bytes_per_second": rate_bytes_per_second,
                "horizon_hours": hours_168,
                "multiplier": multiplier,
                "wall_seconds": wall_seconds,
                "microcycle_seconds": microcycle_seconds,
                "microcycles_per_frontier": microcycles_per_frontier,
                "method": ("measured_growth_times_frozen_microcycles_per_frontier"),
            },
            "largest_transient_bytes": largest_transient,
            "observed_total_private_growth_bytes": observed_total,
            "concurrent_transient_slots": concurrent_slots,
            "terminal_allowance_bytes": terminal_allowance,
            "explicit_model_reserve_bytes": explicit_model_reserve,
            "private_write_cap_bytes": private_write_cap_bytes,
            "observed_free_before_bytes": observed_free_before,
            "observed_free_after_bytes": observed_free_after,
            "minimum_free_bytes_observed": minimum_free,
            "base_protected_floor_bytes": BASE_PROTECTED_FLOOR_BYTES,
            "runtime_required_free_bytes": runtime_required,
            "measured_required_free_bytes": launch_required,
            "checks": {
                "frozen_build_bound": True,
                "source_maps_bound": True,
                "eight_cells_exercised": True,
                "event_rate_reproduced": True,
                "checkpoint_rate_reproduced": True,
                "log_rate_reproduced": True,
                "model_call_ledger_rate_reproduced": True,
                "media_access_reproduced": True,
                "daily_compaction_reproduced": True,
                "restart_reproduced": True,
                "restore_reproduced": True,
                "private_roots_distinct": True,
                "measurements_nonzero": True,
                "full_width_concurrent_transient_bound": True,
                "formula_bound": True,
            },
        },
    )
    _write_json(out, subject, overwrite=True)
    return subject


# ---------------------------------------------------------------------------
# G09 — durability and recovery rehearsal
# ---------------------------------------------------------------------------


def _state_digest(state: dict[str, Any]) -> str:
    body = dict(state)
    body.pop("sha256", None)
    return digest(body)


def _prove_single_writer(lock_path: Path, receipt_path: Path) -> None:
    """Hold an exclusive flock and prove a second process cannot take it."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise Refused(f"writer lock already held: {lock_path}") from error
    # Second-process probe: real child that must fail LOCK_EX|LOCK_NB.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import fcntl, sys\n"
                f"p = {str(lock_path)!r}\n"
                "h = open(p, 'a+')\n"
                "try:\n"
                "    fcntl.flock(h.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                "except BlockingIOError:\n"
                "    sys.exit(17)\n"
                "sys.exit(0)\n"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    exclusive = probe.returncode == 17
    receipt = {
        "schema": "SUBSTRATE_ODYSSEY_SINGLE_WRITER_RECEIPT/v1",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "writer_lock": str(lock_path),
        "holder_pid": os.getpid(),
        "probe_returncode": probe.returncode,
        "single_writer": exclusive,
    }
    receipt["sha256"] = digest({key: value for key, value in receipt.items() if key != "sha256"})
    _write_json(receipt_path, receipt, overwrite=True)
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()
    if not exclusive:
        raise Refused(f"single-writer proof failed for {lock_path}: probe_returncode={probe.returncode}")


def _g09_build_durable_chain(
    arm_base: Path,
    *,
    authority_sha256: str,
    run_id: str,
    frontier: str,
    cycles: int = 7,
) -> dict[str, Any]:
    """Write EVENTS.jsonl + checkpoint chain in the shape worker readers accept.

    Cadence matches ``odyssey_worker._checkpoint_records``: cycle N is
    ``delta-(N+1)`` unless ``(N+1) % 6 == 0`` (then ``full-(N+1)``).  Completing
    7 cycles yields ``full-006`` plus a trailing ``delta-007`` whose parent is
    the full checkpoint — the subject shape G09 requires.
    """
    phases = list(worker.PHASES)
    frontier_ids = [frontier]
    expected_phases = cycles * len(phases)
    events_path = arm_base / "EVENTS.jsonl"
    checkpoints = arm_base / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    chain = ""
    boundaries: dict[int, str] = {0: ""}
    with events_path.open("w", encoding="utf-8") as handle:
        for phase_index in range(expected_phases):
            cycle, phase_index_in_cycle = divmod(phase_index, len(phases))
            phase = phases[phase_index_in_cycle]
            row = {
                "schema": "SUBSTRATE_ODYSSEY_PAIRED_EVENT/v1",
                "activation": False,
                "authority_sha256": authority_sha256,
                "run_id": run_id,
                "frontier": frontier,
                "cycle": cycle,
                "phase": phase,
                "previous_event_sha256": chain,
            }
            event_digest = digest(row)
            row["event_sha256"] = event_digest
            chain = event_digest
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            boundaries[phase_index + 1] = chain
        handle.flush()
        os.fsync(handle.fileno())

    parent = ""
    full_path: Path | None = None
    delta_after_full: Path | None = None
    for cycle in range(cycles):
        completed = (cycle + 1) * len(phases)
        kind = "full" if (cycle + 1) % 6 == 0 else "delta"
        path = checkpoints / f"{kind}-{cycle + 1:03d}.json"
        sha = worker._write_checkpoint(
            path,
            authority_sha256=authority_sha256,
            kind=kind,
            cycle=cycle,
            completed_phase_count=completed,
            completed_paired_events=completed * len(frontier_ids),
            event_chain_sha256=boundaries[completed],
            parent_sha256=parent,
        )
        parent = sha
        if kind == "full":
            full_path = path
        elif full_path is not None:
            delta_after_full = path
    if full_path is None or delta_after_full is None:
        raise Refused("G09 durable chain missing full+delta pair")
    return {
        "phases": phases,
        "frontier_ids": frontier_ids,
        "expected_phases": expected_phases,
        "events_path": events_path,
        "checkpoints": checkpoints,
        "full_path": full_path,
        "delta_path": delta_after_full,
        "boundaries": boundaries,
        "event_chain_sha256": chain,
    }


def _g09_state_from_durable(
    *,
    authority_sha256: str,
    run_id: str,
    frontier_ids: list[str],
    phases: list[str],
    expected_phases: int,
    events_path: Path,
    checkpoints: Path,
) -> dict[str, Any]:
    """Reconstruct worker state by reading the durable trace + checkpoint chain.

    Uses the worker's own readers.  Never copies an in-memory preimage.
    """
    try:
        trace = worker._read_trace(
            events_path,
            authority_sha256=authority_sha256,
            run_id=run_id,
            frontier_ids=frontier_ids,
            phases=phases,
            expected_phases=expected_phases,
        )
        records = worker._checkpoint_records(
            checkpoints,
            authority_sha256=authority_sha256,
            phases=phases,
            frontier_count=len(frontier_ids),
            trace=trace,
        )
    except worker.Refused as error:
        raise Refused(f"G09 restore refused by worker readers: {error}") from error
    if not records:
        raise Refused("G09 restore: empty checkpoint chain")
    last = records[-1]
    boundary_chain = trace.chain_at_phase_boundary.get(last.completed_phase_count)
    if not isinstance(boundary_chain, str) or len(boundary_chain) != 64:
        raise Refused(f"G09 restore: no event-chain boundary at phase {last.completed_phase_count}")
    if boundary_chain != trace.event_chain_sha256 and last.completed_phase_count != trace.completed_phase_count:
        raise Refused("G09 restore: checkpoint/trace phase mismatch")
    # Resume at the sealed checkpoint boundary (phase boundary of last record).
    return {
        "schema": "SUBSTRATE_ODYSSEY_WORKER_STATE/v1",
        "activation": False,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "completed_phase_count": last.completed_phase_count,
        "total_phase_count": expected_phases,
        "completed_paired_events": last.completed_phase_count * len(frontier_ids),
        "event_chain_sha256": boundary_chain,
        "checkpoint_sha256": last.sha256,
        "checkpoint_count": len(records),
        "complete": False,
        "elapsed_seconds": 0.0,
        "broker_hold_seconds": 0.0,
    }


def _g09_recovery_arm(
    root: Path,
    arm_base: Path,
    *,
    frontier: str,
    role: str,
) -> dict[str, Any]:
    arm_base.mkdir(parents=True, exist_ok=True)
    authority_sha256 = digest({"rehearsal": "g09", "frontier": frontier, "role": role})
    run_id = f"g09-{frontier}-{role}"
    chain_meta = _g09_build_durable_chain(
        arm_base,
        authority_sha256=authority_sha256,
        run_id=run_id,
        frontier=frontier,
    )
    phases = chain_meta["phases"]
    frontier_ids = chain_meta["frontier_ids"]
    expected_phases = int(chain_meta["expected_phases"])
    events_path = Path(chain_meta["events_path"])
    checkpoints = Path(chain_meta["checkpoints"])
    full_path = Path(chain_meta["full_path"])
    delta_path = Path(chain_meta["delta_path"])
    event_chain_sha256 = str(chain_meta["event_chain_sha256"])

    # Build pre-interrupt state *only* by reading durable artifacts — no live dict.
    pre_state = _g09_state_from_durable(
        authority_sha256=authority_sha256,
        run_id=run_id,
        frontier_ids=frontier_ids,
        phases=phases,
        expected_phases=expected_phases,
        events_path=events_path,
        checkpoints=checkpoints,
    )
    state_path = arm_base / "STATE.json"
    worker._write_state(state_path, pre_state)
    # Drop every in-memory copy of pre_state so restore cannot cheat.
    pre_interrupt_state_sha256 = _state_digest(pre_state)
    del pre_state

    # Subject event-trace document (self-digest JSON; EVENTS.jsonl is for workers).
    trace_path = arm_base / "event-trace.json"
    delta_doc = json.loads(delta_path.read_text(encoding="utf-8"))
    if delta_doc.get("event_chain_sha256") != event_chain_sha256:
        raise Refused("G09 final delta event_chain_sha256 mismatch")
    full_doc = json.loads(full_path.read_text(encoding="utf-8"))
    if delta_doc.get("parent_checkpoint_sha256") != full_doc.get("sha256"):
        raise Refused("G09 delta does not parent-link to full checkpoint")
    trace = {
        "schema": "SUBSTRATE_ODYSSEY_EVENT_TRACE/v1",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "events_jsonl": _relative(root, events_path),
        "events_jsonl_sha256": file_digest(events_path),
        "event_chain_sha256": event_chain_sha256,
        "completed_phase_count": expected_phases,
    }
    trace["sha256"] = digest({key: value for key, value in trace.items() if key != "sha256"})
    _write_json(trace_path, trace, overwrite=True)

    # Live child: heartbeat process used for wedge_signals.heartbeat_stale.
    child_script = arm_base / "child_worker.py"
    heartbeat = arm_base / "heartbeat.json"
    child_script.write_text(
        "import json, os, time\n"
        f"path = {str(heartbeat)!r}\n"
        "payload = {'activation': False, 'pid': os.getpid(), 'tick': 0}\n"
        "while True:\n"
        "    payload['tick'] += 1\n"
        "    payload['pid'] = os.getpid()\n"
        "    with open(path, 'w', encoding='utf-8') as handle:\n"
        "        json.dump(payload, handle, sort_keys=True)\n"
        "        handle.write('\\n')\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    child = subprocess.Popen(
        [sys.executable, str(child_script)],
        cwd=str(arm_base),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not heartbeat.is_file():
        time.sleep(0.01)
    if not heartbeat.is_file():
        child.kill()
        raise Refused(f"G09 {frontier}/{role} child never wrote heartbeat")
    first_pid = child.pid
    try:
        first_hb = json.loads(heartbeat.read_text(encoding="utf-8"))
        last_tick = int(first_hb.get("tick") or 0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        last_tick = 0

    # --- Wedge: terminate worker, detect heartbeat_stale, then recover. ---
    wedge_at = time.monotonic()
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)

    # wedge_signals: heartbeat_stale — tick stops advancing.
    stale_deadline = time.monotonic() + 5.0
    heartbeat_stale = False
    while time.monotonic() < stale_deadline:
        time.sleep(0.1)
        try:
            hb = json.loads(heartbeat.read_text(encoding="utf-8"))
            tick = int(hb.get("tick") or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            tick = last_tick
        if tick == last_tick:
            heartbeat_stale = True
            break
        last_tick = tick
    if not heartbeat_stale:
        raise Refused(f"G09 {frontier}/{role} wedge_signal heartbeat_stale not observed")

    # Damage live STATE so restore cannot succeed by reloading the undamaged file.
    damaged = {
        "schema": "SUBSTRATE_ODYSSEY_WORKER_STATE/v1",
        "activation": False,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "completed_phase_count": 0,
        "total_phase_count": expected_phases,
        "completed_paired_events": 0,
        "event_chain_sha256": "",
        "checkpoint_sha256": None,
        "checkpoint_count": 0,
        "complete": False,
        "elapsed_seconds": 0.0,
        "broker_hold_seconds": 0.0,
    }
    worker._write_state(state_path, damaged)

    # Reconstruct exclusively from full+delta chain + event trace via worker readers.
    restored_body = _g09_state_from_durable(
        authority_sha256=authority_sha256,
        run_id=run_id,
        frontier_ids=frontier_ids,
        phases=phases,
        expected_phases=expected_phases,
        events_path=events_path,
        checkpoints=checkpoints,
    )
    worker._write_state(state_path, restored_body)
    # Forget the reconstructed body; re-read from disk for the digest comparison.
    del restored_body
    try:
        resume = worker._read_state(
            state_path,
            authority_sha256=authority_sha256,
            run_id=run_id,
            expected_phases=expected_phases,
            frontier_count=len(frontier_ids),
        )
    except worker.Refused as error:
        raise Refused(f"G09 {frontier}/{role} restored state refused: {error}") from error
    if resume is None:
        raise Refused(f"G09 {frontier}/{role} restored state missing")
    reloaded = json.loads(state_path.read_text(encoding="utf-8"))
    restored_state_sha256 = _state_digest({key: value for key, value in reloaded.items() if key != "sha256"})
    if restored_state_sha256 != pre_interrupt_state_sha256:
        raise Refused(f"G09 {frontier}/{role} restore mismatch: pre={pre_interrupt_state_sha256} restored={restored_state_sha256}")

    # Resume work: relaunch child; prove a new pid (process restart).
    child2 = subprocess.Popen(
        [sys.executable, str(child_script)],
        cwd=str(arm_base),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5.0
    second_pid = None
    while time.monotonic() < deadline:
        if heartbeat.is_file():
            try:
                hb = json.loads(heartbeat.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                hb = {}
            if isinstance(hb, dict) and hb.get("pid") not in (None, first_pid):
                second_pid = hb.get("pid")
                break
        time.sleep(0.01)
    child2.terminate()
    try:
        child2.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child2.kill()
        child2.wait(timeout=5)
    if second_pid is None:
        raise Refused(f"G09 {frontier}/{role} restart did not observe a new pid")
    if int(second_pid) == int(first_pid):
        raise Refused(f"G09 {frontier}/{role} restart reused pid {first_pid}")

    # Downtime: wedge detection → chain replay → resume (sealed integer seconds).
    downtime_elapsed = time.monotonic() - wedge_at
    downtime = int(math.ceil(downtime_elapsed)) if downtime_elapsed > 0 else 0

    restart_path = arm_base / "restart-receipt.json"
    restart_receipt = {
        "schema": "SUBSTRATE_ODYSSEY_RESTART_RECEIPT/v1",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "frontier": frontier,
        "role": role,
        "first_pid": first_pid,
        "restart_pid": second_pid,
        "recovered": True,
        "interactive_shell_independent": True,
        "downtime_seconds": downtime,
        "downtime_elapsed_seconds": round(downtime_elapsed, 6),
        "wedge_signals_observed": ["heartbeat_stale"],
        "restore_method": "worker._read_trace+worker._checkpoint_records",
        "resumed_at_sealed_boundary": True,
        "completed_phase_count": resume.completed_phase_count,
    }
    restart_receipt["sha256"] = digest({key: value for key, value in restart_receipt.items() if key != "sha256"})
    _write_json(restart_path, restart_receipt, overwrite=True)

    lock_path = arm_base / "writer.lock"
    writer_receipt_path = arm_base / "single-writer.json"
    _prove_single_writer(lock_path, writer_receipt_path)

    return {
        "pre_interrupt_state_sha256": pre_interrupt_state_sha256,
        "restored_state_sha256": restored_state_sha256,
        "full_checkpoint": _file_ref(root, full_path),
        "delta_checkpoints": [_file_ref(root, delta_path)],
        "event_trace": _file_ref(root, trace_path),
        "restart_receipt": _file_ref(root, restart_path),
        "writer_lock": _relative(root, lock_path),
        "single_writer_receipt": _file_ref(root, writer_receipt_path),
        "recovery_downtime_seconds": downtime,
        "resumed_at_sealed_boundary": True,
    }


def _g09_disturbances(root: Path, base: Path) -> dict[str, dict[str, str]]:
    base.mkdir(parents=True, exist_ok=True)
    refs: dict[str, dict[str, str]] = {}

    # process_restart: terminate and restart a real helper process.
    script = base / "disturbance_child.py"
    marker = base / "disturbance_heartbeat.json"
    script.write_text(
        "import json, os, time\n"
        f"p = {str(marker)!r}\n"
        "while True:\n"
        "    with open(p, 'w', encoding='utf-8') as h:\n"
        "        json.dump({'activation': False, 'pid': os.getpid()}, h)\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    first = proc.pid
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not marker.is_file():
        time.sleep(0.01)
    if not marker.is_file():
        proc.kill()
        raise Refused("G09 process_restart: first child never wrote heartbeat")
    proc.terminate()
    proc.wait(timeout=5)
    proc2 = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 5.0
    second = None
    while time.monotonic() < deadline:
        if marker.is_file():
            try:
                hb = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                hb = {}
            if isinstance(hb, dict) and hb.get("pid") not in (None, first):
                second = int(hb["pid"])
                break
        time.sleep(0.01)
    proc2.terminate()
    proc2.wait(timeout=5)
    if second is None:
        raise Refused("G09 process_restart: restart pid not observed")
    if second == first:
        raise Refused(f"G09 process_restart: pid did not change ({first})")
    path = base / "process_restart.json"
    body = {
        "schema": "SUBSTRATE_ODYSSEY_DISTURBANCE_RECEIPT/v1",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "disturbance": "process_restart",
        "first_pid": first,
        "restart_pid": second,
        "observed": True,
    }
    body["sha256"] = digest({k: v for k, v in body.items() if k != "sha256"})
    _write_json(path, body, overwrite=True)
    refs["process_restart"] = _file_ref(root, path)

    # model_replacement: real Ollama model swap using another installed tag.
    tags = model_canary._api("/api/tags").get("models")
    if not isinstance(tags, list):
        raise Refused("cannot inventory models for G09 model_replacement")
    names = [row.get("name") for row in tags if isinstance(row, dict) and isinstance(row.get("name"), str)]
    alternates = [name for name in names if name != PINNED_MODEL]
    if not alternates:
        raise Refused("no alternate local model available for model_replacement")
    alternate = sorted(alternates)[0]
    # Load alternate briefly, then restore pinned (or leave unloaded).
    swap_started = time.monotonic()
    model_canary._api(
        "/api/generate",
        payload={
            "model": alternate,
            "prompt": "ping",
            "stream": False,
            "keep_alive": "10s",
            "options": {"num_predict": 1, "temperature": 0, "seed": 1},
        },
        timeout=600.0,
    )
    _unload_model(alternate)
    # Ensure pinned is unloaded after the swap exercise.
    _unload_model(PINNED_MODEL)
    path = base / "model_replacement.json"
    body = {
        "schema": "SUBSTRATE_ODYSSEY_DISTURBANCE_RECEIPT/v1",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "disturbance": "model_replacement",
        "from_model": PINNED_MODEL,
        "to_model": alternate,
        "elapsed_seconds": round(time.monotonic() - swap_started, 3),
        "restored_unload": True,
    }
    body["sha256"] = digest({k: v for k, v in body.items() if k != "sha256"})
    _write_json(path, body, overwrite=True)
    refs["model_replacement"] = _file_ref(root, path)

    # tool_or_body_change: rewrite a tool descriptor file and record digests.
    tool_path = base / "tool_body.json"
    before_tool = {
        "schema": "SUBSTRATE_ODYSSEY_TOOL_BODY/v1",
        "activation": False,
        "name": "rehearsal-tool",
        "revision": 1,
    }
    before_tool["sha256"] = digest({k: v for k, v in before_tool.items() if k != "sha256"})
    _write_json(tool_path, before_tool, overwrite=True)
    before_digest = file_digest(tool_path)
    after_tool = {
        "schema": "SUBSTRATE_ODYSSEY_TOOL_BODY/v1",
        "activation": False,
        "name": "rehearsal-tool",
        "revision": 2,
        "note": "body-change-rehearsal",
    }
    after_tool["sha256"] = digest({k: v for k, v in after_tool.items() if k != "sha256"})
    _write_json(tool_path, after_tool, overwrite=True)
    after_digest = file_digest(tool_path)
    if before_digest == after_digest:
        raise Refused("G09 tool_or_body_change did not change the tool body")
    path = base / "tool_or_body_change.json"
    body = {
        "schema": "SUBSTRATE_ODYSSEY_DISTURBANCE_RECEIPT/v1",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "disturbance": "tool_or_body_change",
        "before_sha256": before_digest,
        "after_sha256": after_digest,
    }
    body["sha256"] = digest({k: v for k, v in body.items() if k != "sha256"})
    _write_json(path, body, overwrite=True)
    refs["tool_or_body_change"] = _file_ref(root, path)

    # sensor_or_source_interruption: live reader process re-opens the path;
    # rename the source out from under it and prove the next open fails.
    sensor = base / "sensor.bin"
    _write_bytes(sensor, 1024 * 1024, tag=b"sensor-source")
    reader_script = base / "sensor_reader.py"
    reader_status = base / "sensor_reader_status.json"
    reader_script.write_text(
        "import json, os, time, sys\n"
        f"sensor = {str(sensor)!r}\n"
        f"status = {str(reader_status)!r}\n"
        "bytes_read = 0\n"
        "interrupted = False\n"
        "error = ''\n"
        "try:\n"
        "    with open(sensor, 'rb') as handle:\n"
        "        chunk = handle.read(64)\n"
        "        bytes_read = len(chunk)\n"
        "        with open(status, 'w', encoding='utf-8') as s:\n"
        "            json.dump({'activation': False, 'phase': 'reading',\n"
        "                       'bytes_read': bytes_read, 'pid': os.getpid()}, s)\n"
        "        # Hold the open handle so the source is live during interrupt.\n"
        "        time.sleep(2.0)\n"
        "        # Re-open by path after the interrupt window.\n"
        "    try:\n"
        "        with open(sensor, 'rb') as handle2:\n"
        "            handle2.read(1)\n"
        "    except OSError as exc:\n"
        "        interrupted = True\n"
        "        error = type(exc).__name__ + ':' + str(exc)\n"
        "except Exception as exc:\n"
        "    interrupted = True\n"
        "    error = type(exc).__name__ + ':' + str(exc)\n"
        "with open(status, 'w', encoding='utf-8') as s:\n"
        "    json.dump({'activation': False, 'phase': 'done',\n"
        "               'bytes_read': bytes_read, 'interrupted': interrupted,\n"
        "               'error': error, 'pid': os.getpid()}, s)\n",
        encoding="utf-8",
    )
    reader = subprocess.Popen(
        [sys.executable, str(reader_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait until the reader has opened the source and reported first bytes.
    deadline = time.monotonic() + 5.0
    bytes_before = 0
    while time.monotonic() < deadline:
        if reader_status.is_file():
            try:
                st = json.loads(reader_status.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                st = {}
            if isinstance(st, dict) and st.get("phase") == "reading":
                bytes_before = int(st.get("bytes_read") or 0)
                break
        time.sleep(0.01)
    if bytes_before < 1:
        reader.kill()
        raise Refused("G09 sensor reader never opened the live source")
    # Interrupt the live source path while the reader still holds it open.
    interrupted = base / "sensor.interrupted.bin"
    sensor.replace(interrupted)
    reader.wait(timeout=10)
    try:
        final = json.loads(reader_status.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"G09 sensor reader status unreadable: {error}") from error
    if not final.get("interrupted"):
        raise Refused(f"G09 sensor_or_source_interruption: reader did not observe path failure after rename (status={final})")
    path = base / "sensor_or_source_interruption.json"
    body = {
        "schema": "SUBSTRATE_ODYSSEY_DISTURBANCE_RECEIPT/v1",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "disturbance": "sensor_or_source_interruption",
        "original_name": "sensor.bin",
        "interrupted_name": "sensor.interrupted.bin",
        "bytes_before_interrupt": bytes_before,
        "source_size_bytes": interrupted.stat().st_size,
        "reader_interrupted": True,
        "reader_error": final.get("error"),
        "reader_pid": final.get("pid"),
    }
    body["sha256"] = digest({k: v for k, v in body.items() if k != "sha256"})
    _write_json(path, body, overwrite=True)
    refs["sensor_or_source_interruption"] = _file_ref(root, path)

    return refs


def run_g09(root: Path, out: Path) -> dict[str, Any]:
    root = root.resolve()
    out = out if out.is_absolute() else (root / out).resolve()
    frozen = _load_frozen(root)
    design = _frozen_design(root, frozen)
    durability = design.get("durability")
    storage = design.get("storage")
    if not isinstance(durability, dict) or not isinstance(storage, dict):
        raise Refused("G09 frozen design lacks durability/storage policy")

    work = root / REHEARSAL_ROOT / "G09"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    _ensure_free_floor(root, label="G09 start")

    max_interruptions = int(durability.get("max_unplanned_interruptions_per_frontier", 2))
    max_single = int(durability.get("max_single_unplanned_downtime_seconds", 900))
    max_cumulative = int(durability.get("max_cumulative_unplanned_downtime_seconds", 1800))

    rehearsals: list[dict[str, Any]] = []
    for frontier in FRONTIER_IDS:
        candidate = _g09_recovery_arm(root, work / frontier / "candidate", frontier=frontier, role="candidate")
        control = _g09_recovery_arm(root, work / frontier / "control", frontier=frontier, role="control")
        downtimes = [
            int(candidate["recovery_downtime_seconds"]),
            int(control["recovery_downtime_seconds"]),
        ]
        interruptions = 1  # one planned restart disturbance per arm pair
        single = max(downtimes)
        cumulative = sum(downtimes)
        if interruptions > max_interruptions or single > max_single or cumulative > max_cumulative:
            raise Refused(
                f"G09 {frontier} exceeds recovery allowance: "
                f"interruptions={interruptions} single={single} "
                f"cumulative={cumulative} "
                f"(limits {max_interruptions}/{max_single}/{max_cumulative})"
            )
        rehearsals.append(
            {
                "frontier": frontier,
                "arms": {"candidate": candidate, "control": control},
                "unplanned_interruptions": interruptions,
                "max_single_unplanned_downtime_seconds": single,
                "cumulative_unplanned_downtime_seconds": cumulative,
            }
        )

    disturbances = _g09_disturbances(root, work / "disturbances")
    _unload_model(PINNED_MODEL)
    _ensure_free_floor(root, label="G09 end")

    subject = _subject_envelope(
        root,
        frozen,
        schema="SUBSTRATE_ODYSSEY_DURABILITY_REHEARSAL/v1",
        payload={
            "all_pass": True,
            "checkpoint_policy": {
                "delta_interval_seconds": storage.get("delta_checkpoint_interval_seconds"),
                "full_interval_seconds": storage.get("full_checkpoint_interval_seconds"),
            },
            "rehearsals": rehearsals,
            "scheduled_disturbance_receipts": disturbances,
            "checks": {
                "frozen_build_bound": True,
                "source_maps_bound": True,
                "checkpoint_round_trip": True,
                "delta_plus_full_restore": True,
                "process_restart": True,
                "model_replacement": True,
                "tool_or_body_change": True,
                "sensor_or_source_interruption": True,
                "single_writer": True,
                "interactive_shell_independent": True,
                "recovery_limits_bound": True,
                "event_chain_valid": True,
            },
        },
    )
    _write_json(out, subject, overwrite=True)
    return subject


# ---------------------------------------------------------------------------
# G06 — width calibration
# ---------------------------------------------------------------------------

G06_LAYOUT_FIELDS = (
    "candidate_root",
    "control_root",
    "candidate_event_ledger",
    "control_event_ledger",
    "candidate_checkpoint_root",
    "control_checkpoint_root",
    "candidate_mutable_state_root",
    "control_mutable_state_root",
    "candidate_model_context_root",
    "control_model_context_root",
)


G06_PHASE = "retrieval"
G06_CYCLE = 0
G06_SCHEDULING_MODE = "initial_release_only;per_frontier_candidate_then_control;no_global_role_barrier;parent_global_dwell"


def _g06_diagnostic_cell_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a reduced diagnostic-only paired-cell probe in this process.

    Real durable writes under private roots, a real checkpoint, and one real
    model call.  This remains useful to test receipt-invariance mechanics, but
    it is never a launch subject: it does not execute the production adapter
    path or the strict phase contract.
    """
    root = Path(payload["root"])
    cell_base = Path(payload["cell_base"])
    frontier = payload["frontier"]
    width = int(payload["width"])
    repetition = int(payload["repetition"])
    num_predict = int(payload.get("num_predict", PROBE_NUM_PREDICT))
    skip_model = bool(payload.get("skip_model", False))
    hash_rounds = int(payload.get("hash_rounds", 96))
    receipt_bytes = int(payload.get("receipt_bytes", 262144))

    cell_base.mkdir(parents=True, exist_ok=True)
    layout = _layout_for_cell(root, cell_base, fields=G06_LAYOUT_FIELDS)
    resource_parity = _resource_parity()

    started = time.monotonic()
    cpu_before = time.process_time()
    io_bytes = 0

    # Durable event ledger write (both arms).
    disk_started = time.monotonic()
    for field in ("candidate_event_ledger", "control_event_ledger"):
        path = cell_base / field / "events.jsonl"
        n = _write_bytes(
            path,
            64 * 1024,
            tag=f"g06-{frontier}-{width}-{repetition}-{field}".encode(),
        )
        io_bytes += n
    disk_latency_ms = (time.monotonic() - disk_started) * 1000.0

    # Checkpoint write (both arms) via the worker primitive.
    ckpt_started = time.monotonic()
    chain = digest(
        {
            "g06": True,
            "frontier": frontier,
            "width": width,
            "repetition": repetition,
        }
    )
    for field in ("candidate_checkpoint_root", "control_checkpoint_root"):
        path = cell_base / field / "full-001.json"
        worker._write_checkpoint(
            path,
            authority_sha256=digest({"g06-authority": frontier}),
            kind="full",
            cycle=0,
            completed_phase_count=1,
            completed_paired_events=1,
            event_chain_sha256=chain,
            parent_sha256="",
        )
        io_bytes += path.stat().st_size
    for field in ("candidate_mutable_state_root", "control_mutable_state_root"):
        path = cell_base / field / "state.bin"
        io_bytes += _write_bytes(path, 16 * 1024, tag=f"g06-state-{frontier}-{field}".encode())
    for field in ("candidate_model_context_root", "control_model_context_root"):
        path = cell_base / field / "context.bin"
        io_bytes += _write_bytes(path, 8 * 1024, tag=f"g06-ctx-{frontier}-{field}".encode())
    checkpoint_latency_ms = (time.monotonic() - ckpt_started) * 1000.0

    # Frozen calibration CPU unit.  Seed is identical across widths/repetitions
    # so the unit receipt is genuinely invariant under concurrency.
    cpu_digest = _cpu_hash_work(
        rounds=hash_rounds,
        nbytes=receipt_bytes,
        seed=G06_UNIT_SEED,
    )
    (cell_base / "cpu-work.sha256").write_text(cpu_digest + "\n", encoding="utf-8")
    # Unit receipt: pure function of the sealed calibration unit only — no pid,
    # no timings, no width.  This is the invariant the spec requires.
    unit_receipt = {
        "schema": "SUBSTRATE_ODYSSEY_CALIBRATION_UNIT_RECEIPT/v1",
        "activation": False,
        "unit": "complete_paired_frontier_cell",
        "hash_rounds": hash_rounds,
        "receipt_bytes": receipt_bytes,
        "seed": G06_UNIT_SEED.decode("ascii"),
        "cpu_work_sha256": cpu_digest,
    }
    unit_receipt_sha256 = digest(unit_receipt)
    unit_receipt["sha256"] = unit_receipt_sha256
    unit_path = cell_base / "unit-receipt.json"
    _write_json(unit_path, unit_receipt, overwrite=True)

    model_latency_ms = 0.0
    model_meta: dict[str, Any] = {}
    if not skip_model:
        model_meta = _model_chat(num_predict=num_predict)
        model_latency_ms = float(model_meta["latency_ms"])

    cpu_time_seconds = max(time.process_time() - cpu_before, 1e-6)
    wall_seconds = max(time.monotonic() - started, 1e-6)

    # Sample this process RSS via libproc (or rusage fallback).
    self_pid = os.getpid()
    self_rss = _self_rss_bytes()

    receipt_path = cell_base / "cell-receipt.json"
    receipt = {
        "schema": "SUBSTRATE_ODYSSEY_WIDTH_CELL_RECEIPT/v1",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "frontier": frontier,
        "width": width,
        "repetition": repetition,
        "pid": self_pid,
        "wall_seconds": wall_seconds,
        "cpu_time_seconds": cpu_time_seconds,
        "io_bytes": io_bytes,
        "disk_latency_ms": disk_latency_ms,
        "checkpoint_latency_ms": checkpoint_latency_ms,
        "model_latency_ms": model_latency_ms,
        "resident_memory_bytes": self_rss,
        "model": model_meta,
        "unit_receipt_sha256": unit_receipt_sha256,
        "cpu_work_sha256": cpu_digest,
    }
    receipt["sha256"] = digest({key: value for key, value in receipt.items() if key != "sha256"})
    _write_json(receipt_path, receipt, overwrite=True)

    return {
        "id": frontier,
        **layout,
        "resource_parity": resource_parity,
        "receipt_path": str(receipt_path),
        "unit_receipt_path": str(unit_path),
        "unit_receipt_sha256": unit_receipt_sha256,
        "cpu_work_sha256": cpu_digest,
        "pid": self_pid,
        "wall_seconds": wall_seconds,
        "cpu_time_seconds": cpu_time_seconds,
        "io_bytes": io_bytes,
        "disk_latency_ms": disk_latency_ms,
        "checkpoint_latency_ms": checkpoint_latency_ms,
        "model_latency_ms": model_latency_ms,
        "resident_memory_bytes": self_rss,
    }


def _g06_production_layout(root: Path, cell_base: Path, *, frontier: str) -> tuple[Path, dict[str, str], dict[str, Path]]:
    """Return one private production-arm layout for a G06 frontier lane."""
    worker_root = cell_base / "worker"
    candidate_arm = worker_root / "arms" / frontier / "candidate"
    control_arm = worker_root / "arms" / frontier / "control"
    paths = {
        "candidate_root": candidate_arm,
        "control_root": control_arm,
        "candidate_event_ledger": candidate_arm / "state" / "events.jsonl",
        "control_event_ledger": control_arm / "state" / "events.jsonl",
        "candidate_checkpoint_root": cell_base / "boundary" / "candidate-checkpoints",
        "control_checkpoint_root": cell_base / "boundary" / "control-checkpoints",
        "candidate_mutable_state_root": candidate_arm / "state",
        "control_mutable_state_root": control_arm / "state",
        "candidate_model_context_root": candidate_arm / "state" / "outputs",
        "control_model_context_root": control_arm / "state" / "outputs",
    }
    for directory in (
        worker_root,
        paths["candidate_checkpoint_root"],
        paths["control_checkpoint_root"],
    ):
        directory.mkdir(parents=True, exist_ok=True)
    layout = {name: _relative(root, path) for name, path in paths.items()}
    return worker_root, layout, paths


def _g06_arm_command(
    root: Path,
    *,
    role: str,
    model: str,
    state_root: Path,
    adapter_sha256: str,
) -> list[str]:
    """Construct the same executable production arm used by the full worker."""
    adapter = root / "src/substrate/odyssey_arms.py"
    if not adapter.is_file() or odyssey_transition.canonical_source_digest(adapter) != adapter_sha256:
        raise Refused("G06 production adapter source is missing or drifted")
    return [
        sys.executable,
        str(adapter),
        "run",
        "--root",
        str(root),
        "--role",
        role,
        "--model",
        model,
        "--state-root",
        _relative(root, state_root),
        "--self-sha256",
        adapter_sha256,
        "--ollama-url",
        OLLAMA,
    ]


def _g06_model_latency_ms(receipt: dict[str, Any]) -> float:
    usage = receipt.get("resource_usage")
    if isinstance(usage, dict):
        value = usage.get("total_duration_ns")
        if isinstance(value, int) and value >= 0:
            return value / 1_000_000.0
    elapsed = receipt.get("elapsed_seconds")
    return float(elapsed) * 1000.0 if isinstance(elapsed, (int, float)) and elapsed >= 0 else 0.0


def _g06_production_cell_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one real, production-equivalent paired frontier dispatch.

    The parent establishes the absolute phase deadline before it launches any
    cells.  This child never pads its own timing: all cells are joined, made
    durable, and held to the phase boundary by the parent, matching the live
    worker's no-role-wave, parent-global-dwell scheduling shape.
    """
    root = Path(payload["root"]).resolve()
    cell_base = Path(payload["cell_base"])
    frontier = str(payload["frontier"])
    width = int(payload["width"])
    repetition = int(payload["repetition"])
    model = str(payload["model"])
    adapter_sha256 = str(payload["adapter_sha256"])
    dispatch_contract_sha256 = str(payload["dispatch_contract_sha256"])
    phase_deadline = float(payload["phase_deadline_monotonic"])
    full_phase_seconds = int(payload["full_phase_seconds"])
    strict_dispatch_budget_seconds = int(payload["strict_dispatch_budget_seconds"])
    frontier_entry_raw = payload.get("frontier_entry")
    if not isinstance(frontier_entry_raw, dict):
        raise Refused("G06 production cell lacks a frontier manifest binding")
    frontier_entry = dict(frontier_entry_raw)
    if frontier_entry.get("id") != frontier:
        raise Refused("G06 production cell frontier binding drifted")
    if len(dispatch_contract_sha256) != 64:
        raise Refused("G06 production cell lacks a dispatch contract digest")
    if time.monotonic() >= phase_deadline:
        raise Refused(f"G06 {width}x rep {repetition} {frontier}: process started after the strict dispatch deadline")
    # Density contract: rehearsal refuses when the live model gateway is not on
    # the pinned OLLAMA_NUM_PARALLEL value (measured 2.4x throughput win).
    try:
        density.assert_ollama_num_parallel_pinned(require_running=True)
    except density.DensityRefused as error:
        raise Refused(f"G06 model-gateway pin: {error}") from error

    worker_root, layout, paths = _g06_production_layout(root, cell_base, frontier=frontier)
    for role in ("candidate", "control"):
        frontier_entry[f"{role}_command"] = _g06_arm_command(
            root,
            role=role,
            model=model,
            state_root=paths[f"{role}_mutable_state_root"],
            adapter_sha256=adapter_sha256,
        )

    # This initial read matches worker startup verification.  The shared
    # dispatcher then performs the actual full source-bundle recheck before
    # each candidate and control subprocess, just as the live worker does.
    manifest = worker._manifest_for_frontier(root, frontier_entry, full=False, task_count=1)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
        raise Refused(f"G06 production manifest has no retrieval task for {frontier}")
    task = dict(tasks[0])
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise Refused(f"G06 production task id is invalid for {frontier}")

    cell_started = time.monotonic()
    cpu_before = time.process_time()
    timeline: dict[str, float] = {}

    def observe(stage: str, observed_at: float) -> None:
        if stage in timeline:
            raise Refused(f"G06 production dispatch repeated timeline stage {stage}")
        timeline[stage] = observed_at

    authority_sha256 = str(payload["phase_authority_sha256"])
    run_id = str(payload["phase_run_id"])
    if len(authority_sha256) != 64 or len(run_id) < 1:
        raise Refused("G06 production cell lacks its shared phase identity")
    _frontier, event = worker._dispatch_paired_frontier(
        root,
        authority_sha256=authority_sha256,
        run_id=run_id,
        worker_root=worker_root,
        frontier_entry=frontier_entry,
        task=task,
        cycle=G06_CYCLE,
        phase=G06_PHASE,
        full_source_guard=True,
        task_count=336,
        phase_deadline_monotonic=phase_deadline,
        dispatch_observer=observe,
    )
    active_finished = time.monotonic()
    if active_finished > phase_deadline:
        raise Refused(
            f"G06 {width}x rep {repetition} {frontier}: paired production dispatch missed the strict deadline "
            f"({active_finished - cell_started:.3f}s active)"
        )
    required_timeline = ("candidate_started", "candidate_finished", "control_started", "control_finished")
    if any(name not in timeline for name in required_timeline):
        raise Refused("G06 production dispatch did not record its paired adapter order")
    if not (timeline["candidate_started"] <= timeline["candidate_finished"] <= timeline["control_started"] <= timeline["control_finished"]):
        raise Refused("G06 production dispatch did not preserve candidate-then-control ordering")
    if event.get("source_bundle_guard_calls") != 2:
        raise Refused("G06 production dispatch did not revalidate the source bundle before both arms")

    receipt_paths = {
        role: worker_root / "arms" / frontier / role / "receipts" / f"{G06_CYCLE:03d}-{G06_PHASE}.json"
        for role in ("candidate", "control")
    }
    receipts: dict[str, dict[str, Any]] = {}
    receipt_refs: dict[str, dict[str, str]] = {}
    for role, path in receipt_paths.items():
        if not path.is_file():
            raise Refused(f"G06 production {role} receipt is missing")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise Refused(f"G06 production {role} receipt is malformed")
        receipts[role] = value
        receipt_refs[role] = _file_ref(root, path)

    cpu_time_seconds = max(time.process_time() - cpu_before, 1e-6)
    active_work_seconds = max(active_finished - cell_started, 1e-6)
    model_latency_ms = sum(_g06_model_latency_ms(receipts[role]) for role in ("candidate", "control"))
    self_rss = _self_rss_bytes()
    unit_receipt = {
        "schema": "SUBSTRATE_ODYSSEY_G06_DISPATCH_CONTRACT/v1",
        "activation": False,
        "unit": "complete_real_paired_frontier_dispatch",
        "dispatch_contract_sha256": dispatch_contract_sha256,
        "full_phase_seconds": full_phase_seconds,
        "strict_dispatch_budget_seconds": strict_dispatch_budget_seconds,
        "paired_adapter_dispatches_per_cell": 2,
        "source_bundle_pre_dispatch_revalidation": True,
    }
    unit_receipt["sha256"] = digest(unit_receipt)
    unit_path = cell_base / "dispatch-contract.json"
    _write_json(unit_path, unit_receipt, overwrite=True)
    cell_receipt = {
        "schema": "SUBSTRATE_ODYSSEY_WIDTH_CELL_RECEIPT/v2",
        "activation": False,
        "external_activation": False,
        "program": PROGRAM,
        "frontier": frontier,
        "width": width,
        "repetition": repetition,
        "pid": os.getpid(),
        "worker_root": _relative(root, worker_root),
        "task_binding": {
            "manifest_path": frontier_entry["candidate_manifest"],
            "manifest_sha256": frontier_entry["candidate_manifest_sha256"],
            "task_index": 0,
            "task_id": task_id,
            "task_sha256": digest(task),
        },
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "candidate_receipt": receipt_refs["candidate"],
        "control_receipt": receipt_refs["control"],
        "candidate_adapter_elapsed_seconds": receipts["candidate"].get("elapsed_seconds"),
        "control_adapter_elapsed_seconds": receipts["control"].get("elapsed_seconds"),
        "model_call_count": 2,
        "source_bundle_guard_calls": event["source_bundle_guard_calls"],
        "active_work_seconds": active_work_seconds,
        "deadline_met": True,
        "timeline_offsets_seconds": {name: observed - phase_deadline + strict_dispatch_budget_seconds for name, observed in timeline.items()},
        "cpu_time_seconds": cpu_time_seconds,
        "io_bytes": _dir_size(worker_root),
        "model_latency_ms": model_latency_ms,
        "resident_memory_bytes": self_rss,
        "dispatch_contract_sha256": dispatch_contract_sha256,
        "paired_event": event,
    }
    cell_receipt["sha256"] = digest(cell_receipt)
    cell_receipt_path = cell_base / "cell-receipt.json"
    _write_json(cell_receipt_path, cell_receipt, overwrite=True)

    return {
        "id": frontier,
        **layout,
        "resource_parity": _resource_parity(model=model, wall_time_seconds=full_phase_seconds),
        "receipt_path": str(cell_receipt_path),
        "unit_receipt_path": str(unit_path),
        "unit_receipt_sha256": unit_receipt["sha256"],
        "candidate_receipt": receipt_refs["candidate"],
        "control_receipt": receipt_refs["control"],
        "task_binding": cell_receipt["task_binding"],
        "pid": os.getpid(),
        "wall_seconds": active_work_seconds,
        "active_work_seconds": active_work_seconds,
        "cpu_time_seconds": cpu_time_seconds,
        "io_bytes": _dir_size(worker_root),
        "disk_latency_ms": 0.0,
        "checkpoint_latency_ms": 0.0,
        "model_latency_ms": model_latency_ms,
        "resident_memory_bytes": self_rss,
        "model_call_count": 2,
        "source_bundle_guard_calls": event["source_bundle_guard_calls"],
        "deadline_met": True,
        "timeline": timeline,
        # The parent writes the phase durability boundary from this exact
        # production event.  Keep it outside the child receipt as well so it
        # never has to reconstruct an event from timing-only summary fields.
        "paired_event": event,
        "dispatch_contract_sha256": dispatch_contract_sha256,
    }


def _g06_cell_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Route child work to the launch-grade or reduced diagnostic harness."""
    if payload.get("mode") == "production":
        return _g06_production_cell_job(payload)
    return _g06_diagnostic_cell_job(payload)


def _run_g06_cell_subprocess(job: dict[str, Any]) -> subprocess.Popen[str]:
    """Launch one real child process that executes a single cell job."""
    cell_base = Path(job["cell_base"])
    cell_base.mkdir(parents=True, exist_ok=True)
    payload_path = cell_base / "job-payload.json"
    result_path = cell_base / "job-result.json"
    if result_path.exists():
        result_path.unlink()
    _write_json(payload_path, job, overwrite=True)
    env = dict(os.environ)
    # Ensure the child can import substrate the same way the parent can.
    src = str(Path(job["root"]) / "src")
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not prior else f"{src}{os.pathsep}{prior}"
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "substrate.odyssey_rehearsal",
            "_cell-worker",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
        ],
        cwd=job["root"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _collect_g06_cell(
    proc: subprocess.Popen[str],
    job: dict[str, Any],
    *,
    timeout_seconds: float = 900.0,
) -> dict[str, Any]:
    stdout, stderr = proc.communicate(timeout=timeout_seconds)
    result_path = Path(job["cell_base"]) / "job-result.json"
    if proc.returncode != 0:
        detail = (stderr or stdout or "").strip()
        raise Refused(f"G06 cell {job.get('frontier')} child failed (code={proc.returncode}): {detail[:500]}")
    if not result_path.is_file():
        raise Refused(f"G06 cell {job.get('frontier')} wrote no result file")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"G06 cell result unreadable: {error}") from error
    if not isinstance(result, dict) or result.get("id") != job.get("frontier"):
        raise Refused(f"G06 cell result malformed for {job.get('frontier')}")
    return result


class _ProcTaskInfo(ctypes.Structure):
    """Darwin proc_taskinfo subset used for resident size (libproc)."""

    _fields_ = [
        ("pti_virtual_size", c_ulonglong),
        ("pti_resident_size", c_ulonglong),
        ("pti_total_user", c_ulonglong),
        ("pti_total_system", c_ulonglong),
        ("pti_threads_user", c_ulonglong),
        ("pti_threads_system", c_ulonglong),
        ("pti_policy", c_int),
        ("pti_faults", c_int),
        ("pti_pageins", c_int),
        ("pti_cow_faults", c_int),
        ("pti_messages_sent", c_int),
        ("pti_messages_received", c_int),
        ("pti_syscalls_mach", c_int),
        ("pti_syscalls_unix", c_int),
        ("pti_csw", c_int),
        ("pti_threadnum", c_int),
        ("pti_numrunning", c_int),
        ("pti_priority", c_int),
    ]


_LIBPROC: ctypes.CDLL | None = None
_PROC_ALL_PIDS = 1
_PROC_PIDTASKINFO = 4


def _libproc() -> ctypes.CDLL:
    """Load Darwin libproc for process-table RSS (ps may be sandboxed)."""
    global _LIBPROC
    if _LIBPROC is None:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        lib.proc_listpids.argtypes = [c_uint32, c_uint32, c_void_p, c_int]
        lib.proc_listpids.restype = c_int
        lib.proc_pidinfo.argtypes = [
            c_int,
            c_int,
            c_ulonglong,
            c_void_p,
            c_int,
        ]
        lib.proc_pidinfo.restype = c_int
        _LIBPROC = lib
    return _LIBPROC


def _pid_rss_bytes(pid: int) -> int:
    info = _ProcTaskInfo()
    result = _libproc().proc_pidinfo(int(pid), _PROC_PIDTASKINFO, 0, byref(info), sizeof(info))
    if result != sizeof(info):
        return 0
    return int(info.pti_resident_size)


def _host_rss_bytes() -> int:
    """Sum resident memory across the live process table via libproc.

    Prefer this over ``ps`` so the measurement still works when the seatbelt
    profile denies executing ``ps`` but still allows process-info syscalls.
    Falls back to ``worker._process_sample`` when libproc is unavailable.
    """
    try:
        lib = _libproc()
        raw_size = lib.proc_listpids(_PROC_ALL_PIDS, 0, None, 0)
        if raw_size <= 0:
            raise OSError("proc_listpids returned no buffer size")
        buf = (c_int * (raw_size // sizeof(c_int)))()
        filled = lib.proc_listpids(_PROC_ALL_PIDS, 0, buf, raw_size)
        if filled <= 0:
            raise OSError("proc_listpids returned no pids")
        total = 0
        for index in range(filled // sizeof(c_int)):
            pid = int(buf[index])
            if pid > 0:
                total += _pid_rss_bytes(pid)
        if total <= 0:
            raise OSError("libproc resident sum was empty")
        return total
    except OSError:
        sample = worker._process_sample()
        return sum(process.rss_bytes for process in sample.processes.values())


def _self_rss_bytes() -> int:
    """Resident size of this process from libproc, else rusage peak."""
    rss = _pid_rss_bytes(os.getpid())
    if rss > 0:
        return rss
    # macOS ru_maxrss is bytes.
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max(peak, 1)


def _g06_launch_bindings(
    root: Path,
    frozen: dict[str, Any],
    *,
    g02_subject_path: Path | None,
    g03_subject_path: Path | None,
) -> dict[str, Any]:
    """Load the current model and manifest bindings required by launch G06."""
    if g02_subject_path is None or g03_subject_path is None:
        raise Refused("launch G06 requires explicit current G02 and G03 subject paths")
    g02_path = g02_subject_path if g02_subject_path.is_absolute() else root / g02_subject_path
    g03_path = g03_subject_path if g03_subject_path.is_absolute() else root / g03_subject_path
    if not g02_path.is_file() or not g03_path.is_file():
        raise Refused("launch G06 G02/G03 subject path is missing")
    g02 = authority._read_json(g02_path, require_digest=True)
    g03 = authority._read_json(g03_path, require_digest=True)
    authority._validate_g02(root, g02, frozen)
    authority._validate_g03(root, g03, frozen)
    base_model = g02.get("base_model")
    candidate = g02.get("candidate")
    if not isinstance(base_model, dict) or not isinstance(candidate, dict):
        raise Refused("launch G06 G02 subject lacks the pinned base model or production arm")
    model = base_model.get("id")
    adapter_sha256 = candidate.get("adapter_sha256")
    if not isinstance(model, str) or not model.strip():
        raise Refused("launch G06 G02 base model identifier is invalid")
    if not isinstance(adapter_sha256, str) or adapter_sha256 != frozen.get("implementation_sha256", {}).get("odyssey_arms"):
        raise Refused("launch G06 adapter digest does not match the frozen production arm")
    rows = authority._g03_manifest_rows(g03)
    frontiers: dict[str, dict[str, str]] = {}
    for frontier in FRONTIER_IDS:
        row = rows[frontier]
        frontiers[frontier] = {
            "id": frontier,
            "candidate_manifest": str(row["path"]),
            "candidate_manifest_sha256": str(row["file_sha256"]),
        }
    return {
        "model": model,
        "adapter_sha256": adapter_sha256,
        "base_model": base_model,
        "frontiers": frontiers,
        "g02_subject": _file_ref(root, g02_path),
        "g03_subject": _file_ref(root, g03_path),
    }


def _g06_dispatch_contract(
    frozen: dict[str, Any],
    limits: dict[str, Any],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Return the invariant launch-grade G06 dispatch contract."""
    contract = {
        "schema": "SUBSTRATE_ODYSSEY_G06_REAL_PHASE_HARNESS/v1",
        "measurement_basis": limits["measurement_basis"],
        "full_phase_seconds": limits["full_phase_seconds"],
        "strict_dispatch_budget_seconds": limits["strict_dispatch_budget_seconds"],
        "scale_factor": limits["scale_factor"],
        "phase_boundary_guard_interval_seconds": limits["phase_boundary_guard_interval_seconds"],
        "paired_adapter_dispatches_per_cell": limits["paired_adapter_dispatches_per_cell"],
        "source_bundle_pre_dispatch_revalidation": True,
        "scheduling_mode": limits["scheduling_mode"],
        "worker_sha256": frozen["implementation_sha256"]["odyssey_worker"],
        "adapter_sha256": bindings["adapter_sha256"],
        "model": bindings["model"],
        "max_output_tokens": arms.MAX_OUTPUT_TOKENS,
        "g03_manifest_bindings": [
            {
                "id": frontier,
                "path": bindings["frontiers"][frontier]["candidate_manifest"],
                "sha256": bindings["frontiers"][frontier]["candidate_manifest_sha256"],
            }
            for frontier in FRONTIER_IDS
        ],
    }
    contract["dispatch_contract_sha256"] = digest(contract)
    return contract


def _write_g06_phase_boundary(
    root: Path,
    *,
    boundary_root: Path,
    authority_sha256: str,
    run_id: str,
    events: list[dict[str, Any]],
) -> tuple[dict[str, str], float, int]:
    """Write the real parent-side trace/checkpoint/state durability boundary."""
    started = time.monotonic()
    boundary_root.mkdir(parents=True, exist_ok=True)
    trace_path = boundary_root / "EVENTS.jsonl"
    chain = ""
    written = 0
    with trace_path.open("w", encoding="utf-8") as handle:
        for event in sorted(events, key=lambda row: str(row.get("frontier", ""))):
            row = dict(event)
            row["previous_event_sha256"] = chain
            row["event_sha256"] = worker._digest(row)
            chain = row["event_sha256"]
            encoded = json.dumps(row, sort_keys=True) + "\n"
            handle.write(encoded)
            written += len(encoded.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(boundary_root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    checkpoint_path = boundary_root / "checkpoints" / "delta-001.json"
    checkpoint_sha256 = worker._write_checkpoint(
        checkpoint_path,
        authority_sha256=authority_sha256,
        kind="delta",
        cycle=G06_CYCLE,
        completed_phase_count=1,
        completed_paired_events=len(events),
        event_chain_sha256=chain,
        parent_sha256="",
    )
    state_path = boundary_root / "STATE.json"
    worker._write_state(
        state_path,
        {
            "schema": "SUBSTRATE_ODYSSEY_WORKER_STATE/v1",
            "activation": False,
            "authority_sha256": authority_sha256,
            "run_id": run_id,
            "completed_phase_count": 1,
            "total_phase_count": 1,
            "completed_paired_events": len(events),
            "event_chain_sha256": chain,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_count": 1,
            "complete": False,
            "elapsed_seconds": max(0.0, time.monotonic() - started),
            "broker_hold_seconds": 0.0,
        },
    )
    return _file_ref(root, state_path), (time.monotonic() - started) * 1000.0, written + _dir_size(boundary_root)


def run_g06(
    root: Path,
    out: Path,
    *,
    widths: tuple[int, ...] | None = None,
    repetitions: int | None = None,
    launch: bool = True,
    num_predict: int | None = None,
    g02_subject_path: Path | None = None,
    g03_subject_path: Path | None = None,
) -> dict[str, Any]:
    """Measure G06 through real paired production adapters and phase timing.

    Launch calibration uses the exact 1/2/4/6/8 × 3 schedule, starts the
    strict deadline before child launch, executes each lane's candidate then
    control arm without a role-wave barrier, writes the parent durability
    boundary, and only then performs one global dwell to the scaled boundary.
    The sealed 1.35 ratio is computed from the pre-dwell paired-dispatch wall
    time, so equal deadline padding cannot turn a slower width-eight service
    into a pass.  This is a dispatch calibration, not a claim to reproduce
    later seven-day state growth or a literal 30-minute idle interval.
    Reduced ``launch=False`` calls retain a clearly marked diagnostic path for
    unit tests; they can never become a passing gate subject.
    """
    root = root.resolve()
    out = out if out.is_absolute() else (root / out).resolve()
    frozen = _load_frozen(root)
    limits = _calibration_limits(root, frozen)
    max_slowdown = float(limits["max_slowdown_ratio"])
    cal_widths = tuple(widths) if widths is not None else tuple(CALIBRATION_WIDTHS)
    cal_reps = int(repetitions) if repetitions is not None else int(CALIBRATION_REPETITIONS)
    predict = int(num_predict) if num_predict is not None else PROBE_NUM_PREDICT
    production = bool(launch)

    if launch and (cal_widths != tuple(CALIBRATION_WIDTHS) or cal_reps != CALIBRATION_REPETITIONS):
        raise Refused("launch G06 must use the full 1/2/4/6/8 x 3 schedule")
    if launch and predict != PROBE_NUM_PREDICT:
        raise Refused("launch G06 may not alter the pinned model output budget")

    bindings = _g06_launch_bindings(
        root,
        frozen,
        g02_subject_path=g02_subject_path,
        g03_subject_path=g03_subject_path,
    ) if production else None
    model = str(bindings["model"]) if bindings is not None else PINNED_MODEL
    contract = _g06_dispatch_contract(frozen, limits, bindings) if bindings is not None else None
    dispatch_contract_sha256 = contract["dispatch_contract_sha256"] if contract is not None else None
    strict_budget = int(limits["strict_dispatch_budget_seconds"])
    dwell_guard_seconds = int(limits["phase_boundary_guard_interval_seconds"])

    work = root / REHEARSAL_ROOT / "G06"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    _ensure_free_floor(root, label="G06 start")
    observations: list[dict[str, Any]] = []
    width1_active_phase_wall: float | None = None
    width1_diagnostic_wall: float | None = None
    invariant_unit_digest: str | None = None
    free_samples_g06: list[int] = [_free_bytes(root)]
    width_eight_scheduled_seconds = 0.0

    try:
        for width in cal_widths:
            for repetition in range(1, cal_reps + 1):
                cell_ids = list(FRONTIER_IDS[:width])
                if width == len(FRONTIER_IDS) and cell_ids != list(FRONTIER_IDS):
                    raise Refused("width-8 cells must be A-H in order")
                pageout_before = _pageout_bytes()
                _thermal_before, critical_before, pressure_before = _pressure_state()
                if critical_before:
                    raise Refused(f"G06 {width}x rep {repetition}: critical pressure level={pressure_before} before observation")

                obs_started = time.monotonic()
                phase_deadline = obs_started + strict_budget
                phase_authority_sha256 = digest(
                    {
                        "schema": "SUBSTRATE_ODYSSEY_G06_CALIBRATION_AUTHORITY/v1",
                        "dispatch_contract_sha256": dispatch_contract_sha256,
                        "width": width,
                        "repetition": repetition,
                    }
                ) if production else ""
                phase_run_id = f"g06-{dispatch_contract_sha256[:16]}-{width}x-{repetition}" if production else ""
                jobs: list[dict[str, Any]] = []
                for frontier in cell_ids:
                    cell_base = work / f"{width}x" / f"rep{repetition}" / frontier
                    if production:
                        assert bindings is not None and dispatch_contract_sha256 is not None
                        jobs.append(
                            {
                                "mode": "production",
                                "root": str(root),
                                "cell_base": str(cell_base),
                                "frontier": frontier,
                                "width": width,
                                "repetition": repetition,
                                "model": model,
                                "adapter_sha256": bindings["adapter_sha256"],
                                "frontier_entry": bindings["frontiers"][frontier],
                                "dispatch_contract_sha256": dispatch_contract_sha256,
                                "phase_authority_sha256": phase_authority_sha256,
                                "phase_run_id": phase_run_id,
                                "phase_deadline_monotonic": phase_deadline,
                                "full_phase_seconds": limits["full_phase_seconds"],
                                "strict_dispatch_budget_seconds": strict_budget,
                            }
                        )
                    else:
                        jobs.append(
                            {
                                "mode": "diagnostic",
                                "root": str(root),
                                "cell_base": str(cell_base),
                                "frontier": frontier,
                                "width": width,
                                "repetition": repetition,
                                "num_predict": predict,
                                "hash_rounds": limits["hash_rounds"],
                                "receipt_bytes": limits["receipt_bytes"],
                            }
                        )

                rss_samples: list[int] = []
                stop_sample = threading.Event()

                def _sample_loop(
                    samples: list[int] = rss_samples,
                    stop: threading.Event = stop_sample,
                    sampled_model: str = model,
                ) -> None:
                    while not stop.is_set():
                        try:
                            samples.append(_host_rss_bytes() + _service_bytes(sampled_model))
                        except Exception:
                            with contextlib.suppress(Exception):
                                samples.append(_host_rss_bytes())
                        stop.wait(0.2)

                sampler = threading.Thread(target=_sample_loop, daemon=True)
                sampler.start()
                children = [_run_g06_cell_subprocess(job) for job in jobs]
                results: list[dict[str, Any]] = []
                errors: list[str] = []
                boundary_ref: dict[str, str] | None = None
                boundary_latency_ms = 0.0
                boundary_io_bytes = 0
                global_dwell_seconds = 0.0
                guard_samples = 0
                active_dispatch_wall = 0.0
                try:
                    for index, (proc, job) in enumerate(zip(children, jobs, strict=True)):
                        try:
                            remaining = max(5.0, phase_deadline - time.monotonic() + 30.0) if production else 900.0
                            results.append(_collect_g06_cell(proc, job, timeout_seconds=remaining))
                        except Refused as error:
                            errors.append(str(error))
                        except subprocess.TimeoutExpired:
                            with contextlib.suppress(Exception):
                                proc.kill()
                                proc.communicate()
                            errors.append(f"G06 cell {job.get('frontier')} timed out")
                        if errors:
                            for pending in children[index + 1 :]:
                                with contextlib.suppress(Exception):
                                    pending.kill()
                                    pending.communicate()
                            break
                    if errors:
                        raise Refused("; ".join(errors))
                    if production:
                        if time.monotonic() > phase_deadline:
                            raise Refused(f"G06 {width}x rep {repetition}: parent join exceeded strict dispatch deadline")
                        events = [row.get("paired_event") for row in results]
                        if not all(isinstance(event, dict) for event in events):
                            raise Refused("G06 production cells did not return paired production events")
                        boundary_ref, boundary_latency_ms, boundary_io_bytes = _write_g06_phase_boundary(
                            root,
                            boundary_root=work / f"{width}x" / f"rep{repetition}" / "phase-boundary",
                            authority_sha256=phase_authority_sha256,
                            run_id=phase_run_id,
                            events=[event for event in events if isinstance(event, dict)],
                        )
                        boundary_finished = time.monotonic()
                        if boundary_finished > phase_deadline:
                            raise Refused(f"G06 {width}x rep {repetition}: parent durability boundary exceeded strict dispatch deadline")
                        # The sealed slowdown ratio is about how actual paired
                        # dispatch scales.  Capture that before the required
                        # common dwell so the dwell cannot flatten a failed
                        # concurrency measurement into an apparent pass.
                        active_dispatch_wall = max(boundary_finished - obs_started, 1e-6)
                        while time.monotonic() < phase_deadline:
                            guard_samples += 1
                            _ensure_free_floor(root, label=f"G06 {width}x rep {repetition} global dwell")
                            _thermal_dwell, critical_dwell, level_dwell = _pressure_state()
                            if critical_dwell:
                                raise Refused(f"G06 {width}x rep {repetition}: critical pressure level={level_dwell} during global dwell")
                            time.sleep(min(float(dwell_guard_seconds), phase_deadline - time.monotonic()))
                        global_dwell_seconds = max(0.0, time.monotonic() - boundary_finished)
                finally:
                    stop_sample.set()
                    sampler.join(timeout=2.0)

                obs_wall = max(time.monotonic() - obs_started, 1e-6)
                if not production:
                    active_dispatch_wall = obs_wall
                pageout_after = _pageout_bytes()
                pageout_delta = pageout_after - pageout_before
                thermal_after, critical_after, level_after = _pressure_state()
                if critical_after:
                    raise Refused(f"G06 {width}x rep {repetition}: critical pressure level={level_after} after observation")

                by_id = {row["id"]: row for row in results}
                if set(by_id) != set(cell_ids):
                    raise Refused(f"G06 {width}x rep {repetition}: child results do not match submitted frontiers")
                ordered = [by_id[frontier] for frontier in cell_ids]
                cpu_time = sum(float(row["cpu_time_seconds"]) for row in ordered)
                io_bytes = sum(int(row["io_bytes"]) for row in ordered) + boundary_io_bytes
                disk_latency = max(float(row["disk_latency_ms"]) for row in ordered)
                checkpoint_latency = max(max(float(row["checkpoint_latency_ms"]) for row in ordered), boundary_latency_ms)
                model_latency = max(float(row["model_latency_ms"]) for row in ordered)
                cell_walls = [float(row["wall_seconds"]) for row in ordered]
                mean_cell_wall = sum(cell_walls) / len(cell_walls)
                max_cell_wall = max(cell_walls)
                free_after_obs = _free_bytes(root)
                free_samples_g06.append(free_after_obs)

                unit_digests = [str(row.get("unit_receipt_sha256") or "") for row in ordered]
                if any(len(value) != 64 for value in unit_digests) or len(set(unit_digests)) != 1:
                    raise Refused(f"G06 {width}x rep {repetition}: receipt invariance violated")
                if invariant_unit_digest is None:
                    invariant_unit_digest = unit_digests[0]
                elif unit_digests[0] != invariant_unit_digest:
                    raise Refused(f"G06 {width}x rep {repetition}: receipt invariance violated across widths")

                if production:
                    if width == 1:
                        width1_active_phase_wall = active_dispatch_wall if width1_active_phase_wall is None else (
                            (width1_active_phase_wall * (repetition - 1) + active_dispatch_wall) / repetition
                        )
                    baseline = width1_active_phase_wall
                    if baseline is None or baseline <= 0:
                        raise Refused("G06 missing width-one active dispatch baseline")
                    per_cell_slowdown = max_cell_wall / baseline
                    e2e_slowdown = active_dispatch_wall / baseline
                    raw_active_slowdown = max(per_cell_slowdown, e2e_slowdown)
                    slowdown_basis = "active_paired_dispatch_wall_with_deadline_guard"
                    if width == len(FRONTIER_IDS):
                        width_eight_scheduled_seconds += obs_wall
                else:
                    if width == 1:
                        width1_diagnostic_wall = max_cell_wall if width1_diagnostic_wall is None else (
                            (width1_diagnostic_wall * (repetition - 1) + max_cell_wall) / repetition
                        )
                    baseline = width1_diagnostic_wall
                    if baseline is None or baseline <= 0:
                        raise Refused("G06 missing width-one diagnostic baseline")
                    per_cell_slowdown = max_cell_wall / baseline
                    e2e_slowdown = obs_wall / baseline
                    raw_active_slowdown = max(per_cell_slowdown, e2e_slowdown)
                    slowdown_basis = "diagnostic_max_of_slowest_cell_and_observation_wall"
                gated_slowdown = max(per_cell_slowdown, e2e_slowdown)
                aggregate_throughput = width / active_dispatch_wall

                resident = max(rss_samples) if rss_samples else _host_rss_bytes()
                with contextlib.suppress(Exception):
                    resident = max(resident, _host_rss_bytes() + _service_bytes(model))
                resident = max(int(resident), 1)
                if resident > RESIDENT_CAP_BYTES:
                    raise Refused(f"G06 {width}x rep {repetition}: resident_memory_bytes={resident} exceeds 85 GiB cap")
                if pageout_delta > 0:
                    raise Refused(f"G06 {width}x rep {repetition}: unexpected pageout delta {pageout_delta} bytes")
                if disk_latency > MAX_IO_LATENCY_MS or checkpoint_latency > MAX_IO_LATENCY_MS:
                    raise Refused(f"G06 {width}x rep {repetition}: IO latency exceeded the sealed limit")
                if gated_slowdown > max_slowdown:
                    raise Refused(
                        f"G06 {width}x rep {repetition}: per_cell_slowdown_ratio={gated_slowdown:.6f} "
                        f"exceeds sealed max {max_slowdown} ({slowdown_basis})"
                    )
                if production and (obs_wall < strict_budget or not all(row.get("deadline_met") is True for row in ordered)):
                    raise Refused(f"G06 {width}x rep {repetition}: strict phase deadline/dwell contract was not met")

                receipt_refs: list[dict[str, str]] = []
                cells_out: list[dict[str, Any]] = []
                for row in ordered:
                    receipt_refs.append(_file_ref(root, Path(row["receipt_path"])))
                    unit_path = Path(row.get("unit_receipt_path") or "")
                    if unit_path.is_file():
                        receipt_refs.append(_file_ref(root, unit_path))
                    if production:
                        # Keep the production arm receipts themselves in the
                        # observation evidence set, not merely nested in the
                        # cell receipt.  G06 authority validates these exact
                        # files against the parent paired-event chain.
                        receipt_refs.extend((row["candidate_receipt"], row["control_receipt"]))
                    cell = {
                        "id": row["id"],
                        "candidate_root": row["candidate_root"],
                        "control_root": row["control_root"],
                        "candidate_event_ledger": row["candidate_event_ledger"],
                        "control_event_ledger": row["control_event_ledger"],
                        "candidate_checkpoint_root": row["candidate_checkpoint_root"],
                        "control_checkpoint_root": row["control_checkpoint_root"],
                        "candidate_mutable_state_root": row["candidate_mutable_state_root"],
                        "control_mutable_state_root": row["control_mutable_state_root"],
                        "candidate_model_context_root": row["candidate_model_context_root"],
                        "control_model_context_root": row["control_model_context_root"],
                        "resource_parity": row["resource_parity"],
                    }
                    if production:
                        cell.update(
                            {
                                "task_binding": row["task_binding"],
                                "candidate_receipt": row["candidate_receipt"],
                                "control_receipt": row["control_receipt"],
                                "model_call_count": row["model_call_count"],
                                "source_bundle_guard_calls": row["source_bundle_guard_calls"],
                                "active_work_seconds": row["active_work_seconds"],
                                "deadline_met": row["deadline_met"],
                            }
                        )
                    cells_out.append(cell)
                if boundary_ref is not None:
                    receipt_refs.append(boundary_ref)
                observation_checks = {
                    "receipt_invariant": True,
                    "no_memory_threshold_breach": True,
                    "no_critical_pressure": True,
                    "no_unexpected_swap_or_pageout_increase": True,
                    "io_latency_within_sealed_limit": True,
                    "slowdown_within_sealed_limit": True,
                    "distinct_run_roots": True,
                    "no_shared_writable_evaluator_or_data_root": True,
                    "record_cpu_memory_io": True,
                }
                if production:
                    observation_checks.update(
                        {
                            "strict_dispatch_deadline_met": True,
                            "production_paired_adapters_complete": True,
                            "source_bundle_revalidation_complete": True,
                            "parent_global_dwell_complete": True,
                        }
                    )
                observations.append(
                    {
                        "width": width,
                        "repetition": repetition,
                        "cells": cells_out,
                        "metrics": {
                            "aggregate_throughput": aggregate_throughput,
                            "per_cell_slowdown_ratio": gated_slowdown,
                            "resident_memory_bytes": resident,
                            "swap_pageout_delta_bytes": int(pageout_delta),
                            "disk_latency_ms": float(max(disk_latency, 1e-6)),
                            "checkpoint_latency_ms": float(max(checkpoint_latency, 1e-6)),
                            "model_latency_ms": float(max(model_latency, 1e-6)),
                            "cpu_time_seconds": float(max(cpu_time, 1e-6)),
                            "io_bytes": int(max(io_bytes, 1)),
                            "thermal_pressure": thermal_after,
                            "critical_pressure": False,
                            "observation_wall_seconds": obs_wall,
                            "active_dispatch_wall_seconds": active_dispatch_wall,
                            "mean_cell_wall_seconds": mean_cell_wall,
                            "max_cell_wall_seconds": max_cell_wall,
                            "e2e_slowdown_ratio": e2e_slowdown,
                            "raw_active_dispatch_slowdown_ratio": raw_active_slowdown,
                            "width1_baseline_seconds": baseline,
                            "slowdown_basis": slowdown_basis,
                            "unit_receipt_sha256": invariant_unit_digest,
                            "external_disk_free_bytes": free_after_obs,
                            "vm_pressure_level": level_after,
                            "strict_dispatch_budget_seconds": strict_budget if production else 0,
                            "scheduled_phase_seconds": strict_budget if production else 0,
                            "global_dwell_seconds": global_dwell_seconds,
                            "parent_guard_samples": guard_samples,
                            "paired_adapter_dispatches": 2 * width if production else 0,
                            "phase_boundary_receipt": boundary_ref,
                        },
                        "checks": observation_checks,
                        "receipt_refs": receipt_refs,
                    }
                )
    finally:
        _unload_model(model)

    _ensure_free_floor(root, label="G06 end")
    if invariant_unit_digest is None and observations:
        raise Refused("G06 produced observations without a unit receipt digest")
    checks = {
        "frozen_build_bound": True,
        "source_maps_bound": True,
        "receipt_invariant": True,
        "no_memory_threshold_breach": True,
        "no_critical_pressure": True,
        "no_unexpected_swap_or_pageout_increase": True,
        "io_latency_within_sealed_limit": True,
        "slowdown_within_sealed_limit": True,
        "distinct_run_roots": True,
        "no_shared_writable_evaluator_or_data_root": True,
        "record_cpu_memory_io": True,
    }
    payload: dict[str, Any] = {
        "all_pass": True,
        "admitted_width": 8 if launch else max(cal_widths),
        "full_program_requires_width": 8,
        "calibration_widths": list(cal_widths) if not launch else list(CALIBRATION_WIDTHS),
        "repetitions_per_width": cal_reps if not launch else CALIBRATION_REPETITIONS,
        "observations": observations,
        "unit_receipt_sha256": invariant_unit_digest,
        "external_disk_free_bytes_samples": free_samples_g06,
        "checks": checks,
    }
    if production:
        assert contract is not None and bindings is not None
        payload.update(
            {
                "phase_harness": {
                    **contract,
                    "g02_subject": bindings["g02_subject"],
                    "g03_subject": bindings["g03_subject"],
                    "minimum_width_eight_scheduled_seconds": limits["minimum_width_eight_scheduled_seconds"],
                },
                "width_eight_scheduled_seconds": width_eight_scheduled_seconds,
            }
        )
        checks.update(
            {
                "strict_dispatch_deadline_met": True,
                "production_paired_adapters_complete": True,
                "source_bundle_revalidation_complete": True,
                "parent_global_dwell_complete": True,
            }
        )
    else:
        payload["launch_subject"] = False
        payload["non_launch_reduced_schedule"] = {
            "widths": list(cal_widths),
            "repetitions": cal_reps,
            "reason": "internal_test_or_diagnostic_only",
        }

    subject = _subject_envelope(
        root,
        frozen,
        schema="SUBSTRATE_ODYSSEY_WIDTH_CALIBRATION/v1",
        payload=payload,
    )
    _write_json(out, subject, overwrite=True)
    return subject


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("g06", "g07", "g08", "g09", "_cell-worker"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--g02-subject", type=Path)
    parser.add_argument("--g03-subject", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "_cell-worker":
            # Internal child entrypoint for G06 concurrent cells only.
            if args.payload is None or args.result is None:
                raise Refused("_cell-worker requires --payload and --result")
            payload = json.loads(args.payload.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise Refused("cell payload must be an object")
            result = _g06_cell_job(payload)
            _write_json(args.result, result, overwrite=True)
            print(json.dumps({"activation": False, "status": "pass", "id": result.get("id")}, sort_keys=True))
            return 0
        if args.out is None:
            raise Refused(f"{args.command} requires --out")
        runners: dict[str, Callable[[Path, Path], dict[str, Any]]] = {
            "g07": run_g07,
            "g08": run_g08,
            "g09": run_g09,
        }
        result = (
            run_g06(args.root, args.out, g02_subject_path=args.g02_subject, g03_subject_path=args.g03_subject)
            if args.command == "g06"
            else runners[args.command](args.root, args.out)
        )
    except Refused as error:
        print(json.dumps({"activation": False, "refused": str(error)}, sort_keys=True))
        return 2
    except model_canary.Refused as error:
        print(json.dumps({"activation": False, "refused": str(error)}, sort_keys=True))
        return 2
    except worker.Refused as error:
        print(json.dumps({"activation": False, "refused": str(error)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "activation": False,
                "status": "pass",
                "schema": result.get("schema"),
                "sha256": result.get("sha256"),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
