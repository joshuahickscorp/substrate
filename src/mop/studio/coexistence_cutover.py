"""Deterministic evidence builder for the Hawking coexistence cutover.

The cutover is an operational safety receipt, never a scientific promotion.  A
one-time active-observer snapshot records the process-level observation that fixed
the v5 command-observer false positive.  The final v2 document remains unavailable
until the same v6 run publishes a sealed resumable receipt and the campaign appends
the matching ``resumable-leg`` event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import psutil

from ..config import REPO_ROOT
from . import campaign_supervisor as campaign
from . import external_coexistence as coexistence
from . import local_throttle as throttle

CUTOVER_SCHEMA = "mop-local-throttle-external-coexistence-cutover/v2"
ACTIVE_OBSERVER_SCHEMA = "mop-local-throttle-active-observer-evidence/v1"
READINESS_SCHEMA = "mop-local-throttle-external-coexistence-readiness/v1"
CUTOVER_SEAL = "cutover_sha256"
ACTIVE_OBSERVER_SEAL = "evidence_sha256"

V1_CUTOVER_PATH = Path("proof/LOCAL_THROTTLE_HAWKING_COEXISTENCE_CUTOVER_V1.json")
DEFAULT_OUTPUT_PATH = Path("proof/LOCAL_THROTTLE_HAWKING_COEXISTENCE_CUTOVER.json")
V6_CAMPAIGN_ID = "mac-studio-substrate-phase1-coexistence-10k-v6"
V6_STEP_ID = "edcm1_official"
V6_TASK_ID = "edcm1_official_cpu"
V6_SEED2_RUN_ID = "mac-studio-substrate-phase1-coexistence-10k-v6-edcm1_official_cpu-20260712T191606Z-leg01"
V6_CAMPAIGN_DIR = Path("runs/mac_studio_campaign") / V6_CAMPAIGN_ID
V6_STATUS_PATH = V6_CAMPAIGN_DIR / "current_status.json"
V6_EVENTS_PATH = V6_CAMPAIGN_DIR / "events.jsonl"
V6_ACTIVE_OBSERVER_PATH = V6_CAMPAIGN_DIR / "evidence/v6-active-observer.json"

V5_CAMPAIGN_ID = "mac-studio-substrate-phase1-coexistence-10k-v5"
V5_RUN_ID = "mac-studio-substrate-phase1-coexistence-10k-v5-edcm1_official_cpu-20260712T185754Z-leg01"
V5_RUN_DIR = Path("runs/local_throttle") / V5_RUN_ID
V5_RECEIPT_PATH = V5_RUN_DIR / "run_receipt.json"
V5_ARCHIVED_RECEIPT_PATH = V5_RUN_DIR / "artifacts/EDCM1_EVENT_TRIGGERED_COALITION_V3.seed-2026071101.json"
V5_ARCHIVED_CHECKPOINT_PATH = (
    V5_RUN_DIR / "artifacts/EDCM1_EVENT_TRIGGERED_COALITION_V3.seed-2026071101.checkpoint.json"
)
V5_CAMPAIGN_STATUS_PATH = Path("runs/mac_studio_campaign") / V5_CAMPAIGN_ID / "current_status.json"
V5_CAMPAIGN_EVENTS_PATH = Path("runs/mac_studio_campaign") / V5_CAMPAIGN_ID / "events.jsonl"
V5_ROUTER_STATUS_PATH = Path(
    "runs/mac_studio_null_safe_router/mac-studio-substrate-null-safe-coexistence-v5/current_status.json"
)

V1_FILE_SHA256 = "4a7cedeeead0f32f3541b163e5ec3e3904f4f75068eed2565625ad1e7fe51eae"
V1_CUTOVER_SHA256 = "013db5986364b68dc834012a84c9723355e809c4db7a61de9c352adaf707ff42"
V5_RECEIPT_FILE_SHA256 = "1131a7de42ec6b91ace54b6fea9a4fb8c90d37559e927b11809e757fdd362adb"
V5_RECEIPT_PAYLOAD_SHA256 = "5eb7227d6339a18707f4f2e109fcb0703c73cf056d9137b8a5ff05e49ef47a09"
V5_ARCHIVED_RECEIPT_SHA256 = "9cfd95100a560507c6f8076d17eb5be9d8b3eb88136625f023b95425eab43e29"
V5_ARCHIVED_CHECKPOINT_SHA256 = "fc42b306666fdc4674e014b8b1fc08189f71f13cc896253004d4b8f465de8a60"
V5_GATE_SEED = 2026071101
V6_GATE_SEED = 2026071102
V5_GATE_ROW_SHA256 = "8311eb31fdbd897c9190f880f125cbbceeb0cedcfdeb54e112d19f554c24dba9"
V5_OBSERVER_IMPLEMENTATION_SHA256 = "9516355d31f9c3f3814bc94e9281c6d15d1d49d7895a5cc1a53187cfcea9bac3"
V6_COMPATIBLE_LOADED_THROTTLE_SHA256 = frozenset(
    {
        "09f29e76dc6211e5a0ca918a16bda3e5f81c035853ae68f2e58e45d9ac926c91",
        "b6111a018c7da7a2809cfec144333e3731a06894a78d75465dccea21bb9e99ba",
        "3bbafdbfe02ecd10812f60be73ff9dbc035e2adf530a3bf5d5296fa54d0cb735",
    }
)
V5_OBSERVER_PROBLEM = (
    "active run mac-studio-substrate-phase1-coexistence-10k-v5-"
    "edcm1_official_cpu-20260712T185754Z-leg01: registered child command differs "
    "from task declaration"
)

AUTHORITY_PATHS = (
    "src/mop/studio/local_throttle.py",
    "src/mop/studio/external_coexistence.py",
    "src/mop/studio/task_policy_authority.py",
    "src/mop/studio/campaign_supervisor.py",
    "src/mop/studio/null_safe_campaign_router.py",
    "src/mop/studio/coexistence_cutover.py",
    "scripts/local_execution_throttle.py",
    "scripts/mop_campaign.py",
    "scripts/mop_null_safe_campaign.py",
    "scripts/build_local_throttle_hawking_cutover.py",
    "configs/local_execution_throttle.yaml",
    "configs/campaign/substrate_coexistence_task_overlay.yaml",
    "configs/campaign/substrate_task_overlay.yaml",
    "configs/campaign/mac_studio_substrate_phase1_null_safe_10k.json",
    "configs/campaign/mac_studio_substrate_null_safe_router.json",
)
AUDIT_PATHS = (
    "docs/audits/local_throttle_hawking_coexistence.md",
    "tests/unit/test_external_coexistence.py",
    "tests/unit/test_campaign_supervisor.py",
    "tests/unit/test_local_execution_throttle.py",
    "tests/unit/test_substrate_coexistence_overlay.py",
    "tests/unit/test_substrate_campaign_plan.py",
    "tests/unit/test_null_safe_p6_campaign_router.py",
    "tests/unit/test_coexistence_cutover.py",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CUTOVER_KEYS = {
    "schema",
    "created_at",
    "supersedes",
    "reason",
    "cutover_precondition",
    "external_profile_authority",
    "authority_snapshot",
    "runtime_gates",
    "task_envelope",
    "calibrations",
    "scientific_configuration_changed",
    "scientific_promotion",
    CUTOVER_SEAL,
}


class EvidenceError(ValueError):
    pass


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _safe_path(root: Path, relative: str | Path) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise EvidenceError(f"unsafe repository-relative path: {relative}")
    root = root.resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root):
        raise EvidenceError(f"path escapes repository: {relative}")
    return target


def stable_bytes(root: Path, relative: str | Path) -> bytes:
    path = _safe_path(root, relative)
    try:
        before_path = path.lstat()
        if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
            raise EvidenceError(f"evidence source is not a regular non-symlink: {relative}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before_fd = os.fstat(descriptor)
            chunks: list[bytes] = []
            while chunk := os.read(descriptor, 1024 * 1024):
                chunks.append(chunk)
            after_fd = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = path.lstat()
    except OSError as exc:
        raise EvidenceError(f"cannot snapshot evidence source {relative}: {exc}") from exc
    if not stat.S_ISREG(before_fd.st_mode) or not (
        _identity(before_path) == _identity(before_fd) == _identity(after_fd) == _identity(after_path)
    ):
        raise EvidenceError(f"evidence source changed while read: {relative}")
    return b"".join(chunks)


def _read_json(root: Path, relative: str | Path) -> tuple[dict[str, Any], bytes]:
    raw = stable_bytes(root, relative)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON evidence source {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON evidence source is not an object: {relative}")
    return value, raw


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_binding(root: Path, relative: str | Path) -> dict[str, object]:
    raw = stable_bytes(root, relative)
    return {"path": str(relative), "bytes": len(raw), "sha256": _sha256_bytes(raw)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _validate_seal(payload: Mapping[str, Any], field: str, label: str) -> str:
    core = dict(payload)
    declared = core.pop(field, None)
    _require(isinstance(declared, str) and _SHA256.fullmatch(declared) is not None, f"{label} seal missing")
    _require(canonical_sha256(core) == declared, f"{label} self-seal mismatch")
    return declared


def _sealed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    core = dict(payload)
    core.pop(field, None)
    return {**core, field: canonical_sha256(core)}


def _event_rows(raw: bytes, label: str) -> list[tuple[dict[str, Any], int, bytes]]:
    rows: list[tuple[dict[str, Any], int, bytes]] = []
    prefix = bytearray()
    for index, line in enumerate(raw.splitlines(keepends=True), 1):
        _require(line.endswith(b"\n"), f"{label} event row {index} is not newline terminated")
        prefix.extend(line)
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceError(f"{label} event row {index} is invalid: {exc}") from exc
        _require(isinstance(event, dict), f"{label} event row {index} is not an object")
        _validate_seal(event, "event_sha256", f"{label} event row {index}")
        rows.append((event, len(prefix), bytes(prefix)))
    _require(bool(rows), f"{label} event log is empty")
    return rows


def _atomic_write(path: Path, raw: bytes, *, immutable: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = path.read_bytes()
                _require(existing == raw, f"immutable evidence already exists with different bytes: {path}")
            else:
                _fsync_directory(path.parent)
        else:
            os.replace(temporary, path)
            _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _default_process_probe(pid: int) -> dict[str, Any]:
    try:
        process = psutil.Process(pid)
        return {
            "pid": pid,
            "create_time": process.create_time(),
            "exe": process.exe(),
            "cmdline": process.cmdline(),
        }
    except (psutil.Error, OSError) as exc:
        raise EvidenceError(f"active child process identity is unavailable: {exc}") from exc


def capture_active_observer(
    root: Path = REPO_ROOT,
    *,
    destination: Path = V6_ACTIVE_OBSERVER_PATH,
    status_path: Path = V6_STATUS_PATH,
    events_path: Path = V6_EVENTS_PATH,
    process_probe: Callable[[int], Mapping[str, Any]] = _default_process_probe,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:

    status, status_raw = _read_json(root, status_path)
    _validate_seal(status, "status_sha256", "v6 campaign status")
    _require(status.get("schema") == campaign.STATUS_SCHEMA, "v6 campaign status schema mismatch")
    _require(status.get("campaign_id") == V6_CAMPAIGN_ID, "v6 campaign id mismatch")
    _require(status.get("state") == "observing_existing", "v6 campaign is not observing an existing run")
    _require(status.get("problems") == [], "v6 campaign observer reports problems")
    lanes_value = status.get("active_lanes")
    _require(
        isinstance(lanes_value, list) and len(lanes_value) == 1, "v6 must expose exactly one active lane"
    )
    lanes = cast(list[Any], lanes_value)
    lane_value = lanes[0]
    _require(isinstance(lane_value, dict), "v6 active lane is invalid")
    lane = cast(dict[str, Any], lane_value)
    _require(lane.get("task_id") == V6_TASK_ID, "v6 active lane task mismatch")
    _require(lane.get("problems") == [], "v6 active lane reports problems")
    _require(lane.get("status") == "running", "v6 active lane is not running")
    run_id = lane.get("run_id")
    _require(
        isinstance(run_id, str) and run_id.startswith(f"{V6_CAMPAIGN_ID}-{V6_TASK_ID}-"), "v6 run id mismatch"
    )
    declared_value = lane.get("command")
    _require(
        isinstance(declared_value, list) and all(isinstance(item, str) for item in declared_value),
        "v6 declared command is invalid",
    )
    declared = cast(list[str], declared_value)
    expected_task = throttle.load_policy(root / "configs/local_execution_throttle.yaml").task(V6_TASK_ID)
    _require(tuple(declared) == expected_task.command, "v6 active lane command drifted from policy")

    child_pid_value = lane.get("child_pid")
    _require(isinstance(child_pid_value, int) and child_pid_value > 0, "v6 active child pid is invalid")
    child_pid = cast(int, child_pid_value)
    process = dict(process_probe(child_pid))
    observed_value = process.get("cmdline")
    _require(
        isinstance(observed_value, list) and all(isinstance(item, str) for item in observed_value),
        "v6 observed command is invalid",
    )
    observed = cast(list[str], observed_value)
    prefix_size = len(throttle.TASKPOLICY_COEXISTENCE_PREFIX)
    post_exec = list(expected_task.command[prefix_size:])
    _require(list(observed) == post_exec, "v6 child is not the exact pinned post-taskpolicy command")
    _require(process.get("pid") == child_pid, "v6 observed child pid mismatch")

    events_raw = stable_bytes(root, events_path)
    rows = _event_rows(events_raw, "v6 campaign")
    matching = [
        row
        for row in rows
        if row[0].get("event") == "governor-launched"
        and row[0].get("run_id") == run_id
        and row[0].get("step_id") == V6_STEP_ID
    ]
    _require(len(matching) == 1, "v6 governor-launched event is missing or ambiguous")
    target, prefix_bytes, prefix_raw = matching[0]
    prefix_rows = [row[0] for row in rows if row[1] <= prefix_bytes]
    _require(
        not any(row.get("event") == "integrity_hold" for row in prefix_rows),
        "v6 held before active observation",
    )

    supervisor_value = status.get("supervisor")
    _require(isinstance(supervisor_value, dict), "v6 supervisor authority is missing")
    supervisor = cast(dict[str, Any], supervisor_value)
    _require(
        supervisor.get("implementation_sha256")
        == _file_binding(root, "src/mop/studio/campaign_supervisor.py")["sha256"],
        "v6 supervisor implementation binding mismatch",
    )
    _require(
        supervisor.get("loaded_throttle_sha256")
        == _file_binding(root, "src/mop/studio/local_throttle.py")["sha256"],
        "v6 loaded throttle binding mismatch",
    )

    core: dict[str, Any] = {
        "schema": ACTIVE_OBSERVER_SCHEMA,
        "observed_at": now_fn().astimezone(UTC).isoformat(),
        "campaign_id": V6_CAMPAIGN_ID,
        "step_id": V6_STEP_ID,
        "task_id": V6_TASK_ID,
        "run_id": run_id,
        "campaign_status": {
            "path": str(status_path),
            "bytes": len(status_raw),
            "sha256": _sha256_bytes(status_raw),
            "document": status,
        },
        "event_prefix": {
            "path": str(events_path),
            "bytes": prefix_bytes,
            "sha256": _sha256_bytes(prefix_raw),
            "target_event": target,
        },
        "process_observation": {
            "pid": child_pid,
            "create_time": process.get("create_time"),
            "exe": process.get("exe"),
            "cmdline": observed,
            "cmdline_sha256": canonical_sha256(observed),
            "match_mode": "exact-pinned-post-taskpolicy-exec",
        },
        "observer_result": {
            "campaign_problems": [],
            "active_lane_problems": [],
            "registered_child_accepted": True,
            "integrity_hold_before_observation": False,
        },
        "scientific_promotion": False,
    }
    evidence = _sealed(core, ACTIVE_OBSERVER_SEAL)
    _atomic_write(_safe_path(root, destination), canonical_bytes(evidence) + b"\n", immutable=True)
    return evidence


def validate_active_observer_snapshot(root: Path = REPO_ROOT) -> dict[str, Any]:
    evidence, raw = _read_json(root, V6_ACTIVE_OBSERVER_PATH)
    _validate_seal(evidence, ACTIVE_OBSERVER_SEAL, "v6 active observer evidence")
    _require(evidence.get("schema") == ACTIVE_OBSERVER_SCHEMA, "active observer evidence schema mismatch")
    _require(evidence.get("campaign_id") == V6_CAMPAIGN_ID, "active observer campaign mismatch")
    _require(evidence.get("step_id") == V6_STEP_ID, "active observer step mismatch")
    _require(evidence.get("task_id") == V6_TASK_ID, "active observer task mismatch")
    result = evidence.get("observer_result")
    _require(
        result
        == {
            "campaign_problems": [],
            "active_lane_problems": [],
            "registered_child_accepted": True,
            "integrity_hold_before_observation": False,
        },
        "active observer result is not the reviewed success state",
    )
    status_binding_value = evidence.get("campaign_status")
    _require(isinstance(status_binding_value, dict), "active observer status binding missing")
    status_binding = cast(dict[str, Any], status_binding_value)
    status_value = status_binding.get("document")
    _require(isinstance(status_value, dict), "active observer embedded status missing")
    status = cast(dict[str, Any], status_value)
    _validate_seal(status, "status_sha256", "embedded v6 campaign status")
    _require(status.get("schema") == campaign.STATUS_SCHEMA, "embedded v6 status schema mismatch")
    _require(status.get("campaign_id") == V6_CAMPAIGN_ID, "embedded v6 campaign id mismatch")
    _require(
        status.get("state") == "observing_existing" and status.get("problems") == [],
        "embedded v6 status is not clean",
    )
    lanes_value = status.get("active_lanes")
    _require(
        isinstance(lanes_value, list)
        and len(lanes_value) == 1
        and isinstance(lanes_value[0], dict)
        and lanes_value[0].get("problems") == [],
        "embedded v6 lane is not clean",
    )
    lanes = cast(list[dict[str, Any]], lanes_value)
    lane = lanes[0]
    _require(lane.get("task_id") == V6_TASK_ID, "embedded v6 lane task mismatch")
    _require(lane.get("run_id") == evidence.get("run_id"), "embedded v6 lane run mismatch")
    _require(
        status_binding.get("sha256") == canonical_sha256(status)
        or status_binding.get("sha256") == _sha256_bytes(canonical_bytes(status) + b"\n"),
        "embedded status byte binding mismatch",
    )
    _require(
        status_binding.get("bytes") == len(canonical_bytes(status) + b"\n"),
        "embedded status byte count mismatch",
    )
    supervisor_value = status.get("supervisor")
    _require(isinstance(supervisor_value, dict), "embedded v6 supervisor authority missing")
    supervisor = cast(dict[str, Any], supervisor_value)
    _require(
        supervisor.get("implementation_sha256")
        == _file_binding(root, "src/mop/studio/campaign_supervisor.py")["sha256"],
        "embedded v6 supervisor implementation drifted",
    )
    current_throttle_sha256 = _file_binding(root, "src/mop/studio/local_throttle.py")["sha256"]
    _require(
        supervisor.get("loaded_throttle_sha256") in V6_COMPATIBLE_LOADED_THROTTLE_SHA256
        and current_throttle_sha256 in V6_COMPATIBLE_LOADED_THROTTLE_SHA256,
        "embedded v6 loaded throttle is outside the reviewed compatibility set",
    )
    process_value = evidence.get("process_observation")
    _require(isinstance(process_value, dict), "active observer process binding missing")
    process = cast(dict[str, Any], process_value)
    _require(process.get("pid") == lane.get("child_pid"), "embedded v6 process pid mismatch")
    declared_value = lane.get("command")
    _require(isinstance(declared_value, list), "embedded v6 declared command missing")
    declared = cast(list[Any], declared_value)
    expected_observed = declared[len(throttle.TASKPOLICY_COEXISTENCE_PREFIX) :]
    _require(
        process.get("cmdline") == expected_observed,
        "embedded process command is not the pinned post-exec suffix",
    )
    _require(
        process.get("cmdline_sha256") == canonical_sha256(expected_observed),
        "embedded process command hash mismatch",
    )
    prefix_value = evidence.get("event_prefix")
    _require(isinstance(prefix_value, dict), "active observer event prefix missing")
    prefix = cast(dict[str, Any], prefix_value)
    target_value = prefix.get("target_event")
    _require(isinstance(target_value, dict), "active observer target event missing")
    target = cast(dict[str, Any], target_value)
    _validate_seal(target, "event_sha256", "active observer target event")
    _require(target.get("event") == "governor-launched", "active observer target event mismatch")
    _require(target.get("campaign_id") == V6_CAMPAIGN_ID, "active observer event campaign mismatch")
    _require(target.get("step_id") == V6_STEP_ID, "active observer event step mismatch")
    _require(target.get("task_id") == V6_TASK_ID, "active observer event task mismatch")
    _require(target.get("run_id") == evidence.get("run_id"), "active observer event run mismatch")
    prefix_bytes_value = prefix.get("bytes")
    _require(
        isinstance(prefix_bytes_value, int) and prefix_bytes_value > 0,
        "active observer prefix size is invalid",
    )
    prefix_bytes = cast(int, prefix_bytes_value)
    live_events = stable_bytes(root, V6_EVENTS_PATH)
    _require(prefix_bytes <= len(live_events), "active observer prefix exceeds the live event log")
    prefix_raw = live_events[:prefix_bytes]
    _require(prefix.get("sha256") == _sha256_bytes(prefix_raw), "active observer prefix hash mismatch")
    prefix_rows = _event_rows(prefix_raw, "active observer campaign prefix")
    _require(prefix_rows[-1][0] == target, "active observer target is not the bound prefix terminus")
    _require(
        not any(row[0].get("event") == "integrity_hold" for row in prefix_rows),
        "active observer prefix contains an integrity hold",
    )
    return {
        "path": str(V6_ACTIVE_OBSERVER_PATH),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
        "evidence_sha256": evidence[ACTIVE_OBSERVER_SEAL],
        "observed_at": evidence["observed_at"],
        "run_id": evidence["run_id"],
        "process_observation": process,
        "observer_result": result,
    }


def _v5_first_seed_evidence(root: Path) -> dict[str, Any]:
    receipt, receipt_raw = _read_json(root, V5_RECEIPT_PATH)
    _require(_sha256_bytes(receipt_raw) == V5_RECEIPT_FILE_SHA256, "v5 governor receipt file hash mismatch")
    _require(
        _validate_seal(receipt, "payload_sha256", "v5 governor receipt") == V5_RECEIPT_PAYLOAD_SHA256,
        "v5 governor receipt payload hash mismatch",
    )
    outcome = campaign.probe_run_outcome(V5_RUN_ID, root / "runs/local_throttle")
    _require(
        outcome is not None and outcome.status == "resumable-invocation-boundary",
        "v5 governor receipt is not a valid resumable boundary",
    )
    outcome = cast(campaign.RunOutcome, outcome)
    _require(outcome.final_returncode == 2, "v5 governor return code mismatch")
    _require(not outcome.admission_denied_reasons, "v5 governor admission was denied")
    _require(receipt.get("admission", {}).get("allowed") is True, "v5 admission is not allowed")
    _require(
        receipt.get("admission", {}).get("consecutive_good_samples") == 3, "v5 admission hysteresis mismatch"
    )
    invocations_value = receipt.get("invocations")
    _require(
        isinstance(invocations_value, list) and len(invocations_value) == 1, "v5 invocation count mismatch"
    )
    invocations = cast(list[dict[str, Any]], invocations_value)
    _require(invocations[0].get("returncode") == 2, "v5 invocation return code mismatch")

    artifact, artifact_raw = _read_json(root, V5_ARCHIVED_RECEIPT_PATH)
    checkpoint, checkpoint_raw = _read_json(root, V5_ARCHIVED_CHECKPOINT_PATH)
    _require(_sha256_bytes(artifact_raw) == V5_ARCHIVED_RECEIPT_SHA256, "v5 archived receipt hash mismatch")
    _require(
        _sha256_bytes(checkpoint_raw) == V5_ARCHIVED_CHECKPOINT_SHA256, "v5 archived checkpoint hash mismatch"
    )
    artifact_seal = _validate_seal(artifact, "receipt_sha256", "v5 archived EDCM receipt")
    checkpoint_seal = _validate_seal(checkpoint, "checkpoint_sha256", "v5 archived EDCM checkpoint")
    _require(artifact.get("schema") == "mop-edcm1-receipt/v3", "v5 EDCM receipt schema mismatch")
    _require(artifact.get("execution_status") == "partial", "v5 EDCM status mismatch")
    _require(artifact.get("resumable") is True, "v5 EDCM receipt is not resumable")
    _require(artifact.get("all_ok") is False, "v5 partial receipt cannot claim all_ok")
    _require(artifact.get("problems") == ["execution_incomplete"], "v5 EDCM problems mismatch")
    _require(artifact.get("completed_gate_seeds") == [V5_GATE_SEED], "v5 completed gate seed mismatch")
    _require(artifact.get("completed_heldout_seeds") == [], "v5 heldout work unexpectedly completed")
    _require(artifact.get("scientific_promotion") is False, "v5 partial receipt claims promotion")
    row_hashes = artifact.get("checkpoint_binding", {}).get("gate_row_sha256")
    _require(row_hashes == [V5_GATE_ROW_SHA256], "v5 gate row hash mismatch")
    _require(checkpoint.get("gate_row_sha256") == row_hashes, "v5 receipt/checkpoint row join mismatch")
    _require(
        artifact.get("checkpoint_binding", {}).get("checkpoint_sha256") == checkpoint_seal,
        "v5 checkpoint self-seal join mismatch",
    )
    files = invocations[0].get("checkpoint_after", {}).get("files")
    observed_files = (
        {row.get("sha256") for row in files if isinstance(row, dict)} if isinstance(files, list) else set()
    )
    _require(
        observed_files == {V5_ARCHIVED_RECEIPT_SHA256, V5_ARCHIVED_CHECKPOINT_SHA256},
        "v5 governor output snapshot mismatch",
    )
    progress_value = receipt.get("progress_authority")
    _require(isinstance(progress_value, dict), "v5 progress authority missing")
    progress = cast(dict[str, Any], progress_value)
    child_resource = progress.get("child_resource")
    _require(isinstance(child_resource, dict), "v5 child resource evidence missing")
    return {
        "classification": "resumable-first-seed-success",
        "run_id": V5_RUN_ID,
        "governor_receipt": {
            "path": str(V5_RECEIPT_PATH),
            "bytes": len(receipt_raw),
            "sha256": V5_RECEIPT_FILE_SHA256,
            "payload_sha256": V5_RECEIPT_PAYLOAD_SHA256,
            "status": receipt["status"],
            "final_returncode": receipt["final_returncode"],
            "started_at": receipt["started_at"],
            "finished_at": receipt["finished_at"],
            "wall_seconds": receipt["wall_seconds"],
            "progress_authority_sha256": canonical_sha256(progress),
            "checkpoint_aggregate_sha256": outcome.checkpoint_sha256,
            "child_resource": child_resource,
            "admission": receipt["admission"],
        },
        "archived_receipt": {
            "path": str(V5_ARCHIVED_RECEIPT_PATH),
            "bytes": len(artifact_raw),
            "sha256": V5_ARCHIVED_RECEIPT_SHA256,
            "receipt_sha256": artifact_seal,
        },
        "archived_checkpoint": {
            "path": str(V5_ARCHIVED_CHECKPOINT_PATH),
            "bytes": len(checkpoint_raw),
            "sha256": V5_ARCHIVED_CHECKPOINT_SHA256,
            "checkpoint_sha256": checkpoint_seal,
        },
        "completed_gate_seeds": [V5_GATE_SEED],
        "gate_row_sha256": [V5_GATE_ROW_SHA256],
        "scientific_result": "not_evaluated",
        "scientific_promotion": False,
    }


def _v5_observer_hold_evidence(root: Path) -> dict[str, Any]:
    status, status_raw = _read_json(root, V5_CAMPAIGN_STATUS_PATH)
    _validate_seal(status, "status_sha256", "v5 campaign status")
    _require(status.get("campaign_id") == V5_CAMPAIGN_ID, "v5 campaign id mismatch")
    _require(status.get("state") == "integrity_hold", "v5 campaign did not retain the calibration hold")
    _require(status.get("problems") == [V5_OBSERVER_PROBLEM], "v5 observer hold problem mismatch")
    supervisor_value = status.get("supervisor")
    _require(isinstance(supervisor_value, dict), "v5 supervisor binding missing")
    supervisor = cast(dict[str, Any], supervisor_value)
    _require(
        supervisor.get("implementation_sha256") == V5_OBSERVER_IMPLEMENTATION_SHA256,
        "v5 observer implementation hash mismatch",
    )
    events_raw = stable_bytes(root, V5_CAMPAIGN_EVENTS_PATH)
    events = _event_rows(events_raw, "v5 campaign")
    event_names = [row[0].get("event") for row in events]
    _require(
        event_names == ["supervisor-start", "governor-launched", "integrity_hold", "supervisor-stop"],
        "v5 observer event sequence mismatch",
    )
    launched = events[1][0]
    hold = events[2][0]
    stopped = events[3][0]
    _require(launched.get("run_id") == V5_RUN_ID, "v5 launched run id mismatch")
    _require(hold.get("problem") == V5_OBSERVER_PROBLEM, "v5 hold event problem mismatch")
    v5_receipt, _ = _read_json(root, V5_RECEIPT_PATH)
    _require(
        str(stopped.get("at")) < str(v5_receipt.get("finished_at")),
        "v5 child did not finish after observer stop",
    )
    router, router_raw = _read_json(root, V5_ROUTER_STATUS_PATH)
    _validate_seal(router, "status_sha256", "v5 router status")
    _require(router.get("state") == "failure_hold", "v5 router did not retain the observer hold")
    return {
        "classification": "observer-false-positive-calibration",
        "campaign_status": {
            "path": str(V5_CAMPAIGN_STATUS_PATH),
            "bytes": len(status_raw),
            "sha256": _sha256_bytes(status_raw),
            "status_sha256": status["status_sha256"],
            "state": status["state"],
            "problem": V5_OBSERVER_PROBLEM,
        },
        "observer_implementation_sha256": V5_OBSERVER_IMPLEMENTATION_SHA256,
        "event_log": {
            "path": str(V5_CAMPAIGN_EVENTS_PATH),
            "bytes": len(events_raw),
            "sha256": _sha256_bytes(events_raw),
            "event_sha256": [row[0]["event_sha256"] for row in events],
            "supervisor_stop_at": stopped["at"],
        },
        "governed_child_finished_at": v5_receipt["finished_at"],
        "router_status": {
            "path": str(V5_ROUTER_STATUS_PATH),
            "bytes": len(router_raw),
            "sha256": _sha256_bytes(router_raw),
            "state": router["state"],
            "problems": router["problems"],
        },
        "interpretation": (
            "observer calibration only; corroboration requires the v6 active and resumable evidence"
        ),
        "scientific_promotion": False,
    }


def _no_launch_calibrations(root: Path) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for version in (2, 3, 4):
        campaign_path = Path(
            f"runs/mac_studio_campaign/mac-studio-substrate-phase1-coexistence-10k-v{version}/current_status.json"
        )
        router_path = Path(
            f"runs/mac_studio_null_safe_router/mac-studio-substrate-null-safe-coexistence-v{version}/current_status.json"
        )
        campaign_status, campaign_raw = _read_json(root, campaign_path)
        router_status, router_raw = _read_json(root, router_path)
        _validate_seal(campaign_status, "status_sha256", f"v{version} no-launch campaign status")
        _validate_seal(router_status, "status_sha256", f"v{version} no-launch router status")
        _require(campaign_status.get("state") == "drained", f"v{version} campaign was not drained")
        _require(campaign_status.get("active_lanes") == [], f"v{version} campaign launched a lane")
        _require(campaign_status.get("problems") == [], f"v{version} campaign has problems")
        _require(router_status.get("state") == "drained", f"v{version} router was not drained")
        _require(
            router_status.get("active_lanes") in (None, []),
            f"v{version} router observed a lane",
        )
        _require(router_status.get("problems") == [], f"v{version} router has problems")
        trials.append(
            {
                "version": version,
                "owned_task_launched": False,
                "campaign": {
                    "path": str(campaign_path),
                    "sha256": _sha256_bytes(campaign_raw),
                    "status_sha256": campaign_status["status_sha256"],
                },
                "router": {
                    "path": str(router_path),
                    "sha256": _sha256_bytes(router_raw),
                    "status_sha256": router_status["status_sha256"],
                },
            }
        )
    return {
        "classification": "no-launch-safety-calibration",
        "trials": trials,
        "finding": "pressure gates drained without launching an owned lane",
        "scientific_promotion": False,
    }


def _validate_seed2_continuity(artifact: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    _validate_seal(artifact, "receipt_sha256", "v6 archived EDCM receipt")
    checkpoint_seal = _validate_seal(checkpoint, "checkpoint_sha256", "v6 archived EDCM checkpoint")
    _require(artifact.get("schema") == "mop-edcm1-receipt/v3", "v6 EDCM receipt schema mismatch")
    _require(artifact.get("execution_status") == "partial", "v6 EDCM receipt is not partial")
    _require(artifact.get("resumable") is True, "v6 EDCM receipt is not resumable")
    _require(
        artifact.get("completed_gate_seeds") == [V5_GATE_SEED, V6_GATE_SEED], "v6 did not land exactly seed 2"
    )
    _require(artifact.get("completed_heldout_seeds") == [], "v6 unexpectedly entered heldout evaluation")
    row_hashes = artifact.get("checkpoint_binding", {}).get("gate_row_sha256")
    _require(isinstance(row_hashes, list) and len(row_hashes) == 2, "v6 gate row count mismatch")
    _require(row_hashes[0] == V5_GATE_ROW_SHA256, "v6 checkpoint lost the archived v5 prefix")
    _require(checkpoint.get("gate_row_sha256") == row_hashes, "v6 receipt/checkpoint row join mismatch")
    _require(
        artifact.get("checkpoint_binding", {}).get("checkpoint_sha256") == checkpoint_seal,
        "v6 checkpoint self-seal join mismatch",
    )
    _require(artifact.get("scientific_promotion") is False, "v6 partial receipt claims scientific promotion")


def _output_hashes(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    invocations_value = receipt.get("invocations")
    _require(
        isinstance(invocations_value, list) and len(invocations_value) == 1, "v6 invocation count mismatch"
    )
    invocations = cast(list[dict[str, Any]], invocations_value)
    files = invocations[0].get("checkpoint_after", {}).get("files")
    _require(isinstance(files, list) and len(files) == 2, "v6 checkpoint snapshot file count mismatch")
    result: dict[str, dict[str, Any]] = {}
    for row in files:
        _require(isinstance(row, dict), "v6 checkpoint snapshot row is invalid")
        path = row.get("path")
        digest = row.get("sha256")
        _require(
            isinstance(path, str) and isinstance(digest, str), "v6 checkpoint snapshot binding is invalid"
        )
        result[path] = row
    required = {
        "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json",
        "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.checkpoint.json",
    }
    _require(set(result) == required, "v6 checkpoint snapshot paths mismatch")
    return result


def _archive_terminal_file(
    root: Path,
    *,
    source: str,
    destination: Path,
    expected_sha256: str,
    write: bool,
) -> tuple[dict[str, Any], bytes]:
    destination_path = _safe_path(root, destination)
    if destination_path.is_file():
        raw = stable_bytes(root, destination)
    else:
        raw = stable_bytes(root, source)
        _require(
            _sha256_bytes(raw) == expected_sha256,
            f"v6 live output no longer matches terminal receipt: {source}",
        )
        _require(write, f"v6 terminal artifact is not archived yet: {destination}")
        _atomic_write(destination_path, raw, immutable=True)
    _require(_sha256_bytes(raw) == expected_sha256, f"v6 archived output hash mismatch: {destination}")
    document = json.loads(raw)
    _require(isinstance(document, dict), f"v6 archived output is not an object: {destination}")
    return document, raw


def _v6_terminal_evidence(root: Path, active: Mapping[str, Any], *, archive: bool) -> dict[str, Any]:
    events_raw = stable_bytes(root, V6_EVENTS_PATH)
    rows = _event_rows(events_raw, "v6 campaign")
    active_run_id = active.get("run_id")
    _require(
        isinstance(active_run_id, str) and active_run_id.startswith(f"{V6_CAMPAIGN_ID}-"),
        "v6 active observer run id is invalid",
    )
    run_id = V6_SEED2_RUN_ID
    matching = [
        row
        for row in rows
        if row[0].get("event") == "resumable-leg"
        and row[0].get("run_id") == run_id
        and row[0].get("step_id") == V6_STEP_ID
    ]
    _require(len(matching) == 1, "v6 matching resumable-leg event has not landed")
    target, prefix_bytes, prefix_raw = matching[0]
    prefix_rows = [row[0] for row in rows if row[1] <= prefix_bytes]
    _require(
        not any(row.get("event") == "integrity_hold" for row in prefix_rows),
        "v6 observer entered integrity_hold before resumable-leg",
    )
    _require(
        isinstance(active.get("observed_at"), str) and str(target.get("at")) < str(active["observed_at"]),
        "v6 active observation does not follow the seed-2 resumable event",
    )

    receipt_path = Path("runs/local_throttle") / str(run_id) / "run_receipt.json"
    receipt, receipt_raw = _read_json(root, receipt_path)
    payload_sha = _validate_seal(receipt, "payload_sha256", "v6 governor receipt")
    outcome = campaign.probe_run_outcome(str(run_id), root / "runs/local_throttle")
    _require(
        outcome is not None and outcome.status == "resumable-invocation-boundary",
        "v6 receipt is not a valid resumable invocation boundary",
    )
    outcome = cast(campaign.RunOutcome, outcome)
    _require(
        outcome.final_returncode == 2 and not outcome.admission_denied_reasons,
        "v6 resumable receipt return/admission mismatch",
    )
    _require(
        target.get("checkpoint_sha256") == outcome.checkpoint_sha256, "v6 event/receipt checkpoint mismatch"
    )
    output_rows = _output_hashes(receipt)
    archive_dir = Path("runs/local_throttle") / str(run_id) / "artifacts"
    artifact_destination = archive_dir / f"EDCM1_EVENT_TRIGGERED_COALITION_V3.seed-{V6_GATE_SEED}.json"
    checkpoint_destination = (
        archive_dir / f"EDCM1_EVENT_TRIGGERED_COALITION_V3.seed-{V6_GATE_SEED}.checkpoint.json"
    )
    artifact_row = output_rows["proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json"]
    checkpoint_row = output_rows["proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.checkpoint.json"]
    artifact, artifact_raw = _archive_terminal_file(
        root,
        source="proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json",
        destination=artifact_destination,
        expected_sha256=str(artifact_row["sha256"]),
        write=archive,
    )
    checkpoint, checkpoint_raw = _archive_terminal_file(
        root,
        source="proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.checkpoint.json",
        destination=checkpoint_destination,
        expected_sha256=str(checkpoint_row["sha256"]),
        write=archive,
    )
    _validate_seed2_continuity(artifact, checkpoint)
    return {
        "classification": "observer-fix-corroborated-by-resumable-seed2",
        "run_id": run_id,
        "event_prefix": {
            "path": str(V6_EVENTS_PATH),
            "bytes": prefix_bytes,
            "sha256": _sha256_bytes(prefix_raw),
            "target_event": target,
            "integrity_hold_before_target": False,
        },
        "governor_receipt": {
            "path": str(receipt_path),
            "bytes": len(receipt_raw),
            "sha256": _sha256_bytes(receipt_raw),
            "payload_sha256": payload_sha,
            "status": receipt["status"],
            "final_returncode": receipt["final_returncode"],
            "checkpoint_aggregate_sha256": outcome.checkpoint_sha256,
        },
        "archived_receipt": {
            "path": str(artifact_destination),
            "bytes": len(artifact_raw),
            "sha256": _sha256_bytes(artifact_raw),
            "receipt_sha256": artifact["receipt_sha256"],
        },
        "archived_checkpoint": {
            "path": str(checkpoint_destination),
            "bytes": len(checkpoint_raw),
            "sha256": _sha256_bytes(checkpoint_raw),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        },
        "completed_gate_seeds": [V5_GATE_SEED, V6_GATE_SEED],
        "gate_row_sha256": artifact["checkpoint_binding"]["gate_row_sha256"],
        "scientific_result": "not_evaluated",
        "scientific_promotion": False,
    }


def _supersedes(root: Path) -> dict[str, Any]:
    v1, raw = _read_json(root, V1_CUTOVER_PATH)
    _require(_sha256_bytes(raw) == V1_FILE_SHA256, "preserved v1 file hash mismatch")
    _require(
        v1.get("schema") == "mop-local-throttle-external-coexistence-cutover/v1",
        "preserved v1 schema mismatch",
    )
    _require(
        _validate_seal(v1, CUTOVER_SEAL, "preserved v1 cutover") == V1_CUTOVER_SHA256,
        "preserved v1 cutover hash mismatch",
    )
    return {
        "path": str(V1_CUTOVER_PATH),
        "schema": v1["schema"],
        "bytes": len(raw),
        "sha256": V1_FILE_SHA256,
        "cutover_sha256": V1_CUTOVER_SHA256,
    }


def _authority_snapshot(root: Path) -> dict[str, Any]:
    return {
        "runtime_and_configuration": [_file_binding(root, path) for path in AUTHORITY_PATHS],
        "audits_and_tests": [_file_binding(root, path) for path in AUDIT_PATHS],
    }


def _external_profile_authority() -> dict[str, Any]:
    profile = coexistence.HawkingSerialCPUProfile.create(
        root=throttle.HAWKING_ROOT,
        python_executable=throttle.HAWKING_PYTHON,
    )
    return profile.authority()


def _task_envelope(root: Path) -> dict[str, Any]:
    policy = throttle.load_policy(root / "configs/local_execution_throttle.yaml")
    tasks = [policy.task(task_id) for task_id in sorted(throttle.EXTERNAL_COEXISTENCE_TASKS)]
    for task in tasks:
        _require(
            not throttle._external_coexistence_task_problems(task),
            f"coexistence task drifted: {task.task_id}",
        )
    return {
        "task_ids": [task.task_id for task in tasks],
        "lane": "cpu",
        "accelerator": "none",
        "cpu_cores": 1,
        "estimated_unified_memory_gb": throttle.TASKPOLICY_COEXISTENCE_CAP_GB,
        "taskpolicy_prefix": list(throttle.TASKPOLICY_COEXISTENCE_PREFIX),
        "taskpolicy_memory_mib": 4096,
        "taskpolicy_process_control": "kill",
        "darwin_background_priority": True,
        "thread_environment_cap": 1,
        "scheduler_exclusive": True,
        "producer_task_ids": sorted(throttle.SEED_BOUNDARY_TASKS),
        "producer_invocations_per_governor_leg": 1,
        "p6_coexistence_allowed": False,
        "commands": [
            {"task_id": task.task_id, "command_sha256": canonical_sha256(list(task.command))}
            for task in tasks
        ],
    }


def build_document(root: Path = REPO_ROOT, *, archive_terminal: bool = False) -> dict[str, Any]:
    active = validate_active_observer_snapshot(root)
    terminal = _v6_terminal_evidence(root, active, archive=archive_terminal)
    core: dict[str, Any] = {
        "schema": CUTOVER_SCHEMA,
        "created_at": terminal["event_prefix"]["target_event"]["at"],
        "supersedes": _supersedes(root),
        "reason": "replace an absolute Hawking veto with an exact pressure-gated one-core coexistence lane",
        "cutover_precondition": {
            "external_process_ownership": "observation-only; Hawking is never signaled",
            "no_owned_lane_overlap": True,
            "v5_first_seed_archived": True,
            "v6_observer_fix_corroborated": True,
            "scientific_claim_made": False,
        },
        "external_profile_authority": _external_profile_authority(),
        "authority_snapshot": _authority_snapshot(root),
        "runtime_gates": {
            "external_cpu_percent_cap": coexistence.MAXIMUM_AGGREGATE_CPU_PERCENT,
            "maximum_load_per_logical_cpu": 0.85,
            "maximum_swap_used_gb": 0.0,
            "maximum_transient_cpu_utilization_fraction": 1.0,
            "minimum_available_memory_gb": 40.0,
            "minimum_available_memory_percent": 40.0,
            "minimum_memory_pressure_free_percent": 75.0,
            "power": "AC",
            "thermal": "normal",
        },
        "task_envelope": _task_envelope(root),
        "calibrations": {
            "no_launch": _no_launch_calibrations(root),
            "v5_first_seed_resumable": _v5_first_seed_evidence(root),
            "v5_observer_hold": _v5_observer_hold_evidence(root),
            "v6_active_observer": active,
            "v6_terminal_resumable": terminal,
        },
        "scientific_configuration_changed": False,
        "scientific_promotion": False,
    }
    return _sealed(core, CUTOVER_SEAL)


def write_cutover(root: Path = REPO_ROOT, output: Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    document = build_document(root, archive_terminal=True)
    _atomic_write(_safe_path(root, output), canonical_bytes(document) + b"\n", immutable=False)
    return document


def validate_cutover(root: Path = REPO_ROOT, output: Path = DEFAULT_OUTPUT_PATH) -> dict[str, Any]:
    document, _ = _read_json(root, output)
    _require(set(document) == _CUTOVER_KEYS, "cutover root keys mismatch")
    _require(document.get("schema") == CUTOVER_SCHEMA, "cutover schema mismatch")
    _validate_seal(document, CUTOVER_SEAL, "cutover")
    expected = build_document(root, archive_terminal=False)
    _require(document == expected, "cutover does not exactly match live authorities and immutable evidence")
    return document


def readiness_report(root: Path = REPO_ROOT) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, probe in (
        ("preserved_v1", lambda: _supersedes(root)),
        ("no_launch_calibrations", lambda: _no_launch_calibrations(root)),
        ("v5_first_seed", lambda: _v5_first_seed_evidence(root)),
        ("v5_observer_hold", lambda: _v5_observer_hold_evidence(root)),
        ("v6_active_observer", lambda: validate_active_observer_snapshot(root)),
    ):
        try:
            probe()
        except EvidenceError as exc:
            checks[name] = {"ready": False, "problem": str(exc)}
        else:
            checks[name] = {"ready": True, "problem": None}
    active: dict[str, Any] | None = None
    try:
        active = validate_active_observer_snapshot(root)
        _v6_terminal_evidence(root, active, archive=False)
    except EvidenceError as exc:
        checks["v6_terminal_resumable"] = {"ready": False, "problem": str(exc)}
    else:
        checks["v6_terminal_resumable"] = {"ready": True, "problem": None}
    ready = all(row["ready"] for row in checks.values())
    core = {
        "schema": READINESS_SCHEMA,
        "cutover_ready": ready,
        "checks": checks,
        "scientific_promotion": False,
    }
    return {**core, "report_sha256": canonical_sha256(core)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--capture-active-observer", action="store_true")
    modes.add_argument("--check-readiness", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.capture_active_observer:
            result = capture_active_observer(REPO_ROOT)
        elif args.check_readiness:
            result = readiness_report(REPO_ROOT)
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0 if result["cutover_ready"] else 2
        elif args.verify_only:
            result = validate_cutover(REPO_ROOT, args.out)
        else:
            result = write_cutover(REPO_ROOT, args.out)
    except EvidenceError as exc:
        print(json.dumps({"status": "blocked", "problem": str(exc)}, sort_keys=True, indent=2))
        return 2
    print(json.dumps({"status": "ok", "schema": result["schema"]}, sort_keys=True, indent=2))
    return 0


__all__ = [
    "ACTIVE_OBSERVER_SCHEMA",
    "CUTOVER_SCHEMA",
    "EvidenceError",
    "build_document",
    "canonical_sha256",
    "capture_active_observer",
    "main",
    "readiness_report",
    "validate_active_observer_snapshot",
    "validate_cutover",
    "write_cutover",
]
