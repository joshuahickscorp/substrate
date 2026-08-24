"""Deterministic storage bridge for the Tangible Sandbox continuity lane.

The bridge is infrastructure-only.  It chooses a configured run root in a
stable order, proves the volume has the filesystem semantics the worker needs,
rehearses the same checkpoint writer for one hour, and then hands the real lane
to a separate launchd-owned worker.  Every unsafe condition is a terminal
refusal for that bridge attempt; no historical trace or outcome is rewritten.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from substrate import sandbox_campaign as base
from substrate import sandbox_config as C
from substrate import sandbox_execution as execution

ROOT = base.ROOT
EVIDENCE = base.EVIDENCE
RUNS = base.RUNS

VERSION = "1.0.0"
ROOT_NAME = "storage-bridge"
REHEARSAL_SECONDS = 60 * 60
HEARTBEAT_SECONDS = 60
MINIMUM_OWN_RUN_GROWTH_BYTES = 1 * 1024 * 1024
MINIMUM_TRANSIENT_BYTES = 1 * 1024 * 1024
ESTIMATION_MARGIN_BYTES = 1 * C.GIB
CLEAN_CLONE_MARGIN_BYTES = int(2.5 * C.GIB)
DYNAMIC_PLAN_PATH = EVIDENCE / "SUBSTRATE_SANDBOX_DYNAMIC_STORAGE_PLAN.json"


def _source_path() -> Path:
    return Path(__file__).resolve()


def _sha_file(path: Path) -> str:
    return execution._sha_file(path)


def _path_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _bridge_root(run_id: str) -> Path:
    return RUNS / ROOT_NAME / run_id


def _candidate_roots() -> list[Path]:
    """Return explicit candidates only; unknown mounted disks are never touched."""

    configured = os.environ.get("SUBSTRATE_TANGIBLE_RUN_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(RUNS)
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _protected_floor(total_bytes: int) -> int:
    return C.disk_floor_bytes(total_bytes)


def _directory_bytes(path: Path) -> int:
    """Count allocated files once per inode for a local, bounded receipt tree."""

    seen: set[tuple[int, int]] = set()
    total = 0
    if not path.exists():
        return total
    for candidate in path.rglob("*"):
        try:
            stat = candidate.stat()
        except OSError:
            continue
        if not candidate.is_file() or (stat.st_dev, stat.st_ino) in seen:
            continue
        seen.add((stat.st_dev, stat.st_ino))
        total += stat.st_size
    return total


def _user_reserve_bytes() -> int:
    """Read an explicit operator reserve; the protected floor remains mandatory."""

    raw = os.environ.get("SUBSTRATE_TANGIBLE_USER_RESERVE_GIB", "0")
    try:
        reserve_gib = float(raw)
    except ValueError as error:
        raise base.Refused("SUBSTRATE_TANGIBLE_USER_RESERVE_GIB must be numeric") from error
    if reserve_gib < 0 or reserve_gib > 10_000:
        raise base.Refused("SUBSTRATE_TANGIBLE_USER_RESERVE_GIB is out of range")
    return int(reserve_gib * C.GIB)


def _pending_acquisition_bytes() -> int:
    path = EVIDENCE / "SUBSTRATE_SANDBOX_ACQUISITION_RESULT.json"
    if not path.is_file():
        return sum(int(row["bytes"]) for row in C.CORE_BINARY_ASSETS)
    receipt = base.load_json(path)
    archives = receipt.get("archives", [])
    if not isinstance(archives, list):
        raise base.Refused("acquisition receipt archives are invalid")
    complete = {str(row.get("filename")) for row in archives if row.get("state") == "VALIDATED"}
    return sum(
        int(row["bytes"])
        for row in C.CORE_BINARY_ASSETS
        if str(row["filename"]) not in complete
    )


def _observed_lane_bytes() -> int:
    """Measure only lane receipts, never unrelated volume churn."""

    attempts = sorted(RUNS.glob("longitudinal-invalidated-*"))
    return max((_directory_bytes(path) for path in attempts), default=0)


def _tracked_tree_bytes() -> int:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "-rl", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise base.Refused("cannot measure the terminal clean-clone tree")
    sizes = []
    for line in result.stdout.splitlines():
        columns = line.split(maxsplit=3)
        if len(columns) >= 3 and columns[1] == "blob" and columns[2].isdigit():
            sizes.append(int(columns[2]))
    return sum(sizes)


def _storage_budget(root: Path) -> dict[str, Any]:
    """Calculate phase-specific demand from receipts rather than project size."""

    usage = shutil.disk_usage(root)
    protected_floor = _protected_floor(usage.total)
    user_reserve = _user_reserve_bytes()
    observed_lane = _observed_lane_bytes()
    own_growth = max(MINIMUM_OWN_RUN_GROWTH_BYTES, observed_lane * 8)
    pending = _pending_acquisition_bytes()
    clean_clone = max(CLEAN_CLONE_MARGIN_BYTES, _tracked_tree_bytes())
    maintained_floor = max(protected_floor, user_reserve)
    continuity_required = (
        maintained_floor
        + pending
        + own_growth
        + MINIMUM_TRANSIENT_BYTES
        + ESTIMATION_MARGIN_BYTES
    )
    terminal_required = continuity_required + clean_clone
    project_device = ROOT.stat().st_dev
    return {
        "volume": {
            "root": _path_reference(root),
            "capacity_bytes": usage.total,
            "free_bytes": usage.free,
            "device": root.stat().st_dev,
        },
        "static_project": {
            "root_device": project_device,
            "same_filesystem_as_run_root": project_device == root.stat().st_dev,
            "treatment": "already reflected in free_bytes; not subtracted again",
        },
        "components": {
            "protected_floor_bytes": protected_floor,
            "user_reserve_bytes": user_reserve,
            "maintained_free_floor_bytes": maintained_floor,
            "pending_acquisition_bytes": pending,
            "observed_largest_invalid_lane_bytes": observed_lane,
            "p95_own_run_growth_bytes": own_growth,
            "peak_transient_bytes": MINIMUM_TRANSIENT_BYTES,
            "estimation_margin_bytes": ESTIMATION_MARGIN_BYTES,
            "post_lane_clean_clone_bytes": clean_clone,
        },
        "continuity_required_free_bytes": continuity_required,
        "terminal_required_free_bytes": terminal_required,
        "continuity_headroom_bytes": usage.free - continuity_required,
        "terminal_headroom_bytes": usage.free - terminal_required,
        "continuity_admitted": usage.free >= continuity_required,
        "terminal_admitted": usage.free >= terminal_required,
    }


def _probe_volume(root: Path, *, required_free_bytes: int) -> dict[str, Any]:
    """Exercise the exact filesystem properties used by the continuity writer."""

    probe = root / f".storage-probe-{uuid.uuid4().hex}"
    checks: dict[str, bool] = {
        "directories": False,
        "ordinary_write_and_fsync": False,
        "sparse_file": False,
        "atomic_rename": False,
        "advisory_lock": False,
        "checkpoint_restore": False,
        "staging_delete": False,
    }
    error: str | None = None
    before: int | None = None
    after: int | None = None
    total: int | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(root)
        before, total = usage.free, usage.total
        stage = probe / "staging"
        checkpoint_dir = probe / "checkpoints"
        stage.mkdir(parents=True)
        checkpoint_dir.mkdir()
        checks["directories"] = True
        ordinary = stage / "ordinary.bin"
        with ordinary.open("xb") as handle:
            handle.write(b"substrate-storage-bridge\n" * 2048)
            handle.flush()
            os.fsync(handle.fileno())
        checks["ordinary_write_and_fsync"] = ordinary.stat().st_size > 0
        sparse = stage / "sparse.bin"
        with sparse.open("xb") as handle:
            handle.truncate(2 * 1024 * 1024)
            handle.flush()
            os.fsync(handle.fileno())
        checks["sparse_file"] = sparse.stat().st_size == 2 * 1024 * 1024
        temporary = checkpoint_dir / "checkpoint.tmp"
        checkpoint = checkpoint_dir / "checkpoint.json"
        payload = {"goal": "storage rehearsal", "checkpoint": 0, "activation": False}
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint)
        checks["atomic_rename"] = checkpoint.is_file() and not temporary.exists()
        with checkpoint.open("r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        checks["advisory_lock"] = True
        checks["checkpoint_restore"] = json.loads(checkpoint.read_text()) == payload
        shutil.rmtree(stage)
        checks["staging_delete"] = not stage.exists()
        after = shutil.disk_usage(root).free
    except (OSError, ValueError, json.JSONDecodeError) as caught:
        error = f"{type(caught).__name__}: {caught}"
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    if total is None:
        try:
            usage = shutil.disk_usage(root)
            before, after, total = usage.free, usage.free, usage.total
        except OSError:
            pass
    floor = _protected_floor(total or 0) if total else None
    return {
        "root": _path_reference(root),
        "filesystem_device": os.stat(root).st_dev if root.exists() else None,
        "capacity_bytes": total,
        "free_bytes_before": before,
        "free_bytes_after": after,
        "protected_floor_bytes": floor,
        "required_free_bytes": required_free_bytes,
        "checks": checks,
        "all_filesystem_checks_pass": all(checks.values()),
        "free_target_pass": after is not None and after >= required_free_bytes,
        "error": error,
        "eligible": error is None
        and all(checks.values())
        and after is not None
        and after >= required_free_bytes,
    }


def _select_volume() -> dict[str, Any]:
    probes = []
    for root in _candidate_roots():
        budget = _storage_budget(root)
        probes.append(
            {
                **_probe_volume(
                    root, required_free_bytes=budget["continuity_required_free_bytes"]
                ),
                "budget": budget,
            }
        )
    selected = next((probe for probe in probes if probe["eligible"]), None)
    return {
        "candidate_order": [_path_reference(path) for path in _candidate_roots()],
        "probes": probes,
        "selected": selected,
        "selection_rule": "first explicit candidate with every filesystem check and its measured continuity budget",
    }


def _load_seal() -> tuple[Path, dict[str, Any]]:
    seal_path, seal = execution._active_continuity_seal()
    if not seal_path.is_file():
        raise base.Refused("storage bridge requires an immutable storage bridge seal")
    if seal.get("schema") not in {
        "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_SEAL",
        "SUBSTRATE_SANDBOX_DYNAMIC_STORAGE_BRIDGE_SEAL",
        "SUBSTRATE_SANDBOX_DYNAMIC_STORAGE_BRIDGE_LAUNCH_REPAIR_SEAL",
    }:
        raise base.Refused("invalid storage bridge seal")
    if seal.get("storage_bridge_source_sha256") != _sha_file(_source_path()):
        raise base.Refused("storage bridge source drifted after its seal")
    execution._validate_continuity_seal()
    return seal_path, seal


def preflight() -> dict[str, Any]:
    """Write a new, additive authority without changing the historic preflight."""

    selection = _select_volume()
    document = base.authority(
        "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_PREFLIGHT",
        {
            "storage_bridge_version": VERSION,
            **selection,
            "historical_preflight_rewritten": False,
            "candidate_or_outcome_data_changed": False,
            "activation": False,
        },
        status="admitted" if selection["selected"] else "refused",
    )
    execution._write_json(EVIDENCE / "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_PREFLIGHT.json", document)
    return document


def dynamic_plan() -> dict[str, Any]:
    """Persist the current phase-specific storage model for the dynamic seal."""

    budget = _storage_budget(RUNS)
    document = base.authority(
        "SUBSTRATE_SANDBOX_DYNAMIC_STORAGE_PLAN",
        {
            "storage_bridge_version": VERSION,
            "budget": budget,
            "admitted": budget["continuity_admitted"],
            "dynamic_policy": {
                "project_bytes_counted_twice": False,
                "own_run_growth_is_measured_separately_from_external_drift": True,
                "clean_clone_reserved_only_for_terminal_phase": True,
                "user_reserve_environment": "SUBSTRATE_TANGIBLE_USER_RESERVE_GIB",
            },
            "activation": False,
        },
        status="admitted" if budget["continuity_admitted"] else "refused",
    )
    execution._write_json(DYNAMIC_PLAN_PATH, document)
    return document


def _write_state(path: Path, value: dict[str, Any]) -> None:
    execution._write_json(path, value)


def _rehearse(
    root: Path, *, state_path: Path, continuity_required_free_bytes: int
) -> dict[str, Any]:
    """Run the actual checkpoint writer for a bounded, monitored hour."""

    rehearsal_root = state_path.parent / "rehearsal"
    workspace = rehearsal_root / "workspace"
    staging = rehearsal_root / "staging"
    if rehearsal_root.exists():
        raise base.Refused("storage rehearsal root already exists")
    rehearsal_root.mkdir(parents=True)
    initial = shutil.disk_usage(root).free
    target = continuity_required_free_bytes
    started = time.monotonic()
    first = execution._continuity_work(
        scheduled_hour=0,
        event="start",
        activity="project_intake",
        workspace=workspace,
    )
    restored = execution._run_restart_recovery(ROOT / first["checkpoint"])
    ticks = 0
    while True:
        elapsed = time.monotonic() - started
        free = shutil.disk_usage(root).free
        if free < target:
            raise base.Refused("storage rehearsal crossed the verified free-space target")
        _write_state(
            state_path,
            {
                "schema": "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_STATE",
                "status": "rehearsing",
                "elapsed_seconds": round(elapsed, 3),
                "target_seconds": REHEARSAL_SECONDS,
                "disk_free_bytes": free,
                "required_free_bytes": target,
                "ticks": ticks,
                "activation": False,
            },
        )
        if elapsed >= REHEARSAL_SECONDS:
            break
        staging.mkdir(parents=True, exist_ok=True)
        with (staging / "artifact-writer.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"tick": ticks, "elapsed": round(elapsed, 3)}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        ticks += 1
        time.sleep(min(HEARTBEAT_SECONDS, REHEARSAL_SECONDS - elapsed))
    final = shutil.disk_usage(root).free
    elapsed = max(time.monotonic() - started, 0.001)
    growth = max(0, initial - final)
    projected_growth = int(growth * (C.LONGITUDINAL_HOURS * 3600 / elapsed))
    protected = _protected_floor(shutil.disk_usage(root).total)
    projected_final = final - projected_growth
    shutil.rmtree(staging, ignore_errors=True)
    if projected_final < target:
        raise base.Refused("storage rehearsal projects a final free space below the protected floor")
    return {
        "root": _path_reference(root),
        "actual_elapsed_seconds": round(elapsed, 3),
        "checkpoint_writer": first,
        "checkpoint_restore": restored,
        "initial_free_bytes": initial,
        "final_free_bytes": final,
        "observed_growth_bytes": growth,
        "projected_24h_growth_bytes": projected_growth,
        "required_free_bytes": target,
        "protected_floor_bytes": protected,
        "projected_final_free_bytes": projected_final,
        "ticks": ticks,
        "all_pass": True,
    }


def _job_plist(*, label: str, manifest: Path, stdout: Path, stderr: Path) -> dict[str, Any]:
    return {
        "Label": label,
        "ProgramArguments": [
            sys.executable,
            "-m",
            "substrate.sandbox",
            "storage-bridge-worker",
            "--manifest",
            str(manifest),
        ],
        "WorkingDirectory": str(ROOT),
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(ROOT / "src"),
            "SUBSTRATE_REPOSITORY_ROOT": str(ROOT),
            "SUBSTRATE_STORAGE_BRIDGE_SUPERVISOR": "launchd",
        },
        "KeepAlive": False,
        "RunAtLoad": False,
        "ProcessType": "Adaptive",
        "ThrottleInterval": 60,
        "StandardOutPath": str(stdout),
        "StandardErrorPath": str(stderr),
        "AbandonProcessGroup": False,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    root = (RUNS / ROOT_NAME).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise base.Refused("storage bridge manifest must be a bridge-local receipt")
    manifest = base.load_json(resolved)
    if manifest.get("schema") != "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_MANIFEST":
        raise base.Refused("invalid storage bridge manifest")
    _, seal = _load_seal()
    if manifest.get("storage_bridge_seal_sha256") != seal.get("sha256"):
        raise base.Refused("bridge manifest is not bound to the active storage seal")
    if manifest.get("storage_bridge_source_sha256") != _sha_file(_source_path()):
        raise base.Refused("bridge manifest source identity drifted")
    return manifest


def launch() -> dict[str, Any]:
    """Detach a preflight→rehearsal→longitudinal coordinator through launchd."""

    if sys.platform != "darwin" or not Path("/bin/launchctl").is_file():
        raise base.Refused("storage bridge launch requires macOS launchd")
    if (RUNS / "longitudinal").exists():
        raise base.Refused("existing longitudinal root must be completed or invalidated")
    seal_path, seal = _load_seal()
    selection = _select_volume()
    selected = selection["selected"]
    if selected is None:
        raise base.Refused("no configured storage candidate passed the bridge preflight")
    root = Path(selected["root"])
    # The current implementation writes its continuity receipts beneath RUNS;
    # accepting another root would falsely claim isolation without relocating it.
    if root.resolve() != RUNS.resolve():
        raise base.Refused("configured run root must equal the active campaign RUNS root")
    run_id = f"bridge-{uuid.uuid4().hex}"
    bridge_root = _bridge_root(run_id)
    bridge_root.mkdir(parents=True, exist_ok=False)
    manifest_path = bridge_root / "manifest.json"
    state_path = bridge_root / "state.json"
    plist_path = bridge_root / "launchd.plist"
    stdout = bridge_root / "stdout.log"
    stderr = bridge_root / "stderr.log"
    label = f"org.substrate.tangible-sandbox-r2.storage.{run_id}"
    domain = f"gui/{os.getuid()}"
    manifest = base.authority(
        "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_MANIFEST",
        {
            "storage_bridge_version": VERSION,
            "run_id": run_id,
            "launchd_label": label,
            "launchd_domain": domain,
            "storage_bridge_seal": _path_reference(seal_path),
            "storage_bridge_seal_sha256": seal["sha256"],
            "storage_bridge_source_sha256": _sha_file(_source_path()),
            "execution_source_sha256": _sha_file(Path(execution.__file__)),
            "run_root": selected["root"],
            "run_root_device": selected["filesystem_device"],
            "preflight": selection,
            "dynamic_budget": selected["budget"],
            "rehearsal_seconds": REHEARSAL_SECONDS,
            "completion_authority": "longitudinal worker result only",
            "bridge_exit_is_not_experimental_completion": True,
        },
        status="prepared",
    )
    _write_state(
        manifest_path,
        manifest,
    )
    _write_state(
        state_path,
        {
            "schema": "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_STATE",
            "run_id": run_id,
            "status": "launch_prepared",
            "manifest_sha256": manifest["sha256"],
            "bridge_exit_is_not_experimental_completion": True,
            "activation": False,
        },
    )
    with plist_path.open("wb") as handle:
        plistlib.dump(
            _job_plist(label=label, manifest=manifest_path, stdout=stdout, stderr=stderr),
            handle,
            sort_keys=True,
        )
    bootstrap = execution._launchctl(["bootstrap", domain, str(plist_path)])
    if not bootstrap["ok"]:
        _write_state(
            state_path,
            {
                "schema": "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_STATE",
                "run_id": run_id,
                "status": "launch_failed",
                "manifest_sha256": manifest["sha256"],
                "bootstrap": bootstrap,
                "activation": False,
            },
        )
        raise base.Refused(f"storage bridge bootstrap failed: {bootstrap['stderr']}")
    kickstart = execution._launchctl(["kickstart", "-p", f"{domain}/{label}"])
    document = base.authority(
        "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_LAUNCH",
        {
            "run_id": run_id,
            "launchd_label": label,
            "launchd_domain": domain,
            "manifest": _path_reference(manifest_path),
            "manifest_sha256": manifest["sha256"],
            "plist": _path_reference(plist_path),
            "plist_sha256": _sha_file(plist_path),
            "bootstrap": bootstrap,
            "kickstart": kickstart,
            "launch_succeeded": kickstart["ok"],
            "completion_authority": "longitudinal worker result only",
            "bridge_exit_is_not_experimental_completion": True,
        },
        status="launched" if kickstart["ok"] else "launch_failed",
    )
    _write_state(bridge_root / "launch.json", document)
    if not kickstart["ok"]:
        execution._launchctl(["bootout", f"{domain}/{label}"])
        raise base.Refused(f"storage bridge kickstart failed: {kickstart['stderr']}")
    return document


def worker(manifest_path: Path) -> dict[str, Any]:
    """Perform the bounded rehearsal, then detach the real longitudinal lane."""

    if os.environ.get("SUBSTRATE_STORAGE_BRIDGE_SUPERVISOR") != "launchd":
        raise base.Refused("storage bridge worker must be owned by its user launchd agent")
    manifest = _load_manifest(manifest_path)
    root = Path(manifest["run_root"])
    bridge_root = manifest_path.resolve().parent
    state_path = bridge_root / "state.json"
    previous_handlers: dict[signal.Signals, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise RuntimeError(f"storage bridge received signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous_handlers[signum] = signal.signal(signum, interrupted)
    try:
        _write_state(
            state_path,
            {
                "schema": "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_STATE",
                "run_id": manifest["run_id"],
                "status": "rehearsing",
                "bridge_pid": os.getpid(),
                "manifest_sha256": manifest["sha256"],
                "activation": False,
            },
        )
        budget = manifest.get("dynamic_budget", {})
        required_free = budget.get("continuity_required_free_bytes")
        if not isinstance(required_free, int) or required_free <= 0:
            raise base.Refused("bridge manifest has no valid dynamic continuity budget")
        rehearsal = _rehearse(
            root,
            state_path=state_path,
            continuity_required_free_bytes=required_free,
        )
        _write_state(bridge_root / "rehearsal.json", rehearsal)
        os.environ["SUBSTRATE_TANGIBLE_MIN_FREE_BYTES"] = str(required_free)
        longitudinal_launch = execution.launch_longitudinal_supervised()
        result = base.authority(
            "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_RESULT",
            {
                "run_id": manifest["run_id"],
                "status": "longitudinal_detached",
                "rehearsal": rehearsal,
                "longitudinal_launch": longitudinal_launch,
                "completion_authority": "longitudinal worker result only",
                "bridge_exit_is_not_experimental_completion": True,
            },
            status="longitudinal_detached",
        )
        _write_state(state_path, result)
        _write_state(bridge_root / "result.json", result)
        return result
    except BaseException as error:
        result = base.authority(
            "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_RESULT",
            {
                "run_id": manifest["run_id"],
                "status": "refused_or_interrupted",
                "exception": f"{type(error).__name__}: {error}",
                "completion_authority": "longitudinal worker result only",
                "bridge_exit_is_not_experimental_completion": True,
            },
            status="refused_or_interrupted",
        )
        _write_state(state_path, result)
        _write_state(bridge_root / "result.json", result)
        raise
    finally:
        for handled, previous in previous_handlers.items():
            signal.signal(handled, previous)


def status() -> dict[str, Any]:
    root = RUNS / ROOT_NAME
    manifests = sorted(root.glob("*/manifest.json"), key=lambda path: path.stat().st_mtime)
    if not manifests:
        raise base.Refused("no storage bridge launch exists")
    manifest_path = manifests[-1]
    manifest = base.load_json(manifest_path)
    state_path = manifest_path.parent / "state.json"
    state = base.load_json(state_path) if state_path.is_file() else {}
    launchd = execution._launchctl(
        ["print", f"{manifest['launchd_domain']}/{manifest['launchd_label']}"]
    )
    longitudinal = None
    with contextlib.suppress(base.Refused):
        longitudinal = execution.longitudinal_supervision_status()
    return {
        "schema": "SUBSTRATE_SANDBOX_STORAGE_BRIDGE_STATUS",
        "run_id": manifest["run_id"],
        "state": state,
        "launchd": launchd,
        "longitudinal": longitudinal,
        "bridge_exit_is_not_experimental_completion": True,
        "activation": False,
    }
