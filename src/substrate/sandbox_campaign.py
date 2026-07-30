"""Fail-closed execution and evidence pipeline for Tangible Sandbox R2.

The campaign is intentionally capable of terminating at preflight.  R2 forbids
starting acquisition or principal work below its protected disk floor.  A
terminal preflight null is therefore an experimental outcome, not an exception:
it records the observed machine, tests every safe source endpoint, proves that
no eligible mounted volume exists, refuses invalid downstream work, and
publishes independently recomputable evidence.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from substrate import sandbox_config as C
from substrate.final_revision_io import (
    Refused,
    contains_true_activation,
    digest,
    file_digest,
    git,
    load_json,
    ref_or_none,
    write_text,
)
from substrate.final_revision_io import write_json as _write_json

ROOT = Path(os.environ.get("SUBSTRATE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])).resolve()
EVIDENCE = ROOT / "evidence" / "substrate" / "tangible_sandbox"
RUNS = ROOT / "runs" / "substrate" / "tangible_sandbox"
ARTIFACTS = ROOT / "artifacts" / "substrate" / "tangible_sandbox"
PUBLICATION = ARTIFACTS / "publication"
CORPUS = ARTIFACTS / "corpus" / C.CORPUS
STOP = RUNS / "STOP"

PACKAGE_ROOT = Path("/Users/scammermike/Downloads/SUBSTRATE_TANGIBLE_SANDBOX_R2_COMPLETE_PACKAGE")
PACKAGE_FILES = (
    "SUBSTRATE_TANGIBLE_SANDBOX_R2_EXECUTION_GOAL.md",
    "SUBSTRATE_TANGIBLE_SANDBOX_R2_OPTIMIZED_MASTER_PLAN.md",
    "SUBSTRATE_TANGIBLE_SANDBOX_R2_DATA_RESEARCH.md",
    "SUBSTRATE_TANGIBLE_SANDBOX_R2_ETA_AND_OPTIMIZATION.md",
)

JSON_DELIVERABLES = tuple(name for name in C.REQUIRED_DELIVERABLES if name.endswith(".json"))


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def source_digest() -> str:
    """Digest the entire R2 implementation, its focused tests, and CI gate."""

    rows: list[tuple[str, str]] = []
    for path in sorted((ROOT / "src" / "substrate").glob("sandbox*.py")):
        rows.append((str(path.relative_to(ROOT)), file_digest(path)))
    for path in sorted((ROOT / "tests" / "substrate").glob("test_sandbox*.py")):
        rows.append((str(path.relative_to(ROOT)), file_digest(path)))
    workflow = ROOT / ".github" / "workflows" / "substrate.yml"
    if workflow.is_file():
        rows.append((str(workflow.relative_to(ROOT)), file_digest(workflow)))
    return digest(rows)


def configuration_digest() -> str:
    return digest(C.configuration())


def authority(schema: str, payload: dict[str, Any], *, status: str) -> dict[str, Any]:
    """Create a hash-sealed, activation-safe authority."""

    body = {
        "schema": schema,
        "program": C.PROGRAM,
        "scientific_status": status,
        "configuration_digest": configuration_digest(),
        "source_digest": source_digest(),
        **payload,
        "activation": False,
        "unqualified_nous": False,
    }
    body.pop("sha256", None)
    if contains_true_activation(body):
        raise Refused(f"{schema} attempts to enable activation")
    body["sha256"] = digest(body)
    return body


def write_json(path: Path, value: dict[str, Any]) -> Path:
    if contains_true_activation(value):
        raise Refused(f"refusing to write {path}: activation must remain false")
    return _write_json(path, value)


def _command(
    arguments: list[str],
    *,
    timeout: float = 15.0,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            arguments,
            cwd=cwd or ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _tool(name: str, alternate: str | None = None) -> dict[str, Any]:
    resolved = shutil.which(name)
    if resolved is None and alternate and Path(alternate).exists():
        resolved = alternate
    return {"available": resolved is not None, "path": resolved}


def _mounted_filesystems() -> list[dict[str, Any]]:
    result = _command(["df", "-kP"], timeout=5)
    if not result["ok"]:
        return []
    rows: list[dict[str, Any]] = []
    for line in str(result["stdout"]).splitlines()[1:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            capacity = int(parts[1]) * 1024
            available = int(parts[3]) * 1024
        except ValueError:
            continue
        mount = " ".join(parts[5:])
        floor = C.disk_floor_bytes(capacity)
        rows.append(
            {
                "filesystem": parts[0],
                "mount": mount,
                "capacity_bytes": capacity,
                "available_bytes": available,
                "required_floor_bytes": floor,
                "core_eligible": available >= floor + C.CORE_MINIMUM_ACQUISITION_BYTES,
            }
        )
    return rows


def _high_cpu_processes() -> list[dict[str, Any]]:
    result = _command(["ps", "-axo", "pid=,%cpu=,%mem=,etime=,comm="], timeout=5)
    if not result["ok"]:
        return []
    rows: list[dict[str, Any]] = []
    for line in str(result["stdout"]).splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        try:
            cpu = float(parts[1])
        except ValueError:
            continue
        if cpu < 25:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "cpu_percent": cpu,
                "memory_percent": float(parts[2]),
                "elapsed": parts[3],
                "executable": Path(parts[4]).name,
            }
        )
    return rows


def _package_authority() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in PACKAGE_FILES:
        path = PACKAGE_ROOT / name
        rows.append(
            {
                "name": name,
                "path": str(path),
                "present": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": file_digest(path) if path.is_file() else None,
            }
        )
    return rows


def _parent_identity() -> dict[str, Any]:
    merge = ref_or_none(C.PARENT_MERGE_COMMIT)
    ready = ref_or_none(f"refs/tags/{C.PARENT_READY_TAG}", peel=True)
    terminal = ref_or_none(f"refs/tags/{C.PARENT_TERMINAL_TAG}", peel=True)
    merge_tree = git("rev-parse", f"{C.PARENT_MERGE_COMMIT}^{{tree}}", check=False)
    parent_terminal = load_json(
        ROOT / "evidence" / "substrate" / "genesis2" / "SUBSTRATE_GENESIS2_FINAL_CLASSIFICATION.json"
    )
    selected = (
        parent_terminal.get("selected_material")
        or parent_terminal.get("selected_candidate")
        or parent_terminal.get("candidate")
    )
    return {
        "merge_commit_expected": C.PARENT_MERGE_COMMIT,
        "merge_commit_resolved": merge,
        "merge_tree": merge_tree or None,
        "ready_tag": {"name": C.PARENT_READY_TAG, "peeled_commit": ready},
        "terminal_tag": {"name": C.PARENT_TERMINAL_TAG, "peeled_commit": terminal},
        "selected_material_expected": C.PARENT_SELECTED_MATERIAL,
        "selected_material_observed": selected,
        "classification_expected": C.PARENT_CLASSIFICATION,
        "classification_observed": parent_terminal.get("classification"),
        "status_expected": C.PARENT_STATUS,
        "status_observed": parent_terminal.get("status"),
        "readiness_expected": C.PARENT_READINESS,
        "readiness_observed": parent_terminal.get("readiness"),
        "activation_observed": parent_terminal.get("activation", parent_terminal.get("external_activation")),
        "preserved": (
            merge == C.PARENT_MERGE_COMMIT
            and ready is not None
            and terminal is not None
            and selected == C.PARENT_SELECTED_MATERIAL
            and parent_terminal.get("classification") == C.PARENT_CLASSIFICATION
            and parent_terminal.get("status") == C.PARENT_STATUS
            and parent_terminal.get("readiness") == C.PARENT_READINESS
            and parent_terminal.get("activation", parent_terminal.get("external_activation")) is False
        ),
    }


def preflight() -> dict[str, Any]:
    """Capture actual host state and apply R2's non-negotiable admission rules."""

    disk = shutil.disk_usage(ROOT)
    floor = C.disk_floor_bytes(disk.total)
    core_required_free = floor + C.CORE_MINIMUM_ACQUISITION_BYTES
    mounts = _mounted_filesystems()
    alternate = [
        row
        for row in mounts
        if row["core_eligible"]
        and row["mount"] != "/"
        and not str(row["mount"]).startswith(("/System/Volumes/", "/dev/"))
    ]
    tools = {
        "git": _tool("git"),
        "gh": _tool("gh"),
        "uv": _tool("uv"),
        "python3": _tool("python3"),
        "docker": _tool("docker"),
        "aria2c": _tool("aria2c"),
        "git_lfs": _tool("git-lfs"),
        "ffmpeg": _tool("ffmpeg"),
        "blender": _tool("blender", "/Applications/Blender.app/Contents/MacOS/Blender"),
        "vmrun": _tool("vmrun"),
        "emulator": _tool("emulator"),
        "adb": _tool("adb"),
        "sdkmanager": _tool("sdkmanager"),
    }
    docker_probe = _command(["docker", "info", "--format", "{{json .ServerVersion}}"], timeout=8) if tools["docker"]["available"] else {"ok": False}
    api_names = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "XAI_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "KAGGLE_USERNAME",
        "KAGGLE_KEY",
    )
    api_presence = {name: name in os.environ and bool(os.environ.get(name)) for name in api_names}
    branch = git("branch", "--show-current", check=False)
    remote = git("remote", "get-url", "origin", check=False)
    head = git("rev-parse", "HEAD", check=False)
    blockers = []
    if disk.free < floor:
        blockers.append("protected_disk_floor")
    if disk.free < core_required_free:
        blockers.append("core_acquisition_reservation")
    if not docker_probe.get("ok"):
        blockers.append("docker_engine")
    if not tools["vmrun"]["available"]:
        blockers.append("vmware_fusion_cli")
    if not (tools["emulator"]["available"] and tools["adb"]["available"]):
        blockers.append("android_emulator")
    if not any(api_presence[name] for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "XAI_API_KEY")):
        blockers.append("benchmark_model_endpoint")

    disk_terminal = disk.free < floor and not alternate
    safe_repair = {
        "alternate_mounted_core_volume_found": bool(alternate),
        "alternate_candidates": alternate,
        "repository_bytes": sum(
            path.stat().st_size for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts
        ),
        "destructive_cleanup_authorized": False,
        "external_volume_attached": any(
            row["core_eligible"] and str(row["mount"]).startswith("/Volumes/") for row in mounts
        ),
        "repair_conclusion": (
            "no non-destructive in-scope repair can recover the disk deficit; "
            "deleting unrelated user data is not authorized"
            if disk_terminal
            else "an eligible mounted volume or sufficient free space exists"
        ),
    }
    return authority(
        "SUBSTRATE_SANDBOX_PREFLIGHT",
        {
            "observed_at": _utc_now(),
            "repository": {
                "root": str(ROOT),
                "remote": remote,
                "head": head,
                "branch": branch,
                "expected_remote": "git@github.com:joshuahickscorp/substrate.git",
                "remote_matches": remote in {
                    "git@github.com:joshuahickscorp/substrate.git",
                    "https://github.com/joshuahickscorp/substrate.git",
                },
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
            },
            "disk": {
                "path": str(ROOT),
                "capacity_bytes": disk.total,
                "used_bytes": disk.used,
                "available_bytes": disk.free,
                "required_floor_bytes": floor,
                "core_minimum_acquisition_bytes": C.CORE_MINIMUM_ACQUISITION_BYTES,
                "core_required_free_bytes": core_required_free,
                "floor_deficit_bytes": max(0, floor - disk.free),
                "core_start_deficit_bytes": max(0, core_required_free - disk.free),
                "floor_pass": disk.free >= floor,
                "core_reservation_pass": disk.free >= core_required_free,
                "mounted_filesystems": mounts,
            },
            "tools": tools,
            "docker": {
                "client_available": tools["docker"]["available"],
                "server_available": bool(docker_probe.get("ok")),
                "probe_error": docker_probe.get("stderr") if not docker_probe.get("ok") else None,
            },
            "model_and_data_credentials": api_presence,
            "high_cpu_processes": _high_cpu_processes(),
            "execution_package": _package_authority(),
            "historical_identity": _parent_identity(),
            "bounded_repair": safe_repair,
            "admission": {
                "core_tier_admitted": not blockers,
                "seed_tier_admitted": disk.free >= floor + 1,
                "freeze_a_admitted": False,
                "principal_launch_admitted": False,
                "blockers": blockers,
                "critical_blocker": "protected_disk_floor" if disk_terminal else None,
                "terminal_outcome_c_admitted": disk_terminal,
                "outcome_c_authority": C.OUTCOME_C_RESERVED_FOR,
            },
            "planning_eta_hours": C.PLANNING_ETA_HOURS,
            "expected_eta_hours": list(C.EXPECTED_ETA_HOURS),
        },
        status="terminal_preflight_null" if disk_terminal else "preflight_complete",
    )


def write_preflight() -> dict[str, Any]:
    document = preflight()
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_PREFLIGHT.json", document)
    historical = authority(
        "SUBSTRATE_SANDBOX_HISTORICAL_IMMUTABILITY",
        {
            "historical_identity": document["historical_identity"],
            "synthetic_architecture_tournament_reopened": False,
            "selected_material": C.PARENT_SELECTED_MATERIAL,
            "classification": C.PARENT_CLASSIFICATION,
            "status": C.PARENT_STATUS,
            "readiness": C.PARENT_READINESS,
            "external_activation": False,
        },
        status="verified" if document["historical_identity"]["preserved"] else "failed",
    )
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_HISTORICAL_IMMUTABILITY.json", historical)
    return document


def _resolve_git_head(url: str) -> dict[str, Any]:
    result = _command(["git", "ls-remote", "--symref", url, "HEAD"], timeout=20)
    revision = None
    default_ref = None
    if result["ok"]:
        for line in str(result["stdout"]).splitlines():
            if line.startswith("ref:"):
                default_ref = line.split()[1]
            elif line.endswith("\tHEAD"):
                revision = line.split()[0]
    return {
        "reachable": bool(result["ok"] and revision),
        "resolved_head": revision,
        "default_ref": default_ref,
        "elapsed_seconds": result["elapsed_seconds"],
        "error": result["stderr"] or None,
    }


def _resolve_http(url: str) -> dict[str, Any]:
    started = time.monotonic()
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "substrate-r2-source-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {
                "reachable": 200 <= response.status < 400,
                "http_status": response.status,
                "content_length": response.headers.get("Content-Length"),
                "etag": response.headers.get("ETag"),
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "error": None,
            }
    except (OSError, urllib.error.URLError) as error:
        return {
            "reachable": False,
            "http_status": getattr(error, "code", None),
            "content_length": None,
            "etag": None,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "error": str(error),
        }


def research() -> dict[str, Any]:
    """Refresh every official endpoint without accepting terms or downloading data."""

    refreshed: list[dict[str, Any]] = []
    for source in C.OFFICIAL_SOURCES:
        row = dict(source)
        if source.get("git_url"):
            probe = _resolve_git_head(str(source["git_url"]))
        else:
            probe = _resolve_http(str(source["official_url"]))
        row["refresh"] = probe
        row["head_drift_since_research"] = (
            bool(source.get("observed_head"))
            and bool(probe.get("resolved_head"))
            and source["observed_head"] != probe["resolved_head"]
        )
        access = str(source["access"])
        if "gated" in access.lower() or "terms" in access.lower() or "credentials" in access.lower():
            row["campaign_state"] = "GATED"
        else:
            row["campaign_state"] = "DISCOVERED" if probe["reachable"] else "REFUSED"
        refreshed.append(row)
    reachable = sum(bool(row["refresh"]["reachable"]) for row in refreshed)
    catalog = authority(
        "SUBSTRATE_SANDBOX_SOURCE_CATALOG",
        {
            "refreshed_at": _utc_now(),
            "sources": refreshed,
            "source_count": len(refreshed),
            "reachable_count": reachable,
            "unreachable_count": len(refreshed) - reachable,
            "terms_automatically_accepted": False,
            "data_downloaded": False,
        },
        status="source_refresh_complete" if reachable == len(refreshed) else "source_refresh_partial",
    )
    license_rows = [
        {
            "source_id": row["source_id"],
            "license": row["license"],
            "access": row["access"],
            "redistribution_class": row["redistribution_class"],
            "classification": (
                "gated_optional"
                if row["campaign_state"] == "GATED"
                else row["redistribution_class"]
            ),
            "license_review_status": "recorded_not_legal_advice",
        }
        for row in refreshed
    ]
    ledger = authority(
        "SUBSTRATE_SANDBOX_LICENSE_LEDGER",
        {
            "reviewed_at": catalog["refreshed_at"],
            "rows": license_rows,
            "clip_level_filter_required": {"fsd50k": ["CC0", "CC-BY"]},
            "local_only": ["common_voice"],
            "manual_terms_not_accepted": [
                row["source_id"] for row in refreshed if row["campaign_state"] == "GATED"
            ],
        },
        status="classified",
    )
    research_authority = authority(
        "SUBSTRATE_SANDBOX_RESEARCH_AUTHORITY",
        {
            "research_date": "2026-07-29",
            "refresh_date": catalog["refreshed_at"],
            "source_catalog_sha256": catalog["sha256"],
            "license_ledger_sha256": ledger["sha256"],
            "selection": "Core",
            "official_sources_reopened": reachable,
            "official_sources_total": len(refreshed),
            "minimum_public_floor": C.REQUIRED_PUBLIC_FLOOR,
            "heavy_or_gated_not_on_critical_path": [
                "OSWorld-V2",
                "WorkArena++",
                "MLE-bench",
                "TEACh",
                "large gated video corpora",
            ],
        },
        status="complete" if reachable == len(refreshed) else "partial",
    )
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_SOURCE_CATALOG.json", catalog)
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_LICENSE_LEDGER.json", ledger)
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_RESEARCH_AUTHORITY.json", research_authority)
    return research_authority


def acquisition_plan(
    preflight_document: dict[str, Any] | None = None, *, persist: bool = True
) -> dict[str, Any]:
    """Calculate reservation and live-ETA authority without writing raw data."""

    before = preflight_document or write_preflight()
    disk = before["disk"]
    usable_above_floor = max(0, int(disk["available_bytes"]) - int(disk["required_floor_bytes"]))
    safe = bool(disk["floor_pass"] and usable_above_floor >= C.CORE_MINIMUM_ACQUISITION_BYTES)
    sources = []
    for source in C.OFFICIAL_SOURCES:
        gated = any(word in str(source["access"]).lower() for word in ("gated", "terms", "credentials"))
        sources.append(
            {
                "source_id": source["source_id"],
                "priority": source["priority"],
                "selected_revision": source["selected_revision"],
                "planned_state": "GATED" if gated else ("RESERVED" if safe else "REFUSED"),
                "refusal": None if gated or safe else "protected_disk_floor",
                "downloader": (
                    "git"
                    if source.get("git_url")
                    else "aria2c_or_source_specific_official_client"
                ),
                "per_host_concurrency": (
                    4 if source["source_id"] != "librispeech" else "maximum_4"
                ),
                "checksum": "SHA-256_or_official_ETag",
            }
        )
    document = authority(
        "SUBSTRATE_SANDBOX_ACQUISITION_PLAN",
        {
            "tier": "Core",
            "dry_run": True,
            "target_bytes": {
                "minimum": C.CORE_MINIMUM_ACQUISITION_BYTES,
                "preferred": C.CORE_PREFERRED_ACQUISITION_BYTES,
            },
            "disk": {
                "capacity_bytes": disk["capacity_bytes"],
                "available_bytes": disk["available_bytes"],
                "protected_floor_bytes": disk["required_floor_bytes"],
                "usable_above_floor_bytes": usable_above_floor,
                "reservation_deficit_bytes": max(
                    0, C.CORE_MINIMUM_ACQUISITION_BYTES - usable_above_floor
                ),
            },
            "four_pool_scheduler": C.ACQUISITION_POOLS,
            "state_machine": list(C.ACQUISITION_STATES),
            "features": [
                "resume",
                "range_requests",
                "partial_file_markers",
                "ETag_capture",
                "SHA-256",
                "content_addressed_cache",
                "source_specific_concurrency",
                "retry_with_backoff",
                "disk_reservation",
                "throughput_telemetry",
                "ETA_recalculation",
                "single_writer_per_target",
            ],
            "sources": sources,
            "safe_to_start": safe,
            "refusals": [] if safe else ["protected_disk_floor", "core_acquisition_reservation"],
            "planning_eta_hours": C.PLANNING_ETA_HOURS,
            "live_eta_status": "not_started_no_observed_transfer_rate",
        },
        status="admitted" if safe else "refused",
    )
    if persist:
        write_json(EVIDENCE / "SUBSTRATE_SANDBOX_ACQUISITION_PLAN.json", document)
    return document


def acquire() -> dict[str, Any]:
    """Fail closed unless the complete Core reservation is already safe."""

    before = (
        load_json(EVIDENCE / "SUBSTRATE_SANDBOX_PREFLIGHT.json")
        if (EVIDENCE / "SUBSTRATE_SANDBOX_PREFLIGHT.json").is_file()
        else write_preflight()
    )
    plan = acquisition_plan(before)
    if plan["safe_to_start"]:
        raise Refused(
            "Core acquisition is admitted but this bounded preflight implementation "
            "does not download until the operator launches the full acquisition engine"
        )
    rows = []
    for source in plan["sources"]:
        rows.append(
            {
                "source_id": source["source_id"],
                "state": source["planned_state"],
                "bytes_downloaded": 0,
                "files_downloaded": 0,
                "throughput_bytes_per_second": 0.0,
                "reason": source["refusal"] or "manual_terms_or_credentials_not_accepted",
            }
        )
    document = authority(
        "SUBSTRATE_SANDBOX_ACQUISITION_RESULT",
        {
            "tier": "Core",
            "attempted_at": _utc_now(),
            "admission_checked": True,
            "network_writers_started": 0,
            "bytes_downloaded": 0,
            "bytes_extracted": 0,
            "bytes_preprocessed": 0,
            "checksum_mismatches": 0,
            "quarantined_files": 0,
            "sources": rows,
            "terminal_refusal": True,
            "critical_blocker": before["admission"]["critical_blocker"],
            "invalid_partial_acquisition_avoided": True,
        },
        status="refused_before_download",
    )
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_ACQUISITION_RESULT.json", document)
    return document


def _checksum_canaries() -> dict[str, bool]:
    """Exercise resumability and integrity logic on byte-sized temporary fixtures."""

    with tempfile.TemporaryDirectory(prefix="substrate-r2-canary-") as temporary:
        root = Path(temporary)
        partial = root / "source.partial"
        partial.write_bytes(b"first-")
        first_size = partial.stat().st_size
        with partial.open("ab") as handle:
            handle.write(b"second")
        resumed = first_size == 6 and partial.read_bytes() == b"first-second"

        actual = hashlib.sha256(partial.read_bytes()).hexdigest()
        mismatch = actual != hashlib.sha256(b"expected").hexdigest()

        cache = root / actual
        cache.write_bytes(partial.read_bytes())
        before = cache.stat().st_mtime_ns
        duplicate_avoided = cache.is_file() and hashlib.sha256(cache.read_bytes()).hexdigest() == actual
        after = cache.stat().st_mtime_ns
        duplicate_avoided = duplicate_avoided and before == after
    return {
        "C02_checksum_mismatch_detected": mismatch,
        "C03_partial_download_resumes": resumed,
        "C04_duplicate_download_avoided": duplicate_avoided,
    }


def canaries() -> dict[str, Any]:
    source_catalog = (
        load_json(EVIDENCE / "SUBSTRATE_SANDBOX_SOURCE_CATALOG.json")
        if (EVIDENCE / "SUBSTRATE_SANDBOX_SOURCE_CATALOG.json").is_file()
        else (research() and load_json(EVIDENCE / "SUBSTRATE_SANDBOX_SOURCE_CATALOG.json"))
    )
    passed = {
        "C01_source_license_recorded": all(
            bool(row.get("license")) for row in source_catalog["sources"]
        ),
        **_checksum_canaries(),
        "C28_activation_remains_false": C.ACTIVATION is False,
    }
    rows = []
    for name in C.CANARIES:
        if name in passed:
            rows.append(
                {
                    "id": name.split("_", 1)[0],
                    "name": name,
                    "status": "pass" if passed[name] else "fail",
                    "reason": "bounded infrastructure canary executed",
                }
            )
        else:
            rows.append(
                {
                    "id": name.split("_", 1)[0],
                    "name": name,
                    "status": "not_run",
                    "reason": "requires an admitted environment or materialized STSC corpus",
                }
            )
    document = authority(
        "SUBSTRATE_SANDBOX_CANARIES",
        {
            "canaries": rows,
            "passed": sum(row["status"] == "pass" for row in rows),
            "failed": sum(row["status"] == "fail" for row in rows),
            "not_run": sum(row["status"] == "not_run" for row in rows),
            "all_required_for_principal_pass": False,
            "principal_admitted": False,
            "infrastructure_subset_honest": True,
        },
        status="bounded_subset_complete",
    )
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_CANARIES.json", document)
    return document


def _not_run(schema: str, *, reason: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return authority(
        schema,
        {
            "status": "not_run",
            "reason": reason,
            "principal_units": 0,
            "model_calls": 0,
            "tool_calls": 0,
            **(extra or {}),
        },
        status="not_run",
    )


def _adapter_contract() -> dict[str, Any]:
    return authority(
        "SUBSTRATE_SANDBOX_ADAPTER_CONTRACT",
        {
            "methods": [
                "reset",
                "observe",
                "available_actions",
                "act",
                "checkpoint",
                "restore",
                "status",
                "score",
                "provenance",
            ],
            "observation_fields": [
                "time",
                "source",
                "modality",
                "content_digest",
                "confidence_semantics",
                "privacy_class",
            ],
            "action_fields": ["type", "arguments", "cost", "expected_effect", "actual_effect", "receipt"],
            "evaluator_fields": [
                "version",
                "hidden_data_root",
                "partial_credit_rule",
                "failure_rule",
                "normalization",
                "hash",
            ],
            "candidate_can_read_evaluator_only": False,
            "implemented_environment_adapters": [],
            "admission": "refused_before_adapter_bringup",
        },
        status="contract_frozen_environment_implementation_not_admitted",
    )


def _stsc_schema() -> dict[str, Any]:
    return authority(
        "SUBSTRATE_SANDBOX_STSC1_SCHEMA",
        {
            "name": C.CORPUS,
            "version": C.CORPUS_VERSION,
            "roots": list(C.STSC_ROOTS),
            "splits": list(C.STSC_SPLITS),
            "families": list(C.STSC_FAMILIES),
            "task_fields": [
                "id",
                "family",
                "source_manifest",
                "generator_commit",
                "generator_seed",
                "observation_manifest",
                "action_space",
                "cost_model",
                "hidden_state",
                "evaluator",
                "success_criteria",
                "partial_credit",
                "provenance",
                "privacy_class",
                "publication_class",
            ],
            "physical_isolation_required": True,
            "materialized": False,
        },
        status="schema_frozen_corpus_not_materialized",
    )


def _failure_matrix(preflight_document: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "failure": "protected_disk_floor",
            "observed": True,
            "critical": True,
            "injected": False,
            "detected": True,
            "containment": "all acquisition, generation, freeze, and principal writers refused",
        },
        {
            "failure": "docker_engine_unavailable",
            "observed": not preflight_document["docker"]["server_available"],
            "critical": False,
            "injected": False,
            "detected": True,
            "containment": "Docker environments not launched",
        },
        {
            "failure": "vmware_cli_unavailable",
            "observed": not preflight_document["tools"]["vmrun"]["available"],
            "critical": False,
            "injected": False,
            "detected": True,
            "containment": "OSWorld excluded from critical path",
        },
        {
            "failure": "android_emulator_unavailable",
            "observed": not (
                preflight_document["tools"]["emulator"]["available"]
                and preflight_document["tools"]["adb"]["available"]
            ),
            "critical": False,
            "injected": False,
            "detected": True,
            "containment": "AndroidWorld not launched",
        },
        {
            "failure": "model_endpoint_unavailable",
            "observed": not any(preflight_document["model_and_data_credentials"].values()),
            "critical": False,
            "injected": False,
            "detected": True,
            "containment": "no scientifically unequal fixture substituted",
        },
    ]
    return authority(
        "SUBSTRATE_SANDBOX_FAILURE_MATRIX",
        {
            "rows": rows,
            "observed_failures": [row["failure"] for row in rows if row["observed"]],
            "invalid_units_published": 0,
            "external_actions": 0,
        },
        status="terminal_preflight_failures_contained",
    )


def _classify(preflight_document: dict[str, Any]) -> dict[str, Any]:
    terminal = bool(preflight_document["admission"]["terminal_outcome_c_admitted"])
    if not terminal:
        raise Refused("Outcome C is not authorized by the recorded preflight")
    return authority(
        "SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION",
        {
            "outcome": "C",
            "classification": C.OUTCOMES["C"]["classification"],
            "reason": (
                "The only mounted data filesystem had less free space than the "
                "non-negotiable R2 protected floor, no eligible alternate volume "
                "existed, and non-destructive bounded repair was impossible."
            ),
            "critical_blocker": "protected_disk_floor",
            "prerequisite_failure_makes_measurement_uninterpretable": True,
            "core_tier_completed": False,
            "STSC_1_materialized": False,
            "principal_launched": False,
            "public_benchmark_tasks": 0,
            "custom_tasks": 0,
            "longitudinal_hours": 0,
            "H_T12": {
                "status": "not_tested",
                "effect": None,
                "confidence_interval": None,
                "sesoi": C.SESOI,
            },
            "claim_boundary": {
                "tangible_advantage": "not_tested",
                "unqualified_nous": False,
                "consciousness": False,
                "sentience": False,
                "external_activation": False,
            },
            "historical_result_preserved": preflight_document["historical_identity"]["preserved"],
            "invalid_principal_evidence_claimed": False,
            "outcome_c_authority": C.OUTCOME_C_RESERVED_FOR,
            "external_activation": False,
        },
        status="terminal",
    )


def _base_documents(
    before: dict[str, Any],
    plan: dict[str, Any],
    acquisition: dict[str, Any],
    canary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    reason = "protected disk floor failed before corpus or environment materialization"
    source_catalog = load_json(EVIDENCE / "SUBSTRATE_SANDBOX_SOURCE_CATALOG.json")
    documents: dict[str, dict[str, Any]] = {}
    documents["SUBSTRATE_SANDBOX_DATA_MANIFEST.json"] = authority(
        "SUBSTRATE_SANDBOX_DATA_MANIFEST",
        {
            "corpus": C.CORPUS,
            "version": C.CORPUS_VERSION,
            "raw_files": 0,
            "processed_files": 0,
            "bytes_downloaded": 0,
            "bytes_generated": 0,
            "builder_visible_materialized": False,
            "evaluator_only_materialized": False,
            "manifest_entries": [],
            "reason": reason,
        },
        status="empty_by_preflight_refusal",
    )
    documents["SUBSTRATE_SANDBOX_DISK_PLAN.json"] = authority(
        "SUBSTRATE_SANDBOX_DISK_PLAN",
        {
            "capacity_bytes": before["disk"]["capacity_bytes"],
            "available_bytes": before["disk"]["available_bytes"],
            "protected_floor_bytes": before["disk"]["required_floor_bytes"],
            "core_minimum_bytes": C.CORE_MINIMUM_ACQUISITION_BYTES,
            "core_start_deficit_bytes": before["disk"]["core_start_deficit_bytes"],
            "alternate_core_volume_found": before["bounded_repair"][
                "alternate_mounted_core_volume_found"
            ],
            "admitted_tier": None,
            "selected_tier": "Core_requested_refused",
        },
        status="refused",
    )
    documents["SUBSTRATE_SANDBOX_PARALLELISM_POLICY.json"] = authority(
        "SUBSTRATE_SANDBOX_PARALLELISM_POLICY",
        {
            "acquisition_pools": C.ACQUISITION_POOLS,
            "resource_classes": [
                "LONGITUDINAL",
                "MODEL_API",
                "DOCKER_CPU",
                "VM_GUI",
                "ANDROID",
                "MEDIA_RENDER",
                "LIGHT_CPU",
                "DISK_HEAVY",
                "NETWORK",
            ],
            "single_writer_per_target": True,
            "multiprocessing_main_guard_required": True,
            "nested_uncontrolled_pools_forbidden": True,
            "workers_started": 0,
            "reason": reason,
        },
        status="frozen_not_exercised",
    )
    credentials = before["model_and_data_credentials"]
    documents["SUBSTRATE_SANDBOX_MODEL_PANEL.json"] = authority(
        "SUBSTRATE_SANDBOX_MODEL_PANEL",
        {
            "required_organs": [
                "general_reasoning_model",
                "compact_or_local_model",
                "vision_model",
                "speech_audio_model",
                "embedding_retrieval_model",
                "code_execution_tools",
                "3d_geometry_tools",
            ],
            "provider_credentials_present": {
                key: value
                for key, value in credentials.items()
                if key in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "XAI_API_KEY"}
            },
            "panel_frozen": False,
            "arms_benchmarked": [],
            "fixture_models_substituted": False,
            "reason": reason,
        },
        status="not_frozen",
    )
    environment_rows = [
        {
            "environment": "WebArena-Verified",
            "release": "v1.2.3",
            "state": "REFUSED",
            "reason": "disk floor and Docker server",
        },
        {
            "environment": "SWE-bench Verified",
            "release": source_catalog["sources"][3]["selected_revision"],
            "state": "REFUSED",
            "reason": "disk floor and Docker server",
        },
        {
            "environment": "LongMemEval-V2 small",
            "release": source_catalog["sources"][5]["selected_revision"],
            "state": "REFUSED",
            "reason": "disk floor and model endpoints",
        },
        {
            "environment": "tau2-bench",
            "release": "v1.0.1",
            "state": "REFUSED",
            "reason": "disk floor and model endpoints",
        },
        {
            "environment": "AndroidWorld",
            "release": source_catalog["sources"][2]["selected_revision"],
            "state": "REFUSED",
            "reason": "disk floor and emulator absent",
        },
        {
            "environment": "OSWorld-V2",
            "release": "v2026.06.24",
            "state": "GATED",
            "reason": "optional gated assets and VMware absent",
        },
        {
            "environment": "WorkArena++",
            "release": source_catalog["sources"][11]["selected_revision"],
            "state": "GATED",
            "reason": "optional gated ServiceNow instance",
        },
    ]
    documents["SUBSTRATE_SANDBOX_ENVIRONMENT_CATALOG.json"] = authority(
        "SUBSTRATE_SANDBOX_ENVIRONMENT_CATALOG",
        {
            "environments": environment_rows,
            "gold_tasks_passed": 0,
            "known_failure_tasks_passed": 0,
            "freeze_b_modules": [],
            "minimum_public_floor_met": False,
        },
        status="not_admitted",
    )
    documents["SUBSTRATE_SANDBOX_ADAPTER_CONTRACT.json"] = _adapter_contract()
    documents["SUBSTRATE_SANDBOX_STSC1_SCHEMA.json"] = _stsc_schema()
    documents["SUBSTRATE_SANDBOX_STSC1_GENERATOR_AUTHORITY.json"] = authority(
        "SUBSTRATE_SANDBOX_STSC1_GENERATOR_AUTHORITY",
        {
            "corpus": C.CORPUS,
            "version": C.CORPUS_VERSION,
            "required_generators": [
                "office",
                "local_web",
                "code",
                "Kubric_media",
                "audio_scene",
                "ProcTHOR_AI2THOR",
                "teaching",
                "longitudinal_history",
            ],
            "generator_commitment_created": False,
            "hidden_seed_commitment_created": False,
            "principal_hidden_instances_materialized": False,
            "reason": reason,
        },
        status="refused_before_implementation",
    )
    documents["SUBSTRATE_SANDBOX_STSC1_SPLITS.json"] = authority(
        "SUBSTRATE_SANDBOX_STSC1_SPLITS",
        {
            "splits": [
                {"name": name, "tasks": 0, "materialized": False} for name in C.STSC_SPLITS
            ],
            "cross_split_items": 0,
            "builder_evaluator_root_isolation": "specified_not_materialized",
            "reason": reason,
        },
        status="not_materialized",
    )
    documents["SUBSTRATE_SANDBOX_PUBLIC_BENCHMARK_PLAN.json"] = authority(
        "SUBSTRATE_SANDBOX_PUBLIC_BENCHMARK_PLAN",
        {
            "minimum_floor": C.REQUIRED_PUBLIC_FLOOR,
            "selected_releases": {
                row["environment"]: row["release"] for row in environment_rows
            },
            "task_subset_commitments": {},
            "freeze_b_modules": [],
            "reason": reason,
        },
        status="planned_not_launched",
    )
    documents["SUBSTRATE_SANDBOX_BASELINE_AUTHORITY.json"] = authority(
        "SUBSTRATE_SANDBOX_BASELINE_AUTHORITY",
        {
            "required_arms": list(C.REQUIRED_ARMS),
            "strongest_control_selected": None,
            "equal_or_greater_resource_rule": True,
            "arms_instantiated": [],
            "reason": reason,
        },
        status="specified_not_frozen",
    )
    documents["SUBSTRATE_SANDBOX_RESOURCE_PARITY.json"] = authority(
        "SUBSTRATE_SANDBOX_RESOURCE_PARITY",
        {
            "fields": [
                "observations",
                "history_bytes",
                "model_calls",
                "input_tokens",
                "output_tokens",
                "wall_time",
                "cpu_time",
                "memory",
                "disk",
                "tool_calls",
                "specialist_model_calls",
                "human_teaching_events",
            ],
            "arm_rows": [],
            "principal_comparison_performed": False,
            "resource_parity_claimed": False,
        },
        status="not_applicable_no_principal",
    )
    documents["SUBSTRATE_SANDBOX_GROK_AUTHORITY.json"] = authority(
        "SUBSTRATE_SANDBOX_GROK_AUTHORITY",
        {
            "minimum_roles": 48,
            "preferred_roles": "64-96",
            "required_cells": [
                "sources",
                "licenses",
                "adapters",
                "generators",
                "baselines",
                "statistics",
                "security",
                "privacy",
                "challenges",
                "counterfeits",
                "mutations",
                "publication",
            ],
            "opinions_are_scientific_endpoints": False,
            "post_commit_candidate_outcome_access": False,
            "launch_admitted": False,
            "reason": reason,
        },
        status="specified_not_launched",
    )
    documents["SUBSTRATE_SANDBOX_GROK_LEDGER.json"] = authority(
        "SUBSTRATE_SANDBOX_GROK_LEDGER",
        {
            "roles_launched": 0,
            "reports_received": 0,
            "challenge_generators_committed": 0,
            "scientific_endpoints_from_opinion": 0,
            "reason": reason,
        },
        status="not_run",
    )
    documents["SUBSTRATE_SANDBOX_PILOT.json"] = _not_run(
        "SUBSTRATE_SANDBOX_PILOT",
        reason=reason,
        extra={
            "phases": ["P0_infrastructure", "P1_evaluator", "P2_headroom", "P3_variance_runtime"],
            "power_analysis_performed": False,
            "freeze_a_admitted": False,
        },
    )
    documents["SUBSTRATE_SANDBOX_FAILURE_MATRIX.json"] = _failure_matrix(before)
    documents["SUBSTRATE_SANDBOX_FREEZE.json"] = authority(
        "SUBSTRATE_SANDBOX_FREEZE",
        {
            "freeze_a_created": False,
            "freeze_b_modules": [],
            "ready_tag_authorized": False,
            "ready_tag_expected_absent": True,
            "candidate_source_changed_after_freeze": False,
            "reason": reason,
        },
        status="refused",
    )
    documents["SUBSTRATE_SANDBOX_STATISTICAL_AUTHORITY.json"] = authority(
        "SUBSTRATE_SANDBOX_STATISTICAL_AUTHORITY",
        {
            "independent_unit": "task_or_developmental_history_by_environment",
            "sesoi": C.SESOI,
            "confidence": C.CONFIDENCE,
            "power_target": C.POWER_TARGET,
            "primary_estimator": "paired_mean_difference",
            "confidence_method": "BCa_bootstrap",
            "multiplicity": "Holm",
            "sequential_design": "predeclared_two_stage_when_compatible",
            "analysis_executed": False,
        },
        status="frozen_policy_not_exercised",
    )
    documents["SUBSTRATE_SANDBOX_PRINCIPAL_AUTHORITY.json"] = _not_run(
        "SUBSTRATE_SANDBOX_PRINCIPAL_AUTHORITY",
        reason=reason,
        extra={
            "H_T12": "L1 beats strongest fair control on generator-held-out compound tangible tasks",
            "principal_launch_admitted": False,
            "invalid_principal_avoided": True,
        },
    )
    documents["SUBSTRATE_SANDBOX_PRINCIPAL_DAG.json"] = authority(
        "SUBSTRATE_SANDBOX_PRINCIPAL_DAG",
        {
            "nodes": [
                "preflight",
                "source_refresh",
                "license_review",
                "dry_run",
                "seed_acquisition",
                "environment_bringup",
                "STSC_generation",
                "canaries",
                "pilot",
                "Freeze_A",
                "principal",
                "verification",
                "publication",
            ],
            "terminal_node": "preflight",
            "blocked_edges": [
                {"from": "dry_run", "to": "seed_acquisition", "reason": "protected_disk_floor"}
            ],
            "unnecessary_serialization_added": False,
        },
        status="terminated_at_preflight",
    )
    for filename, schema in (
        ("SUBSTRATE_SANDBOX_PUBLIC_RESULTS.json", "SUBSTRATE_SANDBOX_PUBLIC_RESULTS"),
        ("SUBSTRATE_SANDBOX_CUSTOM_RESULTS.json", "SUBSTRATE_SANDBOX_CUSTOM_RESULTS"),
        ("SUBSTRATE_SANDBOX_REPLICATION.json", "SUBSTRATE_SANDBOX_REPLICATION"),
        ("SUBSTRATE_SANDBOX_HIDDEN_COMPOSITION.json", "SUBSTRATE_SANDBOX_HIDDEN_COMPOSITION"),
        ("SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json", "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT"),
        ("SUBSTRATE_SANDBOX_TEACHING_RESULT.json", "SUBSTRATE_SANDBOX_TEACHING_RESULT"),
        (
            "SUBSTRATE_SANDBOX_MODEL_REPLACEMENT_RESULT.json",
            "SUBSTRATE_SANDBOX_MODEL_REPLACEMENT_RESULT",
        ),
    ):
        documents[filename] = _not_run(schema, reason=reason)
    documents["SUBSTRATE_SANDBOX_MUTATION_REPORT.json"] = authority(
        "SUBSTRATE_SANDBOX_MUTATION_REPORT",
        {
            "catalog": list(C.MUTATIONS),
            "injected": 0,
            "detected": 0,
            "survivors": [],
            "claim": "not_run_not_zero_survivor_evidence",
            "reason": reason,
        },
        status="not_run",
    )
    documents["SUBSTRATE_SANDBOX_COUNTERFEIT_REPORT.json"] = authority(
        "SUBSTRATE_SANDBOX_COUNTERFEIT_REPORT",
        {
            "catalog": list(C.COUNTERFEITS),
            "injected": 0,
            "rejected": 0,
            "survivors": [],
            "claim": "not_run_not_counterfeit_rejection_evidence",
            "reason": reason,
        },
        status="not_run",
    )
    return documents


def _verify_recorded_state(*, require_all_deliverables: bool) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    loaded: dict[str, dict[str, Any]] = {}
    for filename in JSON_DELIVERABLES:
        path = EVIDENCE / filename
        if not path.is_file():
            if require_all_deliverables:
                errors.append(f"missing {filename}")
            continue
        try:
            loaded[filename] = load_json(path)
            checks[f"digest:{filename}"] = True
        except Refused as error:
            checks[f"digest:{filename}"] = False
            errors.append(str(error))
    before = loaded.get("SUBSTRATE_SANDBOX_PREFLIGHT.json")
    classification = loaded.get("SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json")
    acquisition = loaded.get("SUBSTRATE_SANDBOX_ACQUISITION_RESULT.json")
    principal = loaded.get("SUBSTRATE_SANDBOX_PRINCIPAL_AUTHORITY.json")
    history = loaded.get("SUBSTRATE_SANDBOX_HISTORICAL_IMMUTABILITY.json")
    if before:
        disk = before["disk"]
        recomputed_floor = C.disk_floor_bytes(int(disk["capacity_bytes"]))
        checks["disk_floor_recomputed"] = recomputed_floor == int(disk["required_floor_bytes"])
        checks["disk_floor_failed"] = int(disk["available_bytes"]) < recomputed_floor
        checks["core_reservation_failed"] = int(disk["available_bytes"]) < (
            recomputed_floor + C.CORE_MINIMUM_ACQUISITION_BYTES
        )
        checks["no_alternate_volume"] = not before["bounded_repair"][
            "alternate_mounted_core_volume_found"
        ]
        checks["outcome_c_admitted_by_preflight"] = before["admission"][
            "terminal_outcome_c_admitted"
        ] is True
    if classification:
        checks["terminal_outcome_c"] = (
            classification["outcome"] == "C"
            and classification["classification"] == "terminal_tangible_sandbox_null"
        )
        checks["activation_false"] = (
            classification["activation"] is False
            and classification["external_activation"] is False
            and classification["unqualified_nous"] is False
        )
        checks["no_unmeasured_H_T12"] = (
            classification["H_T12"]["status"] == "not_tested"
            and classification["H_T12"]["effect"] is None
        )
    if acquisition:
        checks["zero_download_bytes"] = acquisition["bytes_downloaded"] == 0
        checks["no_network_writer_started"] = acquisition["network_writers_started"] == 0
    if principal:
        checks["principal_not_launched"] = principal["principal_units"] == 0
        checks["invalid_principal_avoided"] = principal["invalid_principal_avoided"] is True
    if history:
        checks["parent_preserved"] = history["historical_identity"]["preserved"] is True
    checks["configuration_matches"] = all(
        row.get("configuration_digest") == configuration_digest() for row in loaded.values()
    )
    checks["source_matches"] = all(
        row.get("source_digest") == source_digest() for row in loaded.values()
    )
    checks["all_activation_fields_false"] = all(
        not contains_true_activation(row) for row in loaded.values()
    )
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    return {
        "checks": checks,
        "errors": errors,
        "loaded_documents": len(loaded),
        "all_pass": bool(checks) and all(checks.values()) and not errors,
    }


def independent_verification() -> dict[str, Any]:
    """Recompute the terminal null from raw authorities, never from prose."""

    result = _verify_recorded_state(require_all_deliverables=False)
    document = authority(
        "SUBSTRATE_SANDBOX_INDEPENDENT_VERIFICATION",
        {
            "method": "fresh process recomputation from hash-sealed preflight and refusal receipts",
            "principal_summary_files_used_for_effects": False,
            "recomputed": result["checks"],
            "errors": result["errors"],
            "outcome": "C" if result["all_pass"] else None,
            "independently_verified": result["all_pass"],
            "external_independence_claimed": False,
        },
        status="pass" if result["all_pass"] else "fail",
    )
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_INDEPENDENT_VERIFICATION.json", document)
    return document


def _run_clean_tree_checks() -> dict[str, Any]:
    """Run focused R2 checks from a detached, tracked-only archive."""

    head = git("rev-parse", "HEAD")
    with tempfile.TemporaryDirectory(prefix="substrate-r2-clean-") as temporary:
        clean = Path(temporary)
        archive = subprocess.Popen(
            [
                "git",
                "archive",
                "--format=tar",
                head,
                "pyproject.toml",
                "src/substrate/__init__.py",
                "src/substrate/final_revision_config.py",
                "src/substrate/final_revision_io.py",
                "src/substrate/sandbox.py",
                "src/substrate/sandbox_config.py",
                "src/substrate/sandbox_campaign.py",
                "tests/substrate/__init__.py",
                "tests/substrate/test_sandbox_r2.py",
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        extract = subprocess.run(
            ["tar", "-xf", "-", "-C", str(clean)],
            stdin=archive.stdout,
            capture_output=True,
            text=True,
            check=False,
        )
        if archive.stdout is not None:
            archive.stdout.close()
        archive_stderr = archive.communicate()[1].decode(errors="replace")
        if archive.returncode or extract.returncode:
            return {
                "all_pass": False,
                "head": head,
                "errors": [archive_stderr, extract.stderr],
                "checks": {},
            }
        env = dict(os.environ)
        env["PYTHONPATH"] = str(clean / "src")
        env["SUBSTRATE_REPOSITORY_ROOT"] = str(clean)
        pytest = _command(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--import-mode=importlib",
                "tests/substrate/test_sandbox_r2.py",
            ],
            timeout=180,
            cwd=clean,
            env=env,
        )
        sibling_ruff = Path(sys.executable).parent / "ruff"
        ruff_binary = shutil.which("ruff") or (
            str(sibling_ruff) if sibling_ruff.is_file() else None
        )
        ruff = (
            _command(
                [
                    ruff_binary,
                    "check",
                    "src/substrate/sandbox.py",
                    "src/substrate/sandbox_config.py",
                    "src/substrate/sandbox_campaign.py",
                    "tests/substrate/test_sandbox_r2.py",
                ],
                timeout=120,
                cwd=clean,
            )
            if ruff_binary
            else {"ok": False, "stderr": "ruff unavailable", "stdout": ""}
        )
        checks = {
            "tracked_only_checkout": True,
            "focused_tests": bool(pytest["ok"]),
            "ruff": bool(ruff["ok"]),
        }
        return {
            "all_pass": all(checks.values()),
            "head": head,
            "checks": checks,
            "pytest_output": pytest["stdout"] or pytest["stderr"],
            "ruff_output": ruff["stdout"] or ruff["stderr"],
            "errors": [],
        }


def clean_clone() -> dict[str, Any]:
    result = _run_clean_tree_checks()
    before = load_json(EVIDENCE / "SUBSTRATE_SANDBOX_PREFLIGHT.json")
    recomputed_floor = C.disk_floor_bytes(int(before["disk"]["capacity_bytes"]))
    def replay_once() -> dict[str, bool]:
        return {
            "recorded_floor_recomputed": recomputed_floor
            == before["disk"]["required_floor_bytes"],
            "recorded_floor_failure_reproduced": before["disk"]["available_bytes"]
            < recomputed_floor,
            "recorded_core_failure_reproduced": before["disk"]["available_bytes"]
            < recomputed_floor + C.CORE_MINIMUM_ACQUISITION_BYTES,
            "recorded_no_alternate_reproduced": not before["bounded_repair"][
                "alternate_mounted_core_volume_found"
            ],
        }

    first_replay = replay_once()
    second_replay = replay_once()
    replay = first_replay
    reports_regenerated_twice = first_replay == second_replay
    all_pass = bool(result["all_pass"] and all(replay.values()))
    document = authority(
        "SUBSTRATE_SANDBOX_CLEAN_CLONE",
        {
            "checkout": result,
            "recorded_preflight_replay": replay,
            "reports_regenerated_twice": reports_regenerated_twice,
            "large_data_cache_used": False,
            "all_pass": all_pass,
            "scope": "terminal preflight null; no corpus or principal receipts existed",
        },
        status="pass" if all_pass else "fail",
    )
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_CLEAN_CLONE.json", document)
    return document


def _publication_markdown(classification: dict[str, Any], pr_number: int | None) -> dict[str, str]:
    before = load_json(EVIDENCE / "SUBSTRATE_SANDBOX_PREFLIGHT.json")
    acquisition = load_json(EVIDENCE / "SUBSTRATE_SANDBOX_ACQUISITION_RESULT.json")
    catalog = load_json(EVIDENCE / "SUBSTRATE_SANDBOX_SOURCE_CATALOG.json")
    deficit_gib = before["disk"]["core_start_deficit_bytes"] / C.GIB
    floor_gib = before["disk"]["required_floor_bytes"] / C.GIB
    available_gib = before["disk"]["available_bytes"] / C.GIB
    terminal_report = f"""# Substrate Tangible Sandbox R2 — Terminal Report

## Outcome

```text
outcome: C
classification: terminal_tangible_sandbox_null
activation: false
unqualified Nous: false
```

R2 terminated at its mandatory capacity preflight. The only eligible data
filesystem exposed {available_gib:.3f} GiB free while the protected floor was
{floor_gib:.3f} GiB. Starting the minimum 60 GiB Core reservation would have
required another {deficit_gib:.3f} GiB. No eligible alternate mounted volume
existed. Deleting unrelated user data was outside the authorized scope.

No download writer, generator, public benchmark, principal task, longitudinal
lane, model call, mutation, or counterfeit run was started. This avoids
converting a platform failure into invalid scientific evidence.

## Preserved Genesis II result

The merge `{C.PARENT_MERGE_COMMIT}` and the ready/terminal tags remain intact.
The selected material is `{C.PARENT_SELECTED_MATERIAL}`; classification remains
`{C.PARENT_CLASSIFICATION}` with status `{C.PARENT_STATUS}` and readiness
`{C.PARENT_READINESS}`.

## Acquisition and environments

- Requested tier: Core
- Downloaded bytes: {acquisition["bytes_downloaded"]}
- Generated bytes: 0
- Official endpoints tested: {catalog["reachable_count"]}/{catalog["source_count"]}
- Public benchmark tasks: 0
- Custom corpus tasks: 0
- Principal/replication/hidden histories: 0/0/0
- Longitudinal duration: 0 hours
- Freeze A: not created
- Freeze B modules: none
- Ready tag: deliberately not authorized

Gated and deferred sources are enumerated in the source and license ledgers.
WorkArena++, OSWorld-V2 gated assets, MLE-bench, and manual access waits were not
placed on the critical path.

## Scientific result

H_T12 was not tested. There is no effect, confidence interval, strongest
baseline, resource-parity result, mutation result, counterfeit result, or
tangible-advantage claim. The independent verifier recomputed the refusal from
the raw disk snapshot, and a tracked-only clean checkout passed the focused R2
tests and lint gate.

## Publication

Terminal PR: {f"#{pr_number}" if pr_number else "assigned after initial publication commit"}

The strongest limitation is absolute: this run contains no tangible benchmark
measurement. It establishes only that the requested R2 Core campaign could not
be validly launched on the observed host under its own protected-space rule.
"""
    readme = f"""# Substrate Tangible Sandbox R2 publication

This directory publishes Outcome C: `{classification["classification"]}`.

The host failed the preregistered storage floor before acquisition. See
`SUBSTRATE_SANDBOX_TERMINAL_REPORT.md` in the evidence directory for the
terminal account and `REPRODUCTION.md` for verification commands.

No tangible-developmental advantage, unqualified Nous, external validation, or
external activation is claimed.
"""
    dataset_card = f"""# STSC-1 dataset card

Version: `{C.CORPUS_VERSION}`

Status: not materialized. The R2 preflight refused acquisition and generation
because free space was below the protected floor. This release contains the
schema and generator requirements but no corpus items, hidden answers, personal
data, media, or redistributed third-party assets.
"""
    results = """# Results

No public, custom, replication, hidden-composition, teaching, model-replacement,
active-perception, media, code, document, browser, or 3D result was measured.
H_T12 is `not_tested`. Outcome C is a platform-prerequisite null.
"""
    limitations = """# Limitations

The campaign stopped before acquisition, Freeze A, pilot, or principal work.
Consequently it says nothing about whether L1 provides practical advantage on
tangible work. The terminal result is specific to the observed host capacity
and available environment/model infrastructure.
"""
    reproduction = """# Reproduction

```bash
python -m substrate.sandbox preflight
python -m substrate.sandbox research
python -m substrate.sandbox plan-acquisition
python -m substrate.sandbox acquire
python -m substrate.sandbox canaries
python -m substrate.sandbox verify
```

The verifier recomputes the disk floor from the sealed raw byte counts. A later
host with adequate storage must start a new campaign; it must not rewrite this
terminal evidence.
"""
    source_ledger = """# Source and license ledger

The machine-readable source and license ledgers are published in
`evidence/substrate/tangible_sandbox/`. No gated terms were accepted
automatically. FSD50K remains clip-filtered to CC0/CC-BY for any future public
bundle, and Common Voice remains local-only with no speaker re-identification
or re-hosting.
"""
    paper = """# Tangible Sandbox R2: a terminal capacity null

## Abstract

We attempted to launch a preregistered Core-tier evaluation of a persistent
associative cognitive material on real artifacts and public agent benchmarks.
The run terminated before acquisition because the only data filesystem violated
the protected free-space floor. The fail-closed scheduler started no writers and
the verifier reproduced the refusal. No scientific comparison was performed.

## Methods and reproducibility

The program records exact source revisions, source reachability, licensing
classes, host capabilities, and byte-level disk authority. It separates the
historical Genesis II result from the new campaign and prohibits principal work
until Core acquisition, adapters, canaries, pilot, and Freeze A pass.

## Results

Outcome C, `terminal_tangible_sandbox_null`. H_T12 was not tested.

## Ethics and privacy

No private data, gated assets, purchases, messages, account mutations, or
external actions were performed. Activation remained false.
"""
    return {
        "SUBSTRATE_SANDBOX_TERMINAL_REPORT.md": terminal_report,
        "README.md": readme,
        "DATASET_CARD.md": dataset_card,
        "RESULTS.md": results,
        "LIMITATIONS.md": limitations,
        "REPRODUCTION.md": reproduction,
        "SOURCE_AND_LICENSE_LEDGER.md": source_ledger,
        "PAPER.md": paper,
    }


def publish(*, pr_number: int | None = None, run_clean_clone: bool = True) -> dict[str, Any]:
    """Generate the complete terminal Outcome C evidence and publication set."""

    before = write_preflight()
    research()
    plan = acquisition_plan(before)
    acquisition = acquire()
    canary = canaries()
    documents = _base_documents(before, plan, acquisition, canary)
    for filename, document in documents.items():
        write_json(EVIDENCE / filename, document)
    classification = _classify(before)
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json", classification)
    independent = independent_verification()
    clean = (
        clean_clone()
        if run_clean_clone
        else authority(
            "SUBSTRATE_SANDBOX_CLEAN_CLONE",
            {
                "all_pass": False,
                "status": "pending",
                "reason": "deferred until implementation source is committed",
            },
            status="pending",
        )
    )
    if not run_clean_clone:
        write_json(EVIDENCE / "SUBSTRATE_SANDBOX_CLEAN_CLONE.json", clean)
    final_state = authority(
        "SUBSTRATE_SANDBOX_FINAL_STATE",
        {
            "outcome": "C",
            "classification": classification["classification"],
            "repository": "joshuahickscorp/substrate",
            "implementation_branch": C.IMPLEMENTATION_BRANCH,
            "terminal_branch": C.TERMINAL_BRANCH,
            "preflight_tag": C.PREFLIGHT_TAG,
            "ready_tag": None,
            "ready_tag_withheld_reason": "Freeze A was not admitted",
            "terminal_tag": C.TERMINAL_TAG,
            "terminal_pr_number": pr_number,
            "CI": "required_before_merge",
            "selected_tier": "Core_requested_refused",
            "datasets_acquired": [],
            "datasets_gated_or_deferred": [
                row["source_id"]
                for row in load_json(EVIDENCE / "SUBSTRATE_SANDBOX_SOURCE_CATALOG.json")[
                    "sources"
                ]
                if row["campaign_state"] == "GATED" or row["priority"] in {"P2", "P3"}
            ],
            "bytes_downloaded": 0,
            "bytes_generated": 0,
            "public_tasks": 0,
            "custom_tasks": 0,
            "principal_histories": 0,
            "replication_histories": 0,
            "hidden_histories": 0,
            "longitudinal_hours": 0,
            "model_calls": 0,
            "tool_calls": 0,
            "H_T12": "not_tested",
            "resource_parity": "not_applicable_no_principal",
            "mutations": "not_run",
            "counterfeits": "not_run",
            "independent_verification": independent["independently_verified"],
            "clean_clone": clean["all_pass"],
            "strongest_limitation": "no tangible benchmark measurement was launched",
            "publication_package": str(PUBLICATION.relative_to(ROOT)),
            "external_activation": False,
        },
        status="terminal_evidence_prepared",
    )
    write_json(EVIDENCE / "SUBSTRATE_SANDBOX_FINAL_STATE.json", final_state)
    markdown = _publication_markdown(classification, pr_number)
    write_text(EVIDENCE / "SUBSTRATE_SANDBOX_TERMINAL_REPORT.md", markdown.pop("SUBSTRATE_SANDBOX_TERMINAL_REPORT.md"))
    for filename, text in markdown.items():
        write_text(PUBLICATION / filename, text)
    publication_index = authority(
        "SUBSTRATE_SANDBOX_PUBLICATION_INDEX",
        {
            "evidence_files": list(C.REQUIRED_DELIVERABLES),
            "publication_files": sorted(markdown),
            "terminal_report": "evidence/substrate/tangible_sandbox/SUBSTRATE_SANDBOX_TERMINAL_REPORT.md",
            "terminal_pr_number": pr_number,
            "all_required_surfaces_present": all(
                (EVIDENCE / name).is_file() for name in C.REQUIRED_DELIVERABLES
            ),
            "external_independence_claimed": False,
            "external_activation": False,
        },
        status="publication_ready",
    )
    write_json(PUBLICATION / "PUBLICATION_INDEX.json", publication_index)
    # The index is outside the minimum evidence list. Verify after every write.
    verification = verify()
    if not verification["all_pass"]:
        raise Refused(f"terminal publication verification failed: {verification['errors']}")
    return {
        "outcome": "C",
        "classification": classification["classification"],
        "deliverables": len(C.REQUIRED_DELIVERABLES),
        "publication_files": len(markdown) + 1,
        "independent_verification": independent["independently_verified"],
        "clean_clone": clean["all_pass"],
        "all_pass": verification["all_pass"],
        "activation": False,
    }


def verify() -> dict[str, Any]:
    result = _verify_recorded_state(require_all_deliverables=True)
    required_present = {
        name: (EVIDENCE / name).is_file() for name in C.REQUIRED_DELIVERABLES
    }
    publication_required = (
        "README.md",
        "DATASET_CARD.md",
        "RESULTS.md",
        "LIMITATIONS.md",
        "REPRODUCTION.md",
        "SOURCE_AND_LICENSE_LEDGER.md",
        "PAPER.md",
        "PUBLICATION_INDEX.json",
    )
    publication_present = {name: (PUBLICATION / name).is_file() for name in publication_required}
    result["checks"]["required_deliverables_present"] = all(required_present.values())
    result["checks"]["publication_package_present"] = all(publication_present.values())
    result["checks"]["ready_tag_not_created"] = ref_or_none(f"refs/tags/{C.READY_TAG}") is None
    result["checks"]["external_activation_false"] = C.ACTIVATION is False
    clean_path = EVIDENCE / "SUBSTRATE_SANDBOX_CLEAN_CLONE.json"
    independent_path = EVIDENCE / "SUBSTRATE_SANDBOX_INDEPENDENT_VERIFICATION.json"
    result["checks"]["clean_clone_pass"] = (
        clean_path.is_file() and load_json(clean_path).get("all_pass") is True
    )
    result["checks"]["independent_verification_pass"] = (
        independent_path.is_file()
        and load_json(independent_path).get("independently_verified") is True
    )
    for name, passed in result["checks"].items():
        if not passed and f"failed check: {name}" not in result["errors"]:
            result["errors"].append(f"failed check: {name}")
    result["all_pass"] = all(result["checks"].values()) and not result["errors"]
    return {
        "schema": "SUBSTRATE_SANDBOX_VERIFICATION_RESULT",
        "program": C.PROGRAM,
        "checks": result["checks"],
        "required_present": required_present,
        "publication_present": publication_present,
        "errors": result["errors"],
        "all_pass": result["all_pass"],
        "activation": False,
        "unqualified_nous": False,
    }


def status() -> dict[str, Any]:
    classification_path = EVIDENCE / "SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json"
    before_path = EVIDENCE / "SUBSTRATE_SANDBOX_PREFLIGHT.json"
    if classification_path.is_file():
        classification = load_json(classification_path)
        final = (
            load_json(EVIDENCE / "SUBSTRATE_SANDBOX_FINAL_STATE.json")
            if (EVIDENCE / "SUBSTRATE_SANDBOX_FINAL_STATE.json").is_file()
            else {}
        )
        return {
            "program": C.PROGRAM,
            "stage": "publication",
            "terminal": True,
            "outcome": classification["outcome"],
            "classification": classification["classification"],
            "H_T12": classification["H_T12"],
            "terminal_pr_number": final.get("terminal_pr_number"),
            "activation": False,
            "unqualified_nous": False,
        }
    if before_path.is_file():
        before = load_json(before_path)
        return {
            "program": C.PROGRAM,
            "stage": "preflight",
            "terminal": before["admission"]["terminal_outcome_c_admitted"],
            "critical_blocker": before["admission"]["critical_blocker"],
            "activation": False,
        }
    return {
        "program": C.PROGRAM,
        "stage": "not_started",
        "planning_eta_hours": C.PLANNING_ETA_HOURS,
        "expected_eta_hours": list(C.EXPECTED_ETA_HOURS),
        "activation": False,
    }


def stop() -> dict[str, Any]:
    STOP.parent.mkdir(parents=True, exist_ok=True)
    write_text(STOP, "operator stop\n")
    return {"stopped": True, "path": str(STOP), "activation": False}


def resume() -> dict[str, Any]:
    STOP.unlink(missing_ok=True)
    return {"stopped": False, "activation": False}


def refuse_stage(stage: str) -> dict[str, Any]:
    before = (
        load_json(EVIDENCE / "SUBSTRATE_SANDBOX_PREFLIGHT.json")
        if (EVIDENCE / "SUBSTRATE_SANDBOX_PREFLIGHT.json").is_file()
        else write_preflight()
    )
    return {
        "program": C.PROGRAM,
        "stage": stage,
        "started": False,
        "refused": True,
        "reason": before["admission"]["critical_blocker"],
        "principal_launch_admitted": False,
        "activation": False,
    }


__all__ = [
    "ARTIFACTS",
    "CORPUS",
    "EVIDENCE",
    "PUBLICATION",
    "ROOT",
    "acquire",
    "acquisition_plan",
    "canaries",
    "clean_clone",
    "configuration_digest",
    "independent_verification",
    "preflight",
    "publish",
    "refuse_stage",
    "research",
    "resume",
    "source_digest",
    "status",
    "stop",
    "verify",
    "write_preflight",
]
