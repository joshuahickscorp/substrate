
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import plistlib
import re
import secrets
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import psutil

from mop.config import REPO_ROOT

RUNS_ROOT = REPO_ROOT / "runs/generation1"
PROOF_ROOT = REPO_ROOT / "proof"
OUTPUT_ROOT = REPO_ROOT / "reports/telegram_notifier"
STATE = OUTPUT_ROOT / "state.json"
LOCK = OUTPUT_ROOT / "notifier.lock"
LOG = OUTPUT_ROOT / "notifier.log"
ERROR_LOG = OUTPUT_ROOT / "notifier.error.log"
PLIST = Path.home() / "Library/LaunchAgents/com.mop.generation1.telegram.plist"
LABEL = "com.mop.generation1.telegram"

TOKEN_SERVICE = "com.hawking.doctorv5.telegram.bot-token"
CHAT_SERVICE = "com.hawking.doctorv5.telegram.chat-id"
KEYCHAIN_ACCOUNT = "hawking"

STATE_SCHEMA = "mop-generation1-telegram-notifier-state/v1"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_MESSAGE_CHARS = 4000
POLL_SECONDS = 120
MILESTONE_PERCENT_STEP = 25
STALL_GRACE_SECONDS = 10 * 60
PROCESS_IDENTITY_TOLERANCE_SECONDS = 0.01
TERMINAL_STATES = {"complete", "failure_hold", "integrity_hold", "failed", "drained", "stopped"}
FAILURE_STATES = {"failure_hold", "integrity_hold", "failed", "stopped"}

PROGRAM_LABELS = {
    "generation1-c3-d1-router-redesign-screen-v1": "C3 Router Redesign",
    "generation1-c3-d1-expanded-canary-v1": "C3 Dispatch Canary",
    "generation1-c3-d1-frozen-producer-challenge-v1": "D1 Frozen Replication",
    "general-run": "General Run",
    "generation1-successor-mechanics-extended-v1": "General Run: Mechanics",
    "generation1-successor-evidence-chain-v2": "General Run: Adopter",
    "generation1-successor-evidence-chain-v3": "General Run: Adopter",
    "generation1-successor-evidence-chain-v4": "General Run: Adopter",
    "generation1-successor-evidence-chain-v5": "General Run: Adopter",
    "generation1-successor-evidence-chain-v6": "General Run: Adopter",
    "generation1-successor-evidence-chain-v7": "General Run: Adopter",
    "generation1-successor-horizon-v1": "General Run: Horizon 1",
    "generation1-successor-horizon-v2": "General Run: Horizon 2",
    "generation1-successor-extension-chain-v1": "General Run: Horizon Waiter",
    "generation1-successor-extension-chain-v2": "General Run: Horizon Waiter",
    "generation1-successor-extension-chain-v3": "General Run: Horizon Waiter",
    "generation1-categorized-batch-extension-chain-v1": "General Run: Categorized Waiter",
    "generation1-successor-categorized-batch-wave-v1": "General Run: Categorized Wave",
    "generation1-consolidated-final-campaign-v1": "General Run: Final",
}

CATEGORY_MECHANISMS = {
    "formation_trace": ("C0", "E1"),
    "communication_repair": ("V1", "M1", "K1"),
    "memory_plasticity": ("R1", "P1R"),
    "action_simulation": ("A1", "S1"),
    "construction": ("G1",),
    "dispatch_redesign": ("D1",),
    "uncertainty_curiosity": ("U1", "N1"),
}

_WAVE_CAPSULE_RE = re.compile(r"^w(\d+)_(.+)$")

NEXT_LABELS = {
    "freeze_best_variant_for_untouched_confirmation_design": "D1 confirmation",
    "iterate_router_representation_before_confirmation": "D1 redesign",
    "design_confirmatory_preregistration": "D1 confirmation",
    "redesign_visible_router_before_confirmation": "D1 redesign",
    "run_v1_m1_g1_sibling_batch": "V1/M1/G1 batch",
    "implement_lane_specific_scientific_producers_and_verifiers": "Final verifier build",
    "interpret_consolidated_result_and_author_independent_verifiers": "Interpret final result",
}


class MOPNotifierError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    source = path.resolve(strict=True)
    if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_JSON_BYTES:
        raise MOPNotifierError(f"unsafe or oversized JSON input: {source}")
    try:
        value = json.loads(source.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MOPNotifierError(f"invalid JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise MOPNotifierError(f"JSON root is not an object: {source}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sealed(value: Mapping[str, Any], field: str) -> bool:
    core = {key: item for key, item in value.items() if key != field}
    return value.get(field) == canonical_sha256(core)


def _valid_status(value: Mapping[str, Any]) -> bool:
    schema = value.get("schema")
    program_id = value.get("program_id")
    return bool(
        isinstance(schema, str)
        and schema.startswith("mop-generation1-")
        and "status/" in schema
        and isinstance(program_id, str)
        and program_id.startswith("generation1-")
        and isinstance(value.get("state"), str)
        and isinstance(value.get("capsules"), dict)
        and _sealed(value, "status_sha256")
    )


def _keychain_get(service: str) -> str | None:
    result = subprocess.run(
        [
            "/usr/bin/security",
            "find-generic-password",
            "-a",
            KEYCHAIN_ACCOUNT,
            "-s",
            service,
            "-w",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _telegram(token: str, method: str, payload: Mapping[str, Any]) -> Any:
    if not re.fullmatch(r"[0-9]{6,16}:[A-Za-z0-9_-]{20,}", token):
        raise MOPNotifierError("Telegram bot token shape is invalid")
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "mop-generation1-notifier/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read(2 * 1024 * 1024)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MOPNotifierError(f"Telegram {method} request failed: {type(exc).__name__}") from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MOPNotifierError(f"Telegram {method} returned invalid JSON") from exc
    if not isinstance(value, dict) or value.get("ok") is not True:
        description = value.get("description") if isinstance(value, dict) else None
        raise MOPNotifierError(f"Telegram {method} refused: {description or 'unknown error'}")
    return value.get("result")


def send_message(text: str) -> dict[str, Any]:
    token = _keychain_get(TOKEN_SERVICE)
    chat_id = _keychain_get(CHAT_SERVICE)
    if not token or not chat_id:
        raise MOPNotifierError("existing Hawking Telegram credentials are unavailable")
    result = _telegram(
        token,
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text[:MAX_MESSAGE_CHARS],
            "disable_web_page_preview": "true",
        },
    )
    if not isinstance(result, dict) or not isinstance(result.get("message_id"), int):
        raise MOPNotifierError("Telegram response lacks a message ID")
    return {"message_id": result["message_id"], "sent_at": _now()}


def _new_state() -> dict[str, Any]:
    core = {
        "schema": STATE_SCHEMA,
        "created_at": _now(),
        "updated_at": _now(),
        "primed": False,
        "delivered": {},
    }
    return {**core, "state_sha256": canonical_sha256(core)}


def load_state(path: Path = STATE) -> dict[str, Any]:
    if not path.exists():
        return _new_state()
    value = read_json(path)
    if value.get("schema") != STATE_SCHEMA or not _sealed(value, "state_sha256"):
        raise MOPNotifierError("MOP notifier state identity is invalid")
    return value


def save_state(value: Mapping[str, Any], path: Path = STATE) -> None:
    core = {key: item for key, item in value.items() if key != "state_sha256"}
    core["updated_at"] = _now()
    atomic_write_json(path, {**core, "state_sha256": canonical_sha256(core)})


def _artifact_hashes(statuses: list[dict[str, Any]]) -> set[str]:
    hashes: set[str] = set()
    for status in statuses:
        for capsule in status.get("capsules", {}).values():
            for artifact in capsule.get("artifacts", []):
                digest = artifact.get("sha256")
                if isinstance(digest, str):
                    hashes.add(digest)
    return hashes


def _capsule_complete(row: Mapping[str, Any]) -> bool:
    artifacts = row.get("artifacts")
    artifacts_ok = (
        isinstance(artifacts, list)
        and bool(artifacts)
        and all(isinstance(artifact, dict) and artifact.get("all_ok") is True for artifact in artifacts)
    )
    return row.get("returncode") == 0 and artifacts_ok and isinstance(row.get("finished_at"), str)


def _progress_percent(complete: int, total: int) -> int | None:

    if total <= 0:
        return None
    return round(100 * min(1.0, max(0.0, complete / total)))


def _milestone_bucket(complete: int, total: int, *, percent: int = MILESTONE_PERCENT_STEP) -> int:

    if total <= 0:
        return 0
    fraction = min(1.0, max(0.0, complete / total))
    return min(100 // percent, int(fraction * 100 // percent))


def _next_milestone_complete(complete: int, total: int, *, percent: int = MILESTONE_PERCENT_STEP) -> int:

    if total <= 0:
        return complete
    bucket = _milestone_bucket(complete, total, percent=percent)
    next_bucket = min(100 // percent, bucket + 1)
    if next_bucket <= bucket:
        return total
    threshold_fraction = next_bucket * percent / 100.0
    return min(total, max(complete, math.ceil(threshold_fraction * total)))


def _event_eta(status: Mapping[str, Any], complete: int, total: int) -> dict[str, float | None]:
    adaptive = status.get("adaptive_execution")
    if not isinstance(adaptive, dict):
        return {"block_seconds": None, "session_seconds": None}
    average = _finite(adaptive.get("average_rung_seconds"))
    next_rung = _finite(adaptive.get("next_rung_seconds"))
    session = _finite(adaptive.get("eta_seconds"))
    workers_value = adaptive.get("workers")
    workers = workers_value if isinstance(workers_value, int) and not isinstance(workers_value, bool) else 1
    remaining_to_milestone = max(0, _next_milestone_complete(complete, total) - complete)
    per_rung = next_rung if next_rung is not None else average
    block = None if per_rung is None else per_rung * remaining_to_milestone / max(1, workers)
    return {"block_seconds": block, "session_seconds": session}


def _status_age_seconds(status: Mapping[str, Any]) -> float | None:
    value = status.get("updated_at") or status.get("created_at")
    if not isinstance(value, str):
        return None
    try:
        observed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if observed.tzinfo is None:
        return None
    return (dt.datetime.now(dt.UTC) - observed.astimezone(dt.UTC)).total_seconds()


def _supervisor_identity_state(identity: Any) -> str:

    if not isinstance(identity, Mapping):
        return "unknown"
    pid = identity.get("pid")
    create_time = identity.get("create_time")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(create_time, bool)
        or not isinstance(create_time, int | float)
        or not math.isfinite(float(create_time))
        or float(create_time) <= 0
    ):
        return "unknown"
    try:
        process = psutil.Process(pid)
        if not math.isclose(
            process.create_time(),
            float(create_time),
            rel_tol=0.0,
            abs_tol=PROCESS_IDENTITY_TOLERANCE_SECONDS,
        ):
            return "gone"
        if process.status() == psutil.STATUS_ZOMBIE:
            return "gone"
        return "alive" if process.is_running() else "gone"
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return "gone"
    except (psutil.AccessDenied, OSError, ValueError):
        return "unknown"


def _stall_event(
    status: Mapping[str, Any],
    *,
    program_id: str,
    complete: int,
    total: int,
) -> dict[str, Any] | None:

    state = str(status.get("state", "unknown"))
    identity = status.get("supervisor")
    age = _status_age_seconds(status)
    if (
        state in TERMINAL_STATES
        or age is None
        or age < STALL_GRACE_SECONDS
        or _supervisor_identity_state(identity) != "gone"
    ):
        return None
    root = canonical_sha256(
        {
            "state": state,
            "status_created_at": status.get("created_at"),
            "supervisor": identity,
        }
    )
    problems = list(status.get("problems", [])) if isinstance(status.get("problems"), list) else []
    problems.append("exact supervisor process is gone and status has been stale for at least 10 minutes")
    return {
        "event_id": f"program/{program_id}/supervisor-stall/{root}",
        "kind": "failure",
        "program_id": program_id,
        "state": "supervisor_stall",
        "progress": {"complete": complete, "total": total},
        "eta": _event_eta(status, complete, total),
        "problems": problems,
        "status_state": state,
        "status_created_at": status.get("created_at"),
    }


def _adaptive_summary(status: Mapping[str, Any]) -> dict[str, Any] | None:

    adaptive = status.get("adaptive_execution")
    if not isinstance(adaptive, Mapping):
        return None
    return {"workers": adaptive.get("workers"), "mode": adaptive.get("mode")}


def _capsule_finish_times(capsules: Mapping[str, Any]) -> list[dt.datetime]:

    times: list[dt.datetime] = []
    for row in capsules.values():
        value = row.get("finished_at") if isinstance(row, Mapping) else None
        if not isinstance(value, str):
            continue
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is not None:
            times.append(parsed.astimezone(dt.UTC))
    times.sort()
    return times


def _synthetic_adaptive(capsules: Mapping[str, Any], *, total: int, complete: int) -> dict[str, Any] | None:

    times = _capsule_finish_times(capsules)
    average_rung_seconds = None
    if len(times) >= 2:
        elapsed = (times[-1] - times[0]).total_seconds()
        average_rung_seconds = elapsed / max(1, len(times) - 1)

    workers: int | None = None
    mode: str | None = None
    try:
        import dataclasses

        from mop.studio import dynamic_worker_controller as controller

        sample = controller.sample_host_state()
        mode = "hawking_active" if sample.state.hawking_active else "hawking_idle"
        settled_state = dataclasses.replace(
            sample.state, current_workers=sample.bounds["comfortable_target"]
        )
        workers = controller.recommended_workers(settled_state)
    except Exception:  # pragma: no cover - best-effort telemetry, never fatal
        pass

    if average_rung_seconds is None and workers is None:
        return None
    eta_seconds = None
    if average_rung_seconds is not None:
        eta_seconds = average_rung_seconds * max(0, total - complete) / max(1, workers or 1)
    return {
        "average_rung_seconds": average_rung_seconds,
        "eta_seconds": eta_seconds,
        "workers": workers,
        "mode": mode,
    }


def _with_effective_adaptive(
    status: Mapping[str, Any], capsules: Mapping[str, Any], *, total: int, complete: int
) -> Mapping[str, Any]:

    if isinstance(status.get("adaptive_execution"), Mapping):
        return status
    synthetic = _synthetic_adaptive(capsules, total=total, complete=complete)
    if synthetic is None:
        return status
    return {**status, "adaptive_execution": synthetic}


def _classify_mode(mode: Any) -> str | None:

    if not isinstance(mode, str):
        return None
    lowered = mode.lower()
    if "hawking" in lowered:
        return "hawking"
    if "idle" in lowered or "opportunistic" in lowered:
        return "idle"
    return None


def _collect_throttle_events(
    last_seen_mode: Mapping[str, Any],
    *,
    runs_root: Path = RUNS_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, str]]:

    events: list[dict[str, Any]] = []
    updated: dict[str, str] = {
        str(key): str(value)
        for key, value in last_seen_mode.items()
        if isinstance(value, str)
    }
    if not runs_root.exists():
        return events, updated
    for path in sorted(runs_root.glob("*/current_status.json")):
        try:
            status = read_json(path)
        except (MOPNotifierError, OSError):
            continue
        if not _valid_status(status):
            continue
        adaptive = status.get("adaptive_execution")
        if not isinstance(adaptive, Mapping):
            continue
        current = _classify_mode(adaptive.get("mode"))
        if current is None:
            continue
        program_id = str(status.get("program_id") or path.parent.name)
        previous = updated.get(program_id)
        updated[program_id] = current
        if previous is None or previous == current:
            continue
        workers = adaptive.get("workers")
        events.append(
            {
                "event_id": f"throttle-shift/{program_id}/{current}",
                "kind": "throttle",
                "program_id": program_id,
                "mode": current,
                "previous_mode": previous,
                "workers": workers if isinstance(workers, int) and not isinstance(workers, bool) else None,
            }
        )
    return events, updated


def _lane_short_name(lane_id: str) -> str:

    return lane_id[3:] if lane_id.startswith("G1-") else lane_id


def _collect_subtask_events(
    last_seen_lanes: Mapping[str, Any],
    last_seen_capsules: Mapping[str, Any],
    *,
    runs_root: Path = RUNS_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:

    events: list[dict[str, Any]] = []
    lanes_updated: dict[str, str] = {
        str(key): str(value) for key, value in last_seen_lanes.items() if isinstance(value, str)
    }
    capsules_updated: dict[str, str] = {
        str(key): str(value) for key, value in last_seen_capsules.items() if isinstance(value, str)
    }
    if not runs_root.exists():
        return events, lanes_updated, capsules_updated
    for path in sorted(runs_root.glob("*/current_status.json")):
        try:
            status = read_json(path)
        except (MOPNotifierError, OSError):
            continue
        if not _valid_status(status):
            continue
        program_id = str(status.get("program_id") or path.parent.name)
        lane_progress = status.get("lane_progress")
        if isinstance(lane_progress, Mapping):
            for lane_id, row in lane_progress.items():
                if not isinstance(lane_id, str) or not isinstance(row, Mapping):
                    continue
                complete = row.get("complete")
                total = row.get("total")
                if (
                    not isinstance(complete, int)
                    or isinstance(complete, bool)
                    or not isinstance(total, int)
                    or isinstance(total, bool)
                    or total <= 0
                ):
                    continue
                current = "complete" if complete >= total else "incomplete"
                key = f"{program_id}/{lane_id}"
                previous = lanes_updated.get(key)
                lanes_updated[key] = current
                if previous is None or previous == current or current != "complete":
                    continue
                events.append(
                    {
                        "event_id": f"subtask/lane/{program_id}/{lane_id}",
                        "kind": "subtask",
                        "program_id": program_id,
                        "lane_id": lane_id,
                        "lane_short": _lane_short_name(lane_id),
                        "progress": {"complete": complete, "total": total},
                    }
                )
        if _program_label(program_id).startswith("General Run"):
            raw_capsules = status.get("capsules")
            capsules: dict[str, Any] = raw_capsules if isinstance(raw_capsules, dict) else {}
            for capsule_id, row in capsules.items():
                if not isinstance(capsule_id, str) or not isinstance(row, Mapping):
                    continue
                match = _WAVE_CAPSULE_RE.match(capsule_id)
                if match is None:
                    continue
                current = "complete" if _capsule_complete(row) else "incomplete"
                key = f"{program_id}/{capsule_id}"
                previous = capsules_updated.get(key)
                capsules_updated[key] = current
                if previous is None or previous == current or current != "complete":
                    continue
                events.append(
                    {
                        "event_id": f"subtask/capsule/{program_id}/{capsule_id}",
                        "kind": "subtask",
                        "program_id": program_id,
                        "capsule_id": capsule_id,
                        "wave_epoch": match.group(1),
                        "category": match.group(2),
                    }
                )
    return events, lanes_updated, capsules_updated


def collect_events(
    *,
    runs_root: Path = RUNS_ROOT,
    proof_root: Path = PROOF_ROOT,
) -> list[dict[str, Any]]:

    events: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("*/current_status.json")) if runs_root.exists() else []:
        try:
            status = read_json(path)
        except (MOPNotifierError, OSError):
            continue
        if not _valid_status(status):
            continue
        statuses.append(status)
        program_id = str(status.get("program_id") or path.parent.name)
        raw_capsules = status.get("capsules")
        capsules: dict[str, Any] = raw_capsules if isinstance(raw_capsules, dict) else {}
        completed = [name for name, row in capsules.items() if _capsule_complete(row)]
        raw_counts = status.get("counts")
        counts: dict[str, Any] = raw_counts if isinstance(raw_counts, dict) else {}
        declared_complete = counts.get("complete")
        declared_total = counts.get("total")
        complete = (
            declared_complete
            if isinstance(declared_complete, int)
            and not isinstance(declared_complete, bool)
            and declared_complete == len(completed)
            else len(completed)
        )
        total = (
            declared_total
            if isinstance(declared_total, int)
            and not isinstance(declared_total, bool)
            and declared_total == len(capsules)
            else len(capsules)
        )
        effective_status = _with_effective_adaptive(status, capsules, total=total, complete=complete)
        adaptive_summary = _adaptive_summary(effective_status)
        for ordinal, capsule_id in enumerate(completed, start=1):
            row = capsules[capsule_id]
            artifact_roots = [
                artifact.get("sha256") for artifact in row.get("artifacts", []) if isinstance(artifact, dict)
            ]
            root = canonical_sha256(artifact_roots)
            rung_event = {
                "event_id": f"capsule/{program_id}/{capsule_id}/{root}",
                "kind": "rung",
                "program_id": program_id,
                "capsule_id": capsule_id,
                "state": "complete",
                "attempts": row.get("attempts"),
                "finished_at": row.get("finished_at"),
                "progress": {"complete": ordinal, "total": total},
                "eta": _event_eta(effective_status, ordinal, total),
                "artifacts": row.get("artifacts", []),
                "problems": status.get("problems", []),
            }
            if adaptive_summary is not None:
                rung_event["adaptive"] = adaptive_summary
            events.append(rung_event)
        state = str(status.get("state", "unknown"))
        if state in TERMINAL_STATES:
            terminal_root = canonical_sha256(
                {
                    "state": state,
                    "completed": completed,
                    "problems": status.get("problems", []),
                }
            )
            terminal_event = {
                "event_id": f"program/{program_id}/{state}/{terminal_root}",
                "kind": "failure" if state in FAILURE_STATES else "terminal",
                "program_id": program_id,
                "state": state,
                "progress": {"complete": complete, "total": total},
                "eta": _event_eta(effective_status, complete, total),
                "problems": status.get("problems", []),
            }
            if adaptive_summary is not None:
                terminal_event["adaptive"] = adaptive_summary
            events.append(terminal_event)
        else:
            stalled = _stall_event(
                effective_status,
                program_id=program_id,
                complete=complete,
                total=total,
            )
            if stalled is not None:
                events.append(stalled)

    bound_hashes = _artifact_hashes(statuses)
    if proof_root.exists():
        for path in sorted(proof_root.glob("GENERATION1*.json")):
            if ".shard_" in path.name or path.stat().st_size > MAX_JSON_BYTES:
                continue
            digest = sha256_file(path)
            if digest in bound_hashes:
                continue
            try:
                proof = read_json(path)
            except (MOPNotifierError, OSError):
                continue
            if proof.get("complete") is False or proof.get("execution_complete") is False:
                continue
            events.append(
                {
                    "event_id": f"proof/{path.name}/{digest}",
                    "kind": "proof",
                    "path": str(path.relative_to(REPO_ROOT))
                    if path.is_relative_to(REPO_ROOT)
                    else str(Path("proof") / path.relative_to(proof_root)),
                    "file_sha256": digest,
                    "schema": proof.get("schema"),
                    "campaign_id": proof.get("campaign_id") or proof.get("pilot_id"),
                    "summary": proof_summary(proof),
                }
            )
    return events


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def proof_summary(proof: Mapping[str, Any]) -> list[str]:

    lines: list[str] = []
    grid = proof.get("grid")
    if isinstance(grid, dict):
        completed = grid.get("completed_cell_count") or grid.get("completed_seed_count")
        expected = grid.get("expected_cell_count") or grid.get("expected_seed_count")
        if completed is not None and expected is not None:
            lines.append(f"coverage {completed}/{expected}")
    overall = proof.get("overall")
    if isinstance(overall, dict):
        names = (
            "learned_dispatch",
            "routed",
            "difficulty_static",
            "global_static",
            "random_actor",
            "oracle_actor",
        )
        metrics = []
        for name in names:
            row = overall.get(name)
            value = row.get("mean") if isinstance(row, dict) else row
            if _finite(value) is not None:
                metrics.append(f"{name} {_fmt(value)}")
        if metrics:
            lines.append(" | ".join(metrics))
    differences = proof.get("learned_dispatch_differences")
    if isinstance(differences, dict):
        metrics = [
            f"{name} {_fmt(value)}" for name, value in differences.items() if _finite(value) is not None
        ]
        if metrics:
            lines.append("deltas: " + " | ".join(metrics[:5]))
    decision = proof.get("decision")
    if isinstance(decision, dict):
        selected = []
        for key in (
            "verdict",
            "context_labeled_frozen_routing_confirmed",
            "ready_to_preregister_g1_c3_learned_dispatch",
            "scientific_confirmation",
            "all_pilot_lanes_executed",
            "c3_confirmed",
        ):
            if key in decision:
                selected.append(f"{key}={decision[key]}")
        if selected:
            lines.append("decision: " + "; ".join(selected))
        next_action = decision.get("next_action")
        if isinstance(next_action, str):
            lines.append(f"Next: {NEXT_LABELS.get(next_action, 'review')}")
    problems = proof.get("problems")
    if isinstance(problems, list) and problems:
        lines.append(f"problems: {len(problems)} ({str(problems[0])[:180]})")
    return lines[:4]


def host_health(root: Path = REPO_ROOT) -> dict[str, Any]:
    def command(argv: list[str]) -> str:
        try:
            result = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=8)
        except (OSError, subprocess.TimeoutExpired):
            return "unknown"
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    pressure = command(["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    swap = command(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    match = re.search(r"used\s*=\s*([0-9.]+)([MG])", swap)
    swap_mb = float(match.group(1)) * (1024 if match and match.group(2) == "G" else 1) if match else None
    stat = os.statvfs(root)
    return {
        "pressure": int(pressure) if pressure.isdigit() else None,
        "swap_mb": swap_mb,
        "disk_free_gb": stat.f_bavail * stat.f_frsize / 1_000_000_000,
    }


def _program_label(program_id: Any) -> str:
    identifier = str(program_id)
    if identifier in PROGRAM_LABELS:
        return PROGRAM_LABELS[identifier]
    simplified = re.sub(r"^generation1-", "", identifier)
    simplified = re.sub(r"-v\d+$", "", simplified)
    return simplified.replace("-", " ").title() or "MOP"


def _health_label(health: Mapping[str, Any]) -> str:
    pressure = health.get("pressure")
    disk = _finite(health.get("disk_free_gb"))
    if pressure == 1 and disk is not None and disk >= 20.0:
        return "good"
    if pressure is None or disk is None:
        return "unknown"
    return "check"


def _errors_line(event: Mapping[str, Any]) -> str:
    problems = event.get("problems")
    if isinstance(problems, list) and problems:
        return f"Errors: {len(problems)} | {str(problems[0])[:240]}"
    return "Errors: none"


def _format_duration(value: Any) -> str:
    seconds = _finite(value)
    if seconds is None:
        return "estimating"
    if seconds <= 0:
        return "done"
    if seconds < 90:
        return f"{max(1, math.ceil(seconds))}s"
    minutes = max(1, math.ceil(seconds / 60.0))
    hours, remainder = divmod(minutes, 60)
    return f"{hours}h {remainder}m" if hours else f"{remainder}m"


def _worker_line(event: Mapping[str, Any]) -> str | None:

    adaptive = event.get("adaptive")
    if not isinstance(adaptive, Mapping):
        return None
    workers = adaptive.get("workers")
    has_workers = isinstance(workers, int) and not isinstance(workers, bool)
    mode = adaptive.get("mode")
    has_mode = isinstance(mode, str) and bool(mode.strip())
    if not has_workers and not has_mode:
        return None
    workers_text = str(workers) if has_workers else "?"
    if has_mode:
        lowered = mode.lower()
        if "idle" in lowered or "opportunistic" in lowered:
            label = "idle burst"
        elif "hawking" in lowered:
            label = "Hawking active"
        else:
            label = mode
        return f"Workers: {workers_text} ({label})"
    return f"Workers: {workers_text}"


def _throttle_message(event: Mapping[str, Any]) -> str:

    label = _program_label(event.get("program_id"))
    workers = event.get("workers")
    workers_text = str(workers) if isinstance(workers, int) and not isinstance(workers, bool) else "?"
    mode = event.get("mode")
    if mode == "idle":
        return f"⚙️ MOP {label}: Hawking cleared, ramping to {workers_text} workers"
    if mode == "hawking":
        return f"⚙️ MOP {label}: Hawking active, ceding to {workers_text} workers"
    return f"⚙️ MOP {label}: throttle mode {mode}"


def _subtask_message(event: Mapping[str, Any]) -> str:

    lane_short = event.get("lane_short")
    if isinstance(lane_short, str):
        progress = event.get("progress") or {}
        complete = progress.get("complete")
        total = progress.get("total")
        counts = (
            f" ({complete}/{total})"
            if isinstance(complete, int)
            and not isinstance(complete, bool)
            and isinstance(total, int)
            and not isinstance(total, bool)
            else ""
        )
        return f"General Run: {lane_short} sub-task complete{counts}"
    epoch = event.get("wave_epoch")
    category = event.get("category")
    mechanisms = CATEGORY_MECHANISMS.get(category) if isinstance(category, str) else None
    mech_suffix = f" ({', '.join(mechanisms)})" if mechanisms else ""
    return f"General Run: W{epoch} {category} sub-task complete{mech_suffix}"


def _progress_line(progress: Mapping[str, Any]) -> str:
    complete, total = progress.get("complete"), progress.get("total")
    percent = (
        _progress_percent(complete, total)
        if isinstance(complete, int)
        and not isinstance(complete, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        else None
    )
    suffix = f" ({percent}%)" if percent is not None else ""
    return f"Progress: {complete}/{total}{suffix}"


def format_event(event: Mapping[str, Any]) -> str:
    kind = event["kind"]
    if kind == "throttle":
        return _throttle_message(event)[:MAX_MESSAGE_CHARS]
    if kind == "subtask":
        return _subtask_message(event)[:MAX_MESSAGE_CHARS]
    if kind == "rung":
        progress = event["progress"]
        lines = [
            f"🧪 MOP {_program_label(event['program_id'])}",
            _progress_line(progress),
        ]
        eta = event.get("eta") or {}
        lines += [
            f"Next {MILESTONE_PERCENT_STEP}%: {_format_duration(eta.get('block_seconds'))}",
            f"Full queue: {_format_duration(eta.get('session_seconds'))}",
        ]
        worker_line = _worker_line(event)
        if worker_line is not None:
            lines.append(worker_line)
    elif kind in {"terminal", "failure"}:
        symbol = "⚠️" if kind == "failure" else "✅"
        progress = event["progress"]
        lines = [
            f"{symbol} MOP {_program_label(event['program_id'])}: {event['state']}",
            _progress_line(progress),
        ]
        if kind == "terminal" and str(event.get("program_id")) in PROGRAM_LABELS:
            lines.append("Chain stage complete")
        worker_line = _worker_line(event)
        if worker_line is not None:
            lines.append(worker_line)
    else:
        lines = [
            "📊 MOP result ready",
            *event.get("summary", []),
        ]
    health = host_health()
    lines += [
        f"Health: {_health_label(health)}",
        _errors_line(event),
    ]
    return "\n".join(lines)[:MAX_MESSAGE_CHARS]


def _should_send(event: Mapping[str, Any]) -> bool:
    if event.get("kind") != "rung":
        return True
    progress = event.get("progress") or {}
    complete = progress.get("complete")
    total = progress.get("total")
    if (
        not isinstance(complete, int)
        or isinstance(complete, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
    ):
        return False
    complete = min(max(complete, 0), total)
    return _milestone_bucket(complete, total) > _milestone_bucket(complete - 1, total)


def prime(*, state_path: Path = STATE) -> dict[str, Any]:
    state = load_state(state_path)
    events = collect_events()
    delivered = dict(state.get("delivered", {}))
    for event in events:
        delivered.setdefault(
            event["event_id"],
            {"status": "primed-existing-no-message", "recorded_at": _now()},
        )
    state["delivered"] = delivered
    state["primed"] = True
    save_state(state, state_path)
    return {"primed": True, "existing_events_suppressed": len(delivered)}


def run_once(
    *,
    sender: Callable[[str], dict[str, Any]] = send_message,
    state_path: Path = STATE,
    runs_root: Path = RUNS_ROOT,
) -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "already-running", "sent": 0}
        state = load_state(state_path)
        if state.get("primed") is not True:
            raise MOPNotifierError("MOP notifier must be primed before delivery")
        delivered = dict(state.get("delivered", {}))
        throttle_events: list[dict[str, Any]] = []
        try:
            previous_modes = state.get("last_seen_mode")
            previous_modes = previous_modes if isinstance(previous_modes, Mapping) else {}
            throttle_events, current_modes = _collect_throttle_events(previous_modes, runs_root=runs_root)
            state["last_seen_mode"] = current_modes
        except (MOPNotifierError, OSError, ValueError):
            throttle_events = []
        subtask_events: list[dict[str, Any]] = []
        try:
            previous_lanes = state.get("last_seen_complete_lanes")
            previous_lanes = previous_lanes if isinstance(previous_lanes, Mapping) else {}
            previous_capsules = state.get("last_seen_complete_capsules")
            previous_capsules = previous_capsules if isinstance(previous_capsules, Mapping) else {}
            subtask_events, lanes_now, capsules_now = _collect_subtask_events(
                previous_lanes, previous_capsules, runs_root=runs_root
            )
            state["last_seen_complete_lanes"] = lanes_now
            state["last_seen_complete_capsules"] = capsules_now
        except (MOPNotifierError, OSError, ValueError):
            subtask_events = []
        save_state(state, state_path)
        sent = 0
        for event in [*throttle_events, *subtask_events, *collect_events()]:
            if event["event_id"] in delivered:
                continue
            if not _should_send(event):
                delivered[event["event_id"]] = {
                    "status": "suppressed-nonmilestone",
                    "recorded_at": _now(),
                }
                state["delivered"] = delivered
                save_state(state, state_path)
                continue
            receipt = sender(format_event(event))
            delivered[event["event_id"]] = {"status": "delivered", **receipt}
            state["delivered"] = delivered
            save_state(state, state_path)
            sent += 1
        return {"status": "ok", "sent": sent, "known_events": len(delivered)}


def install_launch_agent() -> dict[str, Any]:
    if not _keychain_get(TOKEN_SERVICE) or not _keychain_get(CHAT_SERVICE):
        raise MOPNotifierError("existing Hawking Telegram credentials are unavailable")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    document = {
        "Label": LABEL,
        "ProgramArguments": [
            sys.executable,
            str((REPO_ROOT / "scripts/mop_telegram_notifier.py").resolve()),
            "run",
        ],
        "WorkingDirectory": str(REPO_ROOT),
        "StartInterval": POLL_SECONDS,
        "RunAtLoad": True,
        "ProcessType": "Background",
        "StandardOutPath": str(LOG),
        "StandardErrorPath": str(ERROR_LOG),
        "EnvironmentVariables": {"PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])},
    }
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    temporary = PLIST.with_name(f".{PLIST.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            plistlib.dump(document, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, PLIST)
    finally:
        temporary.unlink(missing_ok=True)
    domain = f"gui/{os.getuid()}"
    subprocess.run(["/bin/launchctl", "bootout", domain, str(PLIST)], capture_output=True, check=False)
    loaded = subprocess.run(
        ["/bin/launchctl", "bootstrap", domain, str(PLIST)],
        text=True,
        capture_output=True,
        check=False,
    )
    if loaded.returncode != 0:
        raise MOPNotifierError(f"launchd refused the MOP notifier: {loaded.stderr.strip()[:200]}")
    return {
        "installed": True,
        "label": LABEL,
        "interval_seconds": POLL_SECONDS,
        "plist": str(PLIST),
        "credentials": "shared Hawking macOS Keychain entries",
    }


def status() -> dict[str, Any]:
    state = load_state()
    visible = {event["event_id"] for event in collect_events()}
    return {
        "token_available": _keychain_get(TOKEN_SERVICE) is not None,
        "chat_available": _keychain_get(CHAT_SERVICE) is not None,
        "primed": state.get("primed") is True,
        "known_events": len(state.get("delivered", {})),
        "currently_visible_events": len(visible),
        "launch_agent_installed": PLIST.exists(),
        "interval_seconds": POLL_SECONDS,
        "state": str(STATE),
    }
