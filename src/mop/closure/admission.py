"""Clean terminal admission for the generation1-general-run-closure observer.

The closure program must never launch its replay/derivation before the live General Run is terminal,
clean, and stable. This module is the gate. It is read-only: it reads the sealed General Run status and
state and refuses on anything partial, stale, reconciled, or merely plausible.

Cross-checkout authority. A General Run is sealed by the orchestrator bytes it actually ran from (call it
implementation SHA A). A closure program may run from a DIFFERENT checkout whose orchestrator bytes are SHA
B. The orchestrator's own ``validate_general_run_status`` binds authority to the CURRENT checkout's
orchestrator (SHA B), so validating a SHA-A status from a SHA-B checkout would refuse on
"implementation authority drifted" before the terminal-state check is ever reached, even at ``complete``.

This gate instead binds to the HISTORICAL authority embedded in the run. It reads the status's recorded
``parent_implementation`` (path and SHA), resolves that path to immutable bytes under an
``implementation_root`` (the tree the General Run actually ran from), requires those bytes to exist and hash
exactly to the recorded SHA, requires the supervisor authority to agree, and only then runs the
orchestrator's full structural validator against the recorded authority (by pinning the orchestrator's
``IMPLEMENTATION_PATH`` to the verified historical bytes for the duration of the call). Every other check
(seal, schema, safety, capsule inventory, supervisor PID/create-time identity, counts, timestamps, terminal
state, stability, no live writer) is preserved exactly. If the historical bytes are missing or altered, the
gate refuses.

A clean terminal admission requires ALL of:

- the recorded ``parent_implementation`` path resolves to immutable bytes that hash exactly to the recorded
  SHA, and the supervisor authority agrees (historical authority is real, not assumed identical to now);
- the status validates under the orchestrator's own validator against that historical authority, with no
  drift (seal, schema, identity, safety, capsule inventory, supervisor authority, counts);
- the state is exactly ``complete`` (the ONLY safe terminal; ``failure_hold`` / ``integrity_hold`` /
  ``drained`` are unsafe terminals and are refused);
- the status is stable across two independent reads separated in time (identical status seal);
- the durable state file replays exactly to the terminal status: its self-seal is valid, its field set,
  schema, and program identity are exact, its ``status`` projects to the status ``state`` (the field is
  named ``status`` in the state file and ``state`` in the status file), every straight-through field
  (stage, supervisor, historical implementation authority, timestamps, execution mode, problems,
  reprofile, admission) agrees, the combined legacy and compute capsules equal the status capsules, and
  the independently derived counts equal the status counts;
- no live General Run worker or supervisor process is still writing into the run root.

Nothing here signals, restarts, relabels, or modifies any process, source, or sealed artifact.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mop.studio import general_run_orchestrator as gro
from mop.studio.general_run_orchestrator import (
    DEFAULT_ROOT,
    STATE_FILE,
    STATUS_FILE,
    UNSAFE_TERMINAL_STATES,
    GeneralRunRefused,
    read_general_run_status,
    validate_general_run_status,
)

CLEAN_TERMINAL_STATE = "complete"


@dataclass
class AdmissionDecision:
    """The result of the terminal admission gate. ``admitted`` is only true on a clean, stable, complete."""

    admitted: bool
    general_run_state: str | None
    refusals: list[str] = field(default_factory=list)
    status_seal: str | None = None
    stable_across_reads: bool = False
    live_writers: int = 0
    checked_at: str = ""
    observed_state_raw: str | None = None
    historical_authority_sha256: str | None = None
    historical_bytes_available: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "general_run_state": self.general_run_state,
            "observed_state_raw": self.observed_state_raw,
            "clean_terminal_state_required": CLEAN_TERMINAL_STATE,
            "refusals": self.refusals,
            "status_seal": self.status_seal,
            "stable_across_reads": self.stable_across_reads,
            "live_writers": self.live_writers,
            "checked_at": self.checked_at,
            "historical_authority_sha256": self.historical_authority_sha256,
            "historical_bytes_available": self.historical_bytes_available,
        }


def _count_live_general_run_writers(exclude_pids: set[int] | None = None) -> int:
    """Best-effort count of live General Run supervisor/worker processes still able to write the run root."""

    try:
        import psutil  # type: ignore
    except Exception:
        return -1
    markers = (
        "mop:general-run",
        "general-run:adopter",
        "mop-supervisor:generation1",
        "mop-final-mechanic",
        "mop-g1-horizon",
    )
    exclude = exclude_pids or set()
    count = 0
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] in exclude:
                continue
            hay = " ".join([proc.info.get("name") or "", " ".join(proc.info.get("cmdline") or [])])
            if any(m in hay for m in markers):
                count += 1
        except Exception:
            continue
    return count


def _resolve_historical_implementation(
    status: dict[str, Any], implementation_root: Path
) -> tuple[Path | None, str | None, list[str]]:
    """Resolve the run's recorded implementation authority to immutable, hash-matching bytes.

    Returns ``(resolved_path, recorded_sha, refusals)``. The resolved path is returned only when the
    recorded path resolves to real bytes whose SHA-256 equals the recorded SHA and the supervisor authority
    agrees; otherwise ``refusals`` is populated and the resolved path is None.
    """

    refusals: list[str] = []
    parent = status.get("parent_implementation")
    supervisor = status.get("supervisor")
    if not isinstance(parent, dict) or not isinstance(supervisor, dict):
        refusals.append("status lacks a well-formed parent_implementation or supervisor authority")
        return None, None, refusals
    recorded_path = parent.get("path")
    recorded_sha = parent.get("sha256")
    if not isinstance(recorded_path, str) or not isinstance(recorded_sha, str) or len(recorded_sha) != 64:
        refusals.append("recorded implementation authority is malformed")
        return None, recorded_sha if isinstance(recorded_sha, str) else None, refusals

    # the supervisor authority must agree with the parent authority (no reconciled or mixed shell)
    if (
        supervisor.get("implementation_path") != recorded_path
        or supervisor.get("implementation_sha256") != recorded_sha
    ):
        refusals.append("supervisor authority does not agree with the recorded parent implementation")
        return None, recorded_sha, refusals

    candidate = Path(recorded_path)
    resolved = candidate if candidate.is_absolute() else (implementation_root / candidate)
    resolved = resolved.resolve()
    if not resolved.is_file():
        refusals.append(
            f"recorded implementation bytes are not available at {resolved}; cannot replay authority"
        )
        return None, recorded_sha, refusals
    actual = gro.sha256_file(resolved)
    if actual != recorded_sha:
        refusals.append(
            "recorded implementation bytes were altered: on-disk SHA does not match the sealed authority"
        )
        return None, recorded_sha, refusals
    return resolved, recorded_sha, refusals


@contextlib.contextmanager
def _orchestrator_authority_pinned_to(implementation_path: Path) -> Iterator[None]:
    """Pin the orchestrator's ``IMPLEMENTATION_PATH`` to verified historical bytes for one validation.

    The orchestrator's validator hashes its module ``IMPLEMENTATION_PATH`` to derive the authority it
    compares against. Pinning it to the already-verified historical file lets the exact structural validator
    run against the run's own historical authority (SHA A), not the current checkout (SHA B). Restored in a
    finally so no global state leaks.
    """

    previous = gro.IMPLEMENTATION_PATH
    gro.IMPLEMENTATION_PATH = implementation_path
    try:
        yield
    finally:
        gro.IMPLEMENTATION_PATH = previous


def _validate_state_authority(state_doc: dict[str, Any], status: dict[str, Any]) -> list[str]:
    """Replay the run's durable state file and require it to project to the exact terminal status.

    The orchestrator's durable ``program_state.json`` is the source of truth; the public
    ``current_status.json`` is a pure projection of it (see the orchestrator's ``_publish`` /
    ``_status_payload`` / ``_counts``). Critically the durable state stores the terminal value under the
    field name ``status`` and the projection republishes it as ``state``, so a naive ``state_doc["state"]``
    check reads ``None`` on a real terminal state file. This reproduces the whole projection independently
    and refuses on any disagreement, so a terminal admission can never be granted on a status whose backing
    state does not exactly produce it. Returns a list of refusals; empty means the state replays exactly.
    """

    refusals: list[str] = []

    # a. the durable state file must carry its own valid self-seal (canonical_sha256 over its core)
    try:
        gro._validate_seal(state_doc, "state_sha256", "general-run state")
    except gro.GeneralRunRefused as exc:
        refusals.append(f"state file self-seal is invalid: {exc}")
        return refusals
    except Exception as exc:  # noqa: BLE001
        refusals.append(f"state file seal unverifiable: {type(exc).__name__}: {exc}")
        return refusals

    # b. the exact durable state schema: precise field set, schema tag, and program identity
    if set(state_doc) != gro.STATE_FIELDS:
        refusals.append("state file fields drifted from the durable state schema")
        return refusals
    if state_doc.get("schema") != gro.STATE_SCHEMA:
        refusals.append("state file schema tag drifted")
    if state_doc.get("program_id") != gro.PROGRAM_ID:
        refusals.append("state file program identity drifted")
    if state_doc.get("program_id") != status.get("program_id"):
        refusals.append("state file program identity disagrees with the status program identity")

    # c. the terminal value: the state file field ``status`` projects to the status file field ``state``
    if state_doc.get("status") != status.get("state"):
        refusals.append(
            f"state file status {state_doc.get('status')!r} disagrees with the terminal status state "
            f"{status.get('state')!r}"
        )

    # d. every field the status projects straight through from the state must agree exactly. This covers
    # stage, supervisor, the historical implementation authority, the timestamps, the execution mode,
    # problems, reprofile, and the recorded admission field.
    for key in (
        "created_at",
        "updated_at",
        "finished_at",
        "execution_enabled",
        "stage",
        "supervisor",
        "parent_implementation",
        "reprofile",
        "last_admission",
        "problems",
    ):
        if state_doc.get(key) != status.get(key):
            refusals.append(f"state file {key} does not project to the status {key}")

    # e. capsule projection: the combined legacy and compute capsule maps must equal the status capsules
    legacy = state_doc.get("legacy_capsules")
    compute = state_doc.get("compute_capsules")
    if not isinstance(legacy, dict) or not isinstance(compute, dict):
        refusals.append("state file capsule inventories are not well-formed")
        return refusals
    if set(legacy) & set(compute):
        refusals.append("state file legacy and compute capsule ids overlap")
    if {**legacy, **compute} != status.get("capsules"):
        refusals.append("combined state capsules do not project to the status capsules")

    # f. counts derived independently from the state capsule maps and stage pointer (mirrors _counts)
    stage = state_doc.get("stage")
    stage_index = gro.STAGES.index(stage) if stage in gro.STAGES else 0
    derived_counts = {
        "legacy_complete": sum(
            1 for row in legacy.values() if isinstance(row, dict) and row.get("status") == "complete"
        ),
        "legacy_total": len(legacy),
        "compute_complete": sum(
            1 for row in compute.values() if isinstance(row, dict) and row.get("status") == "complete"
        ),
        "compute_total": len(compute),
        "stage_index": stage_index,
        "stage_total": len(gro.STAGES),
    }
    if derived_counts != status.get("counts"):
        refusals.append("independently derived status counts disagree with the status counts")

    return refusals


def evaluate_admission(
    *,
    root: Path = DEFAULT_ROOT,
    repo_root: Path | None = None,
    implementation_root: Path | None = None,
    stability_wait_seconds: float = 2.0,
    now_iso: str = "",
) -> AdmissionDecision:
    """Evaluate clean terminal admission of the live General Run. Read-only; refuses on any doubt.

    ``implementation_root`` is the tree the General Run actually ran from, used to resolve the run's recorded
    implementation authority to real bytes. It defaults to the repo root two levels above ``root``
    (``<repo>/runs/generation1/<program>`` -> ``<repo>``), which is where the recorded relative path resolves.
    """

    root = Path(root)
    if implementation_root is None:
        implementation_root = (
            root.resolve().parents[2] if len(root.resolve().parents) >= 3 else root.resolve()
        )
    implementation_root = Path(implementation_root).resolve()
    repo_root = (repo_root or implementation_root).resolve()
    decision = AdmissionDecision(admitted=False, general_run_state=None, checked_at=now_iso)

    # best-effort raw state for transparency, read directly (never trusted for admission, only reported)
    try:
        import json

        raw = json.loads((Path(root) / STATUS_FILE).read_text(encoding="utf-8"))
        decision.observed_state_raw = raw.get("state")
    except Exception:  # noqa: BLE001
        decision.observed_state_raw = None

    # 1. read the sealed status (seal + schema validated) without the current-checkout authority check
    try:
        status = read_general_run_status(root)
    except GeneralRunRefused as exc:
        decision.refusals.append(f"status seal/schema did not validate: {exc}")
        return decision
    except FileNotFoundError:
        decision.refusals.append("General Run status file is absent; nothing to close")
        return decision
    except Exception as exc:  # noqa: BLE001 (any read error is a refusal, never a crash)
        decision.refusals.append(f"status unreadable: {type(exc).__name__}: {exc}")
        return decision
    decision.status_seal = status.get("status_sha256")

    # 2. bind to the run's HISTORICAL implementation authority (resolve to immutable, hash-matching bytes)
    historical_impl, recorded_sha, auth_refusals = _resolve_historical_implementation(
        status, implementation_root
    )
    decision.historical_authority_sha256 = recorded_sha
    decision.historical_bytes_available = historical_impl is not None
    if historical_impl is None:
        decision.refusals.extend(auth_refusals)
        return decision

    # 3. full structural validation against the recorded historical authority (not the current checkout)
    try:
        with _orchestrator_authority_pinned_to(historical_impl):
            state = validate_general_run_status(status, repo_root=implementation_root)
    except GeneralRunRefused as exc:
        decision.refusals.append(f"status did not validate against its historical authority: {exc}")
        return decision
    decision.general_run_state = state

    # 4. must be the clean terminal, not an unsafe terminal or a non-terminal
    if state in UNSAFE_TERMINAL_STATES:
        decision.refusals.append(f"General Run reached an UNSAFE terminal state {state!r}; closure refuses")
        return decision
    if state != CLEAN_TERMINAL_STATE:
        decision.refusals.append(f"General Run is not terminal (state={state!r}); closure is deferred")
        return decision

    # 5. the durable state file must replay exactly to the terminal status projection
    state_path = Path(root) / STATE_FILE
    try:
        import json

        state_doc = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        decision.refusals.append(f"state file unreadable: {exc}")
        return decision
    state_refusals = _validate_state_authority(state_doc, status)
    if state_refusals:
        decision.refusals.extend(state_refusals)
        return decision

    # 6. stability across two independent reads, re-bound to the same historical authority
    time.sleep(max(0.0, stability_wait_seconds))
    try:
        status2 = read_general_run_status(root)
        impl2, _sha2, refusals2 = _resolve_historical_implementation(status2, implementation_root)
        if impl2 is None:
            decision.refusals.extend(refusals2)
            return decision
        with _orchestrator_authority_pinned_to(impl2):
            validate_general_run_status(status2, repo_root=implementation_root)
    except Exception as exc:  # noqa: BLE001
        decision.refusals.append(f"status re-read did not validate: {exc}")
        return decision
    decision.stable_across_reads = status2.get("status_sha256") == decision.status_seal
    if not decision.stable_across_reads:
        decision.refusals.append("status seal changed across reads; General Run is not stable")
        return decision

    # 7. no live writer may still be writing the run root
    decision.live_writers = _count_live_general_run_writers()
    if decision.live_writers > 0:
        decision.refusals.append(
            f"{decision.live_writers} live General Run processes still writing the run root"
        )
        return decision

    decision.admitted = True
    return decision


__all__ = ["AdmissionDecision", "evaluate_admission", "CLEAN_TERMINAL_STATE", "STATUS_FILE"]
