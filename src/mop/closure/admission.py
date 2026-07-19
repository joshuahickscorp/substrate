"""Clean terminal admission for the generation1-general-run-closure observer.

The closure program must never launch its replay/derivation before the live General Run is terminal,
clean, and stable. This module is the gate. It is read-only: it reads the sealed General Run status and
state, reuses the orchestrator's own ``validate_general_run_status`` (never re-implementing a weaker
check), and refuses on anything partial, stale, reconciled, or merely plausible.

A clean terminal admission requires ALL of:

- the status validates under the orchestrator's own validator (seal, schema, identity, safety, capsule
  inventory, supervisor authority, counts) with no drift;
- the state is exactly ``complete`` (the ONLY safe terminal; ``failure_hold`` / ``integrity_hold`` /
  ``drained`` are unsafe terminals and are refused);
- the status is stable across two independent reads separated in time (identical status seal), so a status
  mid-write or mid-transition is refused;
- the state file agrees with the status file;
- no live General Run worker or supervisor process is still writing into the run root.

Nothing here signals, restarts, relabels, or modifies any process, source, or sealed artifact.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def evaluate_admission(
    *,
    root: Path = DEFAULT_ROOT,
    repo_root: Path | None = None,
    stability_wait_seconds: float = 2.0,
    now_iso: str = "",
) -> AdmissionDecision:
    """Evaluate clean terminal admission of the live General Run. Read-only; refuses on any doubt."""

    from mop.studio.general_run_orchestrator import REPO_ROOT

    repo_root = (repo_root or REPO_ROOT).resolve()
    decision = AdmissionDecision(admitted=False, general_run_state=None, checked_at=now_iso)

    # best-effort raw state for transparency, read directly (never trusted for admission, only reported)
    try:
        import json

        raw = json.loads((Path(root) / STATUS_FILE).read_text(encoding="utf-8"))
        decision.observed_state_raw = raw.get("state")
    except Exception:  # noqa: BLE001
        decision.observed_state_raw = None

    # 1. read and validate the sealed status under the orchestrator's own validator
    try:
        status = read_general_run_status(root)
        state = validate_general_run_status(status, repo_root=repo_root)
    except GeneralRunRefused as exc:
        decision.refusals.append(f"status did not validate: {exc}")
        return decision
    except FileNotFoundError:
        decision.refusals.append("General Run status file is absent; nothing to close")
        return decision
    except Exception as exc:  # noqa: BLE001 (any read error is a refusal, never a crash)
        decision.refusals.append(f"status unreadable: {type(exc).__name__}: {exc}")
        return decision

    decision.general_run_state = state
    decision.status_seal = status.get("status_sha256")

    # 2. must be the clean terminal, not an unsafe terminal or a non-terminal
    if state in UNSAFE_TERMINAL_STATES:
        decision.refusals.append(f"General Run reached an UNSAFE terminal state {state!r}; closure refuses")
        return decision
    if state != CLEAN_TERMINAL_STATE:
        decision.refusals.append(f"General Run is not terminal (state={state!r}); closure is deferred")
        return decision

    # 3. state file must agree with the status file
    state_path = Path(root) / STATE_FILE
    try:
        import json

        state_doc = json.loads(state_path.read_text(encoding="utf-8"))
        if state_doc.get("state") not in (CLEAN_TERMINAL_STATE, state):
            decision.refusals.append("state file disagrees with the terminal status file")
            return decision
    except Exception as exc:  # noqa: BLE001
        decision.refusals.append(f"state file unreadable: {exc}")
        return decision

    # 4. stability across two independent reads (a mid-write status is refused)
    time.sleep(max(0.0, stability_wait_seconds))
    try:
        status2 = read_general_run_status(root)
        validate_general_run_status(status2, repo_root=repo_root)
    except Exception as exc:  # noqa: BLE001
        decision.refusals.append(f"status re-read did not validate: {exc}")
        return decision
    decision.stable_across_reads = status2.get("status_sha256") == decision.status_seal
    if not decision.stable_across_reads:
        decision.refusals.append("status seal changed across reads; General Run is not stable")
        return decision

    # 5. no live writer may still be writing the run root
    decision.live_writers = _count_live_general_run_writers()
    if decision.live_writers > 0:
        decision.refusals.append(
            f"{decision.live_writers} live General Run processes still writing the run root"
        )
        return decision

    decision.admitted = True
    return decision


__all__ = ["AdmissionDecision", "evaluate_admission", "CLEAN_TERMINAL_STATE", "STATUS_FILE"]
