"""Cross-check continuity segment receipts against the owned checkpoint.

Post-freeze supplementary verifier. Writes no evidence. Not a required
deliverable and not a scored endpoint.

## What an earlier version of this script got wrong

It claimed a hand-authored receipt "has to satisfy all five checks
simultaneously against a file mtime the author does not control after the fact".
That was false, and an adversarial review demonstrated it by forging a passing
twelve-hour receipt set in under a millisecond. The wall-clock check computed
`mtime - receipt["process_started_unix"]` and compared it to
`receipt["duration_seconds"]` -- all three quantities chosen by whoever writes
the file, with `os.utime` supplying the third. The chain checks compared
receipt-declared strings to each other. Nothing opened the checkpoint at all,
and a forged run root containing no checkpoint whatsoever passed.

Receipt-declared fields cannot authenticate themselves. This version therefore
treats them as unauthenticated claims and anchors the check on the one artifact
in the run root that is not free to author: the checkpoint.

## What is actually verified

`EventSourcedKernel.restore` refuses a checkpoint whose `checkpoint_digest` does
not match the document, whose activation is not false, or that contains a single
event whose digest does not match its content. So restoring the checkpoint
validates a content-addressed event chain rather than a set of assertions.

Anchored checks -- a forger must produce a valid kernel checkpoint to pass:

- the checkpoint must exist and restore, which re-verifies every event digest
- the restored projection's recomputed `state_integrity_digest()` must equal the
  final segment's declared `state_integrity_after`
- the scheduled continuity events must be present in the restored projection:
  the long-lived goal, the model replacement, the body change, the defeated
  belief from the conflicting correction, and the history-dependent task
- consecutive segments must chain: successor `state_integrity_before` equals
  predecessor `state_integrity_after`, and only the first segment constructs

Unanchored checks -- these detect accident, not forgery, and are reported as
such: declared duration against file mtime, consolidation cadence, iteration
throughput band, and monotonic non-overlapping start times. They catch a hung
segment, a truncated run, a cadence bug, or receipts copied from the wrong
directory. They do not resist an adversary and are not counted toward
`forgery_resistant_all_pass`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from substrate import final_revision_io as io
from substrate.final_revision_kernel import EventSourcedKernel

CADENCE_SECONDS = 60.0
CADENCE_TOLERANCE = 3
WALL_CLOCK_TOLERANCE = 120.0
MIN_ITERATIONS_PER_SECOND = 10_000.0
MAX_ITERATIONS_PER_SECOND = 50_000_000.0

SCHEDULED_GOALS = ("old-project", "history-dependent-new-task")
SCHEDULED_MODEL = "model-c"
SCHEDULED_BODY = "changed-body-v2"
SCHEDULED_DEFEATED_BELIEF = "old-project-ready"


def check_segment(path: Path) -> dict:
    receipt = json.loads(path.read_text())
    mtime = path.stat().st_mtime
    duration = float(receipt["duration_seconds"])
    started = float(receipt["process_started_unix"])
    observed_wall = mtime - started
    consolidations = int(receipt["background_consolidation_events"])
    expected_consolidations = int(duration // CADENCE_SECONDS)
    iterations = int(receipt["iterations"])
    rate = iterations / duration if duration else 0.0
    unanchored = {
        "wall_clock_covers_declared_duration": observed_wall >= duration - WALL_CLOCK_TOLERANCE,
        "wall_clock_not_absurdly_longer": observed_wall <= duration + WALL_CLOCK_TOLERANCE,
        "consolidation_cadence_matches_duration": abs(consolidations - expected_consolidations) <= CADENCE_TOLERANCE,
        "iteration_rate_plausible": MIN_ITERATIONS_PER_SECOND <= rate <= MAX_ITERATIONS_PER_SECOND,
        "activation_false": receipt.get("activation") is False,
        "process_boundary_declared": receipt.get("process_boundary") is True,
        "state_integrity_advanced": receipt["state_integrity_before"] != receipt["state_integrity_after"],
    }
    return {
        "segment": int(receipt["segment"]),
        "duration_seconds": duration,
        "observed_wall_seconds": observed_wall,
        "consolidations": consolidations,
        "expected_consolidations": expected_consolidations,
        "iterations": iterations,
        "iterations_per_second": rate,
        "restored_from_prior_process": receipt.get("restored_from_prior_process"),
        "state_integrity_before": receipt["state_integrity_before"],
        "state_integrity_after": receipt["state_integrity_after"],
        "process_started_unix": started,
        "receipt_mtime": mtime,
        "unanchored_checks": unanchored,
        "unanchored_all_pass": all(unanchored.values()),
    }


def check_checkpoint(root: Path, rows: list[dict]) -> dict:
    path = root / "checkpoint.json"
    anchored: dict[str, object] = {
        "checkpoint_present": path.is_file(),
        "checkpoint_restores": False,
        "state_integrity_matches_final_segment": False,
        "scheduled_goals_present": False,
        "scheduled_model_replacement_present": False,
        "scheduled_body_change_present": False,
        "scheduled_correction_defeated_belief": False,
    }
    detail: dict[str, object] = {}
    if not path.is_file():
        detail["reason"] = "no checkpoint.json in the run root; receipts cannot be anchored to anything"
        return {"checks": anchored, "detail": detail}

    document = json.loads(path.read_text())
    checkpoint = document.get("checkpoint", document)
    try:
        kernel = EventSourcedKernel.restore(checkpoint)
    except Exception as error:  # noqa: BLE001 - any refusal is a failed anchor
        detail["restore_error"] = f"{type(error).__name__}: {error}"
        return {"checks": anchored, "detail": detail}
    anchored["checkpoint_restores"] = True

    recomputed = kernel.state_integrity_digest()
    detail["recomputed_state_integrity"] = recomputed
    detail["event_count"] = len(checkpoint.get("events", []))
    if rows:
        final_after = rows[-1]["state_integrity_after"]
        detail["final_segment_state_integrity_after"] = final_after
        anchored["state_integrity_matches_final_segment"] = recomputed == final_after

    goals = kernel.query("goals")
    goal_keys = set(goals) if isinstance(goals, dict) else set()
    anchored["scheduled_goals_present"] = all(goal in goal_keys for goal in SCHEDULED_GOALS)
    detail["goals"] = sorted(goal_keys)

    models = kernel.query("model_fabric")["models"]
    anchored["scheduled_model_replacement_present"] = SCHEDULED_MODEL in models
    detail["models"] = sorted(models)

    body = kernel.query("body_and_tools")["body"]
    anchored["scheduled_body_change_present"] = body.get("embodiment") == SCHEDULED_BODY
    detail["embodiment"] = body.get("embodiment")

    belief = kernel.query("beliefs").get(SCHEDULED_DEFEATED_BELIEF)
    anchored["scheduled_correction_defeated_belief"] = bool(belief) and belief.get("defeated") is True
    detail["defeated_belief"] = belief

    return {"checks": anchored, "detail": detail}


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: continuity_receipt_consistency.py RUN_ROOT")
    root = Path(sys.argv[1])
    rows = [check_segment(path) for path in sorted(root.glob("segment-*.json"))]
    if not rows:
        raise SystemExit(f"no segment receipts under {root}")

    chain = {}
    for previous, current in zip(rows, rows[1:]):
        key = f"segment_{previous['segment']}_to_{current['segment']}"
        chain[key] = {
            "successor_restored_from_prior_process": current["restored_from_prior_process"] is True,
            "state_integrity_chains": current["state_integrity_before"] == previous["state_integrity_after"],
            "start_times_monotonic": current["process_started_unix"] > previous["process_started_unix"],
            "segments_do_not_overlap": current["process_started_unix"] >= previous["process_started_unix"] + previous["duration_seconds"] - WALL_CLOCK_TOLERANCE,
        }
    first_segment_constructs = rows[0]["restored_from_prior_process"] is False
    total = sum(row["duration_seconds"] for row in rows)
    checkpoint = check_checkpoint(root, rows)

    forgery_resistant = all(checkpoint["checks"].values()) and all(
        link["state_integrity_chains"] and link["successor_restored_from_prior_process"] for link in chain.values()
    ) and first_segment_constructs

    document = io.authority(
        "substrate-final-revision-continuity-receipt-consistency/v2",
        {
            "purpose": "anchor continuity segment receipts to the owned checkpoint, which restores only if its digest and every event digest validate, so a receipt set is not accepted on its own word",
            "status_note": "post-freeze supplementary verifier; not a required deliverable and not a scored endpoint",
            "receipt_fields_are_unauthenticated_claims": True,
            "run_root": str(root),
            "segments_present": len(rows),
            "segments": rows,
            "chain": chain,
            "checkpoint_anchor": checkpoint,
            "first_segment_constructs_rather_than_restores": first_segment_constructs,
            "total_declared_duration_seconds": total,
            "declared_duration_meets_12_hour_minimum": total >= 43200,
            "forgery_resistant_all_pass": forgery_resistant,
            "unanchored_all_pass": all(row["unanchored_all_pass"] for row in rows),
            "all_pass": forgery_resistant and all(row["unanchored_all_pass"] for row in rows),
        },
        status="complete",
    )
    print(json.dumps(document, sort_keys=True, indent=1))
    raise SystemExit(0 if document["all_pass"] else 1)


if __name__ == "__main__":
    main()
