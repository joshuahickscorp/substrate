"""Independent consistency check over continuity segment receipts.

Post-freeze supplementary verifier. Writes no evidence.

The lane trusts a segment receipt that is already on disk and sums its
self-declared `duration_seconds`. Nothing cross-checks a receipt against real
elapsed time, process identity, or the checkpoint chain, so a pre-planted
receipt would be accepted. This checks the properties a forged receipt would
have to get right and a real one gets for free:

1. wall-clock: the file's mtime must be at least `duration_seconds` after the
   recorded `process_started_unix`, within tolerance
2. cadence: background consolidation events must match the declared duration at
   the 60-second cadence the lane uses
3. throughput: iterations per second must be positive and within a plausible
   band for a SHA-256 loop, so a receipt cannot claim hours of work at an
   impossible or trivial rate
4. chain: segment N+1 must restore from a prior process and open on the state
   integrity digest segment N closed with
5. ordering: segment start times must be monotonic and non-overlapping

A receipt authored by hand has to satisfy all five simultaneously against a file
mtime the author does not control after the fact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from substrate import final_revision_io as io

CADENCE_SECONDS = 60.0
CADENCE_TOLERANCE = 3
WALL_CLOCK_TOLERANCE = 120.0
MIN_ITERATIONS_PER_SECOND = 10_000.0
MAX_ITERATIONS_PER_SECOND = 50_000_000.0


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
    checks = {
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
        "checks": checks,
        "all_pass": all(checks.values()),
    }


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

    document = io.authority(
        "substrate-final-revision-continuity-receipt-consistency/v1",
        {
            "purpose": "cross-check continuity segment receipts against wall clock, cadence, throughput and the checkpoint chain, so a pre-planted receipt is not accepted on its own word",
            "status_note": "post-freeze supplementary verifier; not a required deliverable and not a scored endpoint",
            "run_root": str(root),
            "segments_present": len(rows),
            "segments": rows,
            "chain": chain,
            "first_segment_constructs_rather_than_restores": first_segment_constructs,
            "total_declared_duration_seconds": total,
            "meets_12_hour_minimum": total >= 43200,
            "all_pass": (
                all(row["all_pass"] for row in rows)
                and all(all(link.values()) for link in chain.values())
                and first_segment_constructs
            ),
        },
        status="complete",
    )
    print(json.dumps(document, sort_keys=True, indent=1))
    raise SystemExit(0 if document["all_pass"] else 1)


if __name__ == "__main__":
    main()
