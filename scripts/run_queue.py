#!/usr/bin/env python
"""Drive the Volume IV campaign queue. Reads campaign/run_queue.yaml, resolves tiers and
dependencies, runs the enabled legs.

Usage:
  python scripts/run_queue.py --dry-run                 # resolve + print the plan
  python scripts/run_queue.py --tiers C                 # run Tier C legs at toy scale
  python scripts/run_queue.py --tiers C,E --max 1       # include env legs, 1 run/leg
On the Studio: --tiers C,E,R --full (drops toy_overrides via the leg sweeps).
"""

from __future__ import annotations

import argparse
import json
import sys

from devsys.harness.queue import run_queue


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tiers", default="C", help="comma list of enabled tiers (C,E,R)")
    ap.add_argument("--full", action="store_true", help="full scale (drop toy_overrides)")
    ap.add_argument("--max", type=int, default=1, help="max runs per leg (toy validation)")
    ap.add_argument("--run-disabled", action="store_true")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    summary = run_queue(
        dry_run=a.dry_run,
        enabled_tiers=set(a.tiers.split(",")),
        toy=not a.full,
        max_runs_per_leg=a.max,
        run_disabled=a.run_disabled,
    )
    summary.pop("raw", None)  # keep stdout readable
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
