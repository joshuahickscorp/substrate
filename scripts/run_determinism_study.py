#!/usr/bin/env python
"""Leg 11A: run the determinism characterization and print the report as JSON to stdout.

Runs each representative config (small E1, small I4) N times single-threaded/serial and
reports per-config byte-identity rate and metric spread, plus the CPU-vs-Metal verdict.

Usage: python scripts/run_determinism_study.py [--reps 3]
Logs go to stderr; the JSON report goes to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys

from mop.studies.determinism_study import determinism_study


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    print(f"[determinism_study] {args.reps} reps, single-threaded serial CPU", file=sys.stderr)
    out = determinism_study(reps=args.reps, toy=True)
    for c in out["configs"]:
        print(
            f"[determinism_study] {c['name']} {c['metric']}: "
            f"byte_identical={c['byte_identical']} rate={c['byte_identical_rate']:.2f} "
            f"max_abs={c['max_abs']:.3e} std={c['std']:.3e}",
            file=sys.stderr,
        )
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
