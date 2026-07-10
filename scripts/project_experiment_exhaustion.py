#!/usr/bin/env python
"""Execute and inventory the complete non-F experiment surface.

Examples:
  python scripts/project_experiment_exhaustion.py --all
  python scripts/project_experiment_exhaustion.py --execute-classes --max-workers 6
  python scripts/project_experiment_exhaustion.py --ledger

Runs are resumable under ``runs/project_exhaustion/current``.  Registry-only Studio labels
are never treated as measured hardware boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.project_exhaustion import (
    DEFAULT_EXECUTION_ROOT,
    DEFAULT_LEDGER_PATH,
    execute_blocked_class_smokes,
    execute_cpu_classes,
    execute_diagnostics,
    worker_execute_experiment,
    write_ledger,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="run CPU classes, diagnostics, then write ledger")
    ap.add_argument("--execute-classes", action="store_true")
    ap.add_argument("--execute-blocked-smokes", action="store_true")
    ap.add_argument("--execute-diagnostics", action="store_true")
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_LEDGER_PATH)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=300.0, help="per-class timeout seconds")
    ap.add_argument("--wall-seconds", type=float, default=5100.0)
    ap.add_argument("--worker-experiment", help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker_experiment:
        if args.run_dir is None:
            ap.error("--worker-experiment requires --run-dir")
        report = worker_execute_experiment(args.worker_experiment, args.run_dir, args.seed)
        print("PROJECT_EXHAUSTION_WORKER=" + json.dumps(report, sort_keys=True))
        return 0

    execute_classes = args.all or args.execute_classes
    execute_blocked_smokes = args.all or args.execute_blocked_smokes
    execute_diagnostic_rows = args.all or args.execute_diagnostics
    emit_ledger = (
        args.all or args.ledger or not (execute_classes or execute_blocked_smokes or execute_diagnostic_rows)
    )
    summary: dict[str, object] = {}
    if execute_classes:
        summary["classes"] = execute_cpu_classes(
            args.execution_root,
            seed=args.seed,
            max_workers=args.max_workers,
            timeout_s=args.timeout,
            wall_seconds=args.wall_seconds,
            script_path=Path(__file__).resolve(),
        )
    if execute_blocked_smokes:
        summary["blocked_smokes"] = execute_blocked_class_smokes(
            args.execution_root,
            seed=args.seed,
            timeout_s=args.timeout,
            script_path=Path(__file__).resolve(),
        )
    if execute_diagnostic_rows:
        summary["diagnostics"] = execute_diagnostics(args.execution_root, seed=args.seed)
    if args.all:
        emit_ledger = True
    if emit_ledger:
        ledger = write_ledger(args.out, args.execution_root)
        summary["ledger"] = {
            "path": str(args.out),
            "coverage": ledger["coverage"],
            "resumable_queue": ledger["resumable_queue"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
