#!/usr/bin/env python
"""Inspect or run declared local work through the five-hour host-aware throttle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studio.local_throttle import (
    DEFAULT_POLICY,
    ThrottleRefused,
    collect_host_telemetry,
    dry_run_decision,
    load_policy,
    run_task,
    write_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list validated task resource declarations")
    snapshot = sub.add_parser("snapshot", help="collect one host telemetry snapshot")
    snapshot.add_argument("--out", type=Path)
    decide = sub.add_parser("decide", help="make a dry-run admission decision; never launches work")
    decide.add_argument("--task", required=True)
    decide.add_argument("--samples", type=int)
    decide.add_argument("--interval-seconds", type=float)
    decide.add_argument("--out", type=Path)
    run = sub.add_parser("run", help="run or dry-run one declared task")
    run.add_argument("--task", required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--execute", action="store_true", help="launch the declared command")
    run.add_argument("--out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = load_policy(args.policy)
        if args.command == "list":
            payload = {
                "policy": str(policy.path),
                "sha256": policy.sha256,
                "profile": policy.profile_name,
                "execution_order": policy.execution_order,
                "tasks": {key: asdict(value) for key, value in policy.tasks.items()},
            }
        elif args.command == "snapshot":
            payload = collect_host_telemetry(policy)
        elif args.command == "decide":
            payload = dry_run_decision(
                policy.task(args.task),
                policy,
                samples=args.samples,
                interval_seconds=args.interval_seconds,
            )
        elif args.command == "run" and not args.execute:
            payload = dry_run_decision(policy.task(args.task), policy)
            payload["mode"] = "run-dry-run"
            payload["run_id"] = args.run_id
        elif args.command == "run":
            payload = run_task(policy.task(args.task), policy, run_id=args.run_id)
        else:  # pragma: no cover
            raise AssertionError(args.command)
        output = getattr(args, "out", None)
        if output:
            out = output if output.is_absolute() else REPO_ROOT / output
            write_receipt(out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.command == "run" and args.execute:
            return 0 if payload.get("status") == "complete" else 2
        if "admission" in payload:
            return 0 if payload["admission"].get("allowed") else 2
        return 0
    except (ThrottleRefused, OSError, ValueError) as exc:
        print(json.dumps({"all_ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
