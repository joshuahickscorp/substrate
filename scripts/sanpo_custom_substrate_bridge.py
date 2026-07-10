#!/usr/bin/env python
"""Verify SANPO input or run the future two-stage portable-substrate evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.substrate.sanpo_bridge import (
    SanpoBridgeRefused,
    evaluate_development_artifact,
    evaluate_official_test_once,
    run_preflight,
)

DEFAULT_PLAN = Path("configs/custom_substrate/sanpo_natural_bridge_v1.json")
DEFAULT_PREFLIGHT_PROOF = Path("proof/SANPO_CUSTOM_SUBSTRATE_BRIDGE_PREFLIGHT.json")
DEFAULT_SELECTION_PROOF = Path("proof/SANPO_CUSTOM_SUBSTRATE_DEVELOPMENT_SELECTION.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight",
        help="hash every source file and decode train+validation only; never load a model",
    )
    preflight.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    preflight.add_argument("--proof", type=Path, default=DEFAULT_PREFLIGHT_PROOF)

    development = subparsers.add_parser(
        "evaluate-development",
        help="future stage 1: verify one portable artifact and evaluate train+validation only",
    )
    development.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    development.add_argument("--artifact-dir", type=Path, required=True)
    development.add_argument("--selection-receipt", type=Path, default=DEFAULT_SELECTION_PROOF)
    development.add_argument("--device", default="cpu")

    official_test = subparsers.add_parser(
        "evaluate-official-test-once",
        help="future stage 2: consume the fixed one-shot ledger and decode the two test clips",
    )
    official_test.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    official_test.add_argument("--artifact-dir", type=Path, required=True)
    official_test.add_argument("--selection-receipt", type=Path, default=DEFAULT_SELECTION_PROOF)
    official_test.add_argument("--device", default="cpu")
    official_test.add_argument(
        "--unlock-official-test",
        action="store_true",
        help="required explicit consent; the fixed one-shot receipt prevents a second attempt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = run_preflight(args.plan, args.proof)
        elif args.command == "evaluate-development":
            result = evaluate_development_artifact(
                args.plan,
                args.artifact_dir,
                args.selection_receipt,
                device=args.device,
            )
        else:
            result = evaluate_official_test_once(
                args.plan,
                args.artifact_dir,
                args.selection_receipt,
                unlock_official_test=args.unlock_official_test,
                device=args.device,
            )
    except (SanpoBridgeRefused, OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "all_ok": False,
                    "refused": True,
                    "command": args.command,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
