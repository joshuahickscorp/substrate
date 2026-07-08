#!/usr/bin/env python
"""Write a gated daemon plan for a Studio claim ledger update."""

from __future__ import annotations

import argparse
import json
import sys

from mop.config import REPO_ROOT
from mop.studio.claim_plan import build_claim_daemon_plan, write_claim_daemon_plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build verdict/artifact/ledger daemon plan")
    parser.add_argument("--null-card", required=True)
    parser.add_argument("--run-receipt", required=True)
    parser.add_argument("--verifier-receipt", default=None)
    parser.add_argument("--verdict", default="PUBLISH-POSITIVE")
    parser.add_argument("--verdict-gate-out", required=True)
    parser.add_argument("--artifact-index-out", required=True)
    parser.add_argument("--artifact-path", action="append", default=[], help="extra receipt path")
    parser.add_argument("--copy-dir", default=None)
    parser.add_argument("--no-require-durable", action="store_true", help="draft only")
    parser.add_argument(
        "--ledger-cmd-json",
        required=True,
        help='JSON array command, for example ["python","scripts/studio_wave0_report.py","--apply"]',
    )
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "studio_claim_plan.json"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    try:
        ledger_cmd = json.loads(args.ledger_cmd_json)
    except json.JSONDecodeError as e:
        print(
            json.dumps({"ok": False, "error": f"ledger-cmd-json parse failed: {e}"}, indent=2),
            file=sys.stderr,
        )
        return 2
    if not isinstance(ledger_cmd, list) or not all(isinstance(part, str) for part in ledger_cmd):
        print(
            json.dumps({"ok": False, "error": "ledger-cmd-json must be a JSON array of strings"}, indent=2),
            file=sys.stderr,
        )
        return 2

    try:
        plan = build_claim_daemon_plan(
            null_card=args.null_card,
            run_receipt=args.run_receipt,
            verifier_receipt=args.verifier_receipt,
            verdict=args.verdict,
            verdict_gate_out=args.verdict_gate_out,
            artifact_index_out=args.artifact_index_out,
            artifact_paths=args.artifact_path,
            copy_dir=args.copy_dir,
            require_durable=not args.no_require_durable,
            ledger_cmd=ledger_cmd,
            python=args.python,
        )
        write_claim_daemon_plan(plan, args.out)
    except ValueError as e:
        print(json.dumps({"ok": False, "error": str(e)}, indent=2), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "out": args.out,
                "jobs": [job["id"] for job in plan["jobs"]],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
