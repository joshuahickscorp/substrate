#!/usr/bin/env python
"""Validate that a verdict is ledger-ready."""

from __future__ import annotations

import argparse
import json
import sys

from mop.config import REPO_ROOT
from mop.falsification.verdict_gate import build_verdict_gate, write_verdict_gate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="gate a MOP verdict before ledgering")
    p.add_argument("--null-card", required=True, help="proof/NULL_CARDS/<claim>.md")
    p.add_argument("--run-receipt", required=True, help="raw run JSON receipt")
    p.add_argument("--verifier-receipt", default=None, help="independent verifier JSON receipt")
    p.add_argument("--verdict", default=None, help="override the card verdict for this gate")
    p.add_argument(
        "--non-strict-card",
        action="store_true",
        help="allow TODO placeholders in the null card, only for draft dry-runs",
    )
    p.add_argument("--out", default=str(REPO_ROOT / "runs" / "verdict_gate.json"))
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    gate = build_verdict_gate(
        null_card_path=args.null_card,
        run_receipt_path=args.run_receipt,
        verifier_receipt_path=args.verifier_receipt,
        declared_verdict=args.verdict,
        strict_card=not args.non_strict_card,
    )
    write_verdict_gate(gate, args.out)
    print(json.dumps(gate, indent=2, default=str))
    return 0 if gate["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
