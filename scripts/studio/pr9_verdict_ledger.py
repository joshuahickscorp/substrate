#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mop.studio.pr9_verdict import (  # noqa: E402
    DEFAULT_DR1_CACHE,
    DEFAULT_NULL_CARD,
    build_pr9_verdict_ledger,
    load_json,
    write_pr9_verdict_ledger,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build PR9 verdict ledger from result/state receipts")
    parser.add_argument("--result", default=str(_ROOT / "runs" / "mot" / "pr9_continual_backprop.json"))
    parser.add_argument(
        "--state",
        default=str(_ROOT / "runs" / "mot" / "pr9_continual_backprop.json.state.json"),
    )
    parser.add_argument("--null-card", default=str(_ROOT / DEFAULT_NULL_CARD))
    parser.add_argument("--dr1-cache", default=DEFAULT_DR1_CACHE)
    parser.add_argument("--out", default=str(_ROOT / "runs" / "mot" / "pr9_verdict_ledger.json"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    ledger = build_pr9_verdict_ledger(
        result=load_json(args.result),
        state=load_json(args.state),
        null_card_path=args.null_card,
        dr1_cache=args.dr1_cache,
    )
    write_pr9_verdict_ledger(ledger, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "all_ok": ledger["all_ok"],
                "status": ledger["status"],
                "decision": ledger["decision"],
                "process_c_licensed": ledger["process_c_licensed"],
                "problems": ledger["problems"],
            },
            indent=2,
        )
    )
    return 0 if ledger["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
