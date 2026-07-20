#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mop.studio.atlas_verdict import (  # noqa: E402
    DEFAULT_NULL_CARD,
    build_atlas_verdict_ledger,
    load_json,
    write_atlas_verdict_ledger,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="build atlas verdict ledger from dense-gate and atlas receipts"
    )
    parser.add_argument("--atlas", default=str(_ROOT / "runs" / "mot" / "atlas_multi_encoder_grid.json"))
    parser.add_argument("--dense-gate", default=str(_ROOT / "runs" / "mot" / "dense_atlas_cache_gate.json"))
    parser.add_argument("--null-card", default=str(_ROOT / DEFAULT_NULL_CARD))
    parser.add_argument("--out", default=str(_ROOT / "runs" / "mot" / "atlas_verdict_ledger.json"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    ledger = build_atlas_verdict_ledger(
        atlas=load_json(args.atlas),
        dense_gate=load_json(args.dense_gate),
        null_card_path=args.null_card,
    )
    write_atlas_verdict_ledger(ledger, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "all_ok": ledger["all_ok"],
                "status": ledger["status"],
                "decision": ledger["decision"],
                "claim_status": ledger["claim_status"],
                "problems": ledger["problems"],
            },
            indent=2,
        )
    )
    return 0 if ledger["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
