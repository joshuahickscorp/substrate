#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mop.studio.dr1_source_intake import (  # noqa: E402
    DEFAULT_FACTORS,
    build_dr1_source_intake,
    load_source_card,
    write_dr1_source_intake,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validate DR1 source layout, captions, and source card")
    parser.add_argument("--source", required=True, help="dir of <cell>/<clip> plus captions.json")
    parser.add_argument("--source-card", default=None, help="JSON source card with license/provenance proof")
    parser.add_argument("--factors", default=",".join(DEFAULT_FACTORS), help="comma-list composable factors")
    parser.add_argument("--min-per-cell", type=int, default=16)
    parser.add_argument(
        "--out",
        default=str(_ROOT / "runs" / "studio_dr1" / "dr1_source_intake.json"),
        help="receipt output path",
    )
    parser.add_argument(
        "--allow-missing-source-card",
        action="store_true",
        help="record source-card absence without failing the intake gate",
    )
    args = parser.parse_args(argv)

    card = None
    card_problem = None
    try:
        card = load_source_card(args.source_card)
    except Exception as e:  # noqa: BLE001
        card_problem = f"could not load source card: {e}"

    factors = tuple(f for f in args.factors.split(",") if f)
    receipt = build_dr1_source_intake(
        source=args.source,
        factors=factors,
        min_per_cell=args.min_per_cell,
        source_card=card,
        source_card_path=args.source_card,
        require_source_card=not args.allow_missing_source_card,
    )
    if card_problem:
        receipt["problems"].append(card_problem)
        receipt["all_ok"] = False
    write_dr1_source_intake(receipt, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "all_ok": receipt["all_ok"],
                "source": receipt["source"],
                "caption_recoverability": receipt["caption_recoverability"],
                "problems": receipt["problems"],
            },
            indent=2,
        )
    )
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
