#!/usr/bin/env python
"""Generate and validate the DR1 source-card provenance receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mop.studio.dr1_source_intake import (  # noqa: E402
    build_dr1_source_card,
    load_source_card,
    validate_dr1_source_card,
    write_dr1_source_card,
    write_dr1_source_card_validation,
)

DEFAULT_CARD = _ROOT / "runs" / "studio_dr1" / "dr1_source_card.json"
DEFAULT_VALIDATION = _ROOT / "runs" / "studio_dr1" / "dr1_source_card_validation.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build or validate a DR1 real-video source card")
    sub = parser.add_subparsers(dest="cmd", required=True)

    template = sub.add_parser("template", help="write a populated source-card JSON file")
    template.add_argument("--source-id", required=True)
    template.add_argument("--license", required=True, dest="license_name")
    template.add_argument("--allowed-use", required=True)
    template.add_argument("--non-overlap-proof", required=True)
    template.add_argument("--provenance-tag", default="natural-video")
    template.add_argument("--clip-count", type=int, default=None)
    template.add_argument("--requires-manual-license", action="store_true")
    template.add_argument("--accepted-terms", action="store_true")
    template.add_argument("--license-url", default=None)
    template.add_argument("--source-url", default=None)
    template.add_argument("--notes", default=None)
    template.add_argument("--out", default=str(DEFAULT_CARD))

    validate = sub.add_parser("validate", help="write a source-card validation receipt")
    validate.add_argument("path", nargs="?", default=str(DEFAULT_CARD))
    validate.add_argument("--expected-clip-count", type=int, default=None)
    validate.add_argument("--out", default=str(DEFAULT_VALIDATION))

    args = parser.parse_args(argv)
    if args.cmd == "template":
        card = build_dr1_source_card(
            source_id=args.source_id,
            license_name=args.license_name,
            allowed_use=args.allowed_use,
            non_overlap_proof=args.non_overlap_proof,
            provenance_tag=args.provenance_tag,
            clip_count=args.clip_count,
            requires_manual_license=args.requires_manual_license,
            accepted_terms=args.accepted_terms,
            license_url=args.license_url,
            source_url=args.source_url,
            notes=args.notes,
        )
        write_dr1_source_card(card, args.out)
        receipt = validate_dr1_source_card(card, source_card_path=args.out)
        print(
            json.dumps(
                {
                    "out": args.out,
                    "all_ok": receipt["all_ok"],
                    "problems": receipt["problems"],
                },
                indent=2,
            )
        )
        return 0 if receipt["all_ok"] else 1

    card = None
    load_problem = None
    try:
        card = load_source_card(args.path)
    except Exception as e:  # noqa: BLE001
        load_problem = f"could not load source card: {e}"
    receipt = validate_dr1_source_card(
        card,
        source_card_path=args.path,
        expected_clip_count=args.expected_clip_count,
    )
    if load_problem:
        receipt["problems"].append(load_problem)
        receipt["all_ok"] = False
    write_dr1_source_card_validation(receipt, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "all_ok": receipt["all_ok"],
                "source_card": receipt["source_card"],
                "problems": receipt["problems"],
            },
            indent=2,
        )
    )
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
