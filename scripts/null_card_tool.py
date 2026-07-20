#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.falsification.null_cards import (
    extract_card_yaml,
    generate_from_experiment,
    render_card,
    schema,
    validate_card,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="null-card generator and validator")
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("generate", help="generate a draft card from registry/experiments.yaml")
    pg.add_argument("exp_id")
    pg.add_argument("--out", default=None, help="optional output path; omit to print")
    pg.add_argument("--overwrite", action="store_true")

    pv = sub.add_parser("validate", help="validate one card's fenced yaml block")
    pv.add_argument("path")
    pv.add_argument("--strict", action="store_true", help="also reject TODO placeholders")

    sub.add_parser("schema", help="print the null-card schema")

    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    if args.cmd == "generate":
        card = generate_from_experiment(args.exp_id)
        md = render_card(card)
        if args.out is None:
            print(md)
            return 0
        out = Path(args.out)
        if out.exists() and not args.overwrite:
            print(f"{out} exists; pass --overwrite to replace it", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(json.dumps({"out": str(out), "exp_id": args.exp_id, "problems": validate_card(card)}))
        return 0
    if args.cmd == "validate":
        card = extract_card_yaml(Path(args.path).read_text())
        problems = validate_card(card, strict=bool(args.strict))
        print(json.dumps({"path": args.path, "clean": not problems, "problems": problems}, indent=2))
        return 0 if not problems else 1
    if args.cmd == "schema":
        print(json.dumps(schema(), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
