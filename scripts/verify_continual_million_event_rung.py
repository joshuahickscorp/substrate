#!/usr/bin/env python
"""Independently verify a completed P6 progressive rung receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.continual_million_event_verify import write_verification_receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = write_verification_receipt(args.source, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "source_rung": receipt["source_rung"]["rung"],
                "verification_complete": receipt["verification_complete"],
                "verdict": receipt["independent_recompute"].get("decision", {}).get("verdict"),
                "payload_sha256": receipt["payload_sha256"],
                "errors": receipt["errors"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["verification_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
