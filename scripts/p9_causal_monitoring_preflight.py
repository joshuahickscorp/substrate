#!/usr/bin/env python
"""Build or exactly replay-verify the bounded P9 causal-monitoring mechanics receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.studies.p9_causal_monitoring import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    verify_preflight_receipt,
    write_preflight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify",
        type=Path,
        help="rebuild the deterministic core and verify this existing receipt instead of writing",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.verify is not None:
        receipt = json.loads(args.verify.resolve().read_text(encoding="utf-8"))
        audit = verify_preflight_receipt(receipt, args.config.resolve())
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["verified"] else 1
    receipt = write_preflight(args.config.resolve(), args.out.resolve())
    print(
        f"wrote {args.out}: {receipt['status']}, units={len(receipt['units'])}, "
        f"core={receipt['deterministic_core_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
