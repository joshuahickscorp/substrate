#!/usr/bin/env python
"""Run the bounded, no-weight P7 rendered action-world-model mechanics preflight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mop.studies.action_world_model import DEFAULT_CONFIG, DEFAULT_OUTPUT, write_preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    receipt = write_preflight(args.config.resolve(), args.out.resolve())
    print(
        f"wrote {args.out}: {receipt['status']}, units={len(receipt['units'])}, "
        f"core={receipt['deterministic_core_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
