#!/usr/bin/env python3
"""Execute the f36 and f37 deterministic integration broadcast toy beds."""

from __future__ import annotations

import argparse
from pathlib import Path

from mop.studies.integration_broadcast_runs import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    assert_receipt,
    write_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = write_receipt(args.out, args.config)
    assert_receipt(receipt)
    print(args.out)
    print(receipt["payload_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
