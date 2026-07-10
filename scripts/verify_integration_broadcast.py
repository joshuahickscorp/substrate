#!/usr/bin/env python3
"""Independently re-execute and adversarially verify the f36 and f37 toy beds."""

from __future__ import annotations

import argparse
from pathlib import Path

from mop.studies.integration_broadcast_verify import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    DEFAULT_RUN,
    write_verification,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = write_verification(args.out, args.run, args.config)
    print(args.out)
    print(receipt["payload_sha256"])
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
