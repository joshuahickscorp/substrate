#!/usr/bin/env python3
"""Run the bounded local action-environment preflights and write durable receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.local_action_environment import (
    DEFAULT_CONFIG,
    DEFAULT_PROOF,
    DEFAULT_RUN_DIR,
    DEFAULT_RUN_RECEIPT,
    write_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--run-receipt", type=Path, default=DEFAULT_RUN_RECEIPT)
    parser.add_argument("--proof", type=Path, default=DEFAULT_PROOF)
    args = parser.parse_args()
    result = write_preflight(
        config_path=args.config,
        run_dir=args.run_dir,
        run_receipt=args.run_receipt,
        proof_path=args.proof,
    )
    print(
        json.dumps(
            {
                "proof": str(args.proof),
                "run_receipt": str(args.run_receipt),
                "seed_count": result["seed_count"],
                "all_mechanics_verified": result["all_mechanics_verified"],
                "scientific_promotion_allowed": result["scientific_promotion_allowed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
