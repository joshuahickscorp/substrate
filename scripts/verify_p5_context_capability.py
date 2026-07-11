#!/usr/bin/env python3
"""Independently recompute and adversarially verify the completed P5 pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studies.p5_context_verify import (
    DEFAULT_CHALLENGE,
    DEFAULT_CONFIG,
    DEFAULT_PRIMARY,
    DEFAULT_PRIMARY_RUN_DIR,
    DEFAULT_VERIFICATION,
    P5VerificationRefused,
    write_verification,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--primary-run-dir", type=Path, default=DEFAULT_PRIMARY_RUN_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fresh-challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--out", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        receipt = write_verification(
            args.out,
            args.primary,
            args.primary_run_dir,
            args.config,
            args.fresh_challenge,
            repo_root=REPO_ROOT,
        )
    except P5VerificationRefused as exc:
        print(f"P5 verification refused: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "primary_profile": receipt["primary_profile"],
                "verification_complete": receipt["verification_complete"],
                "classification": receipt["classification"],
                "prerequisite_ready": receipt["prerequisite_ready"],
                "all_ok": receipt["all_ok"],
                "payload_sha256": receipt["payload_sha256"],
                "output": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["all_ok"] is True and receipt["verification_complete"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
