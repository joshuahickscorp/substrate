#!/usr/bin/env python3
"""Run the governed fresh-training challenge for favorable P5 pilot contrasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studies.p5_context_challenge import (
    DEFAULT_RUN_DIR,
    challenge_exit_code,
    run_fresh_challenge,
)
from mop.studies.p5_context_verify import (
    DEFAULT_CHALLENGE,
    DEFAULT_CONFIG,
    DEFAULT_PRIMARY,
    DEFAULT_PRIMARY_RUN_DIR,
    P5VerificationRefused,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--primary-run-dir", type=Path, default=DEFAULT_PRIMARY_RUN_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    from scripts.p5_context_capability import HEAVY_PROCESS_MARKERS, assert_heavy_lane_free

    assert_heavy_lane_free(HEAVY_PROCESS_MARKERS + ("p5_context_fresh_challenge.py",))
    try:
        receipt = run_fresh_challenge(
            args.primary,
            args.primary_run_dir,
            args.config,
            args.run_dir,
            args.out,
            args.device,
            repo_root=REPO_ROOT,
        )
    except P5VerificationRefused as exc:
        print(f"P5 fresh challenge refused: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "complete": receipt["complete"],
                "resumable": receipt["resumable"],
                "all_ok": receipt["all_ok"],
                "patterns": len(receipt["patterns"]),
                "training_runs": len(receipt["training_runs"]),
                "payload_sha256": receipt["payload_sha256"],
                "output": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return challenge_exit_code(receipt)


if __name__ == "__main__":
    raise SystemExit(main())
