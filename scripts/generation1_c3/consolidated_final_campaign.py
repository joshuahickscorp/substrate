#!/usr/bin/env python3
"""Run the detached conditional Generation 1 final campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.generation1_consolidated_final_campaign import (
    DEFAULT_MANIFEST,
    DEFAULT_RESULT,
    DEFAULT_ROOT,
    run_campaign,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--wait-seconds", type=int, default=120)
    args = parser.parse_args()
    status = run_campaign(
        root=args.root,
        manifest_path=args.manifest,
        result_path=args.result,
        wait_seconds=args.wait_seconds,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["state"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
