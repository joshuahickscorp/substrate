#!/usr/bin/env python3
"""Run the extended deterministic Generation 1 successor-mechanics queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.generation1_successor_mechanics_queue import DEFAULT_RESULT, DEFAULT_ROOT, run_queue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    status = run_queue(root=args.root, result_path=args.result)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["state"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
