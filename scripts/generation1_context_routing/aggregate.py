#!/usr/bin/env python3
"""Aggregate the complete Generation 1 C2 frozen-routing cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.generation1_context_routing import aggregate_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    result = aggregate_result(arguments.config, arguments.work_root, arguments.out)
    print(
        json.dumps(
            {
                "path": str(arguments.out),
                "complete": result["complete"],
                "decision": result["decision"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
