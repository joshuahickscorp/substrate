#!/usr/bin/env python3
"""Run the adaptive parallel independent Generation 1 C2 verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.generation1_context_routing_verify_parallel import verify_result_parallel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--idle-workers", type=int, required=True)
    parser.add_argument("--hawking-workers", type=int, required=True)
    arguments = parser.parse_args()
    result = verify_result_parallel(
        arguments.config,
        arguments.result,
        arguments.out,
        idle_workers=arguments.idle_workers,
        hawking_workers=arguments.hawking_workers,
    )
    print(
        json.dumps(
            {
                "path": str(arguments.out),
                "verification_complete": result["verification_complete"],
                "problems": result["problems"],
                "parallel_execution": result["parallel_execution"],
                "verification_sha256": result["verification_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["verification_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
