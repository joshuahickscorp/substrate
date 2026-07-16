#!/usr/bin/env python3
"""Independently verify the Generation 1 C2 frozen-routing result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.generation1_context_routing_verify import verify_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    result = verify_result(arguments.config, arguments.result, arguments.out)
    print(
        json.dumps(
            {
                "path": str(arguments.out),
                "verification_complete": result["verification_complete"],
                "problems": result["problems"],
                "verification_sha256": result["verification_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["verification_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
