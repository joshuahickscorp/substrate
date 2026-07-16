#!/usr/bin/env python3
"""Run the restart-safe Generation 1 C1 competence atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.generation1_competence_atlas import MAX_SEED_WORKERS, run_atlas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-workers", type=int, choices=range(1, MAX_SEED_WORKERS + 1), default=1)
    arguments = parser.parse_args()
    result = run_atlas(
        arguments.config,
        arguments.work_root,
        arguments.out,
        seed_workers=arguments.seed_workers,
    )
    print(
        json.dumps(
            {
                "path": str(arguments.out),
                "complete": result["complete"],
                "completed_seed_count": result["grid"]["completed_seed_count"],
                "decision": result["decision"],
                "atlas_sha256": result["atlas_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
