#!/usr/bin/env python3
"""Run one restart-safe adaptive Generation 1 C2 routing shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.generation1_context_routing import run_shard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--idle-workers", type=int, required=True)
    parser.add_argument("--hawking-workers", type=int, required=True)
    parser.add_argument("--max-new-cells", type=int)
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    resources = config.get("adaptive_resources") or {}
    if arguments.idle_workers != resources.get("idle_workers") or arguments.hawking_workers != resources.get(
        "hawking_workers"
    ):
        raise SystemExit("command worker declaration differs from the sealed C2 config")
    result = run_shard(
        arguments.config,
        arguments.work_root,
        arguments.out,
        arguments.shard_index,
        max_new_cells=arguments.max_new_cells,
    )
    print(
        json.dumps(
            {
                "path": str(arguments.out),
                "shard_index": result["shard_index"],
                "complete": result["complete"],
                "grid": result["grid"],
                "shard_sha256": result["shard_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
