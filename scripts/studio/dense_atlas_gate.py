#!/usr/bin/env python
"""Validate the paired dense real/random-init caches before the full atlas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mop.studio.dense_atlas_gate import (  # noqa: E402
    DEFAULT_EXPECTED_DIM,
    DEFAULT_MIN_TOKENS,
    DEFAULT_RANDOMINIT_CACHE,
    DEFAULT_REAL_CACHE,
    build_dense_atlas_cache_gate,
    write_dense_atlas_cache_gate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validate paired dense atlas cache manifests")
    parser.add_argument("--real-cache", default=DEFAULT_REAL_CACHE)
    parser.add_argument("--randominit-cache", default=DEFAULT_RANDOMINIT_CACHE)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--expected-dim", type=int, default=DEFAULT_EXPECTED_DIM)
    parser.add_argument("--out", default=str(_ROOT / "runs" / "mot" / "dense_atlas_cache_gate.json"))
    args = parser.parse_args(argv)

    receipt = build_dense_atlas_cache_gate(
        real_cache=args.real_cache,
        randominit_cache=args.randominit_cache,
        min_count=args.min_count,
        min_tokens=args.min_tokens,
        expected_dim=args.expected_dim,
    )
    write_dense_atlas_cache_gate(receipt, args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "all_ok": receipt["all_ok"],
                "real_cache": receipt["real_cache"]["path"],
                "randominit_cache": receipt["randominit_cache"]["path"],
                "problems": receipt["problems"],
            },
            indent=2,
        )
    )
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
