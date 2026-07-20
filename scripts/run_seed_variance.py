#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys

from mop.studies.seed_variance import seed_variance


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Leg 11B seed-variance / statistical power")
    p.add_argument("--max-seeds", type=int, default=8, help="largest seed count to test")
    p.add_argument("--full", action="store_true", help="use full E1 scale (slower) instead of toy")
    args = p.parse_args(argv)

    print(f"[run_seed_variance] max_seeds={args.max_seeds} toy={not args.full}", file=sys.stderr)
    out = seed_variance(max_seeds=args.max_seeds, toy=not args.full)
    for row in out["per_S"]:
        print(
            f"[run_seed_variance] S={row['S']} mean_gap={row['mean_gap']:+.4f} "
            f"sem={row['sem']:.4f} sign_stable={row['sign_stable']}",
            file=sys.stderr,
        )
    print(f"[run_seed_variance] recommended_seeds={out['recommended_seeds']}", file=sys.stderr)
    print(f"[run_seed_variance] provenance={out['provenance']}", file=sys.stderr)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
