#!/usr/bin/env python
"""Run the performance microbenchmarks and report them. NON-science: short, deterministic,
machine-relative timings for the hot paths (preprocess, store IO, buffer sampling, faiss-vs-brute
retrieve, a learner step, sweep expansion, manifest write). Separate from the test suite; this is
the human entry point.

Prints a JSON list and a small Markdown table to stdout, writes runs/microbench.md. Logs to stderr.

Usage: python scripts/bench.py [--reps 3]
"""

from __future__ import annotations

import argparse
import json

from mop.bench import render_md, run_benches
from mop.config import REPO_ROOT
from mop.logging_utils import get_logger

log = get_logger("bench")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3, help="runs per bench; the median is reported")
    args = ap.parse_args(argv)

    log.info("running %d microbenchmarks, reps=%d", 8, args.reps)
    records = run_benches(reps=args.reps)

    md = render_md(records)
    out_md = REPO_ROOT / "runs" / "microbench.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    log.info("wrote %s (%d benches)", out_md, len(records))

    print(json.dumps(records, indent=2))
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
