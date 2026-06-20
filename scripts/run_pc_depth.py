#!/usr/bin/env python
"""Run Leg 8C: predictive-coding-vs-depth. Trains backprop and predictive-coding heads of the
same depth-L MLP on a fixed multiclass latent task and reports the accuracy gap per depth. The
expected finding is that PC tracks backprop when shallow and falls behind as depth grows.

Usage: .venv/bin/python scripts/run_pc_depth.py            # depths 1,2,3,4
       .venv/bin/python scripts/run_pc_depth.py --full     # larger task/budget
Logs to stderr; JSON result to stdout.
"""

from __future__ import annotations

import json
import sys

from devsys.studies.pc_depth import pc_depth


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    toy = "--full" not in argv
    print(f"[8C] running pc_depth toy={toy} depths=(1,2,3,4)", file=sys.stderr)
    out = pc_depth(depths=(1, 2, 3, 4), toy=toy)
    for d in out["per_depth"]:
        print(
            f"[8C] depth={d['depth']} bp_acc={d['bp_acc']:.3f} pc_acc={d['pc_acc']:.3f} gap={d['gap']:.3f}",
            file=sys.stderr,
        )
    print(f"[8C] gap_widens_with_depth={out['gap_widens_with_depth']}", file=sys.stderr)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
