#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.config import REPO_ROOT
from mop.logging_utils import get_logger
from mop.studies.cost_projection import (
    cost_projection,
    load_timings_from_checkpoint,
    render_md,
)

log = get_logger("project_cost")

DEFAULT_TIMINGS = {
    "e1_baseline": 4.0,
    "e2_replay": 7.5,
    "e3_plasticity": 5.0,
    "e4_neuromod": 4.5,
    "e7_sparse": 6.0,
    "e8_dendritic": 3.0,
    "e9_local": 3.0,
    "i4_backprop_alts": 9.0,
    "e6_relational": 8.0,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="dir of run_pool manifest json files")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--full-seed", type=int, default=5)
    args = ap.parse_args(argv)

    if args.ckpt:
        timings = load_timings_from_checkpoint(Path(args.ckpt))
        log.info("loaded %d measured timings from %s", len(timings), args.ckpt)
        if not timings:
            log.info("no usable manifests in checkpoint dir; using built-in placeholder timings")
            timings = dict(DEFAULT_TIMINGS)
    else:
        log.info("no --ckpt given; using built-in placeholder timings (%d)", len(DEFAULT_TIMINGS))
        timings = dict(DEFAULT_TIMINGS)

    proj = cost_projection(timings, workers=args.workers, full_seed=args.full_seed)

    out_md = REPO_ROOT / "runs" / "cost_projection.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_md(proj))
    log.info(
        "wrote %s | grand serial %.1fs, parallel wall %.1fs (%.2fh) over %d workers",
        out_md,
        proj["grand_total_serial_s"],
        proj["grand_total_parallel_wall_s"],
        proj["grand_total_parallel_wall_s"] / 3600,
        proj["workers"],
    )
    print(json.dumps(proj, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
