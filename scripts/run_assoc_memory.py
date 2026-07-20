#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys

from mop.studies.assoc_memory import assoc_memory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--corruption", type=float, default=0.3)
    ap.add_argument("--beta", type=float, nargs="+", default=[8.0, 16.0, 32.0])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--thresh", type=float, default=0.9)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="larger K cap (slower)")
    a = ap.parse_args(argv)

    print(f"[8B] dim={a.dim} corruption={a.corruption} betas={a.beta} steps={a.steps}", file=sys.stderr)
    out = assoc_memory(
        dim=a.dim,
        betas=tuple(a.beta),
        corruption=a.corruption,
        steps=a.steps,
        thresh=a.thresh,
        floor=a.floor,
        seed=a.seed,
        toy=not a.full,
    )
    hop, ff = out["hopfield"]["capacity"], out["feedforward"]["capacity"]
    wins = out["hopfield_wins"]
    print(f"[8B] capacity hopfield={hop} feedforward={ff} hopfield_wins={wins}", file=sys.stderr)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
