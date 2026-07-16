#!/usr/bin/env python3
"""Run the bounded non-promotable Generation 1 C3/D1 dispatch canary."""

from __future__ import annotations

import argparse
from pathlib import Path

from mop.studies.generation1_c3_dispatch import (
    atomic_write_json,
    pilot_config,
    run_pilot,
    validate_result,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-seed-start", type=int, default=20270001)
    parser.add_argument("--train-seed-count", type=int, default=1)
    parser.add_argument("--heldout-seed-start", type=int, default=20271001)
    parser.add_argument("--heldout-seed-count", type=int, default=1)
    parser.add_argument("--difficulty-indices", default="0")
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-test", type=int, default=90)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--actor-epochs", type=int, default=1)
    parser.add_argument("--router-epochs", type=int, default=12)
    args = parser.parse_args()
    config = pilot_config(
        train_seed_start=args.train_seed_start,
        train_seed_count=args.train_seed_count,
        heldout_seed_start=args.heldout_seed_start,
        heldout_seed_count=args.heldout_seed_count,
        difficulty_indices=tuple(int(value) for value in args.difficulty_indices.split(",")),
        n_train=args.n_train,
        n_test=args.n_test,
        n_classes=args.n_classes,
        dim=args.dim,
        actor_epochs=args.actor_epochs,
        router_epochs=args.router_epochs,
    )
    result = run_pilot(config)
    validate_result(result, config)
    atomic_write_json(args.out, result)
    print(
        f"C3/D1 canary complete: cells={result['grid']['completed_cell_count']} "
        f"seal={result['result_sha256']} out={args.out}"
    )


if __name__ == "__main__":
    main()
