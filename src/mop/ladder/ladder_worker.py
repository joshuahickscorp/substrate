"""Subprocess entrypoint that runs one Stage 3 mechanism demonstration and seals a receipt.

The orchestrator spawns this module once per (epoch, seed) work item with the exact CLI:
    python -m mop.ladder.ladder_worker --epoch <key> --seed <int> --reps <int> --out <path>
It runs the demonstration `reps` times as deliberate repeated deterministic work so the campaign
has real CPU to schedule, asserts every repetition produced the identical result digest, and writes
a sealed receipt JSON atomically to the output path. Nondeterminism across repetitions is treated as
a failure: the worker exits nonzero and writes no receipt.

Wall clock time is never placed inside the sealed payload so the receipt stays deterministic; timing
is printed to stderr only. The run registry imports each mechanism's bed and runner on demand, so a
worker pays only for the one epoch it runs. That matters because the campaign schedules many concurrent
workers. The runner mints a mechanics-demonstration RunReceipt; this worker never emits a confirmation.

Claim scope: deterministic programmatic mechanics only; no capability or natural-data claim.
House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from typing import Any

from ..substrate.events import canonical_sha256
from .stage3_registry import run_demonstration

RECEIPT_SCHEMA = "mop-ladder-worker-receipt/v1"


def build_receipt(epoch: str, seed: int, reps: int) -> dict[str, Any]:
    """Run the demonstration `reps` times, assert digest identity, and return a sealed receipt.

    Raises ValueError on a non positive reps count and RuntimeError if any repetition drifts.
    """

    if not isinstance(reps, int) or isinstance(reps, bool) or reps < 1:
        raise ValueError("reps must be a positive integer")
    first = run_demonstration(epoch, seed)
    reference_digest = first.digest()
    for index in range(1, reps):
        repeated = run_demonstration(epoch, seed)
        if repeated.digest() != reference_digest:
            raise RuntimeError(
                f"nondeterministic demonstration for epoch {epoch!r} seed {seed} at rep {index}: "
                f"{repeated.digest()} != {reference_digest}"
            )
    core: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "epoch": epoch,
        "seed": seed,
        "reps": reps,
        "mechanism_id": first.mechanism_id,
        "stage": first.stage,
        "requirement_id": first.requirement_id,
        "verdict": first.verdict,
        "kind": first.kind,
        "is_confirmation": first.is_confirmation,
        "controls_cleared": list(first.controls_cleared),
        "result_digest": reference_digest,
        "claim_scope": first.claim_scope,
    }
    receipt = dict(core)
    receipt["receipt_sha256"] = canonical_sha256(core)
    return receipt


def _write_atomic(path: str, receipt: dict[str, Any]) -> None:
    """Serialize the receipt canonically and replace the target path atomically."""

    data = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle_fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".ladder-worker-", suffix=".tmp")
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mop.ladder.ladder_worker")
    parser.add_argument("--epoch", required=True, help="Stage 3 epoch key to demonstrate")
    parser.add_argument("--seed", type=int, required=True, help="nonnegative demonstration seed")
    parser.add_argument("--reps", type=int, required=True, help="repeated deterministic runs (>= 1)")
    parser.add_argument("--out", required=True, help="path to write the sealed receipt JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse the fixed CLI, run the demonstration, and seal a receipt. Returns the exit code."""

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits on a malformed CLI; convert to a return code.
        return exc.code if isinstance(exc.code, int) else 2
    started = time.perf_counter()
    try:
        receipt = build_receipt(args.epoch, args.seed, args.reps)
        _write_atomic(args.out, receipt)
    except Exception as exc:
        print(f"ladder_worker failure: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    print(
        f"ladder_worker ok epoch={args.epoch} seed={args.seed} reps={args.reps} "
        f"elapsed_s={elapsed:.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
