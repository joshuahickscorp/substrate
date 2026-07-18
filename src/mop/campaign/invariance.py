"""Unified campaign engine: receipt-invariance and throughput measurement.

Two mandate requirements meet here:

* prove that worker count and scheduling order do not alter receipt bytes for a deterministic workload, and
* choose worker widths from measured throughput, not a decorative process count.

``run_invariance_sweep`` shards a deterministic seeded-hash workload (the campaign's dominant CPU class)
across several worker widths using one ProcessPoolExecutor per width, merges each width's shard digests by
an order-independent reduction, and asserts every width produces the byte-identical merged receipt. It also
records wall-clock throughput per width so a materially different workload can pick its measured optimum
rather than inheriting the seeded-hash ceiling blindly.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from .runners import NodeContext, RunResult, register_runner


def _shard_digest(args: tuple[int, int, int]) -> str:
    """Partition-independent seeded-hash reduction over one shard [lo, hi). Pure and picklable.

    Each item's digest is XOR-folded into a 32-byte accumulator. XOR is commutative and associative, so the
    per-item reduction is independent of how items are partitioned into shards: this is what makes the final
    receipt genuinely invariant to worker count, not merely to shard ordering."""

    lo, hi, seed = args
    acc = bytearray(32)
    for i in range(lo, hi):
        h = hashlib.sha256(f"{seed}:{i}".encode()).digest()
        # a few rounds so the workload is genuinely CPU bound like the campaign's mechanics hashing
        for _ in range(8):
            h = hashlib.sha256(h).digest()
        for k in range(32):
            acc[k] ^= h[k]
    return acc.hex()


def _merged_receipt(shard_digests: list[str]) -> str:
    """XOR-fold the shard reductions into the final receipt. Width-invariant: the folded result equals the
    XOR over ALL items regardless of how they were partitioned across workers."""

    acc = bytearray(32)
    for digest in shard_digests:
        raw = bytes.fromhex(digest)
        for k in range(32):
            acc[k] ^= raw[k]
    return acc.hex()


def run_invariance_sweep(n_items: int, widths: list[int], seed: int) -> dict[str, Any]:
    """Run the deterministic workload at each width; prove the merged receipt is width-invariant."""

    if n_items <= 0 or not widths:
        raise ValueError("n_items must be positive and widths nonempty")
    per_width: list[dict[str, Any]] = []
    receipts: set[str] = set()
    for width in sorted(set(widths)):
        n_shards = max(width, 1)
        bounds = [
            (round(k * n_items / n_shards), round((k + 1) * n_items / n_shards), seed)
            for k in range(n_shards)
        ]
        bounds = [(lo, hi, s) for (lo, hi, s) in bounds if hi > lo]
        started = time.perf_counter()
        if width == 1:
            digests = [_shard_digest(b) for b in bounds]
        else:
            with ProcessPoolExecutor(max_workers=width) as pool:
                digests = list(pool.map(_shard_digest, bounds))
        elapsed = time.perf_counter() - started
        receipt = _merged_receipt(digests)
        receipts.add(receipt)
        per_width.append(
            {
                "width": width,
                "seconds": round(elapsed, 4),
                "throughput_items_per_s": round(n_items / elapsed, 1) if elapsed > 0 else 0.0,
                "receipt": receipt,
            }
        )
    receipt_invariant = len(receipts) == 1
    optimum = max(per_width, key=lambda r: r["throughput_items_per_s"])
    return {
        "n_items": n_items,
        "widths": sorted(set(widths)),
        "seed": seed,
        "per_width": per_width,
        "receipt_invariant": receipt_invariant,
        "receipt": next(iter(receipts)) if receipt_invariant else None,
        "optimum_width": optimum["width"],
        "optimum_throughput_items_per_s": optimum["throughput_items_per_s"],
    }


@register_runner("invariance.receipt_and_throughput")
def invariance_node_runner(params: dict[str, Any], ctx: NodeContext) -> RunResult:
    """A real node: prove receipt invariance across worker widths and record the throughput optimum."""

    n_items = int(params.get("n_items", 4000))
    widths = list(params.get("widths", [1, 2, 4, 8]))
    result = run_invariance_sweep(n_items, widths, ctx.seed)
    content = {
        "schema": "mop-campaign-invariance/v1",
        "node_id": ctx.node_id,
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
        **result,
    }
    path, seal = ctx.seal_json(f"{ctx.node_id}.json", content)
    verdict = "receipt_invariant" if result["receipt_invariant"] else "receipt_variant"
    return RunResult(
        artifact_path=str(path),
        seal=seal,
        verdict=verdict,
        is_null=not result["receipt_invariant"],
        detail={"optimum_width": result["optimum_width"]},
    )
