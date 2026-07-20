
from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

DENSITY_SCHEMA = "mop-density-block/v1"

COST_KEYS = ("params", "flops", "active_flops", "bytes", "seconds", "updates", "peak_rss_bytes")


@contextmanager
def timed() -> Iterator[dict]:
    out: dict = {}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["seconds"] = time.perf_counter() - start


def density_block(
    capability: Mapping[str, float],
    *,
    primary: str | None = None,
    params: float | None = None,
    flops: float | None = None,
    active_flops: float | None = None,
    bytes: float | None = None,  # noqa: A002 (doctrine name: retention per byte, alignment per GB)
    seconds: float | None = None,
    updates: float | None = None,
    peak_rss_bytes: float | None = None,
) -> dict:
    if not capability:
        raise ValueError("density_block needs at least one capability metric")
    cap = {str(k): float(v) for k, v in capability.items()}
    primary = str(primary) if primary is not None else next(iter(cap))
    if primary not in cap:
        raise ValueError(f"primary metric {primary!r} not in capability keys {sorted(cap)}")

    provided = {
        "params": params,
        "flops": flops,
        "active_flops": active_flops,
        "bytes": bytes,
        "seconds": seconds,
        "updates": updates,
        "peak_rss_bytes": peak_rss_bytes,
    }
    cost = {k: float(v) for k, v in provided.items() if v is not None}
    if not cost:
        raise ValueError(
            "density_block needs at least one cost metric (params, flops, active_flops, bytes, "
            "seconds, updates, peak_rss_bytes): an unpriced capability is not a result"
        )
    for name, value in cost.items():
        if value < 0:
            raise ValueError(f"cost {name!r} is negative ({value}); a negative cost is a bug")

    score = cap[primary]
    density = {f"{primary}_per_{name}": score / value for name, value in cost.items() if value > 0}
    return {
        "schema": DENSITY_SCHEMA,
        "primary": primary,
        "capability": cap,
        "cost": cost,
        "density": density,
    }
