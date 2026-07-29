"""Parallel execution of genesis history runs.

Histories are independent by construction — that is what makes the history the
resampling unit — so they parallelise without any shared state. Each worker
rebuilds its own history from the same coordinates, which also makes the whole
run reproducible from the coordinates alone rather than from a shared cache.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from substrate import genesis_config as C
from substrate import genesis_harness as H
from substrate import genesis_history as HI
from substrate import genesis_material as M
from substrate import genesis_tournament as T

DEFAULT_ENVELOPE = "1GB"
DEFAULT_OPERATION_BUDGET = 8_000_000
DEFAULT_DURABLE_WRITE_BUDGET = 8_192


def _load_materials() -> None:
    """Import every module that registers a material."""
    import substrate.genesis_controls  # noqa: F401
    import substrate.genesis_k_advanced  # noqa: F401
    import substrate.genesis_k_basic  # noqa: F401
    import substrate.genesis_k_structural  # noqa: F401
    import substrate.genesis_reference  # noqa: F401


def _cell(job: tuple[str, int, str, str, tuple[str, ...], str, int, int]) -> list[dict[str, Any]]:
    family, history_id, split, seed_namespace, arms, envelope, operation_budget, durable_write_budget = job
    _load_materials()
    unit = HI.build_history(family=family, split=split, history_id=history_id, seed_namespace=seed_namespace)
    factories = T._factories(list(arms), unit)
    result = H.run_history(
        history_id=history_id,
        family=family,
        arms=factories,
        observations=unit.observations,
        alternative_observations=unit.alternative_observations,
        probes=unit.probes,
        judge=unit.judge,
        envelope=envelope,
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
    )
    probe_count = len(unit.probes.development) + len(unit.probes.retention) + len(unit.probes.scoring)
    rows = []
    for arm, arm_run in result["runs"].items():
        rows.append(
            {
                "family": family,
                "history_id": history_id,
                "split": split,
                "arm": arm,
                "score": arm_run.score,
                "retention_score": arm_run.retention_score,
                "development_score": arm_run.development_score,
                "committed": sum(1 for receipt in arm_run.receipts if receipt.committed),
                "refused": sum(1 for receipt in arm_run.receipts if not receipt.committed),
                "rolled_back": arm_run.rolled_back,
                "compute": arm_run.cost.get("compute", 0),
                "peak_bytes": arm_run.peak_resident_bytes,
                "exhausted": arm_run.exhausted,
                "mechanism": arm_run.mechanism,
                "stream_transform": arm_run.stream_transform,
                "wall_clock_seconds": arm_run.wall_clock_seconds,
                "probe_count": probe_count,
            }
        )
    return rows


def run_split(
    *,
    arms: Sequence[str],
    families: Sequence[str],
    histories: Sequence[int],
    split: str,
    seed_namespace: str,
    envelope: str = DEFAULT_ENVELOPE,
    operation_budget: int = DEFAULT_OPERATION_BUDGET,
    durable_write_budget: int = DEFAULT_DURABLE_WRITE_BUDGET,
    workers: int | None = None,
    record_store: str = "record_store_null",
    enforce_scale: bool = True,
) -> dict[str, Any]:
    """Run every arm over every family and history of one split, in parallel."""
    if record_store not in arms:
        raise T.TournamentRefused("the record-store null must run alongside every candidate")
    workers = workers or max(1, min(12, (os.cpu_count() or 4) - 2))
    jobs = [
        (family, history_id, split, seed_namespace, tuple(arms), envelope, operation_budget, durable_write_budget)
        for family in families
        for history_id in histories
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for produced in pool.map(_cell, jobs, chunksize=1):
            rows.extend(produced)

    episodes = sum(row["probe_count"] for row in rows)
    in_range = C.TOURNAMENT_MINIMUM_EPISODES <= episodes <= C.TOURNAMENT_MAXIMUM_EPISODES
    if enforce_scale and not in_range:
        raise T.TournamentRefused(
            f"episode count {episodes} outside the frozen range "
            f"[{C.TOURNAMENT_MINIMUM_EPISODES}, {C.TOURNAMENT_MAXIMUM_EPISODES}]"
        )

    return {
        "rows": rows,
        "arms": list(arms),
        "families": list(families),
        "histories": list(histories),
        "split": split,
        "seed_namespace": seed_namespace,
        "envelope": envelope,
        "episodes": episodes,
        "episodes_in_frozen_range": in_range,
        "scale_enforced": enforce_scale,
        "operation_budget": operation_budget,
        "durable_write_budget": durable_write_budget,
        "record_store": record_store,
        "workers": workers,
        "activation": False,
    }


def run_envelopes(
    *,
    arms: Sequence[str],
    families: Sequence[str],
    histories: Sequence[int],
    seed_namespace: str,
    envelopes: Sequence[str] | None = None,
    operation_budget: int = DEFAULT_OPERATION_BUDGET,
    durable_write_budget: int = DEFAULT_DURABLE_WRITE_BUDGET,
    workers: int | None = None,
) -> dict[str, Any]:
    """Stage 4. The same work under every memory envelope.

    Size is a cost variable, not the goal. This reports absolute capability per
    envelope and the measured footprint that bought it, so a material that
    needs a larger envelope to reach the same score is visible as such.
    """
    envelopes = list(envelopes or C.MEMORY_ENVELOPES)
    per_envelope: dict[str, Any] = {}
    for envelope in envelopes:
        result = run_split(
            arms=arms,
            families=families,
            histories=histories,
            split="train",
            seed_namespace=seed_namespace,
            envelope=envelope,
            operation_budget=operation_budget,
            durable_write_budget=durable_write_budget,
            workers=workers,
            enforce_scale=False,
        )
        by_arm: dict[str, dict[str, float]] = {}
        for row in result["rows"]:
            entry = by_arm.setdefault(row["arm"], {"score": 0.0, "peak_bytes": 0.0, "compute": 0.0, "cells": 0.0, "exhausted": 0.0})
            entry["score"] += row["score"]
            entry["peak_bytes"] = max(entry["peak_bytes"], float(row["peak_bytes"]))
            entry["compute"] += float(row["compute"])
            entry["cells"] += 1.0
            entry["exhausted"] += 1.0 if row["exhausted"] else 0.0
        per_envelope[envelope] = {
            arm: {
                "mean_score": entry["score"] / entry["cells"],
                "peak_resident_bytes": entry["peak_bytes"],
                "mean_compute": entry["compute"] / entry["cells"],
                "exhausted_cells": int(entry["exhausted"]),
                "envelope_bytes": C.ENVELOPE_BYTES[envelope],
                "learning_per_added_byte": (entry["score"] / entry["cells"]) / max(1.0, entry["peak_bytes"]),
            }
            for arm, entry in sorted(by_arm.items())
        }
    return {
        "envelopes": envelopes,
        "per_envelope": per_envelope,
        "families": list(families),
        "histories": list(histories),
        "note": "peak_resident_bytes is the material's own measured packed footprint, not process RSS",
        "activation": False,
    }


def tournament_arms() -> list[str]:
    """Every registered material: candidates, controls, baselines and instruments."""
    _load_materials()
    return sorted(M.registered())
