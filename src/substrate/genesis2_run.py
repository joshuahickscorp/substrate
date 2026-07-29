"""Parallel, resumable execution for Genesis II developmental histories."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import asdict
from typing import Any, cast

from substrate import genesis2_config as C2
from substrate import genesis2_harness as H2
from substrate import genesis2_io as IO2
from substrate import genesis_history as HI
from substrate import genesis_tournament as T

DEFAULT_ENVELOPE = "1GB"
DEFAULT_OPERATION_BUDGET = 8_000_000
DEFAULT_DURABLE_WRITE_BUDGET = 8_192


def _load_materials() -> None:
    import substrate.genesis2_controls  # noqa: F401
    import substrate.genesis2_material  # noqa: F401


def _worker_init() -> None:
    with suppress(OSError, PermissionError):
        os.nice(cast(int, C2.EXECUTION_POLICY["campaign_worker_nice"]))
    _load_materials()


def _row(
    *,
    family: str,
    history_id: int,
    split: str,
    arm: str,
    run: H2.ArmRun,
    observation_count: int,
    probe_count: int,
) -> dict[str, Any]:
    committed = [receipt for receipt in run.receipts if receipt.committed]
    refused = [receipt for receipt in run.receipts if not receipt.committed]
    return {
        "family": family,
        "history_id": history_id,
        "split": split,
        "arm": arm,
        "score": run.score,
        "retention_score": run.retention_score,
        "development_score": run.development_score,
        "committed": len(committed),
        "refused": len(refused),
        "rolled_back": run.rolled_back,
        "receipt_digest": IO2.digest([asdict(receipt) for receipt in run.receipts]),
        "compute": run.cost.get("compute", 0),
        "durable_writes": run.cost.get("plasticity", 0),
        "retrieval_cost": run.cost.get("retrieval_cost", 0),
        "peak_bytes": run.peak_resident_bytes,
        "exhausted": run.exhausted,
        "mechanism": run.mechanism,
        "stream_transform": run.stream_transform,
        "stream_digest": run.stream_digest,
        "wall_clock_seconds": run.wall_clock_seconds,
        "observation_count": observation_count,
        "probe_count": probe_count,
        "opportunity": run.opportunity,
        "ledger": run.ledger,
        "mechanisms": run.mechanisms,
        "activation": False,
    }


def _run_unit(
    *,
    unit: T.Unit,
    arms: Sequence[str],
    split: str,
    envelope: str,
    operation_budget: int,
    durable_write_budget: int,
    byte_budget: int | None,
) -> list[dict[str, Any]]:
    _load_materials()
    result = H2.run_history(
        history_id=unit.history_id,
        family=unit.family,
        arms=T._factories(arms, unit),
        observations=unit.observations,
        alternative_observations=unit.alternative_observations,
        probes=unit.probes,
        judge=unit.judge,
        envelope=envelope,
        operation_budget=operation_budget,
        durable_write_budget=durable_write_budget,
        byte_budget=byte_budget,
    )
    probe_count = len(unit.probes.development) + len(unit.probes.retention) + len(unit.probes.scoring)
    return [
        _row(
            family=unit.family,
            history_id=unit.history_id,
            split=split,
            arm=arm,
            run=run,
            observation_count=len(unit.observations),
            probe_count=probe_count,
        )
        for arm, run in result["runs"].items()
    ]


def _cell(job: tuple[str, int, str, str, tuple[str, ...], str, int, int, int | None]) -> list[dict[str, Any]]:
    family, history_id, split, seed_namespace, arms, envelope, operations, writes, byte_budget = job
    unit = HI.build_history(
        family=family,
        split=split,
        history_id=history_id,
        seed_namespace=seed_namespace,
    )
    return _run_unit(
        unit=unit,
        arms=arms,
        split=split,
        envelope=envelope,
        operation_budget=operations,
        durable_write_budget=writes,
        byte_budget=byte_budget,
    )


def _composed_cell(
    job: tuple[str, str, int, str, tuple[str, ...], str, int, int, int | None],
) -> list[dict[str, Any]]:
    left, right, history_id, seed_namespace, arms, envelope, operations, writes, byte_budget = job
    unit = HI.build_composed_history(
        left=left,
        right=right,
        history_id=history_id,
        seed_namespace=seed_namespace,
    )
    return _run_unit(
        unit=unit,
        arms=arms,
        split="hidden_composition",
        envelope=envelope,
        operation_budget=operations,
        durable_write_budget=writes,
        byte_budget=byte_budget,
    )


def _default_workers() -> int:
    available = os.cpu_count() or 4
    return max(1, min(8, available - 4))


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
    byte_budget: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run one ordinary split; each job rebuilds its sealed history by digest."""
    if not arms or not families or not histories:
        raise ValueError("arms, families, and histories must all be non-empty")
    if "oracle" not in arms:
        raise ValueError("every reported split requires the oracle headroom arm")
    workers = workers or _default_workers()
    jobs = [
        (
            family,
            int(history_id),
            split,
            seed_namespace,
            tuple(arms),
            envelope,
            int(operation_budget),
            int(durable_write_budget),
            byte_budget,
        )
        for family in families
        for history_id in histories
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        for produced in pool.map(_cell, jobs, chunksize=1):
            rows.extend(produced)
    episodes = sum(int(row["observation_count"]) + int(row["probe_count"]) for row in rows)
    return {
        "rows": rows,
        "arms": list(arms),
        "families": list(families),
        "histories": list(histories),
        "split": split,
        "seed_namespace": seed_namespace,
        "envelope": envelope,
        "byte_budget": byte_budget,
        "operation_budget": operation_budget,
        "durable_write_budget": durable_write_budget,
        "workers": workers,
        "developmental_histories": len(jobs),
        "challenge_units": len(jobs) * HI.UNITS_PER_HISTORY,
        "episodes": episodes,
        "complete": len(rows) == len(jobs) * len(arms),
        "activation": False,
    }


def run_composed_split(
    *,
    arms: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    histories: Sequence[int],
    seed_namespace: str,
    envelope: str = DEFAULT_ENVELOPE,
    operation_budget: int = DEFAULT_OPERATION_BUDGET,
    durable_write_budget: int = DEFAULT_DURABLE_WRITE_BUDGET,
    byte_budget: int | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    if "oracle" not in arms:
        raise ValueError("every reported split requires the oracle headroom arm")
    workers = workers or _default_workers()
    jobs = [
        (
            left,
            right,
            int(history_id),
            seed_namespace,
            tuple(arms),
            envelope,
            int(operation_budget),
            int(durable_write_budget),
            byte_budget,
        )
        for left, right in pairs
        for history_id in histories
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        for produced in pool.map(_composed_cell, jobs, chunksize=1):
            rows.extend(produced)
    episodes = sum(int(row["observation_count"]) + int(row["probe_count"]) for row in rows)
    return {
        "rows": rows,
        "arms": list(arms),
        "pairs": [list(pair) for pair in pairs],
        "histories": list(histories),
        "split": "hidden_composition",
        "seed_namespace": seed_namespace,
        "envelope": envelope,
        "byte_budget": byte_budget,
        "operation_budget": operation_budget,
        "durable_write_budget": durable_write_budget,
        "workers": workers,
        "developmental_histories": len(jobs),
        "challenge_units": len(jobs) * HI.UNITS_PER_HISTORY * 2,
        "episodes": episodes,
        "complete": len(rows) == len(jobs) * len(arms),
        "activation": False,
    }


def summarise(result: Mapping[str, Any]) -> dict[str, Any]:
    """Compact arm and mechanism accounting without discarding raw rows."""
    rows = list(result["rows"])
    by_arm: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    arms: dict[str, Any] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        count = len(arm_rows)
        activations: dict[str, int] = {}
        state_changes: dict[str, int] = {}
        for row in arm_rows:
            mechanisms = row.get("mechanisms", {})
            for name, value in mechanisms.get("activations", {}).items():
                activations[name] = activations.get(name, 0) + int(value)
            for name, value in mechanisms.get("state_changes", {}).items():
                state_changes[name] = state_changes.get(name, 0) + int(value)
        arms[arm] = {
            "mean_score": sum(float(row["score"]) for row in arm_rows) / count,
            "mean_development_score": sum(float(row["development_score"]) for row in arm_rows) / count,
            "mean_retention_score": sum(float(row["retention_score"]) for row in arm_rows) / count,
            "mean_compute": sum(float(row["compute"]) for row in arm_rows) / count,
            "mean_peak_bytes": sum(float(row["peak_bytes"]) for row in arm_rows) / count,
            "mean_committed": sum(float(row["committed"]) for row in arm_rows) / count,
            "mean_refused": sum(float(row["refused"]) for row in arm_rows) / count,
            "exhausted_count": sum(1 for row in arm_rows if row["exhausted"]),
            "cells": count,
            "mechanism_activations": dict(sorted(activations.items())),
            "mechanism_state_changes": dict(sorted(state_changes.items())),
        }
    return {
        "arms": arms,
        "rows": len(rows),
        "episodes": int(result.get("episodes", 0)),
        "complete": bool(result.get("complete")),
        "activation": False,
    }


def demo() -> None:
    _load_materials()
    result = run_split(
        arms=("L9_minimal_sufficient_field", C2.CANONICAL_S2_ID, "record_store_null", "oracle"),
        families=("tool_acquisition",),
        histories=(0,),
        split="train",
        seed_namespace="genesis2-run-demo",
        workers=1,
    )
    assert result["complete"]
    assert len(result["rows"]) == 4
    assert summarise(result)["arms"]["oracle"]["mean_score"] == 1.0
    print("genesis2 parallel runner self-check passed")


if __name__ == "__main__":
    demo()
