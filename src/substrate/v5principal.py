"""Frozen, resumable developmental DAG for the Substrate v5 campaign."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import os
import statistics
import time
from typing import Any

from substrate import v5config as C
from substrate import v5experiment as E
from substrate import v5io as io

SHARDS = 4
PHASES_PER_SHARD = 5
SPLIT_SEEDS = {
    "principal": tuple(range(5_000, 5_048)),
    "replication": tuple(range(6_000, 6_016)),
    "open_world_review": tuple(range(7_000, 7_016)),
}


class Refused(RuntimeError):
    """A principal unit, checkpoint, or launch violated the frozen authority."""


@dataclasses.dataclass(frozen=True)
class WorkUnit:
    split: str
    history_seed: int
    arm: str
    shard: int

    def __post_init__(self) -> None:
        if self.split not in SPLIT_SEEDS:
            raise Refused(f"unknown split {self.split!r}")
        if self.history_seed not in SPLIT_SEEDS[self.split]:
            raise Refused("history seed outside frozen split")
        if self.arm not in C.ARMS:
            raise Refused(f"unknown arm {self.arm!r}")
        if not 0 <= self.shard < SHARDS:
            raise Refused("shard outside frozen DAG")

    @property
    def identity(self) -> str:
        return (
            f"{self.split}-{self.history_seed}-{self.arm}-"
            f"shard{self.shard:02d}"
        )

    @property
    def phase_indices(self) -> tuple[int, ...]:
        start = self.shard * PHASES_PER_SHARD
        return tuple(range(start, start + PHASES_PER_SHARD))

    @property
    def dependency(self) -> str | None:
        if self.shard == 0:
            return None
        return dataclasses.replace(self, shard=self.shard - 1).identity

    @property
    def event_count(self) -> int:
        return len(self.phase_indices) * E.EPISODES_PER_PHASE

    def document(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "hypotheses": list(C.HYPOTHESES),
            "arm": self.arm,
            "history_seed": self.history_seed,
            "split": self.split,
            "phase_indices": list(self.phase_indices),
            "phases": [C.PHASES[index] for index in self.phase_indices],
            "modalities": sorted(
                {
                    modality
                    for index in self.phase_indices
                    for modality in E.PHASE_MODALITIES[index]
                }
            ),
            "models": "registered model-equivalent modules selected by the v5 fabric",
            "body": "desktop_body or seeded_3d_body",
            "inputs": [E.generator_manifest()["generator_digest"]],
            "outputs": [
                f"units/{self.identity}.json",
                f"checkpoints/{self.identity}.json",
            ],
            "dependencies": [self.dependency] if self.dependency else [],
            "resource_class": "cpu_small",
            "worker_class": "deterministic_developmental_history",
            "native_thread_budget": 1,
            "accelerator_requirement": "none",
            "timeout_seconds": 120,
            "retry": "one deterministic retry; preserve both failure receipts",
            "checkpoint": f"checkpoints/{self.identity}.json",
            "artifact_family": self.split,
            "claim_ceiling": "multimodal_nous_ready_for_review",
            "event_count": self.event_count,
            "activation": False,
        }


def work_units(split: str | None = None) -> list[WorkUnit]:
    splits = (split,) if split else tuple(SPLIT_SEEDS)
    return [
        WorkUnit(name, seed, arm, shard)
        for name in splits
        for seed in SPLIT_SEEDS[name]
        for arm in C.ARMS
        for shard in range(SHARDS)
    ]


def _initial_state(unit: WorkUnit) -> dict[str, Any]:
    identity = E.history_identity(unit.split, unit.history_seed, unit.arm)
    return {
        "entity_identity": identity,
        "birth_identity": identity,
        "completed_phase": -1,
        "developmental_events": 0,
        "semantic_memories": 0,
        "procedural_memories": 0,
        "tracked_objects": 0,
        "unfinished_goals": ["return-to-scene"],
        "model_identity": "vision-temporal-alpha",
        "model_replacements": 0,
        "body_identity": "desktop-body",
        "body_changes": 0,
        "sensor_interruptions": 0,
        "restorations": 0,
        "activation": False,
    }


def _checkpoint_body(
    unit: WorkUnit,
    predecessor: dict[str, Any] | None,
    phases: list[dict[str, Any]],
) -> dict[str, Any]:
    if predecessor is None:
        state = _initial_state(unit)
    else:
        state = dict(predecessor["state"])
        if (
            unit.arm != "fresh_reset"
            and state.get("entity_identity")
            != E.history_identity(unit.split, unit.history_seed, unit.arm)
        ):
            raise Refused("predecessor changed continuing entity identity")
        if int(state.get("completed_phase", -1)) != unit.phase_indices[0] - 1:
            raise Refused("predecessor phase does not match DAG dependency")
    if unit.arm == "fresh_reset" and unit.shard > 0:
        state = _initial_state(unit)
        state["entity_identity"] = E.history_identity(
            unit.split,
            unit.history_seed + unit.shard * 100_000,
            unit.arm,
        )
    state["completed_phase"] = unit.phase_indices[-1]
    state["developmental_events"] = int(state["developmental_events"]) + sum(
        int(row["episodes"]) for row in phases
    )
    state["semantic_memories"] = int(state["semantic_memories"]) + sum(
        int(row["accuracy"] * row["episodes"]) for row in phases
    )
    state["procedural_memories"] = int(state["procedural_memories"]) + sum(
        bool(row["mechanisms_active"]) for row in phases
    )
    state["tracked_objects"] = max(
        int(state["tracked_objects"]),
        3 + sum("video" in row["modalities"] for row in phases),
    )
    if 13 in unit.phase_indices:
        state["sensor_interruptions"] = int(state["sensor_interruptions"]) + 1
        state["restorations"] = int(state["restorations"]) + 1
    if 14 in unit.phase_indices:
        state["model_identity"] = "vision-temporal-beta"
        state["model_replacements"] = int(state["model_replacements"]) + 1
    if 15 in unit.phase_indices:
        state["body_identity"] = "seeded-3d-body"
        state["body_changes"] = int(state["body_changes"]) + 1
    predecessor_digest = predecessor.get("state_digest") if predecessor else None
    state_digest = E.transition_digest(
        str(predecessor_digest) if predecessor_digest else None,
        str(state["entity_identity"]),
        phases,
    )
    return {
        "schema": "substrate-v5-developmental-checkpoint/v1",
        "unit": unit.document(),
        "predecessor_checkpoint": predecessor_digest,
        "state": state,
        "state_digest": state_digest,
        "checkpoint_exact": True,
        "activation": False,
    }


def execute_unit(
    unit: WorkUnit,
    predecessor: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    phases = [
        E.phase_result(
            split=unit.split,
            history_seed=unit.history_seed,
            arm=unit.arm,
            phase_index=index,
        )
        for index in unit.phase_indices
    ]
    checkpoint = _checkpoint_body(unit, predecessor, phases)
    state = checkpoint["state"]
    receipt = {
        "schema": "substrate-v5-principal-unit/v1",
        "unit": unit.document(),
        "predecessor_checkpoint": checkpoint["predecessor_checkpoint"],
        "phase_results": phases,
        "summary": {
            "mean_accuracy": statistics.fmean(
                float(row["accuracy"]) for row in phases
            ),
            "mean_utility": statistics.fmean(
                float(row["utility"]) for row in phases
            ),
            "mean_cost": statistics.fmean(
                float(row["mean_cost"]) for row in phases
            ),
            "mean_uncertainty": statistics.fmean(
                float(row["mean_uncertainty"]) for row in phases
            ),
            "mechanisms_active": sorted(
                {
                    mechanism
                    for row in phases
                    for mechanism in row["mechanisms_active"]
                }
            ),
            "modalities": sorted(
                {modality for row in phases for modality in row["modalities"]}
            ),
            "events": sum(int(row["episodes"]) for row in phases),
            "entity_identity": state["entity_identity"],
            "birth_identity": state["birth_identity"],
            "model_identity": state["model_identity"],
            "body_identity": state["body_identity"],
            "unfinished_goals": state["unfinished_goals"],
            "state_digest": checkpoint["state_digest"],
            "checkpoint_exact": True,
        },
        "source_generator_digest": E.generator_manifest()["generator_digest"],
        "activation": False,
    }
    return receipt, checkpoint


def validate(
    receipt: dict[str, Any],
    checkpoint: dict[str, Any],
    unit: WorkUnit,
    predecessor: dict[str, Any] | None = None,
) -> bool:
    expected_receipt, expected_checkpoint = execute_unit(unit, predecessor)
    return (
        receipt == expected_receipt
        and checkpoint == expected_checkpoint
        and receipt.get("activation") is False
        and checkpoint.get("activation") is False
        and all(
            row.get("commitment_precedes_target") is True
            and row.get("raw_observation_excludes_target") is True
            for row in receipt.get("phase_results", [])
        )
    )


def _relative(unit: WorkUnit, family: str) -> str:
    return f"{unit.split}/{family}/{unit.identity}.json"


def _load_if_valid(
    unit: WorkUnit,
    predecessor: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    receipt_path = io.RUNS / _relative(unit, "units")
    checkpoint_path = io.RUNS / _relative(unit, "checkpoints")
    if not receipt_path.is_file() or not checkpoint_path.is_file():
        return None
    try:
        receipt = dict(io.load_json(receipt_path))
        checkpoint = dict(io.load_json(checkpoint_path))
    except io.Refused:
        return None
    for document in (receipt, checkpoint):
        document.pop("program", None)
        document.pop("sha256", None)
    return (receipt, checkpoint) if validate(
        receipt,
        checkpoint,
        unit,
        predecessor,
    ) else None


def _chain(split: str, history_seed: int, arm: str) -> list[
    tuple[dict[str, Any], dict[str, Any]]
]:
    predecessor = None
    rows = []
    for shard in range(SHARDS):
        unit = WorkUnit(split, history_seed, arm, shard)
        receipt, checkpoint = execute_unit(unit, predecessor)
        rows.append((receipt, checkpoint))
        predecessor = checkpoint
    return rows


def _worker(arguments: tuple[str, int, str]) -> tuple[
    tuple[str, int, str],
    list[tuple[dict[str, Any], dict[str, Any]]],
]:
    return arguments, _chain(*arguments)


def prepare() -> dict[str, Any]:
    units = work_units()
    manifest = {
        "schema": "substrate-v5-principal-dag/v1",
        "units": [unit.document() for unit in units],
        "unit_count": len(units),
        "developmental_histories": sum(len(values) for values in SPLIT_SEEDS.values()),
        "principal_histories": len(SPLIT_SEEDS["principal"]),
        "arms": list(C.ARMS),
        "phases": list(C.PHASES),
        "sensory_events_or_cognitive_episodes": sum(
            unit.event_count for unit in units
        ),
        "generator": E.generator_manifest(),
        "frozen": True,
        "activation": False,
    }
    resource_plan = {
        "schema": "substrate-v5-resource-plan/v1",
        "worker_candidates": [1, 2, 4, 8, 12, 16],
        "selected_workers": min(8, os.cpu_count() or 1),
        "native_threads_per_worker": 1,
        "central_authoritative_publisher": True,
        "accelerator_required": False,
        "minimum_free_disk_gib": 25,
        "activation": False,
    }
    io.config_json("principal_manifest.json", manifest)
    io.seal("SUBSTRATE_V5_PRINCIPAL_DAG.json", manifest)
    io.seal("SUBSTRATE_V5_RESOURCE_PLAN.json", resource_plan)
    io.seal(
        "SUBSTRATE_V5_WORKER_AUTHORITY.json",
        {
            "schema": "substrate-v5-worker-authority/v1",
            "workers_write_staging_only": True,
            "publisher_validates_and_publishes_atomically": True,
            "duplicate_units_refused": True,
            "worker_candidates": resource_plan["worker_candidates"],
            "selected_workers": resource_plan["selected_workers"],
            "activation": False,
        },
    )
    return {"manifest": manifest, "resource_plan": resource_plan}


def run(
    split: str | None = None,
    *,
    workers: int | None = None,
) -> dict[str, Any]:
    prepare()
    splits = (split,) if split else tuple(SPLIT_SEEDS)
    if any(name not in SPLIT_SEEDS for name in splits):
        raise Refused("unknown principal split")
    if io.STOP.exists():
        raise Refused("v5 stop switch is present")
    selected_workers = workers or min(8, os.cpu_count() or 1)
    selected_workers = max(1, min(16, int(selected_workers)))
    chains = [
        (name, seed, arm)
        for name in splits
        for seed in SPLIT_SEEDS[name]
        for arm in C.ARMS
    ]
    started = time.perf_counter()
    published = 0
    failed: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=selected_workers,
    ) as executor:
        futures = {executor.submit(_worker, row): row for row in chains}
        for future in concurrent.futures.as_completed(futures):
            arguments = futures[future]
            try:
                _, results = future.result()
                predecessor = None
                for shard, (receipt, checkpoint) in enumerate(results):
                    unit = WorkUnit(*arguments, shard)
                    if not validate(receipt, checkpoint, unit, predecessor):
                        raise Refused(f"worker returned invalid unit {unit.identity}")
                    io.run_json(_relative(unit, "units"), receipt)
                    io.run_json(_relative(unit, "checkpoints"), checkpoint)
                    predecessor = checkpoint
                    published += 1
            except Exception as error:  # noqa: BLE001 - failure receipt is required
                failed.append(
                    {
                        "chain": list(arguments),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    elapsed = time.perf_counter() - started
    expected = sum(
        len(SPLIT_SEEDS[name]) * len(C.ARMS) * SHARDS for name in splits
    )
    result = {
        "schema": "substrate-v5-principal-execution/v1",
        "splits": list(splits),
        "expected_units": expected,
        "published_units": published,
        "failed_attempts": failed,
        "all_terminal": published == expected and not failed,
        "workers": selected_workers,
        "wall_seconds": elapsed,
        "units_per_second": published / elapsed if elapsed else None,
        "sensory_events_or_cognitive_episodes": published
        * PHASES_PER_SHARD
        * E.EPISODES_PER_PHASE,
        "activation": False,
    }
    io.seal("SUBSTRATE_V5_PRINCIPAL_AUTHORITY.json", result)
    if failed:
        raise Refused(f"{len(failed)} principal chains failed")
    return result


def status() -> dict[str, Any]:
    expected = work_units()
    valid = 0
    split_counts: dict[str, dict[str, int]] = {}
    for name in SPLIT_SEEDS:
        split_counts[name] = {
            "expected": len(work_units(name)),
            "present": 0,
        }
    for unit in expected:
        receipt = io.RUNS / _relative(unit, "units")
        checkpoint = io.RUNS / _relative(unit, "checkpoints")
        if receipt.is_file() and checkpoint.is_file():
            valid += 1
            split_counts[unit.split]["present"] += 1
    return {
        "schema": "substrate-v5-principal-status/v1",
        "expected": len(expected),
        "present": valid,
        "remaining": len(expected) - valid,
        "splits": split_counts,
        "complete": valid == len(expected),
        "activation": False,
    }
